from src.utils.privacy import disable_telemetry
disable_telemetry()

import os
from pathlib import Path
from typing import Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import BaseModel

from src.core.emperor_brain import EmperorBrain
from src.infrastructure.database import Database
from src.infrastructure.vector_store import VectorStore
from src.memory.condensation import CondensationManager
from src.models.schemas import Session, Message, User
from src.utils.config import load_config
from src.utils.auth import get_user_id_from_token, optional_auth, security

app = FastAPI(title="Stoic Emperor", docs_url="/api/docs")

config = load_config()
db = Database(config["database"]["url"])
vectors = VectorStore(config["database"]["url"])
brain = EmperorBrain(config=config)
condensation = CondensationManager(db, config)

ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
DEFAULT_USER_ID = "default_user"
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")


def get_current_user_id(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> str:
    if ENVIRONMENT == "development" and not credentials:
        return DEFAULT_USER_ID

    if not credentials:
        raise HTTPException(status_code=401, detail="Authentication required")

    try:
        from src.utils.auth import get_user_id_from_token
        return get_user_id_from_token(credentials)
    except Exception as e:
        if ENVIRONMENT == "development":
            return DEFAULT_USER_ID
        raise

static_path = Path(__file__).parent / "static"
templates_path = Path(__file__).parent / "templates"
app.mount("/static", StaticFiles(directory=static_path), name="static")
templates = Jinja2Templates(directory=templates_path)


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    session_id: str
    message_id: str


class SessionInfo(BaseModel):
    id: str
    created_at: datetime
    message_count: int


class MessageInfo(BaseModel):
    id: str
    role: str
    content: str
    created_at: datetime


class ProfileInfo(BaseModel):
    version: int
    content: str
    created_at: datetime
    consensus_reached: Optional[bool] = None
    stability_score: Optional[float] = None


class AnalysisStatus(BaseModel):
    uncondensed_tokens: int
    condensation_threshold: int
    summary_count: int
    has_profile: bool


class UserInfo(BaseModel):
    id: str
    name: Optional[str] = None
    created_at: datetime


class UpdateNameRequest(BaseModel):
    name: str


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request, "active_page": "chat"})


@app.get("/login")
async def login(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


@app.get("/history")
async def history(request: Request):
    return templates.TemplateResponse("history.html", {"request": request, "active_page": "history"})


@app.get("/analysis")
async def analysis(request: Request):
    return templates.TemplateResponse("analysis.html", {"request": request, "active_page": "analysis"})


@app.get("/api/config")
async def get_config():
    return {
        "supabase_url": SUPABASE_URL,
        "supabase_anon_key": SUPABASE_ANON_KEY,
        "environment": ENVIRONMENT
    }


@app.get("/api/user", response_model=UserInfo)
async def get_user(user_id: str = Depends(get_current_user_id)):
    user = db.get_or_create_user(user_id)
    return UserInfo(id=user.id, name=user.name, created_at=user.created_at)


@app.put("/api/user/name", response_model=UserInfo)
async def update_user_name(request: UpdateNameRequest, user_id: str = Depends(get_current_user_id)):
    user = db.update_user_name(user_id, request.name)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return UserInfo(id=user.id, name=user.name, created_at=user.created_at)


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, user_id: str = Depends(get_current_user_id)):
    user = db.get_or_create_user(user_id)

    if request.session_id:
        session = db.get_session(request.session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
    else:
        session = db.get_latest_session(user.id)
        if not session:
            session = Session(user_id=user.id)
            db.create_session(session)

    history = db.get_session_messages(session.id)

    user_msg = Message(session_id=session.id, role="user", content=request.message)
    db.save_message(user_msg)
    history.append(user_msg)

    retrieved_context = _retrieve_context(request.message, user.id)
    response = brain.respond(
        user_message=request.message,
        conversation_history=history,
        retrieved_context=retrieved_context
    )

    emperor_msg = Message(
        session_id=session.id,
        role="emperor",
        content=response.response_text,
        psych_update=response.psych_update
    )
    db.save_message(emperor_msg)

    _maybe_condense_and_analyze(user.id)

    return ChatResponse(
        response=response.response_text,
        session_id=session.id,
        message_id=emperor_msg.id
    )


@app.post("/api/sessions", response_model=SessionInfo)
async def create_session(user_id: str = Depends(get_current_user_id)):
    user = db.get_or_create_user(user_id)
    session = Session(user_id=user.id)
    db.create_session(session)
    return SessionInfo(id=session.id, created_at=session.created_at, message_count=0)


@app.get("/api/sessions", response_model=list[SessionInfo])
async def list_sessions(user_id: str = Depends(get_current_user_id)):
    user = db.get_or_create_user(user_id)
    rows = db.get_user_sessions_with_counts(user.id)
    return [
        SessionInfo(
            id=row["id"],
            created_at=row["created_at"],
            message_count=row["message_count"],
        )
        for row in rows
    ]


@app.get("/api/sessions/{session_id}/messages", response_model=list[MessageInfo])
async def get_session_messages(session_id: str):
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    messages = db.get_session_messages(session_id)
    return [
        MessageInfo(id=m.id, role=m.role, content=m.content, created_at=m.created_at)
        for m in messages
    ]


@app.get("/api/profile", response_model=Optional[ProfileInfo])
async def get_profile(user_id: str = Depends(get_current_user_id)):
    user = db.get_or_create_user(user_id)
    profile = db.get_latest_profile(user.id)
    if not profile:
        return None

    consensus_reached = None
    stability_score = None
    if profile.get("consensus_log"):
        log = profile["consensus_log"]
        consensus_reached = log.get("consensus_reached")
        stability_score = log.get("stability_score")

    created_at = profile["created_at"]
    if isinstance(created_at, str):
        created_at = datetime.fromisoformat(created_at)

    return ProfileInfo(
        version=profile["version"],
        content=profile["content"],
        created_at=created_at,
        consensus_reached=consensus_reached,
        stability_score=stability_score
    )


@app.get("/api/analysis/status", response_model=AnalysisStatus)
async def get_analysis_status(user_id: str = Depends(get_current_user_id)):
    user = db.get_or_create_user(user_id)
    uncondensed = condensation.get_uncondensed_messages(user.id)
    uncondensed_tokens = sum(condensation.estimate_tokens(m.content) for m in uncondensed)
    summaries = db.get_condensed_summaries(user.id)
    profile = db.get_latest_profile(user.id)

    return AnalysisStatus(
        uncondensed_tokens=uncondensed_tokens,
        condensation_threshold=condensation.chunk_threshold_tokens,
        summary_count=len(summaries),
        has_profile=profile is not None
    )


def _maybe_condense_and_analyze(user_id: str) -> None:
    try:
        did_condense = condensation.maybe_condense(user_id, verbose=False)
        if did_condense:
            _maybe_update_profile(user_id)
    except Exception:
        pass


def _maybe_update_profile(user_id: str) -> None:
    min_summaries = config.get("aegean_consensus", {}).get("min_summaries_for_profile", 3)
    summaries = db.get_condensed_summaries(user_id)
    if len(summaries) < min_summaries:
        return

    profile = db.get_latest_profile(user_id)
    if profile:
        from datetime import datetime
        last_profile_time = profile["created_at"]
        if isinstance(last_profile_time, str):
            last_profile_time = datetime.fromisoformat(last_profile_time)
        new_summaries = [s for s in summaries if s.created_at > last_profile_time]
        if len(new_summaries) < 2:
            return

    try:
        from src.cli.analyze import main as run_analysis_main
        run_analysis_main(user_id=user_id, force=True, show=False)
    except Exception:
        pass


def _retrieve_context(user_message: str, user_id: str) -> dict:
    context = {"stoic": [], "psych": [], "insights": [], "episodic": []}

    try:
        expanded = brain.expand_query(user_message)
        query_terms = [t.strip() for t in expanded.split(",")]
        query_text = " ".join(query_terms) if query_terms else user_message
    except Exception:
        query_text = user_message

    try:
        stoic_results = vectors.query("stoic_wisdom", query_texts=[query_text], n_results=3)
        if stoic_results.get("documents") and stoic_results["documents"][0]:
            context["stoic"] = stoic_results["documents"][0]
    except Exception:
        pass

    try:
        psych_results = vectors.query("psychoanalysis", query_texts=[query_text], n_results=3)
        if psych_results.get("documents") and psych_results["documents"][0]:
            context["psych"] = psych_results["documents"][0]
    except Exception:
        pass

    try:
        insight_results = vectors.query(
            "semantic",
            query_texts=[query_text],
            n_results=5,
            where={"user_id": user_id}
        )
        if insight_results.get("documents") and insight_results["documents"][0]:
            context["insights"] = insight_results["documents"][0]
    except Exception:
        pass

    return context
