from src.utils.privacy import disable_telemetry
disable_telemetry()

import os
from pathlib import Path
from typing import Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import BaseModel

from src.core.emperor_brain import EmperorBrain
from src.infrastructure.database import Database
from src.infrastructure.vector_store import VectorStore
from src.models.schemas import Session, Message, User
from src.utils.config import load_config
from src.utils.auth import get_user_id_from_token, optional_auth, security

app = FastAPI(title="Stoic Emperor", docs_url="/api/docs")

config = load_config()
db = Database(config["database"]["url"])
vectors = VectorStore(config["database"]["url"])
brain = EmperorBrain(config=config)

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
app.mount("/static", StaticFiles(directory=static_path), name="static")


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
    sessions_since_analysis: int
    threshold: int
    can_analyze: bool
    has_profile: bool


@app.get("/")
async def index():
    return FileResponse(static_path / "index.html")


@app.get("/api/config")
async def get_config():
    return {
        "supabase_url": SUPABASE_URL,
        "supabase_anon_key": SUPABASE_ANON_KEY,
        "environment": ENVIRONMENT
    }


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
    with db._connection() as conn:
        rows = conn.execute(
            """SELECT s.id, s.created_at, COUNT(m.id) as message_count
               FROM sessions s
               LEFT JOIN messages m ON s.id = m.session_id
               WHERE s.user_id = ?
               GROUP BY s.id
               ORDER BY s.created_at DESC""",
            (user.id,)
        ).fetchall()
        return [
            SessionInfo(
                id=row["id"],
                created_at=datetime.fromisoformat(row["created_at"]),
                message_count=row["message_count"]
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
    
    return ProfileInfo(
        version=profile["version"],
        content=profile["content"],
        created_at=datetime.fromisoformat(profile["created_at"]),
        consensus_reached=consensus_reached,
        stability_score=stability_score
    )


@app.get("/api/analysis/status", response_model=AnalysisStatus)
async def get_analysis_status(user_id: str = Depends(get_current_user_id)):
    user = db.get_or_create_user(user_id)
    sessions_since = db.count_sessions_since_last_analysis(user.id)
    threshold = config.get("aegean_consensus", {}).get("sessions_between_analysis", 5)
    profile = db.get_latest_profile(user.id)
    
    return AnalysisStatus(
        sessions_since_analysis=sessions_since,
        threshold=threshold,
        can_analyze=sessions_since >= threshold,
        has_profile=profile is not None
    )


@app.post("/api/analyze", response_model=ProfileInfo)
async def run_analysis(user_id: str = Depends(get_current_user_id)):
    from src.cli.analyze import main as run_analysis_main
    user = db.get_or_create_user(user_id)
    
    run_analysis_main(user_id=user.id, force=True, show=False)
    
    profile = db.get_latest_profile(user.id)
    if not profile:
        raise HTTPException(status_code=500, detail="Analysis failed to generate profile")
    
    consensus_reached = None
    stability_score = None
    if profile.get("consensus_log"):
        log = profile["consensus_log"]
        consensus_reached = log.get("consensus_reached")
        stability_score = log.get("stability_score")
    
    return ProfileInfo(
        version=profile["version"],
        content=profile["content"],
        created_at=datetime.fromisoformat(profile["created_at"]),
        consensus_reached=consensus_reached,
        stability_score=stability_score
    )


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
