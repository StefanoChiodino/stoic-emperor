import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
    force=True,
)
logger = logging.getLogger("stoic_emperor")
logger.setLevel(logging.DEBUG)
print("=== PYTHON PROCESS STARTED ===", file=sys.stderr, flush=True)
logger.info("Python started")  # pragma: no cover

from src.utils.privacy import disable_telemetry

disable_telemetry()

from src.utils.config import load_env

load_env()

import asyncio
import os
from datetime import datetime
from pathlib import Path

logger.info("Starting imports...")  # pragma: no cover

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from src.models.schemas import Message, SemanticInsight, Session
from src.utils.auth import security
from src.utils.config import load_config

logger.info("Imports complete, creating app...")  # pragma: no cover

from contextlib import asynccontextmanager


def _check_env_vars():  # pragma: no cover
    required = ["DATABASE_URL"]
    recommended = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]

    for var in required:
        value = os.getenv(var)
        if value is None:
            logger.error(f"ENV VAR {var} is NOT SET (None)")
        elif value == "":
            logger.error(f"ENV VAR {var} is SET BUT EMPTY")
        else:
            logger.info(f"ENV VAR {var} is set (length={len(value)}, starts={value[:20]}...)")

    missing_recommended = [v for v in recommended if not os.getenv(v)]
    if missing_recommended:
        logger.warning(f"Missing recommended env vars: {missing_recommended}")

    port = os.getenv("PORT", "not set")
    logger.info(f"PORT={port}, ENVIRONMENT={os.getenv('ENVIRONMENT', 'not set')}")


@asynccontextmanager
async def lifespan(app):  # pragma: no cover
    print("=== LIFESPAN START ===", file=sys.stderr, flush=True)
    logger.info("FastAPI lifespan starting")
    _check_env_vars()
    yield
    print("=== LIFESPAN END ===", file=sys.stderr, flush=True)
    logger.info("FastAPI shutdown")


app = FastAPI(title="Stoic Emperor", docs_url="/api/docs", lifespan=lifespan)

logger.info("App created")  # pragma: no cover

_state = {"initialized": False, "config": {}, "db": None, "vectors": None, "brain": None, "condensation": None}


def _init():  # pragma: no cover
    if _state["initialized"]:
        return
    from src.core.emperor_brain import EmperorBrain
    from src.infrastructure.database import Database
    from src.infrastructure.vector_store import VectorStore
    from src.memory.condensation import CondensationManager
    from src.memory.episodic import EpisodicMemory

    _state["config"] = load_config()
    _state["db"] = Database(_state["config"]["database"]["url"])
    _state["vectors"] = VectorStore(_state["config"]["database"]["url"])
    _state["brain"] = EmperorBrain(config=_state["config"])
    _state["condensation"] = CondensationManager(_state["db"], _state["config"])
    _state["episodic"] = EpisodicMemory(
        _state["db"], _state["vectors"], _state["config"]["memory"]["max_context_tokens"]
    )
    _state["initialized"] = True


@app.get("/health")
async def health():
    logger.info("Health check called")
    return {"status": "ok", "port": os.getenv("PORT", "not set")}


ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
DEFAULT_USER_ID = "default_user"
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")


def get_current_user_id(credentials: HTTPAuthorizationCredentials | None = Depends(security)) -> str:
    if not credentials:
        raise HTTPException(status_code=401, detail="Authentication required")

    from src.utils.auth import get_user_id_from_token

    return get_user_id_from_token(credentials)


static_path = Path(__file__).parent / "static"
templates_path = Path(__file__).parent / "templates"
app.mount("/static", StaticFiles(directory=static_path), name="static")
templates = Jinja2Templates(directory=templates_path)


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


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
    consensus_reached: bool | None = None
    stability_score: float | None = None


class AnalysisStatus(BaseModel):
    uncondensed_tokens: int
    condensation_threshold: int
    summary_count: int
    has_profile: bool


class UserInfo(BaseModel):
    id: str
    name: str | None = None
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


@app.get("/terms")
async def terms(request: Request):
    return templates.TemplateResponse("terms.html", {"request": request})


@app.get("/api/config")
async def get_config():
    return {"supabase_url": SUPABASE_URL, "supabase_anon_key": SUPABASE_ANON_KEY, "environment": ENVIRONMENT}


@app.get("/api/user", response_model=UserInfo)
async def get_user(user_id: str = Depends(get_current_user_id)):
    _init()
    user = _state["db"].get_or_create_user(user_id)
    return UserInfo(id=user.id, name=user.name, created_at=user.created_at)


@app.put("/api/user/name", response_model=UserInfo)
async def update_user_name(request: UpdateNameRequest, user_id: str = Depends(get_current_user_id)):
    _init()
    user = _state["db"].update_user_name(user_id, request.name)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return UserInfo(id=user.id, name=user.name, created_at=user.created_at)


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, user_id: str = Depends(get_current_user_id)):
    _init()
    db = _state["db"]
    brain = _state["brain"]
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
    history_with_user = history + [user_msg]

    retrieved_context = _retrieve_context(request.message, user.id)
    profile = db.get_latest_profile(user.id)
    if profile:
        retrieved_context["profile"] = profile["content"]

    summaries = _state["condensation"].get_context_summaries(user.id, token_budget=2000)
    if summaries:
        retrieved_context["narrative"] = "\n\n".join(s.content for s in summaries)

    response = await asyncio.to_thread(
        brain.respond,
        user_message=request.message,
        conversation_history=history_with_user,
        retrieved_context=retrieved_context,
    )

    db.save_message(user_msg)
    emperor_msg = Message(
        session_id=session.id, role="emperor", content=response.response_text, psych_update=response.psych_update
    )
    db.save_message(emperor_msg)

    try:
        _state["episodic"].store_turn(
            user_id=user.id,
            session_id=session.id,
            user_message=request.message,
            emperor_response=response.response_text,
        )
    except Exception:
        pass

    if response.psych_update.semantic_assertions:
        vectors = _state["vectors"]
        for assertion in response.psych_update.semantic_assertions:
            if assertion.confidence >= 0.5:
                try:
                    insight = SemanticInsight(
                        user_id=user.id,
                        source_message_id=emperor_msg.id,
                        assertion=assertion.text,
                        confidence=assertion.confidence,
                    )
                    db.save_semantic_insight(insight)
                    vectors.add(
                        collection="semantic",
                        ids=[insight.id],
                        documents=[assertion.text],
                        metadatas=[
                            {
                                "user_id": user.id,
                                "source_message_id": emperor_msg.id,
                                "confidence": assertion.confidence,
                            }
                        ],
                    )
                except Exception:
                    pass

    _maybe_condense_and_analyze(user.id)

    return ChatResponse(response=response.response_text, session_id=session.id, message_id=emperor_msg.id)


@app.post("/api/sessions", response_model=SessionInfo)
async def create_session(user_id: str = Depends(get_current_user_id)):
    _init()
    db = _state["db"]
    user = db.get_or_create_user(user_id)
    session = Session(user_id=user.id)
    db.create_session(session)
    return SessionInfo(id=session.id, created_at=session.created_at, message_count=0)


@app.get("/api/sessions", response_model=list[SessionInfo])
async def list_sessions(user_id: str = Depends(get_current_user_id)):
    _init()
    db = _state["db"]
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
    _init()
    db = _state["db"]
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    messages = db.get_session_messages(session_id)
    return [MessageInfo(id=m.id, role=m.role, content=m.content, created_at=m.created_at) for m in messages]


@app.get("/api/profile", response_model=ProfileInfo | None)
async def get_profile(user_id: str = Depends(get_current_user_id)):
    _init()
    db = _state["db"]
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
        stability_score=stability_score,
    )


@app.get("/api/analysis/status", response_model=AnalysisStatus)
async def get_analysis_status(user_id: str = Depends(get_current_user_id)):
    _init()
    db = _state["db"]
    condensation = _state["condensation"]
    user = db.get_or_create_user(user_id)
    uncondensed = condensation.get_uncondensed_messages(user.id)
    uncondensed_tokens = sum(condensation.estimate_tokens(m.content) for m in uncondensed)
    summaries = db.get_condensed_summaries(user.id)
    profile = db.get_latest_profile(user.id)

    return AnalysisStatus(
        uncondensed_tokens=uncondensed_tokens,
        condensation_threshold=condensation.chunk_threshold_tokens,
        summary_count=len(summaries),
        has_profile=profile is not None,
    )


def _maybe_condense_and_analyze(user_id: str) -> None:  # pragma: no cover
    try:
        did_condense = _state["condensation"].maybe_condense(user_id, verbose=False)
        if did_condense:
            _maybe_update_profile(user_id)
    except Exception:
        pass


def _maybe_update_profile(user_id: str) -> None:  # pragma: no cover
    config = _state["config"]
    db = _state["db"]
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


def _retrieve_context(user_message: str, user_id: str) -> dict:  # pragma: no cover
    brain = _state["brain"]
    vectors = _state["vectors"]
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
        insight_results = vectors.query("semantic", query_texts=[query_text], n_results=5, where={"user_id": user_id})
        if insight_results.get("documents") and insight_results["documents"][0]:
            context["insights"] = insight_results["documents"][0]
    except Exception:
        pass

    try:
        episodic_results = vectors.query("episodic", query_texts=[query_text], n_results=3, where={"user_id": user_id})
        if episodic_results.get("documents") and episodic_results["documents"][0]:
            context["episodic"] = episodic_results["documents"][0]
    except Exception:
        pass

    return context
