from src.utils.privacy import disable_telemetry
disable_telemetry()

from pathlib import Path
from typing import Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.core.emperor_brain import EmperorBrain
from src.infrastructure.database import Database
from src.infrastructure.vector_store import VectorStore
from src.models.schemas import Session, Message
from src.utils.config import load_config

app = FastAPI(title="Stoic Emperor", docs_url="/api/docs")

config = load_config()
db = Database(config["paths"]["sqlite_db"])
vectors = VectorStore(config["paths"]["vector_db"])
brain = EmperorBrain(config=config)

DEFAULT_USER_ID = "default_user"

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


@app.get("/")
async def index():
    return FileResponse(static_path / "index.html")


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    user = db.get_or_create_user(DEFAULT_USER_ID)

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
async def create_session():
    user = db.get_or_create_user(DEFAULT_USER_ID)
    session = Session(user_id=user.id)
    db.create_session(session)
    return SessionInfo(id=session.id, created_at=session.created_at, message_count=0)


@app.get("/api/sessions", response_model=list[SessionInfo])
async def list_sessions():
    user = db.get_or_create_user(DEFAULT_USER_ID)
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
