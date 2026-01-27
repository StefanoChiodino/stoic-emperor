import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
from contextlib import contextmanager

from src.models.schemas import User, Session, Message, SemanticInsight, PsychUpdate

SCHEMA_VERSION = 1

MIGRATIONS = {
    1: """
        CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER PRIMARY KEY,
            applied_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            user_id TEXT REFERENCES users(id),
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            metadata TEXT
        );

        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            session_id TEXT REFERENCES sessions(id),
            role TEXT,
            content TEXT,
            psych_update TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            semantic_processed_at TEXT
        );

        CREATE TABLE IF NOT EXISTS profiles (
            id TEXT PRIMARY KEY,
            user_id TEXT REFERENCES users(id),
            version INTEGER,
            content TEXT,
            consensus_log TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS semantic_insights (
            id TEXT PRIMARY KEY,
            user_id TEXT REFERENCES users(id),
            source_message_id TEXT REFERENCES messages(id),
            assertion TEXT,
            confidence REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
        CREATE INDEX IF NOT EXISTS idx_messages_semantic ON messages(semantic_processed_at);
        CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
        CREATE INDEX IF NOT EXISTS idx_insights_user ON semantic_insights(user_id);
    """
}


class Database:
    def __init__(self, db_path: str = "./data/stoic_emperor.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._run_migrations()

    @contextmanager
    def _connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _run_migrations(self) -> None:
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'")
            if not cursor.fetchone():
                current_version = 0
            else:
                cursor.execute("SELECT MAX(version) FROM schema_version")
                result = cursor.fetchone()
                current_version = result[0] if result[0] else 0

            for version in range(current_version + 1, SCHEMA_VERSION + 1):
                if version in MIGRATIONS:
                    cursor.executescript(MIGRATIONS[version])
                    cursor.execute(
                        "INSERT INTO schema_version (version) VALUES (?)",
                        (version,)
                    )

    def create_user(self, user: User) -> User:
        with self._connection() as conn:
            conn.execute(
                "INSERT INTO users (id, created_at) VALUES (?, ?)",
                (user.id, user.created_at.isoformat())
            )
        return user

    def get_user(self, user_id: str) -> Optional[User]:
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE id = ?", (user_id,)
            ).fetchone()
            if row:
                return User(id=row["id"], created_at=datetime.fromisoformat(row["created_at"]))
        return None

    def get_or_create_user(self, user_id: str) -> User:
        user = self.get_user(user_id)
        if not user:
            user = User(id=user_id)
            self.create_user(user)
        return user

    def create_session(self, session: Session) -> Session:
        with self._connection() as conn:
            conn.execute(
                "INSERT INTO sessions (id, user_id, created_at, metadata) VALUES (?, ?, ?, ?)",
                (session.id, session.user_id, session.created_at.isoformat(), json.dumps(session.metadata))
            )
        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if row:
                return Session(
                    id=row["id"],
                    user_id=row["user_id"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                    metadata=json.loads(row["metadata"]) if row["metadata"] else {}
                )
        return None

    def get_latest_session(self, user_id: str) -> Optional[Session]:
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE user_id = ? ORDER BY created_at DESC LIMIT 1",
                (user_id,)
            ).fetchone()
            if row:
                return Session(
                    id=row["id"],
                    user_id=row["user_id"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                    metadata=json.loads(row["metadata"]) if row["metadata"] else {}
                )
        return None

    def save_message(self, message: Message) -> Message:
        with self._connection() as conn:
            psych_json = message.psych_update.model_dump_json() if message.psych_update else None
            semantic_at = message.semantic_processed_at.isoformat() if message.semantic_processed_at else None
            conn.execute(
                """INSERT INTO messages 
                   (id, session_id, role, content, psych_update, created_at, semantic_processed_at) 
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (message.id, message.session_id, message.role, message.content,
                 psych_json, message.created_at.isoformat(), semantic_at)
            )
        return message

    def get_session_messages(self, session_id: str) -> List[Message]:
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY created_at",
                (session_id,)
            ).fetchall()
            return [self._row_to_message(row) for row in rows]

    def get_unprocessed_messages(self, user_id: str) -> List[Message]:
        with self._connection() as conn:
            rows = conn.execute(
                """SELECT m.* FROM messages m
                   JOIN sessions s ON m.session_id = s.id
                   WHERE s.user_id = ? AND m.semantic_processed_at IS NULL AND m.psych_update IS NOT NULL
                   ORDER BY m.created_at""",
                (user_id,)
            ).fetchall()
            return [self._row_to_message(row) for row in rows]

    def mark_message_processed(self, message_id: str) -> None:
        with self._connection() as conn:
            conn.execute(
                "UPDATE messages SET semantic_processed_at = ? WHERE id = ?",
                (datetime.now().isoformat(), message_id)
            )

    def save_semantic_insight(self, insight: SemanticInsight) -> SemanticInsight:
        with self._connection() as conn:
            conn.execute(
                """INSERT INTO semantic_insights 
                   (id, user_id, source_message_id, assertion, confidence, created_at) 
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (insight.id, insight.user_id, insight.source_message_id,
                 insight.assertion, insight.confidence, insight.created_at.isoformat())
            )
        return insight

    def get_user_insights(self, user_id: str) -> List[SemanticInsight]:
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM semantic_insights WHERE user_id = ? ORDER BY created_at DESC",
                (user_id,)
            ).fetchall()
            return [
                SemanticInsight(
                    id=row["id"],
                    user_id=row["user_id"],
                    source_message_id=row["source_message_id"],
                    assertion=row["assertion"],
                    confidence=row["confidence"],
                    created_at=datetime.fromisoformat(row["created_at"])
                )
                for row in rows
            ]

    def count_sessions_since_last_analysis(self, user_id: str) -> int:
        with self._connection() as conn:
            last_profile = conn.execute(
                "SELECT MAX(created_at) as last FROM profiles WHERE user_id = ?",
                (user_id,)
            ).fetchone()
            
            if last_profile and last_profile["last"]:
                count = conn.execute(
                    "SELECT COUNT(*) as cnt FROM sessions WHERE user_id = ? AND created_at > ?",
                    (user_id, last_profile["last"])
                ).fetchone()
            else:
                count = conn.execute(
                    "SELECT COUNT(*) as cnt FROM sessions WHERE user_id = ?",
                    (user_id,)
                ).fetchone()
            
            return count["cnt"] if count else 0

    def get_latest_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        with self._connection() as conn:
            row = conn.execute(
                """SELECT content, version, created_at, consensus_log 
                   FROM profiles WHERE user_id = ? ORDER BY version DESC LIMIT 1""",
                (user_id,)
            ).fetchone()
            if row:
                return {
                    "content": row["content"],
                    "version": row["version"],
                    "created_at": row["created_at"],
                    "consensus_log": json.loads(row["consensus_log"]) if row["consensus_log"] else None
                }
        return None

    def _row_to_message(self, row: sqlite3.Row) -> Message:
        psych = None
        if row["psych_update"]:
            psych = PsychUpdate(**json.loads(row["psych_update"]))
        semantic_at = datetime.fromisoformat(row["semantic_processed_at"]) if row["semantic_processed_at"] else None
        return Message(
            id=row["id"],
            session_id=row["session_id"],
            role=row["role"],
            content=row["content"],
            psych_update=psych,
            created_at=datetime.fromisoformat(row["created_at"]),
            semantic_processed_at=semantic_at
        )
