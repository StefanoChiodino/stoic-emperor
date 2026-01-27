import json
import os
import sqlite3
from datetime import datetime
from typing import Optional, List, Dict, Any
from contextlib import contextmanager
from urllib.parse import urlparse
from pathlib import Path

from src.models.schemas import User, Session, Message, SemanticInsight, PsychUpdate, CondensedSummary

SCHEMA_VERSION = 3

SQLITE_MIGRATIONS = {
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
    """,
    2: """
        CREATE TABLE IF NOT EXISTS condensed_summaries (
            id TEXT PRIMARY KEY,
            user_id TEXT REFERENCES users(id),
            level INTEGER,
            content TEXT,
            period_start TEXT,
            period_end TEXT,
            source_message_count INTEGER,
            source_word_count INTEGER,
            source_summary_ids TEXT,
            consensus_log TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_summaries_user_level ON condensed_summaries(user_id, level);
        CREATE INDEX IF NOT EXISTS idx_summaries_period ON condensed_summaries(user_id, period_end);
    """,
    3: """
        ALTER TABLE users ADD COLUMN email TEXT UNIQUE;
        ALTER TABLE users ADD COLUMN password_hash TEXT;
        CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
    """
}

POSTGRES_MIGRATIONS = {
    1: """
        CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER PRIMARY KEY,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            user_id TEXT REFERENCES users(id),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata JSONB
        );

        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            session_id TEXT REFERENCES sessions(id),
            role TEXT,
            content TEXT,
            psych_update JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            semantic_processed_at TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS profiles (
            id TEXT PRIMARY KEY,
            user_id TEXT REFERENCES users(id),
            version INTEGER,
            content TEXT,
            consensus_log JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS semantic_insights (
            id TEXT PRIMARY KEY,
            user_id TEXT REFERENCES users(id),
            source_message_id TEXT REFERENCES messages(id),
            assertion TEXT,
            confidence REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
        CREATE INDEX IF NOT EXISTS idx_messages_semantic ON messages(semantic_processed_at);
        CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
        CREATE INDEX IF NOT EXISTS idx_insights_user ON semantic_insights(user_id);
    """,
    2: """
        CREATE TABLE IF NOT EXISTS condensed_summaries (
            id TEXT PRIMARY KEY,
            user_id TEXT REFERENCES users(id),
            level INTEGER,
            content TEXT,
            period_start TIMESTAMP,
            period_end TIMESTAMP,
            source_message_count INTEGER,
            source_word_count INTEGER,
            source_summary_ids JSONB,
            consensus_log JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_summaries_user_level ON condensed_summaries(user_id, level);
        CREATE INDEX IF NOT EXISTS idx_summaries_period ON condensed_summaries(user_id, period_end);
    """,
    3: """
        ALTER TABLE users ADD COLUMN email TEXT UNIQUE;
        ALTER TABLE users ADD COLUMN password_hash TEXT;
        CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
    """
}


class Database:
    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or os.getenv("DATABASE_URL", "sqlite:///./data/stoic_emperor.db")
        
        parsed = urlparse(self.database_url)
        self.is_postgres = parsed.scheme in ("postgresql", "postgres")
        
        if self.is_postgres:
            import psycopg2
            from psycopg2 import pool as pg_pool
            self._psycopg2 = psycopg2
            self._pool = pg_pool.ThreadedConnectionPool(
                minconn=2,
                maxconn=10,
                dsn=self.database_url
            )
        else:
            db_path = self.database_url.replace("sqlite:///", "")
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            self._sqlite_path = db_path
        
        self._run_migrations()

    @contextmanager
    def _connection(self):
        if self.is_postgres:
            from psycopg2.extras import RealDictCursor
            conn = self._pool.getconn()
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                self._pool.putconn(conn)
        else:
            conn = sqlite3.connect(self._sqlite_path)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            finally:
                conn.close()

    def _run_migrations(self) -> None:
        migrations = POSTGRES_MIGRATIONS if self.is_postgres else SQLITE_MIGRATIONS
        
        with self._connection() as conn:
            if self.is_postgres:
                from psycopg2.extras import RealDictCursor
                cursor = conn.cursor(cursor_factory=RealDictCursor)
                cursor.execute(
                    "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name='schema_version')"
                )
                table_exists = cursor.fetchone()["exists"]
            else:
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'")
                table_exists = cursor.fetchone() is not None
            
            if not table_exists:
                current_version = 0
            else:
                cursor.execute("SELECT MAX(version) as max FROM schema_version")
                result = cursor.fetchone()
                if self.is_postgres:
                    current_version = result["max"] if result and result["max"] else 0
                else:
                    current_version = result[0] if result and result[0] else 0

            for version in range(current_version + 1, SCHEMA_VERSION + 1):
                if version in migrations:
                    if self.is_postgres:
                        cursor.execute(migrations[version])
                        cursor.execute("INSERT INTO schema_version (version) VALUES (%s)", (version,))
                    else:
                        cursor.executescript(migrations[version])
                        cursor.execute("INSERT INTO schema_version (version) VALUES (?)", (version,))
            
            cursor.close()

    def _placeholder(self) -> str:
        return "%s" if self.is_postgres else "?"

    def _get_cursor(self, conn, dict_cursor: bool = False):
        if self.is_postgres and dict_cursor:
            from psycopg2.extras import RealDictCursor
            return conn.cursor(cursor_factory=RealDictCursor)
        return conn.cursor()

    def _row_to_dict(self, row) -> Optional[Dict[str, Any]]:
        if row is None:
            return None
        if self.is_postgres:
            return dict(row)
        return dict(row)

    def _parse_timestamp(self, value) -> Optional[datetime]:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        return datetime.fromisoformat(value)

    def create_user(self, user: User) -> User:
        ph = self._placeholder()
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"INSERT INTO users (id, email, password_hash, created_at) VALUES ({ph}, {ph}, {ph}, {ph})",
                (user.id, user.email, user.password_hash,
                 user.created_at.isoformat() if not self.is_postgres else user.created_at)
            )
            cursor.close()
        return user

    def get_user(self, user_id: str) -> Optional[User]:
        ph = self._placeholder()
        with self._connection() as conn:
            cursor = self._get_cursor(conn, dict_cursor=True)
            cursor.execute(f"SELECT * FROM users WHERE id = {ph}", (user_id,))
            row = cursor.fetchone()
            cursor.close()
            if row:
                return User(
                    id=row["id"],
                    email=row.get("email"),
                    password_hash=row.get("password_hash"),
                    created_at=self._parse_timestamp(row["created_at"])
                )
        return None

    def get_user_by_email(self, email: str) -> Optional[User]:
        ph = self._placeholder()
        with self._connection() as conn:
            cursor = self._get_cursor(conn, dict_cursor=True)
            cursor.execute(f"SELECT * FROM users WHERE email = {ph}", (email,))
            row = cursor.fetchone()
            cursor.close()
            if row:
                return User(
                    id=row["id"],
                    email=row.get("email"),
                    password_hash=row.get("password_hash"),
                    created_at=self._parse_timestamp(row["created_at"])
                )
        return None

    def get_or_create_user(self, user_id: str) -> User:
        user = self.get_user(user_id)
        if not user:
            user = User(id=user_id)
            self.create_user(user)
        return user

    def create_session(self, session: Session) -> Session:
        ph = self._placeholder()
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"INSERT INTO sessions (id, user_id, created_at, metadata) VALUES ({ph}, {ph}, {ph}, {ph})",
                (session.id, session.user_id, 
                 session.created_at.isoformat() if not self.is_postgres else session.created_at,
                 json.dumps(session.metadata))
            )
            cursor.close()
        return session

    def _parse_metadata(self, value) -> dict:
        if value is None:
            return {}
        if isinstance(value, dict):
            return value
        return json.loads(value)

    def get_session(self, session_id: str) -> Optional[Session]:
        ph = self._placeholder()
        with self._connection() as conn:
            cursor = self._get_cursor(conn, dict_cursor=True)
            cursor.execute(f"SELECT * FROM sessions WHERE id = {ph}", (session_id,))
            row = cursor.fetchone()
            cursor.close()
            if row:
                return Session(
                    id=row["id"],
                    user_id=row["user_id"],
                    created_at=self._parse_timestamp(row["created_at"]),
                    metadata=self._parse_metadata(row["metadata"])
                )
        return None

    def get_latest_session(self, user_id: str) -> Optional[Session]:
        ph = self._placeholder()
        with self._connection() as conn:
            cursor = self._get_cursor(conn, dict_cursor=True)
            cursor.execute(
                f"SELECT * FROM sessions WHERE user_id = {ph} ORDER BY created_at DESC LIMIT 1",
                (user_id,)
            )
            row = cursor.fetchone()
            cursor.close()
            if row:
                return Session(
                    id=row["id"],
                    user_id=row["user_id"],
                    created_at=self._parse_timestamp(row["created_at"]),
                    metadata=self._parse_metadata(row["metadata"])
                )
        return None

    def save_message(self, message: Message) -> Message:
        ph = self._placeholder()
        with self._connection() as conn:
            cursor = conn.cursor()
            psych_dict = message.psych_update.model_dump() if message.psych_update else None
            created = message.created_at.isoformat() if not self.is_postgres else message.created_at
            semantic = message.semantic_processed_at.isoformat() if message.semantic_processed_at and not self.is_postgres else message.semantic_processed_at
            cursor.execute(
                f"""INSERT INTO messages 
                   (id, session_id, role, content, psych_update, created_at, semantic_processed_at) 
                   VALUES ({ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph})""",
                (message.id, message.session_id, message.role, message.content,
                 json.dumps(psych_dict) if psych_dict else None, created, semantic)
            )
            cursor.close()
        return message

    def get_session_messages(self, session_id: str) -> List[Message]:
        ph = self._placeholder()
        with self._connection() as conn:
            cursor = self._get_cursor(conn, dict_cursor=True)
            cursor.execute(
                f"SELECT * FROM messages WHERE session_id = {ph} ORDER BY created_at",
                (session_id,)
            )
            rows = cursor.fetchall()
            cursor.close()
            return [self._row_to_message(row) for row in rows]

    def get_unprocessed_messages(self, user_id: str) -> List[Message]:
        ph = self._placeholder()
        with self._connection() as conn:
            cursor = self._get_cursor(conn, dict_cursor=True)
            cursor.execute(
                f"""SELECT m.* FROM messages m
                   JOIN sessions s ON m.session_id = s.id
                   WHERE s.user_id = {ph} AND m.semantic_processed_at IS NULL AND m.psych_update IS NOT NULL
                   ORDER BY m.created_at""",
                (user_id,)
            )
            rows = cursor.fetchall()
            cursor.close()
            return [self._row_to_message(row) for row in rows]

    def mark_message_processed(self, message_id: str) -> None:
        ph = self._placeholder()
        now = datetime.now().isoformat() if not self.is_postgres else datetime.now()
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"UPDATE messages SET semantic_processed_at = {ph} WHERE id = {ph}",
                (now, message_id)
            )
            cursor.close()

    def save_semantic_insight(self, insight: SemanticInsight) -> SemanticInsight:
        ph = self._placeholder()
        with self._connection() as conn:
            cursor = conn.cursor()
            created = insight.created_at.isoformat() if not self.is_postgres else insight.created_at
            cursor.execute(
                f"""INSERT INTO semantic_insights 
                   (id, user_id, source_message_id, assertion, confidence, created_at) 
                   VALUES ({ph}, {ph}, {ph}, {ph}, {ph}, {ph})""",
                (insight.id, insight.user_id, insight.source_message_id,
                 insight.assertion, insight.confidence, created)
            )
            cursor.close()
        return insight

    def get_user_insights(self, user_id: str) -> List[SemanticInsight]:
        ph = self._placeholder()
        with self._connection() as conn:
            cursor = self._get_cursor(conn, dict_cursor=True)
            cursor.execute(
                f"SELECT * FROM semantic_insights WHERE user_id = {ph} ORDER BY created_at DESC",
                (user_id,)
            )
            rows = cursor.fetchall()
            cursor.close()
            return [
                SemanticInsight(
                    id=row["id"],
                    user_id=row["user_id"],
                    source_message_id=row["source_message_id"],
                    assertion=row["assertion"],
                    confidence=row["confidence"],
                    created_at=self._parse_timestamp(row["created_at"])
                )
                for row in rows
            ]

    def count_sessions_since_last_analysis(self, user_id: str) -> int:
        ph = self._placeholder()
        with self._connection() as conn:
            cursor = self._get_cursor(conn, dict_cursor=True)
            cursor.execute(
                f"SELECT MAX(created_at) as last FROM profiles WHERE user_id = {ph}",
                (user_id,)
            )
            last_profile = cursor.fetchone()
            
            if last_profile and last_profile["last"]:
                cursor.execute(
                    f"SELECT COUNT(*) as cnt FROM sessions WHERE user_id = {ph} AND created_at > {ph}",
                    (user_id, last_profile["last"])
                )
                count = cursor.fetchone()
            else:
                cursor.execute(
                    f"SELECT COUNT(*) as cnt FROM sessions WHERE user_id = {ph}",
                    (user_id,)
                )
                count = cursor.fetchone()
            
            cursor.close()
            return count["cnt"] if count else 0

    def get_latest_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        ph = self._placeholder()
        with self._connection() as conn:
            cursor = self._get_cursor(conn, dict_cursor=True)
            cursor.execute(
                f"""SELECT content, version, created_at, consensus_log 
                   FROM profiles WHERE user_id = {ph} ORDER BY version DESC LIMIT 1""",
                (user_id,)
            )
            row = cursor.fetchone()
            cursor.close()
            if row:
                return {
                    "content": row["content"],
                    "version": row["version"],
                    "created_at": row["created_at"],
                    "consensus_log": self._parse_metadata(row["consensus_log"])
                }
        return None

    def _row_to_message(self, row) -> Message:
        psych = None
        psych_data = row["psych_update"]
        if psych_data:
            if isinstance(psych_data, str):
                psych_data = json.loads(psych_data)
            psych = PsychUpdate(**psych_data)
        return Message(
            id=row["id"],
            session_id=row["session_id"],
            role=row["role"],
            content=row["content"],
            psych_update=psych,
            created_at=self._parse_timestamp(row["created_at"]),
            semantic_processed_at=self._parse_timestamp(row["semantic_processed_at"])
        )

    def save_condensed_summary(self, summary: CondensedSummary) -> CondensedSummary:
        ph = self._placeholder()
        with self._connection() as conn:
            cursor = conn.cursor()
            if self.is_postgres:
                ps, pe, ca = summary.period_start, summary.period_end, summary.created_at
            else:
                ps = summary.period_start.isoformat()
                pe = summary.period_end.isoformat()
                ca = summary.created_at.isoformat()
            cursor.execute(
                f"""INSERT INTO condensed_summaries 
                   (id, user_id, level, content, period_start, period_end, 
                    source_message_count, source_word_count, source_summary_ids, 
                    consensus_log, created_at) 
                   VALUES ({ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph})""",
                (summary.id, summary.user_id, summary.level, summary.content,
                 ps, pe, summary.source_message_count, summary.source_word_count,
                 json.dumps(summary.source_summary_ids),
                 json.dumps(summary.consensus_log) if summary.consensus_log else None, ca)
            )
            cursor.close()
        return summary

    def get_condensed_summaries(self, user_id: str, level: Optional[int] = None) -> List[CondensedSummary]:
        ph = self._placeholder()
        with self._connection() as conn:
            cursor = self._get_cursor(conn, dict_cursor=True)
            if level is not None:
                cursor.execute(
                    f"""SELECT * FROM condensed_summaries 
                       WHERE user_id = {ph} AND level = {ph} 
                       ORDER BY period_start""",
                    (user_id, level)
                )
            else:
                cursor.execute(
                    f"""SELECT * FROM condensed_summaries 
                       WHERE user_id = {ph} 
                       ORDER BY level, period_start""",
                    (user_id,)
                )
            rows = cursor.fetchall()
            cursor.close()
            return [self._row_to_condensed_summary(row) for row in rows]

    def get_messages_in_range(
        self, 
        user_id: str, 
        start_date: Optional[datetime] = None, 
        end_date: Optional[datetime] = None
    ) -> List[Message]:
        ph = self._placeholder()
        with self._connection() as conn:
            cursor = self._get_cursor(conn, dict_cursor=True)
            if not self.is_postgres:
                start_date = start_date.isoformat() if start_date else None
                end_date = end_date.isoformat() if end_date else None
            
            if start_date and end_date:
                cursor.execute(
                    f"""SELECT m.* FROM messages m
                       JOIN sessions s ON m.session_id = s.id
                       WHERE s.user_id = {ph} AND m.created_at >= {ph} AND m.created_at <= {ph}
                       ORDER BY m.created_at""",
                    (user_id, start_date, end_date)
                )
            elif start_date:
                cursor.execute(
                    f"""SELECT m.* FROM messages m
                       JOIN sessions s ON m.session_id = s.id
                       WHERE s.user_id = {ph} AND m.created_at >= {ph}
                       ORDER BY m.created_at""",
                    (user_id, start_date)
                )
            elif end_date:
                cursor.execute(
                    f"""SELECT m.* FROM messages m
                       JOIN sessions s ON m.session_id = s.id
                       WHERE s.user_id = {ph} AND m.created_at <= {ph}
                       ORDER BY m.created_at""",
                    (user_id, end_date)
                )
            else:
                cursor.execute(
                    f"""SELECT m.* FROM messages m
                       JOIN sessions s ON m.session_id = s.id
                       WHERE s.user_id = {ph}
                       ORDER BY m.created_at""",
                    (user_id,)
                )
            rows = cursor.fetchall()
            cursor.close()
            return [self._row_to_message(row) for row in rows]

    def get_recent_messages(self, user_id: str, limit: int = 20) -> List[Message]:
        ph = self._placeholder()
        with self._connection() as conn:
            cursor = self._get_cursor(conn, dict_cursor=True)
            cursor.execute(
                f"""SELECT m.* FROM messages m
                   JOIN sessions s ON m.session_id = s.id
                   WHERE s.user_id = {ph}
                   ORDER BY m.created_at DESC
                   LIMIT {ph}""",
                (user_id, limit)
            )
            rows = cursor.fetchall()
            cursor.close()
            return list(reversed([self._row_to_message(row) for row in rows]))

    def _row_to_condensed_summary(self, row) -> CondensedSummary:
        source_ids = row["source_summary_ids"]
        if isinstance(source_ids, str):
            source_ids = json.loads(source_ids)
        elif source_ids is None:
            source_ids = []
        
        consensus = row["consensus_log"]
        if isinstance(consensus, str):
            consensus = json.loads(consensus)
        
        return CondensedSummary(
            id=row["id"],
            user_id=row["user_id"],
            level=row["level"],
            content=row["content"],
            period_start=self._parse_timestamp(row["period_start"]),
            period_end=self._parse_timestamp(row["period_end"]),
            source_message_count=row["source_message_count"],
            source_word_count=row["source_word_count"],
            source_summary_ids=source_ids,
            consensus_log=consensus,
            created_at=self._parse_timestamp(row["created_at"])
        )
