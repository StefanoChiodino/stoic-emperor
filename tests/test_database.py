import pytest
from datetime import datetime

from src.infrastructure.database import Database
from src.models.schemas import User, Session, Message, SemanticInsight, PsychUpdate


class TestDatabaseInitialization:
    def test_database_creates_tables(self, test_db_path):
        db = Database(test_db_path)
        
        with db._connection() as conn:
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            table_names = [t["name"] for t in tables]
        
        assert "users" in table_names
        assert "sessions" in table_names
        assert "messages" in table_names
        assert "profiles" in table_names
        assert "semantic_insights" in table_names
        assert "schema_version" in table_names

    def test_schema_version_recorded(self, test_db_path):
        db = Database(test_db_path)
        
        with db._connection() as conn:
            version = conn.execute(
                "SELECT MAX(version) as v FROM schema_version"
            ).fetchone()
        
        assert version["v"] >= 1


class TestUserOperations:
    def test_create_user(self, test_db_path):
        db = Database(test_db_path)
        user = User(id="test_user_1")
        
        created = db.create_user(user)
        
        assert created.id == "test_user_1"

    def test_get_user(self, test_db_path):
        db = Database(test_db_path)
        user = User(id="test_user_2")
        db.create_user(user)
        
        retrieved = db.get_user("test_user_2")
        
        assert retrieved is not None
        assert retrieved.id == "test_user_2"

    def test_get_nonexistent_user(self, test_db_path):
        db = Database(test_db_path)
        
        retrieved = db.get_user("nonexistent")
        
        assert retrieved is None

    def test_get_or_create_user_creates(self, test_db_path):
        db = Database(test_db_path)
        
        user = db.get_or_create_user("new_user")
        
        assert user.id == "new_user"

    def test_get_or_create_user_gets_existing(self, test_db_path):
        db = Database(test_db_path)
        original = db.get_or_create_user("existing_user")
        
        retrieved = db.get_or_create_user("existing_user")
        
        assert retrieved.id == original.id


class TestSessionOperations:
    def test_create_session(self, test_db_path):
        db = Database(test_db_path)
        db.create_user(User(id="user_1"))
        session = Session(user_id="user_1", metadata={"source": "test"})
        
        created = db.create_session(session)
        
        assert created.user_id == "user_1"
        assert created.metadata["source"] == "test"

    def test_get_session(self, test_db_path):
        db = Database(test_db_path)
        db.create_user(User(id="user_2"))
        session = Session(id="session_1", user_id="user_2")
        db.create_session(session)
        
        retrieved = db.get_session("session_1")
        
        assert retrieved is not None
        assert retrieved.user_id == "user_2"

    def test_get_latest_session(self, test_db_path):
        db = Database(test_db_path)
        db.create_user(User(id="user_3"))
        
        session1 = Session(user_id="user_3")
        session2 = Session(user_id="user_3")
        db.create_session(session1)
        db.create_session(session2)
        
        latest = db.get_latest_session("user_3")
        
        assert latest is not None
        assert latest.id == session2.id


class TestMessageOperations:
    def test_save_message(self, test_db_path, sample_psych_update):
        db = Database(test_db_path)
        db.create_user(User(id="user_m"))
        session = Session(id="session_m", user_id="user_m")
        db.create_session(session)
        
        msg = Message(
            session_id="session_m",
            role="emperor",
            content="Test response",
            psych_update=sample_psych_update
        )
        
        saved = db.save_message(msg)
        
        assert saved.id == msg.id

    def test_get_session_messages(self, test_db_path):
        db = Database(test_db_path)
        db.create_user(User(id="user_msgs"))
        session = Session(id="session_msgs", user_id="user_msgs")
        db.create_session(session)
        
        db.save_message(Message(session_id="session_msgs", role="user", content="Hello"))
        db.save_message(Message(session_id="session_msgs", role="emperor", content="Greetings"))
        
        messages = db.get_session_messages("session_msgs")
        
        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[1].role == "emperor"

    def test_get_unprocessed_messages(self, test_db_path, sample_psych_update):
        db = Database(test_db_path)
        db.create_user(User(id="user_unproc"))
        session = Session(id="session_unproc", user_id="user_unproc")
        db.create_session(session)
        
        db.save_message(Message(
            session_id="session_unproc",
            role="emperor",
            content="Response",
            psych_update=sample_psych_update
        ))
        
        unprocessed = db.get_unprocessed_messages("user_unproc")
        
        assert len(unprocessed) == 1

    def test_mark_message_processed(self, test_db_path, sample_psych_update):
        db = Database(test_db_path)
        db.create_user(User(id="user_proc"))
        session = Session(id="session_proc", user_id="user_proc")
        db.create_session(session)
        
        msg = Message(
            session_id="session_proc",
            role="emperor",
            content="Response",
            psych_update=sample_psych_update
        )
        db.save_message(msg)
        
        db.mark_message_processed(msg.id)
        
        unprocessed = db.get_unprocessed_messages("user_proc")
        assert len(unprocessed) == 0


class TestSemanticInsightOperations:
    def test_save_semantic_insight(self, test_db_path):
        db = Database(test_db_path)
        db.create_user(User(id="user_insight"))
        
        insight = SemanticInsight(
            user_id="user_insight",
            source_message_id="msg_1",
            assertion="User struggles with father relationship",
            confidence=0.85
        )
        
        saved = db.save_semantic_insight(insight)
        
        assert saved.assertion == "User struggles with father relationship"

    def test_get_user_insights(self, test_db_path):
        db = Database(test_db_path)
        db.create_user(User(id="user_insights"))
        
        db.save_semantic_insight(SemanticInsight(
            user_id="user_insights",
            source_message_id="msg_1",
            assertion="Insight 1",
            confidence=0.8
        ))
        db.save_semantic_insight(SemanticInsight(
            user_id="user_insights",
            source_message_id="msg_2",
            assertion="Insight 2",
            confidence=0.9
        ))
        
        insights = db.get_user_insights("user_insights")
        
        assert len(insights) == 2


class TestProfileTracking:
    def test_count_sessions_no_profile(self, test_db_path):
        db = Database(test_db_path)
        db.create_user(User(id="user_count"))
        
        for i in range(3):
            db.create_session(Session(user_id="user_count"))
        
        count = db.count_sessions_since_last_analysis("user_count")
        
        assert count == 3
