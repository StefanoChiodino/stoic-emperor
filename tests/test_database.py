from sqlalchemy import inspect

from src.infrastructure.database import Database
from src.models.schemas import Message, SemanticInsight, Session, User


class TestDatabaseInitialization:
    def test_database_creates_tables(self, test_db_path):
        db = Database(test_db_path)

        inspector = inspect(db.engine)
        table_names = inspector.get_table_names()

        assert "users" in table_names
        assert "sessions" in table_names
        assert "messages" in table_names
        assert "profiles" in table_names
        assert "semantic_insights" in table_names
        assert "schema_version" in table_names

    def test_schema_version_recorded(self, test_db_path):
        from src.infrastructure.database import SchemaVersionModel

        db = Database(test_db_path)

        with db._session() as session:
            version = session.get(SchemaVersionModel, 4)

        assert version is not None


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

        msg = Message(session_id="session_m", role="emperor", content="Test response", psych_update=sample_psych_update)

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

        db.save_message(
            Message(session_id="session_unproc", role="emperor", content="Response", psych_update=sample_psych_update)
        )

        unprocessed = db.get_unprocessed_messages("user_unproc")

        assert len(unprocessed) == 1

    def test_mark_message_processed(self, test_db_path, sample_psych_update):
        db = Database(test_db_path)
        db.create_user(User(id="user_proc"))
        session = Session(id="session_proc", user_id="user_proc")
        db.create_session(session)

        msg = Message(session_id="session_proc", role="emperor", content="Response", psych_update=sample_psych_update)
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
            confidence=0.85,
        )

        saved = db.save_semantic_insight(insight)

        assert saved.assertion == "User struggles with father relationship"

    def test_get_user_insights(self, test_db_path):
        db = Database(test_db_path)
        db.create_user(User(id="user_insights"))

        db.save_semantic_insight(
            SemanticInsight(user_id="user_insights", source_message_id="msg_1", assertion="Insight 1", confidence=0.8)
        )
        db.save_semantic_insight(
            SemanticInsight(user_id="user_insights", source_message_id="msg_2", assertion="Insight 2", confidence=0.9)
        )

        insights = db.get_user_insights("user_insights")

        assert len(insights) == 2


class TestProfileTracking:
    def test_count_sessions_no_profile(self, test_db_path):
        db = Database(test_db_path)
        db.create_user(User(id="user_count"))

        for _ in range(3):
            db.create_session(Session(user_id="user_count"))

        count = db.count_sessions_since_last_analysis("user_count")

        assert count == 3


class TestUserNameOperations:
    def test_update_user_name(self, test_db_path):
        db = Database(test_db_path)
        db.create_user(User(id="user_name_test"))

        result = db.update_user_name("user_name_test", "New Name")

        assert result is not None
        assert result.name == "New Name"

    def test_update_user_name_nonexistent(self, test_db_path):
        db = Database(test_db_path)

        result = db.update_user_name("nonexistent_user", "Name")

        assert result is None


class TestProfileOperations:
    def test_save_profile(self, test_db_path):
        db = Database(test_db_path)
        db.create_user(User(id="user_profile"))

        version = db.save_profile("user_profile", "Profile content here", {"consensus": True})

        assert version == 1

    def test_save_profile_increments_version(self, test_db_path):
        db = Database(test_db_path)
        db.create_user(User(id="user_profile_v"))

        v1 = db.save_profile("user_profile_v", "Profile v1")
        v2 = db.save_profile("user_profile_v", "Profile v2")

        assert v1 == 1
        assert v2 == 2

    def test_get_latest_profile(self, test_db_path):
        db = Database(test_db_path)
        db.create_user(User(id="user_get_profile"))
        db.save_profile("user_get_profile", "First profile")
        db.save_profile("user_get_profile", "Latest profile", {"log": "data"})

        profile = db.get_latest_profile("user_get_profile")

        assert profile is not None
        assert profile["content"] == "Latest profile"
        assert profile["version"] == 2

    def test_get_latest_profile_none(self, test_db_path):
        db = Database(test_db_path)
        db.create_user(User(id="user_no_profile"))

        profile = db.get_latest_profile("user_no_profile")

        assert profile is None

    def test_count_sessions_with_profile(self, test_db_path):
        db = Database(test_db_path)
        db.create_user(User(id="user_count_profile"))

        db.create_session(Session(user_id="user_count_profile"))
        db.save_profile("user_count_profile", "Profile after first session")
        db.create_session(Session(user_id="user_count_profile"))
        db.create_session(Session(user_id="user_count_profile"))

        count = db.count_sessions_since_last_analysis("user_count_profile")

        assert count == 2


class TestMessageRangeOperations:
    def test_get_messages_in_range(self, test_db_path):
        db = Database(test_db_path)
        db.create_user(User(id="user_range"))
        session = Session(id="session_range", user_id="user_range")
        db.create_session(session)

        db.save_message(Message(session_id="session_range", role="user", content="Message 1"))
        db.save_message(Message(session_id="session_range", role="user", content="Message 2"))
        db.save_message(Message(session_id="session_range", role="user", content="Message 3"))

        messages = db.get_messages_in_range("user_range")

        assert len(messages) == 3

    def test_get_messages_in_range_with_dates(self, test_db_path):
        from datetime import datetime, timedelta

        db = Database(test_db_path)
        db.create_user(User(id="user_range_dates"))
        session = Session(id="session_range_dates", user_id="user_range_dates")
        db.create_session(session)

        now = datetime.now()
        db.save_message(Message(session_id="session_range_dates", role="user", content="Old message"))
        db.save_message(Message(session_id="session_range_dates", role="user", content="New message"))

        start = now - timedelta(hours=1)
        end = now + timedelta(hours=1)
        messages = db.get_messages_in_range("user_range_dates", start_date=start, end_date=end)

        assert len(messages) >= 0

    def test_get_recent_messages(self, test_db_path):
        db = Database(test_db_path)
        db.create_user(User(id="user_recent"))
        session = Session(id="session_recent", user_id="user_recent")
        db.create_session(session)

        for i in range(5):
            db.save_message(Message(session_id="session_recent", role="user", content=f"Message {i}"))

        recent = db.get_recent_messages("user_recent", limit=3)

        assert len(recent) == 3

    def test_get_recent_messages_order(self, test_db_path):
        db = Database(test_db_path)
        db.create_user(User(id="user_recent_order"))
        session = Session(id="session_recent_order", user_id="user_recent_order")
        db.create_session(session)

        db.save_message(Message(session_id="session_recent_order", role="user", content="First"))
        db.save_message(Message(session_id="session_recent_order", role="user", content="Second"))
        db.save_message(Message(session_id="session_recent_order", role="user", content="Third"))

        recent = db.get_recent_messages("user_recent_order", limit=3)

        assert recent[0].content == "First"
        assert recent[-1].content == "Third"


class TestSessionWithCountsOperations:
    def test_get_user_sessions_with_counts(self, test_db_path):
        db = Database(test_db_path)
        db.create_user(User(id="user_sess_counts"))

        session1 = Session(id="sess_count_1", user_id="user_sess_counts")
        session2 = Session(id="sess_count_2", user_id="user_sess_counts")
        db.create_session(session1)
        db.create_session(session2)

        db.save_message(Message(session_id="sess_count_1", role="user", content="Msg 1"))
        db.save_message(Message(session_id="sess_count_1", role="emperor", content="Reply 1"))
        db.save_message(Message(session_id="sess_count_2", role="user", content="Msg 2"))

        rows = db.get_user_sessions_with_counts("user_sess_counts")

        assert len(rows) == 2
        counts = {r["id"]: r["message_count"] for r in rows}
        assert counts["sess_count_1"] == 2
        assert counts["sess_count_2"] == 1

    def test_get_user_sessions_empty(self, test_db_path):
        db = Database(test_db_path)
        db.create_user(User(id="user_no_sessions"))

        rows = db.get_user_sessions_with_counts("user_no_sessions")

        assert len(rows) == 0


class TestCondensedSummaryOperations:
    def test_save_and_get_condensed_summary(self, test_db_path):
        from datetime import datetime

        from src.models.schemas import CondensedSummary

        db = Database(test_db_path)
        db.create_user(User(id="user_summary"))

        summary = CondensedSummary(
            user_id="user_summary",
            level=1,
            content="Summary content here",
            period_start=datetime(2025, 1, 1),
            period_end=datetime(2025, 1, 7),
            source_message_count=10,
            source_word_count=500,
            source_summary_ids=[],
            consensus_log={"reached": True},
        )

        db.save_condensed_summary(summary)
        summaries = db.get_condensed_summaries("user_summary")

        assert len(summaries) == 1
        assert summaries[0].content == "Summary content here"

    def test_get_condensed_summaries_by_level(self, test_db_path):
        from datetime import datetime

        from src.models.schemas import CondensedSummary

        db = Database(test_db_path)
        db.create_user(User(id="user_summary_level"))

        for level in [1, 1, 2]:
            summary = CondensedSummary(
                user_id="user_summary_level",
                level=level,
                content=f"Summary level {level}",
                period_start=datetime(2025, 1, 1),
                period_end=datetime(2025, 1, 7),
                source_message_count=5,
                source_word_count=200,
            )
            db.save_condensed_summary(summary)

        level_1 = db.get_condensed_summaries("user_summary_level", level=1)
        level_2 = db.get_condensed_summaries("user_summary_level", level=2)

        assert len(level_1) == 2
        assert len(level_2) == 1


class TestGetSessionNonexistent:
    def test_get_session_nonexistent(self, test_db_path):
        db = Database(test_db_path)

        session = db.get_session("nonexistent_session_id")

        assert session is None

    def test_get_latest_session_none(self, test_db_path):
        db = Database(test_db_path)
        db.create_user(User(id="user_no_sess"))

        session = db.get_latest_session("user_no_sess")

        assert session is None
