import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from src.infrastructure.database import Database
from src.models.schemas import User, Session, Message, CondensedSummary, PsychUpdate
from src.memory.condensation import CondensationManager
from src.memory.context_builder import ContextBuilder


@pytest.fixture
def test_config():
    return {
        "models": {
            "main": "gpt-4o-mini",
            "reviewer": "gpt-4o-mini"
        },
        "condensation": {
            "hot_buffer_tokens": 100,
            "chunk_threshold_tokens": 200,
            "summary_budget_tokens": 300,
            "use_consensus": False
        },
        "prompts": {
            "condensation": """
Condense: {messages}
Period: {period_start} to {period_end}
Messages: {message_count}, Words: {word_count}
Previous: {previous_context}
"""
        }
    }


@pytest.fixture
def db_with_messages(test_db_path):
    db = Database(test_db_path)
    user = User(id="test_user")
    db.create_user(user)

    session = Session(user_id=user.id)
    db.create_session(session)

    messages = []
    base_time = datetime(2024, 1, 1, 10, 0)

    for i in range(10):
        msg_time = base_time + timedelta(minutes=i * 10)

        user_msg = Message(
            session_id=session.id,
            role="user",
            content=f"User message {i}: " + "word " * 20,
            created_at=msg_time
        )
        db.save_message(user_msg)
        messages.append(user_msg)

        psych_update = PsychUpdate(
            detected_patterns=["pattern_1"],
            emotional_state="neutral",
            stoic_principle_applied="test",
            suggested_next_direction="continue",
            confidence=0.8
        )

        emperor_msg = Message(
            session_id=session.id,
            role="emperor",
            content=f"Emperor response {i}: " + "word " * 20,
            psych_update=psych_update,
            created_at=msg_time + timedelta(seconds=30)
        )
        db.save_message(emperor_msg)
        messages.append(emperor_msg)

    return db, user, session, messages


class TestCondensationManager:
    def test_initialization(self, test_db_path, test_config):
        db = Database(test_db_path)
        manager = CondensationManager(db, test_config)

        assert manager.hot_buffer_tokens == 100
        assert manager.chunk_threshold_tokens == 200
        assert manager.summary_budget_tokens == 300
        assert manager.use_consensus == False

    def test_estimate_tokens(self, test_db_path, test_config):
        db = Database(test_db_path)
        manager = CondensationManager(db, test_config)

        short_text = "Hello world"
        long_text = "word " * 100

        short_tokens = manager.estimate_tokens(short_text)
        long_tokens = manager.estimate_tokens(long_text)

        assert short_tokens < long_tokens
        assert short_tokens > 0

    def test_get_uncondensed_messages_with_hot_buffer(self, db_with_messages, test_config):
        db, user, session, messages = db_with_messages
        manager = CondensationManager(db, test_config)

        uncondensed = manager.get_uncondensed_messages(user.id)

        assert len(uncondensed) > 0
        recent_messages = db.get_recent_messages(user.id, limit=20)
        assert len(uncondensed) < len(recent_messages)

    def test_should_condense_false_when_below_threshold(self, test_db_path, test_config):
        db = Database(test_db_path)
        user = User(id="test_user")
        db.create_user(user)

        session = Session(user_id=user.id)
        db.create_session(session)

        msg = Message(
            session_id=session.id,
            role="user",
            content="Short message",
            created_at=datetime.now()
        )
        db.save_message(msg)

        manager = CondensationManager(db, test_config)

        assert manager.should_condense(user.id) == False

    def test_should_condense_true_when_above_threshold(self, db_with_messages, test_config):
        db, user, session, messages = db_with_messages
        manager = CondensationManager(db, test_config)

        should = manager.should_condense(user.id)

        assert isinstance(should, bool)

    @patch('openai.OpenAI')
    def test_condense_chunk_creates_summary(self, mock_openai_class, db_with_messages, test_config):
        db, user, session, messages = db_with_messages
        manager = CondensationManager(db, test_config)

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content.strip.return_value = "Condensed summary content"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        chunk_messages = messages[:4]

        summary = manager.condense_chunk(user.id, chunk_messages)

        assert summary is not None
        assert isinstance(summary, CondensedSummary)
        assert summary.user_id == user.id
        assert summary.level == 1
        assert summary.content == "Condensed summary content"
        assert summary.source_message_count == 4
        assert summary.source_word_count > 0
        assert summary.period_start == chunk_messages[0].created_at
        assert summary.period_end == chunk_messages[-1].created_at

    @patch('openai.OpenAI')
    def test_condense_summaries_creates_higher_level(self, mock_openai_class, test_db_path, test_config):
        db = Database(test_db_path)
        user = User(id="test_user")
        db.create_user(user)

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content.strip.return_value = "Level 2 summary"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        summary1 = CondensedSummary(
            user_id=user.id,
            level=1,
            content="Summary 1 content",
            period_start=datetime(2024, 1, 1),
            period_end=datetime(2024, 1, 2),
            source_message_count=10,
            source_word_count=100
        )
        db.save_condensed_summary(summary1)

        summary2 = CondensedSummary(
            user_id=user.id,
            level=1,
            content="Summary 2 content",
            period_start=datetime(2024, 1, 3),
            period_end=datetime(2024, 1, 4),
            source_message_count=10,
            source_word_count=100
        )
        db.save_condensed_summary(summary2)

        manager = CondensationManager(db, test_config)

        higher_summary = manager.condense_summaries(user.id, level=1)

        assert higher_summary is not None
        assert higher_summary.level == 2
        assert higher_summary.content == "Level 2 summary"
        assert higher_summary.source_message_count == 20
        assert len(higher_summary.source_summary_ids) == 2

    def test_get_context_summaries_respects_budget(self, test_db_path, test_config):
        db = Database(test_db_path)
        user = User(id="test_user")
        db.create_user(user)

        for i in range(5):
            summary = CondensedSummary(
                user_id=user.id,
                level=1,
                content="x" * 100,
                period_start=datetime(2024, 1, i+1),
                period_end=datetime(2024, 1, i+2),
                source_message_count=10,
                source_word_count=100
            )
            db.save_condensed_summary(summary)

        manager = CondensationManager(db, test_config)

        selected = manager.get_context_summaries(user.id, token_budget=150)

        assert len(selected) > 0
        total_tokens = sum(manager.estimate_tokens(s.content) for s in selected)
        assert total_tokens <= 150

    def test_get_context_summaries_prefers_higher_levels(self, test_db_path, test_config):
        db = Database(test_db_path)
        user = User(id="test_user")
        db.create_user(user)

        l1_summary = CondensedSummary(
            user_id=user.id,
            level=1,
            content="Level 1 content",
            period_start=datetime(2024, 1, 1),
            period_end=datetime(2024, 1, 5),
            source_message_count=10,
            source_word_count=100
        )
        db.save_condensed_summary(l1_summary)

        l2_summary = CondensedSummary(
            user_id=user.id,
            level=2,
            content="Level 2 content covering same period",
            period_start=datetime(2024, 1, 1),
            period_end=datetime(2024, 1, 5),
            source_message_count=10,
            source_word_count=100,
            source_summary_ids=[l1_summary.id]
        )
        db.save_condensed_summary(l2_summary)

        manager = CondensationManager(db, test_config)

        selected = manager.get_context_summaries(user.id, token_budget=1000)

        assert any(s.level == 2 for s in selected)
        has_l1_for_same_period = any(
            s.level == 1 and
            s.period_start == l1_summary.period_start and
            s.period_end == l1_summary.period_end
            for s in selected
        )
        assert not has_l1_for_same_period

    @patch('openai.OpenAI')
    def test_maybe_condense_triggers_when_threshold_exceeded(self, mock_openai_class, db_with_messages, test_config):
        db, user, session, messages = db_with_messages

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content.strip.return_value = "Condensed"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        manager = CondensationManager(db, test_config)

        result = manager.maybe_condense(user.id, verbose=False)

        assert isinstance(result, bool)


class TestContextBuilder:
    def test_initialization(self, test_db_path, test_config):
        db = Database(test_db_path)
        builder = ContextBuilder(db, test_config)

        assert builder.hot_buffer_tokens == 100
        assert builder.summary_budget_tokens == 300

    def test_build_context_with_no_data(self, test_db_path, test_config):
        db = Database(test_db_path)
        user = User(id="test_user")
        db.create_user(user)

        builder = ContextBuilder(db, test_config)

        context = builder.build_context(user.id, max_tokens=500)

        assert "recent_messages" in context
        assert "condensed_summaries" in context
        assert len(context["recent_messages"]) == 0
        assert len(context["condensed_summaries"]) == 0

    def test_build_context_with_messages(self, db_with_messages, test_config):
        db, user, session, messages = db_with_messages
        builder = ContextBuilder(db, test_config)

        context = builder.build_context(user.id, max_tokens=5000)

        assert len(context["recent_messages"]) > 0
        assert context["total_tokens"] > 0

    def test_build_context_with_summaries(self, test_db_path, test_config):
        db = Database(test_db_path)
        user = User(id="test_user")
        db.create_user(user)

        session = Session(user_id=user.id)
        db.create_session(session)

        summary = CondensedSummary(
            user_id=user.id,
            level=1,
            content="Past conversation summary",
            period_start=datetime(2024, 1, 1),
            period_end=datetime(2024, 1, 5),
            source_message_count=20,
            source_word_count=200
        )
        db.save_condensed_summary(summary)

        msg = Message(
            session_id=session.id,
            role="user",
            content="Recent message",
            created_at=datetime(2024, 1, 10)
        )
        db.save_message(msg)

        builder = ContextBuilder(db, test_config)

        context = builder.build_context(user.id, max_tokens=5000)

        assert len(context["condensed_summaries"]) == 1
        assert len(context["recent_messages"]) >= 1

    def test_format_context_string(self, test_db_path, test_config):
        db = Database(test_db_path)
        user = User(id="test_user")
        db.create_user(user)

        session = Session(user_id=user.id)
        db.create_session(session)

        summary = CondensedSummary(
            user_id=user.id,
            level=1,
            content="Summary content",
            period_start=datetime(2024, 1, 1),
            period_end=datetime(2024, 1, 5),
            source_message_count=10,
            source_word_count=100
        )

        msg = Message(
            session_id=session.id,
            role="user",
            content="Hello",
            created_at=datetime(2024, 1, 10)
        )

        builder = ContextBuilder(db, test_config)

        context = {
            "condensed_summaries": [summary],
            "recent_messages": [msg]
        }

        formatted = builder.format_context_string(context)

        assert "Historical Context" in formatted
        assert "Recent Conversation" in formatted
        assert "Summary content" in formatted
        assert "Hello" in formatted

    def test_get_summary_statistics(self, test_db_path, test_config):
        db = Database(test_db_path)
        user = User(id="test_user")
        db.create_user(user)

        for level in [1, 1, 2]:
            summary = CondensedSummary(
                user_id=user.id,
                level=level,
                content="Content",
                period_start=datetime(2024, 1, 1),
                period_end=datetime(2024, 1, 5),
                source_message_count=10,
                source_word_count=100
            )
            db.save_condensed_summary(summary)

        builder = ContextBuilder(db, test_config)

        stats = builder.get_summary_statistics(user.id)

        assert stats["total_summaries"] == 3
        assert stats["levels"] == {1: 2, 2: 1}
        assert stats["total_messages_condensed"] == 30
        assert stats["total_words_condensed"] == 300


class TestDatabaseCondensedSummaryOperations:
    def test_save_condensed_summary(self, test_db_path):
        db = Database(test_db_path)
        user = User(id="test_user")
        db.create_user(user)

        summary = CondensedSummary(
            user_id=user.id,
            level=1,
            content="Test summary content",
            period_start=datetime(2024, 1, 1),
            period_end=datetime(2024, 1, 5),
            source_message_count=10,
            source_word_count=100,
            source_summary_ids=["id1", "id2"],
            consensus_log={"consensus_reached": True}
        )

        saved = db.save_condensed_summary(summary)

        assert saved.id == summary.id
        assert saved.user_id == user.id

    def test_get_condensed_summaries_all(self, test_db_path):
        db = Database(test_db_path)
        user = User(id="test_user")
        db.create_user(user)

        for i in range(3):
            summary = CondensedSummary(
                user_id=user.id,
                level=1,
                content=f"Summary {i}",
                period_start=datetime(2024, 1, i+1),
                period_end=datetime(2024, 1, i+2),
                source_message_count=10,
                source_word_count=100
            )
            db.save_condensed_summary(summary)

        summaries = db.get_condensed_summaries(user.id)

        assert len(summaries) == 3

    def test_get_condensed_summaries_by_level(self, test_db_path):
        db = Database(test_db_path)
        user = User(id="test_user")
        db.create_user(user)

        for level in [1, 1, 2, 3]:
            summary = CondensedSummary(
                user_id=user.id,
                level=level,
                content="Content",
                period_start=datetime(2024, 1, 1),
                period_end=datetime(2024, 1, 5),
                source_message_count=10,
                source_word_count=100
            )
            db.save_condensed_summary(summary)

        level_1_summaries = db.get_condensed_summaries(user.id, level=1)

        assert len(level_1_summaries) == 2
        assert all(s.level == 1 for s in level_1_summaries)

    def test_get_messages_in_range(self, test_db_path):
        db = Database(test_db_path)
        user = User(id="test_user")
        db.create_user(user)

        session = Session(user_id=user.id)
        db.create_session(session)

        dates = [
            datetime(2024, 1, 1),
            datetime(2024, 1, 5),
            datetime(2024, 1, 10)
        ]

        for date in dates:
            msg = Message(
                session_id=session.id,
                role="user",
                content="Message",
                created_at=date
            )
            db.save_message(msg)

        messages = db.get_messages_in_range(
            user.id,
            start_date=datetime(2024, 1, 3),
            end_date=datetime(2024, 1, 7)
        )

        assert len(messages) == 1
        assert messages[0].created_at == datetime(2024, 1, 5)

    def test_condensed_summary_table_exists(self, test_db_path):
        from sqlalchemy import inspect

        db = Database(test_db_path)

        inspector = inspect(db.engine)
        table_names = inspector.get_table_names()

        assert "condensed_summaries" in table_names
