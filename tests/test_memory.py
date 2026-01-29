import pytest

from src.infrastructure.database import Database
from src.infrastructure.vector_store import VectorStore
from src.memory.episodic import EpisodicMemory, estimate_tokens
from src.memory.retrieval import UnifiedRetriever
from src.memory.semantic import SemanticMemory
from src.models.schemas import Message, Session, User


class TestTokenEstimation:
    def test_estimate_tokens_basic(self):
        text = "Hello world"
        tokens = estimate_tokens(text)
        assert tokens > 0
        assert tokens < 100

    def test_estimate_tokens_longer_text(self):
        text = "This is a longer piece of text that should have more tokens. " * 10
        tokens = estimate_tokens(text)
        assert tokens > 50


class TestEpisodicMemory:
    @pytest.fixture
    def episodic_setup(self, test_db_path, test_vector_path):
        db = Database(test_db_path)
        vectors = VectorStore(test_vector_path)
        db.create_user(User(id="episodic_user"))
        session = Session(id="episodic_session", user_id="episodic_user")
        db.create_session(session)

        messages = [
            ("user", "I've been feeling anxious lately."),
            ("emperor", "Tell me more about this anxiety."),
            ("user", "It's mostly about work deadlines."),
            ("emperor", "What is within your control regarding these deadlines?"),
            ("user", "I suppose my effort and planning."),
        ]

        for role, content in messages:
            db.save_message(Message(session_id="episodic_session", role=role, content=content))

        return db, vectors, EpisodicMemory(db, vectors, max_context_tokens=500)

    def test_get_recent_context(self, episodic_setup):
        db, vectors, memory = episodic_setup

        recent = memory.get_recent_context("episodic_session")

        assert len(recent) > 0
        assert len(recent) <= 5

    def test_get_recent_context_respects_token_limit(self, episodic_setup):
        db, vectors, memory = episodic_setup
        memory.max_context_tokens = 50

        recent = memory.get_recent_context("episodic_session")

        total_tokens = sum(estimate_tokens(m.content) for m in recent)
        assert total_tokens <= 100

    def test_store_turn(self, episodic_setup):
        db, vectors, memory = episodic_setup

        memory.store_turn(
            user_id="episodic_user",
            session_id="episodic_session",
            user_message="New question",
            emperor_response="New response",
        )

        count = vectors.count("episodic")
        assert count >= 1

    def test_search_past_conversations(self, episodic_setup):
        db, vectors, memory = episodic_setup

        memory.store_turn(
            user_id="episodic_user",
            session_id="episodic_session",
            user_message="Anxiety about work",
            emperor_response="Focus on what you can control",
        )

        results = memory.search_past_conversations("episodic_user", "work anxiety")

        assert isinstance(results, list)


class TestSemanticMemory:
    @pytest.fixture
    def semantic_setup(self, test_db_path, test_vector_path):
        db = Database(test_db_path)
        vectors = VectorStore(test_vector_path)

        from unittest.mock import MagicMock

        brain = MagicMock()
        brain.extract_semantic_insights = MagicMock(
            return_value=[{"text": "User struggles with work anxiety", "confidence": 0.85}]
        )

        return db, vectors, SemanticMemory(db, vectors, brain)

    def test_get_relevant_insights_empty(self, semantic_setup):
        db, vectors, memory = semantic_setup

        insights = memory.get_relevant_insights("nonexistent_user", "anything")

        assert insights == []

    def test_get_all_insights_empty(self, semantic_setup):
        db, vectors, memory = semantic_setup
        db.create_user(User(id="insight_user"))

        insights = memory.get_all_insights("insight_user")

        assert insights == []


class TestUnifiedRetriever:
    @pytest.fixture
    def retriever_setup(self, test_db_path, test_vector_path):
        db = Database(test_db_path)
        vectors = VectorStore(test_vector_path)

        from unittest.mock import MagicMock

        brain = MagicMock()
        brain.expand_query = MagicMock(return_value="anxiety, worry, stress, work")
        brain.extract_semantic_insights = MagicMock(return_value=[])
        brain.config = {"models": {"main": "gpt-4o"}}

        db.create_user(User(id="retriever_user"))
        session = Session(id="retriever_session", user_id="retriever_user")
        db.create_session(session)

        return db, vectors, UnifiedRetriever(db, vectors, brain)

    def test_retrieve_returns_context(self, retriever_setup):
        db, vectors, retriever = retriever_setup

        context = retriever.retrieve(
            user_id="retriever_user", session_id="retriever_session", user_message="I'm worried about my job"
        )

        assert context is not None
        assert hasattr(context, "recent_messages")
        assert hasattr(context, "stoic_wisdom")
        assert hasattr(context, "psychoanalysis")

    def test_retrieve_to_dict(self, retriever_setup):
        db, vectors, retriever = retriever_setup

        context = retriever.retrieve(
            user_id="retriever_user", session_id="retriever_session", user_message="Test message"
        )

        context_dict = context.to_dict()

        assert "stoic" in context_dict
        assert "psych" in context_dict
        assert "insights" in context_dict
