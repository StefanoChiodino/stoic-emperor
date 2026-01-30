import os
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def test_client(temp_dir):
    os.environ["ENVIRONMENT"] = "development"

    mock_db = MagicMock()
    mock_vectors = MagicMock()
    mock_brain = MagicMock()
    mock_condensation = MagicMock()
    mock_episodic = MagicMock()

    mock_user = MagicMock()
    mock_user.id = "test_user_123"
    mock_db.get_or_create_user.return_value = mock_user
    mock_db.get_latest_profile.return_value = None
    mock_db.get_condensed_summaries.return_value = []

    mock_condensation.get_uncondensed_messages.return_value = []
    mock_condensation.chunk_threshold_tokens = 2000

    mock_state = {
        "initialized": True,
        "config": {"sessions_between_analysis": 5},
        "db": mock_db,
        "vectors": mock_vectors,
        "brain": mock_brain,
        "condensation": mock_condensation,
        "episodic": mock_episodic,
    }

    with patch("src.web.api._state", mock_state), patch("src.web.api._init"):
        from fastapi.testclient import TestClient

        from src.web.api import app

        yield TestClient(app), mock_db


class TestProfileEndpoint:
    def test_get_profile_no_profile(self, test_client):
        client, mock_db = test_client
        mock_db.get_latest_profile.return_value = None

        response = client.get("/api/profile")

        assert response.status_code == 200
        assert response.json() is None

    def test_get_profile_with_profile(self, test_client):
        client, mock_db = test_client
        mock_db.get_latest_profile.return_value = {
            "version": 3,
            "content": "# Profile\n\nTest content",
            "created_at": "2025-01-15T10:00:00",
            "consensus_log": {"consensus_reached": True, "stability_score": 0.85},
        }

        response = client.get("/api/profile")

        assert response.status_code == 200
        data = response.json()
        assert data["version"] == 3
        assert data["consensus_reached"] is True
        assert data["stability_score"] == 0.85


class TestAnalysisStatusEndpoint:
    def test_analysis_status_not_ready(self, test_client):
        client, mock_db = test_client
        mock_db.get_latest_profile.return_value = None
        mock_db.get_condensed_summaries.return_value = []

        response = client.get("/api/analysis/status")

        assert response.status_code == 200
        data = response.json()
        assert data["uncondensed_tokens"] == 0
        assert data["summary_count"] == 0
        assert data["has_profile"] is False

    def test_analysis_status_ready(self, test_client):
        client, mock_db = test_client
        mock_db.get_latest_profile.return_value = {"version": 1}
        mock_db.get_condensed_summaries.return_value = [MagicMock(), MagicMock(), MagicMock()]

        response = client.get("/api/analysis/status")

        assert response.status_code == 200
        data = response.json()
        assert data["summary_count"] == 3
        assert data["has_profile"] is True


class TestChatEndpoint:
    def test_chat_creates_response(self, test_client):
        client, mock_db = test_client
        from src.models.schemas import PsychUpdate

        mock_session = MagicMock()
        mock_session.id = "session_123"
        mock_db.get_latest_session.return_value = mock_session
        mock_db.get_session_messages.return_value = []

        from src.web.api import _state

        mock_response = MagicMock()
        mock_response.response_text = "Greetings, seeker of wisdom."
        mock_response.psych_update = PsychUpdate(
            detected_patterns=[],
            emotional_state="curious",
            stoic_principle_applied="presence",
            suggested_next_direction="continue exploration",
            confidence=0.8,
            semantic_assertions=[],
        )
        _state["brain"].respond.return_value = mock_response
        _state["brain"].expand_query.return_value = "hello"
        _state["condensation"].get_context_summaries.return_value = []

        response = client.post("/api/chat", json={"message": "Hello"})

        assert response.status_code == 200
        data = response.json()
        assert data["response"] == "Greetings, seeker of wisdom."
        assert data["session_id"] == "session_123"


class TestIndexRoute:
    def test_index_returns_html(self, test_client):
        client, _ = test_client

        response = client.get("/")

        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]


class TestHealthEndpoint:
    def test_health_returns_ok(self, test_client):
        client, _ = test_client

        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestConfigEndpoint:
    def test_get_config(self, test_client):
        client, _ = test_client

        response = client.get("/api/config")

        assert response.status_code == 200
        data = response.json()
        assert "supabase_url" in data
        assert "environment" in data


class TestUserEndpoint:
    def test_get_user(self, test_client):
        client, mock_db = test_client
        from datetime import datetime

        mock_user = MagicMock()
        mock_user.id = "test_user_123"
        mock_user.name = None
        mock_user.created_at = datetime.now()
        mock_db.get_or_create_user.return_value = mock_user

        response = client.get("/api/user")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "test_user_123"

    def test_update_user_name(self, test_client):
        client, mock_db = test_client
        from datetime import datetime

        mock_user = MagicMock()
        mock_user.id = "test_user_123"
        mock_user.name = "New Name"
        mock_user.created_at = datetime.now()
        mock_db.update_user_name.return_value = mock_user

        response = client.put("/api/user/name", json={"name": "New Name"})

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "New Name"

    def test_update_user_name_not_found(self, test_client):
        client, mock_db = test_client
        mock_db.update_user_name.return_value = None

        response = client.put("/api/user/name", json={"name": "New Name"})

        assert response.status_code == 404


class TestSessionsEndpoint:
    def test_create_session(self, test_client):
        client, mock_db = test_client

        response = client.post("/api/sessions")

        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert "created_at" in data

    def test_list_sessions(self, test_client):
        client, mock_db = test_client
        from datetime import datetime

        mock_db.get_user_sessions_with_counts.return_value = [
            {"id": "sess1", "created_at": datetime.now(), "message_count": 5},
            {"id": "sess2", "created_at": datetime.now(), "message_count": 3},
        ]

        response = client.get("/api/sessions")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2

    def test_get_session_messages(self, test_client):
        client, mock_db = test_client

        mock_session = MagicMock()
        mock_session.id = "sess_123"
        mock_db.get_session.return_value = mock_session

        from src.models.schemas import Message

        mock_db.get_session_messages.return_value = [
            Message(session_id="sess_123", role="user", content="Hello"),
            Message(session_id="sess_123", role="emperor", content="Greetings"),
        ]

        response = client.get("/api/sessions/sess_123/messages")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2

    def test_get_session_messages_not_found(self, test_client):
        client, mock_db = test_client
        mock_db.get_session.return_value = None

        response = client.get("/api/sessions/nonexistent/messages")

        assert response.status_code == 404


class TestLoginRoute:
    def test_login_returns_html(self, test_client):
        client, _ = test_client

        response = client.get("/login")

        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]


class TestHistoryRoute:
    def test_history_returns_html(self, test_client):
        client, _ = test_client

        response = client.get("/history")

        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]


class TestAnalysisRoute:
    def test_analysis_returns_html(self, test_client):
        client, _ = test_client

        response = client.get("/analysis")

        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]


class TestChatWithNewSession:
    def test_chat_creates_new_session(self, test_client):
        client, mock_db = test_client
        from src.models.schemas import PsychUpdate

        mock_db.get_latest_session.return_value = None
        mock_db.get_session_messages.return_value = []

        from src.web.api import _state

        mock_response = MagicMock()
        mock_response.response_text = "A response"
        mock_response.psych_update = PsychUpdate(
            detected_patterns=[],
            emotional_state="neutral",
            stoic_principle_applied="",
            suggested_next_direction="",
            confidence=0.5,
            semantic_assertions=[],
        )
        _state["brain"].respond.return_value = mock_response
        _state["brain"].expand_query.return_value = "query"
        _state["condensation"].get_context_summaries.return_value = []

        response = client.post("/api/chat", json={"message": "Hello"})

        assert response.status_code == 200
        mock_db.create_session.assert_called()

    def test_chat_with_existing_session_id(self, test_client):
        client, mock_db = test_client
        from src.models.schemas import PsychUpdate

        mock_session = MagicMock()
        mock_session.id = "existing_session"
        mock_db.get_session.return_value = mock_session
        mock_db.get_session_messages.return_value = []

        from src.web.api import _state

        mock_response = MagicMock()
        mock_response.response_text = "Response"
        mock_response.psych_update = PsychUpdate(
            detected_patterns=[],
            emotional_state="calm",
            stoic_principle_applied="",
            suggested_next_direction="",
            confidence=0.7,
            semantic_assertions=[],
        )
        _state["brain"].respond.return_value = mock_response
        _state["brain"].expand_query.return_value = "expanded"
        _state["condensation"].get_context_summaries.return_value = []

        response = client.post("/api/chat", json={"message": "Hi", "session_id": "existing_session"})

        assert response.status_code == 200

    def test_chat_session_not_found(self, test_client):
        client, mock_db = test_client
        mock_db.get_session.return_value = None

        response = client.post("/api/chat", json={"message": "Hi", "session_id": "nonexistent"})

        assert response.status_code == 404


class TestChatWithSemanticAssertions:
    def test_chat_stores_semantic_assertions(self, test_client):
        client, mock_db = test_client
        from src.models.schemas import PsychUpdate, SemanticAssertion

        mock_session = MagicMock()
        mock_session.id = "sess_assert"
        mock_db.get_latest_session.return_value = mock_session
        mock_db.get_session_messages.return_value = []

        from src.web.api import _state

        mock_response = MagicMock()
        mock_response.response_text = "Response with assertions"
        mock_response.psych_update = PsychUpdate(
            detected_patterns=[],
            emotional_state="reflective",
            stoic_principle_applied="acceptance",
            suggested_next_direction="continue",
            confidence=0.8,
            semantic_assertions=[
                SemanticAssertion(text="User values family", confidence=0.9),
                SemanticAssertion(text="User has work stress", confidence=0.85),
            ],
        )
        _state["brain"].respond.return_value = mock_response
        _state["brain"].expand_query.return_value = "query"
        _state["condensation"].get_context_summaries.return_value = []

        response = client.post("/api/chat", json={"message": "I value my family but work is stressful"})

        assert response.status_code == 200
        assert mock_db.save_semantic_insight.call_count == 2


class TestChatWithProfile:
    def test_chat_includes_profile(self, test_client):
        client, mock_db = test_client
        from src.models.schemas import PsychUpdate

        mock_session = MagicMock()
        mock_session.id = "sess_profile"
        mock_db.get_latest_session.return_value = mock_session
        mock_db.get_session_messages.return_value = []
        mock_db.get_latest_profile.return_value = {"content": "User profile content"}

        from src.web.api import _state

        mock_response = MagicMock()
        mock_response.response_text = "Response"
        mock_response.psych_update = PsychUpdate(
            detected_patterns=[],
            emotional_state="calm",
            stoic_principle_applied="",
            suggested_next_direction="",
            confidence=0.5,
            semantic_assertions=[],
        )
        _state["brain"].respond.return_value = mock_response
        _state["brain"].expand_query.return_value = "query"
        _state["condensation"].get_context_summaries.return_value = []

        response = client.post("/api/chat", json={"message": "Hello"})

        assert response.status_code == 200


class TestChatWithSummaries:
    def test_chat_includes_summaries(self, test_client):
        client, mock_db = test_client
        from src.models.schemas import PsychUpdate

        mock_session = MagicMock()
        mock_session.id = "sess_sum"
        mock_db.get_latest_session.return_value = mock_session
        mock_db.get_session_messages.return_value = []

        from src.web.api import _state

        mock_summary = MagicMock()
        mock_summary.content = "Previous conversation summary"
        _state["condensation"].get_context_summaries.return_value = [mock_summary]

        mock_response = MagicMock()
        mock_response.response_text = "Response"
        mock_response.psych_update = PsychUpdate(
            detected_patterns=[],
            emotional_state="calm",
            stoic_principle_applied="",
            suggested_next_direction="",
            confidence=0.5,
            semantic_assertions=[],
        )
        _state["brain"].respond.return_value = mock_response
        _state["brain"].expand_query.return_value = "query"

        response = client.post("/api/chat", json={"message": "Hello"})

        assert response.status_code == 200
