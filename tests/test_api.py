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
