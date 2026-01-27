import pytest
import os
from unittest.mock import patch, MagicMock


@pytest.fixture
def test_client(temp_dir):
    os.environ["ENVIRONMENT"] = "development"
    
    with patch("src.web.api.config") as mock_config, \
         patch("src.web.api.db") as mock_db, \
         patch("src.web.api.vectors"), \
         patch("src.web.api.brain"):
        
        mock_config.get.return_value = {"sessions_between_analysis": 5}
        mock_config.__getitem__ = lambda self, key: {"sessions_between_analysis": 5}
        
        mock_user = MagicMock()
        mock_user.id = "test_user_123"
        mock_db.get_or_create_user.return_value = mock_user
        mock_db.get_latest_profile.return_value = None
        mock_db.count_sessions_since_last_analysis.return_value = 2
        
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
            "consensus_log": {
                "consensus_reached": True,
                "stability_score": 0.85
            }
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
        mock_db.count_sessions_since_last_analysis.return_value = 2
        mock_db.get_latest_profile.return_value = None
        
        response = client.get("/api/analysis/status")
        
        assert response.status_code == 200
        data = response.json()
        assert data["sessions_since_analysis"] == 2
        assert data["can_analyze"] is False
        assert data["has_profile"] is False

    def test_analysis_status_ready(self, test_client):
        client, mock_db = test_client
        mock_db.count_sessions_since_last_analysis.return_value = 7
        mock_db.get_latest_profile.return_value = {"version": 1}
        
        response = client.get("/api/analysis/status")
        
        assert response.status_code == 200
        data = response.json()
        assert data["sessions_since_analysis"] == 7
        assert data["can_analyze"] is True
        assert data["has_profile"] is True


class TestChatEndpoint:
    def test_chat_creates_response(self, test_client):
        client, mock_db = test_client
        
        mock_session = MagicMock()
        mock_session.id = "session_123"
        mock_db.get_latest_session.return_value = mock_session
        mock_db.get_session_messages.return_value = []
        
        with patch("src.web.api.brain") as mock_brain:
            mock_response = MagicMock()
            mock_response.response_text = "Greetings, seeker of wisdom."
            mock_response.psych_update = None
            mock_brain.respond.return_value = mock_response
            
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
