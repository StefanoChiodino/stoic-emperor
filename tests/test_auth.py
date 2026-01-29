import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from src.utils.auth import (
    get_supabase_jwt_secret,
    get_user_id_from_token,
    optional_auth,
    verify_supabase_token,
)


class TestGetSupabaseJwtSecret:
    def test_returns_secret_when_set(self):
        with patch.dict(os.environ, {"SUPABASE_JWT_SECRET": "test-secret-key"}):
            result = get_supabase_jwt_secret()
            assert result == "test-secret-key"

    def test_raises_when_not_set(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("SUPABASE_JWT_SECRET", None)
            with pytest.raises(HTTPException) as exc_info:
                get_supabase_jwt_secret()
            assert exc_info.value.status_code == 500
            assert "not configured" in exc_info.value.detail


class TestVerifySupabaseToken:
    def test_valid_token(self):
        from jose import jwt

        secret = "test-secret-key-for-jwt"
        token = jwt.encode({"sub": "user123", "aud": "authenticated"}, secret, algorithm="HS256")

        with patch.dict(os.environ, {"SUPABASE_JWT_SECRET": secret}):
            payload = verify_supabase_token(token)
            assert payload["sub"] == "user123"

    def test_invalid_token(self):
        with patch.dict(os.environ, {"SUPABASE_JWT_SECRET": "secret"}):
            with pytest.raises(HTTPException) as exc_info:
                verify_supabase_token("invalid-token")
            assert exc_info.value.status_code == 401
            assert "Invalid token" in exc_info.value.detail

    def test_wrong_secret(self):
        from jose import jwt

        token = jwt.encode({"sub": "user123", "aud": "authenticated"}, "correct-secret", algorithm="HS256")

        with patch.dict(os.environ, {"SUPABASE_JWT_SECRET": "wrong-secret"}):
            with pytest.raises(HTTPException) as exc_info:
                verify_supabase_token(token)
            assert exc_info.value.status_code == 401


class TestGetUserIdFromToken:
    def test_extracts_user_id(self):
        from jose import jwt

        secret = "test-secret"
        token = jwt.encode({"sub": "user-abc-123", "aud": "authenticated"}, secret, algorithm="HS256")

        credentials = MagicMock()
        credentials.credentials = token

        with patch.dict(os.environ, {"SUPABASE_JWT_SECRET": secret}):
            user_id = get_user_id_from_token(credentials)
            assert user_id == "user-abc-123"

    def test_missing_sub_claim(self):
        from jose import jwt

        secret = "test-secret"
        token = jwt.encode({"aud": "authenticated"}, secret, algorithm="HS256")

        credentials = MagicMock()
        credentials.credentials = token

        with patch.dict(os.environ, {"SUPABASE_JWT_SECRET": secret}):
            with pytest.raises(HTTPException) as exc_info:
                get_user_id_from_token(credentials)
            assert exc_info.value.status_code == 401
            assert "missing user ID" in exc_info.value.detail


class TestOptionalAuth:
    def test_returns_none_when_no_credentials(self):
        result = optional_auth(None)
        assert result is None

    def test_returns_user_id_with_valid_credentials(self):
        from jose import jwt

        secret = "test-secret"
        token = jwt.encode({"sub": "user123", "aud": "authenticated"}, secret, algorithm="HS256")

        credentials = MagicMock()
        credentials.credentials = token

        with patch.dict(os.environ, {"SUPABASE_JWT_SECRET": secret}):
            result = optional_auth(credentials)
            assert result == "user123"

    def test_returns_none_with_invalid_credentials(self):
        credentials = MagicMock()
        credentials.credentials = "invalid-token"

        with patch.dict(os.environ, {"SUPABASE_JWT_SECRET": "secret"}):
            result = optional_auth(credentials)
            assert result is None
