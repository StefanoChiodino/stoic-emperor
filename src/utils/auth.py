import os
from functools import lru_cache

import httpx
from fastapi import HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

security = HTTPBearer(auto_error=False)

SUPABASE_URL = os.getenv("SUPABASE_URL", "")


def get_supabase_jwt_secret() -> str | None:
    return os.getenv("SUPABASE_JWT_SECRET")


@lru_cache(maxsize=1)
def get_supabase_jwks() -> dict:
    if not SUPABASE_URL:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="SUPABASE_URL not configured")
    jwks_url = f"{SUPABASE_URL.rstrip('/')}/auth/v1/.well-known/jwks.json"
    response = httpx.get(jwks_url, timeout=10)
    response.raise_for_status()
    return response.json()


def verify_supabase_token(token: str) -> dict:
    try:
        # First, peek at the token header to determine algorithm
        unverified_header = jwt.get_unverified_header(token)
        alg = unverified_header.get("alg", "HS256")

        if alg.startswith("HS"):
            # HMAC algorithms use the JWT secret
            secret = get_supabase_jwt_secret()
            if not secret:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="SUPABASE_JWT_SECRET not configured for HS* algorithm",
                )
            payload = jwt.decode(token, secret, algorithms=[alg], audience="authenticated")
        elif alg.startswith("ES") or alg.startswith("RS"):
            # Asymmetric algorithms use JWKS public keys
            jwks = get_supabase_jwks()
            payload = jwt.decode(token, jwks, algorithms=[alg], audience="authenticated")
        else:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f"Unsupported algorithm: {alg}")

        return payload
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {e!s}",
        ) from None
    except httpx.HTTPError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch JWKS: {e!s}",
        ) from None


def get_user_id_from_token(credentials: HTTPAuthorizationCredentials) -> str:
    token = credentials.credentials
    payload = verify_supabase_token(token)
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token: missing user ID")
    return user_id


def optional_auth(credentials: HTTPAuthorizationCredentials | None = Security(security)) -> str | None:
    if not credentials:
        return None
    try:
        return get_user_id_from_token(credentials)
    except HTTPException:
        return None
