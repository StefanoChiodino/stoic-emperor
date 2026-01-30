import os

from fastapi import HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

security = HTTPBearer(auto_error=False)


def get_supabase_jwt_secret() -> str:
    secret = os.getenv("SUPABASE_JWT_SECRET")
    if not secret:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="SUPABASE_JWT_SECRET not configured"
        )
    return secret


def verify_supabase_token(token: str) -> dict:
    try:
        secret = get_supabase_jwt_secret()
        # Supabase uses HS256 by default, but allow common alternatives
        payload = jwt.decode(token, secret, algorithms=["HS256", "HS384", "HS512"], audience="authenticated")
        return payload
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {e!s}",
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
