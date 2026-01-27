import os
import secrets
from datetime import datetime, timedelta
from typing import Optional

import bcrypt
from fastapi import HTTPException, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError

security = HTTPBearer(auto_error=False)

JWT_ALGORITHM = "HS256"
JWT_EXPIRY_DAYS = 30


def get_jwt_secret() -> str:
    secret = os.getenv("JWT_SECRET")
    if not secret:
        secret = secrets.token_hex(32)
        os.environ["JWT_SECRET"] = secret
    return secret


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode(), password_hash.encode())


def create_access_token(user_id: str) -> str:
    secret = get_jwt_secret()
    expires = datetime.utcnow() + timedelta(days=JWT_EXPIRY_DAYS)
    payload = {
        "sub": user_id,
        "exp": expires,
        "iat": datetime.utcnow()
    }
    return jwt.encode(payload, secret, algorithm=JWT_ALGORITHM)


def decode_jwt(token: str) -> dict:
    try:
        secret = get_jwt_secret()
        payload = jwt.decode(token, secret, algorithms=[JWT_ALGORITHM])
        return payload
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {str(e)}"
        )


def verify_token(credentials: HTTPAuthorizationCredentials) -> dict:
    token = credentials.credentials
    payload = decode_jwt(token)
    
    if "exp" in payload:
        exp = datetime.fromtimestamp(payload["exp"])
        if exp < datetime.now():
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
    
    return payload


def get_user_id_from_token(credentials: HTTPAuthorizationCredentials) -> str:
    payload = verify_token(credentials)
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token: missing user ID"
        )
    return user_id


def optional_auth(credentials: Optional[HTTPAuthorizationCredentials] = Security(security)) -> Optional[str]:
    if not credentials:
        return None
    try:
        return get_user_id_from_token(credentials)
    except HTTPException:
        return None
