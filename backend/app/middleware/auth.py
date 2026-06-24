"""JWT validation middleware using Supabase."""
from __future__ import annotations
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from app.database import get_supabase_admin

_bearer = HTTPBearer()


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
) -> dict:
    """Validate the Supabase JWT by calling Supabase auth.get_user()."""
    token = credentials.credentials
    admin = get_supabase_admin()
    
    try:
        response = admin.auth.get_user(token)
        if not response or not response.user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
            )
        return {
            "sub": response.user.id,
            "email": response.user.email,
            "user_metadata": response.user.user_metadata or {},
            "access_token": token,
        }
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )


def get_user_id(payload: dict = Depends(get_current_user)) -> str:
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing user id"
        )
    return user_id
