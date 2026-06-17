from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.middleware.auth import get_current_user
from app.models.auth import AuthResponse, DietaryProfile, LoginRequest, RegisterRequest, UserProfile
from app.models.common import ApiResponse
from app.services.auth.service import (
    current_user_profile,
    login_user,
    logout_user,
    register_user,
    update_user_profile,
)

router = APIRouter(prefix="/auth", tags=["auth"])
limiter = Limiter(key_func=get_remote_address)


@router.post("/register", response_model=ApiResponse[AuthResponse])
@limiter.limit("10/minute")
async def register(request: Request, body: RegisterRequest):
    """Register a new user with email, password, and dietary preferences."""
    return register_user(body)


@router.post("/login", response_model=ApiResponse[AuthResponse])
@limiter.limit("20/minute")
async def login(request: Request, body: LoginRequest):
    """Authenticate and return a JWT."""
    return login_user(body)


@router.post("/logout", response_model=ApiResponse[None])
async def logout(payload: dict = Depends(get_current_user)):
    """Invalidate the current session."""
    return logout_user(payload)


@router.get("/me", response_model=ApiResponse[UserProfile])
async def get_me(payload: dict = Depends(get_current_user)):
    """Return the current user's profile."""
    return current_user_profile(payload)


@router.patch("/me", response_model=ApiResponse[UserProfile])
async def update_me(
    body: DietaryProfile,
    payload: dict = Depends(get_current_user),
):
    """Update the current user's dietary profile."""
    return update_user_profile(body, payload)
