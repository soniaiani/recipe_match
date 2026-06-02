"""Authentication business logic."""
from __future__ import annotations

from fastapi import HTTPException, status

from app.database import get_supabase, get_supabase_admin
from app.models.auth import (
    AuthResponse,
    DietaryProfile,
    LoginRequest,
    RegisterRequest,
    UserProfile,
)
from app.models.common import ApiResponse


def dietary_from_meta(meta: dict) -> DietaryProfile:
    return DietaryProfile(**{
        field: meta.get(field, [] if field == "excluded_ingredients" else False)
        for field in DietaryProfile.model_fields
    })


def build_user_profile(user_obj: dict) -> UserProfile:
    meta = user_obj.get("user_metadata") or {}
    return UserProfile(
        id=user_obj["id"],
        email=user_obj["email"],
        dietary=dietary_from_meta(meta),
    )


def register_user(body: RegisterRequest) -> ApiResponse[AuthResponse]:
    client = get_supabase()
    try:
        res = client.auth.sign_up({
            "email": body.email,
            "password": body.password,
            "options": {"data": body.dietary.model_dump()},
        })
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    if not res.user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Registration failed. Please try again.",
        )

    if not res.session:
        return ApiResponse(data=None, error="confirm_email")

    return ApiResponse(data=AuthResponse(
        access_token=res.session.access_token,
        user=build_user_profile(res.user.model_dump()),
    ))


def login_user(body: LoginRequest) -> ApiResponse[AuthResponse]:
    client = get_supabase()
    try:
        res = client.auth.sign_in_with_password({"email": body.email, "password": body.password})
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc))

    if not res.session or not res.user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    return ApiResponse(data=AuthResponse(
        access_token=res.session.access_token,
        user=build_user_profile(res.user.model_dump()),
    ))


def logout_user() -> ApiResponse[None]:
    client = get_supabase()
    try:
        client.auth.sign_out()
    except Exception:
        pass
    return ApiResponse(data=None)


def current_user_profile(payload: dict) -> ApiResponse[UserProfile]:
    meta = payload.get("user_metadata") or {}
    profile = UserProfile(
        id=payload["sub"],
        email=payload.get("email", ""),
        dietary=dietary_from_meta(meta),
    )
    return ApiResponse(data=profile)


def update_user_profile(body: DietaryProfile, payload: dict) -> ApiResponse[UserProfile]:
    admin = get_supabase_admin()
    user_id = payload["sub"]
    try:
        res = admin.auth.admin.update_user_by_id(user_id, {"user_metadata": body.model_dump()})
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    updated_meta = res.user.user_metadata or {}
    profile = UserProfile(
        id=user_id,
        email=payload.get("email", ""),
        dietary=dietary_from_meta(updated_meta),
    )
    return ApiResponse(data=profile)
