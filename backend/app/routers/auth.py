from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException, Request, status
from slowapi import Limiter
from slowapi.util import get_remote_address
from app.database import get_supabase
from app.middleware.auth import get_current_user, get_user_id
from app.models.schemas import (
    ApiResponse, AuthResponse, DietaryProfile, LoginRequest,
    RegisterRequest, UserProfile,
)

router = APIRouter(prefix="/auth", tags=["auth"])
limiter = Limiter(key_func=get_remote_address)


def _dietary_from_meta(meta: dict) -> DietaryProfile:
    return DietaryProfile(**{
        field: meta.get(field, [] if field == "excluded_ingredients" else False)
        for field in DietaryProfile.model_fields
    })


def _build_user_profile(user_obj: dict) -> UserProfile:
    meta = user_obj.get("user_metadata") or {}
    return UserProfile(
        id=user_obj["id"],
        email=user_obj["email"],
        dietary=_dietary_from_meta(meta),
    )


@router.post("/register", response_model=ApiResponse[AuthResponse])
@limiter.limit("10/minute")
async def register(request: Request, body: RegisterRequest):
    """Register a new user with email, password, and dietary preferences."""
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
        # Email confirmation is enabled — user was created, awaiting confirmation.
        return ApiResponse(data=None, error="confirm_email")

    return ApiResponse(data=AuthResponse(
        access_token=res.session.access_token,
        user=_build_user_profile(res.user.model_dump()),
    ))


@router.post("/login", response_model=ApiResponse[AuthResponse])
@limiter.limit("20/minute")
async def login(request: Request, body: LoginRequest):
    """Authenticate and return a JWT."""
    client = get_supabase()
    try:
        res = client.auth.sign_in_with_password({"email": body.email, "password": body.password})
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc))

    if not res.session or not res.user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    return ApiResponse(data=AuthResponse(
        access_token=res.session.access_token,
        user=_build_user_profile(res.user.model_dump()),
    ))


@router.post("/logout", response_model=ApiResponse[None])
async def logout(payload: dict = Depends(get_current_user)):
    """Invalidate the current session."""
    client = get_supabase()
    try:
        client.auth.sign_out()
    except Exception:
        pass
    return ApiResponse(data=None)


@router.get("/me", response_model=ApiResponse[UserProfile])
async def get_me(payload: dict = Depends(get_current_user)):
    """Return the current user's profile."""
    client = get_supabase()
    try:
        res = client.auth.get_user(payload.get("_token", ""))
    except Exception:
        res = None

    meta = payload.get("user_metadata") or {}
    profile = UserProfile(
        id=payload["sub"],
        email=payload.get("email", ""),
        dietary=_dietary_from_meta(meta),
    )
    return ApiResponse(data=profile)


@router.patch("/me", response_model=ApiResponse[UserProfile])
async def update_me(
    body: DietaryProfile,
    payload: dict = Depends(get_current_user),
):
    """Update the current user's dietary profile."""
    from app.database import get_supabase_admin
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
        dietary=_dietary_from_meta(updated_meta),
    )
    return ApiResponse(data=profile)
