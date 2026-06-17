"""For You recommendation endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Request

from app.middleware.auth import get_current_user
from app.models.common import ApiResponse
from app.models.foryou import ForYouResponse, TasteProfileRecipesResponse, TasteProfileResponse
from app.services.foryou.service import build_for_you_response
from app.services.foryou.taste_profile import build_taste_profile_response
from app.services.foryou.taste_profile_recipes import build_taste_profile_recipes_response

router = APIRouter(prefix="/foryou", tags=["foryou"])


@router.get("", response_model=ApiResponse[ForYouResponse])
async def for_you(request: Request, payload: dict = Depends(get_current_user)):
    """Return hybrid personalized recipe recommendations."""
    return await build_for_you_response(request, payload)


@router.get("/taste-profile", response_model=ApiResponse[TasteProfileResponse])
async def taste_profile(request: Request, payload: dict = Depends(get_current_user)):
    """Return the user's cluster-based taste profile."""
    return await build_taste_profile_response(request, payload)


@router.get("/taste-profile/recipes", response_model=ApiResponse[TasteProfileRecipesResponse])
async def taste_profile_recipes(
    request: Request,
    exclude_recipe_ids: list[int] = Query(default=[]),
    payload: dict = Depends(get_current_user),
):
    """Return quality-gated personalized recipes from the user's dominant clusters."""
    return await build_taste_profile_recipes_response(
        request,
        payload,
        set(exclude_recipe_ids),
    )
