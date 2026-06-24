"""Ingredient Explorer endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from app.middleware.auth import get_current_user
from app.models.explorer import (
    ExplorerExpandRequest,
    ExplorerExpandResponse,
    ExplorerRecommendResponse,
    ExplorerSearchResponse,
    ExplorerStartRequest,
    ExplorerStartResponse,
)
from app.models.common import ApiResponse
from app.services.explorer.service import (
    expand_explorer_session,
    recommend_recipes_from_ingredients,
    search_ingredient_names,
    start_explorer_session,
)

router = APIRouter(prefix="/explorer", tags=["explorer"])


@router.post("/start", response_model=ApiResponse[ExplorerStartResponse])
def start_explorer(
    body: ExplorerStartRequest,
    payload: dict = Depends(get_current_user),
) -> ApiResponse[ExplorerStartResponse]:
    return start_explorer_session(body, payload)


@router.post("/expand", response_model=ApiResponse[ExplorerExpandResponse])
def expand_explorer(
    body: ExplorerExpandRequest,
    payload: dict = Depends(get_current_user),
) -> ApiResponse[ExplorerExpandResponse]:
    return expand_explorer_session(body, payload)


@router.post("/recommend", response_model=ApiResponse[ExplorerRecommendResponse])
def recommend_from_ingredients(
    body: ExplorerExpandRequest,
    payload: dict = Depends(get_current_user),
) -> ApiResponse[ExplorerRecommendResponse]:
    return recommend_recipes_from_ingredients(body, payload)


@router.get("/search", response_model=ApiResponse[ExplorerSearchResponse])
def search_ingredients(
    q: str = Query(default="", min_length=0),
    payload: dict = Depends(get_current_user),
) -> ApiResponse[ExplorerSearchResponse]:
    return search_ingredient_names(q, payload)
