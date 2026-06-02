"""Bayesian recommendation session endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from app.middleware.auth import get_current_user_optional
from app.models.common import ApiResponse
from app.models.recommendations import (
    RecAnswerRequest,
    RecAnswerResponse,
    RecInteractionRequest,
    RecResultsResponse,
    RecSessionStartRequest,
    RecSessionStartResponse,
)
from app.services.recommendation_service import (
    delete_recipe_interaction,
    get_recommendation_results,
    record_recipe_interaction,
    start_recommendation_session,
    submit_recommendation_answer,
)

router = APIRouter(prefix="/recommendations", tags=["recommendations"])


@router.post("/session/start", response_model=ApiResponse[RecSessionStartResponse])
async def start_session(
    request: Request,
    body: RecSessionStartRequest | None = None,
    payload: dict | None = Depends(get_current_user_optional),
):
    """Create a new Bayesian recommendation session and return the first question."""
    return start_recommendation_session(request, body, payload)


@router.post(
    "/session/{session_id}/answer",
    response_model=ApiResponse[RecAnswerResponse],
)
async def submit_answer(
    session_id: str,
    body: RecAnswerRequest,
    request: Request,
):
    """Submit an answer; returns next question or final results."""
    return submit_recommendation_answer(session_id, body, request)


@router.get(
    "/session/{session_id}/results",
    response_model=ApiResponse[RecResultsResponse],
)
async def get_results(session_id: str, request: Request):
    """Return top-10 results for a completed or in-progress session."""
    return get_recommendation_results(session_id, request)


@router.post("/interaction", response_model=ApiResponse[None])
async def record_interaction(
    body: RecInteractionRequest,
    payload: dict | None = Depends(get_current_user_optional),
):
    """Save a recipe interaction (view, save, cook, skip)."""
    return record_recipe_interaction(body, payload)


@router.delete("/interaction/{recipe_id}/{interaction_type}", response_model=ApiResponse[None])
async def delete_interaction(
    recipe_id: int,
    interaction_type: str,
    payload: dict | None = Depends(get_current_user_optional),
):
    """Delete interactions of a type for the current user and recipe."""
    return delete_recipe_interaction(recipe_id, interaction_type, payload)
