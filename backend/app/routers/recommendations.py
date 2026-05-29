"""Bayesian recommendation session endpoints."""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.database import get_supabase_admin
from app.middleware.auth import get_current_user_optional
from app.models.schemas import (
    ApiResponse,
    DietaryProfile,
    RecAnswerRequest,
    RecAnswerResponse,
    RecInteractionRequest,
    RecProgress,
    RecQuestion,
    RecResultsResponse,
    RecScoredRecipe,
    RecSessionStartRequest,
    RecSessionStartResponse,
)
from app.recommender.engine import (
    BayesianSession,
    MAX_QUESTIONS,
    QUESTION_BANK,
    restore_session,
    select_next_question,
)
from app.recommender.filters import filter_excluded_ingredients

router = APIRouter(prefix="/recommendations", tags=["recommendations"])

_SESSION_TTL = 3600  # 1 hour
_VALID_INTERACTION_TYPES = {"view", "save", "cook", "skip"}
_INTERACTION_WEIGHTS = {"view": 0.5, "save": 2.0, "cook": 3.0, "skip": -0.5}
_QUESTION_IDS = {q["id"] for q in QUESTION_BANK}
logger = logging.getLogger(__name__)


def _match_score_percent(score: float) -> float:
    return round(min(max(float(score), 0.0), 100.0), 1)


# ── IN-MEMORY SESSION STORE ───────────────────────────────────────────────────

@dataclass
class _RecSession:
    session_id: str
    recipe_ids: list[int]
    answers: dict[str, Any] = field(default_factory=dict)
    question_order: list[str] = field(default_factory=list)
    user_id: str | None = None
    created_at: float = field(default_factory=time.monotonic)
    updated_at: float = field(default_factory=time.monotonic)


_sessions: dict[str, _RecSession] = {}


def _evict() -> None:
    now = time.monotonic()
    expired = [sid for sid, s in _sessions.items() if now - s.updated_at > _SESSION_TTL]
    for sid in expired:
        del _sessions[sid]


def _get_session(session_id: str) -> _RecSession:
    _evict()
    sess = _sessions.get(session_id)
    if not sess:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    return sess


# ── HELPERS ───────────────────────────────────────────────────────────────────

def _require_engine(request: Request) -> tuple[list[dict], dict[str, float]]:
    recipes = getattr(request.app.state, "rec_recipes", [])
    weights = getattr(request.app.state, "rec_weights", {})
    if not recipes:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Recommendation engine not ready. Try again in a moment.",
        )
    return recipes, weights


def _filter_by_dietary(recipes: list[dict], dietary: dict[str, Any]) -> list[dict]:
    out = recipes
    if dietary.get("is_vegetarian"):
        out = [r for r in out if r.get("is_vegetarian")]
    if dietary.get("is_vegan"):
        out = [r for r in out if r.get("is_vegan")]
    if dietary.get("is_gluten_free"):
        out = [r for r in out if r.get("is_gluten_free")]
    if dietary.get("is_dairy_free"):
        out = [r for r in out if r.get("is_dairy_free")]
    return filter_excluded_ingredients(out, dietary.get("excluded_ingredients"))


def _dietary_from_payload(payload: dict) -> dict[str, Any]:
    meta = payload.get("user_metadata") or {}
    return DietaryProfile(
        is_vegetarian=meta.get("is_vegetarian", False),
        is_vegan=meta.get("is_vegan", False),
        is_gluten_free=meta.get("is_gluten_free", False),
        is_dairy_free=meta.get("is_dairy_free", False),
        excluded_ingredients=meta.get("excluded_ingredients", []),
    ).model_dump()


def _resolve_dietary(
    body: RecSessionStartRequest | None,
    payload: dict | None,
) -> dict[str, Any]:
    if payload:
        return _dietary_from_payload(payload)
    if body and body.dietary:
        return body.dietary.model_dump()
    return DietaryProfile().model_dump()


def _q_to_schema(q: dict[str, Any]) -> RecQuestion:
    return RecQuestion(
        id=q["id"],
        type=q["type"],
        options=q.get("options"),
        any_option=q.get("any_option"),
    )


def _build_results(session: BayesianSession) -> list[RecScoredRecipe]:
    ranked = session.top(n=10, min_match_score=50.0)
    return [
        RecScoredRecipe(
            id=r["id"],
            name=r.get("name", ""),
            image_url=r.get("image_url"),
            meal_type=r.get("meal_type"),
            cuisine=r.get("cuisine"),
            protein_type=r.get("protein_type"),
            match_score=_match_score_percent(score),
        )
        for r, score in ranked
    ]


def _session_recipes(sess: _RecSession, all_recipes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    recipe_ids = set(sess.recipe_ids)
    return [recipe for recipe in all_recipes if recipe["id"] in recipe_ids]


def _done_response(bay: BayesianSession) -> ApiResponse[RecAnswerResponse]:
    ranked = _build_results(bay)
    return ApiResponse(data=RecAnswerResponse(
        status="done",
        results=ranked,
        results_count=len(ranked),
        questions_asked=bay.q,
        entropy=round(bay.entropy(), 3),
    ))


def _validate_interaction_type(interaction_type: str) -> None:
    if interaction_type not in _VALID_INTERACTION_TYPES:
        allowed = ", ".join(sorted(_VALID_INTERACTION_TYPES))
        raise HTTPException(status_code=400, detail=f"interaction_type must be one of: {allowed}")


def _persist_session_async(sess: _RecSession, bay: BayesianSession, completed: bool) -> None:
    """Fire-and-forget persistence to recommendation_sessions."""
    try:
        admin = get_supabase_admin()
        top_ids = [r["id"] for r, _ in bay.top(n=10, min_match_score=0.0)]
        payload: dict[str, Any] = {
            "id": sess.session_id,
            "user_id": sess.user_id,
            "answers": sess.answers,
            "question_order": sess.question_order,
            "questions_asked": bay.q,
            "entropy_final": round(bay.entropy(), 4),
            "top_recipe_ids": top_ids,
        }
        if completed:
            from datetime import datetime, timezone
            payload["completed_at"] = datetime.now(timezone.utc).isoformat()
        admin.table("recommendation_sessions").upsert(payload).execute()
    except Exception as exc:
        logger.warning("Failed to persist recommendation session %s: %s", sess.session_id, exc)


# ── ENDPOINTS ─────────────────────────────────────────────────────────────────

@router.post("/session/start", response_model=ApiResponse[RecSessionStartResponse])
async def start_session(
    request: Request,
    body: RecSessionStartRequest | None = None,
    payload: dict | None = Depends(get_current_user_optional),
):
    """Create a new Bayesian recommendation session and return the first question."""
    all_recipes, weights = _require_engine(request)
    user_id = payload.get("sub") if payload else None

    # Authenticated profile metadata is the source of truth for hard filters.
    dietary = _resolve_dietary(body, payload)

    filtered = _filter_by_dietary(all_recipes, dietary)
    if not filtered:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No recipes match your dietary requirements.",
        )

    session_id = str(uuid.uuid4())
    bay = BayesianSession(filtered, weights)

    first_q = select_next_question(bay)
    if not first_q:
        raise HTTPException(status_code=500, detail="No questions available")

    sess = _RecSession(
        session_id=session_id,
        recipe_ids=[r["id"] for r in filtered],
        user_id=user_id,
    )
    _sessions[session_id] = sess

    return ApiResponse(data=RecSessionStartResponse(
        session_id=session_id,
        question=_q_to_schema(first_q),
        progress=RecProgress(current=1, max=MAX_QUESTIONS),
    ))


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
    all_recipes, weights = _require_engine(request)
    sess = _get_session(session_id)

    if body.question_id not in _QUESTION_IDS:
        raise HTTPException(status_code=400, detail="Unknown question_id")

    bay = restore_session(sess.answers, _session_recipes(sess, all_recipes), weights)
    q = next((q for q in QUESTION_BANK if q["id"] == body.question_id), None)
    if not q:
        raise HTTPException(status_code=400, detail="Unknown question_id")

    bay.update(q, body.answer)

    # Persist updated state
    sess.answers[body.question_id] = body.answer
    sess.question_order.append(body.question_id)
    sess.updated_at = time.monotonic()

    if bay.should_stop():
        _persist_session_async(sess, bay, completed=True)
        return _done_response(bay)

    next_q = select_next_question(bay)
    if not next_q:
        _persist_session_async(sess, bay, completed=True)
        return _done_response(bay)

    _persist_session_async(sess, bay, completed=False)
    return ApiResponse(data=RecAnswerResponse(
        status="continue",
        question=_q_to_schema(next_q),
        entropy=round(bay.entropy(), 3),
        questions_asked=bay.q,
        progress=RecProgress(current=bay.q + 1, max=MAX_QUESTIONS),
    ))


@router.get(
    "/session/{session_id}/results",
    response_model=ApiResponse[RecResultsResponse],
)
async def get_results(session_id: str, request: Request):
    """Return top-10 results for a completed or in-progress session."""
    all_recipes, weights = _require_engine(request)
    sess = _get_session(session_id)

    bay = restore_session(sess.answers, _session_recipes(sess, all_recipes), weights)
    ranked = _build_results(bay)
    return ApiResponse(data=RecResultsResponse(results=ranked, results_count=len(ranked)))


@router.post("/interaction", response_model=ApiResponse[None])
async def record_interaction(
    body: RecInteractionRequest,
    payload: dict | None = Depends(get_current_user_optional),
):
    """Save a recipe interaction (view, save, cook, skip)."""
    _validate_interaction_type(body.interaction_type)

    user_id = payload.get("sub") if payload else None

    try:
        admin = get_supabase_admin()
        admin.table("recipe_interactions").insert({
            "user_id": user_id,
            "recipe_id": body.recipe_id,
            "interaction_type": body.interaction_type,
            "weight": _INTERACTION_WEIGHTS[body.interaction_type],
        }).execute()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return ApiResponse(data=None)


@router.delete("/interaction/{recipe_id}/{interaction_type}", response_model=ApiResponse[None])
async def delete_interaction(
    recipe_id: int,
    interaction_type: str,
    payload: dict | None = Depends(get_current_user_optional),
):
    """Delete interactions of a type for the current user and recipe."""
    _validate_interaction_type(interaction_type)

    user_id = payload.get("sub") if payload else None
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")

    try:
        admin = get_supabase_admin()
        (
            admin.table("recipe_interactions")
            .delete()
            .eq("user_id", user_id)
            .eq("recipe_id", recipe_id)
            .eq("interaction_type", interaction_type)
            .execute()
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return ApiResponse(data=None)
