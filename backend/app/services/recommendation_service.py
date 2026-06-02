"""Business logic for interactive Bayesian recommendations."""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from fastapi import HTTPException, Request, status

from app.database import get_supabase_admin
from app.models.auth import DietaryProfile
from app.models.common import ApiResponse
from app.models.recommendations import (
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

_SESSION_TTL = 3600
_VALID_INTERACTION_TYPES = {"view", "save", "cook", "skip"}
_INTERACTION_WEIGHTS = {"view": 0.5, "save": 2.0, "cook": 3.0, "skip": -0.5}
_QUESTION_IDS = {question["id"] for question in QUESTION_BANK}

logger = logging.getLogger(__name__)


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


def _match_score_percent(score: float) -> float:
    return round(min(max(float(score), 0.0), 100.0), 1)


def _evict_expired_sessions() -> None:
    now = time.monotonic()
    expired = [
        session_id
        for session_id, session in _sessions.items()
        if now - session.updated_at > _SESSION_TTL
    ]
    for session_id in expired:
        del _sessions[session_id]


def _get_session(session_id: str) -> _RecSession:
    _evict_expired_sessions()
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )
    return session


def _require_engine(request: Request) -> tuple[list[dict[str, Any]], dict[str, float]]:
    recipes = getattr(request.app.state, "rec_recipes", [])
    weights = getattr(request.app.state, "rec_weights", {})
    if not recipes:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Recommendation engine not ready. Try again in a moment.",
        )
    return recipes, weights


def _filter_by_dietary(
    recipes: list[dict[str, Any]],
    dietary: dict[str, Any],
) -> list[dict[str, Any]]:
    out = recipes
    if dietary.get("is_vegetarian"):
        out = [recipe for recipe in out if recipe.get("is_vegetarian")]
    if dietary.get("is_vegan"):
        out = [recipe for recipe in out if recipe.get("is_vegan")]
    if dietary.get("is_gluten_free"):
        out = [recipe for recipe in out if recipe.get("is_gluten_free")]
    if dietary.get("is_dairy_free"):
        out = [recipe for recipe in out if recipe.get("is_dairy_free")]
    return filter_excluded_ingredients(out, dietary.get("excluded_ingredients"))


def _dietary_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
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
    payload: dict[str, Any] | None,
) -> dict[str, Any]:
    if payload:
        return _dietary_from_payload(payload)
    if body and body.dietary:
        return body.dietary.model_dump()
    return DietaryProfile().model_dump()


def _q_to_schema(question: dict[str, Any]) -> RecQuestion:
    return RecQuestion(
        id=question["id"],
        type=question["type"],
        options=question.get("options"),
        any_option=question.get("any_option"),
    )


def _build_results(session: BayesianSession) -> list[RecScoredRecipe]:
    ranked = session.top(n=10, min_match_score=50.0)
    return [
        RecScoredRecipe(
            id=recipe["id"],
            name=recipe.get("name", ""),
            image_url=recipe.get("image_url"),
            meal_type=recipe.get("meal_type"),
            cuisine=recipe.get("cuisine"),
            protein_type=recipe.get("protein_type"),
            match_score=_match_score_percent(score),
        )
        for recipe, score in ranked
    ]


def _session_recipes(
    session: _RecSession,
    all_recipes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    recipe_ids = set(session.recipe_ids)
    return [recipe for recipe in all_recipes if recipe["id"] in recipe_ids]


def _done_response(bayesian_session: BayesianSession) -> ApiResponse[RecAnswerResponse]:
    ranked = _build_results(bayesian_session)
    return ApiResponse(data=RecAnswerResponse(
        status="done",
        results=ranked,
        results_count=len(ranked),
        questions_asked=bayesian_session.q,
        entropy=round(bayesian_session.entropy(), 3),
    ))


def _persist_session(session: _RecSession, bayesian_session: BayesianSession, completed: bool) -> None:
    """Persist the in-memory session summary to Supabase."""
    try:
        admin = get_supabase_admin()
        top_ids = [
            recipe["id"]
            for recipe, _ in bayesian_session.top(n=10, min_match_score=0.0)
        ]
        payload: dict[str, Any] = {
            "id": session.session_id,
            "user_id": session.user_id,
            "answers": session.answers,
            "question_order": session.question_order,
            "questions_asked": bayesian_session.q,
            "entropy_final": round(bayesian_session.entropy(), 4),
            "top_recipe_ids": top_ids,
        }
        if completed:
            payload["completed_at"] = datetime.now(timezone.utc).isoformat()
        admin.table("recommendation_sessions").upsert(payload).execute()
    except Exception as exc:
        logger.warning(
            "Failed to persist recommendation session %s: %s",
            session.session_id,
            exc,
        )


def _restore_bayesian_session(
    session: _RecSession,
    all_recipes: list[dict[str, Any]],
    weights: dict[str, float],
) -> BayesianSession:
    return restore_session(session.answers, _session_recipes(session, all_recipes), weights)


def _question_by_id(question_id: str) -> dict[str, Any]:
    question = next((item for item in QUESTION_BANK if item["id"] == question_id), None)
    if not question:
        raise HTTPException(status_code=400, detail="Unknown question_id")
    return question


def start_recommendation_session(
    request: Request,
    body: RecSessionStartRequest | None,
    payload: dict[str, Any] | None,
) -> ApiResponse[RecSessionStartResponse]:
    all_recipes, weights = _require_engine(request)
    user_id = payload.get("sub") if payload else None

    dietary = _resolve_dietary(body, payload)
    filtered = _filter_by_dietary(all_recipes, dietary)
    if not filtered:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No recipes match your dietary requirements.",
        )

    bayesian_session = BayesianSession(filtered, weights)
    first_question = select_next_question(bayesian_session)
    if not first_question:
        raise HTTPException(status_code=500, detail="No questions available")

    session_id = str(uuid.uuid4())
    _sessions[session_id] = _RecSession(
        session_id=session_id,
        recipe_ids=[recipe["id"] for recipe in filtered],
        user_id=user_id,
    )

    return ApiResponse(data=RecSessionStartResponse(
        session_id=session_id,
        question=_q_to_schema(first_question),
        progress=RecProgress(current=1, max=MAX_QUESTIONS),
    ))


def submit_recommendation_answer(
    session_id: str,
    body: RecAnswerRequest,
    request: Request,
) -> ApiResponse[RecAnswerResponse]:
    all_recipes, weights = _require_engine(request)
    session = _get_session(session_id)

    if body.question_id not in _QUESTION_IDS:
        raise HTTPException(status_code=400, detail="Unknown question_id")

    bayesian_session = _restore_bayesian_session(session, all_recipes, weights)
    question = _question_by_id(body.question_id)
    bayesian_session.update(question, body.answer)

    session.answers[body.question_id] = body.answer
    session.question_order.append(body.question_id)
    session.updated_at = time.monotonic()

    if bayesian_session.should_stop():
        _persist_session(session, bayesian_session, completed=True)
        return _done_response(bayesian_session)

    next_question = select_next_question(bayesian_session)
    if not next_question:
        _persist_session(session, bayesian_session, completed=True)
        return _done_response(bayesian_session)

    _persist_session(session, bayesian_session, completed=False)
    return ApiResponse(data=RecAnswerResponse(
        status="continue",
        question=_q_to_schema(next_question),
        entropy=round(bayesian_session.entropy(), 3),
        questions_asked=bayesian_session.q,
        progress=RecProgress(current=bayesian_session.q + 1, max=MAX_QUESTIONS),
    ))


def get_recommendation_results(
    session_id: str,
    request: Request,
) -> ApiResponse[RecResultsResponse]:
    all_recipes, weights = _require_engine(request)
    session = _get_session(session_id)
    bayesian_session = _restore_bayesian_session(session, all_recipes, weights)
    ranked = _build_results(bayesian_session)
    return ApiResponse(data=RecResultsResponse(
        results=ranked,
        results_count=len(ranked),
    ))


def validate_interaction_type(interaction_type: str) -> None:
    if interaction_type not in _VALID_INTERACTION_TYPES:
        allowed = ", ".join(sorted(_VALID_INTERACTION_TYPES))
        raise HTTPException(
            status_code=400,
            detail=f"interaction_type must be one of: {allowed}",
        )


def record_recipe_interaction(
    body: RecInteractionRequest,
    payload: dict[str, Any] | None,
) -> ApiResponse[None]:
    validate_interaction_type(body.interaction_type)
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


def delete_recipe_interaction(
    recipe_id: int,
    interaction_type: str,
    payload: dict[str, Any] | None,
) -> ApiResponse[None]:
    validate_interaction_type(interaction_type)
    user_id = payload.get("sub") if payload else None
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )

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
