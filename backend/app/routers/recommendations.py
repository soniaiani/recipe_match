"""Bayesian recommendation session endpoints."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np
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
    question_by_id,
    restore_session,
    select_next_question,
)
from app.recommender.filters import filter_excluded_ingredients
from app.recommender.embeddings import encode_texts
from app.recommender.semantic_rerank import semantic_candidate_pool, semantic_rerank

router = APIRouter(prefix="/recommendations", tags=["recommendations"])

_SESSION_TTL = 3600  # 1 hour
SEMANTIC_POOL_SIZE = 500
PROTEIN_INFERENCE_THRESHOLD = 0.85
PROTEIN_SEMANTIC_MIN_SCORE = 0.45
PROTEIN_SEMANTIC_MIN_MARGIN = 0.08
PROTEIN_OPTION_TEXTS = {
    "chicken": "chicken recipe poultry",
    "beef_pork": "beef or pork recipe red meat",
    "fish_seafood": "fish or seafood recipe shrimp salmon",
    "meatless": "meatless vegetarian vegan recipe",
}


# ── IN-MEMORY SESSION STORE ───────────────────────────────────────────────────

@dataclass
class _RecSession:
    session_id: str
    recipe_ids: list[int]
    answers: dict[str, Any] = field(default_factory=dict)
    question_order: list[str] = field(default_factory=list)
    user_id: str | None = None
    semantic_query: str | None = None
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


def _q_to_schema(q: dict[str, Any]) -> RecQuestion:
    return RecQuestion(
        id=q["id"],
        type=q["type"],
        options=q.get("options"),
        any_option=q.get("any_option"),
    )


def _build_results(session: BayesianSession, semantic_query: str | None = None) -> list[RecScoredRecipe]:
    if semantic_query:
        session.answers["semantic_query"] = semantic_query
    try:
        ranked = semantic_rerank(session, n=10, candidate_n=150, beta=0.75, min_match_score=50.0)
    except Exception:
        ranked = session.top(n=10, min_match_score=50.0)

    return [
        RecScoredRecipe(
            id=r["id"],
            name=r.get("name", ""),
            image_url=r.get("image_url"),
            meal_type=r.get("meal_type"),
            cuisine=r.get("cuisine"),
            protein_type=r.get("protein_type"),
            match_score=score,
        )
        for r, score in ranked
    ]


def _attach_semantic_context(session: BayesianSession, semantic_query: str | None) -> None:
    if semantic_query:
        setattr(session, "semantic_query", semantic_query)


def _maybe_infer_protein_type(sess: _RecSession, bay: BayesianSession) -> None:
    if not sess.semantic_query:
        return
    if "protein_type" in sess.answers:
        return
    if "meal_type" not in sess.answers or "cuisine" not in sess.answers:
        return

    semantic_protein = _infer_protein_from_query_embedding(sess.semantic_query)
    if semantic_protein:
        _apply_inferred_protein(sess, bay, semantic_protein)
        return

    candidates = []
    meal_type = sess.answers["meal_type"]
    cuisines = sess.answers["cuisine"]
    selected_cuisines = cuisines if isinstance(cuisines, list) else [cuisines]

    for recipe in bay.recipes:
        if recipe.get("meal_type") != meal_type:
            continue
        if selected_cuisines != ["any"] and recipe.get("cuisine") not in selected_cuisines:
            continue
        protein = recipe.get("protein_type")
        if protein:
            candidates.append(protein)

    if len(candidates) < 20:
        return

    counts: dict[str, int] = {}
    for protein in candidates:
        counts[protein] = counts.get(protein, 0) + 1

    protein, count = max(counts.items(), key=lambda item: item[1])
    confidence = count / len(candidates)
    if confidence < PROTEIN_INFERENCE_THRESHOLD:
        return

    _apply_inferred_protein(sess, bay, protein)


def _infer_protein_from_query_embedding(query: str | None) -> str | None:
    query = (query or "").strip()
    if len(query) < 3:
        return None

    labels = list(PROTEIN_OPTION_TEXTS.keys())
    texts = [query] + [PROTEIN_OPTION_TEXTS[label] for label in labels]
    embeddings = encode_texts(texts)
    query_embedding = embeddings[0]
    option_embeddings = embeddings[1:]
    scores = option_embeddings @ query_embedding
    order = np.argsort(-scores)
    best_idx = int(order[0])
    second_idx = int(order[1])
    best_score = float(scores[best_idx])
    margin = best_score - float(scores[second_idx])

    if best_score >= PROTEIN_SEMANTIC_MIN_SCORE and margin >= PROTEIN_SEMANTIC_MIN_MARGIN:
        return labels[best_idx]
    return None


def _apply_inferred_protein(sess: _RecSession, bay: BayesianSession, protein: str) -> None:
    if "protein_type" in sess.answers:
        return

    question = question_by_id("protein_type")
    if not question:
        return

    answer = [protein]
    bay.update(question, answer)
    sess.answers["protein_type"] = answer
    sess.question_order.append("protein_type")


def _persist_session_async(sess: _RecSession, bay: BayesianSession, completed: bool) -> None:
    """Fire-and-forget persistence to recommendation_sessions."""
    try:
        if sess.semantic_query:
            bay.answers["semantic_query"] = sess.semantic_query
        admin = get_supabase_admin()
        try:
            top_ids = [
                r["id"]
                for r, _ in semantic_rerank(bay, n=10, candidate_n=150, beta=0.75, min_match_score=0.0)
            ]
        except Exception:
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
    except Exception:
        pass  # non-critical


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

    # Resolve dietary from explicit request body first, then authenticated user metadata.
    if body and body.dietary:
        dietary = body.dietary.model_dump()
    elif payload:
        meta = payload.get("user_metadata") or {}
        dietary = DietaryProfile(
            is_vegetarian=meta.get("is_vegetarian", False),
            is_vegan=meta.get("is_vegan", False),
            is_gluten_free=meta.get("is_gluten_free", False),
            is_dairy_free=meta.get("is_dairy_free", False),
            excluded_ingredients=meta.get("excluded_ingredients", []),
        ).model_dump()
    else:
        dietary = DietaryProfile().model_dump()

    filtered = _filter_by_dietary(all_recipes, dietary)
    if not filtered:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No recipes match your dietary requirements.",
        )

    semantic_query = body.semantic_query.strip() if body and body.semantic_query else None
    if semantic_query:
        try:
            pooled = semantic_candidate_pool(filtered, semantic_query, pool_n=SEMANTIC_POOL_SIZE)
            if len(pooled) >= 20:
                filtered = pooled
        except Exception:
            pass

    session_id = str(uuid.uuid4())
    bay = BayesianSession(filtered, weights)
    _attach_semantic_context(bay, semantic_query)

    first_q = select_next_question(bay)
    if not first_q:
        raise HTTPException(status_code=500, detail="No questions available")

    sess = _RecSession(
        session_id=session_id,
        recipe_ids=[r["id"] for r in filtered],
        user_id=user_id,
        semantic_query=semantic_query,
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

    # Validate question_id
    valid_ids = {q["id"] for q in QUESTION_BANK}
    if body.question_id not in valid_ids:
        raise HTTPException(status_code=400, detail="Unknown question_id")

    # Filter to this session's recipe subset
    recipe_id_set = set(sess.recipe_ids)
    recipes = [r for r in all_recipes if r["id"] in recipe_id_set]

    # Reconstruct session and apply new answer
    bay = restore_session(sess.answers, recipes, weights)
    _attach_semantic_context(bay, sess.semantic_query)
    q = next((q for q in QUESTION_BANK if q["id"] == body.question_id), None)
    if not q:
        raise HTTPException(status_code=400, detail="Unknown question_id")

    bay.update(q, body.answer)

    # Persist updated state
    sess.answers[body.question_id] = body.answer
    sess.question_order.append(body.question_id)
    sess.updated_at = time.monotonic()

    _maybe_infer_protein_type(sess, bay)

    if bay.should_stop():
        _persist_session_async(sess, bay, completed=True)
        ranked = _build_results(bay, sess.semantic_query)
        return ApiResponse(data=RecAnswerResponse(
            status="done",
            results=ranked,
            results_count=len(ranked),
            questions_asked=bay.q,
            entropy=round(bay.entropy(), 3),
        ))

    next_q = select_next_question(bay)
    if not next_q:
        _persist_session_async(sess, bay, completed=True)
        ranked = _build_results(bay, sess.semantic_query)
        return ApiResponse(data=RecAnswerResponse(
            status="done",
            results=ranked,
            results_count=len(ranked),
            questions_asked=bay.q,
            entropy=round(bay.entropy(), 3),
        ))

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

    recipe_id_set = set(sess.recipe_ids)
    recipes = [r for r in all_recipes if r["id"] in recipe_id_set]

    bay = restore_session(sess.answers, recipes, weights)
    _attach_semantic_context(bay, sess.semantic_query)
    ranked = _build_results(bay, sess.semantic_query)
    return ApiResponse(data=RecResultsResponse(results=ranked, results_count=len(ranked)))


@router.post("/interaction", response_model=ApiResponse[None])
async def record_interaction(
    body: RecInteractionRequest,
    payload: dict | None = Depends(get_current_user_optional),
):
    """Save a recipe interaction (view, like, save, cook, skip)."""
    valid_types = {"view", "like", "save", "cook", "skip"}
    if body.interaction_type not in valid_types:
        raise HTTPException(status_code=400, detail=f"interaction_type must be one of {valid_types}")

    user_id = payload.get("sub") if payload else None

    weights = {"view": 0.5, "like": 1.5, "save": 2.0, "cook": 3.0, "skip": -0.5}

    try:
        admin = get_supabase_admin()
        admin.table("recipe_interactions").insert({
            "user_id": user_id,
            "recipe_id": body.recipe_id,
            "interaction_type": body.interaction_type,
            "weight": weights.get(body.interaction_type, 1.0),
        }).execute()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return ApiResponse(data=None)
