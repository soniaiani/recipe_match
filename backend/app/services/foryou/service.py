"""For You recommendation orchestration."""
from __future__ import annotations

from typing import Any

from fastapi import Request

from app.database import get_supabase_admin
from app.models.common import ApiResponse
from app.models.foryou import ForYouResponse
from app.models.recipes import RecipeSummary
from app.services.foryou.common import (
    RECOMMENDATION_COUNT,
    hard_filter,
    load_recipes,
)
from app.services.foryou.profiles import (
    build_answers_profile,
    build_explorer_intents,
    build_interaction_profile,
    build_semantic_profiles,
)
from app.services.foryou.ranking import (
    compute_rank_scores,
    compute_weights,
    load_popular_recipes,
    promote_recent_matches,
    score_candidate,
    select_candidates,
    select_cluster_aware_top,
    select_diverse_top,
    semantic_scores_for_profiles,
)


def _flatten_explorer_intents(explorer_intents: list[list[str]]) -> list[str]:
    return [
        ingredient
        for intent in explorer_intents
        for ingredient in intent
    ]


def _cluster_context(request: Request) -> tuple[bool, dict[int, int]]:
    recipe_clusters = getattr(request.app.state, "recipe_clusters", {}) or {}
    cluster_aware = bool(getattr(request.app.state, "cluster_aware_fyp", False))
    return cluster_aware, recipe_clusters


def _to_for_you_response(recipes: list[dict[str, Any]]) -> ApiResponse[ForYouResponse]:
    return ApiResponse(data=ForYouResponse(
        recipes=[RecipeSummary(**recipe) for recipe in recipes]
    ))


def _select_final_recipes(
    ranked: list[dict[str, Any]],
    scores: dict[int, float],
    cluster_aware: bool,
    recipe_clusters: dict[int, int],
) -> list[dict[str, Any]]:
    if cluster_aware and recipe_clusters:
        return select_cluster_aware_top(
            ranked,
            scores,
            recipe_clusters,
            RECOMMENDATION_COUNT,
        )
    return select_diverse_top(ranked, RECOMMENDATION_COUNT)


def _cold_start_recommendations(
    admin: Any,
    recipes: list[dict[str, Any]],
    cluster_aware: bool,
    recipe_clusters: dict[int, int],
) -> list[dict[str, Any]]:
    ranked = load_popular_recipes(admin, recipes)
    return _select_final_recipes(
        ranked,
        compute_rank_scores(ranked),
        cluster_aware,
        recipe_clusters,
    )


def _candidate_scores(
    candidates: list[dict[str, Any]],
    interaction_profile: dict[str, Any],
    answers_profile: dict[str, Any],
    semantic_scores: dict[int, float],
    weights: dict[str, float],
) -> dict[int, float]:
    return {
        int(recipe["id"]): score_candidate(
            recipe,
            interaction_profile,
            answers_profile,
            semantic_scores,
            weights,
        )
        for recipe in candidates
        if recipe.get("id") is not None
    }


def _rank_by_score(
    candidates: list[dict[str, Any]],
    scores: dict[int, float],
) -> list[dict[str, Any]]:
    return sorted(
        candidates,
        key=lambda recipe: scores.get(int(recipe.get("id", -1)), 0.0),
        reverse=True,
    )


def _personalized_weights(
    n_interactions: int,
    n_sessions: int,
    explorer_ingredients: list[str],
) -> dict[str, float]:
    if n_interactions <= 0 and n_sessions <= 0 and explorer_ingredients:
        return {"interactions": 0.0, "answers": 0.0, "semantic": 1.0}
    return compute_weights(n_interactions, n_sessions)


def _personalized_recommendations(
    admin: Any,
    recipes: list[dict[str, Any]],
    interaction_profile: dict[str, Any],
    answers_profile: dict[str, Any],
    explorer_intents: list[list[str]],
    cluster_aware: bool,
    recipe_clusters: dict[int, int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, float]]:
    n_interactions = int(interaction_profile.get("n_interactions") or 0)
    n_sessions = int(answers_profile.get("n_sessions") or 0)
    explorer_ingredients = _flatten_explorer_intents(explorer_intents)

    semantic_profiles = build_semantic_profiles(
        interaction_profile,
        answers_profile,
        explorer_intents,
    )
    semantic_scores = semantic_scores_for_profiles(admin, semantic_profiles)
    weights = _personalized_weights(n_interactions, n_sessions, explorer_ingredients)

    candidates = select_candidates(
        recipes,
        semantic_scores,
        interaction_profile,
        answers_profile,
    )
    scores = _candidate_scores(
        candidates,
        interaction_profile,
        answers_profile,
        semantic_scores,
        weights,
    )
    final_recipes = _select_final_recipes(
        _rank_by_score(candidates, scores),
        scores,
        cluster_aware,
        recipe_clusters,
    )
    final_recipes = promote_recent_matches(
        final_recipes,
        interaction_profile,
        RECOMMENDATION_COUNT,
    )
    return final_recipes, candidates, weights


async def build_for_you_response(request: Request, payload: dict) -> ApiResponse[ForYouResponse]:
    """Build hybrid personalized recipe recommendations."""
    admin = get_supabase_admin()
    user_id = payload["sub"]
    meta = payload.get("user_metadata") or {}

    interaction_profile = build_interaction_profile(user_id, admin)
    answers_profile = build_answers_profile(user_id, admin)
    saved_ids = interaction_profile.get("saved_ids", set()) or set()

    recipes = hard_filter(load_recipes(request, admin), saved_ids, meta)

    n_interactions = int(interaction_profile.get("n_interactions") or 0)
    n_sessions = int(answers_profile.get("n_sessions") or 0)
    explorer_intents = build_explorer_intents(
        user_id,
        admin,
        meta.get("excluded_ingredients", []),
    )
    explorer_ingredients = _flatten_explorer_intents(explorer_intents)
    cluster_aware, recipe_clusters = _cluster_context(request)

    if n_interactions <= 0 and n_sessions <= 0 and not explorer_ingredients:
        print(f"[foryou] cold start for user={user_id}")
        final_recipes = _cold_start_recommendations(
            admin,
            recipes,
            cluster_aware,
            recipe_clusters,
        )
        return _to_for_you_response(final_recipes)

    final_recipes, candidates, weights = _personalized_recommendations(
        admin,
        recipes,
        interaction_profile,
        answers_profile,
        explorer_intents,
        cluster_aware,
        recipe_clusters,
    )

    print(
        "[foryou] user="
        f"{user_id} interactions={n_interactions} sessions={n_sessions} "
        f"explorer_ingredients={len(explorer_ingredients)} "
        f"candidates={len(candidates)} final={len(final_recipes)} "
        f"cluster_aware={cluster_aware} weights={weights}"
    )
    return _to_for_you_response(final_recipes)
