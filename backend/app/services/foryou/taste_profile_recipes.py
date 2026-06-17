"""Personalized recipe recommendations for the user's dominant taste clusters."""
from __future__ import annotations

from typing import Any

from fastapi import Request

from app.database import get_supabase_admin
from app.models.common import ApiResponse
from app.models.foryou import TasteClusterRecipes, TasteProfileRecipesResponse
from app.models.recipes import RecipeSummary
from app.services.foryou.common import hard_filter, load_recipes
from app.services.foryou.profiles import (
    build_answers_profile,
    build_explorer_intents,
    build_interaction_profile,
    build_semantic_profiles,
)
from app.services.foryou.ranking import (
    compute_weights,
    score_candidate,
    semantic_scores_for_profiles,
)
from app.services.foryou.taste_profile import (
    _build_declared_source_profiles,
    _fetch_cached_profile,
    _fetch_explorer_session_count,
    _fetch_profile_events,
    _numeric_profile,
)

_RECIPES_PER_CLUSTER = 3
_MIN_RECIPES_PER_CLUSTER = 2
_RELATIVE_QUALITY_THRESHOLD = 0.90


async def build_taste_profile_recipes_response(
    request: Request,
    payload: dict[str, Any],
    exclude_recipe_ids: set[int],
) -> ApiResponse[TasteProfileRecipesResponse]:
    user_id = payload.get("sub")
    if not user_id:
        return ApiResponse(data=TasteProfileRecipesResponse())

    recipe_clusters = getattr(request.app.state, "recipe_clusters", {}) or {}
    recipe_vectors = getattr(request.app.state, "recipe_cluster_vectors", {}) or {}
    cluster_centroids = getattr(request.app.state, "recipe_cluster_centroids", {}) or {}
    cluster_profiles = getattr(request.app.state, "recipe_cluster_profiles", {}) or {}
    recipes = getattr(request.app.state, "rec_recipes", []) or []
    model_version = getattr(request.app.state, "recipe_cluster_model_version", None)
    if (
        not recipe_clusters
        or not recipe_vectors
        or not cluster_centroids
        or not cluster_profiles
        or not recipes
        or not model_version
    ):
        return ApiResponse(data=TasteProfileRecipesResponse())

    admin = get_supabase_admin()
    events = _fetch_profile_events(user_id, admin)
    answers_profile = build_answers_profile(user_id, admin)
    meta = payload.get("user_metadata") or {}
    explorer_intents = build_explorer_intents(
        user_id,
        admin,
        meta.get("excluded_ingredients", []),
    )
    source_profiles = _build_declared_source_profiles(
        admin,
        recipes,
        recipe_vectors,
        answers_profile,
        explorer_intents,
        _fetch_explorer_session_count(user_id, admin),
    )
    cache_row = _fetch_cached_profile(user_id, model_version, admin)
    numeric = _numeric_profile(
        events,
        recipe_vectors,
        cluster_centroids,
        cluster_profiles,
        previous_behavior_centroid=(
            cache_row.get("behavior_centroid_vector") or cache_row.get("centroid_vector")
            if cache_row else None
        ),
        behavior_centroid_updated_at=(
            cache_row.get("behavior_centroid_updated_at") or cache_row.get("centroid_updated_at")
            if cache_row else None
        ),
        find_centroid=source_profiles["find_centroid"],
        find_support=source_profiles["find_support"],
        explorer_centroid=source_profiles["explorer_centroid"],
        explorer_support=source_profiles["explorer_support"],
    )
    if numeric is None:
        return ApiResponse(data=TasteProfileRecipesResponse())

    interaction_profile = build_interaction_profile(user_id, admin)
    interacted_ids = {
        int(event["recipe_id"])
        for event in events
        if event.get("recipe_id") is not None
    }
    filtered = hard_filter(
        load_recipes(request, admin),
        interacted_ids | set(interaction_profile.get("saved_ids", set()) or set()),
        meta,
    )

    semantic_profiles = build_semantic_profiles(
        interaction_profile,
        answers_profile,
        explorer_intents,
    )
    semantic_scores = semantic_scores_for_profiles(admin, semantic_profiles)
    explorer_ingredients = [ingredient for intent in explorer_intents for ingredient in intent]
    weights = _personalized_weights(
        int(interaction_profile.get("n_interactions") or 0),
        int(answers_profile.get("n_sessions") or 0),
        explorer_ingredients,
    )
    scores = {
        int(recipe["id"]): score_candidate(
            recipe,
            interaction_profile,
            answers_profile,
            semantic_scores,
            weights,
        )
        for recipe in filtered
        if recipe.get("id") is not None
    }
    recipes_by_cluster: dict[int, list[dict[str, Any]]] = {}
    top_cluster_ids = {cluster.cluster_id for cluster in numeric["top_clusters"]}
    for recipe in filtered:
        recipe_id = int(recipe["id"])
        cluster_id = recipe_clusters.get(recipe_id)
        if cluster_id in top_cluster_ids:
            recipes_by_cluster.setdefault(cluster_id, []).append(recipe)

    selected_ids: set[int] = set()
    cluster_results: list[TasteClusterRecipes] = []
    for cluster in numeric["top_clusters"]:
        selected = _select_quality_recipes(
            recipes_by_cluster.get(cluster.cluster_id, []),
            scores,
            exclude_recipe_ids | selected_ids,
        )
        if len(selected) < _MIN_RECIPES_PER_CLUSTER:
            continue
        selected_ids.update(int(recipe["id"]) for recipe in selected)
        cluster_results.append(TasteClusterRecipes(
            cluster_id=cluster.cluster_id,
            recipes=[RecipeSummary(**recipe) for recipe in selected],
        ))

    return ApiResponse(data=TasteProfileRecipesResponse(clusters=cluster_results))


def _personalized_weights(
    n_interactions: int,
    n_sessions: int,
    explorer_ingredients: list[str],
) -> dict[str, float]:
    if n_interactions <= 0 and n_sessions <= 0 and explorer_ingredients:
        return {"interactions": 0.0, "answers": 0.0, "semantic": 1.0}
    return compute_weights(n_interactions, n_sessions)


def _select_quality_recipes(
    recipes: list[dict[str, Any]],
    scores: dict[int, float],
    excluded_ids: set[int],
) -> list[dict[str, Any]]:
    ranked = sorted(
        recipes,
        key=lambda recipe: scores.get(int(recipe["id"]), 0.0),
        reverse=True,
    )
    if not ranked:
        return []
    best_score = scores.get(int(ranked[0]["id"]), 0.0)
    if best_score <= 0:
        return []
    threshold = best_score * _RELATIVE_QUALITY_THRESHOLD
    return [
        recipe
        for recipe in ranked
        if int(recipe["id"]) not in excluded_ids
        and scores.get(int(recipe["id"]), 0.0) >= threshold
    ][:_RECIPES_PER_CLUSTER]
