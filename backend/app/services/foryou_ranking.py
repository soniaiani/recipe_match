"""Scoring, candidate selection, and diversification for For You."""
from __future__ import annotations

from collections import Counter
from typing import Any

from app.recommender.embeddings import encode_text
from app.services.foryou_common import (
    BOOLEAN_FEATURES,
    CANDIDATE_POOL_LIMIT,
    CATEGORICAL_FEATURES,
    CLUSTER_DIVERSITY_ALPHA,
    PROFILE_POOL_KEEP,
    RECENT_PROMOTION_COUNT,
    RECENT_PROMOTION_MIN_SCORE,
    RECENT_PROMOTION_SLOTS,
    RECENT_SIGNAL_BONUS,
    RECOMMENDATION_COUNT,
    SEMANTIC_POOL_KEEP,
    SEMANTIC_RPC_COUNT,
    MIN_GLOBAL_POPULARITY_USERS,
    clamp,
    parse_ingredients,
)


def compute_weights(n_interactions: int, n_sessions: int) -> dict[str, float]:
    if n_interactions <= 0 and n_sessions <= 0:
        return {"interactions": 0.0, "answers": 0.0, "semantic": 0.0}
    if n_interactions <= 0:
        return {"interactions": 0.0, "answers": 0.60, "semantic": 0.40}
    if n_sessions <= 0:
        return {"interactions": 0.55, "answers": 0.0, "semantic": 0.45}

    return {"interactions": 0.55, "answers": 0.25, "semantic": 0.20}

def _interaction_score(recipe: dict[str, Any], profile: dict[str, Any]) -> float:
    raw = 0.0
    max_possible = 0.0

    for feature in CATEGORICAL_FEATURES:
        freqs = profile.get("categorical", {}).get(feature, {}) or {}
        if not freqs:
            continue
        max_possible += 1.0
        value = recipe.get(feature)
        if value:
            raw += float(freqs.get(str(value), 0.0))

    for feature in BOOLEAN_FEATURES:
        mean = float((profile.get("booleans", {}) or {}).get(feature, 0.0))
        if mean <= 0:
            continue
        max_possible += mean
        if recipe.get(feature) is True:
            raw += mean

    if max_possible <= 0:
        return 0.0
    return clamp(raw / max_possible)

def _answers_score(recipe: dict[str, Any], profile: dict[str, Any]) -> float:
    raw = 0.0
    max_possible = 0.0

    for feature in CATEGORICAL_FEATURES:
        values = profile.get("categorical", {}).get(feature, []) or []
        if not values:
            continue
        max_possible += 1.0
        if recipe.get(feature) in values:
            raw += 1.0

    for feature, answer in (profile.get("booleans", {}) or {}).items():
        max_possible += 1.0
        if answer == "yes":
            if recipe.get(feature) is True:
                raw += 1.0
        elif answer == "no":
            if recipe.get(feature) is True:
                raw -= 0.5
            else:
                raw += 0.5

    if max_possible <= 0:
        return 0.0
    return clamp(raw / max_possible)

def score_candidate(
    recipe: dict[str, Any],
    interaction_profile: dict[str, Any],
    answers_profile: dict[str, Any],
    semantic_scores: dict[int, float],
    weights: dict[str, float],
) -> float:
    recipe_id = int(recipe.get("id", -1))
    semantic_score = semantic_scores.get(recipe_id, 0.0)
    base_score = (
        weights["interactions"] * _interaction_score(recipe, interaction_profile)
        + weights["answers"] * _answers_score(recipe, answers_profile)
        + weights["semantic"] * semantic_score
    )
    if int(interaction_profile.get("n_interactions") or 0) > 0:
        base_score += RECENT_SIGNAL_BONUS * _recent_interaction_score(recipe, interaction_profile)
    return base_score

def _profile_match_score(
    recipe: dict[str, Any],
    interaction_profile: dict[str, Any],
    answers_profile: dict[str, Any],
) -> float:
    return _interaction_score(recipe, interaction_profile) + _answers_score(recipe, answers_profile)

def _recent_interaction_score(recipe: dict[str, Any], profile: dict[str, Any]) -> float:
    recent_categorical = profile.get("recent_categorical", {}) or {}
    recent_ingredients = set(profile.get("recent_ingredients", []) or [])
    if not recent_categorical and not recent_ingredients:
        return 0.0

    categorical_raw = 0.0
    categorical_possible = 0
    for feature in CATEGORICAL_FEATURES:
        values = recent_categorical.get(feature, {}) or {}
        if not values:
            continue
        categorical_possible += 1
        value = recipe.get(feature)
        if value:
            categorical_raw += float(values.get(str(value), 0.0))
    categorical_score = categorical_raw / categorical_possible if categorical_possible else 0.0

    ingredient_score = 0.0
    if recent_ingredients:
        recipe_ingredients = set(parse_ingredients(recipe.get("ingredients_clean") or recipe.get("ingredients")))
        overlap = len(recipe_ingredients & recent_ingredients)
        ingredient_score = min(overlap / 5.0, 1.0)

    if categorical_possible and recent_ingredients:
        return clamp(0.65 * categorical_score + 0.35 * ingredient_score)
    return clamp(categorical_score or ingredient_score)

def _semantic_scores(admin: Any, profile_text: str) -> dict[int, float]:
    if not profile_text:
        return {}
    try:
        embedding = encode_text(profile_text)
        rows = (
            admin.rpc(
                "match_recipes_by_embedding",
                {
                    "query_embedding": [float(x) for x in embedding.tolist()],
                    "match_count": SEMANTIC_RPC_COUNT,
                },
            )
            .execute()
            .data
            or []
        )
    except Exception as exc:
        print(f"[foryou] semantic RPC failed: {exc}")
        return {}

    semantic: dict[int, float] = {}
    for row in rows:
        recipe_id = row.get("id")
        if recipe_id is None:
            continue
        similarity = float(row.get("similarity") or 0.0)
        semantic[int(recipe_id)] = clamp((similarity + 1.0) / 2.0)
    return semantic

def semantic_scores_for_profiles(admin: Any, profile_texts: list[str]) -> dict[int, float]:
    """Combine multiple semantic profile queries with OR-style max similarity."""
    combined: dict[int, float] = {}
    for profile_text in profile_texts:
        for recipe_id, score in _semantic_scores(admin, profile_text).items():
            combined[recipe_id] = max(combined.get(recipe_id, 0.0), score)
    return combined

def select_candidates(
    recipes: list[dict[str, Any]],
    semantic_scores: dict[int, float],
    interaction_profile: dict[str, Any],
    answers_profile: dict[str, Any],
) -> list[dict[str, Any]]:
    by_id = {int(recipe["id"]): recipe for recipe in recipes if recipe.get("id") is not None}

    semantic_ids = [
        recipe_id
        for recipe_id, _ in sorted(semantic_scores.items(), key=lambda item: -item[1])
        if recipe_id in by_id
    ][:SEMANTIC_POOL_KEEP]

    profile_ranked = sorted(
        by_id.values(),
        key=lambda recipe: _profile_match_score(recipe, interaction_profile, answers_profile),
        reverse=True,
    )[:PROFILE_POOL_KEEP]

    selected: dict[int, dict[str, Any]] = {}
    for recipe_id in semantic_ids:
        selected[recipe_id] = by_id[recipe_id]
    for recipe in profile_ranked:
        selected[int(recipe["id"])] = recipe
        if len(selected) >= CANDIDATE_POOL_LIMIT:
            break
    return list(selected.values())[:CANDIDATE_POOL_LIMIT]

def select_diverse_top(recipes: list[dict[str, Any]], limit: int = RECOMMENDATION_COUNT) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    cuisine_counts: Counter = Counter()
    meal_counts: Counter = Counter()

    for recipe in recipes:
        cuisine = recipe.get("cuisine") or "unknown"
        meal_type = recipe.get("meal_type") or "unknown"
        if cuisine_counts[cuisine] >= 3 or meal_counts[meal_type] >= 3:
            continue
        selected.append(recipe)
        cuisine_counts[cuisine] += 1
        meal_counts[meal_type] += 1
        if len(selected) >= limit:
            return selected

    if len(selected) < limit:
        selected_ids = {recipe.get("id") for recipe in selected}
        for recipe in recipes:
            if recipe.get("id") in selected_ids:
                continue
            selected.append(recipe)
            if len(selected) >= limit:
                break
    return selected

def select_cluster_aware_top(
    recipes: list[dict[str, Any]],
    scores: dict[int, float],
    clusters: dict[int, int],
    limit: int = RECOMMENDATION_COUNT,
    alpha: float = CLUSTER_DIVERSITY_ALPHA,
) -> list[dict[str, Any]]:
    """Greedy cluster-aware reranking with a linear soft penalty."""
    if not clusters:
        return select_diverse_top(recipes, limit)

    remaining = list(recipes)
    selected: list[dict[str, Any]] = []
    cluster_counts: Counter = Counter()

    while remaining and len(selected) < limit:
        best_index = 0
        best_adjusted = -float("inf")

        for idx, recipe in enumerate(remaining):
            recipe_id = recipe.get("id")
            if recipe_id is None:
                adjusted = -float("inf")
            else:
                recipe_id_int = int(recipe_id)
                cluster_id = clusters.get(recipe_id_int)
                penalty = alpha * cluster_counts[cluster_id] if cluster_id is not None else 0.0
                adjusted = scores.get(recipe_id_int, 0.0) - penalty

            if adjusted > best_adjusted:
                best_adjusted = adjusted
                best_index = idx

        chosen = remaining.pop(best_index)
        selected.append(chosen)
        chosen_id = chosen.get("id")
        if chosen_id is not None:
            cluster_id = clusters.get(int(chosen_id))
            if cluster_id is not None:
                cluster_counts[cluster_id] += 1

    return selected

def promote_recent_matches(
    selected: list[dict[str, Any]],
    interaction_profile: dict[str, Any],
    limit: int = RECOMMENDATION_COUNT,
) -> list[dict[str, Any]]:
    """Make recent user actions visible in the first screen of FYP."""
    if int(interaction_profile.get("n_interactions") or 0) <= 0:
        return selected
    if not interaction_profile.get("recent_ingredients") and not interaction_profile.get("recent_categorical"):
        return selected

    first_screen_ids = {
        int(recipe["id"])
        for recipe in selected[: max(RECENT_PROMOTION_SLOTS) + 1]
        if recipe.get("id") is not None
    }
    by_recent_match = sorted(
        (
            recipe
            for recipe in selected
            if recipe.get("id") is not None and int(recipe["id"]) not in first_screen_ids
        ),
        key=lambda recipe: _recent_interaction_score(recipe, interaction_profile),
        reverse=True,
    )
    promoted: list[dict[str, Any]] = []
    for recipe in by_recent_match:
        recipe_id = recipe.get("id")
        if recipe_id is None:
            continue
        if _recent_interaction_score(recipe, interaction_profile) < RECENT_PROMOTION_MIN_SCORE:
            continue
        promoted.append(recipe)
        if len(promoted) >= RECENT_PROMOTION_COUNT:
            break

    if not promoted:
        return selected

    promoted_ids = {int(recipe["id"]) for recipe in promoted if recipe.get("id") is not None}
    output = [
        recipe
        for recipe in selected
        if recipe.get("id") is not None and int(recipe["id"]) not in promoted_ids
    ]
    for slot, recipe in zip(RECENT_PROMOTION_SLOTS, promoted):
        output.insert(min(slot, len(output)), recipe)
    return output[:limit]

def _neutral_cold_start_recipes(recipes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deterministic catalog ranking when global popularity is too sparse."""
    def quality_key(recipe: dict[str, Any]) -> tuple[float, int]:
        total_minutes = recipe.get("total_minutes")
        minutes_score = 0.0
        if isinstance(total_minutes, (int, float)) and total_minutes > 0:
            minutes_score = max(0.0, 1.0 - min(float(total_minutes), 120.0) / 120.0)

        score = (
            (1.0 if recipe.get("image_url") else 0.0)
            + (0.5 if recipe.get("description") else 0.0)
            + (0.4 if recipe.get("is_quick") else 0.0)
            + minutes_score
        )
        return (score, int(recipe.get("id") or 0))

    return sorted(recipes, key=quality_key, reverse=True)

def load_popular_recipes(admin: Any, recipes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {int(recipe["id"]): recipe for recipe in recipes if recipe.get("id") is not None}
    try:
        rows = (
            admin.table("recipe_interactions")
            .select("recipe_id,interaction_type,user_id")
            .eq("interaction_type", "cook")
            .execute()
            .data
            or []
        )
    except Exception as exc:
        print(f"[foryou] failed to load popularity: {exc}")
        rows = []

    counts: Counter = Counter()
    user_ids: set[str] = set()
    for row in rows:
        recipe_id = row.get("recipe_id")
        if recipe_id is not None:
            counts[int(recipe_id)] += 1
        if row.get("user_id"):
            user_ids.add(str(row["user_id"]))

    try:
        saved_rows = (
            admin.table("saved_recipes")
            .select("recipe_id,user_id")
            .execute()
            .data
            or []
        )
    except Exception as exc:
        print(f"[foryou] failed to load saved popularity: {exc}")
        saved_rows = []

    for row in saved_rows:
        recipe_id = row.get("recipe_id")
        if recipe_id is not None:
            counts[int(recipe_id)] += 1
        if row.get("user_id"):
            user_ids.add(str(row["user_id"]))

    if len(user_ids) < MIN_GLOBAL_POPULARITY_USERS:
        print(
            "[foryou] global popularity too sparse "
            f"users={len(user_ids)}; using neutral cold start"
        )
        return _neutral_cold_start_recipes(recipes)

    return sorted(
        by_id.values(),
        key=lambda recipe: (counts.get(int(recipe["id"]), 0), int(recipe["id"])),
        reverse=True,
    )

def compute_rank_scores(recipes: list[dict[str, Any]]) -> dict[int, float]:
    total = max(len(recipes) - 1, 1)
    return {
        int(recipe["id"]): 1.0 - (idx / total)
        for idx, recipe in enumerate(recipes)
        if recipe.get("id") is not None
    }
