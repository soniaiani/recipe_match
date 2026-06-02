"""For You - hybrid personalized recommendations."""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from itertools import combinations
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, Request

from app.database import get_supabase_admin
from app.middleware.auth import get_current_user
from app.models.schemas import ApiResponse, ForYouResponse, RecipeSummary
from app.recommender.embeddings import encode_text
from app.recommender.filters import filter_excluded_ingredients, normalize_excluded_ingredients

router = APIRouter(prefix="/foryou", tags=["foryou"])

_CATEGORICAL_FEATURES = ("meal_type", "protein_type", "cuisine")
_BOOLEAN_FEATURES = (
    "is_spicy",
    "is_sweet",
    "is_quick",
    "needs_oven",
    "needs_stovetop",
    "is_no_cook",
    "has_pasta",
    "has_rice",
    "has_potato",
    "has_tomato_base",
    "has_cream_base",
    "has_cheese",
    "has_broth_base",
    "has_mushroom",
    "has_leafy_greens",
    "has_beans_legumes",
    "has_fruit",
    "has_nuts",
    "has_chocolate",
    "has_asian_sauce",
)
_ALL_PROFILE_FEATURES = _CATEGORICAL_FEATURES + _BOOLEAN_FEATURES

_SUMMARY_FIELDS = (
    "id,name,description,image_url,meal_type,cuisine,"
    "total_minutes,is_vegetarian,is_vegan,is_gluten_free,is_dairy_free,is_quick"
)
_RECIPE_FIELD_LIST = (
    "id",
    "name",
    "description",
    "image_url",
    "meal_type",
    "cuisine",
    "total_minutes",
    "is_vegetarian",
    "is_vegan",
    "is_gluten_free",
    "is_dairy_free",
    "is_quick",
    "protein_type",
    "ingredients",
    "ingredients_clean",
    *_BOOLEAN_FEATURES,
)
_RECIPE_FIELDS = ",".join(dict.fromkeys(_RECIPE_FIELD_LIST))
_RECOMMENDATION_COUNT = 20
_SEMANTIC_RPC_COUNT = 300
_SEMANTIC_POOL_KEEP = 100
_PROFILE_POOL_KEEP = 200
_CANDIDATE_POOL_LIMIT = 300
_RECENT_SESSIONS_LIMIT = 20
_MIN_EXPLORER_SESSIONS = 5
_MIN_INGREDIENT_FREQUENCY = 0.30
_MAX_EXPLORER_INTENTS = 5
_MAX_EXPLORER_INTENT_TERMS = 6
_CLUSTER_DIVERSITY_ALPHA = 0.06
_RECENT_INTERACTION_LIMIT = 8
_RECENT_SIGNAL_BONUS = 0.24
_RECENT_SIGNAL_RANK_DECAY = 0.30
_RECENT_PROMOTION_COUNT = 3
_RECENT_PROMOTION_MIN_SCORE = 0.45
_RECENT_PROMOTION_SLOTS = (1, 3, 5)
_MIN_GLOBAL_POPULARITY_USERS = 3
_PANTRY_HARD_INGREDIENTS = {
    "salt",
    "water",
    "oil",
    "olive oil",
    "vegetable oil",
    "cooking spray",
    "black pepper",
    "pepper",
}
_PANTRY_SOFT_INGREDIENTS = {
    "sugar",
    "butter",
    "all-purpose flour",
    "flour",
    "egg",
    "milk",
}
_PANTRY_SOFT_WEIGHT = 0.35

_INTERACTION_WEIGHTS = {
    "view": 0.5,
    "save": 2.0,
    "cook": 3.0,
}

_BOOLEAN_TEXT = {
    "is_spicy": "spicy",
    "is_sweet": "sweet",
    "is_quick": "quick easy",
    "needs_oven": "oven baked",
    "needs_stovetop": "stovetop cooked",
    "is_no_cook": "no cook",
    "has_pasta": "pasta",
    "has_rice": "rice",
    "has_potato": "potato",
    "has_tomato_base": "tomato",
    "has_cream_base": "creamy",
    "has_cheese": "cheese",
    "has_broth_base": "broth soup",
    "has_mushroom": "mushroom",
    "has_leafy_greens": "leafy greens",
    "has_beans_legumes": "beans legumes",
    "has_fruit": "fruit",
    "has_nuts": "nuts",
    "has_chocolate": "chocolate",
    "has_asian_sauce": "asian sauce",
}


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return min(max(value, lo), hi)


def _parse_dt(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str) and value:
        try:
            normalized = value.replace("Z", "+00:00")
            dt = datetime.fromisoformat(normalized)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return datetime.now(timezone.utc)


def _days_ago(value: Any) -> float:
    created_at = _parse_dt(value)
    return max((datetime.now(timezone.utc) - created_at).total_seconds() / 86400.0, 0.0)


def _interaction_recency_multiplier(days: float, rank: int) -> float:
    multiplier = 1.0
    if days <= 1.0:
        multiplier *= 1.75
    elif days <= 7.0:
        multiplier *= 1.35
    elif days <= 30.0:
        multiplier *= 1.10

    if rank < 3:
        multiplier *= 1.50
    elif rank < _RECENT_INTERACTION_LIMIT:
        multiplier *= 1.20
    return multiplier


def _parse_ingredients(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(item).strip().lower() for item in value if str(item).strip()]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(item).strip().lower() for item in parsed if str(item).strip()]
        except Exception:
            pass
        return [
            part.strip().lower()
            for part in value.replace("\n", ",").split(",")
            if part.strip()
        ]
    return []


def _profile_ingredient_weight(ingredient: str) -> float:
    if ingredient in _PANTRY_HARD_INGREDIENTS:
        return 0.0
    if ingredient in _PANTRY_SOFT_INGREDIENTS:
        return _PANTRY_SOFT_WEIGHT
    return 1.0


def _valid_answer(value: Any) -> bool:
    if value in (None, "skip", "unknown", "any"):
        return False
    if value == ["any"]:
        return False
    if isinstance(value, list):
        return any(item not in ("skip", "unknown", "any") for item in value)
    return True


def _normalize_answer_values(value: Any) -> list[str]:
    if not _valid_answer(value):
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item not in ("skip", "unknown", "any")]
    return [str(value)]


def _dietary_filter(recipes: list[dict[str, Any]], meta: dict[str, Any]) -> list[dict[str, Any]]:
    out = recipes
    if meta.get("is_vegetarian"):
        out = [recipe for recipe in out if recipe.get("is_vegetarian")]
    if meta.get("is_vegan"):
        out = [recipe for recipe in out if recipe.get("is_vegan")]
    if meta.get("is_gluten_free"):
        out = [recipe for recipe in out if recipe.get("is_gluten_free")]
    if meta.get("is_dairy_free"):
        out = [recipe for recipe in out if recipe.get("is_dairy_free")]
    return out


def _hard_filter(
    recipes: list[dict[str, Any]],
    saved_ids: set[int],
    meta: dict[str, Any],
) -> list[dict[str, Any]]:
    filtered = [
        recipe
        for recipe in recipes
        if recipe.get("id") is not None and int(recipe["id"]) not in saved_ids
    ]
    filtered = _dietary_filter(filtered, meta)
    return filter_excluded_ingredients(filtered, meta.get("excluded_ingredients", []))


def _load_recipes(request: Request, admin: Any) -> list[dict[str, Any]]:
    recipes = getattr(request.app.state, "rec_recipes", None)
    if recipes:
        return list(recipes)

    loaded: list[dict[str, Any]] = []
    offset = 0
    while True:
        page = (
            admin.table("recipes")
            .select(_RECIPE_FIELDS)
            .range(offset, offset + 999)
            .execute()
            .data
            or []
        )
        loaded.extend(page)
        if len(page) < 1000:
            break
        offset += 1000
    return loaded


def build_interaction_profile(user_id: str, admin: Any) -> dict[str, Any]:
    """Build weighted implicit profile from interactions plus saved-only fallback."""
    empty = {
        "n_interactions": 0,
        "saved_ids": set(),
        "categorical": {feature: {} for feature in _CATEGORICAL_FEATURES},
        "recent_categorical": {feature: {} for feature in _CATEGORICAL_FEATURES},
        "booleans": {feature: 0.0 for feature in _BOOLEAN_FEATURES},
        "ingredients": [],
        "recent_ingredients": [],
    }

    try:
        interaction_rows = (
            admin.table("recipe_interactions")
            .select(f"recipe_id,interaction_type,weight,created_at,recipes({_RECIPE_FIELDS})")
            .eq("user_id", user_id)
            .in_("interaction_type", list(_INTERACTION_WEIGHTS.keys()))
            .order("created_at", desc=True)
            .execute()
            .data
            or []
        )
        saved_rows = (
            admin.table("saved_recipes")
            .select(f"recipe_id,saved_at,recipes({_RECIPE_FIELDS})")
            .eq("user_id", user_id)
            .order("saved_at", desc=True)
            .execute()
            .data
            or []
        )
    except Exception as exc:
        print(f"[foryou] failed to build interaction profile: {exc}")
        return empty

    saved_ids = {int(row["recipe_id"]) for row in saved_rows if row.get("recipe_id") is not None}
    real_save_ids = {
        int(row["recipe_id"])
        for row in interaction_rows
        if row.get("recipe_id") is not None and row.get("interaction_type") == "save"
    }

    events: list[tuple[dict[str, Any], float]] = []
    recent_events: list[tuple[dict[str, Any], float]] = []
    for rank, row in enumerate(interaction_rows):
        recipe = row.get("recipes")
        interaction_type = row.get("interaction_type")
        if not recipe or interaction_type not in _INTERACTION_WEIGHTS:
            continue
        base_weight = _INTERACTION_WEIGHTS[interaction_type]
        days = _days_ago(row.get("created_at"))
        weight = (
            base_weight
            * math.exp(-0.02 * days)
            * _interaction_recency_multiplier(days, rank)
        )
        events.append((recipe, weight))
        recent_events.append((recipe, weight * math.exp(-_RECENT_SIGNAL_RANK_DECAY * rank)))

    for rank, row in enumerate(saved_rows):
        recipe_id = row.get("recipe_id")
        recipe = row.get("recipes")
        if recipe_id is None or not recipe or int(recipe_id) in real_save_ids:
            continue
        days = _days_ago(row.get("saved_at"))
        weight = (
            _INTERACTION_WEIGHTS["save"]
            * math.exp(-0.02 * days)
            * _interaction_recency_multiplier(days, rank)
        )
        events.append((recipe, weight))
        recent_events.append((recipe, weight * math.exp(-_RECENT_SIGNAL_RANK_DECAY * rank)))

    total_weight = sum(weight for _, weight in events)
    if total_weight <= 0:
        empty["saved_ids"] = saved_ids
        return empty

    categorical_counts = {feature: Counter() for feature in _CATEGORICAL_FEATURES}
    recent_categorical_counts = {feature: Counter() for feature in _CATEGORICAL_FEATURES}
    boolean_sums = {feature: 0.0 for feature in _BOOLEAN_FEATURES}
    ingredient_counts: Counter = Counter()
    recent_ingredient_counts: Counter = Counter()

    for recipe, weight in events:
        for feature in _CATEGORICAL_FEATURES:
            value = recipe.get(feature)
            if value:
                categorical_counts[feature][str(value)] += weight
        for feature in _BOOLEAN_FEATURES:
            if recipe.get(feature) is True:
                boolean_sums[feature] += weight
        for ingredient in _parse_ingredients(recipe.get("ingredients_clean") or recipe.get("ingredients")):
            ingredient_weight = _profile_ingredient_weight(ingredient)
            if ingredient_weight > 0:
                ingredient_counts[ingredient] += weight * ingredient_weight

    recent_total_weight = sum(weight for _, weight in recent_events)
    if recent_total_weight > 0:
        for recipe, weight in recent_events:
            for feature in _CATEGORICAL_FEATURES:
                value = recipe.get(feature)
                if value:
                    recent_categorical_counts[feature][str(value)] += weight
            for ingredient in _parse_ingredients(recipe.get("ingredients_clean") or recipe.get("ingredients")):
                ingredient_weight = _profile_ingredient_weight(ingredient)
                if ingredient_weight > 0:
                    recent_ingredient_counts[ingredient] += weight * ingredient_weight

    return {
        "n_interactions": len(events),
        "saved_ids": saved_ids,
        "categorical": {
            feature: {
                value: count / total_weight
                for value, count in counts.items()
            }
            for feature, counts in categorical_counts.items()
        },
        "recent_categorical": {
            feature: {
                value: count / recent_total_weight
                for value, count in counts.items()
            }
            for feature, counts in recent_categorical_counts.items()
        } if recent_total_weight > 0 else {feature: {} for feature in _CATEGORICAL_FEATURES},
        "booleans": {
            feature: boolean_sums[feature] / total_weight
            for feature in _BOOLEAN_FEATURES
        },
        "ingredients": [value for value, _ in ingredient_counts.most_common(15)],
        "recent_ingredients": [value for value, _ in recent_ingredient_counts.most_common(12)],
    }


def build_answers_profile(user_id: str, admin: Any) -> dict[str, Any]:
    """Aggregate completed adaptive answers into stable active preferences."""
    empty = {
        "n_sessions": 0,
        "categorical": {feature: [] for feature in _CATEGORICAL_FEATURES},
        "booleans": {},
    }
    try:
        rows = (
            admin.table("recommendation_sessions")
            .select("answers,completed_at")
            .eq("user_id", user_id)
            .not_.is_("completed_at", "null")
            .order("completed_at", desc=True)
            .limit(_RECENT_SESSIONS_LIMIT)
            .execute()
            .data
            or []
        )
    except Exception as exc:
        print(f"[foryou] failed to build answers profile: {exc}")
        return empty

    value_counts: dict[str, Counter] = defaultdict(Counter)
    answered_counts: Counter = Counter()

    for row in rows:
        answers = row.get("answers") or {}
        if isinstance(answers, str):
            try:
                answers = json.loads(answers)
            except Exception:
                answers = {}
        if not isinstance(answers, dict):
            continue

        for feature in _ALL_PROFILE_FEATURES:
            answer = answers.get(feature)
            values = _normalize_answer_values(answer)
            if not values:
                continue
            answered_counts[feature] += 1
            for value in set(values):
                value_counts[feature][value] += 1

    categorical: dict[str, list[str]] = {feature: [] for feature in _CATEGORICAL_FEATURES}
    booleans: dict[str, str] = {}

    for feature in _CATEGORICAL_FEATURES:
        answered = answered_counts[feature]
        if answered <= 0:
            continue
        active_values = []
        for value, count in value_counts[feature].items():
            share = (count + 1.0) / (answered + 2.0)
            if share >= 0.40:
                active_values.append(value)
        categorical[feature] = active_values

    for feature in _BOOLEAN_FEATURES:
        answered = answered_counts[feature]
        if answered <= 0:
            continue
        yes_count = value_counts[feature].get("yes", 0)
        no_count = value_counts[feature].get("no", 0)
        yes_share = (yes_count + 1.0) / (answered + 2.0)
        no_share = (no_count + 1.0) / (answered + 2.0)
        if yes_share >= 0.40 or no_share >= 0.40:
            booleans[feature] = "yes" if yes_count >= no_count else "no"

    return {
        "n_sessions": len(rows),
        "categorical": categorical,
        "booleans": booleans,
    }


def build_explorer_intents(
    user_id: str,
    admin: Any,
    excluded_ingredients: Any = None,
) -> list[list[str]]:
    """Extract frequent Explorer ingredient associations as separate intents."""
    excluded = set(normalize_excluded_ingredients(excluded_ingredients))
    try:
        rows = (
            admin.table("explorer_sessions")
            .select("chain")
            .eq("user_id", user_id)
            .order("updated_at", desc=True)
            .limit(50)
            .execute()
            .data
            or []
        )
    except Exception as exc:
        print(f"[foryou] failed to build explorer profile: {exc}")
        return []

    if len(rows) < _MIN_EXPLORER_SESSIONS:
        return []

    n_sessions = len(rows)
    ingredient_counts: Counter = Counter()
    pair_counts: Counter = Counter()

    for row in rows:
        chain = [
            str(ingredient).strip().lower()
            for ingredient in (row.get("chain") or [])
            if str(ingredient).strip()
            and str(ingredient).strip().lower() not in excluded
        ]
        unique_chain = sorted(set(chain))
        if not unique_chain:
            continue
        for ingredient in unique_chain:
            ingredient_counts[ingredient] += 1
        for left, right in combinations(unique_chain, 2):
            pair_counts[(left, right)] += 1

    frequent_pairs = [
        pair
        for pair, count in pair_counts.items()
        if count / n_sessions >= _MIN_INGREDIENT_FREQUENCY
    ]

    graph: dict[str, set[str]] = defaultdict(set)
    for left, right in frequent_pairs:
        graph[left].add(right)
        graph[right].add(left)

    intents: list[list[str]] = []
    seen: set[str] = set()
    for ingredient in sorted(graph, key=lambda item: -ingredient_counts[item]):
        if ingredient in seen:
            continue
        stack = [ingredient]
        component: set[str] = set()
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            stack.extend(graph[current] - component)
        seen.update(component)
        intents.append(
            sorted(component, key=lambda item: -ingredient_counts[item])[:_MAX_EXPLORER_INTENT_TERMS]
        )

    frequent = [
        ingredient
        for ingredient, count in ingredient_counts.items()
        if count / n_sessions >= _MIN_INGREDIENT_FREQUENCY
    ]
    frequent.sort(key=lambda ingredient: -ingredient_counts[ingredient])
    used = {ingredient for intent in intents for ingredient in intent}
    for ingredient in frequent:
        if ingredient not in used:
            intents.append([ingredient])
        if len(intents) >= _MAX_EXPLORER_INTENTS:
            break

    return intents[:_MAX_EXPLORER_INTENTS]


def build_explorer_profile(
    user_id: str,
    admin: Any,
    excluded_ingredients: Any = None,
) -> list[str]:
    """Compatibility helper returning flattened frequent Explorer ingredients."""
    seen: set[str] = set()
    ingredients: list[str] = []
    for intent in build_explorer_intents(user_id, admin, excluded_ingredients):
        for ingredient in intent:
            if ingredient not in seen:
                seen.add(ingredient)
                ingredients.append(ingredient)
    return ingredients


def build_semantic_profile(
    interaction_profile: dict[str, Any],
    answers_profile: dict[str, Any],
    explorer_ingredients: list[str] | None = None,
) -> str:
    parts: list[str] = []

    if explorer_ingredients:
        parts.extend(explorer_ingredients[:15])

    interaction_ingredients = interaction_profile.get("ingredients") or []
    parts.extend(interaction_ingredients[:10])

    for feature in _CATEGORICAL_FEATURES:
        freqs = interaction_profile.get("categorical", {}).get(feature, {}) or {}
        parts.extend(value for value, _ in sorted(freqs.items(), key=lambda item: -item[1])[:3])
        parts.extend(answers_profile.get("categorical", {}).get(feature, []) or [])

    interaction_booleans = interaction_profile.get("booleans", {}) or {}
    for feature, value in interaction_booleans.items():
        if value >= 0.40:
            parts.append(_BOOLEAN_TEXT.get(feature, feature))

    for feature, answer in (answers_profile.get("booleans", {}) or {}).items():
        if answer == "yes":
            parts.append(_BOOLEAN_TEXT.get(feature, feature))

    seen: set[str] = set()
    unique_parts = []
    for part in parts:
        normalized = str(part).strip().lower()
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique_parts.append(normalized)
    return " ".join(unique_parts)


def build_semantic_profiles(
    interaction_profile: dict[str, Any],
    answers_profile: dict[str, Any],
    explorer_intents: list[list[str]] | None = None,
) -> list[str]:
    """Build one semantic query per Explorer intent, combined with stable profile signals."""
    if not explorer_intents:
        profile = build_semantic_profile(interaction_profile, answers_profile)
        return [profile] if profile else []

    profiles = []
    for intent in explorer_intents[:_MAX_EXPLORER_INTENTS]:
        profile = build_semantic_profile(interaction_profile, answers_profile, intent)
        if profile:
            profiles.append(profile)

    if profiles:
        return profiles
    fallback = build_semantic_profile(interaction_profile, answers_profile)
    return [fallback] if fallback else []


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

    for feature in _CATEGORICAL_FEATURES:
        freqs = profile.get("categorical", {}).get(feature, {}) or {}
        if not freqs:
            continue
        max_possible += 1.0
        value = recipe.get(feature)
        if value:
            raw += float(freqs.get(str(value), 0.0))

    for feature in _BOOLEAN_FEATURES:
        mean = float((profile.get("booleans", {}) or {}).get(feature, 0.0))
        if mean <= 0:
            continue
        max_possible += mean
        if recipe.get(feature) is True:
            raw += mean

    if max_possible <= 0:
        return 0.0
    return _clamp(raw / max_possible)


def _answers_score(recipe: dict[str, Any], profile: dict[str, Any]) -> float:
    raw = 0.0
    max_possible = 0.0

    for feature in _CATEGORICAL_FEATURES:
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
    return _clamp(raw / max_possible)


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
        base_score += _RECENT_SIGNAL_BONUS * _recent_interaction_score(recipe, interaction_profile)
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
    for feature in _CATEGORICAL_FEATURES:
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
        recipe_ingredients = set(_parse_ingredients(recipe.get("ingredients_clean") or recipe.get("ingredients")))
        overlap = len(recipe_ingredients & recent_ingredients)
        ingredient_score = min(overlap / 5.0, 1.0)

    if categorical_possible and recent_ingredients:
        return _clamp(0.65 * categorical_score + 0.35 * ingredient_score)
    return _clamp(categorical_score or ingredient_score)


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
                    "match_count": _SEMANTIC_RPC_COUNT,
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
        semantic[int(recipe_id)] = _clamp((similarity + 1.0) / 2.0)
    return semantic


def _semantic_scores_for_profiles(admin: Any, profile_texts: list[str]) -> dict[int, float]:
    """Combine multiple semantic profile queries with OR-style max similarity."""
    combined: dict[int, float] = {}
    for profile_text in profile_texts:
        for recipe_id, score in _semantic_scores(admin, profile_text).items():
            combined[recipe_id] = max(combined.get(recipe_id, 0.0), score)
    return combined


def _select_candidates(
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
    ][:_SEMANTIC_POOL_KEEP]

    profile_ranked = sorted(
        by_id.values(),
        key=lambda recipe: _profile_match_score(recipe, interaction_profile, answers_profile),
        reverse=True,
    )[:_PROFILE_POOL_KEEP]

    selected: dict[int, dict[str, Any]] = {}
    for recipe_id in semantic_ids:
        selected[recipe_id] = by_id[recipe_id]
    for recipe in profile_ranked:
        selected[int(recipe["id"])] = recipe
        if len(selected) >= _CANDIDATE_POOL_LIMIT:
            break
    return list(selected.values())[:_CANDIDATE_POOL_LIMIT]


def _diverse_top(recipes: list[dict[str, Any]], limit: int = _RECOMMENDATION_COUNT) -> list[dict[str, Any]]:
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


def _cluster_aware_top(
    recipes: list[dict[str, Any]],
    scores: dict[int, float],
    clusters: dict[int, int],
    limit: int = _RECOMMENDATION_COUNT,
    alpha: float = _CLUSTER_DIVERSITY_ALPHA,
) -> list[dict[str, Any]]:
    """Greedy cluster-aware reranking with a linear soft penalty."""
    if not clusters:
        return _diverse_top(recipes, limit)

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


def _promote_recent_matches(
    selected: list[dict[str, Any]],
    interaction_profile: dict[str, Any],
    limit: int = _RECOMMENDATION_COUNT,
) -> list[dict[str, Any]]:
    """Make recent user actions visible in the first screen of FYP."""
    if int(interaction_profile.get("n_interactions") or 0) <= 0:
        return selected
    if not interaction_profile.get("recent_ingredients") and not interaction_profile.get("recent_categorical"):
        return selected

    first_screen_ids = {
        int(recipe["id"])
        for recipe in selected[: max(_RECENT_PROMOTION_SLOTS) + 1]
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
        if _recent_interaction_score(recipe, interaction_profile) < _RECENT_PROMOTION_MIN_SCORE:
            continue
        promoted.append(recipe)
        if len(promoted) >= _RECENT_PROMOTION_COUNT:
            break

    if not promoted:
        return selected

    promoted_ids = {int(recipe["id"]) for recipe in promoted if recipe.get("id") is not None}
    output = [
        recipe
        for recipe in selected
        if recipe.get("id") is not None and int(recipe["id"]) not in promoted_ids
    ]
    for slot, recipe in zip(_RECENT_PROMOTION_SLOTS, promoted):
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


def _popular_recipes(admin: Any, recipes: list[dict[str, Any]]) -> list[dict[str, Any]]:
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

    if len(user_ids) < _MIN_GLOBAL_POPULARITY_USERS:
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


def _rank_scores(recipes: list[dict[str, Any]]) -> dict[int, float]:
    total = max(len(recipes) - 1, 1)
    return {
        int(recipe["id"]): 1.0 - (idx / total)
        for idx, recipe in enumerate(recipes)
        if recipe.get("id") is not None
    }


@router.get("", response_model=ApiResponse[ForYouResponse])
async def for_you(request: Request, payload: dict = Depends(get_current_user)):
    """Return hybrid personalized recipe recommendations."""
    admin = get_supabase_admin()
    user_id = payload["sub"]
    meta = payload.get("user_metadata") or {}

    interaction_profile = build_interaction_profile(user_id, admin)
    answers_profile = build_answers_profile(user_id, admin)
    saved_ids = interaction_profile.get("saved_ids", set()) or set()

    recipes = _load_recipes(request, admin)
    recipes = _hard_filter(recipes, saved_ids, meta)

    n_interactions = int(interaction_profile.get("n_interactions") or 0)
    n_sessions = int(answers_profile.get("n_sessions") or 0)
    explorer_intents = build_explorer_intents(
        user_id,
        admin,
        meta.get("excluded_ingredients", []),
    )
    explorer_ingredients = [
        ingredient
        for intent in explorer_intents
        for ingredient in intent
    ]
    cluster_aware = bool(getattr(request.app.state, "cluster_aware_fyp", False))
    recipe_clusters = getattr(request.app.state, "recipe_clusters", {}) or {}

    if n_interactions <= 0 and n_sessions <= 0 and not explorer_ingredients:
        print(f"[foryou] cold start for user={user_id}")
        ranked = _popular_recipes(admin, recipes)
        ranked_scores = _rank_scores(ranked)
        if cluster_aware and recipe_clusters:
            final_recipes = _cluster_aware_top(
                ranked,
                ranked_scores,
                recipe_clusters,
                _RECOMMENDATION_COUNT,
            )
        else:
            final_recipes = _diverse_top(ranked, _RECOMMENDATION_COUNT)
        return ApiResponse(data=ForYouResponse(
            recipes=[RecipeSummary(**recipe) for recipe in final_recipes]
        ))

    semantic_profiles = build_semantic_profiles(
        interaction_profile,
        answers_profile,
        explorer_intents,
    )
    semantic_scores = _semantic_scores_for_profiles(admin, semantic_profiles)
    weights = compute_weights(n_interactions, n_sessions)
    if n_interactions <= 0 and n_sessions <= 0 and explorer_ingredients:
        weights = {"interactions": 0.0, "answers": 0.0, "semantic": 1.0}
    candidates = _select_candidates(recipes, semantic_scores, interaction_profile, answers_profile)

    candidate_scores = {
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
    ranked = sorted(
        candidates,
        key=lambda recipe: candidate_scores.get(int(recipe.get("id", -1)), 0.0),
        reverse=True,
    )

    if cluster_aware and recipe_clusters:
        final_recipes = _cluster_aware_top(
            ranked,
            candidate_scores,
            recipe_clusters,
            _RECOMMENDATION_COUNT,
        )
    else:
        final_recipes = _diverse_top(ranked, _RECOMMENDATION_COUNT)
    final_recipes = _promote_recent_matches(
        final_recipes,
        interaction_profile,
        _RECOMMENDATION_COUNT,
    )

    print(
        "[foryou] user="
        f"{user_id} interactions={n_interactions} sessions={n_sessions} "
        f"explorer_ingredients={len(explorer_ingredients)} "
        f"candidates={len(candidates)} final={len(final_recipes)} "
        f"cluster_aware={cluster_aware} weights={weights}"
    )
    return ApiResponse(data=ForYouResponse(
        recipes=[RecipeSummary(**recipe) for recipe in final_recipes]
    ))
