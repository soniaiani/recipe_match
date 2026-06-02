"""Shared constants and helpers for For You recommendations."""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from typing import Any

from fastapi import Request

from app.recommender.filters import filter_excluded_ingredients
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


CATEGORICAL_FEATURES = _CATEGORICAL_FEATURES
BOOLEAN_FEATURES = _BOOLEAN_FEATURES
ALL_PROFILE_FEATURES = _ALL_PROFILE_FEATURES
SUMMARY_FIELDS = _SUMMARY_FIELDS
RECIPE_FIELDS = _RECIPE_FIELDS
RECOMMENDATION_COUNT = _RECOMMENDATION_COUNT
SEMANTIC_RPC_COUNT = _SEMANTIC_RPC_COUNT
SEMANTIC_POOL_KEEP = _SEMANTIC_POOL_KEEP
PROFILE_POOL_KEEP = _PROFILE_POOL_KEEP
CANDIDATE_POOL_LIMIT = _CANDIDATE_POOL_LIMIT
RECENT_SESSIONS_LIMIT = _RECENT_SESSIONS_LIMIT
MIN_EXPLORER_SESSIONS = _MIN_EXPLORER_SESSIONS
MIN_INGREDIENT_FREQUENCY = _MIN_INGREDIENT_FREQUENCY
MAX_EXPLORER_INTENTS = _MAX_EXPLORER_INTENTS
MAX_EXPLORER_INTENT_TERMS = _MAX_EXPLORER_INTENT_TERMS
CLUSTER_DIVERSITY_ALPHA = _CLUSTER_DIVERSITY_ALPHA
RECENT_INTERACTION_LIMIT = _RECENT_INTERACTION_LIMIT
RECENT_SIGNAL_BONUS = _RECENT_SIGNAL_BONUS
RECENT_SIGNAL_RANK_DECAY = _RECENT_SIGNAL_RANK_DECAY
RECENT_PROMOTION_COUNT = _RECENT_PROMOTION_COUNT
RECENT_PROMOTION_MIN_SCORE = _RECENT_PROMOTION_MIN_SCORE
RECENT_PROMOTION_SLOTS = _RECENT_PROMOTION_SLOTS
MIN_GLOBAL_POPULARITY_USERS = _MIN_GLOBAL_POPULARITY_USERS
PANTRY_HARD_INGREDIENTS = _PANTRY_HARD_INGREDIENTS
PANTRY_SOFT_INGREDIENTS = _PANTRY_SOFT_INGREDIENTS
PANTRY_SOFT_WEIGHT = _PANTRY_SOFT_WEIGHT
INTERACTION_WEIGHTS = _INTERACTION_WEIGHTS
BOOLEAN_TEXT = _BOOLEAN_TEXT

clamp = _clamp
days_ago = _days_ago
interaction_recency_multiplier = _interaction_recency_multiplier
normalize_answer_values = _normalize_answer_values
parse_ingredients = _parse_ingredients
profile_ingredient_weight = _profile_ingredient_weight
hard_filter = _hard_filter
load_recipes = _load_recipes
