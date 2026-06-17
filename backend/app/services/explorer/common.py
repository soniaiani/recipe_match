"""Shared Ingredient Explorer constants and parsing helpers."""
from __future__ import annotations

import json
from typing import Any, TypeVar

MAX_EXPAND_CANDIDATES = 250
T = TypeVar("T")

PANTRY_HARD = {
    "salt", "pepper", "black pepper", "water", "oil",
    "olive oil", "vegetable oil", "cooking spray",
}

PANTRY_SOFT = {
    "sugar", "all-purpose flour", "butter", "milk", "egg",
}

RECIPE_FIELDS = (
    "id,name,description,image_url,prep_time,cook_time,total_time,servings,"
    "meal_type,cuisine,protein_type,ingredients_clean,ingredients_clean_str,"
    "is_vegetarian,is_vegan,is_gluten_free,is_dairy_free"
)


def normalize_ingredient(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).lower().strip()
    return normalized or None


def normalize_many(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        normalized = normalize_ingredient(value)
        if normalized and normalized not in seen:
            seen.add(normalized)
            output.append(normalized)
    return output


def parse_ingredients(raw: Any) -> set[str]:
    if not raw:
        return set()
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError:
        return set()
    if not isinstance(parsed, list):
        return set()
    return {
        normalized
        for item in parsed
        if (normalized := normalize_ingredient(item))
    }


def chunks(items: list[T], size: int):
    for start in range(0, len(items), size):
        yield items[start:start + size]
