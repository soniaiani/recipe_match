from __future__ import annotations

import json
import re
from typing import Any


def normalize_excluded_ingredients(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []

    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        normalized = re.sub(r"\s+", " ", str(value).strip().lower())
        if normalized and normalized not in seen:
            seen.add(normalized)
            out.append(normalized)
    return out


def recipe_uses_excluded_ingredient(recipe: dict[str, Any], excluded: list[str]) -> bool:
    if not excluded:
        return False

    clean = recipe.get("ingredients_clean") or []
    if isinstance(clean, str):
        try:
            parsed = json.loads(clean)
            clean = parsed if isinstance(parsed, list) else [clean]
        except Exception:
            clean = [clean]

    raw_parts = [recipe.get("ingredients") or ""]
    raw_parts.extend(str(item) for item in clean if item)
    haystack = " ".join(raw_parts).lower()

    return any(ingredient in haystack for ingredient in excluded)


def filter_excluded_ingredients(
    recipes: list[dict[str, Any]],
    excluded_ingredients: Any,
) -> list[dict[str, Any]]:
    excluded = normalize_excluded_ingredients(excluded_ingredients)
    if not excluded:
        return recipes
    return [
        recipe for recipe in recipes
        if not recipe_uses_excluded_ingredient(recipe, excluded)
    ]
