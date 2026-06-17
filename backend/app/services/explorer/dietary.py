"""Dietary filtering helpers for Ingredient Explorer."""
from __future__ import annotations

from typing import Any

from fastapi import HTTPException, status

from app.services.recommendations.filters import (
    normalize_excluded_ingredients,
    recipe_uses_excluded_ingredient,
)


def dietary_from_payload(payload: dict | None) -> dict[str, Any]:
    meta = (payload or {}).get("user_metadata") or {}
    return {
        "is_vegetarian": bool(meta.get("is_vegetarian", False)),
        "is_vegan": bool(meta.get("is_vegan", False)),
        "is_gluten_free": bool(meta.get("is_gluten_free", False)),
        "is_dairy_free": bool(meta.get("is_dairy_free", False)),
        "excluded_ingredients": normalize_excluded_ingredients(
            meta.get("excluded_ingredients", [])
        ),
    }


def recipe_matches_dietary(recipe: dict[str, Any], dietary: dict[str, Any]) -> bool:
    if dietary.get("is_vegetarian") and not recipe.get("is_vegetarian"):
        return False
    if dietary.get("is_vegan") and not recipe.get("is_vegan"):
        return False
    if dietary.get("is_gluten_free") and not recipe.get("is_gluten_free"):
        return False
    if dietary.get("is_dairy_free") and not recipe.get("is_dairy_free"):
        return False
    return not recipe_uses_excluded_ingredient(
        recipe,
        dietary.get("excluded_ingredients", []),
    )


def compatible_rows(
    parsed_rows: list[tuple[dict[str, Any], set[str]]],
    dietary: dict[str, Any],
) -> list[tuple[dict[str, Any], set[str]]]:
    return [
        (recipe, ingredients)
        for recipe, ingredients in parsed_rows
        if recipe_matches_dietary(recipe, dietary)
    ]


def validate_selected_allowed(selected: list[str], dietary: dict[str, Any]) -> None:
    excluded = set(dietary.get("excluded_ingredients", []))
    blocked = sorted(set(selected) & excluded)
    if blocked:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Excluded ingredient selected: {', '.join(blocked)}",
        )
