"""Recipe and ingredient statistics cache for Ingredient Explorer."""
from __future__ import annotations

import math
from typing import Any

from app.database import get_supabase_admin
from app.services.explorer.common import RECIPE_FIELDS, parse_ingredients

_RECIPE_INGREDIENT_CACHE: list[tuple[dict[str, Any], set[str]]] | None = None
_RECIPE_DETAIL_CACHE: list[tuple[dict[str, Any], set[str]]] | None = None
_INGREDIENT_IDF: dict[str, float] = {}


def _fetch_all_recipe_ingredients() -> list[dict[str, Any]]:
    admin = get_supabase_admin()
    recipes: list[dict[str, Any]] = []
    offset = 0
    while True:
        page = (
            admin.table("recipes")
            .select(
                "id,ingredients_clean,"
                "is_vegetarian,is_vegan,is_gluten_free,is_dairy_free"
            )
            .range(offset, offset + 999)
            .execute()
            .data or []
        )
        recipes.extend(page)
        if len(page) < 1000:
            break
        offset += 1000
    return recipes


def _fetch_all_recipe_details() -> list[dict[str, Any]]:
    admin = get_supabase_admin()
    recipes: list[dict[str, Any]] = []
    offset = 0
    while True:
        page = (
            admin.table("recipes")
            .select(RECIPE_FIELDS)
            .range(offset, offset + 999)
            .execute()
            .data or []
        )
        recipes.extend(page)
        if len(page) < 1000:
            break
        offset += 1000
    return recipes


def recipe_ingredient_rows() -> list[tuple[dict[str, Any], set[str]]]:
    global _RECIPE_INGREDIENT_CACHE
    if _RECIPE_INGREDIENT_CACHE is None:
        _RECIPE_INGREDIENT_CACHE = [
            (recipe, parse_ingredients(recipe.get("ingredients_clean")))
            for recipe in _fetch_all_recipe_ingredients()
        ]
    return _RECIPE_INGREDIENT_CACHE


def recipe_detail_rows() -> list[tuple[dict[str, Any], set[str]]]:
    global _RECIPE_DETAIL_CACHE
    if _RECIPE_DETAIL_CACHE is None:
        _RECIPE_DETAIL_CACHE = [
            (recipe, parse_ingredients(recipe.get("ingredients_clean")))
            for recipe in _fetch_all_recipe_details()
        ]
    return _RECIPE_DETAIL_CACHE


def ingredient_idf() -> dict[str, float]:
    return _INGREDIENT_IDF


def warm_explorer_cache() -> None:
    recipe_ingredient_rows()


def warm_ingredient_idf() -> None:
    global _INGREDIENT_IDF
    admin = get_supabase_admin()
    rows: list[dict[str, Any]] = []
    offset = 0
    while True:
        page = (
            admin.table("ingredient_stats")
            .select("ingredient,recipe_count,total_recipes")
            .range(offset, offset + 999)
            .execute()
            .data
            or []
        )
        rows.extend(page)
        if len(page) < 1000:
            break
        offset += 1000

    _INGREDIENT_IDF = {}
    for row in rows:
        idf = math.log(row["total_recipes"] / max(row["recipe_count"], 1))
        _INGREDIENT_IDF[row["ingredient"]] = round(idf, 6)
    print(f"[explorer] Loaded IDF for {len(_INGREDIENT_IDF)} ingredients.")
