from __future__ import annotations
import json
from typing import Any

from fastapi import APIRouter, HTTPException, status
from app.database import get_supabase_admin
from app.models.schemas import (
    ApiResponse,
    IngredientSuggestionsResponse,
    RecipeDetail,
    ShoppingListResponse,
    SimilarRecipe,
    SimilarRecipesResponse,
)

router = APIRouter(prefix="/recipes", tags=["recipes"])

_DETAIL_FIELDS = (
    "id,name,description,prep_time,cook_time,total_time,servings,"
    "ingredients,ingredients_clean,directions,image_url,meal_type,"
    "protein_type,cuisine,total_minutes,is_vegetarian,is_vegan,"
    "is_gluten_free,is_dairy_free,is_nut_free,is_quick,is_spicy,"
    "is_sweet,needs_oven,needs_stovetop,is_no_cook"
)
_SIMILAR_LIMIT = 20
_INGREDIENT_FIELDS = "ingredients,ingredients_clean"
_INGREDIENT_PAGE_SIZE = 1000
_INGREDIENT_OPTIONS_CACHE: list[str] | None = None


def _normalize_ingredient(value: Any) -> str:
    return " ".join(str(value).strip().lower().split())


def _parse_clean_ingredients(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [_normalize_ingredient(item) for item in value]
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            decoded = None
        if isinstance(decoded, list):
            return [_normalize_ingredient(item) for item in decoded]
    return []


def _parse_raw_ingredients(value: Any) -> list[str]:
    if not isinstance(value, str):
        return []
    return [
        _normalize_ingredient(item)
        for item in value.replace("|", "\n").replace(";", "\n").split("\n")
    ]


def _load_ingredient_options() -> list[str]:
    global _INGREDIENT_OPTIONS_CACHE
    if _INGREDIENT_OPTIONS_CACHE is not None:
        return _INGREDIENT_OPTIONS_CACHE

    admin = get_supabase_admin()
    ingredients: set[str] = set()
    start = 0

    while True:
        end = start + _INGREDIENT_PAGE_SIZE - 1
        res = admin.table("recipes").select(_INGREDIENT_FIELDS).range(start, end).execute()
        rows = res.data or []
        for row in rows:
            clean_items = _parse_clean_ingredients(row.get("ingredients_clean"))
            raw_items = _parse_raw_ingredients(row.get("ingredients"))
            for item in clean_items or raw_items:
                if item:
                    ingredients.add(item)
        if len(rows) < _INGREDIENT_PAGE_SIZE:
            break
        start += _INGREDIENT_PAGE_SIZE

    _INGREDIENT_OPTIONS_CACHE = sorted(ingredients)
    return _INGREDIENT_OPTIONS_CACHE


def _fetch_recipe(recipe_id: int) -> dict:
    admin = get_supabase_admin()
    res = admin.table("recipes").select(_DETAIL_FIELDS).eq("id", recipe_id).single().execute()
    if not res.data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Recipe not found")
    return res.data


@router.get("/ingredients", response_model=ApiResponse[IngredientSuggestionsResponse])
async def get_ingredient_suggestions(q: str = "", limit: int = 20):
    """Return ingredient suggestions from all recipe ingredients."""
    safe_limit = max(1, min(limit, 50))
    query = _normalize_ingredient(q)
    options = _load_ingredient_options()
    suggestions = [
        ingredient
        for ingredient in options
        if not query or query in ingredient
    ][:safe_limit]
    return ApiResponse(data=IngredientSuggestionsResponse(ingredients=suggestions))


@router.get("/{recipe_id}", response_model=ApiResponse[RecipeDetail])
async def get_recipe(recipe_id: int):
    """Get full recipe details."""
    data = _fetch_recipe(recipe_id)
    # ingredients_clean may be a JSONB list from Supabase
    clean = data.get("ingredients_clean") or []
    if isinstance(clean, str):
        import json
        try:
            clean = json.loads(clean)
        except Exception:
            clean = []
    data["ingredients_clean"] = clean
    return ApiResponse(data=RecipeDetail(**data))


@router.get("/{recipe_id}/shopping-list", response_model=ApiResponse[ShoppingListResponse])
async def get_shopping_list(recipe_id: int):
    """Return ingredients with quantities for a recipe."""
    data = _fetch_recipe(recipe_id)
    raw: str = data.get("ingredients") or ""
    ingredients = [s.strip() for s in raw.replace("|", "\n").replace(";", "\n").split("\n") if s.strip()]
    return ApiResponse(data=ShoppingListResponse(
        recipe_id=recipe_id,
        recipe_name=data.get("name", ""),
        ingredients=ingredients,
    ))


@router.get("/{recipe_id}/similar", response_model=ApiResponse[SimilarRecipesResponse])
async def get_similar_recipes(recipe_id: int, limit: int = _SIMILAR_LIMIT):
    """Return recipes semantically similar to this recipe using pgvector embeddings."""
    admin = get_supabase_admin()
    safe_limit = max(1, min(limit, 50))

    current = admin.table("recipes").select("id,embedding").eq("id", recipe_id).single().execute()
    if not current.data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Recipe not found")
    if current.data.get("embedding") is None:
        return ApiResponse(data=SimilarRecipesResponse(recipes=[]))

    res = admin.rpc("match_similar_recipes", {
        "target_recipe_id": recipe_id,
        "match_count": safe_limit,
    }).execute()

    return ApiResponse(data=SimilarRecipesResponse(
        recipes=[SimilarRecipe(**row) for row in (res.data or [])]
    ))
