from __future__ import annotations

from fastapi import APIRouter

from app.models.common import ApiResponse
from app.models.recipes import (
    IngredientSuggestionsResponse,
    RecipeDetail,
    ShoppingListResponse,
    SimilarRecipesResponse,
)
from app.services.recipe_service import (
    get_ingredient_suggestion_response,
    get_recipe_detail_response,
    get_shopping_list_response,
    get_similar_recipes_response,
)

router = APIRouter(prefix="/recipes", tags=["recipes"])


@router.get("/ingredients", response_model=ApiResponse[IngredientSuggestionsResponse])
async def get_ingredient_suggestions(q: str = "", limit: int = 20):
    """Return ingredient suggestions from all recipe ingredients."""
    return get_ingredient_suggestion_response(q, limit)


@router.get("/{recipe_id}", response_model=ApiResponse[RecipeDetail])
async def get_recipe(recipe_id: int):
    """Get full recipe details."""
    return get_recipe_detail_response(recipe_id)


@router.get("/{recipe_id}/shopping-list", response_model=ApiResponse[ShoppingListResponse])
async def get_shopping_list(recipe_id: int):
    """Return ingredients with quantities for a recipe."""
    return get_shopping_list_response(recipe_id)


@router.get("/{recipe_id}/similar", response_model=ApiResponse[SimilarRecipesResponse])
async def get_similar_recipes(recipe_id: int, limit: int = 20):
    """Return recipes semantically similar to this recipe using pgvector embeddings."""
    return get_similar_recipes_response(recipe_id, limit)
