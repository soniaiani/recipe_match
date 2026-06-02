from __future__ import annotations

from fastapi import APIRouter, Depends, status

from app.middleware.auth import get_user_id
from app.models.common import ApiResponse
from app.models.saved import SaveRecipeRequest, SavedRecipe
from app.services.saved_service import (
    list_saved_response,
    save_recipe_response,
    saved_in_collection_response,
    unsave_recipe_response,
)

router = APIRouter(prefix="/saved", tags=["saved"])


@router.post("", response_model=ApiResponse[SavedRecipe], status_code=status.HTTP_201_CREATED)
async def save_recipe(body: SaveRecipeRequest, user_id: str = Depends(get_user_id)):
    """Save a recipe. Creates 'Saved' collection if no collection specified."""
    return save_recipe_response(body, user_id)


@router.delete("/{recipe_id}", response_model=ApiResponse[None])
async def unsave_recipe(recipe_id: int, user_id: str = Depends(get_user_id)):
    """Remove a saved recipe."""
    return unsave_recipe_response(recipe_id, user_id)


@router.get("", response_model=ApiResponse[list[SavedRecipe]])
async def list_saved(user_id: str = Depends(get_user_id)):
    """List all saved recipes with recipe summaries."""
    return list_saved_response(user_id)


@router.get("/collections/{collection_id}", response_model=ApiResponse[list[SavedRecipe]])
async def saved_in_collection(collection_id: str, user_id: str = Depends(get_user_id)):
    """List saved recipes in a specific collection."""
    return saved_in_collection_response(collection_id, user_id)
