"""Saved recipe schemas."""
from __future__ import annotations

from pydantic import BaseModel

from app.models.recipes import RecipeSummary


class SaveRecipeRequest(BaseModel):
    recipe_id: int
    collection_id: str | None = None


class SavedRecipe(BaseModel):
    id: str
    recipe_id: int
    collection_id: str | None = None
    saved_at: str
    recipe: RecipeSummary | None = None
