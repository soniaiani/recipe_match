"""Recipe schemas."""
from __future__ import annotations

from pydantic import BaseModel, Field


class RecipeSummary(BaseModel):
    id: int
    name: str
    description: str | None = None
    image_url: str | None = None
    meal_type: str | None = None
    cuisine: str | None = None
    total_minutes: float | None = None
    is_vegetarian: bool = False
    is_vegan: bool = False
    is_gluten_free: bool = False
    is_dairy_free: bool = False
    is_quick: bool = False


class RecipeDetail(RecipeSummary):
    prep_time: str | None = None
    cook_time: str | None = None
    total_time: str | None = None
    servings: int | None = None
    ingredients: str | None = None
    ingredients_clean: list[str] = Field(default_factory=list)
    directions: str | None = None
    protein_type: str | None = None
    is_nut_free: bool = False
    is_spicy: bool = False
    is_sweet: bool = False
    needs_oven: bool = False
    needs_stovetop: bool = False
    is_no_cook: bool = False


class ShoppingListResponse(BaseModel):
    recipe_id: int
    recipe_name: str
    ingredients: list[str]


class IngredientSuggestionsResponse(BaseModel):
    ingredients: list[str]


class SimilarRecipe(RecipeSummary):
    similarity: float


class SimilarRecipesResponse(BaseModel):
    recipes: list[SimilarRecipe]
