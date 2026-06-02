"""Pydantic schemas for Ingredient Explorer."""
from __future__ import annotations

from pydantic import BaseModel


class ExplorerStartRequest(BaseModel):
    ingredient: str


class ExplorerExpandRequest(BaseModel):
    selected_ingredients: list[str]
    session_id: str | None = None
    finalize: bool = False


class ExplorerSuggestion(BaseModel):
    ingredient: str
    score: float | None = None
    ppmi_score: float | None = None


class ExplorerStartResponse(BaseModel):
    center: str
    suggestions: list[ExplorerSuggestion]
    recipe_count: int


class ExplorerExpandResponse(BaseModel):
    suggestions: list[ExplorerSuggestion]
    recipe_count: int
    relaxed: bool


class ExplorerRecipe(BaseModel):
    id: int
    name: str
    description: str | None = None
    image_url: str | None = None
    prep_time: str | None = None
    cook_time: str | None = None
    total_time: str | None = None
    servings: int | None = None
    meal_type: str | None = None
    cuisine: str | None = None
    protein_type: str | None = None
    ingredients_clean_str: str | None = None
    is_vegetarian: bool | None = None
    is_vegan: bool | None = None
    is_gluten_free: bool | None = None
    is_dairy_free: bool | None = None
    jaccard_score: float


class ExplorerRecommendResponse(BaseModel):
    recipes: list[ExplorerRecipe]
    recipe_count: int
    relaxed: bool


class ExplorerSearchResponse(BaseModel):
    ingredients: list[str]
