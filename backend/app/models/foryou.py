"""For You schemas."""
from __future__ import annotations

from pydantic import BaseModel, Field

from app.models.recipes import RecipeSummary


class ForYouResponse(BaseModel):
    recipes: list[RecipeSummary]


class TasteCard(BaseModel):
    title: str
    text: str
    traits: list[str] = Field(default_factory=list)
    ingredients: list[str] = Field(default_factory=list)
    recipes: list[str] = Field(default_factory=list)


class TasteAttributeTrait(BaseModel):
    name: str
    global_recipe_count: int = 0
    prevalence: float
    global_prevalence: float
    lift: float
    score: float
    is_globally_common: bool = False


class TasteCategoricalTrait(BaseModel):
    value: str
    prevalence: float
    global_prevalence: float
    lift: float
    is_distinctive: bool


class TasteCluster(BaseModel):
    cluster_id: int
    weight: float
    similarity: float
    dominant_cuisine: str | None = None
    dominant_meal_type: str | None = None
    dominant_protein_type: str | None = None
    top_ingredients: list[str] = Field(default_factory=list)
    top_ingredient_traits: list[TasteAttributeTrait] = Field(default_factory=list)
    top_boolean_traits: list[TasteAttributeTrait] = Field(default_factory=list)
    categorical_traits: dict[str, TasteCategoricalTrait] = Field(default_factory=dict)
    representative_recipes: list[str] = Field(default_factory=list)


class TasteProfileResponse(BaseModel):
    status: str
    compact_summary: str | None = None
    description: str | None = None
    taste_cards: list[TasteCard] = Field(default_factory=list)
    top_clusters: list[TasteCluster] = Field(default_factory=list)
    generated_at: str | None = None
    source: str = "none"


class TasteClusterRecipes(BaseModel):
    cluster_id: int
    recipes: list[RecipeSummary] = Field(default_factory=list)


class TasteProfileRecipesResponse(BaseModel):
    clusters: list[TasteClusterRecipes] = Field(default_factory=list)
