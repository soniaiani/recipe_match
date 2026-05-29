from __future__ import annotations
from typing import Any, Generic, TypeVar, Union, List
from pydantic import BaseModel, EmailStr, Field, field_validator

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Response envelope
# ---------------------------------------------------------------------------

class ApiResponse(BaseModel, Generic[T]):
    data: T | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

class DietaryProfile(BaseModel):
    is_vegetarian: bool = False
    is_vegan: bool = False
    is_gluten_free: bool = False
    is_dairy_free: bool = False
    excluded_ingredients: list[str] = Field(default_factory=list)

    @field_validator("excluded_ingredients")
    @classmethod
    def normalize_excluded_ingredients(cls, v: list[str]) -> list[str]:
        seen: set[str] = set()
        normalized: list[str] = []
        for item in v:
            value = " ".join(str(item).strip().lower().split())
            if value and value not in seen:
                seen.add(value)
                normalized.append(value)
        return normalized


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    dietary: DietaryProfile = Field(default_factory=DietaryProfile)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=128)


class UserProfile(BaseModel):
    id: str
    email: str
    dietary: DietaryProfile


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserProfile


# ---------------------------------------------------------------------------
# Recipes
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Session — Find Your Recipe
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Collections
# ---------------------------------------------------------------------------

class CollectionCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)


class Collection(BaseModel):
    id: str
    user_id: str
    name: str
    created_at: str


# ---------------------------------------------------------------------------
# Saved recipes
# ---------------------------------------------------------------------------

class SaveRecipeRequest(BaseModel):
    recipe_id: int
    collection_id: str | None = None


class SavedRecipe(BaseModel):
    id: str
    recipe_id: int
    collection_id: str | None = None
    saved_at: str
    recipe: RecipeSummary | None = None


# ---------------------------------------------------------------------------
# For You
# ---------------------------------------------------------------------------

class ForYouResponse(BaseModel):
    recipes: list[RecipeSummary]


# ---------------------------------------------------------------------------
# Recommendation Session (Bayesian engine)
# ---------------------------------------------------------------------------

class RecProgress(BaseModel):
    current: int
    max: int


class RecQuestion(BaseModel):
    id: str
    type: str                        # "categorical" | "multiselect" | "boolean"
    options: list[str] | None = None
    any_option: str | None = None    # present on multiselect questions


class RecSessionStartRequest(BaseModel):
    dietary: DietaryProfile | None = None


class RecSessionStartResponse(BaseModel):
    session_id: str
    question: RecQuestion
    progress: RecProgress


class RecAnswerRequest(BaseModel):
    question_id: str
    # categorical/boolean → str; multiselect → list[str]
    answer: Union[str, List[str]]

    @field_validator("answer")
    @classmethod
    def answer_not_empty(cls, v: Any) -> Any:
        if isinstance(v, list) and len(v) == 0:
            raise ValueError("answer list must not be empty")
        if isinstance(v, str) and not v:
            raise ValueError("answer string must not be empty")
        return v


class RecScoredRecipe(BaseModel):
    id: int
    name: str
    image_url: str | None = None
    meal_type: str | None = None
    cuisine: str | None = None
    protein_type: str | None = None
    match_score: float  # 0–100, percentage of answered weights matched


class RecAnswerResponse(BaseModel):
    status: str                          # "continue" | "done"
    question: RecQuestion | None = None
    entropy: float | None = None
    questions_asked: int | None = None
    progress: RecProgress | None = None
    results: list[RecScoredRecipe] | None = None
    results_count: int | None = None


class RecInteractionRequest(BaseModel):
    recipe_id: int
    interaction_type: str  # "view" | "save" | "cook" | "skip"


class RecResultsResponse(BaseModel):
    results: list[RecScoredRecipe]
    results_count: int
