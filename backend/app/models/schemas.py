"""Compatibility exports for API schemas.

Prefer importing schemas from their domain modules:
`app.models.auth`, `app.models.recipes`, `app.models.recommendations`, etc.
"""
from __future__ import annotations

from app.models.auth import (
    AuthResponse,
    DietaryProfile,
    LoginRequest,
    RegisterRequest,
    UserProfile,
)
from app.models.collections import Collection, CollectionCreate
from app.models.common import ApiResponse
from app.models.foryou import ForYouResponse
from app.models.recipes import (
    IngredientSuggestionsResponse,
    RecipeDetail,
    RecipeSummary,
    ShoppingListResponse,
    SimilarRecipe,
    SimilarRecipesResponse,
)
from app.models.recommendations import (
    RecAnswerRequest,
    RecAnswerResponse,
    RecInteractionRequest,
    RecProgress,
    RecQuestion,
    RecResultsResponse,
    RecScoredRecipe,
    RecSessionStartRequest,
    RecSessionStartResponse,
)
from app.models.saved import SaveRecipeRequest, SavedRecipe

__all__ = [
    "ApiResponse",
    "AuthResponse",
    "Collection",
    "CollectionCreate",
    "DietaryProfile",
    "ForYouResponse",
    "IngredientSuggestionsResponse",
    "LoginRequest",
    "RecAnswerRequest",
    "RecAnswerResponse",
    "RecInteractionRequest",
    "RecProgress",
    "RecQuestion",
    "RecResultsResponse",
    "RecScoredRecipe",
    "RecSessionStartRequest",
    "RecSessionStartResponse",
    "RecipeDetail",
    "RecipeSummary",
    "RegisterRequest",
    "SaveRecipeRequest",
    "SavedRecipe",
    "ShoppingListResponse",
    "SimilarRecipe",
    "SimilarRecipesResponse",
    "UserProfile",
]
