"""Authentication and user profile schemas."""
from __future__ import annotations

from pydantic import BaseModel, EmailStr, Field, field_validator


class DietaryProfile(BaseModel):
    is_vegetarian: bool = False
    is_vegan: bool = False
    is_gluten_free: bool = False
    is_dairy_free: bool = False
    excluded_ingredients: list[str] = Field(default_factory=list)

    @field_validator("excluded_ingredients")
    @classmethod
    def normalize_excluded_ingredients(cls, value: list[str]) -> list[str]:
        seen: set[str] = set()
        normalized: list[str] = []
        for item in value:
            ingredient = " ".join(str(item).strip().lower().split())
            if ingredient and ingredient not in seen:
                seen.add(ingredient)
                normalized.append(ingredient)
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
