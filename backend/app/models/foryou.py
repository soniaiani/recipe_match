"""For You schemas."""
from __future__ import annotations

from pydantic import BaseModel

from app.models.recipes import RecipeSummary


class ForYouResponse(BaseModel):
    recipes: list[RecipeSummary]
