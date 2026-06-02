"""Interactive recommendation schemas."""
from __future__ import annotations

from typing import Any, List, Union

from pydantic import BaseModel, field_validator

from app.models.auth import DietaryProfile


class RecProgress(BaseModel):
    current: int
    max: int


class RecQuestion(BaseModel):
    id: str
    type: str
    options: list[str] | None = None
    any_option: str | None = None


class RecSessionStartRequest(BaseModel):
    dietary: DietaryProfile | None = None


class RecSessionStartResponse(BaseModel):
    session_id: str
    question: RecQuestion
    progress: RecProgress


class RecAnswerRequest(BaseModel):
    question_id: str
    answer: Union[str, List[str]]

    @field_validator("answer")
    @classmethod
    def answer_not_empty(cls, value: Any) -> Any:
        if isinstance(value, list) and len(value) == 0:
            raise ValueError("answer list must not be empty")
        if isinstance(value, str) and not value:
            raise ValueError("answer string must not be empty")
        return value


class RecScoredRecipe(BaseModel):
    id: int
    name: str
    image_url: str | None = None
    meal_type: str | None = None
    cuisine: str | None = None
    protein_type: str | None = None
    match_score: float


class RecAnswerResponse(BaseModel):
    status: str
    question: RecQuestion | None = None
    entropy: float | None = None
    questions_asked: int | None = None
    progress: RecProgress | None = None
    results: list[RecScoredRecipe] | None = None
    results_count: int | None = None


class RecInteractionRequest(BaseModel):
    recipe_id: int
    interaction_type: str


class RecResultsResponse(BaseModel):
    results: list[RecScoredRecipe]
    results_count: int
