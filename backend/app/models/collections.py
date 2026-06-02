"""Collection schemas."""
from __future__ import annotations

from pydantic import BaseModel, Field


class CollectionCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)


class Collection(BaseModel):
    id: str
    user_id: str
    name: str
    created_at: str
