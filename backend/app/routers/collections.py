from __future__ import annotations

from fastapi import APIRouter, Depends, status

from app.middleware.auth import get_user_id
from app.models.collections import Collection, CollectionCreate
from app.models.common import ApiResponse
from app.services.collections.service import (
    create_user_collection,
    delete_user_collection,
    list_user_collections,
)

router = APIRouter(prefix="/collections", tags=["collections"])


@router.get("", response_model=ApiResponse[list[Collection]])
async def list_collections(user_id: str = Depends(get_user_id)):
    """List all collections for the current user."""
    return list_user_collections(user_id)


@router.post("", response_model=ApiResponse[Collection], status_code=status.HTTP_201_CREATED)
async def create_collection(body: CollectionCreate, user_id: str = Depends(get_user_id)):
    """Create a new collection."""
    return create_user_collection(body, user_id)


@router.delete("/{collection_id}", response_model=ApiResponse[None])
async def delete_collection(collection_id: str, user_id: str = Depends(get_user_id)):
    """Delete a collection (recipes are unlinked, not deleted)."""
    return delete_user_collection(collection_id, user_id)
