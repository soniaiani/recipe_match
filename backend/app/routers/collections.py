from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException, status
from app.database import get_supabase_admin
from app.middleware.auth import get_user_id
from app.models.schemas import ApiResponse, Collection, CollectionCreate

router = APIRouter(prefix="/collections", tags=["collections"])


@router.get("", response_model=ApiResponse[list[Collection]])
async def list_collections(user_id: str = Depends(get_user_id)):
    """List all collections for the current user."""
    admin = get_supabase_admin()
    res = admin.table("collections").select("*").eq("user_id", user_id).order("created_at").execute()
    return ApiResponse(data=[Collection(**c) for c in (res.data or [])])


@router.post("", response_model=ApiResponse[Collection], status_code=status.HTTP_201_CREATED)
async def create_collection(body: CollectionCreate, user_id: str = Depends(get_user_id)):
    """Create a new collection."""
    admin = get_supabase_admin()
    res = admin.table("collections").insert({"user_id": user_id, "name": body.name}).execute()
    if not res.data:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create collection")
    return ApiResponse(data=Collection(**res.data[0]))


@router.delete("/{collection_id}", response_model=ApiResponse[None])
async def delete_collection(collection_id: str, user_id: str = Depends(get_user_id)):
    """Delete a collection (recipes are unlinked, not deleted)."""
    admin = get_supabase_admin()
    res = admin.table("collections").delete().eq("id", collection_id).eq("user_id", user_id).execute()
    if not res.data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Collection not found")
    return ApiResponse(data=None)
