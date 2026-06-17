"""Collection business logic."""
from __future__ import annotations

from fastapi import HTTPException, status

from app.database import get_supabase_admin
from app.models.collections import Collection, CollectionCreate
from app.models.common import ApiResponse


def list_user_collections(user_id: str) -> ApiResponse[list[Collection]]:
    admin = get_supabase_admin()
    res = admin.table("collections").select("*").eq("user_id", user_id).order("created_at").execute()
    return ApiResponse(data=[Collection(**collection) for collection in (res.data or [])])


def create_user_collection(body: CollectionCreate, user_id: str) -> ApiResponse[Collection]:
    admin = get_supabase_admin()
    res = admin.table("collections").insert({"user_id": user_id, "name": body.name}).execute()
    if not res.data:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create collection",
        )
    return ApiResponse(data=Collection(**res.data[0]))


def delete_user_collection(collection_id: str, user_id: str) -> ApiResponse[None]:
    admin = get_supabase_admin()
    res = admin.table("collections").delete().eq("id", collection_id).eq("user_id", user_id).execute()
    if not res.data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Collection not found")
    return ApiResponse(data=None)
