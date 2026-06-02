"""Saved recipe business logic."""
from __future__ import annotations

from fastapi import HTTPException, status

from app.database import get_supabase_admin
from app.models.common import ApiResponse
from app.models.recipes import RecipeSummary
from app.models.saved import SaveRecipeRequest, SavedRecipe

_SUMMARY_FIELDS = (
    "id,name,description,image_url,meal_type,cuisine,"
    "total_minutes,is_vegetarian,is_vegan,is_gluten_free,is_dairy_free,is_quick"
)
_SAVE_INTERACTION_WEIGHT = 2.0


def _get_or_create_default_collection(user_id: str) -> str:
    admin = get_supabase_admin()
    res = admin.table("collections").select("id").eq("user_id", user_id).eq("name", "Saved").execute()
    if res.data:
        return res.data[0]["id"]
    created = admin.table("collections").insert({"user_id": user_id, "name": "Saved"}).execute()
    return created.data[0]["id"]


def _log_save_interaction(user_id: str, recipe_id: int) -> None:
    try:
        get_supabase_admin().table("recipe_interactions").insert({
            "user_id": user_id,
            "recipe_id": recipe_id,
            "interaction_type": "save",
            "weight": _SAVE_INTERACTION_WEIGHT,
        }).execute()
    except Exception as exc:
        print(f"[saved] failed to log save interaction: {exc}")


def _saved_recipe_from_row(row: dict, include_recipe: bool = False) -> SavedRecipe:
    recipe_data = row.get("recipes") if include_recipe else None
    return SavedRecipe(
        id=row["id"],
        recipe_id=row["recipe_id"],
        collection_id=row.get("collection_id"),
        saved_at=str(row["saved_at"]),
        recipe=RecipeSummary(**recipe_data) if recipe_data else None,
    )


def save_recipe_response(body: SaveRecipeRequest, user_id: str) -> ApiResponse[SavedRecipe]:
    admin = get_supabase_admin()
    collection_id = body.collection_id or _get_or_create_default_collection(user_id)

    try:
        res = admin.table("saved_recipes").insert({
            "user_id": user_id,
            "recipe_id": body.recipe_id,
            "collection_id": collection_id,
        }).execute()
    except Exception as exc:
        if "unique" in str(exc).lower():
            existing = (
                admin.table("saved_recipes")
                .select("id,recipe_id,collection_id,saved_at")
                .eq("user_id", user_id)
                .eq("recipe_id", body.recipe_id)
                .limit(1)
                .execute()
            )
            if existing.data:
                return ApiResponse(data=_saved_recipe_from_row(existing.data[0]))
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Recipe already saved")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    _log_save_interaction(user_id, body.recipe_id)
    return ApiResponse(data=_saved_recipe_from_row(res.data[0]))


def unsave_recipe_response(recipe_id: int, user_id: str) -> ApiResponse[None]:
    admin = get_supabase_admin()
    res = admin.table("saved_recipes").delete().eq("user_id", user_id).eq("recipe_id", recipe_id).execute()
    if not res.data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Saved recipe not found")
    return ApiResponse(data=None)


def list_saved_response(user_id: str) -> ApiResponse[list[SavedRecipe]]:
    admin = get_supabase_admin()
    res = (
        admin.table("saved_recipes")
        .select(f"id,recipe_id,collection_id,saved_at,recipes({_SUMMARY_FIELDS})")
        .eq("user_id", user_id)
        .order("saved_at", desc=True)
        .execute()
    )
    return ApiResponse(data=[
        _saved_recipe_from_row(row, include_recipe=True)
        for row in (res.data or [])
    ])


def saved_in_collection_response(collection_id: str, user_id: str) -> ApiResponse[list[SavedRecipe]]:
    admin = get_supabase_admin()
    res = (
        admin.table("saved_recipes")
        .select(f"id,recipe_id,collection_id,saved_at,recipes({_SUMMARY_FIELDS})")
        .eq("user_id", user_id)
        .eq("collection_id", collection_id)
        .order("saved_at", desc=True)
        .execute()
    )
    return ApiResponse(data=[
        _saved_recipe_from_row(row, include_recipe=True)
        for row in (res.data or [])
    ])
