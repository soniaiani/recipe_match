from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException, status
from app.database import get_supabase_admin
from app.middleware.auth import get_user_id
from app.models.schemas import ApiResponse, RecipeSummary, SavedRecipe, SaveRecipeRequest

router = APIRouter(prefix="/saved", tags=["saved"])

_SUMMARY_FIELDS = (
    "id,name,description,image_url,meal_type,cuisine,"
    "total_minutes,is_vegetarian,is_vegan,is_gluten_free,is_dairy_free,is_quick"
)


def _get_or_create_default_collection(user_id: str) -> str:
    """Return 'Saved' collection id, creating it if it doesn't exist."""
    admin = get_supabase_admin()
    res = admin.table("collections").select("id").eq("user_id", user_id).eq("name", "Saved").execute()
    if res.data:
        return res.data[0]["id"]
    created = admin.table("collections").insert({"user_id": user_id, "name": "Saved"}).execute()
    return created.data[0]["id"]


@router.post("", response_model=ApiResponse[SavedRecipe], status_code=status.HTTP_201_CREATED)
async def save_recipe(body: SaveRecipeRequest, user_id: str = Depends(get_user_id)):
    """Save a recipe. Creates 'Saved' collection if no collection specified."""
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
                row = existing.data[0]
                return ApiResponse(data=SavedRecipe(
                    id=row["id"],
                    recipe_id=row["recipe_id"],
                    collection_id=row.get("collection_id"),
                    saved_at=str(row["saved_at"]),
                ))
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Recipe already saved")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    row = res.data[0]
    return ApiResponse(data=SavedRecipe(
        id=row["id"],
        recipe_id=row["recipe_id"],
        collection_id=row.get("collection_id"),
        saved_at=str(row["saved_at"]),
    ))


@router.delete("/{recipe_id}", response_model=ApiResponse[None])
async def unsave_recipe(recipe_id: int, user_id: str = Depends(get_user_id)):
    """Remove a saved recipe."""
    admin = get_supabase_admin()
    res = admin.table("saved_recipes").delete().eq("user_id", user_id).eq("recipe_id", recipe_id).execute()
    if not res.data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Saved recipe not found")
    return ApiResponse(data=None)


@router.get("", response_model=ApiResponse[list[SavedRecipe]])
async def list_saved(user_id: str = Depends(get_user_id)):
    """List all saved recipes with recipe summaries."""
    admin = get_supabase_admin()
    res = (
        admin.table("saved_recipes")
        .select(f"id,recipe_id,collection_id,saved_at,recipes({_SUMMARY_FIELDS})")
        .eq("user_id", user_id)
        .order("saved_at", desc=True)
        .execute()
    )
    saved = []
    for row in res.data or []:
        recipe_data = row.get("recipes")
        saved.append(SavedRecipe(
            id=row["id"],
            recipe_id=row["recipe_id"],
            collection_id=row.get("collection_id"),
            saved_at=str(row["saved_at"]),
            recipe=RecipeSummary(**recipe_data) if recipe_data else None,
        ))
    return ApiResponse(data=saved)


@router.get("/collections/{collection_id}", response_model=ApiResponse[list[SavedRecipe]])
async def saved_in_collection(collection_id: str, user_id: str = Depends(get_user_id)):
    """List saved recipes in a specific collection."""
    admin = get_supabase_admin()
    res = (
        admin.table("saved_recipes")
        .select(f"id,recipe_id,collection_id,saved_at,recipes({_SUMMARY_FIELDS})")
        .eq("user_id", user_id)
        .eq("collection_id", collection_id)
        .order("saved_at", desc=True)
        .execute()
    )
    saved = []
    for row in res.data or []:
        recipe_data = row.get("recipes")
        saved.append(SavedRecipe(
            id=row["id"],
            recipe_id=row["recipe_id"],
            collection_id=row.get("collection_id"),
            saved_at=str(row["saved_at"]),
            recipe=RecipeSummary(**recipe_data) if recipe_data else None,
        ))
    return ApiResponse(data=saved)
