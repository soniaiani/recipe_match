"""For You — personalized recommendations based on saved recipe features."""
from __future__ import annotations
from collections import Counter
from typing import Any
from fastapi import APIRouter, Depends
from app.database import get_supabase_admin
from app.middleware.auth import get_current_user
from app.models.schemas import ApiResponse, ForYouResponse, RecipeSummary
from app.recommender.filters import filter_excluded_ingredients

router = APIRouter(prefix="/foryou", tags=["foryou"])

_SUMMARY_FIELDS = (
    "id,name,description,image_url,meal_type,cuisine,"
    "total_minutes,is_vegetarian,is_vegan,is_gluten_free,is_dairy_free,is_quick"
)
_FILTER_FIELDS = _SUMMARY_FIELDS + ",ingredients,ingredients_clean"
_PROFILE_FIELDS = "meal_type,cuisine,protein_type,is_spicy,is_sweet,is_quick"
_RECOMMENDATION_COUNT = 20


@router.get("", response_model=ApiResponse[ForYouResponse])
async def for_you(payload: dict = Depends(get_current_user)):
    """Return personalized recipe recommendations based on saved recipe history."""
    admin = get_supabase_admin()
    user_id = payload["sub"]
    meta = payload.get("user_metadata") or {}
    excluded_ingredients = meta.get("excluded_ingredients", [])

    # Fetch saved recipe IDs and their feature profiles
    saved_res = (
        admin.table("saved_recipes")
        .select(f"recipe_id,recipes({_PROFILE_FIELDS})")
        .eq("user_id", user_id)
        .execute()
    )
    saved_rows = saved_res.data or []

    if not saved_rows:
        # Cold start: return high-scoring popular recipes
        res = (
            admin.table("recipes")
            .select(_FILTER_FIELDS)
            .limit(_RECOMMENDATION_COUNT * 3)
            .execute()
        )
        recipes = filter_excluded_ingredients(res.data or [], excluded_ingredients)[:_RECOMMENDATION_COUNT]
        return ApiResponse(data=ForYouResponse(
            recipes=[RecipeSummary(**r) for r in recipes]
        ))

    saved_ids = {row["recipe_id"] for row in saved_rows}
    profiles = [row["recipes"] for row in saved_rows if row.get("recipes")]
    if not profiles:
        res = (
            admin.table("recipes")
            .select(_FILTER_FIELDS)
            .limit(_RECOMMENDATION_COUNT * 3)
            .execute()
        )
        recipes = filter_excluded_ingredients(res.data or [], excluded_ingredients)[:_RECOMMENDATION_COUNT]
        return ApiResponse(data=ForYouResponse(
            recipes=[RecipeSummary(**r) for r in recipes]
        ))

    # Build preference profile from saved recipes
    meal_counter: Counter = Counter()
    cuisine_counter: Counter = Counter()
    spicy_count = 0
    sweet_count = 0
    quick_count = 0

    for p in profiles:
        if p.get("meal_type"):
            meal_counter[p["meal_type"]] += 1
        if p.get("cuisine"):
            cuisine_counter[p["cuisine"]] += 1
        if p.get("is_spicy"):
            spicy_count += 1
        if p.get("is_sweet"):
            sweet_count += 1
        if p.get("is_quick"):
            quick_count += 1

    total = len(profiles)
    top_meals = [m for m, _ in meal_counter.most_common(3)]
    top_cuisines = [c for c, _ in cuisine_counter.most_common(3)]

    # Fetch candidates matching top preferences, excluding already saved
    query = admin.table("recipes").select(_FILTER_FIELDS)

    if top_meals:
        query = query.in_("meal_type", top_meals)

    if top_cuisines:
        query = query.in_("cuisine", top_cuisines)

    if spicy_count / total > 0.5:
        query = query.eq("is_spicy", True)

    if quick_count / total > 0.6:
        query = query.eq("is_quick", True)

    res = query.limit(_RECOMMENDATION_COUNT * 3).execute()
    candidates = [r for r in (res.data or []) if r["id"] not in saved_ids]
    candidates = filter_excluded_ingredients(candidates, excluded_ingredients)

    # Score candidates by how well they match the preference profile
    def _preference_score(recipe: dict[str, Any]) -> float:
        score = 0.0
        if recipe.get("meal_type") in top_meals:
            score += meal_counter.get(recipe["meal_type"], 0) / total * 3.0
        if recipe.get("cuisine") in top_cuisines:
            score += cuisine_counter.get(recipe["cuisine"], 0) / total * 2.0
        if recipe.get("is_spicy") and spicy_count / total > 0.4:
            score += 1.0
        if recipe.get("is_sweet") and sweet_count / total > 0.4:
            score += 1.0
        if recipe.get("is_quick") and quick_count / total > 0.4:
            score += 1.0
        return score

    ranked = sorted(candidates, key=_preference_score, reverse=True)[:_RECOMMENDATION_COUNT]
    return ApiResponse(data=ForYouResponse(recipes=[RecipeSummary(**r) for r in ranked]))
