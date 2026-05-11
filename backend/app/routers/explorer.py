from __future__ import annotations
import math
import json
from statistics import mean
from typing import Any, TypeVar

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel

from app.database import get_supabase_admin
from app.middleware.auth import get_current_user_optional
from app.models.schemas import ApiResponse

router = APIRouter(prefix="/explorer", tags=["explorer"])
MAX_EXPAND_CANDIDATES = 250
T = TypeVar("T")

_LOG_K = math.log(15)  # Shifted PPMI, k=5 — Levy & Goldberg (2014)

_PANTRY_HARD = {
    "salt", "pepper", "black pepper", "water", "oil",
    "olive oil", "vegetable oil", "cooking spray",
}

_PANTRY_SOFT = {
    "sugar", "all-purpose flour", "butter", "milk", "egg",
}

_RECIPE_INGREDIENT_CACHE: list[tuple[dict[str, Any], set[str]]] | None = None
_RECIPE_DETAIL_CACHE: list[tuple[dict[str, Any], set[str]]] | None = None
_INGREDIENT_IDF: dict[str, float] = {}

RECIPE_FIELDS = (
    "id,name,description,image_url,prep_time,cook_time,total_time,servings,"
    "meal_type,cuisine,protein_type,ingredients_clean,ingredients_clean_str,"
    "is_vegetarian,is_vegan,is_gluten_free,is_dairy_free"
)


class ExplorerStartRequest(BaseModel):
    ingredient: str


class ExplorerExpandRequest(BaseModel):
    selected_ingredients: list[str]


class ExplorerSuggestion(BaseModel):
    ingredient: str
    score: float | None = None
    ppmi_score: float | None = None


class ExplorerStartResponse(BaseModel):
    center: str
    suggestions: list[ExplorerSuggestion]
    recipe_count: int


class ExplorerExpandResponse(BaseModel):
    suggestions: list[ExplorerSuggestion]
    recipe_count: int
    relaxed: bool


class ExplorerRecipe(BaseModel):
    id: int
    name: str
    description: str | None = None
    image_url: str | None = None
    prep_time: str | None = None
    cook_time: str | None = None
    total_time: str | None = None
    servings: int | None = None
    meal_type: str | None = None
    cuisine: str | None = None
    protein_type: str | None = None
    ingredients_clean_str: str | None = None
    is_vegetarian: bool | None = None
    is_vegan: bool | None = None
    is_gluten_free: bool | None = None
    is_dairy_free: bool | None = None
    jaccard_score: float


class ExplorerRecommendResponse(BaseModel):
    recipes: list[ExplorerRecipe]
    recipe_count: int
    relaxed: bool


class ExplorerSearchResponse(BaseModel):
    ingredients: list[str]


def _normalize_ingredient(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).lower().strip()
    return normalized or None


def _normalize_many(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        normalized = _normalize_ingredient(value)
        if normalized and normalized not in seen:
            seen.add(normalized)
            output.append(normalized)
    return output


def _parse_ingredients(raw: Any) -> set[str]:
    if not raw:
        return set()
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError:
        return set()
    if not isinstance(parsed, list):
        return set()
    return {
        normalized
        for item in parsed
        if (normalized := _normalize_ingredient(item))
    }


def _fetch_all_recipe_ingredients() -> list[dict[str, Any]]:
    admin = get_supabase_admin()
    recipes: list[dict[str, Any]] = []
    offset = 0
    while True:
        page = (
            admin.table("recipes")
            .select("id,ingredients_clean")
            .range(offset, offset + 999)
            .execute()
            .data or []
        )
        recipes.extend(page)
        if len(page) < 1000:
            break
        offset += 1000
    return recipes


def _fetch_all_recipe_details() -> list[dict[str, Any]]:
    admin = get_supabase_admin()
    recipes: list[dict[str, Any]] = []
    offset = 0
    while True:
        page = (
            admin.table("recipes")
            .select(RECIPE_FIELDS)
            .range(offset, offset + 999)
            .execute()
            .data or []
        )
        recipes.extend(page)
        if len(page) < 1000:
            break
        offset += 1000
    return recipes


def _recipe_ingredient_rows() -> list[tuple[dict[str, Any], set[str]]]:
    global _RECIPE_INGREDIENT_CACHE
    if _RECIPE_INGREDIENT_CACHE is None:
        _RECIPE_INGREDIENT_CACHE = [
            (recipe, _parse_ingredients(recipe.get("ingredients_clean")))
            for recipe in _fetch_all_recipe_ingredients()
        ]
    return _RECIPE_INGREDIENT_CACHE


def _recipe_detail_rows() -> list[tuple[dict[str, Any], set[str]]]:
    global _RECIPE_DETAIL_CACHE
    if _RECIPE_DETAIL_CACHE is None:
        _RECIPE_DETAIL_CACHE = [
            (recipe, _parse_ingredients(recipe.get("ingredients_clean")))
            for recipe in _fetch_all_recipe_details()
        ]
    return _RECIPE_DETAIL_CACHE


def warm_explorer_cache() -> None:
    _recipe_ingredient_rows()

def warm_ingredient_idf() -> None:
    global _INGREDIENT_IDF
    admin = get_supabase_admin()
    rows = (
        admin.table("ingredient_stats")
        .select("ingredient,recipe_count,total_recipes")
        .execute()
        .data or []
    )
    for row in rows:
        idf = math.log(row["total_recipes"] / max(row["recipe_count"], 1))
        _INGREDIENT_IDF[row["ingredient"]] = round(idf, 6)
    print(f"[explorer] Loaded IDF for {len(_INGREDIENT_IDF)} ingredients.")


def _matching_recipes(
    selected: list[str],
    parsed_rows: list[tuple[dict[str, Any], set[str]]],
) -> tuple[list[tuple[dict[str, Any], set[str]]], bool]:
    selected_set = set(selected)
    matching = [
        (recipe, ingredients)
        for recipe, ingredients in parsed_rows
        if selected_set.issubset(ingredients)
    ]
    return matching, False


def _fetch_ppmi_scores(candidates: set[str], selected: list[str]) -> dict[str, list[float]]:
    admin = get_supabase_admin()
    scores: dict[str, list[float]] = {candidate: [] for candidate in candidates}
    candidate_list = list(candidates)

    for candidate_batch in _chunks(candidate_list, 100):
        rows = (
            admin.table("ingredient_graph")
            .select("ingredient_a,ppmi_score")
            .in_("ingredient_a", candidate_batch)
            .in_("ingredient_b", selected)
            .execute()
            .data or []
        )
        for row in rows:
            scores.setdefault(row["ingredient_a"], []).append(float(row["ppmi_score"]))
    return scores


def _chunks(items: list[T], size: int):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def _candidate_counts_from_matching(
    matching: list[tuple[dict[str, Any], set[str]]],
    selected_set: set[str],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for _, ingredients in matching:
        for ingredient in ingredients - selected_set:
            counts[ingredient] = counts.get(ingredient, 0) + 1

    ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    return dict(ranked[:MAX_EXPAND_CANDIDATES])


def _filter_candidate_counts_by_context(
    candidate_counts: dict[str, int],
    min_count: int,
    exclude_soft_pantry: bool,
) -> dict[str, int]:
    return {
        ingredient: count
        for ingredient, count in candidate_counts.items()
        if count >= min_count
        and ingredient not in _PANTRY_HARD
        and (not exclude_soft_pantry or ingredient not in _PANTRY_SOFT)
    }


def _graph_suggestions(selected: list[str], selected_set: set[str]) -> list[ExplorerSuggestion]:
    admin = get_supabase_admin()
    rows_by_ingredient: dict[str, list[float]] = {}
    for ingredient in selected:
        rows = (
            admin.table("ingredient_graph")
            .select("ingredient_b,ppmi_score")
            .eq("ingredient_a", ingredient)
            .order("ppmi_score", desc=True)
            .limit(25)
            .execute()
            .data or []
        )
        for row in rows:
            candidate = _normalize_ingredient(row.get("ingredient_b"))
            if not candidate or candidate in selected_set:
                continue
            rows_by_ingredient.setdefault(candidate, []).append(float(row["ppmi_score"]))

    suggestions = [
        ExplorerSuggestion(ingredient=ingredient, score=round(mean(scores), 6))
        for ingredient, scores in rows_by_ingredient.items()
    ]
    suggestions.sort(key=lambda item: item.score or 0.0, reverse=True)
    return suggestions[:5]


@router.post("/start", response_model=ApiResponse[ExplorerStartResponse])
def start_explorer(
    body: ExplorerStartRequest,
    _payload: dict | None = Depends(get_current_user_optional),
) -> ApiResponse[ExplorerStartResponse]:
    ingredient = _normalize_ingredient(body.ingredient)
    if not ingredient:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Ingredient is required"
        )

    admin = get_supabase_admin()
    stats = (
        admin.table("ingredient_stats")
        .select("recipe_count")
        .eq("ingredient", ingredient)
        .limit(1)
        .execute()
        .data or []
    )
    if not stats:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Ingredient not found"
        )

    selected = [ingredient]
    selected_set = {ingredient}

    matching, _ = _matching_recipes(selected, _recipe_ingredient_rows())

    if not matching:
        return ApiResponse(data=ExplorerStartResponse(
            center=ingredient,
            suggestions=[],
            recipe_count=int(stats[0]["recipe_count"]),
        ))

    candidate_counts = _candidate_counts_from_matching(matching, selected_set)
    candidate_counts = _filter_candidate_counts_by_context(
        candidate_counts,
        min_count=20,
        exclude_soft_pantry=True,
    )
    ppmi_scores = _fetch_ppmi_scores(set(candidate_counts), selected)
    n_matching = len(matching)

    scored: list[ExplorerSuggestion] = []
    for candidate, count in candidate_counts.items():
        frequency_score = count / n_matching
        idf = _INGREDIENT_IDF.get(candidate, 1.0)  # fallback 1.0 dacă lipsește
        tfidf_score = frequency_score * idf
        scores = ppmi_scores.get(candidate, [])
        avg_ppmi = mean(scores) if scores else 0.0
        shifted_ppmi = max(avg_ppmi - _LOG_K, 0.0)
        final_score = (0.7 * tfidf_score) + (0.3 * shifted_ppmi)
        scored.append(ExplorerSuggestion(
            ingredient=candidate,
            score=round(final_score, 6),
        ))

    scored.sort(key=lambda s: s.score or 0.0, reverse=True)

    return ApiResponse(data=ExplorerStartResponse(
        center=ingredient,
        suggestions=scored[:5],
        recipe_count=int(stats[0]["recipe_count"]),
    ))


@router.post("/expand", response_model=ApiResponse[ExplorerExpandResponse])
def expand_explorer(
    body: ExplorerExpandRequest,
    _payload: dict | None = Depends(get_current_user_optional),
) -> ApiResponse[ExplorerExpandResponse]:
    selected = _normalize_many(body.selected_ingredients)
    if not selected:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="selected_ingredients is required")

    matching, relaxed = _matching_recipes(selected, _recipe_ingredient_rows())
    selected_set = set(selected)

    if not matching:
        return ApiResponse(data=ExplorerExpandResponse(
            suggestions=[],
            recipe_count=0,
            relaxed=relaxed,
        ))

    if len(matching) <= 1:
        return ApiResponse(data=ExplorerExpandResponse(
            suggestions=[],
            recipe_count=len(matching),
            relaxed=relaxed,
        ))

    candidate_counts = _candidate_counts_from_matching(matching, selected_set)
    candidate_counts = _filter_candidate_counts_by_context(
        candidate_counts,
        min_count=5,
        exclude_soft_pantry=False,
    )
    ppmi_scores = _fetch_ppmi_scores(set(candidate_counts), selected)
    n_matching = len(matching)
    scored: list[ExplorerSuggestion] = []
    for candidate, count in candidate_counts.items():
        frequency_score = count / n_matching
        idf = _INGREDIENT_IDF.get(candidate, 1.0)  # fallback 1.0 dacă lipsește
        tfidf_score = frequency_score * idf
        scores = ppmi_scores.get(candidate, [])
        avg_ppmi = mean(scores) if scores else 0.0
        shifted_ppmi = max(avg_ppmi - _LOG_K, 0.0)
        final_score = (0.7 * tfidf_score) + (0.3 * shifted_ppmi)
        scored.append(ExplorerSuggestion(
            ingredient=candidate,
            score=round(final_score, 6),
        ))
    scored.sort(key=lambda item: item.score or 0.0, reverse=True)
    return ApiResponse(data=ExplorerExpandResponse(
        suggestions=scored[:5],
        recipe_count=len(matching),
        relaxed=relaxed,
    ))


@router.post("/recommend", response_model=ApiResponse[ExplorerRecommendResponse])
def recommend_from_ingredients(
    body: ExplorerExpandRequest,
    _payload: dict | None = Depends(get_current_user_optional),
) -> ApiResponse[ExplorerRecommendResponse]:
    selected = _normalize_many(body.selected_ingredients)
    if not selected:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="selected_ingredients is required")

    matching, relaxed = _matching_recipes(selected, _recipe_detail_rows())
    selected_set = set(selected)
    rows: list[ExplorerRecipe] = []
    for recipe, ingredients in matching:
        union = selected_set | ingredients
        jaccard = len(selected_set & ingredients) / len(union) if union else 0.0
        rows.append(ExplorerRecipe(**recipe, jaccard_score=round(jaccard, 6)))

    rows.sort(key=lambda recipe: recipe.jaccard_score, reverse=True)
    return ApiResponse(data=ExplorerRecommendResponse(
        recipes=rows[:10],
        recipe_count=len(matching),
        relaxed=relaxed,
    ))


@router.get("/search", response_model=ApiResponse[ExplorerSearchResponse])
def search_ingredients(
    q: str = Query(default="", min_length=0),
    _payload: dict | None = Depends(get_current_user_optional),
) -> ApiResponse[ExplorerSearchResponse]:
    query = q.lower().strip()
    if not query:
        return ApiResponse(data=ExplorerSearchResponse(ingredients=[]))

    rows = (
        get_supabase_admin()
        .table("ingredient_stats")
        .select("ingredient")
        .ilike("ingredient", f"{query}%")
        .order("recipe_count", desc=True)
        .limit(10)
        .execute()
        .data or []
    )
    return ApiResponse(data=ExplorerSearchResponse(
        ingredients=[row["ingredient"] for row in rows],
    ))
