from __future__ import annotations
import math
import warnings
from pathlib import Path
from datetime import datetime, timezone
from statistics import mean
from typing import Any, TypeVar

import joblib
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel

from app.database import get_supabase_admin
from app.middleware.auth import get_current_user_optional
from app.models.schemas import ApiResponse
from app.services.explorer_scoring import (
    candidate_counts_from_ingredient_sets,
    filter_candidate_counts_by_context,
    normalize_ingredient,
    normalize_many,
    PANTRY_SOFT,
    parse_ingredients,
    score_expand_candidates,
    score_start_candidates,
)

router = APIRouter(prefix="/explorer", tags=["explorer"])
T = TypeVar("T")

_RECIPE_INGREDIENT_CACHE: list[tuple[dict[str, Any], set[str]]] | None = None
_RECIPE_DETAIL_CACHE: list[tuple[dict[str, Any], set[str]]] | None = None
_INGREDIENT_IDF: dict[str, float] = {}
_INGREDIENT_RECIPE_COUNT: dict[str, int] = {}
_INGREDIENT_RECIPE_IDS: dict[str, set[int]] = {}
_TOTAL_RECIPES = 0
_EXPLORER_LTR_MODEL: Any | None = None
_EXPLORER_LTR_FEATURES: list[str] | None = None
_EXPLORER_LTR_LOAD_FAILED = False
_EXPLORER_LTR_MODEL_PATH = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "explorer_ltr_v0_random_base.joblib"
)

RECIPE_FIELDS = (
    "id,name,description,image_url,prep_time,cook_time,total_time,servings,"
    "meal_type,cuisine,protein_type,ingredients_clean,ingredients_clean_str,"
    "is_vegetarian,is_vegan,is_gluten_free,is_dairy_free"
)


class ExplorerStartRequest(BaseModel):
    ingredient: str


class ExplorerExpandRequest(BaseModel):
    selected_ingredients: list[str]
    session_id: str | None = None
    finalize: bool = False


class ExplorerSuggestion(BaseModel):
    ingredient: str
    score: float | None = None
    ppmi_score: float | None = None
    scoring_method: str | None = None


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
    global _RECIPE_INGREDIENT_CACHE, _INGREDIENT_RECIPE_IDS, _TOTAL_RECIPES
    if _RECIPE_INGREDIENT_CACHE is None:
        _RECIPE_INGREDIENT_CACHE = [
            (recipe, parse_ingredients(recipe.get("ingredients_clean")))
            for recipe in _fetch_all_recipe_ingredients()
        ]
        _TOTAL_RECIPES = len(_RECIPE_INGREDIENT_CACHE)
        _INGREDIENT_RECIPE_IDS = {}
        for idx, (_, ingredients) in enumerate(_RECIPE_INGREDIENT_CACHE):
            for ingredient in ingredients:
                _INGREDIENT_RECIPE_IDS.setdefault(ingredient, set()).add(idx)
    return _RECIPE_INGREDIENT_CACHE


def _recipe_detail_rows() -> list[tuple[dict[str, Any], set[str]]]:
    global _RECIPE_DETAIL_CACHE
    if _RECIPE_DETAIL_CACHE is None:
        _RECIPE_DETAIL_CACHE = [
            (recipe, parse_ingredients(recipe.get("ingredients_clean")))
            for recipe in _fetch_all_recipe_details()
        ]
    return _RECIPE_DETAIL_CACHE


def warm_explorer_cache() -> None:
    _recipe_ingredient_rows()

def warm_ingredient_idf() -> None:
    global _INGREDIENT_IDF, _INGREDIENT_RECIPE_COUNT, _TOTAL_RECIPES
    admin = get_supabase_admin()
    rows: list[dict[str, Any]] = []
    offset = 0
    while True:
        page = (
            admin.table("ingredient_stats")
            .select("ingredient,recipe_count,total_recipes")
            .range(offset, offset + 999)
            .execute()
            .data
            or []
        )
        rows.extend(page)
        if len(page) < 1000:
            break
        offset += 1000

    _INGREDIENT_IDF = {}
    _INGREDIENT_RECIPE_COUNT = {}
    _TOTAL_RECIPES = 0
    for row in rows:
        idf = math.log(row["total_recipes"] / max(row["recipe_count"], 1))
        ingredient = row["ingredient"]
        recipe_count = int(row["recipe_count"])
        total_recipes = int(row["total_recipes"])
        _INGREDIENT_IDF[ingredient] = round(idf, 6)
        _INGREDIENT_RECIPE_COUNT[ingredient] = recipe_count
        _TOTAL_RECIPES = max(_TOTAL_RECIPES, total_recipes)
    print(f"[explorer] Loaded IDF for {len(_INGREDIENT_IDF)} ingredients.")


def warm_explorer_ltr_model() -> None:
    _load_explorer_ltr_model()


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
    scores_by_selected: dict[str, dict[str, float]] = {candidate: {} for candidate in candidates}
    candidate_list = list(candidates)

    for candidate_batch in _chunks(candidate_list, 100):
        rows = (
            admin.table("ingredient_graph")
            .select("ingredient_a,ingredient_b,ppmi_score")
            .in_("ingredient_a", candidate_batch)
            .in_("ingredient_b", selected)
            .execute()
            .data or []
        )
        for row in rows:
            scores_by_selected.setdefault(row["ingredient_a"], {})[row["ingredient_b"]] = float(row["ppmi_score"])
    return {
        candidate: [
            scores_by_selected.get(candidate, {}).get(selected_ingredient, 0.0)
            for selected_ingredient in selected
        ]
        for candidate in candidates
    }


def _load_explorer_ltr_model() -> tuple[Any, list[str]] | None:
    global _EXPLORER_LTR_MODEL, _EXPLORER_LTR_FEATURES, _EXPLORER_LTR_LOAD_FAILED
    if _EXPLORER_LTR_MODEL is not None and _EXPLORER_LTR_FEATURES is not None:
        return _EXPLORER_LTR_MODEL, _EXPLORER_LTR_FEATURES
    if _EXPLORER_LTR_LOAD_FAILED:
        return None
    if not _EXPLORER_LTR_MODEL_PATH.exists():
        _EXPLORER_LTR_LOAD_FAILED = True
        print(f"[explorer] LTR model not found: {_EXPLORER_LTR_MODEL_PATH}")
        return None
    try:
        artifact = joblib.load(_EXPLORER_LTR_MODEL_PATH)
        _EXPLORER_LTR_MODEL = artifact["model"]
        _EXPLORER_LTR_FEATURES = list(artifact["feature_columns"])
        print(f"[explorer] Loaded LTR model: {_EXPLORER_LTR_MODEL_PATH.name}")
        return _EXPLORER_LTR_MODEL, _EXPLORER_LTR_FEATURES
    except Exception as exc:
        _EXPLORER_LTR_LOAD_FAILED = True
        print(f"[explorer] failed to load LTR model: {exc}")
        return None


def _chunks(items: list[T], size: int):
    for start in range(0, len(items), size):
        yield items[start:start + size]


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
            candidate = normalize_ingredient(row.get("ingredient_b"))
            if not candidate or candidate in selected_set:
                continue
            rows_by_ingredient.setdefault(candidate, []).append(float(row["ppmi_score"]))

    suggestions = [
        ExplorerSuggestion(ingredient=ingredient, score=round(mean(scores), 6))
        for ingredient, scores in rows_by_ingredient.items()
    ]
    suggestions.sort(key=lambda item: item.score or 0.0, reverse=True)
    return suggestions[:5]


def _score_start_candidates(
    candidate_counts: dict[str, int],
    n_matching: int,
) -> list[ExplorerSuggestion]:
    scored = [
        ExplorerSuggestion(
            ingredient=candidate,
            score=round(score, 6),
            scoring_method="statistical",
        )
        for candidate, score in score_start_candidates(candidate_counts, n_matching)
    ]
    return scored


def _ppmi_and_jaccard_features(
    candidate: str,
    selected: list[str],
) -> tuple[float, float, float, float]:
    candidate_recipe_ids = _INGREDIENT_RECIPE_IDS.get(candidate, set())
    ppmi_values: list[float] = []
    jaccard_scores: list[float] = []

    for selected_ingredient in selected:
        selected_recipe_ids = _INGREDIENT_RECIPE_IDS.get(selected_ingredient, set())
        co_occurrence = len(candidate_recipe_ids & selected_recipe_ids)
        if co_occurrence > 0 and _TOTAL_RECIPES:
            p_candidate = len(candidate_recipe_ids) / _TOTAL_RECIPES
            p_selected = len(selected_recipe_ids) / _TOTAL_RECIPES
            p_pair = co_occurrence / _TOTAL_RECIPES
            pmi = math.log2(p_pair / (p_candidate * p_selected))
            ppmi_values.append(max(pmi, 0.0))
        union_size = len(candidate_recipe_ids | selected_recipe_ids)
        if union_size > 0:
            jaccard_scores.append(co_occurrence / union_size)

    return (
        mean(ppmi_values) if ppmi_values else 0.0,
        max(ppmi_values) if ppmi_values else 0.0,
        mean(jaccard_scores) if jaccard_scores else 0.0,
        max(jaccard_scores) if jaccard_scores else 0.0,
    )


def _ltr_feature_rows(
    candidate_counts: dict[str, int],
    selected: list[str],
    n_matching: int,
    statistical_scores: dict[str, float],
    statistical_ranks: dict[str, int],
) -> list[list[float]]:
    recipes_with_any_seed = set().union(
        *(_INGREDIENT_RECIPE_IDS.get(ingredient, set()) for ingredient in set(selected))
    )
    selected_recipe_ids_count = len(recipes_with_any_seed)
    candidate_count = len(candidate_counts)
    rows: list[list[float]] = []

    for candidate, count in candidate_counts.items():
        frequency_score = count / n_matching
        candidate_idf = _INGREDIENT_IDF.get(candidate, 1.0)
        tfidf_score = frequency_score * candidate_idf
        degree = float(_INGREDIENT_RECIPE_COUNT.get(candidate, 0))
        ppmi_avg, max_ppmi, avg_jaccard, max_jaccard = _ppmi_and_jaccard_features(
            candidate,
            selected,
        )
        feature_values = {
            "frequency_score": frequency_score,
            "tfidf_score": tfidf_score,
            "ppmi_avg": ppmi_avg,
            "max_ppmi_to_seed": max_ppmi,
            "avg_jaccard_to_seed": avg_jaccard,
            "max_jaccard_to_seed": max_jaccard,
            "idf": candidate_idf,
            "degree": degree,
            "seed_count": len(selected),
            "candidate_count": candidate_count,
            "matching_recipe_count": n_matching,
            "recipe_count_ratio": (
                n_matching / selected_recipe_ids_count
                if selected_recipe_ids_count else 0.0
            ),
            "pantry_flag": int(candidate in PANTRY_SOFT),
            "statistical_score": statistical_scores.get(candidate, 0.0),
            "rank_in_statistical": statistical_ranks.get(candidate, candidate_count + 1),
        }
        rows.append([float(feature_values[feature]) for feature in _EXPLORER_LTR_FEATURES or []])
    return rows


def _score_expand_candidates(
    candidate_counts: dict[str, int],
    selected: list[str],
    n_matching: int,
) -> list[ExplorerSuggestion]:
    if not candidate_counts:
        return []
    aligned_ppmi_scores = _fetch_ppmi_scores(set(candidate_counts), selected)
    statistical_ppmi_scores = {
        candidate: [score for score in scores if score > 0]
        for candidate, scores in aligned_ppmi_scores.items()
    }
    statistical_ranked = score_expand_candidates(
        candidate_counts,
        n_matching,
        _INGREDIENT_IDF,
        statistical_ppmi_scores,
    )

    if len(selected) >= 2 and (model_info := _load_explorer_ltr_model()):
        model, _features = model_info
        try:
            statistical_scores = {
                candidate: score
                for candidate, score, _ in statistical_ranked
            }
            statistical_ranks = {
                candidate: rank
                for rank, (candidate, _, _) in enumerate(statistical_ranked, start=1)
            }
            feature_rows = _ltr_feature_rows(
                candidate_counts,
                selected,
                n_matching,
                statistical_scores,
                statistical_ranks,
            )
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="X does not have valid feature names",
                    category=UserWarning,
                )
                probabilities = model.predict_proba(feature_rows)[:, 1]
            scored = [
                ExplorerSuggestion(
                    ingredient=candidate,
                    score=round(float(probability), 6),
                    ppmi_score=round(statistical_ppmi, 6),
                    scoring_method="ml_ltr",
                )
                for (candidate, _, statistical_ppmi), probability in zip(statistical_ranked, probabilities)
            ]
            scored.sort(key=lambda item: item.score or 0.0, reverse=True)
            return scored
        except Exception as exc:
            print(f"[explorer] LTR scoring failed, using statistical fallback: {exc}")

    return [
        ExplorerSuggestion(
            ingredient=candidate,
            score=round(score, 6),
            ppmi_score=round(ppmi_score, 6),
            scoring_method="statistical",
        )
        for candidate, score, ppmi_score in statistical_ranked
    ]


def _upsert_explorer_session(
    session_id: str,
    user_id: str,
    selected: list[str],
    finalize: bool = False,
) -> None:
    """Persist one row per Explorer session."""
    try:
        now = datetime.now(timezone.utc).isoformat()
        data = {
            "id": session_id,
            "user_id": user_id,
            "chain": selected,
            "updated_at": now,
        }
        if finalize:
            data["finalized_at"] = now

        get_supabase_admin().table("explorer_sessions").upsert(
            data,
            on_conflict="id",
        ).execute()
    except Exception as exc:
        print(f"[explorer] failed to upsert session: {exc}")


@router.post("/start", response_model=ApiResponse[ExplorerStartResponse])
def start_explorer(
    body: ExplorerStartRequest,
    _payload: dict | None = Depends(get_current_user_optional),
) -> ApiResponse[ExplorerStartResponse]:
    ingredient = normalize_ingredient(body.ingredient)
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

    candidate_counts = candidate_counts_from_ingredient_sets(
        (ingredients for _, ingredients in matching),
        selected_set,
    )
    candidate_counts = filter_candidate_counts_by_context(
        candidate_counts,
        min_count=20,
        exclude_soft_pantry=True,
    )
    n_matching = len(matching)
    scored = _score_start_candidates(candidate_counts, n_matching)

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
    selected = normalize_many(body.selected_ingredients)
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

    candidate_counts = candidate_counts_from_ingredient_sets(
        (ingredients for _, ingredients in matching),
        selected_set,
    )
    candidate_counts = filter_candidate_counts_by_context(
        candidate_counts,
        min_count=5,
        exclude_soft_pantry=False,
    )
    n_matching = len(matching)
    scored = _score_expand_candidates(candidate_counts, selected, n_matching)

    if _payload and body.session_id and len(selected) >= 2:
        user_id = _payload.get("sub")
        if user_id:
            _upsert_explorer_session(
                session_id=body.session_id,
                user_id=user_id,
                selected=selected,
                finalize=False,
            )

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
    selected = normalize_many(body.selected_ingredients)
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

    if _payload and body.session_id:
        user_id = _payload.get("sub")
        if user_id:
            _upsert_explorer_session(
                session_id=body.session_id,
                user_id=user_id,
                selected=selected,
                finalize=True,
            )

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
