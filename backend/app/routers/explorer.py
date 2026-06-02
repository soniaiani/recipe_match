from __future__ import annotations
import math
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from statistics import pstdev
from typing import Any, TypeVar

from fastapi import APIRouter, Depends, HTTPException, Query, status
import joblib
import pandas as pd
from pydantic import BaseModel

from app.database import get_supabase_admin
from app.middleware.auth import get_current_user_optional
from app.models.schemas import ApiResponse
from app.recommender.filters import (
    normalize_excluded_ingredients,
    recipe_uses_excluded_ingredient,
)

router = APIRouter(prefix="/explorer", tags=["explorer"])
MAX_EXPAND_CANDIDATES = 250
T = TypeVar("T")

EXPAND_PPMI_SHIFT_K = 1.0
EXPAND_TFIDF_ALPHA = 0.8
_EXPAND_LOG_K = math.log2(EXPAND_PPMI_SHIFT_K)

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
_EXPLORER_LTR_MODEL: Any | None = None
_EXPLORER_LTR_FEATURE_COLUMNS: list[str] = []
_EXPLORER_LTR_MODEL_PATH = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "explorer_ltr_final_no_graph.joblib"
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
            .select(
                "id,ingredients_clean,"
                "is_vegetarian,is_vegan,is_gluten_free,is_dairy_free"
            )
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


def _dietary_from_payload(payload: dict | None) -> dict[str, Any]:
    meta = (payload or {}).get("user_metadata") or {}
    return {
        "is_vegetarian": bool(meta.get("is_vegetarian", False)),
        "is_vegan": bool(meta.get("is_vegan", False)),
        "is_gluten_free": bool(meta.get("is_gluten_free", False)),
        "is_dairy_free": bool(meta.get("is_dairy_free", False)),
        "excluded_ingredients": normalize_excluded_ingredients(
            meta.get("excluded_ingredients", [])
        ),
    }


def _recipe_matches_dietary(recipe: dict[str, Any], dietary: dict[str, Any]) -> bool:
    if dietary.get("is_vegetarian") and not recipe.get("is_vegetarian"):
        return False
    if dietary.get("is_vegan") and not recipe.get("is_vegan"):
        return False
    if dietary.get("is_gluten_free") and not recipe.get("is_gluten_free"):
        return False
    if dietary.get("is_dairy_free") and not recipe.get("is_dairy_free"):
        return False
    return not recipe_uses_excluded_ingredient(
        recipe,
        dietary.get("excluded_ingredients", []),
    )


def _compatible_rows(
    parsed_rows: list[tuple[dict[str, Any], set[str]]],
    dietary: dict[str, Any],
) -> list[tuple[dict[str, Any], set[str]]]:
    return [
        (recipe, ingredients)
        for recipe, ingredients in parsed_rows
        if _recipe_matches_dietary(recipe, dietary)
    ]


def _validate_selected_allowed(selected: list[str], dietary: dict[str, Any]) -> None:
    excluded = set(dietary.get("excluded_ingredients", []))
    blocked = sorted(set(selected) & excluded)
    if blocked:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Excluded ingredient selected: {', '.join(blocked)}",
        )


def warm_explorer_cache() -> None:
    _recipe_ingredient_rows()


def warm_ingredient_idf() -> None:
    global _INGREDIENT_IDF
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
    for row in rows:
        idf = math.log(row["total_recipes"] / max(row["recipe_count"], 1))
        _INGREDIENT_IDF[row["ingredient"]] = round(idf, 6)
    print(f"[explorer] Loaded IDF for {len(_INGREDIENT_IDF)} ingredients.")


def warm_explorer_ltr_model() -> None:
    global _EXPLORER_LTR_MODEL, _EXPLORER_LTR_FEATURE_COLUMNS
    if _EXPLORER_LTR_MODEL is not None:
        return

    if not _EXPLORER_LTR_MODEL_PATH.exists():
        print(f"[explorer] LTR model not found at {_EXPLORER_LTR_MODEL_PATH}; using statistical fallback.")
        return

    try:
        artifact = joblib.load(_EXPLORER_LTR_MODEL_PATH)
        _EXPLORER_LTR_MODEL = artifact["model"]
        _EXPLORER_LTR_FEATURE_COLUMNS = list(artifact["feature_columns"])
        print(
            "[explorer] Loaded LTR model "
            f"{artifact.get('model_name', 'unknown')} "
            f"with features={_EXPLORER_LTR_FEATURE_COLUMNS}."
        )
    except Exception as exc:
        _EXPLORER_LTR_MODEL = None
        _EXPLORER_LTR_FEATURE_COLUMNS = []
        print(f"[explorer] failed to load LTR model: {exc}")


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
    excluded_ingredients: list[str] | None = None,
) -> dict[str, int]:
    excluded = set(excluded_ingredients or [])
    return {
        ingredient: count
        for ingredient, count in candidate_counts.items()
        if count >= min_count
        and ingredient not in _PANTRY_HARD
        and ingredient not in excluded
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


def _min_max_normalize(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    lower = min(values.values())
    upper = max(values.values())
    if upper <= lower:
        return {key: 0.0 for key in values}
    return {
        key: (value - lower) / (upper - lower)
        for key, value in values.items()
    }


def _safe_zscore(value: float, average: float, std: float) -> float:
    return (value - average) / std if std > 0 else 0.0


def _score_start_candidates(
    candidate_counts: dict[str, int],
    n_matching: int,
) -> list[ExplorerSuggestion]:
    scored = [
        ExplorerSuggestion(
            ingredient=candidate,
            score=round(count / n_matching, 6),
        )
        for candidate, count in candidate_counts.items()
    ]
    scored.sort(key=lambda item: item.score or 0.0, reverse=True)
    return scored


def _score_expand_candidates_ltr(
    candidate_counts: dict[str, int],
    selected: list[str],
    n_matching: int,
) -> list[ExplorerSuggestion] | None:
    warm_explorer_ltr_model()
    if _EXPLORER_LTR_MODEL is None or not _EXPLORER_LTR_FEATURE_COLUMNS:
        return None

    raw_features: dict[str, dict[str, float]] = {}
    for candidate, count in candidate_counts.items():
        frequency_score = count / n_matching
        idf = _INGREDIENT_IDF.get(candidate, 1.0)
        raw_features[candidate] = {
            "frequency_score": frequency_score,
            "idf": idf,
            "seed_count": float(len(selected)),
            "pantry_flag": float(int(candidate in _PANTRY_SOFT)),
            "tfidf_score": frequency_score * idf,
        }

    frequency_values = [
        features["frequency_score"]
        for features in raw_features.values()
    ]
    tfidf_values = [
        features["tfidf_score"]
        for features in raw_features.values()
    ]
    frequency_mean = mean(frequency_values) if frequency_values else 0.0
    tfidf_mean = mean(tfidf_values) if tfidf_values else 0.0
    frequency_std = pstdev(frequency_values) if len(frequency_values) > 1 else 0.0
    tfidf_std = pstdev(tfidf_values) if len(tfidf_values) > 1 else 0.0

    rows = []
    candidates = []
    for candidate, features in raw_features.items():
        frequency_score = features["frequency_score"]
        tfidf_score = features["tfidf_score"]
        rows.append({
            **features,
            "freq_zscore": _safe_zscore(
                frequency_score,
                frequency_mean,
                frequency_std,
            ),
            "tfidf_zscore": _safe_zscore(
                tfidf_score,
                tfidf_mean,
                tfidf_std,
            ),
        })
        candidates.append(candidate)

    try:
        frame = pd.DataFrame(rows)
        scores = _EXPLORER_LTR_MODEL.predict_proba(
            frame[_EXPLORER_LTR_FEATURE_COLUMNS]
        )[:, 1]
    except Exception as exc:
        print(f"[explorer] LTR inference failed; using statistical fallback: {exc}")
        return None

    scored = [
        ExplorerSuggestion(
            ingredient=candidate,
            score=round(float(score), 6),
        )
        for candidate, score in zip(candidates, scores)
    ]
    scored.sort(
        key=lambda item: (
            -(item.score or 0.0),
            -raw_features[item.ingredient]["tfidf_score"],
            -raw_features[item.ingredient]["frequency_score"],
            item.ingredient,
        )
    )
    return scored


def _score_expand_candidates(
    candidate_counts: dict[str, int],
    selected: list[str],
    n_matching: int,
) -> list[ExplorerSuggestion]:
    if len(selected) >= 2:
        ltr_scored = _score_expand_candidates_ltr(
            candidate_counts,
            selected,
            n_matching,
        )
        if ltr_scored is not None:
            return ltr_scored

    ppmi_scores = _fetch_ppmi_scores(set(candidate_counts), selected)
    tfidf: dict[str, float] = {}
    shifted_ppmi: dict[str, float] = {}

    for candidate, count in candidate_counts.items():
        frequency_score = count / n_matching
        idf = _INGREDIENT_IDF.get(candidate, 1.0)
        tfidf[candidate] = frequency_score * idf

        scores = ppmi_scores.get(candidate, [])
        avg_ppmi = mean(scores) if scores else 0.0
        shifted_ppmi[candidate] = max(avg_ppmi - _EXPAND_LOG_K, 0.0)

    tfidf_norm = _min_max_normalize(tfidf)
    ppmi_norm = _min_max_normalize(shifted_ppmi)
    scored = [
        ExplorerSuggestion(
            ingredient=candidate,
            score=round(
                EXPAND_TFIDF_ALPHA * tfidf_norm.get(candidate, 0.0)
                + (1.0 - EXPAND_TFIDF_ALPHA) * ppmi_norm.get(candidate, 0.0),
                6,
            ),
        )
        for candidate in candidate_counts
    ]
    scored.sort(key=lambda item: item.score or 0.0, reverse=True)
    return scored


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
    ingredient = _normalize_ingredient(body.ingredient)
    if not ingredient:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Ingredient is required"
        )
    dietary = _dietary_from_payload(_payload)
    _validate_selected_allowed([ingredient], dietary)

    compatible_rows = _compatible_rows(_recipe_ingredient_rows(), dietary)
    selected = [ingredient]
    selected_set = {ingredient}
    matching, _ = _matching_recipes(selected, compatible_rows)

    if not matching:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Ingredient not found in recipes compatible with your dietary requirements."
        )

    candidate_counts = _candidate_counts_from_matching(matching, selected_set)
    candidate_counts = _filter_candidate_counts_by_context(
        candidate_counts,
        min_count=20,
        exclude_soft_pantry=True,
        excluded_ingredients=dietary["excluded_ingredients"],
    )
    n_matching = len(matching)
    scored = _score_start_candidates(candidate_counts, n_matching)

    return ApiResponse(data=ExplorerStartResponse(
        center=ingredient,
        suggestions=scored[:5],
        recipe_count=n_matching,
    ))


@router.post("/expand", response_model=ApiResponse[ExplorerExpandResponse])
def expand_explorer(
    body: ExplorerExpandRequest,
    _payload: dict | None = Depends(get_current_user_optional),
) -> ApiResponse[ExplorerExpandResponse]:
    selected = _normalize_many(body.selected_ingredients)
    if not selected:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="selected_ingredients is required")
    dietary = _dietary_from_payload(_payload)
    _validate_selected_allowed(selected, dietary)

    matching, relaxed = _matching_recipes(
        selected,
        _compatible_rows(_recipe_ingredient_rows(), dietary),
    )
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
        excluded_ingredients=dietary["excluded_ingredients"],
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
    selected = _normalize_many(body.selected_ingredients)
    if not selected:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="selected_ingredients is required")
    dietary = _dietary_from_payload(_payload)
    _validate_selected_allowed(selected, dietary)

    matching, relaxed = _matching_recipes(
        selected,
        _compatible_rows(_recipe_detail_rows(), dietary),
    )
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

    dietary = _dietary_from_payload(_payload)
    excluded = set(dietary["excluded_ingredients"])
    counts: dict[str, int] = {}
    for _, ingredients in _compatible_rows(_recipe_ingredient_rows(), dietary):
        for ingredient in ingredients:
            if ingredient in excluded or ingredient in _PANTRY_HARD:
                continue
            if ingredient.startswith(query):
                counts[ingredient] = counts.get(ingredient, 0) + 1

    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:10]
    return ApiResponse(data=ExplorerSearchResponse(
        ingredients=[ingredient for ingredient, _ in ranked],
    ))
