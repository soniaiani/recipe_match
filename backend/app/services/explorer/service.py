from __future__ import annotations
from datetime import datetime, timezone
from typing import Any

from fastapi import HTTPException, status

from app.database import get_supabase_admin
from app.models.explorer import (
    ExplorerExpandRequest,
    ExplorerExpandResponse,
    ExplorerRecipe,
    ExplorerRecommendResponse,
    ExplorerSearchResponse,
    ExplorerStartRequest,
    ExplorerStartResponse,
    ExplorerSuggestion,
)
from app.models.common import ApiResponse
from app.services.explorer.cache import (
    ingredient_idf,
    recipe_detail_rows,
    recipe_ingredient_rows,
    warm_explorer_cache,
    warm_ingredient_idf,
)
from app.services.explorer.common import (
    PANTRY_HARD,
    PANTRY_SOFT,
    normalize_ingredient,
    normalize_many,
)
from app.services.explorer.dietary import (
    compatible_rows,
    dietary_from_payload,
    validate_selected_allowed,
)
from app.services.explorer.ltr import (
    score_expand_candidates_ltr,
    warm_explorer_ltr_model,
)
from app.services.explorer.scoring import (
    candidate_counts_from_ingredient_sets,
    filter_candidate_counts_by_context,
    score_start_candidates,
)


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


def _candidate_counts_from_matching(
    matching: list[tuple[dict[str, Any], set[str]]],
    selected_set: set[str],
) -> dict[str, int]:
    return candidate_counts_from_ingredient_sets(
        (ingredients for _, ingredients in matching),
        selected_set,
    )


def _score_start_candidates(
    candidate_counts: dict[str, int],
    n_matching: int,
) -> list[ExplorerSuggestion]:
    return [
        ExplorerSuggestion(
            ingredient=ingredient,
            score=round(score, 6),
        )
        for ingredient, score in score_start_candidates(candidate_counts, n_matching)
    ]


def _score_expand_candidates(
    candidate_counts: dict[str, int],
    selected: list[str],
    n_matching: int,
) -> list[ExplorerSuggestion]:
    if len(selected) >= 2:
        ltr_scored = score_expand_candidates_ltr(
            candidate_counts,
            selected,
            n_matching,
            ingredient_idf(),
            PANTRY_SOFT,
        )
        if ltr_scored is not None:
            return ltr_scored

    return _score_start_candidates(candidate_counts, n_matching)


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


def start_explorer_session(
    body: ExplorerStartRequest,
    payload: dict | None = None,
) -> ApiResponse[ExplorerStartResponse]:
    ingredient = normalize_ingredient(body.ingredient)
    if not ingredient:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Ingredient is required"
        )
    dietary = dietary_from_payload(payload)
    validate_selected_allowed([ingredient], dietary)

    compatible_recipe_rows = compatible_rows(recipe_ingredient_rows(), dietary)
    selected = [ingredient]
    selected_set = {ingredient}
    matching, _ = _matching_recipes(selected, compatible_recipe_rows)

    if not matching:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Ingredient not found in recipes compatible with your dietary requirements."
        )

    candidate_counts = _candidate_counts_from_matching(matching, selected_set)
    candidate_counts = filter_candidate_counts_by_context(
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


def expand_explorer_session(
    body: ExplorerExpandRequest,
    payload: dict | None = None,
) -> ApiResponse[ExplorerExpandResponse]:
    selected = normalize_many(body.selected_ingredients)
    if not selected:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="selected_ingredients is required")
    dietary = dietary_from_payload(payload)
    validate_selected_allowed(selected, dietary)

    matching, relaxed = _matching_recipes(
        selected,
        compatible_rows(recipe_ingredient_rows(), dietary),
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
    candidate_counts = filter_candidate_counts_by_context(
        candidate_counts,
        min_count=5,
        exclude_soft_pantry=False,
        excluded_ingredients=dietary["excluded_ingredients"],
    )
    n_matching = len(matching)
    scored = _score_expand_candidates(candidate_counts, selected, n_matching)

    if payload and body.session_id and len(selected) >= 2:
        user_id = payload.get("sub")
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


def recommend_recipes_from_ingredients(
    body: ExplorerExpandRequest,
    payload: dict | None = None,
) -> ApiResponse[ExplorerRecommendResponse]:
    selected = normalize_many(body.selected_ingredients)
    if not selected:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="selected_ingredients is required")
    dietary = dietary_from_payload(payload)
    validate_selected_allowed(selected, dietary)

    matching, relaxed = _matching_recipes(
        selected,
        compatible_rows(recipe_detail_rows(), dietary),
    )
    selected_set = set(selected)
    rows: list[ExplorerRecipe] = []
    for recipe, ingredients in matching:
        union = selected_set | ingredients
        jaccard = len(selected_set & ingredients) / len(union) if union else 0.0
        rows.append(ExplorerRecipe(**recipe, jaccard_score=round(jaccard, 6)))

    rows.sort(key=lambda recipe: recipe.jaccard_score, reverse=True)

    if payload and body.session_id:
        user_id = payload.get("sub")
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


def search_ingredient_names(
    q: str = "",
    payload: dict | None = None,
) -> ApiResponse[ExplorerSearchResponse]:
    query = q.lower().strip()
    if not query:
        return ApiResponse(data=ExplorerSearchResponse(ingredients=[]))

    dietary = dietary_from_payload(payload)
    excluded = set(dietary["excluded_ingredients"])
    counts: dict[str, int] = {}
    for _, ingredients in compatible_rows(recipe_ingredient_rows(), dietary):
        for ingredient in ingredients:
            if ingredient in excluded or ingredient in PANTRY_HARD:
                continue
            if ingredient.startswith(query):
                counts[ingredient] = counts.get(ingredient, 0) + 1

    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:10]
    return ApiResponse(data=ExplorerSearchResponse(
        ingredients=[ingredient for ingredient, _ in ranked],
    ))



