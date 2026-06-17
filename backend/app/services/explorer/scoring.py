"""Statistical scoring helpers for Ingredient Explorer suggestions."""
from __future__ import annotations

from typing import Iterable

from app.services.explorer.common import (
    MAX_EXPAND_CANDIDATES,
    PANTRY_HARD,
    PANTRY_SOFT,
    normalize_ingredient,
    normalize_many,
    parse_ingredients,
)

def candidate_counts_from_ingredient_sets(
    matching_ingredients: Iterable[set[str]],
    selected_set: set[str],
    min_count: int | None = None,
    exclude_soft_pantry: bool = False,
    excluded_ingredients: list[str] | None = None,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for ingredients in matching_ingredients:
        for ingredient in ingredients - selected_set:
            counts[ingredient] = counts.get(ingredient, 0) + 1

    if min_count is not None:
        counts = filter_candidate_counts_by_context(
            counts,
            min_count=min_count,
            exclude_soft_pantry=exclude_soft_pantry,
            excluded_ingredients=excluded_ingredients,
        )

    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return dict(ranked[:MAX_EXPAND_CANDIDATES])


def filter_candidate_counts_by_context(
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
        and ingredient not in PANTRY_HARD
        and ingredient not in excluded
        and (not exclude_soft_pantry or ingredient not in PANTRY_SOFT)
    }


def score_start_candidates(
    candidate_counts: dict[str, int],
    n_matching: int,
) -> list[tuple[str, float]]:
    scored = [
        (candidate, count / n_matching)
        for candidate, count in candidate_counts.items()
    ]
    scored.sort(key=lambda item: (-item[1], item[0]))
    return scored
