"""Statistical scoring helpers for Ingredient Explorer suggestions."""
from __future__ import annotations

import math
from statistics import mean
from typing import Iterable

from app.services.explorer_common import (
    MAX_EXPAND_CANDIDATES,
    PANTRY_HARD,
    PANTRY_SOFT,
    normalize_ingredient,
    normalize_many,
    parse_ingredients,
)

EXPAND_PPMI_SHIFT_K = 1.0
EXPAND_TFIDF_ALPHA = 0.8
EXPAND_LOG_K = math.log2(EXPAND_PPMI_SHIFT_K)


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


def min_max_normalize(values: dict[str, float]) -> dict[str, float]:
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


def score_expand_candidates(
    candidate_counts: dict[str, int],
    n_matching: int,
    idf: dict[str, float],
    ppmi_scores: dict[str, list[float]],
    alpha_tfidf: float = EXPAND_TFIDF_ALPHA,
    log_k: float = EXPAND_LOG_K,
) -> list[tuple[str, float, float]]:
    tfidf: dict[str, float] = {}
    shifted_ppmi: dict[str, float] = {}

    for candidate, count in candidate_counts.items():
        frequency_score = count / n_matching
        tfidf[candidate] = frequency_score * idf.get(candidate, 1.0)

        scores = ppmi_scores.get(candidate, [])
        avg_ppmi = mean(scores) if scores else 0.0
        shifted_ppmi[candidate] = max(avg_ppmi - log_k, 0.0)

    tfidf_norm = min_max_normalize(tfidf)
    ppmi_norm = min_max_normalize(shifted_ppmi)
    scored = [
        (
            candidate,
            alpha_tfidf * tfidf_norm.get(candidate, 0.0)
            + (1.0 - alpha_tfidf) * ppmi_norm.get(candidate, 0.0),
            shifted_ppmi.get(candidate, 0.0),
        )
        for candidate in candidate_counts
    ]
    scored.sort(key=lambda item: (-item[1], item[0]))
    return scored
