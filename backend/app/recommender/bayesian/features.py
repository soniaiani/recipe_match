"""Feature weighting and likelihood helpers for Bayesian recommendations."""
from __future__ import annotations

import math
from collections import Counter
from typing import Any

from app.recommender.bayesian.config import (
    ADAPTIVE_QUESTIONS,
    P_CORRECT,
    P_NOISE,
)


def compute_feature_mi(recipes: list[dict[str, Any]]) -> dict[str, float]:
    """Compute mutual-information weights for adaptive questions."""
    n = len(recipes)
    if n == 0:
        return {}

    targets = [
        (recipe.get("cuisine", "unknown"), recipe.get("protein_type", "unknown"))
        for recipe in recipes
    ]
    target_counts = Counter(targets)

    mi_scores: dict[str, float] = {}
    for question in ADAPTIVE_QUESTIONS:
        feature = question["feature"]
        question_id = question["id"]
        feature_value = question.get("feature_value")
        feature_values = [
            _matches_feature_value(recipe, feature, feature_value)
            for recipe in recipes
        ]
        mi_scores[question_id] = _mutual_information(
            feature_values,
            targets,
            target_counts,
            n,
        )

    max_mi = max(mi_scores.values()) if mi_scores else 1.0
    if max_mi <= 0:
        max_mi = 1.0

    normalized = {
        question_id: round(0.3 + (mi / max_mi) * 2.7, 4)
        for question_id, mi in mi_scores.items()
    }
    normalized["meal_type"] = 2.0
    normalized["cuisine"] = 1.8
    normalized["protein_type"] = 1.6
    return normalized


def get_feature_value_bool(recipe: dict[str, Any], question: dict[str, Any]) -> bool:
    return _matches_feature_value(
        recipe,
        question["feature"],
        question.get("feature_value"),
    )


def compute_likelihood(
    recipe: dict[str, Any],
    question: dict[str, Any],
    answer: Any,
) -> float:
    question_type = question["type"]
    feature = question["feature"]

    if question_type == "categorical":
        if answer in ("skip", "unknown"):
            return 1.0
        return P_CORRECT if recipe.get(feature) == answer else P_NOISE

    if question_type == "multiselect":
        if answer == ["any"] or answer == "any" or answer == "unknown":
            return 1.0
        selected = answer if isinstance(answer, list) else [answer]
        return P_CORRECT if recipe.get(feature) in selected else P_NOISE

    if answer in ("skip", "unknown"):
        return 1.0
    match = get_feature_value_bool(recipe, question)
    if answer == "yes":
        return P_CORRECT if match else P_NOISE
    if answer == "no":
        return P_CORRECT if not match else P_NOISE
    return 1.0


def _matches_feature_value(
    recipe: dict[str, Any],
    feature: str,
    feature_value: Any,
) -> bool:
    recipe_value = recipe.get(feature)
    if isinstance(feature_value, bool):
        return bool(recipe_value) == feature_value
    return recipe_value == feature_value


def _mutual_information(
    feature_values: list[bool],
    targets: list[tuple[Any, Any]],
    target_counts: Counter,
    n: int,
) -> float:
    joint_counts: Counter = Counter(zip(feature_values, targets))
    feature_counts = Counter(feature_values)

    mi = 0.0
    for (feature_value, target), count in joint_counts.items():
        p_joint = count / n
        p_feature = feature_counts[feature_value] / n
        p_target = target_counts[target] / n
        if p_joint > 0 and p_feature > 0 and p_target > 0:
            mi += p_joint * math.log2(p_joint / (p_feature * p_target))
    return max(mi, 0.0)
