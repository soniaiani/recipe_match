"""Question bank and constants for the Bayesian recommendation engine."""
from __future__ import annotations

import math
from typing import Any

MAX_QUESTIONS = 15
MIN_QUESTIONS_BEFORE_STOP = 4
TOP_N = 10
POSTERIOR_TEMPERATURE = 1.25
STABILITY_TOP_K = 3
STABILITY_MIN_OVERLAP = 2

P_CORRECT = 0.75
P_NOISE = 0.05

ENTROPY_STOP_THRESHOLD = math.log2(50)
DRINK_MIN_QUESTIONS_BEFORE_STOP = 10
DRINK_ENTROPY_STOP_THRESHOLD = math.log2(250)
DRINK_TOP10_PROB_STOP_THRESHOLD = 0.18

CUISINE_OPTIONS = [
    "italian",
    "asian",
    "mexican",
    "french",
    "mediterranean",
    "indian",
    "american",
    "other",
]
PROTEIN_OPTIONS = ["chicken", "beef_pork", "fish_seafood", "meatless", "other_meat"]
MEAL_TYPE_OPTIONS = [
    "appetizer",
    "breakfast",
    "dessert",
    "drink",
    "lunch_dinner",
    "salad_side",
    "snack",
    "soup",
    "condiment",
]

QUESTION_BANK: list[dict[str, Any]] = [
    {
        "id": "meal_type",
        "type": "categorical",
        "feature": "meal_type",
        "fixed": True,
        "order": 1,
        "options": MEAL_TYPE_OPTIONS,
    },
    {
        "id": "protein_type",
        "type": "multiselect",
        "feature": "protein_type",
        "fixed": True,
        "order": 3,
        "options": PROTEIN_OPTIONS,
        "any_option": "any",
    },
    {
        "id": "cuisine",
        "type": "multiselect",
        "feature": "cuisine",
        "fixed": True,
        "order": 2,
        "options": CUISINE_OPTIONS,
        "any_option": "any",
    },
    {"id": "is_spicy", "type": "boolean", "feature": "is_spicy", "feature_value": True},
    {"id": "is_sweet", "type": "boolean", "feature": "is_sweet", "feature_value": True},
    {"id": "is_quick", "type": "boolean", "feature": "is_quick", "feature_value": True},
    {"id": "needs_oven", "type": "boolean", "feature": "needs_oven", "feature_value": True},
    {"id": "needs_stovetop", "type": "boolean", "feature": "needs_stovetop", "feature_value": True},
    {"id": "is_no_cook", "type": "boolean", "feature": "is_no_cook", "feature_value": True},
    {"id": "has_pasta", "type": "boolean", "feature": "has_pasta", "feature_value": True},
    {"id": "has_rice", "type": "boolean", "feature": "has_rice", "feature_value": True},
    {"id": "has_potato", "type": "boolean", "feature": "has_potato", "feature_value": True},
    {"id": "has_tomato_base", "type": "boolean", "feature": "has_tomato_base", "feature_value": True},
    {"id": "has_cream_base", "type": "boolean", "feature": "has_cream_base", "feature_value": True},
    {"id": "has_cheese", "type": "boolean", "feature": "has_cheese", "feature_value": True},
    {"id": "has_broth_base", "type": "boolean", "feature": "has_broth_base", "feature_value": True},
    {"id": "has_mushroom", "type": "boolean", "feature": "has_mushroom", "feature_value": True},
    {"id": "has_leafy_greens", "type": "boolean", "feature": "has_leafy_greens", "feature_value": True},
    {"id": "has_beans_legumes", "type": "boolean", "feature": "has_beans_legumes", "feature_value": True},
    {"id": "has_fruit", "type": "boolean", "feature": "has_fruit", "feature_value": True},
    {"id": "has_nuts", "type": "boolean", "feature": "has_nuts", "feature_value": True},
    {"id": "has_chocolate", "type": "boolean", "feature": "has_chocolate", "feature_value": True},
    {"id": "has_asian_sauce", "type": "boolean", "feature": "has_asian_sauce", "feature_value": True},
]

FIXED_QUESTIONS = sorted(
    [question for question in QUESTION_BANK if question.get("fixed")],
    key=lambda question: question.get("order", 99),
)
ADAPTIVE_QUESTIONS = [
    question
    for question in QUESTION_BANK
    if not question.get("fixed")
]

COOKING_METHOD_EXCLUSIONS: dict[str, set[str]] = {
    "is_no_cook": {"needs_oven", "needs_stovetop"},
    "needs_oven": {"needs_stovetop", "is_no_cook"},
    "needs_stovetop": {"needs_oven", "is_no_cook"},
}


def question_by_id(question_id: str) -> dict[str, Any] | None:
    return next((question for question in QUESTION_BANK if question["id"] == question_id), None)


def excluded_question_ids(answers: dict[str, Any]) -> set[str]:
    excluded: set[str] = set()
    for question_id, blocked in COOKING_METHOD_EXCLUSIONS.items():
        if answers.get(question_id) == "yes":
            excluded.update(blocked)
    return excluded - set(answers.keys())
