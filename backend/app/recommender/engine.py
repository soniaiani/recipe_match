"""Compatibility facade for the Bayesian recommendation engine."""
from __future__ import annotations

from app.recommender.bayesian.config import (
    ADAPTIVE_QUESTIONS as _ADAPTIVE_QS,
    COOKING_METHOD_EXCLUSIONS,
    CUISINE_OPTIONS,
    DRINK_ENTROPY_STOP_THRESHOLD,
    DRINK_MIN_QUESTIONS_BEFORE_STOP,
    DRINK_TOP10_PROB_STOP_THRESHOLD,
    ENTROPY_STOP_THRESHOLD,
    FIXED_QUESTIONS as _FIXED_QS,
    MAX_QUESTIONS,
    MEAL_TYPE_OPTIONS,
    MIN_QUESTIONS_BEFORE_STOP,
    P_CORRECT,
    P_NOISE,
    POSTERIOR_TEMPERATURE,
    PROTEIN_OPTIONS,
    QUESTION_BANK,
    STABILITY_MIN_OVERLAP,
    STABILITY_TOP_K,
    TOP_N,
    excluded_question_ids as _excluded_question_ids,
    question_by_id,
)
from app.recommender.bayesian.features import (
    compute_feature_mi,
    compute_likelihood,
    get_feature_value_bool,
)
from app.recommender.bayesian.questions import (
    restore_session,
    select_next_question,
)
from app.recommender.bayesian.session import BayesianSession

__all__ = [
    "BayesianSession",
    "COOKING_METHOD_EXCLUSIONS",
    "CUISINE_OPTIONS",
    "DRINK_ENTROPY_STOP_THRESHOLD",
    "DRINK_MIN_QUESTIONS_BEFORE_STOP",
    "DRINK_TOP10_PROB_STOP_THRESHOLD",
    "ENTROPY_STOP_THRESHOLD",
    "MAX_QUESTIONS",
    "MEAL_TYPE_OPTIONS",
    "MIN_QUESTIONS_BEFORE_STOP",
    "P_CORRECT",
    "P_NOISE",
    "POSTERIOR_TEMPERATURE",
    "PROTEIN_OPTIONS",
    "QUESTION_BANK",
    "STABILITY_MIN_OVERLAP",
    "STABILITY_TOP_K",
    "TOP_N",
    "_ADAPTIVE_QS",
    "_FIXED_QS",
    "_excluded_question_ids",
    "compute_feature_mi",
    "compute_likelihood",
    "get_feature_value_bool",
    "question_by_id",
    "restore_session",
    "select_next_question",
]
