"""Question selection and session reconstruction."""
from __future__ import annotations

from typing import Any

import numpy as np

from app.services.recommendations.bayesian.config import (
    ADAPTIVE_QUESTIONS,
    FIXED_QUESTIONS,
    excluded_question_ids,
    question_by_id,
)
from app.services.recommendations.bayesian.features import get_feature_value_bool
from app.services.recommendations.bayesian.session import BayesianSession


def select_next_question(session: BayesianSession) -> dict[str, Any] | None:
    excluded_ids = excluded_question_ids(session.answers)

    fixed_candidates = [
        question
        for question in FIXED_QUESTIONS
        if question["id"] not in session.asked_ids and question["id"] not in excluded_ids
    ]
    if fixed_candidates:
        return fixed_candidates[0]

    candidates = [
        question
        for question in ADAPTIVE_QUESTIONS
        if question["id"] not in session.asked_ids and question["id"] not in excluded_ids
    ]
    if not candidates:
        return None

    if _has_consecutive_no_or_any_answers(session):
        return max(candidates, key=lambda question: _prevalence_boosted_score(session, question))

    return max(candidates, key=lambda question: session.expected_entropy_reduction(question))


def restore_session(
    answers: dict[str, Any],
    recipes: list[dict[str, Any]],
    weights: dict[str, float],
) -> BayesianSession:
    """Replay saved answers to reconstruct Bayesian state."""
    session = BayesianSession(recipes, weights)
    for question_id, answer in answers.items():
        question = question_by_id(question_id)
        if question:
            session.update(question, answer)
    return session


def _has_consecutive_no_or_any_answers(session: BayesianSession) -> bool:
    recent = list(session.answers.values())[-3:]
    return sum(1 for answer in recent if answer == "no" or answer == ["any"]) >= 3


def _prevalence_boosted_score(session: BayesianSession, question: dict[str, Any]) -> float:
    information_gain = session.expected_entropy_reduction(question)
    if question["type"] != "boolean":
        return information_gain

    relevant_recipes = _relevant_recipes(session)
    if not relevant_recipes:
        return information_gain

    count = sum(1 for recipe in relevant_recipes if get_feature_value_bool(recipe, question))
    prevalence = count / len(relevant_recipes)
    boost = max(1.0, 1.0 + 2.0 * (1.0 - abs(prevalence - 0.35) / 0.35))
    return information_gain * boost


def _relevant_recipes(session: BayesianSession) -> list[dict[str, Any]]:
    probabilities = session.probs()
    uniform = 1.0 / session.n
    relevant_idx = np.where(probabilities > uniform * 0.01)[0]
    return [session.recipes[index] for index in relevant_idx]
