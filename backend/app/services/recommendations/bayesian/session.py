"""Bayesian session state and scoring."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from app.services.recommendations.bayesian.config import (
    DRINK_ENTROPY_STOP_THRESHOLD,
    DRINK_MIN_QUESTIONS_BEFORE_STOP,
    DRINK_TOP10_PROB_STOP_THRESHOLD,
    ENTROPY_STOP_THRESHOLD,
    FIXED_QUESTIONS,
    MAX_QUESTIONS,
    MIN_QUESTIONS_BEFORE_STOP,
    POSTERIOR_TEMPERATURE,
    STABILITY_MIN_OVERLAP,
    STABILITY_TOP_K,
    TOP_N,
    question_by_id,
)
from app.services.recommendations.bayesian.features import (
    compute_likelihood,
    get_feature_value_bool,
)


class BayesianSession:
    def __init__(self, recipes: list[dict[str, Any]], weights: dict[str, float]) -> None:
        self.recipes = recipes
        self.n = len(recipes)
        self.log_probs = np.full(self.n, -math.log(self.n))
        self.answers: dict[str, Any] = {}
        self.asked_ids: set[str] = set()
        self.q = 0
        self.weights = weights
        self.top_history: list[tuple[int, ...]] = []

    def probs(self) -> np.ndarray:
        log_p = self.log_probs / POSTERIOR_TEMPERATURE
        log_p -= log_p.max()
        probabilities = np.exp(log_p)
        return probabilities / probabilities.sum()

    def update(self, question: dict[str, Any], answer: Any) -> None:
        self.q += 1
        self.answers[question["id"]] = answer
        self.asked_ids.add(question["id"])

        if answer == "skip" or answer == ["any"] or answer == "any":
            self.top_history.append(self._top_ids())
            return

        likelihoods = np.array([
            compute_likelihood(recipe, question, answer)
            for recipe in self.recipes
        ])
        weight = self.weights.get(question["id"], 1.0)
        self.log_probs += weight * np.log(likelihoods + 1e-10)
        self.log_probs -= self.log_probs.max()
        self.top_history.append(self._top_ids())

    def entropy(self) -> float:
        probabilities = self.probs()
        non_zero = probabilities[probabilities > 1e-10]
        return float(-np.sum(non_zero * np.log2(non_zero)))

    def should_stop(self) -> bool:
        if self.q <= len(FIXED_QUESTIONS):
            return False
        if self.q < MIN_QUESTIONS_BEFORE_STOP:
            return False
        if self._recent_answers_are_uninformative():
            return True
        if self.q >= MAX_QUESTIONS:
            return True

        entropy_ok = self.entropy() < ENTROPY_STOP_THRESHOLD
        top_k_prob = self._top_probability_mass(10)
        concentrated = top_k_prob > 0.25
        stable = self.is_top_stable()

        if self._drink_session_can_stop(top_k_prob, stable):
            return True

        return sum([entropy_ok, concentrated, stable]) >= 2

    def is_top_stable(self) -> bool:
        if len(self.top_history) < 2:
            return False
        current = set(self.top_history[-1])
        previous = set(self.top_history[-2])
        return len(current & previous) >= STABILITY_MIN_OVERLAP

    def expected_entropy_reduction(self, question: dict[str, Any]) -> float:
        probabilities = self.probs()
        current_entropy = self.entropy()
        relevant_idx = self._relevant_indices(probabilities)
        relevant_probabilities = probabilities[relevant_idx]
        relevant_probabilities = relevant_probabilities / relevant_probabilities.sum()

        expected_entropy = 0.0
        for answer in _possible_answers(question):
            likelihoods = np.array([
                compute_likelihood(self.recipes[i], question, answer)
                for i in relevant_idx
            ])
            answer_probability = float(np.dot(likelihoods, relevant_probabilities))
            if answer_probability < 1e-12:
                continue
            expected_entropy += answer_probability * _updated_entropy(
                relevant_probabilities,
                likelihoods,
            )

        return max(0.0, current_entropy - expected_entropy)

    def compute_answer_match_score(self, recipe: dict[str, Any]) -> float:
        total_weight = 0.0
        matched_weight = 0.0

        for question_id, answer in self.answers.items():
            question = question_by_id(question_id)
            if _answer_is_ignored(answer) or not question:
                continue
            if question["type"] == "boolean" and answer not in ("yes", "no"):
                continue

            weight = self.weights.get(question_id, 1.0)
            total_weight += weight
            if _recipe_matches_answer(recipe, question, answer):
                matched_weight += weight

        if total_weight == 0:
            return 0.0
        return round((matched_weight / total_weight) * 100, 1)

    def compute_posterior_score(self, recipe_index: int) -> float:
        probabilities = self.probs()
        uniform = 1.0 / self.n
        lift = float(probabilities[recipe_index]) / uniform
        scaling = math.log2(self.n)
        return round(100.0 * lift / (lift + scaling), 1)

    def top(
        self,
        n: int = TOP_N,
        min_match_score: float = 50.0,
    ) -> list[tuple[dict[str, Any], float]]:
        probabilities = self.probs()
        candidate_idx = self._hard_filtered_indices()
        candidate_idx.sort(key=lambda index: -probabilities[index])

        scored = [
            (self.recipes[index], self.compute_posterior_score(index))
            for index in candidate_idx
        ]
        if min_match_score <= 0:
            return scored[:n]
        return _thresholded_results(scored, n, min_match_score)

    def _top_ids(self, k: int = STABILITY_TOP_K) -> tuple[int, ...]:
        probabilities = self.probs()
        top_idx = probabilities.argsort()[-k:][::-1]
        return tuple(int(self.recipes[index]["id"]) for index in top_idx)

    def _recent_answers_are_uninformative(self) -> bool:
        recent = list(self.answers.values())[-4:]
        return (
            self.q > len(FIXED_QUESTIONS) + 2
            and len(recent) >= 4
            and all(answer in ("skip", "any") or answer == ["any"] for answer in recent)
        )

    def _top_probability_mass(self, count: int) -> float:
        probabilities = self.probs()
        return float(probabilities[probabilities.argsort()[-count:]].sum())

    def _drink_session_can_stop(self, top_k_prob: float, stable: bool) -> bool:
        if self.answers.get("meal_type") != "drink":
            return False
        if self.q < DRINK_MIN_QUESTIONS_BEFORE_STOP:
            return False

        relaxed_entropy_ok = self.entropy() < DRINK_ENTROPY_STOP_THRESHOLD
        relaxed_concentrated = top_k_prob > DRINK_TOP10_PROB_STOP_THRESHOLD
        return relaxed_entropy_ok or sum([relaxed_entropy_ok, relaxed_concentrated, stable]) >= 2

    def _relevant_indices(self, probabilities: np.ndarray) -> np.ndarray:
        uniform = 1.0 / self.n
        relevant_idx = np.where(probabilities > uniform * 0.01)[0]
        if len(relevant_idx) < 10:
            relevant_idx = probabilities.argsort()[-100:]
        return relevant_idx

    def _hard_filtered_indices(self) -> list[int]:
        hard_filtered = [
            index
            for index, recipe in enumerate(self.recipes)
            if _recipe_passes_hard_answers(recipe, self.answers)
        ]
        return hard_filtered if len(hard_filtered) >= 3 else list(range(self.n))


def _possible_answers(question: dict[str, Any]) -> list[Any]:
    question_type = question["type"]
    if question_type == "categorical":
        return question.get("options", []) + ["skip"]
    if question_type == "multiselect":
        return [[option] for option in question.get("options", [])] + [["any"]]
    return ["yes", "no"]


def _updated_entropy(probabilities: np.ndarray, likelihoods: np.ndarray) -> float:
    log_updated = np.log(probabilities + 1e-10) + np.log(likelihoods + 1e-10)
    if POSTERIOR_TEMPERATURE > 1.0:
        log_updated /= POSTERIOR_TEMPERATURE
    log_updated -= log_updated.max()
    updated = np.exp(log_updated)
    updated /= updated.sum()
    non_zero = updated[updated > 1e-10]
    return float(-np.sum(non_zero * np.log2(non_zero)))


def _answer_is_ignored(answer: Any) -> bool:
    return answer == "skip" or answer == ["any"] or answer == "any"


def _recipe_matches_answer(recipe: dict[str, Any], question: dict[str, Any], answer: Any) -> bool:
    if question["type"] == "categorical":
        return recipe.get(question["feature"]) == answer
    if question["type"] == "multiselect":
        selected = answer if isinstance(answer, list) else [answer]
        return recipe.get(question["feature"]) in selected
    if question["type"] == "boolean":
        recipe_matches = get_feature_value_bool(recipe, question)
        return (answer == "yes" and recipe_matches) or (answer == "no" and not recipe_matches)
    return False


def _recipe_passes_hard_answers(recipe: dict[str, Any], answers: dict[str, Any]) -> bool:
    for question_id, answer in answers.items():
        if answer in ("skip", "no") or answer == ["any"] or answer == "any":
            continue
        question = question_by_id(question_id)
        if not question:
            continue
        if question["type"] == "categorical" and recipe.get(question["feature"]) != answer:
            return False
        if question["type"] == "multiselect":
            selected = answer if isinstance(answer, list) else [answer]
            if recipe.get(question["feature"]) not in selected:
                return False
    return True


def _thresholded_results(
    scored: list[tuple[dict[str, Any], float]],
    n: int,
    min_match_score: float,
) -> list[tuple[dict[str, Any], float]]:
    thresholds = [70.0, 60.0, min_match_score, 40.0, 30.0, 0.0]
    seen: set[float] = set()
    for threshold in thresholds:
        if threshold in seen:
            continue
        seen.add(threshold)
        results = [(recipe, score) for recipe, score in scored if score >= threshold]
        if len(results) >= 3 or threshold == 0.0:
            return results[:n]
    return scored[:n]
