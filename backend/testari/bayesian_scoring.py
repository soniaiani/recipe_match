"""
Bayesian scoring pentru Find Your Recipe.
Inlocuieste scoring-ul weighted cu un model probabilistic principial.

La fiecare raspuns, actualizeaza distributia de probabilitate peste retete
folosind regula lui Bayes:
    P(r | a1...ak) ∝ P(ak | r) × P(r | a1...ak-1)

Referinte:
- Akinator-style adaptive questioning (Flajolet et al. 2011)
- Bayesian active learning for recommendation (Houlsby et al. 2012)
"""
from __future__ import annotations
import math
import numpy as np
from typing import Any

# ── NOISE MODEL ───────────────────────────────────────────────────────────────
# Probabilitatea ca utilizatorul sa raspunda corect
P_CORRECT = 0.90
# Probabilitatea de raspuns eronat (noise)
P_NOISE   = 0.05
# P_CORRECT + P_NOISE < 1.0 — diferenta e probabilitatea de raspuns incert

# ── PRAG DE OPRIRE ────────────────────────────────────────────────────────────
# Opreste cand distributia s-a concentrat pe mai putin de 50 retete candidate
# H < log2(50) ≈ 5.64 biti
ENTROPY_STOP_THRESHOLD = math.log2(50)
MIN_QUESTIONS = 3
MAX_QUESTIONS = 9


def compute_likelihood(
    recipe: dict[str, Any],
    question: dict[str, Any],
    answer: str,
) -> float:
    """
    P(answer | recipe) — likelihood-ul raspunsului dat reteta.

    Pentru boolean: 0.90 daca raspunsul se potriveste cu reteta, 0.05 altfel.
    Pentru categorial: 0.90 / n_selected daca reteta e in selectie, 0.05 altfel.
    Pentru skip: 1.0 — nu modifica distributia.
    """
    if answer == "skip":
        return 1.0

    feature = question["feature"]
    feature_value = question.get("feature_value")
    q_type = question["type"]

    # valoarea feature-ului din reteta
    recipe_val = recipe.get(feature)

    if q_type == "boolean":
        # potrivire booleana directa sau prin feature_value
        if feature_value is None:
            recipe_matches = bool(recipe_val)
        elif isinstance(feature_value, bool):
            recipe_matches = bool(recipe_val) == feature_value
        else:
            recipe_matches = recipe_val == feature_value

        if answer == "yes":
            return P_CORRECT if recipe_matches else P_NOISE
        else:  # answer == "no"
            return P_CORRECT if not recipe_matches else P_NOISE

    else:  # categorial (meal_type)
        # select-all = skip
        all_opts = question.get("options", [])
        if all_opts and set(answer) >= set(all_opts):
            return 1.0

        selected = answer if isinstance(answer, list) else [answer]
        recipe_matches = recipe_val in selected

        if recipe_matches:
            return P_CORRECT
        else:
            return P_NOISE


class BayesianSession:
    """
    Sesiune de recomandare bazata pe Bayesian updating.

    Mentine o distributie de probabilitate P(r) peste toate retetele
    si o actualizeaza la fiecare raspuns al utilizatorului.
    """

    def __init__(self, recipes: list[dict[str, Any]]) -> None:
        self.recipes = recipes
        self.n = len(recipes)
        # distributie initiala uniforma in log-space pentru stabilitate numerica
        self._log_probs = np.full(self.n, -math.log(self.n))
        self._recipe_index = {r["id"]: i for i, r in enumerate(recipes)}
        self.answers: dict[str, str] = {}
        self.asked_ids: set[str] = set()
        self.question_number = 0

    @property
    def probs(self) -> np.ndarray:
        """Distributia de probabilitate normalizata."""
        log_p = self._log_probs - self._log_probs.max()
        p = np.exp(log_p)
        return p / p.sum()

    def update(self, question: dict[str, Any], answer: str) -> None:
        """
        Actualizeaza distributia dupa raspunsul utilizatorului.
        P(r | answer) ∝ P(answer | r) × P(r)
        """
        self.answers[question["id"]] = answer
        self.asked_ids.add(question["id"])
        self.question_number += 1

        if answer == "skip":
            return  # skip nu modifica distributia

        # calculeaza log-likelihood pentru fiecare reteta
        log_likelihoods = np.array([
            math.log(compute_likelihood(r, question, answer) + 1e-10)
            for r in self.recipes
        ])

        # actualizare Bayesiana in log-space
        self._log_probs = self._log_probs + log_likelihoods
        # renormalizeaza pentru stabilitate numerica
        self._log_probs = self._log_probs - self._log_probs.max()

    def entropy(self) -> float:
        """Entropia distributiei curente H = -Σ P(r) log₂ P(r)."""
        p = self.probs
        # elimina probabilitatile zero pentru stabilitate
        p_nonzero = p[p > 1e-10]
        return float(-np.sum(p_nonzero * np.log2(p_nonzero)))

    def should_stop(self) -> bool:
        """
        Opreste cand distributia s-a concentrat suficient.
        Criteriu: H < log2(50) ≈ 5.64 biti
        """
        if self.question_number < MIN_QUESTIONS:
            return False
        if self.question_number >= MAX_QUESTIONS:
            return True
        return self.entropy() < ENTROPY_STOP_THRESHOLD

    def top_n(self, n: int = 10) -> list[tuple[dict[str, Any], float]]:
        """
        Returneaza top-n retete cu probabilitatile lor ca scoruri (0-100%).
        """
        p = self.probs
        top_indices = p.argsort()[-n:][::-1]
        return [
            (self.recipes[i], round(float(p[i]) * 100 * self.n / n, 2))
            for i in top_indices
        ]

    def top_n_scored(self, n: int = 10) -> list[tuple[dict[str, Any], float]]:
        """
        Returneaza top-n retete cu scoruri normalizate 0-100%.
        Scorul 100% = reteta cu probabilitate maxima.
        """
        p = self.probs
        top_indices = p.argsort()[-n:][::-1]
        max_p = p[top_indices[0]] if len(top_indices) > 0 else 1.0
        return [
            (self.recipes[i], round(float(p[i] / max_p) * 100, 2))
            for i in top_indices
        ]

    def expected_entropy_reduction(
        self,
        question: dict[str, Any],
        question_bank: list[dict[str, Any]],
    ) -> float:
        """
        Calculeaza reducerea asteptata de entropie daca punem aceasta intrebare.

        E[H_after] = Σ_a P(a) × H(distributie dupa raspuns a)
        Scorul = H_current - E[H_after]  (mai mare = mai bun)

        Aceasta e selectia optima Bayesiana a intrebarii.
        """
        p = self.probs
        current_h = self.entropy()

        answers_to_check = ["yes", "no"] if question["type"] == "boolean" else \
                           (question.get("options", []) + [["skip"]])

        expected_h_after = 0.0

        for answer in answers_to_check:
            answer_str = answer if isinstance(answer, str) else answer[0]

            # P(a) = Σ_r P(a|r) × P(r)
            likelihoods = np.array([
                compute_likelihood(r, question, answer_str)
                for r in self.recipes
            ])
            p_answer = float(np.dot(likelihoods, p))

            if p_answer < 1e-10:
                continue

            # distributia dupa raspuns a
            log_p_updated = self._log_probs + np.log(likelihoods + 1e-10)
            log_p_updated = log_p_updated - log_p_updated.max()
            p_updated = np.exp(log_p_updated)
            p_updated = p_updated / p_updated.sum()

            # entropia distributiei actualizate
            p_nz = p_updated[p_updated > 1e-10]
            h_after = float(-np.sum(p_nz * np.log2(p_nz)))

            expected_h_after += p_answer * h_after

        return current_h - expected_h_after


def select_next_question(
    session: BayesianSession,
    question_bank: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """
    Selecteaza urmatoarea intrebare care maximizeaza reducerea asteptata de entropie.
    Aceasta e echivalenta cu Information Gain maximization.
    """
    fixed_qs = sorted(
        [q for q in question_bank if q.get("fixed")],
        key=lambda q: q.get("order", 99),
    )
    adaptive_qs = [q for q in question_bank if not q.get("fixed")]

    # intrebarile fixe merg primele
    fixed_idx = session.question_number
    if fixed_idx < len(fixed_qs):
        return fixed_qs[fixed_idx]

    # selectie adaptiva: maximizeaza Information Gain
    candidates = [q for q in adaptive_qs if q["id"] not in session.asked_ids]
    if not candidates:
        return None

    return max(
        candidates,
        key=lambda q: session.expected_entropy_reduction(q, question_bank),
    )