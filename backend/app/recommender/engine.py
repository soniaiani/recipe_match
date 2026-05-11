"""Bayesian recommendation engine — extracted from recipe_recc_v4.py."""
from __future__ import annotations

import math
from collections import Counter
from typing import Any

import numpy as np

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
MAX_QUESTIONS = 15
MIN_QUESTIONS_BEFORE_STOP = 4
TOP_N = 10
POSTERIOR_TEMPERATURE = 1.25
STABILITY_TOP_K = 3
STABILITY_MIN_OVERLAP = 2

P_CORRECT = 0.90
P_NOISE   = 0.05

# Stop when H < log2(15) ≈ 3.9 bits (focused on ~15 candidates)
ENTROPY_STOP_THRESHOLD = math.log2(50)

# ── QUESTION BANK ─────────────────────────────────────────────────────────────
CUISINE_OPTIONS   = ["italian", "asian", "mexican", "french", "mediterranean", "indian", "american", "other"]
PROTEIN_OPTIONS   = ["chicken", "beef_pork", "fish_seafood", "meatless"]
MEAL_TYPE_OPTIONS = ["appetizer", "breakfast", "dessert", "drink", "lunch_dinner",
                     "salad_side", "snack", "soup", "condiment"]

QUESTION_BANK: list[dict[str, Any]] = [
    {
        "id": "meal_type",
        "type": "categorical",
        "feature": "meal_type",
        "fixed": True,
        "order": 1,
        "options": MEAL_TYPE_OPTIONS,
    },
    # Adaptive questions
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
    
    {"id": "is_spicy",              "type": "boolean", "feature": "is_spicy",              "feature_value": True},
    {"id": "is_sweet",              "type": "boolean", "feature": "is_sweet",              "feature_value": True},
    {"id": "is_quick",              "type": "boolean", "feature": "is_quick",              "feature_value": True},
    {"id": "needs_oven",            "type": "boolean", "feature": "needs_oven",            "feature_value": True},
    {"id": "needs_stovetop",        "type": "boolean", "feature": "needs_stovetop",        "feature_value": True},
    {"id": "is_no_cook",            "type": "boolean", "feature": "is_no_cook",            "feature_value": True},
    {"id": "has_pasta",             "type": "boolean", "feature": "has_pasta",             "feature_value": True},
    {"id": "has_rice",              "type": "boolean", "feature": "has_rice",              "feature_value": True},
    {"id": "has_potato",            "type": "boolean", "feature": "has_potato",            "feature_value": True},
    {"id": "has_tomato_base",       "type": "boolean", "feature": "has_tomato_base",       "feature_value": True},
    {"id": "has_cream_base",        "type": "boolean", "feature": "has_cream_base",        "feature_value": True},
    {"id": "has_cheese",            "type": "boolean", "feature": "has_cheese",            "feature_value": True},
    {"id": "has_broth_base",        "type": "boolean", "feature": "has_broth_base",        "feature_value": True},
    {"id": "has_mushroom",          "type": "boolean", "feature": "has_mushroom",          "feature_value": True},
    {"id": "has_leafy_greens",      "type": "boolean", "feature": "has_leafy_greens",      "feature_value": True},
    {"id": "has_beans_legumes",     "type": "boolean", "feature": "has_beans_legumes",     "feature_value": True},
    {"id": "has_fruit",             "type": "boolean", "feature": "has_fruit",             "feature_value": True},
    {"id": "has_nuts",              "type": "boolean", "feature": "has_nuts",              "feature_value": True},
    {"id": "has_chocolate",         "type": "boolean", "feature": "has_chocolate",         "feature_value": True},
    {"id": "has_asian_sauce",       "type": "boolean", "feature": "has_asian_sauce",       "feature_value": True},
]

_FIXED_QS    = sorted([q for q in QUESTION_BANK if q.get("fixed")], key=lambda q: q.get("order", 99))
_ADAPTIVE_QS = [q for q in QUESTION_BANK if not q.get("fixed")]


COOKING_METHOD_EXCLUSIONS: dict[str, set[str]] = {
    "is_no_cook": {"needs_oven", "needs_stovetop"},
    "needs_oven": {"needs_stovetop", "is_no_cook"},
    "needs_stovetop": {"needs_oven", "is_no_cook"},
}


def question_by_id(qid: str) -> dict[str, Any] | None:
    return next((q for q in QUESTION_BANK if q["id"] == qid), None)


def _excluded_question_ids(answers: dict[str, Any]) -> set[str]:
    excluded: set[str] = set()
    for question_id, blocked in COOKING_METHOD_EXCLUSIONS.items():
        if answers.get(question_id) == "yes":
            excluded.update(blocked)
    return excluded - set(answers.keys())


# ── MI-BASED WEIGHTS ──────────────────────────────────────────────────────────
def compute_feature_mi(recipes: list[dict[str, Any]]) -> dict[str, float]:
    """
    Calculates I(feature; cuisine, protein_type) — mutual information between
    each adaptive feature and the (cuisine, protein_type) pair.

    This measures how well each feature discriminates between recipe types
    within a meal_type subset, which is the relevant target after the first
    fixed question is answered.
    """
    n = len(recipes)
    if n == 0:
        return {}

    mi_scores: dict[str, float] = {}

    targets = [
        (r.get("cuisine", "unknown"), r.get("protein_type", "unknown"))
        for r in recipes
    ]
    t_cnt = Counter(targets)

    for q in _ADAPTIVE_QS:
        feature = q["feature"]
        qid = q["id"]
        feature_value = q.get("feature_value")

        if isinstance(feature_value, bool):
            fvals = [bool(r.get(feature)) == feature_value for r in recipes]
        else:
            fvals = [r.get(feature) == feature_value for r in recipes]

        joint: Counter = Counter(zip(fvals, targets))
        f_cnt = Counter(fvals)

        mi = 0.0
        for (fv, tgt), cnt in joint.items():
            p_joint = cnt / n
            p_f = f_cnt[fv] / n
            p_t = t_cnt[tgt] / n
            if p_joint > 0 and p_f > 0 and p_t > 0:
                mi += p_joint * math.log2(p_joint / (p_f * p_t))

        mi_scores[qid] = max(mi, 0.0)

    max_mi = max(mi_scores.values()) if mi_scores else 1.0
    if max_mi <= 0:
        max_mi = 1.0

    normalized = {
        qid: round(0.3 + (mi / max_mi) * 2.7, 4)
        for qid, mi in mi_scores.items()
    }
    normalized["meal_type"] = 3.0
    return normalized

# ── HELPERS ───────────────────────────────────────────────────────────────────
def get_feature_value_bool(recipe: dict[str, Any], question: dict[str, Any]) -> bool:
    feature = question["feature"]
    feature_value = question.get("feature_value")
    recipe_val = recipe.get(feature)
    if isinstance(feature_value, bool):
        return bool(recipe_val) == feature_value
    return recipe_val == feature_value


# ── BAYESIAN LIKELIHOOD ───────────────────────────────────────────────────────
def compute_likelihood(
    recipe: dict[str, Any],
    question: dict[str, Any],
    answer: Any,
) -> float:
    q_type = question["type"]
    feature = question["feature"]

    if q_type == "categorical":
        if answer in ("skip", "unknown"):
            return 1.0
        return P_CORRECT if recipe.get(feature) == answer else P_NOISE

    if q_type == "multiselect":
        if answer == ["any"] or answer == "any" or answer == "unknown":
            return 1.0
        selected = answer if isinstance(answer, list) else [answer]
        return P_CORRECT if recipe.get(feature) in selected else P_NOISE

    # boolean
    if answer in ("skip", "unknown"):
        return 1.0
    match = get_feature_value_bool(recipe, question)
    if answer == "yes":
        return P_CORRECT if match else P_NOISE
    if answer == "no":
        return P_CORRECT if not match else P_NOISE
    return 1.0


# ── BAYESIAN SESSION ──────────────────────────────────────────────────────────
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
        p = np.exp(log_p)
        return p / p.sum()

    def _top_ids(self, k: int = STABILITY_TOP_K) -> tuple[int, ...]:
        p = self.probs()
        top_idx = p.argsort()[-k:][::-1]
        return tuple(int(self.recipes[i]["id"]) for i in top_idx)

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

        w = self.weights.get(question["id"], 1.0)
        self.log_probs += w * np.log(likelihoods + 1e-10)
        self.log_probs -= self.log_probs.max()
        self.top_history.append(self._top_ids())

    def entropy(self) -> float:
        p = self.probs()
        p_nz = p[p > 1e-10]
        return float(-np.sum(p_nz * np.log2(p_nz)))

    def should_stop(self) -> bool:
        # Nu opri niciodată în timpul sau imediat după întrebările fixe
        if self.q <= len(_FIXED_QS):
            return False
        if self.q < MIN_QUESTIONS_BEFORE_STOP:
            return False
        recent = list(self.answers.values())[-4:]
        if (self.q > len(_FIXED_QS) + 2 and 
            len(recent) >= 4 and
            all(a in ("skip", "any") or a == ["any"] for a in recent)):
            return True
        if self.q >= MAX_QUESTIONS:
            return True

        entropy_ok = self.entropy() < ENTROPY_STOP_THRESHOLD

        p = self.probs()
        top_k_prob = float(p[p.argsort()[-10:]].sum())
        concentrated = top_k_prob > 0.25

        conditions_met = sum([entropy_ok, concentrated, self.is_top_stable()])
        return conditions_met >= 2

    def is_top_stable(self) -> bool:
        if len(self.top_history) < 2:
            return False
        current = set(self.top_history[-1])
        previous = set(self.top_history[-2])
        return len(current & previous) >= STABILITY_MIN_OVERLAP

    def expected_entropy_reduction(self, question: dict[str, Any]) -> float:
        p = self.probs()
        current_h = self.entropy()

        uniform = 1.0 / self.n
        relevant_idx = np.where(p > uniform * 0.01)[0]
        if len(relevant_idx) < 10:
            relevant_idx = p.argsort()[-100:]

        p_relevant = p[relevant_idx]
        p_relevant = p_relevant / p_relevant.sum()

        q_type = question["type"]
        if q_type == "categorical":
            possible_answers = question.get("options", []) + ["skip"]
        elif q_type == "multiselect":
            possible_answers = [[opt] for opt in question.get("options", [])] + [["any"]]
        else:
            possible_answers = ["yes", "no"]

        expected_h = 0.0
        w = self.weights.get(question["id"], 1.0)

        for answer in possible_answers:
            likelihoods = np.array([
                compute_likelihood(self.recipes[i], question, answer)
                for i in relevant_idx
            ])
            p_answer = float(np.dot(likelihoods, p_relevant))
            if p_answer < 1e-12:
                continue

            log_updated = np.log(p_relevant + 1e-10) + w * np.log(likelihoods + 1e-10)
            if POSTERIOR_TEMPERATURE > 1.0:
                log_updated /= POSTERIOR_TEMPERATURE
            log_updated -= log_updated.max()
            p_updated = np.exp(log_updated)
            p_updated /= p_updated.sum()

            p_nz = p_updated[p_updated > 1e-10]
            h_after = float(-np.sum(p_nz * np.log2(p_nz)))
            expected_h += p_answer * h_after

        return max(0.0, current_h - expected_h)

    def compute_answer_match_score(self, recipe: dict[str, Any]) -> float:
        total_weight = 0.0
        matched_weight = 0.0

        for qid, answer in self.answers.items():
            if answer == "skip" or answer == ["any"] or answer == "any":
                continue

            q = question_by_id(qid)
            if not q:
                continue

            if q["type"] == "boolean" and answer not in ("yes", "no"):
                continue

            w = self.weights.get(qid, 1.0)
            total_weight += w

            if q["type"] == "categorical":
                if recipe.get(q["feature"]) == answer:
                    matched_weight += w
            elif q["type"] == "multiselect":
                selected = answer if isinstance(answer, list) else [answer]
                if recipe.get(q["feature"]) in selected:
                    matched_weight += w
            elif q["type"] == "boolean":
                recipe_matches = get_feature_value_bool(recipe, q)
                if (answer == "yes" and recipe_matches) or (answer == "no" and not recipe_matches):
                    matched_weight += w

        if total_weight == 0:
            return 0.0
        return round((matched_weight / total_weight) * 100, 1)
    
    def compute_posterior_score(self, recipe_index: int) -> float:
        p = self.probs()
        uniform = 1.0 / self.n
        lift = float(p[recipe_index]) / uniform
        # k scales with the corpus size.
        k = math.log2(self.n)
        return round(100.0 * lift / (lift + k), 1)

    def top(self, n: int = TOP_N, min_match_score: float = 50.0) -> list[tuple[dict[str, Any], float]]:
        p = self.probs()

        # Hard filter: exclude recipes that fail categorical/multiselect answers
        hard_filter = []
        for i, recipe in enumerate(self.recipes):
            include = True
            for qid, answer in self.answers.items():
                if answer in ("skip", "no") or answer == ["any"] or answer == "any":
                    continue
                q = question_by_id(qid)
                if not q:
                    continue
                if q["type"] == "categorical":
                    if recipe.get(q["feature"]) != answer:
                        include = False
                        break
                elif q["type"] == "multiselect":
                    selected = answer if isinstance(answer, list) else [answer]
                    if recipe.get(q["feature"]) not in selected:
                        include = False
                        break
            if include:
                hard_filter.append(i)

        candidate_idx = hard_filter if len(hard_filter) >= 3 else list(range(self.n))
        candidate_idx.sort(key=lambda i: -p[i])

        scored = [
            (self.recipes[i], self.compute_posterior_score(i))
         for i in candidate_idx
        ]

        if min_match_score <= 0:
            return scored[:n]

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


# ── QUESTION SELECTION ────────────────────────────────────────────────────────
def select_next_question(session: BayesianSession) -> dict[str, Any] | None:
    excluded_ids = _excluded_question_ids(session.answers)

    fixed_candidates = [
        q for q in _FIXED_QS
        if q["id"] not in session.asked_ids and q["id"] not in excluded_ids
    ]
    if fixed_candidates:
        return fixed_candidates[0]

    candidates = [
        q for q in _ADAPTIVE_QS
        if q["id"] not in session.asked_ids and q["id"] not in excluded_ids
    ]
    if getattr(session, "semantic_query", None):
        balanced_candidates = [
            q for q in candidates
            if q["type"] != "boolean" or _semantic_pool_question_is_useful(session, q)
        ]
        if balanced_candidates:
            candidates = balanced_candidates

    if not candidates:
        return None

    # If 3 consecutive no/any answers, boost prevalence-near-35%
    recent = list(session.answers.values())[-3:]
    consecutive_no = sum(1 for a in recent if a == "no" or a == ["any"])

    if consecutive_no >= 3:
        p = session.probs()
        uniform = 1.0 / session.n
        rel_idx = np.where(p > uniform * 0.01)[0]
        rel_recipes = [session.recipes[i] for i in rel_idx]

        def prevalence_boosted_score(q: dict[str, Any]) -> float:
            eig = session.expected_entropy_reduction(q)
            if q["type"] != "boolean" or not rel_recipes:
                return eig
            count = sum(1 for r in rel_recipes if get_feature_value_bool(r, q))
            prev = count / len(rel_recipes)
            boost = max(1.0, 1.0 + 2.0 * (1.0 - abs(prev - 0.35) / 0.35))
            return eig * boost

        return max(candidates, key=prevalence_boosted_score)

    return max(candidates, key=lambda q: session.expected_entropy_reduction(q))


def _semantic_pool_question_is_useful(session: BayesianSession, question: dict[str, Any]) -> bool:
    p = session.probs()
    uniform = 1.0 / session.n
    relevant_idx = np.where(p > uniform * 0.01)[0]
    if len(relevant_idx) < 10:
        relevant_idx = p.argsort()[-min(100, session.n):]
    if len(relevant_idx) == 0:
        return True

    prevalence = sum(
        1 for idx in relevant_idx
        if get_feature_value_bool(session.recipes[int(idx)], question)
    ) / len(relevant_idx)
    return 0.10 < prevalence < 0.85


# ── SESSION RECONSTRUCTION ────────────────────────────────────────────────────
def restore_session(
    answers: dict[str, Any],
    recipes: list[dict[str, Any]],
    weights: dict[str, float],
) -> BayesianSession:
    """Replay saved answers to reconstruct Bayesian state (numpy arrays not serialisable)."""
    session = BayesianSession(recipes, weights)
    for qid, answer in answers.items():
        q = question_by_id(qid)
        if q:
            session.update(q, answer)
    return session
