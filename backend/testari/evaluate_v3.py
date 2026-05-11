from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from typing import Any

import numpy as np

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATASET_PATH = r"D:\folosire_api_claude\dataset_llm_labeled_with_ingredient_features.json"
MAX_QUESTIONS = 9
MIN_QUESTIONS_BEFORE_STOP = 3
TOP_N = 10

# Bayesian noise model
P_CORRECT = 0.90
P_NOISE = 0.05

# FIX #3: Entropy threshold calibrat dinamic în BayesianSession
# log2(8397) ≈ 13.0 biți inițial; vrem să oprim când incertitudinea scade sub ~log2(20)
ENTROPY_STOP_THRESHOLD = math.log2(20)  # ~4.32 biți (în loc de log2(50)=5.64)

# FIX #4: Scala pentru normalizarea semantic_boost față de info_gain tipic
# Va fi calibrată dinamic în select_next_question_bayesian
SEMANTIC_BOOST_SCALE = 0.10  # fracție din info_gain mediu

# ── QUESTION BANK ─────────────────────────────────────────────────────────────
QUESTION_BANK: list[dict[str, Any]] = [
    {
        "id": "meal_type",
        "type": "categorical",
        "feature": "meal_type",
        "fixed": True,
        "order": 1,
        "options": [
            "appetizer", "breakfast", "dessert", "drink", "lunch_dinner",
            "salad_side", "snack", "soup", "condiment"
        ],
    },

    # Protein
    {"id": "is_chicken",  "type": "boolean", "feature": "protein_type", "feature_value": "chicken"},
    {"id": "is_beef",     "type": "boolean", "feature": "protein_type", "feature_value": "beef_pork"},
    {"id": "is_fish",     "type": "boolean", "feature": "protein_type", "feature_value": "fish_seafood"},
    {"id": "is_meatless", "type": "boolean", "feature": "protein_type", "feature_value": "meatless"},

    # Cuisine
    {"id": "is_italian",       "type": "boolean", "feature": "cuisine", "feature_value": "italian"},
    {"id": "is_asian",         "type": "boolean", "feature": "cuisine", "feature_value": "asian"},
    {"id": "is_mexican",       "type": "boolean", "feature": "cuisine", "feature_value": "mexican"},
    {"id": "is_french",        "type": "boolean", "feature": "cuisine", "feature_value": "french"},
    {"id": "is_mediterranean", "type": "boolean", "feature": "cuisine", "feature_value": "mediterranean"},
    {"id": "is_indian",        "type": "boolean", "feature": "cuisine", "feature_value": "indian"},
    {"id": "is_american",      "type": "boolean", "feature": "cuisine", "feature_value": "american"},

    # Preference features
    {"id": "is_spicy",       "type": "boolean", "feature": "is_spicy",       "feature_value": True},
    {"id": "is_sweet",       "type": "boolean", "feature": "is_sweet",       "feature_value": True},
    {"id": "is_quick",       "type": "boolean", "feature": "is_quick",       "feature_value": True},
    {"id": "needs_oven",     "type": "boolean", "feature": "needs_oven",     "feature_value": True},
    {"id": "needs_stovetop", "type": "boolean", "feature": "needs_stovetop", "feature_value": True},
    {"id": "is_no_cook",     "type": "boolean", "feature": "is_no_cook",     "feature_value": True},

    # Ingredient features
    {"id": "has_pasta",            "type": "boolean", "feature": "has_pasta",            "feature_value": True},
    {"id": "has_rice",             "type": "boolean", "feature": "has_rice",             "feature_value": True},
    {"id": "has_potato",           "type": "boolean", "feature": "has_potato",           "feature_value": True},
    {"id": "has_tomato_base",      "type": "boolean", "feature": "has_tomato_base",      "feature_value": True},
    {"id": "has_cream_base",       "type": "boolean", "feature": "has_cream_base",       "feature_value": True},
    {"id": "has_cheese",           "type": "boolean", "feature": "has_cheese",           "feature_value": True},
    {"id": "has_broth_base",       "type": "boolean", "feature": "has_broth_base",       "feature_value": True},
    {"id": "has_mushroom",         "type": "boolean", "feature": "has_mushroom",         "feature_value": True},
    {"id": "has_leafy_greens",     "type": "boolean", "feature": "has_leafy_greens",     "feature_value": True},
    {"id": "has_beans_legumes",    "type": "boolean", "feature": "has_beans_legumes",    "feature_value": True},
    {"id": "has_fruit",            "type": "boolean", "feature": "has_fruit",            "feature_value": True},
    {"id": "has_nuts",             "type": "boolean", "feature": "has_nuts",             "feature_value": True},
    {"id": "has_chocolate",        "type": "boolean", "feature": "has_chocolate",        "feature_value": True},
    {"id": "has_tortilla",         "type": "boolean", "feature": "has_tortilla",         "feature_value": True},
    {"id": "has_spicy_ingredient", "type": "boolean", "feature": "has_spicy_ingredient", "feature_value": True},
    {"id": "has_asian_sauce",      "type": "boolean", "feature": "has_asian_sauce",      "feature_value": True},
]

_FIXED_QS = sorted(
    [q for q in QUESTION_BANK if q.get("fixed")],
    key=lambda q: q.get("order", 99)
)
_ADAPTIVE_QS = [q for q in QUESTION_BANK if not q.get("fixed")]


FEATURE_WEIGHTS = {
    "meal_type":   3.00,
    "is_chicken":  2.05,
    "is_beef":     2.05,
    "is_fish":     2.05,
    "is_meatless": 2.05,

    "is_italian":       1.15,
    "is_asian":         1.15,
    "is_mexican":       1.15,
    "is_french":        1.15,
    "is_mediterranean": 1.15,
    "is_indian":        1.15,
    "is_american":      1.15,

    "is_sweet":      1.40,
    "needs_oven":    1.00,
    "is_spicy":      0.95,
    "is_no_cook":    0.95,
    "needs_stovetop":0.95,
    "is_quick":      0.80,

    "has_asian_sauce":  1.05,
    "has_tomato_base":  1.00,
    "has_cheese":       0.95,
    "has_broth_base":   0.90,
    "has_spicy_ingredient": 0.90,

    "has_pasta":   1.20,
    "has_rice":    1.10,
    "has_tortilla":1.05,

    "has_fruit":        0.75,
    "has_potato":       0.75,
    "has_nuts":         0.75,
    "has_cream_base":   0.80,
    "has_chocolate":    0.85,
    "has_mushroom":     0.70,
    "has_beans_legumes":0.70,
    "has_leafy_greens": 0.70,
}

# ── SYNTHETIC USERS ───────────────────────────────────────────────────────────
SYNTHETIC_USERS = [
    {"name": "Dessert american",        "meal_type": "dessert",      "cuisine": "american"},
    {"name": "Soup chicken",            "meal_type": "soup",         "protein_type": "chicken"},
    {"name": "Breakfast sweet",         "meal_type": "breakfast",    "is_sweet": True},
    {"name": "Lunch meatless",          "meal_type": "lunch_dinner", "protein_type": "meatless"},
    {"name": "Snack quick",             "meal_type": "snack",        "is_quick": True},

    {"name": "Italian pasta lunch",     "meal_type": "lunch_dinner", "cuisine": "italian",        "has_pasta": True},
    {"name": "Asian chicken spicy",     "meal_type": "lunch_dinner", "protein_type": "chicken",   "cuisine": "asian",    "is_spicy": True},
    {"name": "Mexican beef tortilla",   "meal_type": "lunch_dinner", "protein_type": "beef_pork", "has_tortilla": True},
    {"name": "Chocolate dessert oven",  "meal_type": "dessert",      "has_chocolate": True,        "needs_oven": True},
    {"name": "Creamy soup meatless",    "meal_type": "soup",         "protein_type": "meatless",  "has_cream_base": True},
    {"name": "Asian rice fish",         "meal_type": "lunch_dinner", "protein_type": "fish_seafood","has_rice": True},
    {"name": "Mediterranean salad",     "meal_type": "salad_side",   "cuisine": "mediterranean",  "is_no_cook": True},
    {"name": "Quick beef american",     "meal_type": "lunch_dinner", "protein_type": "beef_pork", "cuisine": "american", "is_quick": True},
    {"name": "Indian spicy meatless",   "meal_type": "lunch_dinner", "protein_type": "meatless",  "cuisine": "indian",   "is_spicy": True},
    {"name": "Breakfast oven sweet",    "meal_type": "breakfast",    "needs_oven": True,           "is_sweet": True},

    {"name": "Italian chicken pasta stovetop", "meal_type": "lunch_dinner", "protein_type": "chicken",   "cuisine": "italian", "has_pasta": True},
    {"name": "Asian spicy soy stovetop",       "meal_type": "lunch_dinner", "cuisine": "asian",           "is_spicy": True,     "has_asian_sauce": True},
    {"name": "Mexican tortilla cheese beef",   "meal_type": "lunch_dinner", "protein_type": "beef_pork",  "cuisine": "mexican", "has_cheese": True},
    {"name": "Chocolate dessert sweet oven",   "meal_type": "dessert",      "is_sweet": True,             "has_chocolate": True,"needs_oven": True},
    {"name": "Creamy mushroom soup stovetop",  "meal_type": "soup",         "protein_type": "meatless",   "has_mushroom": True, "has_cream_base": True},
]

# ── HELPERS ───────────────────────────────────────────────────────────────────
def get_feature_value(recipe: dict[str, Any], question: dict[str, Any]) -> Any:
    feature = question["feature"]
    feature_value = question.get("feature_value")
    recipe_val = recipe.get(feature)

    if feature_value is None:
        return recipe_val
    if isinstance(feature_value, bool):
        return bool(recipe_val) == feature_value
    return recipe_val == feature_value


def recipe_matches_user(recipe: dict[str, Any], user_profile: dict[str, Any]) -> bool:
    for key, val in user_profile.items():
        if key == "name" or val is None:
            continue

        recipe_val = recipe.get(key)

        if key in {"meal_type", "protein_type", "cuisine"}:
            if recipe_val != val:
                return False
        elif isinstance(val, bool):
            if val is True and bool(recipe_val) is not True:
                return False
        else:
            if recipe_val != val:
                return False

    return True


def simulate_answer(user_profile: dict[str, Any], question: dict[str, Any]) -> list[str]:
    q_type = question["type"]
    feature = question["feature"]
    feature_value = question.get("feature_value")

    if q_type == "categorical":
        if feature not in user_profile:
            return ["skip"]
        user_val = user_profile[feature]
        if user_val is None:
            return ["skip"]
        return [user_val]

    if feature not in user_profile:
        user_val = False
    else:
        user_val = user_profile[feature]

    if user_val is None:
        return ["skip"]

    if feature_value is None:
        return ["yes"] if bool(user_val) else ["no"]
    if isinstance(feature_value, bool):
        return ["yes"] if bool(user_val) == feature_value else ["no"]
    return ["yes"] if user_val == feature_value else ["no"]


def question_by_id(qid: str) -> dict[str, Any] | None:
    return next((q for q in QUESTION_BANK if q["id"] == qid), None)


# ── METRICS ───────────────────────────────────────────────────────────────────
def compute_ndcg_correct(
    ranked: list[tuple[dict[str, Any], float]],
    all_recipes: list[dict[str, Any]],
    user_profile: dict[str, Any],
    k: int = 10
) -> float:
    relevances = [
        1.0 if recipe_matches_user(recipe, user_profile) else 0.0
        for recipe, _ in ranked[:k]
    ]
    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))
    total_relevant = sum(1 for recipe in all_recipes if recipe_matches_user(recipe, user_profile))
    ideal_relevances = [1.0] * min(k, total_relevant)
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_relevances))
    return dcg / idcg if idcg > 0 else 0.0


def compute_overlap(
    ranked: list[tuple[dict[str, Any], float]],
    user_profile: dict[str, Any],
    k: int = 10
) -> float:
    top = ranked[:k]
    if not top:
        return 0.0
    matches = sum(1 for recipe, _ in top if recipe_matches_user(recipe, user_profile))
    return matches / min(k, len(top))


# ── MARGINAL / WEIGHTED SCORING ───────────────────────────────────────────────
def score_recipes_weighted(
    recipes: list[dict[str, Any]],
    answers: dict[str, str]
) -> dict[int, float]:
    answered_max = sum(
        FEATURE_WEIGHTS.get(qid, 0.0)
        for qid, answer in answers.items()
        if answer != "skip" and question_by_id(qid)
    )

    if answered_max <= 0:
        return {recipe["id"]: 0.0 for recipe in recipes}

    scores: dict[int, float] = {}

    for recipe in recipes:
        raw = 0.0
        for qid, answer in answers.items():
            if answer == "skip":
                continue
            q = question_by_id(qid)
            if not q:
                continue
            weight = FEATURE_WEIGHTS.get(qid, 0.0)

            if q["type"] == "categorical":
                if recipe.get(q["feature"]) == answer:
                    raw += weight
                continue

            match = get_feature_value(recipe, q)

            if answer == "yes" and match:
                raw += weight
            elif answer == "no" and match:
                # FIX #5: penalizare mai puternică pentru "no" — weight întreg în loc de weight/2
                raw -= weight

        scores[recipe["id"]] = round(max(0.0, raw) / answered_max * 100.0, 4)

    return scores


def top_n_weighted(
    scores: dict[int, float],
    recipes_by_id: dict[int, dict[str, Any]],
    n: int = 10
) -> list[tuple[dict[str, Any], float]]:
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [(recipes_by_id[rid], sc) for rid, sc in ranked[:n] if rid in recipes_by_id]


def marginal_entropy(question: dict[str, Any], recipes: list[dict[str, Any]]) -> float:
    if not recipes:
        return 0.0
    values = [str(get_feature_value(recipe, question)) for recipe in recipes]
    counts = Counter(values)
    total = len(values)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def should_stop_entropy(
    q_num: int,
    candidates: list[dict[str, Any]],
    recipes: list[dict[str, Any]],
    scores: dict[int, float],
    entropy_threshold: float = 0.5
) -> bool:
    if q_num >= MAX_QUESTIONS:
        return True
    if q_num < MIN_QUESTIONS_BEFORE_STOP:
        return False
    if not candidates:
        return True

    if scores:
        sorted_scores = sorted(scores.values(), reverse=True)
        threshold_index = max(0, len(sorted_scores) // 3)
        threshold = sorted_scores[threshold_index]
        relevant_pool = [
            recipe for recipe in recipes
            if scores.get(recipe["id"], 0.0) >= threshold
        ]
    else:
        relevant_pool = recipes

    if not relevant_pool:
        relevant_pool = recipes

    max_entropy = max(marginal_entropy(q, relevant_pool) for q in candidates)
    return max_entropy < entropy_threshold


# ── JMIM — FIX #2: calculat pe pool-ul filtrat (top 33%) ─────────────────────
def discretize_scores(scores: dict[int, float], recipes: list[dict[str, Any]]) -> list[str]:
    labels = []
    for recipe in recipes:
        score = scores.get(recipe["id"], 0.0)
        if score > 66:
            labels.append("high")
        elif score > 33:
            labels.append("mid")
        else:
            labels.append("low")
    return labels


def joint_mi(xi: list[str], xs: list[str], y: list[str]) -> float:
    n = len(y)
    if n == 0:
        return 0.0
    j3 = Counter(zip(xi, xs, y))
    jxy = Counter(zip(xi, xs))
    cy = Counter(y)
    mi = 0.0
    for (a, b, c), count in j3.items():
        p_joint = count / n
        p_x = jxy[(a, b)] / n
        p_y = cy[c] / n
        if p_joint > 0 and p_x > 0 and p_y > 0:
            mi += p_joint * math.log2(p_joint / (p_x * p_y))
    return mi


def jmim_score(
    question: dict[str, Any],
    recipes: list[dict[str, Any]],
    asked_ids: set[str],
    scores: dict[int, float],
    answers: dict[str, str],
    smoothing: float = 1e-4
) -> float:
    # FIX #2: filtrăm la pool-ul relevant (top 33%) înainte de JMIM
    if scores:
        sorted_scores = sorted(scores.values(), reverse=True)
        threshold = sorted_scores[max(0, len(sorted_scores) // 3)]
        relevant_pool = [r for r in recipes if scores.get(r["id"], 0.0) >= threshold] or recipes
    else:
        relevant_pool = recipes

    valid_asked_ids = [
        qid for qid in asked_ids
        if answers.get(qid) != "skip"
    ]

    if not valid_asked_ids:
        return marginal_entropy(question, relevant_pool)

    xi = [str(get_feature_value(recipe, question)) for recipe in relevant_pool]
    y = discretize_scores(scores, relevant_pool)

    min_jmi = float("inf")
    for asked_qid in valid_asked_ids:
        asked_q = question_by_id(asked_qid)
        if not asked_q:
            continue
        xs = [str(get_feature_value(recipe, asked_q)) for recipe in relevant_pool]
        value = max(joint_mi(xi, xs, y), smoothing)
        min_jmi = min(min_jmi, value)

    if min_jmi == float("inf") or min_jmi < smoothing * 10:
        return marginal_entropy(question, relevant_pool)

    return min_jmi


# ── BAYESIAN ──────────────────────────────────────────────────────────────────
def compute_likelihood(
    recipe: dict[str, Any],
    question: dict[str, Any],
    answer: str
) -> float:
    if answer == "skip":
        return 1.0
    if question["type"] == "categorical":
        recipe_val = recipe.get(question["feature"])
        return P_CORRECT if recipe_val == answer else P_NOISE
    match = get_feature_value(recipe, question)
    if answer == "yes":
        return P_CORRECT if match else P_NOISE
    if answer == "no":
        return P_CORRECT if not match else P_NOISE
    return 1.0


class BayesianSession:
    def __init__(self, recipes: list[dict[str, Any]]) -> None:
        self.recipes = recipes
        self.n = len(recipes)
        self.log_probs = np.full(self.n, -math.log(self.n))
        self.answers: dict[str, str] = {}
        self.asked_ids: set[str] = set()
        self.q = 0

    def probs(self) -> np.ndarray:
        log_p = self.log_probs - self.log_probs.max()
        p = np.exp(log_p)
        return p / p.sum()

    def update(self, question: dict[str, Any], answer: str) -> None:
        self.q += 1
        self.answers[question["id"]] = answer
        self.asked_ids.add(question["id"])

        if answer == "skip":
            return

        likelihoods = np.array([
            compute_likelihood(recipe, question, answer)
            for recipe in self.recipes
        ])
        self.log_probs += np.log(likelihoods + 1e-10)
        self.log_probs -= self.log_probs.max()

    def entropy(self) -> float:
        p = self.probs()
        p_nonzero = p[p > 1e-10]
        return float(-np.sum(p_nonzero * np.log2(p_nonzero)))

    def should_stop(self) -> bool:
        if self.q < MIN_QUESTIONS_BEFORE_STOP:
            return False
        if self.q >= MAX_QUESTIONS:
            return True
        # FIX #3: prag calibrat la log2(20) ≈ 4.32 în loc de log2(50)
        return self.entropy() < ENTROPY_STOP_THRESHOLD

    def expected_entropy_reduction(self, question: dict[str, Any]) -> float:
        current_h = self.entropy()
        p = self.probs()

        if question["type"] == "categorical":
            possible_answers = question.get("options", [])
        else:
            possible_answers = ["yes", "no", "skip"]

        expected_h = 0.0
        for answer in possible_answers:
            likelihoods = np.array([
                compute_likelihood(recipe, question, answer)
                for recipe in self.recipes
            ])
            p_answer = float(np.dot(likelihoods, p))
            if p_answer < 1e-12:
                continue

            log_updated = self.log_probs + np.log(likelihoods + 1e-10)
            log_updated -= log_updated.max()
            p_updated = np.exp(log_updated)
            p_updated /= p_updated.sum()

            p_nonzero = p_updated[p_updated > 1e-10]
            h_after = float(-np.sum(p_nonzero * np.log2(p_nonzero)))
            expected_h += p_answer * h_after

        return current_h - expected_h

    # FIX #5 (Bayesian): scorul relativ normalizat la max, nu 100 fix
    def top(self, n: int = 10) -> list[tuple[dict[str, Any], float]]:
        p = self.probs()
        idx = p.argsort()[-n:][::-1]
        if len(idx) == 0:
            return []
        # Returnează probabilitățile absolute (nu normate la max=100)
        # Scalăm la [0,100] față de probabilitatea uniformă de referință
        uniform_p = 1.0 / self.n
        return [
            (self.recipes[i], round(float(p[i] / uniform_p), 2))
            for i in idx
        ]


# FIX #4: semantic_boost calibrat față de info_gain real
_SEMANTIC_BOOST_RAW = {
    "has_pasta":            0.35,
    "has_rice":             0.30,
    "has_potato":           0.25,
    "has_tomato_base":      0.35,
    "has_cream_base":       0.35,
    "has_cheese":           0.30,
    "has_broth_base":       0.25,
    "has_mushroom":         0.20,
    "has_leafy_greens":     0.20,
    "has_beans_legumes":    0.20,
    "has_fruit":            0.25,
    "has_nuts":             0.20,
    "has_chocolate":        0.25,
    "has_tortilla":         0.30,
    "has_spicy_ingredient": 0.25,
    "has_asian_sauce":      0.30,
}
# Normalizăm raw boost-urile la [0, 1] față de max raw value
_MAX_RAW_BOOST = max(_SEMANTIC_BOOST_RAW.values())
_SEMANTIC_BOOST_NORMALIZED = {
    k: v / _MAX_RAW_BOOST for k, v in _SEMANTIC_BOOST_RAW.items()
}


def question_priority_score(session: BayesianSession, question: dict[str, Any], avg_info_gain: float) -> float:
    info_gain = session.expected_entropy_reduction(question)
    qid = question["id"]

    # FIX #4: boost-ul semantic e o fracție din info_gain mediu, nu o constantă absolută
    normalized_boost = _SEMANTIC_BOOST_NORMALIZED.get(qid, 0.0)
    semantic_boost = normalized_boost * avg_info_gain * SEMANTIC_BOOST_SCALE

    return info_gain + semantic_boost


def select_next_question_bayesian(session: BayesianSession) -> dict[str, Any] | None:
    if session.q < len(_FIXED_QS):
        return _FIXED_QS[session.q]

    candidates = [q for q in _ADAPTIVE_QS if q["id"] not in session.asked_ids]
    if not candidates:
        return None

    # FIX #4: calculăm info_gain mediu pe un sample de candidați pentru calibrare
    sample_gains = [session.expected_entropy_reduction(q) for q in candidates[:10]]
    avg_info_gain = sum(sample_gains) / len(sample_gains) if sample_gains else 1.0

    return max(candidates, key=lambda q: question_priority_score(session, q, avg_info_gain))


# ── SESSION RUNNERS ───────────────────────────────────────────────────────────
def count_relevant_recipes(recipes: list[dict[str, Any]], user_profile: dict[str, Any]) -> int:
    return sum(1 for recipe in recipes if recipe_matches_user(recipe, user_profile))


def run_session(
    recipes: list[dict[str, Any]],
    user_profile: dict[str, Any],
    mode: str,
    verbose: bool = False
) -> dict[str, Any]:

    if mode == "bayesian":
        session = BayesianSession(recipes)
        question_order: list[str] = []

        while not session.should_stop():
            question = select_next_question_bayesian(session)
            if question is None:
                break

            answer = simulate_answer(user_profile, question)[0]
            session.update(question, answer)
            question_order.append(question["id"])

            if verbose:
                print(f"      Q{session.q}: {question['id']} -> {answer} | H={session.entropy():.4f}")

        ranked = session.top(TOP_N)

        return {
            "questions_asked": session.q,
            "question_order": question_order,
            "total_relevant": count_relevant_recipes(recipes, user_profile),
            "ndcg_10": round(compute_ndcg_correct(ranked, recipes, user_profile, TOP_N), 4),
            "overlap_10": round(compute_overlap(ranked, user_profile, TOP_N), 4),
            # FIX #5: AvgScore acum e relativ la probabilitate uniformă (>1 = mai bun ca random)
            "avg_top10_score": round(sum(score for _, score in ranked) / len(ranked) if ranked else 0.0, 2),
            "entropy": round(session.entropy(), 4),
            "top": [
                {
                    "name": recipe.get("name", ""),
                    "score": score,
                    "meal_type": recipe.get("meal_type"),
                    "protein_type": recipe.get("protein_type"),
                    "cuisine": recipe.get("cuisine"),
                }
                for recipe, score in ranked
            ],
        }

    # Marginal / JMIM
    recipes_by_id = {recipe["id"]: recipe for recipe in recipes}
    answers: dict[str, str] = {}
    asked_ids: set[str] = set()
    question_order: list[str] = []
    q_num = 0

    while True:
        scores = score_recipes_weighted(recipes, answers)

        if q_num < len(_FIXED_QS):
            question = _FIXED_QS[q_num]
        else:
            candidates = [q for q in _ADAPTIVE_QS if q["id"] not in asked_ids]
            if not candidates:
                break

            if mode == "jmim":
                # FIX #2: JMIM acum filtrează intern la top 33%
                question = max(
                    candidates,
                    key=lambda q: jmim_score(q, recipes, asked_ids, scores, answers)
                )
            else:
                sorted_scores = sorted(scores.values(), reverse=True)
                threshold = sorted_scores[max(0, len(sorted_scores) // 3)] if sorted_scores else 0.0
                relevant_pool = [
                    recipe for recipe in recipes
                    if scores.get(recipe["id"], 0.0) >= threshold
                ] or recipes
                question = max(candidates, key=lambda q: marginal_entropy(q, relevant_pool))

        answer = simulate_answer(user_profile, question)[0]
        answers[question["id"]] = answer
        asked_ids.add(question["id"])
        question_order.append(question["id"])
        q_num += 1

        if verbose:
            print(f"      Q{q_num}: {question['id']} -> {answer}")

        scores = score_recipes_weighted(recipes, answers)
        remaining = [q for q in _ADAPTIVE_QS if q["id"] not in asked_ids]

        if should_stop_entropy(q_num, remaining, recipes, scores):
            break

    final_scores = score_recipes_weighted(recipes, answers)
    ranked = top_n_weighted(final_scores, recipes_by_id, TOP_N)

    return {
        "questions_asked": len(question_order),
        "question_order": question_order,
        "total_relevant": count_relevant_recipes(recipes, user_profile),
        "ndcg_10": round(compute_ndcg_correct(ranked, recipes, user_profile, TOP_N), 4),
        "overlap_10": round(compute_overlap(ranked, user_profile, TOP_N), 4),
        "avg_top10_score": round(sum(score for _, score in ranked) / len(ranked) if ranked else 0.0, 2),
        "entropy": None,
        "top": [
            {
                "name": recipe.get("name", ""),
                "score": score,
                "meal_type": recipe.get("meal_type"),
                "protein_type": recipe.get("protein_type"),
                "cuisine": recipe.get("cuisine"),
            }
            for recipe, score in ranked
        ],
    }


# ── DATA LOADING ──────────────────────────────────────────────────────────────
def load_recipes(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as file:
        raw = json.load(file)

    recipes: list[dict[str, Any]] = []
    expected_features = {
        "meal_type", "protein_type", "cuisine",
        "is_spicy", "is_sweet", "is_quick", "needs_oven", "needs_stovetop", "is_no_cook",
        "has_pasta", "has_rice", "has_potato", "has_tomato_base", "has_cream_base",
        "has_cheese", "has_broth_base", "has_mushroom", "has_leafy_greens",
        "has_beans_legumes", "has_fruit", "has_nuts", "has_chocolate",
        "has_tortilla", "has_spicy_ingredient", "has_asian_sauce",
    }

    for idx, entry in enumerate(raw):
        features = entry.get("llm_features")
        if not features or entry.get("llm_failed"):
            continue

        recipe = {"id": idx, "name": entry.get("Name", "")}
        for feature in expected_features:
            recipe[feature] = features.get(feature, False)

        recipes.append(recipe)

    return recipes


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["marginal", "jmim", "bayesian", "both", "all"],
        default="all"
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--show-top", action="store_true")
    parser.add_argument("--sample", type=int, default=0)
    args = parser.parse_args()

    print(f"Încarc dataset din {DATASET_PATH}...")
    recipes = load_recipes(DATASET_PATH)

    if args.sample > 0:
        recipes = recipes[:args.sample]

    modes = {
        "both": ["marginal", "jmim"],
        "all":  ["marginal", "jmim", "bayesian"],
    }.get(args.mode, [args.mode])

    print(f"Rețete: {len(recipes)} | Utilizatori sintetici: {len(SYNTHETIC_USERS)}")
    print(f"Max întrebări: {MAX_QUESTIONS} | Entropy stop: {ENTROPY_STOP_THRESHOLD:.3f} biți")
    print()

    all_results: dict[str, list[dict[str, Any]]] = {}

    for mode in modes:
        print("=" * 60)
        print(f"MOD: {mode.upper()}")
        print("=" * 60)

        results: list[dict[str, Any]] = []

        for user in SYNTHETIC_USERS:
            if args.verbose:
                print(f"  {user['name']}:")

            result = run_session(recipes, user, mode, verbose=args.verbose)
            result["user"] = user["name"]
            results.append(result)

            print(
                f"  {user['name'][:35]:<35} | "
                f"Q: {result['questions_asked']:>2} | "
                f"Rel: {result['total_relevant']:>4} | "
                f"NDCG: {result['ndcg_10']:.3f} | "
                f"Overlap: {result['overlap_10']:.2f} | "
                f"AvgScore: {result['avg_top10_score']:.1f}x"
            )

            if args.show_top:
                print("      Top:")
                for item in result["top"][:5]:
                    print(f"        - {item['name']} | {item['score']:.1f}x")

        all_results[mode] = results

        avg_q       = sum(r["questions_asked"] for r in results) / len(results)
        avg_ndcg    = sum(r["ndcg_10"]         for r in results) / len(results)
        avg_overlap = sum(r["overlap_10"]       for r in results) / len(results)
        avg_score   = sum(r["avg_top10_score"]  for r in results) / len(results)

        print(
            f"\n  MEDII: Q={avg_q:.2f} | "
            f"NDCG={avg_ndcg:.4f} | "
            f"Overlap={avg_overlap:.4f} | "
            f"AvgScore={avg_score:.1f}x\n"
        )

    if len(modes) > 1:
        print("=" * 60)
        print("COMPARAȚIE FINALĂ")
        print("=" * 60)

        baseline_mode = modes[0]
        baseline = all_results[baseline_mode]

        for mode in modes[1:]:
            other = all_results[mode]
            print(f"\n  {baseline_mode.upper()} vs {mode.upper()}:")

            for metric, label in [
                ("questions_asked", "Nr. întrebări"),
                ("ndcg_10",         "NDCG@10"),
                ("overlap_10",      "Overlap@10"),
                ("avg_top10_score", "Avg Score (x uniform)"),
            ]:
                b_avg = sum(r[metric] for r in baseline) / len(baseline)
                o_avg = sum(r[metric] for r in other)   / len(other)
                diff  = o_avg - b_avg

                better = (metric != "questions_asked" and diff > 0) or (
                    metric == "questions_asked" and diff < 0
                )
                direction = "↑ mai bun" if better else ("↓ mai slab" if diff != 0 else "= egal")

                print(
                    f"    {label:<22} "
                    f"{baseline_mode}={b_avg:.4f} | "
                    f"{mode}={o_avg:.4f} | "
                    f"Δ={diff:+.4f} {direction}"
                )

    output_path = "evaluation_results_v2.json"
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(all_results, file, ensure_ascii=False, indent=2)

    print(f"\nRezultate salvate în {output_path}")


if __name__ == "__main__":
    main()