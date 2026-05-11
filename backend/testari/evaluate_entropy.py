"""
Evaluare comparativa: entropie marginala vs JMIM vs Bayesian.

Usage:
    python evaluate_entropy.py --mode marginal
    python evaluate_entropy.py --mode jmim
    python evaluate_entropy.py --mode bayesian
    python evaluate_entropy.py --mode all
    python evaluate_entropy.py --mode all --verbose
"""
from __future__ import annotations
import argparse
import json
import math
import numpy as np
from collections import Counter
from typing import Any

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATASET_PATH = r"D:\folosire_api_claude\dataset_llm_labeled.json"
MAX_QUESTIONS = 9
MIN_QUESTIONS_BEFORE_STOP = 3
TOP_N = 10

# Bayesian noise model
P_CORRECT = 0.90
P_NOISE   = 0.05
ENTROPY_STOP_THRESHOLD = math.log2(50)  # ~5.64 biti

# ── QUESTION BANK ─────────────────────────────────────────────────────────────
QUESTION_BANK: list[dict[str, Any]] = [
    {
        "id": "meal_type", "type": "categorical", "feature": "meal_type",
        "fixed": True, "order": 1,
        "options": ["appetizer","breakfast","dessert","drink","lunch_dinner","salad_side","snack","soup","condiment"],
    },
    {"id": "is_chicken",       "type": "boolean", "feature": "protein_type",   "feature_value": "chicken",      "fixed": False, "options": ["yes","no","skip"]},
    {"id": "is_beef_pork",     "type": "boolean", "feature": "protein_type",   "feature_value": "beef_pork",    "fixed": False, "options": ["yes","no","skip"]},
    {"id": "is_fish",          "type": "boolean", "feature": "protein_type",   "feature_value": "fish_seafood", "fixed": False, "options": ["yes","no","skip"]},
    {"id": "is_meatless",      "type": "boolean", "feature": "protein_type",   "feature_value": "meatless",     "fixed": False, "options": ["yes","no","skip"]},
    {"id": "is_italian",       "type": "boolean", "feature": "cuisine",        "feature_value": "italian",      "fixed": False, "options": ["yes","no","skip"]},
    {"id": "is_asian",         "type": "boolean", "feature": "cuisine",        "feature_value": "asian",        "fixed": False, "options": ["yes","no","skip"]},
    {"id": "is_mexican",       "type": "boolean", "feature": "cuisine",        "feature_value": "mexican",      "fixed": False, "options": ["yes","no","skip"]},
    {"id": "is_french",        "type": "boolean", "feature": "cuisine",        "feature_value": "french",       "fixed": False, "options": ["yes","no","skip"]},
    {"id": "is_mediterranean", "type": "boolean", "feature": "cuisine",        "feature_value": "mediterranean","fixed": False, "options": ["yes","no","skip"]},
    {"id": "is_indian",        "type": "boolean", "feature": "cuisine",        "feature_value": "indian",       "fixed": False, "options": ["yes","no","skip"]},
    {"id": "is_american",      "type": "boolean", "feature": "cuisine",        "feature_value": "american",     "fixed": False, "options": ["yes","no","skip"]},
    {"id": "is_spicy",         "type": "boolean", "feature": "is_spicy",       "feature_value": True,           "fixed": False, "options": ["yes","no","skip"]},
    {"id": "is_sweet",         "type": "boolean", "feature": "is_sweet",       "feature_value": True,           "fixed": False, "options": ["yes","no","skip"]},
    {"id": "is_quick",         "type": "boolean", "feature": "is_quick",       "feature_value": True,           "fixed": False, "options": ["yes","no","skip"]},
    {"id": "needs_oven",       "type": "boolean", "feature": "needs_oven",     "feature_value": True,           "fixed": False, "options": ["yes","no","skip"]},
    {"id": "needs_stovetop",   "type": "boolean", "feature": "needs_stovetop", "feature_value": True,           "fixed": False, "options": ["yes","no","skip"]},
    {"id": "is_no_cook",       "type": "boolean", "feature": "is_no_cook",     "feature_value": True,           "fixed": False, "options": ["yes","no","skip"]},
]

FEATURE_WEIGHTS: dict[str, float] = {
    "meal_type": 3.00, "is_sweet": 3.00,
    "is_chicken": 2.05, "is_beef_pork": 2.05, "is_fish": 2.05, "is_meatless": 2.05,
    "needs_oven": 1.01,
    "is_italian": 0.98, "is_asian": 0.98, "is_mexican": 0.98, "is_french": 0.98,
    "is_mediterranean": 0.98, "is_indian": 0.98, "is_american": 0.98,
    "is_no_cook": 0.89, "needs_stovetop": 0.74, "is_quick": 0.51, "is_spicy": 0.47,
}

_FIXED_QS   = sorted([q for q in QUESTION_BANK if q.get("fixed")],  key=lambda q: q.get("order", 99))
_ADAPTIVE_QS = [q for q in QUESTION_BANK if not q.get("fixed")]

# ── UTILIZATORI SINTETICI ─────────────────────────────────────────────────────
SYNTHETIC_USERS = [
    {"name": "Desert lover",           "meal_type": "dessert",      "is_sweet": True,  "cuisine": "american"},
    {"name": "Asian spicy chicken",    "meal_type": "lunch_dinner", "protein_type": "chicken",    "cuisine": "asian",          "is_spicy": True},
    {"name": "Quick veggie breakfast", "meal_type": "breakfast",    "protein_type": "meatless",   "is_quick": True},
    {"name": "Italian beef pasta",     "meal_type": "lunch_dinner", "protein_type": "beef_pork",  "cuisine": "italian"},
    {"name": "Mediterranean soup",     "meal_type": "soup",         "protein_type": "meatless",   "cuisine": "mediterranean"},
    {"name": "Mexican spicy snack",    "meal_type": "snack",        "is_spicy": True,  "cuisine": "mexican"},
    {"name": "French dessert",         "meal_type": "dessert",      "cuisine": "french",           "is_sweet": True},
    {"name": "Quick fish dinner",      "meal_type": "lunch_dinner", "protein_type": "fish_seafood","is_quick": True},
    {"name": "Vegan no-cook salad",    "meal_type": "salad_side",   "protein_type": "meatless",   "is_no_cook": True},
    {"name": "Oven beef american",     "meal_type": "lunch_dinner", "protein_type": "beef_pork",  "needs_oven": True,  "cuisine": "american"},
    {"name": "Indian vegetarian",      "meal_type": "lunch_dinner", "protein_type": "meatless",   "cuisine": "indian"},
    {"name": "Sweet breakfast quick",  "meal_type": "breakfast",    "is_sweet": True,  "is_quick": True},
    {"name": "Asian no-cook drink",    "meal_type": "drink",        "is_no_cook": True,"cuisine": "asian"},
    {"name": "Stovetop chicken soup",  "meal_type": "soup",         "protein_type": "chicken",    "needs_stovetop": True},
    {"name": "Spicy beef mexican",     "meal_type": "lunch_dinner", "protein_type": "beef_pork",  "cuisine": "mexican",        "is_spicy": True},
    {"name": "American sweet dessert", "meal_type": "dessert",      "cuisine": "american",         "is_sweet": True},
    {"name": "Quick meatless lunch",   "meal_type": "lunch_dinner", "protein_type": "meatless",   "is_quick": True},
    {"name": "Italian fish oven",      "meal_type": "lunch_dinner", "protein_type": "fish_seafood","cuisine": "italian",        "needs_oven": True},
    {"name": "Mediterranean salad",    "meal_type": "salad_side",   "cuisine": "mediterranean",   "is_no_cook": True},
    {"name": "Asian spicy stovetop",   "meal_type": "lunch_dinner", "cuisine": "asian",            "is_spicy": True,            "needs_stovetop": True},
    {"name": "French chicken oven",    "meal_type": "lunch_dinner", "protein_type": "chicken",    "cuisine": "french",         "needs_oven": True},
    {"name": "Indian spicy meatless",  "meal_type": "lunch_dinner", "protein_type": "meatless",   "cuisine": "indian",         "is_spicy": True},
    {"name": "American beef sweet",    "meal_type": "lunch_dinner", "protein_type": "beef_pork",  "cuisine": "american",       "is_sweet": True},
    {"name": "Quick asian drink",      "meal_type": "drink",        "is_quick": True,  "cuisine": "asian"},
    {"name": "Italian sweet dessert",  "meal_type": "dessert",      "cuisine": "italian",          "is_sweet": True},
    {"name": "Mexican fish quick",     "meal_type": "lunch_dinner", "protein_type": "fish_seafood","cuisine": "mexican",        "is_quick": True},
    {"name": "Stovetop meatless soup", "meal_type": "soup",         "protein_type": "meatless",   "needs_stovetop": True},
    {"name": "American chicken oven",  "meal_type": "lunch_dinner", "protein_type": "chicken",    "cuisine": "american",       "needs_oven": True},
    {"name": "Sweet no-cook snack",    "meal_type": "snack",        "is_sweet": True,  "is_no_cook": True},
    {"name": "Spicy asian meatless",   "meal_type": "lunch_dinner", "protein_type": "meatless",   "cuisine": "asian",          "is_spicy": True},
    {"name": "Quick beef american",    "meal_type": "lunch_dinner", "protein_type": "beef_pork",  "cuisine": "american",       "is_quick": True},
    {"name": "Mediterranean fish",     "meal_type": "lunch_dinner", "protein_type": "fish_seafood","cuisine": "mediterranean"},
    {"name": "Breakfast sweet oven",   "meal_type": "breakfast",    "is_sweet": True,  "needs_oven": True},
    {"name": "Indian quick meatless",  "meal_type": "lunch_dinner", "protein_type": "meatless",   "cuisine": "indian",         "is_quick": True},
    {"name": "French sweet dessert",   "meal_type": "dessert",      "cuisine": "french",           "is_sweet": True,            "needs_oven": True},
    {"name": "Asian beef stovetop",    "meal_type": "lunch_dinner", "protein_type": "beef_pork",  "cuisine": "asian",          "needs_stovetop": True},
    {"name": "American snack quick",   "meal_type": "snack",        "cuisine": "american",         "is_quick": True},
    {"name": "Spicy mexican chicken",  "meal_type": "lunch_dinner", "protein_type": "chicken",    "cuisine": "mexican",        "is_spicy": True},
    {"name": "No-cook mediterranean",  "meal_type": "salad_side",   "protein_type": "meatless",   "cuisine": "mediterranean",  "is_no_cook": True},
    {"name": "Sweet asian drink",      "meal_type": "drink",        "is_sweet": True,  "cuisine": "asian"},
    {"name": "Italian meatless quick", "meal_type": "lunch_dinner", "protein_type": "meatless",   "cuisine": "italian",        "is_quick": True},
    {"name": "Beef oven american",     "meal_type": "lunch_dinner", "protein_type": "beef_pork",  "needs_oven": True,          "cuisine": "american"},
    {"name": "Chicken soup stovetop",  "meal_type": "soup",         "protein_type": "chicken",    "needs_stovetop": True,      "is_spicy": True},
    {"name": "Quick sweet breakfast",  "meal_type": "breakfast",    "is_quick": True,  "is_sweet": True,                      "protein_type": "meatless"},
    {"name": "Indian fish spicy",      "meal_type": "lunch_dinner", "protein_type": "fish_seafood","cuisine": "indian",        "is_spicy": True},
    {"name": "French soup meatless",   "meal_type": "soup",         "protein_type": "meatless",   "cuisine": "french"},
    {"name": "Mexican sweet snack",    "meal_type": "snack",        "cuisine": "mexican",          "is_sweet": True},
    {"name": "Asian no-cook salad",    "meal_type": "salad_side",   "cuisine": "asian",            "is_no_cook": True},
    {"name": "American fish quick",    "meal_type": "lunch_dinner", "protein_type": "fish_seafood","cuisine": "american",      "is_quick": True},
    {"name": "Spicy indian beef",      "meal_type": "lunch_dinner", "protein_type": "beef_pork",  "cuisine": "indian",         "is_spicy": True},
]

# ── HELPERS ───────────────────────────────────────────────────────────────────
def get_feature_value(recipe: dict, question: dict) -> Any:
    feature = question["feature"]
    feature_value = question.get("feature_value")
    recipe_val = recipe.get(feature)
    if feature_value is None:
        return recipe_val
    if isinstance(feature_value, bool):
        return bool(recipe_val) == feature_value
    return recipe_val == feature_value


def recipe_matches_user(recipe: dict, user_profile: dict) -> bool:
    for key, val in user_profile.items():
        if key == "name":
            continue
        recipe_val = recipe.get(key)
        if isinstance(val, bool):
            if bool(recipe_val) != val:
                return False
        elif isinstance(val, str):
            if recipe_val != val:
                return False
    return True


def simulate_answer(user_profile: dict, question: dict) -> list[str]:
    q_type = question["type"]
    feature = question["feature"]
    feature_value = question.get("feature_value")

    if q_type == "categorical":
        user_val = user_profile.get(feature)
        if user_val is None:
            return question["options"]
        return [user_val]

    if feature_value is not None and not isinstance(feature_value, bool):
        user_feature_val = user_profile.get(feature)
        if user_feature_val is None:
            return ["skip"]
        return ["yes"] if user_feature_val == feature_value else ["no"]

    user_val = user_profile.get(feature)
    if user_val is None:
        return ["skip"]
    return ["yes"] if user_val else ["no"]


# ── SCORING (WEIGHTED) ────────────────────────────────────────────────────────
def score_recipes(recipes: list[dict], answers: dict[str, list[str]]) -> dict[int, float]:
    answered_max = 0.0
    for qid, selected in answers.items():
        weight = FEATURE_WEIGHTS.get(qid, 0.0)
        if weight == 0.0 or not selected or selected[0] == "skip":
            continue
        question = next((q for q in QUESTION_BANK if q["id"] == qid), None)
        if question and question["type"] == "categorical":
            all_opts = question.get("options", [])
            if all_opts and set(selected) >= set(all_opts):
                continue
        answered_max += weight

    if answered_max == 0:
        return {r["id"]: 0.0 for r in recipes}

    scores = {}
    for recipe in recipes:
        raw = 0.0
        for qid, selected in answers.items():
            weight = FEATURE_WEIGHTS.get(qid, 0.0)
            if weight == 0.0:
                continue
            question = next((q for q in QUESTION_BANK if q["id"] == qid), None)
            if not question:
                continue
            if question["type"] == "boolean":
                answer = selected[0] if selected else "skip"
                if answer == "skip":
                    continue
                matches = get_feature_value(recipe, question)
                raw += weight if answer == "yes" and matches else \
                       (-weight / 2 if answer == "no" and matches else 0.0)
            else:
                fval = recipe.get(question["feature"])
                all_opts = question.get("options", [])
                if all_opts and set(selected) >= set(all_opts):
                    continue
                raw += weight if fval in selected else 0.0
        scores[recipe["id"]] = round(max(0.0, raw) / answered_max * 100, 4)
    return scores


def top_n_weighted(scores: dict, recipes_by_id: dict, n: int = 10) -> list[tuple]:
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [(recipes_by_id[rid], sc) for rid, sc in ranked[:n] if rid in recipes_by_id]


# ── ENTROPIE MARGINALA ────────────────────────────────────────────────────────
def marginal_entropy(question: dict, recipes: list[dict]) -> float:
    values = [str(get_feature_value(r, question)) for r in recipes]
    counts = Counter(values)
    total = len(recipes)
    if total == 0:
        return 0.0
    entropy = 0.0
    for c in counts.values():
        if c > 0:
            p = c / total
            entropy -= p * math.log2(p)
    return entropy


# ── JMIM ─────────────────────────────────────────────────────────────────────
def discretize_scores(scores: dict, recipes: list[dict]) -> list[str]:
    return ["high" if scores.get(r["id"], 0) > 66
            else "mid" if scores.get(r["id"], 0) > 33
            else "low" for r in recipes]


def joint_mi(xi: list, xs: list, y: list) -> float:
    n = len(y)
    if n == 0:
        return 0.0
    j3 = Counter(zip(xi, xs, y))
    jxy = Counter(zip(xi, xs))
    cy = Counter(y)
    mi = 0.0
    for (a, b, c), cnt in j3.items():
        pj, pxy, py = cnt/n, jxy[(a,b)]/n, cy[c]/n
        if pj > 0 and pxy > 0 and py > 0:
            mi += pj * math.log2(pj / (pxy * py))
    return mi


def jmim_score(question, recipes, asked_ids, scores, answers, smoothing=1e-4):
    if not asked_ids:
        return marginal_entropy(question, recipes)
    xi = [str(get_feature_value(r, question)) for r in recipes]
    y  = discretize_scores(scores, recipes)
    min_jmi = float("inf")
    for asked_qid in asked_ids:
        if answers.get(asked_qid, ["skip"])[0] == "skip":
            continue
        asked_q = next((q for q in QUESTION_BANK if q["id"] == asked_qid), None)
        if not asked_q:
            continue
        xs = [str(get_feature_value(r, asked_q)) for r in recipes]
        jmi = max(joint_mi(xi, xs, y), smoothing)
        if jmi < min_jmi:
            min_jmi = jmi
    if min_jmi < smoothing * 10:
        return marginal_entropy(question, recipes)
    return min_jmi


# ── BAYESIAN SESSION ──────────────────────────────────────────────────────────
def compute_likelihood(recipe: dict, question: dict, answer: str) -> float:
    if answer == "skip":
        return 1.0
    feature = question["feature"]
    feature_value = question.get("feature_value")
    recipe_val = recipe.get(feature)
    q_type = question["type"]

    if q_type == "boolean":
        if feature_value is None:
            recipe_matches = bool(recipe_val)
        elif isinstance(feature_value, bool):
            recipe_matches = bool(recipe_val) == feature_value
        else:
            recipe_matches = recipe_val == feature_value
        if answer == "yes":
            return P_CORRECT if recipe_matches else P_NOISE
        else:
            return P_CORRECT if not recipe_matches else P_NOISE
    else:
        all_opts = question.get("options", [])
        selected = answer if isinstance(answer, list) else [answer]
        if all_opts and set(selected) >= set(all_opts):
            return 1.0
        return P_CORRECT if recipe_val in selected else P_NOISE


class BayesianSession:
    def __init__(self, recipes: list[dict]) -> None:
        self.recipes = recipes
        self.n = len(recipes)
        self._log_probs = np.full(self.n, -math.log(self.n))
        self.answers: dict[str, str] = {}
        self.asked_ids: set[str] = set()
        self.question_number = 0

    @property
    def probs(self) -> np.ndarray:
        log_p = self._log_probs - self._log_probs.max()
        p = np.exp(log_p)
        return p / p.sum()

    def update(self, question: dict, answer: str) -> None:
        self.answers[question["id"]] = answer
        self.asked_ids.add(question["id"])
        self.question_number += 1
        if answer == "skip":
            return
        log_l = np.array([
            math.log(compute_likelihood(r, question, answer) + 1e-10)
            for r in self.recipes
        ])
        self._log_probs = self._log_probs + log_l
        self._log_probs = self._log_probs - self._log_probs.max()

    def entropy(self) -> float:
        p = self.probs
        p_nz = p[p > 1e-10]
        return float(-np.sum(p_nz * np.log2(p_nz)))

    def should_stop(self) -> bool:
        if self.question_number < MIN_QUESTIONS_BEFORE_STOP:
            return False
        if self.question_number >= MAX_QUESTIONS:
            return True
        return self.entropy() < ENTROPY_STOP_THRESHOLD

    def top_n_scored(self, n: int = 10) -> list[tuple]:
        p = self.probs
        top_idx = p.argsort()[-n:][::-1]
        max_p = p[top_idx[0]] if len(top_idx) > 0 else 1.0
        return [(self.recipes[i], round(float(p[i] / max_p) * 100, 2)) for i in top_idx]

    def expected_entropy_reduction(self, question: dict) -> float:
        p = self.probs
        current_h = self.entropy()
        answers_to_check = ["yes", "no", "skip"] if question["type"] == "boolean" \
                           else question.get("options", [])
        expected_h = 0.0
        for answer in answers_to_check:
            likelihoods = np.array([
                compute_likelihood(r, question, answer) for r in self.recipes
            ])
            p_answer = float(np.dot(likelihoods, p))
            if p_answer < 1e-10:
                continue
            log_p_upd = self._log_probs + np.log(likelihoods + 1e-10)
            log_p_upd -= log_p_upd.max()
            p_upd = np.exp(log_p_upd)
            p_upd /= p_upd.sum()
            p_nz = p_upd[p_upd > 1e-10]
            h_after = float(-np.sum(p_nz * np.log2(p_nz)))
            expected_h += p_answer * h_after
        return current_h - expected_h


def select_next_question_bayesian(session: BayesianSession) -> dict | None:
    fixed_idx = session.question_number
    if fixed_idx < len(_FIXED_QS):
        return _FIXED_QS[fixed_idx]
    candidates = [q for q in _ADAPTIVE_QS if q["id"] not in session.asked_ids]
    if not candidates:
        return None
    return max(candidates, key=lambda q: session.expected_entropy_reduction(q))


# ── METRICI ───────────────────────────────────────────────────────────────────
def compute_ndcg_correct(ranked: list[tuple], all_recipes: list[dict], user_profile: dict, k: int = 10) -> float:
    relevances = [
        1.0 if recipe_matches_user(r, user_profile) else 0.0
        for r, _ in ranked[:k]
    ]

    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))

    total_relevant = sum(
        1 for r in all_recipes
        if recipe_matches_user(r, user_profile)
    )

    ideal_relevances = [1.0] * min(k, total_relevant)

    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_relevances))

    return dcg / idcg if idcg > 0 else 0.0


def compute_overlap(ranked: list[tuple], user_profile: dict, k: int = 10) -> float:
    matches = sum(1 for r, _ in ranked[:k] if recipe_matches_user(r, user_profile))
    return matches / min(k, len(ranked))


# ── CRITERIU DE OPRIRE (MARGINAL/JMIM) ───────────────────────────────────────
def should_stop_entropy(q_num: int, candidates: list, recipes: list,
                        scores: dict, entropy_threshold: float = 0.5) -> bool:
    if q_num > MAX_QUESTIONS:
        return True
    if q_num <= MIN_QUESTIONS_BEFORE_STOP:
        return False
    if not candidates:
        return True
    if scores and len(scores) > 10:
        threshold = sorted(scores.values(), reverse=True)[len(scores) // 3]
        relevant = [r for r in recipes if scores.get(r["id"], 0) >= threshold]
    else:
        relevant = recipes
    max_entropy = max(marginal_entropy(q, relevant) for q in candidates)
    return max_entropy < entropy_threshold


# ── RULARE SESIUNE ────────────────────────────────────────────────────────────
def run_session(recipes: list[dict], user_profile: dict, mode: str,
                verbose: bool = False) -> dict:

    # ── BAYESIAN ──
    if mode == "bayesian":
        session = BayesianSession(recipes)
        while not session.should_stop():
            question = select_next_question_bayesian(session)
            if question is None:
                break
            answer = simulate_answer(user_profile, question)
            answer_str = answer[0] if len(answer) == 1 else answer
            if verbose:
                print(f"      Q{session.question_number+1}: {question['id']} -> {answer}")
            session.update(question, answer_str if isinstance(answer_str, str) else answer_str[0])
        ranked = session.top_n_scored(TOP_N)
        return {
            "questions_asked": session.question_number,
            "question_order": list(session.asked_ids),
            "ndcg_10": round(compute_ndcg_correct(ranked, user_profile), 4),
            "overlap_10": round(compute_overlap(ranked, user_profile), 4),
            "avg_top10_score": round(sum(sc for _, sc in ranked) / len(ranked) if ranked else 0, 2),
        }

    # ── MARGINAL / JMIM ──
    recipes_by_id = {r["id"]: r for r in recipes}
    answers: dict[str, list[str]] = {}
    asked_ids: set[str] = set()
    q_num = 1
    questions_asked = []

    while True:
        scores = score_recipes(recipes, answers) if answers else {r["id"]: 0.0 for r in recipes}

        # selectie intrebare
        fixed_idx = q_num - 1
        if fixed_idx < len(_FIXED_QS):
            question = _FIXED_QS[fixed_idx]
        else:
            candidates = [q for q in _ADAPTIVE_QS if q["id"] not in asked_ids]
            if not candidates:
                break
            if mode == "jmim" and answers:
                question = max(candidates, key=lambda q: jmim_score(
                    q, recipes, asked_ids, scores, answers))
            else:
                question = max(candidates, key=lambda q: marginal_entropy(q, recipes))

        answer = simulate_answer(user_profile, question)
        if verbose:
            print(f"      Q{q_num}: {question['id']} -> {answer}")

        answers[question["id"]] = answer
        asked_ids.add(question["id"])
        questions_asked.append(question["id"])
        scores = score_recipes(recipes, answers)
        q_num += 1

        remaining = [q for q in _ADAPTIVE_QS if q["id"] not in asked_ids]
        if should_stop_entropy(q_num, remaining, recipes, scores):
            break

    final_scores = score_recipes(recipes, answers)
    final_ranked = top_n_weighted(final_scores, recipes_by_id, TOP_N)
    return {
        "questions_asked": len(questions_asked),
        "question_order": questions_asked,
        "ndcg_10": round(compute_ndcg_correct(ranked, recipes, user_profile), 4),
        "overlap_10": round(compute_overlap(final_ranked, user_profile), 4),
        "avg_top10_score": round(sum(sc for _, sc in final_ranked) / len(final_ranked) if final_ranked else 0, 2),
    }


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["marginal","jmim","bayesian","both","all"],
                        default="all")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--sample", type=int, default=0,
                        help="Testeaza pe primele N retete (0=toate)")
    args = parser.parse_args()

    print(f"Incarc dataset din {DATASET_PATH}...")
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    recipes = []
    for idx, entry in enumerate(raw):
        features = entry.get("llm_features")
        if not features or entry.get("llm_failed"):
            continue
        recipes.append({
            "id": idx,
            "name": entry.get("Name", ""),
            "meal_type":     features.get("meal_type"),
            "protein_type":  features.get("protein_type"),
            "cuisine":       features.get("cuisine"),
            "is_spicy":      features.get("is_spicy", False),
            "is_sweet":      features.get("is_sweet", False),
            "is_quick":      features.get("is_quick", False),
            "needs_oven":    features.get("needs_oven", False),
            "needs_stovetop":features.get("needs_stovetop", False),
            "is_no_cook":    features.get("is_no_cook", False),
        })

    if args.sample > 0:
        recipes = recipes[:args.sample]
        print(f"Testez pe primele {args.sample} retete")

    print(f"Retete: {len(recipes)} | Utilizatori sintetici: {len(SYNTHETIC_USERS)}")
    print()

    modes = {"both": ["marginal","jmim"], "all": ["marginal","jmim","bayesian"]}.get(
        args.mode, [args.mode])

    all_results: dict[str, list[dict]] = {}

    for mode in modes:
        print(f"{'='*60}")
        print(f"MOD: {mode.upper()}")
        print(f"{'='*60}")
        results = []
        for user in SYNTHETIC_USERS:
            if args.verbose:
                print(f"  {user['name']}:")
            result = run_session(recipes, user, mode, verbose=args.verbose)
            result["user"] = user["name"]
            results.append(result)
            print(f"  {user['name'][:35]:<35} | "
                  f"Q: {result['questions_asked']} | "
                  f"NDCG: {result['ndcg_10']:.3f} | "
                  f"Overlap: {result['overlap_10']:.2f} | "
                  f"AvgScore: {result['avg_top10_score']:.1f}%")

        all_results[mode] = results
        avg_q     = sum(r["questions_asked"] for r in results) / len(results)
        avg_ndcg  = sum(r["ndcg_10"]         for r in results) / len(results)
        avg_over  = sum(r["overlap_10"]       for r in results) / len(results)
        avg_score = sum(r["avg_top10_score"]  for r in results) / len(results)
        print(f"\n  MEDII: Q={avg_q:.2f} | NDCG={avg_ndcg:.4f} | "
              f"Overlap={avg_over:.4f} | AvgScore={avg_score:.1f}%\n")

    if len(modes) > 1:
        print(f"{'='*60}")
        print("COMPARATIE FINALA")
        print(f"{'='*60}")
        baseline = all_results[modes[0]]
        for mode in modes[1:]:
            other = all_results[mode]
            print(f"\n  {modes[0].upper()} vs {mode.upper()}:")
            for metric, label in [("questions_asked","Nr. intrebari"),
                                   ("ndcg_10","NDCG@10"),
                                   ("overlap_10","Overlap@10"),
                                   ("avg_top10_score","Avg Score (%)")]:
                b_avg = sum(r[metric] for r in baseline) / len(baseline)
                o_avg = sum(r[metric] for r in other)    / len(other)
                diff  = o_avg - b_avg
                better = (metric != "questions_asked" and diff > 0) or \
                         (metric == "questions_asked" and diff < 0)
                direction = "↑ mai bun" if better else ("↓ mai slab" if diff != 0 else "= egal")
                print(f"    {label:<20} {modes[0]}={b_avg:.4f} | {mode}={o_avg:.4f} | "
                      f"Δ={diff:+.4f} {direction}")

    with open("evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print("\nRezultate salvate in evaluation_results.json")


if __name__ == "__main__":
    main()