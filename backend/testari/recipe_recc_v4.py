from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from typing import Any

import numpy as np

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATASET_PATH = r"D:\folosire_api_claude\dataset_llm_labeled_with_ingredient_features.json"
MAX_QUESTIONS = 15
MIN_QUESTIONS_BEFORE_STOP = 3
TOP_N = 10

# Bayesian noise model — absoare răspunsuri greșite
P_CORRECT = 0.90
P_NOISE   = 0.05

# Early stopping: se oprește când H < log2(K) unde K = număr rețete "echivalente" rămase
# log2(15) ≈ 3.9 biți — înseamnă că suntem concentrați pe ~15 candidate
ENTROPY_STOP_THRESHOLD = math.log2(15)

# ── QUESTION BANK ─────────────────────────────────────────────────────────────
# meal_type rămâne categorial single-select (e fundamental)
# cuisine și protein_type devin MULTI-SELECT
# restul rămân boolean

CUISINE_OPTIONS  = ["italian", "asian", "mexican", "french", "mediterranean", "indian", "american", "other"]
PROTEIN_OPTIONS  = ["chicken", "beef_pork", "fish_seafood", "meatless"]
MEAL_TYPE_OPTIONS = ["appetizer", "breakfast", "dessert", "drink", "lunch_dinner",
                     "salad_side", "snack", "soup", "condiment"]

QUESTION_BANK: list[dict[str, Any]] = [
    # Fixed Q1 — single-select categorial
    {
        "id": "meal_type",
        "type": "categorical",
        "feature": "meal_type",
        "fixed": True,
        "order": 1,
        "options": MEAL_TYPE_OPTIONS,
    },
    # Fixed Q2 — multi-select (nou)
    {
        "id": "protein_type",
        "type": "multiselect",
        "feature": "protein_type",
        "fixed": True,
        "order": 2,
        "options": PROTEIN_OPTIONS,
        "any_option": "any",   # răspuns special = skip / orice
    },
    # Fixed Q3 — multi-select (nou)
    {
        "id": "cuisine",
        "type": "multiselect",
        "feature": "cuisine",
        "fixed": True,
        "order": 3,
        "options": CUISINE_OPTIONS,
        "any_option": "any",
    },



    # Adaptive boolean questions
    {"id": "is_spicy",       "type": "boolean", "feature": "is_spicy",       "feature_value": True},
    {"id": "is_sweet",       "type": "boolean", "feature": "is_sweet",       "feature_value": True},
    {"id": "is_quick",       "type": "boolean", "feature": "is_quick",       "feature_value": True},
    {"id": "needs_oven",     "type": "boolean", "feature": "needs_oven",     "feature_value": True},
    {"id": "needs_stovetop", "type": "boolean", "feature": "needs_stovetop", "feature_value": True},
    {"id": "is_no_cook",     "type": "boolean", "feature": "is_no_cook",     "feature_value": True},

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

_FIXED_QS    = sorted([q for q in QUESTION_BANK if q.get("fixed")], key=lambda q: q.get("order", 99))
_ADAPTIVE_QS = [q for q in QUESTION_BANK if not q.get("fixed")]


def question_by_id(qid: str) -> dict[str, Any] | None:
    return next((q for q in QUESTION_BANK if q["id"] == qid), None)


# ── WEIGHTS CALCULATE DIN DATASET (Mutual Information) ───────────────────────
def compute_feature_mi(recipes: list[dict[str, Any]]) -> dict[str, float]:
    """
    Calculează MI condiționat: I(feature; cuisine+protein_type | meal_type)
    
    Logică: după ce știm meal_type, cât de mult discriminează feature-ul
    între subtipurile de rețete (cuisine + protein)? Asta face has_pasta,
    has_tortilla etc. mult mai relevante decât MI global față de meal_type.
    """
    n = len(recipes)
    if n == 0:
        return {}

    mi_scores: dict[str, float] = {}

    # Target condiționat: (cuisine, protein_type) — identitatea subtipului
    # Condiție: meal_type
    for q in _ADAPTIVE_QS:
        feature = q["feature"]
        qid = q["id"]
        feature_value = q.get("feature_value")

        if isinstance(feature_value, bool):
            fvals = [bool(r.get(feature)) == feature_value for r in recipes]
        else:
            fvals = [r.get(feature) == feature_value for r in recipes]

        meal_types = [r.get("meal_type", "unknown") for r in recipes]
        targets = [
            (r.get("cuisine", "unknown"), r.get("protein_type", "unknown"))
            for r in recipes
        ]

        # Grupăm pe meal_type și calculăm MI în fiecare grup
        # MI_cond = sum_{m} P(meal_type=m) * I(feature; target | meal_type=m)
        meal_groups: dict[str, list[int]] = {}
        for i, mt in enumerate(meal_types):
            meal_groups.setdefault(mt, []).append(i)

        cond_mi = 0.0
        for mt, indices in meal_groups.items():
            p_mt = len(indices) / n
            if p_mt < 1e-10:
                continue

            fv_group  = [fvals[i]   for i in indices]
            tgt_group = [targets[i] for i in indices]
            n_g = len(indices)

            joint: Counter = Counter(zip(fv_group, tgt_group))
            f_cnt  = Counter(fv_group)
            tgt_cnt = Counter(tgt_group)

            mi_g = 0.0
            for (fv, tgt), cnt in joint.items():
                p_joint = cnt / n_g
                p_f   = f_cnt[fv]  / n_g
                p_tgt = tgt_cnt[tgt] / n_g
                if p_joint > 0 and p_f > 0 and p_tgt > 0:
                    mi_g += p_joint * math.log2(p_joint / (p_f * p_tgt))

            cond_mi += p_mt * max(mi_g, 0.0)

        mi_scores[qid] = cond_mi

    # Normalizăm la [0.3, 3.0]
    max_mi = max(mi_scores.values()) if mi_scores else 1.0
    if max_mi <= 0:
        max_mi = 1.0

    normalized: dict[str, float] = {}
    for qid, mi in mi_scores.items():
        normalized[qid] = round(0.3 + (mi / max_mi) * 2.7, 4)

    # Fixe
    normalized["meal_type"]    = 4.0
    normalized["protein_type"] = 3.0
    normalized["cuisine"]      = 2.5

    return normalized

#--Vizualizare selectie--
def visualize_question_selection(
    session: BayesianSession,
    step: int,
    output_path: str = "question_selection.json"
) -> None:
    candidates = [q for q in _ADAPTIVE_QS if q["id"] not in session.asked_ids]
    if not candidates:
        return

    p = session.probs()
    uniform = 1.0 / session.n
    rel_idx = np.where(p > uniform * 0.01)[0]
    rel_recipes = [session.recipes[i] for i in rel_idx]

    rows = []
    for q in candidates:
        eig = session.expected_entropy_reduction(q)
        count = sum(1 for r in rel_recipes if get_feature_value_bool(r, q)) if q["type"] == "boolean" else 0
        prev = count / len(rel_recipes) if rel_recipes else 0
        rows.append({
            "question_id": q["id"],
            "eig": round(eig, 4),
            "prevalence": round(prev, 3),
            "relevant_pool_size": len(rel_recipes),
        })

    rows.sort(key=lambda x: -x["eig"])
    if rows:
        rows[0]["selected"] = True
    for r in rows[1:]:
        r["selected"] = False

    data = {
        "step": step,
        "entropy_before": round(session.entropy(), 4),
        "asked_so_far": list(session.asked_ids),
        "candidates": rows,
    }

    try:
        with open(output_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        existing = []

    existing.append(data)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)

# ── HELPERS ───────────────────────────────────────────────────────────────────
def get_feature_value_bool(recipe: dict[str, Any], question: dict[str, Any]) -> bool:
    """Returnează True dacă rețeta satisface condiția întrebării boolean."""
    feature = question["feature"]
    feature_value = question.get("feature_value")
    recipe_val = recipe.get(feature)

    if isinstance(feature_value, bool):
        return bool(recipe_val) == feature_value
    return recipe_val == feature_value


def recipe_matches_user(recipe: dict[str, Any], user_profile: dict[str, Any]) -> bool:
    for key, val in user_profile.items():
        if key == "name" or val is None:
            continue

        recipe_val = recipe.get(key)

        if key == "meal_type":
            if recipe_val != val:
                return False
        elif key in {"protein_type", "cuisine"}:
            # Multi-select: rețeta trebuie să fie într-una din opțiunile alese
            if isinstance(val, list):
                if val and recipe_val not in val:
                    return False
            else:
                if recipe_val != val:
                    return False
        elif isinstance(val, bool):
            if val is True and bool(recipe_val) is not True:
                return False
        else:
            if recipe_val != val:
                return False

    return True


def simulate_answer(user_profile: dict[str, Any], question: dict[str, Any]) -> Any:
    """
    Simulează răspunsul utilizatorului pentru evaluare.
    Returnează:
    - string pentru categorical/boolean
    - list[str] pentru multiselect
    """
    q_type = question["type"]
    feature = question["feature"]

    if q_type == "categorical":
        return user_profile.get(feature, "skip") or "skip"

    if q_type == "multiselect":
        user_val = user_profile.get(feature)
        if user_val is None:
            return ["any"]
        if isinstance(user_val, list):
            return user_val if user_val else ["any"]
        return [user_val]  # string → list

    # Boolean
    feature_value = question.get("feature_value")
    user_val = user_profile.get(feature)

    if user_val is None:
        return "no"
    if isinstance(feature_value, bool):
        return "yes" if bool(user_val) == feature_value else "no"
    return "yes" if user_val == feature_value else "no"


# ── BAYESIAN LIKELIHOOD ───────────────────────────────────────────────────────
def compute_likelihood(
    recipe: dict[str, Any],
    question: dict[str, Any],
    answer: Any,
) -> float:
    """
    Calculează P(answer | recipe) cu model de zgomot.

    Pentru multiselect (cuisine, protein_type):
      - Dacă utilizatorul a ales ["italian", "asian"] și rețeta e italiană → P_CORRECT
      - Dacă rețeta nu e în lista aleasă → P_NOISE
      - "any" → likelihood 1.0 (neutru)

    Soft Bayesian: nu eliminăm hard, ci scădem probabilitatea.
    """
    q_type = question["type"]
    feature = question["feature"]

    if q_type == "categorical":
        if answer == "skip":
            return 1.0
        return P_CORRECT if recipe.get(feature) == answer else P_NOISE

    if q_type == "multiselect":
        if answer == ["any"] or answer == "any":
            return 1.0  # utilizatorul nu are preferință → neutru
        selected = answer if isinstance(answer, list) else [answer]
        recipe_val = recipe.get(feature)
        # Soft: dacă rețeta e în lista aleasă → P_CORRECT, altfel P_NOISE
        return P_CORRECT if recipe_val in selected else P_NOISE

    # Boolean
    if answer == "skip":
        return 1.0

    match = get_feature_value_bool(recipe, question)
    if answer == "yes":
        return P_CORRECT if match else P_NOISE
    if answer == "no":
        return P_CORRECT if not match else P_NOISE

    return 1.0


# ── BAYESIAN SESSION ──────────────────────────────────────────────────────────
class BayesianSession:
    def relevant_recipes(self, threshold_factor: float = 0.01) -> list[dict[str, Any]]:
        """Returnează rețetele cu prob > threshold față de distribuție uniformă."""
        p = self.probs()
        uniform = 1.0 / self.n
        mask = p > uniform * threshold_factor
        return [self.recipes[i] for i in range(self.n) if mask[i]]
    
    def __init__(self, recipes: list[dict[str, Any]], weights: dict[str, float]) -> None:
        self.recipes = recipes
        self.n = len(recipes)
        self.log_probs = np.full(self.n, -math.log(self.n))
        self.answers: dict[str, Any] = {}
        self.asked_ids: set[str] = set()
        self.q = 0
        self.weights = weights  # MI-based weights

    def probs(self) -> np.ndarray:
        log_p = self.log_probs - self.log_probs.max()
        p = np.exp(log_p)
        return p / p.sum()

    def update(self, question: dict[str, Any], answer: Any) -> None:
        self.q += 1
        self.answers[question["id"]] = answer
        self.asked_ids.add(question["id"])

        # Skip/any → nicio actualizare
        if answer == "skip" or answer == ["any"] or answer == "any":
            return

        likelihoods = np.array([
            compute_likelihood(recipe, question, answer)
            for recipe in self.recipes
        ])

        # Weighted update: întrebările mai informative (weight mai mare)
        # actualizează log_probs mai puternic
        w = self.weights.get(question["id"], 1.0)
        self.log_probs += w * np.log(likelihoods + 1e-10)
        self.log_probs -= self.log_probs.max()

    def entropy(self) -> float:
        p = self.probs()
        p_nz = p[p > 1e-10]
        return float(-np.sum(p_nz * np.log2(p_nz)))

    def should_stop(self) -> bool:
        if self.q < MIN_QUESTIONS_BEFORE_STOP:
            return False
        if self.q >= MAX_QUESTIONS:
            return True
        return self.entropy() < ENTROPY_STOP_THRESHOLD

    def expected_entropy_reduction(self, question: dict[str, Any]) -> float:
        """EIG calculat pe subsetul de rețete cu probabilitate semnificativă."""
        p = self.probs()
        current_h = self.entropy()

        # Luăm doar rețetele cu prob > 1% din uniform — zona unde contează
        uniform = 1.0 / self.n
        relevant_idx = np.where(p > uniform * 0.01)[0]

        if len(relevant_idx) < 10:
            relevant_idx = p.argsort()[-100:]  # fallback: top 100

        p_relevant = p[relevant_idx]
        p_relevant = p_relevant / p_relevant.sum()  # renormalizăm

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
            # Likelihood doar pe subsetul relevant
            likelihoods = np.array([
                compute_likelihood(self.recipes[i], question, answer)
                for i in relevant_idx
            ])

            p_answer = float(np.dot(likelihoods, p_relevant))
            if p_answer < 1e-12:
                continue

            log_updated = np.log(p_relevant + 1e-10) + w * np.log(likelihoods + 1e-10)
            log_updated -= log_updated.max()
            p_updated = np.exp(log_updated)
            p_updated /= p_updated.sum()

            p_nz = p_updated[p_updated > 1e-10]
            h_after = float(-np.sum(p_nz * np.log2(p_nz)))
            expected_h += p_answer * h_after

        return max(0.0, current_h - expected_h)

    def top(self, n: int = 10) -> list[tuple[dict[str, Any], float]]:
        p = self.probs()
        idx = p.argsort()[-n:][::-1]
        uniform_p = 1.0 / self.n
        return [
            (self.recipes[i], round(float(p[i] / uniform_p), 2))
            for i in idx
        ]


def select_next_question(session: BayesianSession) -> dict[str, Any] | None:
    if session.q < len(_FIXED_QS):
        return _FIXED_QS[session.q]

    candidates = [q for q in _ADAPTIVE_QS if q["id"] not in session.asked_ids]
    if not candidates:
        return None

    # Verificăm câte răspunsuri "no" consecutive am primit
    recent = list(session.answers.values())[-3:]
    consecutive_no = sum(1 for a in recent if a == "no" or a == ["any"])

    if consecutive_no >= 3:
        # Mod agresiv: preferă features cu prevalență 20-50% în subsetul relevant
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
            # Boost maxim pentru prevalență ~35% (cea mai discriminativă)
            boost = 1.0 + 2.0 * (1.0 - abs(prev - 0.35) / 0.35)
            boost = max(1.0, boost)
            return eig * boost

        return max(candidates, key=prevalence_boosted_score)

    return max(candidates, key=lambda q: session.expected_entropy_reduction(q))
    


# ── METRICS ───────────────────────────────────────────────────────────────────
def compute_ndcg(
    ranked: list[tuple[dict[str, Any], float]],
    all_recipes: list[dict[str, Any]],
    user_profile: dict[str, Any],
    k: int = 10
) -> float:
    relevances = [1.0 if recipe_matches_user(r, user_profile) else 0.0 for r, _ in ranked[:k]]
    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))
    total_rel = sum(1 for r in all_recipes if recipe_matches_user(r, user_profile))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(k, total_rel)))
    return dcg / idcg if idcg > 0 else 0.0


def compute_overlap(ranked: list[tuple[dict[str, Any], float]], user_profile: dict[str, Any], k: int = 10) -> float:
    top = ranked[:k]
    if not top:
        return 0.0
    matches = sum(1 for r, _ in top if recipe_matches_user(r, user_profile))
    return matches / min(k, len(top))


def count_relevant(recipes: list[dict[str, Any]], user_profile: dict[str, Any]) -> int:
    return sum(1 for r in recipes if recipe_matches_user(r, user_profile))


# ── SESSION RUNNER ────────────────────────────────────────────────────────────
def run_session(
    recipes: list[dict[str, Any]],
    user_profile: dict[str, Any],
    weights: dict[str, float],
    verbose: bool = False,
    export_visualization: bool = False,  # nou
) -> dict[str, Any]:
    session = BayesianSession(recipes, weights)
    question_order: list[str] = []

    while not session.should_stop():
        question = select_next_question(session)
        if question is None:
            break
       

        answer = simulate_answer(user_profile, question)
        session.update(question, answer)
        question_order.append(question["id"])
        
        # Export vizualizare dacă e pas adaptiv
        if export_visualization and session.q >= len(_FIXED_QS):
            visualize_question_selection(session, step=session.q + 1)

        if verbose:
            print(f"      Q{session.q}: {question['id']} -> {answer} | H={session.entropy():.3f} bits")

    ranked = session.top(TOP_N)

    return {
        "questions_asked": session.q,
        "question_order": question_order,
        "total_relevant": count_relevant(recipes, user_profile),
        "ndcg_10": round(compute_ndcg(ranked, recipes, user_profile, TOP_N), 4),
        "overlap_10": round(compute_overlap(ranked, user_profile, TOP_N), 4),
        "avg_top10_score": round(sum(s for _, s in ranked) / len(ranked) if ranked else 0.0, 1),
        "entropy_final": round(session.entropy(), 3),
        "top": [
            {
                "name": r.get("name", ""),
                "score_x_uniform": score,
                "meal_type":    r.get("meal_type"),
                "protein_type": r.get("protein_type"),
                "cuisine":      r.get("cuisine"),
                "has_pasta":    r.get("has_pasta"),
            }
            for r, score in ranked
        ],
    }


# ── SYNTHETIC USERS ───────────────────────────────────────────────────────────
SYNTHETIC_USERS = [
    # ── SIMPLE (1-2 constrângeri) ──────────────────────────────────────────
    {"name": "Dessert any",             "meal_type": "dessert"},
    {"name": "Soup any",                "meal_type": "soup"},
    {"name": "Breakfast any",           "meal_type": "breakfast"},
    {"name": "Snack any",               "meal_type": "snack"},
    {"name": "Drink any",               "meal_type": "drink"},
    {"name": "Appetizer any",           "meal_type": "appetizer"},
    {"name": "Salad any",               "meal_type": "salad_side"},

    {"name": "Lunch chicken",           "meal_type": "lunch_dinner", "protein_type": "chicken"},
    {"name": "Lunch beef",              "meal_type": "lunch_dinner", "protein_type": "beef_pork"},
    {"name": "Lunch fish",              "meal_type": "lunch_dinner", "protein_type": "fish_seafood"},
    {"name": "Lunch meatless",          "meal_type": "lunch_dinner", "protein_type": "meatless"},
    {"name": "Dessert sweet",           "meal_type": "dessert",      "is_sweet": True},
    {"name": "Breakfast quick",         "meal_type": "breakfast",    "is_quick": True},
    {"name": "Snack no cook",           "meal_type": "snack",        "is_no_cook": True},

    # ── CUISINE (fără protein specificat) ─────────────────────────────────
    {"name": "Italian any",             "meal_type": "lunch_dinner", "cuisine": "italian"},
    {"name": "Asian any",               "meal_type": "lunch_dinner", "cuisine": "asian"},
    {"name": "Mexican any",             "meal_type": "lunch_dinner", "cuisine": "mexican"},
    {"name": "Indian any",              "meal_type": "lunch_dinner", "cuisine": "indian"},
    {"name": "Mediterranean any",       "meal_type": "salad_side",   "cuisine": "mediterranean"},
    {"name": "French any",              "meal_type": "lunch_dinner", "cuisine": "french"},
    {"name": "American any",            "meal_type": "lunch_dinner", "cuisine": "american"},
    {"name": "American dessert",        "meal_type": "dessert",      "cuisine": "american"},

    # ── INGREDIENT-DRIVEN (fără cuisine) ──────────────────────────────────
    {"name": "Pasta lunch",             "meal_type": "lunch_dinner", "has_pasta": True},
    {"name": "Rice lunch",              "meal_type": "lunch_dinner", "has_rice": True},
    {"name": "Tortilla lunch",          "meal_type": "lunch_dinner", "has_tortilla": True},
    {"name": "Chocolate dessert",       "meal_type": "dessert",      "has_chocolate": True},
    {"name": "Creamy soup",             "meal_type": "soup",         "has_cream_base": True},
    {"name": "Cheesy snack",            "meal_type": "snack",        "has_cheese": True},
    {"name": "Mushroom soup",           "meal_type": "soup",         "has_mushroom": True},
    {"name": "Fruity breakfast",        "meal_type": "breakfast",    "has_fruit": True},
    {"name": "Nutty dessert",           "meal_type": "dessert",      "has_nuts": True},
    {"name": "Spicy lunch",             "meal_type": "lunch_dinner", "is_spicy": True},
    {"name": "Sweet breakfast",         "meal_type": "breakfast",    "is_sweet": True},

    # ── MEDII (cuisine + protein) ──────────────────────────────────────────
    {"name": "Italian chicken",         "meal_type": "lunch_dinner", "protein_type": "chicken",    "cuisine": "italian"},
    {"name": "Asian chicken",           "meal_type": "lunch_dinner", "protein_type": "chicken",    "cuisine": "asian"},
    {"name": "Mexican beef",            "meal_type": "lunch_dinner", "protein_type": "beef_pork",  "cuisine": "mexican"},
    {"name": "Indian meatless",         "meal_type": "lunch_dinner", "protein_type": "meatless",   "cuisine": "indian"},
    {"name": "American beef",           "meal_type": "lunch_dinner", "protein_type": "beef_pork",  "cuisine": "american"},
    {"name": "Asian fish",              "meal_type": "lunch_dinner", "protein_type": "fish_seafood","cuisine": "asian"},
    {"name": "Mediterranean meatless",  "meal_type": "salad_side",   "protein_type": "meatless",   "cuisine": "mediterranean"},
    {"name": "French chicken",          "meal_type": "lunch_dinner", "protein_type": "chicken",    "cuisine": "french"},
    {"name": "Chicken soup",            "meal_type": "soup",         "protein_type": "chicken"},
    {"name": "Meatless soup",           "meal_type": "soup",         "protein_type": "meatless"},

    # ── COMPLEXE (cuisine + protein + ingredient/preference) ──────────────
    {"name": "Italian pasta chicken",   "meal_type": "lunch_dinner", "protein_type": "chicken",    "cuisine": "italian",  "has_pasta": True},
    {"name": "Italian pasta meatless",  "meal_type": "lunch_dinner", "protein_type": "meatless",   "cuisine": "italian",  "has_pasta": True},
    {"name": "Asian spicy chicken",     "meal_type": "lunch_dinner", "protein_type": "chicken",    "cuisine": "asian",    "is_spicy": True},
    {"name": "Asian soy stovetop",      "meal_type": "lunch_dinner", "cuisine": "asian",           "is_spicy": True,      "has_asian_sauce": True},
    {"name": "Mexican tortilla beef",   "meal_type": "lunch_dinner", "protein_type": "beef_pork",  "cuisine": "mexican",  "has_tortilla": True},
    {"name": "Mexican cheese beef",     "meal_type": "lunch_dinner", "protein_type": "beef_pork",  "cuisine": "mexican",  "has_cheese": True},
    {"name": "Indian spicy meatless",   "meal_type": "lunch_dinner", "protein_type": "meatless",   "cuisine": "indian",   "is_spicy": True},
    {"name": "Chocolate oven dessert",  "meal_type": "dessert",      "has_chocolate": True,         "needs_oven": True},
    {"name": "Chocolate sweet dessert", "meal_type": "dessert",      "has_chocolate": True,         "is_sweet": True,      "needs_oven": True},
    {"name": "Creamy mushroom soup",    "meal_type": "soup",         "protein_type": "meatless",   "has_mushroom": True,  "has_cream_base": True},
    {"name": "Creamy meatless soup",    "meal_type": "soup",         "protein_type": "meatless",   "has_cream_base": True},
    {"name": "Quick American beef",     "meal_type": "lunch_dinner", "protein_type": "beef_pork",  "cuisine": "american", "is_quick": True},
    {"name": "Mediterranean no cook",   "meal_type": "salad_side",   "cuisine": "mediterranean",   "is_no_cook": True},
    {"name": "Asian rice fish",         "meal_type": "lunch_dinner", "protein_type": "fish_seafood","has_rice": True},
    {"name": "Breakfast oven sweet",    "meal_type": "breakfast",    "needs_oven": True,            "is_sweet": True},

    # ── FOARTE SPECIFICE (Rel mic, test limită) ────────────────────────────
    {"name": "Italian chicken pasta stovetop", "meal_type": "lunch_dinner", "protein_type": "chicken",    "cuisine": "italian",  "has_pasta": True,      "needs_stovetop": True},
    {"name": "Asian spicy soy fish",           "meal_type": "lunch_dinner", "protein_type": "fish_seafood","cuisine": "asian",    "is_spicy": True,       "has_asian_sauce": True},
    {"name": "Mexican spicy tortilla meatless","meal_type": "lunch_dinner", "protein_type": "meatless",   "cuisine": "mexican",  "has_tortilla": True,   "is_spicy": True},
    {"name": "Chocolate nutty oven dessert",   "meal_type": "dessert",      "has_chocolate": True,         "has_nuts": True,      "needs_oven": True},
    {"name": "Creamy potato soup meatless",    "meal_type": "soup",         "protein_type": "meatless",   "has_cream_base": True,"has_potato": True},
    {"name": "Indian spicy tomato chicken",    "meal_type": "lunch_dinner", "protein_type": "chicken",    "cuisine": "indian",   "is_spicy": True,       "has_tomato_base": True},
    {"name": "French cream chicken oven",      "meal_type": "lunch_dinner", "protein_type": "chicken",    "cuisine": "french",   "has_cream_base": True, "needs_oven": True},
    {"name": "Asian rice meatless quick",      "meal_type": "lunch_dinner", "protein_type": "meatless",   "cuisine": "asian",    "has_rice": True,       "is_quick": True},

    # ── QUICK (is_quick ca constrângere principală) ────────────────────────
    {"name": "Quick lunch any",         "meal_type": "lunch_dinner", "is_quick": True},
    {"name": "Quick chicken",           "meal_type": "lunch_dinner", "protein_type": "chicken",   "is_quick": True},
    {"name": "Quick beef",              "meal_type": "lunch_dinner", "protein_type": "beef_pork",  "is_quick": True},
    {"name": "Quick meatless",          "meal_type": "lunch_dinner", "protein_type": "meatless",   "is_quick": True},
    {"name": "Quick asian",             "meal_type": "lunch_dinner", "cuisine": "asian",           "is_quick": True},
    {"name": "Quick mexican",           "meal_type": "lunch_dinner", "cuisine": "mexican",         "is_quick": True},
    {"name": "Quick american beef",     "meal_type": "lunch_dinner", "protein_type": "beef_pork",  "cuisine": "american", "is_quick": True},
    {"name": "Quick snack",             "meal_type": "snack",        "is_quick": True},
    {"name": "Quick breakfast",         "meal_type": "breakfast",    "is_quick": True},
    {"name": "Quick soup",              "meal_type": "soup",         "is_quick": True},
]


# ── DATA LOADING ──────────────────────────────────────────────────────────────
def load_recipes(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    expected = {
        "meal_type", "protein_type", "cuisine",
        "is_spicy", "is_sweet", "is_quick", "needs_oven", "needs_stovetop", "is_no_cook",
        "has_pasta", "has_rice", "has_potato", "has_tomato_base", "has_cream_base",
        "has_cheese", "has_broth_base", "has_mushroom", "has_leafy_greens",
        "has_beans_legumes", "has_fruit", "has_nuts", "has_chocolate",
        "has_tortilla", "has_spicy_ingredient", "has_asian_sauce",
    }

    recipes = []
    for idx, entry in enumerate(raw):
        features = entry.get("llm_features")
        if not features or entry.get("llm_failed"):
            continue
        recipe = {"id": idx, "name": entry.get("Name", "")}
        for feat in expected:
            recipe[feat] = features.get(feat, False)
        recipes.append(recipe)

    return recipes


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose",   action="store_true")
    parser.add_argument("--show-top",  action="store_true")
    parser.add_argument("--sample",    type=int, default=0)
    parser.add_argument("--show-weights", action="store_true", help="Afișează weights MI calculate")
    args = parser.parse_args()

    print(f"Încarc dataset din {DATASET_PATH}...")
    recipes = load_recipes(DATASET_PATH)

    if args.sample > 0:
        recipes = recipes[:args.sample]

    print(f"Rețete: {len(recipes)} | Utilizatori sintetici: {len(SYNTHETIC_USERS)}")

    # Calculăm weights din dataset (o singură dată)
    print("Calculez weights din Mutual Information...")
    weights = compute_feature_mi(recipes)

    if args.show_weights:
        print("\nWeights calculate (MI-based, normalizate la [0.3, 3.0]):")
        for qid, w in sorted(weights.items(), key=lambda x: -x[1]):
            print(f"  {qid:<30} {w:.4f}")

    print(f"\nMax întrebări: {MAX_QUESTIONS} | Entropy stop: {ENTROPY_STOP_THRESHOLD:.3f} biți")
    print("=" * 70)

    results: list[dict[str, Any]] = []

    import os
    if os.path.exists("question_selection.json"):
        os.remove("question_selection.json")

    for user in SYNTHETIC_USERS:
        if args.verbose:
            print(f"\n  {user['name']}:")

        export_viz = "Mexican spicy tortilla meatless" in user["name"]  # schimbă cu userul dorit
        result = run_session(recipes, user, weights, verbose=args.verbose,
                           export_visualization=export_viz)
        result["user"] = user["name"]
        results.append(result)


        print(
            f"  {user['name'][:35]:<35} | "
            f"Q: {result['questions_asked']:>2} | "
            f"Rel: {result['total_relevant']:>4} | "
            f"NDCG: {result['ndcg_10']:.3f} | "
            f"Overlap: {result['overlap_10']:.2f} | "
            f"H_final: {result['entropy_final']:.2f}b"
        )


        if args.show_top:
            for item in result["top"][:5]:
                print(f"        {item['name'][:50]} | {item['score_x_uniform']:.1f}x")

    avg_q       = sum(r["questions_asked"] for r in results) / len(results)
    avg_ndcg    = sum(r["ndcg_10"]         for r in results) / len(results)
    avg_overlap = sum(r["overlap_10"]       for r in results) / len(results)
    avg_h       = sum(r["entropy_final"]    for r in results) / len(results)

    print("=" * 70)
    print(f"MEDII: Q={avg_q:.2f} | NDCG={avg_ndcg:.4f} | Overlap={avg_overlap:.4f} | H_final={avg_h:.2f}b")




if __name__ == "__main__":
    main()