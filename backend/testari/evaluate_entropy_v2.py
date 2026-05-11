from __future__ import annotations
import argparse
import json
import math
import numpy as np
from collections import Counter
from typing import Any

# ── CONFIG ─────────────────────────────────────────────
DATASET_PATH = r"D:\folosire_api_claude\dataset_llm_labeled.json"
MAX_QUESTIONS = 9
MIN_QUESTIONS_BEFORE_STOP = 3
TOP_N = 10

# Bayesian
P_CORRECT = 0.90
P_NOISE = 0.05
ENTROPY_STOP_THRESHOLD = math.log2(50)

# ── QUESTION BANK ──────────────────────────────────────
QUESTION_BANK = [
    {"id": "meal_type", "type": "categorical", "feature": "meal_type",
     "fixed": True, "order": 1,
     "options": ["appetizer","breakfast","dessert","drink","lunch_dinner","salad_side","snack","soup","condiment"]},

    {"id": "is_chicken", "type": "boolean", "feature": "protein_type", "feature_value": "chicken"},
    {"id": "is_beef", "type": "boolean", "feature": "protein_type", "feature_value": "beef_pork"},
    {"id": "is_fish", "type": "boolean", "feature": "protein_type", "feature_value": "fish_seafood"},
    {"id": "is_meatless", "type": "boolean", "feature": "protein_type", "feature_value": "meatless"},
    {"id": "is_italian", "type": "boolean", "feature": "cuisine", "feature_value": "italian"},
    {"id": "is_asian", "type": "boolean", "feature": "cuisine", "feature_value": "asian"},
    {"id": "is_mexican", "type": "boolean", "feature": "cuisine", "feature_value": "mexican"},
    {"id": "is_spicy", "type": "boolean", "feature": "is_spicy", "feature_value": True},
    {"id": "is_sweet", "type": "boolean", "feature": "is_sweet", "feature_value": True},
    {"id": "is_quick", "type": "boolean", "feature": "is_quick", "feature_value": True},
]

# ── HELPERS ────────────────────────────────────────────
def get_feature_value(recipe: dict[str, Any], question: dict[str, Any]) -> Any:
    f = question["feature"]
    fv = question.get("feature_value")
    val = recipe.get(f)

    if fv is None:
        return val
    if isinstance(fv, bool):
        return bool(val) == fv
    return val == fv


def recipe_matches_user(recipe: dict[str, Any], user: dict[str, Any]) -> bool:
    for k, v in user.items():
        if k == "name":
            continue
        if recipe.get(k) != v:
            return False
    return True


def simulate_answer(user: dict[str, Any], question: dict[str, Any]) -> list[str]:
    val = user.get(question["feature"])
    fv = question.get("feature_value")

    if question["type"] == "categorical":
        return [val] if val is not None else ["skip"]

    if fv is None:
        if val is None:
            return ["skip"]
        return ["yes"] if bool(val) else ["no"]

    if val is None:
        return ["skip"]

    return ["yes"] if val == fv else ["no"]


# ── NDCG CORECT ───────────────────────────────────────
def compute_ndcg_correct(
    ranked: list[tuple[dict[str, Any], float]],
    all_recipes: list[dict[str, Any]],
    user: dict[str, Any],
    k: int = 10
) -> float:
    relevances = [
        1.0 if recipe_matches_user(r, user) else 0.0
        for r, _ in ranked[:k]
    ]

    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))

    total_relevant = sum(
        1 for r in all_recipes if recipe_matches_user(r, user)
    )

    ideal = [1.0] * min(k, total_relevant)
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal))

    return dcg / idcg if idcg > 0 else 0.0


def compute_overlap(
    ranked: list[tuple[dict[str, Any], float]],
    user: dict[str, Any],
    k: int = 10
) -> float:
    top = ranked[:k]
    if not top:
        return 0.0
    matches = sum(1 for r, _ in top if recipe_matches_user(r, user))
    return matches / min(k, len(top))


# ── BAYESIAN ───────────────────────────────────────────
def compute_likelihood(recipe: dict[str, Any], question: dict[str, Any], answer: str) -> float:
    if answer == "skip":
        return 1.0

    match = get_feature_value(recipe, question)

    if answer == "yes":
        return P_CORRECT if match else P_NOISE
    elif answer == "no":
        return P_CORRECT if not match else P_NOISE

    # Pentru întrebări categoriale, answer este valoarea aleasă, ex. lunch_dinner.
    recipe_val = recipe.get(question["feature"])
    return P_CORRECT if recipe_val == answer else P_NOISE


class BayesianSession:
    def __init__(self, recipes: list[dict[str, Any]]) -> None:
        self.recipes = recipes
        self.n = len(recipes)
        self.log_probs = np.full(self.n, -math.log(self.n))
        self.answers: dict[str, str] = {}
        self.asked_ids: set[str] = set()
        self.q = 0

    def probs(self) -> np.ndarray:
        p = np.exp(self.log_probs - self.log_probs.max())
        return p / p.sum()

    def update(self, question: dict[str, Any], answer: str) -> None:
        self.q += 1
        self.answers[question["id"]] = answer
        self.asked_ids.add(question["id"])

        likelihoods = np.array([
            compute_likelihood(r, question, answer)
            for r in self.recipes
        ])

        self.log_probs += np.log(likelihoods + 1e-10)
        self.log_probs -= self.log_probs.max()

    def entropy(self) -> float:
        p = self.probs()
        p = p[p > 1e-10]
        return float(-np.sum(p * np.log2(p)))

    def should_stop(self) -> bool:
        if self.q < MIN_QUESTIONS_BEFORE_STOP:
            return False
        return self.q >= MAX_QUESTIONS or self.entropy() < ENTROPY_STOP_THRESHOLD

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
                compute_likelihood(r, question, answer)
                for r in self.recipes
            ])

            p_answer = float(np.dot(likelihoods, p))
            if p_answer < 1e-12:
                continue

            log_updated = self.log_probs + np.log(likelihoods + 1e-10)
            log_updated -= log_updated.max()

            p_updated = np.exp(log_updated)
            p_updated /= p_updated.sum()

            p_nz = p_updated[p_updated > 1e-10]
            h_after = float(-np.sum(p_nz * np.log2(p_nz)))

            expected_h += p_answer * h_after

        return current_h - expected_h

    def top(self, n: int = 10) -> list[tuple[dict[str, Any], float]]:
        p = self.probs()
        idx = p.argsort()[-n:][::-1]
        max_p = p[idx[0]] if len(idx) > 0 else 1.0
        return [(self.recipes[i], round(float(p[i] / max_p) * 100, 2)) for i in idx]


def select_next_question(session: BayesianSession) -> dict[str, Any] | None:
    fixed_qs = sorted(
        [q for q in QUESTION_BANK if q.get("fixed")],
        key=lambda q: q.get("order", 99)
    )

    if session.q < len(fixed_qs):
        return fixed_qs[session.q]

    candidates = [
        q for q in QUESTION_BANK
        if not q.get("fixed") and q["id"] not in session.asked_ids
    ]

    if not candidates:
        return None

    return max(candidates, key=lambda q: session.expected_entropy_reduction(q))


# ── MAIN ──────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    with open(DATASET_PATH, encoding="utf-8") as f:
        raw = json.load(f)

    recipes: list[dict[str, Any]] = []

    for i, r in enumerate(raw):
        fts = r.get("llm_features")

        if not fts or r.get("llm_failed"):
            continue

        recipes.append({
            "id": i,
            "name": r.get("Name", ""),
            "meal_type": fts.get("meal_type"),
            "protein_type": fts.get("protein_type"),
            "cuisine": fts.get("cuisine"),
            "is_spicy": fts.get("is_spicy", False),
            "is_sweet": fts.get("is_sweet", False),
            "is_quick": fts.get("is_quick", False),
        })

    user = {
    "meal_type": "lunch_dinner",
    "protein_type": "chicken",
    "cuisine": "italian",
    "is_spicy": False,
    "is_sweet": False,
    "is_quick": False
    }
    session = BayesianSession(recipes)

    while not session.should_stop():
        q = select_next_question(session)
        if q is None:
            break

        ans = simulate_answer(user, q)[0]
        session.update(q, ans)

        if args.verbose:
            print(f"Q{session.q}: {q['id']} -> {ans} | H={session.entropy():.4f}")

    ranked = session.top(TOP_N)

    ndcg = compute_ndcg_correct(ranked, recipes, user, TOP_N)
    overlap = compute_overlap(ranked, user, TOP_N)

    print("=" * 60)
    print("BAYESIAN EVALUATION")
    print("=" * 60)
    print(f"Recipes: {len(recipes)}")
    print(f"Questions asked: {session.q}")
    print(f"NDCG@10: {ndcg:.4f}")
    print(f"Overlap@10: {overlap:.4f}")
    print(f"Entropy: {session.entropy():.4f}")
    print()
    print("Top recommendations:")
    for i, (recipe, score) in enumerate(ranked, start=1):
        print(f"{i:02d}. {recipe.get('name', '')} | score={score:.2f}% | "
              f"meal={recipe.get('meal_type')} | protein={recipe.get('protein_type')} | cuisine={recipe.get('cuisine')}")


if __name__ == "__main__":
    main()
