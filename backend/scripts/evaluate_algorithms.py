# backend/scripts/evaluate_algorithms.py
"""
Evaluare comparativă între 3 variante de selecție a întrebărilor:
  A - ordine fixă (fără EER)
  B - random după cele 3 fixe
  C - EER (sistemul actual)

Pe 2 tipuri de utilizatori:
  full  - răspund la toate întrebările
  partial - 50% din booleeni sunt skip
"""
from __future__ import annotations

import importlib
import hashlib
import math
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Any

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import app.recommender.engine
importlib.reload(app.recommender.engine)

from app.database import get_supabase_admin
from app.recommender.engine import (
    MAX_QUESTIONS,
    BayesianSession,
    QUESTION_BANK,
    _ADAPTIVE_QS,
    _FIXED_QS,
    compute_feature_mi,
    question_by_id,
    select_next_question,
)

RANDOM_SEED = 42
TOP_N_EVAL = 10


# ── UTILIZATORI SINTETICI ─────────────────────────────────────────────────────
@dataclass
class SyntheticUser:
    name: str
    profile: dict[str, Any]


USERS = [
    SyntheticUser("Italian chicken pasta", {
        "meal_type": "lunch_dinner",
        "protein_type": ["chicken"],
        "cuisine": ["italian"],
        "is_spicy": "no", "is_sweet": "no", "is_quick": "no",
        "needs_oven": "yes", "needs_stovetop": "no", "is_no_cook": "no",
        "has_pasta": "yes", "has_rice": "no", "has_potato": "no",
        "has_tomato_base": "no", "has_cream_base": "yes", "has_cheese": "yes",
        "has_broth_base": "no", "has_mushroom": "no", "has_leafy_greens": "no",
        "has_beans_legumes": "no", "has_fruit": "no", "has_nuts": "no",
        "has_chocolate": "no", "has_asian_sauce": "no",
    }),
    SyntheticUser("Spicy Asian chicken soup", {
        "meal_type": "soup",
        "protein_type": ["chicken"],
        "cuisine": ["asian"],
        "is_spicy": "yes", "is_sweet": "no", "is_quick": "yes",
        "needs_oven": "no", "needs_stovetop": "yes", "is_no_cook": "no",
        "has_pasta": "no", "has_rice": "no", "has_potato": "no",
        "has_tomato_base": "no", "has_cream_base": "no", "has_cheese": "no",
        "has_broth_base": "yes", "has_mushroom": "yes", "has_leafy_greens": "yes",
        "has_beans_legumes": "no", "has_fruit": "no", "has_nuts": "no",
        "has_chocolate": "no", "has_asian_sauce": "yes",
    }),
    SyntheticUser("Fast fruit breakfast", {
        "meal_type": "breakfast",
        "protein_type": ["meatless"],
        "cuisine": ["american"],
        "is_spicy": "no", "is_sweet": "yes", "is_quick": "yes",
        "needs_oven": "no", "needs_stovetop": "no", "is_no_cook": "yes",
        "has_pasta": "no", "has_rice": "no", "has_potato": "no",
        "has_tomato_base": "no", "has_cream_base": "yes", "has_cheese": "no",
        "has_broth_base": "no", "has_mushroom": "no", "has_leafy_greens": "no",
        "has_beans_legumes": "no", "has_fruit": "yes", "has_nuts": "yes",
        "has_chocolate": "no", "has_asian_sauce": "no",
    }),
    SyntheticUser("Chocolate dessert", {
        "meal_type": "dessert",
        "protein_type": ["meatless"],
        "cuisine": ["french"],
        "is_spicy": "no", "is_sweet": "yes", "is_quick": "no",
        "needs_oven": "yes", "needs_stovetop": "no", "is_no_cook": "no",
        "has_pasta": "no", "has_rice": "no", "has_potato": "no",
        "has_tomato_base": "no", "has_cream_base": "yes", "has_cheese": "no",
        "has_broth_base": "no", "has_mushroom": "no", "has_leafy_greens": "no",
        "has_beans_legumes": "no", "has_fruit": "no", "has_nuts": "yes",
        "has_chocolate": "yes", "has_asian_sauce": "no",
    }),
    SyntheticUser("Mediterranean salad", {
        "meal_type": "salad_side",
        "protein_type": ["meatless"],
        "cuisine": ["mediterranean"],
        "is_spicy": "no", "is_sweet": "no", "is_quick": "yes",
        "needs_oven": "no", "needs_stovetop": "no", "is_no_cook": "yes",
        "has_pasta": "no", "has_rice": "no", "has_potato": "no",
        "has_tomato_base": "yes", "has_cream_base": "no", "has_cheese": "yes",
        "has_broth_base": "no", "has_mushroom": "no", "has_leafy_greens": "yes",
        "has_beans_legumes": "yes", "has_fruit": "no", "has_nuts": "no",
        "has_chocolate": "no", "has_asian_sauce": "no",
    }),
    SyntheticUser("Mexican beef dinner", {
        "meal_type": "lunch_dinner",
        "protein_type": ["beef_pork"],
        "cuisine": ["mexican"],
        "is_spicy": "yes", "is_sweet": "no", "is_quick": "no",
        "needs_oven": "no", "needs_stovetop": "yes", "is_no_cook": "no",
        "has_pasta": "no", "has_rice": "yes", "has_potato": "no",
        "has_tomato_base": "yes", "has_cream_base": "no", "has_cheese": "yes",
        "has_broth_base": "no", "has_mushroom": "no", "has_leafy_greens": "no",
        "has_beans_legumes": "yes", "has_fruit": "no", "has_nuts": "no",
        "has_chocolate": "no", "has_asian_sauce": "no",
    }),
]


# ── GENERARE UTILIZATORI CU SKIP PARȚIAL ─────────────────────────────────────
def make_partial_user(user: SyntheticUser, skip_ratio: float = 0.5) -> SyntheticUser:
    """
    Creează o versiune parțială a utilizatorului unde skip_ratio din
    întrebările booleene adaptive sunt înlocuite cu skip.
    Fixele (meal_type, protein_type, cuisine) rămân neschimbate.
    """
    rng = random.Random(RANDOM_SEED + stable_seed(user.name))
    boolean_keys = [
        q["id"] for q in _ADAPTIVE_QS if q["type"] == "boolean"
    ]
    skip_keys = set(rng.sample(boolean_keys, k=int(len(boolean_keys) * skip_ratio)))

    partial_profile = {}
    for k, v in user.profile.items():
        if k in skip_keys:
            partial_profile[k] = "skip"
        else:
            partial_profile[k] = v

    return SyntheticUser(f"{user.name} [partial]", partial_profile)


def stable_seed(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


# ── RĂSPUNS DIN PROFIL ────────────────────────────────────────────────────────
def answer_for(question: dict, profile: dict) -> Any:
    qid = question["id"]
    q_type = question["type"]

    val = profile.get(qid)
    if val is None:
        val = profile.get(question["feature"])

    if val is None:
        return "skip" if q_type == "boolean" else (["any"] if q_type == "multiselect" else "skip")

    if val == "skip":
        return "skip"

    if q_type == "boolean":
        if val in ("yes", "no"):
            return val
        return "yes" if bool(val) else "no"

    if q_type == "multiselect":
        if val in (["any"], "any"):
            return ["any"]
        return val if isinstance(val, list) else [val]

    return val


# ── SELECȚIE ÎNTREBĂRI — VARIANTELE A și B ────────────────────────────────────
def select_fixed_order(session: BayesianSession) -> dict | None:
    """Varianta A: ordinea fixă din QUESTION_BANK, fără EER."""
    asked = session.asked_ids
    for q in _FIXED_QS:
        if q["id"] not in asked:
            return q
    for q in _ADAPTIVE_QS:
        if q["id"] not in asked:
            return q
    return None


def select_random(session: BayesianSession, rng: random.Random) -> dict | None:
    """Varianta B: random după cele 3 fixe."""
    asked = session.asked_ids
    for q in _FIXED_QS:
        if q["id"] not in asked:
            return q
    candidates = [q for q in _ADAPTIVE_QS if q["id"] not in asked]
    if not candidates:
        return None
    return rng.choice(candidates)


# ── RULARE SESIUNE ────────────────────────────────────────────────────────────
@dataclass
class SessionResult:
    user_name: str
    variant: str
    user_type: str
    questions_asked: int
    converged: bool
    entropy_initial: float
    entropy_final: float
    entropy_total_drop: float
    entropy_per_question: float
    best_top10_match: float
    avg_top10_match: float
    strong_matches_top10: int
    ndcg_10: float
    entropy_trace: list[float] = field(default_factory=list)


def recipe_profile_match_score(recipe: dict[str, Any], profile: dict[str, Any]) -> float:
    checked = 0
    matched = 0

    for question in QUESTION_BANK:
        qid = question["id"]
        feature = question["feature"]
        expected = profile.get(qid, profile.get(feature))

        if expected is None or expected == "skip" or expected == "any" or expected == ["any"]:
            continue

        checked += 1
        if question["type"] == "categorical":
            ok = recipe.get(feature) == expected
        elif question["type"] == "multiselect":
            selected = expected if isinstance(expected, list) else [expected]
            ok = recipe.get(feature) in selected
        else:
            expected_bool = expected == "yes" if expected in ("yes", "no") else bool(expected)
            ok = bool(recipe.get(feature)) == expected_bool

        if ok:
            matched += 1

    if checked == 0:
        return 0.0
    return matched / checked


def dcg(relevances: list[float]) -> float:
    return sum(rel / math.log2(rank + 2) for rank, rel in enumerate(relevances))


def ndcg_at_k(
    ranked_recipes: list[dict[str, Any]],
    all_recipes: list[dict[str, Any]],
    profile: dict[str, Any],
    k: int,
) -> float:
    actual = [recipe_profile_match_score(recipe, profile) for recipe in ranked_recipes[:k]]
    ideal = sorted(
        (recipe_profile_match_score(recipe, profile) for recipe in all_recipes),
        reverse=True,
    )[:k]
    ideal_dcg = dcg(ideal)
    if ideal_dcg <= 0:
        return 0.0
    return dcg(actual) / ideal_dcg


def run_session(
    user: SyntheticUser,
    eval_profile: dict[str, Any],
    recipes: list[dict],
    weights: dict,
    variant: str,
    user_type: str,
    rng: random.Random,
) -> SessionResult:
    session = BayesianSession(recipes, weights)
    h_initial = session.entropy()
    entropy_trace = [h_initial]

    while not session.should_stop() and session.q < MAX_QUESTIONS:
        if variant == "A":
            q = select_fixed_order(session)
        elif variant == "B":
            q = select_random(session, rng)
        else:
            q = select_next_question(session)

        if q is None:
            break

        answer = answer_for(q, user.profile)
        session.update(q, answer)
        entropy_trace.append(session.entropy())

    h_final = session.entropy()
    n = session.q
    drop = h_initial - h_final
    top_recipes = [recipe for recipe, _ in session.top(TOP_N_EVAL, min_match_score=0)]
    top_matches = [recipe_profile_match_score(recipe, eval_profile) for recipe in top_recipes]

    return SessionResult(
        user_name=user.name,
        variant=variant,
        user_type=user_type,
        questions_asked=n,
        converged=session.should_stop() and session.q < MAX_QUESTIONS,
        entropy_initial=round(h_initial, 3),
        entropy_final=round(h_final, 3),
        entropy_total_drop=round(drop, 3),
        entropy_per_question=round(drop / n if n > 0 else 0, 3),
        best_top10_match=round(max(top_matches, default=0.0) * 100, 1),
        avg_top10_match=round((sum(top_matches) / len(top_matches)) * 100 if top_matches else 0.0, 1),
        strong_matches_top10=sum(1 for score in top_matches if score >= 0.90),
        ndcg_10=round(ndcg_at_k(top_recipes, recipes, eval_profile, TOP_N_EVAL), 4),
        entropy_trace=entropy_trace,
    )


# ── RAPORT LEGACY: entropy-only ───────────────────────────────────────────────
def print_entropy_report_legacy(results: list[SessionResult]) -> None:
    variants = ["A", "B", "C"]
    user_types = ["full", "partial"]
    labels = {"A": "Fixed order", "B": "Random", "C": "EER (actual)"}

    print(f"\n{'='*80}")
    print("EVALUARE COMPARATIVĂ — 3 variante × 2 tipuri utilizatori")
    print(f"{'='*80}\n")

    for utype in user_types:
        print(f"── Utilizatori {utype.upper()} ──────────────────────────────────")
        print(f"{'Variant':<16} {'Avg Q':>7} {'Conv%':>7} {'ΔH total':>10} {'ΔH/Q':>8}")
        print("-" * 52)

        for v in variants:
            subset = [r for r in results if r.variant == v and r.user_type == utype]
            if not subset:
                continue
            avg_q = sum(r.questions_asked for r in subset) / len(subset)
            conv = sum(1 for r in subset if r.converged) / len(subset) * 100
            avg_drop = sum(r.entropy_total_drop for r in subset) / len(subset)
            avg_per_q = sum(r.entropy_per_question for r in subset) / len(subset)
            print(f"{labels[v]:<16} {avg_q:>7.1f} {conv:>6.0f}% {avg_drop:>10.3f} {avg_per_q:>8.3f}")
        print()

    print(f"{'='*80}")
    print("DETALII PER UTILIZATOR\n")

    for user in USERS:
        for utype in user_types:
            name = user.name if utype == "full" else f"{user.name} [partial]"
            print(f"  {name}")
            print(f"  {'Variant':<16} {'Q':>4} {'Conv':>5} {'H init':>7} {'H final':>8} {'ΔH':>7} {'ΔH/Q':>7}")
            print(f"  {'-'*56}")
            for v in variants:
                r = next((x for x in results
                          if x.variant == v and x.user_type == utype
                          and x.user_name.replace(" [partial]", "") == user.name), None)
                if r:
                    print(f"  {labels[v]:<16} {r.questions_asked:>4} "
                          f"{'DA' if r.converged else 'nu':>5} "
                          f"{r.entropy_initial:>7.2f} {r.entropy_final:>8.2f} "
                          f"{r.entropy_total_drop:>7.3f} {r.entropy_per_question:>7.3f}")
            print()


# ── RAPORT ────────────────────────────────────────────────────────────────────
def print_report(results: list[SessionResult]) -> None:
    variants = ["A", "B", "C"]
    user_types = ["full", "partial"]
    labels = {"A": "Fixed order", "B": "Random", "C": "EER (actual)"}

    print(f"\n{'='*100}")
    print("EVALUARE COMPARATIVA - entropy + calitatea topului")
    print(f"{'='*100}\n")

    for utype in user_types:
        print(f"-- Utilizatori {utype.upper()} --")
        print(
            f"{'Variant':<16} {'Avg Q':>7} {'Conv%':>7} {'dH total':>9} {'dH/Q':>8} "
            f"{'Best':>8} {'AvgTop':>8} {'Strong':>7} {'NDCG':>7}"
        )
        print("-" * 96)

        for variant in variants:
            subset = [r for r in results if r.variant == variant and r.user_type == utype]
            if not subset:
                continue
            avg_q = sum(r.questions_asked for r in subset) / len(subset)
            conv = sum(1 for r in subset if r.converged) / len(subset) * 100
            avg_drop = sum(r.entropy_total_drop for r in subset) / len(subset)
            avg_per_q = sum(r.entropy_per_question for r in subset) / len(subset)
            best_match = sum(r.best_top10_match for r in subset) / len(subset)
            avg_top = sum(r.avg_top10_match for r in subset) / len(subset)
            strong = sum(r.strong_matches_top10 for r in subset) / len(subset)
            ndcg = sum(r.ndcg_10 for r in subset) / len(subset)
            print(
                f"{labels[variant]:<16} {avg_q:>7.1f} {conv:>6.0f}% {avg_drop:>9.3f} "
                f"{avg_per_q:>8.3f} {best_match:>7.1f}% {avg_top:>7.1f}% "
                f"{strong:>7.1f} {ndcg:>7.3f}"
            )
        print()

    print(f"{'='*100}")
    print("DETALII PER UTILIZATOR\n")

    for base_user in USERS:
        for utype in user_types:
            print(f"  {base_user.name if utype == 'full' else f'{base_user.name} [partial]'}")
            print(
                f"  {'Variant':<16} {'Q':>4} {'Conv':>5} {'H final':>8} {'dH/Q':>7} "
                f"{'Best':>7} {'AvgTop':>7} {'Strg':>5} {'NDCG':>6}"
            )
            print(f"  {'-'*82}")
            for variant in variants:
                result = next(
                    (
                        r for r in results
                        if r.variant == variant
                        and r.user_type == utype
                        and r.user_name.replace(" [partial]", "") == base_user.name
                    ),
                    None,
                )
                if result:
                    print(
                        f"  {labels[variant]:<16} {result.questions_asked:>4} "
                        f"{'DA' if result.converged else 'nu':>5} "
                        f"{result.entropy_final:>8.2f} {result.entropy_per_question:>7.3f} "
                        f"{result.best_top10_match:>6.1f}% {result.avg_top10_match:>6.1f}% "
                        f"{result.strong_matches_top10:>5} {result.ndcg_10:>6.3f}"
                    )
            print()


def main():
    admin = get_supabase_admin()
    recipes = []
    offset = 0
    while True:
        page = admin.table("recipes").select("*").range(offset, offset + 999).execute().data or []
        recipes.extend(page)
        if len(page) < 1000:
            break
        offset += 1000
    print(f"Loaded {len(recipes)} recipes")

    weights = compute_feature_mi(recipes)
    rng = random.Random(RANDOM_SEED)

    all_users = []
    for u in USERS:
        all_users.append(("full", u, u.profile))
        all_users.append(("partial", make_partial_user(u), u.profile))

    results = []
    for utype, user, eval_profile in all_users:
        for variant in ["A", "B", "C"]:
            r = run_session(user, eval_profile, recipes, weights, variant, utype, rng)
            results.append(r)

    print_report(results)


if __name__ == "__main__":
    main()
