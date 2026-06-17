# backend/scripts/compare_fixed_budget.py
"""
Comparație la buget fix de întrebări: Marginal vs JMIM vs Bayesian EER
Toți algoritmii pun exact K întrebări, apoi evaluăm NDCG@10.
"""
from __future__ import annotations

import importlib
import math
import os
import sys
from collections import Counter
from typing import Any

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import app.services.recommendations.bayesian
importlib.reload(app.services.recommendations.bayesian)

from app.database import get_supabase_admin
from app.services.recommendations.bayesian.config import (
    QUESTION_BANK,
    _ADAPTIVE_QS,
    _FIXED_QS,
    question_by_id,
)
from app.services.recommendations.bayesian.features import (
    compute_feature_mi,
    get_feature_value_bool,
)
from app.services.recommendations.bayesian.questions import (
    select_next_question,
)
from app.services.recommendations.bayesian.session import BayesianSession
from app.recommender.semantic_rerank import semantic_candidate_pool, semantic_rerank

TOP_N = 10
BUDGETS = [4, 6, 8, 10, 12]  # bugete de întrebări de evaluat
SCENARIOS = ["normal", "fixed_only", "unknown_adaptive", "text_query"]


# ── UTILIZATORI ───────────────────────────────────────────────────────────────
USERS = [
    {"name": "Italian chicken pasta",    "meal_type": "lunch_dinner", "protein_type": ["chicken"],   "cuisine": ["italian"],       "has_pasta": "yes", "has_cream_base": "yes", "is_spicy": "no"},
    {"name": "Spicy Asian chicken soup", "meal_type": "soup",         "protein_type": ["chicken"],   "cuisine": ["asian"],         "is_spicy": "yes",  "has_broth_base": "yes", "has_asian_sauce": "yes"},
    {"name": "Fast fruit breakfast",     "meal_type": "breakfast",    "protein_type": ["meatless"],  "cuisine": ["american"],      "is_sweet": "yes",  "is_quick": "yes",       "is_no_cook": "yes"},
    {"name": "Chocolate dessert",        "meal_type": "dessert",      "protein_type": ["meatless"],  "cuisine": ["french"],        "has_chocolate": "yes", "has_nuts": "yes",   "needs_oven": "yes"},
    {"name": "Mediterranean salad",      "meal_type": "salad_side",   "protein_type": ["meatless"],  "cuisine": ["mediterranean"], "is_no_cook": "yes","has_tomato_base": "yes"},
    {"name": "Mexican beef dinner",      "meal_type": "lunch_dinner", "protein_type": ["beef_pork"], "cuisine": ["mexican"],       "is_spicy": "yes",  "has_rice": "yes",       "has_cheese": "yes"},
    {"name": "Quick American beef",      "meal_type": "lunch_dinner", "protein_type": ["beef_pork"], "cuisine": ["american"],      "is_quick": "yes",  "has_tomato_base": "yes"},
    {"name": "Indian spicy meatless",    "meal_type": "lunch_dinner", "protein_type": ["meatless"],  "cuisine": ["indian"],        "is_spicy": "yes",  "has_beans_legumes": "yes"},
    {"name": "Creamy mushroom soup",     "meal_type": "soup",         "protein_type": ["meatless"],  "cuisine": ["american"],      "has_cream_base": "yes", "has_mushroom": "yes"},
    {"name": "Breakfast oven sweet",     "meal_type": "breakfast",    "protein_type": ["meatless"],  "cuisine": ["american"],      "needs_oven": "yes","is_sweet": "yes"},
]

SEMANTIC_QUERIES = {
    "Italian chicken pasta": "creamy Italian chicken pasta",
    "Spicy Asian chicken soup": "spicy Asian chicken soup with broth",
    "Fast fruit breakfast": "quick sweet breakfast with fruit and nuts",
    "Chocolate dessert": "chocolate dessert with nuts baked in the oven",
    "Mediterranean salad": "fresh Mediterranean salad with tomato greens and cheese",
    "Mexican beef dinner": "spicy Mexican beef dinner with rice beans and cheese",
    "Quick American beef": "quick American beef dinner with tomato",
    "Indian spicy meatless": "spicy Indian meatless beans legumes dinner",
    "Creamy mushroom soup": "creamy mushroom soup",
    "Breakfast oven sweet": "sweet baked breakfast",
}


# ── RĂSPUNS DIN PROFIL ────────────────────────────────────────────────────────
def answer_for(question: dict, profile: dict) -> Any:
    qid = question["id"]
    q_type = question["type"]
    val = profile.get(qid, profile.get(question["feature"]))
    if val is None:
        return "skip" if q_type == "boolean" else (["any"] if q_type == "multiselect" else "skip")
    if val in ("skip", "unknown"):
        return val
    if q_type == "boolean":
        return val if val in ("yes", "no") else ("yes" if bool(val) else "no")
    if q_type == "multiselect":
        return val if isinstance(val, list) else [val]
    return val


def profile_for_scenario(profile: dict[str, Any], scenario: str) -> dict[str, Any]:
    fixed_ids = {q["id"] for q in _FIXED_QS}
    if scenario == "normal":
        return dict(profile)
    if scenario == "fixed_only":
        return {key: value for key, value in profile.items() if key in fixed_ids}
    if scenario == "unknown_adaptive":
        out = {key: value for key, value in profile.items() if key in fixed_ids}
        for q in _ADAPTIVE_QS:
            out[q["id"]] = "unknown"
        return out
    if scenario == "text_query":
        out = {key: value for key, value in profile.items() if key in fixed_ids}
        out["semantic_query"] = SEMANTIC_QUERIES[profile["name"]]
        return out
    raise ValueError(f"Unknown scenario: {scenario}")


# ── METRICI ───────────────────────────────────────────────────────────────────
def recipe_profile_match(recipe: dict, profile: dict) -> float:
    checked = matched = 0
    for q in QUESTION_BANK:
        expected = profile.get(q["id"], profile.get(q["feature"]))
        if expected is None or expected in ("skip", "any", ["any"]):
            continue
        checked += 1
        if q["type"] == "categorical":
            ok = recipe.get(q["feature"]) == expected
        elif q["type"] == "multiselect":
            selected = expected if isinstance(expected, list) else [expected]
            ok = recipe.get(q["feature"]) in selected
        else:
            exp_bool = expected == "yes" if expected in ("yes", "no") else bool(expected)
            ok = bool(recipe.get(q["feature"])) == exp_bool
        if ok:
            matched += 1
    return matched / checked if checked > 0 else 0.0


def ndcg_at_k(ranked: list[dict], all_recipes: list[dict], profile: dict, k: int = 10) -> float:
    rels = [recipe_profile_match(r, profile) for r in ranked[:k]]
    ideal = sorted([recipe_profile_match(r, profile) for r in all_recipes], reverse=True)[:k]
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rels))
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


# ── RANKING COMUN (după răspunsuri colectate) ─────────────────────────────────
def rank_via_posterior(recipes: list[dict], answers: dict, weights: dict) -> list[dict]:
    """Ranking uniform prin posterior Bayesian pentru toți algoritmii."""
    session = BayesianSession(recipes, weights)
    for qid, ans in answers.items():
        q = question_by_id(qid)
        if q:
            session.update(q, ans)
    return [r for r, _ in session.top(TOP_N, min_match_score=0)]


# ── MARGINAL ENTROPY ──────────────────────────────────────────────────────────
def marginal_entropy(question: dict, recipes: list[dict]) -> float:
    if not recipes:
        return 0.0
    if question["type"] == "boolean":
        vals = [str(get_feature_value_bool(r, question)) for r in recipes]
    else:
        vals = [str(r.get(question["feature"], "unknown")) for r in recipes]
    counts = Counter(vals)
    n = len(vals)
    return -sum((c / n) * math.log2(c / n) for c in counts.values() if c > 0)


def run_marginal_budget(recipes: list[dict], profile: dict, budget: int, weights) -> list[dict]:
    answers: dict[str, Any] = {}
    asked: set[str] = set()

    for q in _FIXED_QS:
        if len(asked) >= budget:
            break
        ans = answer_for(q, profile)
        answers[q["id"]] = ans
        asked.add(q["id"])

    while len(asked) < budget:
        candidates = [q for q in _ADAPTIVE_QS if q["id"] not in asked]
        if not candidates:
            break
        best = max(candidates, key=lambda q: marginal_entropy(q, recipes))
        ans = answer_for(best, profile)
        answers[best["id"]] = ans
        asked.add(best["id"])

    return rank_via_posterior(recipes, answers, weights)


# ── JMIM ──────────────────────────────────────────────────────────────────────
def jmim_score(question: dict, recipes: list[dict], asked_qs: list[dict]) -> float:
    if not asked_qs:
        return marginal_entropy(question, recipes)
    if not recipes:
        return 0.0
    n = len(recipes)

    def vals(q: dict) -> list[str]:
        if q["type"] == "boolean":
            return [str(get_feature_value_bool(r, q)) for r in recipes]
        return [str(r.get(q["feature"], "unknown")) for r in recipes]

    xi = vals(question)
    min_jmi = float("inf")
    for aq in asked_qs:
        xs = vals(aq)
        joint = Counter(zip(xi, xs))
        ci, cs = Counter(xi), Counter(xs)
        mi = sum(
            (cnt / n) * math.log2((cnt / n) / ((ci[a] / n) * (cs[b] / n)))
            for (a, b), cnt in joint.items()
            if cnt > 0 and ci[a] > 0 and cs[b] > 0
        )
        min_jmi = min(min_jmi, max(mi, 1e-6))

    return min_jmi if min_jmi != float("inf") else marginal_entropy(question, recipes)


def run_jmim_budget(recipes: list[dict], profile: dict, budget: int, weights) -> list[dict]:
    answers: dict[str, Any] = {}
    asked: set[str] = set()
    asked_qs: list[dict] = []

    for q in _FIXED_QS:
        if len(asked) >= budget:
            break
        ans = answer_for(q, profile)
        answers[q["id"]] = ans
        asked.add(q["id"])

    while len(asked) < budget:
        candidates = [q for q in _ADAPTIVE_QS if q["id"] not in asked]
        if not candidates:
            break
        best = max(candidates, key=lambda q: jmim_score(q, recipes, asked_qs))
        ans = answer_for(best, profile)
        answers[best["id"]] = ans
        asked.add(best["id"])
        asked_qs.append(best)

    return rank_via_posterior(recipes, answers, weights)


# ── BAYESIAN EER ──────────────────────────────────────────────────────────────
def run_eer_budget(recipes: list[dict], profile: dict, weights: dict, budget: int) -> list[dict]:
    session = BayesianSession(recipes, weights)

    while session.q < budget:
        q = select_next_question(session)
        if q is None:
            break
        ans = answer_for(q, profile)
        session.update(q, ans)

    return [r for r, _ in session.top(TOP_N, min_match_score=0)]


def run_eer_embedding_budget(
    recipes: list[dict],
    profile: dict,
    weights: dict,
    budget: int,
    beta: float = 0.75,
    candidate_n: int = 150,
) -> list[dict]:
    session = BayesianSession(recipes, weights)

    while session.q < budget:
        q = select_next_question(session)
        if q is None:
            break
        ans = answer_for(q, profile)
        session.update(q, ans)

    if profile.get("semantic_query"):
        session.answers["semantic_query"] = profile["semantic_query"]

    return [
        r
        for r, _ in semantic_rerank(
            session,
            n=TOP_N,
            candidate_n=candidate_n,
            beta=beta,
            min_match_score=0,
        )
    ]


def run_eer_embedding_pool_budget(
    recipes: list[dict],
    profile: dict,
    weights: dict,
    budget: int,
    beta: float = 0.75,
    pool_n: int = 500,
    candidate_n: int = 150,
) -> list[dict]:
    query = profile.get("semantic_query")
    pooled = semantic_candidate_pool(recipes, query, pool_n=pool_n)
    session = BayesianSession(pooled, weights)

    while session.q < budget:
        q = select_next_question(session)
        if q is None:
            break
        ans = answer_for(q, profile)
        session.update(q, ans)

    if query:
        session.answers["semantic_query"] = query

    return [
        r
        for r, _ in semantic_rerank(
            session,
            n=TOP_N,
            candidate_n=min(candidate_n, len(pooled)),
            beta=beta,
            min_match_score=0,
        )
    ]


# ── EVALUARE ──────────────────────────────────────────────────────────────────
def evaluate(recipes: list[dict], weights: dict) -> dict:
    """Returnează NDCG mediu per algoritm per buget."""
    algorithms = ["Marginal", "JMIM", "EER", "EER+Emb100", "EER+Emb150", "EER+Pool500"]
    results = {
        scenario: {alg: {b: [] for b in BUDGETS} for alg in algorithms}
        for scenario in SCENARIOS
    }

    for scenario in SCENARIOS:
        for profile in USERS:
            answer_profile = profile_for_scenario(profile, scenario)
            for budget in BUDGETS:
                ranked_m = run_marginal_budget(recipes, answer_profile, budget, weights)
                ranked_j = run_jmim_budget(recipes, answer_profile, budget, weights)
                ranked_e = run_eer_budget(recipes, answer_profile, weights, budget)
                ranked_d100 = run_eer_embedding_budget(
                    recipes, answer_profile, weights, budget, candidate_n=100
                )
                ranked_d150 = run_eer_embedding_budget(
                    recipes, answer_profile, weights, budget, candidate_n=150
                )
                ranked_pool = run_eer_embedding_pool_budget(
                    recipes, answer_profile, weights, budget, pool_n=500, candidate_n=150
                )

                results[scenario]["Marginal"][budget].append(ndcg_at_k(ranked_m, recipes, profile))
                results[scenario]["JMIM"][budget].append(ndcg_at_k(ranked_j, recipes, profile))
                results[scenario]["EER"][budget].append(ndcg_at_k(ranked_e, recipes, profile))
                results[scenario]["EER+Emb100"][budget].append(ndcg_at_k(ranked_d100, recipes, profile))
                results[scenario]["EER+Emb150"][budget].append(ndcg_at_k(ranked_d150, recipes, profile))
                results[scenario]["EER+Pool500"][budget].append(ndcg_at_k(ranked_pool, recipes, profile))

    return results


def print_report(results: dict) -> None:
    algorithms = ["Marginal", "JMIM", "EER", "EER+Emb"]

    print(f"\n{'='*65}")
    print("NDCG@10 LA BUGET FIX DE ÎNTREBĂRI")
    print(f"{'='*65}")
    print(f"{'Algoritm':<12}", end="")
    for b in BUDGETS:
        print(f"  K={b:>2}", end="")
    print()
    print("-" * 55)

    for alg in algorithms:
        print(f"{alg:<12}", end="")
        for b in BUDGETS:
            avg = sum(results[alg][b]) / len(results[alg][b])
            print(f"  {avg:.3f}", end="")
        print()

    print(f"\n{'='*65}")
    print("ÎMBUNĂTĂȚIRE EER vs MARGINAL (ΔNDCG@10)\n")
    print(f"{'':12}", end="")
    for b in BUDGETS:
        print(f"  K={b:>2}", end="")
    print()
    print("-" * 55)

    for alg in ["JMIM", "EER", "EER+Emb"]:
        print(f"{alg} - Marg  ", end="")
        for b in BUDGETS:
            avg_alg = sum(results[alg][b]) / len(results[alg][b])
            avg_base = sum(results["Marginal"][b]) / len(results["Marginal"][b])
            delta = avg_alg - avg_base
            sign = "+" if delta >= 0 else ""
            print(f"  {sign}{delta:.3f}", end="")
        print()

    print(f"\n{'='*65}")
    print("DETALII PER UTILIZATOR LA K=8\n")
    print(f"{'Utilizator':<30} {'Marginal':>9} {'JMIM':>9} {'EER':>9} {'EER+Emb':>9}")
    print("-" * 60)
    for i, profile in enumerate(USERS):
        m = results["Marginal"][8][i]
        j = results["JMIM"][8][i]
        e = results["EER"][8][i]
        d = results["EER+Emb"][8][i]
        best = max(m, j, e, d)
        def fmt(v: float) -> str:
            return f"{'*' if v == best else ' '}{v:.4f}"
        print(f"  {profile['name']:<28} {fmt(m):>9} {fmt(j):>9} {fmt(e):>9} {fmt(d):>9}")


def print_report(results: dict) -> None:
    algorithms = ["Marginal", "JMIM", "EER", "EER+Emb100", "EER+Emb150", "EER+Pool500"]

    for scenario in SCENARIOS:
        scenario_results = results[scenario]
        print(f"\n{'='*75}")
        print(f"NDCG@10 LA BUGET FIX DE INTREBARI - SCENARIU: {scenario}")
        print(f"{'='*75}")
        print(f"{'Algoritm':<14}", end="")
        for budget in BUDGETS:
            print(f"  K={budget:>2}", end="")
        print()
        print("-" * 65)

        for alg in algorithms:
            print(f"{alg:<14}", end="")
            for budget in BUDGETS:
                avg = sum(scenario_results[alg][budget]) / len(scenario_results[alg][budget])
                print(f"  {avg:.3f}", end="")
            print()

        print(f"\n{'='*75}")
        print("DELTA VS EER (NDCG@10)\n")
        print(f"{'':14}", end="")
        for budget in BUDGETS:
            print(f"  K={budget:>2}", end="")
        print()
        print("-" * 65)

        for alg in ["EER+Emb100", "EER+Emb150", "EER+Pool500"]:
            print(f"{alg:<14}", end="")
            for budget in BUDGETS:
                avg_alg = sum(scenario_results[alg][budget]) / len(scenario_results[alg][budget])
                avg_base = sum(scenario_results["EER"][budget]) / len(scenario_results["EER"][budget])
                delta = avg_alg - avg_base
                sign = "+" if delta >= 0 else ""
                print(f"  {sign}{delta:.3f}", end="")
            print()

    scenario_results = results["unknown_adaptive"]
    print(f"\n{'='*75}")
    print("DETALII PER UTILIZATOR LA K=8 - unknown_adaptive\n")
    print(f"{'Utilizator':<30} {'EER':>9} {'Emb100':>9} {'Emb150':>9}")
    print("-" * 60)
    for i, profile in enumerate(USERS):
        e = scenario_results["EER"][8][i]
        d100 = scenario_results["EER+Emb100"][8][i]
        d150 = scenario_results["EER+Emb150"][8][i]
        best = max(e, d100, d150)

        def fmt(value: float) -> str:
            return f"{'*' if value == best else ' '}{value:.4f}"

        print(f"  {profile['name']:<28} {fmt(e):>9} {fmt(d100):>9} {fmt(d150):>9}")


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
    results = evaluate(recipes, weights)
    print_report(results)


if __name__ == "__main__":
    main()
