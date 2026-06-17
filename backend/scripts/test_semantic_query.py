from __future__ import annotations

import argparse
import importlib
import os
import sys
from dataclasses import dataclass
from typing import Any

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import app.services.recommendations.bayesian

importlib.reload(app.services.recommendations.bayesian)

from app.database import get_supabase_admin
from app.services.recommendations.bayesian.config import question_by_id
from app.services.recommendations.bayesian.features import compute_feature_mi
from app.services.recommendations.bayesian.session import BayesianSession
from app.recommender.semantic_rerank import semantic_candidate_pool, semantic_rerank


@dataclass(frozen=True)
class SemanticCase:
    name: str
    semantic_query: str
    answers: dict[str, Any]


CASES = [
    SemanticCase(
        name="Creamy Italian chicken pasta",
        semantic_query="cheesy pasta",
        answers={
            "meal_type": "lunch_dinner",
            "protein_type": ["chicken"],
            "cuisine": ["italian"],
        },
    ),
    SemanticCase(
        name="Spicy Asian chicken soup",
        semantic_query="spicy Asian chicken soup with broth and mushrooms",
        answers={
            "meal_type": "soup",
            "protein_type": ["chicken"],
            "cuisine": ["asian"],
        },
    ),
    SemanticCase(
        name="Quick fruit breakfast",
        semantic_query="quick sweet breakfast with fruit yogurt and nuts",
        answers={
            "meal_type": "breakfast",
            "protein_type": ["meatless"],
            "cuisine": ["american"],
        },
    ),
    SemanticCase(
        name="Chocolate nut dessert",
        semantic_query="baked chocolate dessert with nuts",
        answers={
            "meal_type": "dessert",
            "protein_type": ["meatless"],
            "cuisine": ["french"],
        },
    ),
    SemanticCase(
        name="Fresh Mediterranean salad",
        semantic_query="fresh Mediterranean salad with tomato greens cheese and beans",
        answers={
            "meal_type": "salad_side",
            "protein_type": ["meatless"],
            "cuisine": ["mediterranean"],
        },
    ),
    SemanticCase(
        name="Spicy Mexican beef rice",
        semantic_query="spicy Mexican beef dinner with rice beans tomato and cheese",
        answers={
            "meal_type": "lunch_dinner",
            "protein_type": ["beef_pork"],
            "cuisine": ["mexican"],
        },
    ),
]


def load_recipes() -> list[dict[str, Any]]:
    admin = get_supabase_admin()
    recipes: list[dict[str, Any]] = []
    offset = 0
    while True:
        page = admin.table("recipes").select("*").range(offset, offset + 999).execute().data or []
        recipes.extend(page)
        if len(page) < 1000:
            break
        offset += 1000
    return recipes


def build_session(
    recipes: list[dict[str, Any]],
    weights: dict[str, float],
    answers: dict[str, Any],
) -> BayesianSession:
    session = BayesianSession(recipes, weights)
    for qid, answer in answers.items():
        question = question_by_id(qid)
        if question:
            session.update(question, answer)
    return session


def print_ranked(title: str, rows: list[tuple[dict[str, Any], float]]) -> None:
    print(title)
    print(f"  {'#':<3} {'score':>6}  {'name':<46} {'meal':<13} {'cuisine':<13} {'protein':<12}")
    print("  " + "-" * 100)
    for idx, (recipe, score) in enumerate(rows, start=1):
        print(
            f"  {idx:<3} {score:>6.1f}  "
            f"{str(recipe.get('name', '?'))[:44]:<46} "
            f"{str(recipe.get('meal_type'))[:11]:<13} "
            f"{str(recipe.get('cuisine'))[:11]:<13} "
            f"{str(recipe.get('protein_type'))[:10]:<12}"
        )
    print()


def compare_case(
    case: SemanticCase,
    recipes: list[dict[str, Any]],
    weights: dict[str, float],
    top_n: int,
    candidate_n: int,
    beta: float,
) -> None:
    baseline_session = build_session(recipes, weights, case.answers)
    pooled_recipes = semantic_candidate_pool(recipes, case.semantic_query, pool_n=candidate_n)
    pooled_session = build_session(pooled_recipes, weights, case.answers)
    semantic_session = build_session(recipes, weights, case.answers)
    semantic_session.answers["semantic_query"] = case.semantic_query
    pooled_semantic_session = build_session(pooled_recipes, weights, case.answers)
    pooled_semantic_session.answers["semantic_query"] = case.semantic_query

    baseline = baseline_session.top(n=top_n, min_match_score=0)
    pooled_bayesian = pooled_session.top(n=top_n, min_match_score=0)
    semantic = semantic_rerank(
        semantic_session,
        n=top_n,
        candidate_n=candidate_n,
        beta=beta,
        min_match_score=0,
    )
    pooled_semantic = semantic_rerank(
        pooled_semantic_session,
        n=top_n,
        candidate_n=min(candidate_n, len(pooled_recipes)),
        beta=beta,
        min_match_score=0,
    )

    baseline_ids = [recipe["id"] for recipe, _ in baseline]
    semantic_ids = [recipe["id"] for recipe, _ in semantic]
    pooled_ids = [recipe["id"] for recipe, _ in pooled_semantic]
    promoted = [rid for rid in semantic_ids if rid not in baseline_ids]
    pooled_promoted = [rid for rid in pooled_ids if rid not in baseline_ids]
    overlap = len(set(baseline_ids) & set(semantic_ids))
    pooled_overlap = len(set(baseline_ids) & set(pooled_ids))

    print("=" * 110)
    print(f"CASE: {case.name}")
    print(f"Semantic query: {case.semantic_query!r}")
    print(f"Fixed answers: {case.answers}")
    print(f"Top-{top_n} overlap: {overlap}/{top_n} | New semantic entries: {len(promoted)}")
    print(
        f"Semantic pool size: {len(pooled_recipes)} | "
        f"Pool+rerank overlap: {pooled_overlap}/{top_n} | "
        f"New pool+rerank entries: {len(pooled_promoted)}"
    )
    print()
    print_ranked("Bayesian only", baseline)
    print_ranked(f"Semantic pool + Bayesian (pool_n={candidate_n})", pooled_bayesian)
    print_ranked(f"Bayesian + semantic re-rank (candidate_n={candidate_n}, beta={beta})", semantic)
    print_ranked(
        f"Semantic pool + Bayesian + semantic re-rank (pool_n={candidate_n}, beta={beta})",
        pooled_semantic,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Bayesian-only results with optional semantic_query re-ranking."
    )
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--candidate-n", type=int, default=150)
    parser.add_argument("--beta", type=float, default=0.75)
    parser.add_argument(
        "--case",
        default=None,
        help="Optional case name substring. If omitted, all cases run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    recipes = load_recipes()
    if not recipes:
        raise RuntimeError("No recipes loaded from Supabase.")

    print(f"Loaded recipes: {len(recipes)}")
    weights = compute_feature_mi(recipes)

    cases = CASES
    if args.case:
        needle = args.case.lower()
        cases = [case for case in CASES if needle in case.name.lower()]
        if not cases:
            raise ValueError(f"No semantic cases matched: {args.case!r}")

    for case in cases:
        compare_case(
            case=case,
            recipes=recipes,
            weights=weights,
            top_n=args.top_n,
            candidate_n=args.candidate_n,
            beta=args.beta,
        )


if __name__ == "__main__":
    main()
