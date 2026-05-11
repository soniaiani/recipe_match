from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import app.recommender.engine

importlib.reload(app.recommender.engine)

from app.database import get_supabase_admin
from app.recommender.engine import (
    ENTROPY_STOP_THRESHOLD,
    MAX_QUESTIONS,
    BayesianSession,
    compute_feature_mi,
    question_by_id,
    select_next_question,
)
from app.recommender.semantic_rerank import (
    _infer_feature_intents,
    semantic_candidate_pool,
    semantic_rerank,
)
from app.routers.recommendations import (
    _RecSession,
    _attach_semantic_context,
    _infer_protein_from_query_embedding,
    _maybe_infer_protein_type,
)


@dataclass(frozen=True)
class Scenario:
    name: str
    initial_answers: dict[str, Any]
    target_profile: dict[str, Any]
    semantic_query: str | None = None


SCENARIOS = [
    Scenario(
        name="Classic Bayesian: Italian chicken pasta",
        initial_answers={
            "meal_type": "lunch_dinner",
            "cuisine": ["italian"],
            "protein_type": ["chicken"],
        },
        target_profile={
            "meal_type": "lunch_dinner",
            "cuisine": "italian",
            "protein_type": "chicken",
            "has_pasta": True,
            "has_cream_base": True,
            "has_cheese": True,
            "is_spicy": False,
            "needs_oven": True,
            "is_quick": False,
        },
    ),
    Scenario(
        name="Classic Bayesian: vague dessert",
        initial_answers={"meal_type": "dessert"},
        target_profile={"meal_type": "dessert", "is_sweet": True},
    ),
    Scenario(
        name="Classic Bayesian: sweet quick breakfast",
        initial_answers={"meal_type": "breakfast"},
        target_profile={
            "meal_type": "breakfast",
            "is_sweet": True,
            "has_fruit": True,
            "is_quick": True,
            "needs_oven": False,
            "needs_stovetop": False,
        },
    ),
    Scenario(
        name="Classic Bayesian: spicy Asian chicken soup",
        initial_answers={"meal_type": "soup"},
        target_profile={
            "meal_type": "soup",
            "cuisine": "asian",
            "protein_type": "chicken",
            "has_broth_base": True,
            "is_spicy": True,
            "has_asian_sauce": True,
        },
    ),
    Scenario(
        name="Semantic query: cheese Italian chicken dinner",
        initial_answers={
            "meal_type": "lunch_dinner",
            "cuisine": ["italian"],
            "protein_type": ["chicken"],
        },
        target_profile={
            "meal_type": "lunch_dinner",
            "cuisine": "italian",
            "protein_type": "chicken",
            "has_cheese": True,
        },
        semantic_query="cheese",
    ),
    Scenario(
        name="Semantic query: cheesy pasta without inferred protein",
        initial_answers={
            "meal_type": "lunch_dinner",
            "cuisine": ["italian"],
        },
        target_profile={
            "meal_type": "lunch_dinner",
            "cuisine": "italian",
            "has_pasta": True,
            "has_cheese": True,
        },
        semantic_query="cheesy pasta",
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


def simulate_answer(question: dict[str, Any], target_profile: dict[str, Any]) -> Any:
    q_type = question["type"]
    feature = question["feature"]

    if q_type == "boolean":
        if feature in target_profile:
            return "yes" if target_profile[feature] else "no"
        return "skip"

    if q_type == "multiselect":
        if feature in target_profile:
            value = target_profile[feature]
            return [value] if isinstance(value, str) else value
        return ["any"]

    if q_type == "categorical":
        return target_profile.get(feature, "skip")

    return "skip"


def build_session(
    recipes: list[dict[str, Any]],
    weights: dict[str, float],
    answers: dict[str, Any],
    semantic_query: str | None = None,
) -> BayesianSession:
    session = BayesianSession(recipes, weights)
    _attach_semantic_context(session, semantic_query)
    for qid, answer in answers.items():
        question = question_by_id(qid)
        if question:
            session.update(question, answer)
    if semantic_query:
        session.answers["semantic_query"] = semantic_query
    return session


def print_top(session: BayesianSession, n: int = 5) -> None:
    p = session.probs()
    uniform = 1.0 / session.n
    id_to_idx = {recipe["id"]: idx for idx, recipe in enumerate(session.recipes)}

    print(f"  {'Name':<44} {'Lift':>7} {'Score':>7} {'Cheese':>7}")
    print("  " + "-" * 72)
    for recipe, score in session.top(n, min_match_score=0):
        idx = id_to_idx.get(recipe["id"])
        lift = float(p[idx]) / uniform if idx is not None else 0.0
        cheese = "yes" if recipe.get("has_cheese") else "no"
        print(f"  {recipe.get('name', '?')[:42]:<44} {lift:>7.1f} {score:>6.1f}% {cheese:>7}")


def assert_fixed_question_order(recipes: list[dict[str, Any]], weights: dict[str, float]) -> None:
    session = BayesianSession(recipes, weights)
    q1 = select_next_question(session)
    assert q1 and q1["id"] == "meal_type", f"Expected meal_type first, got {q1}"

    session.update(q1, "lunch_dinner")
    q2 = select_next_question(session)
    assert q2 and q2["id"] == "cuisine", f"Expected cuisine second, got {q2}"

    session.update(q2, ["italian"])
    q3 = select_next_question(session)
    assert q3 and q3["id"] == "protein_type", f"Expected protein_type third, got {q3}"


def assert_semantic_candidate_pool(recipes: list[dict[str, Any]]) -> None:
    pooled = semantic_candidate_pool(recipes, "cheese", pool_n=500)
    assert 20 <= len(pooled) <= 500, f"Unexpected semantic pool size: {len(pooled)}"

    cheese_rate = sum(1 for recipe in pooled[:100] if recipe.get("has_cheese")) / min(len(pooled), 100)
    assert cheese_rate > 0.30, f"Cheese query pool does not contain enough cheese recipes: {cheese_rate:.2f}"


def assert_protein_embedding_inference() -> None:
    cases = {
        "chicken": "chicken",
        "salmon pasta": "fish_seafood",
        "vegetarian curry": "meatless",
        "cheesy pasta": None,
    }
    for query, expected in cases.items():
        actual = _infer_protein_from_query_embedding(query)
        assert actual == expected, f"{query!r}: expected {expected}, got {actual}"


def assert_feature_intents() -> None:
    cases = {
        "cheese": "has_cheese",
        "pasta": "has_pasta",
        "chocolate": "has_chocolate",
        "spicy": "is_spicy",
    }
    for query, expected in cases.items():
        actual = _infer_feature_intents(query)
        assert expected in actual, f"{query!r}: expected {expected} in {actual}"


def assert_semantic_protein_skip(
    recipes: list[dict[str, Any]],
    weights: dict[str, float],
) -> None:
    pooled = semantic_candidate_pool(recipes, "chicken", pool_n=500)
    session = BayesianSession(pooled, weights)
    _attach_semantic_context(session, "chicken")
    sess = _RecSession(
        session_id="test",
        recipe_ids=[recipe["id"] for recipe in pooled],
        semantic_query="chicken",
    )

    meal_q = question_by_id("meal_type")
    cuisine_q = question_by_id("cuisine")
    assert meal_q and cuisine_q

    session.update(meal_q, "lunch_dinner")
    sess.answers["meal_type"] = "lunch_dinner"
    sess.question_order.append("meal_type")
    _maybe_infer_protein_type(sess, session)
    assert "protein_type" not in sess.answers, "Protein should not be inferred before cuisine"

    session.update(cuisine_q, ["italian"])
    sess.answers["cuisine"] = ["italian"]
    sess.question_order.append("cuisine")
    _maybe_infer_protein_type(sess, session)
    assert sess.answers.get("protein_type") == ["chicken"], sess.answers

    next_q = select_next_question(session)
    assert next_q and next_q["id"] != "protein_type", f"Protein question was not skipped: {next_q}"


def assert_cheese_query_rerank(
    recipes: list[dict[str, Any]],
    weights: dict[str, float],
) -> None:
    pooled = semantic_candidate_pool(recipes, "cheese", pool_n=500)
    session = build_session(
        pooled,
        weights,
        {
            "meal_type": "lunch_dinner",
            "cuisine": ["italian"],
            "protein_type": ["chicken"],
        },
        semantic_query="cheese",
    )
    ranked = semantic_rerank(session, n=10, candidate_n=150, beta=0.75, min_match_score=0)
    cheese_count = sum(1 for recipe, _ in ranked if recipe.get("has_cheese"))
    assert cheese_count >= 7, f"Expected at least 7/10 cheese recipes, got {cheese_count}/10"


def run_scenario(
    scenario: Scenario,
    recipes: list[dict[str, Any]],
    weights: dict[str, float],
) -> None:
    print("=" * 90)
    print(f"SCENARIO: {scenario.name}")
    print("=" * 90)

    candidate_recipes = recipes
    if scenario.semantic_query:
        candidate_recipes = semantic_candidate_pool(recipes, scenario.semantic_query, pool_n=500)
        print(f"Semantic query: {scenario.semantic_query!r}")
        print(f"Semantic pool: {len(candidate_recipes)} recipes")

    session = build_session(
        candidate_recipes,
        weights,
        scenario.initial_answers,
        scenario.semantic_query,
    )

    print(f"Initial answers: {scenario.initial_answers}")
    print(f"After {session.q} initial answers:")
    print(f"  Entropy: {session.entropy():.2f} bits")
    print(f"  should_stop: {session.should_stop()}")

    print(
        f"\n{'Q#':<4} {'Question':<24} {'Answer':<18} "
        f"{'Entropy':>8} {'top10':>8} {'stable':>8} {'stop':>6}"
    )
    print("-" * 90)

    while not session.should_stop() and session.q < MAX_QUESTIONS:
        next_q = select_next_question(session)
        if next_q is None:
            print("  [no more questions]")
            break

        answer = simulate_answer(next_q, scenario.target_profile)
        before = session.entropy()
        session.update(next_q, answer)

        p = session.probs()
        top10_prob = float(p[p.argsort()[-10:]].sum())
        print(
            f"{session.q:<4} {next_q['id']:<24} {str(answer):<18} "
            f"{session.entropy():>8.2f} {top10_prob:>8.2f} "
            f"{str(session.is_top_stable()):>8} {str(session.should_stop()):>6}"
        )

        if session.entropy() > before + 1e-9:
            raise AssertionError(f"Entropy increased after {next_q['id']}")

    print(f"\nStopped after {session.q} total questions")
    print(f"Final entropy: {session.entropy():.2f} bits")

    if scenario.semantic_query:
        ranked = semantic_rerank(session, n=5, candidate_n=150, beta=0.75, min_match_score=0)
        print("\nTop 5 semantic reranked recipes:")
        for recipe, score in ranked:
            print(
                f"  {score:>6.1f}  {recipe.get('name', '?')[:48]:<50} "
                f"cheese={bool(recipe.get('has_cheese'))}"
            )
    else:
        print("\nTop 5 Bayesian recipes:")
        print_top(session, n=5)
    print()


def main() -> None:
    print(f"ENTROPY_STOP_THRESHOLD = {ENTROPY_STOP_THRESHOLD:.3f}")
    recipes = load_recipes()
    if not recipes:
        raise RuntimeError("No recipes loaded from Supabase.")

    print(f"Loaded {len(recipes)} recipes")
    weights = compute_feature_mi(recipes)

    print("\nRunning assertions...")
    assert_fixed_question_order(recipes, weights)
    print("  ok fixed question order")
    assert_semantic_candidate_pool(recipes)
    print("  ok semantic candidate pool")
    assert_protein_embedding_inference()
    print("  ok protein embedding inference")
    assert_feature_intents()
    print("  ok semantic feature intents")
    assert_semantic_protein_skip(recipes, weights)
    print("  ok semantic protein skip")
    assert_cheese_query_rerank(recipes, weights)
    print("  ok cheese query rerank")

    print("\nRunning detailed sessions...\n")
    for scenario in SCENARIOS:
        run_scenario(scenario, recipes, weights)

    print("All recommendation session tests completed successfully.")


if __name__ == "__main__":
    main()
