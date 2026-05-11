# backend/scripts/test_no_skip_users.py
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
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
    QUESTION_BANK,
    compute_feature_mi,
    question_by_id,
    select_next_question,
)


BOOLEAN_DEFAULTS = {
    "is_spicy": "no",
    "is_sweet": "no",
    "is_quick": "yes",
    "needs_oven": "no",
    "needs_stovetop": "yes",
    "is_no_cook": "no",
    "has_pasta": "no",
    "has_rice": "no",
    "has_potato": "no",
    "has_tomato_base": "no",
    "has_cream_base": "no",
    "has_cheese": "no",
    "has_broth_base": "no",
    "has_mushroom": "no",
    "has_leafy_greens": "no",
    "has_beans_legumes": "no",
    "has_fruit": "no",
    "has_nuts": "no",
    "has_chocolate": "no",
    "has_asian_sauce": "no",
}


@dataclass(frozen=True)
class SyntheticUser:
    name: str
    profile: dict[str, Any]
    notes: str


USERS = [
    SyntheticUser(
        name="Italian chicken pasta",
        notes="Utilizator hotarat: cina italiana, pui, paste, cremos, fara iute.",
        profile={
            "meal_type": "lunch_dinner",
            "protein_type": ["chicken"],
            "cuisine": ["italian"],
            "is_spicy": "no",
            "is_sweet": "no",
            "is_quick": "no",
            "needs_oven": "yes",
            "needs_stovetop": "no",
            "is_no_cook": "no",
            "has_pasta": "yes",
            "has_rice": "no",
            "has_potato": "no",
            "has_tomato_base": "no",
            "has_cream_base": "yes",
            "has_cheese": "yes",
            "has_broth_base": "no",
            "has_mushroom": "no",
            "has_leafy_greens": "no",
            "has_beans_legumes": "no",
            "has_fruit": "no",
            "has_nuts": "no",
            "has_chocolate": "no",
            "has_asian_sauce": "no",
        },
    ),
    SyntheticUser(
        name="Spicy Asian chicken soup",
        notes="Utilizator hotarat: supa asiatica cu pui, broth, iute, gatita pe plita.",
        profile={
            "meal_type": "soup",
            "protein_type": ["chicken"],
            "cuisine": ["asian"],
            "is_spicy": "yes",
            "is_sweet": "no",
            "is_quick": "yes",
            "needs_oven": "no",
            "needs_stovetop": "yes",
            "is_no_cook": "no",
            "has_pasta": "no",
            "has_rice": "no",
            "has_potato": "no",
            "has_tomato_base": "no",
            "has_cream_base": "no",
            "has_cheese": "no",
            "has_broth_base": "yes",
            "has_mushroom": "yes",
            "has_leafy_greens": "yes",
            "has_beans_legumes": "no",
            "has_fruit": "no",
            "has_nuts": "no",
            "has_chocolate": "no",
            "has_asian_sauce": "yes",
        },
    ),
    SyntheticUser(
        name="Fast fruit breakfast",
        notes="Utilizator hotarat: mic dejun rapid, dulce, fructe, fara gatire.",
        profile={
            "meal_type": "breakfast",
            "protein_type": ["meatless"],
            "cuisine": ["american"],
            "is_spicy": "no",
            "is_sweet": "yes",
            "is_quick": "yes",
            "needs_oven": "no",
            "needs_stovetop": "no",
            "is_no_cook": "yes",
            "has_pasta": "no",
            "has_rice": "no",
            "has_potato": "no",
            "has_tomato_base": "no",
            "has_cream_base": "yes",
            "has_cheese": "no",
            "has_broth_base": "no",
            "has_mushroom": "no",
            "has_leafy_greens": "no",
            "has_beans_legumes": "no",
            "has_fruit": "yes",
            "has_nuts": "yes",
            "has_chocolate": "no",
            "has_asian_sauce": "no",
        },
    ),
    SyntheticUser(
        name="Chocolate dessert",
        notes="Utilizator hotarat: desert dulce cu ciocolata, nuci si cuptor.",
        profile={
            "meal_type": "dessert",
            "protein_type": ["meatless"],
            "cuisine": ["french"],
            "is_spicy": "no",
            "is_sweet": "yes",
            "is_quick": "no",
            "needs_oven": "yes",
            "needs_stovetop": "no",
            "is_no_cook": "no",
            "has_pasta": "no",
            "has_rice": "no",
            "has_potato": "no",
            "has_tomato_base": "no",
            "has_cream_base": "yes",
            "has_cheese": "no",
            "has_broth_base": "no",
            "has_mushroom": "no",
            "has_leafy_greens": "no",
            "has_beans_legumes": "no",
            "has_fruit": "no",
            "has_nuts": "yes",
            "has_chocolate": "yes",
            "has_asian_sauce": "no",
        },
    ),
    SyntheticUser(
        name="Mediterranean salad side",
        notes="Utilizator hotarat: garnitura/salata mediteraneana, fara gatire, branza si verdeturi.",
        profile={
            "meal_type": "salad_side",
            "protein_type": ["meatless"],
            "cuisine": ["mediterranean"],
            "is_spicy": "no",
            "is_sweet": "no",
            "is_quick": "yes",
            "needs_oven": "no",
            "needs_stovetop": "no",
            "is_no_cook": "yes",
            "has_pasta": "no",
            "has_rice": "no",
            "has_potato": "no",
            "has_tomato_base": "yes",
            "has_cream_base": "no",
            "has_cheese": "yes",
            "has_broth_base": "no",
            "has_mushroom": "no",
            "has_leafy_greens": "yes",
            "has_beans_legumes": "yes",
            "has_fruit": "no",
            "has_nuts": "no",
            "has_chocolate": "no",
            "has_asian_sauce": "no",
        },
    ),
    SyntheticUser(
        name="Mexican beef dinner",
        notes="Utilizator hotarat: cina mexicana cu vita/porc, rosii, fasole si iute.",
        profile={
            "meal_type": "lunch_dinner",
            "protein_type": ["beef_pork"],
            "cuisine": ["mexican"],
            "is_spicy": "yes",
            "is_sweet": "no",
            "is_quick": "no",
            "needs_oven": "no",
            "needs_stovetop": "yes",
            "is_no_cook": "no",
            "has_pasta": "no",
            "has_rice": "yes",
            "has_potato": "no",
            "has_tomato_base": "yes",
            "has_cream_base": "no",
            "has_cheese": "yes",
            "has_broth_base": "no",
            "has_mushroom": "no",
            "has_leafy_greens": "no",
            "has_beans_legumes": "yes",
            "has_fruit": "no",
            "has_nuts": "no",
            "has_chocolate": "no",
            "has_asian_sauce": "no",
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


def answer_for(question: dict[str, Any], profile: dict[str, Any]) -> Any:
    qid = question["id"]
    q_type = question["type"]

    if qid in profile:
        return profile[qid]

    feature = question["feature"]
    if feature in profile:
        value = profile[feature]
        if q_type == "boolean":
            if value in ("yes", "no"):
                return value
            return "yes" if bool(value) else "no"
        if q_type == "multiselect":
            return value if isinstance(value, list) else [value]
        return value

    if q_type == "boolean":
        return BOOLEAN_DEFAULTS.get(qid, "no")
    if q_type == "multiselect":
        options = question.get("options") or []
        if not options:
            raise AssertionError(f"Question {qid} has no options")
        return [options[0]]
    if q_type == "categorical":
        options = question.get("options") or []
        if not options:
            raise AssertionError(f"Question {qid} has no options")
        return options[0]

    raise AssertionError(f"Unsupported question type for {qid}: {q_type}")


def assert_informative_answer(qid: str, answer: Any) -> None:
    if answer in ("skip", "any") or answer == ["any"]:
        raise AssertionError(f"{qid} received non-informative answer: {answer!r}")


def assert_valid_answer(question: dict[str, Any], answer: Any) -> None:
    qid = question["id"]
    q_type = question["type"]
    assert_informative_answer(qid, answer)

    if q_type == "boolean":
        if answer not in ("yes", "no"):
            raise AssertionError(f"{qid} expected yes/no, got {answer!r}")
    elif q_type == "multiselect":
        selected = answer if isinstance(answer, list) else [answer]
        unknown = set(selected) - set(question.get("options", []))
        if unknown:
            raise AssertionError(f"{qid} has unknown options: {sorted(unknown)}")
    elif q_type == "categorical":
        if answer not in question.get("options", []):
            raise AssertionError(f"{qid} has unknown option: {answer!r}")


def recipe_answer_match(recipe: dict[str, Any], answers: dict[str, Any]) -> tuple[int, int, list[str]]:
    matched = 0
    checked = 0
    misses: list[str] = []

    for qid, answer in answers.items():
        q = question_by_id(qid)
        if not q:
            continue
        checked += 1
        feature = q["feature"]

        if q["type"] == "categorical":
            ok = recipe.get(feature) == answer
        elif q["type"] == "multiselect":
            selected = answer if isinstance(answer, list) else [answer]
            ok = recipe.get(feature) in selected
        else:
            value = bool(recipe.get(feature))
            ok = value if answer == "yes" else not value

        if ok:
            matched += 1
        else:
            misses.append(qid)

    return matched, checked, misses


def top_summary(session: BayesianSession, id_to_idx: dict[Any, int], n: int) -> list[dict[str, Any]]:
    p = session.probs()
    uniform = 1.0 / session.n
    rows = []
    for recipe, score in session.top(n=n, min_match_score=0):
        idx = id_to_idx.get(recipe["id"])
        lift = float(p[idx]) / uniform if idx is not None else 0.0
        matched, checked, misses = recipe_answer_match(recipe, session.answers)
        rows.append(
            {
                "id": recipe.get("id"),
                "name": recipe.get("name", "?"),
                "meal_type": recipe.get("meal_type"),
                "cuisine": recipe.get("cuisine"),
                "protein_type": recipe.get("protein_type"),
                "posterior_score": score,
                "lift": round(lift, 2),
                "answer_matches": matched,
                "answers_checked": checked,
                "misses": misses,
            }
        )
    return rows


def run_user(
    user: SyntheticUser,
    recipes: list[dict[str, Any]],
    weights: dict[str, float],
    id_to_idx: dict[Any, int],
    max_questions: int,
    top_n: int,
) -> dict[str, Any]:
    session = BayesianSession(recipes, weights)
    entropies = [session.entropy()]
    trace = []

    while not session.should_stop() and session.q < max_questions:
        question = select_next_question(session)
        if question is None:
            break

        answer = answer_for(question, user.profile)
        assert_valid_answer(question, answer)
        entropy_before = session.entropy()
        session.update(question, answer)
        entropy_after = session.entropy()
        entropies.append(entropy_after)

        p = session.probs()
        top10_prob = float(p[p.argsort()[-10:]].sum())
        trace.append(
            {
                "q": session.q,
                "question_id": question["id"],
                "type": question["type"],
                "answer": answer,
                "entropy_before": round(entropy_before, 4),
                "entropy_after": round(entropy_after, 4),
                "delta_entropy": round(entropy_before - entropy_after, 4),
                "top10_probability": round(top10_prob, 4),
                "top_stable": session.is_top_stable(),
                "should_stop": session.should_stop(),
            }
        )

    answers = session.answers
    if any(answer in ("skip", "any") or answer == ["any"] for answer in answers.values()):
        raise AssertionError(f"{user.name}: session contains skip/any answers")
    if len(answers) != len(set(answers)):
        raise AssertionError(f"{user.name}: duplicate question ids in answers")

    final_top = top_summary(session, id_to_idx, top_n)
    best = final_top[0] if final_top else {}
    entropy_drops = [before - after for before, after in zip(entropies, entropies[1:])]

    return {
        "name": user.name,
        "notes": user.notes,
        "questions_asked": session.q,
        "stopped": session.should_stop(),
        "stop_reason": stop_reason(session),
        "entropy_initial": round(entropies[0], 4),
        "entropy_final": round(session.entropy(), 4),
        "entropy_threshold": round(ENTROPY_STOP_THRESHOLD, 4),
        "entropy_total_drop": round(entropies[0] - session.entropy(), 4),
        "entropy_non_increasing_steps": sum(1 for d in entropy_drops if d >= -1e-9),
        "answers": answers,
        "trace": trace,
        "best_recipe": best,
        "top": final_top,
    }


def stop_reason(session: BayesianSession) -> str:
    if session.q >= MAX_QUESTIONS:
        return "max_questions"
    if session.entropy() < ENTROPY_STOP_THRESHOLD:
        return "entropy_or_combined_conditions"
    if session.is_top_stable():
        return "stability_or_combined_conditions"
    if session.should_stop():
        return "engine_should_stop"
    return "not_stopped"


def print_report(results: list[dict[str, Any]], recipe_count: int) -> None:
    print(f"Loaded recipes: {recipe_count}")
    print(f"ENTROPY_STOP_THRESHOLD: {ENTROPY_STOP_THRESHOLD:.3f}")
    print(f"MAX_QUESTIONS: {MAX_QUESTIONS}")
    print()

    for result in results:
        print("=" * 100)
        print(f"USER: {result['name']}")
        print(result["notes"])
        print("-" * 100)
        print(
            f"Questions: {result['questions_asked']} | stopped={result['stopped']} "
            f"| reason={result['stop_reason']} | entropy {result['entropy_initial']:.2f}"
            f" -> {result['entropy_final']:.2f} bits"
        )
        print()
        print(
            f"{'#':<3} {'question':<24} {'type':<12} {'answer':<24} "
            f"{'H before':>9} {'H after':>8} {'drop':>8} {'top10':>8} {'stop':>6}"
        )
        print("-" * 100)
        for row in result["trace"]:
            answer = json.dumps(row["answer"], ensure_ascii=False)
            print(
                f"{row['q']:<3} {row['question_id']:<24} {row['type']:<12} {answer:<24} "
                f"{row['entropy_before']:>9.2f} {row['entropy_after']:>8.2f} "
                f"{row['delta_entropy']:>8.2f} {row['top10_probability']:>8.2f} "
                f"{'yes' if row['should_stop'] else 'no':>6}"
            )
        print()
        print("Top recipes:")
        print(
            f"  {'name':<42} {'meal':<14} {'cuisine':<14} {'protein':<13} "
            f"{'score':>7} {'lift':>7} {'match':>8} misses"
        )
        print("  " + "-" * 96)
        for row in result["top"]:
            match = f"{row['answer_matches']}/{row['answers_checked']}"
            misses = ",".join(row["misses"][:5])
            if len(row["misses"]) > 5:
                misses += ",..."
            print(
                f"  {row['name'][:40]:<42} {str(row['meal_type'])[:12]:<14} "
                f"{str(row['cuisine'])[:12]:<14} {str(row['protein_type'])[:11]:<13} "
                f"{row['posterior_score']:>7.1f} {row['lift']:>7.2f} {match:>8} {misses}"
            )
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run detailed Bayesian recommendation simulations with users that never skip questions."
    )
    parser.add_argument("--max-questions", type=int, default=MAX_QUESTIONS)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path where the full test report is written as JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    recipes = load_recipes()
    if not recipes:
        raise RuntimeError("No recipes loaded from Supabase.")

    weights = compute_feature_mi(recipes)
    id_to_idx = {recipe["id"]: i for i, recipe in enumerate(recipes)}

    missing_questions = [
        q["id"]
        for q in QUESTION_BANK
        if q["id"] not in BOOLEAN_DEFAULTS and q["type"] == "boolean"
    ]
    if missing_questions:
        raise AssertionError(f"BOOLEAN_DEFAULTS missing questions: {missing_questions}")

    results = [
        run_user(
            user=user,
            recipes=recipes,
            weights=weights,
            id_to_idx=id_to_idx,
            max_questions=args.max_questions,
            top_n=args.top_n,
        )
        for user in USERS
    ]

    print_report(results, len(recipes))

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(results, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"JSON report written to: {args.json_out}")

    failures = [
        result["name"]
        for result in results
        if result["questions_asked"] == 0 or not result["top"]
    ]
    if failures:
        raise AssertionError(f"Invalid simulations: {failures}")

    print("All no-skip user simulations completed successfully.")


if __name__ == "__main__":
    main()
