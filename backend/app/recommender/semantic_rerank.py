"""Legacy semantic reranking helpers.

The active Find flow no longer uses free-text semantic queries. This module is
kept for historical evaluation scripts and possible future experiments.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np

from app.recommender.embeddings import encode_recipes, encode_text, encode_texts
from app.recommender.engine import BayesianSession, ENTROPY_STOP_THRESHOLD, question_by_id

DEFAULT_SEMANTIC_BETA = 0.75
DEFAULT_CANDIDATE_N = 150
DEFAULT_POOL_N = 500
HIGH_ENTROPY_THRESHOLD = ENTROPY_STOP_THRESHOLD + 0.75
UNKNOWN_RATIO_THRESHOLD = 0.35
TIE_BREAKER_POINTS = 3.0

MEAL_TEXT = {
    "appetizer": "appetizer",
    "breakfast": "breakfast",
    "dessert": "dessert",
    "drink": "drink",
    "lunch_dinner": "lunch or dinner",
    "salad_side": "salad or side dish",
    "snack": "snack",
    "soup": "soup",
    "condiment": "condiment or sauce",
}

PROTEIN_TEXT = {
    "chicken": "chicken",
    "beef_pork": "beef or pork",
    "fish_seafood": "fish or seafood",
    "meatless": "meatless",
}

CUISINE_TEXT = {
    "italian": "Italian",
    "asian": "Asian",
    "mexican": "Mexican",
    "french": "French",
    "mediterranean": "Mediterranean",
    "indian": "Indian",
    "american": "American",
    "other": "international",
}

BOOLEAN_POSITIVE_TEXT = {
    "is_spicy": "spicy",
    "is_sweet": "sweet",
    "is_quick": "quick and easy",
    "needs_oven": "oven-baked",
    "needs_stovetop": "cooked on the stovetop",
    "is_no_cook": "no-cook",
    "has_pasta": "with pasta",
    "has_rice": "with rice",
    "has_potato": "with potato",
    "has_tomato_base": "with a tomato base",
    "has_cream_base": "creamy",
    "has_cheese": "with cheese",
    "has_broth_base": "brothy",
    "has_mushroom": "with mushrooms",
    "has_leafy_greens": "with leafy greens",
    "has_beans_legumes": "with beans or legumes",
    "has_fruit": "with fruit",
    "has_nuts": "with nuts",
    "has_chocolate": "with chocolate",
    "has_asian_sauce": "with Asian sauce",
}

BOOLEAN_NEGATIVE_TEXT = {
    "is_spicy": "not spicy",
    "is_sweet": "not sweet",
    "is_quick": "not quick",
    "needs_oven": "without oven baking",
    "needs_stovetop": "without stovetop cooking",
    "is_no_cook": "cooked recipe",
    "has_pasta": "without pasta",
    "has_rice": "without rice",
    "has_potato": "without potato",
    "has_tomato_base": "without a tomato base",
    "has_cream_base": "not creamy",
    "has_cheese": "without cheese",
    "has_broth_base": "not brothy",
    "has_chocolate": "without chocolate",
}

FEATURE_INTENT_TEXT = {
    "is_spicy": "spicy hot chili pepper recipe",
    "is_sweet": "sweet dessert sugar recipe",
    "is_quick": "quick easy fast recipe",
    "needs_oven": "oven baked roasted recipe",
    "needs_stovetop": "stovetop skillet pan cooked recipe",
    "is_no_cook": "no cook raw fresh recipe",
    "has_pasta": "pasta noodles spaghetti fettuccine recipe",
    "has_rice": "rice risotto grain recipe",
    "has_potato": "potato potatoes recipe",
    "has_tomato_base": "tomato sauce tomato base recipe",
    "has_cream_base": "creamy cream sauce recipe",
    "has_cheese": "cheese cheesy parmesan mozzarella cheddar recipe",
    "has_broth_base": "broth soup stock recipe",
    "has_mushroom": "mushroom mushrooms recipe",
    "has_leafy_greens": "leafy greens spinach kale salad recipe",
    "has_beans_legumes": "beans legumes lentils chickpeas recipe",
    "has_fruit": "fruit berries apple banana recipe",
    "has_nuts": "nuts almonds walnuts pecans recipe",
    "has_chocolate": "chocolate cocoa brownie recipe",
    "has_asian_sauce": "soy sauce teriyaki hoisin Asian sauce recipe",
}

FEATURE_INTENT_MIN_SCORE = 0.52
FEATURE_INTENT_MIN_MARGIN = 0.10
FEATURE_INTENT_BLEND = 0.35


def build_preference_query(answers: dict[str, Any]) -> str:
    semantic_query = answers.get("semantic_query") or answers.get("craving_text")
    meal = _single_answer(answers.get("meal_type"))
    protein_values = _list_answer(answers.get("protein_type"))
    cuisine_values = _list_answer(answers.get("cuisine"))

    parts: list[str] = []
    if cuisine_values:
        cuisines = ", ".join(CUISINE_TEXT.get(value, str(value)) for value in cuisine_values)
        parts.append(f"{cuisines} recipe")
    else:
        parts.append("recipe")

    if meal:
        parts.append(f"for {MEAL_TEXT.get(meal, meal)}")

    if protein_values:
        proteins = ", ".join(PROTEIN_TEXT.get(value, str(value)) for value in protein_values)
        parts.append(f"with {proteins}")

    preferences: list[str] = []
    for qid, answer in answers.items():
        q = question_by_id(qid)
        if not q or q["type"] != "boolean":
            continue
        if answer == "yes" and qid in BOOLEAN_POSITIVE_TEXT:
            preferences.append(BOOLEAN_POSITIVE_TEXT[qid])
        elif answer == "no" and qid in BOOLEAN_NEGATIVE_TEXT:
            preferences.append(BOOLEAN_NEGATIVE_TEXT[qid])

    text = " ".join(parts)
    if semantic_query:
        text = f"{semantic_query}. {text}"
    if preferences:
        text += ". Preferences: " + ", ".join(preferences)
    return text


def semantic_rerank(
    session: BayesianSession,
    n: int = 10,
    candidate_n: int = DEFAULT_CANDIDATE_N,
    beta: float = DEFAULT_SEMANTIC_BETA,
    min_match_score: float = 50.0,
    mode: str = "auto",
) -> list[tuple[dict[str, Any], float]]:
    candidates = session.top(n=candidate_n, min_match_score=min_match_score)
    if len(candidates) <= 1:
        return candidates[:n]
    if not _has_semantic_signal(session):
        return candidates[:n]

    recipes = [recipe for recipe, _ in candidates]
    bayes_scores = np.array([score for _, score in candidates], dtype=float)
    bayes_norm = _normalize_0_1(bayes_scores)

    query = build_preference_query(session.answers)
    query_embedding = encode_text(query)
    recipe_embeddings = _candidate_embeddings(recipes)
    semantic_scores = recipe_embeddings @ query_embedding
    semantic_norm = _normalize_0_1(semantic_scores)
    semantic_norm = _apply_feature_intent_boost(query, recipes, semantic_norm)

    beta = min(max(beta, 0.0), 1.0)
    if mode == "blend" or (mode == "auto" and _needs_stronger_semantic_signal(session)):
        final_scores = beta * bayes_norm + (1.0 - beta) * semantic_norm
        output_scale = 100.0
    else:
        final_scores = bayes_scores + (semantic_norm * TIE_BREAKER_POINTS)
        output_scale = 1.0

    order = np.argsort(-final_scores)

    return [
        (
            recipes[int(i)],
            round(min(max(float(final_scores[int(i)] * output_scale), 0.0), 100.0), 1),
        )
        for i in order[:n]
    ]


def semantic_candidate_pool(
    recipes: list[dict[str, Any]],
    query: str | None,
    pool_n: int = DEFAULT_POOL_N,
) -> list[dict[str, Any]]:
    query = (query or "").strip()
    if not query or len(query) < 3 or len(recipes) <= pool_n:
        return recipes

    query_embedding = encode_text(query)
    recipe_embeddings = _candidate_embeddings(recipes)
    semantic_scores = recipe_embeddings @ query_embedding
    order = np.argsort(-semantic_scores)
    return [recipes[int(i)] for i in order[:pool_n]]


def _needs_stronger_semantic_signal(session: BayesianSession) -> bool:
    if session.answers.get("semantic_query") or session.answers.get("craving_text"):
        return True

    answers = list(session.answers.values())
    if not answers:
        return False

    unknown_count = sum(1 for answer in answers if answer in ("skip", "unknown", "any") or answer == ["any"])
    unknown_ratio = unknown_count / len(answers)
    return session.entropy() >= HIGH_ENTROPY_THRESHOLD or unknown_ratio >= UNKNOWN_RATIO_THRESHOLD


def _has_semantic_signal(session: BayesianSession) -> bool:
    if session.answers.get("semantic_query") or session.answers.get("craving_text"):
        return True

    informative_boolean_count = 0
    for qid, answer in session.answers.items():
        q = question_by_id(qid)
        if q and q["type"] == "boolean" and answer in ("yes", "no"):
            informative_boolean_count += 1
    return informative_boolean_count >= 3


def _candidate_embeddings(recipes: list[dict[str, Any]]) -> np.ndarray:
    parsed: list[np.ndarray | None] = [_parse_embedding(recipe.get("embedding")) for recipe in recipes]
    missing = [recipes[i] for i, embedding in enumerate(parsed) if embedding is None]
    fallback_embeddings = encode_recipes(missing) if missing else np.empty((0, 384))

    output: list[np.ndarray] = []
    fallback_idx = 0
    for embedding in parsed:
        if embedding is None:
            output.append(fallback_embeddings[fallback_idx])
            fallback_idx += 1
        else:
            output.append(embedding)

    arr = np.vstack(output).astype(float)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, 1e-12)


def _apply_feature_intent_boost(
    query: str,
    recipes: list[dict[str, Any]],
    semantic_norm: np.ndarray,
) -> np.ndarray:
    intents = _infer_feature_intents(query)
    if not intents:
        return semantic_norm

    intent_scores = []
    for recipe in recipes:
        matched = 0
        for qid in intents:
            if bool(recipe.get(qid)):
                matched += 1
        intent_scores.append(matched / len(intents))

    intent_norm = np.array(intent_scores, dtype=float)
    return ((1.0 - FEATURE_INTENT_BLEND) * semantic_norm) + (FEATURE_INTENT_BLEND * intent_norm)


def _infer_feature_intents(query: str) -> list[str]:
    query = (query or "").strip()
    if len(query) < 3:
        return []

    labels = list(FEATURE_INTENT_TEXT.keys())
    texts = [query] + [FEATURE_INTENT_TEXT[label] for label in labels]
    embeddings = encode_texts(texts)
    scores = embeddings[1:] @ embeddings[0]
    order = np.argsort(-scores)

    inferred: list[str] = []
    for rank, idx in enumerate(order[:3]):
        idx = int(idx)
        score = float(scores[idx])
        next_score = float(scores[int(order[rank + 1])]) if rank + 1 < len(order) else 0.0
        if score >= FEATURE_INTENT_MIN_SCORE and score - next_score >= FEATURE_INTENT_MIN_MARGIN:
            inferred.append(labels[idx])

    return inferred


def _parse_embedding(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value.astype(float)
    if isinstance(value, list):
        return np.array(value, dtype=float)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = [float(part) for part in text.strip("[]").split(",") if part.strip()]
        return np.array(parsed, dtype=float)
    return None


def _normalize_0_1(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    lo = float(values.min())
    hi = float(values.max())
    if hi - lo < 1e-12:
        return np.ones_like(values)
    return (values - lo) / (hi - lo)


def _single_answer(value: Any) -> str | None:
    if value in (None, "skip", "unknown", "any") or value == ["any"]:
        return None
    if isinstance(value, list):
        return str(value[0]) if value else None
    return str(value)


def _list_answer(value: Any) -> list[str]:
    if value in (None, "skip", "unknown", "any") or value == ["any"]:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item != "any"]
    return [str(value)]
