"""User profile builders for For You recommendations."""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from itertools import combinations
from typing import Any

from app.recommender.filters import normalize_excluded_ingredients
from app.services.foryou_common import (
    ALL_PROFILE_FEATURES,
    BOOLEAN_FEATURES,
    BOOLEAN_TEXT,
    CATEGORICAL_FEATURES,
    INTERACTION_WEIGHTS,
    MAX_EXPLORER_INTENTS,
    MAX_EXPLORER_INTENT_TERMS,
    MIN_EXPLORER_SESSIONS,
    MIN_INGREDIENT_FREQUENCY,
    RECENT_SESSIONS_LIMIT,
    RECENT_SIGNAL_RANK_DECAY,
    RECIPE_FIELDS,
    days_ago,
    interaction_recency_multiplier,
    normalize_answer_values,
    parse_ingredients,
    profile_ingredient_weight,
)


def build_interaction_profile(user_id: str, admin: Any) -> dict[str, Any]:
    """Build weighted implicit profile from interactions plus saved-only fallback."""
    empty = _empty_interaction_profile()
    try:
        interaction_rows, saved_rows = _fetch_interaction_rows(user_id, admin)
    except Exception as exc:
        print(f"[foryou] failed to build interaction profile: {exc}")
        return empty

    saved_ids = _saved_recipe_ids(saved_rows)
    real_save_ids = _interaction_save_ids(interaction_rows)
    events, recent_events = _collect_profile_events(
        interaction_rows,
        saved_rows,
        real_save_ids,
    )

    if not events:
        empty["saved_ids"] = saved_ids
        return empty

    return _summarize_profile_events(events, recent_events, saved_ids)


def build_answers_profile(user_id: str, admin: Any) -> dict[str, Any]:
    """Aggregate completed adaptive answers into stable active preferences."""
    empty = _empty_answers_profile()
    try:
        rows = _fetch_answer_rows(user_id, admin)
    except Exception as exc:
        print(f"[foryou] failed to build answers profile: {exc}")
        return empty

    value_counts, answered_counts = _aggregate_answer_rows(rows)
    return {
        "n_sessions": len(rows),
        "categorical": _active_categorical_preferences(value_counts, answered_counts),
        "booleans": _active_boolean_preferences(value_counts, answered_counts),
    }


def build_explorer_intents(
    user_id: str,
    admin: Any,
    excluded_ingredients: Any = None,
) -> list[list[str]]:
    """Extract frequent Explorer ingredient associations as separate intents."""
    excluded = set(normalize_excluded_ingredients(excluded_ingredients))
    try:
        rows = _fetch_explorer_rows(user_id, admin)
    except Exception as exc:
        print(f"[foryou] failed to build explorer profile: {exc}")
        return []

    if len(rows) < MIN_EXPLORER_SESSIONS:
        return []

    ingredient_counts, pair_counts = _count_explorer_cooccurrences(rows, excluded)
    intents = _connected_explorer_intents(ingredient_counts, pair_counts, len(rows))
    _append_single_ingredient_intents(intents, ingredient_counts, len(rows))
    return intents[:MAX_EXPLORER_INTENTS]


def build_explorer_profile(
    user_id: str,
    admin: Any,
    excluded_ingredients: Any = None,
) -> list[str]:
    """Compatibility helper returning flattened frequent Explorer ingredients."""
    seen: set[str] = set()
    ingredients: list[str] = []
    for intent in build_explorer_intents(user_id, admin, excluded_ingredients):
        for ingredient in intent:
            if ingredient not in seen:
                seen.add(ingredient)
                ingredients.append(ingredient)
    return ingredients


def build_semantic_profile(
    interaction_profile: dict[str, Any],
    answers_profile: dict[str, Any],
    explorer_ingredients: list[str] | None = None,
) -> str:
    parts: list[str] = []

    if explorer_ingredients:
        parts.extend(explorer_ingredients[:15])

    interaction_ingredients = interaction_profile.get("ingredients") or []
    parts.extend(interaction_ingredients[:10])

    for feature in CATEGORICAL_FEATURES:
        freqs = interaction_profile.get("categorical", {}).get(feature, {}) or {}
        parts.extend(value for value, _ in sorted(freqs.items(), key=lambda item: -item[1])[:3])
        parts.extend(answers_profile.get("categorical", {}).get(feature, []) or [])

    interaction_booleans = interaction_profile.get("booleans", {}) or {}
    for feature, value in interaction_booleans.items():
        if value >= 0.40:
            parts.append(BOOLEAN_TEXT.get(feature, feature))

    for feature, answer in (answers_profile.get("booleans", {}) or {}).items():
        if answer == "yes":
            parts.append(BOOLEAN_TEXT.get(feature, feature))

    return _deduplicate_profile_parts(parts)


def build_semantic_profiles(
    interaction_profile: dict[str, Any],
    answers_profile: dict[str, Any],
    explorer_intents: list[list[str]] | None = None,
) -> list[str]:
    """Build one semantic query per Explorer intent, combined with stable profile signals."""
    if not explorer_intents:
        profile = build_semantic_profile(interaction_profile, answers_profile)
        return [profile] if profile else []

    profiles = [
        profile
        for intent in explorer_intents[:MAX_EXPLORER_INTENTS]
        if (profile := build_semantic_profile(interaction_profile, answers_profile, intent))
    ]
    if profiles:
        return profiles

    fallback = build_semantic_profile(interaction_profile, answers_profile)
    return [fallback] if fallback else []


def _empty_interaction_profile() -> dict[str, Any]:
    return {
        "n_interactions": 0,
        "saved_ids": set(),
        "categorical": {feature: {} for feature in CATEGORICAL_FEATURES},
        "recent_categorical": {feature: {} for feature in CATEGORICAL_FEATURES},
        "booleans": {feature: 0.0 for feature in BOOLEAN_FEATURES},
        "ingredients": [],
        "recent_ingredients": [],
    }


def _fetch_interaction_rows(user_id: str, admin: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    interaction_rows = (
        admin.table("recipe_interactions")
        .select(f"recipe_id,interaction_type,weight,created_at,recipes({RECIPE_FIELDS})")
        .eq("user_id", user_id)
        .in_("interaction_type", list(INTERACTION_WEIGHTS.keys()))
        .order("created_at", desc=True)
        .execute()
        .data
        or []
    )
    saved_rows = (
        admin.table("saved_recipes")
        .select(f"recipe_id,saved_at,recipes({RECIPE_FIELDS})")
        .eq("user_id", user_id)
        .order("saved_at", desc=True)
        .execute()
        .data
        or []
    )
    return interaction_rows, saved_rows


def _saved_recipe_ids(saved_rows: list[dict[str, Any]]) -> set[int]:
    return {
        int(row["recipe_id"])
        for row in saved_rows
        if row.get("recipe_id") is not None
    }


def _interaction_save_ids(interaction_rows: list[dict[str, Any]]) -> set[int]:
    return {
        int(row["recipe_id"])
        for row in interaction_rows
        if row.get("recipe_id") is not None and row.get("interaction_type") == "save"
    }


def _collect_profile_events(
    interaction_rows: list[dict[str, Any]],
    saved_rows: list[dict[str, Any]],
    real_save_ids: set[int],
) -> tuple[list[tuple[dict[str, Any], float]], list[tuple[dict[str, Any], float]]]:
    events: list[tuple[dict[str, Any], float]] = []
    recent_events: list[tuple[dict[str, Any], float]] = []

    for rank, row in enumerate(interaction_rows):
        event = _interaction_event(row, rank)
        if event:
            _append_profile_event(events, recent_events, event, rank)

    for rank, row in enumerate(saved_rows):
        event = _saved_event(row, rank, real_save_ids)
        if event:
            _append_profile_event(events, recent_events, event, rank)

    return events, recent_events


def _interaction_event(row: dict[str, Any], rank: int) -> tuple[dict[str, Any], float] | None:
    recipe = row.get("recipes")
    interaction_type = row.get("interaction_type")
    if not recipe or interaction_type not in INTERACTION_WEIGHTS:
        return None

    days = days_ago(row.get("created_at"))
    weight = (
        INTERACTION_WEIGHTS[interaction_type]
        * math.exp(-0.02 * days)
        * interaction_recency_multiplier(days, rank)
    )
    return recipe, weight


def _saved_event(
    row: dict[str, Any],
    rank: int,
    real_save_ids: set[int],
) -> tuple[dict[str, Any], float] | None:
    recipe_id = row.get("recipe_id")
    recipe = row.get("recipes")
    if recipe_id is None or not recipe or int(recipe_id) in real_save_ids:
        return None

    days = days_ago(row.get("saved_at"))
    weight = (
        INTERACTION_WEIGHTS["save"]
        * math.exp(-0.02 * days)
        * interaction_recency_multiplier(days, rank)
    )
    return recipe, weight


def _append_profile_event(
    events: list[tuple[dict[str, Any], float]],
    recent_events: list[tuple[dict[str, Any], float]],
    event: tuple[dict[str, Any], float],
    rank: int,
) -> None:
    recipe, weight = event
    events.append((recipe, weight))
    recent_events.append((recipe, weight * math.exp(-RECENT_SIGNAL_RANK_DECAY * rank)))


def _summarize_profile_events(
    events: list[tuple[dict[str, Any], float]],
    recent_events: list[tuple[dict[str, Any], float]],
    saved_ids: set[int],
) -> dict[str, Any]:
    total_weight = sum(weight for _, weight in events)
    if total_weight <= 0:
        empty = _empty_interaction_profile()
        empty["saved_ids"] = saved_ids
        return empty

    categorical_counts = _weighted_categorical_counts(events)
    recent_total_weight = sum(weight for _, weight in recent_events)
    recent_categorical_counts = _weighted_categorical_counts(recent_events)
    boolean_sums = _weighted_boolean_sums(events)
    ingredient_counts = _weighted_ingredient_counts(events)
    recent_ingredient_counts = _weighted_ingredient_counts(recent_events)

    return {
        "n_interactions": len(events),
        "saved_ids": saved_ids,
        "categorical": _normalize_feature_counts(categorical_counts, total_weight),
        "recent_categorical": (
            _normalize_feature_counts(recent_categorical_counts, recent_total_weight)
            if recent_total_weight > 0
            else {feature: {} for feature in CATEGORICAL_FEATURES}
        ),
        "booleans": {
            feature: boolean_sums[feature] / total_weight
            for feature in BOOLEAN_FEATURES
        },
        "ingredients": [value for value, _ in ingredient_counts.most_common(15)],
        "recent_ingredients": [value for value, _ in recent_ingredient_counts.most_common(12)],
    }


def _weighted_categorical_counts(events: list[tuple[dict[str, Any], float]]) -> dict[str, Counter]:
    counts = {feature: Counter() for feature in CATEGORICAL_FEATURES}
    for recipe, weight in events:
        for feature in CATEGORICAL_FEATURES:
            value = recipe.get(feature)
            if value:
                counts[feature][str(value)] += weight
    return counts


def _weighted_boolean_sums(events: list[tuple[dict[str, Any], float]]) -> dict[str, float]:
    sums = {feature: 0.0 for feature in BOOLEAN_FEATURES}
    for recipe, weight in events:
        for feature in BOOLEAN_FEATURES:
            if recipe.get(feature) is True:
                sums[feature] += weight
    return sums


def _weighted_ingredient_counts(events: list[tuple[dict[str, Any], float]]) -> Counter:
    counts: Counter = Counter()
    for recipe, weight in events:
        ingredients = parse_ingredients(recipe.get("ingredients_clean") or recipe.get("ingredients"))
        for ingredient in ingredients:
            ingredient_weight = profile_ingredient_weight(ingredient)
            if ingredient_weight > 0:
                counts[ingredient] += weight * ingredient_weight
    return counts


def _normalize_feature_counts(
    counts_by_feature: dict[str, Counter],
    total_weight: float,
) -> dict[str, dict[str, float]]:
    return {
        feature: {
            value: count / total_weight
            for value, count in counts.items()
        }
        for feature, counts in counts_by_feature.items()
    }


def _empty_answers_profile() -> dict[str, Any]:
    return {
        "n_sessions": 0,
        "categorical": {feature: [] for feature in CATEGORICAL_FEATURES},
        "booleans": {},
    }


def _fetch_answer_rows(user_id: str, admin: Any) -> list[dict[str, Any]]:
    return (
        admin.table("recommendation_sessions")
        .select("answers,completed_at")
        .eq("user_id", user_id)
        .not_.is_("completed_at", "null")
        .order("completed_at", desc=True)
        .limit(RECENT_SESSIONS_LIMIT)
        .execute()
        .data
        or []
    )


def _aggregate_answer_rows(rows: list[dict[str, Any]]) -> tuple[dict[str, Counter], Counter]:
    value_counts: dict[str, Counter] = defaultdict(Counter)
    answered_counts: Counter = Counter()

    for row in rows:
        answers = _parse_answer_payload(row.get("answers") or {})
        if not isinstance(answers, dict):
            continue

        for feature in ALL_PROFILE_FEATURES:
            values = normalize_answer_values(answers.get(feature))
            if not values:
                continue
            answered_counts[feature] += 1
            for value in set(values):
                value_counts[feature][value] += 1

    return value_counts, answered_counts


def _parse_answer_payload(answers: Any) -> dict[str, Any]:
    if isinstance(answers, str):
        try:
            answers = json.loads(answers)
        except Exception:
            return {}
    return answers if isinstance(answers, dict) else {}


def _active_categorical_preferences(
    value_counts: dict[str, Counter],
    answered_counts: Counter,
) -> dict[str, list[str]]:
    categorical: dict[str, list[str]] = {feature: [] for feature in CATEGORICAL_FEATURES}
    for feature in CATEGORICAL_FEATURES:
        answered = answered_counts[feature]
        if answered <= 0:
            continue
        categorical[feature] = [
            value
            for value, count in value_counts[feature].items()
            if (count + 1.0) / (answered + 2.0) >= 0.40
        ]
    return categorical


def _active_boolean_preferences(
    value_counts: dict[str, Counter],
    answered_counts: Counter,
) -> dict[str, str]:
    booleans: dict[str, str] = {}
    for feature in BOOLEAN_FEATURES:
        answered = answered_counts[feature]
        if answered <= 0:
            continue

        yes_count = value_counts[feature].get("yes", 0)
        no_count = value_counts[feature].get("no", 0)
        yes_share = (yes_count + 1.0) / (answered + 2.0)
        no_share = (no_count + 1.0) / (answered + 2.0)
        if yes_share >= 0.40 or no_share >= 0.40:
            booleans[feature] = "yes" if yes_count >= no_count else "no"
    return booleans


def _fetch_explorer_rows(user_id: str, admin: Any) -> list[dict[str, Any]]:
    return (
        admin.table("explorer_sessions")
        .select("chain")
        .eq("user_id", user_id)
        .order("updated_at", desc=True)
        .limit(50)
        .execute()
        .data
        or []
    )


def _count_explorer_cooccurrences(
    rows: list[dict[str, Any]],
    excluded: set[str],
) -> tuple[Counter, Counter]:
    ingredient_counts: Counter = Counter()
    pair_counts: Counter = Counter()

    for row in rows:
        unique_chain = _unique_explorer_chain(row.get("chain") or [], excluded)
        if not unique_chain:
            continue
        ingredient_counts.update(unique_chain)
        for left, right in combinations(unique_chain, 2):
            pair_counts[(left, right)] += 1

    return ingredient_counts, pair_counts


def _unique_explorer_chain(chain: list[Any], excluded: set[str]) -> list[str]:
    return sorted({
        normalized
        for ingredient in chain
        if (normalized := str(ingredient).strip().lower())
        and normalized not in excluded
    })


def _connected_explorer_intents(
    ingredient_counts: Counter,
    pair_counts: Counter,
    n_sessions: int,
) -> list[list[str]]:
    graph = _frequent_pair_graph(pair_counts, n_sessions)
    intents: list[list[str]] = []
    seen: set[str] = set()

    for ingredient in sorted(graph, key=lambda item: -ingredient_counts[item]):
        if ingredient in seen:
            continue
        component = _graph_component(ingredient, graph)
        seen.update(component)
        intents.append(
            sorted(component, key=lambda item: -ingredient_counts[item])[:MAX_EXPLORER_INTENT_TERMS]
        )

    return intents


def _frequent_pair_graph(pair_counts: Counter, n_sessions: int) -> dict[str, set[str]]:
    graph: dict[str, set[str]] = defaultdict(set)
    for (left, right), count in pair_counts.items():
        if count / n_sessions >= MIN_INGREDIENT_FREQUENCY:
            graph[left].add(right)
            graph[right].add(left)
    return graph


def _graph_component(seed: str, graph: dict[str, set[str]]) -> set[str]:
    stack = [seed]
    component: set[str] = set()
    while stack:
        current = stack.pop()
        if current in component:
            continue
        component.add(current)
        stack.extend(graph[current] - component)
    return component


def _append_single_ingredient_intents(
    intents: list[list[str]],
    ingredient_counts: Counter,
    n_sessions: int,
) -> None:
    frequent = [
        ingredient
        for ingredient, count in ingredient_counts.items()
        if count / n_sessions >= MIN_INGREDIENT_FREQUENCY
    ]
    frequent.sort(key=lambda ingredient: -ingredient_counts[ingredient])

    used = {ingredient for intent in intents for ingredient in intent}
    for ingredient in frequent:
        if ingredient not in used:
            intents.append([ingredient])
        if len(intents) >= MAX_EXPLORER_INTENTS:
            return


def _deduplicate_profile_parts(parts: list[Any]) -> str:
    seen: set[str] = set()
    unique_parts: list[str] = []
    for part in parts:
        normalized = str(part).strip().lower()
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique_parts.append(normalized)
    return " ".join(unique_parts)
