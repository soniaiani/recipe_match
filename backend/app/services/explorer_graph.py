"""Ingredient graph queries for Ingredient Explorer."""
from __future__ import annotations

from statistics import mean

from app.database import get_supabase_admin
from app.models.explorer import ExplorerSuggestion
from app.services.explorer_common import chunks, normalize_ingredient


def fetch_ppmi_scores(candidates: set[str], selected: list[str]) -> dict[str, list[float]]:
    admin = get_supabase_admin()
    scores: dict[str, list[float]] = {candidate: [] for candidate in candidates}
    candidate_list = list(candidates)

    for candidate_batch in chunks(candidate_list, 100):
        rows = (
            admin.table("ingredient_graph")
            .select("ingredient_a,ppmi_score")
            .in_("ingredient_a", candidate_batch)
            .in_("ingredient_b", selected)
            .execute()
            .data or []
        )
        for row in rows:
            scores.setdefault(row["ingredient_a"], []).append(float(row["ppmi_score"]))
    return scores


def graph_suggestions(selected: list[str], selected_set: set[str]) -> list[ExplorerSuggestion]:
    admin = get_supabase_admin()
    rows_by_ingredient: dict[str, list[float]] = {}
    for ingredient in selected:
        rows = (
            admin.table("ingredient_graph")
            .select("ingredient_b,ppmi_score")
            .eq("ingredient_a", ingredient)
            .order("ppmi_score", desc=True)
            .limit(25)
            .execute()
            .data or []
        )
        for row in rows:
            candidate = normalize_ingredient(row.get("ingredient_b"))
            if not candidate or candidate in selected_set:
                continue
            rows_by_ingredient.setdefault(candidate, []).append(float(row["ppmi_score"]))

    suggestions = [
        ExplorerSuggestion(ingredient=ingredient, score=round(mean(scores), 6))
        for ingredient, scores in rows_by_ingredient.items()
    ]
    suggestions.sort(key=lambda item: item.score or 0.0, reverse=True)
    return suggestions[:5]
