from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Any, TypeVar

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.database import get_supabase_admin

T = TypeVar("T")

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS ingredient_graph (
  ingredient_a   TEXT NOT NULL,
  ingredient_b   TEXT NOT NULL,
  ppmi_score     FLOAT NOT NULL,
  co_occurrence  INTEGER NOT NULL,
  PRIMARY KEY (ingredient_a, ingredient_b)
);

CREATE TABLE IF NOT EXISTS ingredient_stats (
  ingredient     TEXT PRIMARY KEY,
  recipe_count   INTEGER NOT NULL,
  total_recipes  INTEGER NOT NULL
);
"""


def normalize_ingredient(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).lower().strip()
    return normalized or None


def parse_ingredients(raw: Any) -> set[str]:
    if not raw:
        return set()
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError:
        return set()
    if not isinstance(parsed, list):
        return set()
    return {
        normalized
        for item in parsed
        if (normalized := normalize_ingredient(item))
    }


def fetch_all_recipes() -> list[dict[str, Any]]:
    admin = get_supabase_admin()
    recipes: list[dict[str, Any]] = []
    offset = 0
    while True:
        page = (
            admin.table("recipes")
            .select("id,ingredients_clean")
            .range(offset, offset + 999)
            .execute()
            .data or []
        )
        recipes.extend(page)
        if len(page) < 1000:
            break
        offset += 1000
    return recipescorectare_explorer


def chunks(items: list[T], size: int):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def ensure_tables_exist() -> None:
    admin = get_supabase_admin()
    try:
        admin.table("ingredient_stats").select("ingredient").limit(1).execute()
        admin.table("ingredient_graph").select("ingredient_a").limit(1).execute()
    except Exception as exc:
        print("Ingredient Explorer tables are missing or inaccessible.")
        print("Run this SQL in the Supabase SQL Editor, then rerun the script:")
        print(CREATE_TABLE_SQL)
        raise RuntimeError("Missing ingredient explorer tables") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute ingredient PPMI graph into Supabase.")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--min-co-occurrence", type=int, default=5)
    args = parser.parse_args()

    started = time.perf_counter()
    ensure_tables_exist()

    recipes = fetch_all_recipes()
    total_recipes = len(recipes)
    if total_recipes == 0:
        raise RuntimeError("No recipes loaded from Supabase.")

    ingredient_to_recipes: dict[str, set[int]] = {}
    for recipe in recipes:
        recipe_id = int(recipe["id"])
        for ingredient in parse_ingredients(recipe.get("ingredients_clean")):
            ingredient_to_recipes.setdefault(ingredient, set()).add(recipe_id)

    stats_rows = [
        {
            "ingredient": ingredient,
            "recipe_count": len(recipe_ids),
            "total_recipes": total_recipes,
        }
        for ingredient, recipe_ids in ingredient_to_recipes.items()
    ]

    admin = get_supabase_admin()
    for batch in chunks(stats_rows, args.batch_size):
        admin.table("ingredient_stats").upsert(batch, on_conflict="ingredient").execute()

    pair_rows: list[dict[str, Any]] = []
    ingredients = sorted(ingredient_to_recipes)
    stored_pairs = 0

    for ingredient_a, ingredient_b in combinations(ingredients, 2):
        recipes_a = ingredient_to_recipes[ingredient_a]
        recipes_b = ingredient_to_recipes[ingredient_b]
        co_occurrence = len(recipes_a & recipes_b)
        if co_occurrence < args.min_co_occurrence:
            continue

        p_a = len(recipes_a) / total_recipes
        p_b = len(recipes_b) / total_recipes
        p_ab = co_occurrence / total_recipes
        pmi = math.log2(p_ab / (p_a * p_b))
        ppmi = max(pmi, 0.0)
        if ppmi <= 0:
            continue

        row_ab = {
            "ingredient_a": ingredient_a,
            "ingredient_b": ingredient_b,
            "ppmi_score": round(ppmi, 8),
            "co_occurrence": co_occurrence,
        }
        row_ba = {
            **row_ab,
            "ingredient_a": ingredient_b,
            "ingredient_b": ingredient_a,
        }
        pair_rows.extend([row_ab, row_ba])
        stored_pairs += 2

        if len(pair_rows) >= args.batch_size:
            admin.table("ingredient_graph").upsert(
                pair_rows,
                on_conflict="ingredient_a,ingredient_b",
            ).execute()
            pair_rows = []

    if pair_rows:
        admin.table("ingredient_graph").upsert(
            pair_rows,
            on_conflict="ingredient_a,ingredient_b",
        ).execute()

    elapsed = time.perf_counter() - started
    print("Ingredient graph precompute complete.")
    print(f"  total recipes loaded: {total_recipes}")
    print(f"  total unique ingredients found: {len(ingredient_to_recipes)}")
    print(f"  total pairs stored: {stored_pairs}")
    print(f"  elapsed time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
