# backend/scripts/backfill_saved_interactions.py
r"""Backfill recipe_interactions from existing saved_recipes.

This is safe to run multiple times: it skips user/recipe pairs that already
have a "save" interaction.

Run:
    cd D:\recipe_match
    $env:PYTHONIOENCODING='utf-8'
    backend\venv\Scripts\python.exe backend\scripts\backfill_saved_interactions.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(dotenv_path=ROOT / ".env")

from app.database import get_supabase_admin  # noqa: E402

PAGE_SIZE = 1000
SAVE_WEIGHT = 2.0


def load_paginated(table: str, fields: str, **eq_filters: Any) -> list[dict[str, Any]]:
    admin = get_supabase_admin()
    rows: list[dict[str, Any]] = []
    offset = 0
    while True:
        query = admin.table(table).select(fields).range(offset, offset + PAGE_SIZE - 1)
        for column, value in eq_filters.items():
            query = query.eq(column, value)
        page = query.execute().data or []
        rows.extend(page)
        if len(page) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
    return rows


def chunks(rows: list[dict[str, Any]], size: int):
    for start in range(0, len(rows), size):
        yield rows[start:start + size]


def main() -> None:
    admin = get_supabase_admin()

    saved_rows = load_paginated(
        "saved_recipes",
        "user_id,recipe_id,saved_at",
    )
    existing_rows = load_paginated(
        "recipe_interactions",
        "user_id,recipe_id",
        interaction_type="save",
    )
    existing = {
        (row.get("user_id"), int(row["recipe_id"]))
        for row in existing_rows
        if row.get("user_id") and row.get("recipe_id") is not None
    }

    inserts: list[dict[str, Any]] = []
    for row in saved_rows:
        user_id = row.get("user_id")
        recipe_id = row.get("recipe_id")
        if not user_id or recipe_id is None:
            continue

        key = (user_id, int(recipe_id))
        if key in existing:
            continue

        inserts.append({
            "user_id": user_id,
            "recipe_id": int(recipe_id),
            "interaction_type": "save",
            "weight": SAVE_WEIGHT,
            "created_at": row.get("saved_at"),
        })
        existing.add(key)

    print(f"saved_recipes rows: {len(saved_rows)}")
    print(f"existing save interactions: {len(existing_rows)}")
    print(f"new interactions to insert: {len(inserts)}")

    inserted = 0
    for batch in chunks(inserts, PAGE_SIZE):
        admin.table("recipe_interactions").insert(batch).execute()
        inserted += len(batch)
        print(f"Inserted {inserted}/{len(inserts)}")

    print("Done.")


if __name__ == "__main__":
    main()
