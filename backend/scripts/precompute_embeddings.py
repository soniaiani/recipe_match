from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.database import get_supabase_admin
from app.recommender.embeddings import encode_recipes

RECIPE_FIELDS = (
    "id,name,description,meal_type,cuisine,protein_type,"
    "ingredients,ingredients_clean,embedding"
)


def _chunks(items: list[dict], size: int):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def fetch_all_recipes(admin, fields: str) -> list[dict]:
    all_recipes = []
    offset = 0
    page_size = 1000

    while True:
        res = admin.table("recipes")\
            .select(fields)\
            .range(offset, offset + page_size - 1)\
            .execute()

        batch = res.data or []
        all_recipes.extend(batch)

        print(f"  Fetched {len(all_recipes)} recipes so far...")

        if len(batch) < page_size:
            break

        offset += page_size

    return all_recipes


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute recipe embeddings into Supabase pgvector.")
    parser.add_argument("--all", action="store_true", help="Recompute embeddings even when one already exists.")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    admin = get_supabase_admin()

    print("Fetching recipes from Supabase...")
    recipes = fetch_all_recipes(admin, RECIPE_FIELDS)

    if not args.all:
        recipes = [r for r in recipes if r.get("embedding") is None]

    print(f"Recipes to embed: {len(recipes)}")
    if not recipes:
        print("Nothing to do.")
        return

    done = 0
    for batch in _chunks(recipes, args.batch_size):
        embeddings = encode_recipes(batch)
        for recipe, embedding in zip(batch, embeddings):
            admin.table("recipes").update({
                "embedding": [float(x) for x in embedding.tolist()],
            }).eq("id", recipe["id"]).execute()
            done += 1
        print(f"Updated {done}/{len(recipes)}")

    print(f"\nDone. {done} recipes embedded.")


if __name__ == "__main__":
    main()