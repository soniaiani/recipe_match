import os
import json
import math
from dotenv import load_dotenv
from supabase import create_client

load_dotenv(r"D:\recipe_match\backend\.env")

supabase = create_client(
    os.environ.get("SUPABASE_URL"),
    os.environ.get("SUPABASE_KEY")
)

with open(r"D:\folosire_api_claude\dataset_llm_labeled_with_ingredient_features.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

print(f"Total retete: {len(dataset)}")


def clean_float(val):
    try:
        f = float(val)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def clean_str(val):
    if val is None:
        return None
    try:
        if isinstance(val, float) and math.isnan(val):
            return None
        return str(val)
    except:
        return None


def clean_int(val):
    try:
        f = float(val)
        return None if math.isnan(f) else int(f)
    except (TypeError, ValueError):
        return None


batch_size = 100
success = 0
failed = 0

for i in range(0, len(dataset), batch_size):
    batch = dataset[i:i + batch_size]
    rows = []

    for r in batch:
        features = r.get("llm_features") or {}
        try:
            rows.append({
                "name": clean_str(r.get("Name")),
                "description": clean_str(r.get("Description")),
                "prep_time": clean_str(r.get("Prep Time")),
                "cook_time": clean_str(r.get("Cook Time")),
                "total_time": clean_str(r.get("Total Time")),
                "servings": clean_int(r.get("Servings")),
                "ingredients": clean_str(r.get("Ingredients")),
                "image_url": clean_str(r.get("Image URL")),
                "total_minutes": clean_float(r.get("total_minutes")),
                "ingredients_clean": r.get("ingredients_clean", []),
                "ingredients_clean_str": clean_str(r.get("ingredients_clean_str")),
                "directions": clean_str(r.get("directions")),
                "meal_type": features.get("meal_type"),
                "protein_type": features.get("protein_type"),
                "cuisine": features.get("cuisine"),
                "is_vegetarian": features.get("is_vegetarian", False),
                "is_vegan": features.get("is_vegan", False),
                "is_gluten_free": features.get("is_gluten_free", False),
                "is_dairy_free": features.get("is_dairy_free", False),
                "is_nut_free": features.get("is_nut_free", False),
                "is_spicy": features.get("is_spicy", False),
                "is_sweet": features.get("is_sweet", False),
                "is_quick": features.get("is_quick", False),
                "needs_oven": features.get("needs_oven", False),
                "needs_stovetop": features.get("needs_stovetop", False),
                "is_no_cook": features.get("is_no_cook", False),
                "has_pasta": features.get("has_pasta", False),
                "has_rice": features.get("has_rice", False),
                "has_potato": features.get("has_potato", False),
                "has_tomato_base": features.get("has_tomato_base", False),
                "has_cream_base": features.get("has_cream_base", False),
                "has_cheese": features.get("has_cheese", False),
                "has_broth_base": features.get("has_broth_base", False),
                "has_mushroom": features.get("has_mushroom", False),
                "has_leafy_greens": features.get("has_leafy_greens", False),
                "has_beans_legumes": features.get("has_beans_legumes", False),
                "has_fruit": features.get("has_fruit", False),
                "has_nuts": features.get("has_nuts", False),
                "has_chocolate": features.get("has_chocolate", False),
                "has_tortilla": features.get("has_tortilla", False),
                "has_spicy_ingredient": features.get("has_spicy_ingredient", False),
                "has_asian_sauce": features.get("has_asian_sauce", False),
            })
        except Exception as e:
            print(f"  Skip {r.get('Name', '')}: {e}")
            failed += 1

    try:
        supabase.table("recipes").insert(rows).execute()
        success += len(rows)
        print(f"  Importat batch {i // batch_size + 1} -- {success} retete")
    except Exception as e:
        print(f"  Eroare batch {i // batch_size + 1}: {e}")
        failed += len(rows)

print(f"\nFinal: {success} importate, {failed} esuate")