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

INGREDIENT_FEATURES = [
    "has_pasta", "has_rice", "has_potato", "has_tomato_base", "has_cream_base",
    "has_cheese", "has_broth_base", "has_mushroom", "has_leafy_greens",
    "has_beans_legumes", "has_fruit", "has_nuts", "has_chocolate",
    "has_tortilla", "has_spicy_ingredient", "has_asian_sauce",
]

# Fetch toate rețetele din Supabase ca să avem name → id mapping
print("Fetch rețete din Supabase...")
all_db = []
offset = 0
while True:
    res = supabase.table("recipes").select("id, name").range(offset, offset + 999).execute()
    if not res.data:
        break
    all_db.extend(res.data)
    offset += 1000

name_to_id = {r["name"]: r["id"] for r in all_db}
print(f"Găsite {len(name_to_id)} rețete în DB")

# Update batch
batch_size = 50
updated = 0
skipped = 0

for i in range(0, len(dataset), batch_size):
    batch = dataset[i:i + batch_size]
    
    for r in batch:
        name = r.get("Name")
        features = r.get("llm_features") or {}
        
        if name not in name_to_id:
            skipped += 1
            continue
        
        recipe_id = name_to_id[name]
        update_data = {feat: features.get(feat, False) for feat in INGREDIENT_FEATURES}
        
        try:
            supabase.table("recipes").update(update_data).eq("id", recipe_id).execute()
            updated += 1
        except Exception as e:
            print(f"  Eroare {name}: {e}")
            skipped += 1
    
    print(f"  Progres: {updated} actualizate...")

print(f"\nFinal: {updated} actualizate, {skipped} sărite")

# # import json

# # with open(r"D:\folosire_api_claude\dataset_llm_labeled_with_ingredient_features.json", "r", encoding="utf-8") as f:
# #     data = json.load(f)

# # # Caută prima rețetă cu pasta în nume
# # for r in data:
# #     if "Bacon-Wrapped Cherries" in r.get("Name", ""):
# #         print(r["Name"])
# #         print("has_fruit:", r.get("llm_features", {}).get("has_fruit"))
# #         print("ingredients_clean:", r.get("ingredients_clean", [])[:5])
# #         break



