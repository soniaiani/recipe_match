# backend/scripts/analyze_exclusions.py
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.database import get_supabase_admin
from app.services.recommendations.bayesian.config import _ADAPTIVE_QS
from app.services.recommendations.bayesian.features import get_feature_value_bool

MEAL_TYPES = ["dessert", "drink", "breakfast", "soup", "snack", "lunch_dinner", "appetizer", "salad_side", "condiment"]
def load_all_recipes() -> list[dict]:
    admin = get_supabase_admin()
    recipes: list[dict] = []
    offset = 0
    PAGE_SIZE = 1000

    while True:
        resp = (
            admin.table("recipes")
            .select("*")
            .range(offset, offset + PAGE_SIZE - 1)
            .execute()
        )
        page = resp.data or []
        recipes.extend(page)
        if len(page) < PAGE_SIZE:
            break
        offset += PAGE_SIZE

    return recipes
def analyze():
    recipes = load_all_recipes()    
    # grupează pe meal_type
    groups: dict[str, list[dict]] = {mt: [] for mt in MEAL_TYPES}
    for r in recipes:
        mt = r.get("meal_type", "unknown")
        if mt in groups:
            groups[mt].append(r)

    for mt, recs in groups.items():
        if not recs:
            continue
        n = len(recs)
        print(f"\n{mt.upper()} ({n} rețete)")
        print(f"  {'Feature':<28} {'%True':>7}  Verdict")
        print(f"  {'-'*45}")
        for q in _ADAPTIVE_QS:
            vals = [get_feature_value_bool(r, q) for r in recs]
            pct = sum(vals) / n * 100
            # flaghează features cu prevalență <3% — candidați pentru excludere
            flag = "  <-- exclude?" if pct < 3 else ""
            print(f"  {q['id']:<28} {pct:>6.1f}%{flag}")

if __name__ == "__main__":
    analyze()
