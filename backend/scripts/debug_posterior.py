# backend/scripts/debug_posterior.py
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from app.database import get_supabase_admin
from app.recommender.engine import restore_session, compute_feature_mi

admin = get_supabase_admin()
recipes = []
offset = 0
while True:
    page = admin.table('recipes').select('*').range(offset, offset+999).execute().data or []
    recipes.extend(page)
    if len(page) < 1000:
        break
    offset += 1000

# în debug_posterior.py, înlocuiește answers cu:

answers = {
    'meal_type': 'lunch_dinner',
    'protein_type': ['chicken'],
    'cuisine': ['italian'],
    'is_spicy': 'no',
    'has_pasta': 'yes',
    'has_cream_base': 'yes',
}

weights = compute_feature_mi(recipes)
session = restore_session(answers, recipes, weights)
p = session.probs()
uniform = 1.0 / session.n
lifts = p / uniform

print(f'Retete cu lift > 1: {(lifts > 1).sum()}')
print(f'Retete cu lift > 5: {(lifts > 5).sum()}')
print(f'Retete cu lift > 9: {(lifts > 9).sum()}')
print(f'Lift max: {lifts.max():.1f}')
print(f'Lift min: {lifts.min():.6f}')
print(f'Lift std: {lifts.std():.3f}')
print(f'Entropie: {session.entropy():.2f} bits (max={np.log2(len(recipes)):.1f})')
print()

top_idx = p.argsort()[-5:][::-1]
for i in top_idx:
    r = recipes[i]
    name = r.get('name', '?')
    mt = r.get('meal_type')
    cu = r.get('cuisine')
    pt = r.get('protein_type')
    print(f"  {name} | {mt} | {cu} | {pt} | lift={lifts[i]:.2f}")

print()
print("Scoruri posterior (k=4):")
for i in top_idx:
    r = recipes[i]
    p_score = 100.0 * lifts[i] / (lifts[i] + 4.0)
    print(f"  {r.get('name','?')} | lift={lifts[i]:.2f} | posterior_score={p_score:.1f}%")
    