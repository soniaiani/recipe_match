# backend/scripts/analyze_question_bank.py
"""
Analizează question bank-ul:
1. Prevalența fiecărui feature (% rețete cu True)
2. Corelațiile Phi între toate perechile
3. Identifică features redundante și cu prevalență extremă
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import itertools
import numpy as np
from app.database import get_supabase_admin
from app.services.recommendations.bayesian.config import _ADAPTIVE_QS
from app.services.recommendations.bayesian.features import get_feature_value_bool

PAGE_SIZE = 1000


def load_all_recipes() -> list[dict]:
    admin = get_supabase_admin()
    recipes: list[dict] = []
    offset = 0

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


def analyze(recipes: list[dict]) -> None:
    n = len(recipes)
    questions = [(q["id"], q) for q in _ADAPTIVE_QS if q["type"] == "boolean"]

    print(f"\n{'='*60}")
    print(f"Corpus: {n} rețete\n")

    # ── 1. Prevalență ──────────────────────────────────────────
    print("PREVALENȚĂ (features cu <5% sau >95% sunt practic inutile)\n")
    print(f"{'Feature':<30} {'True':>8} {'%':>8}  Verdict")
    print("-" * 60)

    prevalences = {}
    for qid, question in questions:
        vals = np.array([get_feature_value_bool(r, question) for r in recipes])
        pct = vals.mean() * 100
        prevalences[qid] = pct
        if pct < 5 or pct > 95:
            verdict = "⚠️  SLAB (aproape uniform)"
        elif 30 <= pct <= 70:
            verdict = "✅ IDEAL"
        else:
            verdict = "ok"
        print(f"{qid:<30} {int(vals.sum()):>8} {pct:>7.1f}%  {verdict}")

    # ── 2. Corelații Phi ───────────────────────────────────────
    print(f"\n{'='*60}")
    print("CORELAȚII PHI > 0.4 (candidați pentru penalizare/eliminare)\n")
    print(f"{'Feature 1':<28} {'Feature 2':<28} {'Phi':>6}")
    print("-" * 66)

    high_corr_pairs = []
    for (qid1, question1), (qid2, question2) in itertools.combinations(questions, 2):
        v1 = np.array([get_feature_value_bool(r, question1) for r in recipes], dtype=float)
        v2 = np.array([get_feature_value_bool(r, question2) for r in recipes], dtype=float)
        if v1.std() > 0 and v2.std() > 0:
            phi = abs(np.corrcoef(v1, v2)[0, 1])
            if phi > 0.4:
                high_corr_pairs.append((qid1, qid2, phi))

    high_corr_pairs.sort(key=lambda x: -x[2])
    for q1, q2, phi in high_corr_pairs:
        print(f"{q1:<28} {q2:<28} {phi:>6.3f}")

    if not high_corr_pairs:
        print("  Nicio pereche cu phi > 0.4")

    # ── 3. Sumar ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    weak = [qid for qid, pct in prevalences.items() if pct < 5 or pct > 95]
    if weak:
        print(f"⚠️  Features cu prevalență extremă (candidați pentru eliminare):")
        for qid in weak:
            print(f"   - {qid}: {prevalences[qid]:.1f}%")
    else:
        print("✅ Nicio feature cu prevalență extremă.")

    print(f"\nPerechi cu corelație ridicată: {len(high_corr_pairs)}")
    print(f"Acestea sunt candidații pentru QUESTION_CORRELATIONS.\n")

if __name__ == "__main__":
    recipes = load_all_recipes()
    print(f"Loaded {len(recipes)} recipes from Supabase.")
    analyze(recipes)
