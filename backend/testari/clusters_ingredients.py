"""
Clustering pe ingrediente pentru a genera features noi pentru question bank.
Foloseste TF-IDF + KMeans pe ingredients_clean_str.

Usage:
    pip install scikit-learn
    python cluster_ingredients.py
"""
from __future__ import annotations
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from collections import Counter

DATASET_PATH = r"D:\folosire_api_claude\dataset_llm_labeled.json"
N_CLUSTERS = 12  # vom testa 8-16 si alegem pe baza silhouette score
OUTPUT_PATH = r"D:\folosire_api_claude\ingredient_clusters.json"

# ── INCARCARE DATE ────────────────────────────────────────────────────────────
print("Incarc dataset...")
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    raw = json.load(f)

recipes = []
for idx, entry in enumerate(raw):
    features = entry.get("llm_features")
    if not features or entry.get("llm_failed"):
        continue
    ingredients_str = entry.get("ingredients_clean_str", "") or ""
    if not ingredients_str:
        continue
    recipes.append({
        "id": idx,
        "name": entry.get("Name", ""),
        "ingredients_str": ingredients_str,
        "meal_type": features.get("meal_type"),
        "cuisine": features.get("cuisine"),
    })

print(f"Retete cu ingrediente: {len(recipes)}")

# ── TF-IDF PE INGREDIENTE ─────────────────────────────────────────────────────
print("\nCalculez TF-IDF pe ingrediente...")
corpus = [r["ingredients_str"] for r in recipes]

vectorizer = TfidfVectorizer(
    max_features=300,      # top 300 ingrediente
    min_df=10,             # apare in minim 10 retete
    max_df=0.7,            # nu apare in mai mult de 70% din retete (elimina sare etc.)
    ngram_range=(1, 1),
)
X = vectorizer.fit_transform(corpus)
feature_names = vectorizer.get_feature_names_out()
print(f"Vocabular TF-IDF: {len(feature_names)} ingrediente")

# ── GASESTE NUMARUL OPTIM DE CLUSTERE ────────────────────────────────────────
print("\nTestez numarul optim de clustere (8-16)...")
silhouette_scores = {}
for k in range(8, 17):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    score = silhouette_score(X, labels, sample_size=2000, random_state=42)
    silhouette_scores[k] = score
    print(f"  k={k}: silhouette={score:.4f}")

best_k = max(silhouette_scores, key=silhouette_scores.get)
print(f"\nNumar optim de clustere: {best_k} (silhouette={silhouette_scores[best_k]:.4f})")

# ── CLUSTERING FINAL ──────────────────────────────────────────────────────────
print(f"\nRunning KMeans cu k={best_k}...")
km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
labels = km_final.fit_predict(X)

# ── FUNCTIE GENERARE NUME CLUSTER ────────────────────────────────────────────
def _generate_cluster_name(ingredients: list[str], meal_type: str, cuisine: str) -> str:
    ing_set = set(ingredients[:6])
    if "chocolate" in ing_set or "cocoa" in ing_set:
        return "chocolate_baking"
    if "pasta" in ing_set or "spaghetti" in ing_set or "noodle" in ing_set:
        return "pasta_dishes"
    if "chicken" in ing_set or "turkey" in ing_set:
        return "chicken_dishes"
    if "beef" in ing_set or "pork" in ing_set or "sausage" in ing_set:
        return "meat_dishes"
    if "fish" in ing_set or "shrimp" in ing_set or "salmon" in ing_set:
        return "seafood_dishes"
    if "tomato" in ing_set and "bean" in ing_set:
        return "legume_tomato"
    if "cream" in ing_set or "cheese" in ing_set or "milk" in ing_set:
        return "dairy_rich"
    if "rice" in ing_set and cuisine == "asian":
        return "asian_rice"
    if "flour" in ing_set and "sugar" in ing_set:
        return "baked_sweets"
    if "potato" in ing_set or "carrot" in ing_set:
        return "root_vegetables"
    if "lemon" in ing_set or "lime" in ing_set or "vinegar" in ing_set:
        return "citrus_acid"
    if "spice" in ing_set or "cumin" in ing_set or "curry" in ing_set:
        return "spiced_dishes"
    return f"cluster_{meal_type}_{cuisine}"

# ── ANALIZA CLUSTERE ──────────────────────────────────────────────────────────
print("\nAnaliza clustere:")
clusters = {}
for cluster_id in range(best_k):
    cluster_recipes = [recipes[i] for i, l in enumerate(labels) if l == cluster_id]

    # top ingrediente din centroid
    center = km_final.cluster_centers_[cluster_id]
    top_ingredient_indices = center.argsort()[-10:][::-1]
    top_ingredients = [feature_names[i] for i in top_ingredient_indices]

    # distributia meal_type in cluster
    meal_types = Counter(r["meal_type"] for r in cluster_recipes)
    cuisines = Counter(r["cuisine"] for r in cluster_recipes)
    dominant_meal = meal_types.most_common(1)[0][0] if meal_types else "unknown"
    dominant_cuisine = cuisines.most_common(1)[0][0] if cuisines else "unknown"

    # genereaza un nume descriptiv pentru cluster
    cluster_name = _generate_cluster_name(top_ingredients, dominant_meal, dominant_cuisine)

    clusters[cluster_id] = {
        "id": cluster_id,
        "name": cluster_name,
        "size": len(cluster_recipes),
        "top_ingredients": top_ingredients,
        "dominant_meal_type": dominant_meal,
        "dominant_cuisine": dominant_cuisine,
        "meal_distribution": dict(meal_types.most_common(5)),
        "cuisine_distribution": dict(cuisines.most_common(5)),
    }

    print(f"\nCluster {cluster_id} — '{cluster_name}' ({len(cluster_recipes)} retete)")
    print(f"  Top ingrediente: {', '.join(top_ingredients[:6])}")
    print(f"  Dominant: {dominant_meal} / {dominant_cuisine}")
    print(f"  Meal types: {dict(meal_types.most_common(3))}")





# ── ASIGNEAZA CLUSTER LA FIECARE RETETA ──────────────────────────────────────
print("\nAsignez clustere la retete...")
recipe_clusters = {}
for i, recipe in enumerate(recipes):
    recipe_clusters[recipe["id"]] = {
        "cluster_id": int(labels[i]),
        "cluster_name": clusters[int(labels[i])]["name"],
    }

# ── SALVEAZA REZULTATELE ──────────────────────────────────────────────────────
output = {
    "n_clusters": best_k,
    "clusters": clusters,
    "recipe_clusters": recipe_clusters,
    "vocabulary_size": len(feature_names),
    "top_ingredients_vocab": list(feature_names[:50]),
}

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"\nRezultate salvate in {OUTPUT_PATH}")

# ── INTREBARI PENTRU QUESTION BANK ───────────────────────────────────────────
print("\n" + "="*60)
print("INTREBARI NOI PENTRU QUESTION BANK:")
print("="*60)
for cid, cluster in clusters.items():
    print(f'\n{{"id": "has_{cluster["name"]}", "type": "boolean", '
          f'"feature": "ingredient_cluster", "feature_value": {cid}, '
          f'"fixed": False, "text": "Do you want a recipe with {cluster["name"].replace("_", " ")}?", '
          f'"options": ["yes", "no", "skip"]}}')