"""
LDA (Latent Dirichlet Allocation) pe ingrediente cu stopwords eliminate.
Genereaza topic-uri interpretabile pentru question bank.

Usage:
    pip install scikit-learn
    python lda_ingredients.py
"""
from __future__ import annotations
import json
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from collections import Counter

DATASET_PATH = r"D:\folosire_api_claude\dataset_llm_labeled.json"
OUTPUT_PATH  = r"D:\folosire_api_claude\lda_topics.json"
N_TOPICS     = 20
N_TOP_WORDS  = 12

# ── STOPWORDS CULINARE ────────────────────────────────────────────────────────
# Ingrediente universale care apar in aproape toate retetele
# si nu discrimineaza intre tipuri de preparate
CULINARY_STOPWORDS = {
    "salt", "pepper", "water", "oil", "olive oil", "vegetable oil",
    "butter", "garlic", "onion", "black pepper", "white pepper",
    "sugar", "flour", "egg", "milk", "cream", "all purpose flour",
    "baking powder", "baking soda", "vanilla extract", "salt pepper",
    "cooking spray", "nonstick spray", "to taste", "optional",
    "fresh", "dried", "ground", "chopped", "minced", "sliced",
    "diced", "cup", "tablespoon", "teaspoon", "pound", "ounce",
}

# ── INCARCARE DATE ────────────────────────────────────────────────────────────
print("Incarc dataset...")
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    raw = json.load(f)

recipes = []
for idx, entry in enumerate(raw):
    features = entry.get("llm_features")
    if not features or entry.get("llm_failed"):
        continue
    ingredients = entry.get("ingredients_clean", []) or []
    if not ingredients:
        continue

    # elimina stopwords culinare
    clean = [i for i in ingredients if i.lower() not in CULINARY_STOPWORDS]
    if len(clean) < 2:
        continue

    recipes.append({
        "id": idx,
        "name": entry.get("Name", ""),
        "ingredients_filtered": " ".join(clean),
        "meal_type": features.get("meal_type"),
        "cuisine": features.get("cuisine"),
        "protein_type": features.get("protein_type"),
    })

print(f"Retete procesate: {len(recipes)}")

# ── COUNT VECTORIZER ──────────────────────────────────────────────────────────
print("\nVectorizez ingrediente (fara stopwords)...")
corpus = [r["ingredients_filtered"] for r in recipes]

vectorizer = CountVectorizer(
    max_features=400,
    min_df=15,        # apare in minim 15 retete
    max_df=0.60,      # nu apare in mai mult de 60% din retete
    ngram_range=(1, 2),  # unigrams si bigrams
)
X = vectorizer.fit_transform(corpus)
vocab = vectorizer.get_feature_names_out()
print(f"Vocabular: {len(vocab)} termeni")

# ── LDA ───────────────────────────────────────────────────────────────────────
print(f"\nAntrenez LDA cu {N_TOPICS} topic-uri...")
lda = LatentDirichletAllocation(
    n_components=N_TOPICS,
    random_state=42,
    max_iter=20,
    learning_method="batch",
    doc_topic_prior=0.1,   # sparse topics per document
    topic_word_prior=0.01, # sparse words per topic
)
doc_topics = lda.fit_transform(X)
print(f"Perplexity: {lda.perplexity(X):.1f}")

# ── ANALIZA TOPIC-URI ─────────────────────────────────────────────────────────
print("\nTop cuvinte per topic:")
topics = {}
for topic_id, topic_weights in enumerate(lda.components_):
    top_indices = topic_weights.argsort()[-N_TOP_WORDS:][::-1]
    top_words = [vocab[i] for i in top_indices]

    # retete dominante in acest topic
    dominant_recipe_indices = doc_topics[:, topic_id].argsort()[-20:][::-1]
    dominant_recipes = [recipes[i] for i in dominant_recipe_indices]

    meal_dist = Counter(r["meal_type"] for r in dominant_recipes)
    cuisine_dist = Counter(r["cuisine"] for r in dominant_recipes)
    protein_dist = Counter(r["protein_type"] for r in dominant_recipes)

    print(f"\nTopic {topic_id:2d}: {', '.join(top_words[:6])}")
    print(f"          Meal: {dict(meal_dist.most_common(3))}")
    print(f"          Cuisine: {dict(cuisine_dist.most_common(2))}")

    topics[topic_id] = {
        "id": topic_id,
        "top_words": top_words,
        "meal_distribution": dict(meal_dist.most_common(5)),
        "cuisine_distribution": dict(cuisine_dist.most_common(3)),
        "protein_distribution": dict(protein_dist.most_common(3)),
        "sample_recipes": [r["name"] for r in dominant_recipes[:5]],
    }

# ── ASIGNEAZA TOPIC DOMINANT LA FIECARE RETETA ───────────────────────────────
print("\nAsignez topic dominant la fiecare reteta...")
recipe_topics = {}
for i, recipe in enumerate(recipes):
    dominant_topic = int(doc_topics[i].argmax())
    topic_strength = float(doc_topics[i][dominant_topic])
    recipe_topics[recipe["id"]] = {
        "topic_id": dominant_topic,
        "topic_strength": round(topic_strength, 4),
        "top_words": topics[dominant_topic]["top_words"][:4],
    }

# ── DISTRIBUTIA TOPIC-URILOR ──────────────────────────────────────────────────
topic_counts = Counter(v["topic_id"] for v in recipe_topics.values())
print("\nDistributia topic-urilor:")
for tid, count in sorted(topic_counts.items()):
    pct = count / len(recipes) * 100
    print(f"  Topic {tid:2d}: {count:4d} retete ({pct:.1f}%) — {', '.join(topics[tid]['top_words'][:4])}")

# ── SALVEAZA ──────────────────────────────────────────────────────────────────
output = {
    "n_topics": N_TOPICS,
    "vocabulary_size": len(vocab),
    "topics": topics,
    "recipe_topics": recipe_topics,
}
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"\nSalvat in {OUTPUT_PATH}")

# ── INTREBARI SUGERATE PENTRU QUESTION BANK ───────────────────────────────────
print("\n" + "="*60)
print("TOPIC-URI INTERPRETABILE — INTREBARI POSIBILE:")
print("="*60)
for tid, topic in topics.items():
    words = topic["top_words"][:5]
    dominant_meal = list(topic["meal_distribution"].keys())[0] if topic["meal_distribution"] else "?"
    dominant_cuisine = list(topic["cuisine_distribution"].keys())[0] if topic["cuisine_distribution"] else "?"
    samples = topic["sample_recipes"][:3]
    print(f"\nTopic {tid:2d} [{dominant_meal}/{dominant_cuisine}]")
    print(f"  Cuvinte: {', '.join(words)}")
    print(f"  Exemple: {', '.join(samples)}")