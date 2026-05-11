import json
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

DATASET_PATH = r"D:\folosire_api_claude\dataset_llm_labeled_with_ingredient_features.json"

BOOLEAN_FEATURES = [
    "is_spicy",
    "is_sweet",
    "is_quick",
    "needs_oven",
    "needs_stovetop",
    "is_no_cook",

    "has_pasta",
    "has_rice",
    "has_potato",
    "has_tomato_base",
    "has_cream_base",
    "has_cheese",
    "has_broth_base",
    "has_mushroom",
    "has_leafy_greens",
    "has_beans_legumes",
    "has_fruit",
    "has_nuts",
    "has_chocolate",
    "has_tortilla",
    "has_spicy_ingredient",
    "has_asian_sauce",
]

with open(DATASET_PATH, "r", encoding="utf-8") as f:
    raw = json.load(f)

recipes = []

for entry in raw:
    features = entry.get("llm_features")

    if not features or entry.get("llm_failed"):
        continue

    recipe = {
        "meal_type": features.get("meal_type", "other"),
        "protein_type": features.get("protein_type", "meatless"),
        "cuisine": features.get("cuisine", "other"),
    }

    for feature in BOOLEAN_FEATURES:
        recipe[feature] = int(bool(features.get(feature, False)))

    recipes.append(recipe)

# target combinat: categoria semantică mare a rețetei
target_labels = [
    f"{r['meal_type']}|{r['protein_type']}|{r['cuisine']}"
    for r in recipes
]

target = LabelEncoder().fit_transform(target_labels)

# X conține doar features de preferință + ingredient features
X = np.column_stack([
    [r[feature] for r in recipes]
    for feature in BOOLEAN_FEATURES
])

mi_scores = mutual_info_classif(
    X,
    target,
    discrete_features=True,
    random_state=42
)

# normalizare în weights utile pentru algoritmul weighted
max_mi = max(mi_scores) if len(mi_scores) else 1

weights = {
    feature: round(float(score / max_mi), 4) if max_mi > 0 else 0.0
    for feature, score in zip(BOOLEAN_FEATURES, mi_scores)
}

print("Mutual Information față de target combinat:")
print("-" * 70)

for feature, score in sorted(zip(BOOLEAN_FEATURES, mi_scores), key=lambda x: x[1], reverse=True):
    print(f"{feature:<25} MI = {score:.5f} | weight_norm = {weights[feature]:.4f}")

print("\nFEATURE_WEIGHTS pentru cod:")
print("-" * 70)

print("FEATURE_WEIGHTS = {")
print('    "meal_type": 3.00,')
print('    "is_chicken": 2.05,')
print('    "is_beef": 2.05,')
print('    "is_fish": 2.05,')
print('    "is_meatless": 2.05,')
print('    "is_italian": 1.15,')
print('    "is_asian": 1.15,')
print('    "is_mexican": 1.15,')
print('    "is_french": 1.15,')
print('    "is_mediterranean": 1.15,')
print('    "is_indian": 1.15,')
print('    "is_american": 1.15,')

for feature, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
    scaled_weight = round(0.50 + weight * 1.25, 2)
    print(f'    "{feature}": {scaled_weight:.2f},')

print("}")