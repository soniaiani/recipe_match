# backend/scripts/export_final_fyp_cluster_artifacts.py
r"""Export production FYP cluster artifacts for Supabase upload.

Run:
    cd D:\recipe_match
    $env:PYTHONIOENCODING='utf-8'
    backend\venv\Scripts\python.exe backend\scripts\export_final_fyp_cluster_artifacts.py
"""
from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "testari"))
load_dotenv(dotenv_path=ROOT / ".env")

from compare_recipe_clustering import (  # noqa: E402
    BOOLEAN_FEATURES,
    clustering_summary,
    compute_internal_metrics,
    fetch_all_recipes,
    parse_ingredients,
    build_matrix,
)
from evaluate_fyp_cluster_finetune import (  # noqa: E402
    build_hybrid_matrix,
    load_cached_ingredient_embeddings,
    pca_space,
)


MODEL_VERSION = "pantry_downweighted_kmeans_hybrid_bool125_pca25_k260_alpha008_seed42_v1"
WEIGHTS = {
    "ingredients": 0.90,
    "meal_type": 0.20,
    "cuisine": 0.15,
    "protein_type": 0.25,
    "booleans": 1.25,
}
METRICS = {
    "unique_clusters_at_20": 18.525,
    "max_cluster_at_20": 1.740,
    "pairwise_similarity_at_20": 0.6222,
    "score_loss_at_20": 0.0148,
    "ndcg_at_20": 0.9804,
}
PANTRY_HARD = {
    "salt",
    "water",
    "oil",
    "olive oil",
    "vegetable oil",
    "cooking spray",
    "black pepper",
    "pepper",
}
PANTRY_SOFT = {
    "sugar",
    "butter",
    "all-purpose flour",
    "flour",
    "egg",
    "milk",
}
INGREDIENT_PREVALENCE_MIN = 0.20
INGREDIENT_LIFT_MIN = 1.50
BOOLEAN_PREVALENCE_MIN = 0.30
BOOLEAN_LIFT_MIN = 1.25
CATEGORICAL_DISTINCTIVE_LIFT = 1.25
CATEGORICAL_FIELDS = ("cuisine", "meal_type", "protein_type")
MIN_GLOBAL_RECIPE_COUNT = 5
MAX_DISTINCTIVE_CLUSTER_COVERAGE = 0.50


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _l2_normalize(vector: np.ndarray) -> list[float]:
    norm = float(np.linalg.norm(vector))
    if norm <= 0:
        return vector.astype(float).tolist()
    return (vector / norm).astype(float).tolist()


def _ingredient_presence(recipe: dict[str, Any]) -> set[str]:
    return {
        ingredient
        for ingredient in parse_ingredients(recipe.get("ingredients_clean"))
        if ingredient not in PANTRY_HARD
    }


def _global_profile_statistics(recipes: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(recipes)
    ingredient_counts: Counter = Counter()
    boolean_counts: Counter = Counter()
    categorical_counts = {field: Counter() for field in CATEGORICAL_FIELDS}

    for recipe in recipes:
        ingredient_counts.update(_ingredient_presence(recipe))
        for feature in BOOLEAN_FEATURES:
            if recipe.get(feature) is True:
                boolean_counts[feature] += 1
        for field in CATEGORICAL_FIELDS:
            categorical_counts[field][str(recipe.get(field) or "unknown")] += 1

    return {
        "ingredients": {key: value / total for key, value in ingredient_counts.items()},
        "ingredient_counts": dict(ingredient_counts),
        "booleans": {key: value / total for key, value in boolean_counts.items()},
        "boolean_counts": dict(boolean_counts),
        "categorical": {
            field: {key: value / total for key, value in counts.items()}
            for field, counts in categorical_counts.items()
        },
    }


def _distinctive_traits(
    counts: Counter,
    cluster_size: int,
    global_prevalence: dict[str, float],
    global_counts: dict[str, int],
    prevalence_min: float,
    lift_min: float,
    limit: int,
) -> list[dict[str, Any]]:
    traits: list[dict[str, Any]] = []
    for name, count in counts.items():
        prevalence = count / cluster_size
        global_value = global_prevalence.get(name, 0.0)
        global_count = int(global_counts.get(name, 0))
        lift = prevalence / global_value if global_value > 0 else 0.0
        if (
            global_count < MIN_GLOBAL_RECIPE_COUNT
            or prevalence < prevalence_min
            or lift < lift_min
            or lift <= 1
        ):
            continue
        traits.append({
            "name": name,
            "global_recipe_count": global_count,
            "prevalence": round(prevalence, 4),
            "global_prevalence": round(global_value, 4),
            "lift": round(lift, 4),
            "score": round(prevalence * math.log2(lift + 1.0), 4),
            "is_globally_common": False,
        })
    traits.sort(key=lambda item: (-item["score"], -item["prevalence"], item["name"]))
    return traits[:limit]


def _mark_globally_common_traits(rows: list[dict[str, Any]], field: str) -> None:
    selected_cluster_counts = Counter(
        trait["name"]
        for row in rows
        for trait in row[field]
    )
    cluster_count = len(rows)
    for row in rows:
        for trait in row[field]:
            coverage = selected_cluster_counts[trait["name"]] / cluster_count
            trait["is_globally_common"] = coverage > MAX_DISTINCTIVE_CLUSTER_COVERAGE


def _dominant_categorical_trait(
    counts: Counter,
    cluster_size: int,
    global_prevalence: dict[str, float],
) -> dict[str, Any]:
    value, count = counts.most_common(1)[0]
    prevalence = count / cluster_size
    global_value = global_prevalence.get(value, 0.0)
    lift = prevalence / global_value if global_value > 0 else 0.0
    return {
        "value": value,
        "prevalence": round(prevalence, 4),
        "global_prevalence": round(global_value, 4),
        "lift": round(lift, 4),
        "is_distinctive": lift >= CATEGORICAL_DISTINCTIVE_LIFT,
    }


def _cluster_profiles(
    recipes: list[dict[str, Any]],
    cluster_x: np.ndarray,
    labels: np.ndarray,
) -> list[dict[str, Any]]:
    global_stats = _global_profile_statistics(recipes)
    by_cluster: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        by_cluster[int(label)].append(idx)

    rows: list[dict[str, Any]] = []
    for cluster_id, indices in sorted(by_cluster.items()):
        vectors = cluster_x[indices]
        centroid_array = vectors.mean(axis=0)
        centroid = np.asarray(_l2_normalize(centroid_array), dtype=np.float32)
        similarities = vectors @ centroid
        representative = [
            str(recipes[indices[int(local_idx)]].get("name") or "")
            for local_idx in np.argsort(-similarities)[:5]
        ]

        categorical_counts = {
            field: Counter(str(recipes[idx].get(field) or "unknown") for idx in indices)
            for field in CATEGORICAL_FIELDS
        }
        ingredients: Counter = Counter()
        booleans: Counter = Counter()
        for idx in indices:
            ingredients.update(_ingredient_presence(recipes[idx]))
            for feature in BOOLEAN_FEATURES:
                if recipes[idx].get(feature) is True:
                    booleans[feature] += 1

        ingredient_traits = _distinctive_traits(
            ingredients,
            len(indices),
            global_stats["ingredients"],
            global_stats["ingredient_counts"],
            INGREDIENT_PREVALENCE_MIN,
            INGREDIENT_LIFT_MIN,
            8,
        )
        boolean_traits = _distinctive_traits(
            booleans,
            len(indices),
            global_stats["booleans"],
            global_stats["boolean_counts"],
            BOOLEAN_PREVALENCE_MIN,
            BOOLEAN_LIFT_MIN,
            5,
        )
        categorical_traits = {
            field: _dominant_categorical_trait(
                categorical_counts[field],
                len(indices),
                global_stats["categorical"][field],
            )
            for field in CATEGORICAL_FIELDS
        }

        rows.append({
            "model_version": MODEL_VERSION,
            "cluster_id": cluster_id,
            "centroid": _l2_normalize(centroid_array),
            "size": len(indices),
            "dominant_cuisine": categorical_traits["cuisine"]["value"],
            "dominant_meal_type": categorical_traits["meal_type"]["value"],
            "dominant_protein_type": categorical_traits["protein_type"]["value"],
            "top_ingredients": [trait["name"] for trait in ingredient_traits],
            "top_ingredient_traits": ingredient_traits,
            "top_boolean_traits": boolean_traits,
            "categorical_traits": categorical_traits,
            "representative_recipes": [name for name in representative if name],
        })
    _mark_globally_common_traits(rows, "top_ingredient_traits")
    _mark_globally_common_traits(rows, "top_boolean_traits")
    return rows


def main() -> None:
    out_dir = ROOT / "testari" / "production_cluster_artifacts" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_recipes = fetch_all_recipes()
    recipes, _ranking_x = build_matrix(raw_recipes)
    cache_path = ROOT / "testari" / "cluster_embedding_cache" / "ingredient_embeddings_pantry_downweighted.npz"
    ingredient_embeddings = load_cached_ingredient_embeddings(recipes, cache_path)
    hybrid = build_hybrid_matrix(recipes, ingredient_embeddings, WEIGHTS)
    cluster_x, explained = pca_space(hybrid, 25, 42)
    cluster_model = KMeans(
        n_clusters=260,
        random_state=42,
        n_init=20,
        max_iter=300,
    )
    labels = cluster_model.fit_predict(cluster_x).astype(int)
    inertia = float(cluster_model.inertia_)

    summary = clustering_summary(labels)
    internal = compute_internal_metrics(cluster_x, labels, 2000, 42, inertia)
    metrics = {**METRICS, **summary, **internal}

    clusters = [
        {
            "recipe_id": int(recipe["id"]),
            "cluster_id": int(labels[idx]),
            "model_version": MODEL_VERSION,
        }
        for idx, recipe in enumerate(recipes)
    ]
    vectors = [
        {
            "recipe_id": int(recipe["id"]),
            "model_version": MODEL_VERSION,
            "vector": cluster_x[idx].astype(float).tolist(),
        }
        for idx, recipe in enumerate(recipes)
    ]
    profiles = _cluster_profiles(recipes, cluster_x, labels)
    model = {
        "model_version": MODEL_VERSION,
        "params": {
            "algorithm": "KMeans",
            "k": 260,
            "n_init": 20,
            "max_iter": 300,
            "seed": 42,
            "pca_components": 25,
            "pca_explained_variance": explained,
            "weights": WEIGHTS,
            "ingredient_variant": "pantry_downweighted",
            "reranking_alpha": 0.08,
            "cluster_characterization": {
                "score": "prevalence * log2(lift + 1), for lift > 1",
                "ingredient_thresholds": {
                    "prevalence_min": INGREDIENT_PREVALENCE_MIN,
                    "lift_min": INGREDIENT_LIFT_MIN,
                },
                "boolean_thresholds": {
                    "prevalence_min": BOOLEAN_PREVALENCE_MIN,
                    "lift_min": BOOLEAN_LIFT_MIN,
                },
                "categorical_distinctive_lift": CATEGORICAL_DISTINCTIVE_LIFT,
                "ingredient_counting": "presence_per_recipe",
                "minimum_global_recipe_count": MIN_GLOBAL_RECIPE_COUNT,
                "globally_common_if_selected_cluster_coverage_above": MAX_DISTINCTIVE_CLUSTER_COVERAGE,
            },
        },
        "metrics": metrics,
    }

    _write_csv(out_dir / f"recipe_clusters_{MODEL_VERSION}.csv", clusters)
    _write_json(out_dir / f"recipe_cluster_vectors_{MODEL_VERSION}.json", vectors)
    _write_json(out_dir / f"recipe_cluster_profiles_{MODEL_VERSION}.json", profiles)
    _write_json(out_dir / f"cluster_model_{MODEL_VERSION}.json", model)
    _write_json(out_dir / "manifest.json", {
        "model_version": MODEL_VERSION,
        "created_at": datetime.now().isoformat(),
        "files": {
            "clusters_csv": f"recipe_clusters_{MODEL_VERSION}.csv",
            "vectors_json": f"recipe_cluster_vectors_{MODEL_VERSION}.json",
            "profiles_json": f"recipe_cluster_profiles_{MODEL_VERSION}.json",
            "model_json": f"cluster_model_{MODEL_VERSION}.json",
        },
    })
    print(f"Exported production artifacts to {out_dir}")


if __name__ == "__main__":
    main()
