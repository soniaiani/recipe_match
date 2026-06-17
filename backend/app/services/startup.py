"""Application startup loading for recommendation-related state."""
from __future__ import annotations

from typing import Any

from fastapi import FastAPI

from app.services.explorer import service as explorer_service

_REC_RECIPE_FIELDS = (
    "id,name,image_url,meal_type,protein_type,cuisine,total_minutes,"
    "description,ingredients,ingredients_clean,embedding,"
    "is_vegetarian,is_vegan,is_gluten_free,is_dairy_free,"
    "is_spicy,is_sweet,is_quick,needs_oven,needs_stovetop,is_no_cook,"
    "has_pasta,has_rice,has_potato,has_tomato_base,has_cream_base,"
    "has_cheese,has_broth_base,has_mushroom,has_leafy_greens,"
    "has_beans_legumes,has_fruit,has_nuts,has_chocolate,"
    "has_tortilla,has_spicy_ingredient,has_asian_sauce"
)

_FYP_CLUSTER_MODEL_VERSION = "pantry_downweighted_kmeans_hybrid_bool125_pca25_k260_alpha008_seed42_v1"


def initialize_recommendation_state(app: FastAPI) -> None:
    """Load startup state used by Bayesian, Explorer and For You engines."""
    try:
        from app.database import get_supabase_admin
        from app.recommender.engine import compute_feature_mi

        admin = get_supabase_admin()
        recipes = _fetch_all_recipes(admin)

        app.state.rec_recipes = recipes
        app.state.rec_weights = compute_feature_mi(recipes)
        print(f"[startup] Loaded {len(recipes)} recipes for Bayesian engine.")

    except Exception as exc:
        _reset_recommendation_state(app)
        print(f"[startup] WARNING: failed to load Bayesian engine data: {exc}")
        return

    _warm_semantic_embedding_model()
    _warm_explorer_resources()
    _load_recipe_clusters(app, admin)


def _fetch_all_recipes(admin: Any) -> list[dict[str, Any]]:
    recipes: list[dict[str, Any]] = []
    offset = 0
    while True:
        page = (
            admin.table("recipes")
            .select(_REC_RECIPE_FIELDS)
            .range(offset, offset + 999)
            .execute()
            .data or []
        )
        recipes.extend(page)
        if len(page) < 1000:
            return recipes
        offset += 1000


def _warm_semantic_embedding_model() -> None:
    try:
        from app.recommender.embeddings import encode_text

        encode_text("warmup recipe preferences")
        print("[startup] Warmed semantic embedding model.")
    except Exception as exc:
        print(f"[startup] WARNING: failed to warm embedding model: {exc}")


def _warm_explorer_resources() -> None:
    try:
        explorer_service.warm_explorer_cache()
        explorer_service.warm_ingredient_idf()
        explorer_service.warm_explorer_ltr_model()
        print("[startup] Warmed Ingredient Explorer cache.")
    except Exception as exc:
        print(f"[startup] WARNING: failed to warm Ingredient Explorer resources: {exc}")


def _load_recipe_clusters(app: FastAPI, admin: Any) -> None:
    try:
        cluster_rows = _fetch_recipe_clusters(admin)
        app.state.recipe_clusters = {
            int(row["recipe_id"]): int(row["cluster_id"])
            for row in cluster_rows
            if row.get("recipe_id") is not None and row.get("cluster_id") is not None
        }
        app.state.recipe_cluster_vectors = _fetch_recipe_cluster_vectors(admin)
        app.state.recipe_cluster_profiles = _fetch_recipe_cluster_profiles(admin)
        app.state.recipe_cluster_centroids = {
            cluster_id: profile["centroid"]
            for cluster_id, profile in app.state.recipe_cluster_profiles.items()
            if profile.get("centroid")
        }
        app.state.recipe_cluster_model_version = _FYP_CLUSTER_MODEL_VERSION
        app.state.cluster_aware_fyp = bool(app.state.recipe_clusters)
        print(
            "[startup] Loaded "
            f"{len(app.state.recipe_clusters)} recipe clusters for FYP "
            f"model={_FYP_CLUSTER_MODEL_VERSION}; "
            f"vectors={len(app.state.recipe_cluster_vectors)} "
            f"profiles={len(app.state.recipe_cluster_profiles)}."
        )
    except Exception as exc:
        app.state.recipe_clusters = {}
        app.state.recipe_cluster_vectors = {}
        app.state.recipe_cluster_profiles = {}
        app.state.recipe_cluster_centroids = {}
        app.state.recipe_cluster_model_version = None
        app.state.cluster_aware_fyp = False
        print(f"[startup] WARNING: failed to load FYP recipe clusters: {exc}")


def _fetch_recipe_clusters(admin: Any) -> list[dict[str, Any]]:
    cluster_rows: list[dict[str, Any]] = []
    offset = 0
    while True:
        page = (
            admin.table("recipe_clusters")
            .select("recipe_id,cluster_id")
            .eq("model_version", _FYP_CLUSTER_MODEL_VERSION)
            .range(offset, offset + 999)
            .execute()
            .data or []
        )
        cluster_rows.extend(page)
        if len(page) < 1000:
            return cluster_rows
        offset += 1000


def _fetch_recipe_cluster_vectors(admin: Any) -> dict[int, list[float]]:
    try:
        rows: list[dict[str, Any]] = []
        offset = 0
        while True:
            page = (
                admin.table("recipe_cluster_vectors")
                .select("recipe_id,vector")
                .eq("model_version", _FYP_CLUSTER_MODEL_VERSION)
                .range(offset, offset + 999)
                .execute()
                .data or []
            )
            rows.extend(page)
            if len(page) < 1000:
                break
            offset += 1000
        return {
            int(row["recipe_id"]): [float(value) for value in (row.get("vector") or [])]
            for row in rows
            if row.get("recipe_id") is not None and row.get("vector")
        }
    except Exception as exc:
        print(f"[startup] WARNING: failed to load FYP recipe vectors: {exc}")
        return {}


def _fetch_recipe_cluster_profiles(admin: Any) -> dict[int, dict[str, Any]]:
    try:
        rows: list[dict[str, Any]] = []
        offset = 0
        while True:
            page = (
                admin.table("recipe_cluster_profiles")
                .select(
                    "cluster_id,centroid,size,dominant_cuisine,dominant_meal_type,"
                    "dominant_protein_type,top_ingredients,top_ingredient_traits,"
                    "top_boolean_traits,categorical_traits,representative_recipes"
                )
                .eq("model_version", _FYP_CLUSTER_MODEL_VERSION)
                .range(offset, offset + 999)
                .execute()
                .data or []
            )
            rows.extend(page)
            if len(page) < 1000:
                break
            offset += 1000
        return {
            int(row["cluster_id"]): {
                "cluster_id": int(row["cluster_id"]),
                "centroid": [float(value) for value in (row.get("centroid") or [])],
                "size": int(row.get("size") or 0),
                "dominant_cuisine": row.get("dominant_cuisine"),
                "dominant_meal_type": row.get("dominant_meal_type"),
                "dominant_protein_type": row.get("dominant_protein_type"),
                "top_ingredients": row.get("top_ingredients") or [],
                "top_ingredient_traits": row.get("top_ingredient_traits") or [],
                "top_boolean_traits": row.get("top_boolean_traits") or [],
                "categorical_traits": row.get("categorical_traits") or {},
                "representative_recipes": row.get("representative_recipes") or [],
            }
            for row in rows
            if row.get("cluster_id") is not None and row.get("centroid")
        }
    except Exception as exc:
        print(f"[startup] WARNING: enriched FYP cluster profiles unavailable: {exc}")
        return _fetch_legacy_recipe_cluster_profiles(admin)


def _fetch_legacy_recipe_cluster_profiles(admin: Any) -> dict[int, dict[str, Any]]:
    try:
        rows: list[dict[str, Any]] = []
        offset = 0
        while True:
            page = (
                admin.table("recipe_cluster_profiles")
                .select(
                    "cluster_id,centroid,size,dominant_cuisine,dominant_meal_type,"
                    "dominant_protein_type,top_ingredients,representative_recipes"
                )
                .eq("model_version", _FYP_CLUSTER_MODEL_VERSION)
                .range(offset, offset + 999)
                .execute()
                .data or []
            )
            rows.extend(page)
            if len(page) < 1000:
                break
            offset += 1000
        return {
            int(row["cluster_id"]): {
                **row,
                "cluster_id": int(row["cluster_id"]),
                "centroid": [float(value) for value in (row.get("centroid") or [])],
                "size": int(row.get("size") or 0),
                "top_ingredient_traits": [],
                "top_boolean_traits": [],
                "categorical_traits": {},
            }
            for row in rows
            if row.get("cluster_id") is not None and row.get("centroid")
        }
    except Exception as exc:
        print(f"[startup] WARNING: failed to load FYP cluster profiles: {exc}")
        return {}


def _reset_recommendation_state(app: FastAPI) -> None:
    app.state.rec_recipes = []
    app.state.rec_weights = {}
    app.state.recipe_clusters = {}
    app.state.recipe_cluster_vectors = {}
    app.state.recipe_cluster_profiles = {}
    app.state.recipe_cluster_centroids = {}
    app.state.recipe_cluster_model_version = None
    app.state.cluster_aware_fyp = False
