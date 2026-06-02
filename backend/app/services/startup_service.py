"""Application startup loading for recommendation-related state."""
from __future__ import annotations

from typing import Any

from fastapi import FastAPI

from app.services import explorer_service

_REC_RECIPE_FIELDS = (
    "id,name,image_url,meal_type,protein_type,cuisine,"
    "description,ingredients,ingredients_clean,embedding,"
    "is_vegetarian,is_vegan,is_gluten_free,is_dairy_free,"
    "is_spicy,is_sweet,is_quick,needs_oven,needs_stovetop,is_no_cook,"
    "has_pasta,has_rice,has_potato,has_tomato_base,has_cream_base,"
    "has_cheese,has_broth_base,has_mushroom,has_leafy_greens,"
    "has_beans_legumes,has_fruit,has_nuts,has_chocolate,"
    "has_tortilla,has_spicy_ingredient,has_asian_sauce"
)

_FYP_CLUSTER_MODEL_VERSION = "pantry_downweighted_hybrid_c_pca25_k180_seed42_v1"


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

        _warm_semantic_embedding_model()
        _warm_explorer_resources()
        _load_recipe_clusters(app, admin)
    except Exception as exc:
        _reset_recommendation_state(app)
        print(f"[startup] WARNING: failed to load Bayesian engine data: {exc}")


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
    explorer_service.warm_explorer_cache()
    explorer_service.warm_ingredient_idf()
    explorer_service.warm_explorer_ltr_model()
    print("[startup] Warmed Ingredient Explorer cache.")


def _load_recipe_clusters(app: FastAPI, admin: Any) -> None:
    try:
        cluster_rows = _fetch_recipe_clusters(admin)
        app.state.recipe_clusters = {
            int(row["recipe_id"]): int(row["cluster_id"])
            for row in cluster_rows
            if row.get("recipe_id") is not None and row.get("cluster_id") is not None
        }
        app.state.recipe_cluster_model_version = _FYP_CLUSTER_MODEL_VERSION
        app.state.cluster_aware_fyp = bool(app.state.recipe_clusters)
        print(
            "[startup] Loaded "
            f"{len(app.state.recipe_clusters)} recipe clusters for FYP "
            f"model={_FYP_CLUSTER_MODEL_VERSION}."
        )
    except Exception as exc:
        app.state.recipe_clusters = {}
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


def _reset_recommendation_state(app: FastAPI) -> None:
    app.state.rec_recipes = []
    app.state.rec_weights = {}
    app.state.recipe_clusters = {}
    app.state.recipe_cluster_model_version = None
    app.state.cluster_aware_fyp = False
