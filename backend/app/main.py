from __future__ import annotations
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from app.config import get_settings
from app.routers import auth, collections, explorer, foryou, recipes, recommendations, saved

settings = get_settings()

limiter = Limiter(key_func=get_remote_address, default_limits=["200/minute"])

# Fields needed by the Bayesian engine at inference time
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

app = FastAPI(
    title="Recipe Match API",
    version="1.0.0",
    docs_url="/docs" if not settings.is_production else None,
    redoc_url="/redoc" if not settings.is_production else None,
)


@app.on_event("startup")
async def _load_rec_engine() -> None:
    """Load all recipes and pre-compute MI weights once at startup."""
    try:
        from app.database import get_supabase_admin
        from app.recommender.engine import compute_feature_mi

        admin = get_supabase_admin()
        
        recipes = []
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
                break
            offset += 1000

        app.state.rec_recipes = recipes
        app.state.rec_weights = compute_feature_mi(app.state.rec_recipes)
        print(f"[startup] Loaded {len(app.state.rec_recipes)} recipes for Bayesian engine.")

        try:
            from app.recommender.embeddings import encode_text

            encode_text("warmup recipe preferences")
            print("[startup] Warmed semantic embedding model.")
        except Exception as exc:
            print(f"[startup] WARNING: failed to warm embedding model: {exc}")

        explorer.warm_explorer_cache()
        explorer.warm_ingredient_idf()
        explorer.warm_explorer_ltr_model()
        print("[startup] Warmed Ingredient Explorer cache.")

        try:
            cluster_rows = []
            cluster_offset = 0
            while True:
                page = (
                    admin.table("recipe_clusters")
                    .select("recipe_id,cluster_id")
                    .eq("model_version", _FYP_CLUSTER_MODEL_VERSION)
                    .range(cluster_offset, cluster_offset + 999)
                    .execute()
                    .data or []
                )
                cluster_rows.extend(page)
                if len(page) < 1000:
                    break
                cluster_offset += 1000

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
    except Exception as exc:
        app.state.rec_recipes = []
        app.state.rec_weights = {}
        app.state.recipe_clusters = {}
        app.state.recipe_cluster_model_version = None
        app.state.cluster_aware_fyp = False
        print(f"[startup] WARNING: failed to load Bayesian engine data: {exc}")

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS — tighten allowed_origins in production
_ALLOWED_ORIGINS = (
    ["*"] if not settings.is_production
    else ["https://your-production-domain.com"]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["Authorization", "Content-Type"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    # Never leak stack traces in production
    detail = str(exc) if not settings.is_production else "Internal server error"
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"data": None, "error": detail},
    )


app.include_router(auth.router)
app.include_router(recipes.router)
app.include_router(collections.router)
app.include_router(saved.router)
app.include_router(foryou.router)
app.include_router(recommendations.router)
app.include_router(explorer.router)


@app.get("/health")
async def health_check():
    return {"status": "ok"}
