from __future__ import annotations
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from app.config import get_settings
from app.routers import auth, collections, explorer, foryou, recipes, recommendations, saved
from app.services.startup import initialize_recommendation_state

settings = get_settings()

limiter = Limiter(key_func=get_remote_address, default_limits=["200/minute"])

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all recipes and pre-compute MI weights once at startup."""
    initialize_recommendation_state(app)
    yield

app = FastAPI(
    title="Recipe Match API",
    version="1.0.0",
    docs_url="/docs" if not settings.is_production else None,
    redoc_url="/redoc" if not settings.is_production else None,
    lifespan=lifespan,
)


app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS: tighten allowed_origins in production
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
