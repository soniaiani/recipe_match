"""Taste profile generation from FYP cluster space."""
from __future__ import annotations

import asyncio
import json
import math
import re
import socket
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import Request

from app.config import get_settings
from app.database import get_supabase_admin
from app.models.common import ApiResponse
from app.models.foryou import TasteCard, TasteCluster, TasteProfileResponse
from app.services.foryou.common import (
    BOOLEAN_FEATURES,
    CATEGORICAL_FEATURES,
    INTERACTION_WEIGHTS,
    days_ago,
    interaction_recency_multiplier,
)
from app.services.foryou.profiles import (
    build_answers_profile,
    build_explorer_intents,
    build_semantic_profiles,
)
from app.services.foryou.ranking import _answers_score, semantic_scores_for_profiles

_MIN_DISTINCT_RECIPES = 3
_MIN_POSITIVE_WEIGHT = 5.0
_TOP_CLUSTER_COUNT = 3
_CACHE_TTL_HOURS = 24
_FALLBACK_CACHE_TTL_MINUTES = 5
_CACHE_NEW_INTERACTIONS = 5
_GEMINI_MAX_RETRIES = 1
_GEMINI_BACKOFF_SECONDS = 1.5
_GEMINI_COOLDOWN_SECONDS = 300
_PROFILE_TEXT_VERSION = "multi_source_agreement_v1"
_CENTROID_ALPHA_BASE = 0.10
_CENTROID_ALPHA_PER_RECIPE = 0.03
_CENTROID_ALPHA_MAX = 0.30
_SOURCE_RECIPE_LIMIT = 40
_BEHAVIOR_TRUST = 1.00
_FIND_TRUST = 0.75
_EXPLORER_TRUST = 0.40
_BEHAVIOR_SUPPORT_SATURATION = 10
_FIND_SUPPORT_SATURATION = 8
_EXPLORER_SUPPORT_SATURATION = 8
_SOURCE_DISAGREEMENT_FLOOR = 0.10

_gemini_semaphore = asyncio.Semaphore(1)
_gemini_cooldown_until: datetime | None = None


_TASTE_TRAIT_LABELS = {
    "is_spicy": "Lively heat",
    "is_sweet": "Sweet-leaning",
    "is_quick": "Quick favorites",
    "is_no_cook": "Fresh, no-cook",
    "needs_oven": "Oven-baked",
    "needs_stovetop": "Stovetop cooking",
    "has_pasta": "Pasta-centered",
    "has_rice": "Rice-based",
    "has_potato": "Potato comfort",
    "has_tomato_base": "Tomato-rich",
    "has_cream_base": "Creamy comfort",
    "has_cheese": "Cheesy comfort",
    "has_broth_base": "Brothy bowls",
    "has_mushroom": "Earthy mushrooms",
    "has_leafy_greens": "Leafy greens",
    "has_beans_legumes": "Beans and legumes",
    "has_fruit": "Fruit-forward",
    "has_nuts": "Nutty textures",
    "has_chocolate": "Chocolate-rich",
    "has_asian_sauce": "Savory Asian sauces",
    "has_spicy_ingredient": "Naturally spicy",
    "has_tortilla": "Tortilla-based",
}


async def build_taste_profile_response(
    request: Request,
    payload: dict[str, Any],
) -> ApiResponse[TasteProfileResponse]:
    user_id = payload.get("sub")
    if not user_id:
        return ApiResponse(data=_unavailable_response("none"))

    vectors = getattr(request.app.state, "recipe_cluster_vectors", {}) or {}
    centroids = getattr(request.app.state, "recipe_cluster_centroids", {}) or {}
    profiles = getattr(request.app.state, "recipe_cluster_profiles", {}) or {}
    recipes = getattr(request.app.state, "rec_recipes", []) or []
    model_version = getattr(request.app.state, "recipe_cluster_model_version", None)

    if not vectors or not centroids or not profiles or not recipes or not model_version:
        return ApiResponse(data=_unavailable_response("none"))

    admin = get_supabase_admin()
    events = _fetch_profile_events(user_id, admin)
    answers_profile = build_answers_profile(user_id, admin)
    meta = payload.get("user_metadata") or {}
    explorer_intents = build_explorer_intents(
        user_id,
        admin,
        meta.get("excluded_ingredients", []),
    )
    source_profiles = _build_declared_source_profiles(
        admin,
        recipes,
        vectors,
        answers_profile,
        explorer_intents,
        _fetch_explorer_session_count(user_id, admin),
    )
    cache_row = _fetch_cached_profile(user_id, model_version, admin)
    numeric = _numeric_profile(
        events,
        vectors,
        centroids,
        profiles,
        previous_behavior_centroid=(
            cache_row.get("behavior_centroid_vector") or cache_row.get("centroid_vector")
            if cache_row else None
        ),
        behavior_centroid_updated_at=(
            cache_row.get("behavior_centroid_updated_at") or cache_row.get("centroid_updated_at")
            if cache_row else None
        ),
        find_centroid=source_profiles["find_centroid"],
        find_support=source_profiles["find_support"],
        explorer_centroid=source_profiles["explorer_centroid"],
        explorer_support=source_profiles["explorer_support"],
    )
    if numeric is None:
        return ApiResponse(data=TasteProfileResponse(status="insufficient_data"))

    if _cache_is_valid(cache_row, numeric):
        return ApiResponse(data=_response_from_cache(cache_row))

    generated = await _generate_text_profile(numeric["top_clusters"])
    source = generated["source"]
    taste_cards = _enrich_taste_cards(generated["taste_cards"], numeric["top_clusters"])
    response = TasteProfileResponse(
        status="ready",
        compact_summary=generated["compact_summary"],
        description=generated["description"],
        taste_cards=taste_cards,
        top_clusters=numeric["top_clusters"],
        generated_at=datetime.now(timezone.utc).isoformat(),
        source=source,
    )
    _store_cached_profile(user_id, model_version, numeric, response, admin)
    return ApiResponse(data=response)


def _fetch_profile_events(user_id: str, admin: Any) -> list[dict[str, Any]]:
    interaction_rows = (
        admin.table("recipe_interactions")
        .select("recipe_id,interaction_type,weight,created_at")
        .eq("user_id", user_id)
        .in_("interaction_type", list(INTERACTION_WEIGHTS.keys()))
        .order("created_at", desc=True)
        .execute()
        .data
        or []
    )
    saved_rows = (
        admin.table("saved_recipes")
        .select("recipe_id,saved_at")
        .eq("user_id", user_id)
        .order("saved_at", desc=True)
        .execute()
        .data
        or []
    )

    save_ids = {
        int(row["recipe_id"])
        for row in interaction_rows
        if row.get("recipe_id") is not None and row.get("interaction_type") == "save"
    }
    events: list[dict[str, Any]] = []
    for rank, row in enumerate(interaction_rows):
        recipe_id = row.get("recipe_id")
        interaction_type = row.get("interaction_type")
        if recipe_id is None or interaction_type not in INTERACTION_WEIGHTS:
            continue
        weight = float(row.get("weight") or INTERACTION_WEIGHTS[interaction_type])
        if weight <= 0:
            continue
        days = days_ago(row.get("created_at"))
        events.append({
            "recipe_id": int(recipe_id),
            "weight": weight * math.exp(-0.02 * days) * interaction_recency_multiplier(days, rank),
            "created_at": row.get("created_at"),
        })

    for rank, row in enumerate(saved_rows):
        recipe_id = row.get("recipe_id")
        if recipe_id is None or int(recipe_id) in save_ids:
            continue
        days = days_ago(row.get("saved_at"))
        events.append({
            "recipe_id": int(recipe_id),
            "weight": INTERACTION_WEIGHTS["save"] * math.exp(-0.02 * days) * interaction_recency_multiplier(days, rank),
            "created_at": row.get("saved_at"),
        })

    events.sort(key=lambda item: _parse_dt(item.get("created_at")), reverse=True)
    return events


def _fetch_explorer_session_count(user_id: str, admin: Any) -> int:
    try:
        rows = (
            admin.table("explorer_sessions")
            .select("user_id")
            .eq("user_id", user_id)
            .limit(50)
            .execute()
            .data
            or []
        )
        return len(rows)
    except Exception as exc:
        print(f"[taste-profile] failed to count Explorer sessions: {exc}")
        return 0


def _build_declared_source_profiles(
    admin: Any,
    recipes: list[dict[str, Any]],
    recipe_vectors: dict[int, list[float]],
    answers_profile: dict[str, Any],
    explorer_intents: list[list[str]],
    explorer_session_count: int,
) -> dict[str, Any]:
    find_centroid = _find_source_centroid(recipes, recipe_vectors, answers_profile)
    explorer_centroid = _explorer_source_centroid(
        admin,
        recipe_vectors,
        explorer_intents,
    )
    return {
        "find_centroid": find_centroid,
        "find_support": int(answers_profile.get("n_sessions") or 0) if find_centroid else 0,
        "explorer_centroid": explorer_centroid,
        "explorer_support": explorer_session_count if explorer_centroid else 0,
    }


def _find_source_centroid(
    recipes: list[dict[str, Any]],
    recipe_vectors: dict[int, list[float]],
    answers_profile: dict[str, Any],
) -> list[float] | None:
    if int(answers_profile.get("n_sessions") or 0) <= 0:
        return None
    scored: list[tuple[list[float], float, dict[str, Any]]] = []
    for recipe in recipes:
        recipe_id = recipe.get("id")
        if recipe_id is None:
            continue
        vector = recipe_vectors.get(int(recipe_id))
        if not vector:
            continue
        score = float(_answers_score(recipe, answers_profile))
        if score > 0:
            scored.append((vector, score, {"recipe_id": int(recipe_id)}))
    if not scored:
        return None
    scored.sort(key=lambda item: item[1], reverse=True)
    return _weighted_centroid(scored[:_SOURCE_RECIPE_LIMIT], len(scored[0][0]))


def _explorer_source_centroid(
    admin: Any,
    recipe_vectors: dict[int, list[float]],
    explorer_intents: list[list[str]],
) -> list[float] | None:
    if not explorer_intents:
        return None
    empty_interaction_profile = {
        "ingredients": [],
        "categorical": {feature: {} for feature in CATEGORICAL_FEATURES},
        "booleans": {feature: 0.0 for feature in BOOLEAN_FEATURES},
    }
    empty_answers_profile = {
        "categorical": {feature: [] for feature in CATEGORICAL_FEATURES},
        "booleans": {},
    }
    semantic_profiles = build_semantic_profiles(
        empty_interaction_profile,
        empty_answers_profile,
        explorer_intents,
    )
    semantic_scores = semantic_scores_for_profiles(admin, semantic_profiles)
    scored: list[tuple[list[float], float, dict[str, Any]]] = []
    for recipe_id, score in semantic_scores.items():
        vector = recipe_vectors.get(int(recipe_id))
        adjusted_score = max(float(score) - 0.5, 0.0)
        if vector and adjusted_score > 0:
            scored.append((vector, adjusted_score, {"recipe_id": int(recipe_id)}))
    if not scored:
        return None
    scored.sort(key=lambda item: item[1], reverse=True)
    return _weighted_centroid(scored[:_SOURCE_RECIPE_LIMIT], len(scored[0][0]))


def _numeric_profile(
    events: list[dict[str, Any]],
    recipe_vectors: dict[int, list[float]],
    cluster_centroids: dict[int, list[float]],
    cluster_profiles: dict[int, dict[str, Any]],
    previous_behavior_centroid: list[float] | None = None,
    behavior_centroid_updated_at: Any = None,
    find_centroid: list[float] | None = None,
    find_support: int = 0,
    explorer_centroid: list[float] | None = None,
    explorer_support: int = 0,
) -> dict[str, Any] | None:
    weighted_vectors: list[tuple[list[float], float, dict[str, Any]]] = []
    distinct_recipe_ids: set[int] = set()
    positive_weight = 0.0
    latest_interaction_at: str | None = None

    for event in events:
        recipe_id = int(event["recipe_id"])
        vector = recipe_vectors.get(recipe_id)
        if not vector:
            continue
        weight = float(event["weight"])
        if weight <= 0:
            continue
        weighted_vectors.append((vector, weight, event))
        distinct_recipe_ids.add(recipe_id)
        positive_weight += weight
        latest_interaction_at = latest_interaction_at or event.get("created_at")

    behavior_eligible = (
        len(distinct_recipe_ids) >= _MIN_DISTINCT_RECIPES
        and positive_weight >= _MIN_POSITIVE_WEIGHT
    )
    source_dims = (
        len(weighted_vectors[0][0]) if weighted_vectors
        else len(find_centroid or explorer_centroid or [])
    )
    if source_dims <= 0:
        return None

    behavior_centroid: list[float] | None = None
    valid_previous = (
        isinstance(previous_behavior_centroid, list)
        and len(previous_behavior_centroid) == source_dims
        and bool(behavior_centroid_updated_at)
    )
    new_vectors: list[tuple[list[float], float, dict[str, Any]]] = []
    if behavior_eligible and valid_previous:
        watermark = _parse_dt(behavior_centroid_updated_at)
        new_vectors = [
            item
            for item in weighted_vectors
            if item[2].get("created_at") and _parse_dt(item[2]["created_at"]) > watermark
        ]

    if behavior_eligible and valid_previous and new_vectors:
        delta_centroid = _weighted_centroid(new_vectors, source_dims)
        new_distinct_recipes = len({int(item[2]["recipe_id"]) for item in new_vectors})
        alpha = min(
            _CENTROID_ALPHA_MAX,
            _CENTROID_ALPHA_BASE + _CENTROID_ALPHA_PER_RECIPE * new_distinct_recipes,
        )
        behavior_centroid = _l2_normalize([
            (1.0 - alpha) * float(previous_behavior_centroid[idx]) + alpha * delta_centroid[idx]
            for idx in range(source_dims)
        ])
    elif behavior_eligible and valid_previous:
        behavior_centroid = _l2_normalize([float(value) for value in previous_behavior_centroid])
    elif behavior_eligible:
        behavior_centroid = _weighted_centroid(weighted_vectors, source_dims)

    source_weights = _source_fusion_weights(
        behavior_centroid,
        len(distinct_recipe_ids) if behavior_eligible else 0,
        find_centroid,
        find_support,
        explorer_centroid,
        explorer_support,
    )
    centroid = _fuse_source_centroids(
        behavior_centroid,
        find_centroid,
        explorer_centroid,
        source_weights,
    )
    if centroid is None:
        return None

    scored = [
        (cluster_id, _dot(centroid, _l2_normalize(vector)))
        for cluster_id, vector in cluster_centroids.items()
        if vector
    ]
    scored.sort(key=lambda item: item[1], reverse=True)
    top = scored[:_TOP_CLUSTER_COUNT]
    positive_sim_sum = sum(max(similarity, 0.0) for _, similarity in top)

    top_clusters: list[TasteCluster] = []
    for cluster_id, similarity in top:
        profile = cluster_profiles.get(cluster_id, {})
        weight = (
            max(similarity, 0.0) / positive_sim_sum
            if positive_sim_sum > 0
            else 1.0 / max(len(top), 1)
        )
        top_clusters.append(TasteCluster(
            cluster_id=cluster_id,
            weight=round(weight, 4),
            similarity=round(float(similarity), 4),
            dominant_cuisine=profile.get("dominant_cuisine"),
            dominant_meal_type=profile.get("dominant_meal_type"),
            dominant_protein_type=profile.get("dominant_protein_type"),
            top_ingredients=_string_list(profile.get("top_ingredients"))[:8],
            top_ingredient_traits=profile.get("top_ingredient_traits") or [],
            top_boolean_traits=profile.get("top_boolean_traits") or [],
            categorical_traits=profile.get("categorical_traits") or {},
            representative_recipes=_representative_recipe_names(profile.get("representative_recipes"))[:5],
        ))

    signature = _profile_signature(top_clusters)
    return {
        "top_clusters": top_clusters,
        "profile_signature": signature,
        "interaction_count": len(weighted_vectors),
        "positive_weight": round(positive_weight, 4),
        "latest_interaction_at": latest_interaction_at,
        "centroid_vector": centroid,
        "centroid_updated_at": latest_interaction_at,
        "behavior_centroid_vector": behavior_centroid,
        "behavior_centroid_updated_at": latest_interaction_at if behavior_centroid else None,
        "new_interaction_count": len(new_vectors),
        "source_weights": source_weights,
        "source_support": {
            "behavior": len(distinct_recipe_ids) if behavior_eligible else 0,
            "find": find_support if find_centroid else 0,
            "explorer": explorer_support if explorer_centroid else 0,
        },
    }


def _source_fusion_weights(
    behavior_centroid: list[float] | None,
    behavior_support: int,
    find_centroid: list[float] | None,
    find_support: int,
    explorer_centroid: list[float] | None,
    explorer_support: int,
) -> dict[str, float]:
    raw = {
        "behavior": (
            _BEHAVIOR_TRUST * min(behavior_support / _BEHAVIOR_SUPPORT_SATURATION, 1.0)
            if behavior_centroid else 0.0
        ),
        "find": (
            _FIND_TRUST * min(find_support / _FIND_SUPPORT_SATURATION, 1.0)
            if find_centroid else 0.0
        ),
        "explorer": (
            _EXPLORER_TRUST * min(explorer_support / _EXPLORER_SUPPORT_SATURATION, 1.0)
            if explorer_centroid else 0.0
        ),
    }
    if behavior_centroid:
        for key, source in (("find", find_centroid), ("explorer", explorer_centroid)):
            if source:
                agreement = max(_dot(behavior_centroid, _l2_normalize(source)), 0.0)
                raw[key] *= max(_SOURCE_DISAGREEMENT_FLOOR, agreement**2)
    total = sum(raw.values())
    if total <= 0:
        return {key: 0.0 for key in raw}
    return {key: round(value / total, 6) for key, value in raw.items()}


def _fuse_source_centroids(
    behavior_centroid: list[float] | None,
    find_centroid: list[float] | None,
    explorer_centroid: list[float] | None,
    source_weights: dict[str, float],
) -> list[float] | None:
    sources = {
        "behavior": behavior_centroid,
        "find": find_centroid,
        "explorer": explorer_centroid,
    }
    dims = next((len(source) for source in sources.values() if source), 0)
    if dims <= 0:
        return None
    fused = [0.0] * dims
    for key, source in sources.items():
        weight = float(source_weights.get(key) or 0.0)
        if not source or len(source) != dims or weight <= 0:
            continue
        for idx, value in enumerate(source):
            fused[idx] += weight * float(value)
    return _l2_normalize(fused) if any(fused) else None


def _weighted_centroid(
    weighted_vectors: list[tuple[list[float], float, dict[str, Any]]],
    dims: int,
) -> list[float]:
    centroid = [0.0] * dims
    total_weight = 0.0
    for vector, weight, _ in weighted_vectors:
        if len(vector) != dims:
            continue
        total_weight += weight
        for idx, value in enumerate(vector):
            centroid[idx] += float(value) * weight
    if total_weight <= 0:
        return centroid
    return _l2_normalize([value / total_weight for value in centroid])


async def _generate_text_profile(top_clusters: list[TasteCluster]) -> dict[str, Any]:
    global _gemini_cooldown_until

    settings = get_settings()
    fallback = _fallback_text(top_clusters)
    if not settings.gemini_api_key:
        fallback["source"] = "fallback"
        return fallback

    now = datetime.now(timezone.utc)
    if _gemini_cooldown_until and now < _gemini_cooldown_until:
        print(
            "[taste-profile] Gemini cooldown active; using fallback until "
            f"{_gemini_cooldown_until.isoformat()}."
        )
        fallback["source"] = "fallback"
        return fallback

    try:
        async with _gemini_semaphore:
            parsed, used_model = await _call_gemini_with_fallbacks(
                settings.gemini_api_key,
                settings.gemini_model,
                settings.gemini_fallback_models,
                settings.gemini_timeout_seconds,
                top_clusters,
            )
        parsed["source"] = "gemini"
        print(f"[taste-profile] Gemini profile generated successfully model={used_model}.")
        return parsed
    except Exception as exc:
        print(f"[taste-profile] Gemini failed; using fallback: {exc}")
        fallback["source"] = "fallback"
        return fallback


async def _call_gemini_with_fallbacks(
    api_key: str,
    primary_model: str,
    fallback_models: str,
    timeout_seconds: float,
    top_clusters: list[TasteCluster],
) -> tuple[dict[str, Any], str]:
    global _gemini_cooldown_until

    models = _gemini_model_chain(primary_model, fallback_models)
    last_error: Exception | None = None
    for model in models:
        try:
            parsed = await _call_gemini_model_with_retries(
                api_key,
                model,
                timeout_seconds,
                top_clusters,
            )
            return parsed, model
        except _GeminiTransientError as exc:
            last_error = exc
            print(f"[taste-profile] Gemini model unavailable model={model} error={exc}")

    _gemini_cooldown_until = datetime.now(timezone.utc) + timedelta(
        seconds=_GEMINI_COOLDOWN_SECONDS
    )
    if last_error:
        raise last_error
    raise RuntimeError("No Gemini models configured")


def _gemini_model_chain(primary_model: str, fallback_models: str) -> list[str]:
    models = [
        primary_model.strip(),
        *(model.strip() for model in fallback_models.split(",")),
    ]
    return list(dict.fromkeys(model for model in models if model))


async def _call_gemini_model_with_retries(
    api_key: str,
    model: str,
    timeout_seconds: float,
    top_clusters: list[TasteCluster],
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(_GEMINI_MAX_RETRIES + 1):
        try:
            payload = await asyncio.to_thread(
                _call_gemini,
                api_key,
                model,
                timeout_seconds,
                top_clusters,
            )
            try:
                return _parse_gemini_profile(payload)
            except (ValueError, KeyError, IndexError, TypeError) as exc:
                raise _GeminiTransientError(
                    f"Gemini returned invalid JSON: {exc}"
                ) from exc
        except _GeminiTransientError as exc:
            last_error = exc
            if attempt >= _GEMINI_MAX_RETRIES:
                raise
            delay = _GEMINI_BACKOFF_SECONDS * (attempt + 1)
            print(
                "[taste-profile] Gemini transient failure; retrying "
                f"model={model} "
                f"attempt={attempt + 2}/{_GEMINI_MAX_RETRIES + 1} "
                f"in={delay:.1f}s error={exc}"
            )
            await asyncio.sleep(delay)
        except Exception:
            raise

    if last_error:
        raise last_error
    raise RuntimeError("Gemini call failed without an error")


class _GeminiTransientError(RuntimeError):
    pass


def _call_gemini(
    api_key: str,
    model: str,
    timeout_seconds: float,
    top_clusters: list[TasteCluster],
) -> dict[str, Any]:
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )
    prompt = _gemini_prompt(top_clusters)
    body = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
            "responseMimeType": "application/json",
            "maxOutputTokens": 4096,
            "thinkingConfig": {"thinkingBudget": 0},
            "responseSchema": {
                "type": "OBJECT",
                "required": ["compact_summary", "description", "taste_cards"],
                "properties": {
                    "compact_summary": {"type": "STRING"},
                    "description": {"type": "STRING"},
                    "taste_cards": {
                        "type": "ARRAY",
                        "minItems": 3,
                        "maxItems": 3,
                        "items": {
                            "type": "OBJECT",
                            "required": ["title", "text"],
                            "properties": {
                                "title": {"type": "STRING"},
                                "text": {"type": "STRING"},
                            },
                        },
                    },
                },
            },
        },
    }).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=max(float(timeout_seconds), 1.0)) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        if exc.code in (429, 503):
            raise _GeminiTransientError(f"Gemini HTTP {exc.code}: {detail}") from exc
        raise RuntimeError(f"Gemini HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise _GeminiTransientError(f"Gemini network error: {exc.reason}") from exc
    except (TimeoutError, socket.timeout) as exc:
        raise _GeminiTransientError(
            f"Gemini timed out after {float(timeout_seconds):.1f}s"
        ) from exc


def _gemini_prompt(top_clusters: list[TasteCluster]) -> str:
    cluster_payload = [
        {
            "dominant_cuisine": c.dominant_cuisine,
            "dominant_meal_type": c.dominant_meal_type,
            "dominant_protein_type": c.dominant_protein_type,
            "top_ingredients": c.top_ingredients[:6],
            "distinctive_ingredients": [
                trait.model_dump()
                for trait in c.top_ingredient_traits[:8]
            ],
            "distinctive_culinary_traits": [
                trait.model_dump()
                for trait in c.top_boolean_traits[:5]
            ],
            "categorical_context": {
                key: trait.model_dump()
                for key, trait in c.categorical_traits.items()
            },
            "representative_recipes": c.representative_recipes[:3],
        }
        for c in top_clusters
    ]
    return (
        "You write a user's food taste profile in warm, personal language — like a "
        "knowledgeable friend describing their palate. You receive 3 flavor clusters "
        "that describe their recipe history.\n\n"
        "Rules:\n"
        "- Never mention algorithms, embeddings, cluster IDs, percentages, or technical terms\n"
        "- For each cluster write a creative 2-4 word title (evocative, like a magazine "
        "section: 'Fresh & Bright', 'Comfort Bowls', 'Weekend Spice') — never generic labels\n"
        "- Each card's text should feel personal and specific, naming 1-2 concrete "
        "ingredients or dishes from that cluster\n"
        "- Treat distinctive ingredients and culinary traits as strong evidence\n"
        "- If a trait has is_globally_common=true, use it only as supporting context, "
        "not as the central differentiator\n"
        "- Use categorical context carefully: if is_distinctive is false, mention it "
        "only as supporting context, not as a defining preference\n"
        "- compact_summary: one punchy sentence, under 95 characters\n"
        "- description: 2-3 natural sentences about the overall palate, under 380 characters\n"
        "- taste_cards: exactly 3 objects {title, text}, one per cluster in the same order, "
        "title under 30 characters, text under 110 characters\n\n"
        "Return ONLY valid JSON with keys: compact_summary, description, taste_cards.\n\n"
        f"Clusters (in order): {json.dumps(cluster_payload, ensure_ascii=False)}"
    )


def _parse_gemini_profile(payload: dict[str, Any]) -> dict[str, Any]:
    candidate = payload.get("candidates", [{}])[0]
    finish_reason = candidate.get("finishReason")
    if finish_reason and finish_reason not in {"STOP"}:
        raise ValueError(f"Gemini response did not finish cleanly: {finish_reason}")
    text = "".join(
        str(part.get("text") or "")
        for part in (candidate.get("content", {}).get("parts") or [])
    )
    data = json.loads(_extract_json(text))
    compact_summary = str(data.get("compact_summary") or "").strip()
    description = str(data.get("description") or "").strip()
    cards = data.get("taste_cards") or []
    if not compact_summary or not description or not isinstance(cards, list):
        raise ValueError("Gemini response missing required fields")
    taste_cards = []
    for card in cards[:3]:
        title = str((card or {}).get("title") or "").strip()
        text_value = str((card or {}).get("text") or "").strip()
        if title and text_value:
            taste_cards.append({"title": title[:60], "text": text_value[:160]})
    if len(taste_cards) < 2:
        raise ValueError("Gemini response has too few taste cards")
    return {
        "compact_summary": compact_summary[:140],
        "description": description[:520],
        "taste_cards": taste_cards[:3],
    }


def _enrich_taste_cards(
    cards: list[dict[str, Any]],
    top_clusters: list[TasteCluster],
) -> list[TasteCard]:
    result = []
    for i, card in enumerate(cards[:3]):
        cluster = top_clusters[i] if i < len(top_clusters) else None
        result.append(TasteCard(
            title=card.get("title", ""),
            text=card.get("text", ""),
            traits=_taste_trait_labels(cluster),
            ingredients=cluster.top_ingredients[:6] if cluster else [],
            recipes=cluster.representative_recipes[:2] if cluster else [],
        ))
    return result


def _taste_trait_labels(cluster: TasteCluster | None) -> list[str]:
    if cluster is None:
        return []
    return [
        _TASTE_TRAIT_LABELS.get(trait.name, _humanize_trait_name(trait.name))
        for trait in cluster.top_boolean_traits
        if not trait.is_globally_common
    ][:3]


def _humanize_trait_name(name: str) -> str:
    words = re.sub(r"^(has|is|needs)_", "", name).replace("_", " ")
    return words.capitalize()


def _fallback_text(top_clusters: list[TasteCluster]) -> dict[str, Any]:
    primary = top_clusters[0] if top_clusters else None
    ingredients = _unique([
        ingredient
        for cluster in top_clusters
        for ingredient in cluster.top_ingredients[:3]
    ])
    cuisines = _unique([
        cluster.dominant_cuisine
        for cluster in top_clusters
        if cluster.dominant_cuisine
    ])
    meals = _unique([
        cluster.dominant_meal_type
        for cluster in top_clusters
        if cluster.dominant_meal_type
    ])
    proteins = _unique([
        cluster.dominant_protein_type
        for cluster in top_clusters
        if cluster.dominant_protein_type
    ])

    ingredient_text = _join_words(ingredients[:4]) or "distinctive ingredients"
    cuisine_text = _join_words(cuisines[:2]) or "comfort-forward"
    meal_text = _join_words(meals[:2]) or "everyday"
    protein_text = _join_words(proteins[:2]) or "flexible proteins"

    compact = f"You lean toward {cuisine_text} {meal_text} recipes."
    description = (
        f"Your recent activity points to {meal_text} recipes built around "
        f"{ingredient_text}. The strongest signals come from {cuisine_text} "
        f"flavors and {protein_text}, with a mix of familiar dishes and practical meals."
    )
    raw_cards = [
        {"title": "Core flavors", "text": f"Frequent signals include {ingredient_text}."},
        {"title": "Cooking style", "text": f"You tend to choose {meal_text} recipes with {cuisine_text} cues."},
        {"title": "Protein pattern", "text": f"Your strongest protein direction is {protein_text}."},
    ]
    taste_cards = _enrich_taste_cards(raw_cards, top_clusters)
    return {
        "compact_summary": compact[:140],
        "description": description[:520],
        "taste_cards": [card.model_dump() for card in taste_cards],
        "source": "fallback",
    }


def _fetch_cached_profile(user_id: str, model_version: str, admin: Any) -> dict[str, Any] | None:
    try:
        rows = (
            admin.table("user_taste_profiles")
            .select("*")
            .eq("user_id", user_id)
            .eq("model_version", model_version)
            .limit(1)
            .execute()
            .data
            or []
        )
        return rows[0] if rows else None
    except Exception as exc:
        print(f"[taste-profile] cache read failed: {exc}")
        return None


def _cache_is_valid(cache_row: dict[str, Any] | None, numeric: dict[str, Any]) -> bool:
    if not cache_row:
        return False
    if cache_row.get("status") != "ready":
        return False
    if cache_row.get("source") == "fallback":
        generated_at = _parse_dt(cache_row.get("generated_at"))
        if generated_at + timedelta(minutes=_FALLBACK_CACHE_TTL_MINUTES) <= datetime.now(timezone.utc):
            return False
    expires_at = _parse_dt(cache_row.get("expires_at"))
    if expires_at <= datetime.now(timezone.utc):
        return False
    if cache_row.get("profile_signature") != numeric["profile_signature"]:
        return False
    previous_count = int(cache_row.get("interaction_count") or 0)
    return int(numeric["interaction_count"]) - previous_count < _CACHE_NEW_INTERACTIONS


def _store_cached_profile(
    user_id: str,
    model_version: str,
    numeric: dict[str, Any],
    response: TasteProfileResponse,
    admin: Any,
) -> None:
    try:
        now = datetime.now(timezone.utc)
        cache_ttl = (
            timedelta(minutes=_FALLBACK_CACHE_TTL_MINUTES)
            if response.source == "fallback"
            else timedelta(hours=_CACHE_TTL_HOURS)
        )
        admin.table("user_taste_profiles").upsert({
            "user_id": user_id,
            "model_version": model_version,
            "status": response.status,
            "profile_signature": numeric["profile_signature"],
            "interaction_count": numeric["interaction_count"],
            "positive_weight": numeric["positive_weight"],
            "centroid_vector": numeric["centroid_vector"],
            "centroid_updated_at": numeric["centroid_updated_at"],
            "behavior_centroid_vector": numeric["behavior_centroid_vector"],
            "behavior_centroid_updated_at": numeric["behavior_centroid_updated_at"],
            "source_weights": numeric["source_weights"],
            "source_support": numeric["source_support"],
            "compact_summary": response.compact_summary,
            "description": response.description,
            "taste_cards": [card.model_dump() for card in response.taste_cards],
            "top_clusters": [cluster.model_dump() for cluster in response.top_clusters],
            "source": response.source,
            "generated_at": now.isoformat(),
            "expires_at": (now + cache_ttl).isoformat(),
            "updated_at": now.isoformat(),
        }, on_conflict="user_id,model_version").execute()
    except Exception as exc:
        print(f"[taste-profile] cache write failed: {exc}")


def _response_from_cache(row: dict[str, Any]) -> TasteProfileResponse:
    return TasteProfileResponse(
        status="ready",
        compact_summary=row.get("compact_summary"),
        description=row.get("description"),
        taste_cards=[TasteCard(**card) for card in (row.get("taste_cards") or [])],
        top_clusters=[TasteCluster(**cluster) for cluster in (row.get("top_clusters") or [])],
        generated_at=str(row.get("generated_at")) if row.get("generated_at") else None,
        source="cache",
    )


def _profile_signature(top_clusters: list[TasteCluster]) -> str:
    parts = [
        f"{cluster.cluster_id}:{round(cluster.weight / 0.05) * 0.05:.2f}"
        for cluster in top_clusters
    ]
    return f"{_PROFILE_TEXT_VERSION}|" + "|".join(parts)


def _unavailable_response(source: str) -> TasteProfileResponse:
    return TasteProfileResponse(status="unavailable", source=source)


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 0:
        return vector
    return [value / norm for value in vector]


def _dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _extract_json(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped, flags=re.IGNORECASE).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    return stripped


def _parse_dt(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str) and value:
        try:
            normalized = value.replace("Z", "+00:00")
            parsed = datetime.fromisoformat(normalized)
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return datetime.fromtimestamp(0, tz=timezone.utc)


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception:
            return [value]
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if isinstance(item, dict):
            item = item.get("name") or item.get("ingredient") or item.get("value")
        text = str(item).strip()
        if text:
            out.append(text)
    return out


def _representative_recipe_names(value: Any) -> list[str]:
    return _string_list(value)


def _unique(values: list[Any]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = str(value or "").strip().replace("_", " ")
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            out.append(text)
    return out


def _join_words(values: list[str]) -> str:
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    return ", ".join(values[:-1]) + f" and {values[-1]}"
