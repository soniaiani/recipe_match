from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_EMBEDDING_CACHE: dict[str, np.ndarray] = {}

MEAL_TEXT = {
    "appetizer": "appetizer",
    "breakfast": "breakfast",
    "dessert": "dessert",
    "drink": "drink",
    "lunch_dinner": "lunch or dinner",
    "salad_side": "salad or side dish",
    "snack": "snack",
    "soup": "soup",
    "condiment": "condiment or sauce",
}

PROTEIN_TEXT = {
    "chicken": "chicken",
    "beef_pork": "beef or pork",
    "fish_seafood": "fish or seafood",
    "meatless": "meatless",
    "other_meat": "other meat",
}

CUISINE_TEXT = {
    "italian": "Italian",
    "asian": "Asian",
    "mexican": "Mexican",
    "french": "French",
    "mediterranean": "Mediterranean",
    "indian": "Indian",
    "american": "American",
    "other": "international",
}

FEATURE_TEXT = {
    "is_spicy": ("spicy", "not spicy"),
    "is_sweet": ("sweet", "not sweet"),
    "is_quick": ("quick and easy", "not quick"),
    "needs_oven": ("oven-baked", "not oven-baked"),
    "needs_stovetop": ("cooked on the stovetop", "not cooked on the stovetop"),
    "is_no_cook": ("no-cook", "cooked recipe"),
    "has_pasta": ("with pasta", "without pasta"),
    "has_rice": ("with rice", "without rice"),
    "has_potato": ("with potato", "without potato"),
    "has_tomato_base": ("with a tomato base", "without a tomato base"),
    "has_cream_base": ("creamy", "not creamy"),
    "has_cheese": ("with cheese", "without cheese"),
    "has_broth_base": ("brothy", "not brothy"),
    "has_mushroom": ("with mushrooms", "without mushrooms"),
    "has_leafy_greens": ("with leafy greens", "without leafy greens"),
    "has_beans_legumes": ("with beans or legumes", "without beans or legumes"),
    "has_fruit": ("with fruit", "without fruit"),
    "has_nuts": ("with nuts", "without nuts"),
    "has_chocolate": ("with chocolate", "without chocolate"),
    "has_asian_sauce": ("with Asian sauce", "without Asian sauce"),
}


@lru_cache(maxsize=1)
def _get_model():
    backend_dir = Path(__file__).resolve().parents[2]
    cache_root = backend_dir / ".cache"
    cache_root.mkdir(exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_root / "huggingface"))

    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME, cache_folder=str(cache_root / "sentence-transformers"))


def recipe_to_text(recipe: dict[str, Any]) -> str:
    ingredients = recipe.get("ingredients_clean") or recipe.get("ingredients") or ""
    if isinstance(ingredients, str):
        try:
            parsed = json.loads(ingredients)
            if isinstance(parsed, list):
                ingredients = parsed
        except Exception:
            pass
    if isinstance(ingredients, list):
        ingredients = ", ".join(str(item) for item in ingredients if item)

    meal = MEAL_TEXT.get(str(recipe.get("meal_type")), recipe.get("meal_type"))
    cuisine = CUISINE_TEXT.get(str(recipe.get("cuisine")), recipe.get("cuisine"))
    protein = PROTEIN_TEXT.get(str(recipe.get("protein_type")), recipe.get("protein_type"))

    summary_parts = []
    if cuisine:
        summary_parts.append(str(cuisine))
    if protein:
        summary_parts.append(str(protein))
    if meal:
        summary_parts.append(str(meal))

    features = []
    for feature, (positive_text, negative_text) in FEATURE_TEXT.items():
        value = recipe.get(feature)
        if value is True:
            features.append(positive_text)
        elif value is False and feature in {"is_spicy", "is_sweet", "is_no_cook"}:
            features.append(negative_text)

    parts = [
        recipe.get("name"),
        f"A {' '.join(summary_parts)} recipe." if summary_parts else None,
        recipe.get("description"),
        f"Features: {', '.join(features)}." if features else None,
        f"ingredients: {ingredients}" if ingredients else None,
    ]
    return ". ".join(str(part) for part in parts if part)


def encode_recipes(recipes: list[dict[str, Any]]) -> np.ndarray:
    texts = [recipe_to_text(recipe) for recipe in recipes]
    return encode_texts(texts)


def encode_texts(texts: list[str]) -> np.ndarray:
    missing_texts = [text for text in texts if text not in _EMBEDDING_CACHE]
    if missing_texts:
        model = _get_model()
        missing_embeddings = model.encode(
            missing_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        for text, embedding in zip(missing_texts, missing_embeddings):
            _EMBEDDING_CACHE[text] = embedding

    return np.array([_EMBEDDING_CACHE[text] for text in texts])


def encode_text(text: str) -> np.ndarray:
    return encode_texts([text])[0]


def clear_embedding_cache() -> None:
    _EMBEDDING_CACHE.clear()
