"""Learning-to-rank support for Ingredient Explorer suggestions."""
from __future__ import annotations

from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import joblib
import pandas as pd

from app.models.explorer import ExplorerSuggestion

_EXPLORER_LTR_MODEL: Any | None = None
_EXPLORER_LTR_FEATURE_COLUMNS: list[str] = []
_BACKEND_DIR = Path(__file__).resolve().parents[3]
_EXPLORER_LTR_MODEL_PATH = (
    _BACKEND_DIR
    / "models"
    / "explorer_ltr_final_no_graph.joblib"
)
_LEGACY_EXPLORER_LTR_MODEL_PATH = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "explorer_ltr_final_no_graph.joblib"
)


def warm_explorer_ltr_model() -> None:
    global _EXPLORER_LTR_MODEL, _EXPLORER_LTR_FEATURE_COLUMNS
    if _EXPLORER_LTR_MODEL is not None:
        return

    model_path = (
        _EXPLORER_LTR_MODEL_PATH
        if _EXPLORER_LTR_MODEL_PATH.exists()
        else _LEGACY_EXPLORER_LTR_MODEL_PATH
    )
    if not model_path.exists():
        print(f"[explorer] LTR model not found at {_EXPLORER_LTR_MODEL_PATH}; using statistical fallback.")
        return

    try:
        artifact = joblib.load(model_path)
        _EXPLORER_LTR_MODEL = artifact["model"]
        _EXPLORER_LTR_FEATURE_COLUMNS = list(artifact["feature_columns"])
        print(
            "[explorer] Loaded LTR model "
            f"{artifact.get('model_name', 'unknown')} "
            f"with features={_EXPLORER_LTR_FEATURE_COLUMNS}."
        )
    except Exception as exc:
        _EXPLORER_LTR_MODEL = None
        _EXPLORER_LTR_FEATURE_COLUMNS = []
        print(f"[explorer] failed to load LTR model: {exc}")


def score_expand_candidates_ltr(
    candidate_counts: dict[str, int],
    selected: list[str],
    n_matching: int,
    ingredient_idf: dict[str, float],
    pantry_soft: set[str],
) -> list[ExplorerSuggestion] | None:
    warm_explorer_ltr_model()
    if _EXPLORER_LTR_MODEL is None or not _EXPLORER_LTR_FEATURE_COLUMNS:
        return None

    raw_features = _build_raw_ltr_features(
        candidate_counts,
        selected,
        n_matching,
        ingredient_idf,
        pantry_soft,
    )
    rows, candidates = _build_ltr_rows(raw_features)
    scores = _predict_ltr_scores(rows)
    if scores is None:
        return None

    return _rank_ltr_suggestions(candidates, scores, raw_features)


def _build_raw_ltr_features(
    candidate_counts: dict[str, int],
    selected: list[str],
    n_matching: int,
    ingredient_idf: dict[str, float],
    pantry_soft: set[str],
) -> dict[str, dict[str, float]]:
    raw_features: dict[str, dict[str, float]] = {}
    for candidate, count in candidate_counts.items():
        frequency_score = count / n_matching
        idf = ingredient_idf.get(candidate, 1.0)
        raw_features[candidate] = {
            "frequency_score": frequency_score,
            "idf": idf,
            "seed_count": float(len(selected)),
            "pantry_flag": float(int(candidate in pantry_soft)),
            "tfidf_score": frequency_score * idf,
        }
    return raw_features


def _build_ltr_rows(
    raw_features: dict[str, dict[str, float]],
) -> tuple[list[dict[str, float]], list[str]]:
    zscores = _feature_zscores(raw_features)
    rows: list[dict[str, float]] = []
    candidates: list[str] = []

    for candidate, features in raw_features.items():
        rows.append({
            **features,
            "freq_zscore": zscores["frequency_score"][candidate],
            "tfidf_zscore": zscores["tfidf_score"][candidate],
        })
        candidates.append(candidate)

    return rows, candidates


def _feature_zscores(
    raw_features: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    return {
        feature: _zscore_by_candidate(raw_features, feature)
        for feature in ("frequency_score", "tfidf_score")
    }


def _zscore_by_candidate(
    raw_features: dict[str, dict[str, float]],
    feature: str,
) -> dict[str, float]:
    values = [features[feature] for features in raw_features.values()]
    average = mean(values) if values else 0.0
    std = pstdev(values) if len(values) > 1 else 0.0
    return {
        candidate: _safe_zscore(features[feature], average, std)
        for candidate, features in raw_features.items()
    }


def _safe_zscore(value: float, average: float, std: float) -> float:
    return (value - average) / std if std > 0 else 0.0


def _predict_ltr_scores(rows: list[dict[str, float]]) -> Any | None:
    try:
        frame = pd.DataFrame(rows)
        return _EXPLORER_LTR_MODEL.predict_proba(
            frame[_EXPLORER_LTR_FEATURE_COLUMNS]
        )[:, 1]
    except Exception as exc:
        print(f"[explorer] LTR inference failed; using statistical fallback: {exc}")
        return None


def _rank_ltr_suggestions(
    candidates: list[str],
    scores: Any,
    raw_features: dict[str, dict[str, float]],
) -> list[ExplorerSuggestion]:
    suggestions = [
        ExplorerSuggestion(
            ingredient=candidate,
            score=round(float(score), 6),
        )
        for candidate, score in zip(candidates, scores)
    ]
    suggestions.sort(
        key=lambda item: (
            -(item.score or 0.0),
            -raw_features[item.ingredient]["tfidf_score"],
            -raw_features[item.ingredient]["frequency_score"],
            item.ingredient,
        )
    )
    return suggestions
