# backend/scripts/upload_recipe_clusters.py
r"""Upload offline recipe cluster artifacts to Supabase.

Run after applying backend/migrations/006_recipe_clusters.sql:
    cd D:\recipe_match
    $env:PYTHONIOENCODING='utf-8'
    backend\venv\Scripts\python.exe backend\scripts\upload_recipe_clusters.py
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(dotenv_path=ROOT / ".env")

from app.database import get_supabase_admin  # noqa: E402


MODEL_VERSION = "pantry_downweighted_kmeans_hybrid_bool125_pca25_k260_alpha008_seed42_v1"
DEFAULT_ARTIFACT_DIR = ROOT / "testari" / "production_cluster_artifacts"
DEFAULT_MODEL_JSON = DEFAULT_ARTIFACT_DIR / f"cluster_model_{MODEL_VERSION}.json"
DEFAULT_CLUSTERS_CSV = DEFAULT_ARTIFACT_DIR / f"recipe_clusters_{MODEL_VERSION}.csv"
DEFAULT_VECTORS_JSON = DEFAULT_ARTIFACT_DIR / f"recipe_cluster_vectors_{MODEL_VERSION}.json"
DEFAULT_PROFILES_JSON = DEFAULT_ARTIFACT_DIR / f"recipe_cluster_profiles_{MODEL_VERSION}.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload recipe cluster artifacts to Supabase.")
    parser.add_argument("--artifact-dir", type=Path, default=None)
    parser.add_argument("--model-json", type=Path, default=DEFAULT_MODEL_JSON)
    parser.add_argument("--clusters-csv", type=Path, default=DEFAULT_CLUSTERS_CSV)
    parser.add_argument("--vectors-json", type=Path, default=DEFAULT_VECTORS_JSON)
    parser.add_argument("--profiles-json", type=Path, default=DEFAULT_PROFILES_JSON)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument(
        "--description",
        type=str,
        default=(
            "FYP diversification model: pantry-downweighted ingredients, hybrid representation, "
            "PCA(25), KMeans(k=260, n_init=20), soft penalty alpha=0.08."
        ),
    )
    args = parser.parse_args()
    using_default_paths = (
        args.model_json == DEFAULT_MODEL_JSON
        and args.clusters_csv == DEFAULT_CLUSTERS_CSV
        and args.vectors_json == DEFAULT_VECTORS_JSON
        and args.profiles_json == DEFAULT_PROFILES_JSON
    )
    artifact_dir = args.artifact_dir or (_latest_artifact_dir(DEFAULT_ARTIFACT_DIR) if using_default_paths else None)
    if artifact_dir:
        args.model_json = artifact_dir / f"cluster_model_{MODEL_VERSION}.json"
        args.clusters_csv = artifact_dir / f"recipe_clusters_{MODEL_VERSION}.csv"
        args.vectors_json = artifact_dir / f"recipe_cluster_vectors_{MODEL_VERSION}.json"
        args.profiles_json = artifact_dir / f"recipe_cluster_profiles_{MODEL_VERSION}.json"
    return args


def _latest_artifact_dir(root: Path) -> Path | None:
    if not root.exists() or root.is_file():
        return None
    candidates = [
        path
        for path in root.iterdir()
        if path.is_dir()
        and (path / f"cluster_model_{MODEL_VERSION}.json").exists()
        and (path / f"recipe_clusters_{MODEL_VERSION}.csv").exists()
        and (path / f"recipe_cluster_vectors_{MODEL_VERSION}.json").exists()
        and (path / f"recipe_cluster_profiles_{MODEL_VERSION}.json").exists()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_model(path: Path, description: str) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    return {
        "model_version": payload["model_version"],
        "params": payload.get("params") or {},
        "metrics": payload.get("metrics") or {},
        "description": description,
    }


def load_cluster_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append({
                "recipe_id": int(row["recipe_id"]),
                "model_version": row["model_version"],
                "cluster_id": int(row["cluster_id"]),
            })
    return rows


def load_json_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise RuntimeError(f"Expected list in {path}")
    return data


def chunks(rows: list[dict[str, Any]], size: int):
    for start in range(0, len(rows), size):
        yield rows[start:start + size]


def main() -> None:
    args = parse_args()
    admin = get_supabase_admin()

    model = load_model(args.model_json, args.description)
    rows = load_cluster_rows(args.clusters_csv)
    vector_rows = load_json_rows(args.vectors_json)
    profile_rows = load_json_rows(args.profiles_json)
    if not rows:
        raise RuntimeError(f"No cluster rows found in {args.clusters_csv}")

    versions = {row["model_version"] for row in rows}
    if versions != {model["model_version"]}:
        raise RuntimeError(
            "Model JSON and cluster CSV versions differ: "
            f"json={model['model_version']} csv={sorted(versions)}"
        )
    for label, artifact_rows in (("vectors", vector_rows), ("profiles", profile_rows)):
        artifact_versions = {row.get("model_version") for row in artifact_rows}
        if artifact_rows and artifact_versions != {model["model_version"]}:
            raise RuntimeError(
                f"Model JSON and {label} versions differ: "
                f"json={model['model_version']} {label}={sorted(artifact_versions)}"
            )

    print(f"Uploading model: {model['model_version']}")
    admin.table("recipe_cluster_models").upsert(
        model,
        on_conflict="model_version",
    ).execute()

    print(f"Uploading {len(rows)} recipe cluster rows...")
    uploaded = 0
    for batch in chunks(rows, args.batch_size):
        admin.table("recipe_clusters").upsert(
            batch,
            on_conflict="recipe_id,model_version",
        ).execute()
        uploaded += len(batch)
        print(f"Uploaded {uploaded}/{len(rows)}")

    if vector_rows:
        print(f"Uploading {len(vector_rows)} recipe vectors...")
        uploaded = 0
        for batch in chunks(vector_rows, args.batch_size):
            admin.table("recipe_cluster_vectors").upsert(
                batch,
                on_conflict="recipe_id,model_version",
            ).execute()
            uploaded += len(batch)
            print(f"Uploaded vectors {uploaded}/{len(vector_rows)}")

    if profile_rows:
        print(f"Uploading {len(profile_rows)} cluster profiles...")
        uploaded = 0
        for batch in chunks(profile_rows, args.batch_size):
            admin.table("recipe_cluster_profiles").upsert(
                batch,
                on_conflict="model_version,cluster_id",
            ).execute()
            uploaded += len(batch)
            print(f"Uploaded profiles {uploaded}/{len(profile_rows)}")

    print("Done.")


if __name__ == "__main__":
    main()
