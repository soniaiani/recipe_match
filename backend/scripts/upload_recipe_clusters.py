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


DEFAULT_ARTIFACT_DIR = ROOT / "testari" / "fyp_diversification_results" / "20260517_201022"
DEFAULT_MODEL_JSON = DEFAULT_ARTIFACT_DIR / "cluster_model_pantry_downweighted_hybrid_c_pca25_k180_seed42_v1.json"
DEFAULT_CLUSTERS_CSV = DEFAULT_ARTIFACT_DIR / "recipe_clusters_pantry_downweighted_hybrid_c_pca25_k180_seed42_v1.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload recipe cluster artifacts to Supabase.")
    parser.add_argument("--model-json", type=Path, default=DEFAULT_MODEL_JSON)
    parser.add_argument("--clusters-csv", type=Path, default=DEFAULT_CLUSTERS_CSV)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument(
        "--description",
        type=str,
        default=(
            "FYP diversification model: pantry-downweighted ingredients, hybrid_c, "
            "PCA(25), MiniBatchKMeans(k=180), soft penalty alpha=0.06."
        ),
    )
    return parser.parse_args()


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


def chunks(rows: list[dict[str, Any]], size: int):
    for start in range(0, len(rows), size):
        yield rows[start:start + size]


def main() -> None:
    args = parse_args()
    admin = get_supabase_admin()

    model = load_model(args.model_json, args.description)
    rows = load_cluster_rows(args.clusters_csv)
    if not rows:
        raise RuntimeError(f"No cluster rows found in {args.clusters_csv}")

    versions = {row["model_version"] for row in rows}
    if versions != {model["model_version"]}:
        raise RuntimeError(
            "Model JSON and cluster CSV versions differ: "
            f"json={model['model_version']} csv={sorted(versions)}"
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

    print("Done.")


if __name__ == "__main__":
    main()
