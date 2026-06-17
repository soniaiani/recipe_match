# backend/scripts/pregenerate_taste_profiles.py
r"""Pre-generate cached taste profiles for demo users.

Run:
    cd D:\recipe_match
    $env:PYTHONIOENCODING='utf-8'
    backend\venv\Scripts\python.exe backend\scripts\pregenerate_taste_profiles.py --user-id <uuid> --user-id <uuid>
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

from dotenv import load_dotenv
from fastapi import FastAPI

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(dotenv_path=ROOT / ".env")

from app.services.foryou.taste_profile import build_taste_profile_response  # noqa: E402
from app.services.startup import initialize_recommendation_state  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-generate taste profiles for demo users.")
    parser.add_argument("--user-id", action="append", required=True, help="Supabase auth.users UUID")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    app = FastAPI()
    initialize_recommendation_state(app)
    request = SimpleNamespace(app=app)

    for user_id in args.user_id:
        response = await build_taste_profile_response(request, {"sub": user_id})
        data = response.data
        print(
            f"user={user_id} status={data.status if data else 'none'} "
            f"source={data.source if data else 'none'}"
        )


if __name__ == "__main__":
    asyncio.run(main())
