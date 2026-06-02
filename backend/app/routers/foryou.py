"""For You recommendation endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from app.middleware.auth import get_current_user
from app.models.common import ApiResponse
from app.models.foryou import ForYouResponse
from app.services.foryou_service import build_for_you_response

router = APIRouter(prefix="/foryou", tags=["foryou"])


@router.get("", response_model=ApiResponse[ForYouResponse])
async def for_you(request: Request, payload: dict = Depends(get_current_user)):
    """Return hybrid personalized recipe recommendations."""
    return await build_for_you_response(request, payload)
