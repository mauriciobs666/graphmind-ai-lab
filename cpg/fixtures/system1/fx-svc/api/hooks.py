"""Webhook receiver."""
from fastapi import APIRouter

router = APIRouter()


@router.get("/hooks/{token}")
def receive_hook(token: str):
    return {"accepted": bool(token)}
