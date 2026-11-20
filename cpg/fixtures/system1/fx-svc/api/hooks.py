"""Webhook receiver."""
from fastapi import APIRouter


def build_router():
    router = APIRouter()

    @router.get("/hooks/{token}")
    def receive_hook(token: str):
        return {"accepted": bool(token)}

    return router
