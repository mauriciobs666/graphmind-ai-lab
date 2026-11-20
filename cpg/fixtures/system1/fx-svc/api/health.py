"""Liveness endpoint."""
from fastapi import APIRouter


def build_router():
    router = APIRouter()

    @router.get("/health")
    def health():
        return {"status": "ok"}

    return router
