"""Search endpoint."""
from fastapi import APIRouter


def build_router():
    router = APIRouter()

    @router.get("/search")
    def search(q: str = ""):
        return {"query": q, "hits": []}

    return router
