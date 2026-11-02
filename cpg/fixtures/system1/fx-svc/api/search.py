"""Search endpoint."""
from fastapi import APIRouter

router = APIRouter()


@router.get("/search")
def search(q: str = ""):
    return {"query": q, "hits": []}
