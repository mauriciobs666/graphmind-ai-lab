"""Static file reads."""
from fastapi import APIRouter


def build_router():
    router = APIRouter()

    @router.get("/files/{name}")
    def read_file(name: str):
        return {"name": name, "bytes": 0}

    return router
