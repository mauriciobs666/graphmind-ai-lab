"""Static file reads."""
from fastapi import APIRouter

router = APIRouter()


@router.get("/files/{name}")
def read_file(name: str):
    return {"name": name, "bytes": 0}
