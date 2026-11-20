"""Item reads, through an injected store."""
import logging

from fastapi import APIRouter, Depends

from state.stores import Store, get_store

log = logging.getLogger(__name__)


def build_router():
    router = APIRouter()

    @router.get("/{item_id}")
    def read_item(item_id: int, store: Store = Depends(get_store)):
        log.info("GET /api/items/1 ok")
        return store.load(item_id)

    return router
