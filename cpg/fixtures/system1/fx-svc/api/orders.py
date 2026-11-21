"""Order endpoints and the helpers around them."""
import requests
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from core.flow import apply_discount
from db.repo import Repo, get_repo

LEGACY_FEE_URL = "http://legacy:8010/legacy/fee"
TOOLS_RUN_URL = "http://tools:8020/tools/run"


class OrderIn(BaseModel):
    id: int
    item_id: int
    qty: int
    rate: float = 0.0
    total: float = 0.0


def accept_quote(quote, discount):
    """Accept a quote; the discount policy is supplied by the caller."""
    return discount.apply_discount(quote.item_id, quote.qty, quote.rate)


def build_router():
    router = APIRouter()

    @router.post("/v1/orders")
    def create_order(order: OrderIn, repo: Repo = Depends(get_repo)):
        total = apply_discount(order.item_id, order.qty, order.rate)
        order.total = total
        repo.save(order)
        return {"total": total}

    @router.get("/v1/orders/{order_id}")
    def read_order(order_id: int):
        return {"id": order_id}

    return router


def estimate(pricer, item, qty):
    return pricer.price(item, qty)


def peek(cache, key):
    return cache.entries[key]


def legacy_fee():
    return requests.get(LEGACY_FEE_URL, timeout=5).json()


def run_tool():
    return requests.get(TOOLS_RUN_URL, timeout=5).json()
