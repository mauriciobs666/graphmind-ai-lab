"""Order endpoints and the helpers around them."""
import requests
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from fx_svc.core.flow import apply_discount
from fx_svc.db.repo import Repo, get_repo

router = APIRouter(prefix="/v1/orders")

LEGACY_FEE_PATH = "/legacy/fee"
TOOLS_RUN_PATH = "/tools/run"


class OrderIn(BaseModel):
    id: int
    item_id: int
    qty: int
    rate: float = 0.0
    total: float = 0.0


def accept_quote(quote, discount):
    """Accept a quote; the discount policy is supplied by the caller."""
    return discount.apply_discount(quote.item_id, quote.qty, quote.rate)


@router.post("")
def create_order(order: OrderIn, repo: Repo = Depends(get_repo)):
    total = apply_discount(order.item_id, order.qty, order.rate)
    order.total = total
    repo.save(order)
    return {"total": total}


@router.get("/{order_id}")
def read_order(order_id: int):
    return {"id": order_id}


def estimate(pricer, item, qty):
    return pricer.price(item, qty)


def peek(cache, key):
    return cache.entries[key]


def legacy_fee():
    return requests.get(LEGACY_FEE_PATH, timeout=5).json()


def run_tool():
    return requests.get(TOOLS_RUN_PATH, timeout=5).json()
