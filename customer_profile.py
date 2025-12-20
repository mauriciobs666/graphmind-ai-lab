from __future__ import annotations

from typing import Dict, Literal, Optional, TypedDict

from session_manager import ensure_session_id
from cart import cart_has_items

InfoStage = Literal[
    "need_name",
    "awaiting_name",
    "idle",
    "awaiting_address",
    "awaiting_payment",
    "complete",
]


class CustomerProfile(TypedDict):
    customer_name: Optional[str]
    delivery_address: Optional[str]
    payment_method: Optional[str]
    info_stage: InfoStage


_profile_store: Dict[str, CustomerProfile] = {}


def _create_default_profile() -> CustomerProfile:
    return {
        "customer_name": None,
        "delivery_address": None,
        "payment_method": None,
        "info_stage": "need_name",
    }


def _get_profile(session_id: str) -> CustomerProfile:
    if session_id not in _profile_store:
        _profile_store[session_id] = _create_default_profile()
    return _profile_store[session_id]


def get_profile(session_id: Optional[str] = None) -> CustomerProfile:
    session = ensure_session_id(session_id)
    return _get_profile(session)


def get_customer_profile(session_id: Optional[str] = None) -> CustomerProfile:
    session = ensure_session_id(session_id)
    profile = _get_profile(session)
    return {
        "customer_name": profile["customer_name"],
        "delivery_address": profile["delivery_address"],
        "payment_method": profile["payment_method"],
        "info_stage": profile["info_stage"],
    }


def reset_customer_profile(session_id: Optional[str] = None) -> None:
    session = ensure_session_id(session_id)
    _profile_store[session] = _create_default_profile()


def is_order_ready(session_id: Optional[str] = None) -> bool:
    """
    Check if the current session has every customer field plus cart items.
    """

    session = ensure_session_id(session_id)
    profile = _get_profile(session)
    has_profile = all(
        [
            profile.get("customer_name"),
            profile.get("delivery_address"),
            profile.get("payment_method"),
        ]
    )
    return has_profile and cart_has_items(session)
