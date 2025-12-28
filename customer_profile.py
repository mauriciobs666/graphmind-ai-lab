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
    "awaiting_confirmation",
    "complete",
]


class CustomerProfile(TypedDict):
    customer_name: Optional[str]
    delivery_address: Optional[str]
    payment_method: Optional[str]
    info_stage: InfoStage
    order_confirmed: bool


_profile_store: Dict[str, CustomerProfile] = {}


def _create_default_profile() -> CustomerProfile:
    return {
        "customer_name": None,
        "delivery_address": None,
        "payment_method": None,
        "info_stage": "need_name",
        "order_confirmed": False,
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
        "order_confirmed": profile["order_confirmed"],
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
            profile.get("order_confirmed"),
        ]
    )
    return has_profile and cart_has_items(session)


def mark_order_unconfirmed(session_id: Optional[str] = None) -> None:
    """
    Clear the confirmation flag whenever the order changes.
    """

    session = ensure_session_id(session_id)
    profile = _get_profile(session)
    profile["order_confirmed"] = False
    if (
        profile.get("customer_name")
        and profile.get("delivery_address")
        and profile.get("payment_method")
        and cart_has_items(session)
    ):
        profile["info_stage"] = "awaiting_confirmation"
    elif profile.get("info_stage") == "complete":
        profile["info_stage"] = "idle"
