from __future__ import annotations

from typing import Dict, Optional, TypedDict, TYPE_CHECKING

from session_manager import ensure_session_id
from cart import cart_has_items
from utils_common import setup_logger

if TYPE_CHECKING:
    from agent import InfoStage
else:
    InfoStage = str  # runtime placeholder to avoid circular imports


logger = setup_logger("customer_profile")


class CustomerProfile(TypedDict):
    customer_name: Optional[str]
    delivery_address: Optional[str]
    info_stage: InfoStage


_profile_store: Dict[str, CustomerProfile] = {}


def _create_default_profile() -> CustomerProfile:
    return {
        "customer_name": None,
        "delivery_address": None,
        "info_stage": "need_name",
    }


def _get_profile(session_id: str) -> CustomerProfile:
    if session_id not in _profile_store:
        logger.debug("Initializing profile for session %s", session_id)
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
        "info_stage": profile["info_stage"],
    }


def reset_customer_profile(session_id: Optional[str] = None) -> None:
    session = ensure_session_id(session_id)
    _profile_store[session] = _create_default_profile()
    logger.debug("Reset profile to defaults | session=%s", session)


def is_order_ready(session_id: Optional[str] = None) -> bool:
    """
    Check if the current session has every customer field plus cart items.
    """

    session = ensure_session_id(session_id)
    profile = _get_profile(session)
    from cart import cart_is_confirmed  # Avoid circular import at module load time

    has_profile = all(
        [
            profile.get("customer_name"),
            profile.get("delivery_address"),
        ]
    )
    ready = has_profile and cart_has_items(session) and cart_is_confirmed(session)
    logger.debug(
        "Order readiness check | session=%s ready=%s has_profile=%s cart_items=%s cart_confirmed=%s",
        session,
        ready,
        has_profile,
        cart_has_items(session),
        cart_is_confirmed(session),
    )
    return ready


def handle_cart_changed(session_id: Optional[str] = None) -> None:
    """
    Adjust the profile stage when the cart is edited and confirmation resets.
    """

    session = ensure_session_id(session_id)
    profile = _get_profile(session)
    previous_stage = profile.get("info_stage")
    if (
        profile.get("customer_name")
        and profile.get("delivery_address")
        and cart_has_items(session)
    ):
        profile["info_stage"] = "awaiting_confirmation"
    elif profile.get("info_stage") == "complete":
        profile["info_stage"] = "idle"
    if profile.get("info_stage") != previous_stage:
        logger.debug(
            "Profile stage updated after cart change | session=%s %s -> %s",
            session,
            previous_stage,
            profile.get("info_stage"),
        )
