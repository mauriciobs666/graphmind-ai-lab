from __future__ import annotations

from typing import Any, Dict, Optional

from cart import cart_is_confirmed, get_cart_snapshot
from customer_profile import get_profile
from session_manager import ensure_session_id, get_memory


def get_session_snapshot(session_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Return a lightweight snapshot of session state for debugging/support.
    """

    session = ensure_session_id(session_id)
    profile = get_profile(session)
    cart = get_cart_snapshot(session)
    memory = get_memory(session)

    return {
        "session_id": session,
        "info_stage": profile.get("info_stage"),
        "has_name": bool(profile.get("customer_name")),
        "has_address": bool(profile.get("delivery_address")),
        "cart_items": len(cart.get("items", [])),
        "cart_total": cart.get("total", 0.0),
        "cart_confirmed": cart_is_confirmed(session),
        "message_count": len(memory.messages),
    }
