from __future__ import annotations

import json
import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage

from graph import graph
from llm import llm
from session_manager import ensure_session_id
from utils_common import format_currency, setup_logger

logger = setup_logger("cart")


class CartItem(TypedDict):
    flavor: str
    price: float
    quantity: int


class CartState(TypedDict):
    items: List[CartItem]
    order_confirmed: bool


_cart_store: Dict[str, CartState] = {}


def _normalize_text(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value or "")
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return stripped.lower().strip()


def _extract_json_payload(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    payload = text
    if not payload.strip().startswith("{"):
        match = re.search(r"\{.*\}", payload, re.DOTALL)
        if match:
            payload = match.group(0)
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    return data


_QUANTITY_EXTRACTION_PROMPT = SystemMessage(
    content=(
        "You extract the quantity and flavor from pastel orders. "
        "Use only numbers explicitly provided by the customer. "
        "Reply only in JSON with the format "
        '{"flavor": "<flavor without the leading quantity>", "quantity": <number or null>}. '
        "If no clear number is provided, return quantity as null and keep the original flavor. "
        "Do not invent flavors or quantities."
    )
)


def _parse_llm_quantity_response(
    response_text: str
) -> tuple[str, Optional[int]] | None:
    data = _extract_json_payload(response_text)
    if not data:
        return None

    flavor = str(data.get("flavor") or "").strip()
    qty_raw = data.get("quantity")

    qty: Optional[int] = None
    if isinstance(qty_raw, (int, float, str)):
        try:
            qty = int(qty_raw)
            if qty <= 0:
                qty = None
        except (TypeError, ValueError):
            qty = None

    if not flavor:
        return None

    return flavor, qty


_REMOVAL_EXTRACTION_PROMPT = SystemMessage(
    content=(
        "You interpret requests to remove or decrease items from the pastel cart. "
        "Reply only in JSON with the keys: "
        '{"flavor": "<target flavor>", "quantity_to_remove": <number or null>, "remove_all": <true|false>}. '
        "If the customer asks to remove the item entirely (or provides no number), use remove_all=true and quantity_to_remove=null. "
        "If the customer asks to remove/decrease a specific quantity, use remove_all=false and quantity_to_remove with that number. "
        "Do not invent flavors or quantities; use only what is explicit."
    )
)


def _parse_llm_removal_response(
    response_text: str,
) -> tuple[str, Optional[int], bool] | None:
    data = _extract_json_payload(response_text)
    if not data:
        return None

    flavor = str(data.get("flavor") or "").strip()
    remove_all = bool(data.get("remove_all"))
    qty_raw = data.get("quantity_to_remove")

    qty: Optional[int] = None
    if isinstance(qty_raw, (int, float, str)):
        try:
            qty = int(qty_raw)
            if qty <= 0:
                qty = None
        except (TypeError, ValueError):
            qty = None

    if not flavor:
        return None

    return flavor, qty, remove_all


def _extract_quantity_from_flavor(flavor: str) -> tuple[str, Optional[int]]:
    """
    Use the LLM to detect patterns like '2 pasteis de carne' and return ('carne', 2).
    """

    if not flavor:
        return "", None

    user_message = HumanMessage(content=flavor.strip())
    try:
        response = llm.invoke([_QUANTITY_EXTRACTION_PROMPT, user_message])
        parsed = _parse_llm_quantity_response(response.content)
        if parsed:
            logger.debug("LLM quantity extraction | input=%s parsed=%s", flavor, parsed)
            return parsed
    except Exception:
        logger.exception("LLM quantity extraction failed for '%s'.", flavor)

    return flavor.strip(), None


def _lookup_pastel(flavor: str) -> Optional[Dict[str, Any]]:
    target = _normalize_text(flavor)
    if not target:
        return None

    query = """
    MATCH (p:Pastel)
    RETURN p.name AS flavor, p.price AS price
    """
    try:
        result = graph.ro_query(query)
    except Exception:
        logger.exception("Failed to query pastel %s.", flavor)
        return None

    rows = getattr(result, "result_set", [])
    best_match: Optional[Dict[str, Any]] = None
    best_score = 0.0

    for row in rows:
        flavor_raw, price_raw = row[:2]
        normalized = _normalize_text(str(flavor_raw))
        if not normalized:
            continue

        try:
            price_value = float(price_raw)
        except (TypeError, ValueError):
            logger.warning("Invalid price returned for %s: %s", flavor_raw, price_raw)
            continue

        if normalized == target:
            return {"flavor": str(flavor_raw), "price": price_value}

        if target in normalized or normalized in target:
            score = 0.9
        else:
            score = SequenceMatcher(None, target, normalized).ratio()

        if score > best_score:
            best_score = score
            best_match = {"flavor": str(flavor_raw), "price": price_value}

    # Require a reasonable similarity level to avoid random matches.
    if best_match and best_score >= 0.55:
        logger.debug(
            "Pastel lookup match | query=%s match=%s score=%.2f",
            flavor,
            best_match,
            best_score,
        )
        return best_match
    logger.debug(
        "Pastel lookup had no confident match | query=%s score=%.2f", flavor, best_score
    )
    return None


def _create_cart_state() -> CartState:
    return {"items": [], "order_confirmed": False}


def _get_cart_state(session_id: str) -> CartState:
    if session_id not in _cart_store:
        logger.debug("Initializing cart state for session %s", session_id)
        _cart_store[session_id] = _create_cart_state()
    return _cart_store[session_id]


def _get_cart(session_id: str) -> List[CartItem]:
    return _get_cart_state(session_id)["items"]


def cart_is_confirmed(session_id: Optional[str] = None) -> bool:
    session = ensure_session_id(session_id)
    return _get_cart_state(session)["order_confirmed"]


def _notify_profile_cart_changed(session_id: str) -> None:
    try:
        from customer_profile import handle_cart_changed

        handle_cart_changed(session_id)
        logger.debug("Synced profile after cart change | session=%s", session_id)
    except Exception:
        logger.exception("Failed to sync profile after cart change.")


def mark_cart_unconfirmed(session_id: Optional[str] = None) -> None:
    session = ensure_session_id(session_id)
    state = _get_cart_state(session)
    state["order_confirmed"] = False
    logger.debug("Marked cart as unconfirmed | session=%s", session)
    _notify_profile_cart_changed(session)


def set_cart_confirmation(confirmed: bool, session_id: Optional[str] = None) -> None:
    session = ensure_session_id(session_id)
    state = _get_cart_state(session)
    state["order_confirmed"] = bool(confirmed)
    logger.debug("Set cart confirmation | session=%s confirmed=%s", session, confirmed)
    if not confirmed:
        _notify_profile_cart_changed(session)


def cart_has_items(session_id: Optional[str] = None) -> bool:
    session = ensure_session_id(session_id)
    return bool(_get_cart(session))


def get_cart_snapshot(session_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Return a copy of the current cart so the UI can render it.
    """

    session = ensure_session_id(session_id)
    cart = list(_get_cart(session))
    total = sum(item["price"] * item["quantity"] for item in cart)
    return {
        "items": [dict(item) for item in cart],
        "total": total,
    }


def _cart_lines(cart: List[CartItem]) -> List[str]:
    lines = []
    for item in cart:
        subtotal = item["price"] * item["quantity"]
        lines.append(
            f"{item['quantity']}× {item['flavor']} — "
            f"{format_currency(item['price'])} cada (subtotal {format_currency(subtotal)})"
        )
    return lines


def add_to_cart_tool(flavor: str, quantity: Any = 1) -> str:
    """
    Add an item to the session cart after confirming flavor and quantity.
    """

    flavor_hint, parsed_qty = _extract_quantity_from_flavor(flavor)
    try:
        qty = int(quantity)
    except (TypeError, ValueError):
        qty = parsed_qty or 0

    if qty <= 1 and parsed_qty and parsed_qty > 1:
        qty = parsed_qty

    if qty <= 0:
        qty = 1

    flavor_query = flavor_hint or flavor
    logger.debug(
        "Add to cart requested | flavor=%s quantity_arg=%s parsed_qty=%s final_qty=%s flavor_query=%s",
        flavor,
        quantity,
        parsed_qty,
        qty,
        flavor_query,
    )
    pastel = _lookup_pastel(flavor_query)
    if not pastel:
        return "Não encontrei esse sabor no cardápio."

    session_id = ensure_session_id()
    cart = _get_cart(session_id)

    for item in cart:
        if item["flavor"].lower() == pastel["flavor"].lower():
            item["quantity"] += qty
            subtotal = item["price"] * item["quantity"]
            try:
                mark_cart_unconfirmed(session_id)
            except Exception:
                logger.exception("Failed to mark cart as unconfirmed after update.")
            logger.debug(
                "Updated cart item | session=%s flavor=%s new_quantity=%s subtotal=%.2f",
                session_id,
                item["flavor"],
                item["quantity"],
                subtotal,
            )
            return (
                f"Atualizei o carrinho: agora são {item['quantity']}× {item['flavor']} "
                f"(subtotal {format_currency(subtotal)})."
            )

    cart.append(
        {
            "flavor": pastel["flavor"],
            "price": pastel["price"],
            "quantity": qty,
        }
    )
    subtotal = pastel["price"] * qty
    try:
        mark_cart_unconfirmed(session_id)
    except Exception:
        logger.exception("Failed to mark cart as unconfirmed after cart update.")
    logger.debug(
        "Added new cart item | session=%s flavor=%s quantity=%s subtotal=%.2f",
        session_id,
        pastel["flavor"],
        qty,
        subtotal,
    )
    return (
        f"Adicionei {qty}× {pastel['flavor']} ao carrinho "
        f"(subtotal {format_currency(subtotal)})."
    )


def remove_from_cart_tool(flavor: str, quantity: Any = None) -> str:
    """
    Remove an item from the cart or decrease its quantity using LLM extraction.
    """

    session_id = ensure_session_id()
    cart = _get_cart(session_id)
    if not cart:
        return "O carrinho já está vazio."

    user_text = flavor.strip()
    if quantity not in (None, ""):
        user_text = f"{user_text} {quantity}".strip()

    flavor: str = user_text
    remove_qty: Optional[int] = None
    remove_all = False

    if user_text:
        try:
            response = llm.invoke([_REMOVAL_EXTRACTION_PROMPT, HumanMessage(content=user_text)])
            parsed = _parse_llm_removal_response(response.content)
            if parsed:
                flavor, remove_qty, remove_all = parsed
                logger.debug(
                    "LLM removal extraction | input=%s parsed_flavor=%s remove_qty=%s remove_all=%s",
                    user_text,
                    flavor,
                    remove_qty,
                    remove_all,
                )
        except Exception:
            logger.exception("LLM removal extraction failed for '%s'.", user_text)

    if not flavor:
        return "Preciso do sabor para remover do carrinho."

    fallback_flavor, fallback_qty = _extract_quantity_from_flavor(flavor)
    if not remove_qty:
        remove_qty = fallback_qty
    if not remove_all and remove_qty is None:
        remove_all = True
    flavor = fallback_flavor or flavor
    logger.debug(
        "Normalized removal request | flavor=%s remove_qty=%s remove_all=%s",
        flavor,
        remove_qty,
        remove_all,
    )

    target_norm = _normalize_text(flavor)
    match_item: Optional[CartItem] = None
    for item in cart:
        item_norm = _normalize_text(item["flavor"])
        if item_norm == target_norm or target_norm in item_norm or item_norm in target_norm:
            match_item = item
            break

    if not match_item:
        logger.debug("Removal target not found | session=%s flavor_query=%s", session_id, flavor)
        return "Esse sabor não está no carrinho."

    if remove_all or remove_qty is None or remove_qty >= match_item["quantity"]:
        cart.remove(match_item)
        message = f"Removi {match_item['flavor']} do carrinho."
        logger.debug(
            "Removed item from cart | session=%s flavor=%s remove_all=%s",
            session_id,
            match_item["flavor"],
            remove_all or remove_qty is None or remove_qty >= match_item["quantity"],
        )
    else:
        match_item["quantity"] -= remove_qty
        subtotal = match_item["price"] * match_item["quantity"]
        message = (
            f"Atualizei {match_item['flavor']} para {match_item['quantity']}× "
            f"(subtotal {format_currency(subtotal)})."
        )
        logger.debug(
            "Decreased item quantity | session=%s flavor=%s new_quantity=%s subtotal=%.2f",
            session_id,
            match_item["flavor"],
            match_item["quantity"],
            subtotal,
        )

    try:
        mark_cart_unconfirmed(session_id)
    except Exception:
        logger.exception("Failed to mark cart as unconfirmed after removal.")

    return message


def show_cart_tool(_: str = "") -> str:
    """
    Return a human-friendly summary of the cart contents.
    """

    session_id = ensure_session_id()
    cart = _get_cart(session_id)
    if not cart:
        return "O carrinho está vazio."

    total = sum(item["price"] * item["quantity"] for item in cart)
    lines = ["Itens no carrinho:"] + _cart_lines(cart)
    lines.append(f"Total: {format_currency(total)}")
    return "\n\n".join(lines)


def clear_cart_tool(_: str = "") -> str:
    """
    Empty the cart for the active session.
    """

    session_id = ensure_session_id()
    state = _get_cart_state(session_id)
    state["items"] = []
    state["order_confirmed"] = False
    try:
        mark_cart_unconfirmed(session_id)
    except Exception:
        logger.exception("Failed to reset cart confirmation after clearing cart.")
    return "Esvaziei o carrinho. Pode recomeçar o pedido!"
