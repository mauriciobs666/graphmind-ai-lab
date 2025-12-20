from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, TypedDict

from graph import graph
from session_manager import ensure_session_id
from utils_common import setup_logger

logger = setup_logger("cart")


class CartItem(TypedDict):
    sabor: str
    preco: float
    quantidade: int


_cart_store: Dict[str, List[CartItem]] = {}


def _format_currency(value: float) -> str:
    return f"R${value:.2f}".replace(".", ",")


def _normalize_text(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value or "")
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return stripped.lower().strip()


_QUANTITY_PREFIX = re.compile(
    r"""
    ^\s*(\d+)\s*            # leading quantity
    (?:x|vez(?:es)?|past[eé]is?|unidades?|pcs?|pçs?)?\s*  # optional unit markers
    (?:de|do|da)?\s*        # optional filler words
    (.+)$                   # remaining flavor text
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _extract_quantity_from_sabor(sabor: str) -> tuple[str, Optional[int]]:
    """
    Detect patterns like '2 pastéis de carne' and return ('carne', 2).
    """

    if not sabor:
        return "", None

    match = _QUANTITY_PREFIX.match(sabor.strip())
    if not match:
        return sabor.strip(), None

    qty = int(match.group(1))
    remainder = match.group(2).strip()
    if not remainder:
        return sabor.strip(), None
    return remainder, qty


def _lookup_pastel(flavor: str) -> Optional[Dict[str, Any]]:
    target = _normalize_text(flavor)
    if not target:
        return None

    query = """
    MATCH (p:Pastel)
    RETURN p.flavor AS sabor, p.price AS preco
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
        sabor_raw, preco_raw = row[:2]
        normalized = _normalize_text(str(sabor_raw))
        if not normalized:
            continue

        try:
            price_value = float(preco_raw)
        except (TypeError, ValueError):
            logger.warning("Invalid price returned for %s: %s", sabor_raw, preco_raw)
            continue

        if normalized == target:
            return {"sabor": str(sabor_raw), "preco": price_value}

        if target in normalized or normalized in target:
            score = 0.9
        else:
            score = SequenceMatcher(None, target, normalized).ratio()

        if score > best_score:
            best_score = score
            best_match = {"sabor": str(sabor_raw), "preco": price_value}

    # Require a reasonable similarity level to avoid random matches.
    if best_match and best_score >= 0.55:
        return best_match
    return None


def _get_cart(session_id: str) -> List[CartItem]:
    if session_id not in _cart_store:
        _cart_store[session_id] = []
    return _cart_store[session_id]


def cart_has_items(session_id: Optional[str] = None) -> bool:
    session = ensure_session_id(session_id)
    return bool(_get_cart(session))


def get_cart_snapshot(session_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Return a copy of the current cart so the UI can render it.
    """

    session = ensure_session_id(session_id)
    cart = list(_get_cart(session))
    total = sum(item["preco"] * item["quantidade"] for item in cart)
    return {
        "items": [dict(item) for item in cart],
        "total": total,
    }


def _cart_lines(cart: List[CartItem]) -> List[str]:
    lines = []
    for item in cart:
        subtotal = item["preco"] * item["quantidade"]
        lines.append(
            f"{item['quantidade']}× {item['sabor']} — "
            f"{_format_currency(item['preco'])} cada (subtotal {_format_currency(subtotal)})"
        )
    return lines


def add_to_cart_tool(sabor: str, quantidade: Any = 1) -> str:
    """
    Add an item to the session cart after confirming flavor and quantity.
    """

    flavor_hint, parsed_qty = _extract_quantity_from_sabor(sabor)
    try:
        qty = int(quantidade)
    except (TypeError, ValueError):
        qty = parsed_qty or 0

    if qty <= 1 and parsed_qty and parsed_qty > 1:
        qty = parsed_qty

    if qty <= 0:
        qty = 1

    flavor_query = flavor_hint or sabor
    pastel = _lookup_pastel(flavor_query)
    if not pastel:
        return "Não encontrei esse sabor no cardápio."

    session_id = ensure_session_id()
    cart = _get_cart(session_id)

    for item in cart:
        if item["sabor"].lower() == pastel["sabor"].lower():
            item["quantidade"] += qty
            subtotal = item["preco"] * item["quantidade"]
            return (
                f"Atualizei o carrinho: agora são {item['quantidade']}× {item['sabor']} "
                f"(subtotal {_format_currency(subtotal)})."
            )

    cart.append(
        {
            "sabor": pastel["sabor"],
            "preco": pastel["preco"],
            "quantidade": qty,
        }
    )
    subtotal = pastel["preco"] * qty
    return (
        f"Adicionei {qty}× {pastel['sabor']} ao carrinho "
        f"(subtotal {_format_currency(subtotal)})."
    )


def show_cart_tool(_: str = "") -> str:
    """
    Return a human-friendly summary of the cart contents.
    """

    session_id = ensure_session_id()
    cart = _get_cart(session_id)
    if not cart:
        return "O carrinho está vazio."

    total = sum(item["preco"] * item["quantidade"] for item in cart)
    lines = ["Itens no carrinho:"] + _cart_lines(cart)
    lines.append(f"Total: {_format_currency(total)}")
    return "\n".join(lines)


def clear_cart_tool(_: str = "") -> str:
    """
    Empty the cart for the active session.
    """

    session_id = ensure_session_id()
    _cart_store[session_id] = []
    return "Esvaziei o carrinho. Pode recomeçar o pedido!"
