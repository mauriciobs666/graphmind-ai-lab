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
from utils_common import setup_logger

logger = setup_logger("cart")


class CartItem(TypedDict):
    sabor: str
    preco: float
    quantidade: int


class CartState(TypedDict):
    items: List[CartItem]
    order_confirmed: bool


_cart_store: Dict[str, CartState] = {}


def _format_currency(value: float) -> str:
    return f"R${value:.2f}".replace(".", ",")


def _normalize_text(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value or "")
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return stripped.lower().strip()


_QUANTITY_EXTRACTION_PROMPT = SystemMessage(
    content=(
        "Você extrai a quantidade e o sabor de pedidos de pastel. "
        "Use apenas números explicitamente informados pelo cliente. "
        "Responda apenas em JSON no formato "
        '{"sabor": "<sabor sem a quantidade na frente>", "quantidade": <numero ou null>}. '
        "Se não houver número claro, devolva quantidade como null e mantenha o sabor original. "
        "Não invente sabores ou quantidades."
    )
)


def _parse_llm_quantity_response(
    response_text: str
) -> tuple[str, Optional[int]] | None:
    payload = response_text
    if not payload:
        return None

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

    flavor = str(data.get("sabor") or "").strip()
    qty_raw = data.get("quantidade")

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
        "Você interpreta pedidos para remover ou diminuir itens do carrinho de pastéis. "
        "Responda somente em JSON com as chaves: "
        '{"sabor": "<sabor alvo>", "quantidade_remover": <numero ou null>, "remover_tudo": <true|false>}. '
        "Se o cliente pedir para remover totalmente (ou não citar número), use remover_tudo=true e quantidade_remover=null. "
        "Se o cliente pedir para tirar/apenas diminuir uma quantidade específica, use remover_tudo=false e quantidade_remover com esse número. "
        "Não invente sabores nem quantidades; use apenas o que estiver explícito."
    )
)


def _parse_llm_removal_response(
    response_text: str,
) -> tuple[str, Optional[int], bool] | None:
    payload = response_text
    if not payload:
        return None

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

    flavor = str(data.get("sabor") or "").strip()
    remove_all = bool(data.get("remover_tudo"))
    qty_raw = data.get("quantidade_remover")

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


def _extract_quantity_from_sabor(sabor: str) -> tuple[str, Optional[int]]:
    """
    Use the LLM to detect patterns like '2 pastéis de carne' and return ('carne', 2).
    """

    if not sabor:
        return "", None

    user_message = HumanMessage(content=sabor.strip())
    try:
        response = llm.invoke([_QUANTITY_EXTRACTION_PROMPT, user_message])
        parsed = _parse_llm_quantity_response(response.content)
        if parsed:
            logger.debug("LLM quantity extraction | input=%s parsed=%s", sabor, parsed)
            return parsed
    except Exception:
        logger.exception("LLM quantity extraction failed for '%s'.", sabor)

    return sabor.strip(), None


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
        logger.debug(
            "Pastel lookup match | query=%s match=%s score=%.2f", flavor, best_match, best_score
        )
        return best_match
    logger.debug("Pastel lookup had no confident match | query=%s score=%.2f", flavor, best_score)
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
    logger.debug(
        "Add to cart requested | sabor=%s quantidade_arg=%s parsed_qty=%s final_qty=%s flavor_query=%s",
        sabor,
        quantidade,
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
        if item["sabor"].lower() == pastel["sabor"].lower():
            item["quantidade"] += qty
            subtotal = item["preco"] * item["quantidade"]
            try:
                mark_cart_unconfirmed(session_id)
            except Exception:
                logger.exception("Failed to mark cart as unconfirmed after update.")
            logger.debug(
                "Updated cart item | session=%s sabor=%s nova_quantidade=%s subtotal=%.2f",
                session_id,
                item["sabor"],
                item["quantidade"],
                subtotal,
            )
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
    try:
        mark_cart_unconfirmed(session_id)
    except Exception:
        logger.exception("Failed to mark cart as unconfirmed after cart update.")
    logger.debug(
        "Added new cart item | session=%s sabor=%s quantidade=%s subtotal=%.2f",
        session_id,
        pastel["sabor"],
        qty,
        subtotal,
    )
    return (
        f"Adicionei {qty}× {pastel['sabor']} ao carrinho "
        f"(subtotal {_format_currency(subtotal)})."
    )


def remove_from_cart_tool(sabor: str, quantidade: Any = None) -> str:
    """
    Remove um item do carrinho ou diminui sua quantidade usando extração via LLM.
    """

    session_id = ensure_session_id()
    cart = _get_cart(session_id)
    if not cart:
        return "O carrinho já está vazio."

    user_text = sabor.strip()
    if quantidade not in (None, ""):
        user_text = f"{user_text} {quantidade}".strip()

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

    fallback_flavor, fallback_qty = _extract_quantity_from_sabor(flavor)
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
        item_norm = _normalize_text(item["sabor"])
        if item_norm == target_norm or target_norm in item_norm or item_norm in target_norm:
            match_item = item
            break

    if not match_item:
        logger.debug("Removal target not found | session=%s flavor_query=%s", session_id, flavor)
        return "Esse sabor não está no carrinho."

    if remove_all or remove_qty is None or remove_qty >= match_item["quantidade"]:
        cart.remove(match_item)
        message = f"Removi {match_item['sabor']} do carrinho."
        logger.debug(
            "Removed item from cart | session=%s sabor=%s remove_all=%s",
            session_id,
            match_item["sabor"],
            remove_all or remove_qty is None or remove_qty >= match_item["quantidade"],
        )
    else:
        match_item["quantidade"] -= remove_qty
        subtotal = match_item["preco"] * match_item["quantidade"]
        message = (
            f"Atualizei {match_item['sabor']} para {match_item['quantidade']}× "
            f"(subtotal {_format_currency(subtotal)})."
        )
        logger.debug(
            "Decreased item quantity | session=%s sabor=%s nova_quantidade=%s subtotal=%.2f",
            session_id,
            match_item["sabor"],
            match_item["quantidade"],
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

    total = sum(item["preco"] * item["quantidade"] for item in cart)
    lines = ["Itens no carrinho:"] + _cart_lines(cart)
    lines.append(f"Total: {_format_currency(total)}")
    return "\n".join(lines)


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
