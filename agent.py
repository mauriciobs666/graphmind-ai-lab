from __future__ import annotations

from typing import Annotated, Any, Dict, List, Optional, TypedDict, Literal

import re
import unicodedata
from difflib import SequenceMatcher

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import Tool
from langgraph.graph import END, StateGraph, add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from cypher import cypher_qa
from graph import graph
from llm import llm
from utils_common import get_session_id, setup_logger


class CartItem(TypedDict):
    sabor: str
    preco: float
    quantidade: int


_memory_store: Dict[str, InMemoryChatMessageHistory] = {}
_cart_store: Dict[str, List[CartItem]] = {}
_profile_store: Dict[str, "CustomerProfile"] = {}
_active_session_id: Optional[str] = None

# NOTE: keep string literals in sync with InfoStage type hints
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


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    customer_name: Optional[str]
    delivery_address: Optional[str]
    payment_method: Optional[str]
    info_stage: InfoStage

logger = setup_logger("agent")


def get_memory(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in _memory_store:
        _memory_store[session_id] = InMemoryChatMessageHistory()
    return _memory_store[session_id]


def _get_cart(session_id: str) -> List[CartItem]:
    if session_id not in _cart_store:
        _cart_store[session_id] = []
    return _cart_store[session_id]


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


def get_customer_profile(session_id: Optional[str] = None) -> CustomerProfile:
    session = _ensure_session_id(session_id)
    profile = _get_profile(session)
    return {
        "customer_name": profile["customer_name"],
        "delivery_address": profile["delivery_address"],
        "payment_method": profile["payment_method"],
        "info_stage": profile["info_stage"],
    }


def reset_customer_profile(session_id: Optional[str] = None) -> None:
    session = _ensure_session_id(session_id)
    _profile_store[session] = _create_default_profile()


def _ensure_session_id(explicit: Optional[str] = None) -> str:
    """
    Resolve the active session id without touching Streamlit in worker threads.
    """

    global _active_session_id

    if explicit:
        _active_session_id = explicit
        return explicit

    if _active_session_id:
        return _active_session_id

    session_id = get_session_id()
    _active_session_id = session_id
    return session_id


def get_cart_snapshot(session_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Return a copy of the current cart so the UI can render it.
    """

    session = _ensure_session_id(session_id)
    cart = list(_get_cart(session))
    total = sum(item["preco"] * item["quantidade"] for item in cart)
    return {
        "items": [dict(item) for item in cart],
        "total": total,
    }


def _cart_has_items(session_id: Optional[str] = None) -> bool:
    session = _ensure_session_id(session_id)
    return bool(_get_cart(session))


def is_order_ready(session_id: Optional[str] = None) -> bool:
    """
    Check if the current session has every customer field plus cart items.
    """

    session = _ensure_session_id(session_id)
    profile = _get_profile(session)
    has_profile = all(
        [
            profile.get("customer_name"),
            profile.get("delivery_address"),
            profile.get("payment_method"),
        ]
    )
    return has_profile and _cart_has_items(session)


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


_NAME_PATTERNS = [
    re.compile(r"\bmeu nome\s+(?:é|eh)\s+([A-Za-zÀ-ÿ' ]+)", re.IGNORECASE),
    re.compile(r"\bme chamo\s+([A-Za-zÀ-ÿ' ]+)", re.IGNORECASE),
    re.compile(r"\bsou\s+(?:o|a)?\s*([A-Za-zÀ-ÿ' ]+)", re.IGNORECASE),
]


def _clean_simple_response(text: str) -> Optional[str]:
    candidate = text.strip().strip(".!,;:")
    if not candidate:
        return None
    words = candidate.split()
    if not 1 <= len(words) <= 4:
        return None
    if any(ch.isdigit() for ch in candidate):
        return None
    return " ".join(word.capitalize() for word in words)


def _extract_name_from_text(text: str, allow_simple: bool = False) -> Optional[str]:
    if not text:
        return None
    for pattern in _NAME_PATTERNS:
        match = pattern.search(text)
        if match:
            cleaned = match.group(1).strip()
            if cleaned:
                return _clean_simple_response(cleaned) or cleaned
    if allow_simple:
        return _clean_simple_response(text)
    return None


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

    session_id = _ensure_session_id()
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

    session_id = _ensure_session_id()
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

    session_id = _ensure_session_id()
    _cart_store[session_id] = []
    return "Esvaziei o carrinho. Pode recomeçar o pedido!"


def _extract_last_ai_message(messages: List[BaseMessage]) -> AIMessage | None:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return message
    return None


def _extract_last_user_message(messages: List[BaseMessage]) -> HumanMessage | None:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return message
    return None


def _collect_name(state: AgentState):
    info_stage = state.get("info_stage", "need_name")
    customer_name = state.get("customer_name")

    if customer_name:
        if info_stage in {"need_name", "awaiting_name"}:
            return {"info_stage": "idle"}
        return {}

    user_message = _extract_last_user_message(state["messages"])
    allow_simple = info_stage == "awaiting_name"
    extracted = (
        _extract_name_from_text(
            user_message.content if user_message else "",
            allow_simple=allow_simple,
        )
        if user_message
        else None
    )

    if extracted:
        return {"customer_name": extracted, "info_stage": "idle"}

    prompt = (
        "Antes de continuarmos com o pedido, poderia me dizer seu nome?"
        if info_stage == "need_name"
        else "Ainda preciso do seu nome para continuar. Como posso te chamar?"
    )
    return {
        "messages": [AIMessage(content=prompt)],
        "info_stage": "awaiting_name",
    }


def _collect_address(state: AgentState):
    info_stage = state.get("info_stage", "need_name")
    if info_stage in {"need_name", "awaiting_name"}:
        return {}

    if state.get("delivery_address"):
        if info_stage == "awaiting_address":
            return {"info_stage": "idle"}
        return {}

    if not _cart_has_items():
        return {}

    user_message = _extract_last_user_message(state["messages"])
    if info_stage == "awaiting_address" and user_message:
        address = user_message.content.strip()
        if address:
            return {
                "delivery_address": address,
                "info_stage": "idle",
            }

    prompt = (
        "Ótimo, agora me informe o endereço completo para entrega, por favor."
        if info_stage != "awaiting_address"
        else "Não consegui entender o endereço. Pode repetir com rua e número?"
    )
    return {
        "messages": [AIMessage(content=prompt)],
        "info_stage": "awaiting_address",
    }


def _collect_payment(state: AgentState):
    info_stage = state.get("info_stage", "need_name")
    if info_stage in {"need_name", "awaiting_name", "awaiting_address"}:
        return {}

    if not state.get("delivery_address"):
        return {}

    if state.get("payment_method"):
        if info_stage == "awaiting_payment":
            return {"info_stage": "complete"}
        return {}

    if not _cart_has_items():
        return {}

    user_message = _extract_last_user_message(state["messages"])
    if info_stage == "awaiting_payment" and user_message:
        method = user_message.content.strip()
        if method:
            return {
                "payment_method": method,
                "info_stage": "complete",
            }

    prompt = (
        "Qual forma de pagamento você prefere (PIX, cartão na entrega ou dinheiro)?"
        if info_stage != "awaiting_payment"
        else "Pode confirmar a forma de pagamento (PIX, cartão ou dinheiro)?"
    )
    return {
        "messages": [AIMessage(content=prompt)],
        "info_stage": "awaiting_payment",
    }


def _name_condition(state: AgentState) -> Literal["ask", "next"]:
    if state.get("customer_name"):
        return "next"
    if state.get("info_stage") in {"need_name", "awaiting_name"}:
        return "ask"
    return "next"


def _address_condition(state: AgentState) -> Literal["ask", "next"]:
    if state.get("info_stage") == "awaiting_address" and not state.get(
        "delivery_address"
    ):
        return "ask"
    return "next"


def _payment_condition(state: AgentState) -> Literal["ask", "next"]:
    if state.get("info_stage") == "awaiting_payment" and not state.get(
        "payment_method"
    ):
        return "ask"
    return "next"


tools = [
    Tool.from_function(
        name="cardapio",
        description=(
            "Consult the cardápio for flavors, ingredients, and prices. "
            "Use it before answering any pastel-related question."
        ),
        func=cypher_qa,
    ),
    Tool.from_function(
        name="adicionar_carrinho",
        description=(
            "Add a pastel to the cart after confirming the flavor and quantity."
        ),
        func=add_to_cart_tool,
    ),
    Tool.from_function(
        name="ver_carrinho",
        description="Show the cart items and the running total.",
        func=show_cart_tool,
    ),
    Tool.from_function(
        name="limpar_carrinho",
        description="Empty the cart when the customer wants to start over.",
        func=clear_cart_tool,
    ),
]

system_prompt = """
You are the virtual attendant for the Pastel do Mau shop.
Always rely on the cardápio data to suggest flavors, ingredients, and prices.
Maintain an affectionate, upbeat tone—use pequenos elogios ou agradecimentos—while keeping answers enxutas (no more than a short paragraph or two sentences whenever possible).
Discourage questions unrelated to our pastéis or ingredients.
Whenever the customer asks about flavors, ingredients, or prices, consult the `cardapio` tool before replying.
Record orders in the cart with the proper tools:
- `adicionar_carrinho` after confirming flavor and quantity.
- `ver_carrinho` to review the current cart and totals.
- `limpar_carrinho` when the customer wants to start over.
If you do not know the answer, say so honestly.
Keep responses concise and focused even while being warm.
Prices must match the menu and cannot be changed.
Respond to the customer in Brazilian Portuguese.
Use the shared state values when responding:
- `customer_name`: greet the customer by name once available.
- `delivery_address` and `payment_method`: confirm them explicitly after they are recorded so the customer can correct mistakes.
If any of these values are missing the guard nodes will ask the user, so resume the normal conversation once they exist.
"""

_system_message = SystemMessage(content=system_prompt.strip())
_llm_with_tools = llm.bind_tools(tools)


def _call_agent(state: AgentState):
    response = _llm_with_tools.invoke([_system_message, *state["messages"]])
    return {"messages": [response]}


workflow_builder = StateGraph(AgentState)
workflow_builder.add_node("collect_name", _collect_name)
workflow_builder.add_node("collect_address", _collect_address)
workflow_builder.add_node("collect_payment", _collect_payment)
workflow_builder.add_node("agent", _call_agent)
workflow_builder.add_node("tools", ToolNode(tools))
workflow_builder.add_conditional_edges(
    "collect_name",
    _name_condition,
    {"ask": END, "next": "collect_address"},
)
workflow_builder.add_conditional_edges(
    "collect_address",
    _address_condition,
    {"ask": END, "next": "collect_payment"},
)
workflow_builder.add_conditional_edges(
    "collect_payment",
    _payment_condition,
    {"ask": END, "next": "agent"},
)
workflow_builder.add_conditional_edges(
    "agent", tools_condition, {"tools": "tools", "__end__": END}
)
workflow_builder.add_edge("tools", "agent")
workflow_builder.set_entry_point("collect_name")

agent_workflow = workflow_builder.compile()


def generate_response(user_input: str) -> str:
    """
    Generate a response for the given user input using the agent.

    Args:
        user_input (str): The input message from the user.

    Returns:
        str: The agent's response to the user input.
    """

    session_id = _ensure_session_id()
    memory = get_memory(session_id)
    memory.add_user_message(user_input)
    profile = _get_profile(session_id)

    graph_state: AgentState = {
        "messages": memory.messages,
        "customer_name": profile["customer_name"],
        "delivery_address": profile["delivery_address"],
        "payment_method": profile["payment_method"],
        "info_stage": profile["info_stage"],
    }

    try:
        state = agent_workflow.invoke(graph_state)
    except Exception:
        logger.exception("LangGraph workflow failed.")
        return "Não consegui gerar uma resposta no momento."

    messages = state["messages"]
    memory.messages = list(messages)
    new_name = state.get("customer_name")
    if new_name is not None:
        profile["customer_name"] = new_name
    new_address = state.get("delivery_address")
    if new_address is not None:
        profile["delivery_address"] = new_address
    new_payment = state.get("payment_method")
    if new_payment is not None:
        profile["payment_method"] = new_payment
    new_stage = state.get("info_stage")
    if new_stage:
        profile["info_stage"] = new_stage

    ai_message = _extract_last_ai_message(messages)
    if not ai_message:
        return "Não consegui gerar uma resposta no momento."

    return ai_message.content
