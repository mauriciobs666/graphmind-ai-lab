from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

import re
import unicodedata
from difflib import SequenceMatcher

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.tools import Tool
from langgraph.graph import END, MessagesState, StateGraph
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
_active_session_id: Optional[str] = None

logger = setup_logger("agent")


def get_memory(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in _memory_store:
        _memory_store[session_id] = InMemoryChatMessageHistory()
    return _memory_store[session_id]


def _get_cart(session_id: str) -> List[CartItem]:
    if session_id not in _cart_store:
        _cart_store[session_id] = []
    return _cart_store[session_id]


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
    RETURN p.sabor AS sabor, p.preco AS preco
    """
    try:
        result = graph.ro_query(query)
    except Exception:
        logger.exception("Falha ao consultar o pastel %s.", flavor)
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
            logger.warning("Preço inválido retornado para %s: %s", sabor_raw, preco_raw)
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


tools = [
    Tool.from_function(
        name="cardapio",
        description=(
            "Consultar o cardápio sobre sabores, ingredientes e preços. "
            "Use antes de responder dúvidas sobre os pastéis."
        ),
        func=cypher_qa,
    ),
    Tool.from_function(
        name="adicionar_carrinho",
        description=(
            "Adicionar um pastel ao carrinho depois de confirmar sabor e quantidade."
        ),
        func=add_to_cart_tool,
    ),
    Tool.from_function(
        name="ver_carrinho",
        description="Mostrar itens e total acumulado do carrinho.",
        func=show_cart_tool,
    ),
    Tool.from_function(
        name="limpar_carrinho",
        description="Esvaziar o carrinho quando o cliente quiser recomeçar.",
        func=clear_cart_tool,
    ),
]

system_prompt = """
Você é um atendente virtual da pastelaria Pastel do Mau.
Use sempre os dados do cardápio para sugerir sabores, consultar ingredientes e preços.
Seja amigável, use um tom acolhedor e ofereça recomendações baseadas nas preferências do cliente.
Desencoraje perguntas que não estejam relacionadas aos nossos pastéis ou ingredientes disponíveis.
Sempre que o cliente pedir informações sobre sabores, ingredientes ou preços, consulte a tool cardápio antes de responder.
Registre pedidos no carrinho usando as ferramentas apropriadas:
- `adicionar_carrinho` após confirmar sabor e quantidade.
- `ver_carrinho` para revisar os itens e informar totais.
- `limpar_carrinho` quando o cliente quiser recomeçar o pedido.
Se não souber a resposta, admita honestamente.
Seja sucinto e direto ao ponto em suas respostas.
Preços devem ser precisos conforme o cardápio e não podem ser alterados.
"""

_system_message = SystemMessage(content=system_prompt.strip())
_llm_with_tools = llm.bind_tools(tools)


def _call_agent(state: MessagesState):
    response = _llm_with_tools.invoke([_system_message, *state["messages"]])
    return {"messages": [response]}


workflow_builder = StateGraph(MessagesState)
workflow_builder.add_node("agent", _call_agent)
workflow_builder.add_node("tools", ToolNode(tools))
workflow_builder.add_conditional_edges(
    "agent",
    tools_condition,
    {"tools": "tools", "__end__": END},
)
workflow_builder.add_edge("tools", "agent")
workflow_builder.set_entry_point("agent")

agent_workflow = workflow_builder.compile()


def _extract_last_ai_message(messages: List[BaseMessage]) -> AIMessage | None:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return message
    return None


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

    try:
        state = agent_workflow.invoke({"messages": memory.messages})
    except Exception:
        logger.exception("LangGraph workflow failed.")
        return "Não consegui gerar uma resposta no momento."

    messages = state["messages"]
    memory.messages = list(messages)

    ai_message = _extract_last_ai_message(messages)
    if not ai_message:
        return "Não consegui gerar uma resposta no momento."

    return ai_message.content
