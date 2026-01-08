from __future__ import annotations

from typing import Annotated, List, Optional, TypedDict, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import Tool
from langgraph.graph import END, StateGraph, add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from cart import (
    add_to_cart_tool,
    cart_has_items,
    cart_is_confirmed,
    clear_cart_tool,
    get_cart_snapshot,
    remove_from_cart_tool,
    set_cart_confirmation,
    show_cart_tool,
)
from customer_profile import get_customer_profile, get_profile, is_order_ready
from cypher import cypher_qa
from llm import llm
from session_manager import ensure_session_id, get_memory
from utils_common import format_currency, setup_logger

InfoStage = Literal[
    "need_name",
    "awaiting_name",
    "idle",
    "awaiting_address",
    "awaiting_confirmation",
    "complete",
]


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    customer_name: Optional[str]
    delivery_address: Optional[str]
    order_confirmed: bool
    info_stage: InfoStage
    last_intent: Optional[str]

logger = setup_logger("agent")


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


def _classify_intent_with_llm(messages: List[BaseMessage]) -> str:
    """
    Let the LLM decide whether the user wants to edit the cart, provide data, or confirm the order.
    """

    last_user = _extract_last_user_message(messages)
    if not last_user:
        return "unknown"

    instruction = SystemMessage(
        content=(
            "Classify the customer's latest message into exactly one option: "
            "'cart_edit' (when they want to add, remove, or change items, or ask for the menu), "
            "'provide_info' (when they provide data like name or delivery address), "
            "'confirm_order' (when they confirm or approve the order, say you can send it, "
            "say everything is correct, or that you can close the order), "
            "or 'other'. Reply with only one of these words."
        )
    )
    try:
        response = llm.invoke([instruction, last_user])
        label = response.content.strip().lower()
        if label in {"cart_edit", "provide_info", "confirm_order", "other"}:
            logger.debug("Intent classified as: %s", label)
            return label
        logger.debug("LLM returned unrecognized intent label: %s", label)
    except Exception:
        logger.exception("LLM intent classification failed.")
    return "unknown"


def _get_intent(state: AgentState) -> str:
    cached = state.get("last_intent")
    if cached:
        logger.debug("Using cached intent: %s", cached)
        return cached
    intent = _classify_intent_with_llm(state["messages"])
    state["last_intent"] = intent
    logger.debug("Intent cached as: %s", intent)
    return intent


def _summarize_recent_user_messages(
    messages: List[BaseMessage], limit: int = 8
) -> str:
    """
    Keep only recent human messages to reduce prompt size for extraction calls.
    """

    user_messages = [
        message.content for message in messages if isinstance(message, HumanMessage)
    ]
    if not user_messages:
        return ""
    trimmed = user_messages[-limit:]
    return "\n".join(f"Customer: {msg}" for msg in trimmed)


def _has_profile_data(state: AgentState) -> bool:
    return bool(state.get("customer_name")) and bool(state.get("delivery_address"))


def _format_order_summary(state: AgentState) -> str:
    snapshot = get_cart_snapshot()
    items = snapshot.get("items", [])
    total = snapshot.get("total", 0.0)
    lines: list[str] = []
    if items:
        lines.append("Resumo do pedido:")
        for item in items:
            price = format_currency(item.get("price", 0.0))
            qty = item.get("quantity", 1)
            flavor = item.get("flavor", "")
            lines.append(f"- {qty}× {flavor} ({price} cada)")
        lines.append("")
        lines.append(f"Total: {format_currency(total)}")
    address = state.get("delivery_address") or "Não informado"
    lines.append("")
    lines.append(f"Endereço: {address}")
    return "\n".join(lines)


def _extract_field_with_llm(
    messages: List[BaseMessage],
    system_instruction: str,
    *,
    log_label: str = "extraction",
) -> Optional[str]:
    transcript = _summarize_recent_user_messages(messages)
    if not transcript:
        return None

    try:
        response = llm.invoke(
            [SystemMessage(content=system_instruction), HumanMessage(content=transcript)]
        )
    except Exception:
        logger.exception("LLM extraction for %s failed.", log_label)
        return None

    candidate = response.content.strip().strip('"\n ')
    logger.debug("LLM %s candidate: %s", log_label, candidate)
    if not candidate or candidate.upper().startswith("NONE"):
        return None
    return candidate


_NAME_EXTRACTION_PROMPT = (
    "Extract the customer's name using only explicit information from recent messages. "
    "If there is no clear name, reply only with 'NONE'. "
    "If a name exists, reply only with the name (up to four words), with no extra text."
)

_DELIVERY_EXTRACTION_PROMPT = (
    "You extract full addresses (street and number, complement, neighborhood, and city if "
    "present) from a conversation. Use only what the customer provided. "
    "If there is no explicit address, reply only with 'NONE'. "
    "Reply only with the address or 'NONE', with no extra text."
)


def _build_confirmation_prompt(info_stage: InfoStage, summary: str) -> str:
    if info_stage != "awaiting_confirmation":
        return f"{summary}\n\nPosso confirmar o pedido com esses itens e dados?"
    return f"{summary}\n\nPode me confirmar se está tudo certo?"


def _collect_name(state: AgentState):
    has_cart = cart_has_items()
    has_name = bool(state.get("customer_name"))
    logger.debug(
        "collect_name | stage=%s has_cart=%s has_name=%s",
        state.get("info_stage"),
        has_cart,
        has_name,
    )
    if not has_cart:
        return {}

    info_stage = state.get("info_stage", "need_name")
    current_name = state.get("customer_name")

    extracted = _extract_field_with_llm(
        state["messages"], _NAME_EXTRACTION_PROMPT, log_label="customer name"
    )
    if extracted and extracted != current_name:
        return {"customer_name": extracted, "info_stage": "idle"}

    if current_name:
        if info_stage in {"need_name", "awaiting_name"}:
            return {"info_stage": "idle"}
        return {}

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
    has_cart = cart_has_items()
    has_name = bool(state.get("customer_name"))
    has_address = bool(state.get("delivery_address"))
    logger.debug(
        "collect_address | stage=%s has_cart=%s has_name=%s has_address=%s",
        state.get("info_stage"),
        has_cart,
        has_name,
        has_address,
    )
    info_stage = state.get("info_stage", "need_name")
    if info_stage in {"need_name", "awaiting_name"}:
        return {}

    current_address = state.get("delivery_address")
    if has_cart:
        intent = _get_intent(state)
        if intent == "cart_edit" and info_stage not in {"awaiting_address", "awaiting_confirmation"}:
            return {
                "messages": [
                    AIMessage(
                        content="Sem problemas, vamos seguir editando o pedido. O que mais posso adicionar ou alterar?"
                    )
                ],
                "info_stage": "idle",
                "order_confirmed": False,
                "last_intent": intent,
            }

        extracted = _extract_field_with_llm(
            state["messages"], _DELIVERY_EXTRACTION_PROMPT, log_label="delivery address"
        )
        if extracted and extracted != current_address:
            return {
                "delivery_address": extracted,
                "info_stage": "idle",
                "order_confirmed": False,
                "last_intent": intent,
            }

    if current_address:
        if info_stage == "awaiting_address":
            return {"info_stage": "idle"}
        return {}

    if not has_cart:
        return {}

    prompt = (
        "Ótimo, agora me informe o endereço completo para entrega, por favor."
        if info_stage != "awaiting_address"
        else "Não consegui entender o endereço. Pode repetir com rua e número?"
    )
    return {
        "messages": [AIMessage(content=prompt)],
        "info_stage": "awaiting_address",
        "last_intent": state.get("last_intent"),
    }


def _name_condition(state: AgentState) -> Literal["ask", "next"]:
    decision: Literal["ask", "next"]
    has_cart = cart_has_items()
    has_name = bool(state.get("customer_name"))
    if not has_cart:
        decision = "ask"
    elif has_name:
        decision = "next"
    elif state.get("info_stage") in {"need_name", "awaiting_name"}:
        decision = "ask"
    else:
        decision = "next"
    logger.debug(
        "name_condition -> %s | stage=%s has_cart=%s has_name=%s",
        decision,
        state.get("info_stage"),
        has_cart,
        has_name,
    )
    return decision


def _address_condition(state: AgentState) -> Literal["ask", "next"]:
    decision: Literal["ask", "next"]
    if state.get("info_stage") == "awaiting_address" and not state.get(
        "delivery_address"
    ):
        decision = "ask"
    else:
        decision = "next"
    logger.debug(
        "address_condition -> %s | stage=%s has_address=%s",
        decision,
        state.get("info_stage"),
        bool(state.get("delivery_address")),
    )
    return decision


def _confirm_condition(state: AgentState) -> Literal["ask", "next"]:
    decision: Literal["ask", "next"]
    info_stage = state.get("info_stage", "need_name")
    has_cart = cart_has_items()
    has_profile = _has_profile_data(state)
    confirmed = bool(state.get("order_confirmed"))
    awaiting_info = info_stage in {"need_name", "awaiting_name", "awaiting_address"}
    if info_stage == "awaiting_confirmation":
        decision = "ask"
    elif awaiting_info:
        decision = "next"
    elif not has_cart:
        decision = "ask"
    elif confirmed:
        decision = "ask"
    elif has_profile:
        decision = "ask"
    else:
        decision = "next"
    logger.debug(
        "confirm_condition -> %s | stage=%s has_cart=%s confirmed=%s name=%s address=%s",
        decision,
        info_stage,
        has_cart,
        confirmed,
        bool(state.get("customer_name")),
        bool(state.get("delivery_address")),
    )
    return decision


def _confirm_order(state: AgentState):
    has_cart = cart_has_items()
    confirmed = bool(state.get("order_confirmed"))
    logger.debug(
        "confirm_order | stage=%s has_cart=%s confirmed=%s name=%s address=%s",
        state.get("info_stage"),
        has_cart,
        confirmed,
        bool(state.get("customer_name")),
        bool(state.get("delivery_address")),
    )
    info_stage = state.get("info_stage", "need_name")
    if info_stage in {"need_name", "awaiting_name", "awaiting_address"}:
        return {}

    if not has_cart:
        return {}

    if confirmed:
        if info_stage == "awaiting_confirmation":
            return {"info_stage": "complete"}
        return {}

    intent = _get_intent(state)
    if intent == "confirm_order":
        return {
            "order_confirmed": True,
            "info_stage": "complete",
            "messages": [
                AIMessage(
                    content="Pedido confirmado! Muito obrigado por escolher o Pastel do Mau!"
                )
            ],
            "last_intent": intent,
        }

    summary = _format_order_summary(state)
    prompt = _build_confirmation_prompt(info_stage, summary)
    return {
        "messages": [AIMessage(content=prompt)],
        "info_stage": "awaiting_confirmation",
        "last_intent": intent,
    }


def _apply_state_updates(
    state: AgentState,
    profile: dict,
    session_id: str,
) -> tuple[bool, Optional[str]]:
    new_name = state.get("customer_name")
    if new_name is not None:
        if new_name != profile.get("customer_name"):
            logger.debug("Updating profile name: %s -> %s", profile.get("customer_name"), new_name)
        profile["customer_name"] = new_name
    new_address = state.get("delivery_address")
    if new_address is not None:
        if new_address != profile.get("delivery_address"):
            logger.debug(
                "Updating profile address: %s -> %s", profile.get("delivery_address"), new_address
            )
        profile["delivery_address"] = new_address
    confirmed_now = bool(state.get("order_confirmed"))
    if confirmed_now:
        state["info_stage"] = "idle"

    order_confirmed = state.get("order_confirmed")
    if order_confirmed is not None:
        set_cart_confirmation(order_confirmed, session_id)
    new_stage = state.get("info_stage")
    if new_stage:
        if new_stage != profile.get("info_stage"):
            logger.debug("Advancing info_stage: %s -> %s", profile.get("info_stage"), new_stage)
        profile["info_stage"] = new_stage

    return confirmed_now, new_stage


tools = [
    Tool.from_function(
        name="menu",
        description=(
            "Consult the menu for flavors, ingredients, and prices. "
            "Use it before answering any pastel-related question."
        ),
        func=cypher_qa,
    ),
    Tool.from_function(
        name="add_to_cart",
        description=(
            "Add a pastel to the cart after confirming the flavor and quantity."
        ),
        func=add_to_cart_tool,
    ),
    Tool.from_function(
        name="remove_from_cart",
        description=(
            "Remove or decrease the quantity of a pastel already in the cart."
        ),
        func=remove_from_cart_tool,
    ),
    Tool.from_function(
        name="view_cart",
        description="Show the cart items and the running total.",
        func=show_cart_tool,
    ),
    Tool.from_function(
        name="clear_cart",
        description="Empty the cart when the customer wants to start over.",
        func=clear_cart_tool,
    ),
]

system_prompt = """
You are the virtual attendant for Pastel do Mau. Prioritize safety -> accuracy -> friendliness.
Always respond in Brazilian Portuguese, in 1-2 sentences, with good humor and subtle compliments.

Tool rules:
- Always consult `menu` on the first mention of flavors/ingredients/prices or when the customer asks for an item; if it fails or returns empty, say you could not find it and ask them to try another flavor.
- Use `add_to_cart` only after confirming flavor and quantity; show the total with `view_cart` after any change.
- Use `remove_from_cart` to remove items or reduce quantity when the customer asks.
- Use `clear_cart` if the customer wants to start over.

Shared state:
- When it helps the flow, ask if the customer wants to add or remove more items and offer menu details; avoid repeating this in every response. The conversation only ends when the customer confirms they want to place the order.
- In each turn, use only the strictly necessary tools (e.g., menu + add_to_cart + view_cart) and finish with a customer-facing message without more tool calls.
- Collect name/address only after the cart has items; if the customer interrupts to edit the order, return to editing and resume collection afterward.
- Greet by `customer_name` as soon as it is available (once).
- Confirm `delivery_address` when it is provided, allowing corrections.
- If something is missing, follow the normal flow; if you do not know, be honest.
"""

_system_message = SystemMessage(content=system_prompt.strip())
_llm_with_tools = llm.bind_tools(tools)


def _call_agent(state: AgentState):
    response = _llm_with_tools.invoke([_system_message, *state["messages"]])
    return {"messages": [response]}


workflow_builder = StateGraph(AgentState)
workflow_builder.add_node("collect_name", _collect_name)
workflow_builder.add_node("collect_address", _collect_address)
workflow_builder.add_node("confirm_order", _confirm_order)
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
    {"ask": END, "next": "confirm_order"},
)
workflow_builder.add_conditional_edges(
    "confirm_order",
    _confirm_condition,
    {"ask": END, "next": "agent"},
)
workflow_builder.add_conditional_edges(
    "agent", tools_condition, {"tools": "tools", "__end__": "collect_name"}
)
workflow_builder.add_edge("tools", "agent")
workflow_builder.set_entry_point("agent")

agent_workflow = workflow_builder.compile()

def generate_response(user_input: str) -> dict | str:
    """
    Generate a response for the given user input using the agent.

    Args:
        user_input (str): The input message from the user.

    Returns:
        dict | str: Structured response with reply, transition and intent, or raw text on error.
    """

    session_id = ensure_session_id()
    memory = get_memory(session_id)
    logger.debug("User message | session=%s content=%s", session_id, user_input)
    memory.add_user_message(user_input)
    profile = get_profile(session_id)

    logger.debug(
        "Incoming message | session=%s stage=%s cart_confirmed=%s has_cart_items=%s name=%s address=%s",
        session_id,
        profile.get("info_stage"),
        cart_is_confirmed(session_id),
        cart_has_items(session_id),
        bool(profile.get("customer_name")),
        bool(profile.get("delivery_address")),
    )

    graph_state: AgentState = {
        "messages": memory.messages,
        "customer_name": profile["customer_name"],
        "delivery_address": profile["delivery_address"],
        "order_confirmed": cart_is_confirmed(session_id),
        "info_stage": profile["info_stage"],
        "last_intent": None,
    }

    try:
        state = agent_workflow.invoke(graph_state)
    except Exception:
        logger.exception("LangGraph workflow failed.")
        return "Não consegui gerar uma resposta no momento."

    messages = state["messages"]
    memory.messages = list(messages)
    confirmed_now, new_stage = _apply_state_updates(state, profile, session_id)

    ai_message = _extract_last_ai_message(messages)
    if not ai_message:
        return "Não consegui gerar uma resposta no momento."

    reply_text = ai_message.content
    intent_value = state.get("last_intent") or "unknown"
    transition = "finalize" if confirmed_now else "continue"

    logger.debug(
        "Response ready | session=%s intent=%s transition=%s stage=%s cart_confirmed=%s",
        session_id,
        intent_value,
        transition,
        new_stage or profile.get("info_stage"),
        state.get("order_confirmed"),
    )

    logger.debug(
        "Assistant reply | session=%s intent=%s transition=%s content=%s",
        session_id,
        intent_value,
        transition,
        reply_text,
    )

    return {
        "reply": reply_text,
        "transition": transition,
        "intent": intent_value,
    }
