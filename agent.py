from __future__ import annotations

import json
from typing import Annotated, Dict, List, Optional, TypedDict, Literal

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
from prompts import (
    _ADDRESS_PROMPT_INITIAL,
    _ADDRESS_PROMPT_RETRY,
    _CONFIRM_SUCCESS_MESSAGE,
    _DELIVERY_EXTRACTION_PROMPT,
    _EDIT_ORDER_PROMPT,
    _INTENT_CLASSIFICATION_PROMPT,
    _NAME_EXTRACTION_PROMPT,
    _NAME_PROMPT_INITIAL,
    _NAME_PROMPT_RETRY,
    _SYSTEM_PROMPT,
)

InfoStage = Literal[
    "need_name",
    "awaiting_name",
    "idle",
    "awaiting_address",
    "awaiting_confirmation",
    "complete",
]

_NAME_STAGES = {"need_name", "awaiting_name"}
_AWAITING_INFO_STAGES = {"need_name", "awaiting_name", "awaiting_address"}


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    customer_name: Optional[str]
    delivery_address: Optional[str]
    order_confirmed: bool
    info_stage: InfoStage
    last_intent: Optional[str]
    intent_flags: Optional[Dict[str, bool]]

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


def _classify_intent_with_llm(messages: List[BaseMessage]) -> Dict[str, bool]:
    """
    Let the LLM mark which intents are present in the user's latest message.
    """

    last_user = _extract_last_user_message(messages)
    if not last_user:
        return {"cart_edit": False, "provide_info": False, "confirm_order": False, "other": True}

    try:
        response = llm.invoke([_INTENT_CLASSIFICATION_PROMPT, last_user])
        payload = json.loads(response.content.strip())
        flags = {
            "cart_edit": bool(payload.get("cart_edit")),
            "provide_info": bool(payload.get("provide_info")),
            "confirm_order": bool(payload.get("confirm_order")),
            "other": bool(payload.get("other")),
        }
        logger.debug("Intent flags classified as: %s", flags)
        return flags
    except Exception:
        logger.exception("LLM intent classification failed.")
    return {"cart_edit": False, "provide_info": False, "confirm_order": False, "other": True}


def _primary_intent_from_flags(flags: Dict[str, bool]) -> str:
    if flags.get("cart_edit"):
        return "cart_edit"
    if flags.get("confirm_order"):
        return "confirm_order"
    if flags.get("provide_info"):
        return "provide_info"
    return "other"


def _get_intent_flags(state: AgentState) -> Dict[str, bool]:
    cached = state.get("intent_flags")
    if cached:
        logger.debug("Using cached intent flags: %s", cached)
        return cached
    flags = _classify_intent_with_llm(state["messages"])
    state["intent_flags"] = flags
    state["last_intent"] = _primary_intent_from_flags(flags)
    logger.debug("Intent flags cached as: %s", flags)
    return flags


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


def _build_confirmation_prompt(info_stage: InfoStage, summary: str) -> str:
    if info_stage != "awaiting_confirmation":
        return f"{summary}\n\nPosso confirmar o pedido com esses itens e dados?"
    return f"{summary}\n\nPode me confirmar se está tudo certo?"


def _is_collecting_name(info_stage: InfoStage) -> bool:
    return info_stage in _NAME_STAGES


def _is_awaiting_profile_info(info_stage: InfoStage) -> bool:
    return info_stage in _AWAITING_INFO_STAGES


def _collect_name(state: AgentState):
    has_cart = cart_has_items()
    current_name = state.get("customer_name")
    has_name = bool(current_name)
    logger.debug(
        "collect_name | stage=%s has_cart=%s has_name=%s",
        state.get("info_stage"),
        has_cart,
        has_name,
    )
    if not has_cart:
        return {}

    info_stage = state.get("info_stage", "need_name")

    extracted = _extract_field_with_llm(
        state["messages"], _NAME_EXTRACTION_PROMPT, log_label="customer name"
    )
    if extracted and extracted != current_name:
        return {"customer_name": extracted, "info_stage": "idle"}

    if current_name:
        if _is_collecting_name(info_stage):
            return {"info_stage": "idle"}
        return {}

    prompt = (
        _NAME_PROMPT_INITIAL if info_stage == "need_name" else _NAME_PROMPT_RETRY
    )
    return {
        "messages": [AIMessage(content=prompt)],
        "info_stage": "awaiting_name",
    }


def _collect_address(state: AgentState):
    has_cart = cart_has_items()
    logger.debug(
        "collect_address | stage=%s has_cart=%s has_name=%s has_address=%s",
        state.get("info_stage"),
        has_cart,
        bool(state.get("customer_name")),
        bool(state.get("delivery_address")),
    )
    info_stage = state.get("info_stage", "need_name")
    if _is_collecting_name(info_stage):
        return {}

    current_address = state.get("delivery_address")
    if has_cart:
        intent_flags = _get_intent_flags(state)
        primary_intent = _primary_intent_from_flags(intent_flags)
        updates: Dict[str, object] = {"last_intent": primary_intent}

        extracted = _extract_field_with_llm(
            state["messages"], _DELIVERY_EXTRACTION_PROMPT, log_label="delivery address"
        )
        if extracted and extracted != current_address:
            updates.update(
                {
                    "delivery_address": extracted,
                    "info_stage": "idle",
                    "order_confirmed": False,
                }
            )

        if intent_flags.get("cart_edit") and info_stage not in {
            "awaiting_address",
            "awaiting_confirmation",
        }:
            updates.update(
                {
                    "messages": [AIMessage(content=_EDIT_ORDER_PROMPT)],
                    "info_stage": "idle",
                    "order_confirmed": False,
                }
            )
            return updates

        if "delivery_address" in updates:
            return updates

    if current_address:
        if info_stage == "awaiting_address":
            return {"info_stage": "idle"}
        return {}

    if not has_cart:
        return {}

    prompt = (
        _ADDRESS_PROMPT_INITIAL if info_stage != "awaiting_address" else _ADDRESS_PROMPT_RETRY
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
    needs_name = _is_collecting_name(state.get("info_stage", "need_name"))
    if not has_cart or (not has_name and needs_name):
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
    intent_flags = state.get("intent_flags") or {}
    has_cart = cart_has_items()
    has_profile = _has_profile_data(state)
    confirmed = bool(state.get("order_confirmed"))
    awaiting_info = _is_awaiting_profile_info(info_stage)
    if intent_flags.get("cart_edit"):
        decision = "next"
    elif info_stage == "awaiting_confirmation":
        decision = "ask"
    elif awaiting_info:
        decision = "next"
    else:
        should_ask = (not has_cart) or confirmed or has_profile
        decision = "ask" if should_ask else "next"
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
    if _is_awaiting_profile_info(info_stage):
        return {}

    if not has_cart:
        return {}

    if confirmed:
        if info_stage == "awaiting_confirmation":
            return {"info_stage": "complete"}
        return {}

    intent_flags = _get_intent_flags(state)
    primary_intent = _primary_intent_from_flags(intent_flags)
    if intent_flags.get("confirm_order") and not intent_flags.get("cart_edit"):
        return {
            "order_confirmed": True,
            "info_stage": "complete",
            "messages": [
                AIMessage(
                    content=_CONFIRM_SUCCESS_MESSAGE
                )
            ],
            "last_intent": primary_intent,
        }

    summary = _format_order_summary(state)
    prompt = _build_confirmation_prompt(info_stage, summary)
    return {
        "messages": [AIMessage(content=prompt)],
        "info_stage": "awaiting_confirmation",
        "last_intent": primary_intent,
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
    order_confirmed = state.get("order_confirmed")
    if order_confirmed is not None:
        set_cart_confirmation(order_confirmed, session_id)
    new_stage = state.get("info_stage")
    if new_stage:
        if new_stage != profile.get("info_stage"):
            logger.debug("Advancing info_stage: %s -> %s", profile.get("info_stage"), new_stage)
        profile["info_stage"] = new_stage
    last_intent = state.get("last_intent")
    if last_intent is not None:
        profile["last_intent"] = last_intent

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

_system_message = SystemMessage(content=_SYSTEM_PROMPT.strip())
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
        "intent_flags": None,
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
