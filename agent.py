from __future__ import annotations

from typing import Annotated, List, Optional, TypedDict, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import Tool
from langgraph.graph import END, StateGraph, add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from cart import (
    CartItem,
    add_to_cart_tool,
    cart_has_items,
    cart_is_confirmed,
    clear_cart_tool,
    get_cart_snapshot,
    remove_from_cart_tool,
    set_cart_confirmation,
    show_cart_tool,
)
from customer_profile import (
    CustomerProfile,
    get_customer_profile,
    get_profile,
    is_order_ready,
    reset_customer_profile,
)
from cypher import cypher_qa
from llm import llm
from session_manager import ensure_session_id, get_memory
from utils_common import setup_logger

InfoStage = Literal[
    "need_name",
    "awaiting_name",
    "idle",
    "awaiting_address",
    "awaiting_payment",
    "awaiting_confirmation",
    "complete",
]


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    customer_name: Optional[str]
    delivery_address: Optional[str]
    payment_method: Optional[str]
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
            "Classifique a última mensagem do cliente em apenas uma opção: "
            "'cart_edit' (quando quer adicionar, remover ou alterar itens, ou pedir o cardápio), "
            "'provide_info' (quando está passando dados como nome, endereço ou forma de pagamento), "
            "'confirm_order' (quando confirma ou "
            " aprova o pedido, diz que pode enviar, que está tudo certo, que pode fechar), "
            "ou 'other'. Responda somente com uma dessas palavras."
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
    return "\n".join(f"Cliente: {msg}" for msg in trimmed)


def _format_currency(value: float) -> str:
    return f"R${value:.2f}".replace(".", ",")


def _format_order_summary(state: AgentState) -> str:
    snapshot = get_cart_snapshot()
    items = snapshot.get("items", [])
    total = snapshot.get("total", 0.0)
    lines: list[str] = []
    if items:
        lines.append("Resumo do pedido:")
        for item in items:
            preco = _format_currency(item.get("preco", 0.0))
            qty = item.get("quantidade", 1)
            sabor = item.get("sabor", "")
            lines.append(f"- {qty}× {sabor} ({preco} cada)")
        lines.append(f"Total: {_format_currency(total)}")
    address = state.get("delivery_address") or "Não informado"
    payment = state.get("payment_method") or "Não informado"
    lines.append(f"Endereço: {address}")
    lines.append(f"Pagamento: {payment}")
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
    "Extraia o nome do cliente usando somente informações explícitas das mensagens "
    "recentes. Caso não exista um nome claro, responda apenas com 'NONE'. "
    "Se existir, responda apenas com o nome (até quatro palavras), sem texto extra."
)

_DELIVERY_EXTRACTION_PROMPT = (
    "Você extrai endereços completos (rua e número, complemento, bairro e cidade se "
    "existir) a partir de uma conversa. Use somente o que o cliente informou. "
    "Se não houver endereço explícito, responda apenas com 'NONE'. "
    "Responda apenas com o endereço ou 'NONE', sem texto extra."
)

_PAYMENT_EXTRACTION_PROMPT = (
    "Identifique a forma de pagamento preferida do cliente na conversa. "
    "Considere PIX, cartão (crédito/débito/maquininha) ou dinheiro. "
    "Se encontrar, responda exatamente com 'PIX', 'Cartão' ou 'Dinheiro'. "
    "Se não existir essa informação, responda apenas com 'NONE'."
)


def _build_confirmation_prompt(info_stage: InfoStage, summary: str) -> str:
    if info_stage != "awaiting_confirmation":
        return f"{summary}\nPosso confirmar o pedido com esses itens e dados?"
    return f"{summary}\nPode me confirmar se está tudo certo?"


def _collect_name(state: AgentState):
    if not cart_has_items():
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
    info_stage = state.get("info_stage", "need_name")
    if info_stage in {"need_name", "awaiting_name"}:
        return {}

    current_address = state.get("delivery_address")
    if cart_has_items():
        intent = _get_intent(state)
        if intent == "cart_edit" and info_stage not in {
            "awaiting_address",
            "awaiting_payment",
            "awaiting_confirmation",
        }:
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

    if not cart_has_items():
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


def _collect_payment(state: AgentState):
    info_stage = state.get("info_stage", "need_name")
    if info_stage in {"need_name", "awaiting_name", "awaiting_address"}:
        return {}

    if not state.get("delivery_address"):
        return {}

    current_payment = state.get("payment_method")
    if cart_has_items():
        intent = _get_intent(state)
        if intent == "cart_edit" and info_stage not in {"awaiting_payment", "awaiting_confirmation"}:
            return {
                "messages": [
                    AIMessage(
                        content="Claro, voltamos para o carrinho. Qual pastel ou alteração você quer?"
                    )
                ],
                "info_stage": "idle",
                "order_confirmed": False,
                "last_intent": intent,
            }

        extracted = _extract_field_with_llm(
            state["messages"], _PAYMENT_EXTRACTION_PROMPT, log_label="payment method"
        )
        if extracted and extracted != current_payment:
            return {
                "payment_method": extracted,
                "info_stage": "awaiting_confirmation",
                "order_confirmed": False,
                "last_intent": intent,
            }

    if current_payment:
        if info_stage == "awaiting_payment":
            return {"info_stage": "awaiting_confirmation", "order_confirmed": False}
        return {}

    if not cart_has_items():
        return {}

    prompt = (
        "Qual forma de pagamento você prefere (PIX, cartão na entrega ou dinheiro)?"
        if info_stage != "awaiting_payment"
        else "Pode confirmar a forma de pagamento (PIX, cartão ou dinheiro)?"
    )
    return {
        "messages": [AIMessage(content=prompt)],
        "info_stage": "awaiting_payment",
        "last_intent": state.get("last_intent"),
    }


def _name_condition(state: AgentState) -> Literal["ask", "next"]:
    if not cart_has_items():
        return "ask"
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


def _confirm_condition(state: AgentState) -> Literal["ask", "next"]:
    info_stage = state.get("info_stage", "need_name")
    if info_stage == "awaiting_confirmation":
        return "ask"
    if info_stage in {"need_name", "awaiting_name", "awaiting_address", "awaiting_payment"}:
        return "next"
    if not cart_has_items():
        return "ask"
    if state.get("order_confirmed"):
        return "ask"
    if (
        state.get("customer_name")
        and state.get("delivery_address")
        and state.get("payment_method")
    ):
        return "ask"
    return "next"


def _confirm_order(state: AgentState):
    info_stage = state.get("info_stage", "need_name")
    if info_stage in {"need_name", "awaiting_name", "awaiting_address", "awaiting_payment"}:
        return {}

    if not cart_has_items():
        return {}

    if state.get("order_confirmed"):
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
                    content="Pedido confirmado! Vou separar tudo com carinho. Precisa de algo mais?"
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
        name="remover_carrinho",
        description=(
            "Remove ou diminui a quantidade de um pastel existente no carrinho."
        ),
        func=remove_from_cart_tool,
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
Voce e o atendente virtual do Pastel do Mau. Priorize seguranca -> precisao -> simpatia. Fale sempre em portugues do Brasil, em 1-2 frases, com bom humor e elogios sutis.

Regras de ferramentas:
- Sempre consulte `cardapio` na primeira mencao a sabores/ingredientes/precos ou quando o cliente pedir um item; se falhar ou vier vazio, diga que nao achou e peca para tentar outro sabor.
- `adicionar_carrinho` so apos confirmar sabor e quantidade; mostre o total com `ver_carrinho` depois de qualquer mudanca.
- `remover_carrinho` para tirar itens ou diminuir quantidade quando o cliente pedir.
- Use `limpar_carrinho` se o cliente quiser recomecar.

Estado compartilhado:
- Em toda resposta, pergunte se o cliente quer incluir ou remover mais itens e ofereca trazer detalhes sobre sabores/ingredientes/precos do cardapio. A conversa so termina quando o cliente confirmar que quer fechar o pedido.
- Em cada turno, use no maximo as ferramentas estritamente necessarias (ex.: cardapio + adicionar_carrinho + ver_carrinho) e finalize com uma mensagem ao cliente sem novas chamadas de ferramenta.
- Nome/endereco/pagamento so sao coletados depois que o carrinho tiver itens; se o cliente interromper para editar o pedido, volte a editar e retome a coleta depois.
- Cumprimente pelo `customer_name` assim que disponivel (uma vez).
- Confirme `delivery_address` e `payment_method` quando forem informados, permitindo correcoes.
- Se algo faltar, siga o fluxo normal; caso nao saiba, diga honestamente.
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
    {"ask": END, "next": "collect_payment"},
)
workflow_builder.add_conditional_edges(
    "collect_payment",
    _payment_condition,
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
    memory.add_user_message(user_input)
    profile = get_profile(session_id)

    logger.debug(
        "Incoming message | session=%s stage=%s cart_confirmed=%s has_cart_items=%s name=%s address=%s payment=%s",
        session_id,
        profile.get("info_stage"),
        cart_is_confirmed(session_id),
        cart_has_items(session_id),
        bool(profile.get("customer_name")),
        bool(profile.get("delivery_address")),
        bool(profile.get("payment_method")),
    )

    graph_state: AgentState = {
        "messages": memory.messages,
        "customer_name": profile["customer_name"],
        "delivery_address": profile["delivery_address"],
        "payment_method": profile["payment_method"],
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
    new_payment = state.get("payment_method")
    if new_payment is not None:
        if new_payment != profile.get("payment_method"):
            logger.debug(
                "Updating profile payment: %s -> %s", profile.get("payment_method"), new_payment
            )
        profile["payment_method"] = new_payment
    order_confirmed = state.get("order_confirmed")
    if order_confirmed is not None:
        set_cart_confirmation(order_confirmed, session_id)
    new_stage = state.get("info_stage")
    if new_stage:
        if new_stage != profile.get("info_stage"):
            logger.debug("Advancing info_stage: %s -> %s", profile.get("info_stage"), new_stage)
        profile["info_stage"] = new_stage

    ai_message = _extract_last_ai_message(messages)
    if not ai_message:
        return "Não consegui gerar uma resposta no momento."

    reply_text = ai_message.content
    intent_value = state.get("last_intent") or "unknown"
    transition = "finalize" if state.get("order_confirmed") else "continue"

    logger.debug(
        "Response ready | session=%s intent=%s transition=%s stage=%s cart_confirmed=%s",
        session_id,
        intent_value,
        transition,
        new_stage or profile.get("info_stage"),
        state.get("order_confirmed"),
    )

    return {
        "reply": reply_text,
        "transition": transition,
        "intent": intent_value,
    }
