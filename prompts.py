from __future__ import annotations

from langchain_core.messages import SystemMessage

_INTENT_CLASSIFICATION_PROMPT = SystemMessage(
    content=(
        "Classify the customer's latest message into exactly one option: "
        "'cart_edit' (when they want to add, remove, or change items, or ask for the menu), "
        "'provide_info' (when they provide data like name or delivery address), "
        "'confirm_order' (when they confirm or approve the order, say you can send it, "
        "say everything is correct, or that you can close the order), "
        "or 'other'. Reply with only one of these words."
    )
)

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

_NAME_PROMPT_INITIAL = "Antes de continuarmos com o pedido, poderia me dizer seu nome?"
_NAME_PROMPT_RETRY = "Ainda preciso do seu nome para continuar. Como posso te chamar?"
_ADDRESS_PROMPT_INITIAL = "Ótimo, agora me informe o endereço completo para entrega, por favor."
_ADDRESS_PROMPT_RETRY = "Não consegui entender o endereço. Pode repetir com rua e número?"
_EDIT_ORDER_PROMPT = (
    "Sem problemas, vamos seguir editando o pedido. O que mais posso adicionar ou alterar?"
)
_CONFIRM_SUCCESS_MESSAGE = (
    "Pedido confirmado! Muito obrigado por escolher o Pastel do Mau!"
)

_SYSTEM_PROMPT = """
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

