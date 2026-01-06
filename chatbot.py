import streamlit as st
from agent import (
    generate_response,
    get_cart_snapshot,
    get_customer_profile,
    is_order_ready,
)

st.set_page_config("Pastel do Mau", page_icon=":cook:")

def format_currency(value: float) -> str:
    return f"R${value:.2f}".replace(".", ",")

def write_message(role, content, save=True):
    """
    This is a helper function that saves a message to the
    session state and then writes a message to the UI
    """
    # Append to session state
    if save:
        st.session_state.messages.append({"role": role, "content": content})

    # Write to UI
    with st.chat_message(role):
        safe_content = content.replace("$", "\\$")
        st.markdown(safe_content)

# Set up Session State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Bem vindo a loja virtual do Pastel do Mau ! Qual o seu nome? Em que posso ajudar?"},
    ]

def render_sidebar():
    snapshot = get_cart_snapshot()
    profile = get_customer_profile()
    ready = is_order_ready()
    with st.sidebar:
        st.subheader("Cliente")
        st.write(f"Nome: {profile['customer_name'] or '—'}")
        st.write(f"Endereço: {profile['delivery_address'] or '—'}")
        if ready:
            st.markdown(":green-background[Pedido confirmado]")
        st.divider()
        st.subheader("Carrinho")
        items = snapshot["items"]
        if not items:
            st.write("Carrinho vazio.")
        else:
            for item in items:
                st.write(
                    f"{item['quantidade']}× {item['sabor']} — "
                    f"{format_currency(item['preco'])}"
                )
            st.markdown(f"**Total: {format_currency(snapshot['total'])}**")

# Submit handler
def handle_submit(message):
    # Handle the response
    with st.spinner('Pensando...'):
        # Call the agent
        response = generate_response(message)
        reply_text = response.get("reply") if isinstance(response, dict) else response
        write_message('assistant', reply_text)


# Display messages in Session State
for message in st.session_state.messages:
    write_message(message['role'], message['content'], save=False)

# Handle any user input
if question := st.chat_input(". . ."):
    # Display user message in chat message container
    write_message('user', question)

    # Generate a response
    handle_submit(question)

render_sidebar()
