import streamlit as st
from agent import generate_response

st.set_page_config("Pastel do Mau", page_icon=":cook:")

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
        st.markdown(content)

# Set up Session State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Pastel do mau, o mais legal ! Em que posso ajudar?"},
    ]

# Submit handler
def handle_submit(message):
    # Handle the response
    with st.spinner('Thinking...'):
        # Call the agent
        response = generate_response(message)
        write_message('assistant', response)


# Display messages in Session State
for message in st.session_state.messages:
    write_message(message['role'], message['content'], save=False)

# Handle any user input
if question := st.chat_input(". . ."):
    # Display user message in chat message container
    write_message('user', question)

    # Generate a response
    handle_submit(question)
