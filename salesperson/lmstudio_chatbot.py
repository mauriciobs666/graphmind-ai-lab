import os
from typing import List, Dict, Optional

import streamlit as st
from openai import OpenAI


def get_client(base_url: str, api_key: str) -> OpenAI:
    """Create an OpenAI client pointed at LM Studio's OpenAI-compatible API."""
    # LM Studio typically uses http://localhost:1234/v1 and any API key string
    return OpenAI(base_url=base_url.strip(), api_key=api_key.strip() or "not-needed")


def ensure_state():
    if "messages" not in st.session_state:
        st.session_state.messages: List[Dict[str, str]] = []
    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = (
            "You are a helpful AI assistant. Be concise and accurate."
        )
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = os.getenv(
            "LMSTUDIO_MODEL", "lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF"
        )
    if "models_refresh_token" not in st.session_state:
        st.session_state.models_refresh_token = 0


@st.cache_data(show_spinner=False)
def list_models_cached(base_url: str, api_key: str, refresh_token: int) -> List[str]:
    """List available models from LM Studio via the OpenAI-compatible /v1/models.

    The refresh_token is included to allow users to force refresh via a button.
    """
    client = get_client(base_url, api_key)
    try:
        resp = client.models.list()
        items = getattr(resp, "data", resp)
        model_ids: List[str] = []
        for m in items:
            mid: Optional[str] = getattr(m, "id", None)
            if mid is None and isinstance(m, dict):
                mid = m.get("id")
            if mid:
                model_ids.append(mid)
        # Unique + sorted for stable UI
        return sorted(set(model_ids))
    except Exception:
        return []


def sidebar_controls():
    with st.sidebar:
        st.title("LM Studio Chat")

        default_base_url = os.getenv("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
        default_api_key = os.getenv("LMSTUDIO_API_KEY", "not-needed")
        base_url = st.text_input("Base URL", value=default_base_url)
        api_key = st.text_input("API Key", value=default_api_key, type="password")

        st.caption(
            "LM Studio exposes an OpenAI-compatible API. Typically use "
            "`http://localhost:1234/v1` and any API key."
        )

        # Model settings (auto-list with fallback to manual input)
        cols = st.columns([3, 1])
        with cols[0]:
            st.subheader("Model", anchor=False)
        with cols[1]:
            if st.button("Refresh", help="Reload available models from LM Studio"):
                st.session_state.models_refresh_token += 1

        models = list_models_cached(base_url, api_key, st.session_state.models_refresh_token)

        if models:
            # Try to preserve previous selection if still available
            current = st.session_state.selected_model
            index = models.index(current) if current in models else 0
            selected = st.selectbox(
                "Available models",
                options=models,
                index=index,
                help="Models reported by LM Studio at /v1/models",
                key="model_select",
            )
            st.session_state.selected_model = selected
            model = selected
        else:
            st.info(
                "Could not list models from LM Studio. Enter the model name manually."
            )
            model = st.text_input(
                "Model name",
                value=st.session_state.selected_model,
                help="Enter the exact model name shown in LM Studio.",
                key="model_text_input",
            )
            st.session_state.selected_model = model

        temperature = st.slider("Temperature", min_value=0.0, max_value=1.5, value=0.7, step=0.1)
        max_tokens = st.number_input("Max tokens (0 = model default)", min_value=0, value=0, step=50)
        stream = st.toggle("Stream responses", value=True)

        st.markdown("---")
        st.subheader("System Prompt")
        st.session_state.system_prompt = st.text_area(
            "", value=st.session_state.system_prompt, height=120, label_visibility="collapsed"
        )

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear Chat"):
                st.session_state.messages = []
                st.rerun()
        with col2:
            st.caption("Chat history is only kept in this session.")

    return {
        "base_url": base_url,
        "api_key": api_key,
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream,
    }


def render_chat_history():
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"]) 


def build_messages_for_api(system_prompt: str, user_and_assistant_messages: List[Dict[str, str]]):
    messages: List[Dict[str, str]] = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    messages.extend(user_and_assistant_messages)
    return messages


def chat_completion(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    stream: bool = True,
):
    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
    }
    if max_tokens and max_tokens > 0:
        kwargs["max_tokens"] = int(max_tokens)

    return client.chat.completions.create(stream=stream, **kwargs)


def main():
    st.set_page_config(page_title="LM Studio Chat", page_icon="💬", layout="centered")
    ensure_state()
    settings = sidebar_controls()

    # Chat UI
    st.header("Chat with Local LM Studio")
    st.caption(
        "Powered by LM Studio via OpenAI-compatible API. Messages stay local to your machine."
    )

    render_chat_history()

    user_input = st.chat_input("Ask anything…")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Prepare API call
        messages_for_api = build_messages_for_api(
            st.session_state.system_prompt, st.session_state.messages
        )

        try:
            client = get_client(settings["base_url"], settings["api_key"])

            with st.chat_message("assistant"):
                if settings["stream"]:
                    # Stream tokens incrementally
                    placeholder = st.empty()
                    full = ""
                    stream_resp = chat_completion(
                        client,
                        settings["model"],
                        messages_for_api,
                        settings["temperature"],
                        settings["max_tokens"],
                        stream=True,
                    )
                    for chunk in stream_resp:
                        try:
                            delta = chunk.choices[0].delta.content or ""
                        except Exception:
                            delta = ""
                        if delta:
                            full += delta
                            placeholder.markdown(full)

                    st.session_state.messages.append({"role": "assistant", "content": full})
                else:
                    resp = chat_completion(
                        client,
                        settings["model"],
                        messages_for_api,
                        settings["temperature"],
                        settings["max_tokens"],
                        stream=False,
                    )
                    text = resp.choices[0].message.content
                    st.markdown(text)
                    st.session_state.messages.append({"role": "assistant", "content": text})

        except Exception as e:
            st.error(
                "Failed to reach LM Studio or generate a response.\n"
                f"Error: {e}\n\n"
                "Troubleshooting: Ensure LM Studio server is running with the OpenAI API enabled, "
                "the Base URL is correct (e.g., http://localhost:1234/v1), and the model name matches."
            )


if __name__ == "__main__":
    main()
