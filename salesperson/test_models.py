import os
import streamlit as st
from typing import List, Dict, Optional

import urllib.request
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

st.set_page_config(page_title="Test Models Pipeline", page_icon="🧪", layout="wide")

CONFIG_FILE = os.path.join(os.path.dirname(__file__), "test_models.json")

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        except:
            pass
    return {}

def save_config(config_data):
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(config_data, f, indent=4)
    except:
        pass

def ensure_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "system_prompt" not in st.session_state:
        cfg = load_config()
        st.session_state.system_prompt = cfg.get("system_prompt", "You are a helpful AI assistant.")
    if "lm_studio_models_refresh" not in st.session_state:
        st.session_state.lm_studio_models_refresh = 0
    if "session_stats" not in st.session_state:
        st.session_state.session_stats = {"prompt_tokens": 0, "completion_tokens": 0, "total_calls": 0}

@st.cache_data(show_spinner=False)
def list_lmstudio_models(base_url: str, api_key: str, refresh_token: int) -> List[str]:
    try:
        url = base_url.strip()
        if not url.endswith("/v1"):
            url = url.rstrip("/") + "/v1"
        if not url.endswith("/models"):
            url = url + "/models"
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key.strip() or 'not-needed'}"})
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())
            items = data.get("data", [])
            return sorted({m.get("id") for m in items if m.get("id")})
    except Exception:
        return []

def sidebar_controls():
    config = load_config()
    
    with st.sidebar:
        st.title("Model Configuration")
        
        default_provider = config.get("provider", "LM Studio")
        provider_options = ["LM Studio", "OpenAI"]
        provider_index = provider_options.index(default_provider) if default_provider in provider_options else 0
        provider = st.radio("Provider", provider_options, index=provider_index)
        
        st.divider()
        settings = {"provider": provider}
        
        if provider == "LM Studio":
            settings["base_url"] = st.text_input("Base URL", value=config.get("lm_base_url", os.getenv("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1")))
            settings["api_key"] = st.text_input("API Key", value=config.get("lm_api_key", "not-needed"), type="password")
            
            if st.button("Refresh Models"):
                st.session_state.lm_studio_models_refresh += 1
            
            models = list_lmstudio_models(settings["base_url"], settings["api_key"], st.session_state.lm_studio_models_refresh)
            default_lm_model = config.get("lm_model", os.getenv("LMSTUDIO_MODEL", "lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF"))
            
            if models:
                m_idx = models.index(default_lm_model) if default_lm_model in models else 0
                settings["model"] = st.selectbox("Available Models", models, index=m_idx)
            else:
                settings["model"] = st.text_input("Model Name", value=default_lm_model)
                
        elif provider == "OpenAI":
            settings["api_key"] = st.text_input("OpenAI API Key", value=config.get("openai_api_key", os.getenv("OPENAI_API_KEY", "")), type="password")
            settings["model"] = st.text_input("Model Name", value=config.get("openai_model", "gpt-4o"))
            
        st.divider()
        settings["temperature"] = st.slider("Temperature", 0.0, 2.0, config.get("temperature", 0.7), 0.1)
        max_t = st.number_input("Max Tokens (0=default)", 0, None, config.get("max_tokens", 0), 100)
        settings["max_tokens"] = max_t if max_t > 0 else None
        
        st.subheader("System Prompt")
        st.session_state.system_prompt = st.text_area("System Prompt", st.session_state.system_prompt, height=100, label_visibility="collapsed")
        
        if st.button("Save Configuration", use_container_width=True):
            save_data = config.copy()
            save_data["provider"] = provider
            save_data["temperature"] = settings["temperature"]
            save_data["max_tokens"] = max_t
            save_data["system_prompt"] = st.session_state.system_prompt
            
            if provider == "LM Studio":
                save_data["lm_base_url"] = settings["base_url"]
                save_data["lm_api_key"] = settings["api_key"]
                save_data["lm_model"] = settings["model"]
            elif provider == "OpenAI":
                save_data["openai_api_key"] = settings["api_key"]
                save_data["openai_model"] = settings["model"]
                
            save_config(save_data)
            st.success("Configuration saved!", icon="✅")
            
        st.divider()
        
        if st.button("Clear Chat Iteration"):
            st.session_state.messages = []
            st.rerun()
            
        if st.button("Reset Session Stats"):
            st.session_state.session_stats = {"prompt_tokens": 0, "completion_tokens": 0, "total_calls": 0}
            st.rerun()

    return settings

def build_langchain_messages(system_prompt: str, messages: List[Dict[str, str]]):
    msgs = []
    if system_prompt:
        msgs.append(SystemMessage(content=system_prompt))
    for m in messages:
        if m["role"] == "user":
            msgs.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            msgs.append(AIMessage(content=m["content"]))
    return msgs

def call_langchain_client(settings, messages):
    kwargs = {
        "model": settings["model"],
        "temperature": settings["temperature"],
        "api_key": settings["api_key"] or "none",
    }
    if settings.get("base_url"):
        b_url = settings["base_url"].strip()
        if b_url and not b_url.endswith("/v1"):
            b_url = b_url.rstrip("/") + "/v1"
        kwargs["base_url"] = b_url
    
    if settings.get("max_tokens"):
        kwargs["max_tokens"] = settings["max_tokens"]
        
    chat = ChatOpenAI(**kwargs)
    lc_messages = build_langchain_messages(st.session_state.system_prompt, messages)
    
    # === DEBUG CONTEXT PASSED ===
    print("\n" + "="*40)
    print("SENDING TO LLM:")
    for m in lc_messages:
        print(f"[{m.type.upper()}]: {m.content}")
    print("="*40 + "\n")
    # ============================
    
    try:
        resp = chat.invoke(lc_messages)
        text = resp.content
        
        usage = resp.response_metadata.get("token_usage", {})
        p_tok = usage.get("prompt_tokens", 0)
        c_tok = usage.get("completion_tokens", 0)
        
        return str(text), p_tok, c_tok
    except Exception as e:
        return f"Error: {str(e)}", 0, 0

def display_stats_panel():
    with st.sidebar.expander("📊 Session Statistics", expanded=False):
        stats = st.session_state.session_stats
        total_tokens = stats['prompt_tokens'] + stats['completion_tokens']
        
        st.metric("Total API Calls", stats['total_calls'])
        st.metric("Total Tokens", total_tokens)
        st.metric("Prompt Tokens (Read)", stats['prompt_tokens'])
        st.metric("Completion Tokens (Write)", stats['completion_tokens'])
        
        history_ctx = len(st.session_state.messages)
        st.metric("Context Size (Messages)", history_ctx)
        
    with st.sidebar.expander("🔍 Debug: Messages Sent", expanded=False):
        st.json(st.session_state.messages)

def main():
    ensure_state()
    settings = sidebar_controls()
    
    display_stats_panel()

    st.header(f"Testing {settings['provider']}")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    user_input = st.chat_input("Send a message...")
    if user_input:
        msg_payload = {"role": "user", "content": user_input}
        st.session_state.messages.append(msg_payload)
        with st.chat_message("user"):
            st.markdown(user_input)
            
        with st.spinner("Generating..."):
            reply, p_tok, c_tok = call_langchain_client(settings, st.session_state.messages)
            
        with st.chat_message("assistant"):
            st.markdown(reply)
        
        st.session_state.messages.append({"role": "assistant", "content": reply})
        
        # Up stats
        if not reply.startswith("Error:"):
            st.session_state.session_stats["total_calls"] += 1
            st.session_state.session_stats["prompt_tokens"] += p_tok
            st.session_state.session_stats["completion_tokens"] += c_tok
            # Trigger a redraw so stats panel updates immediately
            st.rerun()

if __name__ == "__main__":
    main()
