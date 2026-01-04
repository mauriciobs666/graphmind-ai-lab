from __future__ import annotations

import uuid
from typing import Dict, Optional

import streamlit as st
from langchain_core.chat_history import InMemoryChatMessageHistory

_memory_store: Dict[str, InMemoryChatMessageHistory] = {}
_active_session_id: Optional[str] = None


def ensure_session_id(explicit: Optional[str] = None) -> str:
    """
    Resolve the active session id, creating one in Streamlit session state if needed.
    """

    global _active_session_id

    if explicit:
        _active_session_id = explicit
        return explicit

    if _active_session_id:
        return _active_session_id

    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    _active_session_id = st.session_state.session_id
    return _active_session_id


def get_memory(session_id: Optional[str] = None) -> InMemoryChatMessageHistory:
    session = ensure_session_id(session_id)
    if session not in _memory_store:
        _memory_store[session] = InMemoryChatMessageHistory()
    return _memory_store[session]
