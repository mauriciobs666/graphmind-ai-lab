from __future__ import annotations

import time
import uuid
from typing import Dict, Optional, Tuple

import streamlit as st
from langchain_core.chat_history import InMemoryChatMessageHistory

from utils_common import ensure_session_log_handler, register_ttl_store, set_active_session

_memory_store: Dict[str, Tuple[InMemoryChatMessageHistory, float]] = {}
_active_session_id: Optional[str] = None

register_ttl_store("memory", _memory_store)


def ensure_session_id(explicit: Optional[str] = None) -> str:
    """
    Resolve the active session id, creating one in Streamlit session state if needed.
    """

    global _active_session_id

    if explicit:
        _active_session_id = explicit
        set_active_session(_active_session_id)
        ensure_session_log_handler(_active_session_id)
        return explicit

    if _active_session_id:
        set_active_session(_active_session_id)
        ensure_session_log_handler(_active_session_id)
        return _active_session_id

    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    _active_session_id = st.session_state.session_id
    set_active_session(_active_session_id)
    ensure_session_log_handler(_active_session_id)
    return _active_session_id


def get_memory(session_id: Optional[str] = None) -> InMemoryChatMessageHistory:
    session = ensure_session_id(session_id)
    now = time.time()
    if session not in _memory_store:
        _memory_store[session] = (InMemoryChatMessageHistory(), now)
    memory, _ = _memory_store[session]
    _memory_store[session] = (memory, now)
    return memory
