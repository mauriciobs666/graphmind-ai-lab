from __future__ import annotations

from typing import Dict, Optional

from langchain_core.chat_history import InMemoryChatMessageHistory

from utils_common import get_session_id

_memory_store: Dict[str, InMemoryChatMessageHistory] = {}
_active_session_id: Optional[str] = None


def ensure_session_id(explicit: Optional[str] = None) -> str:
    """
    Resolve the active session id without touching Streamlit in worker threads.
    """

    global _active_session_id

    if explicit:
        _active_session_id = explicit
        return explicit

    if _active_session_id:
        return _active_session_id

    session_id = get_session_id()
    _active_session_id = session_id
    return session_id


def get_memory(session_id: Optional[str] = None) -> InMemoryChatMessageHistory:
    session = ensure_session_id(session_id)
    if session not in _memory_store:
        _memory_store[session] = InMemoryChatMessageHistory()
    return _memory_store[session]
