import logging
import uuid
from typing import Any, List
import streamlit as st
from falkordb.edge import Edge
from falkordb.node import Node
from falkordb.path import Path

def stringify_value(value: Any) -> Any:
    """
    Convert FalkorDB objects to serializable Python types.
    """
    if isinstance(value, Node):
        return {"labels": value.labels, "properties": value.properties}
    if isinstance(value, Edge):
        return {
            "type": value.relation,
            "source": value.src_node.properties,
            "target": value.dest_node.properties,
            "properties": value.properties,
        }
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [stringify_value(item) for item in value]
    return value

def setup_logger(name: str) -> logging.Logger:
    """
    Set up a logger with a standard format.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger

def get_session_id():
    """
    Generate a unique session ID using UUID.
    """
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id