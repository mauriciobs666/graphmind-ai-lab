import streamlit as st
import uuid

def get_session_id():
    """
    Generate a unique session ID using UUID.
    """
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id
