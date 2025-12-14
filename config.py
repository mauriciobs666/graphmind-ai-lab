import streamlit as st

class Config:
    @staticmethod
    def get_openai_api_key():
        return st.secrets.get("OPENAI_API_KEY")

    @staticmethod
    def get_openai_model():
        return st.secrets.get("OPENAI_MODEL")

    @staticmethod
    def get_falkordb_url():
        return st.secrets.get("FALKORDB_URL", "redis://localhost:6379")

    @staticmethod
    def get_falkordb_graph():
        return st.secrets.get("FALKORDB_GRAPH", "kg_pastel")

    @staticmethod
    def get_falkordb_credentials():
        return {
            "host": st.secrets.get("FALKORDB_HOST", "localhost"),
            "port": int(st.secrets.get("FALKORDB_PORT", 6379)),
            "username": st.secrets.get("FALKORDB_USERNAME"),
            "password": st.secrets.get("FALKORDB_PASSWORD"),
        }