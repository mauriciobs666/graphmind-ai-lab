import os

import streamlit as st

class Config:
    @staticmethod
    def _get_value(key: str, default=None):
        value = st.secrets.get(key, None)
        if value is None:
            return os.getenv(key, default)
        return value

    @staticmethod
    def get_openai_api_key():
        return Config._get_value("OPENAI_API_KEY")

    @staticmethod
    def get_openai_model():
        return Config._get_value("OPENAI_MODEL")

    @staticmethod
    def get_falkordb_url():
        return Config._get_value("FALKORDB_URL", "redis://localhost:6379")

    @staticmethod
    def get_falkordb_graph():
        return Config._get_value("FALKORDB_GRAPH", "kg_pastel")

    @staticmethod
    def get_falkordb_credentials():
        return {
            "host": Config._get_value("FALKORDB_HOST", "localhost"),
            "port": int(Config._get_value("FALKORDB_PORT", 6379)),
            "username": Config._get_value("FALKORDB_USERNAME"),
            "password": Config._get_value("FALKORDB_PASSWORD"),
        }

    @staticmethod
    def get_log_level() -> str:
        return Config._get_value("LOG_LEVEL", "DEBUG")
