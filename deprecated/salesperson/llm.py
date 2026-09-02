import streamlit as st
from config import Config

# Create the LLM
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    openai_api_key=Config.get_openai_api_key(),
    model=Config.get_openai_model(),
)

# Create the Embedding model
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    openai_api_key=Config.get_openai_api_key()
)