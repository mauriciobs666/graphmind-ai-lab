from typing import Dict, List

from llm import llm

# Create a set of tools

from langchain_core.tools import Tool
from cypher import cypher_qa

tools = [
    Tool.from_function(
        name="cardapio",
        description="Consultar o cardapio sobre sabores, ingredientes e preços",
        func=cypher_qa,
    ),
]

# Create chat history callback

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


class InMemoryChatMessageHistory(BaseChatMessageHistory):
    def __init__(self):
        self._messages: List[BaseMessage] = []

    @property
    def messages(self) -> List[BaseMessage]:
        return self._messages

    def add_user_message(self, message):
        self._messages.append(message if isinstance(message, BaseMessage) else HumanMessage(content=message))

    def add_ai_message(self, message):
        self._messages.append(message if isinstance(message, BaseMessage) else AIMessage(content=message))

    def set_messages(self, messages: List[BaseMessage]):
        self._messages = list(messages)

    def clear(self):
        self._messages = []


_memory_store: Dict[str, InMemoryChatMessageHistory] = {}


def get_memory(session_id: str) -> BaseChatMessageHistory:
    if session_id not in _memory_store:
        _memory_store[session_id] = InMemoryChatMessageHistory()
    return _memory_store[session_id]

# Create the agent

from langchain.agents import create_agent

system_prompt = """
Você é um atendente virtual da pastelaria Pastel do Mau.
Use sempre os dados do cardapio para sugerir sabores, consultar ingredientes e preços.
Seja amigável, use um tom acolhedor e ofereça recomendações baseadas nas preferências do cliente.
Desencoraje perguntas que não estejam relacionadas aos nossos pastéis ou ingredientes disponíveis.
Sempre que o cliente pedir informações sobre sabores, ingredientes ou preços, consulte a tool cardapio antes de responder.
Se não souber a resposta, admita honestamente.
Seja sucinto e direto ao ponto em suas respostas. 
Preços devem ser precisos conforme o cardapio e não podem ser alterados.
"""

agent_graph = create_agent(
    llm,
    tools=tools,
    system_prompt=system_prompt.strip(),
)

# Create a handler to call the agent

from utils import get_session_id


def _extract_last_ai_message(messages: List[BaseMessage]) -> AIMessage | None:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return message
    return None

"""
Call the conversational agent and return a response for the UI.
"""

def generate_response(user_input: str):
    session_id = get_session_id()
    memory = get_memory(session_id)
    memory.add_user_message(user_input)

    state = agent_graph.invoke({"messages": memory.messages})

    messages = state["messages"]
    memory.set_messages(messages)

    ai_message = _extract_last_ai_message(messages)
    if not ai_message:
        return "Não consegui gerar uma resposta no momento."

    return ai_message.content
