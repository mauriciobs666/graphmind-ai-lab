import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from falkordb.edge import Edge
from falkordb.node import Node
from falkordb.path import Path
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from llm import llm
from graph import get_schema_description, graph

logger = logging.getLogger("tools.cypher")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

CYPHER_GENERATION_TEMPLATE = """
Você é um desenvolvedor especialista em FalkorDB.
Gere apenas uma consulta Cypher para responder a pergunta do usuário
seguindo o esquema descrito abaixo.

Esquema:
{schema}

Regras:
- Utilize somente os labels, relacionamentos e propriedades mostrados no esquema.
- Gere apenas consultas de leitura (MATCH/RETURN).
- Sempre use aliases descritivos com `AS` para cada coluna retornada. Ex: `RETURN p.sabor AS sabor, p.preco AS preco`.
- Não inclua nenhum texto explicativo, apenas a consulta pura.
- Quando precisar comparar textos, utilize sempre comparações case-insensitive com toLower(), por exemplo: `WHERE toLower(i.nome) = toLower("queijo")`.
- SEMPRE que for comparar textos, utilize CONTAINS, por exemplo: `WHERE toLower(i.nome) CONTAINS toLower("queijo")`.
- Retorne sempre explicitamente cada uma de todas as propriedades dos nós e relacionamentos no RETURN.

Pergunta:
{question}
"""

ANSWER_TEMPLATE = """
Você recebeu uma pergunta sobre o grafo de conhecimento.
Use a consulta Cypher e os resultados devolvidos para responder de forma clara.
Se a consulta não trouxe registros, diga que não encontrou informação suficiente.

Pergunta: {question}
Consulta Cypher: {cypher_query}
Resultados: {context}
"""

cypher_prompt = ChatPromptTemplate.from_template(CYPHER_GENERATION_TEMPLATE)
cypher_chain = cypher_prompt | llm | StrOutputParser()

answer_prompt = ChatPromptTemplate.from_template(ANSWER_TEMPLATE)
answer_chain = answer_prompt | llm | StrOutputParser()


def _extract_cypher(text: str) -> str:
    if not text:
        logger.debug("LLM returned empty response for Cypher generation.")
        return ""
    fenced_blocks = re.findall(r"```(?:cypher)?(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced_blocks:
        query = fenced_blocks[-1].strip()
        logger.debug("Extrai bloco Cypher explícito:\n%s", query)
        return query
    query = text.strip()
    logger.debug("Usando resposta completa como Cypher:\n%s", query)
    return query


def _stringify_value(value: Any) -> Any:
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
        return [_stringify_value(item) for item in value]
    return value


def _normalize_header(header: Any, idx: int) -> str:
    if isinstance(header, (list, tuple)):
        header = header[0]
    if isinstance(header, bytes):
        header = header.decode()
    header = str(header) if header is not None else ""
    header = header.strip()
    if not header or header.isdigit():
        return f"col_{idx}"
    if "." in header:
        return header.split(".")[-1]
    return header


def _format_rows(result) -> List[Dict[str, Any]]:
    headers = [_normalize_header(column, idx) for idx, column in enumerate(result.header)]
    rows = []
    for row in result.result_set:
        record = {}
        for idx, header in enumerate(headers):
            record[header] = _stringify_value(row[idx])
        rows.append(record)
    return rows


def _execute_cypher(query: str) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    try:
        logger.debug("Executando Cypher:\n%s", query)
        query_result = graph.ro_query(query)
    except Exception as exc:
        logger.exception("Erro ao executar Cypher.")
        return [], f"Não consegui executar a consulta Cypher: {exc}"
    rows = _format_rows(query_result)
    logger.debug("Consulta retornou %d linha(s).", len(rows))
    return rows, None


def cypher_qa(question: str) -> str:
    logger.info("Pergunta recebida: %s", question)
    schema = get_schema_description()
    logger.debug("Schema resumido:\n%s", schema)
    cypher_suggestion = cypher_chain.invoke({"schema": schema, "question": question})
    logger.debug("Resposta completa do gerador de Cypher:\n%s", cypher_suggestion)
    cypher_query = _extract_cypher(cypher_suggestion)

    if not cypher_query:
        logger.warning("Falha ao gerar consulta Cypher para a pergunta.")
        return "Não consegui gerar uma consulta Cypher para essa pergunta."

    rows, error = _execute_cypher(cypher_query)
    if error:
        logger.error("Erro executando consulta: %s", error)
        return error

    context = json.dumps(rows, ensure_ascii=False, indent=2) if rows else "Nenhum registro encontrado."
    logger.debug("Contexto repassado ao LLM:\n%s", context)

    answer = answer_chain.invoke(
        {"question": question, "cypher_query": cypher_query, "context": context}
    )
    logger.info("Resposta final do atendente: %s", answer)
    return answer
