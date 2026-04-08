import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from llm import llm
from graph import get_schema_description, graph, stringify_value
from utils_common import setup_logger
from prompts import ANSWER_TEMPLATE, CYPHER_GENERATION_TEMPLATE

logger = setup_logger("tools.cypher")

cypher_prompt = ChatPromptTemplate.from_template(CYPHER_GENERATION_TEMPLATE)
cypher_chain = cypher_prompt | llm | StrOutputParser()

answer_prompt = ChatPromptTemplate.from_template(ANSWER_TEMPLATE)
answer_chain = answer_prompt | llm | StrOutputParser()

_ALLOWED_CYPHER_KEYWORDS = {
    "match", "return", "where", "with", "order", "by", "limit", "skip",
    "optional", "distinct", "as", "and", "or", "not", "in", "contains",
    "starts", "ends", "labels", "properties", "collect", "count", "sum",
    "avg", "min", "max", "head", "tail", "size", "unwind", "range",
    "tointeger", "tostring", "tofloat", "split", "replace", "trim",
    "lower", "upper", "left", "right", "substring", "node", "relationship",
    "type", "id", "start", "end", "p", "n", "r", "i", "m", "c",
}
_COMMENT_PATTERN = re.compile(r"//.*?$|/\*.*?\*/", re.MULTILINE | re.DOTALL)
_DANGEROUS_PATTERN = re.compile(
    r"\b(create|delete|set|remove|drop|detach|merge)\b",
    re.IGNORECASE
)


def _extract_cypher(text: str) -> str:
    if not text:
        logger.debug("LLM returned an empty response for Cypher generation.")
        return ""
    fenced_blocks = re.findall(r"```(?:cypher)?(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced_blocks:
        query = fenced_blocks[-1].strip()
        logger.debug("Extracted Cypher block:\n%s", query)
        return query
    query = text.strip()
    logger.debug("Using entire response as Cypher:\n%s", query)
    return query


def _looks_like_cypher(query: str) -> bool:
    if not query:
        return False
    normalized = query.lstrip().lower()
    return "match" in normalized and "return" in normalized


def _validate_safe_cypher(query: str) -> Tuple[bool, Optional[str]]:
    cleaned = _COMMENT_PATTERN.sub("", query)
    if _DANGEROUS_PATTERN.search(cleaned):
        logger.warning("Potentially dangerous Cypher query blocked: %s", query[:100])
        return False, "Query contains disallowed operations."
    normalized = re.sub(r"[^a-z0-9\s]", " ", cleaned.lower())
    words = set(normalized.split())
    allowed = words & _ALLOWED_CYPHER_KEYWORDS
    unknown = words - _ALLOWED_CYPHER_KEYWORDS - {"the", "a", "an", "is", "are", "from", "all"}
    if unknown:
        logger.debug("Unusual Cypher keywords found: %s", unknown)
    return True, None


MENU_STANDARD_QUERY = (
    "MATCH (p:Pastel)-[:FEITO_DE]->(i:Ingrediente)\n"
    "RETURN p.name AS name, p.price AS price, collect(DISTINCT i.name) AS ingredients"
)


def _is_menu_request(question: str) -> bool:
    normalized = question.lower()
    return any(keyword in normalized for keyword in ["cardapio", "cardápio", "menu"])


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
            record[header] = stringify_value(row[idx])
        rows.append(record)
    return rows


def _execute_cypher(query: str) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    Execute a Cypher query against the FalkorDB graph.

    Args:
        query (str): The Cypher query to execute.

    Returns:
        Tuple[List[Dict[str, Any]], Optional[str]]: A tuple containing the query results as a list of dictionaries
        and an optional error message if the query fails.
    """
    try:
        logger.debug("Running Cypher:\n%s", query)
        query_result = graph.ro_query(query)
    except Exception as exc:
        logger.exception("Error running Cypher.")
        return [], f"Não consegui executar a consulta Cypher: {exc}"
    rows = _format_rows(query_result)
    logger.debug("Query returned %d row(s).", len(rows))
    return rows, None


def cypher_qa(question: str) -> str:
    logger.info("Question received: %s", question)
    schema = get_schema_description()
    logger.debug("Schema snapshot:\n%s", schema)
    cypher_suggestion = cypher_chain.invoke({"schema": schema, "question": question})
    logger.debug("Raw Cypher generation output:\n%s", cypher_suggestion)
    cypher_query = _extract_cypher(cypher_suggestion)
    logger.debug("Cypher after extraction:\n%s", cypher_query)
    if _is_menu_request(question):
        logger.debug("Detected menu-style request.")

    if not cypher_query or not _looks_like_cypher(cypher_query):
        logger.warning("Generated text did not look like a valid Cypher query:\n%s", cypher_query)
        if _is_menu_request(question):
            logger.info("Using standard full-menu query for generic menu request.")
            cypher_query = MENU_STANDARD_QUERY
        else:
            return "Não consegui gerar uma consulta Cypher para essa pergunta."
    else:
        is_safe, error_msg = _validate_safe_cypher(cypher_query)
        if not is_safe:
            logger.error("Cypher validation failed: %s", error_msg)
            return "Não consegui executar essa consulta de forma segura."
        logger.debug("Cypher query validated as safe.")

    rows, error = _execute_cypher(cypher_query)
    if error:
        logger.error("Error executing query: %s", error)
        return error
    if not rows:
        logger.debug("Cypher result set empty.")
    else:
        logger.debug("First row sample: %s", rows[0])

    context = json.dumps(rows, ensure_ascii=False, indent=2) if rows else "No records found."
    logger.debug("Context passed to the LLM:\n%s", context)

    answer = answer_chain.invoke(
        {"question": question, "cypher_query": cypher_query, "context": context}
    )
    logger.info("Final answer returned to the user: %s", answer)
    return answer
