import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from llm import llm
from graph import get_schema_description, graph, stringify_value
from utils_common import setup_logger

logger = setup_logger("tools.cypher")

CYPHER_GENERATION_TEMPLATE = """
You are a FalkorDB expert developer.
Generate exactly one Cypher query that answers the user question,
based strictly on the schema below.
Return only the Cypher wrapped in a ```cypher``` fenced code block—no explanations.

Schema:
{schema}

Rules:
- Use only the labels/relationships/properties described above.
- Generate read-only statements (MATCH/RETURN) and keep one `MATCH (p:Pastel)-[:FEITO_DE]->(i:Ingrediente)` block.
- Always return **all** columns with aliases: `RETURN p.flavor AS flavor, p.price AS price, collect(DISTINCT i.name) AS ingredients`.
- When filtering, prefer `p.flavor` for flavor questions. If filtering by ingredient, still collect every ingredient of the pastel (do not limit to the filtered one).
- Always use case-insensitive comparisons with `toLower()`, preferably with `CONTAINS`.
- No explanations or comments—only valid Cypher.
- Always list every property you need explicitly in the RETURN clause.
- Use only the property names shown (e.g., `p.flavor`, `p.price`, `i.name`); do not invent new ones.
- Do not add `LIMIT`, `ORDER BY`, or extra `MATCH`/`OPTIONAL MATCH` clauses unless the user explicitly requests them.
- Aggregate ingredients even when filtering by a single ingredient (never return just the matched ingredient).
- When excluding ingredients, filter *after* collecting them for each pastel (e.g., collect into `ingredients`, then use `WHERE ALL(name IN ingredients WHERE ...)`).
- Example template:
  `MATCH (p:Pastel)-[:FEITO_DE]->(i:Ingrediente)`
  `WHERE toLower(p.flavor) = toLower("Calabresa")`
  `RETURN p.flavor AS flavor, p.price AS price, collect(DISTINCT i.name) AS ingredients`

Question:
{question}
"""

ANSWER_TEMPLATE = """
You are the virtual attendant for “Pastel do Mau”.
Use only the structured data provided below to answer the customer.
- Mention flavors, ingredients, and prices returned in the context.
- If multiple pastéis match, summarize them in natural language (e.g., bullet list or short paragraphs).
- If the context is empty, politely say that you don’t have enough information.
- Do **not** invent data beyond what is shown.
- Respond in Brazilian Portuguese.

Customer Question: {question}
Cypher Query Used: {cypher_query}
Query Results (JSON): {context}
"""

cypher_prompt = ChatPromptTemplate.from_template(CYPHER_GENERATION_TEMPLATE)
cypher_chain = cypher_prompt | llm | StrOutputParser()

answer_prompt = ChatPromptTemplate.from_template(ANSWER_TEMPLATE)
answer_chain = answer_prompt | llm | StrOutputParser()


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


MENU_FALLBACK_QUERY = (
    "MATCH (p:Pastel)-[:FEITO_DE]->(i:Ingrediente)\n"
    "RETURN p.flavor AS flavor, p.price AS price, collect(DISTINCT i.name) AS ingredients"
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

    if not cypher_query or not _looks_like_cypher(cypher_query):
        logger.warning("Generated text did not look like a valid Cypher query:\n%s", cypher_query)
        if _is_menu_request(question):
            logger.info("Using fallback full-menu query for generic menu request.")
            cypher_query = MENU_FALLBACK_QUERY
        else:
            return "Não consegui gerar uma consulta Cypher para essa pergunta."

    rows, error = _execute_cypher(cypher_query)
    if error:
        logger.error("Error executing query: %s", error)
        return error

    context = json.dumps(rows, ensure_ascii=False, indent=2) if rows else "No records found."
    logger.debug("Context passed to the LLM:\n%s", context)

    answer = answer_chain.invoke(
        {"question": question, "cypher_query": cypher_query, "context": context}
    )
    logger.info("Final answer returned to the user: %s", answer)
    return answer
