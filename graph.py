import streamlit as st
from falkordb import FalkorDB

from functools import lru_cache
from typing import Any, Dict, List, Optional

DEFAULT_URL = "redis://localhost:6379"


def _normalize_url(url: str) -> str:
    if not url:
        return DEFAULT_URL
    if url.startswith("falkor://"):
        return "redis://" + url[len("falkor://") :]
    if url.startswith("falkors://"):
        return "rediss://" + url[len("falkors://") :]
    return url


def _build_url(host: str, port: int, username: Optional[str], password: Optional[str]) -> str:
    auth = ""
    if username or password:
        user = username or ""
        pwd = password or ""
        auth = f"{user}:{pwd}@"
    return f"redis://{auth}{host}:{port}"


connection_url = _normalize_url(st.secrets.get("FALKORDB_URL"))
graph_name = st.secrets.get("FALKORDB_GRAPH", "kg_pastel")

if connection_url and connection_url != DEFAULT_URL:
    db = FalkorDB.from_url(connection_url)
else:
    host = st.secrets.get("FALKORDB_HOST", "localhost")
    port = int(st.secrets.get("FALKORDB_PORT", 6379))
    username = st.secrets.get("FALKORDB_USERNAME")
    password = st.secrets.get("FALKORDB_PASSWORD")
    db = FalkorDB(host=host, port=port, username=username, password=password)
    connection_url = _build_url(host, port, username, password)

FALKORDB_URL = connection_url
graph = db.select_graph(graph_name)


@lru_cache(maxsize=1)
def get_schema_description() -> str:
    """
    Return a textual description of the current FalkorDB schema so that
    LLM prompts can reason about the available nodes and relationships.
    """

    def _safe_query(query: str) -> List[List[Any]]:
        try:
            return graph.ro_query(query).result_set
        except Exception:
            return []

    node_labels = [
        row[0] for row in _safe_query("MATCH (n) UNWIND labels(n) AS label RETURN DISTINCT label ORDER BY label")
    ]
    node_props_rows = _safe_query(
        """
        MATCH (n)
        UNWIND labels(n) AS label
        UNWIND keys(n) AS property
        RETURN label, collect(DISTINCT property) AS properties
        """
    )
    node_props: Dict[str, List[str]] = {label: props or [] for label, props in node_props_rows}
    relationship_rows = _safe_query(
        """
        MATCH (start)-[r]->(end)
        RETURN DISTINCT labels(start) AS source,
                        type(r) AS rel_type,
                        labels(end) AS target,
                        keys(r) AS properties
        """
    )

    lines: List[str] = []

    if node_labels:
        lines.append("Node types:")
        for label in node_labels:
            props = ", ".join(sorted(node_props.get(label, []))) or "no explicit properties"
            lines.append(f"- {label} {{ {props} }}")
    else:
        lines.append("Node types: none found.")

    if relationship_rows:
        lines.append("\nRelationships:")
        for source, rel_type, target, rel_props in relationship_rows:
            lhs = ":".join(source) if source else "Unknown"
            rhs = ":".join(target) if target else "Unknown"
            props = ", ".join(sorted(rel_props)) if rel_props else "no properties"
            lines.append(f"- ({lhs})-[:{rel_type}]->({rhs}) {{ {props} }}")
    else:
        lines.append("\nRelationships: none found.")

    return "\n".join(lines)
