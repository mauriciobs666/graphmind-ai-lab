from functools import lru_cache
import logging
from typing import Any, Dict, List

import streamlit as st
from falkordb import FalkorDB
from falkordb.edge import Edge
from falkordb.node import Node
from falkordb.path import Path

from config import Config
from utils_common import setup_logger

DEFAULT_URL = "redis://localhost:6379"
logger = setup_logger("graph")
logger.setLevel(logging.INFO)


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


try:
    connection_url = Config.get_falkordb_url()
    graph_name = Config.get_falkordb_graph()

    if connection_url and connection_url != DEFAULT_URL:
        db = FalkorDB.from_url(connection_url)
    else:
        credentials = Config.get_falkordb_credentials()
        db = FalkorDB(
            host=credentials["host"],
            port=credentials["port"],
            username=credentials["username"],
            password=credentials["password"]
        )

    graph = db.select_graph(graph_name)
except Exception as exc:  # pragma: no cover - defensive fallback for tests/CI
    logger.warning("Could not connect to FalkorDB: %s", exc)

    class _UnavailableGraph:
        def ro_query(self, *args, **kwargs):
            raise RuntimeError(
                "FalkorDB is unavailable. "
                "Verify the database connection before running queries."
            ) from exc

    graph = _UnavailableGraph()


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
