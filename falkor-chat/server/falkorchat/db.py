"""FalkorDB connection + graph selection (DESIGN §14.2 `db.py`).

Thin wrapper over `falkordb-py`. The only responsibilities here are opening a
connection and selecting the per-workspace graph (`ws:{id}`); all Cypher lives
in `repository.py`.
"""

from __future__ import annotations

from falkordb import FalkorDB, Graph

from . import config


def connect(host: str | None = None, port: int | None = None) -> FalkorDB:
    """Open a FalkorDB connection.

    Config is resolved at call time (not import time) so tests/deploys can
    repoint `config.FALKORDB_*` without re-importing this module.
    """
    return FalkorDB(
        host=host if host is not None else config.FALKORDB_HOST,
        port=port if port is not None else config.FALKORDB_PORT,
    )


def workspace_graph(db: FalkorDB, ws: str) -> Graph:
    """Select the per-workspace graph for workspace id `ws` (key ``ws:{ws}``).

    One graph per workspace is a locked decision (AGENTS.md) — never filter by a
    `workspaceId` property inside a shared graph.
    """
    return db.select_graph(f"ws:{ws}")
