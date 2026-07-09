"""Shared pytest fixtures.

Repository tests are **integration** tests against a live FalkorDB `ws:test`
graph (the same isolated-graph approach `scripts/test_queries.sh` uses). The
graph's schema (indexes + constraints) is bootstrapped once per session; node
data is wiped before every test so each test starts from an empty-but-schemaed
graph.

Requires a running FalkorDB (``./scripts/start_falkordb.sh``). If the instance
is unreachable the whole integration suite is skipped with a clear reason,
rather than reporting misleading failures.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from falkorchat import db
from falkorchat.repository import Repository

TEST_WS = "test"
# The `ws:test` vector indexes are bootstrapped at this dimension — matching
# scripts/test_queries.sh (its own isolated-suite dim). A vector-index dimension
# is fixed at creation and cannot be altered in place, so GraphRAG tests build
# stub vectors of exactly this length. (Production GraphRAG workspaces are 1024;
# never hardcode 1024 in these tests.)
TEST_EMBEDDING_DIM = 4
_REPO_ROOT = Path(__file__).resolve().parents[2]
_BOOTSTRAP = _REPO_ROOT / "scripts" / "bootstrap_schema.sh"


def _falkordb_reachable() -> bool:
    try:
        conn = db.connect()
        conn.select_graph("ws:test").query("RETURN 1")
        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def _schema() -> None:
    """Rebuild `ws:test` schema once per session at ``TEST_EMBEDDING_DIM``.

    The whole graph is dropped first so the vector-index dimension is
    deterministic: a vector index's dimension is fixed at creation and cannot be
    altered in place, so a stale wrong-dim index left by a prior run (or by the
    default 1536 bootstrap) would silently break the GraphRAG tests.
    """
    if not _falkordb_reachable():
        pytest.skip("FalkorDB not reachable — start it with ./scripts/start_falkordb.sh")
    try:
        db.connect().select_graph(f"ws:{TEST_WS}").delete()
    except Exception:
        pass  # graph may not exist yet — bootstrap creates it fresh below
    subprocess.run(
        ["bash", str(_BOOTSTRAP), TEST_WS],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "EMBEDDING_DIM": str(TEST_EMBEDDING_DIM)},
    )


@pytest.fixture()
def conn(_schema):
    """A FalkorDB connection with the `ws:test` graph wiped clean (schema kept)."""
    connection = db.connect()
    graph = db.workspace_graph(connection, TEST_WS)
    graph.query("MATCH (n) DETACH DELETE n")
    return connection


@pytest.fixture()
def repo(conn) -> Repository:
    """A Repository over a freshly-wiped `ws:test` graph."""
    return Repository(conn)
