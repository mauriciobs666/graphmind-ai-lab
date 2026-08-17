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
from redis.exceptions import ResponseError

from falkorchat import config, db
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


def _falkordb_reachable(ws: str = TEST_WS) -> bool:
    """True iff FalkorDB responds for `ws:{ws}` — never materializes the graph
    key as a side effect, even when it doesn't exist yet (K-046, mirroring
    `tests/eval/conftest.py`'s own `_falkordb_reachable`, fixed for the
    identical bug at K-026 Unit 2b, B-1).

    Routed via `ro_query` (`GRAPH.RO_QUERY`), never `.query` (`GRAPH.QUERY`):
    per `claude/graph-dba/falkordb-quirks.md`, a write-mode query against a
    nonexistent graph key silently materializes an empty graph key, which is
    exactly the side effect a reachability probe must never have. A
    `ResponseError` containing "empty key" means the server responded but
    `ws:{ws}` doesn't exist yet — that still counts as *reachable*.
    """
    try:
        db.connect().select_graph(f"ws:{ws}").ro_query("RETURN 1")
        return True
    except ResponseError as exc:
        if "empty key" in str(exc):
            return True
        return False
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


@pytest.fixture()
def wf_repo(conn) -> Repository:
    """A Repository over `ws:test` **and** the global `reference` graph, both wiped.

    `conn` wipes only `ws:test` node data; the `reference` graph is global and
    additive (workflow-def authoring writes there, plan F3), so workflow tests
    also wipe its node data here to stay isolated across tests (plan F8 — no
    cross-test collisions). Schema (indexes + constraints) survives DETACH DELETE.
    """
    db.reference_graph(conn).query("MATCH (n) DETACH DELETE n")
    return Repository(conn)


_TESTS_DATA = Path(__file__).resolve().parent / "data"


@pytest.fixture(autouse=True)
def _model_config_env(monkeypatch):
    """K-042: point the two model-config env vars at the offline `tests/data/`
    fixtures for every test, so the suite never depends on a developer's real
    `~/.config/opencode/opencode.json` (the M-2 gate — this repo's real dev box has
    one; the suite must pass on a machine with no such file). `monkeypatch` reverts
    both after each test, so a test that explicitly wants a different (or absent)
    value can still override/delete it locally.

    Sets **both** the env vars (for anything that reads `os.environ` directly, or a
    subprocess that inherits it) **and** `config.OPENCODE_CONFIG_PATH`/
    `config.MODEL_CONFIG_PATH` themselves — `falkorchat.config` resolves its env vars
    once at *import* time (FR-15: read once, no reload path), and `falkorchat.config`
    is already imported by the time this per-test fixture runs, so a bare `setenv`
    alone would never reach `ModelGateway.from_env()`. Same pattern `test_app.py`
    already uses for `config.ENABLE_AGENT` etc.
    """
    opencode_path = str(_TESTS_DATA / "opencode.json")
    model_config_path = str(_TESTS_DATA / "models.json")
    monkeypatch.setenv("FALKORCHAT_OPENCODE_CONFIG", opencode_path)
    monkeypatch.setenv("FALKORCHAT_MODEL_CONFIG", model_config_path)
    monkeypatch.setattr(config, "OPENCODE_CONFIG_PATH", opencode_path)
    monkeypatch.setattr(config, "MODEL_CONFIG_PATH", model_config_path)
