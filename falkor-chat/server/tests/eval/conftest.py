"""Shared fixtures for `server/tests/eval/` (K-026, `docs/plans/graphrag-eval.md`
§5 Unit 2b, design D2).

**`ws:eval` is persistent, not rebuilt per session (D2).** Unlike `ws:test`/
`ws:live`, it is seeded once via `./scripts/seed_eval_corpus.sh` and left alone —
rebuilding it here (with real embedding calls) would make the default, network-free
test suite depend on a live embedder, and would make "did recall regress" ambiguous
between "the retrieval code changed" and "the corpus embeddings drifted between
runs." So the `ws_eval` fixture below is **probe-only**: it never bootstraps, never
seeds, never writes — it only checks readiness and skips cleanly (never fails) when
`ws:eval` isn't there yet, mirroring `test_workflow_live.py`'s skip-never-fail
posture but without the rebuild.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# `tests/eval/__init__.py` (added to fix the `conftest.py` module-name collision
# with `tests/conftest.py` under pytest's default `--import-mode=prepend`) makes
# this directory a package, which moves pytest's own sys.path insertion point up
# to `tests/` instead of `tests/eval/`. `judge.py`/`metrics.py` are test-only
# helper modules imported bare (`from judge import ...`) by the sibling
# `test_*.py` files in this directory, not part of the `falkorchat` package, so
# they need this directory back on `sys.path` explicitly.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest

from falkorchat import db, modelconfig
from falkorchat.repository import Repository

# Matches `scripts/seed_eval_corpus.py`'s own `EVAL_WS = os.environ.get("EVAL_WS",
# "eval")` default — the graph key this fixture probes is `ws:{EVAL_WS}`.
EVAL_WS = os.environ.get("EVAL_WS", "eval")

# §4's corpus target is ~110-130 messages across ~10-12 topics; 50 is a
# conservative floor that distinguishes "looks seeded" from "empty/partial" without
# hardcoding the exact target count here (the corpus is free to grow).
_MIN_MESSAGE_COUNT = 50


def _falkordb_reachable() -> bool:
    try:
        db.connect().select_graph(f"ws:{EVAL_WS}").query("RETURN 1")
        return True
    except Exception:
        return False


def _expected_embedding_dim() -> int:
    """D7 mechanism 1: read the real `config/models.json` directly, bypassing
    `ModelGateway.from_env()` (redirected to the dim-4 test-fixture config for every
    pytest test by `server/tests/conftest.py`'s autouse fixture) — the same pattern
    `test_golden_set_integrity.py` uses."""
    overlay = modelconfig.Overlay.load(str(modelconfig.DEFAULT_MODEL_CONFIG_PATH))
    ref = overlay.default_for("embedding")
    assert ref, (
        f"{modelconfig.DEFAULT_MODEL_CONFIG_PATH} has no defaults.embedding — "
        f"cannot determine ws:eval's expected vector-index dimension"
    )
    dim = overlay.model_settings(ref).get("dim")
    assert isinstance(dim, int), (
        f"{ref!r} in {modelconfig.DEFAULT_MODEL_CONFIG_PATH} has no declared "
        f"integer 'dim' — cannot determine ws:eval's expected vector-index dimension"
    )
    return dim


def _message_count() -> int:
    graph = db.workspace_graph(db.connect(), EVAL_WS)
    res = graph.ro_query("MATCH (m:Message) RETURN count(m)")
    return res.result_set[0][0]


@pytest.fixture(scope="session")
def ws_eval() -> str:
    """Skip (never fail) unless `ws:eval` is reachable and looks fully seeded.

    Readiness = FalkorDB reachable + a `Message` vector index exists at exactly
    the currently-configured embedding dimension + a plausible minimum message
    count. Returns the workspace id (`EVAL_WS`) for convenience.
    """
    if not _falkordb_reachable():
        pytest.skip(
            "FalkorDB not reachable — start it with ./scripts/start_falkordb.sh -d"
        )

    repo = Repository(db.connect())
    actual_dim = repo.read_index_dimension(EVAL_WS, label="Message")
    if actual_dim is None:
        pytest.skip(
            f"ws:{EVAL_WS} has no Message vector index yet — run "
            f"./scripts/seed_eval_corpus.sh"
        )

    expected_dim = _expected_embedding_dim()
    if actual_dim != expected_dim:
        pytest.skip(
            f"ws:{EVAL_WS}'s Message vector index is dim={actual_dim}, but the "
            f"configured embedding model expects dim={expected_dim} — reseed with "
            f"RESEED=1 ./scripts/seed_eval_corpus.sh"
        )

    count = _message_count()
    if count < _MIN_MESSAGE_COUNT:
        pytest.skip(
            f"ws:{EVAL_WS} has only {count} message(s) (< {_MIN_MESSAGE_COUNT}) — "
            f"looks unseeded or partially seeded; run ./scripts/seed_eval_corpus.sh"
        )

    return EVAL_WS
