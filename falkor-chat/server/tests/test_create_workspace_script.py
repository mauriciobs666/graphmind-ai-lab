"""Regression guard for FR-2's enforcement mechanism — `create_workspace.sh`
(§3.2 Option B, `docs/plans/embedding-migration.md` §3.2/§5 step 2).

`create_workspace.sh` is pure wiring (`bootstrap_schema.sh "$@"` then FR-2's pin
operation per given id) — the property worth pinning down with a real test is
the ORDER: bootstrap must run, and succeed, before pin ever touches the graph.
A pure logic test of `bootstrap_schema.sh` or `embedding_migration.pin()` in
isolation cannot see a wiring-order regression in the wrapper that calls both,
so this shells out to the real script against a throwaway workspace — same
subprocess-integration convention as `test_seed_workflows_script.py`.

Requires a live FalkorDB (skipped otherwise, same posture as `conftest.py`'s
`_schema` fixture) and the `server/.venv` (`create_workspace.sh` execs
`pin_workspace_embedding_model.sh`, which execs `server/.venv/bin/python`).
Points `FALKORCHAT_OPENCODE_CONFIG`/`FALKORCHAT_MODEL_CONFIG` at a throwaway
fixture pair (same shape as `test_embedding_migration.py`'s
`_point_model_config_at`) so this test never depends on the ambient dev box's
real `~/.config/opencode/opencode.json`.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
from redis.exceptions import ResponseError

from falkorchat import db
from falkorchat.repository import Repository

_REPO_ROOT = Path(__file__).resolve().parents[2]  # .../falkor-chat
_CREATE_WORKSPACE = _REPO_ROOT / "scripts" / "create_workspace.sh"

# Throwaway workspace id, distinct from ws:test/ws:eval/ws:acme/etc — this test
# creates and fully GRAPH.DELETEs it itself, both before and after, so it never
# collides with any other test's shared graph state.
WS = "k072-create-workspace-guard"
# A second throwaway id, for the multi-id-in-one-call test below — `create_workspace.sh`'s
# own usage/docstring advertises `<wsId> [<wsId> ...]`, so a test that only ever passes
# one id can't see a regression where the pin loop narrows to just the first argument
# (teco's independent mutation, 2026-09-20) while bootstrap still runs against all of them.
WS2 = f"{WS}-2"

_DEFAULT_EMBEDDING_REF = "lmstudio/test-fixture-create-workspace-model"


def _falkordb_reachable() -> bool:
    try:
        db.connect().select_graph(f"ws:{WS}").ro_query("RETURN 1")
        return True
    except ResponseError as exc:
        if "empty key" in str(exc):
            return True
        return False
    except Exception:
        return False


def _drop_ws(ws: str = WS) -> None:
    try:
        db.connect().select_graph(f"ws:{ws}").delete()
    except Exception:
        pass  # graph may not exist — fine, that's the state we want


@pytest.fixture()
def throwaway_ws():
    if not _falkordb_reachable():
        pytest.skip("FalkorDB not reachable — start it with ./scripts/start_falkordb.sh")
    _drop_ws()  # guarantee a genuinely fresh (nonexistent) graph key
    yield WS
    _drop_ws()


@pytest.fixture()
def throwaway_ws_pair():
    """Two distinct, genuinely fresh throwaway workspace ids, for a single
    `create_workspace.sh <id1> <id2>` invocation."""
    if not _falkordb_reachable():
        pytest.skip("FalkorDB not reachable — start it with ./scripts/start_falkordb.sh")
    _drop_ws(WS)
    _drop_ws(WS2)
    yield WS, WS2
    _drop_ws(WS)
    _drop_ws(WS2)


def _model_config_env(tmp_path: Path, *, default_embedding: str) -> dict[str, str]:
    """Same fixture shape as `test_embedding_migration.py`'s
    `_point_model_config_at`, but as env-var overrides for a subprocess rather
    than monkeypatching this process — `create_workspace.sh`'s pin step runs as
    a separate `python` process, so an in-process monkeypatch wouldn't reach it.
    """
    opencode_path = tmp_path / "opencode.json"
    opencode_path.write_text(
        json.dumps(
            {
                "provider": {
                    "lmstudio": {
                        "npm": "@ai-sdk/openai-compatible",
                        "name": "LM Studio (test fixture)",
                        "options": {"baseURL": "http://localhost:1234/v1"},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    model_config_path = tmp_path / "models.json"
    model_config_path.write_text(
        json.dumps({"defaults": {"embedding": default_embedding}}), encoding="utf-8"
    )
    return {
        "FALKORCHAT_OPENCODE_CONFIG": str(opencode_path),
        "FALKORCHAT_MODEL_CONFIG": str(model_config_path),
    }


def test_create_workspace_bootstraps_then_pins(throwaway_ws, tmp_path):
    """The done-condition from `docs/plans/embedding-migration.md` §5 step 2:
    one call leaves the workspace both schema-bootstrapped AND pinned, with no
    separate operator action."""
    env = {
        **os.environ,
        **_model_config_env(tmp_path, default_embedding=_DEFAULT_EMBEDDING_REF),
    }

    result = subprocess.run(
        ["bash", str(_CREATE_WORKSPACE), throwaway_ws],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"create_workspace.sh failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )

    repo = Repository(db.connect())

    # bootstrap ran: the vector index (bootstrap_schema.sh-only DDL) exists.
    dim = repo.read_index_dimension(throwaway_ws, label="Message")
    assert dim is not None, "bootstrap_schema.sh's vector index is missing — bootstrap step didn't run"

    # pin ran: the workspace was pinned to the (fixture) global default.
    overrides = repo.read_model_overrides(throwaway_ws)
    assert overrides["embeddingModel"] == _DEFAULT_EMBEDDING_REF


def test_create_workspace_pins_every_given_id_not_just_the_first(throwaway_ws_pair, tmp_path):
    """`create_workspace.sh <wsId> [<wsId> ...]` — its own usage/docstring promises
    every given id is bootstrapped AND pinned, not only the first. This is the
    coverage gap teco's independent mutation found: narrowing the pin loop from
    `for wid in "$@"` to `for wid in "$1"` still bootstraps every given id
    (bootstrap gets `"$@"` unchanged) but silently leaves every id after the
    first unpinned — exactly the FR-2 gap this mechanism exists to close."""
    ws1, ws2 = throwaway_ws_pair
    env = {
        **os.environ,
        **_model_config_env(tmp_path, default_embedding=_DEFAULT_EMBEDDING_REF),
    }

    result = subprocess.run(
        ["bash", str(_CREATE_WORKSPACE), ws1, ws2],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"create_workspace.sh failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )

    repo = Repository(db.connect())
    for ws in (ws1, ws2):
        dim = repo.read_index_dimension(ws, label="Message")
        assert dim is not None, f"ws:{ws} — bootstrap step didn't run"

        overrides = repo.read_model_overrides(ws)
        assert overrides["embeddingModel"] == _DEFAULT_EMBEDDING_REF, (
            f"ws:{ws} was not pinned — got {overrides['embeddingModel']!r}"
        )


def test_create_workspace_is_idempotent(throwaway_ws, tmp_path):
    env = {
        **os.environ,
        **_model_config_env(tmp_path, default_embedding=_DEFAULT_EMBEDDING_REF),
    }

    for _ in range(2):
        result = subprocess.run(
            ["bash", str(_CREATE_WORKSPACE), throwaway_ws],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, (
            f"create_workspace.sh failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    repo = Repository(db.connect())
    overrides = repo.read_model_overrides(throwaway_ws)
    assert overrides["embeddingModel"] == _DEFAULT_EMBEDDING_REF


def test_create_workspace_rejects_no_args():
    result = subprocess.run(
        ["bash", str(_CREATE_WORKSPACE)], capture_output=True, text=True, timeout=10,
    )
    assert result.returncode != 0
