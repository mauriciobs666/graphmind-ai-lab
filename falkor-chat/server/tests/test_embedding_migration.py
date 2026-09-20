"""Tests for `scripts/embedding_migration.py`'s `pin` subcommand (FR-1/FR-2,
`docs/plans/embedding-migration.md` §5 step 1 / §6 tests 1-4).

`scripts/embedding_migration.py` is not part of the `falkorchat` package (it lives
in `scripts/`, run via `server/.venv/bin/python`, same posture as
`seed_eval_corpus.py`) — imported here the same way `test_seed_workflows_script.py`
reaches into `scripts/`, except by direct module import (needed so a spy/fake repo
can assert on call counts, not just end state) rather than subprocess.

Tests 1/2 are live-FalkorDB integration tests against a fresh `ws:test` graph (the
`repo` fixture, `conftest.py`) — matching `test_repository.py`'s convention for
`write_model_overrides`/`read_model_overrides` themselves. Tests 3-5 are pure-logic,
offline tests against a fake in-memory repo (`_FakeRepo`, same convention
`test_modelconfig.py` already uses for `GraphWorkspaceOverrides`'s own fake), so a
"no write issued" assertion is a spy count, not an inference from end state.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from falkorchat import config
from falkorchat.repository import Repository

_REPO_ROOT = Path(__file__).resolve().parents[2]  # .../falkor-chat
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import embedding_migration  # noqa: E402


# ── the current test-fixture global default (tests/data/models.json) ─────────────
_DEFAULT_EMBEDDING_REF = "lmstudio/text-embedding-qwen3-embedding-0.6b"


# ── a fake, in-memory repo — same convention as test_modelconfig.py's `_FakeRepo`,
# extended with a write side so a "no write issued" assertion is a spy count, not
# an inference from end state alone ────────────────────────────────────────────

class _FakeRepo:
    def __init__(self, initial: dict[str, str | None] | None = None) -> None:
        self._overrides = initial or {
            "agentModel": None, "guardModel": None,
            "embeddingModel": None, "responderModel": None,
        }
        self.write_calls: list[dict[str, Any]] = []

    def read_model_overrides(self, ws: str) -> dict[str, str | None]:
        return dict(self._overrides)

    def write_model_overrides(
        self, ws: str, *, agent=None, guard=None, embedding=None, responder=None,
        at: int, by: str,
    ) -> dict[str, Any]:
        self.write_calls.append(
            {"ws": ws, "agent": agent, "guard": guard, "embedding": embedding,
             "responder": responder, "at": at, "by": by}
        )
        self._overrides = {
            "agentModel": agent, "guardModel": guard,
            "embeddingModel": embedding, "responderModel": responder,
        }
        return {"agent": agent, "guard": guard, "embedding": embedding, "responder": responder}


class _ExplodingRepo:
    """Proves the WARNING path never touches the graph at all — any call raises."""

    def read_model_overrides(self, ws: str):  # pragma: no cover - must never run
        raise AssertionError("pin() must not read the repo when the gateway failed")

    def write_model_overrides(self, ws: str, **kwargs):  # pragma: no cover
        raise AssertionError("pin() must not write the repo when the gateway failed")


def _write_json(path: Path, doc: dict) -> str:
    path.write_text(json.dumps(doc), encoding="utf-8")
    return str(path)


def _point_model_config_at(monkeypatch, tmp_path: Path, *, default_embedding: str) -> None:
    """Repoint `ModelGateway.from_env()` at a fresh overlay declaring
    `defaults.embedding = default_embedding`, same provider fixture as
    `tests/data/opencode.json`. Mirrors `conftest.py`'s `_model_config_env` fixture
    (env var AND the already-imported `config` module attributes, since
    `falkorchat.config` resolves its paths once at import time)."""
    opencode_path = _write_json(
        tmp_path / "opencode.json",
        {
            "provider": {
                "lmstudio": {
                    "npm": "@ai-sdk/openai-compatible",
                    "name": "LM Studio (test fixture)",
                    "options": {"baseURL": "http://localhost:1234/v1"},
                }
            }
        },
    )
    model_config_path = _write_json(
        tmp_path / "models.json",
        {"defaults": {"embedding": default_embedding}},
    )
    monkeypatch.setenv("FALKORCHAT_OPENCODE_CONFIG", opencode_path)
    monkeypatch.setenv("FALKORCHAT_MODEL_CONFIG", model_config_path)
    monkeypatch.setattr(config, "OPENCODE_CONFIG_PATH", opencode_path)
    monkeypatch.setattr(config, "MODEL_CONFIG_PATH", model_config_path)


# ── test 1 (live): no override at all → pinned to the current global default ────

def test_pin_no_override_pins_to_current_global_default(repo: Repository):
    result = embedding_migration.pin("test", repo=repo)

    assert result is not None
    assert result["embeddingModel"] == _DEFAULT_EMBEDDING_REF
    # not "nulled from nothing" — absent stays absent, never coerced to a value.
    assert result["agentModel"] is None
    assert result["guardModel"] is None
    assert result["responderModel"] is None

    on_disk = repo.read_model_overrides("test")
    assert on_disk["embeddingModel"] == _DEFAULT_EMBEDDING_REF
    assert on_disk["agentModel"] is None


def test_pin_is_idempotent_on_a_real_repo(repo: Repository):
    first = embedding_migration.pin("test", repo=repo)
    second = embedding_migration.pin("test", repo=repo)

    assert first == second
    assert second["embeddingModel"] == _DEFAULT_EMBEDDING_REF


# ── test 2 (live): an existing agentModelOverride survives untouched ────────────

def test_pin_preserves_existing_agent_override_byte_identical(repo: Repository):
    repo.write_model_overrides(
        "test", agent="lmstudio/mistralai/ministral-3-3b", at=100, by="test-setup",
    )

    result = embedding_migration.pin("test", repo=repo)

    assert result["agentModel"] == "lmstudio/mistralai/ministral-3-3b"
    assert result["embeddingModel"] == _DEFAULT_EMBEDDING_REF

    on_disk = repo.read_model_overrides("test")
    assert on_disk["agentModel"] == "lmstudio/mistralai/ministral-3-3b"
    assert on_disk["embeddingModel"] == _DEFAULT_EMBEDDING_REF


def test_pin_preserves_guard_and_responder_overrides_in_their_own_slots(
    repo: Repository,
):
    # Regression coverage for the read-before-write discipline on ALL three
    # sibling kinds, not just `agentModel` — a swapped `guard=`/`responder=` pair
    # in the write call is invisible unless both are set, distinct, and non-None
    # on the same workspace beforehand.
    repo.write_model_overrides(
        "test",
        agent="lmstudio/agent-override",
        guard="lmstudio/guard-override",
        responder="lmstudio/responder-override",
        at=100, by="test-setup",
    )

    result = embedding_migration.pin("test", repo=repo)

    assert result["agentModel"] == "lmstudio/agent-override"
    assert result["guardModel"] == "lmstudio/guard-override"
    assert result["responderModel"] == "lmstudio/responder-override"
    assert result["embeddingModel"] == _DEFAULT_EMBEDDING_REF

    on_disk = repo.read_model_overrides("test")
    assert on_disk["agentModel"] == "lmstudio/agent-override"
    assert on_disk["guardModel"] == "lmstudio/guard-override"
    assert on_disk["responderModel"] == "lmstudio/responder-override"
    assert on_disk["embeddingModel"] == _DEFAULT_EMBEDDING_REF


# ── test 3 (offline, spy repo): already-set embeddingModelOverride → true no-op ──

def test_pin_already_set_is_a_true_noop_write_never_issued():
    fake = _FakeRepo({
        "agentModel": "lmstudio/existing-agent", "guardModel": None,
        "embeddingModel": "lmstudio/already-pinned-model", "responderModel": None,
    })
    pre_call_read = fake.read_model_overrides("acme")

    result = embedding_migration.pin("acme", repo=fake)

    assert result == pre_call_read
    assert fake.write_calls == []  # the wasted-write regression this test guards


# ── test 4 (offline, acceptance criteria 1-3): two workspaces, default moves
# between calls — each workspace keeps the default in effect at its own "birth",
# and an already-pinned workspace is unaffected by a later default change ──────

def test_pin_two_workspaces_each_keep_the_default_at_their_own_birth(
    monkeypatch, tmp_path,
):
    _point_model_config_at(monkeypatch, tmp_path, default_embedding="lmstudio/model-a")
    repo_ws1 = _FakeRepo()

    result_ws1 = embedding_migration.pin("ws1", repo=repo_ws1)
    assert result_ws1["embeddingModel"] == "lmstudio/model-a"

    # Global default moves — acceptance criteria: an EXISTING workspace's override
    # is unaffected by a later default change.
    _point_model_config_at(monkeypatch, tmp_path, default_embedding="lmstudio/model-b")

    result_ws1_again = embedding_migration.pin("ws1", repo=repo_ws1)
    assert result_ws1_again["embeddingModel"] == "lmstudio/model-a"  # unchanged
    assert len(repo_ws1.write_calls) == 1  # the second call was a no-op

    # A NEW workspace, pinned after the default moved, pins the default in effect
    # at its own birth.
    repo_ws2 = _FakeRepo()
    result_ws2 = embedding_migration.pin("ws2", repo=repo_ws2)
    assert result_ws2["embeddingModel"] == "lmstudio/model-b"


# ── the WARNING path: a broken/missing FALKORCHAT_OPENCODE_CONFIG never raises ──

def test_pin_missing_opencode_config_warns_and_does_not_raise(monkeypatch, capsys):
    monkeypatch.delenv("FALKORCHAT_OPENCODE_CONFIG", raising=False)
    monkeypatch.setattr(config, "OPENCODE_CONFIG_PATH", None)

    result = embedding_migration.pin("acme", repo=_ExplodingRepo())

    assert result is None
    captured = capsys.readouterr()
    assert "WARNING" in captured.out
    assert "acme" in captured.out


def test_pin_gateway_construction_failure_never_touches_the_repo(monkeypatch):
    monkeypatch.delenv("FALKORCHAT_OPENCODE_CONFIG", raising=False)
    monkeypatch.setattr(config, "OPENCODE_CONFIG_PATH", None)

    # _ExplodingRepo raises AssertionError on ANY read/write call — reaching this
    # line without raising IS the assertion that pin() never touched the graph.
    embedding_migration.pin("acme", repo=_ExplodingRepo())


# ── CLI wiring: the `pin` subparser exists and forwards to `pin()` ──────────────

def test_cli_pin_subcommand_parses_multiple_workspace_ids():
    parser = embedding_migration._build_arg_parser()
    args = parser.parse_args(["pin", "acme", "eval"])
    assert args.command == "pin"
    assert args.workspace == ["acme", "eval"]


def test_cli_requires_a_subcommand():
    parser = embedding_migration._build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
