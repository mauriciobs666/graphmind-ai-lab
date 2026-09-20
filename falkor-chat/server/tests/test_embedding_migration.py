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

from falkorchat import config, db
from falkorchat.repository import Repository

from conftest import TEST_EMBEDDING_DIM, rebuild_vector_indexes

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


def test_cli_migrate_subcommand_parses_flags():
    parser = embedding_migration._build_arg_parser()
    args = parser.parse_args(
        ["migrate", "eval", "lmstudio/new-model", "--batch-size", "10",
         "--i-have-stopped-traffic"]
    )
    assert args.command == "migrate"
    assert args.workspace == "eval"
    assert args.target_ref == "lmstudio/new-model"
    assert args.batch_size == 10
    assert args.traffic_stopped is True


def test_cli_migrate_defaults_batch_size_and_traffic_flag():
    parser = embedding_migration._build_arg_parser()
    args = parser.parse_args(["migrate", "eval", "lmstudio/new-model"])
    assert args.batch_size == 50
    assert args.traffic_stopped is False


# ── D-1 (analyst review + live QA pass, docs/test-reports/embedding-migration-
# report.md): `main()`'s `migrate` branch must surface a `MigrationAbortedError`
# as a clean one-line error + exit 1, never an uncaught traceback — confirmed
# live on all three trigger paths (no traffic-stop flag, interactive "n", an
# undeclared target dim); this pins the fix at the CLI-entry-point level. ──────

def test_cli_migrate_reports_aborted_error_cleanly_instead_of_a_traceback(
    monkeypatch, capsys,
):
    def _explode(*args, **kwargs):
        raise embedding_migration.MigrationAbortedError("traffic must be stopped first")

    monkeypatch.setattr(embedding_migration, "migrate", _explode)
    monkeypatch.setattr(embedding_migration, "_confirm_traffic_stopped", lambda ws: False)

    exit_code = embedding_migration.main(["migrate", "eval", "lmstudio/new-model"])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "ERROR: traffic must be stopped first" in captured.err


# ═══════════════════════════════════════════════════════════════════════════
# `migrate` (FR-3/FR-4/FR-5/FR-8/FR-10, §5 step 4 / §6 tests 5-14).
#
# Always injects a fake gateway (deterministic embeddings, no network — same
# posture the plan's own test strategy names) — a real `Repository` against
# `ws:test` (the `migrate_repo` fixture below), since these ARE the graph
# mechanics `-graph.md` verified live and this unit must exercise for real.
# ═══════════════════════════════════════════════════════════════════════════

_TARGET_REF = "lmstudio/target-embedding-model"
_TARGET_DIM = 8
_OLD_REF = "lmstudio/old-embedding-model"


@pytest.fixture()
def migrate_repo(conn) -> Repository:
    """A `Repository` over `ws:test`, restoring the vector-index dimension to
    `TEST_EMBEDDING_DIM` after the test — unlike every other write in this
    suite, `migrate` itself rebuilds the vector index at a *different*
    dimension, and `conn`'s per-test wipe only clears node data, never
    indexes. Left un-restored, a migrate test would silently break every
    later test in the same session that assumes `ws:test`'s documented dim-4
    index (`conftest.TEST_EMBEDDING_DIM`)."""
    try:
        yield Repository(conn)
    finally:
        rebuild_vector_indexes(conn)


class _FakeResolvedModel:
    def __init__(self, dim: int | None) -> None:
        self.dim = dim


class _FakeResolution:
    def __init__(self, dim: int | None) -> None:
        self.primary = _FakeResolvedModel(dim)


class _FakeEmbedder:
    """Deterministic, network-free stand-in for a real embedder. Two rows with
    the same text embed identically; distinct rows embed distinguishably
    (hash-derived), which is enough to prove "every row got re-embedded" and
    "the right embedder produced this vector" without any real HTTP call."""

    def __init__(self, dim: int, *, raise_after: int | None = None) -> None:
        self.dim = dim
        self.calls = 0
        self._raise_after = raise_after

    def embed(self, text: str) -> list[float]:
        self.calls += 1
        if self._raise_after is not None and self.calls > self._raise_after:
            raise RuntimeError("simulated embedder crash (interrupt/resume test)")
        seed = (abs(hash(text)) % 997) / 997.0
        return [seed + (i * 0.001) for i in range(self.dim)]


class _FakeGateway:
    """Fake `ModelGateway` double. Tracks every `resolve()`/`embedder()` call's
    kwargs so a test can assert `ws=`/`overrides=` were **never** passed
    (§3.3's hard-cap-bypass regression, test 7) — the single most important
    assertion in this file, per the plan's own risk ranking (§7)."""

    def __init__(
        self, *, dims: dict[str, int], embedders: dict[str, _FakeEmbedder] | None = None,
    ) -> None:
        self._dims = dims
        self._embedders = embedders or {ref: _FakeEmbedder(dim) for ref, dim in dims.items()}
        self.resolve_calls: list[dict[str, Any]] = []
        self.embedder_calls: list[dict[str, Any]] = []

    def resolve(self, kind: str, *, requested=None, ws=None, overrides=None):
        self.resolve_calls.append(
            {"kind": kind, "requested": requested, "ws": ws, "overrides": overrides}
        )
        return _FakeResolution(self._dims.get(requested))

    def embedder(self, kind: str, *, requested=None, ws=None, overrides=None):
        self.embedder_calls.append(
            {"kind": kind, "requested": requested, "ws": ws, "overrides": overrides}
        )
        return self._embedders[requested]


class _ExplodingMigrateRepo:
    """Proves the step-0 traffic-stop precondition aborts before ANY graph
    access — every method raises `AssertionError` if called at all (test 6)."""

    def _graph(self, ws: str):  # pragma: no cover - must never run
        raise AssertionError("migrate() must not touch the graph before the traffic-stop check")

    def read_index_dimension(self, ws: str, *, label: str):  # pragma: no cover
        raise AssertionError("migrate() must not touch the graph before the traffic-stop check")

    def read_model_overrides(self, ws: str):  # pragma: no cover
        raise AssertionError("migrate() must not touch the graph before the traffic-stop check")

    def write_model_overrides(self, ws: str, **kwargs):  # pragma: no cover
        raise AssertionError("migrate() must not touch the graph before the traffic-stop check")


class _ExplodingMigrateGateway:
    """Pairs with `_ExplodingMigrateRepo` — proves no HTTP-shaped call is made
    either (test 6 covers "no graph write"; this covers "no model resolution
    or embed call")."""

    def resolve(self, kind: str, **kwargs):  # pragma: no cover
        raise AssertionError("migrate() must not resolve a model before the traffic-stop check")

    def embedder(self, kind: str, **kwargs):  # pragma: no cover
        raise AssertionError("migrate() must not build an embedder before the traffic-stop check")


class _VanishingRowGraph:
    """Wraps a real FalkorDB graph object; the instant the write query for one
    chosen row is about to run, deletes that row via a real Cypher `DETACH
    DELETE` first — reproducing §3.3's "vanished between read and write" race
    against the real engine (a genuine `properties_set == 0` no-op), not a
    faked result object."""

    def __init__(self, real_graph: Any, *, label: str, id_prop: str, vanish_id: str) -> None:
        self._real = real_graph
        self._label = label
        self._id_prop = id_prop
        self._vanish_id = vanish_id
        self._vanished = False

    def ro_query(self, q: str, params: dict | None = None):
        return self._real.ro_query(q, params)

    def query(self, q: str, params: dict | None = None):
        if (
            not self._vanished
            and "SET n.embedding" in q
            and params is not None
            and params.get("id") == self._vanish_id
        ):
            self._vanished = True
            self._real.query(
                f"MATCH (n:{self._label} {{{self._id_prop}: $id}}) DETACH DELETE n",
                {"id": self._vanish_id},
            )
        return self._real.query(q, params)


def _seed_row(
    conn, ws: str, *, label: str, id_prop: str, id_value: str, text: str,
    embedding: list[float] | None = None, embedding_model: str | None = None,
) -> None:
    """Create one `Message`/`Chunk` node directly (bypassing `migrate`/the
    ordinary hot write path), matching the exact property shapes §3.3/§4
    describe: `embedding IS NULL` for a never-embedded row, no `embeddingModel`
    property at all for a row written before this feature existed."""
    graph = db.workspace_graph(conn, ws)
    props: dict[str, Any] = {"id": id_value, "text": text}
    set_clauses = [f"{id_prop}: $id", "text: $text"]
    if embedding is not None:
        set_clauses.append("embedding: vecf32($embedding)")
        props["embedding"] = list(embedding)
    if embedding_model is not None:
        set_clauses.append("embeddingModel: $embeddingModel")
        props["embeddingModel"] = embedding_model
    graph.query(f"CREATE (n:{label} {{{', '.join(set_clauses)}}})", props)


def _old_embedding(dim: int = TEST_EMBEDDING_DIM) -> list[float]:
    return [0.5] * dim


# ── test 6: the step-0 traffic-stop precondition — zero graph/HTTP access ────

def test_migrate_without_traffic_stopped_aborts_before_any_graph_or_http_access():
    with pytest.raises(embedding_migration.MigrationAbortedError):
        embedding_migration.migrate(
            "test", _TARGET_REF, traffic_stopped=False,
            repo=_ExplodingMigrateRepo(), gateway=_ExplodingMigrateGateway(),
        )


def test_migrate_default_repo_and_gateway_never_constructed_without_traffic_stopped(
    monkeypatch,
):
    # Even with repo=/gateway=None (the CLI's real path), _default_repo() and
    # ModelGateway.from_env() must never be called — patch both to explode.
    def _boom(*_a, **_kw):
        raise AssertionError("must not build a repo/gateway before the traffic-stop check")

    monkeypatch.setattr(embedding_migration, "_default_repo", _boom)
    monkeypatch.setattr(
        embedding_migration.ModelGateway, "from_env", classmethod(lambda cls, **kw: _boom())
    )
    with pytest.raises(embedding_migration.MigrationAbortedError):
        embedding_migration.migrate("test", _TARGET_REF, traffic_stopped=False)


# ── test 12: target ref's dim not declared — abort before any HTTP/graph work ─

def test_migrate_missing_target_dim_aborts_before_embedding_or_graph_write(
    migrate_repo, conn,
):
    _seed_row(conn, "test", label="Message", id_prop="msgId", id_value="m1", text="hi")
    gateway = _FakeGateway(dims={})  # target_ref has no declared dim
    with pytest.raises(embedding_migration.MigrationAbortedError, match=_TARGET_REF):
        embedding_migration.migrate(
            "test", _TARGET_REF, traffic_stopped=True, repo=migrate_repo, gateway=gateway,
        )
    # No embedder was ever built, and the row is untouched.
    assert gateway.embedder_calls == []
    on_disk = migrate_repo.read_model_overrides("test")
    assert on_disk["embeddingModel"] is None


# ── test 5 / 5a: full re-embed + index rebuild, including the self-healing
# unembedded-row path ────────────────────────────────────────────────────────

def test_migrate_reembeds_every_row_and_rebuilds_both_indexes(migrate_repo, conn):
    _seed_row(conn, "test", label="Message", id_prop="msgId", id_value="m1",
              text="hello", embedding=_old_embedding())
    _seed_row(conn, "test", label="Message", id_prop="msgId", id_value="m2",
              text="world", embedding=_old_embedding())
    _seed_row(conn, "test", label="Chunk", id_prop="chunkId", id_value="c1",
              text="chunk one", embedding=_old_embedding())
    # 5a: one Message and one Chunk row never embedded at all — self-healing.
    _seed_row(conn, "test", label="Message", id_prop="msgId", id_value="m3", text="never embedded")
    _seed_row(conn, "test", label="Chunk", id_prop="chunkId", id_value="c2", text="never embedded chunk")

    gateway = _FakeGateway(dims={_TARGET_REF: _TARGET_DIM})
    report = embedding_migration.migrate(
        "test", _TARGET_REF, traffic_stopped=True, repo=migrate_repo, gateway=gateway,
    )

    assert report.labels["Message"].migrated == 3
    assert report.labels["Message"].total == 3
    assert report.labels["Chunk"].migrated == 2
    assert report.labels["Chunk"].total == 2
    assert report.labels["Message"].skipped == 0
    assert report.labels["Chunk"].skipped == 0

    graph = db.workspace_graph(conn, "test")
    for label, id_prop, ids in (
        ("Message", "msgId", ("m1", "m2", "m3")),
        ("Chunk", "chunkId", ("c1", "c2")),
    ):
        for row_id in ids:
            res = graph.query(
                f"MATCH (n:{label} {{{id_prop}: $id}}) RETURN n.embeddingModel",
                {"id": row_id},
            )
            assert res.result_set[0][0] == _TARGET_REF

    assert migrate_repo.read_index_dimension("test", label="Message") == _TARGET_DIM
    assert migrate_repo.read_index_dimension("test", label="Chunk") == _TARGET_DIM

    on_disk = migrate_repo.read_model_overrides("test")
    assert on_disk["embeddingModel"] == _TARGET_REF


# ── regression: each row's WRITTEN embedding VALUE must come from that row's
# own text, not its id or another row's text — none of the checks above (row
# counts, `embeddingModel` marker, index dimension) would notice an
# `embedder.embed(row_id)`/swapped-text/shuffled-batch defect, since none of
# them inspect the actual vector. `_FakeEmbedder.embed` is deterministic and
# hash-derived specifically so this is pinnable without a real network call. ──

def test_migrate_writes_the_embedding_computed_from_each_rows_own_text(migrate_repo, conn):
    rows = [
        ("Message", "msgId", "m1", "the quick brown fox"),
        ("Message", "msgId", "m2", "jumps over the lazy dog"),
        ("Chunk", "chunkId", "c1", "an entirely different chunk of text"),
    ]
    for label, id_prop, id_value, text in rows:
        _seed_row(conn, "test", label=label, id_prop=id_prop, id_value=id_value,
                  text=text, embedding=_old_embedding())

    gateway = _FakeGateway(dims={_TARGET_REF: _TARGET_DIM})
    embedding_migration.migrate(
        "test", _TARGET_REF, traffic_stopped=True, repo=migrate_repo, gateway=gateway,
    )

    # A fresh, independent `_FakeEmbedder` instance — never touched by
    # `migrate()` itself — computes the expected vector for each row's own
    # text; `embed()`'s output depends only on `(text, dim)`, never on call
    # count, so this is a true independent oracle, not a tautology.
    reference_embedder = _FakeEmbedder(dim=_TARGET_DIM)
    graph = db.workspace_graph(conn, "test")
    for label, id_prop, id_value, text in rows:
        expected = reference_embedder.embed(text)
        res = graph.query(
            f"MATCH (n:{label} {{{id_prop}: $id}}) RETURN n.embedding",
            {"id": id_value},
        )
        actual = res.result_set[0][0]
        assert actual == pytest.approx(expected, rel=1e-5), (
            f"{label} {id_value!r}'s stored embedding does not match "
            f"embed({text!r}) — wrong text (or id/another row's text) was embedded"
        )


# ── test 7: the hard-cap-bypass regression — the plan's top-priority test ────

def test_migrate_bypasses_the_workspace_hard_cap(migrate_repo, conn):
    _seed_row(conn, "test", label="Message", id_prop="msgId", id_value="m1",
              text="hello", embedding=_old_embedding())
    migrate_repo.write_model_overrides("test", embedding=_OLD_REF, at=1, by="test-setup")

    old_embedder = _FakeEmbedder(dim=TEST_EMBEDDING_DIM)
    new_embedder = _FakeEmbedder(dim=_TARGET_DIM)
    gateway = _FakeGateway(
        dims={_OLD_REF: TEST_EMBEDDING_DIM, _TARGET_REF: _TARGET_DIM},
        embedders={_OLD_REF: old_embedder, _TARGET_REF: new_embedder},
    )

    embedding_migration.migrate(
        "test", _TARGET_REF, traffic_stopped=True, repo=migrate_repo, gateway=gateway,
    )

    assert new_embedder.calls > 0
    assert old_embedder.calls == 0
    assert all(c["ws"] is None and c["overrides"] is None for c in gateway.resolve_calls)
    assert all(c["ws"] is None and c["overrides"] is None for c in gateway.embedder_calls)


# ── test 8: idempotent resume after a mid-batch crash ────────────────────────

def test_migrate_resumes_after_a_mid_batch_crash_without_reprocessing(migrate_repo, conn):
    for i in range(5):
        _seed_row(conn, "test", label="Message", id_prop="msgId", id_value=f"m{i}",
                  text=f"row {i}", embedding=_old_embedding())

    crashing_embedder = _FakeEmbedder(dim=_TARGET_DIM, raise_after=3)
    gateway = _FakeGateway(dims={_TARGET_REF: _TARGET_DIM}, embedders={_TARGET_REF: crashing_embedder})

    with pytest.raises(RuntimeError, match="simulated embedder crash"):
        embedding_migration.migrate(
            "test", _TARGET_REF, traffic_stopped=True, batch_size=50,
            repo=migrate_repo, gateway=gateway,
        )

    # Steps d/e/f never reached: index and override untouched.
    assert migrate_repo.read_index_dimension("test", label="Message") == TEST_EMBEDDING_DIM
    assert migrate_repo.read_model_overrides("test")["embeddingModel"] is None
    unmigrated, total = embedding_migration._count_unmigrated(
        migrate_repo, "test", label="Message", target_ref=_TARGET_REF,
    )
    assert total == 5
    assert 0 < unmigrated < 5  # some rows landed before the crash, not all

    resuming_embedder = _FakeEmbedder(dim=_TARGET_DIM)
    gateway2 = _FakeGateway(dims={_TARGET_REF: _TARGET_DIM}, embedders={_TARGET_REF: resuming_embedder})
    report = embedding_migration.migrate(
        "test", _TARGET_REF, traffic_stopped=True, repo=migrate_repo, gateway=gateway2,
    )

    assert report.labels["Message"].migrated == 5
    assert report.labels["Message"].total == 5
    assert migrate_repo.read_index_dimension("test", label="Message") == _TARGET_DIM
    assert migrate_repo.read_model_overrides("test")["embeddingModel"] == _TARGET_REF
    # The resuming call only re-embedded the rows the crashed call never reached.
    assert resuming_embedder.calls < 5


# ── test 9: the DROP-guard resume hazard — crash between DROP and CREATE ─────

def test_migrate_resume_after_index_dropped_but_not_recreated(migrate_repo, conn):
    # Fully-migrated data already (as if a prior `migrate` call finished the
    # re-embed and count-check steps), but the index DROP happened and the
    # CREATE did not — the exact intermediate state `-graph.md` item 5 flags.
    _seed_row(conn, "test", label="Message", id_prop="msgId", id_value="m1",
              text="hello", embedding=[0.1] * _TARGET_DIM, embedding_model=_TARGET_REF)
    _seed_row(conn, "test", label="Chunk", id_prop="chunkId", id_value="c1",
              text="chunk", embedding=[0.1] * _TARGET_DIM, embedding_model=_TARGET_REF)
    graph = db.workspace_graph(conn, "test")
    graph.query("DROP VECTOR INDEX FOR (n:Message) ON (n.embedding)")
    graph.query("DROP VECTOR INDEX FOR (n:Chunk) ON (n.embedding)")
    assert migrate_repo.read_index_dimension("test", label="Message") is None
    assert migrate_repo.read_index_dimension("test", label="Chunk") is None

    gateway = _FakeGateway(dims={_TARGET_REF: _TARGET_DIM})
    # Must NOT hard-error on an unconditional DROP with nothing to drop.
    report = embedding_migration.migrate(
        "test", _TARGET_REF, traffic_stopped=True, repo=migrate_repo, gateway=gateway,
    )

    assert report.labels["Message"].migrated == 1
    assert report.labels["Chunk"].migrated == 1
    assert migrate_repo.read_index_dimension("test", label="Message") == _TARGET_DIM
    assert migrate_repo.read_index_dimension("test", label="Chunk") == _TARGET_DIM


# ── test: idempotent re-run reports zero rows processed ──────────────────────

def test_migrate_rerun_after_success_is_idempotent(migrate_repo, conn):
    _seed_row(conn, "test", label="Message", id_prop="msgId", id_value="m1",
              text="hello", embedding=_old_embedding())
    gateway1 = _FakeGateway(dims={_TARGET_REF: _TARGET_DIM})
    first = embedding_migration.migrate(
        "test", _TARGET_REF, traffic_stopped=True, repo=migrate_repo, gateway=gateway1,
    )
    assert first.labels["Message"].migrated == 1

    embedder2 = _FakeEmbedder(dim=_TARGET_DIM)
    gateway2 = _FakeGateway(dims={_TARGET_REF: _TARGET_DIM}, embedders={_TARGET_REF: embedder2})
    second = embedding_migration.migrate(
        "test", _TARGET_REF, traffic_stopped=True, repo=migrate_repo, gateway=gateway2,
    )

    assert second.labels["Message"].migrated == 1
    assert second.labels["Message"].total == 1
    assert embedder2.calls == 0  # nothing left to embed


# ── test 10: count check reports the actual nonzero count mid-migration ─────

def test_migrate_count_check_reports_actual_unmigrated_counts(migrate_repo, conn):
    for i in range(3):
        _seed_row(conn, "test", label="Message", id_prop="msgId", id_value=f"m{i}",
                  text=f"row {i}", embedding=_old_embedding())
    unmigrated, total = embedding_migration._count_unmigrated(
        migrate_repo, "test", label="Message", target_ref=_TARGET_REF,
    )
    assert (unmigrated, total) == (3, 3)

    embedding_migration._write_embedding(
        migrate_repo, "test", label="Message", id_prop="msgId", id_value="m0",
        embedding=[0.2] * _TARGET_DIM, target_ref=_TARGET_REF,
    )
    unmigrated, total = embedding_migration._count_unmigrated(
        migrate_repo, "test", label="Message", target_ref=_TARGET_REF,
    )
    assert (unmigrated, total) == (2, 3)


def test_migrate_aborts_before_index_rebuild_when_count_check_finds_leftover_rows(
    monkeypatch, migrate_repo, conn,
):
    # Defense in depth: even if step c (re-embed) had a defect and silently
    # left a row behind, the count check must still catch it and refuse to
    # touch the index/override — never trust the loop having "finished" alone.
    _seed_row(conn, "test", label="Message", id_prop="msgId", id_value="m1",
              text="hello", embedding=_old_embedding())
    monkeypatch.setattr(embedding_migration, "_reembed_label", lambda *a, **kw: 0)

    gateway = _FakeGateway(dims={_TARGET_REF: _TARGET_DIM})
    with pytest.raises(embedding_migration.MigrationAbortedError, match="unmigrated"):
        embedding_migration.migrate(
            "test", _TARGET_REF, traffic_stopped=True, repo=migrate_repo, gateway=gateway,
        )

    assert migrate_repo.read_index_dimension("test", label="Message") == TEST_EMBEDDING_DIM
    assert migrate_repo.read_model_overrides("test")["embeddingModel"] is None


# ── test 11: zero Message/Chunk nodes — completes cleanly, index still rebuilt

def test_migrate_on_an_empty_workspace_completes_cleanly(migrate_repo):
    gateway = _FakeGateway(dims={_TARGET_REF: _TARGET_DIM})
    report = embedding_migration.migrate(
        "test", _TARGET_REF, traffic_stopped=True, repo=migrate_repo, gateway=gateway,
    )
    assert report.labels["Message"].migrated == 0
    assert report.labels["Message"].total == 0
    assert report.labels["Chunk"].migrated == 0
    assert report.labels["Chunk"].total == 0
    assert migrate_repo.read_index_dimension("test", label="Message") == _TARGET_DIM
    assert migrate_repo.read_index_dimension("test", label="Chunk") == _TARGET_DIM


# ── test 13: vanished-row skip-and-log ───────────────────────────────────────

def test_migrate_skips_and_logs_a_row_that_vanishes_between_read_and_write(
    migrate_repo, conn, capsys,
):
    _seed_row(conn, "test", label="Message", id_prop="msgId", id_value="m1",
              text="hello", embedding=_old_embedding())
    _seed_row(conn, "test", label="Message", id_prop="msgId", id_value="m2",
              text="world", embedding=_old_embedding())

    real_graph = migrate_repo._graph("test")
    migrate_repo._graph = lambda ws: _VanishingRowGraph(
        real_graph, label="Message", id_prop="msgId", vanish_id="m1",
    )

    gateway = _FakeGateway(dims={_TARGET_REF: _TARGET_DIM})
    report = embedding_migration.migrate(
        "test", _TARGET_REF, traffic_stopped=True, repo=migrate_repo, gateway=gateway,
    )

    assert report.labels["Message"].skipped == 1
    # The vanished row was never marked migrated; it's correctly absent from
    # `migrated` in the final count.
    assert report.labels["Message"].migrated == 1
    assert report.labels["Message"].total == 1  # m1 was actually deleted
    captured = capsys.readouterr()
    assert "vanished" in captured.out
    assert "m1" in captured.out


# ── test 14: FR-5's read-before-write discipline, on the migrate side ────────

def test_migrate_write_back_preserves_other_override_kinds(migrate_repo, conn):
    _seed_row(conn, "test", label="Message", id_prop="msgId", id_value="m1",
              text="hello", embedding=_old_embedding())
    migrate_repo.write_model_overrides(
        "test", agent="lmstudio/agent-override", guard="lmstudio/guard-override",
        responder="lmstudio/responder-override", embedding=_OLD_REF,
        at=1, by="test-setup",
    )

    gateway = _FakeGateway(dims={_TARGET_REF: _TARGET_DIM})
    embedding_migration.migrate(
        "test", _TARGET_REF, traffic_stopped=True, repo=migrate_repo, gateway=gateway,
    )

    on_disk = migrate_repo.read_model_overrides("test")
    assert on_disk["embeddingModel"] == _TARGET_REF
    assert on_disk["agentModel"] == "lmstudio/agent-override"
    assert on_disk["guardModel"] == "lmstudio/guard-override"
    assert on_disk["responderModel"] == "lmstudio/responder-override"
