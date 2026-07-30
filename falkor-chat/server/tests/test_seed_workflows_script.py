"""Regression guard for K-037: `scripts/seed_workflows.sh`'s triage-literal publish
identity must stay decoupled from `FALKORCHAT_TRIGGER_DEF_KEY`/`_VERSION`.

**The bug this pins:** the script used to read `FALKORCHAT_TRIGGER_DEF_KEY`/`_VERSION` —
the app's real `config.TRIGGER_DEF_KEY`/`_VERSION` `@mention`-trigger override — to decide
what key/version to publish its own inline `triage`-literal step spec under. Overriding
the trigger for a demo/QA session (e.g. to point it at `access-request@v1`) therefore also
redirected the triage literal's publish target: the `WorkflowDef` node reported "already
present — no-op" (it already existed), but the per-step `MERGE` underneath is keyed by
`stepUid`, so 3 spurious `Step`s + a second `START` edge got silently grafted onto the
unrelated, already-published def (`docs/test-reports/web-api-coverage-report.md` Finding
1; `docs/BACKLOG.md` K-037). The fix decoupled the triage-literal's identity into its own
dedicated `FALKORCHAT_TRIAGE_DEF_KEY`/`_VERSION` pair — this test is the durable guard
against someone "simplifying" that back onto `FALKORCHAT_TRIGGER_DEF_KEY` later.

**Network-free, no `live` marker needed.** Unlike `test_workflow_live.py` (which shells
out to this same script but then drives the executor against a real LM Studio), this only
exercises the seed script itself plus structural Cypher reads — no LLM involved. Runs as
part of the ordinary `pytest -q` baseline.

**Throwaway identities, never the real `triage`/`access-request` keys**, even though
`reference` is a shared graph the `wf_repo` fixture already wipes at setup: this keeps the
test's own assertions unambiguous (nothing else could have contributed to
`k037-*-guard`'s step count) and means a solo run of just this module can't collide with
anything a parallel/adjacent workflow test might have left behind. Cleans up its own
throwaway defs at teardown regardless of outcome — see `_cleanup_throwaway_defs` below.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from falkorchat import db
from falkorchat.config import CallContext
from falkorchat.services import Services

CTX = CallContext(ws="test", actor="u1")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SEED_WORKFLOWS = _REPO_ROOT / "scripts" / "seed_workflows.sh"

# Throwaway, test-only identities — distinct from the real `triage`/`access-request` keys
# so this test's writes into the shared `reference` graph are unambiguous and never touch
# production names.
TRIAGE_KEY, TRIAGE_VERSION = "k037-triage-guard", "v1"
TARGET_KEY, TARGET_VERSION = "k037-target-guard", "v1"


def _run_seed(*, trigger_key: str | None = None, trigger_version: str | None = None) -> None:
    """Shell out to the REAL `scripts/seed_workflows.sh` against `ws:test` — the same
    script `start_server.sh` runs on every startup, never a reimplementation of its logic
    (the script's own header flags that drift risk).

    `FALKORCHAT_TRIAGE_DEF_KEY`/`_VERSION` are always passed EXPLICITLY (never left to the
    script's own default) — belt-and-suspenders: this proves the triage-literal's identity
    is actually read from its own dedicated pair on every invocation, not merely
    unaffected by accident because it happened to match a bash default. A regression that
    silently reverted to reading `FALKORCHAT_TRIGGER_DEF_KEY` again would show up
    immediately as the target def changing shape below, precisely because this pins
    `TRIAGE_KEY` and `TARGET_KEY` to two different values.

    `FALKORCHAT_PROCESS_DEF_KEY`/`_VERSION` are likewise pinned to `TARGET_KEY`/`_VERSION`
    on every call, so the "other def" `FALKORCHAT_TRIGGER_DEF_KEY` collides with is always
    this test's own throwaway def, never the real `access-request@v1`.
    """
    env = {
        **os.environ,
        "FALKORCHAT_WS_ID": "test",
        "FALKORCHAT_TRIAGE_DEF_KEY": TRIAGE_KEY,
        "FALKORCHAT_TRIAGE_DEF_VERSION": TRIAGE_VERSION,
        "FALKORCHAT_PROCESS_DEF_KEY": TARGET_KEY,
        "FALKORCHAT_PROCESS_DEF_VERSION": TARGET_VERSION,
    }
    if trigger_key is not None:
        env["FALKORCHAT_TRIGGER_DEF_KEY"] = trigger_key
        env["FALKORCHAT_TRIGGER_DEF_VERSION"] = trigger_version or "v1"
    proc = subprocess.run(
        ["bash", str(_SEED_WORKFLOWS), "test"],
        capture_output=True, text=True, env=env,
    )
    assert proc.returncode == 0, (
        f"seed_workflows.sh failed (exit {proc.returncode})\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )


@pytest.fixture()
def _cleanup_throwaway_defs(wf_repo):
    """`wf_repo` already wiped `reference`/`ws:test` at setup (conftest.py). Also clean up
    at teardown, unconditionally, so a solo run of just this module (e.g. `pytest -k
    k037`) doesn't leave the throwaway `k037-*-guard` defs sitting in the shared
    `reference` graph for whatever runs next.
    """
    yield wf_repo
    conn = db.connect()
    for key, version in ((TRIAGE_KEY, TRIAGE_VERSION), (TARGET_KEY, TARGET_VERSION)):
        db.reference_graph(conn).query(
            "MATCH (d:WorkflowDef {key: $key, version: $version}) "
            "OPTIONAL MATCH (d)-[:HAS_STEP]->(s:Step) "
            "DETACH DELETE d, s",
            {"key": key, "version": version},
        )
        db.workspace_graph(conn, "test").query(
            "MATCH (d:WorkflowDefSnapshot {key: $key, version: $version}) "
            "OPTIONAL MATCH (d)-[:HAS_STEP]->(s:Step) "
            "DETACH DELETE d, s",
            {"key": key, "version": version},
        )


def test_trigger_def_key_override_does_not_graft_onto_the_other_def(_cleanup_throwaway_defs):
    """K-037 regression guard — the backlog item's own test-strategy note, automated.

    Baseline run (no trigger override): `TARGET_KEY`/`_VERSION` (playing
    `access-request@v1`'s role) and `TRIAGE_KEY`/`_VERSION` (playing `triage@v1`'s role)
    are both created fresh, at their canonical shape.

    Collision run: re-seed with `FALKORCHAT_TRIGGER_DEF_KEY`/`_VERSION` pointed at
    `TARGET_KEY`/`_VERSION` — exactly K-037's Pass-B demo/QA override
    (`FALKORCHAT_TRIGGER_DEF_KEY=access-request`), replayed against throwaway keys.
    Pre-fix, this redirected the triage-literal's publish identity onto the target too
    (same env var read for both), grafting 3 spurious `Step`s + a second `START` edge.
    Post-fix, the target must come out byte-for-byte identical to the baseline, and the
    triage-literal's own def must be an unchanged, idempotent no-op.
    """
    services = Services(_cleanup_throwaway_defs)

    _run_seed()

    target_before = services.get_workflow_def_structure(
        CTX, key=TARGET_KEY, version=TARGET_VERSION
    )
    triage_before = services.get_workflow_def_structure(
        CTX, key=TRIAGE_KEY, version=TRIAGE_VERSION
    )
    assert target_before["stepCount"] == 6
    assert "startKeys" not in target_before  # exactly one START edge
    assert triage_before["stepCount"] == 3
    assert "startKeys" not in triage_before

    _run_seed(trigger_key=TARGET_KEY, trigger_version=TARGET_VERSION)

    target_after = services.get_workflow_def_structure(
        CTX, key=TARGET_KEY, version=TARGET_VERSION
    )
    assert target_after == target_before, (
        "FALKORCHAT_TRIGGER_DEF_KEY pointed at the target def changed its structure — "
        f"K-037 regression. before={target_before!r} after={target_after!r}"
    )

    triage_after = services.get_workflow_def_structure(
        CTX, key=TRIAGE_KEY, version=TRIAGE_VERSION
    )
    assert triage_after == triage_before, (
        "the triage-literal's own def changed shape across an idempotent re-run — "
        f"before={triage_before!r} after={triage_after!r}"
    )
