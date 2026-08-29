#!/usr/bin/env bash
# verify_salesperson.sh — check that `reference` and ws:<id> agree on BOTH the
# `salesperson@<version>` def AND its FR-6/FR-9 sibling `order-fulfillment@<version>`
# (K-052/K-053 M6). The one-command form of seed_salesperson.sh's own re-seed
# discipline, scoped to the two defs that script seeds (mirrors
# verify_workflows.sh, which covers `triage`+`access-request`).
#
# Usage:
#   ./scripts/verify_salesperson.sh [<workspaceId>]   # default: $FALKORCHAT_WS_ID or "acme"
#
# ⚠️ STRICTLY READ-ONLY — publishes nothing, materializes nothing, deletes
# nothing. When a def is missing it prints the `seed_salesperson.sh` command
# and exits non-zero; running that command is the operator's decision, not
# this script's. A DIVERGENCE is not repaired by re-seeding either (a
# create-only re-publish cannot overwrite anything, and deleting a snapshot
# breaks live WorkflowRuns via OF_DEF/AT_STEP) — report it instead.
#
# WHY a Python one-shot over the SERVICE LAYER rather than HTTP (mirrors
# verify_workflows.sh's own rationale): the moment this check is most needed is
# right after a `pytest` / `test_queries.sh` run, before uvicorn is back up. It
# drives the same `services.*` methods the REST `/diff` route does.
#
# For EACH def (default versions `salesperson@v3`, `order-fulfillment@v1` —
# override with FALKORCHAT_SALESPERSON_DEF_VERSION /
# FALKORCHAT_ORDER_FULFILLMENT_DEF_VERSION, the same knobs seed_salesperson.sh
# reads), it checks:
#   1. the def is published in `reference` AT THE EXPECTED VERSION;
#   2. the snapshot exists in ws:<id> at that version;
#   3. the two are `inSync` per `services.diff_def_snapshot`, printing every
#      difference;
#   4. the def declares exactly ONE start key (`startKeys` present ⇒ more than
#      one START edge ⇒ a re-publish added one — K-034);
#   5. the topology is exactly what each def specifies:
#        salesperson:       2 steps (assistant/agent, ended/decision), 1 transition;
#        order-fulfillment: 4 steps (placed/decision, fulfilled/human,
#                            delivered/decision, cancelled/decision), 3 transitions.
#
# Exit 0 = all green. Exit 1 = anything missing or divergent.
#
# Env vars (all optional):
#   FALKORDB_HOST                            (default: 127.0.0.1)
#   FALKORDB_PORT                            (default: 6379)
#   FALKORCHAT_WS_ID                         (default: acme)
#   FALKORCHAT_SALESPERSON_DEF_KEY           (default: salesperson)
#   FALKORCHAT_SALESPERSON_DEF_VERSION       (default: v3)
#   FALKORCHAT_ORDER_FULFILLMENT_DEF_KEY     (default: order-fulfillment)
#   FALKORCHAT_ORDER_FULFILLMENT_DEF_VERSION (default: v1)

set -euo pipefail

HOST="${FALKORDB_HOST:-127.0.0.1}"
PORT="${FALKORDB_PORT:-6379}"
WS_ID="${1:-${FALKORCHAT_WS_ID:-acme}}"
SALESPERSON_DEF_KEY="${FALKORCHAT_SALESPERSON_DEF_KEY:-salesperson}"
SALESPERSON_DEF_VERSION="${FALKORCHAT_SALESPERSON_DEF_VERSION:-v3}"
ORDER_FULFILLMENT_DEF_KEY="${FALKORCHAT_ORDER_FULFILLMENT_DEF_KEY:-order-fulfillment}"
ORDER_FULFILLMENT_DEF_VERSION="${FALKORCHAT_ORDER_FULFILLMENT_DEF_VERSION:-v1}"

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SERVER_DIR="$REPO_DIR/server"
VENV_PY="$SERVER_DIR/.venv/bin/python"

if [ ! -x "$VENV_PY" ]; then
  echo "ERROR: server venv not found at $VENV_PY" >&2
  echo "       Create it first:  cd server && python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'" >&2
  exit 1
fi

echo "Checking FalkorDB at ${HOST}:${PORT}..."
redis-cli -h "$HOST" -p "$PORT" PING | grep -q PONG || {
  echo "ERROR: cannot reach FalkorDB at ${HOST}:${PORT}" >&2
  exit 1
}

echo "── verifying workflow defs '${SALESPERSON_DEF_KEY}@${SALESPERSON_DEF_VERSION}' + '${ORDER_FULFILLMENT_DEF_KEY}@${ORDER_FULFILLMENT_DEF_VERSION}': reference vs ws:${WS_ID} ──"

FALKORCHAT_WS_ID="$WS_ID" \
FALKORDB_HOST="$HOST" FALKORDB_PORT="$PORT" \
FALKORCHAT_SALESPERSON_DEF_KEY="$SALESPERSON_DEF_KEY" \
FALKORCHAT_SALESPERSON_DEF_VERSION="$SALESPERSON_DEF_VERSION" \
FALKORCHAT_ORDER_FULFILLMENT_DEF_KEY="$ORDER_FULFILLMENT_DEF_KEY" \
FALKORCHAT_ORDER_FULFILLMENT_DEF_VERSION="$ORDER_FULFILLMENT_DEF_VERSION" \
"$VENV_PY" - <<'PY'
import os
import sys

from redis.exceptions import ResponseError

from falkorchat import config, db
from falkorchat.repository import Repository
from falkorchat.services import Services, WorkflowDefNotFoundError

# The two (key, version) pairs seed_salesperson.sh publishes/materializes —
# expected TOPOLOGY per def, mirroring verify_workflows.sh's own per-def
# structure table, but declared inline here (there is no shared
# DEMO_EXPECTED_DEFS-style constant for this script's own pair — see
# services.DEMO_EXPECTED_DEFS's own comment: it is scoped to the
# triage/access-request pair verify_workflows.sh checks).
DEFS = [
    (
        os.environ["FALKORCHAT_SALESPERSON_DEF_KEY"],
        os.environ["FALKORCHAT_SALESPERSON_DEF_VERSION"],
        {"assistant": "agent", "ended": "decision"},
        "assistant",
        1,
    ),
    (
        os.environ["FALKORCHAT_ORDER_FULFILLMENT_DEF_KEY"],
        os.environ["FALKORCHAT_ORDER_FULFILLMENT_DEF_VERSION"],
        {
            "placed": "decision", "fulfilled": "human",
            "delivered": "decision", "cancelled": "decision",
        },
        "placed",
        3,
    ),
]

services = Services(Repository(db.connect()))
ctx = config.get_context()

ABSENT = {"defPresent": False, "snapshotPresent": False, "inSync": False,
          "differences": [], "differenceCount": 0}


def read(fn, absent=None):
    """Read-only probe: a cold graph key or an absent def is 'nothing there'."""
    try:
        return fn()
    except WorkflowDefNotFoundError:
        return absent
    except ResponseError as exc:
        if "empty key" in str(exc):
            return absent
        raise


failures = []

for key, version, expected_steps, expected_start, expected_transition_count in DEFS:
    label = f"{key}@{version}"
    diff = read(
        lambda: services.diff_def_snapshot(ctx, key=key, version=version), absent=ABSENT,
    )
    def_present, snap_present = diff["defPresent"], diff["snapshotPresent"]

    print(f"\n  {label}")
    snap_label = f"ws:{ctx.ws} snapshot"
    width = max(len("reference def"), len(snap_label), len("in sync"), len("topology"))
    print(f"    {'reference def':<{width}} : {'present' if def_present else 'MISSING'}")
    print(f"    {snap_label:<{width}} : {'present' if snap_present else 'MISSING'}")

    if not def_present:
        failures.append(f"{label}: not published in `reference` at this version")
    if not snap_present:
        failures.append(f"{label}: not materialized into ws:{ctx.ws} at this version")

    if def_present and snap_present:
        if diff["inSync"]:
            print(f"    {'in sync':<{width}} : YES")
        else:
            print(f"    {'in sync':<{width}} : NO ({diff['differenceCount']} differences)")
            for d in diff["differences"]:
                print(f"      - {d['path']}")
                print(f"          def      = {d['def']!r}")
                print(f"          snapshot = {d['snapshot']!r}")
            failures.append(
                f"{label}: reference def and ws:{ctx.ws} snapshot diverge "
                f"({diff['differenceCount']} differences)"
            )

    # Finding-3 tripwire: `startKeys` is emitted only when a root carries more
    # than one START edge, which a create-only re-publish can add (K-034).
    sides = (
        ("reference def",
         lambda: services.get_workflow_def_structure(ctx, key=key, version=version)),
        (f"ws:{ctx.ws} snapshot",
         lambda: services.get_snapshot_structure(ctx, key=key, version=version)),
    )
    for side, reader in sides:
        structure = read(reader)
        if structure and "startKeys" in structure:
            starts = structure["startKeys"]
            print(f"    ⚠ {side}: {len(starts)} START edges {starts} — expected exactly one")
            failures.append(
                f"{label}: {side} has {len(starts)} START edges "
                f"({', '.join(starts)}) — see K-034"
            )

    # Topology sanity — each def's own exact shape (per DEFS above).
    if snap_present:
        snap = services.get_snapshot(ctx, key=key, version=version)
        step_types = {s["key"]: s["type"] for s in snap["steps"]}
        if step_types != expected_steps:
            print(f"    ⚠ step topology: {step_types} (expected {expected_steps})")
            failures.append(f"{label}: unexpected step topology {step_types}")
        if snap["start_key"] != expected_start:
            print(f"    ⚠ start key: {snap['start_key']!r} (expected {expected_start!r})")
            failures.append(f"{label}: unexpected start key {snap['start_key']!r}")
        if len(snap["transitions"]) != expected_transition_count:
            print(
                f"    ⚠ transition count: {len(snap['transitions'])} "
                f"(expected {expected_transition_count})"
            )
            failures.append(
                f"{label}: unexpected transition count {len(snap['transitions'])}"
            )
        else:
            steps_desc = ", ".join(f"{k}/{t}" for k, t in expected_steps.items())
            print(
                f"    {'topology':<{width}} : {len(expected_steps)} steps "
                f"({steps_desc}), {expected_transition_count} transition(s) — OK"
            )

print("")
if failures:
    print("RESULT: FAIL")
    for f in failures:
        print(f"  ✗ {f}")
    print("")
    print("If a def or snapshot is MISSING, re-seed:")
    print(f"  ./scripts/seed_salesperson.sh {ctx.ws}")
    print("If they DIVERGE, do NOT re-seed — a create-only re-publish cannot overwrite")
    print("the stored def, and deleting a snapshot breaks live WorkflowRuns. Report it.")
    sys.exit(1)

print(f"RESULT: OK — {len(DEFS)} defs in sync between `reference` and ws:{ctx.ws}")
PY
