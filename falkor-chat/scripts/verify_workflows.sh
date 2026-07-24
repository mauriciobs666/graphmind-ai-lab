#!/usr/bin/env bash
# verify_workflows.sh — check that `reference` and ws:<id> agree about the seeded
# workflow defs (K-031). The one-command form of the re-seed discipline documented on
# `seed_workflows.sh`.
#
# Usage:
#   ./scripts/verify_workflows.sh [<workspaceId>]   # default: $FALKORCHAT_WS_ID or "acme"
#
# ⚠️ STRICTLY READ-ONLY — this script must never gain a "let me just re-seed that for
# you" fallback. It publishes nothing, materializes nothing, deletes nothing. When a def
# is missing it prints the `seed_workflows.sh` command and exits non-zero; running that
# command is the operator's decision, not this script's. Repairing a *divergence* is not
# a re-seed at all (a create-only re-publish cannot overwrite anything) and deleting a
# snapshot breaks live WorkflowRuns via OF_DEF/AT_STEP — a destructive shared-state op.
#
# WHY a Python one-shot over the SERVICE LAYER rather than HTTP: the moment this check is
# most needed is right after a `pytest` / `test_queries.sh` run, before uvicorn is back
# up. It drives the same `services.*` methods the three REST routes do, so a green run
# here and a green `GET …/diff` mean the same thing.
#
# Per def — `triage@v1` (from config.TRIGGER_DEF_KEY/TRIGGER_DEF_VERSION) and
# `access-request@v1` (from falkorchat.proof_defs.ACCESS_REQUEST_DEF), the same pair
# `seed_workflows.sh` seeds — it checks:
#   1. the def is published in `reference` AT THE EXPECTED VERSION (the version-staleness
#      check the /diff endpoint structurally cannot do — it is version-qualified);
#   2. the snapshot exists in ws:<id> at that version;
#   3. the two are `inSync` per `services.diff_def_snapshot`, printing every difference;
#   4. the def declares exactly ONE start key (`startKeys` present ⇒ more than one START
#      edge ⇒ a re-publish added one — see K-034).
#
# Exit 0 = all green. Exit 1 = anything missing or divergent.
#
# Env vars (all optional):
#   FALKORDB_HOST                  (default: 127.0.0.1)
#   FALKORDB_PORT                  (default: 6379)
#   FALKORCHAT_WS_ID               (default: acme)
#   FALKORCHAT_TRIGGER_DEF_KEY     (default: triage)  — must match config.TRIGGER_DEF_KEY
#   FALKORCHAT_TRIGGER_DEF_VERSION (default: v1)      — must match config.TRIGGER_DEF_VERSION

set -euo pipefail

HOST="${FALKORDB_HOST:-127.0.0.1}"
PORT="${FALKORDB_PORT:-6379}"
WS_ID="${1:-${FALKORCHAT_WS_ID:-acme}}"

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

echo "── verifying workflow defs: reference vs ws:${WS_ID} ──"

FALKORCHAT_WS_ID="$WS_ID" \
FALKORDB_HOST="$HOST" FALKORDB_PORT="$PORT" \
"$VENV_PY" - <<'PY'
import sys

from redis.exceptions import ResponseError

from falkorchat import config, db
from falkorchat.proof_defs import ACCESS_REQUEST_DEF
from falkorchat.repository import Repository
from falkorchat.services import Services, WorkflowDefNotFoundError

# The expected (key, version) pairs come from the SAME sources `seed_workflows.sh`
# publishes from, so this check cannot drift from the seed. Check 1 — "published at
# the EXPECTED version" — is exactly this pairing: the /diff route is version-
# qualified and can never answer it on its own.
DEFS = [
    (config.TRIGGER_DEF_KEY, config.TRIGGER_DEF_VERSION),
    (ACCESS_REQUEST_DEF["key"], ACCESS_REQUEST_DEF["version"]),
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

for key, version in DEFS:
    label = f"{key}@{version}"
    diff = read(
        lambda: services.diff_def_snapshot(ctx, key=key, version=version),
        absent=ABSENT,
    )
    def_present, snap_present = diff["defPresent"], diff["snapshotPresent"]

    print(f"\n  {label}")
    snap_label = f"ws:{ctx.ws} snapshot"
    width = max(len("reference def"), len(snap_label), len("in sync"))
    print(f"    {'reference def':<{width}} : "
          f"{'present' if def_present else 'MISSING'}")
    print(f"    {snap_label:<{width}} : "
          f"{'present' if snap_present else 'MISSING'}")

    if not def_present:
        failures.append(f"{label}: not published in `reference` at this version")
    if not snap_present:
        failures.append(f"{label}: not materialized into ws:{ctx.ws} at this version")

    if def_present and snap_present:
        if diff["inSync"]:
            print(f"    {'in sync':<{width}} : YES")
        else:
            print(f"    {'in sync':<{width}} : NO "
                  f"({diff['differenceCount']} differences)")
            for d in diff["differences"]:
                print(f"      - {d['path']}")
                print(f"          def      = {d['def']!r}")
                print(f"          snapshot = {d['snapshot']!r}")
            failures.append(
                f"{label}: reference def and ws:{ctx.ws} snapshot diverge "
                f"({diff['differenceCount']} differences)"
            )

    # Finding-3 tripwire: `startKeys` is emitted only when a root carries more than
    # one START edge, which a create-only re-publish can add (K-034).
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
            print(f"    ⚠ {side}: {len(starts)} START edges {starts} "
                  f"— expected exactly one")
            failures.append(
                f"{label}: {side} has {len(starts)} START edges "
                f"({', '.join(starts)}) — see K-034"
            )

print("")
if failures:
    print("RESULT: FAIL")
    for f in failures:
        print(f"  ✗ {f}")
    print("")
    print("If a def or snapshot is MISSING, re-seed:")
    print(f"  ./scripts/seed_workflows.sh {ctx.ws}")
    print("If they DIVERGE, do NOT re-seed — a create-only re-publish cannot overwrite")
    print("the stored def, and deleting a snapshot breaks live WorkflowRuns. Report it.")
    sys.exit(1)

print(f"RESULT: OK — {len(DEFS)} defs in sync between `reference` and ws:{ctx.ws}")
PY
