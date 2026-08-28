#!/usr/bin/env bash
# seed_salesperson.sh — publish + materialize the shared "salesperson" demo agent
# (K-052 M6, docs/plans/workflow-catalog-lookup.md §3.3/§3.4).
#
# Usage:
#   ./scripts/seed_salesperson.sh [<workspaceId>]     # default: $FALKORCHAT_WS_ID or "acme"
#
# What it seeds — ONE def, additive-only, idempotent (safe to re-run): published
# as a WorkflowDef into the GLOBAL `reference` graph and materialized as a
# WorkflowDefSnapshot into ws:<id> (the workspace copy the executor actually
# drives). Mirrors seed_workflows.sh's shape exactly (same idempotence-probe
# pattern, same "already present — no-op" reporting), but for one def instead
# of two, imported (never re-typed) from `falkorchat.proof_defs.SALESPERSON_DEF`
# — the SAME constant this repo's `salesperson`-scaffold tests
# (`server/tests/test_salesperson_scaffold.py`) import, so the seeded def and
# the tested def cannot drift (the same anti-drift property `ACCESS_REQUEST_DEF`
# already established).
#
#   salesperson@v1  — kind `conversation`, one `type:'agent'` step (`assistant`,
#                      start, waitsForHuman) with the catalog-lookup tools
#                      (`post_message`, `lookup_product_fact`, `filter_products`),
#                      plus one terminal `decision` step (`ended`) reached only
#                      via a `ctx.endConversation` truthy guard that this
#                      milestone's tools never set (docs/plans/
#                      workflow-catalog-lookup.md §2.4 — see proof_defs.py's own
#                      comment on SALESPERSON_DEF for the full mechanism).
#
# THIS SCRIPT IS EDITED IN PLACE by each sibling capability (K-053 cart/order,
# K-054 durable profile, K-055 NL query generation) to bump
# `SALESPERSON_DEF["version"]`'s import target and re-run — it is the same
# evolving artifact `seed_workflows.sh` itself is across K-022/K-024/etc., not a
# new script per capability (plan §3.4). **Topology stays byte-identical across
# all four versions** (plan §2.5) — only `config.tools`/`systemPrompt` change,
# which is exactly what a version bump is for, so the K-034 409 topology-conflict
# path is never hit by a later sibling's bump.
#
# ⚠️ "Idempotent" means CREATE-ONLY, not update (same caveat as
# seed_workflows.sh): a property-only edit (step config, guard text) to an
# ALREADY-PUBLISHED (key, version) is a silent no-op on re-publish. Landing a def
# edit for THIS version therefore requires either a fresh (unpublished)
# workspace, or — as designed — bumping the version instead of editing `v1` in
# place.
#
# ORDERING — run this AFTER:
#   1. ./scripts/bootstrap_schema.sh <wsId>   (indexes + constraints for `reference` + ws)
#   2. ./scripts/seed_demo.sh <wsId>          (the `assistant` Agent + a channel/thread to @mention)
#   3. ./scripts/seed_catalog.sh              (the Product catalog the demo's tools query)
# It does not touch chat/demo data or the catalog itself.
#
# Env vars (all optional):
#   FALKORDB_HOST                    (default: 127.0.0.1)
#   FALKORDB_PORT                    (default: 6379)
#   FALKORCHAT_WS_ID                 (default: acme)     — workspace id (graph key ws:<id>)
#   FALKORCHAT_SALESPERSON_DEF_KEY     (default: salesperson) — LOCAL to this script
#   FALKORCHAT_SALESPERSON_DEF_VERSION (default: v1)          (K-037-style decoupling:
#                                     no config var reads either of these two — this
#                                     def is never an @mention trigger target in this
#                                     milestone, only ever started/observed directly)

set -euo pipefail

HOST="${FALKORDB_HOST:-127.0.0.1}"
PORT="${FALKORDB_PORT:-6379}"
WS_ID="${1:-${FALKORCHAT_WS_ID:-acme}}"
SALESPERSON_DEF_KEY="${FALKORCHAT_SALESPERSON_DEF_KEY:-salesperson}"
SALESPERSON_DEF_VERSION="${FALKORCHAT_SALESPERSON_DEF_VERSION:-v1}"

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

echo "── seeding workflow def '${SALESPERSON_DEF_KEY}@${SALESPERSON_DEF_VERSION}' into reference + ws:${WS_ID} ──"

# Def content is IMPORTED from falkorchat.proof_defs (SALESPERSON_DEF) — never
# re-typed here. Runtime values (ws/key/version) are read from the environment
# inside Python, never interpolated into the payload.
FALKORCHAT_WS_ID="$WS_ID" \
FALKORDB_HOST="$HOST" FALKORDB_PORT="$PORT" \
FALKORCHAT_SALESPERSON_DEF_KEY="$SALESPERSON_DEF_KEY" \
FALKORCHAT_SALESPERSON_DEF_VERSION="$SALESPERSON_DEF_VERSION" \
"$VENV_PY" - <<'PY'
import os
import sys

from redis.exceptions import ResponseError

from falkorchat import config, db
from falkorchat.proof_defs import SALESPERSON_DEF
from falkorchat.repository import Repository
from falkorchat.services import Services

KEY = os.environ["FALKORCHAT_SALESPERSON_DEF_KEY"]
VERSION = os.environ["FALKORCHAT_SALESPERSON_DEF_VERSION"]

SPEC = {**SALESPERSON_DEF, "key": KEY, "version": VERSION}

services = Services(Repository(db.connect()))
ctx = config.get_context()


def _probe(fn):
    """Idempotence probe (before): a cold graph key (nothing published/
    materialized yet) raises 'Invalid graph operation on empty key' on the
    read — treat that as 'not present' rather than crashing (publish/
    materialize below create the graph)."""
    try:
        return fn()
    except ResponseError as exc:
        if "empty key" in str(exc):
            return None
        raise


def_pre = _probe(lambda: services.get_workflow_def(ctx, key=KEY, version=VERSION))
snap_pre = _probe(lambda: services.get_snapshot(ctx, key=KEY, version=VERSION))

pub = services.publish_workflow_def(ctx, **SPEC)
mat = services.materialize_def(ctx, key=KEY, version=VERSION)

print(
    f"  reference def   {pub['key']}@{pub['version']}  "
    f"steps={pub['stepCount']} transitions={pub['transitionCount']}  "
    f"({'already present — no-op' if def_pre is not None else 'created'})"
)
print(
    f"  ws:{ctx.ws} snapshot {mat['key']}@{mat['version']}  "
    f"steps={mat['stepCount']} transitions={mat['transitionCount']}  "
    f"({'already present — no-op' if snap_pre is not None else 'materialized'})"
)

if services.get_snapshot(ctx, key=KEY, version=VERSION) is None:
    print(
        f"ERROR: snapshot {KEY}@{VERSION} not found after materialize",
        file=sys.stderr,
    )
    sys.exit(1)
PY

echo ""
echo "salesperson@${SALESPERSON_DEF_VERSION} seeded (idempotent, create-only) into ws:${WS_ID}."
echo "Verify with:  ./scripts/verify_salesperson.sh ${WS_ID}"
echo ""
echo "It is a 'conversation'-kind, chat-triggered def (like triage) — NOT startable"
echo "via a bare 'POST /workflow-runs' (that path has no trigger message, so the"
echo "'assistant' step's post_message tool would have no thread to post into)."
echo "To exercise it live, point the @mention trigger at it for this run:"
echo "  FALKORCHAT_TRIGGER_DEF_KEY=${SALESPERSON_DEF_KEY} FALKORCHAT_TRIGGER_DEF_VERSION=${SALESPERSON_DEF_VERSION} \\"
echo "    ./scripts/start_server.sh"
echo "then @mention the agent (config.AGENT_ID) in any thread in ws:${WS_ID}."
