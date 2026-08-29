#!/usr/bin/env bash
# seed_salesperson.sh — publish + materialize the shared "salesperson" demo agent
# AND its FR-6/FR-9 sibling, the "order-fulfillment" process def
# (K-052/K-053 M6, docs/plans/workflow-catalog-lookup.md §3.3/§3.4,
# docs/plans/workflow-cart-and-totals.md §3.4/§4 step 7).
#
# Usage:
#   ./scripts/seed_salesperson.sh [<workspaceId>]     # default: $FALKORCHAT_WS_ID or "acme"
#
# What it seeds — TWO defs, additive-only, idempotent (safe to re-run), mirroring
# seed_workflows.sh's own two-def pattern: each is published as a WorkflowDef into
# the GLOBAL `reference` graph and materialized as a WorkflowDefSnapshot into
# ws:<id> (the workspace copy the executor actually drives). Both defs are
# imported (never re-typed) from `falkorchat.proof_defs` — the SAME constants
# this repo's own tests import (`server/tests/test_salesperson_scaffold.py`,
# `server/tests/test_order_fulfillment.py`), so the seeded defs and the tested
# defs cannot drift.
#
#   salesperson@v2       — kind `conversation`, one `type:'agent'` step
#                           (`assistant`, start, waitsForHuman) with the
#                           catalog-lookup tools (K-052) PLUS the five
#                           cart/order tools (K-053: `view_cart`, `add_to_cart`,
#                           `remove_from_cart`, `clear_cart`, `place_order`),
#                           plus one terminal `decision` step (`ended`) reached
#                           only via a `ctx.endConversation` truthy guard this
#                           milestone's tools never set (see proof_defs.py's own
#                           comment on SALESPERSON_DEF for the full mechanism).
#                           Topology is byte-identical to v1 — only
#                           `config.tools`/`systemPrompt` changed (the
#                           version-bump discipline, docs/plans/
#                           workflow-catalog-lookup.md §2.5) — so this bump
#                           never hits the K-034 409 topology-conflict path.
#   order-fulfillment@v1  — kind `process`, the LLM-FREE order-lifecycle proof
#                           flow (K-053, docs/plans/workflow-cart-and-totals.md
#                           §3.4): four `human`/`decision` steps, three
#                           deterministic `cmp`-guarded transitions, started
#                           over REST (no chat trigger), never by an @mention.
#      placed(decision, parks) -> fulfilled(human) -> delivered(decision, terminal)
#      placed                  -> cancelled(decision, terminal)
#                           `placed` is `type:'decision'` but still declares
#                           `config.waitsForHuman: true` (load-bearing — see
#                           proof_defs.py's own comment on ORDER_FULFILLMENT_DEF
#                           for why a parking step of ANY type needs this flag).
#                           This def's own steps have no side effect on the
#                           `Order` node it manages — advancing `Order.status`
#                           is a separate `services.advance_order` call the
#                           operator-facing caller makes alongside
#                           `submit_workflow_input` (the "two-step, accepted"
#                           pairing, docs/plans/workflow-cart-and-totals-graph.md
#                           §4) — not wired to a REST route in this milestone.
#
# THIS SCRIPT IS EDITED IN PLACE by each sibling capability (K-054 durable
# profile, K-055 NL query generation bump `salesperson`'s own version further;
# K-053 landed `order-fulfillment` alongside it) and re-run — it is the same
# evolving artifact seed_workflows.sh itself is across K-022/K-024/etc., not a
# new script per capability (docs/plans/workflow-catalog-lookup.md §3.4).
#
# ⚠️ "Idempotent" means CREATE-ONLY, not update (same caveat as
# seed_workflows.sh): a property-only edit (step config, guard text) to an
# ALREADY-PUBLISHED (key, version) is a silent no-op on re-publish. Landing a
# def edit for THIS version therefore requires either a fresh (unpublished)
# workspace, or — as designed for `salesperson` — bumping the version instead
# of editing a shipped one in place.
#
# ORDERING — run this AFTER:
#   1. ./scripts/bootstrap_schema.sh <wsId>   (indexes + constraints for `reference` + ws)
#   2. ./scripts/seed_demo.sh <wsId>          (the `assistant` Agent + a channel/thread to @mention)
#   3. ./scripts/seed_catalog.sh              (the Product catalog the demo's tools query)
# It does not touch chat/demo data or the catalog itself.
#
# Env vars (all optional):
#   FALKORDB_HOST                        (default: 127.0.0.1)
#   FALKORDB_PORT                        (default: 6379)
#   FALKORCHAT_WS_ID                     (default: acme)     — workspace id (graph key ws:<id>)
#   FALKORCHAT_SALESPERSON_DEF_KEY       (default: salesperson)      — LOCAL to this script
#   FALKORCHAT_SALESPERSON_DEF_VERSION   (default: v2)              (K-037-style decoupling:
#                                        no config var reads either of these two — this
#                                        def is never an @mention trigger target in this
#                                        milestone, only ever started/observed directly)
#   FALKORCHAT_ORDER_FULFILLMENT_DEF_KEY     (default: order-fulfillment) — LOCAL to
#   FALKORCHAT_ORDER_FULFILLMENT_DEF_VERSION (default: v1)               this script;
#                                        `test_order_fulfillment.py` drives the defaults
#                                        under a `-test` version suffix, so an override
#                                        here seeds a def nothing else refers to.

set -euo pipefail

HOST="${FALKORDB_HOST:-127.0.0.1}"
PORT="${FALKORDB_PORT:-6379}"
WS_ID="${1:-${FALKORCHAT_WS_ID:-acme}}"
SALESPERSON_DEF_KEY="${FALKORCHAT_SALESPERSON_DEF_KEY:-salesperson}"
SALESPERSON_DEF_VERSION="${FALKORCHAT_SALESPERSON_DEF_VERSION:-v2}"
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

echo "── seeding workflow defs '${SALESPERSON_DEF_KEY}@${SALESPERSON_DEF_VERSION}' + '${ORDER_FULFILLMENT_DEF_KEY}@${ORDER_FULFILLMENT_DEF_VERSION}' into reference + ws:${WS_ID} ──"

# Both defs' content is IMPORTED from falkorchat.proof_defs — never re-typed
# here. Runtime values (ws/key/version) are read from the environment inside
# Python, never interpolated into the payload.
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
from falkorchat.proof_defs import ORDER_FULFILLMENT_DEF, SALESPERSON_DEF
from falkorchat.repository import Repository
from falkorchat.services import Services

SALESPERSON_KEY = os.environ["FALKORCHAT_SALESPERSON_DEF_KEY"]
SALESPERSON_VERSION = os.environ["FALKORCHAT_SALESPERSON_DEF_VERSION"]
ORDER_FULFILLMENT_KEY = os.environ["FALKORCHAT_ORDER_FULFILLMENT_DEF_KEY"]
ORDER_FULFILLMENT_VERSION = os.environ["FALKORCHAT_ORDER_FULFILLMENT_DEF_VERSION"]

# The two defs this script seeds, in order — both SHIPPED constants
# (falkorchat.proof_defs), the no-drift property from
# docs/plans/m3-process-flow.md §4.4, applied here the same way
# seed_workflows.sh applies it to ACCESS_REQUEST_DEF. Key/version are
# overridable locally (no config var reads either pair), so each imported
# spec is copied rather than mutated in place.
DEFS = [
    {**SALESPERSON_DEF, "key": SALESPERSON_KEY, "version": SALESPERSON_VERSION},
    {
        **ORDER_FULFILLMENT_DEF, "key": ORDER_FULFILLMENT_KEY,
        "version": ORDER_FULFILLMENT_VERSION,
    },
]

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


for spec in DEFS:
    key, version = spec["key"], spec["version"]
    def_pre = _probe(lambda: services.get_workflow_def(ctx, key=key, version=version))
    snap_pre = _probe(lambda: services.get_snapshot(ctx, key=key, version=version))

    pub = services.publish_workflow_def(ctx, **spec)
    mat = services.materialize_def(ctx, key=key, version=version)

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

    if services.get_snapshot(ctx, key=key, version=version) is None:
        print(
            f"ERROR: snapshot {key}@{version} not found after materialize",
            file=sys.stderr,
        )
        sys.exit(1)
PY

echo ""
echo "salesperson@${SALESPERSON_DEF_VERSION} + ${ORDER_FULFILLMENT_DEF_KEY}@${ORDER_FULFILLMENT_DEF_VERSION} seeded (idempotent, create-only) into ws:${WS_ID}."
echo "Verify with:  ./scripts/verify_salesperson.sh ${WS_ID}"
echo ""
echo "salesperson is a 'conversation'-kind, chat-triggered def (like triage) — NOT"
echo "startable via a bare 'POST /workflow-runs' (that path has no trigger message, so"
echo "the 'assistant' step's post_message tool would have no thread to post into)."
echo "To exercise it live, point the @mention trigger at it for this run:"
echo "  FALKORCHAT_TRIGGER_DEF_KEY=${SALESPERSON_DEF_KEY} FALKORCHAT_TRIGGER_DEF_VERSION=${SALESPERSON_DEF_VERSION} \\"
echo "    ./scripts/start_server.sh"
echo "then @mention the agent (config.AGENT_ID) in any thread in ws:${WS_ID}."
echo ""
echo "order-fulfillment is a 'process'-kind def, started over REST (no chat trigger) —"
echo "  POST /workflow-runs {\"defKey\":\"${ORDER_FULFILLMENT_DEF_KEY}\",\"version\":\"${ORDER_FULFILLMENT_DEF_VERSION}\"}"
echo "then POST /workflow-runs/{runId}/input {\"action\":\"fulfill\"|\"cancel\"|\"deliver\"}"
echo "to advance it. Advancing the Order's own status alongside it is a separate call"
echo "(services.advance_order) — not wired to a REST route in this milestone."
