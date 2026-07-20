#!/usr/bin/env bash
# seed_workflows.sh — publish + materialize the M3 "triage" proof workflow so an
# @mention of the agent starts a run (K-022 / U13).
#
# Usage:
#   ./scripts/seed_workflows.sh [<workspaceId>]     # default: $FALKORCHAT_WS_ID or "acme"
#
# What it seeds (additive-only, idempotent — safe to re-run):
#   * WorkflowDef {triage@v1}          — published into the GLOBAL `reference` graph
#   * WorkflowDefSnapshot {triage@v1}  — materialized into ws:<id> (the workspace copy
#                                        the executor drives)
# The def is the 3-step conversational triage flow from `docs/plans/m3-executor.md`
# §8 (kind `conversation`): intake -> research -> answer, three `type:'agent'` steps.
#   intake  (start, waitsForHuman) --{llm guard}--> research --{"" unconditional, D5}--> answer (terminal)
#
# WHY a Python one-shot over the SERVICE LAYER (not raw redis-cli Cypher like
# seed_demo.sh): publishing a def runs real validation (start-key derivation, step-type
# whitelist, transition-endpoint checks, opaque-JSON config/guard serialization) inside
# `services.publish_workflow_def`. Reimplementing that in bash Cypher would drift from the
# service invariants. Both `publish_workflow_def` and `materialize_def` are idempotent by
# construction (constraint-backed MERGE, immutable per version), so a re-run is a clean
# no-op — no duplicate defs or snapshots.
#
# ORDERING — run this AFTER:
#   1. ./scripts/bootstrap_schema.sh <wsId>   (indexes + constraints for `reference` + ws)
#   2. ./scripts/seed_demo.sh <wsId>          (the `assistant` Agent + a channel/thread to @mention)
# It depends on the workspace graph + its schema existing; it does NOT touch chat or demo data.
#
# The def key/version MUST match the trigger config (config.TRIGGER_DEF_KEY /
# TRIGGER_DEF_VERSION, defaults triage/v1) or the @mention-to-start step never resolves
# the def. Flip the trigger on with FALKORCHAT_WORKFLOW_ENABLED=1 (start_server.sh does this).
#
# Env vars (all optional):
#   FALKORDB_HOST          (default: 127.0.0.1)
#   FALKORDB_PORT          (default: 6379)
#   FALKORCHAT_WS_ID       (default: acme)     — workspace id (graph key ws:<id>)
#   FALKORCHAT_TRIGGER_DEF_KEY     (default: triage)  — must match config.TRIGGER_DEF_KEY
#   FALKORCHAT_TRIGGER_DEF_VERSION (default: v1)      — must match config.TRIGGER_DEF_VERSION

set -euo pipefail

HOST="${FALKORDB_HOST:-127.0.0.1}"
PORT="${FALKORDB_PORT:-6379}"
WS_ID="${1:-${FALKORCHAT_WS_ID:-acme}}"
DEF_KEY="${FALKORCHAT_TRIGGER_DEF_KEY:-triage}"
DEF_VERSION="${FALKORCHAT_TRIGGER_DEF_VERSION:-v1}"

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

echo "── seeding workflow def '${DEF_KEY}@${DEF_VERSION}' into reference + ws:${WS_ID} ──"

# The triage def content lives in the service-layer payload below; the def spec (start-key
# derivation, opaque-JSON serialization, transition validation) is enforced by the service.
# Runtime values (ws/key/version) are read from the environment inside Python — never
# interpolated into the payload — so the shell quoting never touches the def content.
FALKORCHAT_WS_ID="$WS_ID" \
FALKORDB_HOST="$HOST" FALKORDB_PORT="$PORT" \
FALKORCHAT_TRIGGER_DEF_KEY="$DEF_KEY" FALKORCHAT_TRIGGER_DEF_VERSION="$DEF_VERSION" \
"$VENV_PY" - <<'PY'
import os
import sys

from redis.exceptions import ResponseError

from falkorchat import config, db
from falkorchat.repository import Repository
from falkorchat.services import Services

KEY = os.environ["FALKORCHAT_TRIGGER_DEF_KEY"]
VERSION = os.environ["FALKORCHAT_TRIGGER_DEF_VERSION"]

# The triage proof flow (docs/plans/m3-executor.md §8): kind `conversation`, three
# `type:'agent'` steps. `config` is opaque JSON app-side (rule 8 — never filtered in
# Cypher); the executor reads waitsForHuman / systemPrompt / tools / maxIterations.
STEPS = [
    {
        "key": "intake",
        "type": "agent",
        "start": True,
        "config": {
            "waitsForHuman": True,
            # D14 (docs/plans/m3-executor-coordination.md, 2026-07-19): S5
            # (docs/plans/m3-guard-thread-context.md §5) is DELIBERATELY NOT HERE.
            # S5 asked this node to end every turn with a
            # `{"understanding":{request,known,missing}}` object so the intake→research
            # llm guard could run on its *primary* extract-then-judge path
            # (`m3-executor-ml.md` §Q1). Measured live on the shipped chat model
            # (Qwen3-4B) it REGRESSED intake advancement 10/10 → 3/10, so this cut ships
            # the guard on the DS's degraded recent-turns fallback instead — the 10/10
            # path. Do not "helpfully" re-add the understanding-JSON instruction: it is
            # only worth re-landing once the judge is model-robust (fence/prose-tolerant
            # parse) AND calibrated — a K-023 follow-up.
            # The "never pass mentions" rule below is a separate, separately-measured
            # Defect-C mitigation, not part of S5: the node's thread context is folded
            # in as "{displayName}: {text}", so the model reliably reaches for
            # `mentions:["alice"]`, which §4 rejects (it resolves member *ids*) — and a
            # 4B recovers from that rejection by dropping the tool and emitting text,
            # i.e. silently posting nothing.
            "systemPrompt": (
                "You are triaging a user's request in a chat thread.\n\n"
                "Ask the user clarifying questions until you can state their request "
                "precisely; ask one question at a time. Deliver every question by "
                "calling the `post_message` tool — text you merely write is never seen "
                "by anyone. Never pass `mentions`; omit that argument entirely."
            ),
            "tools": ["post_message"],
            "maxIterations": 4,
        },
    },
    {
        "key": "research",
        "type": "agent",
        "config": {
            "systemPrompt": (
                "Retrieve relevant context from the workspace and produce concise "
                "findings grounded only in what you retrieve; if nothing relevant is "
                "found, say so."
            ),
            "tools": ["graphrag_retrieve"],
        },
    },
    {
        "key": "answer",
        "type": "agent",
        "config": {
            # Defect C (docs/plans/m3-executor-coordination.md, Defect C): the answer
            # node reliably produces a good grounded answer but often posts nothing —
            # no PRODUCED edge, AC-4 fails. Two measured mechanisms, both prompt-level:
            #   (a) it emits the answer as final TEXT instead of calling post_message
            #       (D4's 4B tool-calling risk); and
            #   (b) it DOES call post_message but with `mentions:["alice"]` (the folded
            #       "{displayName}: {text}" thread context leaks the display name), §4
            #       rejects it, and the 4B "recovers" by dropping the tool and emitting
            #       text — silently posting nothing.
            # So: mandate the tool call as the node's REQUIRED final action, and forbid
            # `mentions`. This is a prompt-only mitigation of an accepted 4B risk; the
            # durable fix, if it doesn't hold, is an engine-level "terminal node must
            # post" contract (out of scope here — routes to the architect).
            "systemPrompt": (
                "You are delivering the final answer to the user's request in a chat "
                "thread.\n\n"
                "You MUST post your answer by calling the `post_message` tool. This is "
                "the only way the user receives it — an answer you write as plain text "
                "is discarded and the user sees nothing. Do not end your turn until you "
                "have called `post_message`. Never pass `mentions`; omit that argument "
                "entirely (passing a display name there fails and loses your answer).\n\n"
                "Write the `text` as a direct, grounded answer to the user's request, "
                "using the research findings above; cite what you used. If the findings "
                "do not cover the request, say so plainly in the posted message — but "
                "still post it."
            ),
            "tools": ["post_message"],
        },
    },
]

TRANSITIONS = [
    # intake -> research: fuzzy LLM-judged guard (AC-2). False + waitsForHuman => suspend.
    {
        "from": "intake",
        "to": "research",
        "on": "ready",
        "guard": {
            "kind": "llm",
            "text": "the user has provided enough information to research their request",
        },
        "order": 0,
    },
    # research -> answer: "" unconditional (D5). Sufficiency is the node's own abstention,
    # not an LLM guard (no human to unblock a suspend here).
    {
        "from": "research",
        "to": "answer",
        "on": "done",
        "guard": "",
        "order": 0,
    },
    # answer is terminal — no outgoing transition (run -> done, AC-4).
]

services = Services(Repository(db.connect()))
ctx = config.get_context()

# Idempotence probe (before): the def/snapshot MERGE is a structural no-op on re-run;
# report whether each already existed so a re-run is visibly a clean no-op. A cold graph
# key (nothing published/materialized yet) raises "Invalid graph operation on empty key"
# on the read — treat that as "not present" rather than crashing (publish/materialize
# below create the graph).
def _probe(fn):
    try:
        return fn()
    except ResponseError as exc:
        if "empty key" in str(exc):
            return None
        raise

def_pre = _probe(lambda: services.get_workflow_def(ctx, key=KEY, version=VERSION))
snap_pre = _probe(lambda: services.get_snapshot(ctx, key=KEY, version=VERSION))

pub = services.publish_workflow_def(
    ctx,
    key=KEY,
    version=VERSION,
    name="Triage",
    kind="conversation",
    steps=STEPS,
    transitions=TRANSITIONS,
)
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

# Sanity: the trigger resolves the def by (key, version); confirm the snapshot the
# executor drives is readable back from the workspace.
snap = services.get_snapshot(ctx, key=KEY, version=VERSION)
if snap is None:
    print("ERROR: snapshot not found after materialize", file=sys.stderr)
    sys.exit(1)
PY

echo ""
echo "Triage workflow seeded (idempotent). Trigger it by turning the workflow engine on"
echo "(FALKORCHAT_WORKFLOW_ENABLED=1, done by start_server.sh) and @mentioning the agent."
