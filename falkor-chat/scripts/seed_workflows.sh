#!/usr/bin/env bash
# seed_workflows.sh — publish + materialize the M3 proof workflows (K-022 U13, K-024 U4).
#
# Usage:
#   ./scripts/seed_workflows.sh [<workspaceId>]     # default: $FALKORCHAT_WS_ID or "acme"
#
# What it seeds — TWO defs, additive-only, idempotent (safe to re-run). Each is published
# as a WorkflowDef into the GLOBAL `reference` graph and materialized as a
# WorkflowDefSnapshot into ws:<id> (the workspace copy the executor actually drives):
#
#   1. triage@v1          — kind `conversation`, the LLM-driven flow an @mention starts
#                           (docs/plans/m3-executor.md §8): three `type:'agent'` steps,
#      intake (start, waitsForHuman) --{llm guard}--> research --{"" uncond., D5}--> answer (terminal)
#                           Def content is INLINE below (see the two-source note).
#   2. access-request@v1  — kind `process`, the LLM-FREE proof flow (K-024,
#                           docs/plans/m3-process-flow.md §4): six `human`/`decision`/`wait`
#                           steps, six deterministic `cmp`-guarded transitions, started over
#                           REST (`POST /workflow-runs`), never by an @mention.
#      submit(human) -> route(decision) -> approval(human) -> provision(wait) -> activate | rejected
#                           Def content is IMPORTED from `falkorchat.proof_defs`
#                           (ACCESS_REQUEST_DEF) — the same constant the offline acceptance
#                           test `server/tests/test_process_flow.py` drives, so the seeded
#                           def and the tested def cannot drift.
#
# TWO def-source conventions in one script, deliberately (plan §4.4, gate m-9): moving the
# published, live triage literal into `proof_defs.py` during this slice risked a byte-diff
# that `MERGE … ON CREATE SET` would silently swallow. Filed as proposed backlog item K-029
# — "converge seed def sources into proof_defs.py".
#
# WHY a Python one-shot over the SERVICE LAYER (not raw redis-cli Cypher like
# seed_demo.sh): publishing a def runs real validation (start-key derivation, step-type
# whitelist, transition-endpoint checks, the `waitsForHuman` + `cmp`-guard publish
# invariants, opaque-JSON config/guard serialization) inside `services.publish_workflow_def`.
# Reimplementing that in bash Cypher would drift from the service invariants. Both
# `publish_workflow_def` and `materialize_def` are idempotent by construction
# (constraint-backed MERGE, immutable per version), so a re-run is a clean no-op — it
# prints `already present — no-op` for BOTH defs and writes nothing.
#
# ⚠️ "Idempotent" means CREATE-ONLY, not update — for both defs. `repository._PUBLISH_CYPHER`
# is `MERGE (…) ON CREATE SET …`, so EDITING a step config / guard / prompt here (or in
# `proof_defs.py`) and re-running changes NOTHING live: the run reports a clean
# `already present — no-op` while the old content stays. Worse, `reference` (def) and
# `ws:<id>` (snapshot) go stale INDEPENDENTLY — `scripts/test_queries.sh` and the pytest
# `wf_repo` fixture both wipe `reference` but not `ws:<id>` — so a naive re-seed after either
# republishes the NEW def while the workspace keeps the OLD snapshot, a silent split-brain,
# and the snapshot is what the executor drives. Landing a def edit therefore requires an
# explicit act: delete the def + snapshot subgraphs and republish, or bump `key`/`version`
# (for triage, kept in sync with config.TRIGGER_DEF_KEY/TRIGGER_DEF_VERSION — note
# start_server.sh neither forwards nor exports those two vars). Deleting a snapshot breaks
# live WorkflowRuns that point at it via OF_DEF/AT_STEP — a destructive shared-state op.
#
# ORDERING — run this AFTER:
#   1. ./scripts/bootstrap_schema.sh <wsId>   (indexes + constraints for `reference` + ws)
#   2. ./scripts/seed_demo.sh <wsId>          (the `assistant` Agent + a channel/thread to @mention)
# It depends on the workspace graph + its schema existing; it does NOT touch chat or demo data.
# RE-RUN IT after `./scripts/test_queries.sh` or a server pytest run — for DIFFERENT reasons:
#   * test_queries.sh DELETES `reference` at teardown, taking BOTH defs with it (ws:<id> survives);
#   * the pytest `wf_repo` fixture wipes `reference` at fixture SETUP, per workflow test, so a
#     finished pytest session LEAVES BEHIND whatever the last workflow test published. After a
#     pytest run, `already present — no-op` may therefore be reporting a TEST's publish rather
#     than a real seed, while ws:<id> still holds the older snapshot the executor drives.
#     (The acceptance test publishes `access-request@v1-test`, deliberately not the production
#     version, so it cannot be the def you find here.)
#
# The TRIAGE def key/version MUST match the trigger config (config.TRIGGER_DEF_KEY /
# TRIGGER_DEF_VERSION, defaults triage/v1) or the @mention-to-start step never resolves
# the def. Flip the trigger on with FALKORCHAT_WORKFLOW_ENABLED=1 (start_server.sh does this).
# `access-request@v1` needs no config var at all: it is started over REST, so nothing in
# config.py / .env.example / start_server.sh refers to it.
#
# Env vars (all optional):
#   FALKORDB_HOST          (default: 127.0.0.1)
#   FALKORDB_PORT          (default: 6379)
#   FALKORCHAT_WS_ID       (default: acme)     — workspace id (graph key ws:<id>)
#   FALKORCHAT_TRIGGER_DEF_KEY     (default: triage)  — must match config.TRIGGER_DEF_KEY
#   FALKORCHAT_TRIGGER_DEF_VERSION (default: v1)      — must match config.TRIGGER_DEF_VERSION
#   FALKORCHAT_PROCESS_DEF_KEY     (default: access-request) — LOCAL to this script; no
#   FALKORCHAT_PROCESS_DEF_VERSION (default: v1)              config var reads these two,
#                                  and `test_process_flow.py` drives the defaults, so an
#                                  override seeds a def nothing else refers to.

set -euo pipefail

HOST="${FALKORDB_HOST:-127.0.0.1}"
PORT="${FALKORDB_PORT:-6379}"
WS_ID="${1:-${FALKORCHAT_WS_ID:-acme}}"
DEF_KEY="${FALKORCHAT_TRIGGER_DEF_KEY:-triage}"
DEF_VERSION="${FALKORCHAT_TRIGGER_DEF_VERSION:-v1}"
# ⚠️ These two defaults DUPLICATE `ACCESS_REQUEST_DEF["key"]` / `["version"]` (re-gate r-5).
# The splat below (`{**ACCESS_REQUEST_DEF, "key": …, "version": …}`) *overrides* both, so the
# seed never reads the constant's own pair — the duplication is kept in sync by hand. It is
# pinned on one side only, by `test_process_flow.py`'s assertion that the constant is
# ("access-request", "v1"). A key/version bump therefore has to touch all THREE sites:
# these defaults, `proof_defs.py`, and that test.
PROCESS_DEF_KEY="${FALKORCHAT_PROCESS_DEF_KEY:-access-request}"
PROCESS_DEF_VERSION="${FALKORCHAT_PROCESS_DEF_VERSION:-v1}"

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

echo "── seeding workflow defs '${DEF_KEY}@${DEF_VERSION}' + '${PROCESS_DEF_KEY}@${PROCESS_DEF_VERSION}' into reference + ws:${WS_ID} ──"

# The triage def content lives in the service-layer payload below (the access-request def is
# imported from `falkorchat.proof_defs`); the def spec (start-key derivation, opaque-JSON
# serialization, transition + publish-invariant validation) is enforced by the service.
# Runtime values (ws/key/version) are read from the environment inside Python — never
# interpolated into the payload — so the shell quoting never touches the def content.
FALKORCHAT_WS_ID="$WS_ID" \
FALKORDB_HOST="$HOST" FALKORDB_PORT="$PORT" \
FALKORCHAT_TRIGGER_DEF_KEY="$DEF_KEY" FALKORCHAT_TRIGGER_DEF_VERSION="$DEF_VERSION" \
FALKORCHAT_PROCESS_DEF_KEY="$PROCESS_DEF_KEY" \
FALKORCHAT_PROCESS_DEF_VERSION="$PROCESS_DEF_VERSION" \
"$VENV_PY" - <<'PY'
import os
import sys

from redis.exceptions import ResponseError

from falkorchat import config, db
from falkorchat.proof_defs import ACCESS_REQUEST_DEF
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

# The two defs this script seeds, in order. `triage` is the inline literal above;
# `access-request` is the SHIPPED constant (falkorchat.proof_defs) that the offline
# acceptance test drives — the no-drift property from plan §4.4. The process def's
# key/version are overridable locally (they are referenced by no config var), so the
# imported spec is copied rather than mutated in place.
DEFS = [
    {
        "key": KEY,
        "version": VERSION,
        "name": "Triage",
        "kind": "conversation",
        "steps": STEPS,
        "transitions": TRANSITIONS,
    },
    {
        **ACCESS_REQUEST_DEF,
        "key": os.environ["FALKORCHAT_PROCESS_DEF_KEY"],
        "version": os.environ["FALKORCHAT_PROCESS_DEF_VERSION"],
    },
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

    # Sanity: a run resolves its def by (key, version) — the trigger for triage, the
    # REST start body for access-request; confirm the snapshot the executor drives is
    # readable back from the workspace.
    if services.get_snapshot(ctx, key=key, version=version) is None:
        print(
            f"ERROR: snapshot {key}@{version} not found after materialize",
            file=sys.stderr,
        )
        sys.exit(1)
PY

echo ""
echo "Both workflow defs seeded (idempotent, create-only)."
echo "  ${DEF_KEY}@${DEF_VERSION}: trigger it by turning the workflow engine on"
echo "    (FALKORCHAT_WORKFLOW_ENABLED=1, done by start_server.sh) and @mentioning the agent."
echo "  ${PROCESS_DEF_KEY}@${PROCESS_DEF_VERSION}: start it over REST —"
echo "    POST /workflow-runs {\"defKey\":\"${PROCESS_DEF_KEY}\",\"version\":\"${PROCESS_DEF_VERSION}\",\"maxSteps\":24}"
echo "    then POST /workflow-runs/{runId}/input to advance it."
