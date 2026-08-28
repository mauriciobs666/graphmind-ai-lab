"""Proof/demo workflow definitions shipped with the package (K-024 U4).

**Why def *content* lives in the installed package** (plan `docs/plans/m3-process-flow.md`
§4.4): it is *data the acceptance test must import*, and the package is the only artifact
that both an offline test (`server/tests/test_process_flow.py`) and a shell script
(`scripts/seed_workflows.sh`) can read without a subprocess. K-022 U14 learned this the
hard way — `test_workflow_live.py` had to shell out to the seed script precisely because a
copied def spec drifts. An importable constant gets the same no-drift property for free.

Nothing on the request path imports this module: it is ~2 KB of constants and no runtime
behaviour.

`ACCESS_REQUEST_DEF` is the LLM-free `kind:'process'` proof flow (DESIGN §6.3 —
"coordination is workflow"). It exercises every capability the K-024 slice added and
nothing else: `human` / `decision` / `wait` typed handlers (no `agent`, so **no LLM and no
network**), the four deterministic `cmp` ops (`exists` / `in` / `eq` / `truthy`), the
conditional-beats-unconditional transition ordering, a two-way branch where *neither* side
firing re-parks the step, and two terminal outcome nodes.

Shape note: `config`/`guard` are **plain dicts here**. `services.publish_workflow_def`
serializes them to the opaque strings the graph stores (rule 8 — never filtered in Cypher),
so this module never hand-rolls JSON.

Invariants this def deliberately honours (see `falkor-chat/AGENTS.md`):
  * every `human`/`wait` step declares `config.waitsForHuman: true` — enforced at publish;
  * every published def carries **≥ 1 transition** — `repository._PUBLISH_CYPHER` ends in
    a bare `UNWIND $transitions`, which collapses the row stream on `[]`. Since U4b
    `services._validate_def_spec` rejects a transition-less spec with
    `WorkflowDefSpecError` **before any repository call**, so publish is safe. The guard is
    **publish-only**: `materialize_snapshot` reuses the same query with no validation and
    still raises `IndexError` after a partial write (backlog **K-030**, which also proposes
    guarding the `UNWIND` and relaxing this rule so a genuine single-step def is publishable
    again). A terminal step is one with no *outgoing* transition (`activate`, `rejected`),
    never a def with no transitions at all;
  * the key is **`access-request`**, not `onboarding` — that key belongs to long-standing
    test fixtures.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "ACCESS_REQUEST_DEF", "ACCESS_REQUEST_MAX_STEPS",
    "SALESPERSON_DEF", "SALESPERSON_MAX_STEPS",
]


# Declared step budget (plan §4.1 / D-H part c). The privileged-role happy path costs 8
# steps, so 24 leaves 16 spare re-parks; values the def declares invalid (`expects`) are
# rejected at the boundary and cost **nothing**. A caller that omits `maxSteps` falls back
# to the executor's global default of 12 ⇒ only 4 spare re-parks — documented here rather
# than discovered in production.
ACCESS_REQUEST_MAX_STEPS = 24


# ⚠️ These six keys ARE `services.publish_workflow_def`'s keyword signature (n-A): both
# `scripts/seed_workflows.sh` and `server/tests/test_process_flow.py` splat this constant
# with `**`. Adding a field (a `notes`, a `budget`) breaks both with a `TypeError` at run
# time — put anything that is not a publish argument in a module-level constant instead,
# the way `ACCESS_REQUEST_MAX_STEPS` is.
ACCESS_REQUEST_DEF: dict[str, Any] = {
    "key": "access-request",
    "version": "v1",
    "name": "Access request",
    "kind": "process",
    # ── §4.1 steps ──────────────────────────────────────────────────────────────
    "steps": [
        {
            # Parks until the request is filed. `fields` lists the accepted **top-level**
            # input keys (D-H rule 2); the submitted `request` is a nested object accepted
            # whole — validation is on top-level keys only, there is no deep schema.
            "key": "submit",
            "type": "human",
            "start": True,
            "config": {
                "waitsForHuman": True,
                "prompt": "File the access request",
                "fields": ["request"],
                "assignee": "requester",
            },
        },
        {
            # Pure branch, no side effect: privileged roles need approval, standard hires
            # do not. Its semantics are entirely its outgoing guards (#2 / #3).
            "key": "route",
            "type": "decision",
            "config": {},
        },
        {
            # Parks until a manager decides. `expects` makes any other value a **free 400**
            # (D-H rule 3) — a typo can never burn step budget.
            "key": "approval",
            "type": "human",
            "config": {
                "waitsForHuman": True,
                "prompt": "Approve or reject this access request",
                "fields": ["decision"],
                "expects": {"decision": ["approve", "reject"]},
                "assignee": "manager",
            },
        },
        {
            # Parks until the provisioning system signals back. Signal-driven, **not**
            # timer-driven (D-C — there is no scheduler; timers are proposed K-028).
            # Deliberately **no `expects`**, so `{"provisioned": false}` ("not yet") stays
            # expressible: it re-parks and costs one step.
            "key": "provision",
            "type": "wait",
            "config": {"waitsForHuman": True, "signal": "provisioned"},
        },
        # The two terminal outcome nodes: no outgoing transition ⇒ `complete_run` ⇒ the
        # run ends `done`. A rejected request is a completed *process* with a rejected
        # *outcome*; `failed` stays reserved for engine faults and budget exhaustion.
        {"key": "activate", "type": "decision", "config": {}},
        {"key": "rejected", "type": "decision", "config": {}},
    ],
    # ── §4.2 transitions — six ─────────────────────────────────────────────────
    # `on` values are descriptive labels only (F-1); the sort key is
    # `(guard == "", order)`, i.e. conditional guards first with `order` as the
    # intra-class tie-break — which is what makes #2 beat the unconditional #3.
    "transitions": [
        {
            "from": "submit", "to": "route", "on": "filed", "order": 0,
            "guard": {"kind": "cmp", "path": "ctx.request.role", "op": "exists"},
        },
        {
            "from": "route", "to": "approval", "on": "needs_approval", "order": 0,
            "guard": {
                "kind": "cmp", "path": "ctx.request.role", "op": "in",
                "value": ["contractor", "exec"],
            },
        },
        {
            # Unconditional default — fires only if the conditional #2 does not.
            "from": "route", "to": "provision", "on": "auto", "order": 1,
            "guard": "",
        },
        {
            "from": "approval", "to": "provision", "on": "approved", "order": 0,
            "guard": {
                "kind": "cmp", "path": "ctx.decision", "op": "eq", "value": "approve",
            },
        },
        {
            # With #4, a two-way branch where *neither* firing re-parks `approval`.
            "from": "approval", "to": "rejected", "on": "rejected", "order": 1,
            "guard": {
                "kind": "cmp", "path": "ctx.decision", "op": "eq", "value": "reject",
            },
        },
        {
            "from": "provision", "to": "activate", "on": "provisioned", "order": 0,
            "guard": {"kind": "cmp", "path": "ctx.provisioned", "op": "truthy"},
        },
    ],
}


# ── `salesperson` — the shared demo-agent scaffold (K-052 M6) ────────────────
#
# `SALESPERSON_DEF` is the single, shared "salesperson" `WorkflowDef` this document
# names as the canonical scaffold for FOUR sibling capabilities (`docs/plans/
# workflow-catalog-lookup.md` §2.3-§2.5, the document that owns this constant's
# design): catalog lookup (K-052, this landing), cart/order (K-053), durable
# profile (K-054), and NL query generation (K-055). Each sibling **bumps this
# constant's `version`** (`v1` -> `v2` -> `v3` -> `v4`) and republishes the FULL
# cumulative `config.tools`/`systemPrompt` — never edits `v1` in place — because a
# def's topology is immutable per version (`docs/DESIGN.md` §4) but `config.tools`/
# `systemPrompt` are create-only properties: a same-version republish with an
# added tool would silently no-op and the new tool would never reach a running
# agent (plan §2.5). Topology (one `agent` step + the `ended` decision step + the
# one `ctx.endConversation` transition) is deliberately identical across all four
# versions, so the K-034 409 topology-conflict path is never hit by a later
# sibling's version bump.
#
# **Why exactly one conditional transition, not zero and not unconditional**
# (plan §2.4 — binding for all four versions): `_validate_def_spec` requires a def
# to carry >= 1 transition (K-024 U4b, O-6; K-030, still open, would relax this),
# and an *unconditional* (`guard: ""`) transition always fires (`guards.
# evaluate_guard`), which would make the `assistant` step advance every turn
# instead of parking for the next customer message — breaking the whole
# "wait for the customer's next message" design this demo depends on. The
# resolution: one **conditional** transition to a terminal `decision` step,
# guarded on a `ctx` key nothing in this demo's tool set ever sets
# (`ctx.endConversation`). This mirrors the precedented, present-but-unexercised
# `human_handoff` pattern above (`tools.HumanHandoffTool` — "a registered
# capability that signals suspend... present, not exercised") — a genuine
# forward-looking affordance (a future "the agent ends the conversation"
# extension has somewhere to go), not dead code smuggled in to satisfy a
# validator. See `server/tests/test_salesperson_scaffold.py` for the regression
# guard proving this transition never fires across an ordinary multi-turn
# conversation, plus a sanity companion proving the guard mechanism itself is real.
SALESPERSON_DEF: dict[str, Any] = {
    "key": "salesperson",
    "version": "v1",
    "name": "Salesperson",
    "kind": "conversation",
    "steps": [
        {
            "key": "assistant",
            "type": "agent",
            "start": True,
            "config": {
                "waitsForHuman": True,
                "systemPrompt": (
                    "You are a helpful electronics-store assistant chatting with a "
                    "customer.\n\n"
                    "You can answer factual questions about specific products (name, "
                    "category, price) and list products matching a category or price "
                    "range, using your catalog tools. Never guess a price or category "
                    "you have not retrieved from a tool; if nothing matches, say so "
                    "plainly rather than inventing an answer.\n\n"
                    "Deliver every reply by calling the `post_message` tool; text you "
                    "merely write is never seen by the customer. Never pass `mentions`; "
                    "omit that argument entirely."
                ),
                "tools": ["post_message", "lookup_product_fact", "filter_products"],
                "requiredTools": ["post_message"],
                "maxIterations": 8,
            },
        },
        {"key": "ended", "type": "decision", "config": {}},
    ],
    "transitions": [
        {
            "from": "assistant", "to": "ended", "on": "ended", "order": 0,
            "guard": {"kind": "cmp", "path": "ctx.endConversation", "op": "truthy"},
        },
    ],
}

# A `WorkflowRun.maxSteps` budget (not `schemas.MAX_STEPS`, the unrelated
# publish-time step-*count* cap). This demo runs many customer turns over one
# long-lived run (unlike `triage`'s few-turn intake), so a larger budget than
# `ACCESS_REQUEST_MAX_STEPS`'s 24 is appropriate — a tripwire, not an SLA
# (`docs/DESIGN.md` §6.2).
SALESPERSON_MAX_STEPS = 40
