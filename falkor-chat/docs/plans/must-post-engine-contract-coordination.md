# `must-post-engine-contract` (K-027 item 2) — coordination

> **Status:** archived · **Owner:** `teco` · **Tracks:** K-027 item 2 (M3.5)

**Closed 2026-08-17.** All units delivered/gated/accepted; five commits landed
(`7d3effb` plan+plan-gate, `b6b9b53` implementation, `b04682f` diff-scoped re-gate,
`0e3aad7` docs closeout, `b3cb724` graph-dba kaizen inbox). `ws:acme` live-graph
divergence (U9/U10/U11) resolved same day. See the final ledger below for the full
trail.

## Goal

Design and land the **engine-level "must-post" contract** that K-027 item 2 has been carrying
since the D12-B descope: today an `agent`-typed executor node whose entire purpose is to
communicate (granted `post_message` or a future "must-communicate" tool) can end its turn on
plain text with no tool call, and the executor treats that as a perfectly normal, successful
`StepResult` (`on="done"`). On the shipped 4B chat model this is not a rare edge case — the
live-reproduced RCA (`docs/reviews/mention-reply-delivery-rca.md`) found **every** step of a
fresh `triage@v1` run doing this. K-039 item 1 already shipped a narrow, node-local **fallback**
(implicit `post_message` dispatch inside `_run_agent_node` when the loop ends on non-tool-call
text) as an immediate demo-unblocking mitigation — that is explicitly *not* this item (K-039's
own scope note: "Do not fold this into the full K-027 item 2 engine contract — that item is
broader... and `architect`-owned"). This unit is that broader, architect-owned design.

**Scope, per `docs/BACKLOG.md` K-027 item 2 + its "Addendum from the K-025 QA pass":**
1. An engine-level guarantee, not a prompt, that a node whose contract is "must post/communicate"
   actually does — covering **any** such node, not just the terminal one (the addendum found the
   same failure on the non-terminal `intake` node too, in the worse "clarifying question never
   reached the thread" shape).
2. Must sit alongside, not duplicate or fight, K-039 item 1's already-shipped node-local fallback
   (`server/falkorchat/executor.py::_run_agent_node`, the implicit-dispatch branch) — the plan
   needs to say explicitly whether the contract subsumes/replaces that fallback, layers on top of
   it, or leaves it as one of several defenses, and why.
3. Must respect the `_drive_loop` byte-identity SHA-lock (`71055f756280`,
   `docs/archive/plans/m3-process-flow.md` §3.1) — `falkor-chat/AGENTS.md`'s executor-invariants
   block records that `_execute_step`, `_select_transition`, `_trace_step` and `resume` sit
   **outside** that lock, so a design that stays outside `_drive_loop` avoids a lock-break/re-lock
   ceremony entirely; if the plan concludes the lock *must* break, it has to say so explicitly and
   scope the re-lock ceremony (grep `71055f756280` across `AGENTS.md`, `BACKLOG.md` ×2,
   `HISTORY.md` ×2, plus the frozen archive documents listed in K-033's write-up) as part of the
   plan, not as a surprise for the implementer.
4. Out of scope for this unit: judge calibration (K-027 item 3), golden-set expansion (item 4),
   Ministral re-probe (item 5) — those are `data-scientist` territory and unaffected by this
   design. Also out of scope: K-033 (the `maxSteps` off-by-one) and K-035 (argument-key
   shadowing) — related nearby debt, not this contract.

## Inputs for the architect (read, don't paraphrase)

- `falkor-chat/docs/BACKLOG.md` — K-027 (full entry, item 2 + the K-025 QA addendum are the
  primary scope statement) and K-039 (item 1's shipped fallback + item 2's explicit "do not fold
  in" boundary).
- `falkor-chat/docs/reviews/mention-reply-delivery-rca.md` — full RCA, live-reproduced root cause
  and §5 "Suggested fix & prevention" (items 1 and 3 in particular).
- `falkor-chat/docs/DESIGN.md` §6 (workflow engine model: §6.1 definition, §6.2 run, §6.3
  coordination) — the executor's current step/transition contract this design extends.
- `falkor-chat/AGENTS.md` — the executor-invariants block (byte-lock scope, what sits inside vs.
  outside `_drive_loop`) and the `ctx`/`config`/`guard`-opaque rule (rule 8) if the contract
  touches how a node's "must-post" obligation is declared in def config.
- `server/falkorchat/executor.py` — `_run_agent_node`, `_drive_loop`, `_execute_step`,
  `_select_transition`, `_trace_step`, `resume` — read the actual code, not just the docs, for
  exact seam boundaries and the K-039 fallback's current shape.
- `scripts/seed_workflows.sh` — the `triage@v1` literal, to see today's prompt-level mitigation
  this contract is meant to replace/backstop.

## Deliverable

Plan at `falkor-chat/docs/plans/must-post-engine-contract.md`: the engine-level mechanism design
(how a node declares/is recognized as "must-post", where the guarantee is enforced — inside vs.
outside the executor loop, what happens on violation — retry/fail/park/trace-and-continue),
relationship to the K-039 fallback, file/interface-level implementation steps, risks (RAM per
rule 6 — expect none, no new node/index), and test strategy (offline pins mirroring the RCA's
live repro shape, per AGENTS.md's existing convention for this codebase).

## Ledger

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict |
|---|---|---|---|---|---|
| U1 | `architect` | `a8bd7fd19a8b51c33` | accepted | `docs/plans/must-post-engine-contract.md` | `analyst` → approve with suggestions |
| U2 | `analyst` | `aacd0bbad0fb92cd8` | accepted | `docs/reviews/must-post-engine-contract.md` | plan-gate → approve with suggestions |
| U4 | `architect` | `a8bd7fd19a8b51c33` | accepted | plan v2 (`docs/plans/must-post-engine-contract.md`) | folded into U2 verdict |
| U5 | `coder` | `a14818e0b108167b0` | accepted | `executor.py` mechanism + `test_executor_agent.py` (9 tests, added 1 beyond plan's 8) | `analyst` (U8) → approve |
| U6 | `coder` | `a2f65008841aec9b7` | accepted | `services.py` invariant + `test_services.py` tests 9-12 + `seed_workflows.sh` | `analyst` (U8) → approve |
| U7 | `coder` | `a415f85f654cd7bee` | accepted | docs (BACKLOG/HISTORY/DESIGN) + follow-up `ws:acme`-resolved addendum | — |
| U8 | `analyst` | `a0de4f7e8fa4d2ef6` | accepted | `docs/reviews/must-post-engine-contract-impl.md` | diff-scoped re-gate → approve |
| U9 | — | — | superseded | live `reference`/`ws:acme` `triage@v1` divergence — initial user decision: leave as-is (superseded by U10) | paused → user: leave as-is |
| U10 | `graph-dba` | `ad5d36cab5e14d1e1` | accepted | dropped + re-materialized `ws:acme` `triage@v1` snapshot; found + flagged `reference` also missing `access-request@v1` (concurrent `pytest -q` from U8) | — |
| U11 | `teco` | — | accepted | re-seeded `access-request@v1` into `reference` (documented, idempotent, create-only remedy) — `verify_workflows.sh acme`: both defs in sync | — |
| U3 | `cobb` | `a426b88160f8e47a7` | accepted | `skills/cpg-analysis/SKILL.md` update from architect kaizen inbox | — (side unit, not part of K-027 delivery) |
