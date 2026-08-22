# Workflow Timers / Scheduled Wakeups — Coordination

> **Status:** archived · **Owner:** `teco` · **Tracks:** K-028 (follow-up, not an M-gate)

Coordination record for delivering K-028 (`falkor-chat/docs/BACKLOG.md` §K-028) end to end.
This document, not any agent's context window, is the state of record.

## Goal

`wait` steps park and are released only by an external signal on
`POST /workflow-runs/{id}/input` (DESIGN §6.1/§6.3, decision D-C) — there is no scheduler, so an
SLA/escalation step ("if no approval in 48h, escalate") cannot be expressed. K-028 adds a durable
due-time on a parked run plus something that ticks, reusing the **existing** resume CAS
(`resume_run`, QUERIES §12.4) so a timer wakeup and a concurrent human signal cannot double-resume
the same run.

## Stakeholder pre-decisions (2026-08-21)

1. The architect's recommended ticking mechanism (in-process scheduler / external cron sweep
   endpoint / Redis keyspace notifications) is **pre-approved** — proceed without a pause back to
   the stakeholder, unless the plan gate (`analyst`) contests it or a genuine product-behavior
   fork surfaces; then pause with a crisp decision summary.
2. Normal commit authority applies: `teco` commits each verified deliverable by explicit path as
   units close. **No Claude-Session/Co-Authored-By/attribution footer on any commit.**
3. Standing multi-unit process rules apply: two `analyst` gates (plan + diff-scoped re-gate),
   implementers mutation-test their own green-on-arrival tests, serialize same-file units, never
   run a tree-mutating git command (including inside briefs), verify self-reported recovery
   independently, `teco` runs the integrated baseline suite itself.

## CPG freshness (checked at coordination start, 2026-08-21)

`cpg_falkorchat` exists, `builtAt=2026-08-17T00:40:42Z`, no `sourceCommit` (scratch-copy build
pattern, `sourcePath=/tmp/cpg-src/falkor-chat-server`; real counterpart `falkor-chat/server`).
`git log --oneline --since=2026-08-17T00:40:42Z -- falkor-chat/server` → **6 commits** (K-027
work, including engine/executor-adjacent changes) → **stale**. Every unit's brief carries this
verdict; specialists should prefer reading `server/falkorchat/{executor,services,repository}.py`
directly over trusting a broad structural claim from this CPG.

## Suite baselines (to re-verify, not assumed)

- Query suite: `./scripts/test_queries.sh` — last recorded green at 256/256 (DESIGN §7.1).
- Offline pytest: `cd server && .venv/bin/python -m pytest -q` — last recorded green at 1456
  passed, 3 deselected (HISTORY.md 2026-08-21 entry).
- `test_queries.sh` wipes the `reference` graph at teardown — re-seed with
  `scripts/seed_workflows.sh <wsId>` before any downstream check that needs it, per
  `falkor-chat/AGENTS.md`.

## Ledger

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict | Cost |
|---|---|---|---|---|---|---|
| U1 | `architect` | `a5eef1537a14f63dd` | delivered (v2), gate closed | `docs/plans/workflow-timers.md` | `analyst` → **approve with suggestions** | 212k tok/42 tools (v1) |
| U2 | `analyst` (plan gate) | `a17b7b6f7c36a3c79` | delivered (Pass 2) | `docs/reviews/workflow-timers.md` (Pass 1 + Pass 2) | — | 164k tok/49 tools (pass 1) |
| U3a | `coder` | `a2893bc1925ce752d` | delivered, **defect found — holding, not integrated** | core logic (uncommitted working-tree changes) | — | 307k tok / 134 tools |
| U3b | `coder` | — | blocked (on architect fix + re-gate) | wiring: `schemas.py`, `api.py`, `config.py`, `app.py`, `DESIGN.md`/`AGENTS.md` updates, tests 5-6 | `analyst` (diff re-gate, folded into U4) → — | — |
| U1-fix | `architect` | `a5eef1537a14f63dd` | delivered (v3), gate closed | plan v3 — Direction A (ctx-marker `timerFired`, step-scoped, via `resume_run_with_ctx`) | `analyst` (Pass 3) → **approve with suggestions** | — |
| U3a-fix | `coder` | `a2893bc1925ce752d` | delivered | core logic reworked per v3 marker-guard mechanism + both Pass-3 minors (pytest 1507→1524/3 deselected, query 320/320) | `analyst` (diff re-gate, U4) → pending | — |
| U3b | `coder` | `a2893bc1925ce752d` | delivered | wiring (`schemas.py`/`api.py`/`config.py`/`app.py`) + `DESIGN.md`/`start_server.sh` docs (pytest 1524→1528/3 deselected, query 320/320) | `analyst` (diff re-gate, U4) → pending | — |
| U4 | `analyst` (diff re-gate) | `a17b7b6f7c36a3c79` | delivered, gate closed | `docs/reviews/workflow-timers.md` (Diff Re-gate section) | — → **approve with suggestions** | — |
| U3c | `coder` | `a2893bc1925ce752d` | delivered, re-verified by `teco` | ctx-merge `MAX_CONFIG_LEN` bound fixed (pytest 1528→1529/3 deselected) | (no third gate cycle needed) | — |
| U5 | `qa-engineer` | `a0099eda1f8163fc1` | delivered | `docs/test-plans/workflow-timers.md` + `docs/test-reports/workflow-timers-report.md` | — | **PASS, zero defects** — 175k tok / 59 tools |
| U6 | `teco` | — | delivered | HISTORY.md entry, BACKLOG.md K-028 closeout + K-049 filed, 4× `Status: archived` flips, 4 integration commits | — | — |

## CLOSED (2026-08-21)

K-028 delivered end to end. Final state:

- **Plan** (`docs/plans/workflow-timers.md`, v3, archived) — 3 `analyst` gate passes (needs
  changes → approve with suggestions → approve with suggestions, the last a genuine end-to-end
  trace after a load-bearing mechanism defect was found mid-implementation and fixed).
- **Implementation** (`services.py`/`repository.py`/`schemas.py`/`api.py`/`config.py`/`app.py`/
  `executor.py` + tests) — `analyst` diff re-gate: approve with suggestions, independently
  re-verified (own suite runs, own `GRAPH.PROFILE`, own mutation test, recomputed SHA-lock hash).
  One remaining nit (ctx-merge `MAX_CONFIG_LEN` bound) folded in and re-verified directly by
  `teco` — no third diff-gate cycle.
- **QA** (`docs/test-plans/workflow-timers.md` + `docs/test-reports/workflow-timers-report.md`,
  both archived) — **PASS, zero defects**, 12/12 planned test items, driving the real running
  process end to end including the automatic periodic sweep actually ticking.
- **Docs curated**: `DESIGN.md` §6.1/§6.2/§6.3, `QUERIES.md` §12.16, `scripts/start_server.sh`,
  `BACKLOG.md` (K-028 flipped delivered + K-049 filed), `HISTORY.md` (dated entry). All spot-checked
  by `teco` directly (diffs read, not taken on report).
- **Suites** (independently re-run by `teco`, not just reported): offline `pytest -q` 1456 → 1529
  passed, 3 deselected; query suite (`test_queries.sh`) 320/320.
- **Commits** (4, by explicit path, no attribution footer): plan+review docs; implementation
  (code + tests + `QUERIES.md`/`DESIGN.md`/`start_server.sh`); QA docs; this closeout batch
  (`BACKLOG.md`/`HISTORY.md` + the 4 `Status: archived` flips + this doc).

**Open follow-up, not gating K-028**: **K-049** (`falkor-chat/docs/BACKLOG.md`) — an oversized
value on an indexed graph property crashed the shared `falkordb-dev` container outright, found
incidentally during implementation. Routed to `graph-dba`, unrelated to K-028's own correctness.

**A genuine mid-coordination course correction, worth naming explicitly**: the architect's first
gated fix for the churn risk (a mandatory unconditional fallback transition) was approved by two
separate `analyst` review passes, then found — only when an agent tried to actually drive the
mechanism during implementation — to make the entire feature non-functional. `teco` independently
re-verified the defect against live source before routing it back for a redesign (Direction A: a
step-scoped `ctx.timerFired` marker via the existing `resume_run_with_ctx` CAS), which a third,
deliberately more skeptical `analyst` pass traced end to end and confirmed. This is the single
biggest process lesson from this run: **static plan review, however careful, can approve a
mechanism that doesn't actually work — only driving it (a real test, or the running system)
proves it does.** Recorded to `kaizen_team` by `teco` below.

### U3c + integrated baseline (2026-08-21)

All U3 sub-units complete. `teco` independently re-ran the full integrated baseline itself (not
trusting `coder`'s self-report): offline `pytest -q` → **1529 passed, 3 deselected**; re-seeded
`reference`/`ws:acme` (`bootstrap_schema.sh acme` + `seed_workflows.sh acme`, `verify_workflows.sh`
confirms in sync); `./scripts/test_queries.sh` → **320/320**; re-seeded again after (it wipes
`reference` at teardown), environment ready for QA.

**Incident, worth its own backlog item (2026-08-21).** While constructing the U3c bound-violation
test, `coder`'s first attempt used an oversized **step key** (an indexed/constrained property, not
an opaque ctx value) to trigger the length bound — publishing that def **crashed the shared dev
FalkorDB instance outright**: the connection dropped and the `--rm falkordb-dev` container vanished
entirely from `docker ps -a`, reproduced twice across independent restarts. Root cause not
established (`--rm` meant no logs survived the crash instant). `coder` switched the test to an
oversized **ctx** value (opaque, unindexed) instead — confirmed safe, no restart-worthy work
remains in this coordination's own scope. This affects the **shared** dev instance, which also
hosts `cpg_falkorchat`, `kaizen_team`, and other workspaces — real blast radius beyond K-028.
`coder` filed a `kaizen_team` entry flagged for `graph-dba`. `teco` judgment: this is a real
reliability defect (crash-on-oversized-indexed-property write) that deserves an actual backlog
item, not just a kaizen note, given the shared-instance blast radius — filing as **K-049** at
closeout (U6), routed to `graph-dba` for root-cause/hardening. Out of scope for K-028 itself (K-028
never writes to an indexed property with an unbounded value — the fix landed on the opaque-ctx
side, which is genuinely safe).

### U4 outcome (2026-08-21)

`analyst` independently re-verified everything (own suite runs, own `GRAPH.PROFILE`, own
mutation test via a scratch-copy `services.py` load against real FalkorDB, own recomputed
SHA-lock hash) rather than trusting `coder`'s self-report. **Verdict: approve with suggestions.**
One non-blocking nit: `sweep_due_workflow_runs`'s ctx merge doesn't apply the `MAX_CONFIG_LEN`
bound `submit_workflow_input`'s own merge enforces — low-probability (the merged value is the
sweep's own short, def-controlled step key, not attacker-supplied), real in principle, easy fix,
currently untested. `teco` judgment: fold this in now (one more small `coder` dispatch) rather
than file as a fast-follow, since it's cheap and this coordination has repeatedly found that
"cheap, non-blocking" nits are worth closing before they're forgotten. No third diff-gate cycle —
`teco` will re-verify the fix directly.

### U3a+U3b implementation notes (2026-08-21)

Both delivered by `coder` (same agent throughout). A second genuine plan-v3 defect surfaced and
was fixed pragmatically during U3b (not blocking, not routed back to `architect`): the plan
specified `DEFAULT_SWEEP_LIMIT`/`MAX_SWEEP_LIMIT` defined in `services.py` and imported into
`schemas.py` — the reverse of the working `MAX_CONFIG_LEN` precedent it claimed to mirror — a real
circular import on every module-load order. `coder` fixed it by defining the constants in
`schemas.py` and having `services.py` import from there (the non-deadlocking direction, functionally
identical). Flagging for `analyst`'s diff re-gate to confirm the fix is sound and for a possible
one-line plan-doc correction at `teco`'s discretion (non-blocking, plan is historical once
implementation is accepted).

### Pass 2 outcome (2026-08-21)

`analyst` re-gated v2: **approve with suggestions.** All 3 Pass-1 majors + the minor verified
closed against live source. Two new, non-blocking findings surfaced (both consequences of the
Finding-1 fix, not reopenings): (a) new Major — the mandatory default-arm invariant forecloses the
existing "not yet, keep waiting" negative-signal resume pattern (`proof_defs.py`'s `provision`
step) for any step that also adopts a timer; doesn't block the backlog's own approval-step example
(which already rejects any non-`approve`/`reject` value at 400 before reaching the executor) but is
an undocumented landmine for a future def author combining both patterns — fix is to document the
trade-off, not redesign. (b) new Minor — the invariant's "guard normalizes to `''`" check must be
specified against `_serialize_opaque` (what the runtime actually stores/compares), not
`_normalize_opaque` (`_normalize_opaque(None) is None`, not `""`, which would wrongly reject a
default arm expressed by omitting `guard` rather than writing `guard: ""` explicitly).
Coordinator decision: fold both into the implementation units below as cheap additive fixes —
no third plan-gate cycle, per the review's own recommendation.

### CRITICAL — U3a implementation surfaced a load-bearing defect both analyst passes missed (2026-08-21)

`coder` (U3a), while writing the CAS-contention/sweep tests, found that the v2 mandatory-
unconditional-fallback-arm invariant (the fix for Pass-1 Finding 1) makes the feature it exists to
enable **unreachable**: `executor._drive_loop`'s `_select_transition` is called on **every**
evaluation of a step, including the step's very first arrival, not only on a resumed evaluation —
and `guards.evaluate_guard` fires an unconditional (`guard: ""`) transition unconditionally
whenever it is reached (`guards.py:223-224`, `if not guard: return GuardVerdict(decision=True,
...)`). So a `wait`/`human` step that satisfies the new invariant (carries an unconditional
fallback arm) **never reaches OUTCOME B (suspend) at all** — it advances via OUTCOME A on first
evaluation, before ever parking. Independently verified by `teco` by re-reading
`executor.py:471-510` (the while-loop's ordering) and `executor.py:975-990` (`_select_transition`)
directly — confirmed, not taken on `coder`'s word. Neither `analyst` pass caught this: Pass 2
verified the *sort order* (`guard == "", order, to`) resolves ties toward conditional-first, but
did not check whether the unconditional arm also fires on the step's first-ever evaluation, before
any suspend has happened.

**Consequence:** the gated v2 plan's central anti-churn mechanism is unworkable as specified. This
is a technical feasibility defect, not a product-behavior question — the desired end capability
(SLA-escalation timers on a parked step) is unchanged; only the mechanism needs to change. Per the
stakeholder's standing instruction for this coordination ("settle execution mechanics yourself...
escalate only genuine product choices"), `teco` is not pausing to the stakeholder for this —
routing directly back to `architect` for a v3 redesign of this one mechanism, then a fresh
`analyst` Pass 3 gate before resuming implementation. `coder`'s U3a code (the invariant machinery
itself aside) — `_wait_due_at`, `find_due_wait_candidates`, the sweep's CAS-reuse/bucketing flow —
remains valid and is expected to be largely reusable; **held uncommitted** pending the v3 fix.
`coder` also wrote a `kaizen_team` entry (author `coder`) capturing the underlying guard-evaluation
fact.

### v3 resolution (2026-08-21)

`architect` delivered v3 (Direction A: a step-scoped `ctx.timerFired` marker, written atomically
via the existing `resume_run_with_ctx` CAS, replacing v2's broken unconditional-arm invariant with
a genuinely conditional escalation guard). `analyst` Pass 3 traced the full mechanism end to end
against live source (guard resolution/totality, the reserved-key structural enforcement at both
ctx-mutating entry points, the CAS atomicity, the step-scoping fix) and confirmed it actually works
— not a re-approval on faith. **Verdict: approve with suggestions.** Two new cheap minors, neither
blocking: (1) the sweep's step-5.1 "raced" check should also compare `atStepKey`, not just
`status`, to avoid merging a stale scanned `stepKey` onto a run that moved to a different waiting
step in the interim (widens, not creates, the plan's already-accepted "cycle back" residual); (2) a
wrong file:line citation in §5 step 4 (correct: `services.py:1658-1663,1708`). Routed to `coder`
(same agent, resumed) to finish U3a against v3 and fold in both minors.

## Notes

- No prior in-flight state to reconcile: the earlier attempt at this exact goal was killed and
  cleanly reverted (`git log` shows `1502917` reverting `2c50e6e`) before this session began, and
  is treated as not existing — this coordination starts from scratch, not from that content.

## PAUSED — stakeholder decision needed (2026-08-21)

Plan-gate review (`docs/reviews/workflow-timers.md`, `analyst`, agent `a17b7b6f7c36a3c79`)
returned **needs changes**: 3 majors, 1 minor. Full findings in the review doc. Summary:

- **Finding 1 (major, not a stakeholder decision — routes back to `architect` regardless of
  Finding 2's answer):** the sweep's automated resume can trigger unbounded resume→re-park churn
  on a `wait`/`human` step whose only outgoing transition is conditional (no unconditional
  fallback arm) — the one shipped `wait` step (`provision` in `access-request@v1`) has exactly
  this shape. `analyst`'s suggested fix: a new publish-time invariant requiring an unconditional
  default arm on any step declaring a timer key. Fixable within the plan's own constraints.
- **Finding 2 (major, GENUINE PRODUCT-BEHAVIOR FORK — this is the pause trigger):** the plan
  scopes timers to `wait` steps only; `analyst` traced the plan's own justifying quote and found
  it doesn't exist in the backlog text as a scope boundary — it describes `wait`'s *pre-K-028
  baseline* behavior, not an intended restriction. The backlog's actual motivating example
  ("if no approval in 48h, escalate") maps onto the shipped `human`-typed `approval` step, not the
  `wait`-typed `provision` step. Excluding `human` steps means K-028 may ship without closing the
  gap it was filed to close. `analyst` offers two resolutions: (a) confirm `wait`-only is really
  wanted, or (b) broaden scope to both `WAITING_STEP_TYPES` members now, since the plan's own
  analysis shows this costs almost nothing (same parking mechanics, same publish-time check minus
  its type restriction).
- **Finding 3 (major, not a stakeholder decision):** the plan's flagship "injected clock" test
  wires `Services(clock=...)` but never says the test's separately-constructed `WorkflowExecutor`
  must share that same clock callable — as written the test wouldn't test what it claims to.
  Fixable with one added sentence in the plan.
- **Minor 4:** `scripts/start_server.sh`'s header-comment env-var listing is missing from the
  plan's §7 doc-curation checklist.

**Coordinator recommendation:** adopt `analyst`'s option (b) — broaden timer scope to both `wait`
and `human` steps. Reasoning: it directly serves the backlog's own stated purpose (the SLA-
escalation example is a `human`-step scenario), the architect's own plan already concedes the
mechanism generalizes at near-zero cost, and Finding 1's fix (mandatory default-arm invariant)
applies identically to both step types either way — broadening scope adds no new risk surface
beyond what's already being fixed. Recommend folding all of Findings 1/2(b)/3 and Minor 4 into a
single plan-revision dispatch to `architect`, then a fresh (not necessarily full-cost) plan-gate
pass by the same `analyst` (`a17b7b6f7c36a3c79`, via `SendMessage`) before implementation starts.

**Options for the stakeholder:**
1. **(Recommended)** Broaden scope to `wait` + `human` steps (analyst's option b) — proceed
   without further pause once the revision is gated.
2. Keep `wait`-only, accepting the SLA-escalation-on-`human`-approval gap stays open (a narrower,
   still-useful capability; the backlog item's title literally says "workflow timers," so a
   partial delivery is defensible if that's what's wanted) — file the `human` extension as a
   follow-up backlog item instead of building it now.
3. Something else — the stakeholder's call.

U3-U6 are blocked on this decision + the resulting plan revision + re-gate.

### RESUMED — stakeholder decision received (2026-08-21)

Stakeholder confirmed **option 1**: broaden K-028 scope to both `wait` and `human` steps.
Proceeding: single plan-revision dispatch to `architect` (agent `a5eef1537a14f63dd`, resumed by
`SendMessage`) folding in Findings 1, 2(b), 3, and Minor 4 → re-gate by the same `analyst`
(agent `a17b7b6f7c36a3c79`) → implementation → diff re-gate → QA → doc closeout. No further pause
for this decision.
