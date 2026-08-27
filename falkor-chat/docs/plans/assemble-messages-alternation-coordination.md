# `_assemble_messages` role-alternation fix — coordination log (K-048)

> **Status:** archived · **Owner:** `teco` · **Tracks:** K-048 (post-M3 follow-up, not a milestone gate)

Coordinator: `teco`. Started 2026-08-26, on the user's "let's work on K-048".

## Scope

`docs/BACKLOG.md` → `### K-048`. `WorkflowExecutor._assemble_messages`
(`server/falkorchat/executor.py:910-931`) unconditionally appends a trailing `role: user`
`CONTEXT` block after the role-mapped thread turns. When the thread's last turn before this
block is itself `user`-authored (structural on `intake`'s first call, and on `research`→`answer`
handoffs), the request ends with two consecutive `user`-role messages. Live-verified against
LM Studio's `mistralai/ministral-3-3b` (and alias `mistralai_ministral-3-3b-instruct-2512`): hard
`HTTP 400`, Jinja alternation error. Qwen's template tolerates the shape today, so nothing is
on fire in production, but this is model-agnostic message-assembly debt per the backlog item.
Full evidence trail: `docs/plans/ministral-reprobe-ml.md` §4.2/§4.4;
`docs/reviews/ministral-reprobe.md` (approve).

Confirmed **not** inside the SHA-locked `_drive_loop` (`executor.py:451-514`, hash
`71055f756280`) — `_assemble_messages` (`:910`) and its caller `_run_agent_node` (`:615`) sit
outside it; no re-lock ceremony needed for this fix.

**CPG:** `cpg_falkorchat`, built `2026-08-26T22:27:22Z` (scratch source path maps to
`falkor-chat/server`, no `sourceCommit` stamped — git-less scratch build). One commit since
build (`da10d57`, K-049) touches `api.py`/`schemas.py`/`services.py`/tests only, not
`executor.py` — fresh for this unit's purposes.

## Plan (per BACKLOG owner sequence: `architect` → `tdd-engineer`, both analyst-gated)

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict | Cost |
|---|---|---|---|---|---|---|
| U1 | `architect` | `ac543caf570663192` | delivered | `docs/plans/assemble-messages-alternation.md` | `analyst` → — | 90.6k tok, 30 tool uses |
| U1-gate | `analyst` | `a76cf930395b7bfb0` | accepted | `docs/reviews/assemble-messages-alternation.md` | approve with suggestions | 91.9k tok, 25 tool uses |
| U2 | `tdd-engineer` | `a9a8378caa600a0bc` | delivered | code diff (`executor.py` + tests) + HISTORY/BACKLOG updates, uncommitted | `analyst` → — | 122.5k tok, 60 tool uses |
| U2-gate | `analyst` | `a76cf930395b7bfb0` | accepted | `docs/reviews/assemble-messages-alternation.md` (Pass 2) | approve | 106.3k tok, 34 tool uses |

U2 is dependent on U1's approved plan. Each unit's review gate is dispatched immediately after
its delivery (analyst re-uses the same review doc for Pass 2 on U2, per the collision rule for
`reviews/`).

## Log

- 2026-08-26: coordination opened; U1 dispatched to `architect`.
- 2026-08-26: U1 delivered (`docs/plans/assemble-messages-alternation.md`); U1-gate dispatched to
  `analyst`.
- 2026-08-26: U1-gate returned **approve with suggestions** (3 minor findings: docstring
  completeness, a broken internal cross-reference in the plan's own prose, promote one
  "recommended" test to mandatory). Two actionable ones folded directly into U2's brief; the
  prose cross-reference nit had no code implication, left alone.
- 2026-08-26: U2 dispatched to `tdd-engineer` with the approved plan + folded-in suggestions.
- 2026-08-26: U2 delivered — `_append_turn` helper implemented exactly per plan §3, 3 new tests
  (characterization/crash-shape/sibling-shape), full offline suite 1782→1785 passed (exactly +3),
  mutation-tested, `_drive_loop` SHA-lock re-verified unchanged (`71055f756280`), live regression
  pass green (`test_triage_flow_runs_end_to_end_against_live_llm`). `HISTORY.md`/`BACKLOG.md`
  updated in the same change. Left uncommitted for `teco`. Spot-checked the diff against the plan
  myself before gating — matches exactly.
- 2026-08-26: U2-gate (Pass 2, same reviewer resumed) returned **approve**, no new findings — all
  three Pass-1 findings resolved as expected. Independently re-ran the full suite (1785/4
  deselected, matches), reconstructed pre-fix behavior from `git show HEAD:...` to confirm the
  new tests aren't tautological, re-ran the SHA-lock recipe.
- 2026-08-26: committed `executor.py`, `test_executor_agent.py`, `HISTORY.md`, `BACKLOG.md`, the
  plan, both review passes, and this coordination doc as one unit. K-048 fully integrated;
  coordination closed, `Status` → `archived`.
