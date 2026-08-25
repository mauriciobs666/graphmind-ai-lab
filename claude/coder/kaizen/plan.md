# Kaizen — Improvement Plan: coder

> Forward-looking backlog for the `coder` agent.
> Status: 🔵 proposed · 🟡 in-progress · ✅ done (then moved to history.md) · ⚪ rejected/deferred
> Last reviewed: 2026-07-24

## Active

| ID | Added | Priority | Status | Summary |
|------|------------|----------|--------|---------|
| K-002 | 2026-06-20 | medium | 🔵 | Validate the architect→coder handoff end-to-end and confirm the coder can execute an architect plan without re-investigating. |
| K-003 | 2026-08-24 | low | ✅ | `:11` and `:35` prescribe different actions for the same trigger (mid-build plan defect: stop vs. note-and-continue) |
| K-004 | 2026-08-24 | low | ✅ | `:12`'s conventions rule doesn't say whether project norm or local neighbour wins when a file deviates — sibling of `tdd-engineer` K-006, fix together |

> Done: K-001 (2026-07-09) — efficiency-based routing boundary with `tdd-engineer`; descriptions now use objective task-shape triggers (detailed plan → coder; bug fix / safety-net refactor / test work / clear-contract feature → tdd-engineer) and cross-reference symmetrically. See history.md.

### K-002 — End-to-end handoff validation
- **Status:** 🔵 proposed
- **Priority:** medium
- **Rationale:** The architect→coder contract is designed but unproven; the coder should be able to pick up an architect plan cold (isolated context) and build it.
- **Proposed change:** Run a real feature through architect→coder; capture what the plan was missing; feed back into both prompts.
- **Notes:** Paired with architect K-002. **Update 2026-07-08:** the handoff transport is now fixed — the coder receives the architect's plan as a document path (`<component>/docs/plans/<slug>.md`) and reads the file itself; the live validation run remains. **Update 2026-07-09:** the *contract* is now proven live — in the teco K-001 run (falkor-chat M3 slice 1), `tdd-engineer` executed an architect plan cold from its path with no re-investigation (architect K-002 ✅). What remains for this item is coder-specific only: one live run with the **coder** as the implementer half. **Update 2026-07-24 (inbox distillation):** the coder has since run as the implementer half repeatedly in falkor-chat (K-022 Landing 2 M-2, K-024 U2/U4/U4b), and its inbox entries show it reading plan docs by path and reporting blockers rather than guessing (e.g. the zero-transition `IndexError` was "not fixed (out of unit scope); reported to teco" — correct scope discipline). That is strong circumstantial evidence, but the run *reports* weren't reviewed here, so the item stays open pending a deliberate look at one end-to-end architect→coder run.

### K-003 — `:11` and `:35` give the same trigger two different actions
- **Status:** 🔵 proposed
- **Priority:** low
- **Rationale:** Step 1 of "What you optimize for" says *"If you discover the plan is wrong or incomplete mid-build, **stop** and say so with a concrete proposal."* The "Don't silently exceed scope" guardrail says *"Surface **plan defects**, better alternatives, and tempting-but-out-of-scope work **as notes for the user** — don't just do them."* Items 2 and 3 of that list plainly mean note-and-continue, so the collision is item 1 only — but for a mid-build plan defect the two rules say halt and don't-halt. It bites hardest when `coder` runs **delegated**: "stop" there means returning the unit undone, with no ability to ask. Found by `cobb`'s §7 lint during C6 (`claude/docs/plans/prompt-waste-reduction.md`), which correctly dissented from that unit's initial reading that the three scope statements were cleanly distinct.
- **Proposed change:** make the guardrail defer to step 1 on the one overlapping case, e.g. *"…don't just do them. A plan defect that blocks the current step stops the work (step 1); one that doesn't, becomes a note."*
- **Notes:** Deliberately **not** bundled into C6 — it is a rule change, and §4.0's rollback contract keeps rule changes out of a compression commit.

### K-004 — `:12` doesn't say whether the project norm or the local neighbour wins
- **Status:** ✅ **closed 2026-08-25** — all three instances fixed in one commit with a byte-identical sentence, enforced by `audit-team.sh` check 10 (fails at some-but-not-all). See `history.md`.
- **Priority:** low
- **Rationale:** *"Match the language, framework, structure, naming, and idioms **already in the codebase**. Discover conventions by reading **neighboring code**"* mixes a project-scope authority with a local-scope discovery heuristic. They agree in a consistent codebase and diverge in exactly the case worth having a rule for — a file that deviates locally. `:29` ("Its conventions… win over your defaults") is **not** part of the collision: it runs on the project-vs-*agent* axis, and points away from the agent's habits just as `:12` does.
- **Proposed change:** name the tiebreak, e.g. *"…where a file deviates locally from the project norm, match the file — a mixed-style file is worse than either style."*
- **Notes:** Sibling of **`tdd-engineer` K-006** (the stronger instance: two co-equal imperatives) and of the same defect in **`frontend-engineer`** (`:17`/`:56`/`:77`). **Fix all three together or none** — `coder` and `tdd-engineer` are declared routing partners that split work by task shape, so divergent convention-precedence rules across them is an `agent-maintenance` §4 **check-5 boundary-reciprocity** problem, not just a per-file nit.

## Parking lot / ideas
- **Judged and kept, do not re-litigate (2026-08-24, C6 lint).** Three restatements will read as class-7 duplicates to a future dedup sweep; all are keeps under finding 5 ("needed twice", not "said twice"):
  - **"Don't claim what you didn't run" vs. step 5's report procedure** — prohibition vs. the mechanics of an honest report (show output; report `passed`/`skipped`/`deselected`). Same pair `qa-engineer` certified at C5.
  - **"Ask before destructive or environment-changing actions" vs. step 2's bootstrap ask** — two decision points, each carrying its own subagent carve-out. The carve-out duplication is an `agent-maintenance` §4 check-3 **certification requirement**, not style.
  - **"Minimal blast radius" vs. "Don't silently exceed scope"** — scope of edits vs. reporting obligation. (The *third* member of that trio, `:11`, is a genuine contradiction — see K-003.)
- A "definition of done" checklist (suite green, behavior covered, no scope creep, honest run report) the coder self-checks before reporting completion.
- Consider whether the coder should delegate the test-writing step to `tdd-engineer` when strict TDD is required, rather than doing it itself.
