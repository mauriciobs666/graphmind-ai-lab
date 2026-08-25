# Kaizen — Improvement Plan: coder

> Forward-looking backlog for the `coder` agent.
> Status: 🔵 proposed · 🟡 in-progress · ✅ done (then moved to history.md) · ⚪ rejected/deferred
> Last reviewed: 2026-07-24

## Active

| ID | Added | Priority | Status | Summary |
|------|------------|----------|--------|---------|
| K-002 | 2026-06-20 | medium | 🔵 | Validate the architect→coder handoff end-to-end and confirm the coder can execute an architect plan without re-investigating. |
### K-002 — End-to-end handoff validation
- **Status:** 🔵 proposed
- **Priority:** medium
- **Rationale:** The architect→coder contract is designed but unproven; the coder should be able to pick up an architect plan cold (isolated context) and build it.
- **Proposed change:** Run a real feature through architect→coder; capture what the plan was missing; feed back into both prompts.
- **Notes:** Paired with architect K-002. **Update 2026-07-08:** the handoff transport is now fixed — the coder receives the architect's plan as a document path (`<component>/docs/plans/<slug>.md`) and reads the file itself; the live validation run remains. **Update 2026-07-09:** the *contract* is now proven live — in the teco K-001 run (falkor-chat M3 slice 1), `tdd-engineer` executed an architect plan cold from its path with no re-investigation (architect K-002 ✅). What remains for this item is coder-specific only: one live run with the **coder** as the implementer half. **Update 2026-07-24 (inbox distillation):** the coder has since run as the implementer half repeatedly in falkor-chat (K-022 Landing 2 M-2, K-024 U2/U4/U4b), and its inbox entries show it reading plan docs by path and reporting blockers rather than guessing (e.g. the zero-transition `IndexError` was "not fixed (out of unit scope); reported to teco" — correct scope discipline). That is strong circumstantial evidence, but the run *reports* weren't reviewed here, so the item stays open pending a deliberate look at one end-to-end architect→coder run.

## Parking lot / ideas
- **Judged and kept, do not re-litigate (2026-08-24, C6 lint).** Three restatements will read as class-7 duplicates to a future dedup sweep; all are keeps under finding 5 ("needed twice", not "said twice"):
  - **"Don't claim what you didn't run" vs. step 5's report procedure** — prohibition vs. the mechanics of an honest report (show output; report `passed`/`skipped`/`deselected`). Same pair `qa-engineer` certified at C5.
  - **"Ask before destructive or environment-changing actions" vs. step 2's bootstrap ask** — two decision points, each carrying its own subagent carve-out. The carve-out duplication is an `agent-maintenance` §4 check-3 **certification requirement**, not style.
  - **"Minimal blast radius" vs. "Don't silently exceed scope"** — scope of edits vs. reporting obligation. (The *third* member of that trio, `:11`, is a genuine contradiction — see K-003.)
- A "definition of done" checklist (suite green, behavior covered, no scope creep, honest run report) the coder self-checks before reporting completion.
- Consider whether the coder should delegate the test-writing step to `tdd-engineer` when strict TDD is required, rather than doing it itself.
