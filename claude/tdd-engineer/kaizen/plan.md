# Kaizen — Improvement Plan: tdd-engineer

> Forward-looking backlog for the `tdd-engineer` agent.
> Status: 🔵 proposed · 🟡 in-progress · ✅ done (then moved to history.md) · ⚪ rejected/deferred
> Last reviewed: 2026-07-09 (routing boundary with `coder` made efficiency-based — see history.md)

## Active

| ID | Added | Priority | Status | Summary |
|-------|------------|----------|--------|---------|
| K-006 | 2026-08-24 | low | ✅ | "Idiomatic, clean production code" states two conventions rules with no precedence between them. See below — surfaced by the C5 prompt-waste lint, deliberately not bundled into it. |

> Done: K-001, K-002, K-004, K-005 (2026-06-05) — see history.md.
> Deferred: K-003 (2026-06-05) — keep tools unconstrained for flexibility; see history.md.

### K-006 — The conventions bullet never says which convention wins
- **Status:** ✅ **closed 2026-08-25** — all three instances fixed in one commit with a byte-identical sentence, enforced by `audit-team.sh` check 10 (fails at some-but-not-all). See `history.md`.
- **Priority:** low — an ambiguity, not a gap; both readings produce reasonable code.
- **Rationale:** The Principles bullet "**Idiomatic, clean production code**" opens with "Follow the
  language and project conventions you observe" and closes with "Match the surrounding code's
  style." These answer *differently* when the file being edited deviates locally from the project
  norm — and that is a decision point the agent stands at on essentially every edit. The C5
  compression sweep flagged the pair as a class-7 duplicate and kept it precisely because the two
  sentences are not equivalent; the lint's sharper reading is that the non-equivalence is the
  defect. It is an **ambiguity finding wearing class-7 clothes**, which is why a dedup sweep would
  never fix it — the sweep's only available move is to delete one, and deleting either silently
  picks a winner.
- **Proposed change:** state the precedence in one sentence, e.g. "Follow the language and project
  conventions you observe; where the file you're editing deviates locally, match the file, not the
  project norm — a mixed-style file is worse than either style."
- **Provenance:** surfaced by `cobb`'s §7 lint during C5 of `claude/docs/plans/prompt-waste-reduction.md`.
  **Pre-existing, not introduced by C5.** Deliberately *not* folded into that commit: it is a rule
  **change**, and bundling it would make the commit non-revertible as a pure waste-reduction change,
  against §4.0's rollback contract.

### K-003 — Tool permissions decision  ⚪ DEFERRED (2026-06-05)
- **Status:** ⚪ deferred — user chose to keep `tools` unconstrained for now.
- **Decision:** No `tools` key; the agent continues to inherit all tools, preserving flexibility to spawn subagents and fetch docs mid-task. The focused-set restriction (`Read, Edit, Write, Bash, Grep, Glob`) was considered and declined.
- **Revisit if:** the agent's broad tool access causes surprise or unwanted actions in practice.

## Parking lot / ideas
- ~~Handoff symmetry (2026-07-11 team certification): RCA-by-path consumption + reciprocal qa-engineer boundary.~~ *(✅ Resolved same-day: workflow step 1 names the RCA input, the description routes acceptance passes to qa-engineer, and the pair is in `audit-team.sh` `BOUNDARY_PAIRS`. See history.md.)*
- State explicitly that the agent does **not** auto-commit (the harness rule is "commit only when asked") — avoids surprise commits given the "commit-sized increments" language.
- Add a one-liner that coverage % is a guide, not a goal — pin behaviors, don't chase numbers.
- Note on flaky tests: quarantine + diagnose root cause rather than re-run until green.
- ~~Consider whether `opus` is warranted vs. `sonnet` for cost~~ — **resolved 2026-07-27**: the `model` pin was removed team-wide; the agent inherits the session/system default.
- Optional enrichment: a brief nod to advanced test techniques where they fit — table-driven/parameterized tests for boundary sweeps, property-based testing for invariants, and mutation testing as a *coverage-quality* check (does the suite actually catch injected faults?). Low priority; the prompt is deliberately lean, so only add if it earns its keep.
