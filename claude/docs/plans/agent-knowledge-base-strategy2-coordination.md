# Agent knowledge-base strategy — Stages 1-7 planning coordination

> **Status:** archived · **Owner:** `teco` · **Tracks:** K-030 (`claude/cobb/kaizen/plan.md`)

Successor to `claude/docs/plans/agent-knowledge-base-strategy-coordination.md` (archived — that
one coordinated Stage 0 only, executed and closed, commits `6d834f0`/`c00d42d`; not reopened here).

This coordination covers the design work that Stage 0's closing note left open: the plan's
substrate-stage sections (everything from §1 onward, formerly blocked) now need revision against
the requirements doc's 2026-09-17 resolution (commit `9c2a41f`) — `claude/docs/requirements/
agent-knowledge-base-strategy.md` is back to **Ready for design**: Option B chosen on the merits
(not automatically, per the plan's own prior caveat), plus an expanded, two-track scope (FR-8/FR-9
raw-capture migration first, FR-2-FR-7 distilled-KB ingestion second). This coordination produces
the revised design only — no implementation dispatched from it.

**CPG note (recorded once, for whichever unit needs it):** `cpg_falkorchat` exists but is stale for
this purpose — built 2026-09-12 (source `ca25a20`), which predates `document-ingestion2`'s full
shipping (`8906878`, Stage F close) by several stages, so it does not see that feature's later-stage
code (including the new update/delete/list methods that resolved the original Option A/B blocker).
Not rebuilding for this coordination — the shipped feature's own archived plan/review/test-report/
HISTORY entries, plus a direct source read, are more reliable for design-altitude work than the CPG
in its current state.

## Ledger

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict | Cost |
|---|---|---|---|---|---|---|
| U1 | `architect` | `a27ddf9a3faa372ff` | delivered | `claude/docs/plans/agent-knowledge-base-strategy.md` (revised in place) | — → — | 255636 tok / 31 tools |
| U2 | `data-scientist` | `a6847012c0cd2f0bd` | delivered | `claude/docs/plans/agent-knowledge-base-strategy-ml.md` (v2, revised in place) | — → — | 141344 tok / 21 tools |
| U3 | `analyst` | `abff2864f248f6549` | delivered | `claude/docs/reviews/agent-knowledge-base-strategy.md` | `analyst` (`abff2864f248f6549`) → needs changes (1 blocker, 1 major, 2 minor) | 178480 tok / 35 tools |
| U4 | `architect` | `a27ddf9a3faa372ff` (resumed) | delivered | `claude/docs/plans/agent-knowledge-base-strategy.md` (revised in place) | — → — | 357213 tok / 21 tools |
| U5 | `analyst` | `abff2864f248f6549` (resumed) | delivered | `claude/docs/reviews/agent-knowledge-base-strategy.md` (Pass 2, in place) | `analyst` → approve with suggestions (1 new major from interaction check) | 256765 tok / 10 tools |
| U6 | `architect` | `a27ddf9a3faa372ff` (resumed) | accepted | `claude/docs/plans/agent-knowledge-base-strategy.md` (revised in place) | verified directly by `teco` against source (`modelconfig.py:729`, `embedding.py:123-126`, `config.py:276-284`) | 392537 tok / 14 tools |
| U7 | `cobb` | `a248a04110e45955b` | accepted | K-030 entry rewritten, `claude/cobb/kaizen/plan.md` (table row + narrative, both spots) | verified directly by `teco` (`grep -n K-030`, both spots read in full, consistent) | 82074 tok / 8 tools |

## Notes

- Sequencing: U1 (plan revision) → U2 (ML method note revision, against U1's revised plan) → U3
  (design-level review of both together — this feature's substrate design has never had one; only
  Stage 0's implementation got a diff-scoped review) → U4 (fix) → U5 (narrow Pass 2 re-gate,
  same reviewer) → U6 (one more evidence-backed fix from Pass 2's interaction check) → U7
  (backlog documentation curation).
- U1 and U2 each independently decide, and state, revise-in-place vs. successor-document under
  `AGENTS.md`'s document convention (rule 5) for their own owned document — not decided here.
  Stage 0's own section of the current plan is out of scope for either and must not be reopened.
  Both kept the plan/ml-note revised in place (no successor); the review doc is new (no prior
  design-level review of this artifact existed to revise).

## Close-out (2026-09-17)

All seven units accepted. Final state: `claude/docs/plans/agent-knowledge-base-strategy.md` and
`claude/docs/plans/agent-knowledge-base-strategy-ml.md` (Version 2) both revised in place, carrying
the two-track (FR-8/FR-9 → FR-2-FR-7) design against the requirements doc's 2026-09-17 resolution
(commit `9c2a41f`). `analyst`'s design review (`claude/docs/reviews/agent-knowledge-base-strategy.md`)
went Pass 1 needs changes (1 blocker, 1 major, 2 minor) → all four fixed by `architect` → Pass 2
approve with suggestions, with one further factual correction from Pass 2's own interaction check
(a wrong stated justification for the workspace-pinning decision — the decision itself never
changed) independently verified against source by both `architect` and `teco` directly, not taken
on report. K-030's backlog entry (`claude/cobb/kaizen/plan.md`) rewritten to match current state,
left at 🟡 in-progress (design done, no implementation dispatched).

**This coordination produced the revised design only, as scoped — no implementation was
dispatched.** Concretely still open, per the plan's own closing section and K-030's rewritten
entry: a `graph-dba` design note for §4.1's attribution-fix schema, `devops` work to stand up the
dedicated `ws:agent-team` falkor-chat process, and (before Track 2 Stage 7 specifically) `cobb`'s
own sign-off on `skills/agent-kb-retrieval/SKILL.md` as the retrieval-artifact placement — none of
that dispatched here; a future coordination picks up Track 1 Stage 1 implementation.
