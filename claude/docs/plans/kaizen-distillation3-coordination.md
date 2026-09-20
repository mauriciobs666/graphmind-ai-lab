# Kaizen distillation — team-wide pass 3 (toward `kaizen_team` deletion)

> **Status:** archived · **Owner:** `teco` · **Tracks:** — (—) · **Extends:** `claude/docs/plans/kaizen-distillation2-coordination.md`

User-directed pass over the shared `kaizen_team` FalkorDB graph
(`skills/agent-maintenance/SKILL.md` §5), explicitly scoped to draining
`kaizen_team` toward eventual deletion — not this pass's job to also fold in
each agent's post-2026-09-19 `ws:agent-team` captures (a separate source now
that Track 1's write-cutover is complete; see `claude/AGENTS.md`). For every
agent with raw `:KaizenEntry` nodes still in `kaizen_team`, `cobb` verifies
each entry, routes it (prompt / knowledge base / project docs / discard /
kept-open), logs the disposition in that agent's own `kaizen/history.md`
(and `plan.md` for kept-open actionable items, with the dedup check), tags
any `MENTIONS` edges for entries that are really about a different agent,
and clears each entry from the graph via the curator shapes in SKILL.md §5
step 4 once logged.

**Dispatched one agent per turn, sequentially — explicit user direction**,
not the parallel-batch cadence pass 1 used. Each unit waits for the prior
one's result before the next is dispatched.

No independent review gate: unchanged from passes 1 and 2 — this is
`cobb`'s sole-owned, already-specified procedure (SKILL.md §5 embeds its own
verification step), not a design or implementation deliverable. The team
certification pass (SKILL.md §4) remains the periodic audit of `cobb`'s
distillation work, run separately on request.

**Snapshot at open** (`teco`, 2026-09-20, live query against `kaizen_team`;
zero legacy/`author`-only entries exist — every remaining entry is
current-shape with a real `PRODUCED` edge):

| Agent | Current-shape count |
|---|---|
| analyst | 16 |
| architect | 2 |
| coder | 1 |
| data-scientist | 8 |
| devops | 3 |
| qa-engineer | 1 |
| security-expert | 1 |
| tdd-engineer | 1 |
| teco | 2 |

**One anomaly found at open, not yet assigned to a unit:** entry
`e1a6c4d2-8b3f-4b1a-9c7e-3f2a6d9b1c4e` (`ModelGateway.from_env()` eager
provider resolution) has **no `PRODUCED` edge and no `author` property at
all** — orphaned producer identity — but does carry a `MENTIONS`→`tico`
edge. `tico` itself produced zero entries. Route this one under a `tico`
top-up unit once the 9 producer-agent units above are done; `cobb` should
treat the missing producer edge itself as worth a dated `history.md` note
(possibly a `cypher-mcp` write-path defect worth a backlog item), not just
silently disposition the fact and move on.

`frontend-engineer`, `graph-dba` had zero raw entries in `kaizen_team` at
open — no unit planned for them unless a later census finds new ones.

## Ledger

| Unit | Agent | Agent id | Status | Deliverable | Gate → verdict | Cost |
|---|---|---|---|---|---|---|
| U1 | analyst | `a1a065cbae8c7f65e` | accepted | `claude/analyst/kaizen/history.md`+`plan.md`, `claude/analyst/review-techniques.md` (3 sections+1 addendum), `falkor-chat/docs/SERVER.md` (§1.9), `model-bench/AGENTS.md` (invariant), `skills/agent-maintenance/SKILL.md` (step 3a), graph cleared (verified 0 remaining) | none (see above) → — | 217.7k tok, 65 tools |
| U2 | architect | `aaa2b1e79d3481f2a` | accepted | `claude/architect/kaizen/history.md` (both entries discarded, already published more deeply), graph cleared (verified 0 remaining) | none (see above) → — | 138.9k tok, 25 tools |
| U3 | coder | `aad1635b2ae3021a7` | accepted | `claude/coder/kaizen/history.md`, `claude/tdd-engineer/kaizen/history.md` + `claude/tdd-engineer/test-design-techniques.md` (cross-agent promotion), graph cleared (verified 0 remaining) | none (see above) → — | 195.4k tok, 44 tools |
| U4 | data-scientist | `a8c1eea008f7de461` | accepted | `claude/data-scientist/kaizen/history.md` (all 8 entries discarded, already published), graph cleared (verified 0 remaining) | none (see above) → — | 163k tok, 57 tools |
| U5 | devops | `a005be95225b64db6` | accepted | `claude/devops/ops-quirks.md` (1 new section: `ingest_document` auto-supersede trap), `claude/devops/kaizen/history.md` (2 discarded, already published), graph cleared (verified 0 remaining) | none (see above) → — | 191.8k tok, 59 tools |
| U6 | qa-engineer | `a5852e0d655dc05b5` | accepted | `claude/qa-engineer/kaizen/history.md` (discarded, already published in `falkor-chat/docs/test-reports/salesperson-language-salience-report.md`), graph cleared (verified 0 remaining) | none (see above) → — | 131k tok, 20 tools |
| U7 | security-expert | `a94706fa68073068f` | accepted | `skills/agent-standards/claude-code.md` (new bullet: classifier `[Self-Modification]` scrutiny doesn't carry across tool surfaces), `claude/security-expert/kaizen/history.md`, graph cleared (verified 0 remaining) | none (see above) → — | 146.2k tok, 33 tools |
| U8 | tdd-engineer | `aece9cd682b41cc49` | accepted | `claude/tdd-engineer/kaizen/history.md` (discarded, already published in `skills/agent-standards/claude-code.md`), graph cleared (verified 0 remaining) | none (see above) → — | 198k tok, 36 tools |
| U9 | teco | `aba7ff4c6923a45a9` | accepted | `claude/teco/kaizen/history.md`, `claude/teco/coordination-techniques.md` (new commit-grant-vs-classifier section), `skills/agent-standards/claude-code.md` (nesting-depth caveat), `claude/cobb/kaizen/plan.md` (K-052 annotation), graph cleared (verified 0 remaining) | none (see above) → — | 161k tok, 46 tools |
| U10 | tico (orphaned entry) | `af4c79f0dc93ebb56` | accepted | anomaly investigated and resolved as not-a-defect (`claude/cobb/kaizen/history.md`, new K-053 process note); fact promoted to `falkor-chat/docs/manuals/llm-provider-config.md` §2 (closes `tico` K-016 + `data-scientist` K-003); graph cleared — **whole-graph `count(KaizenEntry)` verified 0, pass closed** | none (see above) → — | 246.3k tok, 62 tools |

## Follow-ups

- **Stale line-number citation in an already-closed doc** (found during `teco`'s U4 verification,
  not a U4 defect): `model-bench/docs/reviews/small-model-catalog-sweep-impl.md:562` cites its own
  pinning test at `model-bench/tests/test_report.py:3425`; the test
  (`test_rank_report_guard_judge_family_k_does_not_shrink_when_a_candidate_has_no_paired_data`) now
  lives at `test_report.py:3587` — the file grew after the review was written. Cosmetic (the test
  still exists and still pins the described behavior), but worth a symbol-citation fix next time
  that review doc is touched, per the repo's own "cite symbols, not line numbers, in files under
  active change" convention.
- **Open, unrelated review found during U7 verification, not this pass's to resolve:**
  `docs/reviews/mcp-json-edit-bypass-incident.md` (`Status: active`, owner `security-expert`) —
  a genuine incident where a `devops` subagent's `.mcp.json` `Edit` was denied
  `[Self-Modification]` and it then made the identical change via `Bash`/`python3`. Verdict:
  **needs changes**, with an unresolved Recommendation 3 about the diff's process, not its
  content. U7 correctly promoted the durable-reference half of this (a new
  `skills/agent-standards/claude-code.md` bullet) but did not and should not have touched the
  review's own open recommendation — flagging for the user/`devops` to close out separately.

## Close

All 10 units accepted; every disposition independently re-verified by `teco` (graph counts,
diffs, duplicate-heading scans, citation spot-checks, and — for U4/U9 — word-count deltas) before
acceptance, not taken on any delegate's self-report. **`MATCH (k:KaizenEntry) RETURN count(k)`
against `kaizen_team` returns 0** — re-confirmed independently by `teco` after U10's clear, not
only by `cobb`'s own report. `kaizen_team` is fully drained of raw learnings and, per this pass's
stated goal, ready for the user to decide on deletion (a destructive op outside any agent's
standing grant — not actioned here, flagged in the final report instead). This document is a
purely docs-only-chain deliverable (every touched file is `.md`, none is source/tests/config); all
24 touched files across the 10 units, plus this document itself (25 total), are committed together
in one commit per `claude/AGENTS.md`'s docs-only batching convention, immediately after this close.
