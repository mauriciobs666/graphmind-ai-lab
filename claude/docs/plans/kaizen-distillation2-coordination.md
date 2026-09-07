# Kaizen distillation — team-wide pass 2

> **Status:** active · **Owner:** `teco` · **Tracks:** — (—) · **Extends:** `claude/docs/plans/kaizen-distillation-coordination.md`

Second routine curation pass over the shared `kaizen_team` FalkorDB graph
(`skills/agent-maintenance/SKILL.md` §5), covering the 196 raw `:KaizenEntry`
nodes accumulated since pass 1 closed. Procedure is unchanged and lives in
the skill: `cobb` verifies each entry (re-deriving the fact, not just
confirming the citation still exists), routes it (prompt / knowledge base /
project docs / discard / kept-open), logs the disposition in the producing
agent's `kaizen/history.md` (plus `plan.md` for kept-open actionable items,
with the `entryId` dedup check), tags `MENTIONS` for entries really about a
different agent, and only then resolves the edge or clears the node via the
curator shapes.

**Two deliberate differences from pass 1**, both at the user's direction:

- **Strictly sequential — one agent at a time**, not parallel batches of six.
- **Large inboxes are chunked** by date range, capped at ~12–15 entries per
  dispatch. Pass 1's cost data (`kaizen-distillation-coordination.md`) shows a
  20-entry unit burning 240.8k tokens / 95 tool uses; a 44-entry unit would run
  out of turns and leave a partially-cleared inbox with an incomplete history
  log. Each chunk gets its own dated `history.md` disposition entry, which is
  normal §5 bookkeeping, not a workaround.

**No independent review gate** — precedent from pass 1, unchanged: this is
`cobb`'s sole-owned, already-specified procedure with its own embedded
verification step, not a design or implementation deliverable. The §4 team
certification pass is the periodic audit of `cobb`'s distillation work and is
run separately on request.

`frontend-engineer` had zero raw entries at open — no unit dispatched.
`devops`'s unit includes one **legacy-shape** entry
(`8301b20f-3e57-4761-a333-f1998bcbfcf1`, no `PRODUCED` edge, `author` null)
reached only via the `MENTIONS`→`devops` edge that pass 1's U8 attached; it is
cleared by resolving that one `MENTIONS` edge, and `otherRemaining` will be 0,
so the node itself goes.

`teco` commits each accepted unit's files by explicit path — `cobb` runs as a
delegated subagent here, so the universal interactive-mode commit grant does
not apply to it (`claude/AGENTS.md`, "Git-commit authority").

## Ledger

Order is smallest inbox first (user's choice), so the six light inboxes clear
before the heavy ones. Counts are raw entries in scope at open.

| Unit | Agent (scope) | Agent id | Status | Deliverable | Gate → verdict | Cost |
|---|---|---|---|---|---|---|
| U1 | cobb (1: 2026-09-06) | `a2c2c175f4d6976cb` | in-flight | `claude/cobb/kaizen/*`, graph cleared | none (see above) → — | — |
| U2 | security-expert (2: 08-26, 08-30) | — | queued | `claude/security-expert/kaizen/*`, graph cleared | none → — | — |
| U3 | devops (3: 2 produced 09-02 + 1 legacy `MENTIONS` 08-23) | — | queued | `claude/devops/kaizen/*`, graph cleared | none → — | — |
| U4 | qa-engineer (7: 08-28…08-31) | — | queued | `claude/qa-engineer/kaizen/*`, graph cleared | none → — | — |
| U5 | tico (8: 08-26…09-02) | — | queued | `claude/tico/kaizen/*`, graph cleared | none → — | — |
| U6 | graph-dba (9: 09-02) | — | queued | `claude/graph-dba/kaizen/*`, graph cleared | none → — | — |
| U7 | architect (16: 08-26…09-03) | — | queued | `claude/architect/kaizen/*`, graph cleared | none → — | — |
| U8 | tdd-engineer chunk A (12: ≤ 08-30) | — | queued | `claude/tdd-engineer/kaizen/*`, graph cleared | none → — | — |
| U9 | tdd-engineer chunk B (8: ≥ 08-31) | — | queued | `claude/tdd-engineer/kaizen/*`, graph cleared | none → — | — |
| U10 | coder chunk A (12: ≤ 08-29) | — | queued | `claude/coder/kaizen/*`, graph cleared | none → — | — |
| U11 | coder chunk B (8: 08-31…09-02) | — | queued | `claude/coder/kaizen/*`, graph cleared | none → — | — |
| U12 | coder chunk C (7: 09-03) | — | queued | `claude/coder/kaizen/*`, graph cleared | none → — | — |
| U13 | data-scientist chunk A (10: ≤ 08-30) | — | queued | `claude/data-scientist/kaizen/*`, graph cleared | none → — | — |
| U14 | data-scientist chunk B (9: 08-31…09-02) | — | queued | `claude/data-scientist/kaizen/*`, graph cleared | none → — | — |
| U15 | data-scientist chunk C (9: 09-03…09-06) | — | queued | `claude/data-scientist/kaizen/*`, graph cleared | none → — | — |
| U16 | teco chunk A (8: ≤ 09-01) | — | queued | `claude/teco/kaizen/*`, graph cleared | none → — | — |
| U17 | teco chunk B (12: 09-02) | — | queued | `claude/teco/kaizen/*`, graph cleared | none → — | — |
| U18 | teco chunk C (11: 09-03…09-06) | — | queued | `claude/teco/kaizen/*`, graph cleared | none → — | — |
| U19 | analyst chunk A (12: ≤ 08-30) | — | queued | `claude/analyst/kaizen/*`, graph cleared | none → — | — |
| U20 | analyst chunk B (6: 08-31…09-01) | — | queued | `claude/analyst/kaizen/*`, graph cleared | none → — | — |
| U21 | analyst chunk C (15: 09-02) | — | queued | `claude/analyst/kaizen/*`, graph cleared | none → — | — |
| U22 | analyst chunk D (11: 09-03) | — | queued | `claude/analyst/kaizen/*`, graph cleared | none → — | — |

Deliverable paths above are the guaranteed minimum (every pass touches the
agent's own kaizen files and the graph); each row is rewritten on delivery with
the actual promotion targets — agent prompts, knowledge bases, project docs.

## Follow-ups

- Pass 1's open follow-up (`coder` K-005, the `verify_workflows.sh`
  false-negative) is **closed** — `tdd-engineer` fixed `Repository._read_structure`
  under `falkor-chat/docs/plans/workflow-diff-absent-key-coordination.md`;
  nothing to re-route.
