# Security expert (new agent) — Coordination

> **Status:** archived · **Owner:** `teco` · **Tracks:** — (K-016, cobb backlog)

## Goal

Create the new `security-expert` team member proposed and confirmed via `tico`'s requirements
interview at `claude/docs/requirements/security-expert.md` (Status: Ready for design, confirmed
2026-08-17). Per `claude/AGENTS.md`'s maintenance rule ("a stakeholder proposal for a new team
member is a `tico` interview, not a straight-to-`cobb` request... only once that doc reaches
Ready for design does `cobb` design the actual agent"), that gate has already passed — this
coordination drives the `cobb` design/build pass and its independent review.

This also closes `cobb`'s own backlog item K-016 (`claude/cobb/kaizen/plan.md`), which flagged
the requirements doc as sitting un-actioned since 2026-08-17.

## Ledger

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict |
|---|---|---|---|---|---|
| U1 | `cobb` | `ae5ebc565c80f17b0` | accepted | `claude/security-expert/**`, `claude/README.md`, `claude/AGENTS.md`, `claude/scripts/audit-team.sh`, `claude/{analyst,cobb,devops,teco}` boundary edits, `claude/cobb/kaizen/{plan,history}.md` (K-016 closed) | `analyst` → approve (Pass 2) |
| U2 | `analyst` | `af9b61c1d988188f5` | accepted | `claude/docs/reviews/security-expert.md` | Pass 1: approve with suggestions → Pass 2: **approve** |
| U3 | `cobb` | `ae5ebc565c80f17b0` | accepted | fixes to `guard-exploitation-approval.sh`, FR-10 prompt section, `cpg-analysis/SKILL.md`, `security-expert/kaizen/{plan,history}.md` (new K-004) | `analyst` → approve (Pass 2) |
| U4 | `analyst` | `af9b61c1d988188f5` | accepted | focused re-check of U3's fixes (Pass 2 appended to `claude/docs/reviews/security-expert.md`) | approve |

## Notes

- U1 delivered 2026-08-20. `audit-team.sh`: 110 PASS, 2 pre-existing/unrelated FAIL (a
  personal-info leak in `falkor-chat/docs/test-reports/graphrag-eval-report.md`, committed
  2026-08-16, confirmed untouched by this session — separate follow-up, out of this scope).
- Judgment calls `cobb` flagged for `analyst` to specifically check: (1) standalone
  `guard-exploitation-approval.sh` rather than extending the shared `guard-destructive-ops.sh`
  core; (2) no new `-security` doc-role suffix added to the closed role set — slug-collision
  avoidance handled via prose guidance instead; (3) reciprocal boundary-clause edits limited to
  `analyst`/`cobb`/`devops` (not `qa-engineer`); (4) **`cobb` edited its own `cobb.md` description
  as part of this change** (the reciprocal `security-expert` clause) with no independent review
  of that self-edit — flag this explicitly to `analyst`.

- U1 done-condition: the full FR-1..FR-11 set from the requirements doc reflected in the new
  agent's prompt/tools/hooks, plus the standard "adding an agent" doc-curation checklist from
  `claude/AGENTS.md` (agent source, `kaizen/{plan,history}.md` — no `inbox.md`, FR-12/AC-9 —
  `claude/README.md` catalog entry, name rosters in `claude/AGENTS.md`).
- U2 is the standard cobb-artifact review gate (guardrails: "cobb's agent/skill artifacts →
  analyst"). If `analyst` returns "needs changes", resume U1's `cobb` run by `agentId` rather than
  re-dispatching cold.
- No CPG-freshness check applies to either unit — neither is a code review of a CPG-bearing
  component.

## Close-out

Accepted 2026-08-21. `analyst`'s Pass 2 (same review doc) confirms all three Pass-1 majors
(the `nc`/`ncat`/`netcat` reverse-shell bypass, the `WebFetch`-uncovered FR-10 prompt gap, and
the benign `curl`/`wget` grep false-positives) and the one minor (`cpg-analysis/SKILL.md`
consumer list) are genuinely fixed, verified against `analyst`'s own original reproduction
commands — not on `cobb`'s word. Final verdict: **approve**. Two items were explicitly left
open, not blockers: the `Agent`-tool delegation bypass of hook-gated guards (pre-existing,
team-wide, not specific to this deliverable — a candidate future `cobb`-led cross-cutting look)
and the FR-11 doc-path naming-collision safeguard between `analyst` and `security-expert` reviews
(no live collision today; `security-expert/kaizen/plan.md` already carries the follow-up).
`claude/scripts/audit-team.sh` stayed at 110 PASS / 2 pre-existing-and-unrelated FAIL (a
personal-info leak in `falkor-chat/docs/test-reports/graphrag-eval-report.md`, committed
2026-08-16, confirmed untouched throughout this coordination) across every check in this
coordination — flagging that pre-existing failure as a follow-up for whoever owns that file, out
of this scope.
