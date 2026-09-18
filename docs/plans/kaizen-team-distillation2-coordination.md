# Kaizen-team distillation — pass 2, one producing agent per turn

> **Status:** active · **Owner:** `teco` · **Tracks:** — (stakeholder-triggered `cobb` sweep of the shared `kaizen_team` graph) · **Extends:** `docs/plans/kaizen-team-distillation-coordination.md`

## Context

Stakeholder asked (2026-09-18) to distill the shared `kaizen_team` graph again, dispatching `cobb`
per producing agent. Procedure, conventions and lessons are those of the 2026-09-16 sweep this
document extends — read that document's Context and Close-out rather than this one restating
them. Carried forward unchanged: **fresh `cobb` instance per unit**, **strictly sequential** (units
share `claude/cobb/kaizen/history.md` and cross-agent KBs), **`analyst` diff-scoped gate on every
unit before commit**, largest inbox first, `teco` commits each accepted unit by explicit path.

Two conventions corrected against that precedent, both documentation-grammar:

- Gate reviews land in **one** family document, `docs/reviews/kaizen-team-distillation2.md`, one
  `## U<n>` section per unit (root `AGENTS.md` collision rule 2 — the per-unit `-u<n>` basenames of
  the earlier sweep invented a slug per unit).
- `claude/docs/plans/kaizen-distillation2-coordination.md` (the 2026-09 pass that stayed `active`
  under a "keep going indefinitely, re-query before close" instruction) is flipped to `archived`
  at this open: that instruction is honoured by this lineage — each sweep re-queries fresh at open
  and drains what is there — not by an ever-open ledger. Its own close condition (graph at only the
  one deliberate orphan) was in fact observed 2026-09-10 and again by the 09-16 sweep's U7 sweep.

**Snapshot at open (2026-09-18)**, all current-shape, no legacy `author` entries:

| Agent | Produced | Notes |
|---|---|---|
| `analyst` | 14 | 8 are meta-lessons captured while *gating* the 09-16 sweep; 6 are code facts (model-bench, falkor-chat) |
| `architect` | 2 | one is the falkor-chat `CallContext` actor/ws pin — same seam as two `analyst` entries and the `data-scientist` one |
| `qa-engineer` | 2 | |
| `cobb` · `coder` · `data-scientist` · `tdd-engineer` · `teco` · `tico` | 1 each | |
| `MENTIONS`-only orphans | 3 + 1 | `a3f1c8e2…`→`devops`, `c1f3a9e2…`/`a1c2e3f4…`→`graph-dba`; **`e1a6c4d2…`→`tico` is the deliberate survivor (routing signal for `tico` K-016) and is never cleared** |

Total 28. The graph is live — re-query immediately before each dispatch, pin every brief to explicit
`entryId`s, never "everything this agent has". The `analyst` inbox is split by **theme** rather than
date so each chunk has one discard bar and a disjoint target-file set.

**Documentation-impact scan**: per unit, the producing agent's `kaizen/{history,plan}.md`, its
prompt/KB, and — only when a prompt's routing contract or a KB's existence changes — the
`claude/README.md` catalog row / `claude/AGENTS.md` roster; project-docs promotions land in the
component's own tree (`model-bench/`, `falkor-chat/docs/`), never root `docs/`. No `HISTORY.md`/
`BACKLOG.md`/manual in scope (`claude/` has none; the earlier sweep set that precedent).

## Ledger

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict | Cost |
|---|---|---|---|---|---|---|
| U1 | `cobb` — `analyst` chunk A, 8 meta-lessons from gating the 09-16 sweep: `b4f6c1a2` `a1e2c3d4` `f4b8c2d1` `7f3c9a2e` `f3d9a1c2` `b7e4a1f2` `e2f1a8c3` `f0e8b2a4` | `a958e5e05212be5a1` | accepted | 8/8 dispositioned: 2 sharpenings `claude/analyst/review-techniques.md` (18,881→19,039), 4 sharpenings `skills/agent-maintenance/SKILL.md` §5 (6,366→6,603; 5 entries, 2 merged), 1 discarded (already published twice), 0 kept open; `claude/analyst/kaizen/{history,plan}.md`, `claude/cobb/kaizen/history.md`. Post-clear: `analyst` 6 (exactly U2's ids), orphans 4 — teco re-derived counts and dup-heading scan | `analyst` (`af5a0203abc6441c1`, 111.3k tok · 28 tools) → **needs changes** (`docs/reviews/kaizen-team-distillation2.md` §U1): 1 Major (§5 step-1 timestamp instrument mis-states the clear→gate→commit window; `date` is day-granular), 2 Minor, 2 Info — all 5 fixed by the same `cobb` (Major applied with its own evidence-backed correction to the reviewer's fix: `createdAt` is itself placeholder-midnight for 5 of 6 survivors); teco re-verified each fix against the tree and corrected one figure in `cobb`'s history (4→5 of 6). Post-fix 19,056 / 6,705 words. **Accepted without a Pass-2 re-gate** (disjoint wording fixes; reviewer pre-stated none needed). The gate's own capture `bb058e98…` (`analyst`, 09-18) is a future sweep's | 160.3k+183.1k tok · 39+10 tools (cobb) · 111.3k · 28 (gate) |
| U2 | `cobb` — `analyst` chunk B, 6 code facts: `b3f1b8b4` `a1f3c9e2-6b7d` `a1e6c9d4` `c7f2a815` (model-bench) · `c7e2a814` `a1f3c9e2-7b4d` (falkor-chat `CallContext`) | — | queued | — | `analyst` → — | — |
| U3 | `cobb` — `architect` (2): `a1e6d9f4` `a1f3d9c2` | — | queued | — | `analyst` → — | — |
| U4 | `cobb` — `qa-engineer` (2): `c2e40890` `5a2b5130` | — | queued | — | `analyst` → — | — |
| U5 | `cobb` — `data-scientist` (1): `7a3e9c1b` | — | queued | — | `analyst` → — | — |
| U6 | `cobb` — `tdd-engineer` (1): `a1f3c2e4` | — | queued | — | `analyst` → — | — |
| U7 | `cobb` — `coder` (1): `c2b1f6b4` | — | queued | — | `analyst` → — | — |
| U8 | `cobb` — `teco` (1): `b2f6a1d4` | — | queued | — | `analyst` → — | — |
| U9 | `cobb` — `tico` (1): `f3d8a1c2` | — | queued | — | `analyst` → — | — |
| U10 | `cobb` — `cobb` self-produced (1): `9f2b6e1a` | — | queued | — | `analyst` → — | — |
| U11 | `cobb` — the 3 `MENTIONS`-only orphans (`devops` 1, `graph-dba` 2), one unit as the 09-10 pass's orphan-backlog precedent | — | queued | — | `analyst` → — | — |
