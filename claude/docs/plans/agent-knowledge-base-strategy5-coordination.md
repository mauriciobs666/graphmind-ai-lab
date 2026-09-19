# Agent knowledge-base strategy — K-030 follow-up batch coordination

> **Status:** active · **Owner:** `teco` · **Tracks:** K-030 (`claude/cobb/kaizen/plan.md`)

Three follow-up items surfaced by K-030 Track 2's now-closed coordination
(`claude/docs/plans/agent-knowledge-base-strategy4-coordination.md`, archived), selected by the
user from a menu of flagged-but-undispatched backlog items:

1. **Track 1's own last open item** (unrelated to Track 2): team-wide cutover of the raw-capture
   *write* convention beyond the `cobb`/`teco` pilot — the other 11 agents' "Learning capture"
   sections still write the old `kaizen_team` shape via `mcp__cypher__query`, not
   `ingest_document` against `ws:agent-team`. Confirmed live: `architect.md`'s section still reads
   the pre-cutover form; `teco.md`'s/`cobb.md`'s already match the new one. Folded into the same
   unit: `skills/agent-maintenance/SKILL.md` §5's note that "only `cobb`/`teco` currently write to
   `ws:agent-team`" goes stale the moment this cutover lands and needs updating in the same edit,
   and Stage 9's own review (`-stage9.md`, Pass 3) flagged one small, already-diagnosed orphan-
   document gap in the same file's recovery sequence — same owner, same file, folded in rather than
   run as a separate serialized unit.
2. **Stage 8's DEF-1 finding**: prose/narrative queries retrieve materially worse than code/config
   queries (21% miss rate vs. 0%, Wilson CIs barely overlapping at n=19/n=11). Flagged in
   `claude/docs/test-reports/agent-knowledge-base-strategy-ac2-report.md`'s Feedback section as the
   most actionable follow-up, not fixed there (out of that gate's authority).
3. **The R6/h40 score-instability root-cause check**: `data-scientist`'s U8 consult
   (`claude/docs/plans/agent-knowledge-base-strategy-ml.md`, "Stage 8 Phase 2 addendum") ranked
   "a one-time discrete LM Studio reload between sessions" as the most likely cause and named the
   confirming/falsifying check as a `devops`-level backend-log investigation, not executed there.

## Ledger

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict | Cost |
|---|---|---|---|---|---|---|
| U1 (team-wide write-convention cutover, 11 agents + SKILL.md §5 note + Stage 9's orphan-doc fix) | `cobb` | `a036af9239f8dd423` | in-flight | — | `analyst` → — | — |
| U2 (DEF-1 — diagnose the prose-retrieval quality gap) | `data-scientist` | `a1dbec04fe6cac54d` | in-flight | — | — → — | — |
| U3 (R6/h40 backend-log check — confirm/falsify the discrete-reload hypothesis) | `devops` | `ac42d9af3c8a7a7ed` | in-flight | — | n/a — diagnostic, teco-reverified | — |

U1/U2/U3 are independent (disjoint files: agent prompt files + `skills/agent-maintenance/SKILL.md`
vs. `claude/docs/plans/agent-knowledge-base-strategy-ml.md` vs. no expected file writes) —
dispatched in parallel.

## Notes

- **U1's exact roster** (the 11 agents not yet cut over — every agent except `cobb`/`teco`, which
  Track 1 Stage 4 already piloted): `analyst`, `architect`, `coder`, `data-scientist`, `devops`,
  `frontend-engineer`, `graph-dba`, `qa-engineer`, `security-expert`, `tdd-engineer`, `tico`. Of
  these, 5 have an explicit `tools:` allowlist needing `mcp__falkor-chat-agent-team__ingest_document`
  added (`analyst`, `architect`, `data-scientist` already have `search_documents`/`get_document`
  from Stage 7's U4b fix; `security-expert` and `tico` have neither yet) — the other 6 inherit all
  tools and need no `tools:` line change (`coder`, `devops`, `frontend-engineer`, `graph-dba`,
  `qa-engineer`, `tdd-engineer`).
- **U1's orphan-doc fix, exact scope**: the third interruption point Stage 9's review Pass 3 named
  (`claude/docs/reviews/agent-knowledge-base-strategy4-stage9.md`, "One new, non-blocking
  observation") — between `ingest_document` succeeding and the manifest's `documentId`-overwrite
  write landing, a `None`-triggered re-run re-`ingest`s and orphans the untracked first document.
  Also captured as a `kaizen_team` entry (`analyst`, entryId `b3e6f6b0-0a3f-4e7a-9d2a-5f7c1a9f6e21`)
  — teco-reverified real. This continues that same review file (`reviews/` documents revise in
  place across passes per `AGENTS.md` collision rule 5) as a dated `## Pass 4` once fixed and
  re-gated, not a new review document.
- **U3's data points for devops**: qa-engineer's original AC-2 report (commit `f8558f2`,
  2026-09-19 14:26:54 -03:00) reported R6/h40's second sibling at 0.4201/rank 3; `analyst`'s first
  review pass (commit `1fa49a6`, 2026-09-19 14:41:41 -03:00, only 15 minutes later) reproduced a
  stable-but-different 0.4405/rank 5 twice. `ws:agent-team` is served by the dedicated
  `falkor-chat-agent-team` process (`falkor-chat/scripts/start_agent_team.sh`, port 8200) talking
  to an LM Studio embedding backend at a `base_url` resolved once at process startup per that
  script's own 3-tier fallback (env var → `opencode.local.json` → shared `opencode.json`).
