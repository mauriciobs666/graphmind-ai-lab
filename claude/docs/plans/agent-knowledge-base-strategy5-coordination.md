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
| U1a (write-convention cutover review — diff-scoped) | `analyst` | `ac021e93cccda49a0` | gated | `claude/docs/reviews/agent-knowledge-base-strategy5-u1.md`, commit `ad1f1b7` — teco-reverified both Major findings directly: `list_documents` confirmed hardcoded oldest-first (`repository.py:1353-1386`, `mcp.py:404-412`, no sort param); the "Found one" adopt branch's fall-through to the byte-mismatch `delete_document` path confirmed by reading `SKILL.md`'s actual text (lines ~803-855) — a title collision would delete a different claim's live document; manifest re-checked directly (90 titles, 0 duplicates today) | approve with suggestions (2 Major, non-blocking) — U5 dispatched to fix | 144.2k tok / 33 tools / 270s |
| U5 (fix SKILL.md orphan-sweep: corpus-count guard + collision-safe adopt branch) | `cobb` | `ae77ce414732a6168` | in-flight | — | `analyst` → — (Pass 4, same review file) | — |
| U1 (team-wide write-convention cutover, 11 agents + SKILL.md §5 note + Stage 9's orphan-doc fix) | `cobb` | `a036af9239f8dd423` | accepted | 15 files (11 agent prompts, `claude/AGENTS.md`, `skills/agent-maintenance/SKILL.md`, `claude/cobb/kaizen/history.md`+`plan.md`), commit `209017c` — teco-reverified: 8/11 agent files read in full (framing clauses/special notes correctly preserved), rest diff-stat-consistent; 5 `tools:` additions confirmed exact; `DocumentNotFoundError`/`list_documents` API citations confirmed against `document-ingestion2.md`; `audit-team.sh` clean. **K-052 finding (mid-session `tools:` edit invisible to a nested subagent-of-subagent) independently tested**: a fresh top-level probe from this session succeeded cleanly (tool present, real `ingest_document` call worked) — refines K-052 rather than confirming it as originally stated (caching boundary looks nested-dispatch-specific, not same-session-general); recorded as a `kaizen_team` entry; orphaned test document (no `delete_document` in the probe's grant) cleaned up separately | `analyst` → — (dispatched, U1a) | 282.4k tok / 77 tools / 827s |
| U2 (DEF-1 — diagnose the prose-retrieval quality gap) | `data-scientist` | `a1dbec04fe6cac54d` | accepted | `claude/docs/plans/agent-knowledge-base-strategy-ml.md` "DEF-1 diagnosis" section (Version 4→5), commit `1133329` — teco-reverified: kaizen entry real; document word counts for 3/4 targets checked (minor imprecision in the report's own approximate figures for 2 of them, not worth chasing — teco's own quick check undercounted first, corrected); ~150-200-claim neighborhood tally cross-checked against known per-file counts (~189, in range). Root cause: corpus-neighborhood density (a 150-200-claim stylistically homogeneous region), not query wording or top-K/floor tuning (ruled out by a live `limit=20` re-run). Recommends a bounded `qwen3-embedding:4b` held-out trial first, hybrid lexical+semantic fusion routed to `graph-dba`/`architect` as a future design question, and a cheap `cobb` content edit for the one discovered X1/T1 near-duplicate pair | — → — (advisory, not gated — see U1's precedent for advisory work teco verifies directly) | 174.3k tok / 28 tools / 350s |
| U4 (correct ml.md's Stage 8 Phase 2 addendum per U3's falsification) | `data-scientist` | `a1dbec04fe6cac54d` | accepted | `claude/docs/plans/agent-knowledge-base-strategy-ml.md` (Version 5→6), commit `213db59` — teco-reverified: falsified candidate demoted to "Eliminated" (kept visible, not deleted); concurrent-batching promoted leading-by-elimination with its own original objection preserved, not overclaimed; section-by-section confirmation that sections 2-4's recommendations don't depend on the mechanism, only section 4's "currently-live" framing withdrawn | n/a — advisory correction, teco-verified directly | 216.3k tok / 14 tools / 139s |
| U3 (R6/h40 backend-log check — confirm/falsify the discrete-reload hypothesis) | `devops` | `ac42d9af3c8a7a7ed` | accepted | Diagnostic finding, no file changes (read-only): the discrete-LM-Studio-reload hypothesis is **falsified** — `falkor-chat-agent-team` process running continuously since 2026-09-18 23:12:50 (no restart), LM Studio's own server log shows exactly 5 model-unload events all day, none between 14:00-15:00 -03:00 (nearest: 09:32:10 and 15:44:47), and no LM Studio app-process restart in the window either. Concurrent-request-batching floating-point non-associativity (data-scientist's #2-ranked candidate) is now the leading explanation, not investigated further (no log surface for it; a design question, not a process/log check) — teco independently re-confirmed both the process-start timestamp and the exact unload-event list via direct `ps`/log reads | n/a — diagnostic, teco-reverified directly | 126.2k tok / 28 tools / 200s |

U1/U2/U3 are independent (disjoint files: agent prompt files + `skills/agent-maintenance/SKILL.md`
vs. `claude/docs/plans/agent-knowledge-base-strategy-ml.md` vs. no expected file writes) —
dispatched in parallel.

- **U4 (dispatched — same delegate as U2, resumed):** U3's falsification means `claude/docs/plans/
  agent-knowledge-base-strategy-ml.md`'s "Stage 8 Phase 2 addendum" section currently states a
  now-disproven diagnosis ("most likely — a discrete state change... happening once between
  qa-engineer's run and analyst's first review pass") as its ranked #1 conclusion — that needs a
  correction once U2 finishes (both touch `ml.md`; running them concurrently risks a file
  collision, so U4 is deliberately held rather than dispatched alongside U2). `SKILL.md`'s own
  "Score floor" section doesn't mention the reload hypothesis at all (confirmed via grep — no
  correction needed there), so this is `ml.md`-only.

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
