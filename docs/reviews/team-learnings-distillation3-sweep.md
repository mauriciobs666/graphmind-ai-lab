# Team learnings distillation 3 — sweep report

> **Status:** active · **Owner:** `cobb` · **Tracks:** — · **Extends:** `docs/reviews/kaizen-inbox-distillation2.md`

## Scope and "since the last sweep"

- **Last sweep marker.** The predecessor store `kaizen_team` was drained and deleted by the third team pass
  (`claude/docs/plans/kaizen-distillation3-coordination.md`); the K-030 migration of KB claims into `ws:agent-team`
  is closed in `claude/cobb/scripts/kb-claim-manifest.json` (`_status`, 2026-09-19). No sweep of `ws:agent-team`'s
  raw capture appears in any `kaizen/history.md`, so every raw document there is pending.
- **How raw entries were enumerated.** `mcp__cypher__query(graph='ws:agent-team')` read-only counts (observed):
  470 `Document` nodes; 138 whose `text` starts with `Fact:` (raw entries, created 2026-09-18 to 2026-11-06 by day:
  2, 6, 5, 1, 18, 3, 7, 2, 1, 9, 2, 20, 11, 14, 34, 3). The rest are KB claims. `list_documents` could not be used: it is
  oldest-first, capped at 200, and its 200 returned rows were 198 KB-manifest ids plus 2 raw entries. (Fix recorded in
  `skills/agent-maintenance/SKILL.md` §5 step 1.)
- **Cut.** Oldest 40 of 138 processed (through createdAt 1792625651910); 98 remain (named backlog below).
- **Clearing.** The user approved deleting the stale KB claim `ddfbb4bb` and the 40 dispositioned raw documents (except the kept-open `d88545b5`, so 39). Observed 2026-11-06: 4 raw documents deleted (`4da25923`, `ca7ba5b0`, `5f57a8bb`, `672902db`); the harness classifier then blocked the delete of `ddfbb4bb` and of `d1f94b35` (it allowed neighbouring calls), so I stopped. Cypher read afterwards: 134 raw (`Fact:`) documents remain of 138; both blocked ids still exist; the 7 new claims are all `ready`. Still to delete: 35 dispositioned raw documents (all report ids except the four above and `d88545b5`) plus `ddfbb4bb`.
- **Attribution.** 5 entries show `ingestedByKind: User, ingestedById: u1` (not `produced_by`-attributed); producer
  inferred from content and logged in `cobb`'s history. Marked `u1` below.

## Dispositions (oldest 40 of 138)

Verified = re-derived by me on 2026-11-06 (command or code read), not just the entry's citation. "Doc'd" = already in the cited home.

| # | id (8) | Producer | Fact | Disposition |
|---|---|---|---|---|
| 1 | d1f94b35 | teco | Resume a rate-limit-killed subagent via SendMessage | Rejected: duplicate of `claude/teco/coordination-techniques.md` "resume-versus-fresh" |
| 2 | 4da25923 | cobb | `produced_by` fails loudly on unresolvable agentId | Rejected: doc'd (`agent-maintenance` §5, `falkor-chat/docs/SERVER.md`) |
| 3 | ca7ba5b0 | analyst | Patching a module attr misses `from m import name` binding | Rejected: general Python; the suggested scratch-copy fix is superseded by rows 18 and 20 |
| 4 | 5f57a8bb | u1 | CPG caller query proves caller-exhaustiveness | **Promoted** `skills/cpg-analysis/SKILL.md` (with the bound the entry omitted: by-name, as of build, not dynamic dispatch) |
| 5 | 672902db | coder | Same-second `.pyc` reuse in mutation loops | Rejected: doc'd (`skills/python-web-quirks/SKILL.md` "same-second edit") |
| 6 | 12eb7b6c | u1 | Full suite nondeterministic under concurrent sessions | **Promoted** `falkor-chat/docs/SERVER.md` testing hazards; failure counts not carried (not re-derivable) |
| 7 | 3f4ac750 | data-scientist | Long-running `:8200` server serves stale code | **Promoted** `falkor-chat/docs/SERVER.md` (checked: `start_agent_team.sh` runs `--reload` off) |
| 8 | e34f3143 | u1 | `cmd \| tail; echo $?` reports tail's code | Rejected: generic shell knowledge, no repo specifics |
| 9 | 35649df3 | analyst | ORDER BY null placement, tie order | **Promoted** `claude/graph-dba/falkordb-quirks.md`; verified live (`ws:test`) |
| 10 | 9d10d45b | teco | Compound Bash call denied, split passed | Rejected: not reproducible; classifier behavior covered by teco's "standing commit grant" section |
| 11 | 38cd49cd | architect | BackgroundTasks run one request's tasks sequentially | **Promoted** `skills/python-web-quirks/SKILL.md`; verified (`inspect.getsource`) |
| 12 | 68e28ff8 | analyst | `:8200` log holds only the current launch | **Promoted** `falkor-chat/docs/SERVER.md` §1.10; log still begins at the 2026-10-15 launch header |
| 13 | d0b47b2c | architect | `InvalidURL` is not a `ValueError` | **Promoted** `skills/python-web-quirks/SKILL.md`; verified live (MRO, `urlopen`) |
| 14 | 69d50ac7 | qa-engineer | `lms ps` rc=1 during JIT load; lms polling in log stream | **Merged** into the LM Studio claim, single-source |
| 15 | 8e13f385 | data-scientist | cypher MCP truncates long cells | Rejected: doc'd (`cypher-mcp/README.md` `CYPHER_MCP_MAX_CELL`); I also saw `(+743 chars)` today |
| 16 | 44a4e246 | qa-engineer | `rank_report` drops n=0 model | Rejected: fixed (`model-bench/docs/HISTORY.md` U186; banner in `report.py`) |
| 17 | d9cb242e | graph-dba | server venv has falkordb-py | Rejected: convenience visible from `falkor-chat/AGENTS.md` |
| 18 | 60aba8db | analyst | PYTHONPATH shadow is a no-op | Rejected: doc'd (`analyst/review-techniques.md`, `devops/ops-quirks.md` editable-finder sections) |
| 19 | b7154a94 | u1 | cypher MCP writes are kaizen-shaped only | Rejected: stale (kaizen graph retired) and stated in the tool's own description |
| 20 | 6ce4a232 | u1 | Shadowing needs non-package cwd | Rejected: doc'd, same sections as row 18 |
| 21 | 9d139039 | data-scientist | `search_documents` lexical syntax errors | Rejected: moot, not fixed. The hybrid lexical fusion that fed raw text to RediSearch was reverted (`7a270329`) and `Chunk` has no full-text index; the same hazard still lives in `Message`/`Document`/`Entity` full-text calls (row 26 claim). A mixed-punctuation query returned results today |
| 22 | 681f7f73 | teco | falkor-chat drops INFO under uvicorn | Rejected: doc'd (`SERVER.md` §1.10) |
| 23 | 48ef66f5 | teco | Fence data, not the workspace | **Promoted** `claude/teco/coordination-techniques.md` fencing note (rewritten; it still named `kaizen_team`); claim re-ingested |
| 24 | 6eca9311 | qa-engineer | LM Studio fault shapes | **Merged** into the LM Studio claim |
| 25 | 605a44a9 | qa-engineer | `lms log stream` request signatures | **Merged** into the LM Studio claim |
| 26 | 7f84105d | analyst | Full-text index tokenizes on ASCII only | **Promoted** `falkordb-quirks.md` (one full-text claim with rows 27-29); empty query, `%_%`, `%a-b%`, `%__%` re-run live |
| 27 | 665c8538 | tdd-engineer | Fuzzy-term syntax errors | **Merged** into the full-text claim (corroborated by `fusion.py` docstring) |
| 28 | 0ed79bfd | analyst | Empty full-text query returns 0 rows | **Merged** into the full-text claim; verified live |
| 29 | 5caea049 | tdd-engineer | Empty query / `%_%` / `%__%` | **Merged** into the full-text claim |
| 30 | a2cf142e | devops | `--reload` reloader parent survives | Rejected: double import reproduced, but `ss` listed both PIDs and one TERM ended everything |
| 31 | e7f3f119 | analyst | FastMCP calls `basicConfig(INFO)` | Rejected: doc'd (`SERVER.md` §1.10) |
| 32 | a767dde2 | devops | Same as row 31 | Rejected: duplicate of row 31 |
| 33 | d88545b5 | qa-engineer | Mermaid parse check via Windows npx | **Kept open**: unverified, live run blocked by the classifier; `claude/cobb/kaizen/plan.md` K-062 |
| 34 | c6f11ae1 | tico | Update rule omits the team manual | **Promoted** `claude/AGENTS.md` add/edit/rename rule (rewritten in place) |
| 35 | 92e938a3 | tico | KB drifts from flat files | Verified for 2 cases (stale `ddfbb4bb`, missing 13.1 s claim); fixed by KB sync. Claim "manifest omits 4 files" is wrong (underscore-prefixed keys). Stale delete owed: K-061 |
| 36 | df558abf | qa-engineer | Storefront calls two LLMs | Rejected: superseded; `run["stepModel"]` now makes `query_graph_data` inherit ministral |
| 37 | 2fa57985 | qa-engineer | JIT keeps one model resident | **Merged** into the LM Studio claim (the demo hazard was doc'd; explicit-load detail added) |
| 38 | 410ce035 | architect | Heredoc prose trips destructive-ops `ask` | **Promoted** `teco/coordination-techniques.md`; patterns confirmed in `.claude/settings.json` and the guard script |
| 39 | 4c052e37 | architect | nlq denominator hides structural failures | Rejected: doc'd in the `extraction.py` `score_item` docstring |
| 40 | 860c397b | analyst | `PostToolUse` skips failed calls | **Promoted** `skills/agent-standards/claude-code.md` (docs checked; Bash-non-zero case flagged unverified) |

Note: rows 14, 24, 25, 37 feed the LM Studio claim; rows 26-29 the full-text claim.

## Promotion summary

- **Skills:** `python-web-quirks` (2 sections), `cpg-analysis` (1 paragraph), `agent-standards/claude-code.md` (1 bullet), `agent-maintenance` §5 step 1 (enumeration fix).
- **KB files (+ `ws:agent-team` sync):** `graph-dba/falkordb-quirks.md` (2 claims), `teco/coordination-techniques.md` (fence replaced, readiness replaced, 1 new), `data-scientist/lm-studio-model-notes.md` (1), `devops/ops-quirks.md` (13.1 s claim, missing from the graph, ingested). 7 documents ingested; 6 of the 7 first ended `failed` (embedder evicted, HTTP 400); each was then deleted and re-ingested one at a time and all 7 were observed `ready` on 2026-11-06 (ids in the manifest). Stale `ddfbb4bb` is still served; its delete awaits approval (K-061).
- **Project docs:** `falkor-chat/docs/SERVER.md` (3 bullets), `claude/AGENTS.md` (clause in an existing line; no new line).
- **Counts (of 40):** 12 promoted, 7 merged into those promotions, 1 verified-and-fixed via KB sync (row 35), 1 kept open (row 33), 19 rejected (documented, duplicate, fixed, superseded, unreproduced).

## Remainder backlog (named: team-learnings-distillation3-remainder)

98 raw entries (138 − 40) from createdAt > 1792625651910, i.e. the tail of 2026-10-21 onward through 2026-11-06 (observed per-day counts above). Enumerate with the cypher query in `skills/agent-maintenance/SKILL.md` §5 step 1 and `ORDER BY d.createdAt SKIP 40`. Tracked as `claude/cobb/kaizen/plan.md` K-063.

## Findings for the coordinator

1. 40 raw documents are dispositioned but not deleted (approval needed). K-061: delete stale KB claim `ddfbb4bb…` (denied) and confirm the 7 new claims reach `ready`.
2. `ingestedById: u1` on 5 entries: some client writes without `produced_by`; unidentified.
3. `AGENTS.md` word counts were not reduced; `claude/AGENTS.md` grew by one clause (it was already over the ~2,500-word smell before this sweep).

## Audit result

`bash claude/scripts/audit-team.sh` → `RESULT: FAIL` on two repo-wide checks only: home-path leak (146 files, 743 lines before this report was genericized) and hostname leak (2 files, 2 lines). Not caused by this sweep: `git diff` adds 0 lines containing the user name, and a `git grep -l` for the home path in `HEAD` already lists 139 tracked files. All per-agent, roster, handoff and KB-prefix checks PASS; advisory prompt-size NOTEs for analyst, teco, tico only.

## Files touched

Skills: `skills/python-web-quirks/SKILL.md`, `skills/cpg-analysis/SKILL.md`, `skills/agent-standards/claude-code.md`, `skills/agent-maintenance/SKILL.md`. KB: `claude/graph-dba/falkordb-quirks.md`, `claude/teco/coordination-techniques.md`, `claude/data-scientist/lm-studio-model-notes.md`, `claude/cobb/scripts/kb-claim-manifest.json`. Docs: `falkor-chat/docs/SERVER.md`, `claude/AGENTS.md`. Kaizen: `history.md` of teco, cobb, analyst, coder, data-scientist, tdd-engineer, architect, qa-engineer, graph-dba, devops, tico; `claude/cobb/kaizen/plan.md` (K-061..K-063). No README catalog changed (no agent or skill added, renamed or removed).
