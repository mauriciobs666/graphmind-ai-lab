# Team-learnings follow-ups: KB re-ingest review

> **Status:** active · **Owner:** `analyst` · **Tracks:** K-063, K-102 (—)

## Scope & verdict

Reviewed cobb's uncommitted follow-up pass (2026-11-07): the diffs of `claude/data-scientist/lm-studio-model-notes.md`, `claude/graph-dba/falkordb-quirks.md`, `claude/cobb/scripts/kb-claim-manifest.json`, `claude/{cobb,data-scientist,graph-dba}/kaizen/*`, and the untracked `docs/reviews/team-learnings-followups-sweep.md`. Root `REVIEW.md` applied. All five checks were executed against the live system.

**Verdict: approve with suggestions.** No blocker or major. Two minor findings (stale/inaccurate counts in docs) and one minor observation on the `GRAPH.MEMORY USAGE` bullet.

CPG: considered, not relevant — the change is prose, a JSON manifest and KB documents; the one code question (is there a re-embed route) was a grep of the live `falkor-chat/server` tree.

## Checks run

1. **fd27e4a9 fix: confirmed.** Windows-side dir `/mnt/c/Users/<user>/.lmstudio/server-logs` (`~/.lmstudio` does not exist in WSL): `find -type f` = 138; `grep -rl 'Num Offload Layers'` = exactly 2 (`2026-10/2026-10-30.1.log`, `2026-11/2026-11-07.1.log`). Read the hit lines: all are `"content": "{ \"entities\": [...` extraction-output bodies (10-30 line 3038 follows the "Received request to embed: Fact: LM Studio's per-load GPU placement..." at line 2989, i.e. the raw capture of this fact; 11-07 lines 1034 and 4092 likewise). None is a placement line. The new `.md` wording ("138 files when counted", "only two", "extraction output from ingesting this very fact", "count grows") says only what was observed. The untouched "127 times / 58 times" now reads 128 / 58 in `main.log` (and 159 / 159 in `main.old.log`); it is dated to the check, so not a finding.
2. **GRAPH.MEMORY USAGE: confirmed.** `docker exec falkordb-dev redis-cli GRAPH.MEMORY USAGE <g>`, module `graph` ver 41811: `total_graph_sz_mb` `ws:acme` 1 (874 nodes by `count(n)`), `ws:test` 0 (5 nodes), `ws:eval` 0 (142), `ws:k069probe` 0, `ws:agent-team` 12; every field printed is an integer. Placement at line 1239 is the right bullet (same topic, appended). The clause claims nothing unobserved.
3. **New claims: confirmed.** `d9efef15…` ready, 2 chunks, 0 unembedded; `424a70a2…` ready, 1 chunk, 0 unembedded (cypher on `ws:agent-team`). Full compare (JSON-decoded `get_document` text vs the `.md` section/bullet, `python3 -I`, `==`): both **byte-equal** (1581 and 527 characters). Old claims `fd27e4a9…` and `9bdc41de…` still exist and are `ready`; they are named as superseded in the manifest (`_cobb_followups_2026_11_07._note` plus per-entry `supersedes`). Manifest parses; `git diff` has 0 removed lines, additive only; `git diff --check` clean.
4. **Item A: confirmed.** `MATCH (d:Document {status:'failed'})` = 17. The six named: `4b9c6680`, `15de3552`, `2cc6debd` each 1 chunk, 1 unembedded; `a792f59a`, `7bf10aeb`, `8e05d0f8` fully embedded (1, 1, 2 chunks, 0 unembedded). Re-embed path: `set_chunk_embedding` is called only from `embedding.py:270` (`embed_chunk`), which is called only from `background._safe_embed_chunk` (scheduled at ingest); no `mcp.py` tool or `api.py` route reaches either; `scripts/migrate_embeddings.sh` is a whole-workspace model migration requiring the server stopped. Cobb's "no additive per-document re-embed path" holds. K-102 (`falkor-chat/docs/BACKLOG.md:84`) does record exactly the gap (retry/repair decision) as proposed/unowned. The recommendation ((1) leave, (2) via K-102, root-cause the three embedded-but-failed) is reasonable.
5. **Leaks / history.** `git diff -U0 | grep -E '/home/|mauri|/Users/'` over all added lines: no hits; the untracked report has none either (`/mnt/c/Users/<user>/` is the existing placeholder). Kaizen history entries match the diffs and the report (ids, files, dates, "distillation 5/6 follow-ups 7 and 3", row 15).

## Findings

### F1 — minor — `K-063 (6)` says "six failed raw documents cannot be re-embedded"; only three lack embeddings
`claude/cobb/kaizen/plan.md` K-063 item (6). Verified (check 4): three of the six are fully embedded, so "re-embed" is not their remedy, which the sweep report itself says. As written, item (6) would send a future reader to a repair path for documents that do not need one. Suggest: "three failed raw documents (`4b9c6680`, `15de3552`, `2cc6debd`) have an unembedded chunk and cannot be re-embedded without a repair path (K-102); three more are embedded but failed for an undiagnosed reason."

### F2 — minor — workspace-wide counts in the sweep report and K-102 are stale or do not reconcile
The report says "16 document rows have an unembedded chunk (15 `failed`, 1 `processing`)"; the same query now returns 14 `failed` + 1 `processing` = 15 (verified, cypher `MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk) WHERE c.embedding IS NULL`). K-102's title still says 11 (accurate for its 2026-10-31 date, so not wrong, but the item is not an "open item rewritten" to current state per root `AGENTS.md`). Suggest: restate the report's number as observed at its time of run with the query, and when K-102 is next touched replace its 11 with the current 14 failed + 1 processing (17 `failed` in total, 3 of the six distillation-4 documents among them).

### F3 — minor/observation — the integer-MB clause suggests the bullet's headline may be a rounding artifact
`claude/graph-dba/falkordb-quirks.md:1239-1245`. The bullet's lead claims `indices_sz_mb: 0` "with a live HNSW index holding real vectors" proves under-reporting. Check 2 shows `ws:acme` has VECTOR indexes (`db.indexes()` on `Chunk.embedding` and `Message.embedding`) and reads `indices_sz_mb 0`, while `ws:agent-team` (larger) reads `indices_sz_mb 3`. A 0 on a small graph is therefore explained by the integer rounding the clause now documents, and the "under-reports" title is not shown by the evidence in the file. Not a defect of this pass (the lead predates it and the new clause is accurate), but the two sentences now sit in tension. Suggest graph-dba compare `indices_sz_mb` of a graph with a known large vector index against an `INFO memory` delta, then either narrow the title/lead to "rounds to integer MB" or keep it with that evidence. Inferred, not tested.

## What's solid
- Both re-ingested claims are byte-identical to their sources, `ready`, fully embedded, with a clean additive manifest note naming the superseded ids for the user's delete.
- The new wording in both `.md` files is observation-scoped and dated; no unobserved claim added.
- Item A is correctly stopped at a decision point rather than worked around (a hand-built embedding write would bypass the dimension guard and status bookkeeping).

## Open questions
- Whether the user wants the K-102 repair item opened, or the three embedded-but-failed documents root-caused first (cobb's recommendation: both, in that order).
