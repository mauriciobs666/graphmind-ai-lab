# Implementation review — message entity mentions (K-093, option B)

> **Status:** archived · **Owner:** `analyst` · **Tracks:** K-093

## Pass 1 — build unit (diff-scoped)

### Scope & verdict

**Reviewed:** the uncommitted working-tree diff, measured against baseline `5da38a36`, in `server/falkorchat/repository.py`, `services.py`, `server/tests/test_graphrag.py`, `test_services.py`, `test_tools.py`, `test_responder.py` and `scripts/test_queries.sh`. I checked it against plan v2 §3 (`falkor-chat/docs/plans/message-entity-mentions.md`) and its gate (`falkor-chat/docs/reviews/message-entity-mentions.md`, Pass 1–2, N2 included). I applied root and `falkor-chat/` `REVIEW.md`. The live-spec edits (plan §4) are out of scope; they are the close unit's work.

**Verdict: approve.** No blocker, major or minor findings. One nit concerns the plan's mutant wording, not the code.

CPG: considered, not relevant — `cpg_falkorchat` is stale (per brief), so I used grep, a capture of the text actually sent, and mutants run on a scratch copy.

All mutants ran on a scratch copy of `server/` (package, tests, `pyproject.toml`; `scripts/`, `config/`, `docs/` and `web/` symlinked), using the real venv with the copy first on `sys.path`. The real tree was not touched.

### Checks executed

1. **Query text — byte-identical in both variants. Verified.** I captured the text through a recording fake graph on the real method and compared it with §3.1's two blocks, lines joined by single spaces. Scoped: identical. Unscoped: identical. Parameters are unchanged.
2. **Rebased F1 guard still catches F1. Verified.** Dropping `"WITH seed, score, t "` turns `test_hybrid_search_finds_the_seed_thread_by_index_not_by_walk[c1]` red. Its four plan assertions are unchanged. Green in both variants on the tree, so the EXPLAIN shows `Node By Index Scan | (t:Thread)` for each.
3. **Deletions match the plan exactly. Verified by a name-set diff against `5da38a36`.**
   - `test_graphrag.py` lost the 5 listed tests and the 5 listed helpers. It gained only pins 1 and 2.
   - `test_services.py`, `test_tools.py` and `test_responder.py` have identical `def` sets. Their changes are fixture keys and the pin-3 assertion.
   - N2's three stale comments are gone: the "Entity layer dormant (M2)" header, the `search_chunks` contrast, and the "dormant `relatedContext:[]`" comment.
   - Item arithmetic: the deleted tests were 6 items (one was parametrized ×2) and the new pins add 4. So 3797 → 3795, which matches.
   - `ruff check --select F` is clean on all four touched Python files, so nothing is left orphaned.
4. **Leftover references: none needing work.** I ran an unfiltered grep of `falkor-chat/` (excluding `docs/`) for `relatedContext`, `MENTIONS[^_]`, `expansion`, `co-occurrence` and `co-mention`. Each hit has an owner:
   - pins 2 and 3 (`test_graphrag.py:199`, `test_services.py:2360`);
   - `MAX_MENTIONS` and "role expansion", which are unrelated;
   - `repository.py:2183`, the `ABOUT` docstring the plan keeps by name.

   Repo-wide, `relatedContext` outside docs and kaizen appears only in `cpg/spikes/frontend-fitness/{adjudication,sample}/*.jsonl`. These are frozen code excerpts in eval data, so they need no change.
5. **`test_queries.sh` §6. Verified.**
   - The three copies (`:421`, `:428`, `:433`) no longer carry `CALL {}`.
   - The column was removed from exactly the two copies that carried it (`:421`, `:433`).
   - The expansion assertion, the `count(e)` probe and the always-true "Entity graph is empty" assertion are deleted. So are the three false-premise comments.
   - I normalized literals to parameters and compared each copy with the sent text. They match through `ORDER BY score ASC, msgId ASC LIMIT $limit`. The RETURN projections differ from the method, but they already did at `5da38a36`, so this is not a regression.
6. **Off-brief edits: none.** `git diff --stat` is empty for DESIGN, QUERIES, SERVER, HISTORY and BACKLOG. The only other modified file is the coordination ledger, which is outside this gate.
7. **Suites, run on their own, one after the other.**
   - `./scripts/test_queries.sh`: **493/493**, exit 0. That is the baseline minus the two deleted assertions. It wipes only `ws:test` and `reference-test`.
   - `cd server && .venv/bin/python -m pytest -q`: **3795 passed, 14 deselected, 1 xfailed**. The `wf_repo` tests ran, against `reference-test` (conftest isolation); the live `reference` was not touched and needs no re-seed.

**Pins and mutants (scratch copy).**

| Run | Red |
|---|---|
| Baseline `repository.py` + `services.py` under the tree's tests | pin 1 ×2, pin 2 ×2, pin 3 — **5**, as reported |
| `CALL {}` restored without the column | pin 2 ×2 only |
| Column restored (`relatedContext: []` in the row dict) without `CALL {}` | pin 1 ×2 only |
| Chunk-row `[]` stamp restored | pin 3 |
| `WITH` barrier dropped | F1 guard `[c1]` |
| `msgId` tie-break dropped | `…breaks_exact_score_ties_by_msg_id_asc` |
| `msgId`/`score` swapped in the row dict | 5 hybrid tests |

### Findings

**N1 (nit) — plan §3.2's mutant line merges two independent removals.**
- §3.2 says restoring `CALL { … }` breaks pins 1 and 2. That is true only for the full restore, the subquery plus its column, which is the baseline (5 red).
- The subquery alone reddens only pin 2, and the column alone reddens only pin 1 (table above). The implementer's reading is correct.
- Coverage is complete: each removed element has its own pin, and neither pin is redundant.
- The plan has been executed against, so no edit is owed. The lesson goes into the proposed line below.

### What's solid

- The repository, services and docstring edits are minimal and match §3.3 line for line. The thread-by-index and barrier text is kept.
- The pins are built on kept helpers only (`_seed_thread`, `_sent_hybrid_query`). Pin 2 inspects the captured text that was actually sent, not the source.
- The fixture-key removals in the three FakeRepo-style modules leave no assertion that relied on the key.

### Proposed REVIEW.md lines

Target: root `REVIEW.md`, *Always check*. Source: N1 of this review.

- (addition) When a removal drops several independent elements (a clause and the column it feeds, a field and its stamp), mutate each element alone and name the check each one reddens. A mutant that restores them together proves only that some check catches the pair.

### Open questions

- None.

## Pass 2 — live-spec close check

### Scope & verdict

**Reviewed:** the uncommitted diff to `falkor-chat/docs/DESIGN.md` and `falkor-chat/docs/QUERIES.md`, checked against the code shipped at `3f7e5c2e`. `git diff --quiet 3f7e5c2e -- server scripts` passes, so the tree's code is the shipped code. I used plan §4's close list and §2's row text.

The hunks touch exactly the listed sections: DESIGN §1.2, §5.1, §5.2 and §8, and QUERIES §4, §6, §10 and §14.3. SERVER, README, AGENTS, HISTORY and BACKLOG are untouched.

**Verdict: approve with suggestions.** There is one minor finding. It does not block the close.

CPG: considered, not relevant — `cpg_falkorchat` is stale; I used captured query text, grep and live EXPLAIN.

### Checks executed

1. **QUERIES §6 equals the sent text. Re-derived independently.** I extracted the §6 block, dropped its `//` lines and joined the rest with single spaces. For the unscoped text, I removed the `WITH seed, score, t` and `MATCH (c:Channel …)` lines, as the block's own header comment says. Both texts equal what `Repository.hybrid_search` sends, captured through a recording fake graph.
2. **Claims in the touched sections are true of the code.** I checked each one:
   - DESIGN §5.2 and §8: the row keys `msgId`, `text`, `role` and `score` match `repository.py`.
   - QUERIES §4: "a `Message` has no edge to an `Entity`". Grep of `server/falkorchat` and `scripts` finds `[:ABOUT]` from `Chunk` as the only edge into `Entity` (`repository.py:2190`, `seed_nlq_eval_corpus.py:299`). That edge is documented at §14.5.
   - The §1.2 row's premise (every mention is a fresh `Entity`) matches DESIGN `:363` and `:915`, and `ingestion.py:14`.
   - The QUERIES §10 and §14.3 edits are pure deletions of a contrast.
3. **Nothing still describes the expansion.** I grepped `DESIGN`, `QUERIES`, `SERVER`, `README` and `AGENTS` without a filter, for:
   - `relatedContext`, `MENTIONS[^_]`, `expansion`, `co-occurrence`, `co-mention`, `related message(s)/context` and `dormant`;
   - the cost-model synonyms `fan-in`, `hub entit`, `LIMIT 5`, `per seed`/`per-seed`, `85 ms` and `2-hop`.

   Every hit has an owner: the new §1.2 row, the new §8 and §6 sentences, QUERIES `:2066` (the `ABOUT` sentence the plan keeps), and unrelated uses (`QUERY_MEM_CAPACITY`, per-seed score in §10).
4. **No change history.** There is no `K-093`, "was removed", "no longer" or dated text in the hunks. The row's "never from the dropped read" is register rationale, like the DayBucket row's "dropped when…".
5. **"Detailed in: §8" is valid. Pass-2 N3 resolved.** §8 now says the read has no entity hop and points back to §1.2.
6. **The reversal trigger is falsifiable.** It fires when a requirements document or backlog item names a consumer of per-seed related-message context. That event is observable and greppable, and the row names the consumers to look for.
7. **Growth-stops check: not applicable.** DESIGN and QUERIES are live specs, not `requirements/`, `plans/` or `reviews/` documents.

**The pre-existing barrier claim (QUERIES §6) holds.** I ran EXPLAIN on `ws:acme`, read-only through `mcp__cypher__query`:

| Text | Plan (bottom-up) |
|---|---|
| No `WITH` | `ProcedureCall` → **`Node By Index Scan \| (c:Channel)`** → `Conditional Traverse \| (c)->(t:Thread)` → `Filter` |
| With `WITH` (shipped) | `ProcedureCall` → `Node By Index Scan \| (t:Thread)` → `Project` → `Conditional Traverse \| (t)->(c:Channel)` → `Filter` |
| `threadId` moved into the channel pattern | the same as no `WITH` |

The bullet's "anchors on `Node By Index Scan | (c:Channel)`" is exact, and so is "plans like no barrier". The bullet's with-barrier arrow chain leaves out the `Project` operator, which is a harmless abbreviation.

### Findings

**m1 (minor) — the "each row is the seed alone" fact now has two homes.**
- DESIGN §8 (`:883-884`) and the new QUERIES §6 bullet (`:519-520`) both state it, and both point to §1.2. A fact stated in two spec sections is a finding under root `REVIEW.md`. DESIGN §8 itself says it describes shape "so the two never drift".
- The QUERIES bullet is also imprecise. "No hop leaves the seed besides its thread scope" calls the thread lookup a hop, while the bullet just above says the thread is looked up by index and never walked.
- Both copies are true of the code today.

**Fix:** delete the QUERIES §6 "Each row is the seed alone" bullet. The Cypher block already shows the four-column projection, and DESIGN §8 keeps the clause that §1.2's "Detailed in" needs. Verified by reading.

### Proposed REVIEW.md lines

None. m1 is already covered by root `REVIEW.md`'s two-sections rule.
