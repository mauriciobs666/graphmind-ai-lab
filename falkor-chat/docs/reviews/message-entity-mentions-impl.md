# Implementation review — message entity mentions (K-093, option B)

> **Status:** active · **Owner:** `analyst` · **Tracks:** K-093

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
