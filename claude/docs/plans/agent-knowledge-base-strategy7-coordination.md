# Agent knowledge-base strategy — item 2, hybrid lexical+semantic score fusion coordination

> **Status:** active · **Owner:** `teco` · **Tracks:** K-030 (`claude/cobb/kaizen/plan.md`)

User-authorized follow-up to `agent-knowledge-base-strategy6-coordination.md` (archived, hypothesis
falsified for item 1). Executes **item 2** of `data-scientist`'s DEF-1 diagnosis
(`claude/docs/plans/agent-knowledge-base-strategy-ml.md`, "### Recommendation — what's actually
worth trying" — read there for full context, not restated here): **hybrid lexical (BM25/full-text)
+ semantic score fusion** for `search_documents`/`search_chunks` over `ws:agent-team`, motivated
specifically by queries P1 and G2 sharing "exact-phrase overlap a lexical signal would reward
directly, independent of the embedding model's fine-discrimination ceiling" — and, per strategy6's
own closing summary, further corroborated by that trial's result (P1/G2 sharing a
lexical-overlap-without-embedding-win pattern).

## Pre-dispatch check (teco, 2026-09-20)

FalkorDB reachable (`redis-cli -p 6379 ping` → `PONG`).

**Grounding read, done before scoping** — `falkor-chat/server/falkorchat/repository.py`: `Chunk`
nodes (the standalone-KB search surface, FR-3) carry `chunkId`/`text`/`documentId`/`seq`/
`embedding`; `search_chunks` is vector-ANN-only today (`CALL db.idx.vector.queryNodes('Chunk',
'embedding', ...)`) — **no full-text index exists on `Chunk.text`**, confirmed against
`falkor-chat/scripts/bootstrap_schema.sh` (only `Message.text`, `Entity.name`, `Document.title`
carry `db.idx.fulltext.createNodeIndex` calls today).

**A directly relevant prior finding, not to be silently rediscovered or ignored:**
`falkor-chat/docs/plans/document-ingestion2.md` (§0's dispatch log, and §4.1/§6/§7) records that a
raw RediSearch full-text index on `Document.text` was **live-measured at 2-5x the raw text size in
RAM** (0.27 MB/doc low-entropy, 2.45 MB/doc higher-entropy prose) and **ruled out for that reason**
for a *different* feature (FR-9 near-duplicate candidate narrowing) — replaced there with an
app-side MinHash/LSH-banding fingerprint, which narrows near-duplicate *candidates* but is not a
mechanism for scoring arbitrary query-term relevance, so it is not a substitute for what item 2
actually needs (a real lexical relevance signal to fuse with the vector score). Whether the same
RAM concern recurs at `Chunk.text` granularity over `ws:agent-team`'s actual, much smaller corpus
(~332 claims total, DEF-1's own count) — rather than a general chat workspace's unbounded message
volume — is exactly U1's feasibility question below, not something to assume either way.

## Ledger

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict | Cost |
|---|---|---|---|---|---|---|
| U1 (feasibility + mechanics design) | `graph-dba` | `a925989978c83ba6d` | accepted | `claude/docs/plans/agent-knowledge-base-strategy-graph.md` — teco-reverified: independently re-queried `ws:agent-team` (560 chunks/382,279 chars vs. reported 558/381,039 — natural corpus drift, not a discrepancy), confirmed `Chunk` carries no fulltext index today (only RANGE+VECTOR), confirmed the probe graph `ws:gdba_kbftprobe` no longer exists (`GRAPHS` lists 9, not it), confirmed the dated 2026-09-20 `falkordb-quirks.md` entry landed | n/a — advisory design note, no formal gate (same precedent as strategy5/6 `-ml`/`-graph` addenda) | 183.3k tok / 45 tools / 572s |
| U2 (fusion method note) | `data-scientist` | `a1c485f3d87d2404d` | accepted | "## Item 2 — hybrid lexical+semantic score fusion method (2026-09-20)", `claude/docs/plans/agent-knowledge-base-strategy-ml.md` — corrected post-delivery: `teco` independently re-verified against `skills/agent-kb-retrieval/SKILL.md` and caught the admissibility-gate floor citation as 0.42 (this file's own superseded Stage 8 Phase 1 provisional figure) instead of the current operative 0.43 (Stage 8 Phase 2's landed value); fixed in §2's gate definition, plus a separate, self-caught model-attribution error in §1 (C1's vector rank had been mislabeled as "under the 0.6B model" when rank 14 is actually the unadopted 4B trial's figure — 0.6B leaves C1 absent even at rank 20). Both corrections independently re-verified by `teco` as now accurate (0.43 sourced correctly throughout §2; §1's rank attributions match the DEF-1 Finding 1 table) | n/a — advisory design note, no formal gate (same precedent as U1) | 144.0k+175.1k tok / 9+14 tools (2 rounds) |
| U3 (implementation plan) | `architect` | `a3135fdd9c0888fe5` | accepted | `claude/docs/plans/agent-knowledge-base-strategy7-impl.md` — schema DDL, `Repository.search_chunks_fulltext`, `_fuse_chunk_hits_rrf` + wiring into `Services.search_documents`, full test plan, doc impact. **Revised twice**: (round 1) `teco` caught §3.3's original resolution backwards — plan added `_strip_query_instruction_prefix`/`QUERY_INSTRUCTION_MARKER`, applied only to the lexical call. (round 2, after `analyst`'s Pass 1 gate) fixed both findings: §5.1 item 3's fixture corrected to a genuine two-marker-occurrence string (was a double-backslash typo collapsing it to one — the review's own Appendix A repro, used directly); §3.7 added (new subsection) — the fixed `HYBRID_OVERFETCH_K=20` over-fetch depth now scales to `max(HYBRID_OVERFETCH_K, limit)`, closing the large-`limit` capacity regression, with rationale for rejecting the review's other two named options (document-as-accepted-ceiling; tighten `api.py`'s `le=`) and two new tests (§5.3 items 3-4) plus a corrected item 2. Also fixed the Minor finding (§3.6/Step 3 docstrings: `score is None` disambiguated from "admitted via the lexical gate" — the latter can carry a real, floor-failing `score`). | `analyst` (`af1b8da953548ee50`) → **Pass 2: approve with suggestions** (`claude/docs/reviews/agent-knowledge-base-strategy7-impl.md`). Both Pass 1 findings (blocker, major) confirmed fixed, independently re-derived by the reviewer itself (not re-running Pass 1's own repro). One new Minor surfaced by the fix itself (non-blocking): the depth-scaling fix makes `limit > 20` reachable, but U2 never validated RRF/gate behavior at any depth other than 20 — carried forward to U4/U5's briefs below rather than sent back for a plan edit (architect's context is now large; this is small enough to just carry). | 230.0k+300.7k+249.1k tok / 24+40+10 tools (3 rounds) |
| U4 (implement) | `coder` | `ad9ff4757b4857347` | accepted | **teco-reverified**: re-ran the full suite myself (2873 passed, 14 deselected, matches exactly); confirmed the `Chunk.text` fulltext index live and functional on `ws:agent-team` directly (`db.indexes()`, and `db.idx.fulltext.queryNodes('Chunk','falkordb')` → 35, matching); ran my own independent data-plumbing mutation (swapped `vector_hits`/`lexical_hits` argument order at the `_fuse_chunk_hits_rrf` call site — a silent-corruption class coder's own control-flow-focused mutation table doesn't cover) and confirmed 4 tests catch it, incl. the dedicated end-to-end wiring test; restored the file byte-identical after, re-confirmed clean; confirmed the learning-capture doc (`672902db61f94121aa9fc72c2da7c5dc`) is real and well-formed; spot-checked `DESIGN.md`/`HISTORY.md` diffs, both accurate. **Found and fixed a side effect**: the offline pytest run (mine + coder's) wiped the shared `reference` graph's seeded workflow defs/catalog per this repo's own documented consequence of a default suite run — restored via `bootstrap_schema.sh acme` → `seed_demo.sh` → `seed_catalog.sh` → `seed_workflows.sh acme` → `seed_salesperson.sh acme` (all idempotent/additive), re-verified clean via `verify_workflows.sh`/`verify_salesperson.sh`/`verify_catalog.sh`. **Confirmed disjoint from the concurrent `embedding-migration` session's own in-flight work** (`falkor-chat/scripts/create_workspace.sh`, its edit to `start_server.sh`, its own coordination doc) — none of that is touched by this unit's commit. Schema DDL landed live on `ws:agent-team` (`Chunk.text` FULLTEXT, `OPERATIONAL`; functional smoke query returned 35 matches for `'falkordb'`) + code diff: `Repository.search_chunks_fulltext` (`falkor-chat/server/falkorchat/repository.py`); `_fuse_chunk_hits_rrf`/`_strip_query_instruction_prefix`/4 named constants + wiring into `Services.search_documents` incl. the `depth = max(HYBRID_OVERFETCH_K, limit)` fix (`falkor-chat/server/falkorchat/services.py`); docstring update (`falkor-chat/server/falkorchat/mcp.py`); no change needed to `api.py` (confirmed). Docs: `falkor-chat/docs/DESIGN.md` (fulltext register + the pre-existing `Document.title` omission), `falkor-chat/docs/QUERIES.md` (new §14.3a + §14.4 revision), `skills/agent-kb-retrieval/SKILL.md` (step 4 now server-side-redundant), `falkor-chat/docs/HISTORY.md` (dated entry). Tests: `falkor-chat/server/tests/test_services.py` — 4 direct `_strip_query_instruction_prefix` cases, 9 direct `_fuse_chunk_hits_rrf` cases, 9 `FakeRepo`-level wiring tests (3 rewrites, 6 new) — 19 new test functions, 1 renamed-away (per `git diff \| grep -c '^+def test_'`/`'^-def test_'`). Mutation-tested: wrong `RRF_K`, flipped sort direction, a never-rejecting gate, `rpartition` swap, and a `tail or query` fallback each deliberately introduced and confirmed to fail exactly the tests the plan named (and no others outside the predicted set). Full suite: 2873 passed, 14 deselected (`-m live`), 0 failed — same totals before and after (code-neutral once tests reverted to the delivered state). `ruff check .` clean on every touched file (pre-existing E501s elsewhere, untouched by this diff). No plan defect found — implemented as specified, no deviations. Committed: `e2292fa3`/`1b2a99f0`/`3cf2d350`/`af535ecb` (U1-U4, each by explicit path). | `analyst` (`ad05017f8d25ad27e`) → **approve with suggestions** (`claude/docs/reviews/agent-knowledge-base-strategy7-diff.md`) — teco-reverified both mutations directly (score collapses to `None` on lexical-only admit despite vector presence; `<=`→`<` floor-boundary exclusion), both confirmed to slip through the relevant tests undetected exactly as reported, then restored `services.py` byte-identical. No blocker; one Major (test-coverage gap on `_fuse_chunk_hits_rrf`'s `score`-semantics boundary, two named test additions suggested). **Transient flakiness observed independently** (2 `test_api.py` tests failed on one full-suite run, passed in isolation and on a clean re-run — shared-live-DB-state flakiness per the guardrail, not a regression) — `reference`'s seed data wiped again by these repeated offline runs and re-restored. Dispatching a small follow-up (fresh `coder`, U4's own context too large) to add the two suggested tests before final acceptance. **Follow-up delivered**: `coder` added exactly the two suggested tests to `falkor-chat/server/tests/test_services.py`'s §5.2 block (after `test_fuse_chunk_hits_rrf_rejected_when_both_signals_fail_their_half_of_the_gate`) — `test_fuse_chunk_hits_rrf_vector_score_at_exact_floor_admits` (pins `<=` at the exact floor boundary, no lexical presence) and `test_fuse_chunk_hits_rrf_score_reports_real_vector_value_when_admitted_via_lexical` (present in both signals, floor-failing vector score, admitted via lexical rank 1 — asserts `score` is the real value, not `None`). Mutation-tested both directly against the two named mutations (score collapsed to `None` on lexical-only admit; `<=`→`<` floor-boundary exclusion) — each new test fails exactly its mutation and no other test in the file is affected; reverted, confirmed `git diff` on `services.py` empty. `services.py` untouched (0 lines changed); only `test_services.py` touched (+44 lines, `git diff --stat`). Full `test_services.py` suite: 290 passed (288 + 2 new). `ruff check` clean on `test_services.py`. Not run: the full monorepo suite (deliberately scoped out per this unit's own brief — no live DB dependency, no need to disturb `reference` again for a two-test addition). Uncommitted — leaving commit to `teco` per this unit's own instruction. | — |
| U5 (re-test against DEF-1 criterion) | `data-scientist` | — | queued | new dated section, `agent-knowledge-base-strategy-ml.md` | — → — (teco-reverified) | — |

U1 → U2 → U3 → (analyst gate) → U4 → (analyst gate) → U5: a hard sequential chain — each unit
decides a fact the next one's deliverable must encode (U2's normalization depends on U1's measured
full-text score distribution; U3's plan depends on both being settled; U4 executes U3's plan;
U5 validates U4's actual behavior). Not parallelizable.

## Notes

- **Scope boundary**: this coordination is item 2 only. Item 3 (the cheap `cobb` X1/T1 curation
  edit) remains a separate, unrequested follow-up.
- **`ws:agent-team` is shared, live, production KB state** — U1's measurement work should prefer a
  throwaway probe graph key where feasible (mirroring `document-ingestion2`'s own
  `ws:docprobe`-style precedent) and U4's actual index/code changes land on `ws:agent-team` itself
  only once U3's plan is reviewed and accepted — never experiment directly against production
  KB state.
- **Acceptance criterion carry-forward**: strategy6 used "≥3 of 4 target queries (C1, G2, P1, X1)
  land in top-5" as its bar for item 1. U2 should treat that as a prior to reconsider against the
  hybrid mechanism's own expected effect, not adopted fact — say explicitly whether it still
  applies or a different bar is more appropriate for a structural (not just model-capacity) lever.
- **CPG freshness (retroactive)**: U3's own report revealed `architect` had consulted `cpg_falkorchat`
  for a caller-count impact check — teco should have run the freshness recipe before dispatch (per
  `AGENTS.md`'s centralized-freshness rule) and didn't. Checked after the fact: `builtAt`
  2026-09-18T21:12:42Z, `source-origin`, one commit since (`999141fe`, K-065 salesperson
  systemPrompt language salience) — touches only `proof_defs.py`/`test_salesperson_scaffold.py`,
  neither `repository.py`/`services.py`/`mcp.py`, so U3's "exactly two callers" claim is unaffected.
  No action needed this time, but check freshness *before* dispatching a unit likely to consult a
  CPG going forward, not after.
- **Depth-validation gap (`analyst` Pass 2, non-blocking Minor), carried forward for U4/U5's own
  briefs rather than a plan edit**: U3's `depth = max(HYBRID_OVERFETCH_K, limit)` fix correctly
  restores capacity for a large `limit`, but U2's RRF `k=60`/lexical-rank-≤2-gate reasoning was only
  ever validated at depth=20 — a future caller requesting `limit > 20` exercises RRF/the gate in an
  unvalidated regime. Inert today (the only real caller, `SKILL.md`'s convention, is pinned at
  `limit=5`). U4's brief should carry this as a stated, accepted limitation (not something to fix);
  U5's brief should note its regression run stays scoped to the `limit=5` convention by design, so
  it will not catch a problem that only shows up at a larger `limit` — that's a known gap, not an
  oversight to silently close.
