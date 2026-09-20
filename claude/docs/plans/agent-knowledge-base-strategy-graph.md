# Agent knowledge-base strategy — hybrid lexical+semantic fusion, graph mechanics note

> **Status:** active · **Owner:** `graph-dba` · **Tracks:** K-030 (`claude/cobb/kaizen/plan.md`)

U1 of `claude/docs/plans/agent-knowledge-base-strategy7-coordination.md` (item 2 of
`data-scientist`'s DEF-1 diagnosis, `claude/docs/plans/agent-knowledge-base-strategy-ml.md`):
feasibility, mechanics, score characterization, and migration handling for adding a lexical
(full-text) signal on `Chunk.text` in `ws:agent-team`, to fuse with the existing vector-ANN
`search_documents`/`search_chunks` path.

**All measurement below used a throwaway probe graph, `ws:gdba_kbftprobe`, populated with a
read-only copy of `ws:agent-team`'s real `Chunk` rows (text, and for one test, embeddings too).
`ws:agent-team` itself was never written to.** The probe graph was deleted after each experiment
(`GRAPH.DELETE ws:gdba_kbftprobe`, confirmed `OK` after the final run — nothing left behind).

**Bottom line: feasible, cheap in absolute terms, and it backfills automatically on a populated
label.** A full-text index on `Chunk.text` at `ws:agent-team`'s real current scale (558 chunks,
381,039 chars, 2026-09-20) costs **≈2.9 MiB** of RAM — smaller than the existing vector index on
the same label and label-population. The `document-ingestion2` 2-5x-of-raw-text RAM finding that
ruled out indexing `Document.text` does **not** transfer unchanged to `Chunk.text`: at this
corpus's small-chunk granularity the ratio starts *higher* (≈8x) than that finding's 2-5x, then
**converges down toward 2-5x as chunk count grows** — the opposite of what would make this a
scaling risk. In both single-chunk-granularity runs, the absolute number stayed in the low
single-digit MiB range. Full detail in §1.

---

## 1. Feasibility — RAM cost of a full-text index on `Chunk.text`

### Method

`falkordb-py` (`falkor-chat/server/.venv`, already has the dependency) against the live instance
on `localhost:6379`. For each run: read all `Chunk` rows from `ws:agent-team` via `ro_query`
(read-only, no write), bulk-`UNWIND`-`CREATE` an identical copy into `ws:gdba_kbftprobe`, sample
`redis-cli INFO memory`'s `used_memory` (median of 5 samples, 0.3s apart, after a 1-2s settle) as
the RAM baseline, run `CALL db.idx.fulltext.createNodeIndex('Chunk', 'text')`, settle again, and
re-sample. Delta = index cost. This is the same `INFO memory`-delta methodology
`document-ingestion2`/`llm-provider-config-graph`/`capacity-report.md` already use in this repo —
`GRAPH.MEMORY USAGE` is documented elsewhere in this KB as under-reporting index memory on this
build, so it was not used here either.

### Results

| Run | Chunk count | Raw text (chars) | RAM delta | Ratio to raw text | Bytes/chunk |
|---|---|---|---|---|---|
| 1x (real corpus, as-is) | 558 | 381,039 | 3,006,144 B (2.87 MiB) | **7.89x** | 5,387 |
| 4x (real corpus rows replicated 4x, distinct `chunkId`s) | 2,232 | 1,524,156 | 4,612,904 B (4.40 MiB) | **3.03x** | 2,067 |

(A first, less-controlled 1x pass — before the probe graph was rebuilt clean for the table above —
independently produced 2,955.6 KiB delta at 8.0x ratio; consistent with the table's 1x row, cited
here only to show the number reproduces, not as a second data point in the analysis.)

### Interpretation

- **The ratio is granularity-dependent, and it moves the *right* direction as the corpus grows.**
  `document-ingestion2`'s 2-5x figure was measured on `Document.text` at up to a 500,000-char
  ceiling — a few large documents. RediSearch carries fixed per-document bookkeeping (doc table
  entry, per-field metadata) that amortizes poorly over **many small documents** — `Chunk.text`
  here averages ~683 chars, two orders of magnitude smaller than that ceiling. At 558 chunks the
  ratio comes in *above* the Document-text finding's range (7.9-8.0x); quadrupling the row count
  (to 2,232, still small/short chunks) drops it to 3.0x — squarely inside the original 2-5x band.
  The trend is consistent with "fixed per-doc overhead dominates at low N, amortizes as N grows" —
  the opposite of a runaway-cost risk. I would not extrapolate confidently past this two-point
  trend without a third data point, but I have no reason to expect the ratio to reverse and climb
  back up as the corpus grows further; if anything, expect it to keep drifting toward or below the
  Document-text band.
- **In absolute terms, both runs are small.** 2.87 MiB at the real, current 558-chunk corpus; 4.40
  MiB at a hypothetical 4x-larger one. For comparison, `falkor-chat/docs/test-reports/
  capacity-report.md` §1 measures the existing `Chunk.embedding` vector index (HNSW + range-index
  overhead) at ~6,400 B/vector on top of the 4,096 B raw `vecf32` — **≈10.4 KB/chunk**, i.e. roughly
  ≈5.6 MiB total for `ws:agent-team`'s current 548 embedded chunks. A full-text index on
  `Chunk.text` adds **about half** what the vector index already costs on the same nodes — real,
  worth stating explicitly (rule 6: "call it out"), but not remotely the kind of cost that ruled
  out indexing `Document.text`, and this workspace's whole footprint is nowhere near being
  RAM-bound.
- **Answer to the feasibility question: yes, viable at `ws:agent-team`'s actual scale, and the
  scaling direction is favorable, not adverse.** No alternative/cheaper lexical mechanism is
  needed — a direct `db.idx.fulltext.createNodeIndex('Chunk', 'text')` is the right call. If this
  KB corpus someday grows to the size where `Document.text` indexing was ruled out (workspace-wide,
  chat-message volumes, not a curated technique corpus), re-measure rather than assume — but there
  is no basis to expect that scale here; this is a bounded, curated knowledge base (currently 335
  `Document`s / 558 `Chunk`s), not an open chat log.

---

## 2. Mechanics — combining the vector-ANN and full-text calls

### Both signals are independently confirmed to work over `Chunk` in one Cypher statement

Two shapes tested live, both against the probe graph (populated with a copy of the real chunk
texts *and* their real 1024-dim `embedding` vectors, plus a matching `CREATE VECTOR INDEX` +
`CALL db.idx.fulltext.createNodeIndex`, mirroring `ws:agent-team`'s actual `Chunk` schema):

**Shape A — two `CALL...YIELD` blocks chained through `WITH`, each collected, in ONE statement:**
```cypher
CALL db.idx.vector.queryNodes('Chunk', 'embedding', $k, vecf32($qVec))
YIELD node AS vNode, score AS vScore
WITH collect({chunkId: vNode.chunkId, score: vScore}) AS vecResults
CALL db.idx.fulltext.queryNodes('Chunk', $q)
YIELD node AS fNode, score AS fScore
WITH vecResults, collect({chunkId: fNode.chunkId, score: fScore}) AS ftResults
RETURN vecResults, ftResults
```
Ran clean, returned both lists as separate map-literal collections in one round trip. This
confirms FalkorDB's OpenCypher subset does support **sequencing** two independent procedure calls
in one statement (a second `CALL` after a `WITH` that closes off the first, same "insert a `WITH`
between clause boundaries" rule the quirks file already documents for `OPTIONAL MATCH`→`MATCH`)
— it is not APOC/GDS-gated.

**Shape B — `UNION ALL` of two independently-scored `CALL` blocks, tagged by source:**
```cypher
CALL db.idx.vector.queryNodes('Chunk', 'embedding', $k, vecf32($qVec)) YIELD node, score
RETURN node.chunkId AS chunkId, score AS rawScore, 'vector' AS source
UNION ALL
CALL db.idx.fulltext.queryNodes('Chunk', $q) YIELD node, score
RETURN node.chunkId AS chunkId, score AS rawScore, 'fulltext' AS source
```
Also ran clean — a single flat, tagged row set. **Caveat, load-bearing for U3**: `UNION`/`UNION
ALL` combines two complete top-level query bodies; you cannot append further clauses (a `WITH`,
a `GROUP BY`-style aggregation) *after* the union in this dialect (no `CALL {...}` subquery form —
`count{}` is confirmed unsupported elsewhere in this KB, and I found no evidence of a general
subquery-wrapping construct either). So Shape B gets you a merged, taggable row set for a
result-set-shape probe, but it does **not** get you a single-query per-`chunkId` fusion
(vector-score-and-fulltext-score-side-by-side, GROUP BY id) without going back to Shape A's
collect-then-`UNWIND`-then-regroup pattern — which is expressible (this dialect's `UNWIND` +
`WITH` idiom can synthesize a group-by), but is real query complexity for what is fundamentally a
small, two-list merge.

### Recommendation: two separate `ro_query` calls from Python, fused client-side — matching the existing convention, not a new one

Both single-statement shapes above are *possible*. I recommend **against** building the
normalization/fusion arithmetic inside Cypher, for three concrete reasons, not just "simpler":

1. **The existing code already does exactly this pattern.** `search_chunks`
   (`falkor-chat/server/falkorchat/repository.py`, the `db.idx.vector.queryNodes(...)` call U1's
   brief names) and `Services.search_documents` (`services.py`, over-fetch/filter composition per
   the ML note's Findings) already put result-shaping logic in the Python service layer, not in
   Cypher. A second `search_chunks`-shaped repository method for the full-text side
   (`CALL db.idx.fulltext.queryNodes('Chunk', $q) YIELD node, score RETURN seed.chunkId AS
   chunkId, ... ORDER BY score DESC LIMIT $limit`, mirroring the existing method's shape exactly,
   `score DESC` since higher-is-better here — see §3) is a straight, small addition next to the
   one that exists, not a new architectural pattern.
2. **The fusion math itself (whatever U2 designs — likely per-result-set normalization, or a
   rank-based method like RRF) is easier to write, test, and unit-test in Python** than to encode
   as Cypher `WITH`/`UNWIND`/aggregation chains. `falkor-chat`'s test suite already exercises this
   layer with a `FakeRepo` (`server/tests/test_services.py`, cited in the ML note) — that pattern
   extends directly to a fusion function; a Cypher-embedded fusion formula would have no
   equivalent fast, engine-free test path.
3. **Independent tuning knobs.** Two separate calls let the vector `k` and the full-text
   result-set size vary independently (e.g. over-fetch more on one signal than the other) without
   entangling that decision in the query text itself.

**One round trip vs. two is not a real cost concern here** — both `ro_query` calls are cheap
index lookups (µs-to-ms range, confirmed by the `Query internal execution time` figures on every
probe query in §3, all under 3ms), and `search_documents` already does more than one logical step
(the ANN call, then Python-side over-fetch/limit trimming) without a latency problem worth naming.
If U3 later decides a single round trip is worth it anyway, Shape A above is the concrete fallback
— confirmed working, not a hypothetical.

---

## 3. Score characterization — what `db.idx.fulltext.queryNodes` actually returns

### Method

Ran `CALL db.idx.fulltext.queryNodes('Chunk', $q) YIELD node, score` against the probe graph
(real `Chunk.text` copies, 558 rows) with several real query shapes: multi-term queries built from
words actually drawn from a real chunk's own text (a stand-in for "a situation description that
shares vocabulary with its correct answer"), quoted exact-phrase queries, single common terms, and
a guaranteed no-match term.

### Results

**Multi-term queries (RediSearch default OR-scoring across terms), 4-word queries drawn from a
real chunk's own text:**

| Query terms (from chunk X) | Top hit | Score | Chunk X's own rank/score |
|---|---|---|---|
| "killed work session wide" | chunk X | 2.72 (rank 1) | — (X was top) |
| "delegate measuring instrument tends" | chunk X | 3.75 (rank 1) | — |
| "specific properties scan anchor" | chunk X | 3.50 (rank 1) | — |
| "cleanly reproducible actually close" | a *different* chunk | 1.90 | X: 0.75, rank 2 |
| "catalog sweep five pack" | a *different* chunk | 2.46 | X: 1.21, rank 2 |

**Exact-phrase queries (RediSearch quoted-phrase syntax, `"term1 term2 term3"`):** 1 of 3 tried
phrases (chosen as a contiguous 3-word run from a real chunk) actually recurred verbatim and
matched (2 rows, top score 3.67); the other 2 returned **0 rows** — the phrase words existed in
the source text but not as that exact contiguous run, confirming phrase queries are strict
adjacency, not just term-presence.

**Common-term queries (single word, broad match):**

| Term | Chunks matched | Score range | Mean score |
|---|---|---|---|
| `"the"` | 531 / 558 (95%) | 0.10 – 1.00 | 0.82 |
| `"index"` | 66 / 558 | 0.026 – 3.00 | 0.83 |
| `"graph"` | 108 / 558 | 0.069 – 2.00 | 0.65 |
| `"falkordb"` | 33 / 558 | 0.286 – 4.00 | 1.09 |

**No-match term:** 0 rows, no error (consistent with the already-documented "clean zero-row
result" behavior, quirks file — but confirmed here specifically against an index that genuinely
exists on this label, not the "index doesn't exist at all" false-empty trap that entry warns
about).

### What this means for U2's normalization design

- **Unbounded, not `[0,1]`.** Scores ranged from ~0.026 to 4.00 across these probes — a
  RediSearch TF-IDF-family score (term-frequency × inverse-document-frequency, summed across
  matching query terms), not a normalized similarity. There is no fixed ceiling to calibrate
  against analytically — it has to be normalized empirically, per U2's own method.
- **Magnitude depends on term rarity, not just on relevance.** `"the"` (in 95% of the corpus, near-
  zero IDF) tops out at 1.00; `"falkordb"` (in 6% of the corpus, higher IDF) tops out at 4.00. A
  fixed global threshold on raw full-text score is not meaningful the way the vector floor (0.42
  cosine distance, `skills/agent-kb-retrieval/SKILL.md`) is — the same absolute score means
  different things depending on which terms a given query happens to contain. **Per-query
  normalization (e.g. min-max within that query's own returned result set, or a rank-based method
  like Reciprocal Rank Fusion that never looks at raw magnitude at all) is the safer starting
  point than trying to fix a single raw-score cutoff** — this is a design recommendation for U2 to
  weigh, not a decision I'm making for it.
- **Direction is inverted relative to the vector signal, and this is easy to get backwards.**
  Vector `score` = cosine distance, **lower is better** (0 = identical; `ORDER BY score ASC`,
  confirmed convention throughout `search_chunks`/`hybrid_search`). Full-text `score` here is
  **higher is better** (`ORDER BY score DESC` is what surfaces the true match first — confirmed in
  every table above; sorting `ASC` would rank the *worst* matches first). Any fusion formula must
  normalize both onto the same direction before combining — a detail worth stating explicitly
  because the two existing conventions in this codebase (`search_chunks` vector distance ASC) will
  make `DESC` look like a typo to anyone porting the pattern without reading this.
- **Exact-phrase matching works, and is a plausible lever for the P1/G2 failure mode.** Since
  P1/G2's whole diagnosis is "exact-phrase/lexical overlap the embedding didn't reward directly,"
  the quoted-phrase query form (`"term1 term2 term3"`) — not just the default OR-of-terms form —
  is worth U2 evaluating specifically, since it demonstrably enforces adjacency rather than mere
  term co-occurrence (confirmed above: 1 of 3 real contiguous phrases matched; loose term overlap
  alone would have matched all 3 candidate word groups regardless of adjacency).

---

## 4. Migration — creating the index against `ws:agent-team`'s already-populated `Chunk` label

**Confirmed directly, live: `db.idx.fulltext.createNodeIndex` on an already-populated label
backfills existing nodes automatically — no separate reindex step, no async wait observed at
this corpus's scale.** Checked first against `claude/graph-dba/falkordb-quirks.md` per this
task's own instruction — the file already confirms `db.idx.fulltext.createNodeIndex`/`queryNodes`
"confirmed working" and documents the *index-doesn't-exist* false-empty trap, but had **no
existing entry on populated-label backfill timing specifically** — so this is new, live-verified
ground, not a re-derivation of something already on file.

Sequence run against the probe (558 `Chunk` nodes **created first**, index added **after**):
1. `UNWIND ... CREATE (:Chunk {...})` × 558 — data written with no index present yet.
2. `CALL db.idx.fulltext.createNodeIndex('Chunk', 'text')` — succeeds.
3. `CALL db.indexes() YIELD label, types, status` immediately after (no artificial delay) →
   `status: OPERATIONAL` for the `text` field, `types: {text: [FULLTEXT], embedding: [VECTOR]}`.
4. A full-text query for a term known to exist only in the pre-existing data (`'falkordb'`)
   returned **31 matching chunks in 1.16ms** with **zero settle delay** after the create call.

This is a different lifecycle from `GRAPH.CONSTRAINT CREATE`'s documented async
`PENDING`→`OPERATIONAL` behavior (quirks file, *Indexing, constraints & DDL*) — at this corpus
size, full-text index creation-plus-backfill behaved as effectively synchronous: immediately
`OPERATIONAL`, immediately queryable, immediately correct. **Caveat, not measured**: this was
558 pre-existing nodes averaging ~683 chars each — a small backfill by any standard. I did not
test whether a much larger backfill (tens of thousands of nodes, or much longer text) would ever
surface a transient `status` other than `OPERATIONAL` on this build; if `ws:agent-team` ever grows
to that order of magnitude before this index is created, re-check `db.indexes()`'s `status`
immediately after issuing the create rather than assuming the same instantaneous behavior holds —
cheap to check, not worth blocking on speculatively now.

**One related, confirmed-consistent behavior, not new but worth stating for the implementer:**
re-running `createNodeIndex` on a property that already carries a full-text index is rejected
(`"Attribute 'text' is already indexed"`), the same non-reapplying behavior the quirks file already
documents for `CREATE VECTOR INDEX`. `bootstrap_schema.sh`'s existing pattern (idempotent,
tolerant of the "already indexed" error on repeat runs) already handles this correctly for the
three fulltext indexes it creates today (`Message.text`, `Entity.name`, `Document.title`) — adding
`Chunk.text` as a fourth line follows the exact same, already-proven idiom.

**Answer to the migration question: no special handling needed.** Add one line to
`bootstrap_schema.sh` (`CALL db.idx.fulltext.createNodeIndex('Chunk', 'text')`, alongside the
existing three fulltext lines, §357-364 of that script) and run it against `ws:agent-team` — it
will index all 558 existing chunks in place, immediately queryable, no downtime, no reindex
script, no phased rollout required at this scale.

---

## 5. Live-verified fact logged to the KB

Added to `claude/graph-dba/falkordb-quirks.md` (dated 2026-09-20): `db.idx.fulltext.createNodeIndex`
against an already-populated label backfills existing nodes and reaches `OPERATIONAL` /
becomes correctly queryable with no observed async delay at small scale (558 nodes) — contrasted
with `GRAPH.CONSTRAINT CREATE`'s documented async `PENDING` lifecycle. See that file's *Indexing,
constraints & DDL* section for the committed entry.

---

## Open items for U2/U3, not decided here

- The actual normalization/fusion formula (per-query min-max vs. rank-based RRF vs. something
  else) — U2's call, per §3's data above.
- Whether to use default OR-term full-text queries, quoted-phrase queries, or both per lookup —
  flagged in §3 as a lever specifically relevant to the P1/G2 diagnosis, not decided here.
- Whether the implementation lands as a new `search_chunks_fulltext` repository method mirroring
  the existing `search_chunks` shape (my recommendation, §2) or something else — U3's call.
