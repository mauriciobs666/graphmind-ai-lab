# Agent knowledge-base strategy — item 2, hybrid lexical+semantic fusion implementation plan

> **Status:** active · **Owner:** `architect` · **Tracks:** K-030 (`claude/cobb/kaizen/plan.md`)

U3 of `claude/docs/plans/agent-knowledge-base-strategy7-coordination.md`. Turns U1's feasibility/
mechanics note (`claude/docs/plans/agent-knowledge-base-strategy-graph.md`, `graph-dba`) and U2's
fusion-method decision (`claude/docs/plans/agent-knowledge-base-strategy-ml.md`, "## Item 2 —
hybrid lexical+semantic score fusion method (2026-09-20)", `data-scientist`) into a concrete,
step-by-step build for `coder` (U4). Both inputs are settled — this document does not re-derive
the RRF formula, the `k=60`/equal-weight choice, the OR-term lexical query shape, or the
admissibility gate values (0.43 vector floor OR lexical rank ≤2); it decides how they land in
`falkor-chat`'s actual code.

**No high-stakes fork found.** Neither U1's mechanics nor U2's formula/gate contains a defect —
both are internally consistent with each other and with the live-verified facts they cite (checked
against the actual current source below, not just the notes' prose). The one genuinely destructive-
risk question (does landing the DDL against `ws:agent-team` risk data loss) is closed by U1 §4's own
live-verified finding (backfill-on-create, no reindex step, no async pending window observed at
this scale) — nothing here reopens it.

**Two corrections already applied, in two rounds, before this reached its current state.**
(1) §3.3's first draft misread U2's "lexical query must run on raw, unwrapped text" note as meaning
`search_documents`'s `query` argument already arrives prefix-free — `teco` traced
`skills/agent-kb-retrieval/SKILL.md`'s calling convention independently and found the opposite (a
compliant caller always sends the *wrapped* string; nothing in `falkor-chat` strips it). §3.3/§4
Step 3-4/§5.1 now implement and test an explicit strip step. (2) `analyst`'s Pass 1 gate
(`claude/docs/reviews/agent-knowledge-base-strategy7-impl.md`) found that fix's own §5.1 item-3
test was inert as specified (a double-backslash typo collapsed the fixture to one marker
occurrence, so `partition`/`rpartition` were indistinguishable on it — fixed, §5.1 item 3, using
the review's own verified repro) and that the fixed `HYBRID_OVERFETCH_K=20` over-fetch depth
silently capped `search_documents` well below what `limit` promises for a caller requesting more
than ~20-40 — fixed by scaling the depth (§3.7, §4 Step 4). §6 records both.

**CPG: used `cpg_falkorchat`** — confirmed via call-graph query that `Repository.search_chunks` has
exactly two production callers (`Services.search_documents`, `Services.hybrid_search`) and
`Services.search_documents` has exactly two production callers (`mcp.py:search_documents`,
`api.py:build_router.search_documents`), both of unstructured-`dict` return type (no
`response_model=` pydantic lock at either transport). This is what justifies §3's scope boundary
(fuse only inside `search_documents`, leave `hybrid_search` untouched) and §3's contract call (no
transport-level breaking change) — both are backed by this query, not assumed.

---

## 1. Goal & scope

**Goal.** Give `Services.search_documents` (FR-3 standalone-KB search, the surface
`skills/agent-kb-retrieval/SKILL.md` describes) a second, lexical signal — a full-text index on
`Chunk.text` plus a new repository read over it — fused with the existing vector-ANN signal via
Reciprocal Rank Fusion and gated by U2's admissibility rule, so a query with strong lexical overlap
but a weak embedding match (the P1/G2 failure pattern) has a real chance of surfacing its correct
document.

**In scope:** the `Chunk.text` full-text index (schema + migration), a new repository method for
the lexical read, the fusion+gate function and its wiring into `Services.search_documents`, the
test coverage for all of the above, and the documentation updates that follow directly from the
change (QUERIES.md, DESIGN.md, and one skill-file consequence surfaced below, §3.6).

**Out of scope:**
- `Services.hybrid_search`/`Repository.hybrid_search` (the `Message`+`Chunk` chat-grounding merge,
  FR-2) — the coordination doc scopes item 2 to `search_documents`/`search_chunks` (the FR-3
  standalone-KB surface) specifically; §3.1 below states why leaving `hybrid_search` untouched is
  also the only choice the CPG-confirmed call graph supports cleanly.
- Re-deriving or tuning the fusion formula, the lexical query shape, or the gate thresholds — U2's
  job, done.
- U5's live retrieval-quality re-test — already scoped by U2 §4; this plan only covers the
  implementation-level test coverage for the new code paths (§5 below).
- A phrase-query fallback mode — U2 names it as an evidence-triggered follow-up, not to be built
  now (§2 below states where the hook would go if it's ever needed).

---

## 2. Context & findings

**`falkor-chat/server/falkorchat/repository.py:1232` (`Repository.search_chunks`)** — vector-ANN-
only today:
```cypher
CALL db.idx.vector.queryNodes('Chunk', 'embedding', $k, vecf32($qVec))
YIELD node AS seed, score
WHERE seed.documentCurrent = true
RETURN seed.chunkId AS chunkId, seed.text AS text, seed.documentId AS documentId,
       seed.seq AS seq, score
ORDER BY score ASC
LIMIT $limit
```
`score` = cosine distance, lower is better. The post-`YIELD` `WHERE seed.documentCurrent = true`
excludes superseded-document chunks from the already-ANN-bounded candidate set — which is exactly
why callers over-fetch `k` beyond `limit` (document-ingestion2 Stage C).

**`falkor-chat/server/falkorchat/repository.py:747` (`Repository.search_messages`)** — the existing
full-text shape this plan's new method mirrors, `Chunk` in place of `Message`:
```cypher
CALL db.idx.fulltext.queryNodes('Message', $query)
YIELD node AS m, score
RETURN m.msgId AS msgId, m.threadId AS threadId, m.text AS text, m.createdAt AS createdAt, score
ORDER BY score DESC LIMIT $limit
```
`score` here is RediSearch TF-IDF-family, higher is better — confirmed by U1 §3 to hold identically
for `Chunk.text` once indexed. `Services.search_messages` (`services.py:1060`) wraps the repository
call in `try/except ResponseError: raise InvalidSearchQueryError` — the pattern this plan's new
lexical call in `search_documents` reuses (§3.4).

**`falkor-chat/server/falkorchat/services.py:1360` (`Services.search_documents`)** — today: embeds
`query` via the injected `ModelGateway`, then calls `repository.search_chunks` once with
`k = limit * SEARCH_DOCUMENTS_OVERFETCH` (module constant, `= 3`, `services.py:175`),
`limit = limit`, `timeout = RAG_QUERY_TIMEOUT_MS` (`= 5000`, `services.py:156`). Raises
`SearchNotAvailableError` (503) if no `ModelGateway` is wired — that early-raise is unaffected by
this change (the lexical signal still needs the vector signal to exist at all under this design;
see §3.5).

**`falkor-chat/server/falkorchat/mcp.py:378` / `falkor-chat/server/falkorchat/api.py:243`** — the
two production callers of `services.search_documents`, confirmed exhaustive by the CPG query above.
Neither constrains the return shape with a pydantic `response_model` — `mcp.py` returns
`list[dict[str, Any]]` straight through; `api.py`'s route has no `response_model=` either. This is
why §3.6's return-shape change is additive, not breaking, at the transport layer.

**`falkor-chat/server/tests/test_services.py`** — the `FakeRepo` class (`:70`) is the intended test
seam (U2's own pointer). Relevant existing pieces:
- `FakeRepo.search_chunks` (`:336`): `self.calls.append(("search_chunks", ws, tuple(q_vec), k,
  limit, timeout)); return self.since_rows if self.chunk_rows is None else self.chunk_rows`.
- `FakeRepo.search_messages` (`:219`): the exact idiom for scripting a `ResponseError` — `if
  isinstance(self.since_rows, Exception): raise self.since_rows`.
- Three existing `search_documents` tests reference `search_chunks`'s exact call-argument tuple and
  assert `rows == repo.since_rows` (a bare passthrough) — all three break under this change and are
  named for rewrite in §5.
- `_RankedChunkRepo` (`:1380`) — a `FakeRepo` subclass simulating the real `k`-vs-`limit`-vs-
  `documentCurrent` interaction; its one fixture (`:1409`) has rows with **no `score` key at all**,
  which the new fusion function requires — a fixture update, not a design problem (§5).

**`falkor-chat/scripts/bootstrap_schema.sh:356-373`** — the three existing fulltext lines
(`Message.text`, `Entity.name`, `Document.title`), each an `echo` + idempotent `gquery` pair inside
`bootstrap_workspace()` (workspace-scoped, not `bootstrap_reference()` — `Chunk` is workspace-local
too, so the fourth line lands in the same function).

**`skills/agent-kb-retrieval/SKILL.md`** — the query-side calling convention for `ws:agent-team`:
fixed `limit=5`, a client-side "reject any hit with `score > 0.43`" step (step 4 of the calling
convention), and the query-instruction prefix template that must **never** reach the lexical side
raw (§3.3 below).

**`claude/docs/plans/agent-knowledge-base-strategy-graph.md` (U1)** — feasibility (≈2.9 MiB,
favorable scaling), the two working Cypher shapes for combining vector+fulltext in one statement
(not adopted — §3.1), the score-characterization data (unbounded/TF-IDF/DESC, opposite direction
from vector/ASC), and the live-verified backfill-on-create migration behavior (§4 below cites it
directly, not re-measures it).

**`claude/docs/plans/agent-knowledge-base-strategy-ml.md`, "## Item 2 …" (U2)** — RRF formula
(`k=60`, `w=1`), `K=20` over-fetch per signal, OR-term (not phrase) lexical query shape run against
raw unwrapped text, the admissibility gate (vector ≤0.43 OR lexical rank ≤2), and the
"rank-first-then-gate-with-backfill" ordering — all adopted as specified; see U2's own §1/§2 for the
rationale this plan does not restate.

---

## 3. Design & rationale

### 3.1 Scope boundary: `search_documents` only, not `hybrid_search`

`Repository.search_chunks` has two production callers. `Services.hybrid_search` (FR-2, chat-
grounding retrieval) is the other one, and it is **not** touched by this change — confirmed
deliberate by the coordination doc's own framing ("hybrid lexical + semantic score fusion for
`search_documents`/`search_chunks`... over `ws:agent-team`") and reinforced by the CPG-confirmed
call graph: `search_chunks` itself stays byte-for-byte unchanged (no new parameter, no behavior
change), so `hybrid_search`'s existing call is unaffected by construction, not by a conditional
branch that could silently regress it. Extending fusion to chat-grounding retrieval is a distinct,
unauthorized future decision.

### 3.2 A new repository method, not a Cypher-embedded fusion (per U1 §2)

`Repository.search_chunks_fulltext` mirrors `search_chunks`'s shape exactly, swapping
`db.idx.vector.queryNodes` for `db.idx.fulltext.queryNodes('Chunk', ...)` — U1 confirmed both single-
statement shapes (chained `CALL`s via `WITH`, or `UNION ALL`) work but recommended against Cypher-
side fusion for three reasons (existing-idiom fit, Python testability, independent tuning knobs) —
adopted without modification. Two separate `ro_query` round trips per `search_documents` call is an
accepted, named cost (U1: both are sub-3ms index lookups; not a latency concern at this scale).

### 3.3 The lexical query must strip the query-instruction prefix — resolved, not an open question

**Resolved by `teco`'s independent check before this plan reached `analyst`** (this section
originally mis-resolved it the other way; corrected here). `skills/agent-kb-retrieval/SKILL.md`
step 1/2 is unambiguous, and quoted exactly (its own fenced literal, not paraphrased):

```
f"Instruct: Given a coding agent's description of its current situation, retrieve the distilled technique or rule that applies to it.\nQuery: {situation}"
```

with step 1 stating the caller "substitute[s] your own situation description for `{situation}`"
and step 2, "call `search_documents(query=<the prefixed string>, limit=5)`" — and, load-bearing:
"nothing in `falkor-chat` applies a prefix for you, and **nothing strips one you forget to add**."
A compliant caller therefore always sends the **whole wrapped string** — boilerplate sentence and
all — as `search_documents`'s `query` argument. `Services.search_documents` itself confirms this
is what happens in-process today: it passes `query` straight to `embedder.embed(query)` with no
wrap/strip step anywhere in `falkor-chat/server/falkorchat/*.py` (grepped, confirmed absent).

**This means the raw-text requirement (U2's correctness note) is not satisfied by construction —
it requires an actual strip step, added by this plan, not just a documentation note.** The vector
signal needs the prefix verbatim (Qwen3-Embedding's own asymmetric query/document instruction
convention — stripping it there would be a regression, not a fix); the lexical signal must never
see it. `Services.search_documents` is the one place both forms of `query` are available before
either repository call fires, so the strip lives there (§4 Step 3/4).

**Strip design: split on the first occurrence of the template's own structural marker, not a
full-prefix byte match.** Two candidate techniques, chosen against the template's actual shape:
- **Full fixed-prefix match** (`query.removeprefix(<whole sentence>)`) — exact, but brittle to any
  future rewording of the instruction sentence in `SKILL.md` (a wording tweak that doesn't change
  the template's *shape* would silently stop matching, and the failure is silent — a no-op strip,
  not an error).
- **Marker split** (adopted) — the template's structural boundary is `"\nQuery: "`, immediately
  before `{situation}` begins; this is Qwen3-Embedding's own `Instruct:.../nQuery:...` API shape, a
  more stable anchor than the instruction sentence's wording. Splitting on the **first** occurrence
  (not the last) is the correct choice, not an arbitrary one: the template's own marker is always
  the earliest occurrence in the constructed string, so using `str.partition` (first) rather than
  `str.rpartition` (last) is what keeps a situation description that happens to itself contain the
  literal substring `"Query: "` intact in the returned tail, rather than truncating into the middle
  of it. If the marker is absent entirely (a caller that omitted the prefix, or any future caller
  outside this convention), the string passes through unchanged — nothing here repairs a missing
  prefix, mirroring the skill's own "nothing strips one you forget to add" posture: there's simply
  nothing to strip.

**Named, accepted residual risk:** the marker constant (`QUERY_INSTRUCTION_MARKER`, §4 Step 3) is
now a second, hand-maintained copy of a fragment of `SKILL.md`'s canonical template — that file's
own drift check (`claude/scripts/audit-team.sh`, greps for the *whole* template string) covers only
`claude/`-side prompts, not this `falkor-chat`-side constant. If `SKILL.md`'s template ever drops
the `Instruct:.../nQuery:...` shape entirely (not just rewords the sentence), this constant needs a
matching update with nothing automated to catch a miss — flagged in Risks (§6), not solved here.

### 3.4 Error handling: mirror `search_messages`'s posture exactly

A RediSearch syntax rejection on the lexical call surfaces as `redis.exceptions.ResponseError`.
`search_chunks_fulltext` does not catch it (same as `search_chunks` never catches ANN errors);
`Services.search_documents` wraps it: `except ResponseError as exc: raise
InvalidSearchQueryError(str(exc)) from exc` — the exact idiom `Services.search_messages` already
uses (`services.py:1068-1071`). `InvalidSearchQueryError` is an existing `ServiceError` subclass
that already maps to REST 400 with no new wiring in `app.py`.

### 3.5 `search_documents` still requires a wired `ModelGateway`

The early `SearchNotAvailableError` raise (`self._models is None`) stays first and unconditional.
The admissibility gate's primary path is the vector floor; the lexical OR-term signal is a
secondary, additive contribution (U2 §1's "a chunk absent from a signal's list contributes 0" — the
degenerate, unremarkable case). Making the vector side optional would mean sometimes ranking purely
by an unbounded, uncalibrated lexical score with no floor at all — U1 §3 already establishes that
has no stable meaning across queries. Not attempted.

### 3.6 Return-shape change — additive, not breaking, and it retires one client-side step

Today's `search_documents`/`search_chunks_fulltext`-free row shape:
`{chunkId, text, documentId, seq, score}` (score = cosine distance). Post-fusion, every returned row
gains three fields — `rrfScore` (float, always present), `vectorRank` (`int | None`, 1-indexed
position in the vector-side top-K, `None` if the chunk was absent from the vector signal entirely),
`lexicalRank` (`int | None`, same for the lexical side) — and `score`'s meaning narrows: it is
**only** the vector cosine distance, unconditionally `vector_score.get(chunk_id)` (Step 3's code).
**`score` is `None` if and only if the chunk was absent from the vector signal's own top-K list at
all — not "if and only if admitted via the lexical gate."** Those are two different conditions: a
chunk *present* in the vector top-K but scoring above the 0.43 floor, admitted only because it also
ranks ≤2 lexically, still carries its real (floor-failing) vector distance in `score` — e.g. `0.6`,
not `None`. `score is None` therefore means specifically "no vector hit at all," a strict subset of
"admitted via the lexical half of the gate." This is a deliberate, honest choice, not an oversight:
RRF is rank-based (U2 §1), so there is no single scalar "the fused score" that means the same thing
as the old cosine-distance `score` did — collapsing `rrfScore` into the `score` key would silently
break anyone who still reads `score` expecting a `[0, ~0.5]` cosine-distance range and instead gets
a `~0.03` RRF value.

**Confirmed non-breaking at both transports** (§ CPG note above — no `response_model` lock on
either route), so this is safe to ship without a version bump or a new tool name.

**One real documentation consequence, beyond the brief's own list, flagged here rather than
silently fixed:** `skills/agent-kb-retrieval/SKILL.md` step 4 ("Apply the score floor yourself —
reject any returned hit with `score > 0.43`") is written for a caller of `search_documents`
specifically. Once this plan ships, every row `search_documents` returns has **already** passed the
admissibility gate (vector ≤0.43 OR lexical rank ≤2) server-side — a caller re-applying "`score >
0.43` → reject" on a lexical-gate admit would either crash on the comparison (when `score` is
`None`, the "absent from vector entirely" case) or incorrectly discard a validly-admitted hit
(when `score` is populated but floor-failing, e.g. `0.6` — the "present in vector, admitted via
lexical rank instead" case, §3.6 above). Both variants of a lexical-gate admit break the same
client-side re-check; the skill file
needs a revision (§4 Step 6 below) stating that step 4 is now redundant for `search_documents`
(the gate is enforced server-side) while keeping the value's derivation history intact — this is
this plan's own finding, not something U1/U2 flagged, so `analyst`'s gate should confirm the fix
reads correctly before it ships with U4.

### 3.7 Over-fetch depth scales with `limit` — closing a real capacity regression `analyst` caught

**Finding (Pass 1 review, Major):** a fixed `HYBRID_OVERFETCH_K=20` over-fetch depth, independent
of the caller's own `limit`, means `_fuse_chunk_hits_rrf`'s output can never exceed the union of
two ≤20-row lists (≤40 unique chunks, fewer once overlap and the gate are applied) — regardless of
`limit`. `api.py:245` declares `limit: int = Query(20, ge=1, le=200)` and `mcp.py:378`'s tool takes
an uncapped `limit: int = 20` — both promise up to 200 (REST) or any value (MCP) today, a promise
this plan's original Step 4 design would silently break for any caller requesting more than ~20-40.

**Fix adopted: scale the over-fetch depth to `depth = max(HYBRID_OVERFETCH_K, limit)`**, replacing
every use of the bare `HYBRID_OVERFETCH_K` constant in `search_documents`'s repository calls (§4
Step 4, revised) with this per-call `depth`. `HYBRID_OVERFETCH_K` itself stays as the *floor* — the
one and only value that governs the calling convention this plan was actually designed around
(`skills/agent-kb-retrieval/SKILL.md`'s fixed `limit=5`, always well under 20, so `depth` reduces to
`HYBRID_OVERFETCH_K` unchanged for every call that convention makes) — while a caller outside that
convention requesting more gets a proportionally deeper over-fetch on both signals, preserving the
pre-existing `search_documents(limit=N)` contract: "as many as `N`, corpus permitting," not "as
many as `N`, capped at ~40 regardless of corpus size."

**Why this option over the other two `analyst` named, and rejected, not silently:**
- *Document the ~40-row ceiling as an accepted ceiling (§6), change nothing* — rejected: this would
  leave `api.py`'s own declared `le=200` contract false for any caller who actually exercises it,
  and doesn't touch `mcp.py`'s uncapped tool at all — a documented lie is still a lie a future
  caller can be burned by.
- *Tighten `api.py`'s `le=200` down to something ≤40* — rejected: `search_documents`'s REST/MCP
  surface is not exclusively the agent-KB retrieval convention's own call shape; narrowing the
  transport-level bound to match this one retrieval mechanism's current internal constant couples
  an unrelated validation limit to an implementation detail that may itself change later (the ml
  note's own §RRF-k retune risk), and it still leaves `mcp.py` uncapped/inconsistent with REST
  either way — two transports, one fix, in different files, is a worse shape than one fix in the
  one file that actually causes the ceiling.
- **Scaling the depth** fixes both transports uniformly with a one-line change at the one call site
  that computes it, requires no change to `api.py`/`mcp.py`, and costs nothing for the convention
  this plan was designed around (unchanged behavior at `limit=5`) — proportionally more work only
  for a caller who explicitly asks for more, which is the expected, acceptable cost of that ask.

**Consequence for the vector side's own ANN over-fetch:** `k = depth * SEARCH_DOCUMENTS_OVERFETCH`
(unchanged formula, `depth` substituted for the old bare `HYBRID_OVERFETCH_K`) — the existing
document-ingestion2 Stage C under-fill protection scales the same way, automatically, with no
separate change needed.

---

## 4. Step-by-step implementation

**Step 1 — schema (land and verify before any code change).**

Edit `falkor-chat/scripts/bootstrap_schema.sh`, inside `bootstrap_workspace()`, immediately after
the existing `Document.title` fulltext block (currently ending at line 373):
```bash
  # K-030 item 2 (claude/docs/plans/agent-knowledge-base-strategy7-impl.md): the lexical half
  # of the FR-3 standalone-KB hybrid search fusion (Services.search_documents). Backfills
  # automatically on the already-populated Chunk label — confirmed live, no reindex step, no
  # observed async PENDING window at this corpus's scale (graph-dba,
  # agent-knowledge-base-strategy-graph.md §4).
  echo "[fulltext] Chunk.text"
  gquery "$g" "CALL db.idx.fulltext.createNodeIndex('Chunk', 'text')"
```
Idempotent, same tolerant-of-"already indexed" behavior as its three siblings — no script-logic
change needed.

**Apply it to `ws:agent-team`:**
```bash
cd falkor-chat
EMBEDDING_DIM=1024 ./scripts/bootstrap_schema.sh agent-team
```
(1024 matches `ws:agent-team`'s existing embedding dimension, per `seed_agent_team.sh`'s own
documented convention — re-running the full idempotent DDL set is safe and is what every other
schema addition in this codebase does; there is no narrower "just this one index" invocation and
none is needed.)

**Verify it landed** (no new persistent `verify_*.sh` script — rationale below):
```bash
redis-cli -p 6379 GRAPH.RO_QUERY ws:agent-team \
  "CALL db.indexes() YIELD label, types, status RETURN label, types, status"
```
Expect a `Chunk` row whose `types` map now includes `text: [FULLTEXT]` alongside the existing
`embedding: [VECTOR]`/`chunkId: [RANGE]` entries, `status: OPERATIONAL`. Follow with one functional
smoke query (not just DDL-registration):
```bash
redis-cli -p 6379 GRAPH.RO_QUERY ws:agent-team \
  "CALL db.idx.fulltext.queryNodes('Chunk', 'falkordb') YIELD node RETURN count(node)"
```
expecting a positive count (U1 measured 33 matching chunks for this exact term against the same
corpus, §3 of the graph note) — a `0` here means the index exists but never queries correctly,
which the DDL-registration check alone cannot catch.

**Why no new `scripts/verify_*.sh`:** every existing `verify_*.sh` (`verify_workflows.sh`,
`verify_catalog.sh`, `verify_salesperson.sh`) guards **seeded data** that can drift across re-runs
or across `reference`/`ws:<id>` splits — this is a pure, idempotent DDL addition with nothing
seeded to drift. A persistent script would have nothing repeatable to check beyond what `db.
indexes()` already answers in one line. If the team later wants a standing guard across all four
fulltext indexes (not just this one), that generalization is a reasonable follow-up but is out of
this unit's scope — noted, not built.

**Step 2 — `Repository.search_chunks_fulltext`** (`falkor-chat/server/falkorchat/repository.py`,
immediately after `search_chunks` ends, i.e. after line 1284):
```python
def search_chunks_fulltext(
    self, ws: str, *, query: str, limit: int = 10, timeout: int | None = None,
) -> list[dict[str, Any]]:
    """`Chunk`-only full-text (RediSearch) retrieval — the lexical half of the
    FR-3 standalone-KB hybrid search (K-030 item 2,
    `claude/docs/plans/agent-knowledge-base-strategy7-impl.md`). Read path (`ro_query`).

    Mirrors `search_chunks`'s shape (same post-`YIELD` `WHERE seed.documentCurrent
    = true` superseded-chunk exclusion, same denormalized `documentId`/`seq`) with
    `db.idx.fulltext.queryNodes('Chunk', ...)` in place of
    `db.idx.vector.queryNodes(...)` — the same `Message`-to-`Chunk` label swap
    `search_messages` (§5) already demonstrates for full-text.

    `score` is a RediSearch TF-IDF-family score, **higher is better**
    (`ORDER BY score DESC`) — the OPPOSITE convention from `search_chunks`'s
    cosine-distance `ASC` (`claude/docs/plans/agent-knowledge-base-strategy-graph.md`
    §3). Do not re-sort, and never compare this score directly against
    `search_chunks`'s.

    `query` must be the caller's raw, unwrapped situation text — never a
    vector-only query-instruction prefix (`skills/agent-kb-retrieval/SKILL.md`);
    this method does no stripping itself, mirroring `search_chunks`'s own
    "caller is responsible for what it hands in" posture toward `q_vec`.

    Raises `redis.exceptions.ResponseError` on RediSearch syntax rejection,
    uncaught here — `Services.search_documents` wraps it into
    `InvalidSearchQueryError`, mirroring `Services.search_messages`.
    """
    res = self._graph(ws).ro_query(
        "CALL db.idx.fulltext.queryNodes('Chunk', $query) "
        "YIELD node AS seed, score "
        "WHERE seed.documentCurrent = true "
        "RETURN seed.chunkId AS chunkId, seed.text AS text, "
        "seed.documentId AS documentId, seed.seq AS seq, score "
        "ORDER BY score DESC "
        "LIMIT $limit",
        {"query": query, "limit": limit},
        timeout=timeout,
    )
    return [
        {
            "chunkId": row[0], "text": row[1], "documentId": row[2],
            "seq": row[3], "score": row[4],
        }
        for row in res.result_set
    ]
```

**Step 3 — fusion+gate function and constants** (`falkor-chat/server/falkorchat/services.py`, new
module-level block near `SEARCH_DOCUMENTS_OVERFETCH`, `:175`):
```python
# ── K-030 item 2: hybrid lexical+semantic RRF fusion + admissibility gate ──
# (claude/docs/plans/agent-knowledge-base-strategy7-impl.md; formula decided by
# data-scientist, claude/docs/plans/agent-knowledge-base-strategy-ml.md "Item 2" —
# this file executes it, does not re-derive it.)
RRF_K = 60  # Cormack/Clarke/Buettcher 2009 literature default, not re-derived for
            # this corpus (ml note, Risks) — retune here first if a future eval
            # result is close but not clean.
HYBRID_OVERFETCH_K = 20  # each signal's own top-K depth fed into RRF — the
    # FLOOR for that depth, matching the depth Stage 8's golden-set evaluation
    # already probed to (ml note §1). `search_documents` computes the actual
    # per-call depth as `max(HYBRID_OVERFETCH_K, limit)` (§3.7) — this constant
    # alone does NOT bound the depth for a caller requesting `limit` >20; it
    # only sets the depth used by the `skills/agent-kb-retrieval/SKILL.md`
    # convention's fixed `limit=5`, where it is the effective, unchanged value.
VECTOR_ADMISSIBILITY_FLOOR = 0.43  # canonical value now lives here — moved from
    # being solely a client-side convention (skills/agent-kb-retrieval/SKILL.md,
    # which now cites this constant rather than restating the number). A
    # candidate is admissible if its vector cosine distance is at or under this
    # floor, OR it passes the lexical rank gate below (OR, not AND).
LEXICAL_ADMISSIBILITY_RANK = 2  # ...it ranks at or above this 1-indexed position
    # in the lexical OR-term result list (ml note §2).

# skills/agent-kb-retrieval/SKILL.md's exact fenced query-instruction-prefix
# template (quoted verbatim there, not re-derived here):
#
#   f"Instruct: Given a coding agent's description of its current situation,
#   retrieve the distilled technique or rule that applies to it.\nQuery:
#   {situation}"
#
# A compliant caller of `search_documents` substitutes its own {situation}
# text into this f-string and sends the WHOLE result as `query` (that skill's
# step 1/2 — "nothing strips one you forget to add," i.e. this is not an
# optional extra a caller sometimes omits). The vector signal
# (`embedder.embed(query)`) needs this prefix verbatim; the lexical signal
# must never see it (ml note, Item 2 §2) — the boilerplate terms (`Instruct`,
# `retrieve`, `situation`, `technique`...) would otherwise pollute every
# lexical query's TF-IDF signal with the same constant, query-independent
# term set. `QUERY_INSTRUCTION_MARKER` anchors the strip on the template's
# structural boundary (Qwen3-Embedding's own `Instruct:.../nQuery:...` shape)
# rather than the full instruction sentence, which is more likely to be
# reworded over time — see `_strip_query_instruction_prefix`'s own docstring
# for why the split uses the FIRST occurrence, not the last. **Not a
# decoupled copy, though**: if `SKILL.md`'s template ever drops this
# `\nQuery: ` shape entirely, this constant needs a matching update, with
# nothing automated to catch a miss (that file's own drift check,
# `claude/scripts/audit-team.sh`, greps only `claude/`-side prompts).
QUERY_INSTRUCTION_MARKER = "\nQuery: "


def _strip_query_instruction_prefix(query: str) -> str:
    """Recover raw situation text from a `search_documents` `query` argument
    that (per `skills/agent-kb-retrieval/SKILL.md`'s calling convention) a
    compliant caller always sends already wrapped in the query-instruction
    prefix. Used only for the lexical (full-text) call — the vector call
    keeps `query` unchanged, since Qwen3-Embedding's own asymmetric
    query/document convention needs the prefix verbatim.

    Splits on the FIRST occurrence of `QUERY_INSTRUCTION_MARKER`, not the
    last: the template's own marker is always the earliest occurrence in a
    correctly-constructed `query` string, so anchoring on the first
    occurrence is what keeps a situation description that happens to itself
    contain the literal substring `"Query: "` intact in the returned tail,
    rather than truncating into the middle of it (the failure mode
    `str.rpartition` would introduce). If the marker is absent entirely (a
    caller that omitted the prefix, or any caller outside this convention),
    `query` is returned unchanged — nothing here repairs a missing prefix,
    mirroring the skill's own "nothing strips one you forget to add" posture;
    there is simply nothing to strip.
    """
    _, sep, tail = query.partition(QUERY_INSTRUCTION_MARKER)
    return tail if sep else query


def _fuse_chunk_hits_rrf(
    vector_hits: list[dict[str, Any]],
    lexical_hits: list[dict[str, Any]],
    *, limit: int,
    k: int = RRF_K,
    vector_floor: float = VECTOR_ADMISSIBILITY_FLOOR,
    lexical_rank_gate: int = LEXICAL_ADMISSIBILITY_RANK,
) -> list[dict[str, Any]]:
    """Reciprocal Rank Fusion (equal weights) over `vector_hits` (`search_chunks`
    shape, cosine distance ASC — list position 0 = rank 1) and `lexical_hits`
    (`search_chunks_fulltext` shape, RediSearch score DESC — list position 0 =
    rank 1), then U2's admissibility gate, walking the fused order and keeping
    the first `limit` candidates that pass — continuing past any that don't is
    the backfill (ml note §1/§2; mirrors `Services.search_documents`'s existing
    over-fetch-then-filter idiom, `SEARCH_DOCUMENTS_OVERFETCH`).

    Trusts each input list's own order as its rank; does not re-sort by `score`
    itself (both repository calls already return their own `ORDER BY`). A chunk
    absent from one list contributes 0 to that side's RRF term — plain RRF, no
    special-casing. Ties in fused RRF score break on `chunkId` ascending, for
    determinism.

    Each returned row carries the original `chunkId`/`text`/`documentId`/`seq`
    plus `score` (the vector cosine distance if this chunk was in `vector_hits`
    at all, else `None`), `rrfScore`, `vectorRank`, `lexicalRank` (the latter
    two `None` when absent from that signal). **`score is None` iff the chunk
    was absent from `vector_hits` — it is NOT a synonym for "admitted via the
    lexical gate"**: a chunk present in `vector_hits` with a floor-failing
    score, admitted only because it also passes `lexical_rank_gate`, still
    reports that real (floor-failing) score here, not `None` (§3.6).
    """
    vector_rank = {row["chunkId"]: i + 1 for i, row in enumerate(vector_hits)}
    vector_score = {row["chunkId"]: row["score"] for row in vector_hits}
    lexical_rank = {row["chunkId"]: i + 1 for i, row in enumerate(lexical_hits)}

    by_id: dict[str, dict[str, Any]] = {}
    for row in lexical_hits:
        by_id[row["chunkId"]] = row
    for row in vector_hits:  # vector's own text/documentId/seq wins on overlap
        by_id[row["chunkId"]] = row

    def rrf(chunk_id: str) -> float:
        score = 0.0
        if chunk_id in vector_rank:
            score += 1.0 / (k + vector_rank[chunk_id])
        if chunk_id in lexical_rank:
            score += 1.0 / (k + lexical_rank[chunk_id])
        return score

    fused_order = sorted(by_id, key=lambda cid: (-rrf(cid), cid))

    def admissible(chunk_id: str) -> bool:
        vs = vector_score.get(chunk_id)
        if vs is not None and vs <= vector_floor:
            return True
        lr = lexical_rank.get(chunk_id)
        return lr is not None and lr <= lexical_rank_gate

    out: list[dict[str, Any]] = []
    for chunk_id in fused_order:
        if not admissible(chunk_id):
            continue
        row = dict(by_id[chunk_id])
        row["score"] = vector_score.get(chunk_id)
        row["rrfScore"] = rrf(chunk_id)
        row["vectorRank"] = vector_rank.get(chunk_id)
        row["lexicalRank"] = lexical_rank.get(chunk_id)
        out.append(row)
        if len(out) >= limit:
            break
    return out
```

**Step 4 — wire it into `Services.search_documents`** (`services.py:1360`, replacing the current
single `return self._repo.search_chunks(...)`):
```python
def search_documents(
    self, ctx: CallContext, *, query: str, limit: int = 20,
) -> list[dict[str, Any]]:
    """FR-3 standalone KB search: rank ingested `Chunk`s by similarity to
    `query` — hybrid lexical (full-text, `search_chunks_fulltext`) + semantic
    (vector, `search_chunks`) signals, fused by Reciprocal Rank Fusion and
    gated by `_fuse_chunk_hits_rrf` (K-030 item 2,
    `claude/docs/plans/agent-knowledge-base-strategy7-impl.md`).

    Both signals are over-fetched to `depth = max(HYBRID_OVERFETCH_K, limit)`
    — at least `HYBRID_OVERFETCH_K` (=20, matching the depth this KB's own
    golden-set evaluation already probed to; the effective, unchanged value
    for `skills/agent-kb-retrieval/SKILL.md`'s fixed `limit=5` convention),
    or `limit` itself when a caller requests more than that (REST's
    `le=200`, MCP's uncapped `limit`) — so a large-`limit` caller still gets
    up to `limit` admissible rows when the corpus can supply them, rather
    than being silently capped at a fixed ~20-40-row ceiling regardless of
    corpus size (§3.7; `docs/reviews/agent-knowledge-base-strategy7-impl.md`
    Pass 1 Major finding). The vector call additionally over-fetches its own
    ANN candidate pool via `k = depth * SEARCH_DOCUMENTS_OVERFETCH`, the same
    document-ingestion2 Stage C under-fill protection `search_chunks`'s
    `documentCurrent` post-filter already requires.

    Raises `SearchNotAvailableError` when no `ModelGateway` is wired (503) —
    unchanged; the vector signal is mandatory even though the lexical one
    would not need an embedder, because the admissibility gate's primary path
    depends on a calibrated vector floor (ml note §1/§3; no floor exists for
    lexical scores alone, they are corpus/query-dependent and unbounded).

    Raises `InvalidSearchQueryError` (400) if the lexical call rejects
    `query`'s RediSearch syntax — mirrors `search_messages`.

    **`query` carries the query-instruction prefix verbatim for the vector
    call** (`embedder.embed(query)`, unchanged from today) **but is stripped
    of it for the lexical call** (`_strip_query_instruction_prefix(query)`) —
    a compliant caller always sends the wrapped form
    (`skills/agent-kb-retrieval/SKILL.md` step 1/2; §3.3 above), so this
    method, not the caller, is responsible for recovering the raw situation
    text the lexical signal needs.
    """
    if self._models is None:
        raise SearchNotAvailableError(
            "search_documents requires a configured ModelGateway (no "
            "embedding model wired into this deployment)"
        )
    embedder = self._models.embedder("embedding", ws=ctx.ws)
    q_vec = embedder.embed(query)
    depth = max(HYBRID_OVERFETCH_K, limit)
    vector_hits = self._repo.search_chunks(
        ctx.ws, q_vec=q_vec,
        k=depth * SEARCH_DOCUMENTS_OVERFETCH,
        limit=depth, timeout=RAG_QUERY_TIMEOUT_MS,
    )
    lexical_query = _strip_query_instruction_prefix(query)
    try:
        lexical_hits = self._repo.search_chunks_fulltext(
            ctx.ws, query=lexical_query, limit=depth,
            timeout=RAG_QUERY_TIMEOUT_MS,
        )
    except ResponseError as exc:
        raise InvalidSearchQueryError(str(exc)) from exc
    return _fuse_chunk_hits_rrf(vector_hits, lexical_hits, limit=limit)
```
`ResponseError` is already imported at module top (`services.py:23`) — no new import needed.

**Step 5 — no change to `mcp.py`/`api.py`.** Both call `services.search_documents(ctx, query=...,
limit=...)` with the same signature as today; confirmed by the CPG query (§ above) that these are
the only two callers and neither pins a response schema. Their docstrings/comments do not assert
anything about `score`'s exact semantics that this change would falsify (`mcp.py`'s own docstring,
"`score` is cosine distance — lower is more similar," is now imprecise for a lexical-only admit —
see Step 6).

**Step 6 — documentation, folded into this unit's own done-conditions:**

1. **`mcp.py:378`'s `search_documents` tool docstring** — revise "Returns chunks ordered most-
   similar-first (`score` is cosine distance — lower is more similar)..." to state the fused
   ordering and `score`'s now-conditional meaning (cite §3.6 above rather than re-deriving it).
2. **`falkor-chat/docs/DESIGN.md:616-619`** — the "Full-text index (RediSearch)" bullet currently
   reads `Message.text`, `Entity.name` — add `Chunk.text`. **Also fix, in the same edit, a
   pre-existing omission found while editing this exact line: `Document.title` is a fourth live
   fulltext index (`bootstrap_schema.sh:373`, document-ingestion2 Stage D) never added to this
   register** — a one-word fix sitting on the line being touched anyway; flagged here rather than
   silently folded in, since it's not something U1/U2/this brief asked for.
3. **`falkor-chat/docs/QUERIES.md`** — new subsection immediately after §14.3 (`search_chunks`),
   named **§14.3a "Lexical+semantic RRF fusion for `search_documents`"** — deliberately not
   "hybrid_search," which already names the unrelated `Message`+`Chunk` vector-only merge (§6/§10).
   Content: the `search_chunks_fulltext` Cypher (Step 2), the fusion+gate formula in prose (citing
   the ml note for derivation, not restating the rationale), and the four named constants
   (`RRF_K`, `HYBRID_OVERFETCH_K`, `VECTOR_ADMISSIBILITY_FLOOR`, `LEXICAL_ADMISSIBILITY_RANK`) with
   `services.py` named as their canonical location. Also revise §14.4's prose ("`search_documents`
   embeds `query`... then calls `search_chunks` (§14.3) above") to point at §14.3a instead of
   implying vector-only retrieval.
4. **`skills/agent-kb-retrieval/SKILL.md`** — per §3.6's finding: revise the "Score floor" section
   to state that, for `search_documents` specifically, the 0.43-or-lexical-rank-≤2 admissibility
   gate is now enforced server-side (`falkorchat/services.py`'s `VECTOR_ADMISSIBILITY_FLOOR`/
   `LEXICAL_ADMISSIBILITY_RANK`, cited, not restated) and step 4 of the calling convention is no
   longer a required client action for this tool — keep the value's derivation history (Stage 8
   Phase 1/2, the R6/h40 instability caveat) intact, since it remains historically accurate.
5. **`falkor-chat/docs/HISTORY.md`** — a dated entry once delivered. **Lands with U4's
   implementation, not this plan** — stated explicitly per the brief, not left implicit.

---

## 5. Test strategy

All new coverage lands in `falkor-chat/server/tests/test_services.py` (no live-DB test needed for
the fusion logic itself — it's pure Python; a live smoke check is Step 1's `redis-cli` verification,
not a pytest node). Two tiers: direct unit tests of `_fuse_chunk_hits_rrf` (fast, exhaustive,
mutation-test-oriented), and `Services.search_documents` tests via `FakeRepo` (wiring/composition).
`_fuse_chunk_hits_rrf`, `RRF_K`, `HYBRID_OVERFETCH_K`, `VECTOR_ADMISSIBILITY_FLOOR`,
`LEXICAL_ADMISSIBILITY_RANK` are all directly importable from `falkorchat.services`, same as
`_diff_structures`/`SEARCH_DOCUMENTS_OVERFETCH` already are (`test_services.py:26-54`).

### 5.1 `_strip_query_instruction_prefix` — direct unit tests

`_strip_query_instruction_prefix` and `QUERY_INSTRUCTION_MARKER` are directly importable from
`falkorchat.services`, same as the other new constants/functions.

1. **Prefix present — stripped to the tail.** `query =
   "Instruct: Given a coding agent's description of its current situation, retrieve the "
   "distilled technique or rule that applies to it.\nQuery: real situation text"` →
   `_strip_query_instruction_prefix(query) == "real situation text"`.
2. **Prefix absent — unchanged passthrough.** `_strip_query_instruction_prefix("bare query, no
   prefix") == "bare query, no prefix"` — the graceful-degradation case (§3.3).
3. **Marker embedded in the situation text itself is preserved, not truncated.** **Fixed per
   `analyst`'s Pass 1 finding (`claude/docs/reviews/agent-knowledge-base-strategy7-impl.md`,
   Blocker) — the construction below is Appendix A's verified `q2`, not the plan's original,
   inert `q` (which used a double backslash, `\\nQuery: `, producing only one literal-newline
   marker occurrence — `partition`/`rpartition` returned byte-identical tails on it, so that
   version could not have caught the mutation it claimed to).** The embedded occurrence below
   MUST be a genuine second `\n` (single backslash — the same escaping items 1/2/4 already use
   correctly), not `\\n`:
   ```python
   query = "Instruct: ...\nQuery: does the situation text mention \"\nQuery: \" literally?"
   ```
   This string contains **two** real occurrences of `QUERY_INSTRUCTION_MARKER` (`"\nQuery: "`) —
   confirmed live (Appendix A): `query.count(QUERY_INSTRUCTION_MARKER) == 2`. Assert
   `_strip_query_instruction_prefix(query) == 'does the situation text mention "\nQuery: " '
   'literally?'` (the first-occurrence split, keeping the situation text's own embedded marker
   intact) — and, as the actual pinning assertion, assert this differs from
   `query.rpartition(QUERY_INSTRUCTION_MARKER)[2]` (`'" literally?'`, the last-occurrence split),
   e.g. `assert query.partition(MARKER)[2] != query.rpartition(MARKER)[2]` as a sanity check the
   fixture itself is discriminating before asserting the function's actual output. **Catches a
   mutation swapping `partition` for `rpartition`** — this is the one test in this section that
   exists specifically to pin "first occurrence, not last" (§3.3's own named design choice), and
   with a genuine two-occurrence fixture it now actually does.
4. **Prefix-present tail can legitimately be empty.** `query =
   "Instruct: ...\nQuery: "` (template with nothing substituted) →
   `_strip_query_instruction_prefix(query) == ""`, **not** the original wrapped string — catches a
   naive `tail or query` fallback (falsy-empty-string bug) in place of the `sep`-presence check the
   design actually uses.

### 5.2 `_fuse_chunk_hits_rrf` — direct unit tests

Each test below names the specific defect it exists to catch (per the brief's ask):

1. **Vector-only admit.** One vector hit (rank 1, `score=0.1`), no lexical hits → returned with
   `score=0.1`, `vectorRank=1`, `lexicalRank=None`, `rrfScore == 1/(60+1)`. Baseline correctness.
2. **Lexical-only admit.** A chunk absent from `vector_hits`, present at lexical rank 1 → admitted
   (`score=None`, `vectorRank=None`, `lexicalRank=1`) — proves the OR-gate's lexical half actually
   admits something the vector floor alone never would.
3. **Lexical-only rejection at rank 3.** A chunk absent from `vector_hits`, present at lexical rank
   3 (fails `lexical_rank_gate=2`) → **not** in the output, even though it has a nonzero `rrfScore`
   and would otherwise rank ahead of a weaker admitted candidate. **Catches "a gate that never
   rejects anything"** — the one mutation the brief names explicitly; without this test a gate that
   always returns `True` still passes every other case in this list.
4. **Both-signals-fail rejection.** A chunk with vector `score=0.6` (above the 0.43 floor) *and*
   lexical rank 3 (below the rank-2 gate) → excluded — proves the OR is evaluated on both halves,
   not short-circuited to "admit if present in either list at all" regardless of threshold.
5. **RRF arithmetic, pinned exactly.** Candidate A: vector rank 1, no lexical. Candidate B: vector
   rank 5, lexical rank 1. Assert `rrfScore`s exactly equal `1/61` (A) and `1/65 + 1/61` (B), and
   that B outranks A in `fused_order` (B's sum > A's). **Catches a wrong `k`** (any `k != 60` changes
   both numbers, this test's exact-value assertion fails) **and a flipped sort direction** (sorting
   ascending instead of descending on `-rrf(cid)` would put A first).
6. **Backfill.** Four candidates in fused order; the first two fail the gate, the third and fourth
   pass. Call with `limit=2` → output is exactly `[candidate3, candidate4]`, not `[]` or a
   `len()==0` short-circuit. Directly exercises U2's named backfill requirement.
7. **Empty inputs.** Both lists empty → `[]`, no exception (matches the existing floor's graceful
   degradation).
8. **Deterministic tie-break.** Two candidates constructed to have identical fused RRF scores (e.g.
   both at vector rank 1 in independent, non-overlapping lists is not tie-eligible — construct via
   one at vector-rank-2/lexical-rank-2 vs. one at vector-rank-2/lexical-rank-2 for a different
   chunk, so both accumulate the identical sum) → output order is `chunkId` ascending, stable across
   repeated calls — catches removal of the explicit `(-rrf(cid), cid)` secondary sort key.
9. **`limit` truncation post-gate.** Five candidates all pass the gate; `limit=2` → exactly 2
   returned, the top 2 by fused order — proves `limit` still caps after gating, not before.

### 5.3 `Services.search_documents` — `FakeRepo`-level wiring tests

**Prerequisite `FakeRepo` addition** (`test_services.py`, near `chunk_rows`/`since_rows`, `:107-114`):
```python
self.chunk_fulltext_rows: list[dict] | Exception = []
```
and a new method (mirroring `search_messages`'s exception-scripting idiom, `:219-223`):
```python
def search_chunks_fulltext(self, ws, *, query, limit, timeout=None):
    self.calls.append(("search_chunks_fulltext", ws, query, limit, timeout))
    if isinstance(self.chunk_fulltext_rows, Exception):
        raise self.chunk_fulltext_rows
    return self.chunk_fulltext_rows
```
Defaults to `[]` (not `since_rows`) — the common, unremarkable degenerate case (U2 §2) should be
the fixture default, and reusing `since_rows` would silently couple this new call to whatever other
test in the file happens to have set it for an unrelated reason.

**Existing tests requiring rewrite (items 1, 2, 5), plus the two new depth-scaling tests the §3.7
fix requires (items 3, 4, tagged `**New**`, interleaved here rather than in the "new tests" list
below since they belong next to the existing test they replace/extend):** items 1/2/5 currently
assert a bare `rows == repo.since_rows` passthrough and a single `search_chunks` call-argument
tuple — both assumptions break.

1. `test_search_documents_embeds_the_query_then_searches_chunks` — rewrite to seed exactly one
   vector hit with `score` at or under 0.43 (so it survives the gate) and leave
   `chunk_fulltext_rows` at its `[]` default; call `search_documents` with a plain, unwrapped
   `query="hello", limit=5` (no query-instruction prefix — this test is about the embed/over-fetch
   wiring, not the strip behavior, which gets its own dedicated test, item 8 below; `limit=5` keeps
   `depth = max(HYBRID_OVERFETCH_K, 5) == HYBRID_OVERFETCH_K == 20`, so the floor branch of §3.7's
   formula, not its scaling branch — that gets its own test, item 3 below) and assert **both**
   calls fire — `search_chunks` at `k=HYBRID_OVERFETCH_K*SEARCH_DOCUMENTS_OVERFETCH,
   limit=HYBRID_OVERFETCH_K` (not `limit*SEARCH_DOCUMENTS_OVERFETCH`/`limit`) and
   `search_chunks_fulltext` at `limit=HYBRID_OVERFETCH_K` with `query == "hello"` (unchanged,
   since there is no prefix to strip — the passthrough case) — then assert the final row equals
   the one vector hit plus the three added fields (`rrfScore`, `vectorRank=1`,
   `lexicalRank=None`).
2. **Replace** `test_search_documents_defaults_limit_to_20` with
   `test_search_documents_hybrid_overfetch_depth_is_the_floor_below_it` — call
   `search_documents(..., limit=3)` (a value below the floor) and assert both repository calls
   request depth `HYBRID_OVERFETCH_K` (=20), **not** `limit` (=3) — pins §3.7's `max(...)` floor
   half of the formula (the old test's name/intent, "defaults to 20," no longer describes anything
   meaningful once 20 is a floor rather than a fixed constant or a default parameter value).
3. **New** `test_search_documents_overfetch_depth_scales_up_for_a_large_limit` — call
   `search_documents(..., limit=50)` (a value **above** the floor) and assert both repository
   calls request depth `50` (`search_chunks` at `k=50*SEARCH_DOCUMENTS_OVERFETCH, limit=50`;
   `search_chunks_fulltext` at `limit=50`) — pins §3.7's scaling half of the formula, the actual
   fix for the Major finding; items 2 and 3 together are what "the depth is
   `max(HYBRID_OVERFETCH_K, limit)`," not "the depth is fixed at 20" or "the depth always equals
   `limit`," actually means.
4. **New** `test_search_documents_large_limit_is_not_capped_by_the_hybrid_overfetch_floor` — the
   end-to-end regression test for the Major finding itself (`analyst`'s specific ask), reusing
   `_RankedChunkRepo` (item 5 below) seeded with **50** vector-hit rows, all
   `documentCurrent: True`, all `score` at or under 0.43 (so every one clears the gate), no
   lexical hits. Call `search_documents(..., limit=50)` and assert `len(rows) == 50` — **not**
   capped at ~40. This test fails under the plan's pre-fix Step 4 (fixed `limit=HYBRID_OVERFETCH_K`
   =20 passed to the repository call would slice `_RankedChunkRepo`'s pool down to 20 rows before
   fusion ever sees the other 30) and passes once §3.7's `depth` scaling lands — the one test in
   this section that actually exercises the *repository-call-level* slicing a real `LIMIT $limit`
   Cypher clause would impose, which items 2/3 (call-argument assertions only) do not by
   themselves prove fixes the user-visible symptom.
5. `test_search_documents_overfetch_prevents_under_fill_when_superseded_chunks_rank_first` — update
   `_RankedChunkRepo`'s pool fixture (`:1409-1412`) to add a `"score": 0.1` (or any value ≤0.43) key
   to every row — the fusion function's `vector_score` dict-comprehension requires the key to exist,
   which the current fixture (chunkId/documentCurrent only) does not provide. Keep the assertion on
   `[r["chunkId"] for r in rows] == ["nc0", "nc1", "nc2"]` — the ordering invariant under test is
   unchanged by fusion when there are no lexical hits at all (the default `[]`).

**New tests (continued):**

6. `test_search_documents_raises_invalid_search_query_on_lexical_syntax_error` — seed
   `repo.chunk_fulltext_rows = ResponseError("RediSearch: Syntax error...")`, assert
   `search_documents` raises `InvalidSearchQueryError` — mirrors
   `test_search_messages_maps_syntax_error_to_service_error` (`:2056`) exactly.
7. `test_search_documents_fuses_vector_and_lexical_hits_end_to_end` — seed both `repo.since_rows`
   (vector) and `repo.chunk_fulltext_rows` (lexical) with a small, realistic mixed set (e.g. 2
   vector-only hits, 1 lexical-only hit at rank 1, 1 chunk present in both), call
   `search_documents`, and assert the result's `chunkId` order and field values match what calling
   `_fuse_chunk_hits_rrf` directly on the same two lists would produce. This is the one test that
   would catch a wiring defect the pure-function tests structurally cannot — e.g. passing the two
   lists to `_fuse_chunk_hits_rrf` in swapped order (vector as `lexical_hits`, and vice versa) — since
   §5.2's tests always label their own fixtures correctly by construction.
8. `test_search_documents_strips_query_instruction_prefix_for_lexical_call_only` — call
   `search_documents` with `query` equal to the full, real prefixed template (§3.3's exact
   quote, `{situation}` substituted with e.g. `"raw situation text"`); assert `models.embedded ==
   [<the whole wrapped string>]` (vector side keeps it verbatim) **and** the
   `search_chunks_fulltext` call's `query` argument equals `"raw situation text"` (stripped) — the
   one test that would catch this plan's own originally-wrong resolution of §3.3 reappearing (e.g.
   a future edit that removes the strip call while leaving `_strip_query_instruction_prefix` itself
   correctly tested in isolation, §5.1 — those tests alone cannot catch a call site that stops
   invoking the function).
9. Confirm (no code change, just re-run) `test_search_documents_raises_when_no_models_wired` still
   passes unmodified — the early raise happens before either repository method is ever called.

---

## 6. Risks & open questions

- **§3.3 is resolved, not open — but the resolution itself replaced an earlier, wrong draft of
  this plan, worth recording plainly.** This plan's first draft misread U2's correctness note as
  meaning `search_documents`'s `query` argument arrives prefix-free, and proposed passing it
  unchanged to both signals. `teco` independently traced `skills/agent-kb-retrieval/SKILL.md`'s
  exact calling convention (step 1/2: a compliant caller always sends the fully wrapped string,
  and "nothing strips one you forget to add") plus a direct grep of `falkor-chat/server/
  falkorchat/*.py` (no wrap/strip step exists anywhere today) and found the opposite: a compliant
  caller's `query` is *always* prefixed, so a real strip step was required, not just a corrected
  assumption. §3.3/§4 Step 3-4 now implement `_strip_query_instruction_prefix`. **Residual risk,
  named in §3.3 and not fully closable in code:** the strip anchors on
  `QUERY_INSTRUCTION_MARKER = "\nQuery: "`, a hand-maintained fragment of `SKILL.md`'s canonical
  template with no automated drift check on this `falkor-chat`-side copy (that file's own
  `audit-team.sh` check greps only `claude/`-side prompts) — a future edit to `SKILL.md`'s template
  shape (not just its wording) needs a matching, manual update here.
- **`analyst`'s Pass 1 gate found two further defects in this plan itself, both now fixed, recorded
  here rather than silently folded away.** (1) §5.1 item 3's test (the one specifically meant to
  pin "first occurrence, not last" in `_strip_query_instruction_prefix`) was inert as originally
  written — a double-backslash typo in the fixture's embedded marker occurrence made
  `partition`/`rpartition` return identical results on it, so the test would have passed under
  either implementation; fixed using the review's own live-verified repro (Appendix A), now a
  genuine two-occurrence fixture. (2) The fixed `HYBRID_OVERFETCH_K=20` over-fetch depth silently
  capped `search_documents`'s output at ~20-40 rows regardless of a caller's requested `limit`, a
  real regression against `api.py`'s declared `le=200` contract and `mcp.py`'s uncapped tool.
  **This plan's answer to `analyst`'s own open question** ("which of the three named fixes" — not
  the reviewer's call to make): scale the over-fetch depth to `max(HYBRID_OVERFETCH_K, limit)`
  (§3.7) — chosen over documenting the ceiling as accepted (leaves a declared contract false) or
  tightening `api.py`'s `le=` (fixes only one of two transports, couples an unrelated validation
  bound to an internal retrieval constant) for the reasons §3.7 states in full. Both fixes carry
  their own new/corrected test coverage (§5.1 item 3; §5.3 items 2-4).
- **RRF's magnitude-blindness and the gate's unvalidated rank-2 threshold are U2's own named risks,
  not new ones this plan introduces** — carried forward for visibility, not re-argued: see the ml
  note's "Risks & open questions... item 2 addendum" for the full statement. This plan's test
  suite (§5.2 items 3-4) is what would catch the gate genuinely admitting a false positive, but only
  U5's live re-test against real query/document pairs can confirm the threshold itself is right.
- **No feature flag; direct schema+code change judged low-risk, not gated behind a rollout switch.**
  Rationale: (a) `ws:agent-team` is an internal team KB, not customer-facing traffic — the
  coordination doc's own framing; (b) the DDL addition is confirmed backfilling/idempotent/non-
  disruptive at this scale (U1 §4, live-verified, not inferred); (c) the code change is additive at
  both transports (§3.6, CPG-confirmed no schema lock); (d) rollback is cheap and layer-independent
  — reverting `Services.search_documents` to its pre-fusion single `search_chunks` call fully
  reverses the runtime behavior regardless of whether the fulltext index is ever dropped, and if it
  ever needs to be, `CALL db.idx.fulltext.drop('Chunk', 'text')` is a confirmed real procedure
  (`claude/graph-dba/falkordb-reference.md:28`) — not verified live by U1 in this exercise, but
  documented elsewhere in this KB as an existing capability, not a guess.
- **This plan's own scope addition (§3.6, the `SKILL.md` revision) is a recommendation, not
  something U1/U2 asked for** — flagged plainly so `analyst`'s gate can independently judge whether
  it's in scope for U4 or should be split into its own follow-up; this plan's own position is that
  shipping the code change without it leaves a stale, actively-misleading client instruction in a
  file agents read as an operative convention, which is worse than the small scope increase of
  fixing it in the same pass.
- **U5 depends on this plan's exact constant names/values landing unchanged** (`RRF_K=60`,
  `HYBRID_OVERFETCH_K=20`, `VECTOR_ADMISSIBILITY_FLOOR=0.43`, `LEXICAL_ADMISSIBILITY_RANK=2`) — if
  `coder` or a later revision changes any of them without re-flagging U2, U5's re-test would be
  validating a different mechanism than the one U2 designed. Worth a one-line check at U4's own
  gate, not just U5's.

---

**Traceability.** Inputs: `claude/docs/plans/agent-knowledge-base-strategy-graph.md` (U1,
`graph-dba`), `claude/docs/plans/agent-knowledge-base-strategy-ml.md` "## Item 2 …" (U2,
`data-scientist`). Coordination:
`claude/docs/plans/agent-knowledge-base-strategy7-coordination.md`, U3. Codebase verified directly
(not from either note's paraphrase) against the current `falkor-chat/server/falkorchat/
{repository,services,mcp,api}.py`, `falkor-chat/server/tests/test_services.py`,
`falkor-chat/scripts/bootstrap_schema.sh`, `falkor-chat/docs/{QUERIES,DESIGN}.md`,
`skills/agent-kb-retrieval/SKILL.md`, and one CPG call-graph query against `cpg_falkorchat`
(§ above). No trial run in this unit — every code sample above is a design-level spec for `coder`
to build against, not code that has been executed.
