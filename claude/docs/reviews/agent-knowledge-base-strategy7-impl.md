# Agent knowledge-base strategy — item 2 hybrid fusion, implementation plan review

> **Status:** active · **Owner:** `analyst` · **Tracks:** K-030 (`claude/cobb/kaizen/plan.md`)

## Scope & verdict

Reviewed `claude/docs/plans/agent-knowledge-base-strategy7-impl.md` (U3 of
`claude/docs/plans/agent-knowledge-base-strategy7-coordination.md`) against both upstream inputs
it builds on — `claude/docs/plans/agent-knowledge-base-strategy-graph.md` (U1, `graph-dba`) and
`claude/docs/plans/agent-knowledge-base-strategy-ml.md`'s "## Item 2 …" section (U2,
`data-scientist`) — and against the real code it describes: `falkor-chat/server/falkorchat/
{repository,services,mcp,api,app}.py`, `falkor-chat/server/tests/test_services.py`,
`falkor-chat/scripts/bootstrap_schema.sh`, `falkor-chat/docs/{DESIGN,QUERIES}.md`,
`skills/agent-kb-retrieval/SKILL.md`. Did not re-derive or second-guess the RRF formula, `k=60`,
equal weights, `K=20` over-fetch, the OR-term lexical shape, or the 0.43/rank-≤2 admissibility
gate — those are U2's settled decisions and I checked only whether U3 transcribes them faithfully.
No live-DB changes made; static review only, all claims checked by reading the actual source.

**Verdict: needs changes.** One blocker (§5.1 item 3's test is inert as specified — it cannot
catch the mutation it names) and one major (a real, unaddressed behavioral regression for a
caller requesting a large `limit`) need to be closed before this plan is handed to `coder`.
Everything else the plan claims about the current codebase (call-graph, existing test bodies,
DESIGN.md's fulltext register, the SKILL.md convention, script idempotency) checked out exactly
as stated — this is a strong, well-grounded plan with two concrete gaps, not a shaky one.

**CPG: considered, not relevant.** The plan itself already ran and cited the one CPG query that
matters (`cpg_falkorchat`, call-graph on `search_chunks`/`search_documents`) — I independently
re-verified its conclusion by direct grep of `services.py`/`mcp.py`/`api.py` rather than re-running
the CPG query myself (see Finding "Solid" below); nothing in this review turned up a second
call-graph question a CPG query would answer better than the direct reads I already did.

## Findings

### Blocker — §5.1 item 3's test cannot catch the mutation it exists to pin

The plan's own text names this as "the one test in this section that exists specifically to pin
'first occurrence, not last'" (§5.1 item 3, catching a `partition`→`rpartition` swap). As literally
specified, the situation text's embedded "second occurrence" of the marker is written with a
double backslash (`\\nQuery: `) inside the quoted Python string. Traced in Python directly
(Appendix A): a double backslash produces a literal `\` + `n` character pair, not a real newline —
so the string contains **one** actual occurrence of `QUERY_INSTRUCTION_MARKER` (`"\nQuery: "`,
a real newline), not two. `str.partition` and `str.rpartition` on a string with only one occurrence
return byte-identical tails. Confirmed live: both give `'does the situation text mention
"\\nQuery: " literally?'` — the test would pass whether `_strip_query_instruction_prefix` uses
`partition` or `rpartition`, defeating its stated purpose exactly on the case this plan was
revised once already to get right (§3.3/§6). Items 1, 2, and 4 of the same section use a real
single-backslash `\n` correctly and are unaffected.

**Fix:** the situation-text portion needs a genuine second occurrence of a real newline followed
by `"Query: "` — i.e. the same single-backslash escaping items 1/4 already use, not a double
backslash. Concretely, in actual Python source: build the string so the marker's second occurrence
is `"\nQuery: "` (real newline), not `"\\nQuery: "` (literal backslash-n). Appendix A shows both the
broken and the working construction side by side, confirmed against a live interpreter.

### Major — fixed `HYBRID_OVERFETCH_K=20` silently caps results well below what `limit` promises for a caller requesting more than ~20-40

`Services.search_documents`'s Step 4 code over-fetches both signals to a **fixed** `HYBRID_OVERFETCH_K=20`, independent of the caller's own `limit` — deliberate, per §4/§6. But the REST route
(`api.py:243-248`, confirmed read) declares `limit: int = Query(20, ge=1, le=200)` and the MCP tool
(`mcp.py:378`) takes an uncapped `limit: int = 20` — both allow a caller to request up to 200 (REST)
or any value (MCP) today. Post-fusion, `_fuse_chunk_hits_rrf`'s output can never exceed the size of
the union of two ≤20-row lists (≤40 unique chunks, fewer once overlap and the admissibility gate
are applied) — so a caller requesting `limit=100` silently gets at most ~40 rows regardless of
corpus size, where pre-fusion `search_chunks` alone could return up to `limit` rows if the corpus
had that many. `skills/agent-kb-retrieval/SKILL.md`'s only real caller is pinned at `limit=5`
(no live risk today), but this is a real regression in the API's own declared contract
(`le=200`), and nothing in §5's test plan exercises a `limit` beyond the fixed depth — the
`_fuse_chunk_hits_rrf` unit tests (§5.2) only vary `limit` at or below the candidate pool size, and
the one rewritten wiring test that varies `limit` (§5.3 item 2) uses `limit=3`, smaller than 20 in
the other direction.

**Fix:** name this explicitly in §6 Risks as an accepted, deliberate behavior change (a `limit`
beyond ~20-40 will visibly return fewer rows than requested, not silently the wrong ones — so it's
not a correctness bug, just an undocumented capacity ceiling), or tighten the REST route's `le=200`
to something consistent with the new ceiling, or scale the over-fetch depth
(`max(HYBRID_OVERFETCH_K, limit)`) if under-delivery at a large `limit` is not acceptable. Whichever
call `coder`/`teco` makes, add one test that exercises `limit` **above** `HYBRID_OVERFETCH_K`
(e.g. `limit=50` against a synthetic pool with 50+ addressable candidates) asserting the actual,
now-documented behavior — the missing axis in §5.2/§5.3's otherwise thorough coverage.

### Minor — §3.6's `score=None` framing describes only one of two cases that produce it

§3.6 states `score` "is `None` for a chunk admitted solely via the lexical rank-≤2 gate (no vector
hit at all in the top-20)." The code (Step 3) actually sets `row["score"] = vector_score.get(chunk_id)`
unconditionally, so a chunk that **does** have a vector hit but whose vector score is *above* the
0.43 floor, admitted only because its lexical rank ≤2, gets `score` populated with that
floor-failing vector distance (e.g. `0.6`) — not `None`. This isn't a code bug (the value shown is
honest and matches the field's stated meaning, "vector cosine distance"), but the docstring's
parenthetical ties `score=None` to "no vector hit at all," which is only one of the two ways a
lexical-admitted row can arise. Worth a one-clause fix in the `search_documents`/
`_fuse_chunk_hits_rrf` docstrings so a future reader doesn't assume `score is None` is the same
predicate as "admitted via the lexical gate."

## What's solid

- Every "grounding" claim I independently re-checked against the live source held exactly: the two
  `search_chunks` call sites (`services.py:1132`/`1402`), the two `search_documents` callers
  (`mcp.py:378`, `api.py:243`) with no `response_model=`/schema lock at either, `DESIGN.md`'s
  fulltext register really omitting `Document.title` (`DESIGN.md:616-619`), `search_messages`'s
  exact `ResponseError`→`InvalidSearchQueryError` idiom (`services.py:1068-1071`), and all three
  named existing `test_services.py` tests (`:1334`, `:1358`, `:1403`) with the exact bodies the plan
  describes, including the real absence of a `score` key in `_RankedChunkRepo`'s fixture.
- The RRF arithmetic in §5.2 item 5 checks out exactly (`1/61` and `1/65+1/61`, verified by hand),
  and the sort-direction/tie-break reasoning is correct.
- `_strip_query_instruction_prefix`'s actual design (first-occurrence `partition`, the `sep`-presence
  check rather than `tail or query`) is correct Python and the right choice for the stated reason —
  my finding above is about one illustrative test's construction, not the design or the shipped
  function.
- §3.3's correction (stripping the prefix only for the lexical call) is a faithful, complete fix for
  the bug `teco` caught, and the bootstrap_schema.sh placement/idempotency claims (§4 Step 1) are
  accurate against the real script (`bootstrap_schema.sh:356-373`).
- The self-found `SKILL.md` step-4 scope addition (§3.6) is worth doing in U4, not splitting off: it
  is small, and shipping the server-side gate without it leaves a client convention that will crash
  (`None > 0.43`) or silently mis-discard a validly-admitted row the first time an agent actually
  hits a lexical-only admit — my own independent judgment agrees with the plan's stated position.

## Open questions

- Whichever of the three fixes for the large-`limit` finding is chosen (document, cap tighter, or
  scale the over-fetch) is a real design call, not mine to make for `coder` — flag it back to
  `architect`/`teco` rather than have `coder` decide unilaterally while implementing.

## Pass 2 (2026-09-20)

**Re-check scope:** `architect`'s revision addressing both Pass 1 findings — §5.1 item 3's fixture,
§3.7's new `depth = max(HYBRID_OVERFETCH_K, limit)` scaling (Step 4, both repository call sites),
§5.3's new items 3-4, and §3.6/`_fuse_chunk_hits_rrf`'s docstring fix — plus a fresh pass for any
second-order defect the fixes themselves might have introduced. Not a full re-read of sections
unaffected by the revision (§1-§2, §3.1-§3.5, §4 Steps 1-2/5-6, §5.1 items 1/2/4, §5.2 unchanged).

**Verdict: approve with suggestions.** Both Pass 1 findings are fixed, correctly and completely;
one new Minor surfaced by the fix itself, not blocking.

- **Blocker (§5.1 item 3 inert fixture):** fixed. Re-derived independently, not just re-running the
  reviewer's own Appendix A: built the plan's *current* literal string (line 711) in a fresh Python
  interpreter and confirmed `query.count(QUERY_INSTRUCTION_MARKER) == 2`,
  `query.partition(MARKER)[2] != query.rpartition(MARKER)[2]`, and the partition tail matches the
  plan's own asserted expected value exactly (`'does the situation text mention "\nQuery: " literally?'`).
  The fixture now genuinely discriminates `partition` from `rpartition`.
- **Major (fixed `HYBRID_OVERFETCH_K` capacity ceiling):** fixed. Traced both call sites in Step 4
  (`search_chunks`'s `k=depth*SEARCH_DOCUMENTS_OVERFETCH, limit=depth` and
  `search_chunks_fulltext`'s `limit=depth`, both lines 626-636) — no leftover bare
  `HYBRID_OVERFETCH_K` at a call site; grepped the whole document for every occurrence of the
  constant and confirmed each remaining one is either the module-level floor definition, an
  explanatory comment, or a test assertion at an input where `max(20, limit)` correctly evaluates
  to 20 (limit=3 or 5). §5.3 items 3/4 correctly split the check into two: item 3 pins the
  *formula* (call-argument depth at `limit=50`), item 4 pins the *symptom* end-to-end via
  `_RankedChunkRepo` seeded with 50 rows, asserting `len(rows) == 50` — verified by hand-tracing
  the fake's own `search_chunks(k=150, limit=50)` against a 50-row pool: no truncation, matches. At
  `limit=200` (REST's actual ceiling) the restored formula (`k = depth * SEARCH_DOCUMENTS_OVERFETCH`
  = 600) is numerically identical to the pre-fusion behavior at that same `limit` — this fix
  restores the original contract exactly, not a new approximation of it.
- **Minor (§3.6 `score is None` framing):** fixed, and fixed in both places I'd have wanted it —
  §3.6's prose (lines 249-260) and `_fuse_chunk_hits_rrf`'s own docstring (lines 524-531) both now
  state the two distinct cases (absent-from-vector vs. present-but-floor-failing-admitted-via-lexical)
  explicitly, matching the code's actual behavior.

**New (Minor) — the depth-scaling fix restores capacity correctness but the RRF/gate mechanism is
now unvalidated at any depth other than 20.** U2's own reasoning for `K=20` (`agent-knowledge-base-
strategy-ml.md` "Item 2" §1) is explicit that 20 "matches the depth this file's own DEF-1 diagnosis
... already probed to" and "gives RRF real room" at that specific depth — U2 never reasoned about
`RRF_K=60`'s behavior, or the rank-≤2 lexical gate's looseness, at a fused list depth of 50 or 200.
Today this is inert (the only real caller, `SKILL.md`'s convention, is pinned at `limit=5`, always
below the floor, so `depth` never exceeds 20 in practice) — but it is now a *reachable* code path
with no evaluation behind it, where before the fix it was simply unreachable at scale. Worth one
sentence in §6 naming this explicitly (in the spirit of the existing "U5 depends on this plan's
exact constant names/values landing unchanged" bullet): a future caller that legitimately needs
`limit > 20` is exercising RRF/the gate at a depth U2 never measured, and U5's regression run
(scoped to the 45-pair set at `limit=5`) will not catch a problem that only shows up there.

No other second-order issue found: the admissibility gate's two thresholds (`VECTOR_ADMISSIBILITY_
FLOOR`, `LEXICAL_ADMISSIBILITY_RANK`) are rank/threshold-based and depth-independent by construction
(verified by re-reading `admissible()`, unchanged by this revision); the vector side's `k` scaling
at large `depth` stays well under the known ~10,000-row FalkorDB result cap
(`claude/graph-dba/falkordb-quirks.md`, F2) even at REST's `le=200` ceiling; and `search_chunks_
fulltext`'s `limit=depth` at a large value is a plain `LIMIT` on an already-ranked RediSearch result
set, not a new correctness risk.

## Appendix A — `partition` vs `rpartition` on §5.1 item 3's literal string

```python
>>> MARKER = "\nQuery: "
>>> # As specified in the plan (double backslash in the embedded portion):
>>> q = "Instruct: ...\nQuery: does the situation text mention \"\\nQuery: \" literally?"
>>> q.count(MARKER)
1
>>> q.partition(MARKER)[2] == q.rpartition(MARKER)[2]
True   # test passes regardless of partition vs rpartition — inert

>>> # What it needs to actually be (real second newline, matching items 1/4's own style):
>>> q2 = "Instruct: ...\nQuery: does the situation text mention \"\nQuery: \" literally?"
>>> q2.count(MARKER)
2
>>> q2.partition(MARKER)[2]
'does the situation text mention "\nQuery: " literally?'
>>> q2.rpartition(MARKER)[2]
'" literally?'
>>> q2.partition(MARKER)[2] == q2.rpartition(MARKER)[2]
False  # now the test actually distinguishes the two methods
```
Both snippets run live against Python 3 during this review; outputs are copied verbatim, not
retyped from memory.
