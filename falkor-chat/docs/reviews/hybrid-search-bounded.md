# Review: bounded `Repository.hybrid_search` (U12)

> **Status:** active · **Owner:** `analyst` · **Tracks:** K-098

## Scope & verdict

**Reviewed:** the uncommitted U12 diff to `falkor-chat/server/falkorchat/repository.py`
(`Repository.hybrid_search`), `falkor-chat/server/tests/test_graphrag.py` (eight new tests) and
`falkor-chat/scripts/test_queries.sh` (three §6 hand copies). The baseline is
`docs/plans/graph-config-defaults-graph.md` §3.1, and I checked the evidence in
`docs/reviews/graph-config-defaults-evidence/u12/`. Production callers read: `Services.hybrid_search`
(`services.py:1194-1263`), then `tools.py:368` and `responder.py:104` above it. I applied root
`REVIEW.md` and `falkor-chat/REVIEW.md`. I ran no suites, per the brief: teco's solo run is the suite
evidence. My own executed evidence came from one throwaway container (`analyst-u12-probe`,
`falkordb/falkordb:v4.18.11`, `127.0.0.1:16401`), now removed. I did not touch `falkordb-dev`,
`ws:test` or `reference-test`.

**Verdict: needs changes.** There is one major finding (F1). The memory bound, which is the
unit's purpose, holds in both variants. The scoped variant, though, keeps a latency term that grows
linearly with the number of threads in a channel. That is the class of unbounded term §3.1 set out
to remove, and the new EXPLAIN test pins it as the intended plan. The scoped variant is the
production hot path, because `responder.py:104` passes `channel_id` on every @mention reply. A
one-clause change, which I ran, brings it to a flat ~1 ms. Everything else meets approval: the text
matches §3.1, the return shape is unchanged, the tests read the sent text, and both equivalence
claims hold on the shipped text.

CPG: considered, not relevant — the change is a single Cypher string plus tests, and the questions are about plans and execution in FalkorDB, which a Joern CPG of `falkor-chat` cannot answer.

## Findings

### F1 (major): the scoped variant's per-seed cost is O(threads in the channel), and the EXPLAIN test pins it

*Verified by execution* (Appendix A). The implementer's dispute of §3.1 is correct. When scoped,
FalkorDB folds `MATCH (t:Thread {threadId: seed.threadId})` and
`MATCH (c:Channel {channelId: $channelId})-[:HAS_THREAD]->(t)` into one pattern. It anchors on
`Node By Index Scan | (c:Channel)`, runs `Conditional Traverse | (c)->(t:Thread)` and then
`Filter`s on `threadId`. That happens for **every ANN seed**, including seeds outside the channel.
With 9–10 seeds and k=10, memory stays flat at 528 KiB. Latency does not:

| threads in channel | 1k | 10k | 100k | 1M |
|---|---|---|---|---|
| shipped scoped | 3 ms | 12.5 ms | 109 ms | **1,070 ms** |
| unscoped | 1 ms | 1 ms | 0.9 ms | 0.9 ms |

At k=10 this crosses `RAG_QUERY_TIMEOUT_MS` (5 s) at about 4.7M threads, and sooner at a larger k.
At k=20 and 1M threads it already exceeded the image's default 1 s `TIMEOUT` in my run. It is not
urgent on today's data, where `ws:demo` has about 12k messages in threads of 15 or fewer. Three
things still need fixing:
- (a) The repository docstring says "the seed's thread is found by its `threadId` index", which is
  false for the scoped variant.
- (b) `test_hybrid_search_finds_the_seed_thread_by_index_not_by_walk[c1]` asserts the
  `(c:Channel)` anchor, so a fix would turn it red.
- (c) §3.1's done-condition cannot catch this. `coverage.py` does not vary threads per channel, so
  "graph-dba confirms the bound" would pass.

**Suggested improvement** (a §3.1 text amendment, so it goes to `graph-dba` as plan owner, then
`tdd-engineer`). Put a `WITH` barrier after the anchor:
`MATCH (t:Thread {threadId: seed.threadId}) WITH seed, score, t MATCH (c:Channel {channelId: $channelId})-[:HAS_THREAD]->(t)`.
I ran it. It plans `Node By Index Scan | (t:Thread)` → `Conditional Traverse | (t)->(c:Channel)`,
runs in about 1 ms flat from 1k to 1M threads, and returns rows identical to the shipped text for
channel `c1` and for a decoy channel `c2`. The pattern predicate
`WHERE (t)<-[:HAS_THREAD]-(:Channel {channelId: $channelId})` behaves the same way. Folding the
anchor into the channel pattern (`…->(t:Thread {threadId: seed.threadId})`) does **not** help: it
still anchors on Channel and takes 1.19 s at 1M threads. What catches the fix being wrong:
- `test_hybrid_search_channel_scoped_excludes_other_channels`, which already kills the scope-loss
  mutant M10;
- the `[c1]` EXPLAIN parameter flipped to `Node By Index Scan | (t:Thread)`, plus an assertion that
  `Node By Index Scan | (c:Channel)` is absent.

Separately, `coverage.py` needs a threads-per-channel axis (1k → 1M) before the §3.1 done-condition
re-run. Without it, that re-run is a bound, not proof.

### F2 (minor): the DISTINCT in the subquery is redundant today, but becomes load-bearing as soon as `e` is referenced

*Verified by execution* (Appendix B). M4's equivalence holds on the shipped text. With `e`
unreferenced, the planner fuses the two hops into one
`Optional Conditional Traverse | (seed)->(related:Message)`. That collapses duplicate paths,
including parallel `MENTIONS` edges, which I tested. Once `e` is projected (`WITH related, e`) or
filtered (`WHERE e.entityId IS NOT NULL`), the plan splits into two traverses, and dropping
`DISTINCT` returns `['S','R','S','R']`. With `DISTINCT` kept, the filtered form still returns
`['S','R']`. K-093's contract decision (entity type or confidence filtering) is exactly that edit.
So `test_hybrid_search_lists_a_message_sharing_two_entities_once` pins the engine's fusion, not the
`DISTINCT`. It cannot go red on M4 today. **Suggested improvement:** add one clause to the docstring:
"`DISTINCT` is load-bearing once the subquery references `e`; the planner fuses the hops only while
`e` is unreferenced." K-093's writer should then re-run M4 against its own text. No new test is
needed now, because no test can distinguish M4 on this text.

### F3 (nit): stale caller comment

`services.py:1247-1252` still says the `Thread` match "holds for any message reachable via this
codebase's self-guarding HEAD/NEXT write paths". After U12 it holds for any message whose
`threadId` matches a `Thread`, which is the docstring's precondition. **Suggested improvement:**
change the clause to "any message carrying a `threadId` (every write path stamps it;
`backfill_thread_ids.sh` for older data)".

### Checked and not findings

- **Text against §3.1.** *Verified.* The method's text matches §3.1 clause for clause, both
  variants. Its only addition is the scoped `MATCH` line, which §3.1 marks as optional. The return
  is positional `row[0..4]` → `msgId/text/role/score/relatedContext`, unchanged. The shape
  `Services.hybrid_search` consumes is unchanged too (`{**row, "seedKind": …}`, sort on `score`).
- **The tests read the sent text.** *Verified by reading.* `_sent_hybrid_query` wraps `ro_query` at
  the graph call and unpacks exactly one query. The EXPLAIN and PROFILE tests run the captured
  `(cypher, params)`, not a copy. Each assertion has a named killer in `mutants.txt`, and I checked
  each against the test body:
  - M0, M1 and M12 turn the EXPLAIN or PROFILE tests red.
  - M2, M3 and M8 turn the cap and per-seed tests red.
  - M5, M6 and M7 turn the tie test red.
  - M14 turns the two-entity test red.
  - teco's `m.text→seed.text` mutant was killed by two tests.

  The PROFILE test asserts `Records produced ≤ 5` with one embedded seed and 40 mentioners, and M1
  shows that bound is not vacuous.
- **M13 is equivalent.** *Verified by execution.* Both variants produce identical plans and
  identical rows.
- **`test_queries.sh` hand copies.** *Verified.* I evaluated each copy with bash, normalized the
  literals (`2`→`$k`, the vector→`$qVec`, `'ch1'`→`$channelId`, `LIMIT 5`→`$limit`), and diffed it
  against the captured method text. All three (lines 422, 435, 440) are identical up to the outer
  `RETURN`. The projections differ (missing `AS text`/`AS role`, dropped columns), as they did at
  HEAD. The line-440 change *adds* `MATCH (t:Thread {threadId: seed.threadId})`. That fixes an
  older divergence: HEAD's workspace-wide copy omitted the thread match the method always had.
- **Sweep.** I swept with an unfiltered `git grep` for the clauses
  `MENTIONS]->(e:Entity)` and `HEAD|NEXT*0..`. It found no other code copy. The remaining hits are
  in the live spec (QUERIES.md:476/496-498, DESIGN §8), which belongs to the feature-close writer
  per the ledger. Everything else is frozen evidence or archive.
- **Evidence directory.** *Verified.* `u12/mut/` is absent and nothing in the directory is a
  production copy or a secret. One minor point: `m4_distinct_probe.py` runs
  `MATCH (n) DETACH DELETE n` on `ws:test` against the default host. If someone re-runs it during a
  suite, it wipes the suite's graph. Consider a header warning, since the committed file is an
  invitation to re-run it.

## What's solid

- The bounded subquery, the per-seed scoping and the `msgId` tie-break are correct. Each is pinned
  by a test over the real method that a named mutant turns red.
- The implementer found the §3.1 plan error instead of copying the plan's claim into a test.
- The memory proof is at full fan-out (10 seeds on a 100k hub, plus 100k depth), per
  `falkor-chat/REVIEW.md`, and the old text fails it at 64 MiB. The mutation harness proves it
  imported the mutated copy.

## Open questions

- F1's fix changes §3.1's verbatim text. Should `graph-dba` amend §3.1 and U12 be re-opened? Or
  does the scoped latency term get accepted and recorded in DESIGN §8 at close, with a reversal
  trigger (threads per channel ≳ 100k)? I recommend the amendment: it is one clause, it is verified,
  and the scoped variant is the responder's path.

### Proposed REVIEW.md lines

Target `falkor-chat/REVIEW.md`, *Always check*. This is an addition, not a narrowing. The source is
F1: §3.1 asserted a plan for one variant, and the coverage axes did not include the term the other
variant's real plan scales with.

- `When a query has variants (scoped/unscoped, optional clauses), EXPLAIN each variant's sent text and derive its cost terms from that plan; the bound's coverage probe varies every cardinality the plan's anchor and traversals scan, not only the axes the design named.`

## Appendix

**A. Scoped plan and scaling** (scratch probe on `:16401`). Schema: `bootstrap_schema.sh probe`,
with `EMBEDDING_DIM=4` and `FALKORCHAT_REFERENCE_GRAPH=probe-ref`. Fixture: channel `c1` with N
single-message threads, 10 near-identical embedded seeds spread across them, and 5 decoy seeds in
`c2`. The text was captured from the real method. Each latency is the mean of 3 runs at
`QUERY_MEM_CAPACITY` 64 MiB, and each min cap was found by bisection.
Shipped scoped plan: `Results / Limit / Sort / Project / Apply / Filter / Conditional Traverse | (c)->(t:Thread) / Node By Index Scan | (c:Channel) / ProcedureCall / Aggregate / Limit / Distinct / Project / Optional Conditional Traverse | (seed)->(related:Message) / …`.
At N=1M: shipped 1,067–1,081 ms. Single-pattern form 1,180–1,204 ms (Channel anchor). WITH barrier
1.0–1.1 ms (Thread anchor). Pattern predicate 1.1–1.3 ms (Thread anchor). For `c1` and `c2`, both
fixes returned the same msgId set as the shipped text, and the unscoped set mixed `c1` and `c2`
seeds, so the scope filter is live. The min cap was 528 KiB scoped and 512 KiB unscoped at every N.

**B. M4 probe.** Fixture: seed `s` mentions `e1` twice (parallel edges) and `e2` once. `r` mentions
`e1` once and `e2` twice. Real method text, unscoped.

| variant | relatedContext | plan traverse |
|---|---|---|
| shipped | `['S','R']` | fused `(seed)->(related)` |
| M4 (no DISTINCT) | `['S','R']` | fused |
| M4 + named rel var | `['S','R']` | fused |
| M4 + `WITH related, e` | `['S','R','S','R']` | split `(seed)->(e)`, `(e)->(related)` |
| M4 + `WHERE e.entityId IS NOT NULL` | `['S','R','S','R']` | split |
| DISTINCT kept + same WHERE | `['S','R']` | split |

## Pass 2

Re-gate on the revised tree (still uncommitted). It covers `repository.py`, `services.py`,
`test_graphrag.py`, `test_queries.sh`, the note's §3.1 revision and the new evidence
(`u12/f1_barrier_probe.*`, `u12/mutants.txt` M15, `graph-dba-r4/`). I verified it by execution on
a throwaway container (`analyst-u12-p2`, `:16402`), now removed. I ran no suites.

- **F1: fixed.** The sent scoped text now carries `WITH seed, score, t`. Its EXPLAIN reads
  `Filter / Conditional Traverse | (t)->(c:Channel) / Project / Node By Index Scan | (t:Thread)`.
  With the barrier stripped (M15), it reads `Node By Index Scan | (c:Channel)`. That fails both
  of the revised `[c1]` assertions, which agrees with `mutants.txt` (M15 killed by
  `[c1]` only). Pass 1 already measured this exact string ("anchor then WITH barrier") at 1.0–1.1
  ms with 1M threads per channel, with rows unchanged. `coverage.py`'s `v3` form is the
  sent text exactly, in both variants, so the done-condition's re-run measured the shipped text.
- **F2: fixed.** The docstring, the note's "Why each part" and Pass 1's Appendix B agree that
  `DISTINCT` is redundant while `e` is unreferenced and load-bearing once `e` is referenced. Gap:
  P2-m1.
- **F3: fixed.** Both `services.py` sites (`:1248`, `:1516`) now state the `threadId`
  precondition.
- **Evidence nit: fixed.** `m4_distinct_probe.py` now refuses to run without a port and writes only
  `ws:u12-m4-probe`. `u12/mut/` is still absent.
- **Sent text against §3.1:** identical in both variants, compared as the whitespace-joined §3.1
  block (the optional bracket included or dropped) against the captured text.
- **Docstring plan claims against EXPLAIN:** they match in both variants.
- **`test_queries.sh`:** I re-evaluated the copies with bash. Lines 422, 435 and 440 are identical to
  the sent text up to the outer `RETURN`, after literal normalization, as in Pass 1.

**Growth-stops check (note §3.1, +35/−14 lines).**
- *New scope:* the threads-per-channel coverage axis (plus its bullet and evidence citation), and
  the new done-condition assertion.
- *Rework:* the corrected plan claims, the barrier and `DISTINCT` reasons, and the 576/640 KiB
  line. These replace the old text in place. The rejected-alternative latency is stated once, as a
  design reason.
- *Not clean:* P2-m2.

### P2-m1 (minor): the K-093 obligation is recorded only in a document that will freeze

The note says "K-093's writer re-runs the drop-`DISTINCT` mutant against its own text", but the
K-093 item (`docs/BACKLOG.md:1033`) does not carry it. Its done-condition covers only
`relatedContext` and the spec sections. Once the note is `archived` at close, nothing on K-093's
path surfaces the re-run. **Suggested improvement:** `teco` adds one clause to K-093's
done-condition: "if the expansion's subquery references `e`, re-run the drop-`DISTINCT` mutant
(`docs/plans/graph-config-defaults-graph.md` §3.1)". *Verified by reading K-093's full body.*

### P2-m2 (minor): the note states the no-barrier latency twice, with two values

"Why each part" gives "1,253 ms at 1M threads per channel, against 2.1 ms". The done-condition
gives "1,267 ms for the text without the barrier" and "1.5–2.5 ms". These are one fact with two
values, from two runs. The done-condition has also become a results record. **Suggested
improvement:** cut the done-condition back to its checks (the EXPLAIN assertion plus the
`coverage.py` re-run on the `v3` form, citing `graph-dba-r4/coverage_summary.txt`), and keep the
numbers only in "Why each part". *Verified by reading both sentences.* A nit for the same writer:
one docstring line in `repository.hybrid_search` (`…(e.g. a K-093 filter). Precondition: …`) runs
past the file's wrap width.

**Pass 2 verdict: approve with suggestions.** The blocking concern (F1) is fixed and verified by
execution. P2-m1 and P2-m2 are documentation edits that can land with the close and do not touch
the code. The Pass 1 `REVIEW.md` proposal still stands, verbatim:

- `When a query has variants (scoped/unscoped, optional clauses), EXPLAIN each variant's sent text and derive its cost terms from that plan; the bound's coverage probe varies every cardinality the plan's anchor and traversals scan, not only the axes the design named.`

Target `falkor-chat/REVIEW.md`, *Always check*. It is an addition, not a narrowing, and its source
is F1.
