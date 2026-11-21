# Message entity mentions — drop the co-occurrence expansion

> **Status:** active · **Owner:** `architect` · **Tracks:** K-093 · **Version:** 2

2026-11-21: v2. The stakeholder picked option B (drop). The plan gate's findings are folded in (`docs/reviews/message-entity-mentions.md`).

## 1. Premise

**Nothing writes `(:Message)-[:MENTIONS]->(:Entity)`.** I ran an unfiltered `grep -rn MENTIONS`,
excluding `.git`, `.venv` and `node_modules`. In `server/`, `scripts/` and `web/`, the only
`[:MENTIONS]` texts are these:
- the read in `Repository.hybrid_search`;
- its three hand copies in `scripts/test_queries.sh` §6;
- the test-only raw `CREATE` in `test_graphrag.py` `_mention`.

Every other hit is `MENTIONS_MEMBER`, kaizen prose, or `kiro/DESIGN.md`'s unrelated draft
`Message→Agent` edge. Live, `MATCH ()-[r:MENTIONS]->() RETURN count(r)` is `0` in all seven `ws:*`
graphs. The edge has no DDL.

**Nothing reads `relatedContext`.** Each consumer drops it:
- `AgentResponder._build_prompt` uses only `s['text']`.
- `GraphragRetrieveTool.run` projects `seedId`/`text`/`score`/`documentId`.
- `mcp.py` and `api.py` expose no hybrid search.
- `web/app.js` searches through full-text `/search`.
- `kiro/` and `salesperson/` have zero hits.

So the drop breaks no MCP or REST contract.

ccg impact for `Services.hybrid_search`, verbatim stamp and notices:

> Snapshot 483767a48aecd809219aa738f6c1f8ca, verdict log offset 0. … 862 items listed (2 confirmed,
> 860 possible). Flow-dependent effects in python may be missing from this list. In python the walk
> did not follow data, return-value, argument or control-flow edges; it followed call, link and
> wiring edges. An item's absence here is not evidence that no such effect exists. Reach stopped at
> distance 2 with 60218 unexplored edges.

Production callers at distance 1: `AgentResponder.maybe_respond`, `GraphragRetrieveTool.run`.
The two confirmed items are eval-harness functions that do not read the field. The walk follows no data flow, so the field-level claim rests on the grep.

## 2. Decision and reversal trigger

**Drop the expansion, the `relatedContext` field, and the dormant edge from the schema and the
spec.** I rejected the alternatives for these reasons.

- **Building the writer (A).** It costs one local `extraction`-kind LLM call per posted message on
  the single, already contended GPU. It is also structurally unsound with today's entity model:
  extraction makes every mention a fresh `Entity` (DESIGN §9 "Extract entities"), so each entity
  would carry one `MENTIONS` edge, and the read could only ever return the seed itself.
- **Keeping the field as a hard-coded `[]`.** It needs the same edits as B and preserves only a key
  that no reader consumes.

The reason is dead surface in the spec and the tests, not the no-op's runtime cost, which is about
2 ms (`docs/plans/graph-config-defaults-graph.md` §3.1).

**Where the reversal trigger lives.** DESIGN §1.2 gets a new row. It is the register
`falkor-chat/AGENTS.md` names, and it has a precedent in the "No DayBucket *(rejected alternative)*" row.

| Decision | Rationale / consequence | Detailed in |
|---|---|---|
| **No `Message→Entity` co-occurrence expansion** *(rejected alternative)* | No consumer used per-seed related messages. Under fresh-entity-per-mention extraction the 2-hop read can only return the seed. Reopen when a requirements document or backlog item names a consumer of per-seed related-message context (the responder prompt, a tool output schema, or a UI). Rebuild from a canonical-entity `MENTIONS` target, never from the dropped read. | §8 |

The trigger is an observable event: a document gets written. It is not an eval margin, because
measuring a margin requires building the writer first, so it could not fire before most of the cost
is paid. Once the trigger fires, the `-ml` note for that feature sets the acceptance margin.

## 3. Build unit (`tdd-engineer`)

### 3.1 Target query

Scoped variant. The unscoped variant drops only the `WITH seed, score, t` + `MATCH (c:Channel …)`
pair. The parameters do not change.

```cypher
CALL db.idx.vector.queryNodes('Message', 'embedding', $k, vecf32($qVec))
YIELD node AS seed, score
MATCH (t:Thread {threadId: seed.threadId})
WITH seed, score, t
MATCH (c:Channel {channelId: $channelId})-[:HAS_THREAD]->(t)
RETURN seed.msgId AS msgId, seed.text AS text, seed.role AS role, score
ORDER BY score ASC, msgId ASC
LIMIT $limit
```

```cypher
CALL db.idx.vector.queryNodes('Message', 'embedding', $k, vecf32($qVec))
YIELD node AS seed, score
MATCH (t:Thread {threadId: seed.threadId})
RETURN seed.msgId AS msgId, seed.text AS text, seed.role AS role, score
ORDER BY score ASC, msgId ASC
LIMIT $limit
```

The plan gate EXPLAINed both texts on `ws:acme`. Scoped:
`Node By Index Scan | (t:Thread)` → `Conditional Traverse | (t)->(c:Channel)`. Unscoped:
`Node By Index Scan | (t:Thread)`. So the F1 barrier still holds (`docs/reviews/hybrid-search-bounded.md`).

### 3.2 Red-first pins (done-condition)

Write these first. Each must fail at HEAD.

1. `test_graphrag.py`: the rows of `Repository.hybrid_search` have exactly the keys
   `{msgId, text, role, score}`, parametrized over `channel_id ∈ {None, "c1"}`. This replaces
   `test_hybrid_search_related_context_is_empty_list`.
2. `test_graphrag.py`: `"MENTIONS" not in` the text sent, captured through `_sent_hybrid_query`,
   in both variants.
3. `test_services.py`: `"relatedContext" not in` the merged Chunk row of `Services.hybrid_search`.
   This replaces the assertion under the "same dormant `relatedContext:[]` convention" comment.

Mutants, each of which must turn a pin red:
- restoring the `CALL { … }` subquery breaks pins 1 and 2;
- restoring the Chunk-row `[]` stamp breaks pin 3.

### 3.3 Edits

- **`Repository.hybrid_search`.**
  - Code:
    - send §3.1's text;
    - the returned row dict has four keys.
  - Docstring:
    - drop the `relatedContext`/`MENTIONS_MEMBER` sentence;
    - drop the `CALL { … LIMIT 5 }`/`DISTINCT` paragraph;
    - drop "The seed itself counts among its own related messages";
    - keep the thread-by-index and barrier text.
- **`Repository.search_chunks` docstring.** Drop the "no Entity co-occurrence expansion (`ABOUT` is
  dormant …)" clause. It has nothing left to contrast with, and `ABOUT` is no longer dormant.
- **`Services.hybrid_search`.** Drop the `"relatedContext": []` stamp on Chunk rows. In the
  docstring, drop "Entity co-occurrence expansion", the FR-2 `relatedContext` parenthesis, and the
  "dormant-field convention" sentence.
- **`server/tests/test_graphrag.py`.**
  - Delete these tests:
    - `…caps_related_context_at_five_from_a_hub`
    - `…expands_each_seed_from_its_own_entities`
    - `…lists_a_message_sharing_two_entities_once`
    - `…stops_traversing_a_hub_after_five_per_seed`
    - `…related_context_is_empty_list` (superseded by pin 1)
  - Delete the `MENTIONS` fixture comment block and these helpers:
    - `_mention`
    - `_seed_hub_thread`
    - `_seed_thread_partly_embedded`
    - `_plan_ops`
    - `_records_produced`
  - **Rebase `test_hybrid_search_finds_the_seed_thread_by_index_not_by_walk`.** It builds its
    fixture through `_seed_hub_thread`, which calls `_mention`. Replace that call with
    `_seed_thread(repo, channel_id="c1", thread_id="t1", messages=[("a-seed", "a seed", _pad([1.0]))])`.
    The test EXPLAINs plan text, not data, so the dropped follower and edge do not matter. Its four
    plan assertions stay as they are. It remains the F1 guard.
- **`server/tests/test_services.py`, `test_tools.py`, `test_responder.py`.** Drop the
  `relatedContext` keys from fixture rows.
- **`scripts/test_queries.sh` §6.**
  - Remove `CALL { … }` from the three query copies, and `, [m IN expanded | m.text] AS relatedContext`
    from the two that carry it. Each copy keeps its own literals.
  - Delete the "Entity expansion no-ops" assertion.
  - Delete the `MATCH (e:Entity) RETURN count(e)` probe and its "Entity graph is empty" assertion.
    That assertion can never fail: it matches the `0` in the output footer.
  - Delete the comments that rest on the false "no Entity pipeline in M2" premise: the §6 section
    header comment, "+ Entity expansion" in the full-retrieval comment, and the no-op comment.

**Gates.**
1. The pins are red, then green.
2. `cd server && .venv/bin/python -m pytest -q`.
3. `./scripts/test_queries.sh`.
4. `analyst` impl review.

## 4. Close list (live spec, at feature close)

I derived this from an unfiltered grep over DESIGN, QUERIES, SERVER and the context files for
`MENTIONS[^_]`, `co-occurrence`, `co-mention`, `expansion`, `related messages` and `relatedContext`.

- **DESIGN §1.2**: add the §2 row.
- **DESIGN §5.1**: delete the `(:Message)-[:MENTIONS]->(:Entity)` line and its "dormant schema"
  comment header. The edge is the only entry under that header.
- **DESIGN §5.2**, `Message.embedding` bullet: drop "and entity co-occurrence".
- **DESIGN §8**, message-read bullet: drop the expansion clauses.
- **QUERIES §4**, `MENTIONS_MEMBER` bullet: drop the `MENTIONS`→`Entity` contrast.
- **QUERIES §6**:
  - replace the block with §3.1;
  - delete the "Expansion bounded", "`DISTINCT`" and "Dormant" bullets.
- **QUERIES §10**, `EMITTED` bullet: drop the `MENTIONS`→`Entity` contrast.
- **QUERIES §14.3**: drop "and no Entity expansion".

These stay unchanged:
- QUERIES §14's `ABOUT` "co-occurrence edge" sentence, and the matching
  `Repository.link_chunk_about_entity` docstring. Both are about `ABOUT`.
- SERVER.md, which is clean.
- DESIGN §5.4.

Then the `analyst` close check, the `HISTORY.md` entry, and the removal of K-093 from `BACKLOG.md`.
