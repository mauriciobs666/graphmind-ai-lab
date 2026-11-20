# Message quoting — write `REPLY_TO` from the post front doors

> **Status:** active · **Owner:** `architect` · **Tracks:** K-094 · **Reviews:** `falkor-chat/docs/reviews/message-quoting.md`

## 1. Goal & scope

Add an optional quoted-message id to the two message post front doors. Validate it inside the
§4 write and write it as `(m)-[:REPLY_TO]->(quoted)`, so that `get_message`'s `quotedId` is
written by a front door and read back. This is K-094's done-condition, through the stakeholder's
"add quoting" decision of 2026-11-20 (`falkor-chat/docs/plans/live-spec-defects-coordination.md`).

**In scope:** REST `POST /threads/{tid}/messages` and MCP `send_message`, through
`Services.post_message`, down to `Repository.post_first_message` and
`Repository.post_subsequent_message`.

**Out of scope:**
- the web UI;
- the §10.1 agent-answer writes (`Repository.post_agent_answer*`, `Services.post_agent_answer`),
  which keep their 4-column status row;
- `quotedId` on any read other than `get_message` (§3.5);
- an MCP `get_message` tool.

## 2. Context & findings

- **There is a reader but no writer.** `Repository.get_message` reads
  `OPTIONAL MATCH (m)-[:REPLY_TO]->(quoted:Message)`. Only the suite's QUERIES §4 "Post a reply"
  library case writes the edge. DESIGN §5.1 lists it as dormant schema.
- **The write paths** are the QUERIES §4 v2 bodies, with dispatch as in DESIGN §5.3.
  `Repository._write_status` builds `MessageWriteStatus` positionally from the row, and the
  §10.1 pair goes through the same helper. `Services._dispatch_write` is shared by
  `post_message` and `post_agent_answer`.
- **Front doors.** REST: `api.post_message` reads `schemas.PostMessageIn`. MCP:
  `mcp.send_message`. A third caller, `storefront_api.post_message`, has its own
  `PostMessageIn` and returns `services.post_message`'s `posted` dict unchanged, so its response
  grows too (§3.4).
- **Error envelope.** `app._register_error_handlers` returns `400` for a `ServiceError` outside
  its not-found tuple. On MCP the error is an `isError` result,
  `Error executing tool <tool>: <message>` (SERVER §2.2).
- **Bound parity.** `tests/test_input_bound_parity.py` needs one `MCP_ARGS` entry per MCP
  argument. A REST field needs a `MAPPING` entry only if it is constrained. SERVER §2.2 leaves id
  arguments undeclared because they are only looked up.
- **A null parameter defeats the index.** On an indexed label, `(n:Label {prop: $p})` with
  `$p = NULL` plans `Node By Label Scan` (`claude/graph-dba/falkordb-quirks.md`, "A bare
  pattern-property equality match …"). I measured this for
  `(quoted:Message {msgId: $quotedId})`, and so did the review, at 201 records on a 201-message
  graph: every post without a quote would scan all messages. Rebinding through
  `UNWIND [$quotedId] AS qid` keeps `Node By Index Scan | (quoted:Message)` for both null and a
  real id (§4).
- **`Message.threadId`** is written inline and is unindexed. Pre-K-007 messages have it only after
  `scripts/backfill_thread_ids.sh` has run.

CPG: considered, not relevant — a small change at seams located by symbol in a handful of files; `repository.py` is changing under another unit, so a snapshot graph would be stale (the review used `cpg_falkorchat` to confirm the three `post_message` callers).

## 3. Design

### 3.1 Validation scope: same thread

A quote is valid only when `quoted.threadId = $threadId`, and the check runs inside the write.
The same-workspace check comes for free: the write runs on `ws:{id}`, and an edge cannot cross
graphs. I chose same-thread because:

- it can be relaxed later to same-workspace without breaking anyone; tightening it later could
  not be done that way;
- `REPLY_TO` means a reply inside one conversation, and a cross-thread reference is a citation,
  which is a different feature;
- the storefront participant reset deletes whole threads (DESIGN §9), so with this scope no quote
  is ever left dangling.

### 3.2 The check runs in-query and reports a status column

The quoted message is resolved and checked in the **same `GRAPH.QUERY`** as the write:

- a fifth status column, `quotedFound`, is `true` when no quote was asked for or when the id
  resolves in this thread;
- `ok` gains `AND quoteOk`;
- the `REPLY_TO` create is a nested `FOREACH` inside the guard, like `MENTIONS_MEMBER`.

I rejected a service pre-read that raises before the write. It would cost an extra round trip
per quoting post. It would leave a window between the check and the write. And the write must
check anyway to stay self-guarding. The existing pre-checks (`thread_exists`,
`resolve_member_kinds`) stay, because they name things the status row cannot report.

**Resolution idiom:** `UNWIND [$quotedId] AS qid`, then
`OPTIONAL MATCH (quoted:Message {msgId: qid}) WHERE quoted.threadId = $threadId`.

- A one-element list yields exactly one row, even `[null]`, so no `CASE` guard is needed.
- The variable binding keeps the index scan (§2).
- The repository always sends `quotedId`, as `None` when there is no quote; leaving it out
  raises `Missing parameters`.

I also rejected the quirk file's other fix, two literal texts per path, because it would double
the §4 bodies that can drift apart.

`OPTIONAL MATCH … WHERE` folds "no such msgId" and "msgId in another thread" into
`quoted IS NULL`. Both get one error, and the caller fixes both the same way.

### 3.3 Status-row contract, dispatch and error

| Result | Service action |
|---|---|
| `quotedFound=false`, `hadHead=false`, not `dupMsg` (id missing, in another thread, or the thread is headless) | raise `UnknownQuotedMessageError`. REST answers `400`, MCP returns a tool error, and nothing is written. |
| `quotedFound=false`, `hadHead=true` (first path) | Lost the first-post race. Re-dispatch as subsequent, which then decides the quote. |
| `dupMsg=true` | Idempotent success. `quotedFound` is not consulted. |

The order of checks in `_dispatch_write` is: `written or dup_msg`, then `not author_found`, then
`had_head`, then `not quoted_found`, then `RuntimeError`.

`UnknownQuotedMessageError(ServiceError)` sits beside `UnknownMemberError` and maps to `400` as
the generic case, with no `app.py` change. It is a `400`, like an unresolvable mention, because
the id comes from the request body and is not the path resource. Its detail string is **exact**,
because it is model-read and the tests assert it literally:

```python
f"quoted message {quoted_id!r} is not a message in thread {thread_id!r}; "
"nothing was posted. Retry without the quote, or quote the msgId of a message "
"read from this thread."
```

REST and MCP share this one detail, so it names neither `quotedId` nor `quoted_id`.

`MessageWriteStatus` gains `quoted_found: bool = True` as its last field. The default keeps
three things valid: the §10.1 4-column rows, every existing construction, and the positional
`_write_status`.

### 3.4 What callers pass and get back

- **REST:** `PostMessageIn.quotedId: str | None = None`. It has no `Field` bound, since the value
  is looked up, never stored. `MAPPING` is unchanged.
- **MCP:** `send_message(body, re, mentions=None, frm=None, quoted_id=None)`, with the new
  argument last. `MCP_ARGS` gains
  `("send_message", "quoted_id"): NoBound("id argument, only looked up; REST body.quotedId is unbounded too")`.
  The docstring gains this **exact** paragraph (model-read; asserted literally after whitespace
  normalization):

  > Omit `quoted_id` unless you hold the exact `msgId` of a message in thread `re`, copied from a
  > `read_messages(re=...)` row; never construct or guess one. A `quoted_id` that is not a
  > message in this thread is an error and nothing is posted.

- **Service:** `Services.post_message(ctx, *, thread_id, text, mentions=None, quoted_id=None)`
  passes `quoted_id` through `write_kwargs`.
- **Response:** `posted` gains `"quotedId": quoted_id`, which is `None` when there is no quote.
  The new key appears on REST, MCP and the storefront `POST /shop/api/messages`, where it is
  always `null`. Its consumers (`background._safe_run_workflow`, `trigger.maybe_trigger`, the
  embed scheduling) read keys by name, so the extra key does not affect them. The SPA's
  `postMessage` is a TypeScript cast with no runtime validation.

### 3.5 Reads: `get_message` only, for now

`read_thread` and the §9.1/§9.2 since-reads stay as they are. They are the hot reads, and adding
a field changes their row contract. The stakeholder scoped this change to the write side, and the
UI change that renders quotes needs `read_thread` anyway, so that change can add the field to all
three reads with one profiling pass.

MCP agents can therefore quote but cannot read a quote back. At close, `teco` files a follow-up:
"surface `quotedId` on `read_thread` and §9 reads (+ MCP read-back)".

### 3.6 RAM (rule 6)

Each quoting message gets at most **one** `REPLY_TO` edge (measured `max(count) = 1`). The edge
has no properties, no index and no constraint. A post without a quote adds no data. The only
added read is one `Message.msgId` index lookup, which returns 0 or 1 records.

### 3.7 The two write variants (canonical text for QUERIES §4)

Each variant is the current §4 body plus four additions:

- the `UNWIND`/`OPTIONAL MATCH` pair;
- `quoted` and `quoteOk` carried through the `WITH` chain;
- the nested `REPLY_TO` `FOREACH`;
- the `quotedFound` column.

The parameters are those of §4 plus `$quotedId` (a string or `null`).

**First message (`post_first_message`):**

```cypher
MATCH (t:Thread {threadId: $threadId})
OPTIONAL MATCH (t)-[:HEAD]->(h)
OPTIONAL MATCH (dup:Message {msgId: $msgId})
OPTIONAL MATCH (ua:User  {userId:  $authorId})
OPTIONAL MATCH (aa:Agent {agentId: $authorId})
UNWIND [$quotedId] AS qid
OPTIONAL MATCH (quoted:Message {msgId: qid}) WHERE quoted.threadId = $threadId
WITH t, h, dup, coalesce(ua, aa) AS author, quoted
UNWIND (CASE WHEN $mentions = [] THEN [null] ELSE $mentions END) AS mid
OPTIONAL MATCH (mu:User  {userId:  mid})
OPTIONAL MATCH (ma:Agent {agentId: mid})
WITH t, h, dup, author, quoted, collect(DISTINCT coalesce(mu, ma)) AS mems
WITH t, h, dup, author, quoted, mems,
     ($quotedId IS NULL OR quoted IS NOT NULL) AS quoteOk
WITH t, h, dup, author, quoted, mems, quoteOk,
     (h IS NULL AND dup IS NULL AND author IS NOT NULL AND quoteOk) AS ok
FOREACH (_ IN CASE WHEN ok THEN [1] ELSE [] END |
  CREATE (m:Message {msgId: $msgId, text: $text, role: $role,
                     createdAt: $createdAt, threadId: $threadId})
  CREATE (t)-[:HEAD]->(m)
  CREATE (t)-[:TAIL]->(m)
  CREATE (m)-[:POSTED_BY]->(author)
  SET t.updatedAt = $createdAt
  FOREACH (mem IN mems | CREATE (m)-[:MENTIONS_MEMBER]->(mem))
  FOREACH (q IN CASE WHEN quoted IS NULL THEN [] ELSE [quoted] END |
    CREATE (m)-[:REPLY_TO]->(q))
)
RETURN ok                 AS written,
       h    IS NOT NULL   AS hadHead,
       dup  IS NOT NULL   AS dupMsg,
       author IS NOT NULL AS authorFound,
       quoteOk            AS quotedFound
```

**Subsequent message (`post_subsequent_message`):**

```cypher
MATCH (t:Thread {threadId: $threadId})-[tailRel:TAIL]->(prev:Message)
OPTIONAL MATCH (dup:Message {msgId: $msgId})
OPTIONAL MATCH (ua:User  {userId:  $authorId})
OPTIONAL MATCH (aa:Agent {agentId: $authorId})
UNWIND [$quotedId] AS qid
OPTIONAL MATCH (quoted:Message {msgId: qid}) WHERE quoted.threadId = $threadId
WITH t, tailRel, prev, dup, coalesce(ua, aa) AS author, quoted
UNWIND (CASE WHEN $mentions = [] THEN [null] ELSE $mentions END) AS mid
OPTIONAL MATCH (mu:User  {userId:  mid})
OPTIONAL MATCH (ma:Agent {agentId: mid})
WITH t, tailRel, prev, dup, author, quoted, collect(DISTINCT coalesce(mu, ma)) AS mems
WITH t, tailRel, prev, dup, author, quoted, mems,
     ($quotedId IS NULL OR quoted IS NOT NULL) AS quoteOk
WITH t, tailRel, prev, dup, author, quoted, mems, quoteOk,
     (dup IS NULL AND author IS NOT NULL AND quoteOk) AS ok
FOREACH (_ IN CASE WHEN ok THEN [1] ELSE [] END |
  CREATE (m:Message {msgId: $msgId, text: $text, role: $role,
                     createdAt: $createdAt, threadId: $threadId})
  CREATE (prev)-[:NEXT]->(m)
  DELETE tailRel
  CREATE (t)-[:TAIL]->(m)
  CREATE (m)-[:POSTED_BY]->(author)
  SET t.updatedAt = $createdAt
  FOREACH (mem IN mems | CREATE (m)-[:MENTIONS_MEMBER]->(mem))
  FOREACH (q IN CASE WHEN quoted IS NULL THEN [] ELSE [quoted] END |
    CREATE (m)-[:REPLY_TO]->(q))
)
RETURN ok                 AS written,
       false              AS hadHead,
       dup  IS NOT NULL   AS dupMsg,
       author IS NOT NULL AS authorFound,
       quoteOk            AS quotedFound
```

A `CREATE` endpoint must be a bound node variable (QUERIES §10.1), and `q` is one.

## 4. Live verification (2026-11-20)

I ran both texts verbatim, with `CYPHER`-prefix parameters, on `falkordb-dev` (v4.18.11), in a
throwaway `ws:probe-k094`. It had the four anchor indexes and the `Message.msgId` constraint, and
was `GRAPH.DELETE`d afterwards. `ws:test` and `reference-test` were not touched. The review
ran an overlapping set of cases on its own throwaway key, and the rows agree where the
two sets overlap (its Appendix A).

Rows are `written hadHead dupMsg authorFound quotedFound`, with `T`/`F` for true/false:

| Case | Path | Row |
|---|---|---|
| no quote, empty thread / no quote + mention | first / subsequent | `T F F T T`, no `REPLY_TO` |
| quote m1, same thread | subsequent | `T F F T T`, `m3→m1` |
| unknown id; `''`; other-thread id | subsequent | `F F F T F`, nothing written |
| headless thread, quote other-thread or unknown | first | `F F F T F`, nothing written |
| race (headed thread), valid / unknown quote | first | `F T F T T` / `F T F T F` |
| replay m3 with its quote | subsequent | `F F T T T`, no second edge |
| unknown author, valid quote | subsequent | `F F F F T` |
| quote m2 + mentions `['a1','u1','nope']` | subsequent | `T F F T T`, `m10→m2`, 2 mention edges |

After the refusals, the chain and TAIL were intact. `REPLY_TO` was exactly `m3→m1` and `m10→m2`.
`get_message(m3)` returned `quotedId = m1`. `GRAPH.PROFILE` showed every anchor as a
`Node By Index Scan`, including `(quoted:Message)`, on both paths, for null (0 records) and for a
real id (1 record).

## 5. Implementation steps (for `tdd-engineer`)

Each step goes red, then green, with the suite green at the end of the step. The unit **lands
after the concurrent K-091 unit's commit, or rebases on it** (§7). Re-locate every symbol by
name before editing.

| # | Seam | Files · symbols | Done when |
|---|---|---|---|
| 1 | Repository write | `server/falkorchat/repository.py`: `MessageWriteStatus.quoted_found: bool = True`. `post_first_message`/`post_subsequent_message` gain `quoted_id: str \| None = None`, always send `"quotedId"`, and take the §3.7 texts verbatim. The `# ── §4 Messages` comment gives the `UNWIND [$quotedId]` index reason. The `post_agent_answer` docstring's "Same status-row contract as §4" becomes "§4's first four columns (no `quotedFound`)". `scripts/test_queries.sh`: the 12 §4 copies of §6.2, the M1 index guard, and the new cases. | §6.1 and §6.2 pass. The §10.1 query texts are untouched. |
| 2 | Service dispatch | `server/falkorchat/services.py`: `UnknownQuotedMessageError(ServiceError)` with the §3.3 detail. `Services.post_message` gains `quoted_id`, puts it in `write_kwargs`, and returns `"quotedId"`. `Services._dispatch_write` raises after the `had_head` branch when `not st.quoted_found`, taking the id from `write_kwargs.get("quoted_id")`. | §6.3 passes, and `post_agent_answer` behaves as before. |
| 3 | Front doors | `server/falkorchat/schemas.py` `PostMessageIn.quotedId`. `server/falkorchat/api.py` `post_message` passes `quoted_id=body.quotedId`. `server/falkorchat/mcp.py` `send_message` gains `quoted_id` plus the §3.4 docstring paragraph. `server/tests/test_input_bound_parity.py` gets the `MCP_ARGS` entry. | §6.4 passes. |

## 6. Test strategy

Write the failing tests first, in this order. §6.1 and §6.2 need FalkorDB (`ws:test`).

### 6.1 Repository (`server/tests/test_repository.py`, `repo` fixture)

1. `test_post_subsequent_message_with_quote_writes_reply_to`: the status is
   `written, quoted_found`, and `get_message(new)["quotedId"] == m1`.
2. `test_post_subsequent_message_quote_unknown_id_refuses_nothing_written`: `written=False`,
   `quoted_found=False`, TAIL unchanged, and `get_message(new) is None`.
3. `test_post_subsequent_message_quote_other_thread_refuses`: same assertions as test 2.
4. `test_post_first_message_quote_in_headless_thread_refuses`: `quoted_found=False`,
   `had_head=False`, and no HEAD.
5. `test_post_first_message_quote_on_headed_thread_reports_had_head`: the race row.
6. `test_post_message_without_quote_has_no_reply_to`: on both paths, `quoted_found=True` and
   `quotedId is None`.
7. `test_post_subsequent_message_quote_replay_is_dup_no_second_edge`: `dup_msg=True` and exactly
   one `REPLY_TO`.
8. `test_post_subsequent_message_replay_wins_over_unresolvable_quote`: replay a written msgId with
   a quote that no longer resolves. Expect `dup_msg=True`, `quoted_found=False`, and the chain
   intact. This pins "`dupMsg` wins" at the query level.

### 6.2 `scripts/test_queries.sh`

**Enumeration.** Find the copies by a clause of the query body, never by the status string:

```
grep -n "FOREACH (mem IN mems" scripts/test_queries.sh | grep -v seedIds
```

On today's tree this gives **12 runs**. Every one switches to the §3.7 text for its path. Each
copy inlines its ids as literals and passes only `mentions` as a parameter. So in every copy the
two new clauses keep `$threadId`/`$quotedId` verbatim, and its `CYPHER` prefix gains
`threadId='<that case's thread>' quotedId=<null|'id'>`. The other clauses keep their literals. A
missing `quotedId` fails loudly. A copy left on the old text fails nothing, so check off all 12.
Each `gq` copy's composed status string gains ` quoted=<quoteOk>`, and its expected value gains
` quoted=true`. Copies listed by assertion label:

| # | Assertion label | Path |
|---|---|---|
| 1 | `post first message m1: status row written` | first |
| 2 | `post second message m2 (v2 subsequent)` | subsequent |
| 3 | `post third message m3 (v2 subsequent)` | subsequent |
| 4 | `post reply m4 (v2 subsequent + REPLY_TO in guard)`: rewritten with `quotedId='m3'` in place of its `MATCH (quoted:Message {msgId:'m3'})` library shape. Its `m4 REPLY_TO edge exists` and `get single message m4: quotedId` follow-ups stay. | subsequent |
| 5 | `replay of subsequent write reports dupMsg=true (idempotent)` | subsequent |
| 6 | `late first-post refused with hadHead=true` | first |
| 7 | `unknown author refused with authorFound=false` | subsequent |
| 8 | `v2 subsequent write has no All Node Scan` (`gp` profile; the M1 guard below) | subsequent |
| 9 | `agent posts assistant message (v2 write commits)`: `w=true auth=true` gains `quoted=true` | subsequent |
| 10 | `mentions=[] v2 first-post commits (guard keeps the write alive)` (th2 `mn1`) | first |
| 11 | `mention post mn2 commits (v2)` (th2) | subsequent |
| 12 | `dedup post mn3 commits (v2)` (th2) | subsequent |

**Index guard (mandatory).** This is the plan's one silent regression: reverting to
`{msgId: $quotedId}` label-scans every `Message` on posts without a quote, and every other test
still passes.

- Copy 8 runs the subsequent text with `quotedId=null`. It keeps its `assert_not_contains "All
  Node Scan"` and adds
  `assert_contains "v2 subsequent write: quote lookup is index-anchored (null quotedId)" "Node By Index Scan | (quoted:Message)" "$prof"`.
- Add a twin `gp` case for the first text, on headed `th1` so it is a `hadHead` no-op, also with
  `quotedId=null`, and the same `assert_contains` on its own label.
- The check asserts **presence**, not the absence of `Label Scan`, because an absence check also
  passes on an error string.
- The red control: written before the text swap, the assertion fails, because today's text has
  no `quoted` operator.

**New cases**, placed after the m4 follow-ups unless stated otherwise:

- Subsequent into th1 with `quotedId='nope'` asserts `w=false … quoted=false`, then a `count`
  confirms the msgId was not created.
- First into th2 with `quotedId='m1'` (th2 is still headless at that point) asserts
  `w=false hh=false … quoted=false`, and th2 still has no HEAD.
- **After copy 12**, once th2 has a TAIL: subsequent into th2 with `quotedId='m1'` asserts
  `quoted=false`. Placed earlier, the subsequent anchor misses and the query returns zero rows.

### 6.3 Service (`server/tests/test_services.py`, `FakeRepo`)

`FakeRepo.post_*` accepts `**kw`, so the fake needs no change.

1. `test_post_message_passes_quoted_id_to_write`: the recorded kwargs carry `quoted_id`, and the
   result carries `quotedId`.
2. `test_post_message_without_quote_passes_none_and_returns_null_quoted_id`.
3. `test_post_message_quoted_not_found_raises_unknown_quoted_message`: a subsequent row
   `(F, F, F, T, F)`. Assert `str(exc)` equals the §3.3 detail literally.
4. `test_post_message_quote_had_head_redispatches_before_quote_check`: first `(F, T, F, T, F)`,
   then `OK`, gives success after two writes. This kills the quote-checked-first mutant.
5. `test_post_message_quote_dup_msg_is_idempotent_success`: `(F, F, T, T, F)` does not raise.
6. `test_post_agent_answer_unaffected_by_quote_dispatch`: a 4-argument `MessageWriteStatus`
   succeeds.

### 6.4 Front doors

1. **REST** (`server/tests/test_api.py`): `POST /threads/{tid}/messages` with `quotedId: m1`
   gives `201` and `quotedId == m1`. Then `GET /messages/{new}` gives `quotedId == m1`. This is
   the done-condition, end to end.
2. **REST**: an unknown or other-thread `quotedId` gives `400` with
   `error == "UnknownQuotedMessageError"` and `detail` equal to the §3.3 string, and no new
   message.
3. **REST**: with no `quotedId`, the response is `201` with `quotedId: null`.
4. **MCP** (`server/tests/test_mcp.py`, through `mcp.call_tool` like the existing cases):
   - `call_tool("send_message", {"body": …, "re": …, "quoted_id": m1})` returns `quotedId == m1`;
   - an unknown `quoted_id` gives an `isError` result whose text contains the §3.3 detail, and
     the background scheduler is not called;
   - `list_tools()`: `send_message`'s `inputSchema` lists `quoted_id` as a property that is not
     in `required` and carries no constraint keyword;
   - its `description`, whitespace-normalized (`" ".join(s.split())`), contains the §3.4
     paragraph, also normalized.
5. **Storefront** (`server/tests/test_storefront_api.py`): a successful `POST /shop/api/messages`
   body has `quotedId` set to `None`.
6. `test_input_bound_parity.py` passes.

Done means all of §6 passes, together with the full `pytest -q` and `./scripts/test_queries.sh`.

## 7. Risks

- **Shared files with the concurrent K-091 unit.** Its uncommitted diff touches `repository.py`,
  `services.py`, `scripts/test_queries.sh`, `tests/test_services.py`, `tests/test_repository.py`
  and `tests/test_api.py`, and this plan edits all six. None of its hunks touches a §4 symbol,
  but the unit is serialized after that commit, per §5.
- **Messages without `threadId` can't be quoted.** A pre-K-007 message that has not been
  backfilled is refused as not found. The fix is `backfill_thread_ids.sh`. A traversal check
  would cost O(thread length) per post instead.
- **A model may fabricate an id.** An agent can fill `quoted_id` with a guess. The guess is
  refused loudly with recovery text (§3.3/§3.4), so no data is silently wrong and the message is
  only lost if the agent ignores the error. No model measurement covers the wording. Posting
  without the quote on a bad id was rejected, because it would hide caller mistakes on REST too.
- **`quotedId` length is unbounded.** It is looked up, never stored, and a huge value costs one
  failed index lookup and a `400`.
- **No open stakeholder question.** The scope, the `400` and the read deferral are reversible
  design calls, made above with their reasons.

## 8. Live-spec sentences that change at close

The delegated spec writer applies these. This plan edits none of them.

**DESIGN**
- §5.1: move `(:Message)-[:REPLY_TO]->(:Message)` out of "Read by queries but written by no
  current code path" and into the chat edges, annotated: quoted message, same thread, at most one
  per message, written by the §4 post paths, read by `get_message`.
- §5.3: one sentence saying that either variant optionally creates a same-thread `REPLY_TO` and
  otherwise refuses (`quotedFound=false`).
- §9 "Post message" row: "`MENTIONS_MEMBER` edges (and, for an AI answer, `EMITTED` seeds) ride
  the same query" gains "and an optional same-thread `REPLY_TO`".

**QUERIES**
- §4, first and subsequent bodies: replace with the §3.7 texts.
- §4 "Status-row contract": add the `quotedFound=false` row. On the first path, `hadHead=true`
  takes precedence. Add that the §10.1 rows have no `quotedFound`.
- §4 "Write-path invariants": add the `UNWIND [$quotedId]` reason, citing the quirk file, and
  that a one-element list always yields one row.
- §4 "Post a reply": delete it or turn it into a pointer. Its "*No server write path creates
  `REPLY_TO`*" becomes false.
- §10.1: "Status-row contract is identical to §4" becomes "is §4's first four columns (no
  `quotedFound`)".

**SERVER**
- §1.4 `POST /threads/{tid}/messages` row: "(+ mentions)" becomes "(+ mentions, + optional
  same-thread quote `quotedId`)", plus "an unknown or other-thread `quotedId` → `400`
  `UnknownQuotedMessageError`, nothing written".
- §1.4 `GET /messages/{mid}` row: add "`quotedId` is the quoted message's `msgId`, or `null`".
- §1.4 **Error envelope**: add `UnknownQuotedMessageError` to the `400` examples.
- §2.2 `send_message` row: the signature gains `quoted_id=None`, and the query cell gains
  "(+ optional same-thread quote)".
- §2.2 **Argument bounds**: name `send_message`'s `quoted_id` among the undeclared id arguments.
- Wherever SERVER lists the keys of the post response, including the storefront's, add
  `quotedId`.

## Ready to implement

Implement the plan at `falkor-chat/docs/plans/message-quoting.md`, after the K-091 commit, in
three steps:

1. **Repository:** the §3.7 Cypher, `MessageWriteStatus.quoted_found`, and the
   `test_queries.sh` §6.2 work, which covers 12 copies, the mandatory index guard and the new
   cases.
2. **Service:** `quoted_id` plumbing and `UnknownQuotedMessageError` with the exact §3.3 detail,
   checked after `had_head`.
3. **Front doors:** `PostMessageIn.quotedId`, MCP `quoted_id` with the exact §3.4 docstring
   paragraph, and the parity entry.
