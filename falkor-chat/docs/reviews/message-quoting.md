# Review — message quoting plan (K-094)

> **Status:** archived · **Owner:** `analyst` · **Tracks:** K-094

## Scope & verdict

**Reviewed:** `falkor-chat/docs/plans/message-quoting.md` (untracked, first version), checked
against the working tree on 2026-11-20. I read the `### K-094` backlog item, `falkor-chat/AGENTS.md`
("Message write paths", rule 4), QUERIES §4 and §10.1, DESIGN §5.1/§5.3/§9, SERVER §1.3/§1.4/§2.2,
and the code by symbol: `MessageWriteStatus`, `Repository._write_status`/`post_first_message`/
`post_subsequent_message`/`post_agent_answer*`/`get_message`, `Services._dispatch_write`/
`post_message`, `app._register_error_handlers`, `api.post_message`, `schemas.PostMessageIn`,
`mcp.send_message`, `storefront_api.post_message`, `background._safe_run_workflow`,
`trigger.maybe_trigger`, `test_input_bound_parity.py`, the §4 block of `scripts/test_queries.sh`,
root `REVIEW.md` and `falkor-chat/REVIEW.md`. The stakeholder decision (add quoting, server-side
only) is taken as settled.

**Not run:** the pytest suites and `test_queries.sh`, at the brief's instruction (another unit
owns `ws:test`). **Live probes I ran:** both §3.7 texts, extracted verbatim from the plan file
with `sed`, against the throwaway `ws:probe-k094-review` on `falkordb-dev` (v4.18.11). The key
was `GRAPH.DELETE`d after each of two runs, and `GRAPHS` afterwards confirmed it was gone. No
write touched any other key.

**Verdict: approve with suggestions.** The design is sound, and every premise I was asked to test
held when executed (see Appendix A). Three majors should be fixed before handoff, all in the
test plan and the model-read text: the null-parameter regression guard is optional and sits in
the wrong form, the suite sweep finds only 7 of the 12 §4-body runs, and the new MCP argument
is model-read text with no fabrication guard. None of them needs a design change.

CPG: used cpg_falkorchat — enumerated the production callers of `post_message` (`mcp.send_message`, `api.post_message`, `storefront_api.post_message`), cross-checked against an unfiltered grep of the current tree.

## Findings

### Major

**M1 — The guard against the plan's one silent regression is optional and would be written in a
form that can't catch it** (plan §7 "Null-parameter plan regression", §6.2). *Verified by
execution.* The plan says that reverting to `{msgId: $quotedId}` "silently label-scans … and the
tests still pass", then makes the `GRAPH.EXPLAIN` check optional and phrases it as "expecting no
`Label Scan`". I measured the mutant: with `quotedId=null` the direct form profiles
`Node By Label Scan | (quoted:Message) | Records produced: 201` on a 201-message graph. The
`UNWIND` form profiles `Node By Index Scan | (quoted:Message) | Records produced: 0`.
`test_queries.sh` already has the right home for this check: the case "v2 subsequent write plans
with indexed anchors only" (`gp … assert_not_contains "All Node Scan"`). That assertion can't
see this regression, because the mutant produces a *Label* Scan, not an All Node Scan.
**Fix:** make the check mandatory in step 1. Update that existing profile case to the §3.7
subsequent text with `CYPHER … quotedId=null`, add the same for the first text, and assert
**presence** of `Node By Index Scan | (quoted:Message)` with `assert_contains`. Don't assert the
absence of `Label Scan`: an absence check also passes when `gp` returns an error string. The red
control is the direct form above. Written first against today's text, which has no `quoted`
operator, it is red before step 1 and green after.

**M2 — The prescribed sweep for the suite's §4 copies finds 7 of the 12 runs** (§6.2: "A
`grep -n "assert_contains.*hh="` lists them"). *Verified by grep.* That pattern matches only the
seven cases that compose `hh=`. An unfiltered enumeration
(`grep -n "FOREACH (mem IN mems" scripts/test_queries.sh | grep -v seedIds`) gives 12 runs. The
five that the pattern misses are:

- the indexed-anchors `gp` profile case (M1's home);
- "agent posts assistant message (v2 write commits)", which asserts `w=true auth=true`;
- the three th2 mention posts (`mn1`/`mn2`/`mn3`), which return only `'w='+toString(ok)`.

The plan's prose names the th2 posts, but its grep misses them. A missed copy fails nothing: it
keeps running the old text and passing. At close, QUERIES §4 would call the new bodies
"suite-verified" while the Agent-author and mention paths were verified only on the old text.
The plan also says the copies stay "verbatim except the final `RETURN`", but they inline
`threadId`/`msgId` as literals and send only `mentions` as a parameter. The new
`WHERE quoted.threadId = $threadId` therefore needs `threadId` added to each `CYPHER` prefix (or
inlined). **Fix:** replace the grep with the unfiltered enumeration, list the 12 cases by
assertion label (not line number), and state the `threadId`/`quotedId` prefix change. A forgotten
`quotedId` fails loudly (`ResponseError: Missing parameters`, measured), but a forgotten text
swap does not.

**M3 — The new `send_message` argument is model-read text with no guard against a fabricated id**
(§3.4, step 3; `falkor-chat/REVIEW.md`, "What counts as serious"). *Inferred from the standing
rule and the contract.* MCP callers are LLM agents (Claude subagents, the Kiro demo agent, the
mcp-monitor-woken headless agents). An optional `quoted_id` invites a model to fill it with a
guessed id. Under this design a guessed id turns the **whole post** into a refusal, so the
agent's message is lost unless it understands the error and retries. The plan specifies the
docstring only loosely ("an id outside the thread is an error") and words the detail as
`quoted message … is not a message in thread …`, which tells the agent neither that nothing was
posted nor how to recover. **Fix:** (a) the docstring says to omit `quoted_id` unless you hold the
exact `msgId` from a `read_messages(re=…)` row of this thread, and never to construct or guess one;
(b) the error detail ends with "nothing was posted; retry without `quoted_id`, or with a `msgId`
from `read_messages`"; (c) §6.4 pins both strings with literal assertions. The standing rule
otherwise requires a measurement. If the architect would rather fall back to posting without the
quote, that is a design change to route back. I don't recommend it, because it would hide a
caller's mistake on REST too.

### Minor

**m1 — The storefront response grows too, and §7 says it doesn't.** *Verified by reading.* §7:
"No such client exists in-repo; the storefront SPA uses its own storefront routes." But
`storefront_api.post_message` returns `services.post_message`'s `posted` dict unchanged, so
`POST /shop/api/messages` gains `"quotedId": null` as well. The effect is harmless: the SPA's
`postMessage` is a compile-time TS cast (`salesperson/src/api/endpoints.ts`) with no runtime
validation, and no storefront test pins the response keys (grep). SERVER §1.3 ("answers the
posted message") stays true. **Fix:** correct the §7 sentence, and have step 3 add one
storefront assertion that the 200 body's `quotedId` is `null`. That makes the change deliberate
rather than incidental.

**m2 — §8 misses one spec sentence the change falsifies.** *Verified by grep.* QUERIES §10.1:
"Status-row contract is identical to §4 (zero rows = no TAIL …; `dupMsg=true` = idempotent
replay)". After the change §4 has five columns and §10.1 has four. The plan's §4 contract
addition ("the §10.1 rows have no `quotedFound`") leaves this §10.1 sentence contradicting it.
The same claim sits in the `post_agent_answer` docstring ("Same status-row contract as §4").
**Fix:** add the §10.1 sentence to §8, and have step 1 adjust that docstring. The rest of the
unfiltered sweep over the three spec files (`REPLY_TO`, `quotedId`, `quoted`, `reply`) is covered
by §8 or needs no change: DESIGN §1.2's register row, the naming list at DESIGN line 150, QUERIES
§10's "same convention as `REPLY_TO`", and the storefront §1.3 body sentence.

**m3 — The collision note names only `repository.py`.** *Verified by `git diff --stat`.* The
concurrent unit's uncommitted diff also touches `services.py`, `scripts/test_queries.sh`,
`tests/test_services.py`, `tests/test_repository.py` and `tests/test_api.py`, and this plan edits
every one of them. I checked the hunks and none overlaps a §4 symbol: they are K-091 LSH-band
code (`create_document_with_auto_supersede`, `list_documents_missing_lsh_bands`,
`set_document_lsh_bands`, and the document-ingest call in `services.py`) and the §14.10 suite block. **Fix:** list all six files in §7 and say that the unit lands after the other
unit's commit, or rebases on it. The brief's "serialize shared-file units" practice applies.

**m4 — The MCP test should go through the generated schema** (§6.4.4;
`falkor-chat/REVIEW.md`, MCP input contract). As written, the test calls
`send_message(body, re, quoted_id=…)` directly. **Fix:** drive `mcp.call_tool("send_message",
{…, "quoted_id": …})`, matching the existing `test_mcp.py` cases. Also assert that
`list_tools()`'s `inputSchema` lists `quoted_id` as optional with no constraint keyword, which is
the same fact the parity table's `NoBound` entry claims.

### Nits

**n1** — §6.1 has no repository case for a replay whose quote no longer resolves. I measured that
row as `(written=F, hadHead=F, dupMsg=T, authorFound=T, quotedFound=F)`, with the chain intact. A
literal case would pin the "`dupMsg` wins over `quotedFound`" contract at the query level, not only
in the FakeRepo test §6.3.5. It is low value, because msgIds are server-minted and `dupMsg` is a
backstop.
**n2** — §2's "Service" bullet restates the dispatch loop that DESIGN §5.3 and QUERIES §4 already
specify. Per root `REVIEW.md`, a plan cites spec content rather than restating it; a clause
pointing at §5.3 would do.

## What's solid

- **The resolution idiom works** (premise 1). Measured with `GRAPH.EXPLAIN` and `GRAPH.PROFILE`
  on both paths: `Node By Index Scan | (quoted:Message)` for null (0 records) and for a real id
  (1 record). The direct form label-scans when the parameter is null. All 13 executed cases
  returned exactly one row: `UNWIND [x]` neither collapses nor duplicates rows, including when
  combined with a duplicate/unknown mention list (`['a1','u1','a1','nope']` gives 2 mention edges
  and 1 `REPLY_TO`).
- **Guard ordering and refusal hold** (premise 2). Unknown id, other-thread id, self-quote,
  legacy message without `threadId`, and headless first path were each refused with nothing
  written. Every refused msgId was absent afterwards, th2 still had no HEAD, the chains were
  intact, and there was at most one `REPLY_TO` per message. The dispatch order (`written or
  dup_msg` → `author` → `had_head` → `quoted_found`) is right: the race row F4 was measured as
  `(F,T,F,T,F)`, and the test in §6.3.4 kills the mutant that checks the quote first.
- **Compatibility holds** (premise 3). `MessageWriteStatus` is a frozen dataclass constructed
  positionally by `_write_status`, so the 4-column §10.1 rows and the three `test_services.py`
  constants (`OK`/`DUP`/`HAD_HEAD`, keyword-built) keep working with the default `True`. The
  unfiltered grep of `falkor-chat/` for `MessageWriteStatus|authorFound|author_found|_write_status`
  found no other consumer: one docs mention in `docs/plans/document-ingestion-graph.md`, and the
  `test_queries.sh` hits are composed strings. `posted` consumers read named keys only
  (`background._safe_run_workflow`, `trigger.maybe_trigger`, `api`/`mcp` embed scheduling). Every
  MCP caller in the repo uses named arguments (`call_tool` dicts).
- **The error contract holds** (premise 4). `_handle_service_error` sends every `ServiceError`
  outside its explicit not-found tuple to 400, so a new sibling class needs no `app.py` change.
  `UnknownMemberError`, the closest precedent, gets 400 the same way.
- **The read deferral is coherent with the done-condition** (premise 5). K-094's done-condition
  needs "written by a front door and read back", and that holds through REST
  `POST /threads/{tid}/messages`, then `GET /messages/{mid}` (§6.4.1). The plan states the MCP
  read-back gap and gives it a follow-up item.
- **Same-thread scope, in-query validation and the single-query write** respect rule 4 and the
  locked "two self-guarding paths" decision. The step table cites code by symbol throughout.

## Open questions

None for the stakeholder. M3's alternative (silently dropping a bad quote) would be a design
reversal for `architect` to raise; this review recommends against it.

## Proposed REVIEW.md lines

Target `falkor-chat/REVIEW.md`, `## Always check`. Source: this review's M1 and M2.

- When a plan changes a query whose copies are inlined in `scripts/test_queries.sh`, the sweep
  that finds those copies matches a clause of the query body, never the composed status string.
  The review compares the plan's count with an unfiltered enumeration.
- When a plan's correctness or cost rests on a clause planning as an index scan, a suite case
  asserts the **presence** of `Node By Index Scan | (<var>:<Label>)` for that clause. Absence of
  `All Node Scan` doesn't count: a `Label Scan` regression, or an error string, passes it.

## Appendix A — live probe results (`ws:probe-k094-review`, 2026-11-20)

Status order: `written hadHead dupMsg authorFound quotedFound`. Setup: indexes on
`Message.msgId`/`Thread.threadId`/`User.userId`/`Agent.agentId`, `UNIQUE` on `Message.msgId`,
and 50–200 padding messages in another `threadId`.

| Case | Path | Row |
|---|---|---|
| no quote, empty th1 | first | T F F T T |
| no quote + mention | subsequent | T F F T T |
| quote m1 (same thread) | subsequent | T F F T T |
| quote unknown | subsequent | F F F T F |
| quote other-thread msg | subsequent | F F F T F |
| headless th2, quote m1 | first | F F F T F |
| race (th1 headed), valid quote | first | F T F T T |
| race, unknown quote | first | F T F T F |
| replay m3 with its quote | subsequent | F F T T T |
| quote m2 + mentions `[a1,u1,a1,nope]` | subsequent | T F F T T |
| self-quote (`quotedId = msgId`) | subsequent | F F F T F |
| legacy message without `threadId` | subsequent | F F F T F |
| replay t3 after its quote target stopped resolving | subsequent | F F T T F |

After the run: `REPLY_TO` edges were exactly `m3→m1` and `m9→m2`, `max(count)` per message was 1,
none of the refused msgIds existed, th2 had no HEAD, and th3's chain was `t1→t2→t3` with TAIL t3.
Omitting the `quotedId` parameter raised `ResponseError: Missing parameters`.

Plans, from `GRAPH.PROFILE` with the `CYPHER` prefix, subsequent text, on the 201-message graph:

| Form | `quotedId` | `(quoted:Message)` operator | Records |
|---|---|---|---|
| `UNWIND [$quotedId]` (plan) | null | Node By Index Scan | 0 |
| direct `{msgId: $quotedId}` | null | **Node By Label Scan** | 201 |
| `UNWIND [$quotedId]` (plan) | `'t1'` | Node By Index Scan | 1 |
| direct | `'t1'` | Node By Index Scan | 1 |

`GRAPH.EXPLAIN` of the first text gave the same `Node By Index Scan | (quoted:Message)` for null
and for a real id. No anchor of either text planned a label scan or an all-node scan.

## Pass 2

> 2026-11-20 · plan revised in place, 4,384 → 3,739 words · current tree, K-091's `test_queries.sh` edits included

**Verdict: approve.** Every finding is fixed or deviated soundly. One new nit, which doesn't block.

**Dispositions**
- **M1: fixed** (§6.2 "Index guard (mandatory)"). The check now asserts *presence* of `Node By Index Scan | (quoted:Message)` on copy 8, plus a first-text twin. Rechecked against the thread state the copy order leaves in place: copy 8 runs on th1 with its dup `m2`, a guarded no-op. The twin runs on th1, which has been headed since copy 1, so it is a `hadHead` no-op. The `(quoted:Message)` operator doesn't depend on the data (Pass 1 EXPLAIN, both texts). `gp` is a raw `redis-cli GRAPH.PROFILE` (`test_queries.sh` `gp()`), and its operator line contains the asserted substring. The red controls hold: today's text has no `quoted` operator, and the direct form produces `Node By Label Scan` (Pass 1, Appendix A).
- **M2: fixed** (§6.2 table). I re-ran the plan's enumeration verbatim on the current tree: **12 runs**, at lines 208 215 219 223 245 255 262 273 535 545 552 561. K-091's hunks are all in §14.10, after these. Each of the 12 assertion labels (and both m4 follow-ups) matches exactly one line (`grep -cF` = 1 each). The `threadId`/`quotedId` prefix change is stated.
- **M3: fixed, with a deviation I accept.** (a) The §3.4 docstring paragraph is exact and asserted after normalization. (c) The detail is asserted literally on REST, MCP and the service. **(b) Ruling: the deviation is right.** The detail comes from a single `ServiceError` that both front doors share. Naming `quoted_id` would be wrong on REST, and naming `quotedId` would be wrong on MCP. "nothing was posted. Retry without the quote, or quote the msgId of a message read from this thread" carries both facts the finding asked for. For an MCP caller it reads correctly next to the docstring, which already names `quoted_id` and `read_messages(re=...)`. §7 now records honestly that no model measurement covers the wording.
- **m1: fixed** (§2, §3.4, §6.4.5 storefront `quotedId is None`).
- **m2: fixed** (§8 QUERIES §10.1 line; step 1 changes the `post_agent_answer` docstring).
- **m3: fixed** (§7 names all six files; §5 and "Ready to implement" serialize after the K-091 commit).
- **m4: fixed** (§6.4.4 goes through `call_tool`, and `list_tools()` checks the property, `required` and the absence of a constraint).
- **n1: fixed** (§6.1.8; matches my measured row `F F T T F`).
- **n2: fixed** (§2 now cites DESIGN §5.3 for dispatch instead of restating it).

**Fixes checked against each other**
- **New-case placement.** The th1 `'nope'` case and the headless-th2 first-path case go "after the m4 follow-ups". The follow-ups are split (the edge check at line 227, `get single message m4` at line 310), but both cases are valid anywhere in that span. th2 stays headless until copy 10 (line 545), and a refused write creates nothing and sets no `updatedAt`, so no later count, cursor-pagination or `updatedAt` assertion moves. The cross-thread case after copy 12 runs when th2's TAIL is `mn3`. It returns a row with `quoted=false` (Pass 1 measured the same shape), and it adds no message for the §9.1 th2 reads that follow.
- **The trims (§4 table at 8 rows, alternatives cut) dropped no fact an implementer needs.** The empty-string, headless-unknown and unknown-author cases are folded into rows that still exist. The reasons for rejecting the pre-read and the two-text alternative are kept in one sentence each. §6.1.3 still names the other-thread setup through the test name.

**Growth-stops check.** The document shrank 645 words and adds no new scope. The rework replaced text in place, and there is no finding→resolution table. The fixes are cited as design, not by ID, except §6.2's "the M1 guard" labels, which cite by ID and are permitted. One restated fact is inaccurate: see N3.

**New**
- **N3 (nit): §4's "The review re-ran the same matrix on its own throwaway key and got identical rows" overstates what the review did.** Appendix A ran a different, overlapping matrix. It had no `''` case and no unknown-author case, but it had self-quote, a legacy message without `threadId`, and an unresolvable replay. Its message names also differ (`m9→m2`, not `m10→m2`). Where the cases overlap, the rows agree. **Fix:** replace the sentence with "the review's own matrix (its Appendix A) agreed on every case the two share". Alternatively, drop it, since §2 already cites the measurement.
- **N4 (nit): the §3.3 detail echoes `quoted_id!r` with no length cap**, and §7 says the value is unbounded. A huge guessed id therefore comes back in full in the 400 body and in the model-read MCP error. If the architect wants a cap, truncate the echoed value (for example, the first 64 characters), and have §6.3.3 assert the truncated literal. It is take-or-leave: the only cost is a large error message.
