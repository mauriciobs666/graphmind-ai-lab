# Review — message quoting implementation (K-094)

> **Status:** archived · **Owner:** `analyst` · **Tracks:** K-094

## Scope & verdict

**Reviewed:** the uncommitted `tdd-engineer` diff (`git diff --` on the 13 files named in the
brief), checked against the approved plan `falkor-chat/docs/plans/message-quoting.md` (gated in
`falkor-chat/docs/reviews/message-quoting.md`, Pass 2). The files are `server/falkorchat/`
`repository.py`, `services.py`, `schemas.py`, `api.py`, `mcp.py` and `storefront_api.py`;
`server/tests/` `test_repository.py`, `test_services.py`, `test_api.py`, `test_mcp.py`,
`test_storefront_api.py` and `test_input_bound_parity.py`; and `scripts/test_queries.sh`. The
K-098 U5 files are out of scope (another analyst is gating them). I applied root `REVIEW.md`
and `falkor-chat/REVIEW.md`.

**Suites, which I ran for this gate:**
- `./scripts/test_queries.sh`: **495/495**. It wipes `ws:test` and `reference-test` only.
- full `pytest -q`: **3762 passed, 14 deselected, 1 xfailed**. This includes the `wf_repo`
  tests, which wipe `reference-test`, not the live `reference`, so no re-seed is needed
  (`falkor-chat/AGENTS.md`, "The reference graph's name"). The run includes U5's
  FalkorDB-free tests.

**Verdict: approve.** The shipped code matches the plan's §3.7 texts and dispatch order. All
four departures are sound, and the first two improve on the plan. I found no blocker, major or
minor, and two nits.

CPG: considered, not relevant — the diff is a 13-file seam change I read in full; the Pass 1 `cpg_falkorchat` caller enumeration (three `post_message` callers) still matches an unfiltered grep, and the storefront caller is the one the diff touches.

## Findings

### Nits

**n1 — The first-path `REPLY_TO` `FOREACH` has no positive test, so deleting it survives**
(implementer's departure 4). *Inferred from the contract, and consistent with the implementer's
own statement.* The first path writes only into a headless thread, and a same-thread quote
can't resolve there. The only exception is a planted orphan: a `Message` carrying the thread's
`threadId` but outside its chain. Keeping the branch is right, because it keeps the two §3.7
texts symmetric and QUERIES §4 canonical. **Fix (optional):** add a repository case that plants
such an orphan and asserts that the first path writes `REPLY_TO`, or add one line to the `# ── §4`
comment saying the branch is unreachable today and kept for symmetry.

**n2 — `UnknownQuotedMessageError(write_kwargs.get("quoted_id"), …)` can pass `None` to a
parameter typed `str`** (`Services._dispatch_write`). This is unreachable: `quoted_found` is only
`False` when a quote was sent, and the §10.1 rows default to `True` (pinned by
`test_post_agent_answer_unwritten_four_column_row_is_not_a_quote_error`). Annotating
`str | None` would make that true on the page; there is nothing to test.

## Judgements requested

**Departures.**
1. **The method-level index test is sound, and better than the plan.** I ran a mutant of my own
   on a copy of the package: the bare `{msgId: $quotedId}` form, with the `UNWIND` removed, loaded
   into `Repository` by a pytest plugin. Of the 12 quote tests, exactly
   `test_post_message_quote_lookup_is_index_anchored[None-post_first_message]` and
   `[None-post_subsequent_message]` fail, and all behavioural tests pass. That is precisely the
   blindness §7 predicted. This test EXPLAINs the text the method actually sends, so unlike the
   suite's hand copy it can't drift from the shipped query. Note that a mutant injected *under*
   the test's recording layer (wrapping `_graph`) is invisible to it by construction, which is
   expected and not a gap.
2. **Declaring the storefront partition-guard entry was necessary and correct.**
   `SERVICE_ERRORS_UNREACHABLE` gains the class, and the subclass count goes 15 → 16. The
   unreachability claim is true: `storefront_api.post_message` never passes `quoted_id`, so
   `quotedId=null` makes `quoteOk` true. `test_the_quote_unreachability_claim_is_pinned_to_its_producer`
   checks this on the wire: a body with `quotedId: "nope"` still answers 200 with `quotedId: null`,
   and the storefront `PostMessageIn` has no such field. Threading the field through would turn it
   red. The plan missed this guard; the implementer found it.
3. **Not truncating the echoed id (N4) is acceptable.** N4 was take-or-leave.
4. **Keeping the first-path `REPLY_TO`:** sound (n1).

**Positional `MessageWriteStatus(*row)` with `quoted_found=True` as the default is safe.**
*Traced end to end.* `_dispatch_write` checks the quote only for rows from the write pair it is
given. `post_message` passes `post_first_message`/`post_subsequent_message`, and both RETURN
exactly five columns (`repository.py`; both texts read). `post_agent_answer` passes the §10.1
pair, which returns four columns and never receives `quoted_id`. A 4-column row therefore
reaches the quote check only from §10.1, where the default `True` is correct.

A future regression that drops `quotedFound` from a §4 text fails safe: a refused write then has
`written=False` and falls to `RuntimeError`, not to success. teco's column-swap mutant (7 tests
red) shows that a positional mix-up is also caught.

**The `test_queries.sh` copies match the shipped texts.** *Verified by a structural
comparison.* I captured the exact strings `post_first_message`/`post_subsequent_message` send,
using a recording graph stub. I compared each suite run containing `FOREACH (mem IN mems` (no
`seedIds`) after normalizing whitespace, `\$`, string and number literals and parameters, and
cutting at `RETURN`. All **16** runs match their path's text: the 12 copies, the first-text
profile twin and the 3 new refusal cases. The prefixes carry the right `threadId`/`quotedId`, and
each composed status carries `quoted=`.

The comparator can both fail and pass. On `HEAD`'s suite, all 12 copies report DIFF. On a copy
of the current suite mutated to the bare form, all 16 report DIFF. The plan's prescribed
enumeration still gives 12 on this tree. The two index assertions (`v2 subsequent/first write:
quote lookup is index-anchored (null quotedId)`) passed in my run. Their red control is the
bare form's measured `Node By Label Scan | (quoted:Message)` (plan review, Appendix A).

## What's solid

- **The code matches the plan.** Both §4 texts match §3.7 clause for clause. The dispatch order
  is `written or dup_msg` → `author` → `had_head` → `quoted_found`. The detail string and the MCP
  docstring paragraph match the plan verbatim. The `post_agent_answer` docstring is corrected
  (the plan review's m2). `quotedId` is always sent.
- **The tests assert behaviour, with expected values written as literals rather than read
  from the class.** The detail string is asserted literally in `test_services.py`, `test_api.py`
  and `test_mcp.py`. The MCP tests go through `call_tool`, a connected client session (`isError`,
  nothing scheduled, nothing written) and `list_tools()` (optional, unconstrained, description
  carries the paragraph). Repository tests check TAIL, `get_message`, the `REPLY_TO` set and
  HEAD count on every refusal. The REST test covers the done-condition end to end (post, then
  `GET /messages/{mid}`).
- **The parity entry is in place:** `("send_message", "quoted_id")` is `NoBound`.

## Live-spec close list

The implementer's list (plan §8 plus its additions), checked against the shipped diff and an
unfiltered grep of DESIGN/QUERIES/SERVER for `REPLY_TO`, `quotedId`, `quoted`, `hadHead`,
`status row`, `send_message` and `UnknownMemberError`:

| Spec section | Change | Ruling |
|---|---|---|
| DESIGN §5.1 | move `REPLY_TO` out of dormant schema; annotate same-thread, ≤1, written by §4, read by `get_message` | required |
| DESIGN §5.3 | one sentence: optional same-thread `REPLY_TO`, else refused (`quotedFound=false`) | required |
| DESIGN §9 "Post message" row | "+ an optional same-thread `REPLY_TO`" | required |
| QUERIES §4 bodies | replace with the shipped texts (= plan §3.7, verified above) | required |
| QUERIES §4 status-row contract | `quotedFound=false` row; `hadHead` takes precedence on the first path; §10.1 rows have no `quotedFound` | required |
| QUERIES §4 write-path invariants | `UNWIND [$quotedId]` reason (cite `claude/graph-dba/falkordb-quirks.md`); one-element list → one row; parameter always sent | required |
| QUERIES §4 "Post a reply" | delete or turn into a pointer; its "No server write path creates `REPLY_TO`" is false | required |
| QUERIES §10.1 | "Status-row contract is identical to §4" → "§4's first four columns" | required |
| SERVER §1.4 `POST /threads/{tid}/messages` row | + optional `quotedId`; unresolvable → `400 UnknownQuotedMessageError`, nothing written | required |
| SERVER §1.4 `GET /messages/{mid}` row | `quotedId` = quoted `msgId` or `null` | required |
| SERVER §1.4 Error envelope | add `UnknownQuotedMessageError` to the 400 examples | required |
| SERVER §2.2 tool table, `send_message` row | signature gains `quoted_id=None`; query cell "+ optional same-thread quote" | required |
| SERVER §2.2 Argument bounds | name `send_message` `quoted_id` among the undeclared id arguments | required |
| SERVER, post-response keys | **no SERVER sentence lists `posted`'s keys** (grep). Optional: the §1.4 storefront `POST /messages` sentence ("answers the posted message") may add "its `quotedId` is always `null`; a `quotedId` in the body is ignored" | optional |
| SERVER, storefront unreachable set | **not spec'd by name**: §1.4 (storefront error handling) describes the partition as a measurement, with no class list | no change |

I found nothing beyond the implementer's list. DESIGN §1.2's register row, DESIGN's naming list,
QUERIES §10's "same convention as `REPLY_TO`" and SERVER §1.4's Bounds bullets stay true as
written. Per root `REVIEW.md`, the spec writer checks against this shipped diff, not against the
plan.

## Proposed REVIEW.md lines

Target `falkor-chat/REVIEW.md`, `## Always check`. Source: the plan review's M1/M2
(`falkor-chat/docs/reviews/message-quoting.md`) and this review's departure 1 ruling. The second
line is revised from the plan review: this implementation showed that a check on the suite's
hand copy can drift from the shipped text, and a check on the captured text cannot.

- When a change touches a query whose copies are inlined in `scripts/test_queries.sh`, the sweep
  that finds those copies matches a clause of the query body, never the composed status string.
  In an implementation review, compare each copy with the text the method actually sends, after
  normalizing literals and parameters.
- When a query's correctness or cost rests on a clause planning as an index scan, a pytest
  EXPLAINs the text the method actually sends (captured at the graph call). It asserts the
  presence of `Node By Index Scan | (<var>:<Label>)` for that clause. Absence of `All Node Scan`
  doesn't count: a `Label Scan` regression, or an error string, passes it.

No prune is proposed: no existing line in `falkor-chat/REVIEW.md` is now enforced elsewhere.

## Appendix — evidence

- **Structural copy comparison:** script and outputs under
  `/tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/aae0e7a7-4212-4b57-9541-3e1dca2fac24/scratchpad/k094/impl/`
  (`cmp.py`, `cmp_old.py`, `cmp_mut.py`). Current tree: 16/16 MATCH, at suite lines 209 216 220
  224 246 256 263 274 279 324 331 556 566 573 582 590. `HEAD`: 12/12 DIFF. Bare-form-mutated
  copy: 16/16 DIFF.
- **Bare-form mutant on the method:** a package copy with both `UNWIND [$quotedId]` lines
  removed and `{msgId: qid}` → `{msgId: $quotedId}`, loaded by
  `plug/mutsrc_plugin.py`. Running `pytest tests/test_repository.py -k "quote or reply_to"`
  gave 2 failed (both `[None-…]` index cases) and 10 passed.
- **Suites:** `test_queries.sh` 495/495 (output `impl/tq.out`); `pytest -q` 3762 passed,
  14 deselected, 1 xfailed in 74 s.

## Close check

> 2026-11-20 · step 3 of the feature close: the writer's uncommitted DESIGN/QUERIES/SERVER/BACKLOG edits, checked against shipped `496c5d40` · docs-only, no suites run (U12 owns `ws:test`)

**Verdict: approve with suggestions.** Every contract the shipped diff changed is in the spec,
and the spec says nothing the code contradicts on the §4 bodies, the status row or the front
doors. There are two minor fixes, both one-line rewrites. Neither blocks the close; apply them
before the `HISTORY.md` entry.

**Checks run**
- **The §4 Cypher blocks are byte-equal, modulo whitespace, to the sent text.** *Run.* I loaded
  `repository.py` from `git show 496c5d40:…` (not the working tree: a later unit has since edited
  the §6 hybrid-search query in it), captured the string each method passes to `query`/`ro_query`
  through a recording stub, and compared it with every ```` ```cypher ```` block in QUERIES.md.
  `post_first_message` matches block 12 exactly, `post_subsequent_message` block 13, and
  `get_message` block 16. The parameter sets carry `quotedId`.
  The comparator can fail: it reports the §10.1 `post_agent_answer*` blocks as non-matching. The
  only difference is `{ score:` versus `{score:`, whitespace inside a map literal, and those
  bodies were unchanged by this feature. This is pre-existing and cosmetic, and not a finding for
  this close. My check is independent of the writer's `extract_q.py`, which is a static AST read
  of a file path; mine is a runtime capture of the committed blob.
- **The status-row order matches `_dispatch_write`.** *Traced.* QUERIES §4: "`written` or
  `dupMsg`, then `authorFound`, then `hadHead`, then `quotedFound`" is the code's order. Also
  true as stated: `dupMsg` is success whatever `quotedFound` says, `hadHead` takes precedence on
  the first path, and the §10.1 rows have no `quotedFound`.
- **The cited quirk entry is committed.** *Run.* "A bare pattern-property equality match …" is
  in `HEAD:claude/graph-dba/falkordb-quirks.md` at line 791 (`git log -S`: since `fb10a7a3`). The
  working-tree diff to that file (+16 lines) belongs to others and is not what the spec cites.
- **No change history in the spec.** *Run.* Scanning the added lines for `K-094`, dates and
  "now/previously/used to/was/added/new" finds only "one added block" in §10.1, which predates
  this edit. The "Post a reply" heading keeps its place as a two-line pointer, so the numbering
  holds. The `REPLY_TO` move in DESIGN §5.1 deletes the dormant-schema line rather than
  annotating it "formerly dormant".
- **Absolute words and second copies.** *Run.* An unfiltered grep of the three spec files for
  `REPLY_TO|quotedId|quoted_id|quotedFound` finds 34 lines (DESIGN 7, QUERIES 22, SERVER 5), each read. "at most one per
  message", "never a bare …", "always sent" and "no separate reply query" hold against the
  code. Storefront resets `DETACH DELETE` their victims, so same-thread edges never dangle.
  SERVER's REST-parity addition ("an optional `quotedId` that is `send_message`'s `quoted_id`")
  matches `api.post_message` → `quoted_id=body.quotedId`. One second copy is wrong; see C1.
- **`GAL-I02-K0005` is sound.** *Read.* The ID is registered in `.work-items/registry.jsonl`
  (`new`, type K, parent `GAL-I02`, home `falkor-chat`, slug `quoted-id-read-back`, uncommitted
  with the rest). The body is present-tense and forward-looking, and names the three reads,
  both transports and the profile obligation. Its done-condition can fail either way (a missing
  `quotedId`, or an added scan). It is a correct split of plan §3.5's deferral.

**Findings**

**C1 (minor) — DESIGN §5.3's new clause contradicts QUERIES §4 on a replay.** "an unresolvable
quote (`quotedFound=false` on a row that lost no race) is a caller error" also covers a
`dupMsg=true` replay whose quote no longer resolves. That row is
`(written=F, hadHead=F, dupMsg=T, authorFound=T, quotedFound=F)`, measured in plan review
Appendix A and pinned by `test_post_subsequent_message_replay_wins_over_unresolvable_quote`.
The code and QUERIES §4 treat it as success. **Fix:** "… and an unresolvable quote
(`quotedFound=false` on a row that is neither written, a replay, nor a lost race) is a caller
error". Or drop the clause and cite QUERIES §4's order sentence, which is the one home of the
dispatch order.

**C2 (minor, teco's commit, outside the spec) — `496c5d40` committed a mutated copy of the whole
server package and two full suite copies as review evidence.** These are 44 files under
`docs/reviews/message-quoting-evidence/impl/`: `mut/falkorchat/*`, ~17k lines, including a
`repository.py` with the **bare `{msgId: $quotedId}` form**; plus `mut_tq.sh` and `old_tq.sh`,
~2k lines each. Every unfiltered sweep the REVIEW.md files mandate over `falkor-chat/` now hits a
stale, deliberately wrong copy of production code. A future "no bare form remains" check would
report a false positive, or be filtered around, at which point it becomes a bound. **Fix:** in a
follow-up commit, remove `mut/`, `mut_tq.sh` and `old_tq.sh`, and keep `cmp*.py`, the plugins,
`mutate.py`, the `spec_*.json` files and `tq.out`. The copies are reproducible: `old_tq.sh` is
`git show 496c5d40^:…`, and the mutant is the one-line `sed` stated in this review's Appendix.

**Final `falkor-chat/REVIEW.md` proposals, restated verbatim** (`## Always check`; no prune). C2
does not change either.

- When a change touches a query whose copies are inlined in `scripts/test_queries.sh`, the sweep
  that finds those copies matches a clause of the query body, never the composed status string.
  In an implementation review, compare each copy with the text the method actually sends, after
  normalizing literals and parameters.
- When a query's correctness or cost rests on a clause planning as an index scan, a pytest
  EXPLAINs the text the method actually sends (captured at the graph call). It asserts the
  presence of `Node By Index Scan | (<var>:<Label>)` for that clause. Absence of `All Node Scan`
  doesn't count: a `Label Scan` regression, or an error string, passes it.

Evidence: `/tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/aae0e7a7-4212-4b57-9541-3e1dca2fac24/scratchpad/k094/close/cmpq.py`
(run with the directory as its argument; it reads `repository_496.py`, extracted from the commit).
