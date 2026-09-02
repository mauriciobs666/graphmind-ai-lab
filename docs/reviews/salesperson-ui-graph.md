# The one salesperson UI — graph design note (S0) review

> **Status:** active · **Owner:** `analyst` · **Tracks:** — (M<n> TBD)

## 1. Scope & verdict

**Reviewed:** `docs/plans/salesperson-ui-graph.md` v1.0 (679 lines, `graph-dba`), the S0 gate for
the `salesperson-ui` build, against `docs/plans/salesperson-ui.md` v1.2 §4.3 / §4.6 / §4.8 / §4.9
and its S0/S4 rows in §5.1, `docs/reviews/salesperson-ui.md` **Pass 2 finding N1**,
`falkor-chat/docs/DESIGN.md` §5.1–5.4 / §7 and `falkor-chat/docs/QUERIES.md` §2, §3, §4, §9, §12,
§16, §17.

**Method:** every claim was re-derived, not taken on report. I built an independent fixture on my
own throwaway graphs (`ws:probe-analyst-s0`, `-scale`, `-noconstraint`; all three deleted at the
end of this review — see Appendix A.1 for the recipe), carrying `bootstrap_schema.sh`'s indexes
and UNIQUE constraints, `seed_demo.sh`-shaped non-participant survivors, a **second**
non-participant channel (`shared-promo`) that a participant is also a member of, and six
adversarial participant shapes. `ws:acme`, `ws:test`, `reference`, `ws:probe-s0-reset` and
`probe_u8_rename_dst` were never written to; the falkor-chat pytest suite was not run.

**Verdict: approve with suggestions.**

**I could not defeat either guard.** No graph shape reachable from the shipped code lets a reset
touch anything §4.8 says must survive. G2's provenance marker is genuinely unforgeable:
`Repository.create_channel` (`falkor-chat/server/falkorchat/repository.py:184-198`) writes a fixed
three-property map with no caller-controlled extras, and there is **no `MEMBER_OF`-deleting query
anywhere in the codebase** (`grep -rn "DELETE.*MEMBER_OF\|remove_channel_member"` → zero hits), so
the two states that *would* break scoping cannot be produced. §2.3's **variant B — the load-bearing
one — reproduces exactly**, byte-identically, on my independent fixture (Appendix A.2).

The findings below are one real correctness dependency the note introduces without disclosing it
(F1), two places where the note's own reasoning is inconsistent with what the queries do (F2, F3),
and one unreproducible row in its verification table (F4). None of them is a blocker; all of them
should be fixed in the note before S4 implements it verbatim.

`CPG: used cpg_falkorchat — confirmed the note's "TRIGGERED_BY from a thread message is the
storefront's only run-creation path" claim by enumerating non-test callers of
Services.start_workflow_run and Repository.start_run_untriggered (finding F6).`

---

## 2. Findings

### Major

#### F1 — `u.userId > ''` in `reset_all`'s G1 is a silent under-delete dependency that buys nothing

`§5` line 333 carries the conjunct into the **destructive** query, commented `G1 (+ index anchor,
§8)`. The conjunct is not always true: FalkorDB evaluates `42 > ''` as **`null`**, so a `User`
whose `userId` is not a string is silently dropped from the anchor.

Measured (Appendix A.3): a participant `User {userId: 42, tokenHash: 'h'}` survives `reset_all`
**entirely** — channel, thread, messages, cursors, customer, orders — while the status row reports
`userCount` excluding it. No error, no signal, and the presenter's roster does not list the
participant either, so nothing anywhere surfaces the leak.

And the conjunct buys nothing. `userId > ''` matches *every* row, so it has no selectivity: both
plans scan the whole label. At 52 `User`s the anchors profile as

| form | operator | records | time |
|---|---|---|---|
| `WHERE u.tokenHash IS NOT NULL` | `Node By Label Scan \| (u:User)` | 52 | **0.0036 ms** |
| `WHERE u.userId > '' AND …` | `Node By Index Scan \| (u:User)` | 52 | **0.055 ms** |

— a 15× slowdown at the anchor, identical results, against a query whose total is 215–236 ms.

**Suggested fix:** drop `u.userId > '' AND` from `reset_all`'s `WHERE` (§5). Keep G1 as the bare
`tokenHash IS NOT NULL` it is in `reset_participant`, so the two resets share one predicate and
neither reset's completeness depends on a data-type invariant nothing enforces. See F5 for the
roster half of the same idiom.

#### F2 — `scoped=false` is a *partial* reset, not a no-op, and `reset_all` then orphans the remainder permanently

§4 guards the thread re-mint on `ch IS NULL` (`FOREACH (c IN CASE WHEN ch IS NULL THEN [] …)`) and
describes the status row as *"`scoped=false` means G2 found no owned channel (a real anomaly worth
logging)"*. But the `Customer`/`Cart`/`CartItem`/`Order`/`OrderLine` block below it is **not**
guarded — it is anchored on `$participantId` alone, independently of `ch`.

Measured on a participant whose channel carries a mismatched `participantId`, and again on one with
no `MEMBER_OF` edge into its own correctly-marked channel: `reset_participant` returns
`scoped=false` and **deletes 8 nodes** — the entire commerce subgraph — while the thread, its three
messages, the run/step/trace subtree and both cursors stay. The participant loses their cart and
order history and keeps their transcript: the opposite of what "reset mine" means, reported as a
success row.

`reset_all` then makes it permanent: it deletes the `User` (it is a participant by G1) while G2
leaves the channel unmatched, so the `Channel` + `Thread` + 3 `Message`s + cursors survive with no
anchor any future reset can reach. Verified: after `reset_all`, `ch-D`/`ch-F` and messages
`p-ddd-m1..m3` are still present while `userCount: 3` reports success. That is an AC-2 leak
arriving through AC-5's control — one participant's transcript left visible in a graph the
presenter believes is clean.

**Reachability today is nil** and I want that on the record: `ensure_participant` writes the
`User`, `Channel`, marker and `MEMBER_OF` in one atomic query, and nothing in the codebase deletes
a `MEMBER_OF` edge. This is a defensive-consistency defect, not a live bug.

**Suggested fix, two lines:** wrap the commerce collection in the same `ch`-gate the re-mint uses
(or gate the whole victim list on `ch IS NOT NULL`), so `scoped=false` is a true no-op; and add
`AND ch IS NOT NULL`-equivalent gating to `reset_all`'s `users` list so an unresolvable participant
is left intact-and-collectable rather than beheaded. Whichever way it goes, §4's status-row
paragraph must state what `scoped=false` actually deletes.

#### F3 — §7 names the wrong orphan classes, and hands S7/S10 a done-condition that cannot fail

§7's table asserts that a turn still running after its run was deleted *"writes new
`Message`/`StepRun`/`TraceEvent` rows that no longer hang off anything the next reset can find —
orphans that survive both resets and accumulate."* Every one of those three writes is anchored on a
node the reset deleted, so it matches zero rows and **creates nothing**. Executed against a graph
immediately after `reset_all` (Appendix A.4):

| write (QUERIES.md) | anchor | result | nodes created |
|---|---|---|---|
| `post_message` first-path (§4) | `Thread` | 0 rows | **0** |
| `record_step_and_advance` (§12.2) | `WorkflowRun` | 0 rows | **0** |
| `append_trace_event` (§12.10) | `StepRun` | 0 rows | **0** |
| **`advance_cursor` (§9.3)** | **`mem` (User/Agent)** | 1 row | **1 orphan `ReadCursor`** |

So S7/S10's done-condition — *"a reset issued while a stub-LLM turn is in flight leaves no orphan
`StepRun`/`TraceEvent`/`Message`"* — is vacuously true and tests nothing, while the one writer that
really does orphan is unmentioned. §9.3 `MERGE`s on the **member**, not the thread, so it mints a
`ReadCursor` naming a thread that no longer exists; `rc.threadId IN threadIds` only ever covers the
threads being deleted *now*, so neither reset can collect it, ever. Verified: after
`reset_participant` → `advance_cursor(th-A)` → `reset_participant` → `reset_all`, the orphan
`p-aaa:th-A` is still there.

**Suggested fix:** (i) correct §7's table to the four rows above; (ii) re-word the S7/S10
done-condition to something that can fail — the quiesce is still right, it just protects a
*user-visible* outcome (a `ThreadNotFoundError` mid-turn, a turn that silently produces no reply),
not an orphan-node count; (iii) close the cursor gap in `reset_all` with one clause, which I ran:
`WHERE rc.threadId IN threadIds OR rc.memberId IN pids` — it collects the participant-owned orphan
and leaves every §4.8 survivor cursor untouched (`assistant:demo-welcome`, `u1:demo-welcome`,
`assistant:promo-thread` all intact). An `Agent`-owned orphan still escapes it; a fully complete
sweep needs `OPTIONAL MATCH (t:Thread {threadId: rc.threadId}) … WHERE t IS NULL` (parses and runs
on this build; `EXISTS { … }` and `exists((pattern))` both **fail to parse** — see A.5), which
widens `reset_all`'s contract to "collect every orphan cursor". That widening is a `graph-dba`
judgment call, not mine.

#### F4 — §2.3's variant A does not reproduce against the shipped query text

§1 rests on both guards being *"live-proved load-bearing in §2.3"*, and §12 tells S4 **"Do not
relax either guard for readability. §2.3 is what each one costs."** Variant A's published row (23
deleted, `demo-welcome` **GONE**) is not reproducible from the shipped query minus G2 on a graph
carrying `bootstrap_schema.sh`'s constraints.

With G2 removed, `ch` binds to **every** channel the participant is a member of, and — because the
`WITH u, ch, collect(…)` groups by `ch` — the query row-multiplies: one row per channel. The
re-mint `FOREACH` then fires once per row with the same `$newThreadId`, and the query raises
`unique constraint violation on node of type Thread` and writes **nothing** (verified with the
2-channel and 3-channel adversarial participants; identity and label deltas both empty). I
reproduced the published 23-node delete **only** after dropping the `Thread.threadId` UNIQUE
constraint (Appendix A.2) — which the note's own §7 proves was present on its probe graph.

The conclusion is unaffected — G2 is obviously load-bearing, and variant B and the CONTROL both
reproduce exactly (6 and 17 nodes deleted). But an implementer or re-reviewer who re-runs §2.3 as
published gets an error on row A, and §12's instruction cites that row as the guard's price.

**Suggested fix:** re-run variant A and publish what it actually does, or state that the ablation
was run without the re-mint clause. Either way the note gains a fact it currently does not
document: **`reset_participant` produces one row per matched channel**, and two channels carrying
the same `participantId` therefore make the reset raise permanently for that participant (verified;
`reset_all` still cleans up). That is a fail-safe, and §4 should say so.

### Minor

#### F5 — §8's roster recommendation is a measured regression, not an asymptotic improvement

§8's hand-off to S4 says the conjunct *"keeps the roster O(index) as the `User` label grows."* It
does not: `userId > ''` matches every row, so the "index scan" is a full index traversal and both
plans are O(|User|). Measured at 52 users, 20 runs each: bare form **0.045 ms** server-side
(`Node By Label Scan`, 52 records); conjunct form **0.078 ms** (`Node By Index Scan`, 52 records).
Identical result sets.

The idiom is not fragile *as a read* — it is exactly QUERIES.md §3's `WHERE c.channelId > ''`
precedent, and a missing row in a roster is cosmetic. It is simply not worth taking. **Suggested
fix:** drop §8's hand-off recommendation and §12's "Take §8's roster change" bullet, or keep the
conjunct and correct the justification to "matches §3's house idiom", with the O(index) claim
removed. This is the direct answer to the gate question: *sound, not a trick, and pointless here —
and actively harmful in `reset_all` (F1), where the same non-string `userId` turns a cosmetic
omission into a silent under-delete.*

#### F6 — reset completeness silently depends on S3's `dev_surface=False`

§4 argues `WorkflowRun` is safely reached through `TRIGGERED_BY` because
`trigger.maybe_trigger → start_workflow_run` is the storefront's only run-creation path. CPG-
verified and true: `Services.start_workflow_run` has exactly two non-test callers —
`falkorchat/trigger.py:76` and `falkorchat/api.py:394` — and `Repository.start_run_untriggered` is
called from exactly one place, inside `Services.start_workflow_run` (`services.py:2040`). A run
started through `api.py:394` without a trigger message carries **no `TRIGGERED_BY` edge** and is
unreachable by both resets, along with its `StepRun`s and `TraceEvent`s.

That path is closed in the storefront deployment only because §4.9 move 1 leaves
`api.build_router` unmounted — i.e. S0's reset completeness depends on **S3**. The note attributes
it to "this demo never uses the process path", which is a convention, not the structural property
that actually holds. **Suggested fix:** one sentence in §4 and in the `QUERIES.md` §18 entry naming
`dev_surface=False` as the reason, so a future deployment that re-mounts the dev surface knows it
has re-opened an under-delete.

#### F7 — `ensure_participant`'s `existed` row conflates "existing participant" with "existing non-participant `User`"

§3's status-row table calls `created=false, existed=true` *"idempotent success … the row returns the
**stored** ids/language."* It does so for **any** `User` with that id. Executed against
`seed_demo.sh`'s tokenHash-less `u1`:

```
ensure_participant(participantId='u1', …) → [created=false, existed=true, collided=false,
                                             agentMissing=false, channelId=null,
                                             threadId=null, language=null]   # nothing written
```

A caller following the table treats that as success and builds a `ParticipantRecord` with
`threadId=None` and a token the graph never stored — `resolve_token` will then reject it. The same
row also hides a second case: a replay with a **fresh** `$tokenHash` returns `existed=true` and the
new hash is silently not written (verified: a replay with new `channelId`/`threadId` returns the
stored ones and writes nothing — correct read-through, but only when the caller's token already
matches).

Reachability is essentially nil (`participantId = "p-" + uuid4().hex`), so this is a contract
defect, not a bug. **Suggested fix:** return `existing.tokenHash IS NOT NULL AS existedParticipant`
alongside `existed`, and give the table a fifth row — *id exists but is not a participant → refuse,
member-id collision* — matching how the `collided` row already behaves.

#### F8 — the atomicity claim is correct, but its failure boundary is client-side and unstated

§7's *"Each reset is one `GRAPH.QUERY`, therefore atomic"* holds, and I strengthened it: the
server's `TIMEOUT 1000` (module arg, live) applies **only to reads**. A 1.67 s write completed
untouched while a 1.0 s read was killed with `Query timed out` (A.6). So a slow `reset_all` will
never be truncated server-side.

The real boundary is the client: `falkorchat/config.py:29` sets
`FALKORDB_SOCKET_TIMEOUT` default **10 s**, passed to the `FalkorDB` constructor
(`falkorchat/db.py:44`). At 236 ms that is 42× headroom, but if `reset_all` ever crosses it the
client raises `TimeoutError` **while the server commits the delete**. §7 promises S7/S10 "the query
is all-or-nothing", and §4.8/§5.2 map a reset failure to `503 … nothing changed`. **Suggested fix:**
one line in §7 — a client-side timeout on a reset means *unknown*, not *nothing changed*; the caller
must re-read state rather than report "no change".

#### F9 — an off-chain `Message` survives both resets

§4 argues correctly that the `HEAD → NEXT*0..` walk "cannot escape the thread". It never states the
converse: a `Message` carrying `threadId` but not linked into the chain is invisible to both
resets. Verified: a hand-planted `orphan-1 {threadId:'th-A'}` survives `reset_participant` intact
and would outlive `reset_all` with its thread gone.

QUERIES.md §4's v2 write paths link `HEAD`/`NEXT`/`TAIL` inside the same guarded `FOREACH`, so this
is unreachable — which is exactly why it is worth one sentence in §4 rather than a query change:
the structural walk's safety is bought against a chain invariant, and naming that invariant is what
keeps a future writer from breaking it.

### Nit

- §11's environment paragraph says the probe graph was *"deleted at the end of the unit"*.
  `ws:probe-s0-reset` still exists as a graph key holding 0 nodes (read-only check). Cosmetic, but
  §11 is the note's provenance section and it is the one line in it that is not literally true.

---

## 3. Where I agree, explicitly

The gate asked four yes/no questions. My answers, with what I checked.

- **DDL verdict `NO` (§9) — agree.** Every anchor is already indexed and §8's plans prove each is
  used. On the rejected `UNIQUE` on `Channel.participantId`: the "bites on re-join" hazard does
  **not** materialize on this build — I created the constraint and verified that FalkorDB exempts
  both an **absent** property and an explicit `null` (two `Channel`s of each create cleanly), that
  a genuine duplicate is rejected, and that delete-then-recreate with the same `participantId` is
  clean. So the constraint would have been *safe*; rejecting it is a scope/proportionality call —
  a `bootstrap_schema.sh` change plus an existing-workspace migration for no measured gain — which
  is precisely the reason §9 gives. Correct verdict, correct reasoning.
- **The unshipped 2.2× `reset_all` lever (§8) — agree, and it is worse than the note says.** The
  fast form takes **two** nullable dependencies, not one: `tm.threadId IN tids` on the deliberately
  unindexed `Message.threadId`, *and* `r.startedAt >= 0`, which drops any `WorkflowRun` whose
  `startedAt` is absent. Either null under-deletes silently. 136 ms once or twice per demo, behind
  a stop-intake-and-drain quiesce, is not worth either.
- **The `ReadCursor` label scan over a structural walk (§4) — agree on the trade, correct the
  claim.** Measured 99 records / 0.012 ms at 50 participants (note: 102 / 0.031 ms) — the cost is
  real but negligible, and the walk-from-members alternative genuinely would miss the `Agent`'s
  cursor. What is overstated is *"an orphan cursor is a real defect"* implying the form catches
  them: it catches cursors whose **owner** left, not cursors whose **thread** is already gone,
  which is the class the platform actually produces (F3).
- **§8's `GRAPH.PROFILE` numbers — spot-checked, they stand.** On my own 50×40 rebuild (52 `User`,
  2004 `Message`, 200 each of `WorkflowRun`/`StepRun`/`TraceEvent`, 103 `ReadCursor`, 101 `Order`):
  `reset_all` **215 ms** wall / 3064 nodes / 3253 rels (note: 236 ms / 3188 / 5345 — my fixture has
  fewer edges per message); `reset_participant` 8.8 ms server-side on a 4-run participant (note:
  4.5 ms on a 1-run one), anchoring on `Node By Index Scan | (u:User)`, 1 record, with the single
  `ReadCursor` label scan as documented; `read_thread`, `get_customer_current_order` and
  `order_belongs_to_customer` all anchor on `Node By Index Scan`, no label scan. Same shape, same
  order of magnitude.

## 4. What's solid

- **The core decision — a provenance marker instead of id-equality (§1.1) — is right, and it is
  right for the reason the note gives.** `Repository.create_channel` has no caller-controlled
  property surface, and no code path deletes a `MEMBER_OF` edge, so the marker cannot be forged or
  stripped through the shipped API. Every attack I could construct required a hand-written `SET`.
- **N1 is demonstrated, not asserted.** Variant B reproduces byte-identically on an independent
  fixture: `Channel 5, Thread 5, Message 10, ReadCursor 7, User 5` after the shipped control reset
  and after the guard-free destructive run — identical in all five labels — while
  `demo-welcome`, its three messages and both its cursors are gone in one and intact in the other.
  Every "assert survivors by label" check passes on the destructive run. §12's positive
  identity-assertion done-condition for S4 is the correct and necessary response.
- The reads and the two order primitives behave exactly as documented — including the tie-break
  (`p-aaa-o3` beats `p-aaa-o2` at equal `placedAt`), the all-`null` placeholder from `collect()` on
  a zero-line order, `sum()` returning **type 5 (Double)** even on the empty aggregation, and all
  five `order_belongs_to_customer` cases. The ownership gate's "not optional" framing in §10.2 is
  correct.
- Thread-scoped-not-author-scoped, `WorkspaceConfig` as the must-survive, `Document`/`Chunk`/
  `Entity` surviving *both* resets, the `Agent` node surviving while its `MEMBER_OF` edges go with
  `DETACH DELETE`, the empty-`UNWIND` guards, `FOREACH`-over-`[]` instead of `UNWIND`, and the
  re-mint-before-collect ordering all verified correct on my fixture. Both resets are idempotent;
  `reset_all` on a clean graph returns one all-zeros row.
- §6's framing — *"documentation, **not** the mechanism … it is **not** what S4 should assert on"* —
  is the single most valuable sentence in the note, and §12's hand-off carries it through.

## 5. Open questions

1. **Does the storefront advance read-cursors at all?** §5.2's `GET /shop/api/messages?since=<ms>`
   reads as an explicit-`since` read, not cursor mode, and `ensure_participant` creates no cursor —
   yet §2.2 reports two cursors per participant. If the storefront never advances a cursor, F3's
   orphan is a platform-wide `QUERIES.md` §18 concern only, and the cursor block in both resets is
   pure defence. `graph-dba`/`architect` to confirm which it is; it changes F3's priority, not its
   correctness.
2. **`MAX_QUEUED_QUERIES` under the reset.** The live module runs with `MAX_QUEUED_QUERIES 25`.
   `reset_all` is a ~236 ms stop-the-world write on `ws:demo`; 50 participants polling
   `GET /shop/api/state` every 2 s, at three graph reads per poll, is ~75 queries/s, so ~18 queue
   during the write — under the cap, but not by much, and §4.4's other traffic is on top. I did not
   execute this (it needs concurrent load against the shared instance, out of this review's
   boundary). Recommend S15's load harness assert it explicitly rather than S0 reasoning about it.

---

## Appendix

### A.1 — Reproduction recipe

All work was done on `ws:probe-analyst-s0`, `ws:probe-analyst-s0-scale`,
`ws:probe-analyst-s0-noconstraint`, `ws:probe-analyst-timeout` and `ws:probe-analyst-uniq`. **All
five are deleted** (`GRAPH.LIST` filtered on `analyst`/`timeout`/`uniq` → empty). To rebuild:
create the `bootstrap_schema.sh` index+constraint set for
`User.userId`, `Channel.channelId`, `Thread.threadId`, `Message.msgId`, `WorkflowRun.runId`,
`StepRun.stepRunId`, `TraceEvent.traceId`, `ReadCursor.cursorId`, `Customer.customerId`,
`Order.orderId`, `Agent.agentId`; seed `demo-general`→`demo-welcome`→3 chained messages + 2
cursors, a second non-participant channel `shared-promo`→`promo-thread`→1 message + 1 cursor,
`u1`/`u2` (no `tokenHash`, `u2` carrying `channelId:'demo-general'` + a `MEMBER_OF` edge),
`WorkspaceConfig`, `Document`/`Chunk`/`Entity` with a `DERIVED_FROM` into `dm1`, snapshot+`Step`,
and `u1`'s `Customer`/`Cart`/`CartItem`/`Order`/`OrderLine`; then participants
`p-aaa`/`p-bbb`/`p-ccc` (each `User`+`Channel {participantId}`+`Thread`+3 chained messages
including one `Agent`-authored+2 cursors+`WorkflowRun`/`StepRun`/`TraceEvent`+`Customer`/`Cart`/
`CartItem`/2 `Order`s/3 `OrderLine`s), with `p-ccc` a genuine `MEMBER_OF` of **both**
`demo-general` and `shared-promo`. Adversarial extras: `p-ddd` whose channel carries
`participantId:'p-eee'`; `p-fff` with no `MEMBER_OF` into its own marked channel; `p-ggg` with two
channels both carrying `participantId:'p-ggg'`; a `User {userId: 42, tokenHash:'h'}`; a
`Message {msgId:'orphan-1', threadId:'th-A'}` off the chain.

### A.2 — §2.3 re-run (F4, and the variant-B confirmation)

Same fixture, same shipped text, one guard removed at a time:

| variant | called with | result on a constraint-carrying graph |
|---|---|---|
| CONTROL (both guards) | `p-ccc` | `scoped=true`, 17 nodes deleted, **every named survivor intact** (identity deltas: none) |
| A — G2 removed | `p-ccc` (2 channels) | **`ERROR: unique constraint violation on node of type Thread`** — 0 deleted, 0 created |
| A — G2 removed | `p-ccc` (3 channels) | same error, 0 deleted |
| A — G2 removed, **no `Thread` UNIQUE** | `p-ccc` (2 channels) | 2 status rows, `Nodes created: 2`, **`Nodes deleted: 23`**, `demo-welcome` GONE — the note's published row |
| B — G1+G2 removed | `u2` | `scoped=true`, **6 nodes deleted**, `demo-welcome` + `dm1..dm3` + both cursors GONE |

Control-vs-B label counts, measured from the same baseline
(`Channel 5, Thread 5, Message 13, ReadCursor 9, User 5`):

```
CONTROL (shipped, p-ccc)  : Channel 5, Thread 5, Message 10, ReadCursor 7, User 5
VARIANT B (no guards, u2) : Channel 5, Thread 5, Message 10, ReadCursor 7, User 5
identical? True
identity deltas B-vs-control: demo-welcome 1→0 · dm1/dm2/dm3 3→0 · chain 3→0 · cursors 2→0
```

### A.3 — F1 evidence

```
MATCH (u:User) WHERE u.tokenHash IS NOT NULL                RETURN collect(u.userId)  → [42, p-bbb, p-aaa]
MATCH (u:User) WHERE u.userId > '' AND u.tokenHash IS NOT NULL RETURN collect(u.userId) → [p-aaa, p-bbb]
MATCH (u:User) RETURN u.userId, u.userId > ''               → [42, null] ['p-aaa', true] …
reset_all()  → userCount 2   |  MATCH (u:User) WHERE u.tokenHash IS NOT NULL → [42]   ← survivor
```

Anchor comparison inside `reset_all` on the 50×40 fixture, identical status rows both ways:

```
shipped (userId > '')   Node By Index Scan | (u:User) | Records produced: 52 | 0.054657 ms
without the conjunct    Node By Label Scan | (u:User) | Records produced: 52 | 0.003615 ms
```

### A.4 — F3 evidence (post-`reset_all` writes)

```
post_message first-path  → []          | Message {msgId:'zz1'} count = 0
record_step_and_advance  → []          | StepRun  count = 0
append_trace_event       → []          | TraceEvent count = 0
advance_cursor (§9.3)    → ['assistant:th-A']   ← creates an orphan ReadCursor
```

Proposed one-clause fix, run end-to-end (`reset_participant` → `advance_cursor` on the dead thread
for both a participant and the `Agent` → `reset_all`):

```
shipped                       cursorCount 2 | left: assistant:th-A, p-aaa:th-A, assistant:promo-thread,
                                                    assistant:demo-welcome, u1:demo-welcome
… OR rc.memberId IN pids      cursorCount 3 | left: assistant:th-A, assistant:promo-thread,
                                                    assistant:demo-welcome, u1:demo-welcome
```

### A.5 — Pattern-predicate support on this build (FalkorDB v4.18.11 / module 41811)

```
WHERE NOT EXISTS { MATCH (t:Thread {threadId: rc.threadId}) }  → parse error
WHERE NOT exists((:Thread {threadId: rc.threadId}))            → parse error
OPTIONAL MATCH (t:Thread {threadId: rc.threadId}) WITH rc, t WHERE t IS NULL … → works
```

### A.6 — F8 evidence (server-side `TIMEOUT` applies to reads only)

Live config: `TIMEOUT 1000`, `TIMEOUT_DEFAULT 0`, `TIMEOUT_MAX 0`, `MAX_QUEUED_QUERIES 25`.

```
write  UNWIND range(1,20000000) AS i … CREATE (:Blob …)   → COMPLETED in 1.67 s, 4 rows
read   MATCH (a:Blob),(b:Blob),(c:Blob) RETURN count(*)   → ERROR "Query timed out" at 1.00 s
```

---

## Pass 2 — 2026-09-02, against `docs/plans/salesperson-ui-graph.md` v1.1 (941 lines)

**Verdict: needs changes** — on one defect, and it is a **two-newline fix**. Every substantive
thing in v1.1 is approved: all nine Pass 1 findings are genuinely fixed (not papered over), both
self-caught defects are real and correctly repaired, and the three claims the coordinator asked me
to re-derive all reproduce. The blocker is that the document does not parse as markdown, which for
a note whose contract is *"S4 implements the printed text verbatim"* is a shipping defect rather
than a cosmetic one. Fix P1 and this is an approve.

**Method:** I re-executed the **printed text** this time rather than a transcription — every
`cypher` fenced block extracted from the file and run verbatim against a fresh fixture on my own
`ws:probe-analyst-p2` / `-p2-scale` (both deleted; `ws:acme` re-confirmed at 2 `Channel` /
2 `Thread` / 52 `Message` / 544 `Entity`; `ws:probe-s0-reset`, `ws:probe-s0r2`,
`probe_u8_rename_dst`, `ws:s1v6` untouched; pytest not run). The fixture carries v1.1's three new
shapes (`p-ddd` mismatched marker, `p-eee` no `MEMBER_OF`, off-chain `orphan-1`) plus one v1.1
does not have: a cross-member participant who **owns a read-cursor on `demo-welcome`** (P2).

`CPG: considered, not relevant — Pass 2 re-derives Cypher behaviour by execution against a live
probe graph; the one source-level claim (F6's caller set) was settled by CPG in Pass 1 and is
unchanged in v1.1.`

### Does the printed Cypher execute verbatim?

**The queries do; the document does not deliver them.** Extracted with a tolerant non-greedy fence
regex (a `cypher` opener, first following backtick run as terminator) — which is what a passing
"closed loop" must have used — all six blocks come out clean and **all five runnable ones executed
first time**:
`ensure_participant` → fresh-participant row; `reset_participant('p-aaa')` → `scoped=true`,
`deletedCount 17`; `reset_all_participants` → full status row incl. `unscopedCount`;
`get_customer_current_order` → the `placed` order, 2 lines, `total 17.5`;
`order_belongs_to_customer` → `[true, 'delivered']`. The §8 tuning-lever block is a fragment and
correctly fails standalone (`Query cannot conclude with WITH`). `--` re-confirmed as a parse error
(`MATCH (u:User) -- G1` → `Invalid input 'G'`), `//` fine. So the comment fix is real and the
logic is sound. What is not sound is the fencing — see P1.

### New findings

#### P1 (Blocker) — two closing fences are glued to the last line of code, so the note does not parse as markdown

Line 347 ends with `AS cartItemCount` immediately followed by a three-backtick run on the **same line**;
line 496 does the same after `unscopedCount`.
A CommonMark closing fence must be a line containing only backticks, so neither closes its block.
Six opening fences (177, 280, 436, 731, 776, 814) meet only four standalone closers (209, 736,
786, 818). Parsed with `markdown-it` (`commonmark` preset, tables enabled):

| | fenced blocks | tables rendered | largest block |
|---|---|---|---|
| **as published** | **4** | **7** | **455 lines** (map `[279, 736]`) |
| after un-gluing the two fences | 6 | 11 | 67 lines |

So as published, one code block runs from §4's `reset_participant` to §8, swallowing **all of §5
(including the `reset_all` query itself), §6's keep/delete inventory, §7's quiesce contract and
most of §8** — and **four of the note's eleven tables do not render**, one of which is the
per-label keep/delete inventory S0's scope row explicitly mandates. `reset_all_participants` is
not an extractable block at all under a conformant tool, and anyone copy-pasting §4's query out of
the rendered page gets a stray backtick run appended to the query text, and a parse error.

This also qualifies the closed-loop claim: the loop passed because its extractor tolerated a glued
fence, not because the fences are correct. A loop that re-parses the note the way a reader's
renderer does would have caught it.

**Fix:** insert a newline before each of the two glued fences. Two characters. Worth adding to the
loop: assert the extractor finds **six** `cypher` blocks and that none exceeds ~70 lines — a
runaway block is exactly what a length assertion catches.

#### P2 (Minor) — §6's inventory has no row for the case F3's new sweep actually adds

`OPTIONAL MATCH (u)-[:HAS_CURSOR]->(own:ReadCursor)` is owner-scoped, not thread-scoped, so it also
collects a participant-owned cursor naming a **live, non-participant** thread. Verified with a
cross-member participant holding `p-ccc:demo-welcome`: `reset_participant('p-ccc')` returns
`cursorCount: 3`, `deletedCount: 18` (v1.0's was 17) and `demo-welcome`'s cursor set goes from
`[p-ccc:demo-welcome, assistant:demo-welcome, u1:demo-welcome]` to `[assistant:demo-welcome,
u1:demo-welcome]`. `reset_all` does the same.

**The two resets differ in whether that is right, and the note treats them alike.** For `reset_all`
it is *correct and an unnamed v1.0 bug fix*: the `User` is deleted, so without the sweep that
cursor became an orphan — a class neither of us named in Pass 1. For `reset_participant` the
participant survives, nothing is orphaned, and deleting it is a deliberate widening — it resets
their read position on a thread §4.8 protects. Benign in the storefront (the surfaces that read
cursors are unmounted, review OQ-1), but §6 carries three `ReadCursor` rows and none of them is
this one, while §2.2's control row still reads "its 2 cursors … intact".

**Fix:** add the row — *`ReadCursor` (participant-owned, naming a live non-participant thread) →
deletes / deletes* — and one clause saying it prevents an orphan in `reset_all` and is a
deliberate widening in `reset_participant`.

#### P3 (Minor) — `unscopedCount`'s loudness is asserted in prose, not fixed in a contract

An unscoped participant keeps a **working `tokenHash`** and their whole transcript — verified after
`reset_all`: `[[p-ddd, true, th-D], [p-eee, true, th-E]]`, 6 messages still on `th-D`/`th-E`. They
are not merely "not reset"; they are an *active* participant with a valid token during the next
demo run. §5 says "S10 must surface it" and §12 says "log/alert, never report success" — the right
instruction, but it leaves S10 free to return `200 {clearedParticipants: 49}` with the count
buried, which reproduces exactly the exposure F2 exists to prevent.

**Fix:** make it a contract in §12's hand-off list, not prose in §5 — `unscopedCount > 0` ⇒ the
presenter route returns a non-2xx, **or** a 200 carrying a required `incomplete: true` that the SPA
must render as a banner naming the count.

#### Nits

- **§8's roster timings are inside measurement noise.** 20 runs each at 52 `User`s, client-side
  medians: bare **0.347 ms**, conjunct **0.333 ms** — reversed from the note's 0.082/0.128 and
  within noise either way. The conclusion is right and rests on **correctness** (F1), not speed;
  drop the "~1.6× regression end-to-end" figure rather than defend a number that does not
  reproduce.
- **§2.1 calls `p-ddd`/`p-eee` "the two ways G2 can *legitimately* fail to resolve".** §1.1 line 76
  already establishes both states are unreachable through the shipped API. "Legitimately" reads as
  though they occur in normal operation, which is precisely what makes F2's FR-7 trade-off look
  costlier than it is — see the ruling below. Change the word and cross-reference §1.1 from §5's
  trade-off bullet.

### Ruling on F2's FR-7 trade-off (asked for explicitly)

**Graceful degradation, not a requirements breach. I would not take it to the stakeholder.**

The reasoning turns on one fact the note has but does not deploy here: **the unscoped branch is
unreachable.** §1.1 line 76 establishes it — `ensure_participant` writes `User`+`Channel`+marker+
`MEMBER_OF` in one atomic query, `create_channel` cannot set the marker, and no query anywhere
deletes a `MEMBER_OF` edge. So on a healthy graph `unscopedCount` is always `0` and **both** the
v1.0 and v1.1 behaviours are dead branches. The question is only which failure is better on a graph
that is *already* corrupt:

- **v1.0** satisfied FR-7 nominally (token invalidated, client bounced) while stranding the
  participant's `Channel`+`Thread`+`Message`s **permanently uncollectable**, reported as success —
  a **permanent, silent AC-2 leak**, in the graph §4.8 calls the most expensive mistake available.
- **v1.1** misses FR-7 for that one participant and reports it — the leftover stays **collectable**
  (repair the edge or the marker, re-run `reset_all`).

AC-2 is the stronger requirement here — §4.3 spends five parts on it — and a recoverable, counted
miss beats an unrecoverable, silent one. The trade is correct. Two conditions make it safe rather
than merely arguable, and both are cheap: P3 (make the counter a contract, not a hope), and a
reversal trigger — **if any future change introduces a way to detach a participant from their own
channel, this flips from a dead branch to a live FR-7 hole and must be revisited.** Say that in §5
next to the trade-off bullet.

### Assessment of F3's substitution and decline

**Substitution — accept, it is better than what I proposed.** `(u)-[:HAS_CURSOR]->(own)` is a
traversal off the already-bound `u` rather than my `rc.memberId IN pids` property comparison: same
coverage, no dependency on `memberId` being written (it is only set `ON CREATE`), and no extra
scan. Verified end to end — after a reset, minting `p-aaa:th-A` and `assistant:th-A` against the
dead thread, the next `reset_participant` reports `cursorCount: 1` and leaves
`[assistant:th-A]` (the documented `Agent` residual) while all three `demo-welcome` cursors stay.
The `tcur + [x IN ocur WHERE NOT x IN tcur]` merge de-duplicates correctly: a cursor in both sets
is counted and deleted once (`cursorCount: 2` on a plain `p-aaa` reset, no double `DETACH DELETE`).

**Decline of the complete `t IS NULL` sweep — accept, and the stated reason reproduces.** I planted
a non-participant dangling cursor (`u1:ghost`) and an `Agent` one (`assistant:ghost`); the complete
sweep's candidate set is `[assistant:ghost, u1:ghost]` — it is **owner-blind**, so it reaches
outside §4.8's "reachable from a participant `User`" rule. Declining it in a reset and recording it
for a future workspace-GC job is the right call, and recording *why* is what makes it a decision
rather than an omission.

### Disposition of Pass 1 findings

| # | Disposition | Evidence I re-checked |
|---|---|---|
| **F1** | **Fixed.** | Conjunct gone from both resets and the roster; §12 carries an explicit "Do NOT add `u.userId > ''` anywhere". Direct before/after on a participant with a numeric `userId` and numeric marker: v1.1 text → `userCount 4`, `ch-NUM` and its messages **gone**; v1.0 text on the same fixture → `userCount 3`, `42` still in the participant list with its channel intact. |
| **F2** | **Fixed, both halves.** | `reset_participant` on `p-ddd` and `p-eee`: `scoped=false`, **zero write statistics of any kind**, empty label and identity deltas, all per-class counts `0`, `threadId` returned unchanged. `reset_all`: `unscopedCount: 2`, both left whole (`User`+`Channel`+3 `Message` each), `userCount: 3`. Idempotent re-run keeps reporting `unscopedCount: 2`. Trade-off ruled on above. |
| **F3** | **Fixed; the declined part is correctly declined.** | §7's table now matches what I measured; the four-part done-condition replacement is testable, and (d) is the right direct test. Substitution and decline assessed above. |
| **F4** | **Fixed, and better than I had it.** | Both rows now published. Variant A verbatim → `unique constraint violation on node of type Thread`, empty identity delta. Variant B verbatim on the **v1.1** text → runs clean, 6 deleted; CONTROL and B label counts identical (`Channel 7, Thread 7, Message 17, ReadCursor 10, User 7`) while five identity checks differ (`demo-welcome`, `dm1-3`, the chain, both `demo-welcome` cursors). **The "no ablation needed" claim is confirmed** — B is now the sole and sufficient demonstration of N1. |
| **F5** | **Fixed.** | Recommendation withdrawn in §8 and §12; the O(index) claim is retracted with the no-selectivity reason. See the nit above on the timing figure. |
| **F6** | **Fixed.** | §4 now attributes run-reachability to `dev_surface=False` (S3) rather than to convention, and §12 routes it into the `QUERIES.md` §18 entry. |
| **F7** | **Fixed.** | `existedParticipant` present; all five status rows executed — row 3 (`u1`) returns `existed=true, existedParticipant=false` with null ids and writes nothing; `u1` unmodified; rows 3/4/5 write no `Channel` and no `User`. The added "not a token-rotation path" paragraph is the right extra. |
| **F8** | **Fixed.** | §7 now separates the quiesce timeout from the socket timeout and tells S7/S10 to treat the latter as *unknown*; §12 repeats it. |
| **F9** | **Fixed.** | §4 names the `HEAD`/`NEXT` chain invariant and §6 has the off-chain row; `orphan-1` survives both resets on my fixture, as documented. |
| **Nit** | **Fixed.** | §11 corrects the disposal claim and names `ws:probe-s0-reset` as still present at 0 nodes. |

### The two self-caught defects

- **`--` comments — real, and the fix is complete.** Re-confirmed the parse error independently.
  Every guard comment in the note is now `//`, and every printed block executed. One consequence
  worth stating for the record: **my Pass 1 validation was of a hand-transcription using `//`**, so
  it tested the intended logic, not the published text. Pass 2 closes that gap by executing the
  published text itself — the logic is the same, plus v1.1's changes.
- **The `tcur` collapse — real, reproduced, and the fix is genuinely free.** At 50×40 (2 004
  `Message`, 102 `ReadCursor`, 200 runs), server-side, three runs each: shipped
  **224.7 / 225.6 / 230.1 ms**, 3 250 nodes deleted (note: 235/239/247, 3 250 — same numbers).
  Re-introducing the un-collapsed form measured **665.5 / 675.3 ms** with identical delete counts
  (note: 684–692). `reset_participant` at scale: **4.3 / 4.6 / 5.1 ms**, 63 nodes / 67 rels (note:
  4.0/4.0/6.0, 63 nodes). So F1's anchor change and F3's added sweep are both free, as claimed.

### What v1.1 does notably well

It re-derived rather than accepted. F1's type semantics were re-established from scratch
(`42.5 > ''`, `true > ''`, `'' > ''` — cases I had not run) instead of quoting me; F3 replaced my
proposed clause with a strictly better one and said why; F4 disclosed an undisclosed ablation
rather than quietly re-running it; and F2's answer changed the *status-row contract* so a partial
cannot read as success, which is a deeper fix than the one I asked for. Two defects were found by
re-verification that neither pass would have caught by reading — the `--` comments and the `tcur`
regression — and both are documented with the measurement that proves them. §7's four-part
done-condition replacement is the single best change in the revision: it turns a criterion that
could not fail into four that can.

---

## Pass 3 — 2026-09-02, against v1.2 (closeout)

**Verdict: approve.** P2's narrowing and P3's contract both hold under execution. No new findings.
Probe graph `ws:probe-analyst-p3` created and deleted; `ws:acme`, `reference`, `ws:test` and the
spent probes untouched; pytest not run.

**P1 — confirmed in passing, one command.** Strict CommonMark scan of the finished file: 6 `cypher`
blocks, lengths `[31, 77, 60, 4, 9, 3]`, and **13 tables render** (v1.1: 4 blocks, 7 tables). The
pre-execution shape gate plus its negative control is the right shape — it tests the gate, not just
the file.

### P2 — the narrowing is correct, including under-delete

Fixture: `p-ccc` (cross-member) holding **five** cursors — its own live thread, a dead thread, the
surviving `demo-welcome`, *another participant's live thread*, and one with a `null` `threadId`.

| cursor | after `reset_participant('p-ccc')` | |
|---|---|---|
| `p-ccc:th-C`, `assistant:th-C` | gone | thread deleted now (`tcur`) |
| `p-ccc:th-gone` | **gone** | **(a) F3's orphan class is genuinely still collected** |
| `p-ccc:nullt` (`threadId` unset) | gone | `liveT IS NULL` catches it too — a small bonus |
| `p-ccc:demo-welcome` | **kept** | live read-state, see (c) |
| `p-ccc:th-A` (another participant's live thread) | kept | live thread, owner survives — not an orphan |
| `u1:demo-welcome`, `assistant:demo-welcome` | kept | §4.8 survivors, untouched |

With the note's own three-cursor shape the numbers reproduce **byte-for-byte: `cursorCount: 3`,
`deletedCount: 18`.**

**(b) Nothing left unowned after `reset_all` — and the check is stronger than the one reported.**
`MATCH (rc:ReadCursor) WHERE NOT ()-[:HAS_CURSOR]->(rc)` returns empty, as claimed; so does the
*dangling* test (`OPTIONAL MATCH (t:Thread {threadId: rc.threadId}) … WHERE t IS NULL`). The
`Agent`-owned residual §7 documents does not appear here because the `Agent`'s cursors for
just-deleted threads are caught by the thread-scoped half; it only survives when the thread died in
an *earlier* reset. Worth one clause in §7 — the residual is narrower than stated. `unscopedCount:
2` / `unscopedIds: [p-ddd, p-eee]`, and `unscopedCount` is now `size(unscopedIds)`, so the two can
no longer disagree.

**(c) The surviving `demo-welcome` cursor is live read-state, not leaked cross-participant state.**
After the reset, `p-ccc` still exists, is still `MEMBER_OF [ch-C, shared-promo, demo-general]`, and
`demo-welcome` is still alive with its 3 messages. The node is owned by that participant's own
`User` and names a thread §4.8 *mandates* survives; its payload (`lastReadMsgId`) points into a
shared non-participant thread the participant legitimately belongs to — no other participant's data
is reachable through it. So keeping it is right, and deleting it (v1.1) was the wider behaviour.
`p-ccc:th-A` is the only shape that would carry cross-participant state, and it is unreachable —
§4.3 resolves thread ids server-side from the token, so a participant cannot read another's thread
to acquire a cursor on it — and it self-heals: once `th-A` dies, the cursor becomes dangling and the
`liveT IS NULL` branch collects it on the next reset.

**Zero-cursor raise — fixed, four ways.** `_AR_EXP_UpdateEntityIdx` does not reproduce for a
participant with no cursors at all (`cursorCount: 1`), a second reset with cursors already gone
(`cursorCount: 0`), or the `scoped=false` path with and without cursors. The collect-then-guard
rewrite holds.

### P3 — implementable as written

Every field the five rows key on is present and correctly typed in the status rows I executed:
`scoped=false` arrives as a row with `deletedCount: 0` (so `409` is expressible, and the caller
already holds `$participantId` for the body); zero rows remains the G1 rejection; `unscopedIds`
comes back as a list of strings, so `unresolved` needs no second query; the clean path is
`unscopedCount == 0`; and the duplicate-marker violation propagates as an exception rather than a
status row, so "propagate as `5xx`, do not retry" is the only thing a caller *can* do. No gap.

### Closing

Three passes, four majors and a blocker, all closed by re-derivation rather than assertion — and in
each round the revision found something neither of us was looking for (the `--` comments, the `tcur`
regression, the null-alias raise). The Cypher is safe to implement verbatim.
