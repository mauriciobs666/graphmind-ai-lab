# The one salesperson UI — graph design note (S0)

> **Status:** active · **Owner:** `graph-dba` · **Tracks:** — (M<n> TBD) · **Version:** 1.2 · **Reviews:** `docs/reviews/salesperson-ui-graph.md`

*2026-09-02 — v1.2: revised against that review's `## Pass 2` (needs changes). **P1 (blocker): two closing code fences were glued to their last code line**, so a conformant CommonMark parser saw 4 blocks instead of 6 and swallowed §5-§8 into one 455-line block — `reset_all_participants` was not extractable at all. Each fence is now on its own line, and the extract-and-execute loop that missed it (it tolerated the glued fence, proving the queries ran while the document failed to deliver them) now asserts **block count** and **max block length** and re-parses the finished file with `markdown-it`. P2 narrows `reset_participant`'s own-cursor sweep so it no longer deletes a participant's read-state on a *surviving* thread, while `reset_all` keeps the wide sweep — the two resets have different node lifecycles and the note now says so. P3 turns `unscopedCount > 0` into a response contract and returns `unscopedIds`. Nits: §8's roster timings are re-stated as noise, and §2.1's wording no longer contradicts §1.1. One live bug found and fixed while implementing P2: the liveness lookup dereferenced a null `own` alias and raised on the empty-cursor case.*

*2026-09-02 — v1.1: revised against `docs/reviews/salesperson-ui-graph.md` (approve with suggestions; 4 majors, 5 minors, 1 nit). Both guards survived the review unbroken and are unchanged. F1 drops `u.userId > ''` from `reset_all` (a silent under-delete, since `42 > ''` is `null`); F2 makes `scoped=false` a true no-op and stops `reset_all` beheading an unscoped participant; F3 corrects §7's orphan classes to the one real producer and adds a structural cursor sweep; F4 corrects §2.3's variant-A row, which was run with the re-mint clause ablated. F5–F9 and the nit folded in. Two defects I found while re-verifying: the published Cypher used `--` comments, which this dialect rejects, and the first cut of F3's sweep cost 3× on `reset_all` until its `collect` was collapsed before the `UNWIND`.*

Implements S0 of `docs/plans/salesperson-ui.md` §5.1 — the graph slice of §4.3 (participant
identity & isolation), §4.6 (the two order reads B4 identified), §4.8 (the two resets) and §4.9
(one workspace variable). Gated on `docs/reviews/salesperson-ui.md` **Pass 2, finding N1**.

**Every query below was executed against the live pinned instance** (`falkordb-dev`,
`localhost:6379`, FalkorDB `v4.18.11`, module `41811`; client `falkordb-py` 1.6.1) on throwaway
probe graphs bootstrapped from `falkor-chat/scripts/bootstrap_schema.sh`'s own
`bootstrap_workspace` at `EMBEDDING_DIM=1024` — `ws:probe-s0-reset` for v1.0, `ws:probe-s0r2`
for the v1.1 re-verification. §11 is the verification log. Nothing here is written from the
model in my head.

**Cypher comments in this note are `//`.** `--` is **not** a comment on this build — it fails to
parse (`Invalid input 'G': expected '>' or '('`). v1.0 published the guard comments as `--`,
which meant the printed query text did not parse as printed. Fixed throughout; copy the blocks
verbatim.

**Out of scope, deliberately:** product images (§4.7 keys them to static assets by `productId`
— no graph work at all).

---

## 1. The one thing this note exists to get right

N1's finding, restated as the design constraint: **victims and survivors share the labels
`Channel`, `Thread` and `Message`.** A survivor assertion written by label is therefore
structurally incapable of catching an over-broad delete. §4.8's governing rule is a *scoping*
rule:

> Every `Channel`, `Thread` and `Message` **not reachable from a participant `User`** survives
> both resets — `seed_demo.sh`'s `demo-general`/`demo-welcome` and every message in them
> included. A participant `User` is one carrying `tokenHash`.

S0's scope row requires that predicate **in the Cypher's `MATCH`, not in a convention about how
the query is called**. This note satisfies that with **two independent guards**, both live-proved
load-bearing in §2.3:

| # | Guard | Where | What it stops |
|---|---|---|---|
| **G1** | `WHERE u.tokenHash IS NOT NULL` on the anchor | both resets | any non-participant `User` being used as a reset root — `config.USER_ID`'s lifespan node, `seed_demo.sh`'s `u1`, an operator account, an `Agent` id, an unknown id |
| **G2** | `WHERE ch.participantId = u.userId` on the channel hop | both resets | any `Channel` **not created by this participant's own join** — even one the participant is a genuine `MEMBER_OF` |

### 1.1 Why G2 is a provenance marker and not `ch.channelId = u.channelId`

The obvious formulation of "their own channel" is id-equality against the `User.channelId`
denormalization §4.3 already specifies. It works, but it is a *coincidence* check: it says the
two ids happen to be equal. It cannot distinguish "this channel was minted for this participant"
from "this participant's `channelId` field happens to name a pre-existing channel."

`Channel.participantId`, written **only** by `ensure_participant` (§3), is a *provenance* check.
`demo-general` — and every channel created by any other code path, seed script, or human — has
no `participantId` at all, and `null = anything` evaluates to `null`, never `true`. Such a
channel is therefore **structurally unreachable** as a delete target, regardless of what any
`User` property says, who is a member of it, or how carelessly the repository method is called.

This is one new nullable, unindexed property on `Channel`. It needs no DDL (§9) and it is the
cheapest available way to make the safety a property of the *data*, not of the caller.

**Residual, stated honestly.** A `Channel` that genuinely carries
`participantId = <some participant's userId>` **is** a delete target — that is the definition.
The only way such a node exists is `ensure_participant` creating it, and `ensure_participant`
mints `$channelId` server-side from a uuid with no route accepting a client value (§4.3: "no
storefront route accepts a client-supplied `threadId`, `customerId`, `orderId` or `ws`"). A
pre-existing channel cannot acquire the marker without a hand-written `SET`.

**The S0 gate tested exactly that residual and could not reach it**
(`docs/reviews/salesperson-ui-graph.md` §1): `Repository.create_channel`
(`falkor-chat/server/falkorchat/repository.py:184-198`) writes a fixed three-property map with
no caller-controlled extras, so the marker cannot be *forged*; and there is no
`MEMBER_OF`-deleting query anywhere in the codebase, so a participant's own channel cannot be
*detached* from them either. Both states that would break scoping are unreachable through the
shipped API. That is an independent confirmation of the mechanism, not of this note's prose —
worth carrying forward as the reason G2 is written as provenance rather than id-equality.

---

## 2. The scoping rule, proved by construction and destruction

### 2.1 The probe fixture

`ws:probe-s0-reset` was seeded with a **non-participant survivor subgraph** mirroring
`seed_demo.sh` plus everything §4.8 lists as a must-survive, and **participants** in three
shapes, including two adversarial ones:

*Survivors (no `tokenHash` anywhere):*
`User u1` (the `seed_demo.sh` actor) · `User u2` — **adversarial: carries
`channelId:'demo-general'`, `threadId:'demo-welcome'` and a real `MEMBER_OF` edge into
`demo-general`** · `Agent assistant` · `Channel demo-general` → `Thread demo-welcome` →
3 `Message` on a full `HEAD`/`NEXT`/`TAIL`/`POSTED_BY`/`MENTIONS_MEMBER`/`EMITTED` chain ·
2 `ReadCursor` · `WorkspaceConfig` (the K-042 singleton carrying K-056's Ministral re-point) ·
`Document`/`Chunk`/`Entity` incl. a `Chunk-[:DERIVED_FROM]->Message` edge ·
`WorkflowDefSnapshot salesperson@v6` + `Step` · `Customer u1` + `Cart` + `CartItem` + `Order` +
`OrderLine`.

*Participants (each: `User`+`Channel`+`Thread`, 3 messages incl. one **`Agent`-authored reply**,
2 `ReadCursor`, `WorkflowRun`+`StepRun`+`TraceEvent`, `Customer`+`Cart`+`CartItem`+2 `Order`s+
3 `OrderLine`s):* `p-aaa` · `p-bbb` · **`p-ccc` — adversarial: also a real `MEMBER_OF` of
`demo-general`**.

**v1.1 adds three shapes the review's fixture had and mine did not**, all of which now run in
every regression pass: **`p-ddd`** — a participant whose channel carries a *mismatched*
`participantId`; **`p-eee`** — a participant with **no `MEMBER_OF` edge** into its own correctly
marked channel; and **`orphan-1`** — a `Message` carrying `threadId: 'th-A'` that is **not linked
into the `HEAD`/`NEXT` chain** (F9). The first two are the only two ways G2 can fail to resolve.
**They are unreachable on a healthy graph** — §1.1's finding, which §5 now deploys — and are
seeded precisely because the reset's behaviour on an *already-corrupt* graph is a design decision
that has to be made deliberately rather than fallen into (F2).

**v1.2 adds two more cursors to the cross-member participant `p-ccc`** (P2): one on
`demo-welcome` — a **live thread it does not own** — and one naming `th-gone`, a thread that no
longer exists. They separate the two halves of the cursor rule §4/§5 now state.

A scaled variant (§8) rebuilds this at **50 participants × 40 messages** — 3 281 nodes,
5 490 relationships — with `p-0007` carrying the cross-membership.

### 2.2 The shipped queries, run against that fixture

| Test | Result |
|---|---|
| `reset_participant('p-ccc')` — the cross-member participant | 17 nodes deleted, **all inside `ch-C`**. `demo-general`, `demo-welcome`, its 3 messages, its 2 cursors, `u1`, `u2`, `u1`'s order and `WorkspaceConfig` all intact. `p-aaa`/`p-bbb` untouched (1 thread / 3 messages each). |
| `reset_participant('u1')` — non-participant, no `tokenHash` | **0 rows, 0 deleted, 0 created** — total no-op |
| `reset_participant('u2')` — non-participant whose `channelId` **is** `'demo-general'` | **0 rows, 0 deleted, 0 created** — total no-op |
| `reset_participant('assistant')` — an `Agent` id | **0 rows, 0 deleted, 0 created** |
| `reset_participant('no-such')` — unknown id | **0 rows, 0 deleted, 0 created** |
| `reset_all_participants()` with the cross-member participant present (4 participants incl. one provisioned-but-silent) | 60 nodes deleted. Survivors: `demo-general` is the **only** `Channel`, `demo-welcome` the **only** `Thread`, its 3 messages the **only** `Message`s; both its cursors, `u1`, `u2`, `u1`'s `Customer`+`Cart`+`Order`, `Agent`, `WorkspaceConfig`, `Document`/`Chunk`/`Entity`, snapshot+`Step` all intact; **0** `WorkflowRun`/`StepRun`/`TraceEvent`; **0** participant `User`s |
| `reset_all_participants()` a second time, clean graph | one status row of all-zeros, 0 deleted, survivors unchanged — idempotent |
| `reset_participant` twice in a row on one participant | 1st deletes 17 and mints `th-A2`; 2nd deletes 1 (the empty thread it had just minted) and mints `th-A3`; `ch-A` holds exactly one thread and `User.threadId` tracks it |
| **v1.1 (F2)** `reset_participant('p-ddd')` — channel carries a *mismatched* marker | `scoped=false`, **0 nodes deleted, 0 created, 0 properties set**, empty identity delta — a total no-op |
| **v1.1 (F2)** `reset_participant('p-eee')` — no `MEMBER_OF` into its own marked channel | same: `scoped=false`, total no-op |
| **v1.1 (F2)** `reset_all_participants()` with both unscoped shapes present | `unscopedCount: 2`; both participants left **whole and collectable** (`User`+`Channel`+`Thread`+3 `Message`+run subtree+commerce all intact), `userCount: 3` counting only the scoped ones |
| **v1.1 (F9)** off-chain `orphan-1` | survives both resets, as documented — it is not in the chain the walk follows |

### 2.3 The counterfactual — each guard removed in turn

Same fixture, same call, one guard deleted from the query. This is the evidence that the guards
are load-bearing rather than decorative, **and that a label-based assertion cannot see the
difference**:

*(Re-run in full for v1.1 on `ws:probe-s0r2` and again for v1.2 on `ws:probe-s0r3`; the counts
below are the v1.2 run. They differ from v1.0's because the fixture carries five participants
instead of three, and from v1.1's by the two `p-ccc` cursors §2.1 adds. **Row A is corrected** —
see the note under the table; the review was right that v1.0 published it without disclosing an
ablation.)*

| Variant | Called with | Result | `demo-general` | `demo-welcome` | msg `dm1` | *A label-based survivor check would report* |
|---|---|---|---|---|---|---|
| **CONTROL** — shipped, both guards | `p-ccc` | 18 deleted | alive | **alive** | **alive** | `Channel 6, Message 16, Thread 6` |
| **A-as-shipped** — G2 removed, query otherwise verbatim | `p-ccc` | **raises `unique constraint violation on node of type Thread`; writes nothing** | alive | alive | alive | `Channel 6, Message 19, Thread 6` (unchanged) |
| **A-delete-only** — G2 removed **and the re-mint clause ablated** | `p-ccc` | 25 deleted | alive | **GONE** | **GONE** | `Channel 6, Message 13, Thread 4` |
| **B** — G1 + G2 removed, query otherwise verbatim | `u2` | 6 deleted | alive | **GONE** | **GONE** | **`Channel 6, Message 16, Thread 6`** |

**Variant B is N1 in one line, and it needs no ablation.** Its label counts are **byte-identical
to the control's** — 6 channels, 16 messages, 6 threads — while `demo-welcome`, all three of its
messages and both its read-cursors have been destroyed. Every "assert every §4.8 survivor by
label" check passes on that run. Only a *positive* assertion naming
`demo-general`/`demo-welcome` catches it, which is exactly what S4's done-condition requires.
Variant B runs clean as shipped because `u2` is a member of exactly **one** channel, so the
query does not row-multiply.

**Correction, and the fact it exposes (F4).** v1.0 published row A as "23 deleted,
`demo-welcome` GONE" without stating that it was run with the thread re-mint clause removed.
Re-run verbatim, G2's removal makes `ch` bind to **every** channel the participant is a member
of, and because `WITH u, ch, collect(…)` groups by `ch` **the query returns one row per matched
channel**; the re-mint `FOREACH` then fires once per row with the same `$newThreadId`, and the
whole query aborts on `Thread.threadId`'s UNIQUE constraint, writing nothing. Both behaviours
are now in the table. The conclusion is unchanged — G2 is load-bearing, as row A-delete-only
shows once the abort is removed — but an implementer re-running row A as v1.0 printed it got an
error, and §12 cited that row as the guard's price.

**That row-multiplication is itself a fail-safe, and §4 now says so.** If two `Channel`s ever
carry the same `participantId`, `reset_participant` raises **permanently** for that participant
and writes nothing (verified: old thread and all messages intact, the new thread absent).
`reset_all` still cleans up, because it never re-mints. A duplicate marker therefore fails
loudly and non-destructively rather than double-deleting — the right direction for a query whose
failure mode matters more than its availability.

---

## 3. `ensure_participant` — provisioning, idempotent, one atomic write

```cypher
// $participantId $displayName $tokenHash $language $channelId $threadId
// $threadTitle $agentId $now
OPTIONAL MATCH (existing:User  {userId:  $participantId})
OPTIONAL MATCH (clash:Agent    {agentId: $participantId})
OPTIONAL MATCH (agent:Agent    {agentId: $agentId})
WITH existing, clash, agent,
     (existing IS NULL AND clash IS NULL AND agent IS NOT NULL) AS doCreate
FOREACH (_ IN CASE WHEN doCreate THEN [1] ELSE [] END |
  CREATE (u:User {userId:      $participantId,
                  displayName: $displayName,
                  tokenHash:   $tokenHash,
                  language:    $language,
                  channelId:   $channelId,
                  threadId:    $threadId,
                  joinedAt:    $now})
  CREATE (c:Channel {channelId:     $channelId, name: $displayName,
                     participantId: $participantId, createdAt: $now})
  CREATE (t:Thread  {threadId:  $threadId,  title: $threadTitle,
                     createdAt: $now, updatedAt: $now})
  CREATE (c)-[:HAS_THREAD]->(t)
  CREATE (u)-[:MEMBER_OF     {role: 'member',    joinedAt: $now}]->(c)
  CREATE (agent)-[:MEMBER_OF {role: 'assistant', joinedAt: $now}]->(c)
)
RETURN doCreate                            AS created,
       existing IS NOT NULL                AS existed,
       existing.tokenHash IS NOT NULL      AS existedParticipant,
       clash    IS NOT NULL                AS collided,
       agent    IS NULL                    AS agentMissing,
       CASE WHEN doCreate THEN $channelId ELSE existing.channelId END AS channelId,
       CASE WHEN doCreate THEN $threadId  ELSE existing.threadId  END AS threadId,
       CASE WHEN doCreate THEN $language  ELSE existing.language  END AS language
```

Shape follows QUERIES.md §2's `ensure_user` **guarded-`CREATE`-inside-`FOREACH`** idiom, extended
in three ways: the whole join is **one query** (a partial join would leave an orphan `Channel` in
a graph the presenter roster reads); the demo `Agent` is a **third precondition**; and the status
row carries the resolved `channelId`/`threadId`/`language` so a replay is a read-through.

**Status-row contract** — exactly one row, always (three `OPTIONAL MATCH`es, no anchor `MATCH`,
so it can never zero-row). **`existedParticipant` is a v1.1 addition (F7)**: without it, row 2
fires for *any* `User` holding the id, participant or not, and a caller following the table
would build a `ParticipantRecord` out of three `null`s and a token the graph never stored.

| `created` | `existed` | `existedParticipant` | `collided` | `agentMissing` | Meaning | Caller action | *verified* |
|---|---|---|---|---|---|---|---|
| `true` | `false` | `false` | `false` | `false` | fresh participant written (3 nodes, 3 rels) | success | ✅ |
| `false` | `true` | **`true`** | `false` | `false` | id already a participant — nothing written; row returns the **stored** ids/language | idempotent success (restart-survival read-through, §4.3) | ✅ |
| `false` | `true` | **`false`** | `false` | `false` | id exists but is **not** a participant (no `tokenHash`) — nothing written; `channelId`/`threadId`/`language` all come back `null` | **refuse: member-id collision**, exactly like the `collided` row | ✅ |
| `false` | `false` | `false` | `true` | `false` | id held by an `Agent` — **nothing written** | refuse: member-id collision (QUERIES.md §2's locked namespace rule) | ✅ |
| `false` | `false` | `false` | `false` | `true` | the demo `Agent` is absent — **nothing written** | refuse: `503`, name `seed_demo.sh`. This is §4.9's readiness preflight failing *late*; the preflight should have caught it at boot | ✅ |

Live-verified for all five rows, including that **nothing at all is written** on the two
collision paths and the agent-missing path (`ch-Z`/`th-Z`/`ch-X`/`p-www` all confirmed absent
afterwards). Row 3 was executed against `seed_demo.sh`'s tokenHash-less `u1`.

**One behaviour the `existed` row does not have, and must not be read as having:** a replay
carrying a **fresh** `$tokenHash` returns `existed=true` and the new hash is **not** written.
That is correct for an idempotent ensure (QUERIES.md §2's re-ensure never updates properties),
but it means `ensure_participant` is not a token-rotation path. Verified: a replay with new
`channelId`/`threadId`/`tokenHash` returned the stored ids and wrote nothing.

**Notes for the implementer.**

- Multi-`CREATE` inside `FOREACH` with cross-clause variable binding **works on this build**, and
  an outer-bound variable (`agent`) is a legal `CREATE` relationship endpoint inside the
  `FOREACH` body. (Contrast the known quirk that a *map-projection or list-subscript* expression
  is **not** a legal endpoint — `CREATE (ms[k])-[:NEXT]->(ms[k+1])` errors; re-confirmed here.)
- Idempotency comes from the **status logic**, not from `MERGE` — `MERGE` inside `FOREACH` is not
  standard OpenCypher. The `User.userId` UNIQUE constraint stays the same-label concurrency
  backstop; the residual cross-label race QUERIES.md §2 documents applies unchanged and is
  irrelevant in practice (`participantId = "p-" + uuid4().hex`).
- **`Channel.participantId` is a new property** (nullable, unindexed, no DDL). It must be added
  to `falkor-chat/docs/DESIGN.md` §5.1's arrow notation and to `QUERIES.md` §18 in the same
  change as S4's implementation, or the guard reads as unexplained in six months.
- **K-049 exposure is nil for this write.** The four UNIQUE-constrained properties it touches
  (`User.userId`, `Channel.channelId`, `Thread.threadId`, and `Customer.customerId` via §3.1) are
  all server-minted ids far under 4096 bytes; the participant-supplied `displayName` lands only
  on unconstrained `User.displayName` and `Channel.name`. A service-boundary length bound on
  `displayName` is still ordinary hygiene, but it is not the K-049 crash path.

### 3.1 The profile-name write (§4.10) reuses `upsert_profile` unchanged

`join()` calls `services.save_profile(ctx, name=display_name)` **after** `ensure_participant`,
which runs QUERIES.md §17.1's existing `upsert_profile`. That `MERGE (c:Customer {customerId})`
creates the `Customer` anchor eagerly, exactly as §4.3 wants. **No new Cypher.** `Cart` stays
lazy (`ensure_cart` on first add), as today.

Two writes, not one, is deliberate: `upsert_profile` is a shipped, `[verified]`, constraint-backed
query with its own semantics (`coalesce()`-guarded per-field update) and folding it into the join
would fork it. The window between them is harmless — a crash after `ensure_participant` leaves a
participant whose profile name is unset, which the next `save_profile` fixes and which the UI
already renders as an em-dash.

---

## 4. `reset_participant` — "reset mine"

One atomic query: collect the victim set structurally, mint the replacement `Thread`, repoint
`User.threadId`, then delete. **Every write is gated on `ch IS NOT NULL`, so a participant whose
own channel does not resolve is a total no-op** (v1.1, F2 — v1.0 deleted their commerce subgraph
while keeping the transcript, and reported success).

```cypher
// $participantId $newThreadId $threadTitle $now
MATCH (u:User {userId: $participantId})
WHERE u.tokenHash IS NOT NULL                       // G1
OPTIONAL MATCH (u)-[:MEMBER_OF]->(ch:Channel)
  WHERE ch.participantId = u.userId                 // G2
OPTIONAL MATCH (ch)-[:HAS_THREAD]->(t:Thread)
OPTIONAL MATCH (t)-[:HEAD]->(h:Message)-[:NEXT*0..]->(m:Message)
WITH u, ch, collect(DISTINCT t) AS threads, collect(DISTINCT m) AS msgs

UNWIND (CASE WHEN msgs = [] THEN [null] ELSE msgs END) AS mm
OPTIONAL MATCH (r:WorkflowRun)-[:TRIGGERED_BY]->(mm)
WITH u, ch, threads, msgs, collect(DISTINCT r) AS runs

UNWIND (CASE WHEN runs = [] THEN [null] ELSE runs END) AS rr
OPTIONAL MATCH (rr)-[:HAS_STEP_RUN]->(sr:StepRun)
OPTIONAL MATCH (sr)-[:TRACED]->(te:TraceEvent)
WITH u, ch, threads, msgs, runs,
     collect(DISTINCT sr) AS steps, collect(DISTINCT te) AS traces,
     [x IN threads | x.threadId] AS threadIds

OPTIONAL MATCH (rc:ReadCursor) WHERE rc.threadId IN threadIds
WITH u, ch, threads, msgs, runs, steps, traces, threadIds,
     collect(DISTINCT rc) AS tcur

OPTIONAL MATCH (u)-[:HAS_CURSOR]->(own:ReadCursor)
WITH u, ch, threads, msgs, runs, steps, traces, threadIds, tcur,
     collect(DISTINCT own) AS allOwn

UNWIND (CASE WHEN allOwn = [] THEN [null] ELSE allOwn END) AS oc
OPTIONAL MATCH (liveT:Thread) WHERE oc IS NOT NULL AND liveT.threadId = oc.threadId
WITH u, ch, threads, msgs, runs, steps, traces, tcur,
     collect(DISTINCT CASE WHEN oc IS NOT NULL
                            AND (oc.threadId IN threadIds OR liveT IS NULL)
                           THEN oc END) AS ocur
WITH u, ch, threads, msgs, runs, steps, traces,
     tcur + [x IN ocur WHERE NOT x IN tcur] AS cursors

OPTIONAL MATCH (cust:Customer {customerId: $participantId})
OPTIONAL MATCH (cust)-[:HAS_CART]->(cart:Cart)
OPTIONAL MATCH (cart)-[:HAS_ITEM]->(item:CartItem)
WITH u, ch, threads, msgs, runs, steps, traces, cursors,
     collect(DISTINCT cust) AS custs, collect(DISTINCT cart) AS carts,
     collect(DISTINCT item) AS items

OPTIONAL MATCH (:Customer {customerId: $participantId})-[:PLACED]->(o:Order)
OPTIONAL MATCH (o)-[:HAS_LINE]->(ol:OrderLine)
WITH u, ch, threads, msgs, runs, steps, traces, cursors, custs, carts, items,
     collect(DISTINCT o) AS orders, collect(DISTINCT ol) AS lines

FOREACH (c IN CASE WHEN ch IS NULL THEN [] ELSE [ch] END |
  CREATE (c)-[:HAS_THREAD]->(:Thread {threadId:  $newThreadId,
                                      title:     $threadTitle,
                                      createdAt: $now, updatedAt: $now})
)
SET u.threadId = CASE WHEN ch IS NULL THEN u.threadId ELSE $newThreadId END

WITH u, ch,
     CASE WHEN ch IS NULL THEN []
          ELSE threads + msgs + runs + steps + traces + cursors
                       + custs + carts + items + orders + lines END AS victims,
     size(threads) AS threadCount, size(msgs)    AS messageCount,
     size(runs)    AS runCount,    size(steps)   AS stepRunCount,
     size(traces)  AS traceCount,  size(cursors) AS cursorCount,
     size(orders)  AS orderCount,  size(items)   AS cartItemCount
FOREACH (v IN victims | DETACH DELETE v)
WITH ch, u, victims, ch IS NOT NULL AS scoped,
     threadCount, messageCount, runCount, stepRunCount, traceCount,
     cursorCount, orderCount, cartItemCount
RETURN scoped, u.threadId AS threadId, size(victims) AS deletedCount,
       CASE WHEN scoped THEN threadCount   ELSE 0 END AS threadCount,
       CASE WHEN scoped THEN messageCount  ELSE 0 END AS messageCount,
       CASE WHEN scoped THEN runCount      ELSE 0 END AS runCount,
       CASE WHEN scoped THEN stepRunCount  ELSE 0 END AS stepRunCount,
       CASE WHEN scoped THEN traceCount    ELSE 0 END AS traceCount,
       CASE WHEN scoped THEN cursorCount   ELSE 0 END AS cursorCount,
       CASE WHEN scoped THEN orderCount    ELSE 0 END AS orderCount,
       CASE WHEN scoped THEN cartItemCount ELSE 0 END AS cartItemCount
```

**Why each piece is shaped the way it is.**

- **Messages are reached by the `HEAD` → `NEXT*0..` walk, never by `WHERE m.threadId = $tid`.**
  The walk is *structural*: it can only reach messages inside a thread that is itself inside a
  proven-participant channel. `Message.threadId` is a deliberately unindexed denormalization
  (`DESIGN.md` §5.1), it is `null` on any pre-backfill row, and per the pinned build's
  anchor-pull behaviour a `WHERE` on it would drag the whole plan onto a full `Message` label
  scan. The walk cannot escape the thread: `NEXT` is thread-scoped by construction, and although
  `StepRun` also carries a `:NEXT` type, no `Message`→`StepRun` `NEXT` edge exists, and the
  terminal pattern is labelled `(m:Message)`.
  **The invariant this buys safety against, named (F9):** every `Message` must be linked into
  its thread's `HEAD`/`NEXT`/`TAIL` chain. QUERIES.md §4's v2 write paths do that inside the
  same guarded `FOREACH` as the `CREATE`, so an off-chain message is unreachable through the
  platform — but a hand-planted one **survives both resets** (verified: `orphan-1`, carrying
  `threadId: 'th-A'`, outlives `reset_participant` and would outlive `reset_all` with its thread
  gone). A future writer that creates a `Message` before linking it breaks reset completeness,
  not just thread reads.
- **The delete is thread-scoped, not author-scoped** (§4.8, locked). The `Agent`-authored reply
  living inside the participant's thread **is** deleted — verified: `POSTED_BY` count drops by
  exactly the thread's three messages while every other participant's `Agent` reply survives. An
  author-scoped sweep would orphan those replies and, the moment the `Agent` is the author, cross
  participant boundaries.
- **`WorkflowRun` is reached only through `TRIGGERED_BY` from a thread message.** This is the
  storefront's only run-creation path (`trigger.maybe_trigger` → `start_workflow_run`).
  **And that is a structural property of the storefront deployment, not a convention (v1.1, F6).**
  CPG-verified in the gate review: `Services.start_workflow_run` has exactly two non-test callers
  — `falkorchat/trigger.py:76` and `falkorchat/api.py:394` — and a run started through
  `api.py:394` without a trigger message carries **no `TRIGGERED_BY` edge**, so it and its
  `StepRun`s/`TraceEvent`s are unreachable by both resets. That path is closed only because
  **§4.9 move 1 leaves `api.build_router` unmounted (`dev_surface=False`, S3)**. A deployment
  that re-mounts the dev surface alongside participants has re-opened an under-delete here —
  this dependency must be stated in the `QUERIES.md` §18 entry too.
- **`ReadCursor` is collected from two sources, and the second half is narrower than
  `reset_all`'s (F3 + P2).** There is no `Thread`→`ReadCursor` edge —
  `(member)-[:HAS_CURSOR]->(:ReadCursor {cursorId: "{memberId}:{threadId}"})` — so cursors for
  the threads being deleted *now* are found by a `ReadCursor` label scan filtered on
  `rc.threadId IN threadIds`. That is the one label scan in the plan (102 rows at 50
  participants, 0.02 ms — §8), chosen for completeness over a structural walk from channel
  members, which would miss a cursor whose owner has left the channel. **On top of it,
  `OPTIONAL MATCH (u)-[:HAS_CURSOR]->(own:ReadCursor)` sweeps the participant's own cursors —
  but only those whose thread is being deleted now *or* no longer exists.**

  **Why narrower here than in `reset_all`, and this asymmetry is deliberate.** In "reset mine"
  the `User` **survives**, so a cursor it holds on a thread that also survives is live read-state
  for a membership that still exists — deleting it would silently reset the participant's read
  position in a channel this operation was never asked to touch, and `demo-welcome` is exactly
  such a thread for the cross-member shape. In `reset_all` the `User` is **deleted**, so every
  cursor it owns would be left unowned — a genuine orphan — and the wide sweep is required. The
  liveness test is an `OPTIONAL MATCH` on the indexed `Thread.threadId` (`Node By Index Scan`,
  1 record, 0.011 ms — §8), guarded so the empty-cursor case cannot dereference a null alias.

  Verified on `p-ccc`, which holds three cursors: its own thread's (`p-ccc:th-C`, deleted with
  the thread), one naming a dead thread (`p-ccc:th-gone`, **collected** — F3's orphan class), and
  one on the surviving `demo-welcome` (`p-ccc:demo-welcome`, **kept**) — `cursorCount: 3`,
  `deletedCount: 18`. The `Agent`'s cursor for the deleted thread goes too. Under v1.1's wider
  form the same call deleted `p-ccc:demo-welcome` as well (`cursorCount: 4`); both runs are in
  §11. `assistant:demo-welcome` and `u1:demo-welcome` are untouched by either.
- **Every optional list is collapsed with `collect(DISTINCT …)` and consumed with `FOREACH`,
  never `UNWIND`.** An empty `UNWIND` collapses the row stream on this build and would silently
  drop the mandatory thread re-mint; `FOREACH` over `[]` is a true no-op. The two `UNWIND`s that
  do appear are guarded with the `CASE WHEN … = [] THEN [null]` idiom and each is collapsed by
  its own `collect` before the next one expands, so they do not row-multiply.
- **The replacement `Thread` is created before the delete but collected after.** Clause order
  guarantees `threads` cannot contain it (the `collect` is a blocking aggregation).
- **`Customer` is deleted**, per §4.8's survivor column.

**Status row.** `scoped=false` means G2 found no owned channel. **It is now a guaranteed
no-op**: the victim list is `[]`, the re-mint is skipped, `User.threadId` is untouched, and
every per-class count is forced to `0` so the row cannot read as a partial success. `deletedCount`
is the authoritative field. Verified against both ways G2 can fail to resolve — a channel with a
mismatched marker, and a participant with no `MEMBER_OF` edge into its own marked channel — each
returning `0 nodes deleted, 0 created, 0 properties set` and an empty identity delta. **The
caller must treat `scoped=false` as an alarm, not as success**: nothing was reset, and nothing
will be until the anomaly is repaired. Zero rows (as opposed to `scoped=false`) means G1 rejected
the id — treat that as "not a participant", since it is also what an already-deleted participant
returns.

**Two `Channel`s carrying the same marker make this query raise, permanently.** The `WITH u, ch,
collect(…)` groups by `ch`, so a duplicate marker yields two rows and the re-mint `FOREACH` fires
twice with one `$newThreadId`, aborting on `Thread.threadId`'s UNIQUE constraint with nothing
written (§2.3). That is a fail-safe, not a defect — `reset_all` still collects the participant,
because it never re-mints — but S4 should surface the raise rather than swallow it.

**One product consequence to hand to S7, not a change here.** Deleting the `Customer` also
deletes the profile name §4.10's join wrote, while `User.displayName` survives — so after "reset
mine" the profile panel shows an em-dash for a participant who typed their name at join.
Recommend S7's `reset_participant` service wrapper re-call the existing
`services.save_profile(ctx, name=<User.displayName>)` immediately after the reset, which restores
join parity with **no new Cypher**. Flagged rather than folded in because §4.8's table is
explicit that the `Customer` goes, and I am not relitigating it.

---

## 5. `reset_all_participants` — "reset everyone"

Identical scoping, widened to every participant `User`, and additionally deleting the `User` and
`Channel` nodes themselves (so every participant token is invalidated).

```cypher
MATCH (u:User)
WHERE u.tokenHash IS NOT NULL                       // G1
OPTIONAL MATCH (u)-[:MEMBER_OF]->(ch:Channel)
  WHERE ch.participantId = u.userId                 // G2
OPTIONAL MATCH (ch)-[:HAS_THREAD]->(t:Thread)
OPTIONAL MATCH (t)-[:HEAD]->(h:Message)-[:NEXT*0..]->(m:Message)
WITH collect(DISTINCT CASE WHEN ch IS NOT NULL THEN u END)        AS users,
     collect(DISTINCT ch)                                          AS channels,
     collect(DISTINCT t)                                           AS threads,
     collect(DISTINCT m)                                           AS msgs,
     collect(DISTINCT CASE WHEN ch IS NOT NULL THEN u.userId END)  AS pids,
     collect(DISTINCT CASE WHEN ch IS NULL THEN u.userId END)      AS unscopedIds

UNWIND (CASE WHEN msgs = [] THEN [null] ELSE msgs END) AS mm
OPTIONAL MATCH (r:WorkflowRun)-[:TRIGGERED_BY]->(mm)
WITH users, channels, threads, msgs, pids, unscopedIds, collect(DISTINCT r) AS runs

UNWIND (CASE WHEN runs = [] THEN [null] ELSE runs END) AS rr
OPTIONAL MATCH (rr)-[:HAS_STEP_RUN]->(sr:StepRun)
OPTIONAL MATCH (sr)-[:TRACED]->(te:TraceEvent)
WITH users, channels, threads, msgs, pids, unscopedIds, runs,
     collect(DISTINCT sr) AS steps, collect(DISTINCT te) AS traces,
     [x IN threads | x.threadId] AS threadIds

OPTIONAL MATCH (rc:ReadCursor) WHERE rc.threadId IN threadIds
WITH users, channels, threads, msgs, pids, unscopedIds, runs, steps, traces,
     collect(DISTINCT rc) AS tcur

UNWIND (CASE WHEN users = [] THEN [null] ELSE users END) AS uu
OPTIONAL MATCH (uu)-[:HAS_CURSOR]->(own:ReadCursor)
WITH users, channels, threads, msgs, pids, unscopedIds, runs, steps, traces,
     tcur, collect(DISTINCT own) AS ocur
WITH users, channels, threads, msgs, pids, unscopedIds, runs, steps, traces,
     tcur + [x IN ocur WHERE NOT x IN tcur] AS cursors

OPTIONAL MATCH (cust:Customer) WHERE cust.customerId IN pids
OPTIONAL MATCH (cust)-[:HAS_CART]->(cart:Cart)
OPTIONAL MATCH (cart)-[:HAS_ITEM]->(item:CartItem)
WITH users, channels, threads, msgs, pids, unscopedIds, runs, steps, traces, cursors,
     collect(DISTINCT cust) AS custs, collect(DISTINCT cart) AS carts,
     collect(DISTINCT item) AS items

OPTIONAL MATCH (c2:Customer)-[:PLACED]->(o:Order) WHERE c2.customerId IN pids
OPTIONAL MATCH (o)-[:HAS_LINE]->(ol:OrderLine)
WITH users, channels, threads, msgs, unscopedIds, runs, steps, traces, cursors,
     custs, carts, items, collect(DISTINCT o) AS orders,
     collect(DISTINCT ol) AS lines

WITH users + channels + threads + msgs + runs + steps + traces + cursors
           + custs + carts + items + orders + lines AS victims,
     unscopedIds, size(unscopedIds) AS unscopedCount,
     size(users)   AS userCount,     size(channels) AS channelCount,
     size(threads) AS threadCount,   size(msgs)     AS messageCount,
     size(runs)    AS runCount,      size(steps)    AS stepRunCount,
     size(traces)  AS traceCount,    size(cursors)  AS cursorCount,
     size(custs)   AS customerCount, size(orders)   AS orderCount
FOREACH (v IN victims | DETACH DELETE v)
RETURN userCount, channelCount, threadCount, messageCount, runCount,
       stepRunCount, traceCount, cursorCount, customerCount, orderCount,
       unscopedCount, unscopedIds
```

- **No thread is re-minted** — every participant is bounced to the join screen (§4.8), and
  `ensure_participant` mints a fresh subgraph on their next join.
- **v1.1 (F1): the anchor is the bare `tokenHash IS NOT NULL`.** v1.0 carried a
  `u.userId > '' AND` conjunct here for an index anchor. It is **not always true** —
  live-verified on this build: `'abc' > ''` → `true`, but `42 > ''` → **`null`**, `42.5 > ''` →
  `null`, `true > ''` → `null`, `null > ''` → `null`. A participant `User` whose `userId` is not
  a string was therefore dropped from the anchor and **survived `reset_all` entirely** — channel,
  thread, messages, cursors, commerce — while the status row reported success. Reproduced
  directly: with `{userId: 42, tokenHash: 'h'}` and `{userId: 'p-x', tokenHash: 'h'}` in the
  graph, the bare predicate returns `[42, 'p-x']` and the conjunct form returns `['p-x']`.
  It also bought nothing: `userId > ''` has no selectivity, so both forms visit the whole label
  (§8). Dropping it makes both resets share one predicate and removes a completeness dependency
  on a data-type invariant nothing enforces.
- **v1.1 (F2): an unscoped participant is skipped whole, never beheaded.** `users` and `pids` are
  gated on `ch IS NOT NULL`, and the count of skipped participants is returned as
  **`unscopedCount`**. v1.0 deleted such a `User` (it passes G1) while G2 left its channel
  unmatched, stranding the `Channel` + `Thread` + `Message`s + cursors with no anchor any future
  reset could reach — one participant's transcript left visible in a graph the presenter believes
  is clean, reported as success. Now the participant is left **intact and collectable**: verified
  with both unscoped shapes present, `unscopedCount: 2` and
  `unscopedIds: ['p-ddd', 'p-eee']`, their `User`/`Channel`/`Thread`/3 `Message`/run
  subtree/commerce all still there, `userCount: 3` counting only the scoped ones. **v1.2 returns
  the ids, not just the count** (P3), so a caller can name the affected participants rather than
  only knowing some exist.
  **The trade-off, and why it is graceful degradation rather than an FR-7 breach.** An unscoped
  participant keeps a **working `tokenHash`** and their transcript, and is not bounced. The
  argument §1.1 establishes and this bullet now deploys: **the unscoped branch is unreachable on
  a healthy graph** — `ensure_participant` writes the `User`, `Channel`, marker and `MEMBER_OF`
  in one atomic query; `Repository.create_channel` cannot set the marker; nothing in the codebase
  deletes a `MEMBER_OF` edge. So v1.0's behaviour and v1.1's are **both dead branches**, and the
  only live question is which failure is better on an already-corrupt graph. v1.0 satisfied FR-7
  nominally while stranding a transcript **permanently and silently** — an AC-2 leak reported as
  success. v1.1 misses FR-7 for one participant and **counts it**. AC-2 is the stronger
  requirement, so this is the right way round. `unscopedIds` is what makes it loud, and §12 makes
  surfacing it a contract rather than a preference (P3).

  **Reversal trigger.** If anything ever introduces a way to detach a participant from their own
  channel — a `MEMBER_OF`-deleting query, a channel-transfer feature, an admin tool that rewrites
  `participantId` — this branch stops being dead and the choice must be re-made. A reachable
  "keeps a valid token" path is a materially different proposition from an unreachable one.
- **v1.1 (F3): participant-owned cursors are swept structurally**, via
  `(uu)-[:HAS_CURSOR]->(own)` over the scoped participants, merged with the thread-scoped set.
  **This sweep stays wide — no liveness filter, unlike §4's (P2).** Here the `User` is deleted,
  so any cursor it owns that survived would be **unowned**: a real orphan, including one on a
  surviving non-participant thread. Verified: `p-ccc:demo-welcome` is collected by `reset_all`
  (and kept by `reset_participant`), and afterwards
  `MATCH (rc:ReadCursor) WHERE NOT ()-[:HAS_CURSOR]->(rc)` returns **empty** — no unowned cursor
  is left behind. §4's cursor bullet carries the reasoning for why the two differ.
  **Collapse `tcur` with its own `WITH` *before* the `users` `UNWIND`** — the first cut left the
  `ReadCursor` scan un-collapsed, so the 50-row `UNWIND` multiplied it to ~5 000 rows and
  `reset_all` cost **~690 ms instead of ~240 ms**. Measured, fixed, re-measured (§8); it is the
  "collapse each block before the next `UNWIND` expands" rule, and it bites at scale only.
- **`Customer` scoping is `customerId IN pids`, exact by construction** — `customerId ==
  participantId` (§4.3, and QUERIES.md §16's graph note §1.2). A non-participant `Customer`
  (`u1`'s, in the fixture) is never in `pids`; verified survivor with its `Cart`, `CartItem`,
  `Order` and `OrderLine`.
- **On a clean graph it returns one all-zeros status row, not zero rows** — the global
  `collect()` over an empty match produces exactly one row. That gives the caller an
  unambiguous "nothing to do" instead of an empty result it has to index into. (`unscopedCount`
  is the exception and stays truthful: a graph still holding unscoped participants reports them
  on every call.)
- `config.USER_ID`'s lifespan-created `User` survives because it carries no `tokenHash` — the
  same predicate §5.2's roster already filters on.
- **The presenter's own presenter token is untouched** — it is not a graph object at all
  (§4.3), so `reset_all` cannot invalidate it. The presenter's *participant* token is
  invalidated like everyone else's, which is what FR-4 asks for.


## 6. Keep/delete inventory — documentation, **not** the mechanism

The mechanism is §1's two guards. This table exists so a reader can check the guards produce the
intended outcome; it is **not** what S4 should assert on (§2.3 variant B is why). Every row was
observed on the probe graph.

`bootstrap_schema.sh`'s full label inventory, adjudicated:

| Label | Reset mine | Reset everyone | How it is reached / why |
|---|---|---|---|
| `User` (participant, **scoped**) | **keeps** | **deletes** | the reset root itself; deleting it invalidates the token |
| `User` (participant, **unscoped** — G2 unresolved) | keeps (total no-op) | **keeps**, counted in `unscopedCount` | v1.1 (F2): skipped whole rather than beheaded |
| `User` (non-participant) | keeps | **keeps** | G1 — no `tokenHash` |
| `Channel` (participant's own) | **keeps** | **deletes** | G2 |
| `Channel` (any other) | keeps | **keeps** | G2 — no `participantId` |
| `Thread` (in a participant channel) | **deletes**, one re-minted | **deletes** | `(ch)-[:HAS_THREAD]->` |
| `Thread` (any other) | keeps | **keeps** | unreachable from a participant `User` |
| `Message` (in a participant thread) | **deletes** | **deletes** | `HEAD` → `NEXT*0..` — incl. `Agent`-authored replies (thread-scoped, not author-scoped) |
| `Message` (any other) | keeps | **keeps** | unreachable |
| `Message` **off-chain** (has `threadId`, not in the `HEAD`/`NEXT` chain) | keeps | **keeps** | v1.1 (F9): unreachable by the structural walk. Unproducible through QUERIES.md §4's write paths, which link the chain in the same guarded `FOREACH` — the invariant §4 names |
| `ReadCursor` (for a thread being deleted now, any member) | **deletes** | **deletes** | `rc.threadId IN threadIds`; orphaned by the thread delete |
| `ReadCursor` (participant-owned, naming an *already*-deleted thread) | **deletes** | **deletes** | v1.1 (F3): the structural `(u)-[:HAS_CURSOR]->` sweep |
| `ReadCursor` (participant-owned, on a **surviving non-participant** thread — e.g. a cross-member participant's cursor on `demo-welcome`) | **keeps** | **deletes** | v1.2 (P2): the one deliberate asymmetry. `reset_participant` keeps the `User`, so this is live read-state for a membership that still exists; `reset_all` deletes the `User`, so leaving it would strand an **unowned** cursor. §4/§5 carry the reasoning |
| `ReadCursor` (`Agent`-owned, naming an already-deleted thread) | keeps | **keeps** | documented residual, §7 — sweeping it by owner would reach `assistant:demo-welcome` |
| `WorkflowRun` / `StepRun` / `TraceEvent` | **deletes** | **deletes** | `TRIGGERED_BY` → `HAS_STEP_RUN` → `TRACED` |
| `Customer` / `Cart` / `CartItem` | **deletes** | **deletes** | `customerId = participantId` |
| `Order` / `OrderLine` | **deletes** | **deletes** | `(Customer)-[:PLACED]->` → `HAS_LINE` |
| **`WorkspaceConfig`** | **keeps** | **keeps** | never matched by either query. §4.8: taking it would silently undo K-056's Ministral re-point — the single most expensive mistake available here. Positively asserted alive after both resets |
| `Document` / `Chunk` / `Entity` | **keeps** | **keeps** | survivors of *both*; never matched |
| `WorkflowDefSnapshot` / `Step` | **keeps** | **keeps** | never matched |
| `Agent` | **keeps** | **keeps** | never matched; its `MEMBER_OF` edges into deleted channels go with `DETACH DELETE`, the node does not |
| `Product` | n/a | n/a | lives in `reference`, a different graph — structurally unreachable |

**Edge behaviour at the boundary, verified.**

- A **surviving** `Chunk` that `DERIVED_FROM` a deleted participant `Message`: the `Chunk` and its
  `Document` survive, the edge is gone. Correct — §4.8 keeps the corpus. (§4.4 measure 3 means no
  such `Chunk` should exist in practice; the behaviour is benign if one does.)
- A **surviving** `Message` with an `EMITTED` edge into a deleted one: survivor intact, that one
  edge removed, its other `EMITTED` edges untouched.
- `reference` is never referenced by any query here. `GRAPH.QUERY` operates on one named graph per
  call — this is structural, not a convention.

---

## 7. Quiesce contract — what the graph gives you, and what it does not

**What the graph gives you.** Each reset is **one `GRAPH.QUERY`**, therefore atomic. Live-proved:
a reset made to violate the `Thread.threadId` UNIQUE constraint mid-query raised
`unique constraint violation on node of type Thread` and wrote **nothing** — node count identical
before and after (78/78), the old thread, its messages and its orders all still present,
`User.threadId` unchanged. There is no observable half-reset state.

**The failure boundary is client-side, and it is not "nothing changed" (v1.1, F8).** The gate
review established that the module's `TIMEOUT 1000` argument applies to **reads only** — a
1.67 s write completed untouched while a 1.0 s read was killed — so a slow `reset_all` will never
be truncated server-side. The real bound is `FALKORDB_SOCKET_TIMEOUT`, default **10 s**
(`falkorchat/config.py:29` → `falkorchat/db.py:44`). At ~240 ms that is 40× headroom, but if a
reset ever crosses it **the client raises `TimeoutError` while the server commits the delete**.
§4.8/§5.2 map a reset failure to `503 … nothing changed`; that mapping is correct for a quiesce
timeout and **wrong for a socket timeout**. S7/S10 must treat a client-side timeout on a reset as
*unknown* — re-read state and report from the graph, never report "no change".

**What the graph does not give you.** Atomicity is per *query*, not per *turn*. A turn is many
queries, and FalkorDB serialises writes per graph — so a reset cannot interleave *inside* another
query, but it interleaves freely *between* the queries of an in-flight turn.

**v1.1 (F3) — corrected: only one platform write actually orphans anything.** v1.0 claimed a turn
running past its deleted run writes orphan `Message`/`StepRun`/`TraceEvent` rows. It does not:
all three writes are anchored on nodes the reset deleted, so they match zero rows and create
nothing. Executed against a graph immediately after `reset_all`:

| in-flight write (QUERIES.md) | anchored on | rows | nodes created | consequence |
|---|---|---|---|---|
| `post_message` first-path (§4) | `Thread` | 0 | **0** | service raises `ThreadNotFoundError` — loud, user-visible |
| `record_step_and_advance` (§12.2) | `WorkflowRun` | 0 | **0** | silent no-op (§12 status-move contract) — the turn advances nothing |
| `append_trace_event` (§12.10) | `StepRun` | 0 | **0** | silent no-op |
| **`advance_cursor` (§9.3)** | **the member (`User`/`Agent`)** | **1** | **1 `ReadCursor`** | **the one real orphan producer** |

§9.3 `MERGE`s on the *member*, not the thread, so it mints a `ReadCursor` naming a thread that no
longer exists. §4/§5's structural `HAS_CURSOR` sweep now collects the **participant-owned** case
on the next reset (verified end to end: `p-aaa:th-A` minted after a reset is gone after the next
one, while `assistant:demo-welcome` and `u1:demo-welcome` are untouched).

**The `Agent`-owned residual is real and deliberately not closed.** `assistant:th-A` — the demo
`Agent`'s cursor for a deleted participant thread — escapes both resets, because the `Agent` is a
survivor and sweeping *its* cursors by owner would reach `assistant:demo-welcome`. A complete
sweep is available and parses on this build —
`OPTIONAL MATCH (t:Thread {threadId: rc.threadId}) WITH rc, t WHERE t IS NULL` — but it widens
`reset_all`'s contract from "everything reachable from a participant `User`" to "every dangling
cursor in the workspace", and I verified it would also collect a **non-participant's** dangling
cursor (`u1:ghost2`). That is outside §4.8's scoping rule, so **I am declining it**: the residual
is bounded (at most one per `(Agent, deleted thread)`), it is produced only by the race quiesce
exists to prevent, and no read path is affected — §9.4 point-lookups a cursor by `cursorId` and a
stale one simply never matches a live thread again. The complete form is recorded here so a future
"garbage-collect the workspace" job can adopt it deliberately rather than smuggling it into a
reset.

**So the quiesce is application-level, exactly as §4.8 specifies**, and this note adds only the
graph-side facts that constrain it:

1. **Reset mine** — cancel that participant's queued turn; if one is in flight, wait bounded by
   `FALKORCHAT_STOREFRONT_QUIESCE_S` (default 30 s), then run the single query; on timeout return
   `503` and run nothing. Safe because the query is all-or-nothing.
2. **Reset everyone** — stop intake (`409` on every subsequent post until it completes), drain,
   then run the single query. At 50 participants the write itself blocks the graph for ~240 ms
   (§8) — invisible against a 2 s poll, but it *is* a stop-the-world write on `ws:demo`, so it
   must not be issued while turns are draining.
3. **The order matters and is not interchangeable.** Quiesce → delete. Deleting first and
   draining after produces the silent no-ops above: a turn that consumes an LLM call and posts
   nothing.

**S7/S10's done-condition must be re-worded, because v1.0's cannot fail (F3).** "A reset issued
while a stub-LLM turn is in flight leaves no orphan `StepRun`/`TraceEvent`/`Message`" is
vacuously true — those writes create nothing post-reset whether quiesce works or not. Replace it
with conditions that can fail:

- **(a)** a reset issued while a stub-LLM turn is in flight **completes only after that turn
  finishes** — assert the turn's `WorkflowRun` reached a terminal status *before* the delete, not
  merely that no orphan exists after it;
- **(b)** no participant request 500s with `ThreadNotFoundError` during a reset — the post path
  returns `409` (intake stopped) or succeeds, never an unhandled raise;
- **(c)** no turn is silently dropped: for every accepted post during the reset window, either a
  reply message exists or the client saw a `409`/`503`;
- **(d)** after `reset_all`, **`MATCH (rc:ReadCursor) OPTIONAL MATCH (t:Thread {threadId:
  rc.threadId}) WITH rc, t WHERE t IS NULL RETURN count(rc)` is 0 for participant-owned cursors** —
  this one *does* fail if quiesce is broken, and it is the direct test for F3's real orphan class.


## 8. `GRAPH.PROFILE` — participant-scoped reads at 50 participants × 40 messages

Fixture: **50 participants × 40 messages** — v1.0 on `ws:probe-s0-reset` (3 271 nodes / 5 475
rels), v1.1 re-run on `ws:probe-s0r2` (3 281 / 5 490; +1 `User` and the `orphan-1` message):
2 004 `Message`, 200 each `WorkflowRun`/`StepRun`/`TraceEvent`, 102 `ReadCursor`, 101 `Order`,
151 `OrderLine`, 51 each `Channel`/`Thread`/`Customer`/`Cart`/`CartItem`, 52 `User`. `p-0007` is
the cross-member participant.

| Read | Anchor operator | Records at the anchor | Label scans |
|---|---|---|---|
| `resolve_token` | `Node By Index Scan \| (u:User)` | 1 | none |
| `list_participants` — bare `tokenHash IS NOT NULL` | `Node By Label Scan \| (u:User)` | 52 | 1 |
| `list_participants` — `u.userId > '' AND tokenHash IS NOT NULL` | `Node By Index Scan \| (u:User)` | 52 | none |
| `read_thread` (QUERIES.md §4) | `Node By Index Scan \| (t:Thread)` → `Conditional Variable Length Traverse` | 1 → 40 | none |
| `read_cart` (§16.5) | `Node By Index Scan \| (cart:Cart)` | 1 | none |
| `read_profile` (§17.2) | `Node By Index Scan \| (c:Customer)` | 1 | none |
| `get_customer_current_order` | `Node By Index Scan \| (cust:Customer)` | 1 | none |
| `order_belongs_to_customer` | `Node By Index Scan \| (cust:Customer)` | 1 | none |
| `ensure_participant` (replay) | 3 × `Node By Index Scan` (`User.userId`, `Agent.agentId` ×2) | 1 each | none |

**Every participant-scoped read stays index-backed at 50 participants.** The roster is the one
exception, and **v1.1 reverses v1.0's recommendation about it (F5).**

> **v1.0 told S4 to add `u.userId > ''` to the roster, claiming it "keeps the roster O(index) as
> the `User` label grows". That claim was wrong and the recommendation is withdrawn.**
> `userId > ''` matches every row, so it has **no selectivity**: the "index scan" is a full index
> traversal and both forms are O(|`User`|), visiting all 52 records either way. That is the whole
> argument, and it does not depend on a stopwatch.
>
> **v1.2 nit correction — do not read the timings as evidence of direction.** v1.1 cited "bare
> 0.082 ms vs conjunct 0.128 ms, a ~1.6× regression". The gate review measured the **opposite**
> ordering on its own fixture (bare 0.347 ms vs conjunct 0.333 ms), and my own re-run at 40
> interleaved iterations gives bare median 0.087 ms / mean 0.094 (min 0.075, max 0.319) against
> conjunct median 0.128 / mean 0.174 (min 0.110, max **1.850**). Spreads that overlap that badly,
> with a max an order of magnitude above the median, mean the difference is **noise on a
> sub-millisecond query**, not a measured regression. Both forms are the same complexity class
> and both are free at this scale. **The reason to drop the conjunct is correctness, not
> speed** — §5/F1 — and consistency: both resets and the roster now share one bare predicate, so
> there is no second form for a future editor to copy into a destructive query.
>
> The idiom itself is not unsound — it is QUERIES.md §3's `WHERE c.channelId > ''` house pattern,
> and it works when the property is reliably a non-empty string. It is simply not worth taking
> here, and in a **destructive** query it is actively dangerous.

**The two writes** (v1.1 text, `ws:probe-s0r2`, three runs each):

| Write | Anchor | Label scans | Server time | Effect |
|---|---|---|---|---|
| `reset_participant('p-0007')` | `Node By Index Scan \| (u:User)` (1 rec) | 1 — `(rc:ReadCursor)`, 102 rec, 0.023 ms | **3.9 / 4.3 / 4.8 ms** | 63 nodes / 107 rels deleted, 1 thread created |
| `reset_all_participants()` | `Node By Label Scan \| (u:User)` (52 rec → 50 after `Filter`, 0.0055 ms) | 3 — `ReadCursor` (102), `Customer` ×2 (51 each) | **234 / 241 / 244 ms** | 3 250 nodes / 5 451 rels deleted |

**v1.2's P2 liveness lookup is free.** The added
`OPTIONAL MATCH (liveT:Thread) WHERE oc IS NOT NULL AND liveT.threadId = oc.threadId` plans as
`Node By Index Scan | (liveT:Thread) | Records produced: 1, Execution time: 0.011 ms` — the
`WHERE` form still folds onto `Thread.threadId`'s index rather than degrading to a label scan, so
`reset_participant` is unchanged at ~4 ms and `reset_all` (which does not carry the filter) is
unchanged at ~240 ms.

`reset_all` is unchanged from v1.0's 236 ms despite F1's anchor change and F3's added cursor
sweep — the anchor swap is free (0.0055 ms label scan vs 0.0059 ms index scan on 52 rows), and
the sweep is free **once its `collect` is collapsed before the `users` `UNWIND`**. It was not
free before that: the first cut measured **684–692 ms**, a 3× regression, because the 100-row
`ReadCursor` set was still an open stream when the 50-row `UNWIND` multiplied it. Both numbers
are measured; the shipped text is the fast one.

The channel-ownership guard is visible in `reset_participant`'s plan doing its job:
`Conditional Traverse | (u)->(ch:Channel) | Records produced: 2` (p-0007 is a member of two
channels) followed by `Filter | Records produced: 1`.

**`reset_all`'s hot spot, and a measured tuning lever S4 should NOT take.** ~150 ms of the ~240 ms
sits in the `Aggregate` above `UNWIND msgs (2 000 rows) → OPTIONAL MATCH (r:WorkflowRun)-
[:TRIGGERED_BY]->(mm)`. Replacing that block with the QUERIES.md §12.14 vacuous-predicate trick —

```cypher
WITH users, channels, threads, msgs, pids, [x IN threads | x.threadId] AS tids
OPTIONAL MATCH (r:WorkflowRun)-[:TRIGGERED_BY]->(tm:Message)
  WHERE r.startedAt >= 0 AND tm.threadId IN tids
WITH users, channels, threads, msgs, pids, collect(DISTINCT r) AS runs
```

— anchors on the `WorkflowRun.startedAt` index (200 rows instead of 2 000) and measured
**100–112 ms vs 226–239 ms** over three runs each, identical results, zero orphans.

**Shipping the slower form.** The fast one takes **two** nullable dependencies, not one (a point
the gate review sharpened): `tm.threadId` on the deliberately unindexed, `null`-on-pre-backfill
`Message.threadId`, **and** `r.startedAt >= 0`, which silently drops any `WorkflowRun` whose
`startedAt` is absent. Either null under-deletes with no signal — the same class of defect F1
removed from the anchor. ~140 ms once or twice per demo, behind a stop-intake-drain quiesce, does
not buy that. The scoping is unaffected either way (the fast form can only under-delete, since
`tids` still derives from proven-participant channels); the lever is documented and measured so a
future workload can take it deliberately.


## 9. New indexes / constraints: **NO** — for either

Nothing to create. Every anchor these queries need is already indexed by
`bootstrap_schema.sh`, and §8's profiles prove each one is actually used.

| Property | Verdict | Evidence |
|---|---|---|
| `User.tokenHash` | **no index** | never a scan anchor. In `resolve_token` it is read after a `User.userId` index scan; in the roster and in `reset_all` it is a `Filter` above a `User` label scan. `IS NOT NULL` cannot anchor a range index, and §8/F5 measures the label scan as the *faster* of the two forms at this cardinality |
| `Channel.participantId` (new) | **no index** | only ever a `Filter` on a `ch` already bound by a `MEMBER_OF` traversal from an index-anchored `u`. A `UNIQUE` constraint was considered and **rejected on scope, not on safety** — v1.1 correction: the gate review created the constraint and verified FalkorDB **exempts both an absent property and an explicit `null`**, rejects a genuine duplicate, and handles delete-then-recreate cleanly, so it would have been *safe*. It is still declined because `ensure_participant`'s own guard already prevents a second channel per participant, §4's row-multiplication makes a duplicate marker fail loudly anyway, and adding it means a `bootstrap_schema.sh` change plus an existing-workspace migration for no measured gain |
| `Customer.customerId` | already indexed + UNIQUE | anchors both new order reads — `Node By Index Scan`, 1 record |
| `Order.orderId` | already indexed + UNIQUE | not needed as an anchor: `order_belongs_to_customer` anchors on the customer and filters the ≤N-order fan-out. Order-anchored and Customer-anchored variants were both profiled — identical plans and record counts, so the simpler Customer-anchored form ships |
| `Order.status` | **no index** | still nothing scans or lists orders by status across the workspace (unchanged from QUERIES.md §16's reasoning) |
| `ReadCursor.threadId` | **no index** | the one label scan in the reset plans: 103 records, 0.02 ms at 50 participants, growing O(participants). An index would save microseconds and cost RAM plus a bootstrap change. Revisit only if `ReadCursor` cardinality ever leaves the low hundreds |
| `Message.threadId` | **stays unindexed** | `DESIGN.md` §5.1 is right and nothing here changes it — the resets reach messages structurally, precisely so this property is not load-bearing |

**The `Channel.participantId` property is the only schema addition in this note.** It is
additive, nullable and needs no DDL, so `bootstrap_schema.sh` is untouched and every existing
workspace stays valid. It must be documented in `DESIGN.md` §5.1 and `QUERIES.md` §18.

---

## 10. The two order primitives (review finding B4)

### 10.1 `get_customer_current_order`

```cypher
// $customerId
MATCH (cust:Customer {customerId: $customerId})-[:PLACED]->(o:Order)
WITH o ORDER BY o.placedAt DESC, o.orderId DESC LIMIT 1
OPTIONAL MATCH (o)-[:HAS_LINE]->(l:OrderLine)
RETURN o.orderId AS orderId, o.status AS status,
       o.placedAt AS placedAt, o.updatedAt AS updatedAt,
       collect({productId: l.productId, name: l.name, unitPrice: l.unitPrice,
                quantity: l.quantity, lineTotal: l.lineTotal}) AS lines,
       sum(l.lineTotal) AS total
```

**"Current" is defined as the most recently *placed* order, whatever its status** — not "the most
recent non-terminal order". A demo participant walks `placed → fulfilled → delivered` and AC-7
requires them to *see* the status change; a non-terminal filter would make the order card vanish
at the moment the demo is trying to show it. A subsequent `place_order` supersedes the card.
Verified against a customer holding a `delivered` order at `placedAt 110` and a `placed` one at
`120`: returns the `placed` one, 2 lines, `total = 17.5`.

- **Return shape deliberately mirrors QUERIES.md §16.8's `get_order`** so the repository's
  existing row-shaping applies unchanged, including its two quirks: a **zero-line order** yields
  one all-`null` placeholder entry from `collect()` rather than `[]` (filter client-side, as
  `Repository.get_order` already does), and `sum()` returns a **float** (`0.0` on an empty
  aggregation, never `NULL`) — a `float`-vs-`int` JSON mismatch to expect wherever this feeds a
  response model. Both re-verified here.
- Zero rows means **either** no `Customer` **or** a `Customer` with no orders. That collapse is
  deliberate: `GET /shop/api/state` renders "no order" identically for both. The distinction
  QUERIES.md §17.2 preserves for the profile is available there if it is ever needed.
- `collect()` and `sum()` together over one fan-out are safe here for the same reason §16.8 gives:
  `LIMIT 1` guarantees exactly one `o` for the whole aggregation.
- **Tie-break.** Two orders sharing a `placedAt` millisecond resolve deterministically by
  `orderId DESC` (verified: `p-aaa-o3` beat `p-aaa-o2`). Arbitrary but *stable across polls*,
  which is what matters for a card that repaints every 2 s — the `LIMIT`-without-tiebreak
  instability that bites elsewhere on this build does not apply. One customer placing two orders
  inside one millisecond is not a reachable state through `place_order` anyway.

### 10.2 `order_belongs_to_customer`

```cypher
// $customerId, $orderId
OPTIONAL MATCH (cust:Customer {customerId: $customerId})-[:PLACED]->(o:Order {orderId: $orderId})
RETURN o IS NOT NULL AS owned, o.status AS status
```

Always exactly one row — `owned` is never `null`, so the caller never has to distinguish "no row"
from "not owned". `status` rides along free, letting `advance_own_order` decide `404` (not theirs
/ no such order) before the CAS without a second round trip. Verified across all five cases:
own order → `[True, 'placed']`; another participant's order, the non-participant's order, an
unknown order, and an unknown customer → `[False, None]` in every case.

**This is the ownership check §4.6 requires, and it is not optional.** `services.advance_order`'s
CAS is keyed on `orderId` alone; without this gate any participant who learned another's
`orderId` could cancel their order. The storefront never exposes an `orderId` in a request body
(§5.2), so this is defence in depth — but it is the only thing standing between the two.

---

## 11. Verification log

Environment: FalkorDB `v4.18.11` / module `41811` on Redis 8.x (`falkordb-dev`,
`localhost:6379`); client `falkordb-py` 1.6.1 from `falkor-chat/server/.venv`. Dates: v1.0 and
v1.1 both 2026-09-02. Probe graphs: **`ws:probe-s0-reset`** (v1.0) and **`ws:probe-s0r2`**
(v1.1), each created for this unit and schema-bootstrapped from `bootstrap_schema.sh`'s own
`bootstrap_workspace` at `EMBEDDING_DIM=1024`. **Correcting v1.0's claim that the probe was
"deleted at the end of the unit" (review nit):** its *data* was wiped, but the graph key
survived holding 0 nodes and it is still present, awaiting the stakeholder's cleanup along with
`probe_u8_rename_dst` and `ws:s1v6`. `ws:probe-s0r2`'s disposal is stated in §12.

No other graph was written to in either pass. Re-confirmed after the v1.1 pass: `ws:acme` holds
2 `Channel` / 2 `Thread` / 52 `Message` / 1 `User` / 544 `Entity` / 87 `Chunk` / 29 `Document`,
`reference` 15 `Product` / 19 `Step` / 5 `WorkflowDef`, `ws:test` untouched. Only
`bootstrap_workspace` was extracted and executed, never `bootstrap_reference`. The falkor-chat
pytest suite was not run in either pass.

**v1.0 checks (all still valid unless a v1.1 row supersedes them):**

| # | Check | Result |
|---|---|---|
| 1 | `ensure_participant` status rows; collided / agent-missing paths write **nothing** | ✅ §3 |
| 2 | `ensure_participant` replay returns the **stored** channel/thread/language | ✅ |
| 3 | Multi-`CREATE` + outer-variable endpoint inside `FOREACH` | ✅ works on this build |
| 4 | `reset_participant` victim set, `Agent`-authored reply deleted, other participants untouched | ✅ §2.2 |
| 5 | `reset_participant` on `u1` / `u2` / `assistant` / unknown id — **0 rows, 0 deleted, 0 created** | ✅ §2.2 |
| 6 | Cross-member participant leaves `demo-general`/`demo-welcome` whole | ✅ §2.2 |
| 7 | Both resets idempotent; `reset_all` on a clean graph returns an all-zeros row | ✅ |
| 8 | `reset_all` survivors positively asserted by identity, 0 orphan runs/steps/traces | ✅ §2.2 |
| 9 | Atomicity — a constraint-violating reset writes nothing at all | ✅ §7 |
| 10 | Boundary edges — surviving `Chunk`/`Document` and `EMITTED` after a target is deleted | ✅ §6 |
| 11 | Order primitives — 6 + 5 cases incl. tie-break, zero-line order, cross-participant | ✅ §10 |
| 12 | `CALL db.constraints()` — the four UNIQUE properties the join touches are server-minted | ✅ §3 |

**v1.1 re-verification (`ws:probe-s0r2`), every changed query re-executed:**

| # | Check | Result |
|---|---|---|
| 13 | **F1** — `42 > ''`, `42.5 > ''`, `true > ''`, `null > ''` all `null`; `'abc' > ''` `true`; `'' > ''` `false` | ✅ §5 |
| 14 | **F1** — with `{userId: 42, tokenHash:'h'}` present, the bare predicate matches it and the conjunct form silently drops it | ✅ §5 |
| 15 | **F1/F5** — roster both forms, 20 runs: bare 0.082 ms median / label scan; conjunct 0.128 ms / index scan; identical result sets | ✅ §8 |
| 16 | **F2** — `reset_participant` on a mismatched-marker channel **and** on a participant with no `MEMBER_OF`: `scoped=false`, 0 deleted / 0 created / 0 properties set, empty identity delta | ✅ §4 |
| 17 | **F2** — `reset_all` with both unscoped shapes: `unscopedCount: 2`, both left whole, `userCount: 3` | ✅ §5 |
| 18 | **F3** — post-`reset_all` writes: `post_message` 0 rows/0 nodes, `record_step_and_advance` 0/0, `append_trace_event` 0/0, `advance_cursor` **1 row / 1 orphan `ReadCursor`** | ✅ §7 |
| 19 | **F3** — the structural `HAS_CURSOR` sweep collects a participant-owned orphan across a later reset; `Agent`-owned one persists; both `demo-welcome` cursors untouched | ✅ §7 |
| 20 | **F3** — the declined complete sweep parses and would also collect a **non-participant's** dangling cursor (`u1:ghost2`) | ✅ §7 |
| 21 | **F4** — variant A verbatim **raises and writes nothing**; delete-only ablation deletes 23; variant B runs clean verbatim (6 deleted) | ✅ §2.3 |
| 22 | **F4** — two channels sharing a marker: `reset_participant` raises permanently, nothing written; `reset_all` still collects | ✅ §2.3, §4 |
| 23 | **F7** — all five `ensure_participant` status rows, incl. `u1` returning `existedParticipant=false` with null ids | ✅ §3 |
| 24 | **F9** — off-chain `orphan-1` survives both resets | ✅ §4, §6 |
| 25 | **Own catch** — `--` is not a comment on this build; the v1.0 note published unparseable query text. All comments now `//` | ✅ preamble |
| 26 | **Own catch** — F3's sweep cost 684–692 ms until `tcur` was collapsed before the `users` `UNWIND`; 235–247 ms after | ✅ §5, §8 |
| 27 | Full regression re-run after every change: R1 (both resets + trap), R2 (four no-ops), R3 (idempotency), R4 (`reset_all` ×2), R5 (order primitives) — survivors asserted **by identity**, field by field | ✅ all PASS |

**v1.2 re-verification (`ws:probe-s0r3`, fixture extended with the two `p-ccc` cursors):**

| # | Check | Result |
|---|---|---|
| 28 | **P1** — the finished file re-parsed with `markdown-it` 3.0.0: fenced-block count and max block length (see §12's note and the return message) | ✅ 6 blocks, max 77 lines, 13 tables; negative control re-gluing the fences drops it to 4 blocks / max 513 lines / 9 tables |
| 29 | **P1** — extract-and-execute loop hardened with block-count and max-block-length assertions, then re-run: every block executes and matches the verified query text | ✅ |
| 30 | **P2** — v1.1's wider sweep reproduced (`cursorCount: 4`, `p-ccc:demo-welcome` deleted); v1.2's narrowed sweep keeps it (`cursorCount: 3`, `deletedCount: 18`) while still collecting `p-ccc:th-gone` | ✅ §4 |
| 31 | **P2** — `reset_all` still sweeps every own-cursor incl. `p-ccc:demo-welcome`, and afterwards `MATCH (rc:ReadCursor) WHERE NOT ()-[:HAS_CURSOR]->(rc)` returns **empty** | ✅ §5 |
| 32 | **P2** — the liveness `OPTIONAL MATCH` plans as `Node By Index Scan \| (liveT:Thread)`, 1 record, 0.011 ms; no label scan introduced | ✅ §8 |
| 33 | **P3** — `unscopedIds: ['p-ddd', 'p-eee']` returned alongside `unscopedCount` | ✅ §5 |
| 34 | **Nit** — roster re-timed 40× interleaved: bare median 0.087 / mean 0.094 / max 0.319 ms; conjunct median 0.128 / mean 0.174 / max 1.850 ms. Overlapping spreads; the review measured the reverse ordering. Re-stated as noise | ✅ §8 |
| 35 | **Live bug found and fixed while implementing P2** — the first cut wrote `OPTIONAL MATCH (liveT:Thread {threadId: own.threadId})`, which raises `_AR_EXP_UpdateEntityIdx: No record was given to locate a value with alias own` when the participant holds **no** cursors (R3's second reset). Fixed by collecting `own` first and matching with a guarded `WHERE`; the empty case now passes | ✅ §4 |
| 36 | Full regression re-run on the v1.2 text — R1–R5, plus the F2/F3/F4 suites — survivors asserted by identity, with the cursor assertion made reset-aware (3 demo cursors after `reset_participant`, 2 after `reset_all`) | ✅ all PASS |
| 37 | Scale re-measure at 50×40: `reset_participant` 3.9/4.3/4.8 ms; `reset_all` 234/241/244 ms, 3 250 nodes / 5 451 rels, 0 orphan runs/steps/traces, survivors `demo-general` + `demo-welcome` + 4 messages (3 demo + `orphan-1`) | ✅ §8 |


## 12. Hand-off to S4

- **Implement §3, §4, §5, §10.1 and §10.2 verbatim.** They are the live-verified v1.2 text, not a
  sketch. Every value is a parameter; no string concatenation of caller input anywhere. Cypher
  comments are `//` — `--` does not parse on this build.
- **If you extract these blocks programmatically, assert the extraction itself.** v1.1 shipped two
  closing fences glued to their last code line; a lenient extractor read all six blocks and ran
  them green while a conformant CommonMark parser saw **four**, swallowing §5–§8 into one
  455-line block in which `reset_all_participants` was not extractable at all. The loop that
  guards this note now asserts **block count** and **max block length** alongside execution, and
  re-parses the file with `markdown-it`. Any extractor S4 writes should do the same: a
  three-figure-line "Cypher block" is self-evidently wrong and is the cheapest possible tripwire.
- **Routing:** `ensure_participant`, `reset_participant`, `reset_all_participants` are writes →
  `.query()`. `get_customer_current_order`, `order_belongs_to_customer` are reads →
  `.ro_query()`.
- **`QUERIES.md` §18** gets all five queries plus: the `Channel.participantId` property, the two
  guards and *why* they are provenance rather than id-equality (§1.1), the `dev_surface=False`
  dependency for run completeness (§4), the `HEAD`/`NEXT` chain invariant (§4), and the
  `ReadCursor` orphan class with its `Agent`-owned residual (§7). `DESIGN.md` §5.1's arrow
  notation gains `Channel {channelId, name, participantId, createdAt}`.
- **Do NOT add `u.userId > ''` anywhere** — not to the roster, not to either reset. v1.0
  recommended it; v1.1 withdraws it on measurement (§8) and on correctness (§5). Use the bare
  `WHERE u.tokenHash IS NOT NULL` in all three places.
- **The two anomaly signals are a response contract, not a logging preference (P3).** Both mean
  *a participant could not be resolved and was not reset*, and prose in a design note does not
  survive into an implementation — so S4/S7/S10 must implement these as stated:

  | Signal | Required behaviour | Why |
  |---|---|---|
  | `reset_participant` returns `scoped=false` | **`409 Conflict`**, body naming the participant and a machine-readable code (e.g. `"unscoped_participant"`). **Never `200`.** | Nothing was reset and nothing will be until the graph is repaired. A `200` here is the same class of lie v1.0's partial delete told. |
  | `reset_participant` returns **zero rows** | `404`/`401` per the route's existing not-a-participant handling | Indistinguishable from an already-deleted participant; not an anomaly |
  | `reset_all` returns `unscopedCount > 0` | **`200` with `incomplete: true` and `unresolved: unscopedIds`** in the body — the reset did do everything it could, so it is not an error, but the response must not read as clean | The presenter is about to tell a room the demo is reset. `unscopedIds` names exactly whose state is still live. |
  | `reset_all` returns `unscopedCount == 0` | `200`, no `incomplete` flag | The normal path |
  | Either raises `unique constraint violation on node of type Thread` | propagate as a `5xx`; do **not** retry | The duplicate-marker fail-safe (§4). A retry re-raises forever; the graph needs repair. |

  `unscopedIds` exists in the status row (v1.2) precisely so the `unresolved` field can be
  populated without a second query.
- **A client-side timeout on a reset means *unknown*, not "nothing changed"** — re-read state
  before reporting (§7, F8).
- **S4's positive test** (the one label checks structurally cannot make) is §2.1's fixture: seed a
  non-participant `Channel` + `Thread` + `Message` owned by a `User` with **no `tokenHash`**, and
  — stronger, and recommended — make one participant a genuine `MEMBER_OF` of it. Assert those
  three survive `reset_all` **by identity**, not by label. §2.3 row B is the proof the label form
  passes on a wipe. Add §2.1's three v1.1 shapes too: a mismatched-marker channel, a participant
  with no `MEMBER_OF`, and an off-chain `Message`.
- **S7/S10's quiesce done-condition must be §7's four-part replacement**, not v1.0's
  orphan-count wording, which cannot fail.
- **Do not relax either guard for readability.** §2.3 rows A-delete-only and B are what each one
  costs — read row A's correction note first, so the ablation is not mistaken for the shipped
  behaviour.

**Open items handed onward, not decided here.**

1. §4's note on re-writing the profile name after "reset mine" (S7's service wrapper, existing
   `save_profile` call, no new Cypher).
2. **Does the storefront advance read-cursors at all?** (review OQ-1). §5.2's
   `GET /shop/api/messages?since=<ms>` reads as an explicit-`since` read, not cursor mode, and
   `ensure_participant` creates no cursor — my fixtures seed them defensively. If the storefront
   never calls `advance_cursor`, F3's orphan class is a platform-wide `QUERIES.md` concern and
   both resets' cursor blocks are pure defence. `architect` to confirm; it changes the priority
   of §7's residual, not its correctness. The cursor handling is cheap either way (§8) and is
   staying in regardless.
3. **`MAX_QUEUED_QUERIES 25` under `reset_all`** (review OQ-2). The ~240 ms stop-the-world write
   against 50 participants polling at 2 s is estimated at ~18 queued queries — under the cap, but
   not by much. Not measurable from S0 (it needs concurrent load against the shared instance);
   **S15's load harness should assert it** rather than S0 reasoning about it.

**Probe-graph disposal.** `ws:probe-s0r2` is this pass's throwaway graph. Its data is wiped; the
key is removed by `redis-cli -h localhost -p 6379 GRAPH.DELETE ws:probe-s0r2` — see the hand-back
note accompanying this revision.
