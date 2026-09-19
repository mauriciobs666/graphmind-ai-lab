# Embedding model migration & index rebuild — Graph Design Note

> **Status:** active · **Owner:** `graph-dba` · **Tracks:** — (M6+)

Companion to `falkor-chat/docs/plans/embedding-migration.md` §4 (the six-item handoff). Answers
every item precisely enough for `coder`/`tdd-engineer` to implement `scripts/embedding_migration.py`
with no further Cypher/DDL judgment calls. All Cypher below was live-verified against disposable
scratch graphs (`ws:gdba_embmig_probe`, `ws:gdba_embmig_probe2`, `ws:gdba_embmig_probe3` —
`GRAPH.DELETE`d at the end of this session; no named workspace touched), on the pinned
`falkordb/falkordb:v4.18.11` build (module `41811`).

**Headline result up front: item 6's ordering (re-embed all rows to completion, then rebuild the
index — never the reverse) is SAFE as designed.** No amendment to plan §3.4 needed. But this
verification surfaced a real, previously-undocumented engine bug (below, item 1) that the plan's
own recommended query shape happens to avoid — flagged here so it's never hit by accident in a
future revision of this script.

---

## Item 6 — ordering safety (verify, don't just confirm DDL) — SAFE

**Test:** on a throwaway `ws:gdba_embmig_probe`, bootstrapped with a `Message.msgId` range
index + `UNIQUE` constraint and a `Message.embedding` **dim-4** vector index, 5 nodes were
written at dim 4 with `embeddingModel:'old'`. Then, **one row at a time**, each node's `embedding`
was overwritten to a dim-8 vector with `embeddingModel:'new'` — reproducing exactly the window
item 6 asked about: an index still declared at dimension A while nodes are incrementally rewritten
to dimension B, one row at a time, with both dimensions coexisting under that one index in the
interim. After **every single** rewrite:

- The write itself always succeeded (`Properties set: 2` each time — `embedding` + `embeddingModel`
  — plus a same-count `Properties removed` restating the *quirks* file's existing "overwrite
  reports a removal it didn't perform" entry, harmless).
- The rewritten node stayed fully `MATCH`-able with its new `embeddingModel` value, every time.
- The dim-4 index's ANN query (`db.idx.vector.queryNodes` with a dim-4 probe vector) correctly
  dropped exactly the rewritten node from its results and kept returning every node **not yet**
  rewritten, in the right count, at every step (5 → 4 → 3 → 2 → 1 → 0 as each of the 5 rows was
  migrated) — no error, no crash, no stale/duplicate/missing row, no cross-contamination between
  the old-dim and new-dim rows coexisting in the same label.
- No FalkorDB/Redis-level error, warning, or process instability at any step (`redis-cli PING`
  healthy throughout; no log noise).

This directly confirms DESIGN §7.1's "wrong-dim write silently drops out of ANN, never an error"
fact holds **not just in the documented steady state** but also **incrementally, mid-migration,
with an old-dimension index still attached and old/new-dimension vectors coexisting under it for
an arbitrarily long window** — exactly the write pattern item 6 flagged as unverified. There is no
internal consistency check on this build that behaves differently under this pattern. **Plan
§3.4's sequence stands unamended.**

Verification commands (representative step, `i`=1):
```
MATCH (n:Message {msgId:'m1'})
SET n.embedding = vecf32([0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]), n.embeddingModel = 'new'
```
```
CALL db.idx.vector.queryNodes('Message','embedding',10,vecf32([0.3,0.3,0.3,0.3]))
YIELD node RETURN node.msgId ORDER BY node.msgId
```
(dim-4 probe against the still-dim-4 index; migrated rows absent from the result set, confirmed
at every one of the 5 steps.)

---

## Item 5 — vector-index rebuild DDL

**The pair, generalized to `(ws, label, newDim)`:**

```cypher
DROP VECTOR INDEX FOR (n:{label}) ON (n.embedding)
```
```cypher
CREATE VECTOR INDEX FOR (n:{label}) ON (n.embedding)
OPTIONS {dimension: {newDim}, similarityFunction: 'cosine'}
```
(`{label}`/`{newDim}` are Python f-string interpolation into the query **text** — DDL identifiers
and `OPTIONS` map keys aren't parameterizable on this build, same posture `conftest.py`'s
`rebuild_vector_indexes` already takes; `label` must come from the fixed `("Message", "Chunk")`
tuple only, never free caller input.)

**Confirmed safe over a graph whose nodes already all carry uniformly-`newDim` vectors** — live-
verified directly (`ws:gdba_embmig_probe`, continuing from item 6's end state, all 5 nodes at
dim 8): the drop+recreate pair ran clean, `db.indexes()` read back `dimension: 8` immediately, and
an ANN query with a dim-8 probe vector returned all 5 nodes. This is exactly `conftest.py`'s own
happy path, now confirmed at a *migrated* end-state rather than a freshly-bootstrapped one — no
difference found.

**No other index needs to be dropped/recreated alongside it.** Verified: the range index on
`Message.msgId` (and its `UNIQUE` constraint) was read back via `db.indexes()` immediately after
the vector-index drop+recreate, unchanged (`{msgId: [RANGE], embedding: [VECTOR]}`). The vector
index is independent DDL from the range/full-text indexes on the same label — dropping/recreating
one never touches the others. Nothing else in this migration's write pattern (only `embedding`/
`embeddingModel` are touched) invalidates the full-text index on `Message.text` either.

**`DROP VECTOR INDEX` on a label with zero vectors ever written: succeeds cleanly, same as any
other drop — but `DROP VECTOR INDEX` on a label with NO vector index present at all is a hard
error, not a no-op. This is the one operationally sharp finding here:**

| State when `DROP VECTOR INDEX` is called | Result |
|---|---|
| Index exists, zero nodes of that label at all | `Indices deleted: 1` — succeeds |
| Index exists, nodes exist but none carry `.embedding` | `Indices deleted: 1` — succeeds |
| **No vector index exists on that label/property at all** | `ERR Unable to drop index on :Label(embedding): no such index.` |

Verified live, in that order, on `ws:gdba_embmig_probe`'s `Chunk` label (created with the index,
zero `Chunk` nodes → drop succeeded; recreated, one `Chunk` node with no `.embedding` written →
drop succeeded; dropped a second time with nothing to drop → hard error).

**Consequence for the migration script — this is a real resumability hazard, not just an edge
case:** every real workspace has the vector index from `bootstrap_schema.sh`, so the *normal* path
never hits the missing-index error. But if `migrate` crashes **between** the `DROP` and the
`CREATE`, a naive unconditional retry of "drop, then create" on resume will hard-error on the drop
(nothing left to drop). **The implementer must guard the drop**, either by:
- checking existence first — `Repository.read_index_dimension(ws, label=label) is not None` (this
  method already exists, `repository.py:805`, exactly for this kind of check) — and skipping the
  `DROP` when it's already `None`, or
- catching the specific `ResponseError` text (`"no such index"`) around the `DROP` call and
  treating it as "already dropped, proceed to `CREATE`."

The first option is cleaner (no string-matching a Redis error message) and reuses an existing,
already-tested repository method — recommended. Either way, **do not call `DROP VECTOR INDEX`
unconditionally with no guard** — that's the one shape verified to fail loudly on a legitimate
resume path.

---

## Item 4 — `embeddingModel` marker property: no index, confirmed; no `bootstrap_schema.sh` change needed

**Confirmed: no index.** Concur with the plan's own recommendation. The property is read only by
this migration script's own occasional, operator-invoked full-label scans (items 1 and 3 below) —
never a hot-path filter on the ordinary read/write surface — so there is no query-time selectivity
need to justify the RAM cost of a range index on every `Message`/`Chunk` node (rule 6, "RAM is the
binding constraint"). A range index here would also do nothing for the read-unmigrated-batch
query's actual anchor, which is `msgId`/`chunkId` (item 1) — `embeddingModel` stays a plain,
filtered-but-unindexed property, exactly like `Message.threadId` (DESIGN §7.1, "deliberately
unindexed... nav metadata, not an anchor" — same shape, different reason, same conclusion).

**Confirmed: `bootstrap_schema.sh` needs no change at all.** A plain scalar property needs no DDL
of any kind on this build — it is written the first time any node sets it (`SET n.embeddingModel =
$targetRef`), read back correctly whether present, absent, or explicitly `null` (all render as
`coalesce(n.embeddingModel, '') <> $targetRef` → `true`), and requires no prior `CREATE INDEX`. This
was exercised directly: none of the throwaway graphs in this session's testing ever ran a DDL
statement for `embeddingModel`, and every read/write against it worked exactly as expected from
first use.

---

## Item 3 — count-unmigrated query

One per label, `Message`/`Chunk`, identical shape:

```cypher
CYPHER target=$targetRef
MATCH (n:Message)
WHERE coalesce(n.embeddingModel, '') <> $target
RETURN count(n) AS unmigrated
```
```cypher
MATCH (n:Message) RETURN count(n) AS total
```

(swap `Chunk` in for `Chunk` rows). Two separate statements — FalkorDB refuses multi-statement
queries (quirks file), and there is no single-query shape that returns both counts without a
`CASE`-based conditional aggregate, which buys nothing here for the extra complexity; two cheap
label-scan counts are simpler and just as fast at migration scale.

**Verified clean on a graph with zero `Message` nodes at all**: both queries return `0` with no
error — no special-case needed in the implementation. (`ws:gdba_embmig_probe3`, freshly bootstrapped
with only the `msgId` range index, zero nodes created.)

**Report exactly `migrated = total - unmigrated`, `unmigrated`, `total`** — matches the acceptance
criteria's "migrated/total, zero unmigrated" wording directly, no derived-state ambiguity.

---

## Item 2 — write-one-embedding query

One per label, mirroring `Repository.set_embedding`/`set_chunk_embedding`'s existing shape
(`repository.py:773`, `:1202`) with the marker property added:

```cypher
MATCH (n:Message {msgId: $id})
SET n.embedding = vecf32($embedding), n.embeddingModel = $targetRef
```
(swap `Chunk {chunkId: $id}` for `Chunk` rows.)

**No-op detection:** exactly the pattern the two existing repository methods already use —
`res.properties_set > 0` on the `falkordb-py` query result. Since this query always sets **two**
properties on a match (`embedding` + `embeddingModel`), a genuine no-op (id not found) reports
`properties_set == 0`; a real match reports `2` (or `4`, per the "overwrite reports a removal it
did not perform" quirk's paired counter — `properties_set` itself is unaffected by that quirk, only
`properties_removed` is, so `> 0` stays a correct no-op test). Client-side dimension validation
(`len(embedding) == targetDim`) belongs in the calling script **before** this query runs, mirroring
`set_embedding`'s own `EmbeddingDimensionError` guard — this query itself does not re-validate
length, same division of responsibility as the existing methods.

Do **not** route this through `Repository.set_embedding`/`set_chunk_embedding` directly — those
methods don't set `embeddingModel` and validate against `config.EMBEDDING_DIM`/the caller's
`expected_dim` in a way that isn't this script's target dimension by default. Add two new
repository methods (or inline queries in `embedding_migration.py`) with this exact shape instead;
either is a small implementer choice, not a design fork.

---

## Item 1 — read-unmigrated-batch query: keyset, anchored on the existing `msgId`/`chunkId` index

**Recommended query, one per label:**

```cypher
CYPHER lastId=$lastId target=$targetRef b=$batchSize
MATCH (n:Message)
WHERE n.msgId > $lastId AND coalesce(n.embeddingModel, '') <> $target
RETURN n.msgId AS id, n.text AS text
ORDER BY n.msgId ASC
LIMIT $b
```
(swap `Chunk`/`chunkId` for `Chunk` rows; `$lastId` starts at `''` for the first call of a fresh
`migrate` invocation — every real `msgId`/`chunkId` sorts after the empty string.)

**Confirmed via `GRAPH.PROFILE`: this is the right shape, and it is NOT a full-label-scan-per-batch
the way a naive re-query would be.** `msgId`/`chunkId` already carry a range index (backing their
`UNIQUE` constraint, DESIGN §7.1) — `embeddingModel` does not (item 4) and never will. Profiling on
a 5000-row scratch `Message` label:

| Query shape | Plan | Cost at deep pagination (500 of 5005 rows left, cursor at row ~4500) |
|---|---|---|
| `SKIP`-based (`ORDER BY msgId SKIP $skip LIMIT $b`) | `Node By Label Scan` (5005) → `Filter` (evaluates all 5000 matching) → `Sort` (over the full remaining-after-skip set) → `Skip` → `Limit` | `Sort` alone: **6.5 ms**, over 4550 records, at `skip=4500` — and every earlier page pays this same full-label-scan-then-sort cost too, since nothing shrinks what gets scanned |
| **Keyset (`WHERE msgId > $lastId`)** | `Node By Index Scan` (bounded by `$lastId`) → `Filter` → `Sort` (small) → `Limit` | `Node By Index Scan` produces only **499** records (not 5005) at the equivalent resume point; `Sort` over a handful, **~2.4 ms total** |

The keyset form's `Node By Index Scan` genuinely narrows to "everything after `$lastId`" — verified
by watching `Records produced` on the scan operator shrink from 5005 (at `$lastId=''`) to 499 (at
`$lastId='msg004500'`), matching the exact count of ids strictly greater than the cursor. **The
`SKIP` form never narrows** — every page re-scans and re-filters the entire label and re-sorts the
entire remaining matching set, so total cost across a full migration is roughly quadratic in row
count (`O(rows² / batchSize)`), not linear. At `ws:eval`'s current small scale this difference is
invisible; at `docs/test-reports/capacity-report.md`'s documented larger per-workspace row counts
it would not be. **Use the keyset form — confirmed the right choice, not just "likely fine."**

The `coalesce(embeddingModel,'') <> $target` predicate itself stays an un-indexed `Filter` either
way (item 4 — no index on this property, correctly) — the keyset predicate's job is only to bound
*which rows the index scan has to visit at all*, not to make the marker-property check itself
index-anchored. This composes correctly with the migration's strictly-ascending, one-batch-at-a-
time write order (plan §3.3): because rows below any given `$lastId` are only ever advanced past
once their whole batch has been fully written, resuming from the last successfully-processed id
never skips a row that failed mid-batch — the `coalesce(...) <> $target` clause remains the actual
correctness gate (a row keeps matching until it's genuinely migrated), and `$lastId` is purely a
performance anchor, never trusted alone. On a **fresh `migrate` invocation after a crash**, the
simplest and safest choice is to restart `$lastId` at `''` for that label (accepting one pass of
re-scanning whatever was already completed before the crash, at scan cost, not embed-and-write
cost — cheap) rather than trying to persist a cross-process cursor; the coalesce filter guarantees
correctness regardless of where `$lastId` restarts.

**Live-verified engine bug found during this profiling, flagged here and logged to the quirks
file (below) — does not affect the recommended query, but is a real landmine for any future
revision:** on this build, `<` and `<=` against an **indexed** string property fold into a broken
`Node By Index Scan` that silently returns the **entire label**, not the filtered subset — `WHERE
n.msgId < 'msg000050'` and `WHERE n.msgId <= 'msg000050'` both returned all 100 rows of a 100-row
test label (expected 50/51). `>` and `>=` on the same indexed property are unaffected and return
exactly the right rows (verified 49/1 against the same data) — confirmed this is index-scan-
specific, not a general string-comparison bug, by re-running the identical `<`/`<=` predicate
against an **unindexed** copy of the same property (correct 50/51 there, via a plain `Filter`
plan, no index involved). **The read-unmigrated-batch query above uses only `>`, which is the
verified-safe direction — no design change needed — but this rules out ever "flipping" the
pagination to `<`/`<=` for a descending cursor or a different resume strategy on this build.**

---

## Summary table for the implementer

| Item | Verdict |
|---|---|
| 1. Read-unmigrated-batch | Keyset (`WHERE id > $lastId AND coalesce(...) <> $target ... ORDER BY id LIMIT B`), confirmed by profile — not SKIP. Uses only `>` (verified safe; `<`/`<=` on an indexed property is broken on this build, see quirks entry). |
| 2. Write-one-embedding | `MATCH (n:Label {idProp: $id}) SET n.embedding = vecf32($embedding), n.embeddingModel = $targetRef`; no-op via `properties_set > 0`. |
| 3. Count-unmigrated | Two label-scan counts (`unmigrated` via `coalesce(...) <> $target`, `total` unconditional); verified `0`/`0` on an empty graph. |
| 4. `embeddingModel` indexed? | **No** — confirmed. No `bootstrap_schema.sh` change needed — a plain property needs no DDL. |
| 5. Vector-index rebuild DDL | `DROP VECTOR INDEX` + `CREATE VECTOR INDEX ... OPTIONS {dimension:newDim,...}`, confirmed safe over uniformly-migrated nodes, confirmed no other index needs touching. **Guard the `DROP`**: it succeeds on zero-vector labels but hard-errors if the index doesn't exist at all (a crash-between-drop-and-create resume hazard) — check `read_index_dimension(...) is not None` first. |
| 6. Ordering safety | **SAFE as designed.** Live-verified incremental one-row-at-a-time re-embed under an old-dimension index produces zero errors/corruption at every step; plan §3.4 needs no amendment. |

## New quirks-file entry filed

`claude/graph-dba/falkordb-quirks.md` gains a new dated entry (this session) for the `<`/`<=`
indexed-string-property bug found above — the general engine fact, kept there rather than
duplicated in this project-scoped note per this repo's own convention (`falkor-chat/AGENTS.md`,
"Live-verified FalkorDB facts").
