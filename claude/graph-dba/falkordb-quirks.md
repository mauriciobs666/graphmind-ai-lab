# FalkorDB verified engine quirks — this lab's pinned build

> **Live-verified knowledge base for `graph-dba`.** Facts confirmed by hands-on
> testing against this lab's running FalkorDB instance, not just docs — they
> diverge from the general documentation or from Neo4j assumptions. Treat them as
> **ground truth for this build**.
>
> **This is a cache, not the source of truth.** It is pinned to a specific build:
> `falkordb/falkordb:v4.18.11` (tagged release, 2026-06-24), Redis 8.6.3, graph
> module reporting version **`41811`**. **Re-verify every entry against the live
> instance on any release upgrade**, and re-stamp the date below.
>
> **Verified: 2026-07-05 against the live `falkordb:edge` instance (module 999999,
> tracking `main` past the v4.18.11 tag).** Deployment pinned to `v4.18.11` on
> 2026-07-09; the falkor-chat query suite (193/193, which encodes the DDL/ordering/
> anchor quirks below) re-ran green on the pinned build. Entries not exercised by
> that suite still carry their edge-build verification dates — re-probe individually
> if one becomes load-bearing.

`graph-dba.md` (the always-on prompt) points here and stays lean; this file holds
the perishable, growing fact list. When another lab project accumulates its own
"live-verified FalkorDB facts" against this same build, **fold the generic ones in
here** rather than letting them sit siloed in that project's docs — keep the
project-specific corollaries in the project's own `AGENTS.md`, each pointing back
to the general fact here.

---

## Indexing, constraints & DDL

- **Vector index creation is DDL, not a procedure:**
  ```sql
  CREATE VECTOR INDEX FOR (n:Label) ON (n.prop)
  OPTIONS {dimension: N, similarityFunction: 'cosine'}
  ```
  `db.idx.vector.createNodeIndex` is **not** a registered procedure on this build.
- **Index before constraint, always** — `GRAPH.CONSTRAINT CREATE` requires a
  pre-existing exact-match index on the same property, or it fails with
  `"missing supporting exact-match index"`.
- **`GRAPH.CONSTRAINT CREATE`'s entity-type keyword is `NODE`/`RELATIONSHIP`, not `LABEL`**
  (verified 2026-08-18, module 41811). `GRAPH.CONSTRAINT CREATE <key> UNIQUE LABEL <Label>
  PROPERTIES <n> <prop>...` fails outright with `"Invalid constraint entity type"` — the
  correct syntax is `GRAPH.CONSTRAINT CREATE <key> UNIQUE NODE <Label> PROPERTIES <n>
  <prop>...`. Creation is also **async**: the command returns `PENDING` immediately, not a
  confirmation; poll `CALL db.constraints()` (or `db.indexes()` for the paired index) and
  check `status` for `OPERATIONAL` before relying on the constraint being enforced.
- **Composite constraints** (`PROPERTIES 2 key version`) are supported and operational.
- **A `UNIQUE` node constraint does not constrain nodes that lack the property — and there is no
  such thing as a *stored* null, so "absent" and "explicitly null" are the same state** (verified
  2026-09-08, module `41811`, disposable graph `cobb_u21_probe`, deleted after). Any number of
  nodes with the constrained property absent coexist; `CREATE (:Chan {name:'c', pid:null})`
  reports `Properties set: 1` (the `name` only) and `keys(n)` comes back `[name]` — the null is
  discarded at write, never stored, so the constraint has nothing to compare. Two nodes with the
  same non-null string are correctly rejected (`unique constraint violation on node of type
  Chan`), and `DETACH DELETE` followed by re-`CREATE` of the same value succeeds cleanly — the
  index entry is released with the node, so a delete-then-recreate cycle is not a re-join hazard.
  **Consequence: a `UNIQUE` on a *nullable marker* property is safe** (it constrains exactly the
  rows that carry a value) — but it is emphatically **not** an existence constraint; see the
  `RETURN n.prop` → `null` entry under *Cypher dialect* for the projection side of the same gap.
- **`EXISTS { MATCH … }` and `exists((pattern))` are both unusable — the working orphan-detection
  shape is `OPTIONAL MATCH … WITH x, t WHERE t IS NULL`** (verified 2026-09-08, module `41811`).
  They fail for *different* reasons, which matters when reading the error: `WHERE EXISTS { MATCH
  (n)-[:R]->() }` is a genuine **parse** error (*"Invalid input '(': expected ':', ',' or '}'"*),
  while `WHERE exists((n)-[]->())` parses and then fails at plan time with *"Unable to resolve
  filtered alias '(n)-[]->()'"*. Neither is a subquery this engine supports; the `OPTIONAL MATCH`
  + `IS NULL` anti-join is the portable form and runs clean.
- **Fulltext** (`db.idx.fulltext.createNodeIndex` / `queryNodes`) confirmed working. RediSearch
  fuzzy term syntax also confirmed live: `%term%` (1-edit-distance fuzzy) and `%%term%%` (2-edit)
  both match a typo'd query against an indexed exact string (verified 2026-08-22, module `41811`,
  e.g. `%Acmee%` and `%%Acmeee%%` both still matched an indexed `'Acme Corporation'`); a
  non-matching term correctly returns zero rows.
- **`db.idx.fulltext.queryNodes(...)` against a label with NO fulltext index created at all
  silently returns zero rows — no error, indistinguishable from "no match found"** (verified
  2026-08-25, module `41811`, disposable graph). A probe against an un-bootstrapped
  workspace/label looks exactly like a correctly-working-but-empty search until the index is
  actually created; don't infer "the index exists and this is a true empty result" from a clean
  zero-row response alone — confirm the index itself first (`CALL db.indexes()` or a bootstrap
  script's own idempotent creation) before trusting a fulltext-search miss.
- **`db.labels()` and `db.relationshipTypes()` are asymmetric for zero-data schema elements.** A
  node label registers in `db.labels()` (count 0) as soon as `CREATE INDEX FOR (n:Label) ON (...)`
  runs — it's index metadata, not data. A relationship type registers in `db.relationshipTypes()`
  only once at least one edge of that type has actually been **created**; `CREATE INDEX` has no
  relationship-type equivalent that pre-registers a type before any edge exists. Empirically
  re-confirmed 2026-08-25 on a disposable graph: `db.relationshipTypes()` returns empty right after
  `CREATE INDEX FOR (n:L) ON (n.x)`, then shows the type the instant one edge of it is `CREATE`d.
  Consequence for schema verification: an empty `db.relationshipTypes()` result for a type that
  *should* exist is not proof the schema/indexes are missing — it only proves no edge of that type
  has been written yet; check `db.labels()`/`db.indexes()` for the schema side and
  `db.relationshipTypes()` (or a direct count) for the data side, separately.
- **Relationship-property indexes and `RELATIONSHIP`-scoped `UNIQUE` constraints are fully
  supported** (verified 2026-08-22, module `41811` — previously flagged unverified by
  `falkor-chat/docs/plans/document-ingestion.md`, now settled). `CREATE INDEX FOR ()-[r:TYPE]-()
  ON (r.prop)` and `GRAPH.CONSTRAINT CREATE <key> UNIQUE RELATIONSHIP <TYPE> PROPERTIES <n>
  <prop...>` both work exactly like their `NODE` counterparts (same index-before-constraint
  ordering, same async `PENDING`→`OPERATIONAL` constraint lifecycle); `db.indexes()` reports
  `entitytype: RELATIONSHIP` for the paired index. A query filtering on the indexed relationship
  property profiles as `Edge By Index Scan`, confirmed for a pattern-property match
  (`{prop:$x}`), for a `SET` immediately after one, and for a `WHERE`-filtered global scan —
  directed or undirected pattern, doesn't matter. This makes a property-bearing relationship a
  genuine, index-anchored alternative to a reified "decision record" node for any schema that
  needs a hot-filterable status property between two existing nodes (see
  `falkor-chat/docs/plans/document-ingestion-graph.md` §1 for the full worked comparison,
  including a live RAM measurement showing the two shapes cost about the same — this only settles
  *capability*, not which shape wins on RAM).
- **Vector dimension is enforced at query time and index-membership time, but NOT at write
  time** (verified 2026-07-08, module 999999). A wrong-dimension `SET n.embedding =
  vecf32([...])` is **silently accepted** (`Properties set: 1`, no error) — but the node then
  **drops out of the ANN index** and never appears in `db.idx.vector.queryNodes` results.
  Querying the index with a mismatched query vector *does* error
  (`Vector dimension mismatch, expected N but got M`). There is **no `vec.dimension()`
  function** on this build to check a stored vector's length. Consequence: validate embedding
  length client-side before writing — a buggy worker sending wrong-size vectors produces
  permanently unretrievable nodes with no error surfaced.
  **Re-confirmed 2026-08-10 on the pinned build (module `41811`)**: a 3-dim `vecf32` write
  against a dim-4 index still reports `Properties set: 2`, the node is `MATCH`-able, and it
  never appears in `db.idx.vector.queryNodes` results.
- **`db.indexes()` DOES expose a vector index's dimension** — **corrects the earlier
  "does not expose it" claim**, which was recorded against the edge build (module 999999) and
  is **false on the pinned `v4.18.11` / module `41811`** (verified 2026-08-10). The `options`
  column is a map keyed by property name; a `VECTOR`-typed property's entry carries
  `{dimension, similarityFunction, M, efConstruction, efRuntime}`. Dynamic map-key access and
  a post-`YIELD` `WHERE` both work, and the whole thing runs under `GRAPH.RO_QUERY`
  (replica-routable, zero write risk):
  ```
  GRAPH.RO_QUERY <graph> "CYPHER lbl='Message' prop='embedding'
    CALL db.indexes() YIELD label, types, options
    WHERE label = $lbl AND types[$prop] = ['VECTOR']
    RETURN options[$prop].dimension AS dim"
  ```
  Behaviour at the edges (all verified 2026-08-10): the dimension is reported **before any
  vector is written** (it is index metadata, not data); a label with only `RANGE` indexes
  returns a row with `dim = NULL`; an unknown label returns **zero rows**; and a graph key
  that does not exist yet errors `ERR Invalid graph operation on empty key` rather than
  returning zero rows. This is now the cheap, reliable way to prove an index's dimension —
  prefer it over the mismatched-query-vector probe.
- **`CREATE VECTOR INDEX` on an already-indexed property is rejected, never re-applied**
  (verified 2026-08-10, module `41811`). Re-creating `(:Message) ON (n.embedding)` with a
  *different* `dimension` (or a different `similarityFunction`) returns
  `Attribute 'embedding' is already indexed` and the index **keeps its original options**.
  Operationally sharp: a bootstrap script that re-runs with a changed dimension env var does
  **not** change the dimension, and because `redis-cli` exits 0 on Redis-level errors, a
  `set -e` script sails past it. The only way to change a vector index's dimension is to drop
  and re-create the index.
- **ANN kNN returns *up to* `k`, not exactly `k`** — on a small/near-empty HNSW index,
  `db.idx.vector.queryNodes(…, k, …)` may return fewer than `k` (approximate recall of distant/
  orthogonal candidates). Near neighbors are returned and correctly ordered; don't treat
  "returns exactly k" as an invariant.
- **A value over 4096 bytes written into a `UNIQUE`-constrained property crashes the ENTIRE
  instance — SIGSEGV, not a query error** (verified 2026-08-22 on v4.18.11 / module `41811`, in an
  isolated throwaway container, never against a shared instance). `CREATE`/`MERGE` writing a value
  >4096 bytes into any property backed by `GRAPH.CONSTRAINT CREATE ... UNIQUE` segfaults the whole
  `redis-server` process (signal 11, si_code 1 `SEGV_MAPERR`, faulting address `(nil)` — a
  null-pointer dereference, confirmed `OOMKilled: false`, so not resource exhaustion). Stack:
  `EnforceUniqueEntity` ← `Schema_EnforceConstraints` ← `CommitNewEntities` — it fires specifically
  while the engine checks the `UNIQUE` constraint at commit time, not while indexing or storing.
  **Exact boundary: 4096 bytes safe, 4097 crashes** — binary-searched and reproduced deterministically
  at 4097/4104/4112/4128/4200/4500/5000/6000/7000/8000, safe at 100/1000/4000/4096. **Constraint-
  specific, not index-specific**: a RANGE-indexed property with no constraint is safe at least to
  1MB tested; an unindexed property is safe at least to 8000 bytes. **Per-property for a composite
  constraint** (`PROPERTIES 2 a b`) — two columns each under 4096 (e.g. 3000+3000) are safe; one
  column over 4096 crashes regardless of the other's size. **Independent of write-clause shape** —
  bare `CREATE`, bare `MERGE`, and `MERGE` on a computed string-concatenation expression inside an
  `UNWIND` (falkor-chat's exact `_PUBLISH_CYPHER` shape) all reproduce identically once the final
  value exceeds the threshold. The crashed container does **not** self-heal — `redis-server` dies
  (exit 139); a non-`--rm` container comes back empty (no persisted data) on `docker start`, a
  `--rm` container vanishes from `docker ps -a` entirely. Consequence: any caller-supplied string
  that reaches a `UNIQUE`-constrained property needs an app-side length guard well under 4096
  bytes — a pydantic/REST-only bound is not enough, since any in-process caller (test, script, a
  future MCP tool) bypasses it entirely; the guard belongs at the service/repository boundary too.
  Full write-up incl. crash log: `falkor-chat/docs/reviews/unique-constraint-oversized-value-
  crash-rca.md` (K-049).

## Concurrency & atomicity

- **FalkorDB/Redis serializes write execution per graph — only one write query runs at a time,
  queued in arrival order — and every write query is itself atomic (all-or-nothing; readers never
  see a partial write)** (verified 2026-08-25 against docs.falkordb.com/design/concurrency, and
  live in falkor-chat's shipped `create_entity_with_auto_match`,
  `falkor-chat/server/falkorchat/repository.py:1259`, K-050 M5). Consequence: a check-then-act
  sequence (does a candidate already exist? if not, create + link it) is race-free **for free**
  when folded into ONE `GRAPH.QUERY` — no external lock, queue, or CAS-retry loop needed; two
  concurrent callers can never observe or act on the same intermediate state. Read queries do
  **not** serialize against each other (they run in parallel for throughput), only writes do. This
  only holds **within one `GRAPH.QUERY` call against one graph key** — splitting a check-then-act
  across two round trips (an app-level read, then a separate write) reopens the exact race the
  single-query fold closes.

## Cypher dialect & query behavior

- **Overwriting an existing scalar property reports a *removal* it did not perform** (verified
  2026-09-08, module `41811`, live `cpg_falkorchat`). A targeted
  `MATCH (b:CpgBuildInfo) SET b.MARKER_WRITTEN_AT = '<new>'` over an already-present value replies
  `Properties set: 1` **and** `Properties removed: 1`, while `size(keys(b))` is unchanged (10
  before, 10 after) and every other property is byte-identical. Nothing was removed; the counter
  appears to charge the displaced old value. **Never read `Properties removed` as evidence that a
  key disappeared** — `keys(n)` before/after is the only discriminator. Independently observed
  elsewhere in this lab at wider divergences (13 reported against 5 real removals, 4 against none,
  `skills/cpg-analysis/references/freshness.md`), so the mechanism is not established; only the
  unreliability is.
- **A property a node does not carry projects as `null` — never an error — so a `UNIQUE` constraint
  is no guarantee the key is present in a projection** (verified 2026-09-08, module `41811`,
  read-only against the live `kaizen_team` and `reference` graphs; no probe graph created).
  `MATCH (a:Agent {agentId:'cobb'}) RETURN a.noSuchProperty` returns one row holding `null`,
  `a.noSuchProperty IS NULL` is `true`, and `keys(a)` is `['agentId']` — nothing raises at parse,
  plan or run time. This is the projection half of the constraint entry under *Indexing,
  constraints & DDL*: `CALL db.constraints()` on `reference` returns
  `UNIQUE | Product | ['productId'] | NODE | OPERATIONAL` and **no `MANDATORY` row at all** (four
  constraints, all `UNIQUE`), so a `Product` created without `productId` is accepted and every
  projection of it reads `null`. **A repository projection widened onto a `UNIQUE`-constrained key
  still needs its own null check** before a caller treats that key as an identifier — the
  constraint enforces distinctness among the rows that have a value and says nothing about the
  rest. Observed downstream by `analyst` 2026-09-03: `filter_products` → `list_catalog` carried
  `productId: null` out to the caller with no error (that transient row is gone — all 15 current
  `Product` nodes carry the key).
- **No string-repetition operator** — `CREATE (:T {code: 'x' * 400})` fails with
  `Type mismatch: expected Integer, Float, or Null but was String` (verified v4.18.11,
  falkordb-py 1.6.2). Build wide test-fixture strings in the client and pass them as a
  parameter (`{code: $c}`, `params={"c": "x"*400}`) instead of trying to construct them
  in-query. Second-order trap: the **failed** `GRAPH.QUERY` still materializes the graph
  key, leaving a junk empty graph behind that has to be deleted by hand.
- **A single-quoted string literal rejects the SQL-style doubled-single-quote escape (`''`) for
  an embedded apostrophe — it's a parse error at the second quote (`"Invalid input mismatched
  quote"`), not a silent misparse** (verified 2026-09-10, module `41811`, via the `cypher` MCP
  tool against `kaizen_team`). Escape it with a backslash (`'it\'s a test'`) or just use a
  double-quoted literal instead (`"it's a test"`, no escaping needed) — both confirmed to return
  the correct string. Matters most for a free-text field value (a `KaizenEntry.fact`, a report
  string) built with a contraction/possessive and spliced in as an inline literal: prefer
  switching that one field to double quotes over trying to escape the apostrophe inside single
  quotes.
- **`redis-cli GRAPH.QUERY`'s `CYPHER` preamble needs Cypher *literals*, not bare
  `k=v` pairs** — `CYPHER key=$key ...` bound via `redis-cli`'s trailing `k=v` args
  (`... key=triage`) fails `Failed to parse query parameter 'key' value`; those trailing
  args are not a binding channel at all, and an unquoted bare word in the preamble parses
  as an expression. Write `CYPHER key='triage' version='v1' MATCH …` — quoted literals in
  the preamble — for any shell-driven maintenance query. The Python client's `params=`
  dict is unaffected; this is a `redis-cli`-only trap.
- **A non-aggregated key from an `OPTIONAL MATCH` fan-out is a real grouping key beside a
  `collect(DISTINCT …)`, not a constant you can assume away** — `RETURN d.name, start.key
  AS startKey, collect(DISTINCT {...}) AS steps` returns **one row per distinct `start.key`**
  value reachable by the fan-out, each carrying the *full* aggregate, not one row with the
  aggregate collapsed as the "constant key beside an aggregate" idiom usually assumes
  (verified v4.18.11: 2 `START` edges on one node → 2 rows, `steps` identical on both). A
  consumer doing `result_set[0]` doesn't fail on this — it silently returns an arbitrary
  row. The invariant only holds when the schema guarantees the grouping key has exactly one
  value per match (falkor-chat's `WorkflowDef`→`START` relies on this and is why K-034 exists
  — a `MERGE` with a changed endpoint creates a **second** edge rather than moving the first).
  Treat "a scalar returned beside `collect()` is constant across the fan-out" as a premise to
  verify against the schema's actual cardinality guarantee, not an engine property.
- **`exists()` over a relationship pattern does not scope to an already-bound node — it evaluates
  existentially against the whole graph** (verified 2026-09-10, module `41811`, live
  `kaizen_team`). `MATCH (k:KaizenEntry {entryId:'<id>'}) RETURN
  exists((:Agent)-[:PRODUCED]->(k)) AS hasProduced` returned `true` for a node with **zero**
  `PRODUCED` edges, immediately contradicted in the same session by
  `OPTIONAL MATCH (a:Agent)-[p:PRODUCED]->(k) RETURN a.agentId, count(p)` → `null, 0` on the
  identical node. Re-derivation found the bug is **not** an artifact of leaving the source node
  unbound: binding it directly in the pattern —
  `MATCH (k:KaizenEntry {...}) RETURN exists((k)<-[:PRODUCED]-(:Agent))` — reproduces the same
  false `true`. Treat `exists()` over any relationship pattern as unreliable for "does this
  specific node have an edge" and use `OPTIONAL MATCH (...)-[r:REL]->(k) RETURN count(r)` instead
  — the form `skills/agent-maintenance/SKILL.md` §5's clearing-query step already prescribes, for
  exactly this reason.
- **Cross-graph edges silently no-op** — no error, `MATCH` just returns 0 rows.
  There is nothing to catch.
- **`(:A | :B)` union-label syntax** in a pattern is unverified on this build —
  use `coalesce()` over two label-specific `OPTIONAL MATCH`es instead.
- **`length(path)` in `ORDER BY`** is not supported — order by a property instead.
- **`STARTS WITH` on an indexed string property does NOT plan as an index range scan**
  (verified 2026-07-09, module 999999). A prefix predicate like `WHERE n.key STARTS WITH
  $prefix` on an indexed `n.key` profiles as `Node By Label Scan` + `Filter`, not an index
  scan. Consequence: don't use a synthetic-composite-key prefix (`"{a}:{b}:"`) as an
  index-anchored scoping predicate — model an explicit edge (e.g. a `HAS_STEP` containment
  edge) and traverse from an indexed anchor instead.
- **`STARTS WITH` with a concatenated prefix needs explicit parentheses on the RHS**
  (verified 2026-07-09). `x STARTS WITH $a + ':' + $b` errors *"Type mismatch: expected
  Boolean but was String"* — `STARTS WITH` binds tighter than `+`. Write
  `x STARTS WITH ($a + ':' + $b)`.
- **The regex-match operator `=~` is not supported at all** — a hard error,
  *"FalkorDB does not currently support =~"*, with no degraded or partial form to fall back on
  (verified 2026-09-08, module `41811`). For case-insensitive substring matching use
  `toLower(prop) CONTAINS 'term'`, chained with `OR` across several properties, or
  `WHERE any(k IN keys(n) WHERE toLower(toString(n[k])) CONTAINS 'term')` to sweep a whole node.
- **`algo.*` procedures confirmed:** `BFS`, `WCC`, `pageRank`, `SPpaths`,
  `SSpaths`, `MSF`, `betweenness`, `labelPropagation`.
- **Empty `UNWIND` collapses the row stream.** `WITH x UNWIND [] AS y …` drops
  every row that reached it, even ones written earlier in the same query. Guard
  with `UNWIND (CASE WHEN $list = [] THEN [null] ELSE $list END) AS item` + a
  `FOREACH` that never filters. **This can silently drop an unrelated required
  write downstream in the same query, not just the guarded list's own effects**
  — an empty optional-list parameter (e.g. no mentions/tags on this write) can
  collapse a mandatory `CREATE` that has nothing to do with that list. Always
  test the zero-length-list case end-to-end (does the *required* write still
  happen?), not just that the optional edges are correctly absent.
- **`FOREACH (x IN CASE WHEN … THEN [1] ELSE [] END | CREATE …)`** is the working
  idiom for conditional writes without dropping rows. Nested `FOREACH`, and
  `DELETE` inside a `FOREACH`, both work. **A `FOREACH` body may hold SEVERAL `CREATE`
  clauses whose variables bind across them, and a node variable bound OUTSIDE the `FOREACH`
  is a legal `CREATE` relationship endpoint inside it** — so a whole multi-node subgraph is
  writable under one guard (verified 2026-09-07, module `41811`): one guarded `FOREACH`
  containing `CREATE (u:P{…}) CREATE (c:P{…}) CREATE (t:P{…}) CREATE (c)-[:HAS_THREAD]->(t)
  CREATE (u)-[:MEMBER_OF]->(c) CREATE (agent)-[:MEMBER_OF]->(c)`, with `agent` bound by an
  outer `OPTIONAL MATCH`, created 3 nodes + 3 relationships; the guarded-false path wrote
  nothing. This is the **positive counterpart** to the map-projection restriction below — the
  endpoint must be a *bound node variable*, and an outer-scope binding qualifies, while a
  list subscript does not (`CREATE (ms[k])-[:NEXT]->(ms[k+1])` errors `Invalid input '['`).
  Use it for atomic, idempotent provisioning writes: one statement, one guard, whole subgraph.
  **Two `FOREACH` clauses also chain back-to-back after a single `WITH` — no `WITH` between them —
  and each one sees property values written earlier in the same query, by a preceding `SET` *or*
  by a preceding `FOREACH`** (verified 2026-09-08, module `41811`, disposable graph).
  `MATCH (d:D {id:$id}) SET d.remaining = d.remaining - 1 WITH d
  FOREACH (_ IN CASE WHEN d.remaining = 0 THEN [1] ELSE [] END | SET d.status = 'ready')
  FOREACH (_ IN CASE WHEN d.remaining > 0 THEN [1] ELSE [] END | SET d.status = 'running')`
  reads the *decremented* value in both guards: `2 -> 1` returned `running`, `1 -> 0` returned
  `ready`. Cross-`FOREACH` visibility measured separately — `FOREACH (_ IN [1] | SET d.flag = 7)`
  followed by `FOREACH (_ IN CASE WHEN d.flag = 7 THEN [1] ELSE [] END | ...)` fires. So a
  decrement-then-branch counter update is **one** atomic statement, not a read plus a write.
- **`exists((n)-[:REL]->())` in a pattern returns `true` even when the edge is
  absent** (broken on this build); `count{ … }` subquery syntax is unsupported.
  For existence checks use `OPTIONAL MATCH (n)-[:REL]->(x) RETURN x IS NOT NULL`
  instead.
- **`RETURN DISTINCT <col>` followed by `ORDER BY <expr not in the RETURN list>`
  does NOT error on this build — it silently accepts an ill-defined ordering**
  (verified 2026-08-30, module 41811). Unlike SQL engines that reject an
  unprojected `ORDER BY` column after `DISTINCT` (ambiguous per output row),
  FalkorDB plans it as `Sort` *after* `Distinct` (`GRAPH.EXPLAIN` shows
  `Limit → Sort → Distinct → Project`), carrying the `ORDER BY` expression as a
  hidden column through `Project`. `Distinct` dedupes on the **declared RETURN
  columns only** and keeps whichever row happened to survive dedup — empirically,
  the first-encountered row per key in scan/creation order — so the sort key used
  for a collapsed group is arbitrary and order-dependent, not an error, an
  aggregate, or a stable "first/last" guarantee. Reproduced with two `:Entity`
  nodes sharing `entityId:'A'` but different `name`s: `RETURN DISTINCT e.entityId
  ORDER BY e.name` picked up whichever node's `name` was created/scanned first,
  and the overall row order changed accordingly when creation order was swapped —
  same query, same data, different `name` values feeding the sort depending only
  on insertion order. Treat any `RETURN DISTINCT` + `ORDER BY` where the order
  expression isn't one of the returned columns as a live correctness bug, not a
  style nit: either add the order expression to the `RETURN` list (so it's
  deduped together with the key, making the "arbitrary representative" explicit
  and query-visible) or drop the `ORDER BY` — never assume the engine will reject
  or normalize the ambiguous case for you.
- **The safe shape for "distinct projection, ordered by a column NOT in the projection"
  (the superlative/min-max-by-different-column pattern, e.g. "which product is cheapest") is
  `WITH DISTINCT <returns-cols>, <order_by-col> ORDER BY <order_by-col> LIMIT n RETURN
  <returns-cols>` — dedup on the FULL tuple (every needed column, not just the returned ones),
  in the same clause the `ORDER BY`/`LIMIT` runs in** (verified 2026-08-30, module `41811`,
  disposable `probe_u29f` graph — follow-up to the entry above, which this refines rather than
  contradicts). The naive fix ("reject any query where `ORDER BY` isn't in `RETURN`") is
  over-broad: it also outlaws this shape, which has no other expression in this dialect (single
  `MATCH`, no aggregate returns "the associated column of the row achieving the min/max"). Root
  cause confirmed via `GRAPH.EXPLAIN` on both the buggy and fixed forms: the buggy
  `RETURN DISTINCT p.name ORDER BY p.price` plans `Project → Distinct → Sort → Limit` — `Distinct`
  collapses to the returned column alone *before* `Sort` runs, so a collapsed group's surviving
  `price` (the sort key) is arbitrary/order-dependent. Direct repro: three `:Product` rows
  `Gadget/50`, `Widget/100`, `Widget/3` (true min is `Widget` at 3) — `RETURN DISTINCT p.name
  ORDER BY p.price ASC LIMIT 1` returned **`Gadget` (wrong answer, not just non-deterministic)**
  in one creation order and **`Widget` (right answer)** in the reverse order, same data, same
  query. The fixed shape — `WITH DISTINCT p.name AS c0, p.price AS c1 ORDER BY c1 ASC LIMIT 1
  RETURN c0` — plans `Project → Distinct → Sort → Limit → Project` (`Distinct` now dedupes on the
  `(c0,c1)` tuple, so it never discards the pairing `Sort` needs) and returned the correct
  `Widget` in **both** creation orders. It also does the right thing on genuine duplicates: two
  physically-identical `(name, price)` rows collapse to one (verified with an exact-duplicate
  `Widget/3` pair alongside a same-named-but-different-price `Widget/100` row — the true dup
  collapsed, the distinct-price row did not, and the correct minimum still won). Placing the
  `ORDER BY ... LIMIT` on the same `WITH` that carries the tuple-`DISTINCT` (rather than on a
  later `RETURN`) also empirically tested safe (`WITH DISTINCT c0, c1 RETURN c0 ORDER BY c1 LIMIT
  1` gave identical plans/results) — the load-bearing fix is the tuple-`DISTINCT`, not which
  clause keyword `ORDER BY` attaches to; putting it on the `WITH` is simply the more direct
  compile target and keeps `DISTINCT`/`ORDER BY`/`LIMIT` visually and structurally together.
  **One residual, expected, non-pathological caveat**: a genuine tie in the sort column between
  two *different* rows (e.g. two distinct products both priced at the true minimum) still resolves
  to an insertion-order-dependent winner under `LIMIT 1` — verified, flips between the two tied
  names across creation orders. This is ordinary "no tiebreak column specified" `LIMIT` behavior
  (true of every SQL/Cypher engine, not a FalkorDB defect) and is categorically different from the
  original bug: the original bug picked an arbitrary *sort key* for one logical group and could
  give a **wrong** answer even with no real tie in the data; this residual case only arbitrates a
  **real** tie. Not attempting to fix this residual case here — it's a data-shape ambiguity, not
  an engine defect, and out of scope for the compiler fix this entry informs.
- **`labels(coalesce(a, b))[0]`** subscripting works, for reading the resolved
  label off a `coalesce()` of two optionally-matched nodes.
- **A map-projection cannot be a `CREATE` relationship endpoint** (verified
  2026-07-08, module 999999). `FOREACH (rec IN recs | CREATE (m)-[:R]->(rec.node))`
  where `rec` is a map with a `node` field **errors** (`Invalid input '.': expected
  a label, '{', a parameter or ')'`). The endpoint must be a **bound node
  variable**. To attach per-edge properties while iterating: collect the endpoints
  as **nodes** (`collect(DISTINCT s)`) and pull props from **map parameters keyed by
  the node's own property** — `CREATE (m)-[:R {score: $scoreBy[s.id], rank:
  $rankBy[s.id]}]->(s)`. Dynamic map-parameter indexing by a node property
  (`$scoreBy[s.id]`) works, including inside a `FOREACH`.
- **Two sequential guarded `UNWIND`s** in one query (each followed by its own
  `collect(...)` back to one row) do **not** row-multiply — the first `collect`
  collapses before the second `UNWIND` expands. Pattern: `UNWIND (CASE …) AS a …
  collect(…) AS as  UNWIND (CASE …) AS b … collect(…) AS bs`. Verified for two
  distinct edge blocks (e.g. `MENTIONS_MEMBER` + `EMITTED`) inside one guarded write.
- **Sequential `UNWIND` blocks *without* an intervening collapse row-multiply the
  final `RETURN`** (verified 2026-07-09). `WITH d UNWIND $steps … WITH d UNWIND
  $transitions … RETURN d.key` emits `steps × transitions` duplicate rows. Collapse
  each block back to one row with an aggregation (`WITH d, count(st) AS stepCount`)
  so the query returns a single clean status row. The write itself is unaffected —
  this is a result-cardinality issue, not a correctness one.
- **The same collapse rule applies to a preceding `OPTIONAL MATCH`: a still-open
  `OPTIONAL MATCH` stream gets multiplied by any later `UNWIND`, with no error and identical
  results — only the cost changes** (verified 2026-09-07, module `41811`, disposable graph).
  `MATCH (u:User) WITH collect(DISTINCT u) AS users OPTIONAL MATCH (rc:ReadCursor) WHERE … UNWIND
  users AS uu …` profiles as `Unwind | Records produced: 5100` (100 cursor rows × 51 users);
  giving the `OPTIONAL MATCH` its own `WITH users, collect(DISTINCT rc) AS cursors` **before**
  the `UNWIND` gives `Unwind | Records produced: 51` and the same answer — a 4× drop in the
  downstream `Aggregate` at this trivial scale, measured at **3× wall-clock** (≈690 ms → ≈240 ms)
  on a 50-participant / 2000-message workspace-reset write. Cheap to miss precisely because both
  shapes are *correct*: it shows up only in timing and in `PROFILE`'s records-produced column.
- **Cannot combine an aggregation with a reference to a prior variable in the same
  `WITH`** when building an accumulator. `WITH acc + [x IN collect(DISTINCT c.NAME)
  WHERE NOT x IN acc] AS acc2` fails at runtime with `_AR_EXP_UpdateEntityIdx:
  Unable to locate a value with alias <acc>`. Split into two steps: `WITH acc,
  collect(DISTINCT c.NAME) AS lvl` then `WITH acc + [x IN lvl WHERE NOT x IN acc]
  AS acc2`. Surfaced building a multi-level name-based reachability closure for a
  test-gap query. (Verified 2026-07-19 on v4.18.11 / `cpg_falkorchat`.)

- **An aggregation buried inside a concatenated `RETURN` expression, next to non-aggregated
  terms, silently returns ZERO ROWS** (verified 2026-07-20 on v4.18.11).
  `... OPTIONAL MATCH (r)-[:REL]->(m) RETURN "def="+d.key+" n="+toString(count(m)) AS s`
  returns an empty result set — **no error**, just no rows — because the implicit grouping key
  is the whole concatenated expression, which contains the aggregate. Put the aggregate
  **first** and wrap the non-aggregated terms in `collect(x)[0]`:
  `RETURN "n="+toString(count(m))+" def="+collect(d.key)[0] AS s`. Silent-zero-rows is the
  dangerous shape here — a test asserting on the string just fails with an empty `got:`.

- **`sum(CASE WHEN … THEN 1 ELSE 0 END)` over a zero-row aggregation returns a float `0.0`, never
  Cypher `NULL` — and returns `float`, not `int`, over non-empty input too, via `falkordb-py`**
  (verified 2026-07-31 on v4.18.11 via both `redis-cli GRAPH.QUERY` and `falkordb-py`). A query
  shaped `... WITH r, count(m) AS producedCount RETURN count(r) AS sampleSize, sum(CASE WHEN
  producedCount > 0 THEN 1 ELSE 0 END) AS postedCount` returns `sampleSize=1 <int>,
  postedCount=0.0 <float>` over zero matching rows, and `sampleSize=2 <int>, postedCount=1.0
  <float>` over non-empty input — `count()` stays a clean Python `int` in both cases, only
  `sum()` comes back `float`. Consequence: don't `None`-coalesce a `sum(CASE...)` result
  expecting `NULL` on empty input (it's already a defined `0.0`), and expect a `float`-vs-`int`
  JSON-serialization mismatch (`"postedCount": 1.0` instead of `1`) wherever this shape feeds a
  response model.

- **A global `collect()` over a zero-row `MATCH` still returns exactly ONE row, carrying empty
  lists — never an empty result set** (verified 2026-09-07, module `41811`).
  `MATCH (u:User) WHERE u.userId = 'nobody' WITH collect(DISTINCT u) AS users RETURN size(users),
  users` returns one row with `0` / `[]`. Consequence: an aggregate "status" query written this
  way reports an all-zeros row rather than "no rows", so a caller that treats an empty result set
  as the not-found signal will never see it — check the counts, not `len(result_set)`. Same
  family as the `sum(CASE …)` entry above (a defined zero, not `NULL`/absent).

- **`collect()` of a MAP LITERAL over a zero-row `OPTIONAL MATCH` yields a list holding ONE
  all-null map, not `[]`** (verified 2026-09-07, module `41811`). Collecting the unmatched variable
  itself, or a property of it, correctly yields `[]` — nulls are dropped — but a map literal is a
  non-null value whose *fields* happen to be null, so it survives the collect. Measured on one node
  with no matching edge: `collect(l)` → `[]`, `collect(l.qty)` → `[]`, `collect({q: l.qty})` →
  `[{q: null}]` (size **1**). Consequence: `size(lines) = 0` is not the emptiness test for a
  `collect({...})` fan-out — filter on a required non-null field first
  (`size([x IN lines WHERE x.productId IS NOT NULL])`). Same family as the `sum(CASE …)` entry: a
  defined placeholder, not absence. Surfaced on `falkor-chat`'s `get_order` for a zero-line order.

- **A node pulled OUT of a collected list by index (`head(...)`, `[0]`) stays usable as a value
  and as a bare single-node pattern, but raises the moment it is an endpoint of a RELATIONSHIP
  pattern** (verified 2026-09-07, module `41811`, read-only against a live graph).
  `MATCH (p:Product) WITH head(collect(DISTINCT p)) AS x …` — `RETURN x.name` / `labels(x)` work,
  and even `MATCH (x) WHERE x.name IS NOT NULL RETURN x.name` re-binds it correctly (one row, the
  same node, not a re-scan). But `MATCH (x)-[r]-(y)` or `OPTIONAL MATCH (x)-[:REL]->(c)` fails the
  whole query with `ResponseError: encountered unexpected type in Record; expected Node` — raised
  whether or not any such edge exists. **The trigger is index extraction from the list, not the
  `head()` function**: `collect(DISTINCT p)[0]` fails identically. `UNWIND` is the working route —
  `WITH collect(DISTINCT p) AS xs UNWIND xs AS x WITH x LIMIT 1 OPTIONAL MATCH (x)-[:REL]->(c)`
  runs clean. Consequence: any "dedupe to one node, then traverse from it" step must dedupe via
  `UNWIND` (or re-`MATCH` the node by its id property), never by indexing the collected list — and
  the failure is a hard error, not a wrong result, so an ablation written this way invalidates
  itself rather than changing behavior.

- **Dynamic property access by variable key works: `n[k]` evaluates for a `k` bound from
  `keys(n)`** (verified 2026-09-07, module `41811`, positive and negative control).
  `MATCH (n) WHERE any(k IN keys(n) WHERE n[k] = 'Wireless Mouse Pro') RETURN labels(n), count(n)`
  returned the one real match; the same query with a value present on no node returned `0`, so it
  is not vacuously true either way. Consequence: a whole-graph "this value is stored nowhere, under
  any property, on any label" assertion is one read-only query — no property-name enumeration, no
  per-label sweep. Pair it with the positive control in the same test, or a regression that breaks
  `keys()`/`n[k]` turns the assertion into a silent pass.

- **`count(*)` under-counts parallel edges between the same node pair — bind the relationship
  variable and use `count(r)` instead** (verified 2026-08-25, module `41811`, disposable graph).
  Two identical `(a)-[:REL]->(b)` edges between the same two nodes: `MATCH (a)-[:REL]->(b) RETURN
  count(*)` returns **1**; `MATCH (a)-[r:REL]->(b) RETURN count(r)` correctly returns **2**. Any
  query counting relationships (audit/dedup checks, "how many edges of this type" reporting) must
  bind and count the relationship variable, never `count(*)`, whenever parallel edges between the
  same pair are possible in the schema (e.g. a schema that deliberately never deduplicates
  `RELATES_TO`/`ABOUT`-style edges — falkor-chat K-050 M5).

- **An undirected relationship pattern combined with a predicate on an INDEXED relationship
  property silently degrades to directed — first-declared node treated as the edge's source —
  and returns wrong (too-few) results, no error** (verified 2026-08-25, module `41811`, disposable
  graph; corrects/merges two raw `kaizen_team` entries that reported the symptom without isolating
  the actual trigger — re-derived from scratch, not assumed from their citations). Setup: edge
  stored `(e2)-[:SAME_AS {status:'confirmed'}]->(e1)`, plus `CREATE INDEX FOR ()-[r:SAME_AS]-() ON
  (r.status)`. `MATCH (:Entity{entityId:'e1'})-[r:SAME_AS{status:'confirmed'}]-(:Entity
  {entityId:'e2'}) RETURN count(r)` returns **0**; swapping which node is declared first (`e2`
  before `e1`, everything else identical) returns **1**. `PROFILE` shows why: the property
  predicate folds into an `Edge By Index Scan | [r:SAME_AS]` that scans only the direction implied
  by pattern order, not both. **The index is the trigger, not the property predicate alone** — the
  identical query, same schema, with the index dropped, correctly returns **1** regardless of
  declared node order. Reproduces identically for an inline map filter
  (`{status:'confirmed'}` in the pattern) and for an equivalent separate `WHERE r.status =
  'confirmed'` clause after two prior `MATCH`es bind the same nodes — both forms fold into the
  same directional index scan. A plain undirected pattern with NO relationship-property predicate
  is unaffected even with the index present (correctly symmetric both ways). Directed patterns are
  unaffected in every variant (they're supposed to be direction-sensitive). **Consequence:** any
  undirected-pattern query filtering on an indexed relationship property (a
  `SAME_AS`/`RELATES_TO`-style edge probed from either endpoint) needs two `OPTIONAL MATCH`es, one
  per direction, `coalesce`d — don't trust "the pattern has no arrow" to mean direction-safe once
  that relationship property is indexed. Same index-folding mechanism as the "guarded-CAS WHERE"
  and "two independent WHERE predicates fold into one Index Scan" entries below (Query tuning),
  but here the fold changes the **result**, not just the plan shape.

- **SQL-style `--` is NOT a comment on this build — Cypher comments are `//`** (verified
  2026-09-07, module `41811`). `MATCH (u:User) -- G1` + newline + `RETURN count(u)` errors
  `Invalid input 'G': expected '>' or '('` — the parser reads `--` as the start of an operator,
  not a line comment; the `//` form returns normally. This fails **loudly**, not silently, but
  the error message points at the token after the dashes and reads nothing like "bad comment",
  so it costs a debugging cycle if you don't know the rule.

- **Comparing a non-string value against a string yields `NULL`, so the row is silently dropped
  by a `WHERE`** (verified 2026-09-07, module `41811`). `RETURN 42 > ''`, `42.5 > ''` and
  `true > ''` all return `NULL`; `'abc' > '' → true`, `'' > '' → false`. There is no cross-type
  coercion and no error. Consequence: any predicate that compares a property to a string literal
  is also an implicit *type filter*, and in a destructive query that is a silent under-delete.
  See the `IS NOT NULL` index-anchor entry under *Query tuning* for the specific idiom this
  invalidates.

- **A bare `MATCH` immediately after an `OPTIONAL MATCH` is rejected — insert a `WITH`**
  (verified 2026-09-07, module `41811`). `OPTIONAL MATCH (a:X) MATCH (b:Y) RETURN …` raises
  *"A WITH clause is required to introduce a MATCH clause after an OPTIONAL MATCH."* A
  pass-through `WITH` satisfies it, as does reordering so the required `MATCH` leads. Same
  `WITH`-as-clause-boundary rule the update-clause chaining entry below turns on — the parser
  demands the bridge, and the bridge is always trivially suppliable.

- **FalkorDB refuses a multi-statement query outright — one `GRAPH.QUERY`, one statement.**
  `RETURN 1 AS a; RETURN 2 AS b` comes back as the error *"query with more than one statement is
  not supported"* (re-verified 2026-09-09 through the `cypher` MCP tool against `kaizen_team`).
  Two consequences, pulling opposite ways. For a **design**: there is no clear-then-write, no
  delete-and-recreate, no two-phase anything in a single call — either express it as one statement,
  or accept two round trips with a window in which the node is absent. That is why a singleton
  marker/stamp is a `MERGE … SET`, not a `DELETE` followed by a `CREATE`: the atomic version of
  the latter is unconstructible here. For **injection**: a `;` buys an attacker nothing, so the
  identifier-splice hazard in the next bullet is entirely about what fits inside *one* statement —
  which, as that bullet shows, is more than the grammar first suggests.

- **A single Cypher statement can chain a read clause, `DETACH DELETE`, and another read clause
  — the grammar is not a barrier to an identifier-splice mutation.** An update clause cannot be
  followed *directly* by a read clause (`MATCH (v:Product) DETACH DELETE v WHERE true RETURN 1`
  fails to parse: `Invalid input 'H': expected WITH`), but the bridge the parser demands is
  trivially suppliable: insert `WITH 1 AS x` and `MATCH … DETACH DELETE … WITH … MATCH … RETURN …`
  parses cleanly as **one statement, no semicolon, no injected `;` needed**. `GRAPH.RO_QUERY`
  then refuses it — but at the engine's read-only check *on the parsed plan*, not as a syntax
  error and not by scanning the query text. **Consequence for any query-builder DSL that splices
  an identifier** (a label, variable, property or ordering token) into a read-only template: a
  full mutation payload delivered through that field is a genuinely constructible statement on
  this engine, not something Cypher grammar happens to stop. The splice-point allowlist is
  therefore load-bearing on its own, and the read-only command is the only engine-level backstop
  behind it. (Verified 2026-09-06 on v4.18.11, module `41811`, live against an existing graph;
  same parse-then-reject ordering the `GRAPH.EXPLAIN` entry under *Ops* relies on to syntax-check
  a write.)

## Query tuning

- **`WHERE prop IS NOT NULL` cannot anchor a range index — and the obvious `prop > ''` workaround
  is unsound and buys nothing** (verified 2026-09-07, module `41811`, 51-node disposable graph;
  merges and **corrects** two raw `kaizen_team` entries, the second of which overturned the
  first). `MATCH (u:User) WHERE u.tokenHash IS NOT NULL RETURN count(u)` profiles as
  `Node By Label Scan | (u:User) | Records produced: 51` even with `u.userId` indexed — an
  existence predicate is not a range and gets no index. Adding an "always-true" range conjunct on
  the indexed property (`WHERE u.userId > '' AND …`) *does* upgrade the plan to
  `Node By Index Scan`. **Do not use it.** Two independent reasons: (1) **it is not always true**
  — any non-string `userId` compares to `''` as `NULL` and the row vanishes, silently, with no
  error (the same graph returned **51** rows on the bare label scan and **50** on the
  "equivalent" index-scan form, the missing row being a `userId: 42` integer; see the
  cross-type-comparison entry under *Cypher dialect*), which in a delete is a silent
  under-delete; and (2) **it buys no selectivity** — both plans visit the whole label, and the
  index form measured *slower* (0.128 ms vs 0.082 ms median over 20 runs at 52 nodes; 0.455 ms
  vs 0.277 ms single-shot here). An index scan in the plan is not automatically a win: read
  `Records produced`, not the operator name. If a label genuinely needs an anchored scan, index
  the property you actually filter on, or traverse in from an indexed anchor.

- **A `$param IS NULL OR prop = $param` optional-filter idiom defeats an otherwise-available
  index — even when `$param` is bound to a real, selective value** (verified 2026-08-22 on
  v4.18.11 / module `41811`, at 1000-node scale). `MATCH (a)-[r:SAME_AS]->(b) WHERE $status IS
  NULL OR r.status = $status ...` — the standard "optional filter parameter" idiom for a
  listing endpoint that's sometimes filtered, sometimes not — profiles as `All Node Scan | (a)`
  (1000 records) → `Conditional Traverse` → `Filter`, completely ignoring the `SAME_AS.status`
  relationship index, **even on the call where `$status` is set to a real value** (not just the
  `NULL` "list everything" call, where a full scan is unavoidable anyway). The exact same
  predicate written as a direct pattern-property match on the same parameter —
  `MATCH (a)-[r:SAME_AS {status: $status}]->(b) ...` — profiles as a clean `Edge By Index Scan`
  the moment `$status` carries a real value. Consequence: don't write a single query with an
  `IS NULL OR` guard for an "optionally filtered" listing endpoint on this build — branch at the
  repository layer into two distinct query strings (filtered: direct pattern-property match;
  unfiltered: the same shape minus the property, which costs a full scan regardless of how it's
  written — verified no phrasing avoids that for "list everything with no anchor" queries).
  Surfaced designing `list_matches(status=None, limit)` for `falkor-chat`'s entity-fusion audit
  surface — full comparison at `falkor-chat/docs/plans/document-ingestion-graph.md` §1.7.
  **Refinement — a literal-sentinel `coalesce` DOES keep a RANGE predicate on `Node By Index Scan`
  in every combination, including the fully-unfiltered call; this corrects the "no phrasing avoids
  that" clause above for the range case** (verified 2026-09-07, module `41811`, live
  `GRAPH.EXPLAIN` against an indexed `Product` label). `WHERE p.price >= coalesce($minPrice, -1.0)
  AND p.price <= coalesce($maxPrice, 1e9)` plans `Node By Index Scan` with all params `NULL`, while
  the same filters written as `IS NULL OR` guards plan `Node By Label Scan` even with a real,
  selective value bound. **It does not work for an EQUALITY predicate via a self-referential
  coalesce** — `WHERE p.categoryNormalized = coalesce($category, p.categoryNormalized)` alone plans
  `Node By Label Scan` with `$category` bound and the property indexed, where plain
  `= $category` on that same property is a clean index scan. Two caveats before reaching for it:
  (1) an index scan spanning the entire range buys no *selectivity* on the unfiltered call — read
  `Records produced`, not the operator name (the `IS NOT NULL` entry above); (2) the sentinel bounds
  are a silent type/NULL filter — a row whose property is `NULL` or non-numeric compares to the
  sentinel as `NULL` and vanishes, where the `IS NULL OR` form would have returned it (the
  cross-type-comparison entry under *Cypher dialect*). Sound only where the property is guaranteed
  present and of the sentinel's type. Live use: `falkor-chat`'s `Repository.filter_products`,
  `falkor-chat/docs/QUERIES.md` §15.2.

- **A multi-property `OR` — a genuinely different disjunct on each side, not the same property
  repeated — DOES stay index-anchored per disjunct, unlike the single-property `IS NULL OR`
  idiom above; don't conflate the two shapes** (verified 2026-09-11, module `41811`, disposable
  `ws:docprobe2` graph). Setup: `Document.lshBand0..lshBand7`, eight separate plain RANGE
  indexes (one per property, no composite), 10 `Document` nodes with distinct band values.
  `WHERE d.lshBand0 = $band0 OR d.lshBand1 = $band1 OR ... OR d.lshBand7 = $band7` (8 disjuncts,
  8 different properties, each equality-tested against its own bound parameter) plans a single
  `Node By Index Scan | (d:Document)` with **no** `Node By Label Scan` anywhere in the plan, and
  `Records produced` at that operator tracks the true match count exactly (2 for two planted
  matches on different bands, 1 when only the *last* disjunct — `lshBand7` — matches [ruling out
  "only the first disjunct gets an index"], 0 for an all-miss negative control) — i.e. the engine
  unions per-disjunct index scans rather than falling back to a full scan and filtering. This is
  a different shape from the `$param IS NULL OR prop = $param` "optional filter" entry above
  (same property compared to a possibly-null parameter, which defeats the index even with a real
  value bound) — that finding does **not** generalize to "any `OR` defeats an index on this
  build"; a genuine multi-property disjunct, each side plain `prop = $param` with no `IS NULL`
  guard, is safe and stays index-anchored. Surfaced verifying `falkor-chat`'s
  `find_update_shortlist` band-equality candidate lookup,
  `falkor-chat/docs/plans/document-ingestion2.md` §4 Stage D/§7.

- **A function call wrapped around an indexed property forfeits the index — there are no
  expression/functional indexes on this build** (verified 2026-09-07, module `41811`).
  `MATCH (p:Product) WHERE toLower(p.categoryNormalized) = 'audio'` plans `Node By Label Scan` +
  `Filter`; the identical predicate written `p.categoryNormalized = 'audio'` plans
  `Node By Index Scan`. So case-/whitespace-insensitive matching is done by **precomputing a
  normalized property, indexing that, and normalizing the parameter client-side** before comparing
  — never by normalizing the property inside `WHERE`. (`falkor-chat` carries
  `Product.nameNormalized`/`categoryNormalized` for exactly this; `docs/QUERIES.md` §15.1–15.2.)
- **An `OR` across two label-specific properties as the scan anchor**
  (`WHERE n.propA = $x OR n.propB = $x`) profiles as an `All Node Scan` even when
  both properties are indexed. Use two separate `OPTIONAL MATCH`es (one indexed
  lookup per label) + `coalesce()` instead. The `OR` form is fine once `n` is
  already bound by an indexed/traversal anchor — it's only a scan-anchor problem.

- **A composite keyset-pagination predicate over ONE indexed column still plans as a bare
  `Node By Index Scan`, no residual `Filter`** — `WHERE col > $x OR (col = $x AND tiebreak >
  $y)` (the standard `(col, tiebreak)` keyset-cursor shape) anchors cleanly on `col`'s index
  even though the predicate is a compound OR. Verified on an edge build; re-profile this
  specific shape on engine upgrades before trusting it as a settled fact — it is exactly the
  kind of planner behavior that moves between releases.

- **A `WHERE` predicate on ANY pattern variable can pull the label-scan anchor onto that
  variable's label — even when a much smaller, filter-free label sits elsewhere in the same
  pattern** (verified 2026-07-28 on v4.18.11 / module `41811`). `MATCH (r:Small)-[:REL]->(m:Big)
  WHERE m.someUnindexedProp = $x` anchors on `Node By Label Scan | (m:Big)` — scanning the
  **entire** `Big` label — even with `Small` at 3 rows and `Big` at 20,000. Relative cardinality
  does not drive the anchor choice here; "which variable carries a `WHERE` predicate" does.
  Confirmed independent of `MATCH` clause order/split (single vs. two-`MATCH`) and of pattern
  direction. **The identical pattern with NO `WHERE` at all correctly anchors on the smaller
  label** — so the trap is specifically "one side filtered, the other not," not something
  inherent to two-hop patterns generally. **Fix:** add a second, even functionally-vacuous,
  predicate on the variable you want as anchor (e.g. `WHERE r.someProp >= 0 AND m.filter = $x`)
  — this alone, with **no index required**, redirects the anchor to that variable's (smaller)
  label scan. A supporting range index on top upgrades that label scan to an index scan, but
  **an index alone, with no predicate change, is a confirmed no-op for this shape** — tested
  explicitly (index present, no extra predicate → anchor unchanged, index unused). Surfaced
  building falkor-chat's `find_runs_for_thread`
  (`(WorkflowRun)-[:TRIGGERED_BY]->(Message)`, filtered on `Message.threadId`, unindexed) —
  full write-up with PROFILE output in `falkor-chat/docs/QUERIES.md` §12.14.

- **A guarded-CAS `WHERE` on a *second* indexed property folds INTO the `Node By Index Scan` —
  no residual `Filter` operator** (verified 2026-07-20 on v4.18.11 / `ws:test`).
  `MATCH (r:Run {runId:$id}) WHERE r.status = 'waiting' SET …` with RANGE indexes on both
  `runId` and `status` profiles as `Node By Index Scan | (r:Run)` → `Update` → `Project`, and
  when the CAS fails the scan itself reports `Records produced: 0`. The planner still anchors
  on the **selective** property: with five other `status:'waiting'` rows present the scan
  produced exactly 1 record. Consequence: a zero-row CAS is a *scan-level* miss, so **nothing
  downstream of it runs — no `SET` is partially applied**. Don't read the absent `Filter`
  operator as "the predicate was dropped".

- **Two independently-indexed properties on the same label, both `AND`-ed plain `WHERE`
  predicates (not a `{prop:$x}` pattern-property match) — a numeric range AND a `status IN
  [...]` list — fold into ONE `Node By Index Scan`, not "pick one, filter the other"**
  (verified 2026-07-31 on v4.18.11 / module `41811`, `falkor-chat` `ws:test` + `ws:acme`).
  `MATCH (r:Label) WHERE r.rangeProp >= 0 AND r.tagProp IN [...] AND r.unindexedProp = $x`
  profiles as a single `Node By Index Scan | (r:Label)` whose **output already reflects both
  indexed predicates** — an unindexed third predicate (`unindexedProp`) is the only thing that
  surfaces as a separate `Filter` operator above it. Distinguished from "only one property
  actually anchors, the other rides along coincidentally" by planting a probe row that matches
  one indexed predicate but fails the other (`status:'done'` with `startedAt:-5` against
  `WHERE startedAt >= 0 AND status IN [...]`): the probe was excluded **at the index-scan
  step itself**, before `Filter` ran, proving both predicates are evaluated inside the scan.
  Complements the CAS entry above (which showed one *pattern-property* match + one *`WHERE`*
  match folding together) — this is the same folding behavior for two independent plain
  `WHERE` predicates. Consequence for "which index does this query use": with two indexed
  predicates present, the honest answer is **both, combined** — don't force a single-index
  answer when profiling a query shaped like this. Full isolation-test write-up (drop-each-
  predicate variants + the probe-row test) in `falkor-chat/docs/QUERIES.md` §12.15.

- **A bare label constraint on a relationship-pattern endpoint forces a full `Node By Label Scan`
  on that label, even when the relationship itself carries a selective index and that endpoint
  node has no property predicate of its own** (verified 2026-08-22 on v4.18.11 / module `41811`,
  at 1000-row label cardinality). `MATCH (a:Entity)-[r:SAME_AS {matchId:$id}]->(b:Entity) SET
  r.status=... ` profiles as `Edge By Index Scan | [r:SAME_AS]` (correctly selective, 1 record)
  sitting *underneath* a `Node By Label Scan | (a:Entity)` that reports **1000** records produced
  — the full label — even though the edge index alone already fully determines the match. The
  label scan fires whether the label sits on the start endpoint alone, the end endpoint alone, or
  both; dropping the label from the pattern entirely (`MATCH (a)-[r:SAME_AS {matchId:$id}]->(b)
  SET ...`) collapses the plan to a single clean `Edge By Index Scan`, with the node's actual
  label/properties still readable off the unlabeled pattern variable (`a.entityId`,
  `labels(a)[0]`) — omitting the label from the pattern costs nothing semantically when the
  relationship type already implies what the endpoints are. Consequence: for any query anchored
  on an indexed relationship property (a matchId-style lookup, a status-filtered global listing),
  don't assert the endpoint labels "for clarity" — it silently reintroduces the full label scan
  the relationship index was supposed to avoid. Full worked example (with PROFILE output at both
  1000- and unlabeled-clean shapes): `falkor-chat/docs/plans/document-ingestion-graph.md` §1.4.

## Ops, config & tooling

- **`GRAPH.RO_QUERY`** routes to read replicas — use it for all read-only traffic.
- **`falkordb-py`'s `result.header` is a list of `[type_code, column_name]` pairs, and an
  UN-ALIASED `RETURN` expression is named by its literal expression text** (verified 2026-09-07,
  falkordb-py 1.6.1, module `41811`). `RETURN p.name AS name, p.price AS price` gives
  `[[1,'name'],[1,'price']]`; the same query with no `AS` gives `[[1,'p.name'],[1,'count(p)']]`.
  Consequence for any generic row-to-dict mapper built from `res.header` — a query whose column set
  isn't known ahead of time — expect keys like `"c.name"` and `"count(p)"`, never bare property
  names: alias every column you intend to key on, or key on the expression text deliberately.
  **The hard-error corollary: two identically-named result columns are rejected outright**
  (verified 2026-09-08, module `41811`) — `RETURN d.id, d.id` and `RETURN count(d), count(d)` both
  fail with *"Error: Multiple result columns with the same name are not supported."* It is raised
  during **query validation, not only at execution**: `GRAPH.EXPLAIN` on the same text errors
  identically, so no rows are ever produced. There is still no client-side parse step, so the
  caller meets it as a `redis.exceptions.ResponseError` raised by `.query()`/`.ro_query()` at call
  time — any builder that assembles a projection list from model or user input must check the
  returns list for duplicates itself (`falkor-chat`'s `querygen.compile()` does).
- **Renaming a graph is a plain Redis `RENAME`/`RENAMENX` on its key — atomic, non-destructive,
  and fully supported. `GRAPH.COPY` + `GRAPH.DELETE` is NOT needed and should not be used**
  (verified 2026-09-07, module `41811`, disposable graph). A graph is a single Redis key of type
  `graphdata`, so `RENAME old new` moves it in place: node/edge counts identical, `db.indexes()`
  reports the RANGE index still `OPERATIONAL`, `GRAPH.EXPLAIN` still plans `Node By Index Scan`
  through it, reads **and** writes both work on the new name, and even the query-plan cache
  follows the key (`Cached execution: 1` on the new name for a plan compiled under the old one)
  — the internal `GraphContext` is moved, not rebuilt. The old key is genuinely gone: `EXISTS`
  returns `0` and `GRAPH.RO_QUERY` on it errors *"Invalid graph operation on empty key"*, not a
  lingering alias. **Prefer `RENAMENX`** — it returns `0` and refuses rather than clobbering when
  the target name is taken (`1` when free). The `GRAPH.COPY` + `GRAPH.DELETE` alternative is
  destructive and transiently doubles RAM for no benefit.
- **Queue depth is directly observable: `GRAPH.INFO` reports a `# Waiting queries` section
  alongside `# Running queries`** (verified 2026-09-08, module `41811`, shared dev instance).
  Paired with `GRAPH.CONFIG GET MAX_QUEUED_QUERIES` (default `25` here), that gives an
  *observable* headroom metric — so a load-test or capacity done-condition can assert the peak
  observed queue depth against the cap, instead of degrading to the much weaker "no query was
  rejected", which only goes red once the cap has already been hit. `GRAPH.INFO` takes no graph
  key: it is instance-wide, and also carries an `Object Pool` section.
- **`RESULTSET_SIZE` (default 10000) silently caps *every* result set, including one with an
  explicit larger `LIMIT`** — `GRAPH.CONFIG GET RESULTSET_SIZE` → `10000`; a query with `LIMIT
  50000` against a graph holding 110k+ matching rows still returns only ~10,000, with nothing in
  the response marking it as capped rather than exact. Any tool reporting a result count (a
  wrapper, an MCP tool, a manual script) needs to either raise `RESULTSET_SIZE` for the session or
  document that its reported row count can itself be a silent cap, not the true total — a claim
  like "the reported total is always exact" is false against this default. (Verified 2026-07-30,
  v4.18.11, via the `cypher` MCP tool vs. raw `GRAPH.RO_QUERY`.)
  **A capped reply is structurally indistinguishable from a complete one, which bounds the
  statistics-trailer discriminator below** — re-measured 2026-09-09, module `41811`:
  `GRAPH.RO_QUERY <g> "UNWIND range(1,200000) AS x RETURN x"` → exit 0, **10003 lines** (header +
  10000 rows + `Cached execution` + trailer), last data row `10000`, last line
  `Query internal execution time: 5.220555 milliseconds`. The trailer therefore proves the server
  **ran the query to completion**; it does **not** prove the reply carries every matched row. Those
  are different claims and only the first is checkable from the reply's shape — a caller needing
  **completeness** must assert a row count against its own expectation, or page the query.
- **A destructive op run through a wrapper script used to be invisible to `guard-destructive-ops.sh`
  — fixed 2026-08-08 (C-311), don't assume it's still open.** `pipeline.sh --reset` runs
  `redis-cli ... GRAPH.DELETE` *inside* the script, so the literal string never appeared in the
  outer Bash command the guard inspects (observed 2026-07-30 refreshing `cpg_falkorchat`, even
  backgrounded/`nohup`ed). The guard now also basename+flag-matches `pipeline.sh ... --reset`
  directly (`claude/scripts/guard-destructive-ops.sh`), closing this specific gap. General lesson
  that outlasts this one fix: a Bash-command-string guard is blind to anything a wrapper script
  does *internally* unless the wrapper's own invocation is pattern-matched too — when adding a new
  destructive wrapper script, check whether the guard needs a matching clause for it, don't assume
  the existing `GRAPH.DELETE`/`FLUSHALL` patterns cover it by transitivity.
- **A read via `GRAPH.QUERY` materializes an empty graph key** — running e.g.
  `MATCH (n) RETURN count(n)` against a *non-existent* graph creates the key (it
  then shows up in `GRAPH.LIST` with 0 nodes). `GRAPH.RO_QUERY` on the same
  non-existent graph instead returns `ERR Invalid graph operation on empty key`
  and creates **nothing**. So to test "does this graph already hold data?"
  without side effects, probe with `RO_QUERY` and treat the `empty key` error as
  "absent/empty"; never scan the whole `redis-cli` reply for digits (the
  `Query internal execution time: 0.179153 milliseconds` line makes everything
  look non-empty — parse the count from the lone pure-integer output line).
  Concrete extraction that works: `redis-cli ... GRAPH.QUERY ... --no-raw | awk
  '/^[0-9]+$/{last=$0} END{print last}'`; a naive `grep -oE '[0-9]+' | tail -1`
  instead grabs digits from the stats line and reports a phantom huge count
  (one run misread a real 29,447 as 273,336).
  (Verified 2026-07-17 on v4.18.11; surfaced building the CPG loader, `joern-cpg` skill.)
- **`redis-cli` exits 0 on a server *error reply*, and prints the error text to STDOUT** — so `$?`,
  `||` and `set -e` cannot see a rejected or malformed query, and `… >/dev/null` throws away the
  only evidence there was. Paired control, 2026-09-08, `redis-cli 7.0.15` against
  `localhost:6379` (module `41811`): `redis-cli PING` → stdout `PONG`, exit 0; `redis-cli
  NOTACOMMAND` → stdout `ERR unknown command 'NOTACOMMAND'`, **stderr empty, exit 0**;
  `GRAPH.RO_QUERY kaizen_team "MATCH (n:Agent RETURN count(n)"` → stdout `errMsg: Invalid input
  'R': …`, stderr empty, **exit 0**; and `( set -e; redis-cli GRAPH.RO_QUERY … >/dev/null; echo … )`
  runs its trailing `echo` and exits 0. The one case that *does* exit non-zero is a **connection**
  failure: `redis-cli -p 6399 PING` → empty stdout, `Could not connect to Redis at
  127.0.0.1:6399: Connection refused` on **stderr**, exit 1. That split is what lets the trap
  survive review — `||` and `set -e` visibly work against a down server and silently do nothing
  against a rejected query, so testing the guard the easy way confirms it.
  **And error replies carry no uniform prefix, so classifying the captured text by prefix does not
  rescue it.** Paired control, same session, `GRAPH.RO_QUERY` against `kaizen_team`, checking each
  reply against the `errMsg:*|ERR\ *|WRONGTYPE*` prefix set: a **parse** error
  (`THIS IS NOT CYPHER`) → `errMsg: Invalid input 'T': …`, caught; an **absent graph** →
  `ERR Invalid graph operation on empty key`, caught; but two **runtime** errors come back
  completely bare — `RETURN nosuchfunc(1)` → `Unknown function 'nosuchfunc'` and
  `MATCH (n:KaizenEntry) RETURN keys(n.fact)` → `Type mismatch: expected Map, Node, Edge, or Null
  but was String`. Both **missed** by that prefix set, both exit 0. A prefix-matching wrapper
  therefore reports a real runtime error as success, and the control (a valid query) is the only
  thing that makes the miss visible.
  **What a caller must actually do: assert the intended effect positively, or use a client that
  raises.** Read the write back and check the value you meant to store (`falkordb-py` raises on the
  error reply and is the simplest correct answer). Keep the `||` for the connect case, and treat
  any prefix `case` as a courtesy message, never the check. If you must classify a `redis-cli`
  reply, the discriminator that held across all five probes above is **affirmative**: a successful
  reply carries a header row plus the `Query internal execution time:` trailer, while every error
  reply is a single bare line. **`GRAPH.QUERY` *write* replies carry the trailer too** — measured
  2026-09-09, module `41811`, on the shape a provenance stamp actually uses
  (`MERGE (b:CpgBuildInfo) SET b = {…} RETURN 1`): the run that created the node
  (`Labels added: 1` / `Nodes created: 1` / `Properties set: 2` / `Cached execution: 0` / trailer)
  and the re-run where the `MERGE` matched both end with it, as does a plain `CREATE (:X)`. So the
  discriminator is safe to apply unconditionally to `GRAPH.QUERY` and `GRAPH.RO_QUERY` alike —
  but **only** to those two: `GRAPH.DELETE` answers the bare status `OK` (observed 2026-09-09,
  exit 0) — no trailer at all — so a helper that gates on the trailer must refuse any other
  command rather than mis-judge its reply as a failure.
  **Anchor the test on the reply's LAST line, and require the whole trailer prefix.** Same session:
  a timed-out query answers the single bare line `Query timed out`, which shares its first word
  with the trailer — a loose `Query*` test passes it. And a substring search over the whole reply
  can be satisfied by returned *data* carrying the literal. `${out##*$'\n'}` against
  `"Query internal execution time:"*` is the form that survives both.
  **That discriminator is sound rather than merely observed, because a mid-stream runtime error
  aborts the whole reply.** Re-measured 2026-09-08, module `41811`:
  `GRAPH.RO_QUERY <g> "UNWIND [1,0] AS x RETURN 1/x AS stray"` — whose first row *is* producible —
  comes back as the single line `Division by zero` and nothing else: no `stray` column header, no
  `Cached execution`, no trailer, exit 0. `Query timed out` (this instance's default
  `TIMEOUT 1000`) has the identical shape. The rows already produced are discarded, so **a
  mid-stream abort leaves no partial reply for the trailer test to mistake for a whole one** —
  scoped deliberately to *aborts*, because the generalisation "there is no partial reply to worry
  about" is **false**; see the `RESULTSET_SIZE` bullet above. Requiring the trailer makes a
  *negative* assertion ("this query returned no rows") fail-closed instead of fail-quiet — a
  zero-row success still prints header + blank + `Cached execution: 0` +
  `Query internal execution time:`, and every abort prints one bare line.
  Never `redis-cli … >/dev/null` on a write whose

  failure matters. Live instance of the outer trap, since fixed:
  `skills/joern-cpg/scripts/pipeline.sh:199` at `6012ddb` wrote the `CpgBuildInfo` provenance
  marker as `redis-cli … GRAPH.QUERY "$GRAPH" "$STAMP" >/dev/null`, making a failed stamp invisible
  after a multi-hour build; `9124a1f` replaced it with an `rq()` helper that captures stdout and
  `case`-matches those same three prefixes — which closed the discarded-output half and left the
  bare-runtime-error half open (it returned 0 on `Unknown function`, `Type mismatch:`,
  `Division by zero` and `Query timed out`). **Closed 2026-09-09** by replacing the prefix
  blacklist with the affirmative last-line trailer test above; the probe table, its passing
  controls and the mutation runs are in `claude/graph-dba/kaizen/history.md` (U28).
- **`GRAPH.EXPLAIN` (unlike `GRAPH.QUERY`/`GRAPH.RO_QUERY`) refuses to run against a graph key
  that doesn't exist yet** — `GRAPH.EXPLAIN <key> "<any syntactically valid query>"` against a
  never-created key errors `ERR Invalid graph operation on empty key` and materializes nothing
  (re-verified live 2026-08-18, module `41811`). Consequence: `GRAPH.EXPLAIN` **cannot** be used
  to syntax-check a **write** query when there's no existing graph to point it at. Workaround:
  run the write query via `GRAPH.RO_QUERY` against any existing graph (not necessarily the
  target one) — it parses the query first and only then rejects it for being a write ("graph.
  RO_QUERY is to be executed only on read-only queries"), which proves the Cypher parses with
  zero side effects. `GRAPH.PROFILE` isn't a substitute either — see the entry below (it
  silently *executes* writes rather than explaining them).
- **Bolt port is `65535`** per `GRAPH.CONFIG` (not the Bolt default).
- **Default `TIMEOUT` is 1000ms — and writes ignore it entirely**; a write runs to
  completion regardless of clause or default. Reads enforce it batch-granularly
  (slightly-over queries can slip through). The client `timeout=` pass-through
  (`g.ro_query(q, params=…, timeout=…)`) works and is **uncapped while
  `TIMEOUT_MAX=0`**. Re-measured 2026-09-08 on module `41811`
  (`TIMEOUT 1000`, `TIMEOUT_DEFAULT 0`, `TIMEOUT_MAX 0`): `UNWIND range(1,20000000) AS i WITH i
  WHERE i%5000000=0 CREATE (:Blob {i:i})` ran **1.75 s** untouched, while a 4-way cartesian
  `MATCH` over 400 nodes was killed with `Query timed out` at **exactly 1.00 s**.
  - **So the only bound on a write is the client's `socket_timeout` — and it does not roll back.**
    Measured: a `redis.Redis(socket_timeout=0.5)` client with retries disabled raised
    `redis.exceptions.TimeoutError` at 0.50 s on that same write, and **all four nodes were
    committed** — the server had never stopped. There is no cancel path; a client-side timeout
    abandons the *answer*, not the *work*.
  - **Worse, a retrying client re-applies the write.** A plain `redis.Redis(...)`'s pooled
    connection carries `Retry(ExponentialWithJitterBackoff(), retries=10)` whose supported set
    **includes `TimeoutError`** — so the identical call through one re-issued the non-idempotent
    write repeatedly and left **36 nodes** where the query creates 4. Non-idempotent Cypher over a
    short client timeout multiplies silently.
  - **This lab is not exposed by default: `falkordb-py` disables retry, so a FalkorDB call gets
    one attempt.** Verified 2026-09-08 on **both** venvs in this repo that carry the driver —
    `falkor-chat/server/.venv` (falkordb-py 1.6.1 / redis-py 8.0.1) and `cypher-mcp/.venv`
    (falkordb-py 1.6.2 / redis-py 8.1.0) — with identical results, so this is **not** a version
    difference. It is an *object* difference, and knowing which object you hold is the whole
    trick, because the two accessors disagree by design:
    - `FalkorDB(...).connection` is the **`redis.Redis` client**. It has no `.retry` attribute at
      all — reading one raises `AttributeError: 'Redis' object has no attribute 'retry'` on both
      versions above — and `client.get_retry()` returns **`None`**.
    - That client's **pooled `Connection`**
      (`client.connection_pool.get_connection()`) is where the policy lives:
      `conn.retry._retries == 0`. The same accessor on a plain `redis.Redis`'s pooled connection
      returns **10**, which is the contrast that matters.

    Either reading confirms it; neither is the canonical one. `AttributeError` here means you are
    holding the client, not that the driver changed. Consequence: `falkor-chat` (`db.py:44`,
    `FALKORDB_SOCKET_TIMEOUT` default 10 s) and `cypher-mcp` (`server.py:903`) both get one
    attempt. A helper script or test harness that reaches for bare `redis-py` instead does **not**
    inherit that, and is the shape to watch for.
- **`GRAPH.MEMORY USAGE` under-reports vector-index memory** (reports
  `indices_sz_mb: 0` with a live HNSW index holding real vectors) — size
  vector-heavy workspaces from `INFO memory` deltas instead, until fixed upstream.
- **`GRAPH.PROFILE` is not read-only, and neither `GRAPH.RO_QUERY` nor a `PROFILE`/`EXPLAIN`
  prefix inside a query string is honored as a planning directive — the query just executes.**
  Per docs.falkordb.com/commands/graph.profile: unlike `GRAPH.EXPLAIN`, `GRAPH.PROFILE`
  *executes* the query including any write operations (it only suppresses `RETURN` output, not
  the side effects) — so a `PROFILE`-prefixed write bypasses any `GRAPH.RO_QUERY`-based
  read-only guard. Live-verified on v4.18.11: an `EXPLAIN`/`PROFILE`/`profile` prefix inside a
  `GRAPH.QUERY` or `GRAPH.RO_QUERY` string — including after a `//` or `/* */` comment — is
  silently ignored and the query runs for real, returning results with no error and no plan;
  actual plans come only from the separate `GRAPH.EXPLAIN`/`GRAPH.PROFILE` commands.
  Consequence for any tool that sniffs a directive prefix to route to
  `Graph.explain()`/`Graph.profile()` or to refuse a write: whitespace/case trimming is not
  enough, and a leading comment defeats the sniff. Also: in `falkordb-py` 1.6.x,
  `Graph.explain`/`Graph.profile` take no `timeout` parameter (only `query, params`), unlike
  `query`/`ro_query`. (Verified 2026-07-24/25, surfaced reviewing an MCP read-tool design.)
