# Connected code graph — graph model, store design & sizing

> **Status:** active · **Owner:** `graph-dba` · **Tracks:** — (`docs/requirements/connected-code-graph.md`) · **Extends:** `docs/plans/connected-code-graph.md` · **Version:** 2.1

_Revision 2026-10-26: v2 adds §6 (B1 edge directions and traversal table, B2 graph side, per-requirement schema deltas after S1–S38 and the use-case re-check, new stakeholder questions); §3.1 rule 1, Q3 peak, Q7 edge list and Q10 ACL corrected in place to point at it._

_Revision 2026-10-28 (v2.1): peak-memory figures in §0, Q3, §5 Q-R5, §6.4 and §6.5 restated for the decided default of **2** concurrent previews (FR-23, Q-R8; analyst Pass 3 nit P3-n1); four previews kept only as the stated alternative._

Graph-side review of the architecture proposal `docs/plans/connected-code-graph.md` (v1): answers
to its §6.2 questions (plus one relayed from the ML unit, `docs/plans/connected-code-graph-ml.md`
M4), a concrete FalkorDB model for `ccg_<system>`, and findings for the architect's revision and
the stakeholder. Requirements are still `Interviewing`; nothing here is approved for
implementation.

## 0. Verdict and digest

**Verdict: sound with changes.** The core store decision holds and is now live-verified:
FalkorDB cannot traverse across graph keys (V1), so one summary-level snapshot graph per system is
the right shape. At ~0.115 MB per source file it is ~20× smaller than full CPGs, ~1.5 GB for a
10,000-file system (Q3), so federation is not needed at workplace scale. The swap mechanism exists
and is reader-safe (`RENAME`, V2). What has to change is a set of model rules and one enforcement
point:

- **High:** raw queries are not read-only through `cypher-mcp` as it stands. FR-25 has to be
  enforced by a per-system Redis ACL credential (G1, V5, V6). Constrained ids must be hashes,
  because readable keys from arbitrary codebases can crash the whole instance (G2). Name-matched
  calls must route through a `Name` node and closures (static `COVERS`) must not be materialized,
  or edge counts go quadratic (G3, V13).
- **Medium:** taint stitching and reach are engine algorithms, best materialized as build-time
  `Finding` chains (G4). Verdicts belong in a file of record, applied into the snapshot (G5).
  Overlays are built from bundles, not copied (G6). Swap by `RENAME` with zero kept snapshots
  (G7). The code graph gets a dedicated FalkorDB instance (G8).
- **Compatible with the ML unit's M4:** no vector index on the fact path. Exact indexed anchors
  plus a full-text keyword tier (complete, deterministic result sets) cover it (Q9).
- **Six requirement-level questions** for the stakeholder (§5), chiefly what FR-24 protects
  (content vs. structure) and when a verdict takes effect.

**v2 (§6):** B1 is fixed by keeping storage as is and making traversal direction depend on the
operator, the edge type and the arrival state (normative table, §6.2). B2's graph side is points
plus field-path `DATA` edges, the same schema for shell and prompts (§6.3). Carrying data flow and
control dependence for all use cases is feasible only as a per-method projection: ~0.3 MB per
file, ~3–4 GB per 10k-file snapshot, ~18 GB peak at the decided default of two previews (§6.4). Five new stakeholder
questions (§6.5).

Nothing here requires a source, test or config change now. All findings are design input for the
proposal's revision and P0.

## 1. Live verification log

All probes 2026-10-25 on `falkordb/falkordb:v4.18.11` (module `41811`, Redis 8.6.3). Writes ran
only in a **throwaway container** (`gdba-ccg-probe`, port 6391, `--rm`, stopped after); the shared
`falkordb-dev` instance (6379) was read with `GRAPH.RO_QUERY` / `GRAPH.MEMORY USAGE` only.

| # | Claim probed | Result |
|---|---|---|
| V1 | Cross-graph traversal in one query | **Impossible.** No `USE` clause (`USE other MATCH …` → parse error *"expected USING PERIODIC COMMIT"*); `CALL dbms.procedures()` lists no multi-graph procedure (only `algo.*`, `db.*`, `dbms.*`); cross-graph edges already known to silently no-op (`claude/graph-dba/falkordb-quirks.md`). |
| V2 | Atomic swap by `RENAME next live` while `live` exists and a read is in flight | **Safe.** 3 trials, each with a ~39 s `GRAPH.RO_QUERY` running on `live`: `RENAME` returned in 1–3 ms; a new read immediately saw the new snapshot; the in-flight read **completed normally on the old snapshot** (correct old marker, full count); no error, no crash; the old graph's memory was released afterwards (44 MB → 24 MB). `RENAME`/`RENAMENX` already verified to carry indexes and plan cache (quirks file). |
| V3 | `GRAPH.COPY` cost (overlay option) | 115 MB graph copied in **2.37 s**, indexes carried (`OPERATIONAL`), but the Redis main thread **stalled up to 280 ms** (PING latency) — instance-wide, every graph. |
| V4 | `EXPIRE` on a graph key (overlay TTL) | Works: key gone after TTL, memory freed. |
| V5 | Redis ACL as the read-only / scope enforcement layer | User `-@all +graph.ro_query +graph.explain ~ccg_size1k`: `RO_QUERY` on that key works; another key → `NOPERM … access a key`; `GRAPH.QUERY`/`GRAPH.PROFILE` → `NOPERM … command`; a write through `RO_QUERY` → rejected by the engine. **New quirk:** the Redis 7 read-only key selector `%R~key` blocks even `GRAPH.RO_QUERY` (`NOPERM … access a key`) — the module does not declare its key access as read-only, so use `~key` plus a command allowlist, never `%R~`. |
| V6 | `cypher-mcp` as the FR-25 raw client "unchanged" | **Not read-only for every graph.** `cypher-mcp/server.py` (≈ lines 1018–1036): on an `RO_QUERY` write rejection it calls `authorize_write(cypher, agent)` and, if a kaizen write shape matches, re-runs the text with `GRAPH.QUERY` **against whatever `graph` was passed** — no graph-name check. A producer-write shape aimed at `ccg_<system>` would be executed. |
| V7 | `UNIQUE` constraint created over data that already has duplicates | Constraint goes `PENDING` → **`FAILED`** (`CALL db.constraints()`). Usable as a post-load integrity gate. |
| V8 | `algo.SSpaths` for best-path reach | Works with `relDirection:'incoming'`, a `relTypes` list and `weightProp`; **`pathCount` is the total number of paths returned, default 1** (not per target). With a status weight (confirmed = 1, possible = 1000) it returns, per reached node, the lexicographically best path. Tie-break determinism not established. |
| V9 | Full-text index at 25k nodes | `queryNodes` returns **all** 25,000 matches (no RediSearch 10-row default cap). But a query issued right after `createNodeIndex` on the populated label returned **0 rows, silently** — backfill is asynchronous at this size (refines the quirks entry measured at 558 nodes). |
| V10 | Per-graph `telemetry{<graph>}` keys | **New quirk:** every graph has a companion Redis **stream** `telemetry{<graph>}` (capped at 1,000 entries, no TTL) recording recent queries **with full parameter values** and timings; present on the shared instance for every graph (`telemetry{cpg_falkorchat}`, `telemetry{ws:agent-team}`, …). `RENAME` does not move it. |
| V11 | Summary-graph RAM (synthetic, §3 model) | 1,000 files × (1 `File`, 36 `Symbol`, 210 `CallSite`) + 12k `Name` nodes = **259k nodes / 580k edges → 115 MB** `GRAPH.MEMORY USAGE` (37 MB of it indexes); loaded in 13.6 s by naive per-file `UNWIND` batches over one connection. |
| V12 | Today's full-CPG baseline | `cpg_falkorchat` (2026-10-15 build, 97 files): **387k nodes / 2.64M edges, 255 MB** → ~4.0k nodes / ~27k edges / **~2.6 MB per file** (test-heavy files push it above the `skills/joern-cpg/SKILL.md` 2.7k/18k rule of thumb). Summary-level counts in the same graph: 3,520 first-party named methods (~36/file), 20,360 non-operator call sites (~210/file), 6,866 resolved `CALL` edges (34% of call sites). |
| V13 | Name-match fan-out | In `cpg_falkorchat`, 735 names are both a first-party method name and a call-site callee: 1,396 defs × 8,538 sites → **67,547 (site, candidate) pairs vs 9,934 if routed through one shared `Name` node** (6.8×). `__init__` alone: 147 defs × 304 sites = 44,688 pairs. |
| V14 | Bulk/copy commands present | `COMMAND INFO GRAPH.BULK` and `GRAPH.COPY` both present (flag `write`). |

**Proposed additions to `claude/graph-dba/falkordb-quirks.md`** (V5's `%R~`, V9's async backfill
at scale, V10's telemetry streams, V2's RENAME-under-reader isolation): not written there in this
run, because this unit is scoped to one deliverable; listed here so a later `graph-dba` pass folds
them in.

## 2. Answers to the proposal's §6.2 questions

### Q1 — Cross-graph traversal

**Confirmed: FalkorDB cannot traverse across graph keys in one query** (V1). Every `GRAPH.*`
command names exactly one key; there is no `USE` clause, no multi-graph procedure and no Fabric
equivalent; an edge cannot point into another key. The only way to cross keys is application-side
federation (query A, carry ids, query B), which is §3.13's rejected alternative. So **one
`ccg_<system>` graph per system is the right shape**, and it has a corollary the proposal does not
draw yet: **anything an answer must join with the snapshot has to be inside the snapshot.** The
verdicts (Q6) and an overlay's base facts (Q5) are copied in at assembly time, never joined live
from a second key.

### Q2 — Graph model

Answered in §3: contract kinds as a **property** (FR-26: labels can't be parameterized, so a label
per kind would put protocol names into operator text); `status`/`reason` on **edges**, evidence on
**nodes**; **reified `Link` nodes only for cross-boundary links** (candidate sets, ordered wiring
evidence, verdicts), plain edges for the bulk; a verdict on a non-reified possible fact is
addressable by `linkId` (relationship index) or materialized as a `CALLS`/`DISMISSED` edge; anchor
indexes on `Symbol.nameNorm`, `Symbol.qname`, `File.path` (then a line filter over the file's
symbols), `ContractPoint.keyNorm`, `LogTemplate.textNorm`. A normalized-key **prefix** index is
not offered: `STARTS WITH` never uses an index on this build, so prefix and template matching stay
in the linker and in a client-side anchor matcher.

### Q3 — Sizing

| Shape (per source file) | Nodes | Edges | RAM |
|---|---|---|---|
| Full CPG, `cpg_falkorchat` 2026-10-15 build (V12) | ~4,000 | ~27,000 | ~2.6 MB |
| Full CPG, `skills/joern-cpg/SKILL.md` rule of thumb | ~2,700 | ~18,000 | ~1.7 MB (scaled) |
| Summary model §3, synthetic at measured Python densities (V11) | ~260 | ~580 | **~0.115 MB** |

The summary graph is **~15× fewer nodes, ~45× fewer edges and ~20× less RAM per file** than the
full CPG: about one order of magnitude, the low end of the proposal's "one to two". It holds only
**if** name matches go through `Name` nodes (V13) and no closure is materialized (§3.1 rule 3);
materializing either brings back a quadratic term.

**Workplace projection, 10,000 source files in one system:** ~2.6M nodes / ~5.8M edges,
**~1.2 GB**; budget **~1.5 GB** for what the synthetic graph lacks (`Link`, `ConfigEntry`,
`LogTemplate`, `Finding`, `Dependency`). The same system as full CPGs: ~40M nodes / ~270M edges,
**~26 GB**. Peak memory is not one snapshot, though:

`peak ≈ snapshot × (1 live + 1 building + k concurrent overlays + s per-scope views)` (+ Redis
overhead, ~20%). *(v2: s = 0, since S5/S6 limit raw queries to all-repo readers and protect content only; the snapshot itself grows with FR-11's flow edges, §6.3.)*

With k = 2 previews (the decided default, FR-23) and no per-scope views: ~1.5 GB × 4 × 1.2 ≈ **7 GB** for the v1 model, **~18 GB** for the v2 flow-aware model (§6.4); with four previews, ~11 GB and ~30 GB. Run it on a
**dedicated FalkorDB instance** (`maxmemory` set, `maxmemory-policy noeviction`), not on the shared
`falkordb-dev`: §4 G8.

**When federation (§3.13) becomes necessary:** when one snapshot no longer fits about a third of
a shard's RAM (the rest is the build copy, overlays and headroom). On a 64 GB host that is a
~20 GB snapshot, roughly **150k source files in one system**, an order of magnitude past the
workplace target. The realistic risk is not file count but **unbounded edge families**: name-match
products, materialized closures, "any of N endpoints" expansions. Those are model rules (§3.1),
not a reason to federate. **Caveats:** the synthetic graph uses Python densities from one codebase.
P0 must re-measure on real bundles from each FR-1 frontend, especially Java/C# call-site density,
`Link` counts under runtime-built URLs, and `LogTemplate` volume.

### Q4 — Snapshot swap

**Build into a fresh key, then `RENAME ccg_<system>__build_<id> ccg_<system>`** (V2). This is
live-verified: the rename takes ~1–3 ms, carries indexes and the plan cache, and overwrites the old
key in one step. New reads see the new snapshot immediately. **Reads already running finish on
the old snapshot**, which is freed when the last of them ends. `GRAPH.COPY` is never needed, and
there is no alias mechanism in FalkorDB (a pointer key resolved by the engine would give raw-query
users no stable name, and ACL key patterns would have to follow it).

- **Old snapshots kept in FalkorDB: zero.** The rollback unit is the last-good **bundle set** on
  disk. Re-assembling it costs a few minutes and no Joern, and FR-23 / Out of scope exclude past
  snapshots anyway. If an instant rollback is ever wanted, `MULTI; RENAME live prev; RENAME
  build live; EXEC` keeps one previous snapshot with no window where `live` is absent. That is
  standard Redis transaction semantics, **not live-verified here** (verify in P0), and it costs one
  more snapshot of RAM.
- The overwriting `RENAME` (and an overlay's `EXPIRE`, Q5) **destroys a graph without the literal
  `GRAPH.DELETE`**, so `claude/scripts/guard-destructive-ops.sh` will not see it. The refresh
  runner must only ever rename and expire keys under the `ccg_` prefix, and the target key name
  must come from the system manifest, never from caller input.
- The `:Snapshot` node (written last) gives every answer its `snapshotId` (AC-9, AC-21's
  "snapshot ID unchanged").

### Q5 — Overlay graphs

**Full materialization per overlay is affordable** (~1.5 GB for the 10k-file system, Q3) and is
the only sound option. A delta graph plus engine-side merge **is** federation (Q1): a path that
goes in and out of changed units would have to be stitched across two keys in application code,
losing single-query traversal, the published-query property (R1) and raw-query access to overlays.

Build an overlay **the same way as a snapshot, from bundles**: the base bundle set with the changed
units' bundles substituted, re-linked (proposal §3.9 step 2), then bulk-loaded into
`ccg_<system>__ov_<id>`. Do **not** `GRAPH.COPY` the base and patch it. Copying stalls the whole
instance (V3: 280 ms at 115 MB, so seconds at 1.5 GB), and patching means deleting and re-linking
part of a graph, which is the incremental cross-program re-linking that FR-7's full-rebuild
allowance exists to avoid. Load time measured at 13.6 s per 1k files with naive batches, so ~2–3
min for 10k files, less with `GRAPH.BULK`. That is small next to the Joern rebuild of the changed
units (R14).

**Cleanup:** `EXPIRE` on the overlay key (V4), renewed on each use (e.g. 24 h idle), plus a hard
cap on concurrent overlays per system, because each one is a full snapshot of RAM. The cap is a
config value sized from Q3's peak formula. When the cap is reached, the oldest idle overlay is
evicted by the runner, never by Redis eviction.

### Q6 — Verdict store

**Use a file of record, not a second graph.** The verdict store has no traversal workload: it is
looked up by `linkId` once per assembly. It is also the only state that cannot be rebuilt, and
FalkorDB's durability (RDB/AOF on whatever instance hosts it) is the weakest in the system.
Recommended:

1. **System of record:** an append-only JSONL per system, `{linkId, verdict, why, by, at,
   fpConsumer, fpProvider, endpoints: {consumerSymbolId, providerSymbolId, kind, keyNorm}}`, kept
   beside the system manifest. That gives backup by ordinary file backup or git, an audit trail
   for free, and a reviewable diff.
2. **At assembly** the linker loads it into a dict and applies each verdict to the matching `Link`
   / `READS` / `WRITES` edge, or materializes `CALLS {verdict-confirmed}` / `DISMISSED` for a
   name-matched pair (§3.2). If either fingerprint differs from the current symbol's, the verdict
   **expires**: status back to `possible`, `priorVerdict*` set (FR-29, R6a for confirmations).
3. **The `verdict` command** appends to the file **and** applies the same change to the live
   snapshot in **one** write query, so it shows in the next answer, not only after the next refresh
   (§5 Q-R2). That command is the only writer to a live snapshot and uses its own ACL user (Q10).
4. **Orphans:** a `linkId` that no longer resolves (the code at one end was renamed or deleted)
   is not silently dropped. It is counted on `:Snapshot.orphanedVerdicts` and listable, because a
   rename changes `symbolId` and therefore `linkId`. Without this, a renamed function's dismissed
   link reappears with no "earlier dismissal" note.

The proposal's `ccg_<system>_verdicts` graph can still exist as a read-only mirror for raw
queries, but nothing should depend on it. It adds nothing that `Link.verdict` and `DISMISSED`
inside the snapshot don't already provide.

### Q7 — Query shapes

- **Do not use unbounded variable-length path enumeration** (`-[:CALLS|VIA|TO*1..k]-` returning
  paths). FalkorDB returns **every** path, not one per target, so the result is exponential in
  depth on any graph with fan-in and quickly hits the silent 10,000-row cap.
- **Primary shape: level-synchronous BFS driven by the engine.** One parameterized read per depth
  level, `UNWIND $frontier AS id MATCH (n {…id})<-[r:<edge types and direction per §6.2's normative table>]-(m) …
  RETURN id, type(r), r.status, r.reason, m.<id>`, with dismissed pairs anti-joined. The engine
  keeps a visited set and the **best predecessor** per node: fewest possible hops first, then depth,
  then stable id. That predecessor map **is** the evidence chain, so no path is returned by the
  database. Paging is natural: frontiers go out in batches sized so one reply stays well under
  `RESULTSET_SIZE`, and a reply of exactly 10,000 rows is treated as truncated and re-split.
  Deterministic, bounded and cheap. The depth bound is stated in the answer.
- **Candidate optimization: `algo.SSpaths`** with `relDirection:'incoming'`, a `relTypes` list and
  a status `weightProp` (V8) returns one lightest path per reached node. But `pathCount` is a
  total cap (default 1), results are still subject to `RESULTSET_SIZE`, and tie-break determinism
  is unproven. It also needs a weight property on every edge type traversed. Benchmark it against
  the BFS in P0, and adopt it only if it is faster and passes the FR-19 repeat-diff test.
- **Status propagation:** weakest hop on the best path, computed in the engine from the per-hop
  `r.status` the BFS already returns. Nothing about it needs Cypher path functions.
- **Mixed edge types in one expansion** are a plain `[:A|B|C]` list. Operators differ only in the
  type list and the direction (rule 1), which keeps FR-26's "no per-use-case code" honest.

### Q8 — Bulk load path

§3.5. In short: indexes on the empty fresh key **before** loading (edge batches `MATCH` their
endpoints by id); nodes then edges, `UNWIND $rows` batches of a few hundred to a few thousand rows
over **one** persistent connection (the `joern-cpg` loader's lesson: per-statement processes hit
the 128 KB argv limit and reset storms); no client retries on writes (a retried non-idempotent
`CREATE` duplicates silently; `falkordb-py` disables retry by default, bare `redis-py` does not);
a long socket timeout (writes ignore `TIMEOUT`). Full-text index and `UNIQUE` constraints come
**after** the load and are polled to `OPERATIONAL`: the constraints act as the integrity gate
(V7), and full-text backfill is asynchronous at scale (V9). `GRAPH.BULK` (the FalkorDB bulk-loader
command, present on this build) is the optimization for a fresh key if `UNWIND` throughput is ever
the bottleneck. Measured baseline: 259k nodes / 580k edges loaded in 13.6 s.

### Q9 — The ML unit's M4 question (no approximate vector search on the fact path)

**Compatible, and I agree.** The model in §3 has **no vector index**. Every fact-path anchor is an
exact lookup on an indexed, normalized property (`nameNorm`, `qname`, `File.path`, `keyNorm`,
`LogTemplate.textNorm`). The keyword tier of error and log matching uses the **full-text** index,
which is not approximate: it is an inverted index that returns the **complete** match set (V9:
all 25,000 matches, no top-k cut), so it is deterministic for a given snapshot. Three conditions
keep it on the fact path:

1. **No `LIMIT` or score cut-off** on a full-text query that decides membership. BM25 scores may
   only order (RCA ranking, which FR-19 allows to shift).
2. **Wait for `OPERATIONAL`** before the snapshot is swapped in. Right after creation on a populated
   label it silently returns zero (V9), a false "no match" (the quirks file's no-index-zero-rows
   trap in a new form).
3. **Fuzzy terms (`%term%`) produce *possible* items only**, with `reason = 'fuzzy-text'`, and are
   escaped properly. Stripping metacharacters silently merges tokens (quirks file).

The exact tier (normalized template equality, plus client-side matching of the build-time
template's placeholders) runs first, as M4 proposes. If a vector index is ever wanted for a
convenience feature such as "did you mean" suggestions, it belongs outside `ccg_<system>`, or at
least outside every operator's query set.

### Q10 — Read-only raw queries (FR-25) and access scope (FR-24, proposal R3)

**FR-25 must be enforced by the engine credential, not by the client.** Live evidence:

- `GRAPH.RO_QUERY` does reject writes on the plan (quirks file), but **`cypher-mcp` escalates a
  rejected write to `GRAPH.QUERY`** whenever its text matches one of the six kaizen write shapes,
  **on any graph name** (V6). So "reuse `cypher-mcp` unchanged" (proposal §3.6, §3.12) does not
  give AC-21 as written. And `GRAPH.PROFILE` executes writes, so any client exposing it is a
  write path too.
- **A Redis ACL user per system works as the hard guarantee** (V5):

  ```
  ACL SETUSER ccg_raw_<system> on ><secret> resetkeys ~ccg_<system> ~ccg_<system>__ov_* \
      -@all +graph.ro_query +graph.explain +info
  ```

  Verified: `RO_QUERY` on the key works, every other key → `NOPERM`, `GRAPH.QUERY` / `PROFILE` →
  `NOPERM`, and a write via `RO_QUERY` is rejected by the engine. **Do not use `%R~`** (Redis's
  read-only key selector): this module does not declare its keys read-only, so `%R~` blocks
  `RO_QUERY` itself (V5). The raw client, whether `cypher-mcp` or a raw tool on the new `ccg`
  server, connects **as this user**. Then AC-21 holds even if the client is wrong. Two more users:
  `ccg_build` (`GRAPH.QUERY`, `GRAPH.CONSTRAINT`, `RENAME`, `EXPIRE` on `~ccg_<system>*`, used by
  the refresh runner) and `ccg_verdict` (`GRAPH.QUERY` on the live key, used only by the `verdict`
  command).
- The raw user's key pattern deliberately excludes `telemetry{ccg_<system>}`. That stream records
  recent queries **with their parameters** (V10), so it would expose other askers' questions.

**Access scope (FR-24 / R3).** A graph is the only isolation unit FalkorDB has: no row- or
property-level security, and ACLs work per key and per command. Hence:

- **R3(a), raw access only for principals who can read every repo in the system: agree.** It is
  enforceable as "holds the `ccg_raw_<system>` credential", and it costs nothing.
- **R3(b), one view per access scope**, is buildable as `ccg_<system>__scope_<h>`, each its own key
  and ACL user. It costs one more snapshot of RAM per distinct scope (Q3's `s` term). There are two
  ways to build a view: **drop** the hidden repos (then links through them vanish, contradicting
  FR-24's "affected, no access" for the other clients, so FR-19 breaks across clients), or keep the
  topology and **scrub** code-revealing properties (`path`, `line`, `name`, `qname`, `key`) on
  hidden-repo nodes. Scrubbing matches the engine's redaction but still shows counts and shape.
  Whether that is acceptable is a requirement question (§5 Q-R1).
- Within the engine (three clients), redaction at answer time over the full graph (proposal
  §3.10) is correct and needs nothing from the store beyond the denormalized `repoId` on every
  node (§3.2).

## 3. Proposed graph model for `ccg_<system>`

Conventions: labels `PascalCase`, relationship types `UPPER_SNAKE`, properties `camelCase` (the
`falkor-chat` convention; the Joern `UPPER_CASE` keys stay confined to optional `cpg_*` graphs).
FalkorDB OpenCypher only: no APOC/GDS, index before constraint.

### 3.1 Three design rules

1. **Every dependency edge is stored from the dependent to the dependency** (superseded as a
   *traversal* rule in v2: an edge's traversal direction comes from §6.2's normative table, B1). Then *impact* ("what
   breaks if X changes") is always **incoming** reach from X, and *depends-on* is always outgoing
   reach. One direction per question means one `relDirection` for a traversal, one weight scheme,
   and no per-edge-type direction logic in operators (§3.7 of the proposal).
2. **Evidence lives on nodes; status lives on edges.** Everything an evidence hop must cite with
   file + line (FR-20) is a node carrying `path` + `line`: `Symbol`, `CallSite`, `ConfigEntry`,
   `SchemaEntry`. Edges carry only `status`, `reason` and (where a verdict can target them) a
   `linkId`. An evidence chain is then just the node sequence of a path, rendered hop by hop.
3. **Only direct facts are stored; transitive reach is computed by traversal.** No materialized
   closure (static `COVERS` from every test to every symbol it reaches, stitched cross-component
   taint as edges). Closures are what turn a ~1 GB graph into a 50 GB one (§4, G3). Per-symbol
   *aggregates* of a closure (e.g. `staticCovered`) are fine: they are O(symbols).

### 3.2 Model

```
(:Snapshot {snapshotId, systemId, builtAt, vocabVersion, bundleSchemaVersion, manifestHash})   singleton

(:Repo {repoId, name, headCommit})
(:Unit {unitId, repoId, rootPath, language, builtCommit, builtTree, builtAt,
        buildStatus: 'fresh'|'stale-kept'|'not-mapped', failedAt, failReason})-[:IN_REPO]->(:Repo)
(:Unit)-[:HAS_FILE]->(:File {fileId, repoId, unitId, path, kind: 'code'|'config'|'schema'|'script'|'prompt'})
(:File)-[:DEFINES]->(:Symbol {symbolId, repoId, unitId, kind, name, nameNorm, qname, path,
                              lineStart, lineEnd, fingerprint, isEntry, paramToReturn: [int]})
        (:Symbol:Test …)                      secondary label on test symbols (cheap diagonal matrix)
(:Symbol)-[:HAS_CALL]->(:CallSite {callSiteId, repoId, unitId, path, line, calleeName})

-- calls inside a program
(:CallSite)-[:CALLS {status:'confirmed', reason:'resolved-call'|'verdict-confirmed'}]->(:Symbol)
(:CallSite)-[:CALLS_NAME]->(:Name {nameKey, lang, name})-[:RESOLVES_TO]->(:Symbol)   name-matched = possible
(:CallSite)-[:DISMISSED {linkId, by, at, why}]->(:Symbol)                             live verdict, anti-joined
(:CallSite)-[:ARG_FLOWS {param:int, arg:int}]->(:Symbol)       intra-unit summary: param of enclosing symbol → arg

-- contracts and cross-boundary links (reified, because they carry candidates + several evidence items + verdicts)
(:ContractPoint {cpId, kind, key, keyNorm})                    kind is a PROPERTY: http, rpc, topic, mcp_tool,
                                                               library_api, data_entity, config_key, env_var, …
(:Symbol|:CallSite|:ConfigEntry)-[:PROVIDES]->(:ContractPoint)          (provider: handler, schema, tool def)
(:CallSite|:ConfigEntry)-[:CONSUMES {status, reason}]->(:ContractPoint) (consumer: recognizer output, raw)
(consumer)-[:VIA]->(:Link {linkId, kind, status, reason, matcher, candidateCount, versionSkew,
                           verdict, verdictBy, verdictAt, verdictWhy, priorVerdict, priorVerdictExpiredBecause})
          -[:TO]->(provider)
(:Link)-[:ON]->(:ContractPoint)
(:Link)-[:BOUND_BY {ord:int}]->(:ConfigEntry {entryId, repoId, path, line, keyPath, valueHash})  wiring evidence

-- data (FR-3): entities are ContractPoints of kind data_entity
(:CallSite)-[:READS|WRITES {status, reason, linkId}]->(:ContractPoint {kind:'data_entity'})
(:ContractPoint)-[:DEFINED_BY {ord:int}]->(:SchemaEntry {entryId, path, line, migration})

-- config / scripts as callers (FR-4, FR-5)
(:ConfigEntry)-[:LAUNCHES]->(:File|:Unit)

-- tests and coverage (FR-2 tests, FR-14)
(:Symbol:Test)-[:RAN {source:'runtime'}]->(:Symbol)            only when per-test runtime contexts exist (R11)
Symbol.{staticCovered, runtimeCovered, coverageState: 'both'|'static-only'|'runtime-only'|'neither'}

-- security marks and findings (FR-15)
(:CallSite|:Symbol)-[:MARKED {argIndex}]->(:Mark {markId})     'source:http-param', 'sink:sql', 'sanitizer:…', 'auth-check:…'
(:Unit)-[:DEPENDS_ON {version, manifestPath, line}]->(:Dependency {depId, ecosystem, package})
(:Finding {findingId, kind, status, reason, level})-[:STEP {ord:int, status, reason}]->(hop node)

-- failure-evidence anchors (FR-12, FR-17)
(:CallSite)-[:EMITS]->(:LogTemplate {templateId, textNorm, level})
```

Why each non-obvious choice:

- **`Name` indirection for name-matched calls.** Materializing one `CALLS {status:'possible'}`
  edge per (call site, same-named candidate) is quadratic in the commonest names: measured 6.8×
  the edge count on `cpg_falkorchat`, 44,688 edges from `__init__` alone (V13). Routed through one
  `Name` node it is linear (sites + defs), and the traversal is still two hops. `Name` nodes for
  very common names become dense rows; in GraphBLAS that costs memory linear in the degree, not a
  pointer-chase. The operator still lists every candidate (FR-8), it just doesn't store the
  product. `nameKey` is language-scoped (`py:get` ≠ `java:get`), because cross-language name matches
  are not calls; cross-language reach goes only through `Link`.
- **Contract kind as a property, not a label.** Cypher cannot parameterize a label, so a label per
  kind would put kind names into operator query text, and a new protocol pack would need operator
  edits, which breaks FR-26. A property plus a range index costs nothing at the anchor.
- **Reify only cross-boundary links.** A `Link` node is one node + three edges instead of one
  edge, but links are a tiny fraction of the graph, and they need what a bare edge cannot carry
  well: a candidate set ("may call any of N endpoints"), an ordered list of wiring evidence
  (`BOUND_BY`), and a verdict addressable by a stable id. Intra-program `CALLS` (the bulk of the
  edges) stay plain edges.
- **Verdicts on non-reified possible links.** A `READS`/`WRITES` edge carries `linkId` (a
  relationship index makes it addressable, quirks file). A name-matched call pair has no edge to
  hang a verdict on, so a verdict materializes one: confirmed → a direct `CALLS {reason:
  'verdict-confirmed'}`; dismissed → a `DISMISSED` edge the impact operator anti-joins
  (`OPTIONAL MATCH (cs)-[d:DISMISSED]->(s) … WHERE d IS NULL`, the pattern the quirks file
  prescribes, never `exists()`).
- **Denormalized `repoId`/`unitId`/`path` on `Symbol` and `CallSite`.** Staleness (FR-7a),
  redaction (FR-24) and evidence rendering (FR-20) then need no extra join per hop. The cost is a
  few short strings per node (inside V11's 115 MB per 1k files).
- **Secrets are never stored.** `ConfigEntry.valueHash`, not the value. A secret finding cites
  path + line, which is enough evidence and keeps secrets out of RAM, RDB files and the
  `telemetry{…}` stream (V10).

### 3.3 Status, reason and evidence (FR-8, FR-20)

- `status ∈ {'confirmed','possible'}` on every dependency edge and `Link`; `reason` from the
  proposal's closed, extensible code list. Edge types that are confirmed by construction
  (`HAS_CALL`, `DEFINES`, `RESOLVES_TO`, `CALLS` with `resolved-call`) carry a constant status;
  `CALLS_NAME` is possible by construction.
- **An item's status is the weakest hop on its best path.** The best path is the one minimizing
  (number of possible hops, then length), with ties broken by the hop nodes' stable ids, so the
  chain is deterministic (FR-19). As a traversal weight: confirmed = 1, possible = 1,000.
- **Evidence chain** = the ordered node list of that path, each node rendered as
  `{kind, path, line, unit, repo}` and each edge as `{type, status, reason}`. `Link` hops expand
  to consumer site → `BOUND_BY` entries in order → provider, so wiring files appear on the chain
  (AC-2).
- **Verdict effects** at render time: `Link.verdict = 'dismissed'` hides the hop's items unless
  "show dismissed"; `'confirmed'` upgrades the hop to confirmed with `reason = 'verdict-confirmed'`;
  `priorVerdict*` fields carry FR-29's "earlier dismissed by X at T for 'why'; expired because
  <end> changed".

### 3.4 Indexes and constraints

Every **constrained** id is a 32-hex-character hash of its canonical tuple (e.g.
`symbolId = h(lang, unitId, qname, signature)`, `linkId = h(kind, keyNorm, providerId,
consumerId)`, `callSiteId = h(enclosing symbolId, calleeName, ordinal of that calleeName in the
symbol)` so it survives line moves). This is mandatory, not stylistic: on this build a `UNIQUE`
property crashes the **whole instance** on a value over 4,096 bytes or one with leading/trailing
whitespace (quirks file), and URLs, SQL text and topic names from an arbitrary codebase are exactly
that kind of value. Human-readable keys go in unconstrained properties, stripped at write and at
lookup (an index silently misses an unstripped value, same entry).

```
-- 1. on the empty fresh key, BEFORE any load (edge loading MATCHes endpoints by these)
CREATE INDEX FOR (n:Repo)          ON (n.repoId)
CREATE INDEX FOR (n:Unit)          ON (n.unitId)
CREATE INDEX FOR (n:File)          ON (n.fileId)
CREATE INDEX FOR (n:File)          ON (n.path)          -- anchor: file + line
CREATE INDEX FOR (n:Symbol)        ON (n.symbolId)
CREATE INDEX FOR (n:Symbol)        ON (n.nameNorm)      -- anchor: symbol name (normalized client-side)
CREATE INDEX FOR (n:Symbol)        ON (n.qname)         -- anchor: qualified name, stack frames
CREATE INDEX FOR (n:CallSite)      ON (n.callSiteId)
CREATE INDEX FOR (n:Name)          ON (n.nameKey)
CREATE INDEX FOR (n:ContractPoint) ON (n.cpId)
CREATE INDEX FOR (n:ContractPoint) ON (n.keyNorm)       -- anchor: route / topic / table / config key
CREATE INDEX FOR (n:Link)          ON (n.linkId)
CREATE INDEX FOR (n:ConfigEntry)   ON (n.entryId)
CREATE INDEX FOR (n:SchemaEntry)   ON (n.entryId)
CREATE INDEX FOR (n:Mark)          ON (n.markId)
CREATE INDEX FOR (n:Dependency)    ON (n.depId)
CREATE INDEX FOR (n:Finding)       ON (n.findingId)
CREATE INDEX FOR (n:LogTemplate)   ON (n.templateId)
CREATE INDEX FOR (n:LogTemplate)   ON (n.textNorm)      -- exact tier of failure-evidence matching
CREATE INDEX FOR ()-[r:READS]-()     ON (r.linkId)
CREATE INDEX FOR ()-[r:WRITES]-()    ON (r.linkId)
CREATE INDEX FOR ()-[r:DISMISSED]-() ON (r.linkId)

-- 2. AFTER the load: keyword tier of failure-evidence matching; poll db.indexes() until OPERATIONAL (V9)
CALL db.idx.fulltext.createNodeIndex('LogTemplate', 'textNorm')

-- 3. AFTER the load, as the integrity gate: a duplicate makes the constraint FAILED (V7)
GRAPH.CONSTRAINT CREATE <key> UNIQUE NODE Symbol        PROPERTIES 1 symbolId
GRAPH.CONSTRAINT CREATE <key> UNIQUE NODE CallSite      PROPERTIES 1 callSiteId
GRAPH.CONSTRAINT CREATE <key> UNIQUE NODE ContractPoint PROPERTIES 1 cpId
GRAPH.CONSTRAINT CREATE <key> UNIQUE NODE Link          PROPERTIES 1 linkId
GRAPH.CONSTRAINT CREATE <key> UNIQUE NODE Name          PROPERTIES 1 nameKey
GRAPH.CONSTRAINT CREATE <key> UNIQUE NODE Unit          PROPERTIES 1 unitId
GRAPH.CONSTRAINT CREATE <key> UNIQUE NODE File          PROPERTIES 1 fileId
-- poll CALL db.constraints(): all OPERATIONAL → swap; any FAILED → the build fails, last good stays live
```

Notes on the index set:

- **File + line anchor:** `MATCH (f:File {path:$p})-[:DEFINES]->(s:Symbol) WHERE s.lineStart <= $l
  AND s.lineEnd >= $l`. The index anchors on the path; the line filter runs over the file's ~36
  symbols. No index on `lineStart`: a range index there gains nothing, and `<`/`<=` against an
  indexed property is a known silent-wrong-result trap on this build (verified for strings; not
  worth testing whether integers escape it).
- **No prefix anchors in Cypher.** `STARTS WITH` does not use an index here (quirks file). Route
  prefixes and path templates are matched by the linker at build time, and a user-supplied concrete
  URL is matched client-side against the system's `http` contract keys (thousands of rows at most)
  with the linker's own matcher. Only the resolved `keyNorm` reaches Cypher.
- **No `OR` anchors and no `$p IS NULL OR …` filters.** Both defeat the index on this build;
  operators branch into separate query strings (quirks file, Query tuning).
- **Relationship-index queries leave endpoint labels off** (`MATCH ()-[r:READS {linkId:$id}]->()`):
  a label on an endpoint forces a full label scan, and an undirected pattern with an indexed
  relationship-property predicate silently returns too few rows (both quirks file).
- **No vector index anywhere in `ccg_<system>`.** See Q9 below (the ML unit's M4 question).

### 3.5 Load sequence and swap (one build)

1. Fresh key `ccg_<system>__build_<snapshotId>`; create the step-1 indexes.
2. Load nodes, then edges grouped by (source label, type, target label), as `UNWIND $rows`
   batches over **one** persistent connection with retries disabled (`falkordb-py`'s default), a
   socket timeout of minutes, because writes ignore `TIMEOUT` and a client timeout abandons the
   answer but not the work (quirks file). Optimization when needed: `GRAPH.BULK` (present on this
   build, V14) via the FalkorDB bulk loader, which suits a fresh key.
3. Step-2 full-text index; step-3 constraints; poll both to `OPERATIONAL`.
4. Write the `:Snapshot` node last (its presence means "complete").
5. `RENAME ccg_<system>__build_<snapshotId> ccg_<system>` (V2). Old snapshot freed when its last
   in-flight reader finishes.


## 4. Findings on the proposal

**What I agree with, unchanged:** one summary-level snapshot graph per system (§3.5), confirmed by
V1; facts computed at build time and queries that only traverse (§3.1 rule 1); deep
interprocedural analysis in Joern, not Cypher (§3.5 deviation), which the intraprocedural
`REACHING_DEF` in `skills/joern-cpg/references/cpg-model.md` forces anyway; full per-unit CPGs
optional and off the answer path; build-into-fresh-key-then-swap; overlays as their own graphs;
LLM-suggested links only at build time as *possible* (§3.13); paging past the silent row cap
(§3.6 step 2).

Severity: **High** = breaks a requirement or can take the instance down as designed;
**Medium** = works only with a design change the architect should fold in; **Low** = a
refinement or a correction to a stated figure.

| # | Sev. | Proposal § | Finding | Change recommended |
|---|---|---|---|---|
| G1 | **High** | §3.6 (Power users), §3.12 (`cypher-mcp` reused unchanged), §5.3 (AC-21) | `cypher-mcp` is **not read-only**: a write rejected by `RO_QUERY` that matches one of its six kaizen shapes is re-run through `GRAPH.QUERY` against **any** graph name (V6). AC-21 could be broken by a producer-write shape aimed at `ccg_<system>`. `GRAPH.PROFILE` is a second write path for any client that exposes it. | FR-25 enforced by the **engine credential**: the raw client connects as a per-system ACL user with only `graph.ro_query`/`graph.explain` (+ `info`, §6.4) on `~ccg_<system>` (Q10, V5). Either `cypher-mcp` gains configurable connection credentials (an addition, consistent with "a future access gate would be an addition") or the `ccg` server ships its own raw tool. AC-21 is tested through that credential, not just through the tool. |
| G2 | **High** | §3.3 (stable symbol ID = "language + unit + qualified name + signature"), §3.4 (contract keys), §3.8 | Any human-readable id under a `UNIQUE` constraint can crash the **whole FalkorDB instance** on this build: a value over 4,096 bytes, or one with leading/trailing whitespace, SIGSEGVs `redis-server` (quirks file). Arbitrary codebases produce such strings: long URLs, SQL, generated qualified names, sloppy config values. On a shared instance that takes `falkor-chat` down with it. | **Every constrained id is a fixed-length hash** of its canonical tuple (§3.4); readable keys go in unconstrained, stripped properties. Normalize (strip) at write and at lookup. Pair with G8 (dedicated instance). |
| G3 | **High** | §3.3 (call sites with "callee targets each tagged resolved / name-matched"), §3.4 (tests: reach "computed at build time and stored as `COVERS`"), §3.5 (size estimate) | Two unbounded edge families. (a) One possible-call edge per (site, same-named candidate) is quadratic in common names: 6.8× the edges on `cpg_falkorchat`, 44,688 from `__init__` alone (V13), and worse in a 10k-file Java/C# system (`get`, `toString`, `execute`). (b) Static `COVERS` stored per (test, reached symbol) is a transitive closure, O(tests × reach), easily tens of millions of edges. Either one voids the "1–2 orders of magnitude smaller" claim and is the real path to K5. | (a) Route name matches through a language-scoped `Name` node (§3.2): linear, still two hops, every candidate still listed. (b) Store only direct facts; compute reach by traversal at query time (Q7). Keep per-symbol aggregates (`staticCovered`, `coverageState`) from one multi-source BFS at build. Per-test *runtime* `RAN` edges are direct facts and fine. State §3.1 rule 3 ("only direct facts stored") in the proposal. |
| G4 | Medium | §3.7 (security: "the engine stitches them across `LINKS`"), §3.6 (published query, R1) | Cross-component taint is a **stateful** walk: argument position *i* at a call site has to match parameter *i* of the handler, which flows to argument *j* of the next call. A fixed Cypher variable-length pattern cannot express that per-hop constraint, so this operator (like the BFS of Q7) is an **engine algorithm**, not "a fixed set of parameterized Cypher queries". That also means R1's "every answer publishes the exact query it ran" cannot be one Cypher text for these operators. | Compose cross-component taint **at build time** (the linker already has the summaries) and store each chain as a `Finding` with ordered `STEP` edges (§3.2). Security answers then just read findings, which is deterministic and readable by raw queries. For BFS operators, the "published query" is the per-level parameterized query **plus** the algorithm id/version, and AC-16's raw leg needs that reading (§5 Q-R4). |
| G5 | Medium | §3.5 (`ccg_<system>_verdicts` graph), §3.8 | A separate verdict graph cannot be joined in a query (Q1). The store with no traversal workload sits in the least durable place. And "application at link time" alone means a verdict shows only after the next refresh. A renamed symbol changes `linkId` and silently orphans its verdict. | File of record (append-only JSONL per system) applied at assembly. The `verdict` command also patches the live snapshot in one write, under its own ACL user. Fingerprint mismatch → expire, applied to confirmations too (R6a). Orphaned verdicts counted and listable (Q6). |
| G6 | Medium | §3.9 (overlay), §3.5 | Unspecified mechanics with cost traps: `GRAPH.COPY` + patch stalls the instance (V3) and reintroduces incremental re-linking. A delta-graph merge is federation (Q1). | Build each overlay from bundles into its own key, exactly like a snapshot. TTL with `EXPIRE` renewed on use (V4). A configured cap on concurrent overlays sized from Q3's peak formula. Overlay keys covered by the raw ACL pattern (Q5). |
| G7 | Medium | §3.5 (swap: "alias or rename mechanism is `graph-dba`'s call") | Settled: there is no alias. **`RENAME` over the live key** is atomic and reader-safe (V2). But it frees the old graph, and like `EXPIRE` it destroys data without the literal `GRAPH.DELETE` the lab's destructive-ops hook matches. | Swap by `RENAME`; keep zero old snapshots (rollback = re-assemble last-good bundles); key names only from the system manifest, prefix `ccg_`, never from caller input; the runner holds the only credential allowed to `RENAME`/`EXPIRE` (Q4, Q10). |
| G8 | Medium | §3.5, §6.1 K5 | The proposal does not say which FalkorDB instance hosts `ccg_*`. On the shared `falkordb-dev`, a G2-class crash, a copy stall or a peak of ~18 GB at the default two previews (Q3, §6.4) hits `falkor-chat` and every other component. Snapshots and overlays are rebuildable, so they need no persistence. Verdicts would (G5 moves them to a file). | A **dedicated FalkorDB instance** for the code graph, sized per Q3, `noeviction`, persistence off or infrequent RDB (a restart = re-assemble from bundles, minutes). Topology is mine to design; bring-up is `devops`'s. |
| G9 | Low | §3.11 (refresh), §3.5 | Build completion is underspecified. Full-text indexes backfill asynchronously at scale and silently return zero rows meanwhile (V9); constraints are async too. | Swap only when every index and constraint reads `OPERATIONAL`; a `FAILED` constraint fails the build (duplicates, V7) and the last good stays live (§3.5 load sequence). |
| G10 | Low | §3.13 (federation fallback), §6.1 K5 | The fallback trigger is framed as "summary graph too big". Measured, a 10k-file system is ~1.5 GB (Q3), and the realistic blow-up is G3's edge families, not file count. | Reword K5 and §3.13: federation is warranted only past ~⅓ of a shard per snapshot (~150k files on 64 GB). The live risk is unbounded edge families, controlled by model rules. |
| G11 | Low | §3.6 step 2, §6.2 Q7 | "Bounded variable-length reverse reach … path-returning queries" would enumerate every path (exponential). | Level-synchronous BFS in the engine with best-predecessor chains; `algo.SSpaths` benchmarked in P0 as an optimization only (Q7, V8). |
| G12 | Low | §1 (Scale; "~2,700 nodes / ~18,000 edges per Python source file") | The current `cpg_falkorchat` build measures ~4.0k nodes / ~27k edges / ~2.6 MB per file (V12). Test-heavy code runs ~1.5× the skill's rule of thumb. | Use the measured figure in §1's proving-repo estimate: ~210 Python files → ~0.85M nodes / ~5.7M edges / ~550 MB as full CPGs. |
| G13 | Low | §3.11 (FR-9 egress), §3.10 | FalkorDB keeps a per-graph `telemetry{<graph>}` stream of recent queries **with parameter values** (V10), in RAM and in RDB snapshots. It is not egress, so FR-9 is unaffected, but parameters that carry code text (a stack trace, a diff hunk) persist there, readable by any credential with access to that key. | Keep code text out of query parameters where an anchor id will do. Exclude `telemetry{…}` keys from every non-admin ACL pattern. P0 checks whether the stream can be disabled or capped on this build. |


## 5. Requirement-level findings for the stakeholder

_v2: Q-R1–Q-R6 were decided as S6, S23, S24, S1, S33 and S7 (requirements decision log, 2026-10-26); new questions are in §6.5._

None of these edits `docs/requirements/connected-code-graph.md`. Each is a question or a
clarification for the AC review, routed through `tico`.

- **Q-R1 — FR-24: does "no code shown" also protect the *existence and shape* of hidden code?**
  The engine already reveals existence: an impact in an unreadable repo is shown as "affected, no
  access". If FR-24 protects only **content** (paths, names, lines, source), a per-scope raw view
  that keeps the topology and scrubs those properties satisfies it for raw queries too (Q10,
  R3(b) "scrub" variant), at one extra snapshot of RAM per distinct scope. If FR-24 also protects
  **structure** (counts, which repos call which), raw queries can only be offered under R3(a). That
  is my recommendation for the first release either way, because it is free and enforceable by one
  credential. *Asks for:* one sentence in FR-24 saying which is meant.
- **Q-R2 — FR-28: when does a verdict take effect?** "Shown in later answers" can be read as
  "after the next refresh" or "immediately". Immediately is cheap: the `verdict` command patches
  the live snapshot as well as the store (Q6). *Recommend:* "immediately, and it survives
  refreshes".
- **Q-R3 — FR-29 / R6: a verdict on code that is renamed or moved.** A verdict is tied to the two
  ends' identities. A rename changes an identity, so the earlier verdict can't be matched to the
  "same" link any more. *Recommend accepting:* such a link returns as *possible* **without** the
  "earlier dismissed" note, and the orphaned verdict stays listable ("verdicts whose code no longer
  exists"). Otherwise the requirement implies rename tracking across refreshes, which is
  heuristic and would itself need a *possible* status.
- **Q-R4 — AC-16's raw-query leg (with proposal R1).** Several operators (impact reach, RCA
  candidate generation) are deterministic **algorithms** that issue one query per traversal level.
  They are not a single Cypher statement (G4, Q7). "Running the published query returns the same
  facts" is then true only if the published artifact is the query set **plus** the algorithm and
  its version. *Recommend:* AC-16's raw leg reads "re-running the answer's published query set
  through the raw client reproduces its facts", and security findings, being materialized at build
  time, are directly queryable.
- **Q-R5 — FR-23: how many overlays at once?** Each overlay is a full copy of the system's
  snapshot in memory (~1.5 GB for a 10,000-file system, Q3 and Q5). Unlimited concurrent overlays
  (for example one per in-flight `teco` change) multiply that. *Recommend:* the requirement accepts a
  configurable cap, with the oldest idle overlay retired first, and an answer
  stating when its overlay had to be rebuilt. *(Decided: cap configurable, default 2 — FR-23, S33/Q-R8.)*
- **Q-R6 — FR-25 / AC-21: say where read-only is guaranteed.** As written, AC-21 is satisfied by
  whichever tool is tested. *Recommend:* "rejected by the database for the raw-query credential",
  so the guarantee survives a client bug (G1).
- **Context, no decision needed — deployment footprint.** At workplace scale the connected view
  wants its own FalkorDB instance with on the order of 10 GB of RAM (one live snapshot, one being
  rebuilt, a few overlays). The full statement-level CPGs the team uses today would need ~26 GB for
  the snapshot alone, which is why the proposal's summary-level graph is the right call.

## 6. Revision 2 — analyst B1/B2 and the re-checked requirements

Inputs: `docs/reviews/connected-code-graph.md` (B1, B2, and the graph-side minors m1–m3 and m6,
plus B6), and the stakeholder's decisions S1–S38 and use-case re-check in
`docs/requirements/connected-code-graph.md` (decision log, 2026-10-26). Where this section and
§2–§3 disagree, this section wins. AC-12's "stack trace and log together rank at least as high as
either alone" is treated as an unconfirmed assumption and drives no graph design here.

New live evidence for this revision (2026-10-26). `cpg_falkorchat` was read only; the ACL and
timeout probes ran in throwaway `v4.18.11` containers on ports 6392/6393, stopped after:

| # | Probe | Result |
|---|---|---|
| V15 | Statement-level flow layers in `cpg_falkorchat` (97 files) | `REACHING_DEF` 655,140 (~6,750/file), `CDG` 51,866 (~535/file), `CFG` 296,217, `ARGUMENT` 207,970. Condition nodes (distinct `CDG` sources) 3,219 (~33/file); `METHOD_PARAMETER_IN` 12,714 (~131/file); `RETURN` 3,024 (~31/file); `<operator>.fieldAccess` 16,898 (~174/file). Only 1,549 `REACHING_DEF` edges run directly param/call → call. The rest pass through identifiers and operators, so a summary needs a per-method **projection**, not a copy. |
| V16 | `GRAPH.LIST` under a key-scoped ACL user | Lists **every** graph key on the instance (`ccg_a`, `other_sys`). Key patterns don't filter keyless commands. Confirms review m6. |
| V17 | `falkordb-py` 1.6.x connecting as a restricted user | `FalkorDB(...)` calls `INFO server` (sentinel detection) in its constructor and fails with `NoPermissionError … 'info'` unless the user has `+info`. |
| V18 | `GRAPH.CONFIG SET TIMEOUT_MAX 500`, then raw-user reads | A query asking for `timeout=60000` is **rejected** (*"cannot exceed the TIMEOUT_MAX"*), and one with no timeout is killed at 0.5 s. An instance-wide cap works for every credential. |

### 6.1 B1 — fix

**Agreed: B1 is real.** With one stored direction and "impact = incoming reach", a writer's change
never reaches the readers of what it writes, and RCA never reaches the writer. The same flaw
exists, unnoticed, on request payloads across a `Link`: the consumer's argument value flows **into**
the provider, which is opposite to the contract dependence the `VIA`/`TO` edges encode.

**Why flipping or mirroring edges is not enough.** A shared-state edge plays two roles in opposite
directions:
- **Contract:** reader *and* writer both depend on the entity's schema. Renaming a label (AC-3)
  must list both.
- **Value:** the reader depends on the writer.

Store `WRITES` as entity → writer and AC-3's contract walk loses the writers. Add a mirrored
`WRITTEN_BY` and a plain incoming walk from writer W reaches the entity, then **every other
writer** W2. That is a false impact (W's value change does not affect W2), and it grows with every
hot table or topic. Either way a rule about *how you arrived* at the entity is needed. Once that
rule exists, mirror edges buy nothing but RAM.

**Fix (graph + engine):**
1. **Storage is unchanged.** Every edge stays stored dependent → dependency, as in §3.2, which
   reads as the contract relation.
2. **Traversal direction is per (operator, edge type, arrival state)**, from the normative table in
   §6.2. It replaces §3.1 rule 1 as a traversal rule.
3. **The BFS (Q7) carries a small state per frontier entry:** `value`, `contract`, or `arg(i, path)`
   (a value entering a callee through argument *i*, field path *path*). Each level issues one
   parameterized read per (edge type, direction) the table allows from that state. The state
   machine is part of the algorithm whose name and version the answer publishes in its recipe
   (FR-19, S1).
4. **Shared state is one pattern** covering data entities, message topics, class fields, env
   vars and config keys. Writer `−WRITES→` CP `←READS−` reader. A publisher `WRITES` a topic CP and
   a subscriber `READS` it. Config `WRITES` an `env_var`/`config_key` CP and code `READS` it. B1's
   fix therefore covers messaging and config-value causes (FR-12) with no extra edge type.
5. **Request direction on links:** a consumer call site reached in state `arg(i, …)`
   (its request argument changed) continues `VIA →`, `TO →`, then `HAS_PARAM →` to the handler
   parameter the pack maps that argument to (`Link.paramMap`). This is the value path. The contract
   path (provider changed → consumers) stays `TO ←`, `VIA ←`.
6. **Name hop (review m1):** `CALLS_NAME` + `RESOLVES_TO` are expanded as **one compound hop** in a
   single query that also anti-joins `DISMISSED`. `Name` is never a BFS node or in the visited set.
   **Resolved call sites also carry `CALLS_NAME`**, at one extra edge per resolved site. The
   operator suppresses the candidate equal to the resolved target and lists the others as
   *possible* (`reason: resolution-unverified`) unless P0 measures Joern's resolved-call precision
   above a threshold set by `data-scientist`. Otherwise a wrong Joern resolution is a silent miss
   plus a false *confirmed*, the worse error under FR-8.

**Fixture assertions** (P0/P1 test set, the analyst's two plus two that guard the fix):
- (1) changing writer W lists reader R in another unit (impact, value);
- (2) RCA from R's symptom lists W (backward, value);
- (3) renaming the entity lists R **and** W (contract);
- (4) changing W does **not** list another writer W2 of the same entity unless W2 also reads it.

### 6.2 Normative edge-type × direction table

`←` = traverse against the stored direction (from dependency to dependent), `→` = along it, `—` =
not traversed. "Impact" answers FR-11's forward question. "Depends-on / RCA" answers FR-11's
reverse question and RCA's backward tracing (FR-12): the same walk, with RCA adding ranking. A
contract-mode impact starts at a `ContractPoint` anchor or at a change that alters a provider key,
schema entry or library signature (FR-13). After the first shared-state hop it continues in
`value`.

| Edge (stored dependent → dependency) | Impact, `value` | Impact, `contract` (from a CP) | Depends-on / RCA backward |
|---|---|---|---|
| `CALLS` CS → Symbol | `←` (callee changed → call site). `→` only in `arg(i)` state, into the callee | — | `→` (call site → callee) |
| `CALLS_NAME`+`RESOLVES_TO` (compound) | as `CALLS`, *possible*, `DISMISSED` anti-joined | — | as `CALLS` |
| `HAS_CALL` / `HAS_PARAM` / `HAS_RETURN` / `HAS_COND` Symbol → point | `←` (point → enclosing function, which becomes a reported item). `HAS_PARAM →` only in `arg(i)` state, index-matched | — | `→`. From a `Param`, `←` then `CALLS ←` to callers' argument `i` |
| `DATA {fromPath, toPath}` use → def | `←` (path-matched when the state carries a path) | — | `→` |
| `CONTROLLED_BY` point → `Cond` | `←` (changed condition → guarded points) | — | `→` (which condition chose the branch) |
| `VIA` consumer → `Link`, `TO` `Link` → provider | `←` (provider changed → consumers). `→` only in `arg(i)` state (request value → handler parameter via `paramMap`) | `←` from the provider's CP | `→` (bad response). From a handler `Param`: `←` to the consumer's arguments |
| `READS` CS → CP | `←` only at a CP reached by `WRITES →` | `←` | `→` (reader → CP, state `value`) |
| `WRITES` CS/`ConfigEntry` → CP | `→` (writer → CP), never `←` | `←` (all writers) | `←` only at a CP reached by `READS →` (who wrote the value) |
| `DEFINED_BY` CP → `SchemaEntry` | — | `←` (schema/migration change → CP → `READS`/`WRITES ←`) | `→` (schema-mismatch hypothesis) |
| `CONSUMES` CS/`ConfigEntry` → CP | — | `←` (every consumer of the old contract, matched or not) | `→` (a consumer whose CP has no provider: wiring-error hypothesis) |
| `PROVIDES` provider → CP | `→` when the anchor is a provider whose key changed (switches to `contract`) | — | — |
| `BOUND_BY` `Link` → `ConfigEntry` | `←` (config change → the links it binds → their consumers) | — | `→` (wiring hypothesis: wrong port/URL) |
| `LAUNCHES` `ConfigEntry` → `File`/`Unit` | `←` (launched code changed → the entries that reference it, FR-13) | — | `→` |
| `RAN` `Test` → Symbol | `←` ("which tests reach this change", runtime half) | — | — |
| `DEPENDS_ON` `Unit` → `Dependency`, `FROM_DEP` external Symbol → `Dependency` | `←` (dependency bump → units) | — | `→` ("cause outside the mapped code" branch) |

Containment edges (`IN_REPO`, `HAS_FILE`, `DEFINES`) are never traversed for reach. They render
grouping and evidence only.

**Distance and direct/indirect (FR-11, AC-25).** Distance counts **inter-function links** (call,
link, shared-state, config hops). Intra-function hops (`DATA`, `CONTROLLED_BY`, `HAS_*`) count 0,
otherwise a three-line function would look "farther" than a cross-service call. Direct = distance
1. This is a definition, not a requirement, so it goes to the stakeholder (§6.5 Q-R7). The best path
per item minimizes (possible hops, distance, length), tie-broken by stable ids, and stays
deterministic.

### 6.3 B2 — graph side

**Agreed: positional `ARG_FLOWS {param, arg}` and `paramToReturn` are too weak.** AC-1's own
wrapper needs field paths through two functions: `apiFetch(opts)` → `buildUrl(opts.path, …)` →
`fetch(url, {method: opts.method})` (`salesperson/src/api/client.ts:73,107-118`). The v2 schema
replaces both with **points and access-path data edges**:

```
(:Symbol)-[:HAS_PARAM]->(:Param {pointId, index, name, line})
(:Symbol)-[:HAS_RETURN]->(:Ret {pointId, line})
(:Symbol)-[:HAS_COND]->(:Cond {pointId, line, code})          code truncated to 200 chars, unconstrained
(:CallSite) is itself a point; its result is fromPath 'r', its argument i is toPath 'a<i>'
(point)-[:DATA {fromPath, toPath}]->(point)                   use → def, intra-function
    fromPath ∈ 'p<i>[.f1.f2]' | 'r[.f…]' | ''   toPath ∈ 'a<i>[.f…]' | 'ret[.f…]' | ''
(point)-[:CONTROLLED_BY]->(:Cond)
(point)-[:READS|WRITES {status, reason, linkId}]->(:ContractPoint {kind:'field', key:'<type>.<field>'})
```

AC-1's chain in this schema:
- `fetch` call site `−DATA{toPath:'a0', fromPath:'r'}→` the `buildUrl` call site;
- the `buildUrl` call site `−DATA{toPath:'a0', fromPath:'p0.path'}→` `apiFetch`'s `Param p0`;
- the `fetch` call site `−DATA{toPath:'a1.method', fromPath:'p0.method'}→` `Param p0`;
- inside `buildUrl`, `Ret −DATA{fromPath:'p0'}→ Param p0`.

**Field paths are capped at depth 3.** A longer path is truncated to its prefix and marked
*possible* (`reason: field-truncated`). Matching paths across a hop is an engine-side BFS state
(`arg(i, path)`, §6.1), the same stateful-walk argument as G4. Cypher cannot carry it in a
variable-length pattern.

**Role lifting happens at build time; the graph stores the result and its proof.** The linker
lifts roles over bundles: `fetch` is an HTTP client on `a0`, so `buildUrl`'s `p0` is one, and so is
`apiFetch`'s `p0.path`. The graph receives the outcome as `CONSUMES` from the `endpoints.ts` call
sites, with `reason: lifted-role`, plus the lifting chain as ordered evidence on the `Link`:
`(:Link)-[:LIFTED_VIA {ord}]->(:CallSite|:Symbol)`. That keeps FR-20's file + line at every step
(endpoints.ts call → `apiFetch` → `buildUrl` → `fetch`) without a stateful walk at answer time.

**Field-sensitive classes, without a heap model:** a class field is a shared-state CP of kind
`field`, keyed by declaring type plus field name, with `READS`/`WRITES` from the accessing points.
That is how `self.repo = …` in `__init__` reaches `self.repo.query(…)` in another method. Aliasing
is not modelled, so field links are *possible* (`reason: field-name-match`) unless the receiver
type is resolved.

**Shell and prompts get the same schema (review m4):**
- The `tree-sitter-bash` extractor emits a `Symbol` per shell function, plus one synthetic
  per-file `Symbol {kind:'script', name:'<script>'}` that spans the file.
- A `CallSite` per command invocation, with `calleeName` = the command or function.
- `Param`s for `$1…$n` / `"$@"`, and `DATA` for assignment → use inside a function. `eval`,
  indirect expansion and `source` of a computed path are *possible*, `reason: shell-dynamic`.
- So `gquery() { redis-cli … GRAPH.QUERY "$GRAPH" "$1"; }` yields
  `DATA{toPath:'a4', fromPath:'p0'}`. Lifting then carries the Cypher literal from each of
  `bootstrap_schema.sh`'s 50 `gquery` calls into the embedded-Cypher parser, which emits
  `WRITES` to the label CPs (AC-3).
- Markdown prompts get a synthetic `Symbol {kind:'document'}` per file and a `CallSite` per tool
  mention (`reason: prompt-mention`). Example configs carry `File.role = 'example'`
  (`reason: example-config`, FR-5/S21).
- Runtime-built labels (`falkor-chat/server/falkorchat/querygen.py:488`) produce *possible*
  `READS`/`WRITES` to the bound store's label CPs (`reason: runtime-built`), collapsed in rendering
  as the proposal already describes.

**What this does to B2's cost concern.** Impact and RCA no longer need build-time whole-program
`reachableBy`. They need a **per-method projection** of Joern's intraprocedural DDG/CDG onto the
method's points: which points each point's value derives from, and which conditions guard it.
That is a local graph search per method. Interprocedural composition then happens at answer time
by traversal. Whole-program `reachableBy` stays only for security taint, materialized as `Finding`
chains (G4). The analyst's P0 vertical-slice recommendation stands, and the projection's build
cost is one of the things it must time.

### 6.4 Per requirement: schema, constraint and index changes, and feasibility

| Requirement (post-S1–S38) | Graph change | Index / constraint | Feasibility on FalkorDB |
|---|---|---|---|
| **FR-1, FR-4, FR-5** shell, structured config, prompts, example configs | Synthetic `script`/`document` Symbols, shell `CallSite`/`Param`/`DATA` (§6.3); `ConfigEntry` per structured key; `File.role ∈ {code, example, doc, config, script}` | none new | none |
| **FR-7a** never-built part | `Unit.buildStatus = 'not-mapped'`; a link into it is `(:Link {status:'possible', reason:'target-not-mapped'})-[:TO_UNIT]->(:Unit)` | none | none |
| **FR-11** data flow, control dependence, direct/indirect, distance, grouping, reverse question | `Param`/`Ret`/`Cond` points, `DATA {fromPath,toPath}`, `CONTROLLED_BY`, field CPs (§6.3); `Unit.componentId` for grouping; distance computed by the BFS (§6.2) | `Param/Ret/Cond.pointId` range index (edge-load anchors); UNIQUE after load | **Size, below.** Traversal is the BFS with state. Paging per level keeps every reply under `RESULTSET_SIZE` |
| **FR-12** config/schema/wiring causes, backward flow, recent changes, outside-the-code branch, no-match | Shared-state pattern for config values (§6.1 item 4); column → table `(:ContractPoint)-[:PART_OF]->(:ContractPoint)`; `Symbol.lastChangedAt`/`lastChangedCommit` from `git blame` at build (window per codebase applied at query time); external Symbols `-[:FROM_DEP]->(:Dependency)`; `LogTemplate` unchanged. Ruled-out/confirmed marks are per investigation and engine-side only (nothing persists) | none new | none |
| **FR-13** five contract kinds, contract delta, test/security/config impact, "show existing" | `ContractPoint.kind ∈ {http, rpc, topic, data_entity, mcp_tool, library_api, …}`. `library_api` key = qualified name + signature hash, so a signature change is "old removed + new added", paired by qname. **The delta is computed at overlay build** (the linker holds base and overlay bundles), because base and overlay are separate keys and cannot be joined (V1): `ContractPoint.overlayState ∈ {unchanged, added, removed, changed}`, removed CPs kept **with their consumers' `CONSUMES`**, `Finding.delta ∈ {new, worsened, existing}` | `ContractPoint.overlayState` range index (review listing anchor) | none, provided the delta is materialized; a query-time base↔overlay join is impossible |
| **FR-14** per-test runtime, function level, test kind, tests-for-a-change, risk ranking, stale data | `Symbol:Test.testKind ∈ {unit, integration, e2e}`; `(:CoverageRun {runId, commit, collectedAt, tool, stale})`; `(:Test)-[:RAN {runId}]->(:Symbol)`. A stale run (commit ≠ unit's built commit) is loaded but never merged and is listed. Risk inputs as properties computed at build: `callerCount`, `securityRelevant`, `lastChangedAt` | `CoverageRun.runId` | **Size risk:** `RAN` edges = tests × functions each test runs. ~0.6M edges at 2k tests × 300 functions; ~10M (≈1 GB) at 20k tests × 500. Direct facts, so linear, but they are the next-biggest family after `DATA` |
| **FR-15** risky config, five kinds, severity, git-history secrets, finding verdicts, values never stored | `Finding {kind incl. 'risky-config', severity, severityReason, fingerprint, verdict*}` with `STEP`s to `ConfigEntry` for config findings; `(:SecretSighting {sightingId, fingerprint, path, line, commit, committedAt, removedInCommit, presentAtHead})`. **`fingerprint` must be a keyed hash (HMAC with a per-system key kept outside the graph), never a plain hash**: secrets are often low-entropy or of a known format, so a plain SHA-256 can be brute-forced back to the value, which breaks "never stores a secret's value" | `Finding.findingId` and `SecretSighting.sightingId` UNIQUE (hashes) | none |
| **FR-19** decision-state stamp, recipe, fixed RCA order | `:Snapshot {verdictLogOffset, verdictLogHash}`, advanced **in the same write query** as each verdict patch (one query is atomic, quirks file). The stamp = `snapshotId` + offset. The recipe (queries + algorithm name/version) is engine-side | none | none |
| **FR-24** content-only protection, raw queries for all-repo readers only (S5, S6) | No per-scope views: Q3's `s = 0`, Q10's R3(b) is no longer needed. Redaction stays engine-side over denormalized `repoId` | none | none |
| **FR-25** enforced by the database credential (S7) | Raw ACL user per system: `-@all +graph.ro_query +graph.explain +info ~ccg_<system> ~ccg_<system>__ov_*`. **`+info` is required** because `falkordb-py`'s constructor calls `INFO` (V17). **`GRAPH.LIST` is not granted**: it reveals every key on the instance (V16), so the engine passes graph names. A runaway raw read is bounded by an **instance-wide `TIMEOUT_MAX`** (V18; Redis ACLs have no per-user timeout). The engine's per-level BFS queries fit under the same cap, a second reason the engine never issues unbounded path queries | — | verified (V5, V16–V18) |
| **FR-28/FR-29 + review B6, m2** immediate verdicts, either-side owner, expiry on change/rename/move/signature | Verdict fields on `Link`, `READS`/`WRITES` (by `linkId`) and `Finding`. **B6 fix (agree with the analyst, plus self-healing):** assembly stamps `verdictLogOffset` on the new key. Under a lock shared with the `verdict` command (a Redis `SET ccg_<system>:verdict_lock NX PX`), it replays log entries past that offset onto the new key, then `RENAME`s. The `verdict` command takes the same lock, appends and fsyncs the log, then patches the live key and advances its offset in one query. After a crash between append and patch, the next verdict or swap replays from the offset, so the store and graph re-converge. **m2:** `symbolId = h(lang, repoId, qname[, signature only for overloading languages: Java, Kotlin, C#, C++])`, with no `unitId`, so a moved build root or an added Python parameter doesn't orphan the id. S24 accepts orphaning on rename, move or signature change; orphans stay listable (Q6 item 4) | `READS`/`WRITES`/`DISMISSED` relationship indexes on `linkId` (§3.4) | none. The lock is the one cross-key coordination the design needs, and it lives outside FalkorDB's single-query atomicity |
| **FR-23** preview cap (S33) | Cap is runner config. Sizing below | — | **RAM**, below |
| **Review m3** θ / score floor | Agreed: a deliberate, frozen, recall-calibrated exception, applied **client-side over the complete full-text result set**, never as an index `LIMIT`. Q9's condition 1 stands for everything else | — | — |

**Summary-graph size with data flow and control dependence carried for all use cases** (the
stakeholder's Joern-CFG/DDG note). These are estimates extrapolated from V11 (synthetic, ~93 B
per element plus indexes) and V15 (measured densities). P0 must measure the real projection.

| Model, per source file | Nodes | Edges | RAM | 10k-file snapshot | Peak: 1 live + 1 building + 2 previews (default), +20% | Peak with 4 previews (alternative) |
|---|---|---|---|---|---|---|
| v1 summary (calls, links, data/config) | ~260 | ~580 | ~0.115 MB | ~1.2–1.5 GB | ~7 GB | ~11 GB |
| **v2 summary + projected points** (params, returns, conditions; `DATA` ≈ 3 per point, `CONTROLLED_BY`, field CPs) | **~455** | **~2,400** | **~0.3–0.35 MB** | **~3–4 GB** | **~14–19 GB (≈18 GB)** | ~25–30 GB |
| Statement-level DDG + CDG copied from Joern (`REACHING_DEF` + `CDG` plus their ~3k statement nodes) | ~3,300 | ~7,500 | ~1.2 MB (est.) | ~12 GB | ~58 GB | ~85 GB |
| Full CPG (V12) | ~4,000 | ~27,000 | ~2.6 MB | ~26 GB | not viable | not viable |

**Feasibility verdict:** carrying data flow and control dependence for all use cases is **feasible
on FalkorDB only as the projected point graph**. It roughly triples the summary graph and keeps
one system on one shard comfortably. Copying Joern's statement-level layers is not feasible at
workplace scale with previews. Remaining risks, all P0-measurable:
- (1) the projection's build time per method (B2);
- (2) the BFS state space under field paths, bounded by the depth-3 cap;
- (3) `RAN` volume (FR-14);
- (4) peak RAM with the decided default of two previews (~18 GB), on the dedicated instance that
  G8 already recommends (§6.5 Q-R8, decided).

### 6.5 Questions for the stakeholder

- **Q-R7 — Distance (FR-11, AC-25):** should "distance" count links **between** functions,
  components and data (calls, endpoints, tables, config), with steps inside one function counted as
  zero? *Recommend yes*, otherwise a long function looks farther away than a cross-service call.
- **Q-R8 — Memory for flow-aware impact plus previews (FR-11 × FR-23) — decided: default cap of
  2.** At about 10,000 source files, the view with data flow and control dependencies needs a
  dedicated machine with roughly **18 GB of RAM** at peak for one live view, one rebuild and two
  concurrent change previews. The cap stays configurable: four previews would need ~30 GB
  (a ~32 GB host). Confirm the figure after P0 measures it.
- **Q-R9 — Where there is no data/control flow (FR-11 × FR-1):** Joern has no shell or config
  frontend, and the lab's shell extractor will give only approximate data flow. Some Joern
  frontends may prove weak in P0. Should impact through such code fall back to call-level reach,
  with affected items shown as *possible* and the reason "data/control flow not available for
  <language>"? *Recommend yes.* That honours "never miss" without claiming precision the tool
  doesn't have.
- **Q-R10 — "Component" in FR-11's grouping:** is a component the same thing as a build unit (one
  build root in one language), or a larger grouping you name, such as a top-level folder or a
  service? *Recommend: the build unit by default, overridable by an optional grouping line in the
  system's repository list* (S30 already accepts that list as input).
- **Q-R11 — Secret fingerprints (FR-15):** to make a fingerprint unrecoverable, it must be a keyed
  hash with a per-system key. The consequence is that fingerprints cannot be compared with another
  tool's fingerprints or across systems. *Recommend accepting that.*

Items the review raised that need **no** stakeholder decision and are folded in above: B1 (§6.1–6.2),
B2 graph side (§6.3), B6 (§6.4 FR-28/29 row), m1 (§6.1 item 6), m2 and m3 (§6.4), m6 (V16–V18).
