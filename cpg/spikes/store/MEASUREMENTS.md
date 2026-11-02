# S20 / S21 measurements (inputs to S23's decisions table)

> Measured 2026-11-01/02 on `ccg-falkordb` (`falkordb/falkordb:v4.18.11`, module `41811`, 127.0.0.1:6390), keys
> `ccg_p0spike`, `ccg_p0spike__*` only. S23 owns the final table; this file holds the raw figures and the commands.
> Raw output: `cpg/spikes/store/results/` (`bench_reach.jsonl`, `bench_reach_summary.json`, `swap_probe_*.json`,
> `dep_index_probe.json`, `sspaths_pathcount_probe.json`, `sspaths_aggregated.json`, `synth_build_record.json`,
> `recheck_{digests,ping,timeout}.json`). Code: `synth.py`, `bfs.py`, `bench_reach.py`, `swap_probe.py`,
> `dep_index_probe.py`, `sspaths_agg_check.py`, `recheck_claims.py`; tests `tests/test_store_spikes.py`.
> A figure with no file under `results/` is marked "unrecorded". Gate follow-up: `docs/reviews/connected-code-graph-store-spikes.md`.
> **Hand-off to S23:** the six `falkordb-quirks.md` entries dated 2026-11-02 (+ one added in the gate follow-up) were folded in
> early; S23 must not re-add them. The "default read `TIMEOUT` is 1000 ms" entry restates the older "Default `TIMEOUT` is 1000ms"
> entry (~line 1143); folding the two is left to S23.

## 1. S20 -- BFS vs `algo.SSpaths`

**Graph.** `python spikes/store/synth.py --build --drop-stale` (from `cpg/`): 1,000 files, **453,905 nodes, 2,399,308
edges** (455 / 2,400 per file, graph note §6.4), `GRAPH.MEMORY USAGE` 330 MB (`relation_matrices` 116, `indices` 63),
instance `used_memory` +423,508,696 B over the build (`results/synth_build_record.json`, the build after the OOM below; the
first two builds' figures are unrecorded: 41 s, +423 MB). Loaded per D16: 160 batches, max batch 20,000 (`BATCH_MAX`),
zero-retry connection (the loader's `LiveStore`), indexes on the empty key, nodes then edges (created-count checked
per batch), constraints after, polled to `OPERATIONAL`, `RENAME`; `TIMEOUT_MAX` was 0 (the explicit-timeout branch is
unit-tested with 7,000, not run live). Build time in the record: indexes 0.0 s, nodes 3.7 s, edges 33.1 s, constraints 0.3 s. Call graph: layered DAG, top hub 3,804 callers
= 2.4 % of the 157,093 `CALLS` edges (slice: 2.3 %), 8 % of `CALLS` `possible`, 4 % back-edges. Contract/`Link` rows are
not generated, so `TO <-`/`VIA <-` return nothing here. Six planted cycles (a `cond -> cond -> cond -> cond`
`CONTROLLED_BY` ring, a two-point `DATA` cycle and a `DATA` self-edge in the first function that calls each of the
first six hubs); the full walk from each of those 6 hubs visited all 3 ring conditions and ended (7 levels).
`synth.py` was fixed after the benchmark to stop a planted row repeating a random one (2 of 2.4 M rows); the graph has been
rebuilt since (twice: once for that fix, once after the OOM below) and `results/recheck_digests.json` (every 5th depth-3
anchor, 10 anchors) shows all 10 BFS and all 10 `SSpaths` digests equal to the recorded ones.

**Which BFS.** `cpg/spikes/vertical-slice/impact.py` `walk()`, committed `cb1b2214`, SHA-256
`434801296c9e85d4ba9593af78cd70d609bb43e50a4871040f528d84b8282171`, with S17-L2's `function` state kind
(`bfs.impact_provenance()`; both run records carry the same hash). `bfs.py` wraps it, it does not re-implement it.
Two additions, both by wrapping `impact._expand` for one call: a **level cap**, and the **`CONTROLLED_BY <-` row**
(graph note §6.2; `impact.py` does not follow it, so without it a cond cycle is never walked) in `full` mode only.
`impact.py` raises `ImpactError` when a 1,000-id chunk returns >= 10,000 rows (it did on the first hub tried); `bfs.py`
halves the chunk and retries (graph note Q7's "re-split"): 5 splits at `full` levels 4 and 5 at levels 6 (counted from each
record's last repeat only), none needed for a single id.

**Segments.** `SSpaths` takes one `relDirection`, so it was benchmarked on **call-only reverse reach** (`CALLS` +
`HAS_CALL`, incoming; `levels = 2 * D` for `D` call hops; `weightProp w` 1 / 1000; `maxLen 2D`; `pathCount` 10,000).
The BFS ran the same segment through `impact.walk` with every other row switched off. `full` (every `bfs-v1` row +
`CONTROLLED_BY <-`) has no `SSpaths` counterpart.

**Run.** `python spikes/store/bench_reach.py run --out spikes/store/results/bench_reach.jsonl` then `summarize`.
50 seeded anchors (15 hubs with 293-3,804 callers, 20 mid with 4-20, 15 low with 1-2), 20 repeats per
(configuration, anchor), except hubs at `D = 5` and `full` levels 6 (5 repeats; stated per record). Records 1-62 ran
with the engine's default read timeout; the run crashed once (below) and was resumed with an explicit 60 s timeout on
every read, skipping finished records.
The `D = 5` pooled p95 weights the hub anchors at 5 repeats against 20 for the rest (775 samples, not 1,000); this does not
change the rule-3 call. Keys used are `ccg_p0spike*`, not the plan's reservation-table `ccg_spike_synth*` / `ccg_spike_swap*`
(`p0-2.md` §5.1): the table needs the rename or a ledger note.

| `D` | repeats identical, BFS | repeats identical, `SSpaths` | item sets equal | status / distance / chain agree on the common items | p50 / p95 BFS (s) | p50 / p95 `SSpaths` (s) |
|---|---|---|---|---|---|---|
| 1 | 50/50 | 50/50 | 50/50 | 10,859 / 10,859 each | 0.0012 / 0.033 | 0.0008 / 0.048 |
| 3 | 50/50 | 50/50 | **35/50** | 44,400/44,400; 44,356/44,400; 42,951/44,400 | 0.0048 / 3.70 | 0.0041 / 0.64 |
| 5 | 50/50 | 50/50 | **35/50** | 49,340/49,340; 49,296/49,340; 47,857/49,340 | 0.0078 / 6.29 | 0.0071 / 0.65 |

By class at `D = 3`: hubs 0/15 equal, `SSpaths` returned 2,756 items (median) against the BFS's 19,045 and its reply held
exactly 10,000 rows on 15/15; **85.2 % of the BFS's hub items missing**; mid 20/20 and low 15/15 equal (p95 BFS 0.013 s /
0.004 s, `SSpaths` 0.026 s / 0.002 s). `D = 5`: hubs 0/15 equal, 90.3 % missing, p95 BFS 6.94 s vs `SSpaths` 1.03 s; mid
p95 BFS 0.067 s vs `SSpaths` 0.177 s.

**What caps `SSpaths` (corrected in the gate follow-up).** `pathCount` is a **global, weight-ordered path budget**:
with the aggregation inside the query (one reply row, so `RESULTSET_SIZE` cannot bind), hub `S0689.14`, `maxLen 6`
(`results/sspaths_pathcount_probe.json`, `count(*)` = rows in the engine):

| `pathCount` | paths returned | distinct end nodes | distinct `Symbol` ends | path length min/max | seconds |
|---|---|---|---|---|---|
| 0 | 3,521 | 3,521 | 0 | 1/1 | 0.036 |
| 100 | 100 | 100 | 0 | 1/1 | 0.007 |
| 10,000 | 10,000 | 8,920 | 3,210 | 1/3 | 0.104 |
| 30,000 | 30,000 | 21,497 | 3,210 | 1/3 | 0.251 |
| 100,000 | 100,000 | 54,021 | 19,666 | 1/5 | 0.555 |
| 300,000 | 300,000 | 98,824 | 19,666 | 1/5 | 1.004 |
| 1,000,000 | 946,532 (fewer than the budget: all paths) | 142,811 | **29,746** | 1/6 | 2.092 |

`RESULTSET_SIZE` is a separate cap on the rows of a reply: returning the 1,000,000-budget paths as rows gave exactly
10,000 rows and 4,251 distinct symbols (same file). The benchmark used `pathCount` 10,000 = `RESULTSET_SIZE` and returned
rows, so both caps gave 10,000 and the benchmark could not tell them apart. The row-returning form fails rule 2 at any
`pathCount` above 10,000 because of the row cap; with aggregation it fails below the total number of paths (66 % of the hub's
BFS items at 100,000 and 300,000).

**Aggregated form (`sspaths_agg_check.py`, `results/sspaths_aggregated.json`).** `... YIELD path WITH last(nodes(path)) AS e
WHERE e:Symbol AND e.symbolId <> $id RETURN count(*), collect(DISTINCT e.symbolId)`, `pathCount` 1,000,000, `maxLen 6`,
all 50 anchors, 20 repeats: **the item set equalled the BFS's on 50/50 anchors**, 20/20 repeats identical on 50/50, the
budget was never exhausted by a hub (largest path count 473,266 Symbol-ending paths), pooled p95 0.53 s (hubs 1.89 s, mid
0.002 s, low 0.001 s) against the BFS's 3.70 s. It returns no chain, status or distance (those need the path or a second
query), and the budget needed is not knowable beforehand. **`pathCount` 3,000,000 on the top hub killed the container**
(below). So the benchmarked configuration fails rule 2 as recorded and the BFS stays the engine's walk, but "`SSpaths`
cannot reach the BFS's item set" is **not** what was measured; the open point for S23/the architect is whether an
item-set-only `SSpaths` pre-pass is worth a guarded path budget.

**Instance loss.** The `pathCount` 3,000,000 query (aggregated form, top hub, 2 GB container cap) ended with
`Connection closed by server`; `docker inspect`: exit 137, `OOMKilled true`. The instance has no persistence, so every
key (`ccg_p0spike`, `ccg_p0slice`) was lost. `cpg/infra/ccg-falkordb.sh up` recreated the container (`config: matches D7`,
0 keys); `ccg_p0spike` was rebuilt (record above); **`ccg_p0slice` was not reloaded** (see the follow-up report: R-B names S13
and `teco` grants it). Memory at `pathCount` 1,000,000: `used_memory_peak` 797 MB. Nothing proves 3,000,000 alone caused it
(not repeated).

**Decision (§4.5 rule, `bench_reach.adopt_decision`, depth >= 3), as benchmarked (row-returning, `pathCount` 10,000): the
BFS (`bfs-v1`); `SSpaths` is not adopted.** Rule 1 holds (20/20 identical for every anchor, 5/5 at the `D = 5` hubs). Rule 3
holds on the pooled p95 (0.64 vs 3.70 s at `D = 3`: 83 % lower; 0.65 vs 6.29 s at `D = 5`) but is hub-only: on mid anchors
`SSpaths` is slower at `D = 5`. **Rule 2 fails**: the item sets differ on 15/50 anchors, every hub, because `SSpaths`
returns paths, not the best path per node, and the 10,000-row reply cap hit silently (no error, no flag). Its chains also
differ from the BFS's in 3 % of common items (cost ties and the state-dependent function-level label, which a per-edge
weight cannot express); distance differed on 44 of 44,400.

**Per-level read latencies (for S22), all `bfs-v1` runs, both modes (`perReadMs`)**, ms, n = 101,395 reads, each a
parameterized query of <= 1,000 ids (halved on a full reply): p50 3.9, p95 61.8, **p99 224.0, max 939.8**. By edge
type, p99 / max: `HAS_CALL <-` 11.1 / 738.9 (n 43,395); `DATA <-` 537.6 / 939.8 (n 18,335); `CALLS <-` (tagged
`other`) 271.0 / 935.3 (n 18,905); `CONTROLLED_BY <-` 23.2 / 748.4; `HAS_COND <-` 9.9 / 628.7; `HAS_RETURN <-` 9.4 / 311.5;
`HAS_PARAM <-` 9.9 / 38.8; `TO <-` 3.5 / 5.0. Whole levels (`perLevelMs`, n = 22,660): p50 1.1, p95 417, p99 1,687, max
20,229. Whole answers: `full`, levels 4 (distance <= 2), p95 3.4 s, max 10.1 s, 213,515 states at most; levels 6
(distance <= 3), p95 11.0 s, max 53.5 s, **567,700 states** at most (hub, 29,746 items). Reads were issued from one
client, one at a time; `THREAD_COUNT` 8. The p99 is a hub-frontier figure: mid and low anchors' reads are single
digits of ms.

**One failed read.** During the first pass a read was killed at the engine's default `TIMEOUT` (1,000 ms, a module
argument; `GRAPH.CONFIG GET TIMEOUT`), `ResponseError: Query timed out`, while I was running niced pytest/mutation runs
on the same host. 5 reruns of that anchor (`S0723.30`) without load and without a timeout argument: worst read 83.7 ms (`results/recheck_timeout.json`; runs 42.6-83.7 ms).
Not reproduced, cause unconfirmed; the largest read in the 101,395 recorded is 940 ms. The pooled hub latencies of
records 52-62 (D = 3 hubs, first 11) overlapped those runs and may be inflated; the decision rests on rule 2, not on
latency.

## 2. S21 -- `MULTI`/`RENAME` rollback and telemetry (`python spikes/store/swap_probe.py [--only rollback|telemetry|delete|stall]`)

Graphs `ccg_p0spike__swap_{live,build,prev,...}`; markers `old`/`new` in a `(:Meta)`.

| # | Probe (commands) | Observed |
|---|---|---|
| i | `MULTI; RENAME live prev; RENAME build live; EXEC`, clean | `[True, True]`; `live` = new, `prev` = old, `build` gone |
| ii | same, `build` absent | `[True, ResponseError]`: **not rolled back** -- `live` absent, `prev` = old. Redis does not roll back; the form is unsafe |
| ii-d | the `MULTI` form with `prev` already present | `prev` overwritten silently (`older` lost) |
| e-old | `EVAL <old script: refuse only if build absent> 3 live build prev`, `build` absent / happy / `prev` present / `live` absent (first build) | `'refused: build absent'`, `live` = old / `'swapped'` / **`'swapped'`, `prev` = old, `older` lost** / `ResponseError: no such key`, nothing moved (`live` absent, `build` = new) |
| e-tight | `EVAL <SWAP_SCRIPT: also refuses `live` absent and `prev` present> ...`, same four states | `'refused: build absent'`, markers unchanged / `'swapped'`, `live` = new, `prev` = old / `'refused: prev present'`, `live` = old, `build` = new, `prev` = older / `'refused: live absent'`, `build` = new, nothing moved |
| iii | a ~12 s `GRAPH.RO_QUERY` (49 M-row self-join) on `live`; swap issued at ~1.8 s from another connection | `MULTI/EXEC` 0.46 ms, plain `RENAME` 0.40 ms, `EVAL` 0.39 ms; reader finished on the old snapshot (markers `old`,`old`, count 49,000,000) in all three; warm-connection `PING` max 5.9 / 4.1 / 6.7 ms (control without a swap: 11.1 ms) |

Record: the `MULTI` form is safe only behind guards. The first `EVAL` guarded only `build`; the probe shows it overwrites an
existing `prev` and errors (cleanly, nothing moved) when `live` is absent. `swap_probe.SWAP_SCRIPT` refuses all three
(`WATCH` not probed). A first build, with no `live`, takes a plain `RENAME build live`, not this script;
**the default stays "zero kept snapshots, rollback = re-assemble last good"** -- a kept `prev` also doubles RAM
(synthetic graph 330 MB). The plain `RENAME build live` of V2 remains the swap.

An in-process harness quirk, **not a server finding**: in my first reader/swap harness a `PING` from a *new*
connection started in the same Python process within ~50 ms of the reader thread blocked until the read finished
(11-12 s in each of 4 harness runs, and once in a cold-connection probe whose code is not kept), which made `RENAME` look like it waited 10.6 s. With a warm connection and a 0.3 s head start, or
with the reader and the `PING` in separate processes (`redis-cli`, 6 delays 0-0.3 s after the child signalled "go": 4.7-6.5 ms each, `results/recheck_ping.json`), nothing stalls; long
reads of three shapes left `PING` at 3.0-29.1 ms (`--only stall`, `results/swap_probe_stall.json`). I did not find the cause (client-side suspected).

**Telemetry.** Commands in `swap_probe.py` `probe_telemetry_*`; numbers in `results/swap_probe_telemetry.json`.
- Created lazily: stream exists 0.94 s after a graph's first query, `TYPE stream`, `TTL -1`; a parameter value was
  readable in `XREVRANGE` after 3.0 s.
- Cap: 1,200 queries -> `XLEN` 1004 (default `MAX_INFO_QUERIES 1000`, approximate trimming).
- **Capped/disabled: `GRAPH.CONFIG SET MAX_INFO_QUERIES n` works at runtime**: 10 -> `XLEN` 10; 1 -> 1; **0 -> no stream
  for a new graph, and an existing stream (33 entries) trimmed to 0** (empty key remains). Restored to 1000 and
  verified (`GRAPH.CONFIG GET`). Instance-wide, not persisted; persistence would be a module argument at start-up (the
  loaded module's args are `MAX_QUEUED_QUERIES 25 TIMEOUT 1000 RESULTSET_SIZE 10000`; adding `MAX_INFO_QUERIES 0` is
  **untested** -- it needs a restart, which I did not do). `CMD_INFO` (1) is refused at runtime (`Failed to set config
  value`); no listed knob is named for telemetry (`TELEMETRY*` names: "Unknown configuration field").
- Admin `XTRIM telemetry{g} MAXLEN 10` holds (`XLEN` 10) and the stream regrows (40 after 30 queries); `DEL` is recreated
  by the next query (2.06 s in the recorded run). `RENAME g g2`: old stream gone, new one 2.90 s after the next query on `g2`.
- **After `GRAPH.DELETE g`: `telemetry{g}` is gone at the first poll (1-9 ms; 47-63 ms for a 600k-node graph, i.e. the
  delete itself), 32/32 trials in four shapes (8 each), never reappearing in 6-8 s watched** (shapes: settled stream; a query
  immediately before the delete with no stream yet; a 600k-node graph; stream present + a query immediately before the
  delete; all with `ASYNC_DELETE 1`). The "query immediately before the delete, no stream yet" shape had no stream at the
  delete in 0/8 trials, so its "gone at the first poll" is vacuous: it tests only that no stream appears afterwards. The earlier "one sighting of asynchronous removal" is **not reproduced**; unexplained.
  Operating rule: telemetry streams hold parameter values for up to 1,000 queries and for the graph's life; for
  the engine, set `MAX_INFO_QUERIES 0` (or a small cap) on the instance, keep code text out of parameters (G13), and
  exclude `telemetry{...}` from non-admin ACL patterns; `GRAPH.DELETE` leaves no history behind, `RENAME` resets it.

## 3. Ruling: `Dependency(depId)` index and constraint

`python spikes/store/dep_index_probe.py --sizes 1000,5000,20000` (loader's `edge_statement`, batch <= 20,000, 50 units):

| `Dependency` nodes = `DEPENDS_ON` edges | edge load without index | with `Dependency(depId)` | plan operator without / with |
|---|---|---|---|
| 1,000 | 0.133 s | 0.010 s | `Node By Label Scan` + `Filter` / `Node By Index Scan` |
| 5,000 | 3.17 s | 0.049 s | same |
| 20,000 | **52.3 s** | 0.193 s (index 2 MB) | same |

Quadratic, 270x at 20,000. **Ruled: add `Dependency(depId)` as an index (before the load) and a `UNIQUE` constraint
(after)** -- the loader already dedupes `Dependency` by `depId`, and the constraint is the integrity gate every other id
label has. `ddl_v1.cypher`: 18 indexes, 11 constraints; §1.6 table and D10 updated; `ccg/store/__init__.py`'s
`parse_ddl` unchanged (strict); DDL tests (`tests/test_ddl_live.py`) and the loader tests that pin 17/10
(`test_loader_v1.py`, `test_live_slice.py`) now pin 18/11. Mutants: dropping the index or the constraint line from the
DDL fails `test_ddl_file_matches_section_1_6_statically`; dropping the index also fails the new
`test_loader_v1.py::test_every_edge_endpoint_label_has_an_index_on_its_id_property` (every `EDGE_ORDER` endpoint label must
have an index on its `ID_PROP`). `test_live_slice.py::test_sl3_load_integrity`
(live, on `ccg_p0slice`) now expects 11 constraints; its fixture reloads the slice, so it is the reload (R-B).
