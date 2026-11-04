# Review: connected-code-graph store spikes S20 + S21 (and the `Dependency(depId)` DDL ruling)

> **Status:** active · **Owner:** `analyst` · **Tracks:** P0-15 (S20, S21)

## Pass 1 (2026-11-02)

### Scope & verdict

**Reviewed (uncommitted working tree on `main` at `53492354`):** `cpg/spikes/store/MEASUREMENTS.md` (the claims);
`cpg/spikes/store/{synth,bfs,bench_reach,swap_probe,dep_index_probe}.py`; `cpg/spikes/store/tests/test_store_spikes.py`;
`cpg/spikes/store/results/*`; the diffs of `cpg/ccg/store/ddl_v1.cypher`, `cpg/tests/test_ddl_live.py`,
`cpg/spikes/vertical-slice/tests/{test_loader_v1,test_live_slice}.py`,
`docs/plans/connected-code-graph-p0-graph.md` (§1.6, D10, §6 item 4) and `claude/graph-dba/falkordb-quirks.md`.
**Baseline:** `docs/plans/connected-code-graph-p0-2.md` §4.5 (S20/S21 rules) and rows S20/S21; root `REVIEW.md`
(no `cpg/REVIEW.md` exists).
**Not done:** no `-m live` tests, no writes; on `ccg-falkordb` I ran read-only `GRAPH.RO_QUERY`/`GRAPH.CONFIG GET`/
`SCAN`/`INFO` on `ccg_p0spike` only (each read appends to `telemetry{ccg_p0spike}`, the one side effect).

**Verdict: needs changes.** The S20 *decision* (BFS `bfs-v1`; `SSpaths` not adopted) is right, and it holds more
firmly than the record says. The S21 findings are supported by the raw output. But the mechanism recorded for the
`SSpaths` failure is wrong (B1): the 10,000-row ceiling in the benchmark is `pathCount`, a global path budget, and
not `RESULTSET_SIZE`. That wrong fact is written into four places, among them the team's durable quirks KB. It is a
text fix in four places, and a re-gate only has to read those lines.

CPG: considered, not relevant — the subject is a measurement spike on `ccg-falkordb` keys plus a DDL file; the only loaded CPG is `cpg_falkorchat`, which does not index `cpg/`.

### Checks run

| Check | Result |
|---|---|
| Recompute every S20 table number from `results/bench_reach.jsonl` (250 records, not the summary) | all reproduce: repeats 50/50, item sets 50/35/35, agree counts, pooled p50/p95, per-class medians, per-read and per-level percentiles, `full` p95/max/states (Appendix A) |
| "Missed items", several definitions (hubs) | D=3: pooled 85.2 %, mean 83.7 %, median 80.5 %, per-anchor range 76.6–97.3 %, sym-diff/union 83.7 %; D=5: pooled 90.3 %, mean 90.3 %, median 90.3 %, range 84.9–97.8 %. `SSpaths` returned **0 extras** in all 150 segment records |
| Inequality vs truncation | 30/30 unequal records are truncated; 0 truncated records are equal; 0 untruncated records are unequal |
| Re-run 6 recorded anchors read-only on today's `ccg_p0spike` (2 low, 2 mid, 2 hub; D=3; both sides) | 12/12 digests equal the recorded ones; graph counts 453,905 / 2,399,308, 330 MB reproduce |
| `impact.py` provenance | SHA-256 `434801…8282171` = working tree = `git show cb1b2214:` |
| Live `SSpaths` probes on hub `S0689.14` with in-query aggregation (Appendix B) | `pathCount` is the cap, not `RESULTSET_SIZE` (B1) |
| S21 raw JSON vs table | all rows supported except three timing figures (m2) |
| Instance hygiene | `MAX_INFO_QUERIES` 1000, `TIMEOUT_MAX` 0, no `ccg_p0spike__swap_*`/`__dep_*` keys left |
| Non-live suites: `spikes/store/tests`, `tests/test_ddl_live.py`, `test_loader_v1.py`, `test_live_slice.py` (from `cpg/`) | **151 passed, 25 deselected** |
| Mutants on a scratch copy (Appendix C) | 7 killed, 3 survive (m3) |
| Unfiltered sweep for leftover 17/10 pins | no live pin left; two hits are dated historical records (slice report SL-3, p0-graph §1.1's slice load), no change needed |

### Findings

#### B1 (blocker) — the 10,000-row ceiling is `pathCount`, not `RESULTSET_SIZE`. The recorded mechanism is wrong in four places

**Evidence (verified live, Appendix B).** On hub `S0689.14` (`maxLen 6`) I put an aggregation inside the query, so
the reply was one row and `RESULTSET_SIZE` could not bind. `count(*)` came back equal to `pathCount` every time:
100 → 100, 10,000 → 10,000, 30,000 → 30,000, 100,000 → 100,000, 300,000 → 300,000. So `pathCount` is a **global
budget on paths, in weight order**. `RESULTSET_SIZE` caps only the rows sent back to the client. The benchmark
set `PATH_COUNT = 10_000`, the same value as `RESULTSET_SIZE`, so the two caps could not be told apart. The 1,000,000 probe
(10,000 rows, 4,251 symbols, a different subset from 10,000's 3,210) is the one case where `RESULTSET_SIZE` cut the
reply.

**Where the wrong cause is written:**
- `MEASUREMENTS.md` §1: "cut at `RESULTSET_SIZE` on 15/15", and "the reply is cut at 10,000 rows silently (no error, no flag)".
- `falkordb-quirks.md`, the new `SSpaths` entry: the headline "a reply is cut at `RESULTSET_SIZE` silently".
- `p0-graph.md` §6 item 4: "reply cut at 10,000 rows".
- `bench_reach.py:41`: `# = RESULTSET_SIZE: a larger pathCount returns the same 10,000 rows`.

**Why it matters.** The quirks file is the team's durable KB. A reader of the entry as written would raise
`RESULTSET_SIZE`, or aggregate in the query, and expect `SSpaths` to become complete. It would not. At `pathCount`
100,000 and 300,000 (0.70 s and 1.06 s inside the engine) the hub still reached only 19,666 of its 29,746 BFS items,
and the longest path returned was 5 edges. `SSpaths` lists every lighter path first, so a deep node needs a path
budget far larger than the node count. That is the real reason rule 2 fails, and it makes the BFS decision firmer.

**Fix (`graph-dba`).** Rewrite the four sites to say three things:
- `pathCount` is a global, weight-ordered path budget, exactly met when enough paths exist.
- `RESULTSET_SIZE` is a separate cap on reply rows, avoidable with in-query aggregation.
- Coverage of distance-3 nodes did not converge by `pathCount` 300k: 66 % of the hub's BFS items.

Optionally add that `pathCount: 0` returned only the minimum-weight tier (3,521 length-1 paths, 0 `Symbol`s).
Keep "silently". The `truncated` flag stays valid, because both caps were 10,000.

#### M1 (major) — the EVAL guard only protects against a missing `build`. The record calls the form "safe"

**Evidence.** Raw result ii-d: `RENAME live prev` overwrites an existing `prev` without a word. The guarded `EVAL`
(`swap_probe.py:129`) runs the same two `RENAME`s, so it overwrites an existing `prev` the same way. I inferred
this from the script text; it was not probed. On a first build, `live` is absent: the first `redis.call('RENAME')`
raises, and the script aborts with an error before any write (Redis Lua semantics, also inferred). The record says
"the `MULTI` form is safe only behind an `EXISTS` guard". S22's `lock_replay.py` will reuse that script.

**Fix.** In `MEASUREMENTS.md` §2, and in the matching quirks entry, state the guard's exact reach:
- `build` absent: refused, verified.
- `prev` present: overwritten, inferred.
- `live` absent: error and nothing moved, inferred.

Route the two inferred cases to S22 as probe rows, or add `EXISTS KEYS[3]==0` and `EXISTS KEYS[1]==1` to the
guard. That design choice belongs to S22's owner, not to this review.

#### m1 (minor) — recorded claims with no raw output in `results/`

These are cited as measured but have no file behind them:
- the `pathCount` 1,000,000 probe (4,251 symbols);
- the 10-anchor digest re-check after the rebuild;
- the `redis-cli` 6-delay `PING` probe (5–8 ms);
- the 5 timeout reruns (worst 85 ms);
- the synth build record (41 s, +423 MB).

I re-derived the load-bearing ones myself: 6 anchors re-digested, graph counts and memory read live, `pathCount` 100 and 10,000. **Fix:** save each to
`results/` (a JSON file, or a recorded command with its output), or mark it "observed, not recorded". The
S21 table already uses that style elsewhere.

#### m2 (minor) — three S21 numbers disagree with their raw file

`swap_probe_telemetry.json` records:
- `DEL` recreated after **2.06 s**. `MEASUREMENTS.md` and quirks both say **0.56 s**.
- new stream after `RENAME` **2.90 s**. `MEASUREMENTS.md` says ~2.8 s.

`swap_probe_stall.json` gives per-shape `PING` maxima of **3.0–29.1 ms**; the record says "1.5–29 ms". The 0.56 s
figure probably comes from an earlier run that the results file overwrote (`main` writes one file per `--only`).
**Fix:** quote the file, or keep and cite the earlier run's output. The `query-just-before-delete` shape never had a stream
(`streamExistedBeforeDelete` 0/8), so its "gone at first poll" is vacuous. Say that this shape tests only non-reappearance.

#### m3 (minor) — three mutants survive the spike tests (Appendix C)

- **`truncated: len(rows) >= 10_000` → `>`** survives. That flag is the evidence for "cut on 15/15". At exactly
  10,000 rows the mutant reports `False` everywhere. The only test that reads the flag is the live 40-file test,
  which uses it as a filter on a graph that never truncates.
- **`summarize`'s `missingFraction` numerator swapped to `extraInSSpaths`** survives. `summarize()` has no test,
  yet it produces every number MEASUREMENTS quotes (they match my recompute today; they would not after a rerun).
- **The EVAL `EXISTS` guard deleted** survives. The guard is checked only by the live probe output.

**Fix (`graph-dba` or `tdd-engineer`):**
- a pure test of `sspaths_answer` with a fake graph returning exactly 10,000 and 9,999 rows;
- a `summarize` test over a 3-record synthetic JSONL whose expected values are computed by hand in the test;
- if the script is lifted into S22's `lock_replay.py`, a test there (a fake `EVAL` cannot test Lua, so this belongs
  in S22's live fault tests, not here).

#### m4 (minor) — no guard for the defect class that `Dependency` was

**Evidence.** The quadratic load happened because an `EDGE_ORDER` endpoint label had no index on its `ID_PROP`.
Nothing tests that class. `test_indexes_precede_every_write_and_cover_the_point_labels` checks constraint labels
and `pointId`, not edge endpoints. I ran a 6-line probe on a scratch copy, as an assertion over `L.EDGE_ORDER`:
for each endpoint except `SkippedFile`, require `CREATE INDEX FOR (n:<label>) ON (n.<ID_PROP[label]>)` to be in
`L.index_statements()`. Today it reports none missing. With the `Dependency` index line removed, it reports
`{'Dependency'}`.

**Fix.** Add it as a test in `test_loader_v1.py`. It reddens on that mutant, it is not authored inside the step it guards
(the index is already in), and it catches the next label a P1 phase adds to `EDGE_ORDER`.

#### m5 (minor) — growth-stops check on `docs/plans/connected-code-graph-p0-graph.md`

- D10 is rework, rewritten in place: compliant.
- The §1.6 `Dependency` row is new scope: compliant.
- The trimmed "Deferred" list is rework: compliant.
- **§6 item 4** is new, but it restates `MEASUREMENTS.md`'s facts: BFS beats `SSpaths`, `MULTI` without a guard,
  `MAX_INFO_QUERIES 0`, `GRAPH.DELETE`. One of them carries B1's wrong mechanism.

This is a restated-facts finding: the facts' home is `MEASUREMENTS.md` and then S23's decisions table. **Fix:**
cut item 4 to a pointer: "S20/S21 measurements: `cpg/spikes/store/MEASUREMENTS.md`; `Dependency(depId)` added
(§1.6, D10)".

#### m6 (minor) — `falkordb-quirks.md`: a duplicate entry, and the fold came early

- The new "default read `TIMEOUT` is 1000 ms" entry restates the existing **"Default `TIMEOUT` is 1000ms"** entry
  (around line 1143; `TIMEOUT_DEFAULT 0` / `TIMEOUT_MAX 0` re-measured 2026-09-08). Fold the new evidence (the
  benchmark kill, "every program-issued read passes an explicit `timeout`") into that entry as one line.
- The plan gives the quirks fold to S23 (§4.5, "S23 … folds the new quirks"). Doing it early is fine if `teco` accepts it.
  But S23 must then not re-add the same entries. Record the hand-off in the ledger.

#### n1 (nit) — small inaccuracies

- `swap_probe.py`'s docstring says a `SET` is tried only on unlisted names, but `probe_telemetry_config_effect`
  sets `MAX_INFO_QUERIES` instance-wide. Its Usage line also omits `stall`.
- `bfs.py`'s docstring says "the working-tree version". It is the committed `cb1b2214`.
- "5 splits in the 50-anchor `full` runs" is 5 per level setting (10 total), counted from the last repeat only.
- The "four shapes" parenthetical lists five items: `ASYNC_DELETE 1` is the config, not a shape.
- The pooled D=5 p95 weights hubs at 5 repeats against 20 for everything else. That does not change the rule-3 call.
- `test_items_from_paths_prefers_fewer_possible_hops_then_distance_then_length`: the first `assert … or …` is
  redundant, given the next one.
- Keys `ccg_p0spike*` differ from the reservation table's `ccg_spike_synth*` / `ccg_spike_swap*`
  (`p0-2.md` §5.1). Either align the table or note the rename in the ledger.

### What's solid

- **The S20 decision, and a fair comparison.** Both sides run the same direction (`incoming` vs `CALLS <-`/`HAS_CALL <-`) on the same segment (`levels = maxLen = 2D`).
  - They agree exactly wherever `SSpaths` was not truncated: 120/120 records, 0 extras across all 150.
  - `bfs.py` wraps `impact.walk` (hash-pinned to `cb1b2214`) and does not re-implement it.
  - The BFS's re-split on a full reply is tested, and its mutant is killed.
  - The SSpaths side did not get an equivalent remedy (a larger `pathCount` with in-query aggregation). My probe shows that remedy fails rule 2 anyway (B1), so the decision stands.
- **The numbers.** Every S20 number reproduces from the raw JSONL. Digests reproduce on today's graph.
- **S21 rollback and in-flight probes.** They are well built: markers per key, a no-swap control, a warm second connection. The harness `PING` artefact is honestly separated from server findings.
- **Instance hygiene.** Config was restored and scratch keys were removed (checked live).
- **The DDL change matches the loader.** `node_statement` uses `CREATE`, and `edge_statement` uses `MATCH (b:Dependency {depId: r.t})` (`load.py:410-420`).
  - `build_rows` dedupes `Dependency` by `depId` across every bundle of a build (`load.py:201`, `_add_node`), so the new `UNIQUE` cannot `FAIL` on a well-formed build.
  - `depId` is repository-neutral (`ccg/ids.py` `dep_id`), which makes the cross-bundle dedupe the right scope.
- **The tests.** The DDL tests take their expectation from the plan's §1.6 table, not from the DDL. The new `drop_dep` mutant and the plan-table `Dependency` assertion are real checks: both DDL mutants are killed (Appendix C).
- **The count pins** (18/11, `+1 +1` in the recorder test, the `Dependency` label in the constraint set) are all edited consistently. No live pin remains.

### Open questions

- `test_sl3_load_integrity` (live) now expects 11 constraints, and `ccg_p0slice` has 10 until it is reloaded. Who reloads it, and when?
  - Until then the `-m live` slice suite is red by design.
  - It needs a reservation (R-B), so it is `teco`'s call.

### Proposed REVIEW.md lines

Target: root `REVIEW.md`, heading **Always check**. Source: B1. It changes what a reviewer does: one more probe before accepting an attribution.

- (addition) When a measurement blames a limit on one of several caps that share the value it hit, check that the cause was isolated: one cap varied, or the other made unable to bind (e.g. an aggregation inside the query). A number equal to two caps proves neither.

No prune: no line in root `REVIEW.md` is now enforced by a test, CI job or hook.

### Appendix A — S20 recompute (from `results/bench_reach.jsonl`, nearest-rank percentiles as `bench_reach.percentile`)

| D | BFS id. | SS id. | equal | status / distance / chain on common | p50/p95 BFS | p50/p95 SS | samples |
|---|---|---|---|---|---|---|---|
| 1 | 50 | 50 | 50 | 10,859 / 10,859 / 10,859 of 10,859 | 0.0012 / 0.0329 | 0.0008 / 0.0482 | 1000 |
| 3 | 50 | 50 | 35 | 44,400 / 44,356 / 42,951 of 44,400 | 0.0048 / 3.7026 | 0.0041 / 0.6376 | 1000 |
| 5 | 50 | 50 | 35 | 49,340 / 49,296 / 47,857 of 49,340 | 0.0078 / 6.2869 | 0.0071 / 0.6514 | 775 |

- **Where the disagreements are.** All 44+44 distance disagreements are in truncated hub records. Chain disagreements: hubs 1,447 at each depth; mid 2 (D=3) and 36 (D=5), untruncated, so these are cost ties, which supports the "ties" explanation.
- **Hub medians.** BFS 19,045 (D=3) / 29,630 (D=5); SS 2,756 at both. The D=5 SS reply equals the D=3 one: the path budget is used up by short paths.
- **Per-read.** n 101,395, p50 3.913, p95 61.787, p99 223.995, max 939.765 ms; per-edge p99/max as in MEASUREMENTS.
- **Per-level.** n 22,660, p50 1.121, p95 417.4, p99 1,687.3, max 20,228.6 ms.
- **`full`.** levels 4: p95 3.38 s, max 10.09 s, 213,515 states. Levels 6: p95 10.99 s, max 53.49 s, 567,700 states (hub `S0689.14`, 29,746 items).
- **The crash window.** It falls on records 52–62 (1-based), i.e. the first 11 D=3 hubs. A second `meta` record sits at index 62 (the resume).

### Appendix B — live `SSpaths` probes (read-only, `ccg_p0spike`, hub `S0689.14`, `maxLen 6`, `TIMEOUT 60000`)

```
MATCH (a:Symbol {symbolId:'S0689.14'}) CALL algo.SSpaths({sourceNode:a, relTypes:['CALLS','HAS_CALL'],
  relDirection:'incoming', weightProp:'w', maxLen:6, pathCount:$pc}) YIELD path, pathWeight
WITH last(nodes(path)) AS e, length(path) AS l
RETURN count(*), count(DISTINCT e), count(DISTINCT CASE WHEN e:Symbol THEN e END), min(l), max(l)
```

| pathCount | count(*) | distinct ends | distinct `Symbol` ends | len min/max | engine ms |
|---|---|---|---|---|---|
| 1 | 1 | 1 | 0 | 1/1 | 3.7 |
| 0 | 3,521 | 3,521 | 0 | 1/1 | 7.6 |
| 100 | 100 | 100 | 0 | 1/1 | 27.5 |
| 10,000 | 10,000 | 8,920 | 3,210 | 1/3 | 116.8 |
| 30,000 | 30,000 | 21,497 | 3,210 | 1/3 | 293.4 |
| 100,000 | 100,000 | 54,021 | 19,666 | 1/5 | 702.1 |
| 300,000 | 300,000 | 98,824 | 19,666 | 1/5 | 1,062.8 |

BFS reference for this anchor (recorded): 29,746 items at D=3 and 30,591 at D=5. `used_memory` stayed at 837 MB.

### Appendix C — mutants (scratch copy of `cpg/{ccg,tests,spikes/store,spikes/vertical-slice}`, `PYTHONPATH` pinned to it)

| Mutant | Suite | Result |
|---|---|---|
| `items_from_paths`: drop `k > 0 or` (function-level labelling) | store tests | killed |
| `adopt_decision`: `<=` → `<` | store tests | killed |
| `SplitRows`: `>= RESULT_SET_LIMIT` → `>` | store tests | killed |
| `sspaths_answer.truncated`: `>=` → `>` | store tests | **survives** |
| `summarize.missingFraction` numerator → `extraInSSpaths` | store tests | **survives** |
| EVAL script: `EXISTS` guard removed | store tests | **survives** |
| DDL: `Dependency` constraint line removed | `test_ddl_live.py` | killed |
| DDL: `Dependency` index line removed | `test_ddl_live.py` | killed |
| DDL: `Dependency` index line removed | `test_loader_v1.py` | killed |
| m4 probe (endpoint-label index coverage) on the DDL without `Dependency` | ad-hoc | reports `{'Dependency'}`; on today's DDL, none |

## Pass 2 (2026-11-02)

**Scope.** The uncommitted fixes on top of `4da8e51f`: diffs of `MEASUREMENTS.md`, `bench_reach.py`, `bfs.py`,
`swap_probe.py`, `tests/test_store_spikes.py`, `test_loader_v1.py`, `p0-graph.md` and `falkordb-quirks.md`; the new
`sspaths_agg_check.py` and `recheck_claims.py`; the new `results/*.json` files and the re-run `swap_probe_rollback.json`.
I did reads only on `ccg_p0spike`. Nothing was written to the instance, and no live tests ran.

**Verdict: approve with suggestions.** No blocker or major remains open. The stopping signal is at the end of this pass.

**Correction to Pass 1.** Pass 1 said, in *What's solid* and in B1's "Why it matters", that the remedy
(aggregation plus a larger `pathCount`) "fails rule 2 anyway". **That was wrong.** I extrapolated from `pathCount`
300k, a sweep that had not saturated. At 1,000,000 the aggregated form equals the BFS's item set on 50 of 50 anchors
(recomputed below). The BFS decision still follows, but for a different reason: the aggregated form returns no chain.

### Checks run

| Check | Result |
|---|---|
| `results/sspaths_aggregated.json`, recomputed from the records | 50/50 `setEqual`; 50/50 repeats identical, 20 per anchor; pooled p95 **0.5346 s** (hub 1.889 s, mid 0.0017 s, low 0.0005 s) vs BFS D=3 **3.7026 s** from `bench_reach.jsonl`; its `bfsItems` equal the benchmark's D=3 counts on 50/50 anchors |
| Aggregated form, my own run (hub `S0450.05`, `pathCount` 1,000,000) | 19,969 `Symbol`-ending paths, 13,430 items, set equal to a fresh BFS, 0.10 s |
| "No chain, status or distance" | true by construction: `sspaths_agg_check.QUERY` returns `count(*), collect(DISTINCT e.symbolId)` and nothing else |
| `sspaths_pathcount_probe.json` vs my Pass-1 Appendix B | the 0 to 300,000 rows match; 1,000,000 gives 946,532 paths (< budget), 29,746 symbols, length up to 6; row-returning at 1,000,000 gives 10,000 rows and 4,251 symbols |
| Rebuilt `ccg_p0spike` (container started 10:26, `RestartCount 0`, `OOMKilled false` on the new container) | 453,905 nodes, 2,399,308 edges, 330 MB, 11 constraints `OPERATIONAL` (the benchmarked graph had 10; `Dependency` is now one of them); `ccg_p0slice` absent; no swap or dep keys left; `MAX_INFO_QUERIES` 1000, `TIMEOUT_MAX` 0 |
| Digests on the rebuilt graph, 4 anchors not in `recheck_digests.json` | D=5 hub `S0014.18`, mid `S0156.30`, low `S0113.00`: BFS and SSpaths digests equal the recorded ones (6/6); `full` levels 4 `S0048.27`: equal. `recheck_digests.json` adds 10/10 more |
| Suites, from `cpg/` | touched suites **158 passed, 29 deselected**; the default `cpg` suite **1504 passed, 8 deselected** |
| Mutants on a fresh scratch copy | 8 killed, 1 survives the non-live suite (P2-m2) |

### Dispositions

- **B1 — fixed.** All four sites now name `pathCount` as a global, weight-ordered budget and `RESULTSET_SIZE` as a separate row cap: `MEASUREMENTS.md` §1, quirks, `p0-graph.md` §6 (now a pointer), `bench_reach.py:42`. The per-`pathCount` table matches the probe file and my Pass-1 run.
- **M1 — fixed.** `SWAP_SCRIPT` checks `EXISTS` on all three keys before its first `RENAME` (`swap_probe.py:41-44`). `swap_probe_rollback.json` shows the old script and the tight script in four starting states each, and the record matches the file in all 8 cells. The old script overwrites `prev` (`older` lost) and fails with `no such key` when `live` is absent; both of my Pass-1 inferences are now measured. The tight script refuses `build absent`, `live absent` and `prev present` with markers unchanged.
- **m1 — fixed.** `synth_build_record.json`, `recheck_{digests,ping,timeout}.json` and `sspaths_pathcount_probe.json` exist, and the numbers in the text match them (83.7 ms; PING 4.6–6.5 ms; 423,508,696 B). The first two builds' 41 s is marked "unrecorded". **Residual:** no script that is kept produces `sspaths_pathcount_probe.json`; `bench_reach.py` only cites it.
- **m2 — fixed.** 2.06 s, 2.90 s and 3.0–29.1 ms are quoted from their files, and the vacuous shape is now stated.
- **m3 — fixed.** The `truncated` (`>=`→`>`), `missingFraction` and `chainAgree`→`distanceAgree` mutants of `summarize` are killed, as is deleting the `prev`-present or `live`-absent guard (static text test). A remaining gap is reported as P2-m2.
- **m4 — fixed.** `test_every_edge_endpoint_label_has_an_index_on_its_id_property` is killed both by removing the `Dependency` index line and by renaming `ID_PROP["Dependency"]`.
- **m5 — fixed.** §6 item 4 is a single pointer. Growth-stops: this is rework that replaces text, with no growth.
- **m6 — not fixed, deferred.** The duplicate TIMEOUT entry remains. Folding it is handed to S23 in `MEASUREMENTS.md`'s header, along with "S23 must not re-add" the early entries. That is acceptable as an explicit hand-off.
- **Nits — fixed:**
  - `swap_probe.py` docstring and Usage line;
  - `bfs.py` "committed `cb1b2214`";
  - the splits wording (5 at levels 4, 5 at levels 6);
  - "all with `ASYNC_DELETE 1`";
  - the D=5 weighting note;
  - the redundant `assert … or …` removed;
  - the key-name drift, which is noted for the ledger. **Residual:** the reservation table in `p0-2.md` §5.1 is unchanged; that is `teco`'s.

### New findings

#### P2-m1 (minor) — `pathCountExhausted` measures the wrong count, and the record reads it backwards

**Evidence.** `sspaths_agg_check.answer` returns `"exhausted": r[0] < pc`. But `r[0]` is `count(*)` *after*
`WHERE e:Symbol`, so it counts only the paths that end on a `Symbol`. It is below `pc` even when the engine used up
the whole budget. The flag is `True` on 50/50 records. `MEASUREMENTS.md` §1 calls this "the budget was never exhausted
by a hub (largest path count 473,266 Symbol-ending paths)". Only the top hub is known to be under budget, from the
unfiltered probe (946,532 < 1,000,000). The item-set equality itself is checked directly, so the 50/50 result stands.

**Fix.** Count before the filter (`WITH collect(last(nodes(path))) AS ends, count(*) AS n …`). If that is not done,
rename the flag `symbolPathsBelowBudget` and reword the sentence.

#### P2-m2 (minor) — a mutant that breaks the swap's `RENAME` order survives the non-live suite

**Evidence (run).** I swapped the two `RENAME`s in `SWAP_SCRIPT` (`build`→`live` first, then `live`→`prev`). That
version destroys the old `live` and leaves `prev` = new. `test_swap_script_checks_all_three_keys_before_its_first_rename`
still passes, because it is a text-order check: 58 passed. Only `test_live_swap_script_refuses_before_moving_anything[happy]`
(`-m live`) would catch it, and I did not run it. Neither `MEASUREMENTS.md` nor `results/` shows that this test ran;
the four-state evidence comes from the probe, not the test.

**Fix.** Run the live test once under a swap reservation and record the result. Optionally add one assertion to the
static test: the text `RENAME', KEYS[1], KEYS[3]` comes before `RENAME', KEYS[2], KEYS[1]`.

#### P2-m3 (minor) — the quirks `SSpaths` entry claims a little more than was measured

- "Returning rows instead, **any** `N` above 10,000 gave exactly 10,000 rows and 4,251 symbols": only `N` = 1,000,000
  was measured in row-returning form. The row count follows from `RESULTSET_SIZE`; the 4,251 does not generalize.
- "the row-returning form reached 10–15 % of the BFS's items": those are the pooled figures (14.8 % at D=3, 9.7 % at
  D=5). Per anchor the range is 2.7–23.4 % (D=3) and 2.2–15.1 % (D=5).

**Fix.** Write "at `N` = 1,000,000" and "pooled 10–15 % (per anchor 2–23 %)".

#### P2-m4 (minor, inferred) — how `pathCount` relates to memory is not yet a rule

At `pathCount` 1,000,000 the top hub had only 946,532 paths in total, so a budget of 3,000,000 could not produce
more paths. If that query caused the OOM, memory grows with the *budget* (allocated up front), not with the paths
found. I inferred this and did not test it, because testing it risks a second instance loss. `MEASUREMENTS.md`
correctly says "nothing proves 3,000,000 alone caused it". The quirks rule "never raise it past what the container
can hold" cannot be applied without a number.

**Fix (`graph-dba`, S23).** State the safe ceiling that was observed (1,000,000 → peak 797 MB on this 330 MB graph,
2 GB container). Mark budget-proportional allocation as a hypothesis to probe on a disposable instance, never on the
shared one.

### The decision still follows

The aggregated form passes rule 2 (50/50) and rule 3 (pooled p95 0.53 s against 3.70 s). It cannot pass rule 1 as the
plan words it ("20/20 repeats identical, **chains included**"): it returns no chain, status or distance.

The chain is a requirement of its own:
- **FR-20 / AC-18** (`docs/requirements/connected-code-graph.md:170`, `:427`) require every item's full chain.
- **FR-19**, which the follow-up message cited, is the repeatability requirement. The chain requirement is FR-20, not FR-19.

Bringing the chain back means returning paths. That brings back either the 10,000-row cap or a per-node reduction in
the query. Even untruncated, `SSpaths`' cheapest path disagreed with `bfs-v1`'s chain on 38 mid-anchor items: per-edge
weights cannot express the function-level state rule.

So **keep the BFS**. Whether an item-set-only `SSpaths` pre-pass is worth a guarded path budget is an open design
point for `architect`/S23. `MEASUREMENTS.md` already frames it that way, and the OOM is the cost to weigh.

The OOM itself is recorded: in the quirks `SSpaths` entry (exit 137, `OOMKilled true`, all keys lost), in
`MEASUREMENTS.md` §1 "Instance loss", and in `sspaths_agg_check.py`'s docstring. What I could check (inferred
consistency) is that the container is new, `ccg_p0slice` is gone, and `used_memory_peak` is 798.77 MB. The OOM
itself cannot be re-observed.

### Interactions between the fixes made in this revision

- **The rebuild used the new DDL** (11 constraints instead of the benchmark's 10). Traversal does not touch constraints, and 27 re-digests match (10 anchors × 2 sides theirs, 3 × 2 + 1 `full` mine). No effect.
- **The rollback re-run overwrote `swap_probe_rollback.json`.** The iii numbers changed (0.39–0.46 ms; PING 4.1–6.7 ms, control 11.1 ms), and both `MEASUREMENTS.md` and quirks quote the new run. They are consistent. Pass 1's m2 point holds the other way round: the earlier figures now exist only in git history (`4da8e51f`).
- **The aggregated check compares against a fresh BFS on the same rebuilt graph,** so it is internally consistent. Its BFS counts also equal the old benchmark's counts on all 50 anchors.
- **The OOM made the Pass-1 open question worse.** `ccg_p0slice` no longer exists at all, so every `-m live` slice test is red, not only `test_sl3_load_integrity`. Any unit that reads `ccg_p0slice` (R-B) is blocked until it is reloaded. That is `teco`'s call, unchanged.

### Stopping signal (falsifiable)

This approval stands until one of these happens:
1. `test_live_swap_script_refuses_before_moving_anything` (`-m live`, under a swap reservation) fails any of its 4 cases.
2. The same test passes with P2-m2's swapped-`RENAME` mutant applied.
3. Recomputing `sspaths_aggregated.json` or `bench_reach.jsonl` gives a figure that differs from `MEASUREMENTS.md`.
4. A re-digest of a recorded anchor on `ccg_p0spike` differs.

Run (1) and (2) once and record them; that closes P2-m2.

### Proposed REVIEW.md lines

**The Pass-1 line is still wanted, unchanged.** This pass adds another instance of the class it targets: my own
Pass-1 claim that the remedy fails was drawn from a cap sweep that had not saturated. So, to the line's sentence "A
number equal to two caps proves neither", add: "and a sweep that has not reached the parameter where the result
stops changing proves no negative".

One addition, root `REVIEW.md` → **Always check**. Source: the OOM in this pass (the `pathCount` 3,000,000 read lost
`ccg_p0slice`, which belongs to another unit).
- (addition) A live probe that sweeps a parameter which scales memory or time on a shared instance states its
  ceiling and stop rule before it runs; a probe without one is a finding.

## Pass 3 (2026-11-04) — S22 + S23 (commit `7c5ce861`)

**Scope.** `cpg/spikes/store/{acl_probe,lock_replay,timeout_probe,stall_probe}.py`, `tests/{test_lock_replay,test_stall_probe}.py`,
the five new `results/*.json`, `MEASUREMENTS.md` §4–§5, the quirks diff, and `p0-graph.md` §3.3/§4/§5/§6. The contract is the
p0-2 S22/S23 rows and §4.5. On the instance I used only `ccg_p0spike*` keys. Writes came only from the tests I ran: their
own keys and temporary ACL users, all removed afterwards (checked: `--scan` shows only `ccg_p0slice`, `ACL USERS` shows only `default`).

**Verdict: approve with suggestions.** One major (P3-1) is a text fix in the quirks file and `MEASUREMENTS.md`. Every §6
decision value follows from its measurement.

### Checks run

| Check | Result |
|---|---|
| §6 rows recomputed from `results/*.json` | rows 1, 4, 13 and 11 match Pass 1/2's recomputes; row 5: 10 × 537.6 = 5,376 → 10,000; row 6/7 match `timeout_probe.json` (kills at 3.014 / 3.018 / 1.005 s, 0 committed; 60000 rejected; backfills 1.16 / 1.647 s at cap 800); row 8: totals 0.448–0.464 s, 3 × 0.464 = 1.392 → 2,000 ms; row 10: `acl_matrix.json` selector diffs empty for 4/4 users, `flat+get`/`spec` leak `DEL` for build/verdict, `spec` loses the release; row 12: 13–21 / 29–36 ms; rows 14/15 match `stall_probe.json` (73.1 ms at 100k; 182 / 485 ms; 28,169–28,604 then 27,506 edges/s; 27.0 ms during 49.6 s) |
| §6 rows with no JSON | row 3's 797 MB (carried, `INFO`, unrecorded); row 9's "21 mutants, all killed" (no list or record anywhere); §5's 917 MB peak (consistent with today's `used_memory_peak` 874.40 MiB = 916.9 MB) |
| `test_lock_replay.py -m live`, twice | **8 passed** both runs (11.9 s / 12.0 s), keys cleaned up |
| P2-m2 closure: `test_store_spikes.py -m live -k live_swap_script` | **4 passed**; the swapped-RENAME mutant is now red in both the non-live (new order assertion) and live tests |
| Suites (from `cpg/`) | store tests + `test_loader_v1.py` **96 passed, 18 deselected**; default `cpg` suite **2273 passed, 1 skipped, 10 deselected** |
| My mutants, scratch copy `…/scratchpad/p3mut` (Appendix D) | 11 run: 8 killed (4 of them only by the live tests), 3 survive (P3-3) |
| FalkorDB docs, `TIMEOUT_DEFAULT` (docs.falkordb.com configuration page) | it applies to reads **and writes** and may be set up to `TIMEOUT_MAX` (P3-4) |

### Carried dispositions

- **P2-m1 — fixed.** The flag is renamed `symbolPathsBelowBudget`, and the docstring and §1 say it shows nothing about exhaustion.
- **P2-m2 — fixed, and run here.** There is now a static order assertion. The live 4-case test passes, and the RENAME-order mutant fails it (above).
- **P2-m3 — fixed.** "at `N` = 1,000,000", "pooled 10–15 % (per anchor 2–23 %)".
- **P2-m4 — fixed.** The ceiling is stated as 1,000,000, and memory growth with the budget is marked as a hypothesis for a disposable instance.
- **m6 — fixed.** The duplicate TIMEOUT entry is folded into the existing entry at line 1143.

### Findings

#### P3-1 (major) — the incident's "only minutes-long behaviour" contradicts the incident's own record

**Evidence.**
- **What is claimed.** `MEASUREMENTS.md` §5 and the new quirks stall entry offer the un-indexed write as the explanation:
  "100k rows over a 20k-node label would take tens of minutes while `PING` still answers … would look like an unresponsive
  instance" (marked inferred).
- **The record rules it out.** The incident (`docs/reviews/connected-code-graph-p0-impl.md`, "Incident during this check")
  states that `GRAPH.EXPLAIN` of that write showed `Node By Index Scan` on **both** endpoints. It also says the server did
  not answer `PING`, or `HELLO` on a new connection. A slow write with `PING` still answering would not cause that.
- **The arithmetic is overstated.** At 4.96 ms/row (`stall_probe.json`, 10k rows in 49.55 s), 100k rows take about
  8 minutes, not tens of minutes.
- **Hypotheses not considered:**
  - memory pressure: it was a RAM probe, on a 6 GB VM, in a container with a 2 GB cap;
  - the S21 in-process new-connection artefact (§2), which matches a "new connection does not answer" symptom.

**Why it matters.** The quirks file is the team's durable knowledge base. It now points the next investigator at a cause
the incident had already excluded.

**Fix (`graph-dba`).**
- Replace the inference with: "not reproduced; the incident write was index-backed; open hypotheses: memory pressure, a
  client-side new-connection stall (S21)".
- Keep row 15 as "not reproduced, cause open".

#### P3-2 (minor) — the 100k hard ceiling equals the size of the unexplained incident

Row 14 sets the hard ceiling at 100,000 rows. That was the incident's write size, and that write stalled the instance for
minutes for a cause that is still not known. This probe measured 100k only on an idle instance with ~59 MB in use.
`BATCH_MAX` stays 20,000, so nothing breaks today. **Fix:** state the condition next to the ceiling ("idle instance;
the S03-check incident stall was at this size, cause open"), or set the ceiling at the last size below the incident (50,000)
until the cause is known. That choice is the owner's.

#### P3-3 (minor) — the "21 mutants, all killed" claim has no record, and the stall measurement path has no test

- Row 9 cites "21 mutants of the lock, ACL-rule and stall code, all killed", but nothing records which mutants. That makes
  it an unrecorded figure, under `MEASUREMENTS.md`'s own convention.
- My mutants of the stall **measurement** path all survive the suite, because `test_stall_probe.py` tests only
  `summarize_pings`:
  - the sampler pinging `falkordb-dev` (6379) instead of 6390;
  - the sampler interval changed from 5 ms to 500 ms;
  - the "during" window collapsed to `[t1, t1]`.
- **The recorded run is not affected.** Its own data is a positive control: the worst `PING` grows with write size
  (11.7 → 485 ms), and the sample counts imply about 5.8 ms per sample (609 in 3.54 s). A rerun, though, is unprotected.

**Fix.** Record the 21 mutants (name + killing test), or drop the number from row 9. Add a check to the probe's own
output, asserted in `run()` and not in a unit test: during the 500k step, `during.max_ms` must exceed `baseline.max_ms`.

#### P3-4 (minor) — "an explicit `timeout` is not needed to avoid a kill" holds only while `TIMEOUT_DEFAULT` is 0

The measurement is right for the recorded state (`timeoutDefault 0`, a kill at the cap: 3.014 s). The FalkorDB docs say
`TIMEOUT_DEFAULT` applies to reads **and writes** and may be set to any value up to `TIMEOUT_MAX`. An operator who sets
it to, say, 1,000 would kill every load query that passes no `timeout` at 1 s. The explicit `timeout` is what protects
against that.

D16 (§4) still requires the explicit `timeout`, so the loader rule is not weakened. Only its stated reason is incomplete:
"S22: without one the cap applies anyway". **Fix.** Add "with `TIMEOUT_DEFAULT` 0" to the quirks headline, §4's
parenthesis and §6 row 6. Keep D16's explicit `timeout`.

#### P3-5 (minor) — the TIMEOUT_MAX sizing: the "tenth" clause holds only at 10,000, and one read already broke it

- §4.5's rule is "the smallest round value ≥ 10× the worst p99 per-level read, with each per-level read ≤ a tenth of it".
  At 10,000 the clause holds: the max recorded read is 939.8 ms. At the noted alternative of 6,000 it fails (939.8 > 600)
  unless the BFS chunk shrinks. That is the real argument for 10,000, and the record should say so.
- "939.8 ms is the largest read" also leaves out the S20 read that was **killed** at 1,000 ms under load, which is absent
  from the records because it crashed the run. So one read has already exceeded the tenth.

**Fix.** In §4 / row 5, state that 6,000 would violate the tenth clause, and record the killed read as the known exception.

#### P3-6 (minor) — the telemetry-removal fact now reads two ways

- The quirks entry heading (line 1282) still says "removed **synchronously** … the sighting did not reproduce". The
  2026-11-04 text appended under it says the removal lags by 13–36 ms and that the `--scan` sighting reproduced.
- `MEASUREMENTS.md` §2 says "not reproduced"; §5 says "reproduces".

The fact holds parameter values (the G13 retention concern), so a reader who stops at the heading gets the wrong rule.
**Fix.** Rewrite the heading and §2's sentence to the current fact (a lag of tens of ms, poll). This is the "rewrite,
don't append" rule. The single `--scan` sighting on the ~1M-edge graph is unrecorded; mark it as such.

#### P3-7 (minor) — the proposal's `~ccg_<system>*` pattern crosses systems that share a prefix (for S50; inferred)

- `ccg_build`'s pattern is `~ccg_<s>*`, with flat `+graph.query +rename +expire`. With `<s>` = `shop`, it also matches
  every `ccg_shopify…` key.
- So one system's build credential can write to, rename or expire another system's graphs. This is Redis glob semantics;
  I did not run it.
- The matrix's foreign key, `ccg_p0spike__zforeign`, cannot collide with `ccg_p0spike__acl*`, so the probe could not show this.

**Fix (S50).** Use `~ccg_<s> ~ccg_<s>__*` plus the lock selector, or forbid system names that are prefixes of each other.

#### P3-8 (minor) — the §3.3 rewrite is S17-L2 material bundled into this commit

The new state counts in §3.3 ("98 / 84 / 84 / 82 / 81 / 83", "19 / 5 / 5 / 3 / 2 / 4", "+19 / +21") are not the slice
report §6.8 figures (79 / 65 / 65 / 63 / 62). They have no file in `results/`, and re-deriving them needs a live walk on
`ccg_p0slice`, which this gate may not touch.

Growth-stops: this is rework plus new measured facts, and the plan note has become those facts' only home. The rest of
the note's growth is compliant: §4 rework in place, §5 replaces text with test names, and §6 is the S23 deliverable
(new scope, a table its contract requires).

**Fix.** Put the counts in a result file, or cite the run that produced them. `teco` should route §3.3 to the S17-L2
owner's gate, not this one.

#### Nits

- `acl_probe.SWAP_LUA` is the old `build`-only script, not `swap_probe.SWAP_SCRIPT`. The ACL needs are the same (`EXISTS`, `RENAME`), but the matrix should run the adopted script.
- §4 says `node_batch … timeout 10000` took 0.012 s; the file says 0.013. It also says peak 373 MB; `memoryLog` max is 371.4 MB.
- `timeout_probe` describes cap 800 as "below the uncapped backfill times", but the constraint took 0.206 s uncapped. The full-text index (1.697 s) is the one probe that shows a backfill outliving the cap. The constraint shows it only because it ran slower under the cap (1.16 s).
- `PATCH_QUERY`'s `CASE` keeps a **null** offset null forever (`null < $off` is null). Every `:Snapshot` must carry `verdictLogOffset`, so this belongs in the loader rule (the `:Snapshot` is written last, D16).

### Is "no stall reproduced" an overreach?

The finding itself is sound: 9 single writes, `PING` from a separate process, and the instance answered throughout. The
overreach is only the explanation offered for the incident (P3-1). "About 1 ms of worst `PING` per 1,000 edges" is a
real, linear head-of-line cost at commit. It belongs with the ceiling (P3-2), not with the incident.

### Stopping condition (falsifiable)

This approval stands until any of these happens:
1. A third `test_lock_replay.py -m live` run, or one under concurrent load, fails any of its 8 tests.
2. Any §6 value recomputed from `results/*.json` differs from the table.
3. A re-run of `acl_probe.py` shows any non-empty selector-variant diff.
4. A P1 measurement shows a per-read p99 over 1,000 ms at the `bfs-v1` chunk size under `TIMEOUT_MAX` 10,000. This would break the tenth clause the sizing rests on.

P3-1 should be fixed before S23 is accepted, because the quirks file is read whole by every `graph-dba` session.

### Proposed REVIEW.md lines

Target: root `REVIEW.md`, **Always check**. Source: P3-1, plus Pass 1 B1 / Pass 2's correction (the same class:
a cause asserted without checking the evidence that already exists).
- (addition) An explanation offered for an incident that was not reproduced is checked against the incident's own
  record (its EXPLAIN output, its symptoms). An inference that the record contradicts is a finding.

The Pass 1 and Pass 2 proposals stand as written.

### Appendix D — Pass 3 mutants (scratch copy, `PYTHONPATH` pinned; live runs exclude the ACL-matrix test)

| Mutant | Non-live | Live |
|---|---|---|
| `RELEASE_LUA` → unconditional `DEL` | survives (the fake re-implements compare-and-delete) | **killed** (2 fail) |
| `Lock` ttl silently × 10 | survives | **killed** (5 fail) |
| `Lock.acquire` drops `px` (never expires) | survives | **killed** (5 fail) |
| `PATCH_QUERY` offset made non-monotonic | survives | **killed** (1 fail) |
| `swap` remainder replay removed | **killed** | **killed** |
| `catch_up` skips the last entry | **killed** | **killed** |
| `Log._repair` disabled | **killed** | survives (no live torn-tail test; non-live covers it) |
| `acl_rules` selector: flat `+del` on `ccg_build` | **killed** | — |
| stall sampler pings port 6379 | **survives** | — (no live test) |
| stall sampler interval 5 → 500 ms | **survives** | — |
| stall "during" window → `[t1, t1]` | **survives** | — |
| (P2-m2) `SWAP_SCRIPT` RENAME order swapped | **killed** | **killed** |
