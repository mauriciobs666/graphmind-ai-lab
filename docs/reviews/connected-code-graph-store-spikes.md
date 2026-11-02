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
