# Review: P1-32b gate harness (`gate.py`, `gate_run.py`) against its spec

> **Status:** active · **Owner:** `analyst` · **Tracks:** C-1001 (P1)

## 1. Scope and verdict

Reviewed: `cpg/eval/tools/gate.py` (commit `fe2dbc70`) and `cpg/eval/tools/gate_run.py` (commit `3c39856e`) with their tests. Spec: `docs/plans/connected-code-graph-p1-gates.md` P1-32 "32b" and P1-37; `docs/plans/connected-code-graph-p1-ml.md` §5-§6; `docs/plans/connected-code-graph-eval-protocol-ml.md` §5.1-§5.4, §12.1-§12.3; `docs/plans/connected-code-graph-answer-spec2.md` §3. Root `REVIEW.md` applied. No held-out data, no custodian dir and no service was read or touched (one probe that would have constructed a FalkorDB client with a default port was denied by the harness and not retried; its result is marked inferred below).

**Verdict: needs changes.** Five blockers (B1-B5). The core hit rule, chain replay, stamp, request echo, floor of 35 and the prereg comparison are sound under my plants. The defects are in the data shape the scorer meets on real store rows, in how a harness error is turned into a gate verdict, and in the once-only and crash-safety design of the run verb, which guards an irreversible spend.

CPG: considered, not relevant — evaluation tooling under `cpg/eval/tools/` has no loaded graph of its own; checked by reading and by execution.

Executed:
- `cd cpg && .venv/bin/python -m pytest tests/test_gate.py tests/test_gate_score.py tests/test_gate_fail.py tests/test_gate_run.py -q` → 161 passed, exit 0. `tests/test_gate_run_wiring.py` → 20 passed, exit 0. `test_gate_live.py` not run (needs a store; I touched no service).
- 40-plus plants on a copy in `/tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/dd493cce-be51-46f4-9ad3-5b22c2dd9c32/scratchpad/copy/` (`plants1.py`..`plants8.py`; `GATE_TOOLS_DIR` points `gate_helpers` at the copy). Nothing in the repo was modified. Results are in the appendix.

## 2. Findings, ranked

### Blockers

**B1. The production `run` path cannot build. Verified by reading.** `gate.build_at_eval_commit` defaults `refresh_fn` to `ccg.refresh.refresh` and calls it as `refresh(system, repo_root=dest)`. `refresh()` raises `RefreshError("no store given…")` when `target is None` (`cpg/ccg/refresh.py:409-410`). `gate_run._default_deps` supplies no `refresh_fn` and no `Target`, so every real run raises after the clone is made, and `gate-build-<sha16>` then blocks the retry ("build directory already exists"; plant R3b). It fails before the marker, so nothing is burned. All 20 wiring tests inject `refresh_fn`, so none exercises the default. Fix: build a `Target` in `_default_deps` from the same host/port/credentials as the snapshot, pass `target=`, and add a test that runs the default `refresh_fn` against a stub `Target` and asserts the call carries `target`. Clean up or reuse a partial build dir on failure, as a documented step.

**B2. The once-only marker is keyed on raw list bytes. Verified (plant R1).** `marker = cust / f"gate-run-{sha256(raw bytes)[:16]}.marker"`. Three byte-different forms of the same records each started a second run: `--lists A,B` vs `B,A`; a trailing blank line; the same records re-serialised with another key order. EP §12.1 step 7 forbids re-running the list that was run; a re-export, an editor save or an argument swap defeats the guard. Fix: key on a canonical list identity, e.g. sha256 over the sorted `itemId`s with each record's canonical JSON (`match_keys.canon`), full 64 hex in the file name or payload (the 16-hex prefix is also a truncated key). Test with the three plants above and a duplicated record.

**B3. A crash after the marker burns the run and loses every answer. Verified (plants R3, R6).** The marker is taken, then `run_asks` collects all answers in memory and `run-<sha>.json` is written once at the end. An exception at question k (a `StoreError`, a `KeyError` in `build_request`, a signal) leaves the marker, no `run.json`, and a refused re-run. `main` catches only `Refusal`, `OSError`, `ValueError`, `KeyError`, `TypeError`, `HarnessError`; a `ClientError` is uncaught (traceback, exit 1, the same code `gate.py` uses for "gate result not a pass"). Worse, `Client` opens its reader lazily inside `ask`, so a missing or rotated engine credential first shows up after the marker (inferred from `clients/common.py:372-387`). See §5 for the precise redesign.

**B4. Scoring crashes on production symbol rows. Verified (plant P10).** The loader stores `ownedRegions` flat (`[s0, e0, s1, e1, …]`: `cpg/ccg/assemble/rows.py:529`, `_flatten_regions`; the engine reads it so at `ccg/engine/anchors.py:165-170`; `eval/tools/baseline.py:316-321` converts flat to pairs for exactly this reason). `StoreSnapshot._symbol_row` passes it through, and `match_keys._regions` does `r[0]`/`r[1]` per element, so for a flat list it raises `TypeError: 'int' object is not subscriptable` as soon as a candidate whose `E(qname)` equals the key's `qname` has owned regions. `score` then exits 4 ("bad input: TypeError"). Every test, including the live one (`tests/test_gate_live.py:69`, `ownedRegions: [[28, 32]]`), uses nested pairs the loader never writes, so the offline and live suites cannot see it. Scoring is re-runnable from `run.json`, so this costs a stop, not a spend, but the gate cannot be scored as shipped. Fix: pair the flat list in `_symbol_row` (and compare the answer's flat copy against the stored flat list as today). Build the test fixtures through the loader's row builder so the shape is the producer's.

**B5. A harness error becomes a miss and the verdict is `fail`. Verified (plant P2 "injection edgeType", P7 "miss+defect").** `replay_chain` returns `error` for an `edgeType` outside the allow-list, a bad `dir`/kind, a class-D hop against a snapshot that lacks the edge type, or a class-X hop. EP §5.2.1 (g)/(h) says such an item is "neither a hit nor a miss" and "blocks the gate verdict". `evaluate_item` folds the error into `best = "H2"` and the item returns `outcome: miss`; `decide_gate` then orders `fail` above `blocked`, so a harness defect reports `fail` and starts the irreversible procedure (cluster to dev, P1-43). Fix: an item whose only unresolved candidates ended in a harness error gets `outcome: "error"` (not scored); any `harnessErrors` yields `blocked` ahead of `fail`; engine defects (HA-2) and masked comparisons keep blocking only a pass.

### Majors

**M1. `decide_gate` requires all five strata for a pass; P1-37 does not run all five.** `strata_ok` loops over `STRATA = (http, mcp, scripts, wiring, flow)`. P1-37 runs flow "only for a language whose flag is true after P1-34", otherwise "flow-unavailable, not closed" with held-out flow unexposed. Plant: a perfect run over four strata gives `minimums-not-met`, never `pass`. Fix: the strata in the run are an explicit input recorded in the list/prereg, `strata_ok` loops over those, and the report names the strata not run.

**M2. `score` trusts the recorded pin, and nothing ties `verify` to `score`. Verified (plant P3).** `score_gate` uses `run["snapshotId"]` from `run.json`; a run file whose pin equals the answers' stamp but not the live `:Snapshot` node scores `hit` with no stop. HA-4 compares the stamp to the id read from the node. Separately `score` does not call `preflight`, does not carry the first-contact label, pins, harness pin or scope block into the report (`build_report(..., context=None)`), and does not refuse a dirty tree. Fix: `score` re-reads `snap.snapshot_id()` and stops on inequality with the file's pin; it requires the `run.json` to carry the preflight block that `run` computed (see M5) and prints it in the report.

**M3. The harness self-pin omits files that run the gate.** `HARNESS_FILES` lists `gate.py, match_keys.py, split_manifest.py, noise.py, oracle.py, ac24_smoke.py, export_kit.py`. It does not list `gate_run.py` (asks and builds), `snapshot_export.py` (`RoRunner`, the only snapshot transport) or anything `export_kit.py`-like that the run imports but never uses. `cpg/eval/tools/` is outside `coreTreeHash`, so an edit to `gate_run.py` after the record changes nothing the prereg compares and `harness_pin` reports clean. Fix: derive the pinned list from the import graph of `gate` and `gate_run` under `cpg/eval/tools/` plus `EVAL_COMMIT`, and test with one planted import per axis (module imported at top level, lazily inside a function, via `sys.path`).

**M4. Unit probe is re-derived from the wrong population. Verified by reading `split_manifest.py:138-142` against `gate.py:523-530`.** EP §5.1 fixes the probe at S47 over U (all in-scope records) from S(u), which includes contract anchors whose provider path lies in the unit (`split_manifest.served_sets`), and clusters the probe with the unit. `unit_probe` picks from the record's own `serves[]` and counts items in the list passed in. They agree only if both populations coincide; on a re-gate list (kept plus fresh) they generally do not, and `serves[]` can omit an S(u) point. The delegate's own test pins the held-out-only reading. Fix: take the S47 probe map (`split_manifest.split(...)["probe"]`, custodian-side) as input and fail closed when a unit item has no entry; keep `unit_probe` only as a cross-check that must equal it.

**M5. Run-time ordering (EP §12.3 "record precedes the first build", the M5 rule) is not enforced, and the label is lost.** `preflight` labels `after-adaptation` when no record commit is found but does not refuse; `run_gate` requires only `firstContact.firstContact` and ignores `pf["label"]`, so a harness edited after the record runs (plant R5: output `label: after-adaptation`, stdout only) and `run.json` carries only `snapshotId` and `answers`. See §4 for the exact additions and §3 for ruling (c).

**M6. The ask request goes through the locator resolver and is unverified before the spend.** `cli_request` is faithful in shape (`parse_request` and `_ANCHOR_KEYS` accept it; test `test_the_cli_request_passes_the_clients_own_parser`) but not in semantics: `Client.ask` re-resolves by locator and takes `res.anchors[0]`. Three differences are visible in the code, none run:
- symbol: `path` is dropped, so a qname defined in two files resolves `ambiguous`;
- contract: `resolve_contract` re-normalises an http key in consumer form (`anchors.py:302-306`); a provider-form `keyNorm` with `{}` placeholders may not hit the exact read and then matches by template, possibly several points;
- config: `resolve_config` matches the stored masked `keyPath` exactly ("a segment the build masked cannot be asked for"), while the request passes the gold's raw `keyPath`.

Each yields a synthesized `not-resolved` miss on a held-out anchor. That is the harness's translation failing, charged to the product as a miss on the first and only run. See ruling (a).

**M7. `--moved-out` and the gate kind are operator inputs. Verified (P7).** `decide_gate(result, flags=None)` is gate 1 and `moved_out=None` means no composition requirement. A re-gate scored without `--moved-out` passes composition vacuously; one scored without `--flags` is judged as gate 1. Fix: an explicit `--kind gate|regate`; for a re-gate, require flags and derive `moved_out` from the prior gate's custodian record, refusing when absent.

**M8. `gate.py score` reads lists from any path and the snapshot factories have no port guard.** `_jsonl(Path(f))` for `--lists` has no custodian check (`gate_run._read_lists` has one; I verified a symlink out of custody, a `..` custodian and an outside list all refuse there). `_default_deps.snapshot_factory` and `gate.py score` build `FalkorDB(host, port=a.port)` with `a.port` and `a.graph` defaulting to `None`, with none of `loader.connect`'s forbidden-port check, and independent of `--system` for the `Client`; the two defaults also differ (`Client` 6390). Reading says the library falls back to its own default port; I could not probe it (denied), so this is inferred. Fix: reuse `clients.common.endpoint()` and the loader's guard for both, derive the graph from the system, and put the `_within_custodian` check on `score --lists` and `--run`.

### Minors

- **m1. Chain length counts Symbol hops (ruling 6).** Equals the engine's `distance` for symbol subjects, reports 0 for a wiring hit (`ConfigEntry` subject, config hop counts 1: spec2 §3.3). Read `x["distance"]` and report the Symbol count beside it.
- **m2. Zero-miss cluster interval is `[1.0, 1.0]` (plant: 58 clusters × 2 items).** The Wilson lower bound beside it is right (159/159 → 0.976, 62/62 → 0.942; both match `-p1-ml.md` §6.1), but the degenerate interval invites misreading; print it only when there is a miss.
- **m3. The CLI never supplies `cluster_of`, `exposure_strata` or `context`.** The report's `clusterUnit` says `anchor` and the 0.938 cluster figure cannot be produced. Add a `--clusters` input (custodian map) before P1-37.
- **m4. HA-3 "reported with `lookups`" is dropped** (`check_request_echo` returns it; `score_gate` keeps only the reason). HA-5 and HA-2 entries are counts only; the method owner has nothing per-item to review. Write the per-item list to a custodian file.
- **m5. `check_repeat_cache_hits` accepts an empty or partial repeat set** (plants: `[]` and repeats 2-5 only both return `stop: False`). Assert the expected repeats 1-5 are present.
- **m6. `verdictLogOffset: 0.0` passes `check_stamp`** (`0.0 != 0` is false; only `bool` is excluded). Use `type(x) is int`.
- **m7. The cache-empty check is a constant.** `run_gate` defaults `cache_len=lambda: 0` and `_default_deps` never supplies one; the structured path has no cache, so the check is true by construction. Say so in the docstring or wire the real counter.
- **m8. A chain hop without `id`, or a non-dict hop, raises `KeyError`/`AttributeError`** (`replay_chain`, `evaluate_item`), aborting the whole score instead of one item's harness error. `AttributeError` is outside `main`'s except list, so it exits 1.
- **m9. Exit 1 is overloaded** (a not-pass verdict and an uncaught traceback). Catch the remaining classes in `main` and give them their own code.
- **nit.** Refusal text prints `REFUSED: REFUSED: …` through `_within_custodian`.

## 3. Rulings on the delegate's decisions

1. **Scope set = `exclusions.paths` + `contentCheck` files (vs `noise.py` paths only): accept.** `eval-manifest.yaml` states the content-check files are "deleted from the kit and excluded here", so the census scope includes them; EP/ML §6.3 say the same frame. `noise.py` (`build_frame`, fed `exclusions.paths` only) is the one out of line; `-p1-ml.md` §6.3 says P1-36 applies the same filter. Owe a one-line change so `noise.py` reads `gate.scope_patterns`, plus a test that both frames agree.
2. **Out-of-scope held-out items excluded from recall as `key-error`: accept, with a labelling fix.** It matches §6.3 ("never scored as a hit, a miss or a wrong-possible item … never dropped"). The report counts them under `outOfScope`. It must call them "proposed key-error, unconfirmed": EP §12.1 step 2 requires the stakeholder to confirm a key-error, and the "recall excluding confirmed key errors" figure is a post-triage step the builder does not produce.
3. **H_s per stratum over all scored items: accept** (AC-5 says "at least 8 items per stratum", `-p1-ml.md` §6.1 reads the 75 against counting items). It is hit plus miss only; void and out-of-scope items leave the count, which is right. Subject to M1 (which strata).
4. **Unit anchor probe: reject as built** (M4).
5. **Precedence `stopped` > `fail` > `blocked`: reject for harness errors** (B5); keep `fail` above `blocked` for engine defects and masked comparisons, because EP §5.2.1 says they block a pass, not a miss.
6. **Chain length: accept for symbol subjects, amend** (m1).
7. **`exposure_strata` not wired: accept for P1-37** (gate 1 has no prior exposure) **and block P1-43 on it.** `exposure_rows` writes `{itemId, hit}` only; per-stratum cumulative recall (EP §12.1 step 6) needs the stratum in each exposure row. Add it to the row now so the first file written is the one P1-43 reads.

**What a stricter reading of the spec changes:** B5 (harness error → not scored, `blocked` first); M1 (strata explicit); M2 (live pin, label in report, dirty refusal in `score`); M4 (S47 probe); the HA-3 `lookups` and HA-5 per-entry listing (m4); cluster figure and context (m3); `score` also refuses unless the `run.json` marker state is complete (§5).

**Ruling (a) — synthesized `not-resolved` document with a snapshot-built stamp.** It cannot inflate recall (it can only add a miss). It can mask: HA-4 is vacuous for those anchors (the stamp is the harness's own), so a wrong graph or system behind the `Client` shows as misses, not as a C5 stamp failure, whenever resolution fails first. And a harness translation fault becomes a gate `fail` on the only run. **It should stop, not score:** resolve every asked anchor that exists in the graph read-only through `client.resolve(spec)` before the marker, require exactly one candidate whose id equals the asked id, and refuse (counts only, by status) otherwise; fixes go on dev anchors. Anchors absent from the graph keep the `anchor-not-in-graph` miss. After the marker, an `AnchorNotResolved` for an in-graph anchor is a harness error (`blocked`), not a miss. Mask the config `keyPath` the way the build did before asking (`gate.mask_keypath`).

**Ruling (c) — preflight requires only `firstContact`, not the harness-pin label.** For an irreversible once-per-list run, refuse unless the harness pin is clean too, with an explicit `--allow-after-adaptation` that writes the label into the marker, `run.json` and the report. The spec allows running as "after adaptation", but the first run's label cannot be repaired afterwards. Also refuse (not label) when `record_commit` is `None`.

## 4. Unbuilt item: M5 (record commit precedes the build), and what must be added

M5 (EP §12.3): the record's commit precedes, in `git log`, the first build; at that build the four fields are recomputed from the working tree with no uncommitted change under `cpg/`. `preflight` does the second half. The first half is not enforced. In `gate_run.run_gate`, before `build_at_eval_commit`:
1. Require `record_commit` found; require `git merge-base --is-ancestor <recordCommit> HEAD` and that no commit between it and `HEAD` touches a non-`cpg/eval/**`, non-doc path of `cpg/` (the four fields already imply this; a failed test names it).
2. Record in the marker payload: `recordCommit`, `HEAD`, each pinned harness file's sha256, `snapshotId`, `startedAt`; the marker file is then the audit proof that the record preceded the build.
3. Write the preflight block (label, pins, harness pin, scope, record commit) into `run.json` so `score` can print it as `context`.

Where: `gate_run.py` (1-3), `gate.py` `score` (read and print the block, refuse if it is missing). Test with a record committed after the build commit, and a harness file edited between record and run.

## 5. Crash-safe marker: precise recommendation for B3

Is "burn on any crash" the spec? EP §12.1 step 7 forbids re-running the gate "on the list that failed". A run that never produced answers produced no result and nothing was scored or shown (the verb prints counts only), so resuming one logical run is consistent; `ac24_smoke.take_marker` already admits "only a human removes the marker, after a harness error". The data-scientist should confirm this reading before it is built. Design:
1. **Marker states.** `started` (O_EXCL, as now, canonical key per B2) → `asked` (atomic rename after the last answer) | `failed:<class>` (written by `main` on any exception, counts and class only).
2. **Persist each answer before the next ask.** Append one JSON line per distinct anchor to `run-<key>.answers.jsonl` with flush and fsync. `run.json` is derived from it at the end.
3. **Resume rule (`run --resume`).** Allowed only if the marker is `started` or `failed:<transient class>`, the canonical list key and the marker's `recordCommit`, harness sha256s and `snapshotId` equal the current ones (the node re-read), and no score output exists. It asks only the anchors missing from the answers file. A recorded answer is never re-asked, so a retry cannot be used to pick a better answer. A deterministic harness error (B5 class, a translation failure) is not resumable; a human removes the marker with a ledger entry, as for AC-24.
4. **Before the marker, exercise everything the asks need:** open the engine reader, read the `:Snapshot` id through it and compare with the `RoRunner` id (this also catches the independent `--graph`/`--system` mismatch of M8), run the resolution dry-run of ruling (a).
5. **`score` requires state `asked`.** It reads only the answers file; it stays freely re-runnable.
6. **Mutants:** kill the process (`SIGKILL`) at ask k of n in a test and assert (i) k answers on disk, (ii) resume asks n-k, (iii) a resume after a harness edit is refused, (iv) a changed list byte order is the same key.

## 6. Checks asked for

**Four-field comparison (plant P1).** `defaultDepthBound` `"2"`, `2.0`, `True` against 2/1; `algorithm.version` `"1.20"`, `1.2` (float), an extra key, a string form; upper-case hash; hash with trailing newline; a missing field: every one named the field in `differs`. Dirty list (even with `clean: True`) → `after-adaptation`. The comparison is sound; `engine_observed` hard-codes `"bfs-v1"` (minor: read it from the engine if it exposes it).

**Chains (P2).** Reversed `dir`, `ON`→`TO`, a skipped hop, an absent id, a first hop that is not the asked anchor, a one-hop chain, a null chain: each `miss`. Injection-shaped `edgeType`, bad `dir`, a lower-case kind: harness error, never interpolated. The replay itself is correct (`a, b = prev, h if dir == "->"`, kind pairs from the 21-string table, `CALLS_NAME` two-edge form). `TO_UNIT`/`Unit` (spec2 §3.5) are absent from `EDGES`/`LABELS`; unreachable for a hit (H1 has no unit branch, a `Unit` hop is always last), so no change now.

**Stamp (P3).** `snapshotId` one digit off, `verdictLogOffset` 1, `"0"`, `False`, a changed hash digit, a missing stamp: each `void` and `decide_gate` → `stopped`. `0.0` passes (m6). Pin source: M2.

**Repeats and cache (P4).** `cacheHit` true on repeat 1 → `stop`; on 3 → `voided`; missing provenance, `"false"`, `0` → hit (fail-closed). Empty set passes (m5). Non-empty cache → `Refusal` before any ask.

**H1 masked (P5).** Two different 24-character keys under one parent mask to one string and score `hit` with `maskedComparisons: 1` and verdict `blocked` (HA-5 as specified). Boundary prefix `.`/`[` accepted, `portsX` rejected. Per-entry listing missing (m4).

**H2-H4 (P6).** Item only as group member → hit; group header with empty members → miss; `possible` status with chain → hit; `dismissed`/missing status → miss at `H1`; confirmed with empty chain → miss at `H2`; `#2` duplicate credited through `E`; line outside span → miss; HA-2 `lineStart`/`qname` altered → hit on snapshot values, defects listed, `blocked`; HA-3 second anchor / `scope` / flipped direction / `ambiguous` → miss.

**Floor and re-gate (P7).** 34 counting fresh + 5 non-counting fresh → `all hits, floor not met` (34); 35 counting fresh → `pass`; the 35th `kept-cluster-fresh` → floor not met; `moved_out` larger than the fresh supply → `composition not met`; no fresh items → floor not met. Void → `stopped`; out-of-scope removes items from H. Gate 1 with a miss → `fail`.

**Anchor not in graph (P8).** A contract anchor absent from the graph is not asked and scores a miss `anchor-not-in-graph`; verdict `fail`. A symbol or config anchor is asked regardless (spec limits the rule to contract anchors).

**Marker (P9).** Second call with the same key refused; two keys sharing the 16-hex prefix are treated as one; byte-different same list: B2.

**Report claims, against EP §5.3 and `-p1-ml.md` §6.1.**
- Recall over items, anchors and clusters is computed; an anchor or cluster counts only if all its items are hits. Item recall uses hits over scored items (out-of-scope and void excluded). Cluster recall needs a map the CLI never gives (m3); the ratio interval uses the anchor as unit and implements P0-ML §7.3 (`R`, `V`, t-quantile, clip).
- Zero-miss Wilson bounds reproduce 0.976 (159) and 0.942 (62).
- Chain length: median and p90 of Symbol hops after the first, per stratum, over hit items (ruling 6, m1).
- No figure can print item ids or content. I fed `build_report` a result with ids `ITEMID*`/`ANCHORID*` through gate 1 and re-gate paths: none in the output. `decide_gate`'s `missClusters` holds anchor ids but the CLI never prints the decision. `write_custodian` refuses a root outside `~/.local/share/ccg-eval` (and the `run` verb refuses a symlink or `..` out of it).

## 7. Custody and dirty tree

- `gate.py` reads held-out data through `verify_heldout` (hashes, caller-supplied dir) and `score --lists` (any path, M8). It writes held-out-derived files only via `write_custodian` (guarded) and prints counts, hashes and classes. Refusal messages carry counts (`renormalize_check` returns a count in `differences`; the ids sit in `diffs`, which the message does not print).
- `gate_run.run` reads lists only through `_within_custodian` (resolved path; verified against a symlink in custody pointing outside, an outside list, and a `..` custodian).
- Dirty tree: `run_gate` refuses an untracked file, an ignored file, a staged change and a `cpg/tests` change under `cpg/` (plants), and allows an untracked file outside `cpg/` (EP §12.3 scope). `gate.py score` and `verify` do not refuse (`verify` returns exit 1 for non-first-contact; `score` ignores it): M2.
- Not run-time guarded: nothing stops `score` pointing at a stray list; nothing stops a second `score` with different flags (it is deterministic and re-runnable by design).

## 8. What is solid

- The hit rule H1-H4 with `E`/`h1_symbol_credit` imported, HA-1 replay with the closed 21-string allow-list (no relationship type ever reaches a query unchecked), HA-2/HA-3/HA-4, and the floor and composition logic held under every plant above.
- Fail-closed defaults on cache hit, stamp, prereg field type (`type(a) is type(b)`), duplicate prereg record, uncommitted record.
- The offline suites are fast (161 + 20 in under 2 s), deterministic and green; the test seams (`DictSnapshot`, `World`, injected `refresh_fn`/`ask`) are well chosen and let me plant without touching code.
- Output discipline: counts only on stdout and stderr, custody enforced on the `run` verb.

## 9. Open questions

1. Does the data-scientist accept the resume reading of EP §12.1 step 7 (§5)? It is a spec interpretation, not a code detail.
2. Is a dry-run `resolve` of held-out anchors before the marker acceptable under the §6.2 contact rule? It yields a status and a candidate count, no link or impact; the run verb is an exposed dispatch already. I believe yes; the method owner should confirm.
3. Does the S47 probe map live in the custodian root today (M4)? I did not read it.

### Proposed REVIEW.md lines

Target: root `REVIEW.md`, heading "Always check". Source: B2, B3, B4, M3 of this review.

- A once-only guard (marker, lock, ledger key) for an irreversible step is tested with planted inputs that keep the content and change the bytes (argument order, trailing blank line, key order, a duplicated record); its key is a canonical identity of the content, not a hash of the bytes or a truncated prefix.
- A procedure that spends something irreversible is tested by killing it at each step after the guard is taken: the answers produced so far are on disk before the next step, and the plan states whether a re-run may resume and under which equalities.
- A fixture standing in for stored data is produced by the writer that stores it (the loader's row builder), or carries a test that feeds the writer's output to the reader. A reader tested only on hand-written rows is a bound, not proof.
- A self-pin of tool files is built from the import graph of the entry points, with one planted import per axis (top-level, lazy, path-inserted); a hand-listed set is a bound.

No line is pruned: none of the existing lines is enforced by these checks.

## Appendix: plant results (scratchpad `copy/tests/plants1.py`..`plants8.py`)

| Plant | Result |
|---|---|
| prereg `defaultDepthBound` `"2"` / `2.0` / `True` vs 1 | `differs: [defaultDepthBound]` / same / not equal |
| prereg version `"1.20"`, `1.2` float, algo extra key | `differs: [algorithm]` each |
| chain: dir reversed / ON→TO / skipped hop / absent id / one hop | miss each (no harness error) |
| chain: `edgeType` injection string / `dir` `<->` / kind `link` | harness error each, miss, verdict `fail` (B5) |
| chain hop without `id` | `KeyError`, whole score aborts (m8) |
| stamp snapshotId, offset 1, `"0"`, `False`, hash digit | void each; offset `0.0` passes (m6) |
| run-file pin ≠ live node, answers agree with the file | `hit`, no stop (M2) |
| repeats `[]`; repeats 2-5 only | `stop: False` (m5) |
| group member only / group header only / `possible` / `dismissed` | hit / miss / hit / miss at H1 |
| HA-2 `lineStart` moved; `qname` altered | hit on snapshot values; defects listed; `blocked` |
| HA-3 second anchor / `scope` / direction / ambiguous | miss each; defect `request-echo` for the first three |
| masked keys (2×24 chars) | hit, `maskedComparisons` 1, `blocked` |
| flat `ownedRegions` `[28, 32]` on a matching candidate | `TypeError: 'int' object is not subscriptable` (B4) |
| 34 counting + 5 non-counting fresh / 35 / 35th kept-cluster-fresh | floor not met / pass / floor not met |
| four strata only, all hits | `minimums-not-met` (M1) |
| miss + engine defect | `fail`, `blockers: [engine-defects]` |
| contract anchor absent | not asked, miss `anchor-not-in-graph`, `fail` |
| marker: same call twice; same 16-hex prefix | refused; refused |
| `run` with `--lists` reordered / blank line / re-serialised | second run allowed ×3 (B2) |
| crash in `ask` / `StoreError` mid-run | marker present, no `run.json`, rerun refused; `StoreError` uncaught (B3) |
| refresh raises, then rerun | "build directory already exists" (B1 side effect) |
| uncommitted prereg record | `REFUSED`, exit 2 |
| untracked / ignored / staged / `cpg/tests` change | refused; untracked outside `cpg/` runs |
| symlink out of custody; `..` custodian; outside list | refused |
| harness file edited after the record | runs; `label: after-adaptation` on stdout only; `run.json` has `answers`, `snapshotId` only (M5) |

## Pass 2

Scope: `gate.py` and `gate_run.py` at HEAD (rounds A, A2, B, commits after `3c39856e`; rule text `docs/plans/connected-code-graph-p1-gate-marker-ml.md`). Executed: the seven gate test files, 437 passed, exit 0; plants `plants1..5` re-run on a fresh copy of HEAD (`scratchpad/copy2`), new `plants9..14` (real `SIGKILL` through `os.fork`, not a stand-in). Live tier not run (the delegates' range starts at 16420; I started no service). **Verdict: needs changes** (one new blocker-class state-machine deadlock and four new majors, all small fixes).

### Dispositions of Pass 1

- **B1 fixed.** `_default_deps` builds `Target(LiveStore(build_db, sys), build_db, verdict_db)` with `acl_user`/`_credential`/`connect`/`Target` signatures matching the real modules (introspected); guards refuse port None/6379/6390/16390/16400/0/70000, a non-loopback host and a `--graph` that is not the system key, before any connection (executed). Not proven live with real ACL users (see rehearsal list).
- **B2 fixed.** Reordered `--lists`, blank lines, re-serialised and duplicated records: refused with one marker; a subset list takes a new key; key is the full 64-hex.
- **B3 fixed, with N1/N5.** `SIGKILL` in ask 1, 2, 3 and during the `run.json` write: marker `started`, answers on disk (0/1/2/3), resume asks only the missing (3/2/1/0). Resume refused on a changed snapshot id, an edited-and-committed harness file, a dirty `cpg/`, a different engine snapshot, and a resolve dry-run that now fails; a run without `--resume` on a started marker is refused.
- **B4 fixed.** Flat `[28, 32]` through `StoreSnapshot` scores `hit` (`region_pairs`). A flat list put directly in `DictSnapshot` still raises; the double is the only path left.
- **B5 fixed at item level, not end to end (Q1).** Injection/`dir`/kind/hop-shape faults give `outcome: error`; miss + error → `blocked`; miss + defect → `fail`; error only → `blocked`, 79 scored.
- **M1 fixed** (four strata perfect → `pass`; a scored flow item while withheld → `Refusal`). **M2 fixed** (`score` re-reads the live id, requires marker `asked`, `listKey`, the preflight block, clean tree), but N3. **M3 fixed** (closure now lists `gate_run`, `snapshot_export`, `baseline`, `export_kit`, `EVAL_COMMIT`; lazy, `__import__`, `import_module`, relative imports followed; `runpy`/`exec` is a bound, none today). **M4 fixed** (S47 probe map; a unit item without an entry is a pre-marker `HarnessError`). **M5 fixed** (record ancestor, no non-eval `cpg/` change since; `after-adaptation` refused unless allowed, label in marker and `run.json`). **M6 fixed**, with N2/N4. **M7 fixed** (kind and `movedOut` from the committed record; the two fields are an extension of EP §12.3, which the ledger discloses; `-p1-ml.md` §8 should name them). **M8 fixed.**
- **m1 fixed** (`distance` plus `symbolHops`). **m2 not fixed** (zero-miss cluster interval still `[1.0, 1.0]`). **m3 not fixed** (no `--clusters`). **m4 not fixed** (no per-item HA-3/HA-5 list). **m5 not fixed** (empty repeat set passes). **m6 not fixed** (`verdictLogOffset: 0.0` hits). **m7 not fixed** (`cache_len` default constant; `_default_deps` supplies none). **m8 fixed** (`hop-shape` error). **m9 fixed** (`gate.py` maps other failures to 4; `gate_run` exits 2/3/4/5/6). **nit fixed.** m2-m7 are disclosed as not done; none blocks a pass.
- **Decisions.** (1) noise.py alignment not done (still owed; disclosed). (2) accept, now labelled `proposed, unconfirmed`. (3) unchanged, accept. (4) fixed. (5) fixed. (6) fixed. (7) fixed: exposure rows carry `stratum`.
- **Rulings (a)-(c).** (a) fixed: dry-run `resolve` before the marker, exactly one candidate with the asked id, counts by status, exit 5; an in-graph anchor that does not resolve after the marker is a `HarnessError`; config key masked in the stored `[#…]` form. (b) fixed (state machine, fsynced per-answer lines, torn-line discard, pin-checked resume). (c) fixed.

### Answers to the two questions

**(1) Post-marker harness error: not realised end to end. Executed (plants11).** `ask` raising `HarnessError` at anchor 2 of 3, an engine `StoreError`, or a `ValueError`: marker `failed:deterministic`, 1 answer on disk, no `run.json`, resume refused, `score` refuses (not `asked`). No `blocked` verdict exists anywhere, and the N-1 answers already recorded are unusable. A `failed:deterministic` marker with a class name is an acceptable record of a **run-wide** fault (credential refused, snapshot changed mid-run, disk full), but for a **per-anchor** fault it discards the evidence and the verdict the ruling requires. `gate_run` must catch per anchor: a `HarnessError` becomes the persisted line `{"anchorId", "harnessError": <class>}` (the shape `score_gate` already reads), the loop continues, `run.json` is written, the marker ends `asked`, and `score` reports `blocked`. An engine exception that is not transient on one anchor (build error, answer fails validation) is a product failure on that anchor, so record it as a miss with reason `engine-error`; this needs the data-scientist's ruling, since the gate-marker note covers only harness faults. Keep `failed:deterministic` for the run-wide class and have `gate_run` print its class and the count of recorded answers.

**(2) Config key mask: the delegate is right (executed).** The loader stores `services.[#…].ports[0]` (`rows._mask_key_path`); `answer.mask_identifier` prints `[#]` for a raw key and leaves a stored `[#…]` unchanged; `resolve_config` matches the stored form. Request side fixed. But see N2: the same two forms diverge in H1.

### New findings

**N1 (major, verified). Crash windows in the marker state machine deadlock a spent list.** (a) `SIGKILL` after `run-<key>.json` is renamed in and before the marker becomes `asked` (`gate_run.py` run block): marker stays `started`, `--resume` refuses ("a run file already exists"), and `score` refuses a non-`asked` marker, so a complete run can never be scored. (b) `take_marker` creates the file with `O_EXCL` and writes the payload afterwards; a kill between them leaves an empty marker: re-run says "already run", resume says "marker is unreadable" (plant: 0 asked, both refuse). The rule that no human removes the AC-5 marker makes both permanent. Fix: (a) on resume, if `run-<key>.json` exists, parses, and its `listKey` and `snapshotId` equal the marker's pins, only finalise the marker to `asked` (ask nothing); (b) write the payload to a temp file and `os.link` it into place (atomic, refuses an existing name), or treat an empty marker with no answers file and no run file as `started` with no pins and let resume fill them. Test with `SIGKILL` at both points.

**N2 (major, verified). H1 on a config item with a screen-failing key segment is a false miss.** Plant: stored `services.[#…].ports`, gold `services.<24 a's>.ports`: `outcome: miss`, `maskedComparisons: 1`. `_h1` masks both sides with `gate.mask_keypath` (`[#]` form), but the store, and so `real["keyPath"]`, holds the `[#…]` form, so `[#…]` is never equal to `[#]`. HA-5 expects the comparison to be listed and judged, not to fail silently. Fix: mask the gold with the loader's `rows._mask_key_path` (as `cli_request` now does), compare in the stored form, and list the entry. Plant with the loader-masked store value (`plants10.py`); gold already in stored form hits.

**N3 (major, verified at function level, read at CLI). `score` recomputes the flow flag instead of using the run's.** `run.json` records `flowFlag`, `withheld` and `listKey`, and `score` reads none of the first two; it calls `flow_flag` again. If the committed decision flips to TRUE between run and score, flow items that were never asked score `miss: anchor-not-in-graph` and the verdict is `fail` (plant: flow item absent from the run → miss, `fail`). Fix: gate on `run["flowFlag"]`, refuse a mismatch with the current record, and put `flowFlag` and the withheld counts into the marker pins so a resume cannot change the asked set.

**N4 (major, inferred from reading; needs the rehearsal). The gate build can overwrite the developer's `ccg` work area.** `refresh(system, …)` requires `system` equal to the clone's manifest system, which is `ccg`; `_default_deps.refresh` passes no `work_root`, `keys_root` or `audit_dir`, so bundles, unit-state files and the retention pass (`prune_unit`, `prune_orphans`, `sweep_stale_tmp` in `ccg/atrest.py`) run in the default work area for `ccg`, shared with the dev system. Fix: pass `work_root=<custodian>/gate-work-<key>`, a private `audit_dir`, and refuse a `--port` whose store already holds a `ccg_<system>` graph.

**N5 (minor, verified). The ask key path can reject gated anchors the dry-run will catch.** `http_key(…, "consumer")` changes provider-form keys that carry `{?}`, `{#}` or a `?query` (`GET /x/{?}` → `GET /x/{`; `…/x?y=1` → `…/x`), so the exact read misses; the dry-run refuses (exit 5, nothing spent) and the cluster goes to dev. Safe, but it can shrink a stratum below 8: run the dry-run over dev anchors and count first.

**N6 (minor, read).** No lock on the answers file or marker: two live `--resume` processes both pass the `started` check and append duplicates. The marker holds no hash of the probe map or the answers file. The eval manifest (scope) and the flow decision are read from the working tree, unpinned. `cache_len` is still a constant (m7).

### Cross-fix interactions checked

- Marker name: `first_run_marker`, `_marker_path` and `marker_state` all use `gate-run-<64-hex>.marker`; `gate_run` keys over all records (withheld included) and `score` keys over `--lists` before `withhold_strata`: equal.
- `run.json` fields `score` requires (`preflight.label`, `recordCommit`, `listKey`, `snapshotId`, `probes`) are written by `run_gate`; a resume writes the same block.
- Resume re-runs the dry-run, the cache check (≤ recorded+1), and the pin equality; it skips the build and reuses the store.
- `snapshot_graph`'s forbidden ports apply to the one `--port` used for build, snapshot reader, engine client and `raw` cross-check; a rehearsal port 16430+ passes.
- Import-closure pin covers the new modules; non-`.py` inputs are not pinned (N6).

### Dress rehearsal on dev anchors: what it must cover that the offline tests cannot

Run the whole `run` verb with the real default deps, dev records only (a dev list and a dev probe map under a throwaway custodian area), on your own store at 16430 or above:
1. **Real ACL users and credentials.** Provision `ccg_build_<s>`, `ccg_verdict_<s>`, `ccg_engine_<s>`, `ccg_raw_<s>` on the throwaway with a private `CCG_CREDENTIALS_DIR`; the delegates' live tier uses the default user only. Check that the snapshot reader (plain `FalkorDB(host, port)`, default user) still works or is switched to a read-only role, and that `raw_query`'s `:Snapshot` cross-check passes.
2. **A real `build_at_eval_commit` end to end:** clone at a pinned commit, refresh into the throwaway through the real `Target`, with a planted dev bundle in the default work area to prove N4 (it must survive).
3. **The resolve dry-run on every dev anchor kind** (symbol by location, config with a masked segment, http contract incl. `{}`, `{?}` and a `?query`, an mcp contract, a unit probe): status counts per class, expecting `ok` for all in-graph anchors.
4. **The real engine answer through `score`**: loader-written `ownedRegions` flat, a wiring hit with a masked key (N2), a possible-status hit, a group-member hit, a chain with `CALLS_NAME`, an engine-stamp `snapshotId` equal to the node.
5. **A real `SIGKILL` of the `run` process** at ask k, at the run-file write and at the final marker write (N1), then `--resume`; and a `SIGTERM`.
6. **A planted per-anchor fault** (an anchor whose ask raises) to prove the Q1 behaviour once built, and a planted C5 failure (a stale engine) to prove `stopped`.
7. **Exit codes and streams** from the real process: 0/2/3/4/5/6, and a content scan of stdout and stderr for the dev gold, ids and paths.
8. **Flow flag both ways** (N3) and the first-contact / `--allow-after-adaptation` labels in `run.json` and the report.
9. **`score` afterwards** on the same custodian root: marker `asked`, listKey, live-id re-read, dirty-tree refusal, report with no ids.
10. **Timing and memory**: wall time of the build, the dry-run and the asks at dev scale, against the query timeout of `raw_query`.

### Proposed REVIEW.md lines (Pass 2)

Target: root `REVIEW.md`, "Always check". Source: N1, N2, N3.
- A state machine written to disk is checked for every state reachable by a kill between two of its writes: for each, a planted real `SIGKILL` and an assertion that the next command can finish or refuse with a way forward. A state from which no command leaves is a finding.
- A comparison that normalises both sides is run with the producer's own normalised value on one side, not two raw strings; a test that masks both sides alike cannot see a producer that masks differently.
- A value recorded in a run file that a later step recomputes (a flag, a stratum set, a decision record) is read from the run file, and a mismatch is refused.

## Pass 3

Scope: round C (`38963ccd`) over `gate.py` and `gate_run.py`; baseline `1c00b943`. Executed: the eight gate test files, 491 passed, exit 0; plants `plants9..12`, `14` re-run on a fresh copy of HEAD (`scratchpad/copy3`); new plants `15..18` (real `SIGKILL` through `os.fork` at the marker link, the final marker write and mid-ask; an exception table through `make_ask`; run plus `score_gate` end to end); 16 mutants of my own (`scratchpad/mutate.py`: finalise snapshot and `listKey` checks, `flock`, harness-as-engine precedence, score's flow cross-check, marker overwrite, resume cache check, engine-error as `error`, private dirs, live-graph refusal, probe-map and flow pins, old keyPath masking, withheld check, ENOSPC carve-out) each run against the eight files with `GATE_TOOLS_DIR`: 16 of 16 failed a test (killed). **Verdict: needs changes**: two majors (P3-1, P3-2), both small; everything else from Pass 2 is fixed.

### Dispositions

- **N1 fixed.** (a) `SIGKILL` before the marker becomes `asked`: `--resume` finalises to `asked`, asks nothing. (b) `SIGKILL` before `os.link`: no marker, next `run` proceeds; after the link: complete payload, `--resume` proceeds. Not reachable by a kill any more (the empty marker of `plants14` is hand-made). Finalise after a committed harness edit is refused by the label check unless `--allow-after-adaptation` (then it finalises, label kept).
- **N2 fixed.** Stored `services.[#…].ports` against a raw gold key: `hit`, `maskedComparisons: 1`, so the HA-5 `blocked` applies; gold already in stored form: `hit`, 1.
- **N3 fixed, and the delegate's design is better than my proposal** (below). A flow flip after the run no longer changes what is scored; marker pin and withheld counts are cross-checked.
- **N4 fixed.** `work_root`, `keys_root`, `audit_dir` reach `refresh` under the custodian root, mode `0700`; a store that already holds the system's live graph is refused before the build (nothing asked, no marker); the check is skipped on resume.
- **N5 open by design** (the dry-run is the guard; unchanged). Count the `{?}`/`{#}`/`?query` anchors on dev first.
- **N6 fixed in part.** Fixed: `flock` per list (a held lock refuses; a `SIGKILLed` holder releases it); probe-map sha256 and the flow flag pinned (a changed probe map or flow decision refuses the resume). Still open: no hash of the answers file in the marker; the eval manifest and the flow decision file are read unpinned; `cache_len` is still a constant (m7).
- **Q1 fixed end to end.** Through `make_ask`, run and `score_gate`: engine `ValueError` on one anchor → `miss/engine-error`, `fail`, marker `asked`; `BadRequest` → `error/HarnessError`, `blocked`; both → `blocked`; an engine-error entry is never re-asked on resume (recorded line, then `SIGKILL` at the next ask: the resume asked 2 anchors, not 3).
- **Pass 1 m2-m7** unchanged (open, disclosed).

### Fixes against each other and against A/A2/B

- Marker finalise vs pins: finalise compares `run.json`'s `listKey` and `snapshotId` with the marker only and runs after the preflight, so a harness edit is refused first and a changed live snapshot finalises (it asks nothing; `score` re-reads the live id and refuses). Acceptable.
- Lock vs marker: the lock file (`gate-run-<key>.lock`) is taken before the marker check, released on exit and on process death; it is not a marker and no glob over markers matches it.
- Private build dirs vs resume: created only on a fresh run, after the marker check, so a spent list's dirs are never removed by `_clean_leftover`.
- Engine-error vs harness-error precedence: `blocked` wins in `decide_gate`; a miss beside a harness error is not triaged (matches the ruling).
- `score` vs marker pins: `run["flowFlag"]` and `run["withheld"]` must equal the marker pin and the lists' own withheld counts; `listKey` and marker `asked` unchanged.
- **Gap (P3-2): `run` does not evaluate the refusals `score` can already know.**

### New findings

**P3-1 (major, verified). ENOSPC/EDQUOT inside an ask is recorded as an engine-error miss.** `ask_failure_entry` returns `None` only if `classify(inner) == "transient"`; round C made `classify` return `deterministic` for `OSError` with `ENOSPC`/`EDQUOT`, so such an error raised inside `client.ask` falls through to `EngineAskError` and is persisted `{"outcome": "engine-error", "exceptionClass": "OSError"}` (plant: `OSError(ENOSPC)` on anchor 2 → `miss/engine-error`, verdict `fail`, marker `asked`). The ruling lists "disk" among the run-wide faults that are **not** engine errors (§3, "Which exceptions"), and §1.1.3 classes `OSError` transient. A full disk on the gate host would therefore write a permanent miss on a held-out anchor and start the fail procedure. Fix: `ask_failure_entry` returns `None` for any `OSError` (and `MemoryError`, `RecursionError`: a process-level fault is not placed on the engine); run-wide then ends `failed:transient`, resumable once space is freed. Plant that reddens it: the ENOSPC ask above, expecting propagation and no entry.

**P3-2 (major, verified). A record `score` refuses does not stop `run`.** `gate_run` never calls `gate.gate_kind`; with a committed `ccg-prereg-v1` record that lacks `kind` (or a re-gate without usable `movedOut`) `run` takes the marker and asks, then `score` refuses with "does not state a valid gate kind" (plant: record without `kind` → `run` ok, one marker; `gate_kind` raises). The `ac5-gate-1` record is not committed yet and its `kind` field is an extension of EP §12.3, so this is live. Fix: before the marker, `run` evaluates every refusal of `score` that depends only on pre-marker inputs: `gate_kind` on the committed record (and `--flags` vs kind), the eval-manifest parse, the flow record, the lists' withheld set. Make it a single function both verbs call. Plant: the record without `kind` → `run` refuses with nothing spent.

**P3-3 (minor, read). A deterministic timeout on one anchor can never be recorded.** A `TimeoutError`/socket timeout from the engine is `transient`, so the run ends `failed:transient`; a resume re-asks the same anchor and times out again, without limit. The data-scientist's text calls a timeout transient but says nothing on a repeat on one anchor. Open question for the data-scientist: after k consecutive transient stops at the same anchor (counted in the marker, not by the operator), record `engine-error` (class `TimeoutError`)? Without it a hanging product holds the list in `failed:transient` forever, with no verdict.

### The two disagreements

**N3: `score` uses `run.json`'s flag and never reads the current decision. Accept; my proposed refusal on mismatch is withdrawn.** My proposal would refuse scoring exactly when the committed decision flips after the run, which is the case N3 was about: the run is complete and spent, so refusing would deadlock it. The delegate's rule has the integrity the refusal was meant to give: `run["flowFlag"]` must equal the marker's pin (taken before the first ask, atomic), `run["withheld"]` must equal the withheld counts recomputed from the lists, and a run file without both is refused. The current record can differ without effect on what was asked, and the report need not read it. Mutants `d` and `o` (drop either cross-check) were killed.

**ENOSPC/EDQUOT carved out of `transient`: reject (P3-1).** The reasoning "a full disk fails the same way on a resume" holds only if nobody frees space; the resume rule is built on equal pins, and freeing disk changes none. Marking it `deterministic` spends the list irreversibly on a recoverable fault, and inside an ask it also turns the fault into a recorded miss. If the marker cannot be written on a full disk it stays `started`, which is already resumable, so the carve-out buys nothing. Restore `OSError` as transient per §1.1.3 and make the in-ask policy propagate it. The test that pins the carve-out (mutant `k` is killed by it) must be inverted.

### The engine-error ruling against the code (`-p1-gate-marker-ml.md` §3)

Matches: `outcome: miss`, reason `engine-error`, `fail` tier, in the recall denominator; class name only (`exceptionClass`; no message); classified by where it was raised (`EngineAskError` wraps `client.ask` only; `cli_request` and the dry-run are outside it), not by type; never re-asked on resume; `blocked` beats it; reported on its own line (`engineErrors`: count, by stratum, by class, never folded into harness errors); `BadRequest` and an unresolved in-graph anchor are harness errors. Deviations: P3-1 (disk placed on the engine side) and `MemoryError`/`RecursionError` (a process fault, same fix). Not implemented, and not required of the harness: the `linker-engine` triage entry and EP §5.2.1 row HA-6 (owed by the data-scientist). One more check the ruling does not cover: `NotBuilt` and `NoCredentials` propagate run-wide, which I judge right.

### Stopping condition for further static passes

No further static pass is justified until (1) P3-1 and P3-2 are fixed, each with a plant that is red on the current tree and green after, and the ENOSPC test inverted; (2) the dev dress rehearsal of Pass 2 has run with zero unexplained harness errors and its `score` produced a report; and (3) a coverage test enumerates the axes `run` varies on from the code's own registries, not from a hand list: every `_write_marker`/`first_run_marker` call site crossed with a `SIGKILL` point; every exception family (`OSError` by errno, transient, `ClientError` subclasses, `BaseException`) crossed with the `ask_failure_entry` result; every `Refusal` reachable in `gate.main score` crossed with whether `run` also raises it pre-marker. The claim is falsifiable: it fails if any cell of those three grids has a plant that reddens. After that, a static pass is only warranted by a diff to a file in `gate.HARNESS_FILES` (the closure), and ends after two consecutive passes with no new blocker or major and a green rehearsal. Pass 3 found two majors in 491 tests and 16 killed mutants, so the remaining risk is in what offline tests cannot reach (real ACL users, a real build, the real engine's answers), which is the rehearsal's job, not another read of the code.

### Proposed REVIEW.md lines (Pass 3)

Target: root `REVIEW.md`, "Always check". Source: P3-1, P3-2.
- A classifier that decides whether a failure is the actor's (a recorded miss) or the environment's (retry or stop) is checked with one planted exception per environment family (disk full, out of memory, permission, timeout, interrupt) and with each tier's ruling text beside it; a carve-out from a family the ruling names is a finding unless the ruling is amended.
- Every refusal a later command can raise from inputs already known before an irreversible step is raised by that step's own preflight: the checker lists the later command's refusals and plants the offending input once for each.

## Pass 4

Scope: `fde55826` (round D), `50d059ef` (round E), `cc53b3de` (engine D5; `cpg/ccg/engine/anchors.py`), read with `git show <sha>:<path>`; callers `cpg/ccg/front/resolve.py:260`, `cpg/clients/common.py:223`. Root `REVIEW.md` applied; `cpg/REVIEW.md` does not exist. Executed: the ten gate test files plus `test_engine_anchors.py`, 670 passed (577 for the gate files on my copy), exit 0; live tier `test_gate_rehearsal_defects_live.py`, 3 passed; my own throwaway FalkorDB `v4.18.11` on 16431 (requirepass, real ACL users, removed); 47 mutants of my own over the gate files (38 killed, 9 survived: 3 equivalent, 6 are the gaps in P4-5) and 5 over the engine (4 killed); a 50,000-comparison old-vs-new differential of `resolve_contract`; real `docker pause` of the store under a real `Client.ask`. Scratchpad `p4/` (`live_reader.py`, `live_stall.py`, `sweep_real.py`, `diff_engine.py`, `env_exits.py`, `mutants.py`). No held-out data, no custodian dir read. **Verdict: needs changes**: two majors (P4-1, P4-2), both in how the real `Client`'s exceptions meet round D's classifier; the round E and engine fixes hold.

CPG: considered, not relevant — evaluation tooling under `cpg/eval/tools/` and an engine function read whole; checked by execution.

### Dispositions

- **P3-1 fixed** (`OSError` incl. ENOSPC, `MemoryError`, `RecursionError` propagate, never an engine-error; mutants D1, D3, D4, D20 killed; sweep below: ENOSPC `propagate (failed:transient)`). **P3-2 fixed** (`score_inputs_check`, one function both verbs call; mutants D14-D17, D19, D21 killed; D18 is P4-5).
- **P3-3: implemented, unreachable in production** (P4-1). **Pass 3 stopping condition**: (1) met; (2) the rehearsal report plus the delegate's no-workaround rerun (own observation; I did not rerun the 16-21 minute build) and my live tier and probes; (3) the coverage grids exist but their exception axis is hand-built (builtin classes), which is why P4-1 and P4-2 passed them.
- **Round E, executed.** D1: the reader opened by `reader_db_factory` is read-only (D1 item 3 below). D2: working-tree manifest passed and `manifestSha256` pinned (mutants E9, E10, E11, E20, E22 killed). D3: `CCG_AUDIT_DIR` restored on success, `Exception`, `KeyboardInterrupt`, `SystemExit`, `GeneratorExit`, the SIGTERM `BaseException`, and the non-fresh refusal, each with the variable unset, set and set-empty (18 cells, all restored); a `SIGKILL`ed forked child leaves the parent's environment unchanged. D4: `STALL_DEADLINE_S` is the engine's 10 s, accepted live by the store. D6: see P4-3. D7: marker check first (mutant E19 killed).
- **Item 3, the read-only reader and the probe, executed live.** As `ccg_raw_<s>`: `graph.query CREATE`, `GRAPH.DELETE`, `GRAPH.COPY`, `RENAME`, `SET`, `FLUSHALL`, `GRAPH.LIST`, `CONFIG SET`, `ACL SETUSER` and a read of another graph key are refused by the ACL; `ro_query` of `CREATE`/`MERGE`/`SET`/`DELETE`/`CREATE INDEX` is refused by the server; the live graph was intact afterwards (44 nodes). `live_graph_present`: absent and build-key-only → `False`; present → `True`; a string at the live key (`WRONGTYPE`), a wrong password, no credential, the `verdict` role (`NOPERM`) and a dead port all **raise**. The claim holds for each family; its offline pin does not (P4-5).
- **Item 4, `--system` default.** `a.system or manifest_system(repo) or "ccg"` reads the working tree's manifest (`graphmind-ai-lab`), and `run` refuses an explicit system that disagrees (E20 killed). No silent mis-pick found: `score` has no `--repo` and derives from the script's own checkout, so a `run --repo X` with another manifest makes `score` look for another graph key, which fails loudly (exit 4). Nit only (P4-7).
- **Item 5, the engine fix.** Order normalised → verbatim → templates holds; kind filter holds (M7 killed); no re-read when normalisation changed nothing. Differential over 50,641 asked/stored pairs (stored keys built by `http_key(.., "provider")`): `unresolved → resolved` 922 (the fix); `ambiguous → resolved` 77; `resolved → resolved` with the same id and `via` `path-template → exact` 91; **`resolved → resolved` with a different id 74** (old: the mangled key template-matched an unrelated stored key, e.g. asked `GET /{#}/shop/api` resolved to `GET /{}`; new: its own point). Both callers consume `Resolution` unchanged. So keys that "previously resolved" do change, toward the asked point, and `via` changes; see P4-4 for what is unpinned.

### New findings

**P4-1 (major, verified live and by execution). The repeat-stall rule cannot fire on the real engine's hang.** `_Persisting.__call__` converts only when `isinstance(exc, TimeoutError)` (`gate_run.py:371`). The real `Client.ask` wraps everything the reader raises into `StoreError`. Real run (`live_stall.py`, `docker pause` of the store, reader socket timeout 3 s): the exception is `StoreError ← ReaderError ← redis TimeoutError ← builtins.TimeoutError`; `classify` = `transient`, `isinstance(exc, TimeoutError)` = `False`. Three stops fed through the real `_Stall`/`_Persisting` with a canary that answers: `re-raised StoreError → failed:transient`, `stall` field `None` each time, no entry written. The same three stops with a planted builtin `TimeoutError` (what the rehearsal's `FAULT` and `tests/test_gate_run_crash.py` plant) count 1, 2, then record. So a hanging product holds the list in `failed:transient` forever, the case §1.1.3 and D4's own text exist to prevent; D4 fixed the canary and left the trigger dead. The delegate's `store_timeout()` (`test_gate_run_crash.py:627`, "a store error that WRAPS a timeout ... never converts", parametrised into `test_p33_only_a_timeout_converts...` and `test_p33_a_non_timeout_failure_never_converts...`) pins the opposite of the only form that occurs. Fix: test the chain, not the outer type: walk `__cause__`/`__context__` (bounded) for a `TimeoutError` by name, stopping false on any `ConnectionError`; keep `exceptionClass: "TimeoutError"`. I ran this against the live chain: stops 1-2 count, stop 3 records. Invert the two `store_timeout()` parameters; the data-scientist should confirm the ruling covers a timeout that reaches the engine boundary as `StoreError` (my reading of "raised inside the engine's `ask`").

**P4-2 (major, verified by execution). A refused credential reaches the run as an engine-error miss on every remaining anchor.** Round D's credential row tests `type(inner).__mro__` against `CREDENTIAL_FAULTS` (`gate_run.py`, `ask_failure_entry`), and `inner` is the outer `StoreError`, so the chain's `AuthenticationError` is never seen. Sweep through the real `CypherReader` + `Client` + `make_ask` + `_Persisting` (`sweep_real.py`, a db whose `ro_query` raises X on every call, three anchors):
```
raised by the db        Client raises  classify        make_ask policy
redis ConnectionError   StoreError     transient       propagate (failed:transient)
redis TimeoutError      StoreError     transient       propagate (failed:transient)   <- P4-1
BusyLoadingError        StoreError     transient       propagate
OSError ENOSPC          StoreError     transient       propagate
'Query timed out'       StoreError     deterministic   engine-error on the first stop
AuthenticationError     StoreError     deterministic   engine-error x3 -> run completes, score `fail`
NoPermissionError       StoreError     deterministic   engine-error x3
ReadOnlyError           StoreError     deterministic   engine-error x3
```
A password rotated or a role disabled during a run spends the one-shot list as a product failure; §3 lists a refused credential as run-wide. The pre-marker dry-run narrows the exposure to a change mid-run, not to zero. Fix: check `CREDENTIAL_FAULTS` over the whole chain (as `classify` already walks it). With that change `AuthenticationError` propagates (`raised StoreError` x3, nothing on disk; run against the sweep). `NoPermissionError` and `ReadOnlyError` (a mis-provisioned ACL, a replica) are not in the ruling's list and stay misses; the data-scientist should rule whether they are run-wide (my recommendation: yes, both are the store's state, not the product's).

**P4-3 (minor, verified by execution). The operator remedy covers 3 of the 9 exits between the build and the marker.** `STORE_REMEDY`'s comment says it is "shared by every refusal that can follow the build". Plants through `R.main` with injected deps, build run, no marker taken, store stranded (every fresh rerun is refused by the live-graph probe): engine-snapshot-id mismatch (exit 2), cache not empty (exit 2), and a `ConnectionError`/`RuntimeError` from `snapshot_factory`, `engine_snapshot_id`, `resolve_factory` or `ask_factory` (exit 4 "bad input: ConnectionError") all print no remedy; a transient fault there reads as "retry", and the retry is refused. Fix: append `STORE_REMEDY` to the two refusals and to the exit-4 line when the build has run (one flag set after `build_at_eval_commit`), or let a no-marker rerun reuse the built store when its snapshot id equals the one the build produced (the owner's call; the second saves 16-21 minutes).

**P4-4 (minor, verified by mutant). The engine's order "verbatim before templates" is unpinned.** Mutant M2 (verbatim read moved after the template fallback) passes all 102 tests. A plant that reddens it: stored `GET /api/items/{}` and `GET /api/items/{#}`, asked `GET /api/items/{#}`: tree resolves `{#}` via `exact`; M2 gives `ambiguous` (executed). Add it, plus one `via`/id assertion for the changed template case (the 74 and 91 above), and say in the docstring that a previously template-resolved stored-form key now resolves exactly.

**P4-5 (minor, verified by mutant). Six round D/E checks cannot fail.** Survivors (not equivalent): E7 `units == []` trusted by `require_complete_build` (empty list passes; only a missing attribute is tested); E21 `gate.py score` default system `"ccg"` instead of the manifest's (the D2 default is tested only on `_default_deps`); E23 `_default_deps` ignoring `--repo`; D18 the eval-manifest-unusable refusal in `score_inputs_check` (no test plants an unparseable `eval-manifest.yaml`); E14/E16 the probe reading any `ResponseError` (`WRONGTYPE`, `NOPERM`, a timeout), or any message containing "empty key", as absent (the offline tests pin only the blanket and `AuthenticationError`; live behaviour is right, above). Equivalent survivors: D2, D11, D13. Each gap is one plant: an empty unit list, `main(["score", ...])` with no `--system` and a manifest naming another system, `--repo` pointing at a repo with another manifest, a bad manifest file, and three raising `ro_query` doubles (`ResponseError("WRONGTYPE")`, `ConnectionError`, a plain `Exception("empty key")`).

**P4-6 (nit, verified).** `_audit_dir_env` is not restored if a signal handler raises inside its own `finally` between the two statements (planted: variable left set). The CLI exits right after, so no developer write is redirected; it matters only to a caller that runs `main()` in-process.

**P4-7 (nit, read).** `gate.py score` has no `--repo`, `run` does; `run --repo X` and `score` from another checkout derive different systems and `score` fails loudly (exit 4). The text at `gate_run.py` "at EVAL_COMMIT there is none" goes stale when `EVAL_COMMIT` moves past `ffeac33f` at P2.

### Item 7: assumptions that went stale with the narrowing (no change asked here)

`STRATA_GATED` (four strata), `MIN_COUNTING = 75`, the withheld-strata machinery and the `ac5-gate-1` record (not committed: `grep -c ac5-gate-1 cpg/eval/preregistration.jsonl` = 0) all describe a P1 http+mcp run that will not happen; at P2 the record must pin the files as they are after the fixes to P4-1 to P4-3, so these changes should land before it, not after.

### Stopping condition (falsifiable)

No further static pass until (1) P4-1 and P4-2 are fixed, each with a plant that is red on this tree: the real `Client` over a `CypherReader` whose db raises the nine families of the table, through `make_ask` and `_Persisting` and the marker, with the expected disposition of each row asserted (rows a ruling does not yet fix are named in the test as open, not omitted); (2) the exception axis of that grid is produced by the real `Client` under fault injection at the reader's db, not by hand-built classes: the claim fails if any row's recorded disposition differs from its ruling; (3) P4-3 and the P4-5 plants are in. A pass after that is warranted only by a diff to a file in `gate.HARNESS_FILES`, and ends after two consecutive passes with no new blocker or major. Pass 4 found two majors in 577 gate tests and 38 killed mutants of 47, both in the one place where the tests' exceptions were not the production ones; the next risk is the same seam on the success side (answers the real engine returns), which only a P2 rehearsal with the engine as it will be then can reach.

### Proposed REVIEW.md lines (Pass 4)

Target: root `REVIEW.md`, "Always check". Source: P4-1, P4-2.
- A classifier of failures is run with the exceptions the production client raises, produced by injecting the fault at the client's lowest layer (a double for the database driver) and passed through the production wrappers; a rule keyed on the outermost exception's type is checked against the cause chain. A grid built from hand-made builtin classes is a bound, not proof.
- A rule written for a named exception ("a `TimeoutError`") is checked for whether the production path can ever raise that type at the place the rule reads it; a test that pins the production form as "never converts" is a finding.
