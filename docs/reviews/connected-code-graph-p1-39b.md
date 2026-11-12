# Review: C-1001 P1-39b, the answer builder (`cpg/ccg/engine/answer_build.py`)

> **Status:** active · **Owner:** `analyst` · **Tracks:** C-1001 (P1)

## Scope & verdict

Reviewed, all uncommitted: `cpg/ccg/engine/answer_build.py` (new); `cpg/tests/test_engine_answer.py`, `test_engine_answer_v18.py`, `test_engine_answer_live.py`, `engine_answer_helpers.py` (new); `cpg/tests/test_engine_group.py` (N1 tests); docstring edits in `cpg/README.md`, `cpg/ccg/engine/{__init__,bfs,group}.py`; spec amendments in `docs/plans/connected-code-graph-answer-spec2.md`. Baseline: spec2, `docs/plans/connected-code-graph-p1-answers.md` P1-39, the P1-39a module `cpg/ccg/engine/staleness.py`, the P0 fitness report §a1.2 (re-read). Root `REVIEW.md` read and applied; **`cpg/REVIEW.md` does not exist**.

**Verdict: needs changes.** One blocker (a mangled spec2 table row, trivial to fix). The builder code itself I found correct: no code bug traced. Two majors are test gaps found by mutation (both survive the delegate's own mutant lists).

CPG: considered, not relevant — no loaded `cpg_*` graph indexes `cpg/` (`GRAPHS` lists only `cpg_falkorchat` and `ws:*`); code read directly.

Ran (my own copies of `ccg`, `tests`, `packs`, `infra`, `pyproject.toml`, plus the stored s14a slice bundles, under the session scratchpad; the real tree untouched): answer + v18 + group suites 366 passed / 3 skipped in a copy that lacks `.ccg-work` (the 3 are the slice tests), the v18 file alone in the real tree 8 passed with the slice tests running; `tests/test_engine_*.py tests/test_answer.py tests/slice/test_engine_equivalence.py tests/test_render.py` in the real tree 2016 passed, 128 deselected (live). Live: own throwaway `ccg-falkordb-tw-16390`, scratch `CCG_CONFIG_DIR`, `CCG_REQUIRE_LIVE=1`: `test_engine_answer_live.py` 11 passed; throwaway removed by exact name; `falkordb-dev` and `ccg-falkordb` not touched. Note: the scratch config dir already held a `falkordb-admin.pass` dated before this run (not created by me); I used it only for the throwaway.

**Delegate's six "equivalent" mutants:** I was not given the six diffs. I instead re-ran the delegate's own mutant files (`scratchpad/mutants1-9.py`, 147 mutants) in my copy: 139 killed, 8 survivor lines (7 distinct). Five are equivalent (derived below); **two are not** (F2); that is the "member-Link" claim, now disproved.

## Findings

### F1 · blocker · spec2 §6 table row destroyed (verified: `git diff`, read lines 446-449)

`docs/plans/connected-code-graph-answer-spec2.md:448` is now a stray fragment starting "(depends-on: the hop is the chain's last; impact: ... An item further C6 is closed here. C4's rule is §4, and its implementation is P1-22's |". The N1 edit spliced the tail of the §5.4 row into the old row `| Carry-over C4, C6 | C6 is closed here. C4's rule is §4, and its implementation is P1-22's |` (present in `HEAD`, line 447), deleting its cell. Effect: the replaced-text list loses the C4/C6 closure row and the table breaks (an unterminated duplicate of the N1 sentence sits between rows). The N1 row itself (line 445) is correct. **Fix:** delete line 448 and restore `| Carry-over C4, C6 | C6 is closed here. C4's rule is §4, and its implementation is P1-22's |` after the stamp row.

### F2 · major · the member-Link survivors are not equivalent; the shape is untested (verified: built the counter-shape, ran it)

Two delegate mutants survive the full suite: `stale-evidence-units-from-members-dropped` (the `evidence_units` set built from `items` only, `answer_build.py:402`) and `evidence-chains-listing-only` (`_Evidence(reader, (p.item.chain for p in listing.items))`, `:399`). The argument for equivalence was that a folded member's `Link` hop also sits on a listed item's chain. It does not hold when the listed item is reached by a cheaper route that bypasses the link.

Counter-shape (`scratchpad/an1/minimal.py`, runnable; found first by a random-world fuzz, 1 differing answer in 7,188 builds): depends-on from `F` (javascript, flag-false). `F` calls provider `P` directly (`c_direct`) and also `T` directly (`c_dT`); `F` also reaches `P` through link `lk` (`c_via` VIA→`lk` TO→`P`), which carries lifted evidence in unit `u_lift` and a bound entry in unit `u_cfg`. `P` has a site `c_in_P` resolving to `T` and name-matching `X` (unit `u_x`). `P` and `T` are listed on their one-hop direct chains; `X` folds into group `c_in_P` with the chain `F, c_via, lk, P, c_in_P, X`. That `lk` hop (and its evidence units) is on no listed item.
- Mutant 1 (members dropped from `evidence_units`): `BuildError ... (rule V9)`; unmutated builds and validates.
- Mutant 2 (evidence read from listed items only): `KeyError` on `ev.counts[...]` at `answer_build.py:190` (an uncaught non-`GraphDataError` crash; with a link that has no site count it would be a silently missing `lifted`/`bound`).

Why it matters: the code is right, but its test pins the only member whose evidence is also evidenced elsewhere, so a regression that reads member evidence from `listing.items` ships green. The spec2 §8 intent ("each source, killed by dropping it") covers units reached by a folded member only for the member's own unit (source 5), not for its evidence. **Fix:** add the world above as an answer-level test (asserts `u_lift` and `u_cfg` in `staleness.units`, the member's `Link` hop carries `lifted`/`bound` and `candidateCount`). It kills both mutants.

### F3 · major · staleness source 2 ("a hop's unit on a chain") is not isolated (verified: mutant)

Mutant `S-a` (`ST.staleness_block` fed each item's chain as `chain[:1] + chain[-1:]`, i.e. all interior hops dropped) survives all 369 tests; delegate's `-k source_2` mutant only drops everything but the last hop. In `test_staleness_unit_source_2` the interior call site and the anchor hop are both in `u_ui`, so hop 0 masks the interior hop. spec2 §8 requires "that unit on no other path". An isolating shape exists in the engine's own fixtures: `data_reached_sites_g` with `Y` and its site `cG` moved to a new unit (`scratchpad/an1/interior.py`: units `['', 'y']`, `S-a` then fails V9). **Fix:** have test 2 use a hop unit that is neither the anchor's, an item's, nor an evidence entry's. Source 1 is inherently the last hop's unit (V7), so its test is as isolated as it can be; say so in the test docstring.

### F4 · minor · `bound.launch` fields `mountPath`, `service`, `envVar` have no test (verified: 3 mutants survive)

Mutating the column indices `r[13]` mountPath, `r[12]` envVar, `r[14]` service (`answer_build.py:276-277`) survives (A, B, C below; the controls D, E, F on lifted/keyPath were killed). No test fixture sets those three properties, and the live world has no launch entry, so the column order against `reader.py:110` is checked by nothing (I checked it by reading: it is correct today, protocol 10, port 11, envVar 12, mountPath 13, service 14, argv0 15, edge port 16-18). These are the F-4 masked identifiers. **Fix:** one test per `launchKind` that sets every launch field with distinct values; add one such entry to the live world.

### F5 · minor · stale "not built" statements remain (verified: unfiltered `grep` over `cpg`, non-test)

The rewritten sites (README bfs row, `engine/__init__.py`, `bfs.py` docstring, `group.py`) are correct. Not rewritten: `cpg/ccg/engine/reader.py:20` ("SEAM(P1-39): `snapshot` and `units` are registered and returned raw; the stamp and the staleness block are built there" — they are built, in `staleness.py`/`answer_build.py`); `cpg/README.md:92` flow row ("`1.1`, re-pinned: unpublished until P1-39"). At feature close: spec2 §1.2 ("first happens at P1-39. Before that its literal may be re-pinned") and §3.6 ("1.1 is unpublished") need the same treatment. The sweep owner for the 39a-named three was the delegate; these two were never in that list.

### F6 · minor · D1 workaround costs one read per depends-on symbol (measured)

`_spans` (`answer_build.py:358-371`) issues one `symbol_by_qname` per symbol item lacking a span. Measured on a synthetic world (F calling N distinct callees): N=10 → 11 reads, N=100 → 101, N=300 → 301; every other read count is constant (`units` 3, `snapshot` 2, `calls_out` 1; `has_out:CallSite`/`owner_of:CallSite` are batched). Correctness is fine (matched by id, decoy test passes, qname is indexed `ddl_v1.cypher:37`). teco's D1 ruling already accepts it for P1; recorded here so the follow-up (a batched by-id span read, the graph-note change teco noted) is sized: it is a per-answer round-trip count linear in the depends-on fan-out. The `calleeName` fill is sound: two batched reads, rows length-checked, an unnameable site is a `BuildError`.

### F7 · minor · `table=` has no consumer under the project's own ruling

teco's D2 ruling (and the module docstring) says candidate walks are measured via `Walk.items()`, never a document, so `build_answer(table=)`, `CANDIDATE_VERSION` and most of `test_engine_answer_v18.py`'s candidate-table assertions serve no caller; the docstring itself documents that the documents are refused. Either drop `table=`/`CANDIDATE_VERSION` until P1-39c needs them, or keep them and accept that those assertions are rewritten when the rule picks a shipped table (spec2 §5 and V18 must be revised first, per the docstring; make that an explicit line in P1-39c's plan row).

### F8 · minor · growth-stops check on spec2 (root `AGENTS.md`)

New scope (fine): the stamp-source row (matches `staleness.stamp`: exactly one row of length 5, `type(offset) is int and offset == 0`, hash equal, else `GraphDataError`; "lacks it" is covered by the type check), the N1 clause (matches the chain `CALLS_NAME ->, HAS_PARAM ->, HAS_PARAM <-` and the control, both verified by test), the §8 five-source paragraph. Rework/history growth: the "Revised 2026-11-12:" paragraph (line 5) is a single paragraph that grows one clause per gate and cites gate finding IDs; the convention is one dated line, not a narrative. Replace it with one line. The §8 paragraph is an instruction to the P1-39b delegate ("For P1-39b ... one answer-level test per source"); once delivered it is delivery history; prefer stating the requirement as what the mutation check covers.

## Checks asked for

**(1) Conformance.** Verified by reading and by test: hop state is the walk kind with `accessPath` for `path` only (`answer_build.py:142-148`, parametrized test); `reason` only on a `possible` hop (`:186`, killed mutants `reason-on-confirmed`/`reason-never`); `candidateCount` pair order for `VIA ->`/`<-` (`:190`, swap mutants killed); `site` state and `Unit` hop handled; an unknown `candidate_count` (`bfs.candidate_counts` returns `None`) is `BuildError` (`:298`), a missing/changed anchor is `BuildError` for all three anchor types (`:123-135`, tests incl. decoys); `clean` (NFC, then strip identity keys) runs before `validate` and `canonicalize` (`:418-421`; opposite-order NFD/NFC test). Fold inputs: re-read report §a1.2 table (lines 70-78): Python 110/116 "/ same", TS/JS 102/102 "/ same", Go 108/108 "/ same", Java 47/47 and 50/50, C# 22/25, C++ 5/5, Kotlin unmeasured; `FOLD_MEASURED` (`:65-72`) equals it, and the test derives its expectation from the report text, not the constant. `javascript` and `go` fold; Python (0.948 < 19/20) lists. No defect.

**(2) Five staleness sources.** 1, 3, 4, 5 isolate (each unit appears on no other path; delegate's four mutants killed, re-run). 2 does not (F3). F2 adds a sixth untested source (a member's evidence units).

**(3) N1 test.** Sound: real walk, asserts the three-hop tail, `candidate_site is None`, no groups, and the depends-on control folds into group `c` (`test_engine_group.py:668-690`); delegate's three N1 mutants killed on re-run. The `<-`-branch variant is only reachable by the synthetic second test (the walk's arg-row tail is `CALLS_NAME ->`), which is what spec2 asked for.

**(4) V18 refusal of candidate-table documents.** Correct behaviour, not a spec2 §5 defect. §5 states the labelled-row list is closed ("these and no others") and V18 clauses (c) (flow-unavailable only on a labelled row leaving a `function` hop) and (e) (every exit from `function` is a listed row) encode that; V-flow's labelled `DATA`/`CONTROLLED_BY`/`arg(i)` hops and V-both's `HAS_CALL ->` exits are by definition outside the shipped table. Refusal is limited to flag-false units (asserted over every world and the slice). Only follow-up: F7.

**(5) Mutants.** Re-derivation of my re-run's survivors: `subject-unit-id` (×2 files): equivalent, a `Unit` hop's `unit` fact equals its id (spec2 §3.5, V9). `recipe-queries-before-reads`: vacuous (`queries_early = None`). `recipe-queries-before-staleness-reads`: equivalent, a staleness read implies items exist, so `units` was already read by `group.unit_refs`, and `snapshot` by the stamp; names, not params, are recorded. `verify-after-walk`: equivalent for symbol/contract/config anchors (the walk never raises for them; the same `BuildError` is raised after). `site-names-wanted-filter`: equivalent, only adds facts for unwanted sites with their own graph name, never read. `bound-literal-2`: equivalent only while `DEFAULT_DEPTH_BOUND == 2`; a test patching the constant would kill it (not worth much). The two member-Link survivors: **not equivalent** (F2).
My own mutants, `ccg/engine/answer_build.py`, run with the delegate's runner logic against answer+v18+group:
- A (survived): `mountPath=_str(n, r[13], ...)` → `r[14]`
- B (survived): `envVar=_str(n, r[12], ...)` → `r[13]`
- C (survived): `service=_str(n, r[14], ...)` → `r[13]`
- S-a (survived): `bfs.Item(..., p.item.chain)` fed to `staleness_block` → `p.item.chain[:1] + p.item.chain[-1:]`
- H (survived, equivalent: double guard `span.get("ownedRegions") is not None` at `:323`): `if r[5] is not None` → `if r[4] is not None` in the span fill.
- controls killed: D `verbPath` col 10→9, E lifted `name` col 13→4, F bound `keyPath` col 8→6, S-b/S-c/S-d (other chain slices). (My mutant G was a no-op by construction and is discarded.)

**(6) Stale statements.** F5 (unfiltered sweep; the named three are rewritten, two more remain).

**(7) D1 workaround.** F6.

**(8) Growth-stops.** F8; plus F1.

## What's solid

- The builder is small and total: static error messages (hostile-string tests assert no value leaks), a single `answer.validate` exit, snapshot re-read at the end (`:404-406`), recipe queries recorded after every read.
- Determinism tests are real (shuffled replies, shuffled insertion order, rebuild with different `builtAt`); NFC order test uses a pair whose raw and canonical orders are opposite.
- Live byte-identity of `CypherReader` vs the in-memory reader over six world parts and nine requests; every text the build ran is in the closed query table.
- The report-derived fold test and the stamp-row wording match the code.

## Open questions

- F4/F7: should the launch-field test (and a launch entry in the live world) be part of this unit or of P1-39c? I recommend this unit; it is an hour's work.

### Proposed REVIEW.md lines

Root `REVIEW.md`, "Always check" (source: this review F2, F3 and the prior 39a F4): *A claimed-equivalent mutant is re-derived, and its argument ("another path covers it") is tested by building the counter-shape (a cheaper route to the covering item), by a small random-world comparison where the world generator exists; an argued equivalence is a bound, not proof.* This changes what a reviewer does (the delegate's argument alone was accepted twice). Not enforced by any test.

## Appendix

Reproduction (all under the session scratchpad `.../scratchpad/an1/`): `minimal.py` (F2 counter-shape, mutant 1; `minimal2.py` mutant 2), `interior.py` (F3), `my_mutants.py`, `my_mutants2.py` (mutants A-H, S-a..S-d), `fuzz_member.py` (random-world comparison: seed 1, 60 s: 7,188 builds, 3,003 with folded members, 100 with a `Link` on a member chain, 1 differing), `n1.py` (F6 read counts), `out1`/`out2`/`out3` (the delegate's 147 mutants re-run; survivors only listed in F2/Checks).

## Pass 2

**Verdict: approve with suggestions.** Re-checked by execution against the current tree (fresh scratch copy `scratchpad/an4`, rebuilt because earlier scratch dirs may have been removed; real tree untouched, no live run needed).

Runs: answer+v18+group in the copy 375 passed (slice bundles present); wider engine/answer/render scope in the real tree 2022 passed, 128 deselected.

- F1 fixed: spec2 §6 table rows 431-448 all start with `|` (18 rows, no stray line), the `Carry-over C4, C6` row is restored at line 448 and the N1 row is intact.
- F2 fixed: the delegate's `stale-evidence-units-from-members-dropped` and `evidence-chains-listing-only` re-run, both killed by `test_a_folded_members_link_hop_carries_its_evidence_and_its_units_are_stale_listed_...`.
- F3 fixed: mutant S-a (interior hops dropped) re-run, killed by `test_staleness_unit_source_2_an_interior_hop_in_a_unit_no_item_anchor_hop_or_evidence_entry_is_in`.
- F4 fixed: my mutants A, B, C (launch column indices) re-run, all killed by `test_every_launch_field_is_read_from_its_own_column_with_distinct_values_per_field[...]`.
- F5 fixed in part: `reader.py:20` now says the stamp and staleness block are built in `staleness`/`answer_build`; the README flow row no longer says "unpublished until P1-39" (grep: no hit). spec2 §1.2/§3.6 wording deferred to feature close, as ruled: acceptable, but it must be on the close list.
- F6 not changed (accepted by teco's D1 ruling; follow-up is the batched by-id span read).
- F7 addressed by docstring only: `table=`/`CANDIDATE_VERSION` remain (4 `table=` hits in `answer_build.py`); acceptable, still a minor YAGNI.
- F8 fixed: line 5 is now one dated line; §8 states the requirement as a test list, including the N1 test. Growth check: the §8 paragraph grew by the N1 sentence (new scope, matches `test_engine_group.py`); no restated history or findings-IDs. Note the one-line revision note dropped the earlier clauses entirely, which is what the rule asks for.

No new findings.
