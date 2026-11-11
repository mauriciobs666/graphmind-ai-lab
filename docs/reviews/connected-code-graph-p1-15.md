# P1-15 indexer adoption (code part): analyst gate

> **Status:** active · **Owner:** `analyst` · **Tracks:** C-1001 (P1)

## Scope and verdict

Reviewed: uncommitted `cpg/extractors/indexers/{__init__,scip,run}.py`, `cpg/extractors/joern/normalize/merge_scip.py`,
`cpg/tests/{test_scip_parse,test_scip_merge,test_indexers_run,scip_helpers}.py`, `cpg/tests/data/scip/tiny.scip`, and the two P1-15
rows of `cpg/README.md`. Baseline: P1-15 in `docs/plans/connected-code-graph-p1-extraction.md`, `docs/plans/connected-code-graph-p1-ml.md`
§2.1-§2.3/§8, `-egress-spec3.md` §7.4/R-E9/R-E10, `-egress-spec4.md` §1, `cpg/sandbox/README.md` (watchdog). Root `REVIEW.md` applied; `cpg/REVIEW.md` does
not exist. The corpus measurement and the four outcomes are out of scope (not faulted). A sibling
`docs/reviews/connected-code-graph-p1-15-security.md` exists; I did not rely on it, and two of my blockers independently match its B1/B2.

**Verdict: needs changes.** Three blockers: two on the hostile-input parser (the size cap does not bound CPU or memory), one on the merge (a
correct Joern TypeScript `new X()` site is demoted by the merge on real `scip-typescript` output).

CPG: considered, not relevant — all files are new and uncommitted, so no loaded `cpg_*` graph contains them; verified by reading, probes and live indexer runs.

Checks run (all `cpg/.venv/bin/python`, cwd `cpg/`, `ulimit -v 3000000`, `timeout -s KILL`): the three P1-15 test files, 160 passed / 14 live deselected;
`-m live` on `tests/test_indexers_run.py`, 14 passed (throwaway container mechanism, nothing built; `docker ps -a` before/after shows only
`serene_gagarin`, `falkordb-dev`, `ccg-falkordb`, none created by me). **The full `pytest` was killed by the 160 s cap at about 12 %**, so
"whole suite green" is not established; only the P1-15 files and `-m live` of `test_indexers_run.py` are. Extra live probes: real `scip-typescript 0.6.0`
and `scip-python 0.6.8` runs over toy projects (scratch repos, same `run_indexer` path).

## Blockers

### B1. Quadratic CPU in per-position de-duplication (verified)
`scip.py:320-322`: `slot = refs.setdefault(...); if t not in slot: slot.append(t)`. N occurrences with N distinct symbols at one
`(file, line, col)` cost O(N^2) dataclass comparisons. Measured with `parse_bytes`: N=2 000 (87 KB) 0.42 s; N=8 000 (351 KB) 6.6 s; N=16 000 (709 KB) 25.9 s.
Under the caps (2 M occurrences, 128 MB) this is days. The docstring's "size cap" and the occurrence cap do not bound it; nothing bounds wall time.
**Fix:** de-duplicate on the interned symbol id (`k`) with a per-position `set`, and cap targets per position (count the excess in
`rejected_occurrences`). **Guard:** a test with 50 000 distinct symbols at one position that finishes under a second; it can fail (the current code takes ~minutes).

### B2. Span lists are materialised before the occurrence cap is checked (verified)
`scip.py:~255-262`: a document's occurrences are first collected into `occs` (one tuple per `\x12\x00`, 2 bytes of file), and `occ_count > max_occurrences`
is tested only in the later loop. Probe, `ulimit -v 1500000`: a **12 MB** file (6 M empty occurrences) peaks at **831 MB RSS** before `ScipError`; a **24 MB**
file raises a bare `MemoryError` (not `ScipError`, so a caller catching `ScipError` crashes). At the 128 MB cap this is ~8 GB on a 6 GB host.
The module docstring claims "memory" is bounded by the caps; it is not. (Honest bound: with the cap-checked path, 2 M distinct-position occurrences peak at
1 195 MB RSS, 83 MB file; that is the realistic ceiling and is itself close to the 1.5 GB budget the caller used.)
**Fix:** count occurrences while scanning the document (raise at the cap before appending), and do not retain a per-occurrence tuple list; lower
`DEFAULT_MAX_BYTES` or add a bytes-per-RSS budget so the ceiling is stated. **Guard:** the 24 MB empty-occurrence file under `ulimit -v` must raise
`ScipError`, never `MemoryError`; and a test asserting peak `tracemalloc` for a capped input.

### B3. TypeScript constructor calls: the merge demotes a correct Joern target (verified on real tool output; delegate gap 4)
Real `scip-typescript 0.6.0` on `new Box(2)` emits `...`a.ts`/Box#`<constructor>`().` (chain `('Box','<constructor>')`, callable), not a type symbol.
`merge_scip._resolve` builds `a.ts::Box.<constructor>`; `INIT_NAME["javascript"] = "constructor"` is only used on `is_type`, which TS never is here. The bundle's name is `Box.constructor`
(`assemble_bundle.py` `init_of_class`). Result, run with the real index shape and a Joern-resolved `new` site (`/…/scratchpad/mp.py`): `status=possible, targets=[]`, counts `disagree`. So on every TS
class instantiation Joern got right, the merge turns a `confirmed` target into `possible/name-match` with the target moved to a side list. That breaks
the step's own invariant (`R_m >= R_j`, ml note §2.2) on the very corpus P1-15 measures, and would bias M2 downward.
Mutant `INIT_NAME["javascript"]="constructor2"` **survives** the suite, so nothing pins the JS mapping. Python is fine: real `scip-python 0.6.8` emits `Box#` (type) for `Box()`, and
`Box#put().` resolves; a class with no `__init__` correctly falls to "external".
**Fix:** map a descriptor named `<constructor>` (TS) to `constructor` in `parse_symbol` or `_resolve`, per language. **Guard:** a merge test built from the captured real symbol string
above, expecting `agree`, plus a JS `INIT_NAME` mutant row; store a real-tool `.scip` as a second golden (see M1).

## Majors

### M1. Guards that cannot fail (mutants that survived)
Mutations on a scratch copy (`/…/scratchpad/m`, `PYTHONPATH` verified to resolve to the copy). Killed: outer slack 30→0, `WALL_S` 1800→900 (pin vs `offline-run.sh`), `>=`→`>` on
the 124 judge (boundary rows 2000/1999), size pre-check removed, length bound, int32 bound, `killpg` removed, `SIGTERM` removed, hardlink and image-env mutants, merge drop-others / drop-S / ambiguity / externalTarget pop,
`--project-version` removed. **Survived:**
- `O_NOFOLLOW` removed (`scip.py` open): the docstring and README claim "a symlink is not followed", no test plants one. **Add** a symlink-to-regular-file test.
- `INIT_NAME` javascript value (B3).
- `scip_only` filter `(t.callable or t.is_type)` replaced by `True`: no test plants a non-callable in-corpus reference at an unsited position.
- Read bound `max_bytes + 1 - total` (TOCTOU growth): equivalent-ish while the `st_size` pre-check holds; a test that grows a file between `fstat` and read would pin it. Low value, mention only.
- The golden `tiny.scip` is byte-identical to `scip_helpers.tiny_index()` (verified `cmp`), i.e. produced by the encoder the decoder tests share, not by a tool. Real layout is only covered by the `live` tests (deselected by default). **Add** one small real-tool capture (e.g. the toy TS project above) as a stored golden.

### M2. Disagreement path (delegate gap 1): judgment
The deviation is forced and correct as far as the schema goes: `bundle-v1.schema.json` `callsite` has `targets.minItems>=1 ⇒ status=confirmed ∧ reason=resolved-call`, `reason` has no `resolution-unverified`, and
`additionalProperties:false` leaves no place for candidates. I verified `test_merged_records_validate_against_the_bundle_schema` exercises it. Two consequences the plan should take, not the coder:
1. **The candidates exist only in the return value.** Nothing in the repo consumes `MergeResult.candidates` (no assembler wiring; `grep CALLS_NAME` finds only `answer.py` and the shell extractor). Until a consumer exists, on a disagreement Joern's confirmed target is *absent from the bundle*,
   so "no candidate is dropped" is true of the function and false of the pipeline. P1-15's measurement must read `candidates`, or M2's `R_m >= R_j` is not a bundle property. Route to `architect`: name the consumer and the seam.
2. `reasons.yaml` scopes `resolution-unverified` to "`CALLS_NAME` candidates of a **resolved** site" (and `render/common.py` words it "another possible callee of a call that already resolves"); on a disagreement the site is *unresolved*. Either widen the registry line or
   use the existing `name-match` for those candidates. Not a code defect; a contract gap.

### M3. `positions` seam is unbuilt
`merge_scip(callsites, positions, ...)` needs `callSiteId -> (file rel. to indexed dir, 1-based line, 0-based col)`. The assembler's `cs_info` holds `(cid, sid, pos)` with `pos` from `locate_callees` (line/col, no file) and no caller builds the map. Not faulted for the corpus run, but the done-condition's merge cannot run until it exists; record it as an open item for the measurement unit.
Real-output evidence for column (delegate gap 5): both tools print columns as characters for BMP text (a `café()` call and an `é` string earlier on the line: `b` at 20, `café` at 31, matching code points, not UTF-8 bytes), and **`position_encodings` is empty for both** (the field is undeclared), so `ScipIndex.position_encodings` records nothing useful. Astral characters (UTF-16 units) remain unverified.

## Minors

- **m1. `env_json`/`lockfile` are used before they are validated.** `run.py` hashes `stage / provisioned.env_json` (and `lockfile`) before `tool_argv` validates the grammar; an absolute or `..` value hashes an arbitrary host file (operator-supplied, so low risk) and the `ValueError` arrives after staging. Validate `tool_argv` first.
- **m2. Unbounded runner output** (`_run`'s `communicate()`): container stdout/stderr is buffered in the parent for up to 1 890 s with no memory limit on the container (R-E10). The sibling `joern/run.py` does the same, so it follows convention; the sibling security note rates it higher. Redirect to a file with a cap if the indexer is ever run on less-trusted sources.
- **m3. Delegate gap 7, private imports.** `run.py` imports `_mkdir_private`, `_resolve_commit` (and `descendant_roots`, `stage_unit`, `RUNNER`, `RunnerError`, `StageError`) from `extractors/joern/run.py`. It works (tests pass) but couples to underscored names; promote the two to public or a shared helper in the unit that next edits `joern/run.py`. Nit-level.
- **m4. After TERM succeeds, the process group is not swept** (only the KILL path calls `_killpg`); same note as the sibling review's M2. Low, since the runner owns its container removal.
- **m5. `Provisioned.env_json` path-component check** in `_missing` uses only the last component; the full grammar is only enforced later by `tool_argv`. Harmless once m1 is fixed.
- **m6. Docstring claim vs behaviour**: `scip.py`'s "memory" and "no write-capable call" claims: the second is structurally tested (good); the first is B2.

## Verified against the spec (no finding)

- Argv: `tool_argv` is spec3 §7.4 plus spec4 §1 `--project-version` (mandatory, `scip-python` only, name pattern shared), `scip-typescript` takes only `--cwd`; two-element option form; `/src/<env.json>` path grammar. Version value is the commit sha (deterministic). `runner_argv` accepts only the two indexer frontends; `joern` is refused client-side (parametrised test) and the live test replays 13 non-grammar argvs through the real runner (all refused 64, one record each, no container).
- Timeouts: `OUTER_S = 1800+60+30 = 1890`; the `WALL_S` literal is pinned against `offline-run.sh` by regex; the margin check is made for the real runner and refuses at or below the sum. TERM to the runner, then KILL of the group after `kill_grace_s`; no timer to cancel (blocking wait), so the spec4 "cancel on normal exit" clause is met trivially. Exit 124 raises `RunnerTimeout` and is judged by the record's `durationMs >= wall*1000` (`True/False/None`), boundary-tested; any non-zero exit means no artifact is parsed.
- Stager: absolute / escaping / chained symlinks, hardlinks (checked *before* the copy), FIFO/socket (never opened), root-must-be-plain-dir, collision with the unit's own `node_modules`, second validation after overlay. Own stage dir `<work>/indexer/stage`, asserted different from Joern's.
- Degradation: `Degraded("headers-unavailable", …)`, nothing staged or run; the code exists in `reasons.yaml` (test reads the registry).
- Merge vs ml §2.2 table: all five rows implemented; candidates are kept (J ∪ {S} in `targets` plus `candidates`), ambiguity leaves the record alone, SCIP-only is reported, not minted.
- Hostile-parse items that **hold** (probed): huge length fields, a field past the message end, an 11-byte varint, wire-type 3, and 900 empty documents all give `ScipError` or a counted rejection in microseconds; no recursion (fixed nesting depth); a document path with `..`/absolute/NUL/`\`/`:` drops the document and is used only as a dict key; size pre-check precedes any read; FIFO opened `O_NONBLOCK` and refused.

## Open questions

1. (architect) Who consumes `MergeResult.candidates`, and does `resolution-unverified` widen to unresolved sites (M2)?
2. (teco/coder) After B1-B3 and the m1 reorder, a re-gate only needs the three probes (distinct-symbol position, 24 MB empty-occurrence file under `ulimit -v`, the TS `<constructor>` merge) as tests.

### Proposed REVIEW.md lines
Root `REVIEW.md`, *Always check* (source: B1/B2, this review; recurring class with `docs/reviews/connected-code-graph-p1-15-security.md` B1/B2): "A parser of hostile input is checked with an expansion probe per cap: the cap's unit (bytes, documents, occurrences) against the work and memory one unit of input costs — a 2-byte record that allocates a tuple, a list `in` per insert — under `ulimit -v`; a cap that counts after materialising is a bound, not a guard." No prune proposed.

## Appendix: probe commands
`probe.py N` (distinct symbols at one position), `probe2.py` (2 M occurrences, 83 MB, 1 195 MB RSS, 10.9 s), the 12/24 MB empty-occurrence files, `tsprobe.py`/`pyprobe.py` (live runs of the two real indexers through `run_indexer`), `mp.py` (the B3 merge), `mut.sh` (mutation harness on a scratch copy): all under
`/tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/a56a19ba-3e84-44da-9d76-47eef6534cc4/scratchpad/`.

## Pass 2 (re-gate)

**Verdict: approve with suggestions.** All three blockers are fixed and re-probed; one new major (N1) and two minors are small and do not need a new round
if the coder closes N1. CPG: considered, not relevant (uncommitted files; verified by probes, a live run and mutants).
Run: P1-15 files plus `test_normalize_assemble.py` 422 passed / 14 live deselected; `-m live` of `test_indexers_run.py` 14 passed; `docker ps -a`
unchanged (`serene_gagarin`, `falkordb-dev`, `ccg-falkordb`, none mine). Full `pytest` still not run to completion (cap), so "whole suite green" is not claimed.

### Dispositions
- **B1 (quadratic dedup): fixed.** `refs` holds interned ids per position with a 64-target cap; 50 000 distinct symbols at one position now parse in 0.24 s / 24 MB (`ulimit -v 1500000`).
- **B2 (memory before cap): fixed.** Each document is scanned twice, the first pass only counts. 40 MB of empty occurrences and 40 MB of empty symbol-infos raise `ScipError` at the cap in 0.6 s, 89 MB RSS; 500 000 empty documents hit the document cap, `ScipError`; 5 MB packed-range occurrence is rejected, 22 MB. Caps are now 64 MB / 1 M occurrences / 1 M symbol-infos.
- **B3 (TS constructor): fixed, and (a) the coder is right and my B3 text was wrong on the target name.** The stored real bundle `tests/data/scip/real_ts.bundle.jsonl` (written by `assemble` from a real Joern run: fingerprints, `lastChangedAt`, kind `constructor`) holds `web/a.ts::Box.<init>`, and `normalize/javascript.py:39-72` names a constructor `<init>`. I had taken `.constructor` from the old `INIT_NAME` constant without checking the bundle. `.constructor` (the plan text, `docs/plans/connected-code-graph-p1-extraction.md` P1-15 "Constructor mapping" row, and the architect text) is the error: route a one-token correction to `architect`. Mutant `INIT_NAME javascript -> "constructor"` is killed by `test_the_real_merge_agrees_upgrades_and_leaves_the_rest`; the real `real_ts.scip` golden replaces the circular encoder golden.
- **M1 (surviving mutants): fixed for the named ones.** `O_NOFOLLOW`, `scip_only` filter, and the JS ctor name are now killed; a real-tool golden is stored. **New unpinned constants: see N2.**
- **M2 (disagreement path): fixed as design.** The plan and module state that `candidates` is a merge-time report, that Joern's and SCIP's targets surface via `CALLS_NAME`→`Name`→`RESOLVES_TO`, and that `resolution-unverified` is not widened; Joern's demoted share is reported (`counts["disagree"]`). I accept it; the cost (a Joern-confirmed target demoted to `name-match`) is stated as the rule's, not hidden. The aliased-import limit is recorded.
- **M3 (positions): fixed, (c).** `positions_from_bundle` rebuilds `callSiteId -> (file, line, col)` from `assemble`'s `callsitePositions` (`assemble_bundle.py` diff is 4 lines, that key only); a site whose line disagrees with the record is dropped (mutant killed); real `café` columns (code points) match the real SCIP positions; `_astral_before` flags a character above U+FFFF before the callee and the merge leaves the site alone and counts `astral_skipped` (mutant killed). The skip is conservative and the right call while UTF-16 is unverified. Caveat: `_astral_before` slices by code point on the line the *source* reader returns; a file with `\r`/` ` line splits differs between readers, but a mismatch only fails to skip, and the line-equality check drops most such sites.
- **m1 (validate before read): fixed** (`check_provisioned_names` before any hash, also in `run_indexer`). **m2 output cap: fixed for this module** (`_Capped` readers keep 256 KB head+tail per stream; mutant 256 MB killed). **m4 sweep: fixed** (`_end` sweeps the group after TERM; mutant killed, no PID-reuse guard, nit). **Parent death/Ctrl-C: fixed** (`except BaseException: _end`). **Audit-id and control characters: fixed** (mutants A9, A15 killed). **Provisioned tree bound: fixed in code** (but see N2). **m3 private imports (gap 7): not fixed**, still `_mkdir_private`, `_resolve_commit` from `joern/run.py`; nit, take it when P1-04's file is next edited.

### New findings
**N1 (major, verified). `load_provisioned`'s `~/.config/ccg` refusal is lexical and is bypassed (d).** `run.py:~252`: `root == keys or keys in root.parents` on an unnormalised `Path`. Probe with a fake `HOME` (`scratchpad/lp.py`): `CCG_PROVISION_DIR=$HOME/.config/ccg`, `.../ccg/`, `.../.config/./ccg` are refused; **`$HOME/other/../.config/ccg` and `$HOME/other/lnk` (symlink to `~/.config/ccg`) return a `Provisioned`**, whose tree would be copied into the stage and mounted into the indexer container, i.e. the HMAC keys and ACL credentials the guard exists for. Needs an operator-set env var, so it is a misconfiguration net, not an attack; but the docstring and the plan present it as a refusal. Also: `root.is_symlink()` checks only the last component, so a symlinked `system` directory passes. **Fix:** compare `os.path.normpath` and `os.path.realpath` of both sides (realpath only stats path components, it reads no file content), and refuse a symlinked base/system/unit component; **guard:** a test per bypass above (the `..` form and a symlinked base), each of which fails on the current code. Mutant "guard removed" is killed today by the plain-path test only, which is why the bypass is invisible.
**N2 (minor, verified by mutation). Default constants are unpinned.** The silent-argument mutants `DEFAULT_MAX_BYTES 64 MiB -> 64 GiB` and `MAX_PROVISIONED_ENTRIES 500 000 -> 500 000 000` both **survive** (220 passed): tests pass explicit caps or small limits, never the shipped defaults. Add one test each that reads the default through the public call (a sparse file one byte over the default cap via `parse_scip(path)` with no `max_bytes`; assert the module constants against the documented numbers in the spec/plan). Killed for contrast: outer slack 30->0, the margin guard `<=`->`<`, occurrence/symbol/target caps, grow-after-fstat, `_Capped` keep size, the `1800+60+30 = 1890` relation (via `test_the_outer_bound_exceeds...` and the real-runner refusal).
**N3 (minor, design). A hostile index forges "confirmed" inside the corpus** (security m2): the `.scip` decides `confirmed/resolved-call` and the defining file of a symbol, within `known`. Not fixed and not stated anywhere (grep of the module docstrings and plan finds no mention). Acceptable under R-E9 (trusted pinned sources) but add one sentence to `merge_scip`'s docstring and to the adopt criteria: the M1/M2 truth-data measures bound the benign case only.

### (b) Cap-respecting worst case, judged
Measured, `ulimit -v 1500000`: 1 M occurrences, one symbol, distinct positions 5.0 s / 234 MB; 1 M occurrences with 1 M distinct symbols 11.0 s / 676 MB (the coder's 10.8 s / 723 MB reproduces). Linear, bounded by the caps (64 MB file, 1 M occurrences), fits the 6 GB host and is small against the 1 890 s run bound. **Acceptable;** a wall-clock bound on the parse is not needed once the work is linear and capped. The 1.5 GB budget is respected with ~2x headroom. Not tested: a hostile file at the byte cap with many 2 KB symbols (60 MB, 30 000 symbols: 131 MB, 0.1 s, fine).

### (e) Items not done, ruled
- **Joern `run.py` unbounded pipe (P1-04's file):** confirmed still there (`extractors/joern/run.py:255-276`, `communicate()`). Out of this unit by ownership; **record a backlog item** (cap or file-redirect the Joern runner's output as `_Capped` does), owner `coder`. Not a P1-15 blocker.
- **m5 FIFO planted between `validate_tree(root)` and `copy2`:** not fixed; needs a concurrent local writer in an operator-owned directory; theoretical. The final `validate_tree(stage)` cannot rescue a hung `copy2`. Cheap fix if wanted: copy, then validate the stage only; or open with `O_NONBLOCK`. Accept as is.
- **No wall-clock parse bound:** see (b), accepted.

### Open questions
1. `architect`: correct `.constructor` to `<init>` in the P1-15 plan row (and any copy in the answer/graph notes: `grep -n "\.constructor" docs/plans/connected-code-graph-p1*.md`; I did not sweep the other notes).
2. Coder: N1 with the two failing-first tests; N2 and N3 are one-liners. No further round needed unless N1 is not closed.

### Proposed REVIEW.md lines
Root `REVIEW.md`, *Always check* (source: N1, this review): "A path guard that refuses a protected directory is checked with a `..` form and a symlinked ancestor, not only the plain path; a lexical comparison is a bound, not a guard." Source N2: "A cap is mutated at its shipped default, not at the parameter the tests pass; a cap no test reads at its default is unpinned." No prune proposed.
