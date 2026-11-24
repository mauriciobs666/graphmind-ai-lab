# Connected code graph P1 - spike dispositions report

> **Status:** active · **Owner:** `coder` · **Tracks:** P1-38 (index D7)

Scope: `git grep -n vertical-slice -- . ':!cpg/spikes/vertical-slice' ':!docs'` (tracked files). Observed: **60 hits in 39 files**
(`cpg/spikes/vertical-slice/` itself: 20 tracked files, not counted). `docs/` hits (94 lines) are out of scope per the bullet.
No code was edited: every remaining hit is provenance, oracle, or spike code, and none is a trivially provable re-point.
`cpg/spikes/vertical-slice/README.md` was created (none existed) with a one-line FROZEN header.
Note: `cpg/.ccg-work/vertical-slice/...` (gitignored stored-bundle area, 9 hits) is a different directory from the frozen spike dir.

## Rulings

| Group | Hits (files) | Ruling |
|---|---|---|
| `cpg/spikes/frontend-fitness/s28_run.py` (default `EXTRACT`, help text) | 2 (1) | **Already re-pointed in P1-34**: `--extract` selects the script, and `tests/test_s28_run_extract.py` pins the default as the old script (`OLD`, sha recorded). Default stays old on purpose (reproduces recorded S28 runs). Historical. |
| `contact.json` | 8 (1) | Historical (recorded artifact paths). |
| `make_contact.py`, `tests/test_contact.py` | 1 + 2 (2) | Path-classification of `spikes/vertical-slice/*` for contact records; strings only, must keep matching historical records. Keep. |
| `.sc` header comments (`rdfix`, `rdrelax`, `rdwide`, `s28-extract-bodiless-guard`) | 4 (4) | Historical (reproduction command lines in headers). |
| `s28_core.py` docstring | 1 (1) | Historical provenance. |
| `store/results/bench_reach.jsonl`, `bench_reach_summary.json` | 3 (2) | Historical (recorded results; editing would falsify). |
| `store/MEASUREMENTS.md` | 1 (1) | Historical. |
| Production provenance docstrings: `ccg/assemble/rows.py`, `ccg/packrt/{extract,lift}.py`, `ccg/store/loader.py`, `extractors/joern/normalize/{assemble_bundle,blame,core,javascript,python}.py` | 9 (9) | Keep as-is. Comments only (no import/path dependency); they cite the frozen oracle, which must stay. Editing touches production files for no gain. |
| `extractors/joern/README.md`, `pyproject.toml` (marker text) | 2 (2) | Keep; they refer to `.ccg-work/vertical-slice` / the marker description, not the spike dir. |
| Equivalence/conformance oracles under `cpg/tests/slice/*` (7 files) | 17 | **Live dependency by design** (D7/D9): import the frozen modules as the oracle. Deleting the spike dir breaks them. Keep. |
| `tests/{test_engine_answer_v18,test_bundle,test_rule_pack_spec,packrt_helpers}.py` | 4 (4) | Oracle import (`test_engine_answer_v18`, 1) or `.ccg-work` bundle path (3). Keep. |
| `tests/test_s28_run_extract.py` | 1 (1) | Pins old script sha (P1-34). Keep. |
| `spikes/store/{bfs,synth,lock_replay}.py`, `store/tests/test_store_spikes.py` | 5 (4) | **Open: BACKLOG item** (below). Code imports `vertical-slice` modules via `sys.path`; test asserts the `impact.py` path/sha. |


## Verification

No source edits, so nothing to re-prove. Baseline run of the two dependents from `cpg/`:
`pytest tests/test_s28_run_extract.py spikes/store/tests -q` -> `81 passed, 14 deselected` (exit 0 from pytest; the 14 deselected come from the project's default marker deselection, not inspected further).

## Proposed BACKLOG item (exact text)

> **Delete or vendor `cpg/spikes/vertical-slice/` (P1 D7).** Frozen, not deleted. Deletion is blocked by: (a) `cpg/tests/slice/*` and `tests/test_engine_answer_v18.py` oracle imports of `assemble.py`/`load.py`/`impact.py`/`extract.sc` (trigger: P2 retires the slice equivalence gates, or the oracles are vendored as golden-free fixtures); (b) `cpg/spikes/store/{bfs,synth,lock_replay}.py` `sys.path` imports and `store/tests/test_store_spikes.py` line 404 (`impact.py` path/sha) - owner `coder`; re-point to vendored copies under `cpg/spikes/store/` or retire the store spikes with their results; (c) `s28_run.py` default `EXTRACT` and `tests/test_s28_run_extract.py` pin the old script by design - re-point only if the S28 record is retired. Historical hits (`contact.json`, `store/results/*`, `.sc` headers, provenance docstrings) need no action but go stale on deletion. Done when `git grep -n vertical-slice` outside `docs/` shows only historical hits and the suite is green.

## Not done (needs owner)

- `cpg/README.md` frozen marking (the gate bullet says README marks it frozen): owned by teco/close writers; the spike dir's own README now carries the header.
