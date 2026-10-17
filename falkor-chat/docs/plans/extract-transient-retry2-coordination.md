# Extract/embed transient retry: LM Studio unload-body coverage (D-3) — Coordination

> **Status:** active · **Owner:** `teco` · **Tracks:** K-071 (follow-up of K-067)

Goal: close D-3. The K-071 re-drive (2026-10-17) sent 13 documents through the restarted `:8200` at
once. LM Studio's JIT model swapping answered with unload bodies that
`retry.py:TRANSIENT_DETAIL_MARKERS` does not match (`Model is unloaded.`, `Model was unloaded while the
request was still in queue..`), so 11 documents failed fast instead of retrying. Plan authority: R-1
(§6) + §3.2.1 of `falkor-chat/docs/plans/extract-transient-retry.md` (archived, not edited), the same
basis as D-2 (`21ff9f9b`).
**Done:** the observed unload bodies classify as transient; genuine 400s still fail fast; test rows
added and mutation-checked; `analyst` gate passed; HISTORY entry added.

Baseline: `71826e36`. Evidence: the scratchpad copies `server-log-redrive1.txt` (the `:8200` log since
the restart) and `d3-error-bodies.txt` (the distinct error bodies). CPG `cpg_falkorchat` is stale
(`retry.py` changed since it was built), but this unit doesn't use it.

## Documentation impact

- `falkor-chat/docs/HISTORY.md`: one entry.
- `falkor-chat/docs/BACKLOG.md`: record the K-071 progress.
- The `retry.py` module docstring, if it lists the markers.

## Ledger

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict | Cost |
|---|---|---|---|---|---|---|
| D3 | `tdd-engineer` | `aa3dcec3795c2058c` | accepted | `server/falkorchat/retry.py` (`"model unloaded"` → `"unloaded"`, + `"predict request failed: fetch failed"`) + `server/tests/test_retry.py` (4 transient rows, 1 permanent row); M1–M5 killed; teco re-ran 96 passed, live bodies classify as expected | `analyst` (G3) → approve w/ suggestions; F-1..F-3 applied by D3 (98 passed); teco re-ran 98 passed and the reviewer's surviving phrase-list mutant now kills 2 rows; O-1 → backlog | 68k+74k tok / 8+4 tools |
| G3 | `analyst` | `a58bd61df2ff12b3c` | accepted | `falkor-chat/docs/reviews/extract-transient-retry2.md` → approve w/ suggestions | — | 105k tok / 27 tools |
