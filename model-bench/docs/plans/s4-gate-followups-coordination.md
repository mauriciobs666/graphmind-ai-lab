# S4 code-gate follow-ups — Coordination

> **Status:** archived · **Owner:** `teco` · **Tracks:** — (post-M9 backlog)

Goal: close the three `docs/BACKLOG.md` open items flagged by the S4 code gate
(`model-bench/docs/reviews/small-model-benchmarking-s4.md`): `report.py`'s unrendered counters,
`exec.py`'s `None`-key `order_by` sort, and `validate_pack`'s scorer-resolution check.

Baseline: `5d106332`, suite `1800 passed, 3 deselected` (`.venv/bin/python -m pytest -q`,
`model-bench/` cwd). CPG: no `cpg_model-bench` graph loaded — none consulted.

## Scope ruling at decomposition

- **`validate_pack` scorer check is already delivered** — `modelbench/packs.py::_scorer_problems`,
  wired into `validate_pack`, `docs/HISTORY.md` entry 2026-09-11 (commit `609b090b`, U127). The
  `BACKLOG.md` item is stale; removal only, folded into FX3.
- **`exec.py` lives inside a content-hashed pack** (`packs/nlq-structured-query/tools/exec.py`;
  `packs.py::content_hash`). Whether editing it changes the pack's `contentHash` against existing
  recorded runs was handed to FX1 to verify: **confirmed** (FX1 report; teco re-checked `_pack_relative_paths` excludes only `PROVENANCE.md`/`__pycache__`, and `results/index.csv` has 18 rows for the pack). FalkorDB's own `ORDER BY` puts `null` last ASC / first DESC (teco, read-only `UNWIND` query on `ws:test`).

## Documentation impact

- `model-bench/docs/BACKLOG.md` — remove the `report.py` and `validate_pack` items; **rewrite** the
  `exec.py` item in place as the deferral record FX1's xfail reason points at (review Pass 1 — FX1,
  M1); add review F2 (verdict/reference sections unlabelled if a pack makes an `exactMatchBy*`
  metric a verdict metric) as a new scope-limit item (FX3).
- `model-bench/docs/HISTORY.md` — one entry for the delivered change (FX3).
- `model-bench/docs/manuals/small-model-benchmarking.md` (`tico`) — only if FX2 changes report output a
  manual section describes; FX2 reports whether it does.
- `model-bench/AGENTS.md` — only if an invariant changes (not expected).

## Ledger

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict | Cost |
|---|---|---|---|---|---|---|
| FX1 | `tdd-engineer` | `af1ba79a64ed0712e` | accepted | **deferred** per stakeholder 2026-10-15: `tests/test_tools_exec.py::test_order_by_a_sparse_property_places_nulls_the_way_falkordb_does` (strict xfail); pack hash unchanged `8f975842` | `analyst` (`afd783634b3d17830`) → approve w/ suggestions (M1 → FX3) | 79k+90k tok / 13+5 tools |
| FX2 | `coder` | `a594ef6236e7e92c4` | accepted | `modelbench/report.py` + `tests/test_report.py` (`aba53a11`), reworked for review F1/F3/N2 (F2, N3, N4 → BACKLOG via FX3; N1 kept) | `analyst` (`afd783634b3d17830`) → Pass 1 approve w/ suggestions; Pass 2 **approve** | 159k+186k tok / 56+9 tools |
| FX3 | `coder` | `aaecae86fdc792766` | accepted | `docs/BACKLOG.md`, `docs/HISTORY.md` | teco integration check → verified (items removed/rewritten/added; entry figures match teco's runs) | 84k tok / 13 tools |

FX1 and FX2 are file-disjoint and share no database; they run in parallel. FX3 follows both
(`BACKLOG.md`/`HISTORY.md` are shared by both).
