# model-bench — agent working context

## Project in one sentence

A standalone, human-started harness that measures one local model against one versioned task pack,
stores the result with a full environment fingerprint, and compares runs within a role — see
`README.md` for the human-facing quickstart and `docs/plans/small-model-benchmarking.md` (repo root)
for the full design.

## Current state

**Stage S0 — skeleton only.** `modelbench/` holds `__init__.py` and nothing else; the suite is one
install smoke test; there is no CLI, so `run.sh` fails loudly with a named reason instead of
`exec`-ing into a traceback. `docs/plans/small-model-benchmarking.md` §4 sequences S1–S8; build them
in order, and delete `run.sh`'s S0 guard block when `modelbench/__main__.py` lands in S1.

**`tests/test_package.py` is deliberately the whole suite at S0**, not a placeholder — it pins
`modelbench.__version__` to the installed distribution metadata, which is what stamps `benchVersion`
into every run record (plan §3.4). Keep it as real tests arrive. It also exists because pytest exits
5 (`EXIT_NOTESTSCOLLECTED`) on an empty suite: never restore green by configuring that exit code
away, since a permanent "no tests ran is fine" setting hides a collection breakage later.

## Hard rules (they are design constraints, not preferences)

- **Zero runtime dependencies.** stdlib only — `urllib.request` for HTTP (falkor-chat's own
  precedent in `falkorchat/transport.py`), `json`/`math`/`statistics`/`hashlib`/`subprocess`. Dev
  extras are `pytest` and `ruff`, nothing else. Old results stay reproducible only if the tool that
  produced them still installs. The one stated reversal trigger (plan §3.2): add `numpy` if a pack's
  corpus exceeds ~1 000 documents or scoring exceeds ~5 s.
- **Standalone — FR-23.** No runtime code path reads any path outside `model-bench/`. Golden data
  from `falkor-chat` is *copied in* and versioned here with provenance; the one-way importer
  `scripts/refresh_golden.py` is a human-invoked maintenance script and is never reachable from a
  run. Nothing in `falkor-chat` changes, in either direction, ever.
- **No aggregate across roles, no gate, no scheduler.** Enforced structurally: `load_history()`
  takes a `packId` and there is no API to load across packs. See `README.md`'s three non-features.

## Conventions

- **`live` pytest marker**, copied verbatim from `falkor-chat/server/pyproject.toml`:
  `addopts = '-ra -m "not live"'` deselects real-LM-Studio tests by default, so the standard run is
  network-free even when LM Studio happens to be up. Opt in with `pytest -m live`.
- **Host venv, no Docker** — `setup.sh`/`run.sh` mirror `mcp-monitor/`'s shape (create venv, install
  with dev extra, smoke-import), resolving every path from the script's own location.
- **Four fingerprint fields are operator-attested, not measured** — `lmStudioAppVersion`,
  `kvCacheSetting`, `hostRamGb`, `otherResidentWorkloads` — because no programmatic source exists on
  this LM Studio build (plan §2.3, live-probed). They live in a gitignored `model-bench/host.json`
  and are copied into every run record so a record stays self-contained. Plan §3.4 has the
  staleness trip-wire that keeps them honest.
- **Empty `docs/` subdirectories are held by `.gitkeep`** (repo precedent), so the module
  documentation convention's layout survives a clone before its first document exists.

## Commands

```bash
model-bench/setup.sh                             # create/refresh .venv (idempotent; --recreate)
model-bench/.venv/bin/pytest model-bench -q      # default suite, network-free
model-bench/.venv/bin/pytest model-bench -m live # real LM Studio, opt-in
model-bench/.venv/bin/ruff check model-bench
```

## Documentation map

This component's feature documents live at the **repo root**, not under `model-bench/docs/` —
`docs/requirements/small-model-benchmarking.md` (`tico`) → `docs/plans/small-model-benchmarking.md`
(`architect`) + `docs/plans/small-model-benchmarking-ml.md` (`data-scientist`, the statistics) →
`docs/reviews/small-model-benchmarking.md`. That is deliberate: the feature was specified before the
component existed and its footnote says to leave it there. Everything written *from here on* —
this component's own requirements, plans, reviews, test plans and reports — goes under
`model-bench/docs/`, which is why those subdirectories already exist. `docs/BACKLOG.md` and
`docs/HISTORY.md` here are this component's living logs (module documentation convention, root
`AGENTS.md`).
