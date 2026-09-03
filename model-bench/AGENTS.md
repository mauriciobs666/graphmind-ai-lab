# model-bench — agent working context

## Project in one sentence

A standalone, human-started harness that measures one local model against one versioned task pack,
stores the result with a full environment fingerprint, and compares runs within a role — see
`README.md` for the human-facing quickstart and `docs/plans/small-model-benchmarking.md` (repo root)
for the full design.

## Current state

**Stage S1 — the harness core is built; nothing calls a model yet.** `modelbench/` holds
`fingerprint`, `results`, `stats`, `report`, `packs`, `roles`, `cli` and `__main__`; the CLI ships
`compare` (with `--negative-control`), `index rebuild` and `models --tested`. **S2 owns everything
that touches the outside world** — `lmstudio.py`, `hostinfo.py`, the real pack loader
(`load_pack`/`validate_pack`, content hashing, the AST import allowlist, the row-count identity),
`convo.py`, `tooling.py`, `runner.py`, and the `attest`/`validate`/`run` commands. A test asserts
those three commands are still absent, so the stage boundary is checked rather than promised.
`docs/plans/small-model-benchmarking.md` §4 sequences S2–S8.

**`stats.py` implements `docs/plans/small-model-benchmarking-ml.md` and no other source.** Every
formula, constant, threshold, tolerance and verdict string is that note's, cited by section; the
plan deliberately does not restate them, and neither should this file. Its shape is §3.4's **seven**
binding rules, written so the anti-conservative version does not typecheck: `resolving_power`'s
`design_effect`/`basis`/`unit_kind`/`alpha_family`/`alpha_mdd` are keyword-only **with no
defaults**, `min_detectable_difference` takes `n_effective: float` so a raw observation count
raises, and **Rule 7 is enforced inside `verdict()`** — no path returns `distinguishable` below
`resolving.observable_floor`, compared against the exact float and never the printed one, and it
**raises** on the `mcnemar-exact` path (there it is a theorem) while it demotes-and-names on the
substitute one.

**Four honesty rules that are easy to break silently, and what holds each in place.**

- **Every printed bound takes the rounding direction, the α and the denominator that keep *its
  own* claim true.** One principle, three instances, and two bounds side by side routinely take
  opposite values of the same parameter: the floor **truncates** at the **unadjusted α** over the
  **unfloored** `n_eff`; the MDD **ceilings** at **α/k** over the **floored** one. So
  `ResolvingPower` carries both αs (`alpha_family`, `alpha_mdd`) and the printed line names both;
  `alpha_step` is Holm's data-dependent third and reaches `verdict()` as a parameter, since it is
  known only after ranking. `stats.format_floor_pp` is the *only* place the floor's direction
  lives — assert through it, never with `round(...)` in a test, which asserts the presentation
  layer against itself. Its `+ 1e-12` guard is **defensive** since the floor moved to `6/n`
  (nothing the note prints needs it), kept because `b_min` is a function of α; pin it with the
  code's own `floor(x/precision)`, never `floor(x*1000)`, which has no hazard to find.
- **`RunResult.designEffect`/`basis` and `BinaryMetric.unit` carry no defaults.** In each case the
  value a forgetful caller wants is the anti-conservative one, so a default rebuilds gate B-1 at
  that seam. The legacy fallbacks live in `from_dict` only, where they are §3.4.3 reader rules.
- **A Wilson interval prints only over the analysis unit.** `-ml` §4.4: *"Never print a Wilson
  interval over a turn-pooled count."* `report.py` compares `BinaryMetric.unit` against the role's
  unit kind; a pooled count prints its `k/n` and no interval.
- **Holm needs two passes.** The step a metric is tested at depends on every other member's
  p-value, so `compare_report` computes every paired table first, then `stats.holm_steps`, then the
  verdicts, zipped `strict=True` so a short ladder cannot drop a metric.
  `holm_thresholds` was replaced by `holm_steps` because a threshold without the
  step-down stop is unusable — and the rendered family table carries a `decision` column for the
  same reason.

**The analysis unit is pack data, never a call-site choice.** `report.py` resolves it from
`PackRef.analysisUnit` (§3.3 fixes it by rule as `pairingKey[0]`). `PairedOutcomes.from_units`
raising on a repeated unit id is a **backstop, not the mechanism** — it only fires when the id it
is handed is the *cluster* key, and 48 conversation ids drawn from 12 scripts are all unique.
`tests/test_report.py`'s DC-5(c) fixture asserts the captured argument itself for this reason.

**`tests/test_package.py` is not a placeholder** — it pins `modelbench.__version__` to the
installed distribution metadata, which is what stamps `benchVersion` into every run record (plan
§3.4). Keep it. It also exists because pytest exits 5 (`EXIT_NOTESTSCOLLECTED`) on an empty suite:
never restore green by configuring that exit code away, since a permanent "no tests ran is fine"
setting hides a collection breakage later.

**A public name starting with `test` is collected by pytest as a test** in every module that
imports it — which is why FR-17a's function is `models_with_stored_results`, not `tested_models`.

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

All of these need `model-bench/` as the working directory — the repo has no root pytest config, so
from the repo root pytest ignores this component's `testpaths` and walks the whole monorepo
(measured: 8 collection errors, exit 2).

```bash
./setup.sh                                       # create/refresh .venv (idempotent; --recreate)
./run.sh compare --pack <id>                     # the CLI (S1: compare, index rebuild, models)
.venv/bin/python -m pytest -q                    # default suite, network-free
.venv/bin/python -m pytest -m live               # real LM Studio, opt-in (none exist until S2)
.venv/bin/ruff check .
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
