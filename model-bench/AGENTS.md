# model-bench — agent working context

## Project in one sentence

A standalone, human-started harness that measures one local model against one versioned task pack,
stores the result with a full environment fingerprint, and compares runs within a role — see
`README.md` for the human-facing quickstart and `docs/plans/small-model-benchmarking.md` (repo root)
for the full design.

## Current state

**Stage S2, most of the way through — the outside world is reachable, but no run has ever been
executed end to end.** `modelbench/` holds `fingerprint`, `results`, `stats`, `report`, `roles`,
`cli`, `__main__` and — S2's — `packs` (the real loader: `load_pack`/`validate_pack`, content
hashing, the AST import allowlist, the row-count identity), `lmstudio`, `hostinfo`, `tooling` and
`convo` (`assemble` and the bounded per-turn `drive`); the CLI ships `compare` (with
`--negative-control`), `index rebuild`, `models --tested` and `attest`.

**What S2 still owes, and each one's live consequence.** `runner.py` does not exist, and with it
neither do the timing carriers `results.py` has no declaration of yet — `CallTiming`, `ItemTiming`,
`ItemResult.timing` (still a stored `latencyMs`), `LatencyBlock`, `RunResult.latency` and
`RunResult.attestationTripWire` — so nothing in the tree produces a latency figure and `report.py`'s
latency slots are unreachable. The `validate` and `run` commands are absent, and
`tests/test_cli.py::test_s2s_remaining_commands_are_not_shipped_yet` asserts **those two** exit `2`,
so that half of the stage boundary is checked rather than promised (`attest` shipped and left the
assertion).
`docs/plans/small-model-benchmarking.md` §4 sequences S2–S8; `docs/HISTORY.md` carries the unit
trail.

**The fingerprint has two discriminators and one derived key, and `ARM_KINDS` is deliberately not
derived from the forbidden mapping.** `REQUIRED_BY_SCHEMA[schema]` and `FORBIDDEN_BY_ARM_PROFILE`
are keyed by `armProfile` — `model:chat`, `model:embeddings`, `deterministic` — which `Fingerprint`
derives from `armKind` (still two-valued, and every `armKind == "model"` filter in `results.py` and
`report.py` is unchanged by that) and `callSurface` (required, no default, `None` **iff**
deterministic). Both are members of no required set and are checked **before any mapping is
consulted**: without a surface there is no profile, so there is no contract to report the fields
against. **`from_dict`'s missing-key sentinel for `callSurface` is `""` on a model record and
`None` on a deterministic one, and the split is load-bearing**: `None` is the deterministic arm's
*value*, so a plain `d.get("callSurface")` reports a stored `null` — a surface something had and
lost — as one that was never written, while a blanket `""` reports a correct reference-arm record
as carrying a forbidden surface. Re-deriving `ARM_KINDS` from the profile mapping makes its members the three profiles, so
`armKind == "model"` fails the membership test and **every model record refuses on write** — a green
mapping and a dead harness. The forbidden sets stay a union-minus-mine **set operation, never a
list**; that is what forbids `runtimeName`, `runtimeVersion`, `temperature` and `maxTokens` on an
embeddings arm without anyone typing those four names. And `validate()` checks each residency
element's whole key set (`{id, state}`, both non-empty strings), because the `present` tier checks
presence and **never** element shape — the gap that let a retired fixture element validate and ship
green.

**`stats.py` implements `docs/plans/small-model-benchmarking-ml.md` and no other source.** Every
formula, constant, threshold, tolerance and verdict string is that note's, cited by section; the
plan deliberately does not restate them, and neither should this file. Its shape is §3.4's **seven**
binding rules, written so the anti-conservative version does not typecheck. **`stats.percentile`
is the only percentile or quantile *estimator* in the package** (`-ml` §11.2, Hyndman-Fan type 1,
exact-rational level); `results.py` imports that object rather than defining one, since two copies
is what let `index.csv` compute `latencyMsP95` at the 50th percentile and stay green.
`exact_paired_quantiles` is not a second one — it is `-ml` §3.4 Rule 4's quantile of the *exact
multinomial resample distribution*, the same operator on a known distribution rather than on a
sample, and it shares `percentile`'s level refusals through `_check_level`. **Do not rename it to
make §11.10(3)'s grep read 1**: plan v1.17 rules that residual's target to be two survivors named,
because surviving a rename is the property it exists to have. `resolving_power`'s
`design_effect`/`basis`/`unit_kind`/`alpha_family`/`alpha_mdd` are keyword-only **with no
defaults**, `min_detectable_difference` takes `n_effective: float` so a raw observation count
raises, and **Rule 7 is enforced inside `verdict()`** — no path returns `distinguishable` below
`resolving.observable_floor`, compared against the exact float and never the printed one, and it
**raises** on the `mcnemar-exact` path (there it is a theorem) while it demotes-and-names on the
substitute one.

**Five honesty rules that are easy to break silently, and what holds each in place.**

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
- **Nothing that shapes a decision carries a default.** `RunResult.designEffect`/`basis`,
  `BinaryMetric.unit`, `PackRef.seed`, `holm_steps`' `alpha`, and both bootstraps' `levels` and
  `clamp` are all required: in each case the value a forgetful caller wants is the
  anti-conservative or unreproducible one, so a default rebuilds gate B-1 at that seam. `levels`
  is `(Fraction, Fraction)` because a `k`-member continuous family's level is `alpha/(2k)`, which
  no decimal unit expresses, and the conventional `2.5/97.5` is exactly the value that prints a
  plausible interval beside a family that was never corrected; `clamp`'s `(-1.0, 1.0)` is right
  for a difference of proportions and false for `sep_z`. **`PackRef.seed`'s consumer is `-ml`
  §3.2d's continuous bootstrap alone** — the paired *binary* interval is a closed form that takes
  no seed, so `report.py` neither passes nor prints one, and re-adding a seed parenthetical there
  would name a resample that does not run. `resolving_power` refuses any `design_effect` **not
  `>= 1.0`** at construction, not `<= 0` — below 1 it *inflates* effective *n* and shrinks both
  printed bounds. **The predicate is that way round at all four sites** — `resolving_power`,
  `verdict()`, `envelope_arms`, `paired_cluster_bootstrap` — because `< 1.0` is `False` for a NaN,
  and a NaN widens both bounds to `nan`, which the clamp turns into the full `(-1, 1)` support
  printed as a real interval; do not simplify any of them. Only `verdict()`'s message names itself,
  which is what keeps the layers orderable. The legacy fallbacks live in `from_dict` only, where
  they are §3.4.3 reader rules.
- **An item's outcome for a metric is *declared*, never inferred.** `ItemResult.scored_outcome`
  is the only place that decides, and it has three answers: `metric` absent from `scoreable`, or
  declared `False`, is **no outcome** (the row leaves the paired table and lands in the §4.3
  tally); declared `True` **must** carry a `counts` entry, and one that does not is refused
  (`IncompleteItemRecord`), never read as a zero. `report.py` infers nothing — its two old defaults
  (absent = scoreable, absent count = failure) turned an arm holding no data at all into
  *"+100.0 pp, p=0.002"*. A metric with an empty paired intersection gets an explicit refusal and
  no resolving power, because `n_effective` of zero is not a small sample.
- **Two halves of one contract on S2's scorers**, both enforced at S1 so the net exists before the
  first real scorer does: emit a `counts` entry for every metric you declare scoreable, **and
  derive `aggregates` from the same `items` in one pass**. An arm whose declared `n` for a
  pre-registered verdict metric disagrees with its own scoreable-item count is **excluded from the
  comparison and named** in the `INVALID RESULTS EXCLUDED` block (`report._aggregate_item_mismatches`,
  plan §4 S1 done-condition 10) — never raised, which would abort outside §3.6a's exit-code set,
  and never suppressed per metric, which would leave a partly-trusted arm in the table. A record
  carrying the sibling malformation is quarantined a seam earlier, on read
  (`results._item_problems`); the report-time check catches it too rather than letting
  `scored_outcome`'s refusal escape.
- **A Wilson interval prints only over the analysis unit.** `-ml` §4.4: *"Never print a Wilson
  interval over a turn-pooled count."* `report.py` compares `BinaryMetric.unit` against the role's
  unit kind; a pooled count prints its `k/n` and no interval. **`BinaryMetric.unit` and
  `PackRef.analysisUnit` are different vocabularies** — a denominator noun (`item`, `turn`) against
  a `pairingKey` component name (`itemId`) — so the predicate is always
  `metric.unit == roles.unit_kind(pack.role)`; comparing against `analysisUnit` is never true and
  fails silently.
- **Holm needs two passes.** The step a metric is tested at depends on every other member's
  p-value, so `compare_report` computes every paired table first, then `stats.holm_steps`, then the
  verdicts, zipped `strict=True` so a short ladder cannot drop a metric.
  `holm_thresholds` was replaced by `holm_steps` because a threshold without the
  step-down stop is unusable — and the rendered family table carries a `decision` column for the
  same reason. `holm_steps`' `alpha` is required, so `ALPHA_FAMILY` stays the only `0.05`.

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
- **A guard's reach lives in an asserted constant, not in prose.** A module-level set or table a
  guard consults — a required-key set, an allowlist, an exemption list, a role→unit map — needs a
  test that binds it to another declaration of the same set, or asserts a distinct behavioural
  consequence per member; never merely that the guard accepts what its own constant contains,
  which is true of any constant. **To know you have one, mutate the constant alone both ways: a
  shrink and a widen must each redden.** Shrink-only is not a pin — a fixture is covering it, or
  its widen stays inert until the second declaration is bound. **A table takes a third mutation:
  move a value to another key.** Its keys and its contents are two pins, and a value the test
  reads back out of the table is asserted against itself. Absent that, the docstring may not claim
  a reach (*only*, *every*, *never a sixth*). **Nothing here is exempt, including a constant
  built wholesale from the runtime** — that one binds to its own source, and the equality is not
  circular: what it refuses is an **augmentation** (`<derived> | {extra}`, the shape a hand-added
  exception takes), never a re-derivation, which is equivalent by construction and stays green
  correctly. Audits: `docs/reviews/small-model-benchmarking-impl.md` Pass 14 and Pass 16 — whose
  P16-5 standing exception is withdrawn: it kept `packs._STDLIB_MODULE_NAMES` unpinned, and one
  name appended there silently widens what every pack may import.

## Commands

All of these need `model-bench/` as the working directory — the repo has no root pytest config, so
from the repo root pytest ignores this component's `testpaths` and walks the whole monorepo
(measured: 8 collection errors, exit 2).

```bash
./setup.sh                                       # create/refresh .venv (idempotent; --recreate)
./run.sh compare --pack <id>                     # the CLI (compare, index rebuild, models, attest)
.venv/bin/python -m pytest -q                    # default suite, network-free
.venv/bin/python -m pytest -m live               # real LM Studio, opt-in (3 exist: lmstudio, hostinfo)
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
