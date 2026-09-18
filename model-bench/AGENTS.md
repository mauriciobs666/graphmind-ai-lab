# model-bench — agent working context

## Project in one sentence

A standalone, human-started harness that measures one local model against one versioned task pack,
stores the result with a full environment fingerprint, and compares runs within a role — see
`README.md` for the human-facing quickstart and `docs/plans/small-model-benchmarking.md` (repo root)
for the full design.

## Current state

**All eight stages (S0–S8) are closed — the feature is delivered.** Every one of the five FR-21
roles has a scorer module in `modelbench/scoring/`: `retrieval.py` (`embedder`), `classification.py`
(`guard-judge`), `extraction.py` (`nlq-generator`), `toolcalls.py` — the package's first
`ConversationScorer`, for the multi-turn `tool-caller` role — and `grounding.py` (`chat-responder`,
deterministic layer only per FR-21a; the judged-quality layer stays deferred, `docs/BACKLOG.md`).
Every pack has its own required live run against a real LM Studio. Two of those runs found real
problems and shipped honestly rather than quietly: `tool-caller`'s known-answer validation
(`qwen/qwen3-4b-2507` vs. `mistralai/ministral-3-3b`) did not reach statistical significance at
this pack's sample size on any configuration tried, and ships flagged "not reproduced" rather than
massaged into one; `chat-responder`'s `groundingRate` metric had a real abstention-detection defect
found on its own required live run, fixed over a five-round gated chain, and confirmed fixed on a
fresh re-run. `modelbench/` also holds `fingerprint`, `results`, `stats`, `report`, `roles`, `packs`
(the loader: `load_pack`/`validate_pack`, content hashing, the AST import allowlist), `lmstudio`,
`hostinfo`, `tooling`, `convo` (turn assembly and driving), and `runner` (`RunConfig`, the
`ItemScorer`/`ConversationScorer` scorer-seam Protocols, `run_pack`'s capture-order orchestration).
The CLI ships all six commands (`compare`, `index rebuild`, `models --tested`, `attest`, `validate`,
`run`). To run a pack: `README.md`'s Quick start and "What a pack is, and how to add one."
**Every unit, defect, gate, and live-run result is in `docs/HISTORY.md`** — this section states only
what is true now, not how it got that way.

## Load-bearing invariants

Durable facts about the shipped code that a future change can silently break. `docs/HISTORY.md` has
the story behind each; this section states only the rule.

**The fingerprint has two discriminators and one derived key.** `REQUIRED_BY_SCHEMA[schema]` and
`FORBIDDEN_BY_ARM_PROFILE[armProfile]` are keyed by `armProfile` (`model:chat` / `model:embeddings`
/ `deterministic`), which `Fingerprint` derives from `armKind` (still two-valued — every
`armKind == "model"` filter elsewhere is unaffected) and `callSurface` (required, no default,
`None` **iff** deterministic) — both checked **before any mapping is consulted**, since without a
surface there is no profile to report fields against. **`ARM_KINDS` must never be derived from the
forbidden mapping** — its members would become the three profiles, `armKind == "model"` would fail
membership, and every model record would refuse on write: a green mapping and a dead harness.
`from_dict`'s missing-`callSurface` sentinel is `""` on a model record and `None` on a deterministic
one, and collapsing that split is a real bug shape: it lets a surface something had and lost read as
one that was never written, or a correct reference-arm record read as carrying a forbidden surface.
Forbidden sets are a **set difference, never a list**. `validate()` checks each residency element's
**whole key set** (`{id, state}`), never presence alone — the gap that let a retired fixture element
validate and ship green.

**`stats.py` implements `docs/plans/small-model-benchmarking-ml.md` (`-ml`) and no other source** —
every formula, constant, threshold and verdict string is that note's, cited by section; don't
restate them here. `stats.percentile` is the **only** percentile/quantile estimator in the package
(`-ml` §11.2, Hyndman-Fan type 1); `results.py` imports that object rather than defining a second
one — two copies is what once let `index.csv` compute `latencyMsP95` at the wrong percentile and
stay green. `exact_paired_quantiles` is not a second percentile estimator — it is `-ml` §3.4 Rule
4's quantile of the *exact multinomial resample distribution* — and must never be renamed to make a
stale grep read 1 (plan v1.17 §11.10(3)): surviving a rename is the property it exists to have.
`resolving_power`'s `design_effect`/`basis`/`unit_kind`/`alpha_family`/`alpha_mdd` are keyword-only
with **no defaults**; `min_detectable_difference` takes `n_effective: float`, so a raw observation
count raises. **Rule 7 is enforced inside `verdict()`** — no path returns `distinguishable` below
`resolving.observable_floor` — and it **raises** on the `mcnemar-exact` path (a theorem there) while
it demotes-and-names on the substitute one.

**Five honesty rules that are easy to break silently, and what holds each in place:**

- **Every printed bound takes the rounding direction, α, and denominator that keep *its own* claim
  true.** The floor truncates at the unadjusted α over the unfloored `n_eff`; the MDD ceilings at
  α/k over the floored one — `ResolvingPower` carries both αs and the printed line names both.
  `stats.format_floor_pp` is the *only* place the floor's direction lives; assert through it, never
  via `round(...)` in a test, which asserts the presentation layer against itself.
- **Nothing that shapes a decision carries a default.** `RunResult.designEffect`/`basis`,
  `BinaryMetric.unit`, `PackRef.seed`, `holm_steps`' `alpha`, and both bootstraps' `levels`/`clamp`
  are all required — a default would silently rebuild the retired anti-conservative behavior.
  `PackRef.seed`'s only consumer is the continuous bootstrap; the paired *binary* interval is a
  closed form that takes no seed, so `report.py` never passes or prints one there. `resolving_power`
  refuses any `design_effect` **not `>= 1.0`** (not `<= 0`) — below 1 it inflates effective *n* and
  shrinks both printed bounds. The predicate is that way round at all four sites that use it
  (`resolving_power`, `verdict()`, `envelope_arms`, `paired_cluster_bootstrap`) because `< 1.0` is
  `False` for a NaN, and a NaN would otherwise silently widen both bounds to the full support.
- **An item's outcome for a metric is *declared*, never inferred.** `ItemResult.scored_outcome` has
  three answers only: a metric absent from `scoreable`, or declared `False`, is no outcome; declared
  `True` **must** carry a `counts` entry, and one that doesn't is refused (`IncompleteItemRecord`),
  never read as zero. `report.py` infers nothing.
- **S2's scorers owe two things, both enforced at S1 so the net exists before the first real scorer
  does:** emit a `counts` entry for every metric you declare scoreable, and derive `aggregates` from
  the same `items` in one pass. A mismatch is **excluded from the comparison and named**
  (`report._aggregate_item_mismatches`) — never raised, never silently dropped per metric.
- **A Wilson interval prints only over the analysis unit.** `report.py` compares `BinaryMetric.unit`
  against the role's own unit kind (`roles.unit_kind(pack.role)`) — comparing against
  `PackRef.analysisUnit` instead is a different vocabulary (a denominator noun vs. a `pairingKey`
  component name) and is always false, silently.
- **Holm needs two passes.** `compare_report` computes every paired table first, then
  `stats.holm_steps`, then the verdicts, zipped `strict=True` so a short ladder can't silently drop
  a metric. `holm_steps`' `alpha` is required, so `ALPHA_FAMILY` stays the only `0.05` in the code.

**The analysis unit is pack data, never a call-site choice.** `report.py` resolves it from
`PackRef.analysisUnit` (plan §3.3 fixes it by rule as `pairingKey[0]`). `PairedOutcomes.from_units`
raising on a repeated unit id is a backstop, not the mechanism — it fires only when the id handed to
it is the *cluster* key.

**`tests/test_package.py` is not a placeholder** — it pins `modelbench.__version__` to the installed
distribution metadata, which is what stamps `benchVersion` into every run record. Keep it: pytest
exits 5 (`EXIT_NOTESTSCOLLECTED`) on an empty suite, so configuring that exit code away would hide a
real collection breakage later rather than fix anything.

**A public name starting with `test` is collected by pytest as a test** in every module that imports
it — which is why FR-17a's function is `models_with_stored_results`, not `tested_models`.

**Sentence scoping inside `modelbench/scoring/` is `grounding._SENTENCE_BOUNDARY_RE`** — `.`/`!`/`?`,
except a `.` flanked by digits on both sides — never a bare `.` split (`rfind(".")`, `split(".")`).
This pack domain is decimal-heavy (`4.5`, `18.5%`, dollar amounts, page references), and a bare
split reads the decimal point as a sentence end, truncating whatever span was scoped to that boundary
and silently reopening the abstention-detection misclassification the span exists to prevent. A new
same-sentence heuristic reuses that regex; a second boundary definition is a second copy of the bug.

## Hard rules (they are design constraints, not preferences)

- **Zero runtime dependencies.** stdlib only — `urllib.request` for HTTP (falkor-chat's own
  precedent in `falkorchat/transport.py`), `json`/`math`/`statistics`/`hashlib`/`subprocess`. Dev
  extras are `pytest` and `ruff`, nothing else. Old results stay reproducible only if the tool that
  produced them still installs. The one stated reversal trigger (plan §3.2): add `numpy` if a pack's
  corpus exceeds ~1 000 documents or scoring exceeds ~5 s.
- **Standalone — FR-23.** No runtime code path reads any path outside `model-bench/`. Golden data
  from `falkor-chat` is *copied in* and versioned here with provenance; the one-way importer
  `scripts/refresh_golden.py` is a human-invoked maintenance script and is never reachable from a
  run. Nothing in `falkor-chat` changes, in either direction, ever. **The default test suite holds
  to the same rule too**: 3 tests that live-verify a copied-in golden asset against its real
  `falkor-chat/` source carry `tests/conftest.py`'s `requires_falkor_chat` skip guard — a plain
  `skipif`, deliberately distinct from the `live` marker below (a different precondition) — so
  `pytest -q` skips them cleanly (reported in the summary line) rather than failing when
  `falkor-chat/` is absent or renamed.
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
  this LM Studio build (plan §2.3, live-probed; the S2 `R-1` probe re-confirmed no such source on
  two independent live loads). They live in a gitignored `model-bench/host.json` and are copied into
  every run record so a record stays self-contained. Plan §3.4 has the staleness trip-wire that
  keeps them honest.
- **Empty `docs/` subdirectories are held by `.gitkeep`** (repo precedent), so the module
  documentation convention's layout survives a clone before its first document exists.
- **A guard's reach lives in an asserted constant, not in prose.** A module-level set or table a
  guard consults — a required-key set, an allowlist, an exemption list, a role→unit map — needs a
  test that binds it to another declaration of the same set, or asserts a distinct behavioural
  consequence per member; never merely that the guard accepts what its own constant contains, which
  is true of any constant. **To know you have one, mutate the constant alone both ways: a shrink and
  a widen must each redden.** Shrink-only is not a pin — a fixture is covering it, or its widen stays
  inert until the second declaration is bound. **A table takes a third mutation:** move a value to
  another key. Its keys and its contents are two pins, and a value the test reads back out of the
  table is asserted against itself. Absent that, the docstring may not claim a reach (*only*,
  *every*, *never a sixth*). **Nothing here is exempt, including a constant built wholesale from the
  runtime** — that one binds to its own source, and the equality is not circular: what it refuses is
  an **augmentation** (`<derived> | {extra}`, the shape a hand-added exception takes), never a
  re-derivation, which is equivalent by construction and stays green correctly. Audits:
  `docs/reviews/small-model-benchmarking-impl.md` Pass 14 and Pass 16.

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
component existed and its footnote says to leave it there. Everything written *from here on* — this
component's own requirements, plans, reviews, test plans and reports — goes under
`model-bench/docs/`, which is why those subdirectories already exist. `docs/BACKLOG.md` and
`docs/HISTORY.md` here are this component's living logs (module documentation convention, root
`AGENTS.md`).
