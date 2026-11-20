# model-bench — agent working context

## Project in one sentence

A standalone, human-started harness that measures one local model (or one decision-model
candidate) against one versioned task pack, stores the result with a full environment fingerprint,
and compares runs within a role. `README.md` is the human-facing quickstart.

## Live spec

This component's **live spec** (root `AGENTS.md`, live-spec bullet) is two documents:
`docs/DESIGN.md`, read whole (CLI and exit codes, run lifecycle, LM Studio contract, pack
validation, scoring per role, typed decision points, verdict statistics, reports, and the §11
index of every shipped file), and `docs/FORMATS.md`, read by lookup (pack, stored run record,
fingerprint, candidate, `host.json`, vectors). They state current behavior only; cite them as
`DESIGN.md §n` / `FORMATS.md §n`, and never renumber. A contract change updates them at the
feature's close, and a new shipped file gets a `DESIGN.md` §11 row. History is `docs/HISTORY.md`,
open work `docs/BACKLOG.md`.

## Current state

Seven CLI commands; eight shipped packs covering seven roles (the five roles FR-21 of `docs/requirements/small-model-benchmarking.md` names, plus
`join-in-judge` and `message-type-classifier`); three decision-point packs
(`packs/*-synthetic/`) and the candidates in `candidates/`, run on the `decisions`/`handoff`
surfaces; `benchSchemaVersion` is `1`. Stored `model:decisions`/`model:handoff` records are
committed under `results/runs/`. `tool-caller`'s known-answer validation ships flagged "not
reproduced", and `chat-responder` scores the deterministic layer only (judged quality is
deferred, `docs/BACKLOG.md`). **Every unit, defect, gate and live-run result is in
`docs/HISTORY.md`**; this section states only what is true now.

## Load-bearing invariants

Rules a future change can silently break; each cites the spec section that states the behavior.

- **Fingerprint** (`FORMATS.md` §3). `ARM_KINDS` is never derived from the forbidden-field mapping
  (every model record would refuse on write: a green mapping and a dead harness). The
  missing-`callSurface` sentinel is `""` on a model record and `None` on a deterministic one;
  collapsing the split is a real bug. Forbidden sets are a set difference, never a list. `null`
  fails both tiers. `validate()` checks each residency element's whole key set.
- **`stats.py` implements `docs/plans/small-model-benchmarking-ml.md` and no other source**
  (`DESIGN.md` §8). `stats.percentile` is the only percentile estimator; `results.py` imports it.
  `exact_paired_quantiles` is a quantile of the exact multinomial resample distribution, not a
  second estimator, and must not be renamed to satisfy a stale grep. `resolving_power`'s
  `design_effect`/`basis`/`unit_kind`/`alpha_family`/`alpha_mdd` are keyword-only with no defaults;
  Rule 7 lives inside `verdict()`.
- **`decision_stats.py` implements `docs/plans/typed-decision-models-ml.md` §Q3–§Q6 and no other
  source.** It may call `stats` primitives, never `stats.verdict`/`stats.holm_steps` (either makes
  `holds` easier); `tests/test_decision_stats.py` pins that by an AST read and a patched-out run.
- **A decision point's bar is requirement constants, never pack data** (`DESIGN.md` §7), each
  mutation-pinned. `compare_report` gets the chat arm alone as `runs` and candidates only as
  `replicates` (`cli._replicate_groups`, never last-wins); comparison-wide blocks precede the
  bar-kind split (`report._comparison_block`); drop counts are eval-only; FR-10 is single-arm.
- **Honesty rules** (`DESIGN.md` §10): every printed bound takes its own rounding direction, α and
  denominator (`stats.format_floor_pp` is the only home of the floor's direction; assert through
  it, never via `round(...)`); nothing that shapes a decision carries a default; an item's outcome
  for a metric is declared, never inferred; a Wilson interval prints only over the analysis unit
  (compare `BinaryMetric.unit` to `roles.unit_kind`, never to `PackRef.analysisUnit`); Holm needs
  two passes, zipped `strict=True`.
- **The analysis unit is pack data**, resolved from `PackRef.analysisUnit`; the repeated-unit check
  in `PairedOutcomes.from_units` is a backstop, not the mechanism.
- **A between-item residency snapshot goes through `runner._between_item_residency`**, never a
  bare `lmstudio.residency()`: a failed probe re-probes, and the snapshot is then `[]`, never the
  previous one (`DESIGN.md` §3.5).
- **A stored run record never persists a reply text.** Auditing a scorer against a real reply
  needs a fresh live call with the scorer's own `build_messages` at `temperature: 0.0`.
- **`sessionId` is top level on a stored `RunResult`**, not in `fingerprint`; read `run.sessionId`.
- **`test_every_stored_run_record_still_validates_unchanged`** freezes every stored record: a
  change to what a stored record means is a schema decision (`FORMATS.md` §4.1), not an edit.
- **`tests/test_package.py`** pins `modelbench.__version__` to the installed metadata, which stamps
  `benchVersion`.
- **A public name starting with `test` is collected by pytest** in every module that imports it
  (hence `models_with_stored_results`).
- **Sentence scoping in `scoring/` is `grounding._SENTENCE_BOUNDARY_RE`** (a `.` between digits is
  not a boundary); a new same-sentence heuristic reuses it.
- **`compare --negative-control` duplicates one stored record** as both arms (a wiring check,
  refused on a decision-point pack); `--models` keeps each key's newest run.

## Hard rules

- **Zero runtime dependencies.** Standard library only; dev extras are `pytest` and `ruff`. The one
  reversal trigger: add `numpy` if a pack's corpus exceeds about 1,000 documents or scoring
  exceeds about 5 s.
- **Standalone.** No runtime path reads outside `model-bench/`. Outside data is copied in by the
  one-way, human-invoked `scripts/refresh_golden.py` and `scripts/import_decision_pack.py`. The
  default suite holds to the same rule: tests that live-verify a copied-in asset against its
  `falkor-chat/` source carry `tests/conftest.py`'s `requires_falkor_chat` skip guard. The reverse
  edges are test-only: falkor-chat's and decision-models' suites byte-copy the
  `tests/fixtures/decision-contract/` files they list (`FORMATS.md` §7).
- **No aggregate across roles, no gate, no scheduler**, enforced structurally (`DESIGN.md` §1.2).

## Conventions

- **`live` pytest marker** (`addopts = '-ra -m "not live"'`): real-LM-Studio tests are deselected
  by default; opt in with `pytest -m live`.
- **Host venv, no Docker**: `setup.sh`/`run.sh` mirror `mcp-monitor/`.
- **Four fingerprint fields are operator-attested** (`lmStudioAppVersion`, `kvCacheSetting`,
  `hostRamGb`, `otherResidentWorkloads`): no programmatic source exists on this LM Studio build.
  They live in gitignored `host.json` and are copied into every run record (`DESIGN.md` §3.9).
- **Empty `docs/` subdirectories are held by `.gitkeep`.**
- **A guard's reach lives in an asserted constant, not in prose.** A module-level set or table a
  guard consults needs a test that binds it to another declaration of the same set, or asserts a
  distinct behavioural consequence per member. Mutate the constant alone both ways: a shrink and a
  widen must each redden; a table also takes a third mutation (move a value to another key). A
  docstring may not claim a reach (*only*, *every*, *never a sixth*) without that. Nothing is
  exempt, including a constant built from the runtime: it binds to its own source, and what it
  refuses is an augmentation (`<derived> | {extra}`), never a re-derivation.

## Commands

All need `model-bench/` as the working directory: the repo has no root pytest config.

```bash
./setup.sh                                       # create/refresh .venv (idempotent; --recreate)
./run.sh <command>                               # the CLI (DESIGN.md §2)
.venv/bin/python -m pytest -q                    # default suite, network-free
.venv/bin/python -m pytest -m live               # real LM Studio, opt-in
.venv/bin/ruff check .
```

## Documentation map

The live spec is above. The original feature's documents live at the **repo root**, not under
`model-bench/docs/`: `docs/requirements/small-model-benchmarking.md` → `docs/plans/small-model-benchmarking.md`
+ `docs/plans/small-model-benchmarking-ml.md` (the statistics) → `docs/reviews/small-model-benchmarking.md`,
and the typed-decision-models family (`docs/*/typed-decision-models*.md`). Everything newer
(requirements, plans, reviews, test plans and reports) goes under `model-bench/docs/`.
`docs/BACKLOG.md` and `docs/HISTORY.md` here are this component's living logs (root `AGENTS.md`,
module documentation convention).
