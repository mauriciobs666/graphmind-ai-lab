# Small-Model Benchmarking — Manual-Additions Verification Test Plan

> **Status:** active · **Owner:** `qa-engineer` · **Tracks:** — (manual walkthrough verification)

## Scope & objective

`tico` extended `model-bench/docs/manuals/small-model-benchmarking.md` (currently uncommitted)
with five new pieces: Walkthrough 7a (`rank`), Walkthrough 8 (`index rebuild`), an "On-disk data
shapes" section, a "Configuration & integration" section, and two new FAQ entries. Earlier
walkthroughs (setup, attest, run, compare) were verified live in a prior pass
(`docs/test-reports/small-model-benchmarking-s4..s7-report.md`) and are **out of scope** here.

Objective: walk each new piece of the manual against the real CLI and real stored data under
`model-bench/results/runs/`, exactly as a reader following the manual would, and flag anything the
manual claims that the running tool does not actually do.

## References

- `model-bench/docs/manuals/small-model-benchmarking.md` (working tree diff, uncommitted) — the
  spec under test.
- `model-bench/modelbench/cli.py` and `rank`/`index` argument parsers (`./run.sh rank --help`,
  `./run.sh index --help`) — contract source.
- `model-bench/packs/embedder-graphrag-retrieval/pack.json` (single `headlineMetric: mrr`) and
  `model-bench/packs/guard-judge-understanding/pack.json` (two `verdictMetrics`, no
  `headlineMetric`) — the two pack shapes the manual claims `rank` renders differently.
- `model-bench/results/runs/*.json`, `model-bench/host.json` — real stored data to spot-check the
  "On-disk data shapes" section against.
- `model-bench/docs/HISTORY.md` (S8 close entry) — background; this increment postdates it.

## Risk assessment

Highest risk: the two new CLI-behavior claims (`rank`'s per-metric-table branching, the
`--reference`/`--footprints` flags and their exit-2 error paths, filename-sequence collision
avoidance) because they are new, specific, executable claims that are easy to get subtly wrong in
prose (e.g. exit code, column presence, which flag triggers which table shape). Second: the
on-disk shape claims, because they are checked against real files and any drift between code and
manual is a direct defect. Lower risk: the two FAQ entries, since they mostly restate claims
already tested elsewhere in the manual (exit codes are also documented in Overview per the diff's
own note) — verify but don't need independent new evidence beyond what the CLI runs already show.

**Explicitly not tested:** `run`/`attest`/`validate`/`compare`/`models` (prior pass covers them,
unchanged here); actually launching LM Studio (task states read-only driving of stored results
suffices and no live model claim was added); the mermaid diagram's rendering (visual, not
behavioral — read for factual accuracy only, which is `analyst`'s half per this manual's own
convention, not re-litigated here).

## Test items

| ID | Title | Preconditions | Steps | Expected result | Priority | Type |
|---|---|---|---|---|---|---|
| TP-001 | `rank` on a single-headline-metric pack | `.venv` set up; ≥3 stored results for `embedder-graphrag-retrieval` | `./run.sh rank --pack embedder-graphrag-retrieval` | Exit 0. One table, one row per **distinct model key** (not per file). Descriptive CI shown by default. No verdict/p-value column. Report saved under `reports/`. | High | Functional |
| TP-002 | `rank` on a two-co-equal-metric, no-headline pack | ≥3 distinct model keys with stored `guard-judge-understanding` results | `./run.sh rank --pack guard-judge-understanding` | Exit 0. **Two** tables (one per verdict metric: `falseAdvanceRate`, `falseSuspendRate`), not one merged table. | High | Functional |
| TP-003 | `--reference` with a valid model key | A stored model key for the pack used | `./run.sh rank --pack embedder-graphrag-retrieval --reference <key>` | Exit 0. Adds a verdict column using "distinguishably better/worse"/"not distinguishable" language, Holm-Bonferroni-corrected, no raw uncorrected p-value printed. | High | Functional |
| TP-004 | `--reference` with an unknown model key | none | `./run.sh rank --pack embedder-graphrag-retrieval --reference does/not-exist` | Exit **2**. No new file written under `reports/`. | High | Negative/boundary |
| TP-005 | `--footprints` with a valid file | flat `{"modelKey":"display string"}` JSON | `./run.sh rank --pack embedder-graphrag-retrieval --footprints <file>` | Exit 0. Footprint column renders the string verbatim for matching keys; dash/raw for missing/malformed per-key values. | Medium | Functional |
| TP-006 | `--footprints` with a malformed file | invalid JSON file | `./run.sh rank --pack embedder-graphrag-retrieval --footprints <bad file>` | Exit **2**. | High | Negative/boundary |
| TP-007 | Filename sequencing vs. `compare` | at least one `compare` report already exists for a pack/day | Run `rank` twice same day for the same pack; inspect `reports/` filenames | `rank` files use a `-rank-` segment distinct from `compare`'s own segment; a same-day re-run does not overwrite the earlier `rank` report. | Medium | Functional |
| TP-008 | `index rebuild` regenerates the index | `results/index.csv` state captured before | `./run.sh index rebuild`; compare row count to `ls results/runs/*.json \| wc -l` | Exit 0. `results/index.csv` row count (minus header) equals the number of files in `results/runs/`. | High | Functional |
| TP-009 | `host.json` shape | real `host.json` present | Read `host.json` | Contains exactly the four fields the manual names: `lmStudioAppVersion`, `kvCacheSetting`, `hostRamGb`, `otherResidentWorkloads`. | Medium | Data-shape |
| TP-010 | `results/runs/*.json` shape | real stored run file | Read one real run file | Has `fingerprint`, `aggregates`, `items` keys, and a **top-level** `sessionId` (not nested under `fingerprint`). | Medium | Data-shape |
| TP-011 | FAQ: compare vs. rank | — | Compare FAQ prose against TP-001/003 observations | Guidance ("compare = two-model question; rank = whole field, only rank supports `--reference`") matches observed CLI behavior. | Low | Documentation |
| TP-012 | FAQ: exit-code table | — | Compare FAQ exit-code list against TP-004/TP-006 and `--help`/argparse behavior | `0`/`2`/`3`/`4`/`5` set matches; `2` demonstrably covers the two new `rank` usage errors. | Medium | Documentation |

## Environment & data setup

- Working directory: `model-bench/` (component-local `pytest`/CLI rule — never run from repo
  root).
- `.venv` already present (confirmed `.venv/bin/python -V` → 3.12.3); no setup needed.
- All test items are driven **read-only** against the existing `results/runs/` corpus (95 stored
  files across 5 packs) — no live LM Studio call is needed or made.
- `results/index.csv` does not currently exist and is **not git-tracked** (confirmed via
  `git ls-files results/index.csv` → empty, and it is not `.gitignore`d by name but has no history)
  — so TP-008 has no committed state to preserve, but the run's row-count check still stands in
  for a real before/after diff.

## Entry / exit criteria

- **Entry:** CLI installed and runnable (`./run.sh --help` succeeds); stored results exist for
  both target packs. Confirmed.
- **Exit:** every test item above executed with real command output as evidence, and the working
  tree is checked (`git status`) to be free of unintended mutations to committed files before
  reporting.

## Out of scope

Live LM Studio runs, the four already-verified walkthroughs, the mermaid diagram's rendering
correctness, and the `packs/`/`PROVENANCE.md` factual claims in "Configuration & integration"
(the catalog-sweep numeric claims about `granite-embedding-278m-multilingual` are an `analyst`-half
factual check against `reports/catalog-sweep-2026-09-19-consolidated.md`, not a behavioral one).
