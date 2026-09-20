# Small-Model Benchmarking — Manual-Additions Verification Test Report

> **Status:** active · **Owner:** `qa-engineer` · **Tracks:** — (manual walkthrough verification)

## Summary

Tested the five new pieces `tico` added to `model-bench/docs/manuals/small-model-benchmarking.md`
(working tree, uncommitted at test time): Walkthrough 7a (`rank`), Walkthrough 8 (`index rebuild`),
"On-disk data shapes," "Configuration & integration," and two new FAQ entries. Driven entirely
read-only against the real, already-populated `model-bench/results/runs/` corpus (95 stored files
across 5 packs) — no live LM Studio call was made or needed. `.venv` was already set up
(`.venv/bin/python` 3.12.3); no bootstrap required. Tested 2026-09-20 against the working tree as
of `git rev-parse HEAD` at session start (no relevant commits landed during the run).

**Overall verdict: not fully trustworthy as written.** Two real defects were found, one of them
severe: `rank --reference` **crashes** (uncaught internal error, exit 2, no report written) on any
pack whose verdict metric is continuous rather than boolean — which includes
`embedder-graphrag-retrieval`, the exact pack Walkthrough 7a's own example command uses. And the
manual's central claim about `results/index.csv` being rebuilt automatically by "every command
above" (Walkthrough 8) / "every command that reads or writes a result" (On-disk data shapes) is
**false**: the code path (`rebuild_index` in `modelbench/results.py`) is called only from the
`index rebuild` CLI subcommand — confirmed both by reading `modelbench/cli.py` and by direct
observation (the file did not exist despite dozens of prior `run`/`compare`/`rank`/`attest`
invocations already reflected in `results/runs/`). Everything else tested — the `rank` table
shapes, the per-metric branching for `guard-judge-understanding`, the `--footprints` rendering, the
exit-2 usage-error paths, the filename sequencing, and the on-disk JSON shape claims — held exactly
as documented.

CPG: considered, not relevant — `model-bench` is a small (~15-module) stdlib-only Python component
with no CPG loaded for it in this environment (only `cpg_<component>` graphs for other components
were in scope), and the task is a black-box manual walkthrough, not a call-graph/impact-analysis
question a CPG would answer; the two code-level defects below were found by reading
`modelbench/cli.py`/`results.py`/`report.py` directly, which was faster than building or querying a
graph for a component this size.

## Results table

| ID | Result | Evidence |
|---|---|---|
| TP-001 | **Pass** | `./run.sh rank --pack embedder-graphrag-retrieval` → exit 0, one table, one row per **distinct model key** (5 rows for 12 stored files — `bm25` alone has 6 stored timestamps, correctly collapsed to 1 row), descriptive `95% CI` column, no verdict/p-value column, `wrote reports/embedder-graphrag-retrieval-rank-20260920-03.md`. |
| TP-002 | **Pass** | `./run.sh rank --pack guard-judge-understanding` → exit 0, **two** separate tables (`### falseAdvanceRate`, `### falseSuspendRate`), 16 rows each (19 stored files, 3 models with 2 timestamps each correctly collapsed to 1 row) — matches the "one table per metric" claim exactly. |
| TP-003 | **Fail (defect, see D-1)** | `./run.sh rank --pack embedder-graphrag-retrieval --reference bm25` → exit 2, crash: `model-bench: item 'gr-01' metric 'mrr' lives in 'measures' (a continuous measurement) and has no boolean outcome; call 'scored_value' instead`. Confirmed working on a boolean-outcome pack instead: `./run.sh rank --pack guard-judge-understanding --reference stablelm-zephyr-3b` → exit 0, adds a `#### Reference-anchored family` block with a `decision` column whose observed values are `distinguishable` / `not distinguishable` / `not tested (Holm stops here)` — Holm-corrected, no raw p-value printed, matching the manual's substance if not its exact wording ("distinguishably better/worse" is not literal; the tool never prints a "worse" word, only a signed `diff` column, e.g. `+87.5 pp`). |
| TP-004 | **Pass** | `./run.sh rank --pack guard-judge-understanding --reference 'does/not-exist'` → exit 2, message `reference model 'does/not-exist' has no stored, consistent run for this pack and session`; `reports/` file count unchanged (14 before and after). |
| TP-005 | **Pass** | `./run.sh rank --pack embedder-graphrag-retrieval --footprints <valid file>` → exit 0; matching keys (`bm25`, `text-embedding-qwen3-embedding-0.6b`) rendered their display string verbatim in the footprint column; non-matching keys still showed `—`. |
| TP-006 | **Pass** | `./run.sh rank --pack embedder-graphrag-retrieval --footprints <malformed JSON>` → exit 2, message `--footprints <path>: not valid JSON: Expecting property name enclosed in double quotes: line 1 column 2 (char 1)`. |
| TP-007 | **Pass** | `rank` report filenames use a `-rank-` segment (`embedder-graphrag-retrieval-rank-20260920-0N.md`) distinct from `compare`'s plain `<pack>-<date>-NN.md`; six same-day `rank` invocations against the same pack during this run produced six distinct sequence numbers (`-03` through `-08`, continuing an existing `-01`/`-02` sequence already present from an earlier session), none overwritten. |
| TP-008 | **Pass** | `./run.sh index rebuild` → exit 0, `results/index.csv` created with 95 data rows (96 lines − 1 header) exactly matching `ls results/runs/*.json \| wc -l` = 95. |
| TP-009 | **Pass, with a clarity note** | Real `host.json` present. It has exactly the four named fields (`lmStudioAppVersion`, `kvCacheSetting`, `hostRamGb`, `otherResidentWorkloads`) — but nested under an `attested` object, not top-level (`host.json` also carries `schemaVersion`, `apiBaseUrl`, `attestedAt`, `observedAtAttestation` alongside). The manual's prose ("Four fields that nothing in LM Studio exposes programmatically...") doesn't claim top-level placement, so this isn't a contradiction — just an omission that could be clarified. |
| TP-010 | **Pass** | Real `results/runs/embedder-graphrag-retrieval-bm25-2026-09-11T11:27:30Z.json` has top-level keys `aggregates`, `armKind`, `attestationTripWire`, `basis`, `designEffect`, `fingerprint`, `items`, `latency`, `role`, `runId`, `sessionId` — `fingerprint`/`aggregates`/`items` all present as claimed, and `sessionId` is genuinely top-level (`s3-step2-live-2026-09-11`), not nested inside `fingerprint` (fingerprint has no `sessionId` key). `aggregates` keys (`kind`, `mrr`, `precisionAt1`, `recallAtK`, `separationRaw`, `separationZ`) match the manual's embedder-pack example. `items[0]` keys (`counts`, `detail`, `itemId`, `latencyMs`, `measures`, `outcome`, `pairingKey`, `scoreable`, `timing`) confirm no raw model-reply text field, only outcome/score detail, as claimed. Also spot-checked `pack.json`'s claimed fields (`sampling.seed`/`pairingKey`/`analysisUnit`, `environment.requires`, `data`, `metrics.verdictMetrics`/`headlineMetric`) — all present as described; `packContentHash` is correctly *not* a `pack.json` field (it's computed and stamped into the run's `fingerprint` instead, consistent with the manual's own wording). |
| TP-011 | **Pass** | `compare --help` has no `--reference` flag; only `rank --help` does — confirms "only rank ... can name a single `--reference` model." |
| TP-012 | **Fail (defect, see D-1)** | The `0`/`2`/`3`/`4`/`5` exit-code set itself is accurate and consistent with the manual's pre-existing Overview/FAQ entries for 3/4/5. But grouping `rank --reference`'s crash under the same "usage error, exit 2" umbrella as the deliberate, clean usage errors (TP-004, TP-006) is misleading: those two are engineered `ValueError`/argument-parsing rejections with clear messages and no report written, while the `--reference` + continuous-metric case is an **unhandled internal exception** that happens to also exit 2 — same exit code, very different nature and user experience (a raw internal error string naming an internal method, `scored_value`, that the manual never mentions). |

## Defects

**D-1 — `rank --reference` crashes on any pack whose verdict metric is continuous (e.g. `embedder-graphrag-retrieval`'s `mrr`). Severity: High.**

- **Steps to reproduce:** from `model-bench/`, with `.venv` set up and stored results present for
  `embedder-graphrag-retrieval`: `./run.sh rank --pack embedder-graphrag-retrieval --reference bm25`.
- **Expected:** per the manual's Walkthrough 7a, this should render the ranked table plus a
  Holm-Bonferroni-corrected verdict column against `bm25` (the manual makes no distinction between
  pack/metric kinds for this flag, and its own basic-`rank` example uses this exact pack).
- **Actual:** exit code 2, single stderr line:
  `model-bench: item 'gr-01' metric 'mrr' lives in 'measures' (a continuous measurement) and has no
  boolean outcome; call 'scored_value' instead`. No report file is written.
- **Root cause (read, not fixed):** `rank_report`'s reference-family builder in
  `modelbench/report.py` (~line 1455) unconditionally calls `_paired_rows(reference_run, cand,
  metric, pack)` for every metric in `pack.metrics.verdictMetrics`, and `_paired_rows` always calls
  `item.scored_outcome(metric)` (`modelbench/report.py` line 290-291) — the boolean-only accessor.
  For a continuous metric (one that lives in `ItemResult.measures` rather than `.outcome`), that
  raises `MetricKindError`, which propagates uncaught to the CLI and is reported as a generic exit
  2. By contrast, `compare_report` (the `compare` command) evidently branches correctly by metric
  kind — confirmed live: `./run.sh compare --pack embedder-graphrag-retrieval --models
  bm25,text-embedding-qwen3-embedding-0.6b` renders `mrr` cleanly with no crash — so this is
  specifically a gap in `rank`'s reference-anchored-family code path, which appears to have only
  ever been exercised against boolean-outcome packs (`guard-judge-understanding` worked perfectly,
  TP-003).
- **Impact:** any reader following Walkthrough 7a who tries `--reference` against the pack the
  walkthrough itself demonstrates gets an unexplained crash quoting an internal method name. The
  manual documents no restriction, and none is enforced with a clean message — this is a code
  defect surfaced by testing the manual's claim, not merely a doc gap.
- **Recommendation:** either implement the continuous-metric branch in `rank`'s reference-family
  builder (mirroring whatever `compare_report` already does for `mrr`-style metrics), or, at
  minimum, catch the `MetricKindError` and fail with a clean, documented usage error — and add a
  sentence to the manual noting the restriction if the fix is deferred. Routing to `coder`/
  `tdd-engineer` for the actual change; not fixed here per this agent's role.

**D-2 — "rebuilt automatically" claim for `results/index.csv` is false; only `index rebuild` ever calls it. Severity: High.**

- **Manual claims, both wrong as written:**
  - Walkthrough 8: "every command above rebuilds it automatically as a side effect, so you don't
    normally touch this."
  - On-disk data shapes: "`results/index.csv` — a flat, derived summary ... rebuilt automatically
    by every command that reads or writes a result (and by `index rebuild` directly, if you ever
    need to force it)."
- **Evidence:**
  - Code: `grep -rn "rebuild_index(" modelbench/ tests/` shows `rebuild_index` (defined in
    `modelbench/results.py`) is called from exactly one production call site —
    `modelbench/cli.py`'s `_cmd_index` (the `index rebuild` subcommand) — and nowhere in
    `_cmd_run`, `_cmd_compare`, `_cmd_rank`, `_cmd_attest`, or `_cmd_models`. The other call sites
    are all in `tests/test_results.py`.
  - Live: at the start of this test session, `results/index.csv` **did not exist**, despite the
    environment already holding 95 stored run files spanning 2026-09-11 through 2026-09-20 (i.e.,
    many prior `run`/`attest`/`compare` invocations across several days), and despite this session
    itself running `rank` eight times and `compare` once *before* ever running `index rebuild`. The
    file only appeared after explicitly running `./run.sh index rebuild`.
- **Impact:** a reader who takes Walkthrough 8 at face value ("you don't normally touch this")
  will have a stale-or-missing `index.csv` indefinitely unless something else in their workflow
  happens to call `index rebuild` — the manual actively discourages the one action that's actually
  needed. This is the more consequential of the two defects for a new user, since it's presented as
  reassurance ("no action needed") rather than an edge case.
- **Recommendation:** either add the automatic-rebuild behavior the manual describes (call
  `rebuild_index` at the end of `run`/`attest`, and/or on-demand inside `compare`/`rank` before they
  read the index — though note `compare`/`rank` in this codebase appear to read `results/runs/`
  directly rather than through `index.csv`, so the "reads a result" half of the claim may be moot
  by design), or rewrite both passages to state plainly that `index.csv` is rebuilt **only** by
  `index rebuild`, and that this is the one command you do need to remember to run periodically.
  Given `rebuild_index`'s own docstring already says "Derived and fully regenerable, so it is never
  a second source of truth to keep honest" — i.e., the design intentionally treats it as
  on-demand-only — the doc fix (not a code fix) looks like the right call here, but that's an
  editorial judgment for `tico`/`architect`, not this report's to make.

**Minor note (not a defect):** `host.json`'s four attested fields are nested under an `attested`
key rather than top-level (TP-009). The manual's prose doesn't claim otherwise, so this is a
clarity suggestion, not a correction.

**Minor note (not a defect):** the reference-family "decision" column's actual vocabulary is
`distinguishable` / `not distinguishable` / `not tested (Holm stops here)`, not the manual's
"distinguishably better/worse" / "not distinguishable" (TP-003/TP-012). The signed `diff` column
(e.g. `+87.5 pp`) does carry the direction, so the manual's substance is fine, but "better/worse" is
not a string the tool ever prints — the manual paraphrases rather than quotes.

## Coverage & gaps

**Covered:** every new walkthrough/section/FAQ entry named in the task, driven against real stored
data with a healthy variety (a single-headline-metric pack, a two-co-equal-metric no-headline pack,
valid/invalid `--reference`, valid/invalid `--footprints`, same-day filename collision avoidance,
`index rebuild` row-count correctness, and direct inspection of `host.json`, a real run record, and
`pack.json`).

**Not covered, deliberately (per task scope and the manual's own family, already verified
elsewhere):** the four earlier walkthroughs (setup, attest, run, compare); an actual live LM Studio
`run`; the mermaid diagram's visual rendering; the numeric catalog-sweep claims in "Configuration &
integration" about `granite-embedding-278m-multilingual`'s relative footprint/latency (a factual
check against `reports/catalog-sweep-2026-09-19-consolidated.md`'s own numbers, which is
`analyst`'s half of this manual's review, not qa-engineer's — and a `docs/reviews/` document for
this same manual slug appeared mid-session, consistent with that review running in parallel).

**Residual risk:** D-1 means any user of `--reference` against an embedder-role (or any future
continuous-metric) pack hits the crash; D-2 means `index.csv` will silently drift stale for any
user who trusts the manual's "you don't normally touch this" framing. Both should block treating
this manual increment as final until resolved (either fix the code, or fix the prose — see each
defect's recommendation).

## Feedback & recommendations

- **Testability win:** the CLI's error messages are otherwise excellent — `--footprints` malformed
  JSON and `--reference` unknown-key both fail with precise, quotable, single-line messages that
  match the manual's promises exactly (TP-004, TP-006). D-1's message is jarring by contrast
  precisely because it's an internal accessor name (`scored_value`) leaking through, which is a
  useful signal that it's an unhandled path rather than a deliberate one.
- **Suggested follow-up:** a unit/integration test for `rank --reference` against a
  continuous-metric pack (mirroring the existing boolean-metric coverage implied by
  `tests/test_results.py`'s `rebuild_index` tests) would have caught D-1 well before this manual
  shipped it as a general-purpose flag.
- **Suggested follow-up:** whichever of `coder`/`architect`/`tico` owns the D-2 call should decide
  code-fix vs. doc-fix explicitly and record it — right now the manual and the code actively
  disagree about a specific, checkable claim.
- This report and the accompanying test plan
  (`docs/test-plans/small-model-benchmarking-manual-additions.md`) are new, uncommitted files (this
  agent leaves them uncommitted for the coordinating agent to review and commit). No committed/
  tracked file was modified during this test run (`git status --short .` before and after this
  session shows the same two pre-existing modified files, `AGENTS.md` and
  `docs/manuals/small-model-benchmarking.md`, both already modified by `tico` before this session
  started). `results/index.csv` was created fresh by TP-008 (it did not exist before, is untracked,
  and was not part of any committed state) and was left in place as a normal, regenerable artifact
  of running the tool as documented.
