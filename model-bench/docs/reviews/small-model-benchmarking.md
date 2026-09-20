# Small-Model Benchmarking — Manual Review (extension pass)

> **Status:** active · **Owner:** `analyst` · **Tracks:** — (M8)

## Scope & verdict

Reviewed the newly-added material in `model-bench/docs/manuals/small-model-benchmarking.md`
against real source in `model-bench/`: Walkthrough 7a (`rank`), Walkthrough 8 (`index rebuild`),
the "On-disk data shapes" section, "Configuration & integration", and the two new FAQ entries
(`compare` vs `rank`; the exit-code list). Earlier sections (Overview, Walkthroughs 1-7) were
already reviewed and shipped in a prior pass and were not re-reviewed, per the brief.

Checked against: `modelbench/cli.py`, `modelbench/report.py`, `modelbench/results.py`,
`modelbench/packs.py`, `modelbench/hostinfo.py`, `modelbench/fingerprint.py`,
`modelbench/runner.py`, and `reports/catalog-sweep-2026-09-19-consolidated.md`. Verified live by
running `models --tested`, `rank`, and `index rebuild` against the repo's own 95 stored run
records under `results/runs/` (see Finding 1's evidence).

**Verdict: needs changes** — one blocker (a repeated, confidently-stated, and live-falsified claim
about `index.csv`'s freshness) plus one minor accuracy nit. Everything else checked — the `rank`
command's behavior, the on-disk shapes, the granite-embedding integration numbers, and both FAQ
entries — matches the real code and the cited report exactly.

**CPG:** not applicable — this is a documentation-accuracy review with no code change under it;
`model-bench` has no loaded CPG to consult in any case, but the task itself (checking manual prose
against source) has no code-level component of its own to run a CPG query against.

## Findings

### BLOCKER — "every command rebuilds `results/index.csv` automatically" is false, stated twice

Manual text (Walkthrough 8): *"`results/index.csv` is a derived summary of everything under
`results/runs/` — every command above rebuilds it automatically as a side effect, so you don't
normally touch this."* Repeated in "On-disk data shapes": *"rebuilt automatically by every command
that reads or writes a result (and by `index rebuild` directly, if you ever need to force it)."*

Traced every call site of `rebuild_index` (`modelbench/results.py:1176`) across the whole package:
it is called from exactly one place, `_cmd_index` (`modelbench/cli.py:361`). `store()`
(`results.py:925`), `load_history()` (`results.py:993`), `_cmd_run`, `_cmd_compare`, `_cmd_rank`,
and `_cmd_models` never call it — `compare`/`rank` read stored runs directly via `load_history`,
never through `index.csv` at all.

Confirmed live, not just by grep: the repo's own `results/runs/` holds 95 stored run files (some
from today), yet `results/index.csv` did not exist before this review. Running
`python3 -m modelbench.cli models --tested --pack embedder-graphrag-retrieval` and
`python3 -m modelbench.cli rank --pack embedder-graphrag-retrieval` (both succeeded, the latter
writing a real report to `reports/`) left `results/` still holding **only** a `runs/`
subdirectory — no `index.csv` appeared until `index rebuild` was run directly. The Mermaid diagram
in the same section actually gets this right (its only edge into `Idx` is labelled
`"./run.sh index rebuild"`) — it is the surrounding prose, in two places, that contradicts both the
diagram and the real behavior.

This inverts the command's actual value: a user told "you don't normally touch this" will never
run `index rebuild`, and will find no `index.csv` at all rather than a stale one — the opposite of
"rebuilt automatically." Fix: drop the "every command... rebuilds it automatically" claim in both
places; state plainly that `index rebuild` is the **only** thing that ever writes `index.csv` today
(nothing else reads it either — `compare`/`rank`/`models` all go straight to `results/runs/`), so it
is an optional, on-demand summary a user builds when they want one, not a side effect of normal use.

### MINOR — `packContentHash` does not cover "every byte of the pack directory"

Manual text (On-disk data shapes, `pack.json` bullet): *"`packContentHash` covers every byte of the
pack directory, so any change to a golden item, a prompt, or a tool schema produces a new,
distinguishable version."* `content_hash()` (`modelbench/packs.py:473-487`) explicitly excludes
`PROVENANCE.md`: *"SHA-256 over the sorted relative paths and bytes of every pack file **but**
`PROVENANCE.md`."* The three illustrating examples (golden item, prompt, tool schema) are all
correctly covered — the overclaim is only in the absolute "every byte" phrasing. Suggest: "covers
every byte of the pack directory except `PROVENANCE.md` (so noting a hash's provenance doesn't
itself bump the hash)."

## What's solid

- **`rank` (Walkthrough 7a)**: every claim checked out against `modelbench/report.py` and
  `modelbench/cli.py` — one row per model with a stored, consistent result
  (`_render_one_rank_table`/`rank_report`); `--reference` adds a Holm-Bonferroni-corrected column
  (`_render_reference_family`, `stats.holm_steps`) with no p-value anywhere when omitted; naming an
  absent reference model raises `ValueError` caught as `EXIT_USAGE` (`cli.py:344-350`);
  `--footprints`'s malformed *file* is `RankUsageError` → exit 2 (`cli.py:255-266`) while a
  malformed *value* degrades to `str(v)` (`cli.py:267`); `guard-judge-understanding`'s two
  co-equal verdict metrics (`verdictMetrics=["falseAdvanceRate","falseSuspendRate"]`,
  `headlineMetric=None`, confirmed from its live `pack.json`) genuinely produce one table per
  metric (`rank_report`'s `members = family` when `headlineMetric is None`); the `-rank-` filename
  infix is real and distinct from `compare`'s (`_rank_report_path` vs. `_report_path`).
- **On-disk data shapes**: the `pack.json` field list, `host.json`'s four fields
  (`hostinfo.ATTESTED_FIELD_NAMES`), and the `RunResult` shape (top-level `sessionId`, not inside
  `fingerprint` — confirmed on the dataclass and live in a stored record) all match source exactly.
  The Mermaid diagram's relationships are all correct, including the one the prose elsewhere
  contradicts (see Finding 1).
- **Configuration & integration**: zero `os.environ`/`getenv` reads anywhere in `modelbench/`,
  confirmed by a full-package grep. The granite-embedding-278m-multilingual numbers (~2-3 pp of MRR,
  ~1/8 footprint, ~4.6x lower p95 latency, "well inside the pack's own... resolving power") are an
  accurate restatement of `reports/catalog-sweep-2026-09-19-consolidated.md`'s embedder section
  (0.6317 vs. 0.6610 = 2.93 pp; ~0.28 GB vs. ~2.2-2.5 GB ≈ 1/8-1/9; 12 ms vs. 55 ms ≈ 4.58x) —
  not overstated.
- **FAQ entries**: `compare` genuinely has no `--reference` flag (only `rank` and `run` do;
  `cli.py:96-105` vs. `:110-122`), so "the only one of the two that can name a `--reference`" is
  correct. The exit-code list (0/2/3/4/5, and their triggers) matches `EXIT_*` constants and every
  `RunRefused(exitCode=...)` call site in `runner.py` exactly, including the tool-caller
  write-before-exit-4 ordering.

## Open questions

None — the two findings above are actionable without further input from the caller.
