# model-bench

A standalone, human-started harness that measures **one local model at a time** against **one named
task pack**, stores the result with a full environment fingerprint, and compares it against
previously stored results **for the same role** — with confidence intervals and a visible flag
whenever a comparison is not apples-to-apples.

The point is not a score. The point is that a result taken today still lines up against a model
tested months ago, on this hardware and this data, instead of against a published leaderboard number
for a task nobody here runs.

Requirements: `docs/requirements/small-model-benchmarking.md` (repo root).
Design: `docs/plans/small-model-benchmarking.md` (repo root), plus the statistical method note
`docs/plans/small-model-benchmarking-ml.md`.

## Three things this tool deliberately is not

- **No CI hook, no scheduler, no timer.** `model-bench` only ever runs because a person typed a
  command. Nothing here is wired into any other component's test suite.
- **No pass/fail gate.** There is no threshold, no "bad score" and no non-zero exit for a result you
  did not like. The only non-zero exits are operational: bad arguments, unreachable LM Studio, an
  invalid pack, a missing fingerprint field.
- **No leaderboard and no cross-role aggregate number.** A "tool-caller" score and an "embedder"
  score do not add up to anything, so there is no command that adds them up. Comparison is always
  within one pack, and that is enforced by the shape of the code, not by a convention anyone has to
  remember.

It is also **standalone** (FR-23): no runtime code path reads anything outside `model-bench/` —
not `falkor-chat`'s configuration, not its model gateway, not its golden sets. Golden *data*
originating elsewhere is copied in, versioned here, and carries its provenance; that is data, not a
dependency.

## Status

**All eight stages (S0–S8) are closed — the feature is delivered.** Five scorer modules cover all
five FR-21 roles:
`scoring/retrieval.py` for `embedder` (recall@10 = 37/38 = 0.974,
`docs/test-reports/embedder-self-check-report.md`); `scoring/classification.py` for
`guard-judge-understanding` (85 items, two verdict metrics, four diagnostics,
`reports/guard-judge-understanding-20260911-02.md`); `scoring/extraction.py` for
`nlq-structured-query` (40 items with answerability stamp 34/6, `layer1ExactMatchRate` 34/34 this
run, `reports/nlq-structured-query-20260911-01.md`); `scoring/toolcalls.py`, the first
`ConversationScorer`, for `tool-caller` (per-turn funnel/hazard/argument-decomposition scoring, plus
the prose-vs-native detector's precision/recall) against the full `packs/tool-caller-shop-assistant/`
storefront pack — 12 hand-authored conversation scripts and 20 labelled calibration replies, every
one human-verified by the stakeholder per FR-19; and `scoring/grounding.py` for
`chat-responder-grounded-answers` (deterministic layer only per FR-21a — latency, format,
grounding-by-containment; judged reply quality stays deferred) against 30 hand-authored items,
likewise FR-19 human-verified by the stakeholder. The CLI ships all six commands (`compare`,
`index rebuild`, `models --tested`, `attest`, `validate`, `run`) plus `scripts/refresh_golden.py`
flags (`--check-tables-shape`, `--stamp-answerability`, `--source-git-sha`) wired into `main()`.
What exists now is both halves: the part that decides whether a number may be printed (the
environment fingerprint and its validation, the run store and its quarantine-on-read, the
statistics module, the markdown comparison) and the part that produces one (the LM Studio adapter,
the pack loader, the runner's capture-order orchestration, all six CLI commands, and five
role-specific scorers, including the first `ConversationScorer`).

`tool-caller`'s first live runs are in: a negative control (two independent runs of the same model)
passed cleanly, and the known-answer validation (`qwen/qwen3-4b-2507` vs.
`mistralai/ministral-3-3b`) ran to completion but did not reach statistical significance at this
pack's sample size, on its shipped configuration or either fallback rung tried — so the pack ships
honestly flagged "known-answer validation: not reproduced" rather than a result that was massaged
into significance. Full detail: `docs/test-reports/small-model-benchmarking-s6-report.md`.

`chat-responder`'s first live run found a real defect in its own headline metric before shipping
it: the abstention/containment checklist under-recognized this model's natural phrasing, scoring
15/30 (0.500) `groundingRate` when the true grounded-reply rate was closer to 0.80. Fixed over a
five-round chain, each independently gated, and confirmed on a fresh live re-run: `groundingRate`
now measures **23/30 (0.767)** for `qwen/qwen3-4b-2507` on this pack, with all 8 of the run's
originally-misclassified abstentions now scoring correctly. A separate, narrower containment/
morphology gap (plain literal matching doesn't tolerate ordinary paraphrase, e.g. "4 retries" vs.
"4 retry attempts") was scoped out of that fix and stays open, `docs/BACKLOG.md`. Full detail,
including the original defect narrative kept as historical record:
`docs/test-reports/small-model-benchmarking-s7-report.md`.

Full stage-by-stage history — every unit, defect, and gate — is `docs/HISTORY.md`; open follow-up
work is `docs/BACKLOG.md`.

The default suite (network-free) runs offline; `pytest -m live` opts into the tests that need a
reachable LM Studio.

## Quick start

Run everything with `model-bench/` as the working directory — the repo has no root pytest
configuration, so from the repo root pytest ignores this component's `testpaths` and walks into
other components' suites.

```bash
./setup.sh                        # create .venv, install the package + dev extra
.venv/bin/python -m pytest -q     # the suite (network-free)
.venv/bin/ruff check .            # lint
./run.sh --help                   # the CLI
```

## What the CLI does today

```bash
./run.sh compare --pack <pack-id> [--models a,b] [--session <id>] [--negative-control] [--out <path>]
./run.sh index rebuild            # regenerate results/index.csv from results/runs/
./run.sh models --tested          # models with stored results (never a deterministic arm)
./run.sh attest [--api-base-url <url>] [--set key=value ...]   # write host.json
./run.sh validate --pack <path> [--strict]   # structural pack check, no LM Studio needed
./run.sh run --pack <id> --model <key> [--session <id>] [--reference <key>] [--warmup <n>]
    [--first-call-timeout <s>] [--request-timeout <s>]   # one model x one pack, against live LM Studio
```

`compare` reads `results/runs/`, renders the markdown comparison to `reports/` **and** stdout, and
never overwrites an earlier same-day comparison — the filename carries a two-digit sequence. It
exits `0` whatever the scores, including when every stored record turns out to be invalid: that is
a report, not an operational failure, and the excluded records are named in it with their reasons.

`--negative-control` puts **two copies of one stored record** in the two arms, so `b = c = 0` is
arithmetic rather than a measurement. It proves the mode is wired and nothing more, and the report
it writes says so in a banner at the top — the real negative control is two *independent* runs of
the same model, and that is an acceptance step.

`attest` writes the operator-attested half of the fingerprint (`host.json`). `validate` checks a
pack's structural integrity with no LM Studio connection at all. `run` drives one model through one
pack against a live LM Studio, following the fingerprint's capture order, and stores the result on
success — every shipped pack has a concrete scorer, so `run` no longer refuses on that ground; it
still refuses per §3.6a's closed exit-code set below (bad pack, unreachable LM Studio, stale
fingerprint, a dispatch-censored conversation).

**Exit codes are closed and operational-only** (§3.6a): `0` whenever the tool ran and reported —
*whatever the scores*, including a `compare` that finds every stored record invalid (that prints
`INVALID RESULTS EXCLUDED` and still exits `0`, because it is a report, not a failure). `2` bad
arguments/usage. `3` LM Studio unreachable, reachable without its native `/api/v0` catalog, not
answering the warm-up call within `--first-call-timeout`, or gone when re-probed after a scored
call timed out. `4` an invalid pack — a `validate` failure, a load error, an unmet
`environment.requires`, a `callSurface` the model's catalog type contradicts, or (after `run`'s
artifacts are already written) a `tool-caller` conversation censored by a tool-dispatch failure — a
data-quality finding discovered only once a partial record exists, not an aborted run. `5`
fingerprint incomplete, or `host.json` stale/absent. Nothing else — there is no score-driven exit
code, on purpose (see "Three things this tool deliberately is not," above).

## What a pack is, and how to add one

A **pack** is a directory under `packs/<pack-id>/`, declared by `pack.json`: `packId`,
`packVersion`, `role` (one of the five FR-21 roles — `embedder`, `guard-judge`, `nlq-generator`,
`tool-caller`, `chat-responder`), `scorer` (the module name `run` resolves against
`modelbench/scoring/`), `environment.requires`, a `prompt` block for any chat-surface role, `data`
(the golden files), `sampling` (item or script/replicate counts, the seed, the `pairingKey` and
`analysisUnit` that make a comparison paired), and `metrics` (`verdictMetrics`, `headlineMetric`).
Golden items are copied in as JSONL, each carrying a `provenance` object (`origin`, `originPath`,
`originGitSha`, `copiedAt`, `draftedBy`, `verifiedBy`, `corpusVersion` — FR-19), and every pack
ships its own `PROVENANCE.md` naming the origin file, the origin commit, the copy date, and what
changed on copy. `packContentHash` covers every byte of the pack directory, so any change — a
golden item, a prompt, a tool schema — is a new, distinguishable version.

To add one: create `packs/<new-id>/`, write `pack.json` declaring its role, its scorer name, and
its sampling/metrics contract; copy in golden data with real, FR-19-verified provenance (never
fabricated); implement (or reuse) a scorer module under `modelbench/scoring/` satisfying the
role's scorer protocol — `ItemScorer` for a single-call role, `ConversationScorer` for a
multi-turn one (only `tool-caller` today); then run `./run.sh validate --pack packs/<new-id>
--strict` — no LM Studio needed — before ever driving it live with `./run.sh run`.

**Scope note — the `embedder` pack scores exact cosine, not the production ANN pipeline.**
`embedder-graphrag-retrieval` embeds the pack's fixed corpus and queries, then ranks with
brute-force exact cosine similarity in-process — no ANN index, no FalkorDB. That is deliberate:
the object of measurement is the *model*, and an approximate index would inject pipeline noise
into a model comparison; exact search is also what makes the score-separation metric possible at
all, since it exposes the irrelevant-document scores an ANN index would prune before they could be
measured. The consequence is a real scope boundary: **a model that wins here has not thereby been
shown to win *through* falkor-chat's hybrid ANN retrieval pipeline** — this pack measures the
embedding model in isolation, not the retrieval path a user actually hits.

## Stored results and schema versioning

`benchSchemaVersion` is a small integer in `modelbench/results.py`, starting at `1`, kept
deliberately separate from `benchVersion` (the installed `modelbench` release). It increments only
when the required-field set or the on-disk record shape changes in a way a *reader* must branch
on — never automatically, and never as a side effect of adding a field. **A bump is a deliberate
act**, not a side effect: a new `REQUIRED_BY_SCHEMA` entry, a `docs/HISTORY.md` line, and an
explicit decision about whether existing stored records need a migration. An older-schema record
is never excluded from a comparison on that basis alone — it is validated against the contract it
was written under, and `compare` prints a `SCHEMA VERSIONS IN THIS COMPARISON` banner whenever a
comparison spans more than one schema version, rather than silently mixing them or dropping the
older ones. No schema bump has been needed through S0–S8; `benchSchemaVersion` is still `1`.

Python 3.12, matching every other component. **Zero runtime dependencies** — stdlib only, on
purpose: a benchmarking tool whose own dependency tree can rot is a tool whose old results stop
being reproducible. `pytest` and `ruff` are the only dev extras.

`scripts/s6_walkthrough.py` (`.venv/bin/python scripts/s6_walkthrough.py [--script <scriptId>]`) is
a read-only review aid for the S6 stakeholder pass (`docs/plans/small-model-benchmarking-s6-spec.md`
§2.6 Step 6): drives every `conversations.jsonl` script's `toolRequired` turns against the real,
live `tool-caller-shop-assistant` environment and prints each turn's real dispatch result beside its
scripted `expect` block. Never writes `conversations.jsonl`, `PROVENANCE.md`, or
`provenance.verifiedBy` — filling `verifiedBy` stays the stakeholder's own action.
