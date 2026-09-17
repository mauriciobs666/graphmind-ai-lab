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

**Stages S4, S5 and S6 are closed.** Four scorer modules exist: `scoring/retrieval.py` for
`embedder` (recall@10 = 37/38 = 0.974, `docs/test-reports/embedder-self-check-report.md`);
`scoring/classification.py` for `guard-judge-understanding` (85 items, two verdict metrics, four
diagnostics, `reports/guard-judge-understanding-20260911-02.md`); `scoring/extraction.py` for
`nlq-structured-query` (40 items with answerability stamp 34/6, `layer1ExactMatchRate` 34/34 this
run, `reports/nlq-structured-query-20260911-01.md`); and `scoring/toolcalls.py`, the first
`ConversationScorer`, for `tool-caller` (per-turn funnel/hazard/argument-decomposition scoring, plus
the prose-vs-native detector's precision/recall) against the full `packs/tool-caller-shop-assistant/`
storefront pack — 12 hand-authored conversation scripts and 20 labelled calibration replies, every
one human-verified by the stakeholder per FR-19. The CLI ships all six commands (`compare`,
`index rebuild`, `models --tested`, `attest`, `validate`, `run`) plus `scripts/refresh_golden.py`
flags (`--check-tables-shape`, `--stamp-answerability`, `--source-git-sha`) wired into `main()`.
What exists now is both halves: the part that decides whether a number may be printed (the
environment fingerprint and its validation, the run store and its quarantine-on-read, the
statistics module, the markdown comparison) and the part that produces one (the LM Studio adapter,
the pack loader, the runner's capture-order orchestration, all six CLI commands, and four
role-specific scorers, including the first `ConversationScorer`).

`tool-caller`'s first live runs are in: a negative control (two independent runs of the same model)
passed cleanly, and the known-answer validation (`qwen/qwen3-4b-2507` vs.
`mistralai/ministral-3-3b`) ran to completion but did not reach statistical significance at this
pack's sample size, on its shipped configuration or either fallback rung tried — so the pack ships
honestly flagged "known-answer validation: not reproduced" rather than a result that was massaged
into significance. Full detail: `docs/test-reports/small-model-benchmarking-s6-report.md`. One role
(`chat-responder`, S7) still has no scorer — `docs/plans/small-model-benchmarking.md` §4. The
default suite (network-free) still runs offline; `pytest -m live` opts into the tests that need a
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
success — but it refuses every pack today, since no concrete per-role scorer exists yet (stage
S3's job).

Python 3.12, matching every other component. **Zero runtime dependencies** — stdlib only, on
purpose: a benchmarking tool whose own dependency tree can rot is a tool whose old results stop
being reproducible. `pytest` and `ruff` are the only dev extras.

`scripts/s6_walkthrough.py` (`.venv/bin/python scripts/s6_walkthrough.py [--script <scriptId>]`) is
a read-only review aid for the S6 stakeholder pass (`docs/plans/small-model-benchmarking-s6-spec.md`
§2.6 Step 6): drives every `conversations.jsonl` script's `toolRequired` turns against the real,
live `tool-caller-shop-assistant` environment and prints each turn's real dispatch result beside its
scripted `expect` block. Never writes `conversations.jsonl`, `PROVENANCE.md`, or
`provenance.verifiedBy` — filling `verifiedBy` stays the stakeholder's own action.
