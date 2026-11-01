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
- **No pass/fail gate.** There is no threshold of its own; a decision-point pack's report states the
  requirement's bar and never acts on it. There is no "bad score" and no non-zero exit for a result
  you did not like. The only non-zero exits are operational: bad arguments, unreachable LM Studio,
  an invalid pack, a missing fingerprint field.
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
./run.sh rank --pack <pack-id> [--session <id>] [--reference <key>] [--footprints <path.json>]
    [--out <path>]   # ranked table across every in-scope model with a stored result, not two arms
./run.sh index rebuild            # regenerate results/index.csv from results/runs/
./run.sh models --tested          # arms with stored results: model keys, candidate ids (never a
                                  # deterministic arm)
./run.sh attest [--api-base-url <url>] [--set key=value ...]   # write host.json
./run.sh validate --pack <path> [--strict]   # structural pack check, no LM Studio needed
./run.sh run --pack <id> (--model <key> | --candidate <id>) [--session <id>] [--reference <key>]
    [--warmup <n>] [--first-call-timeout <s>] [--request-timeout <s>]
    # one model (or one decision-model candidate) x one pack, against live LM Studio
```

`compare` reads `results/runs/`, renders the markdown comparison to `reports/` **and** stdout, and
never overwrites an earlier same-day comparison — the filename carries a two-digit sequence. It
exits `0` whatever the scores, including when every stored record turns out to be invalid: that is
a report, not an operational failure, and the excluded records are named in it with their reasons.

`--negative-control` puts **two copies of one stored record** in the two arms, so `b = c = 0` is
arithmetic rather than a measurement. It proves the mode is wired and nothing more, and the report
it writes says so in a banner at the top — the real negative control is two *independent* runs of
the same model, and that is an acceptance step.

`rank` is `compare`'s N-arm sibling: one ranked table per pack (two, for a pack with no single
headline metric — one per `verdictMetrics` member) covering every in-scope model with a stored,
consistent result, each row carrying its own descriptive confidence interval, latency p95, and
opaque footprint string — never a pairwise matrix. Like `compare`, it writes to `reports/` **and**
stdout and never overwrites an earlier same-day report (`-rank-` in the filename keeps its sequence
separate from `compare`'s own). `--reference <key>` additionally renders an optional, pre-registered,
Holm-Bonferroni family of verdicts against that one stored model; omitted, the report states only
descriptive intervals, no p-value anywhere. Naming a model key with no stored run for this pack
(and session, if given) is a usage error, exit `2`, and nothing is written. `--footprints
<path.json>` passes through a flat `{"<modelKey>": "<display string>"}` map verbatim — never
parsed as a number, so a missing or malformed *value* only ever degrades to a wrong-looking cell
(`—`, or a coerced `str()`); an
unparseable or non-object *file* is a usage error, exit `2`.

`attest` writes the operator-attested half of the fingerprint (`host.json`). `validate` checks a
pack's structural integrity with no LM Studio connection at all. `run` drives one model through one
pack against a live LM Studio, following the fingerprint's capture order, and stores the result on
success — every shipped pack has a concrete scorer, so `run` no longer refuses on that ground; it
still refuses per §3.6a's closed exit-code set below (bad pack, unreachable LM Studio, stale
fingerprint, a dispatch-censored conversation).

`run --candidate <id>` (exclusive with `--model`) runs a typed decision-model candidate,
`candidates/<id>.json`, on a decision-point pack's `decisions` surface
(`docs/plans/typed-decision-models.md` §3.3): the pack must carry a `decisionPoint` block and the
`lmstudio-decisions` token in `environment.requires`, the candidate's question must match the
pack's answer space, and its decision model's catalog type must be `decision` — otherwise exit `4`
before any call. The candidate's fit manifest, `candidates/<fitManifest.file>` (one sorted
`input_hash` per line), must match the candidate's declared `sha256` and entry count, and must
hold no input of any pack item — the leakage refusal (AC-3) — also exit `4` before any call.

A candidate **with** a `handoffModel` runs on the `handoff` surface: the decision model answers
each item first, and an item its policy will not decide, or whose input is over the decision
model's limit, goes to the hand-off chat model. The pack must also declare `lmstudio-chat`, the
hand-off model's catalog type must be `llm` or `vlm`, and on a speed-up point (the guard) it must
be the pack's `decisionPoint.incumbent` — otherwise exit `4` before any call. Both models are
warmed up, the decision model first, and both must still be resident afterwards, or the run is
exit `3` before any item. The record is a `model:handoff` fingerprint: the decision model under
the usual names, the hand-off model under `handoff*` names.

On a decision-point pack the runner loads every model the arm uses that is not resident at start
(`POST /api/v1/models/load`) before the warm-ups, so nothing it scores is JIT-loaded. An explicit
load has no TTL, so a model the runner loads **stays resident on the GPU after the run** until you
unload it. A failed load, a candidate's model gone after the warm-ups, or the decision model's
loaded instance config unreadable, is exit `3`; the record carries that config verbatim
(`decisionInstanceConfig`). A failed decision call (a timeout, or any failure that is not the
over-limit one) keeps its item in the run as a failure, counted in the exploratory
`decisionCallFailures` metric, and is never handed off. An over-limit input (one question over
the decision model's physical batch, 512 tokens for laya, which LM Studio refuses with an HTTP 500
rather than truncating) is counted in `overLimit`; with a hand-off model it is handed off,
without one it is kept as a failure. A failed hand-off chat call withholds the item on the guard
(as a failed chat call always has). On the
new-ability roles any failed call is kept and scored as the fail-closed answer (no reply; no
label, a disagreement), a failed hand-off counted in `handoffCallFailures`. The
`handoff` surface also records `handoffShare` (items handed off, over every item not withheld by a
failed hand-off call — an item withheld for a reload is in n) and each verdict metric split into
`…DecidedByModel` and `…HandedOff`, all exploratory. After the last
item a re-check decides the record's `basis`:
`by-construction` if the run stayed on one load of the decision model, `assumed` otherwise. The
record is keyed by its `armKey` — the candidate id — in `compare --models`, `rank`,
`models --tested` and `index.csv`'s `candidateId` column.

On a decision-point pack (any arm), chat calls run at a pinned 180 s: `--request-timeout` with
any other value is refused, exit `4`. Every run records the request timeout it used.

`compare` on a decision-point pack needs `--session` and `--models`: at most one chat arm (on the
guard, the pack's `decisionPoint.incumbent`, with exactly one run in the session) plus candidate
ids. Every run of each candidate in the session is a **replicate**, oldest first; none is picked
last-wins. Exit `2`, and nothing written, for: a missing `--session` or `--models`, a key with no
run in the session or named twice, a second chat arm or one that is not the incumbent, two
incumbent runs, a candidate with two content hashes in the session, arms run at different chat
timeouts, or `--negative-control`. `rank` refuses a decision-point pack outright.

Its report opens with the labels caveat, naming the labeler and the input generator — what FR-5
makes these labels mean is `docs/requirements/typed-decision-models.md`'s FR-5, not restated here —
then one **bar section** (`docs/plans/typed-decision-models.md` §3.5–§3.6,
`typed-decision-models2.md` §3.6). Per candidate: its fit-manifest leakage line, how many candidates
already ran on this evaluation set, its fit outcome. Per replicate: `handoffShare`,
`decisionCallFailures` and `overLimit`; the scored eval n beside FR-11's 200, with every drop
counted; a design-effect line, `design effect: genRunId ICC <±x.xxx> pooled over <k> cells` (or
`not estimable (<reason>) — DE 1.00`) — the items' clustering by generation run (`genRunId`),
one ICC pooled over the family, which widens limits and never moves a decision; the four FR-1
parity cells, each with its detectable difference, method minimum, `overLimit` and its own
`DE <x.xx>`; the FR-2 speed-up ratio; and the side check, which does not affect pass/fail. On a
new-ability pack (FR-10, one arm: no incumbent; a chat arm is a reference whose results no bar
reads, though its timeout and pack content still apply) a replicate adds `handoffCallFailures`, then
its own design-effect line (`classes`), then each true answer's agreement `k/n` with its
per-class Wilson interval (at that class's `DE`),
method minimum and `overLimit` — every class must reach 0.90, and one with no scored item is
`not assessable` — and the mean latency against 350 ms. Then the replicates' disagreement and
one overall state per bar: `met` needs every replicate met and two fresh loads of the decision
model. A replicate whose model was already resident is not a load, and one whose load changed
mid-run prints `not assessable`. Unless
the chat arm (incumbent or reference), every replicate and the pack on disk share one content hash,
every bar prints `not assessable (arms ran on different pack content: …)`: like an off-pin timeout,
that is a property of stored records, so the report states it rather than refusing. The candidates
line counts valid records only, and says how many invalid ones it left out. The generic tables below
cover the chat arm alone. Exit codes are unchanged.

**Exit codes are closed and operational-only** (§3.6a): `0` whenever the tool ran and reported —
*whatever the scores*, including a `compare` that finds every stored record invalid (that prints
`INVALID RESULTS EXCLUDED` and still exits `0`, because it is a report, not a failure). `2` bad
arguments/usage. `3` LM Studio unreachable, reachable without its native `/api/v0` catalog, not
answering the warm-up call within `--first-call-timeout`, failing the embedder's corpus embed
before any item, or gone when re-probed after a scored call timed out; on a decision-point pack
also a failed explicit load or instance-config read.
`4` an invalid pack — a `validate` failure, a load error, an unmet `environment.requires`, a
`callSurface` the model's catalog type contradicts, or (after `run`'s artifacts are already
written) a `tool-caller` conversation censored by a tool-dispatch failure — a data-quality finding
discovered only once a partial record exists, not an aborted run. `5` fingerprint incomplete,
or `host.json` stale/absent. Nothing else — there is no score-driven exit code, on purpose (see
"Three things this tool deliberately is not," above).

## What a pack is, and how to add one

A **pack** is a directory under `packs/<pack-id>/`, declared by `pack.json`: `packId`,
`packVersion`, `role` (one of `roles.ROLES`: the five FR-21 roles — `embedder`, `guard-judge`,
`nlq-generator`, `tool-caller`, `chat-responder` — plus typed-decision-models' `join-in-judge` and
`message-type-classifier`), `scorer` (the module name `run` resolves against
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

**A decision-point pack** is one whose `pack.json` has a `decisionPoint` key
(`docs/plans/typed-decision-models.md` §3.5); no shipped pack has one, and a pack without it
validates exactly as before. With it, `validate` (and so `run`) also requires: a `labels` block with
`source: "llm-unreviewed"`, `inputs: "synthetic"`, and `labeler` and `generator` blocks each with a
`provider` and a `model`, the labeler's not `decisionPoint.incumbent`; a role in the bar table, with
`incumbent` stated (a model id on the guard's speed-up point, `null` on a new-ability one); a
`stratum` on every item, `eval` or `side-check`, never defaulted; at least 200 `eval` items, each
with `origin: "synthetic"` and a non-empty string `genRunId`; and, on `guard-judge` only and there
always, a `side-check` stratum (the human-curated items, exempt from `origin` and `genRunId`). A
deleted key is refused, never defaulted. On a pack scored by `classification` (the guard's), every
`eval` item needs a bool `expected` and a `path` of `understanding` or `turns`, and every
`side-check` item a `path` and a known `tier`, so a run never stops on a malformed item. A guard
decision-point pack's `verdictMetrics` are exactly its four per-kind rates
(`falseAdvanceRateByUnderstanding` … `falseSuspendRateByTurns`), each once, in any order. Each
`eval` guard item scores its per-kind rate (from its label and `path`) plus the pooled
`falseAdvanceRate`/`falseSuspendRate`, now exploratory; a `side-check` item scores only its
exploratory `sideCheck*` metric. A guard hand-off reply is read the way falkor-chat reads its judge
(`guards._coerce_verdict`): only a JSON `true` whose rationale does not contradict it (an un-negated
"not enough", "still missing", …) advances, so `"decision": "true"` suspends. A pack without
`decisionPoint` keeps its looser parse. A new-ability pack (`join-in-judge`,
`message-type-classifier`) is scored by `agreement`, and a guard pack by `classification` — the role
fixes the scorer (`decisions.SCORER_BY_ROLE`), so a pack declaring another is refused. Its
`verdictMetrics` are the per-class agreement metrics (`agreeWhenYes`/`agreeWhenNo`, or
`agree[<label>]` per label), every `eval` item's `expected` must lie in the answer space, and its
hand-off prompt is `packs/<role>-synthetic/prompts/<point>-handoff.md` (the guard's,
`packs/guard-judge-synthetic/prompts/judge.md`, is a byte-copy of `guard-judge-understanding`'s
judge prompt). Golden vectors in `tests/fixtures/decision-contract/` pin this parse and the judge
prompt render, and falkor-chat's own suite checks its functions against byte-copies of them.

**Importing a decision point's data and candidates** (`scripts/import_decision_pack.py`, one-way,
human-invoked, never reached from a run; `docs/plans/typed-decision-models.md` §3.4):

```bash
.venv/bin/python scripts/import_decision_pack.py --role <role> \
    --dataset ../decision-models/data/<role>/<datasetVersion> --pack-version <new version>
.venv/bin/python scripts/import_decision_pack.py --candidate <dir>/<candidateId>.json
```

The first copies the dataset's **eval split only** (the `itemId`s in `splits.json`'s `"eval"`
list) into the existing `packs/<role>-synthetic/`, rewriting `pack.json`, `items.jsonl` and
`PROVENANCE.md` and nothing else: `prompts/` is committed pack data the importer never writes, and
a missing pack directory or prompt is refused. Each eval item keeps its input fields, `expected`,
`origin`, `genRunId`, `genRound`, `seedId`, `labelVotes` (and `threadPosition` on message type);
`labelVotes` must be counts summing to 3 keyed by exactly the answer space as strings (`"true"`/
`"false"` lowercase on a predicate point, each label on message type), or the import is refused
(`docs/plans/typed-decision-models6.md` §3.7, whose contract vector
`tests/fixtures/decision-contract/dataset-shape.jsonl` pins all three dataset files' shapes);
`labels.labeler`/`labels.generator` come from `generation.json`; the guard pack also gets the
85 items of `guard-judge-understanding` as its `side-check` stratum. The pack must validate before
anything is written, its source files must be committed (their git SHAs go into `PROVENANCE.md`),
and an unchanged `--pack-version` is refused. The second copies a candidate and the fit manifest
its `fitManifest.file` names (never the fit report) into `candidates/`, after both load; a file
already there with other bytes is refused. Exit `0` imported, `1` refused, `2` bad arguments.

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
