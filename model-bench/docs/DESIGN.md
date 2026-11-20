# model-bench — Design

> **Status:** active · **Owner:** `coder` · **Tracks:** —

> **How to read this document.** It is part of model-bench's live spec and states the harness **as
> it is now**: the command surface, the run lifecycle, the LM Studio contract, scoring, the typed
> decision-point surfaces, the verdict statistics and the reports. Data formats (pack manifest,
> result record, fingerprint, candidate, `host.json`) are in `model-bench/docs/FORMATS.md`, read
> by lookup. Change history is in `model-bench/docs/HISTORY.md` and open work in
> `model-bench/docs/BACKLOG.md`. The statistical formulas and verdict strings are those of
> `docs/plans/small-model-benchmarking-ml.md` and `docs/plans/typed-decision-models-ml.md`; this
> document states which instrument decides what, and never restates a formula. Section numbers are
> stable and cited from the code's and the context files' pointers.

## Index

| § | Topic |
|---|---|
| 1 | Purpose, non-features, standalone rule, roles |
| 2 | CLI: global rules, exit codes, each command, operator tooling |
| 3 | Run lifecycle: arms, capture order, item loops, timing, storage |
| 4 | LM Studio contract |
| 5 | Pack loading and validation |
| 6 | Scoring, per role |
| 7 | Typed decision points: `decisions` and `handoff` surfaces |
| 8 | Verdict statistics |
| 9 | Reports |
| 10 | Honesty rules the code enforces |
| 11 | Surface index: every shipped file and where its contract lives |

## 1. Purpose, non-features, roles

### 1.1 What it measures

model-bench is a human-started harness that measures **one local model (or one decision-model
candidate) at a time** against **one versioned task pack**, stores the result with a full
environment fingerprint, and compares stored results **within one pack**. A result taken today
lines up against one taken months ago on the same pack, with confidence intervals and a visible
banner whenever the comparison is not like for like.

### 1.2 Three non-features

- **No CI hook, scheduler or timer.** A run exists only because a person typed a command; no other
  component's suite invokes it.
- **No pass/fail gate.** There is no threshold of its own. A decision-point report states the
  requirement's bar and the exit code never reflects it; the only non-zero exits are operational
  (§2.2).
- **No leaderboard and no cross-role aggregate.** `results.load_history` takes a `packId`, and no
  API loads across packs, so a cross-pack number cannot be requested. Metrics of different roles
  are never combined.

### 1.3 Standalone, zero runtime dependencies

No runtime code path reads a path outside `model-bench/`. Data that originates elsewhere
(falkor-chat's golden sets, decision-models' datasets and candidates) is **copied in**, versioned
and given provenance by the one-way, human-invoked `scripts/refresh_golden.py` and
`scripts/import_decision_pack.py` (§2.10); a run never reaches those sources. The runtime is the
Python 3.12 standard library only (`urllib.request` for HTTP); `pytest` and `ruff` are the only
dev extras. The default test suite is network-free (`-m "not live"`); `pytest -m live` opts into
the tests that need a reachable LM Studio.

### 1.4 Roles

`roles.ROLES` is the closed set of seven roles. Each fixes the **unit kind** (the noun of the
analysis-unit denominator), the **analysis-unit field** (`pairingKey[0]` of every pack of the
role), whether a turn is a multi-call loop, and the **scorer module** under
`modelbench/scoring/` that its packs name.

| Role | Unit kind | Analysis-unit field | Multi-call turn | Scorer |
|---|---|---|---|---|
| `embedder` | `query` | `itemId` | no | `retrieval` |
| `guard-judge` | `item` | `itemId` | no | `classification` |
| `nlq-generator` | `item` | `itemId` | no | `extraction` |
| `chat-responder` | `item` | `itemId` | no | `grounding` |
| `tool-caller` | `conversation` | `scriptId` | yes | `toolcalls` |
| `join-in-judge` | `item` | `itemId` | no | `agreement` |
| `message-type-classifier` | `item` | `itemId` | no | `agreement` |

A role outside the set raises `UnknownRole`; there is no default. The last two roles and
`guard-judge`'s decision-point packs are the **decision points** (`BAR_KIND_BY_ROLE`, §7).

## 2. CLI

The CLI is `python -m modelbench` (`modelbench/__main__.py` → `cli.main`), launched by
`./run.sh <command>`. It has **seven commands**: `compare`, `rank`, `index rebuild`,
`models --tested`, `attest`, `validate`, `run`.

### 2.1 Global rules

- Every command except `validate` takes `--root` (default `.`): the directory holding `packs/`,
  `candidates/`, `results/`, `reports/` and `host.json`. `validate` takes a pack **path**.
- `main` returns the exit code (§2.2). An argparse usage error exits `2`; `--help` exits `0`.
- Run with `model-bench/` as the working directory or pass `--root`; `./run.sh` resolves the venv
  from its own location.

### 2.2 Exit codes (closed set, operational only)

| Code | Meaning |
|---|---|
| `0` | The tool ran and reported, **whatever the scores**. A `compare` that finds every stored record invalid prints `INVALID RESULTS EXCLUDED` and still exits `0`. |
| `2` | Bad arguments or usage (§2.3–§2.9 name each case). |
| `3` | LM Studio unreachable; reachable without its native `/api/v0` catalog; the warm-up failing or not answering within `--first-call-timeout`; the embedder's corpus embed failed before any item; a failed between-item residency probe that the re-probe confirms is an outage; on a decision-point pack also a failed explicit load, a model of the arm not resident after the warm-ups, or the decision model's instance config unreadable. |
| `4` | Invalid pack: a `validate` failure, a pack load error, an unmet `environment.requires`, a `callSurface` the catalog `type` contradicts, a tool-calling ineligible model, a candidate refusal (§7), an off-pin `--request-timeout` on a decision-point pack; or, **after the record is already stored**, a `tool-caller` conversation censored by a tool-dispatch failure. |
| `5` | Fingerprint incomplete, or `host.json` absent, invalid or stale; or `residentModelsAtEnd` not capturable after three retries. |

There is no score-driven exit code. An uncaught exception exits `1` (§2.8 names the one
shipped case).

### 2.3 `compare`

```
compare --pack <pack-id> [--models a,b] [--session <id>] [--negative-control] [--out <path>]
```

Reads `<root>/packs/<pack-id>/pack.json` (absent or invalid: exit `4`), loads the pack's stored
history (`load_history`, keyed on the manifest's `packId`), renders the markdown comparison
(§9.1), writes it to `reports/` **and** stdout, and prints `wrote <path>`.

- **Report path.** `reports/<packId>-<YYYYMMDD>-<NN>.md`, `<NN>` a two-digit same-day sequence
  that never overwrites an earlier file (an error past 99). `--out` writes exactly that path.
- **Arm selection, non-decision-point pack.** `--session` keeps the runs whose top-level
  `sessionId` matches. `--models` names arm keys in order; each key resolves to its **newest**
  stored run (`RunResult.armKey`: the candidate id on a candidate run, else the model key); a
  key with no stored run is a usage error (exit `2`, nothing written). Without `--models`, every
  valid run of the pack is an arm, in file-name order (run ids sort by arm key, then start time).
  The verdict compares the **first two** arms; the arms table lists all of them.
- **`--negative-control`** duplicates the first selected record as both arms, so `b = c = 0` by
  construction. It is a wiring smoke check, the report opens with a banner saying so, and it is
  refused (exit `2`) on a decision-point pack.
- **Decision-point pack** (the manifest has a `decisionPoint` key): the whole pack is loaded, and
  `--session` **and** `--models` are required. `--models` names at most one chat arm plus
  candidate ids; every run of each candidate in the session is a **replicate**, oldest
  `startedAt` first, and none is picked last-wins (§7.5, §9.4). Exit `2` with nothing written
  for: a missing `--session` or `--models`; a key with no run in the session; a key named twice;
  a second chat arm, or a chat arm that is not the pack's `decisionPoint.incumbent` (when the
  incumbent is non-null); a chat arm with other than exactly one run in the session; a candidate
  with more than one `candidateContentHash` in the session; arms whose
  `requestTimeoutSeconds` differ; `--negative-control`.
- Exit `0` whatever the scores; invalid or inconsistent records are excluded and named in the
  report (§9.2).

### 2.4 `rank`

```
rank --pack <pack-id> [--session <id>] [--reference <key>] [--footprints <path.json>] [--out <path>]
```

The N-arm sibling of `compare`: one ranked table per pack over every in-scope arm (§9.3). Arm
selection is `--session` filtering then the newest stored run per `armKey`. The report path is
`reports/<packId>-rank-<YYYYMMDD>-<NN>.md` (same sequence rule; `--out` overrides). Exit `2`,
nothing written, for: a `--footprints` file that is unreadable, not JSON or not a JSON object; a
`--reference` key with no stored, consistent run; any decision-point pack (use `compare`).
`--footprints` is a flat `{"<modelKey>": "<display string>"}` map passed through verbatim to the
footprint column; a non-string value is coerced with `str()`, never parsed as a number.

### 2.5 `index rebuild`

Regenerates `results/index.csv` from `results/runs/` (columns in `FORMATS.md` §4.3) and prints
its path. The file is derived and fully regenerable; an unparseable record is skipped.

### 2.6 `models --tested`

```
models --tested [--pack <pack-id>] [--role <role>]
```

Prints one **arm key** per line for every model arm with a stored record (`armKind == "model"`,
so a deterministic reference arm is never listed): the candidate id on a candidate run, else the
model key. Filtered by pack id and role when given.

### 2.7 `attest`

```
attest [--api-base-url <url>] [--set key=value ...]
```

Writes `host.json` (`FORMATS.md` §6): gathers the four operator-attested fields, probes LM
Studio, and writes only if the probe answers `api-v0`. `--set` is repeatable over `lmStudioAppVersion`, `kvCacheSetting`,
`hostRamGb` (an integer) and `otherResidentWorkloads` (comma-separated); a field not given is
prompted for on stdin. Exit `2`: a malformed or unknown `--set`, a non-integer `hostRamGb`, stdin
exhausted with fields unset, an empty base URL. Exit `3`: the probe is not `api-v0` (nothing is
written). The default base URL is `http://localhost:1234`.

### 2.8 `validate`

```
validate --pack <path> [--strict]
```

Structural pack integrity with **no LM Studio connection** (§5.2). Prints
`<packId> <packVersion> (<role>): valid` and exits `0`, or prints each problem on stderr and
exits `4`; an unloadable pack exits `4`. `--strict` is accepted but unimplemented: it raises
`NotImplementedError`, an uncaught exception (traceback, exit `1`), rather than silently doing
nothing.

### 2.9 `run`

```
run --pack <id> (--model <key> | --candidate <id>) [--session <id>] [--reference <key>]
    [--warmup <n>] [--first-call-timeout <s>] [--request-timeout <s>]
```

One arm × one pack against a live LM Studio. `--model` and `--candidate` are mutually exclusive
and one is required. The command loads and validates the pack (`validate_pack`; failure exit
`4`), reads `host.json` (exit `5`), builds the client from `host.json`'s `apiBaseUrl` (`run`
has no URL flag), runs `run_pack` (§3), stores the record, prints `stored: <path>`, and maps a
`RunRefused` to its exit code. `--warmup <n>` adds `n` untimed warm-up calls after the mandatory
one; `--first-call-timeout` defaults to 300 s; `--request-timeout` defaults to 120 s on a
non-decision-point pack, and a value other than 180 on a decision-point pack is refused with exit
`4` (§3.6). `--session` is recorded as the run's `sessionId`, which `compare` and `rank` filter on.
`--reference` is accepted and ignored by `run`: nothing records it (the reference key that
matters is `rank --reference`).

After the record is stored, on a pack whose scorer defines a `deterministic_arm` (the
`embedder` pack's BM25 arm) a second record is stored and `stored (deterministic arm): <path>`
printed. A `tool-caller` run that censored any conversation by a dispatch failure then prints a
`PACK DISPATCH FAILURES` block (`<scriptId> turn <n>: <tool> — <reason>` per line) and exits
`4`; the record is already on disk.

### 2.10 Operator tooling

| File | Contract |
|---|---|
| `setup.sh` | Creates or refreshes `model-bench/.venv` (Python 3.12 or newer, `PYTHON` overrides the interpreter), installs the package editable with the `dev` extra, and import-smokes it. Idempotent; `--recreate` rebuilds; unknown option exits `2`. |
| `run.sh` | `exec`s `python -m modelbench "$@"` from the component venv, resolved from the script's own location; exits `1` with a message when the venv is missing. |
| `run-sweep.sh <plan-file> <session-id> <log-file>` | Reads `model|pack` lines, runs `./run.sh run --pack <pack> --model <model> --session <session>` for each, never stopping on a failed run, appending one `START` and one `OK`/`FAIL` line per attempt to the log as it goes (a failure's output follows, indented). Exit `3` from a run triggers an independent `curl` of `/api/v0/models` at `http://localhost:1234`; two consecutive confirmed outages halt the sweep with exit `42`. Ends with a `SWEEP DONE` line. |
| `scripts/refresh_golden.py --pack <dir>` | One-way importer of the `embedder`, `guard-judge-understanding` and `nlq-structured-query` packs' golden data from falkor-chat (any other pack id is refused, naming the known ones). Modes: default re-import (refused under an unchanged `packVersion`, rewrites `PROVENANCE.md`); `--check-origins` (read-only drift report, exit `0` iff nothing drifted); `--embed-corpus --model <key>` (the one live mode: embeds `corpus.jsonl` and writes `corpus.embeddings.json`, same version gate); `--check-tables-shape --source-git-sha <sha>`; `--stamp-answerability`. Also `--repo-root` (the monorepo root; default: the script's own ancestor) and, for `--embed-corpus` only, `--api-base-url` and `--timeout` (seconds for each of the warm-up and the batched embed call; default `300`). Exit `1` refusal, `2` bad arguments. Never imported by a run path. |
| `scripts/import_decision_pack.py` | `--role <r> --dataset <dir> --pack-version <v>` copies a decision-models dataset's eval split into `packs/<r>-synthetic/` (`FORMATS.md` §1.6); `--candidate <file>` copies a candidate and its fit manifest into `candidates/` (`FORMATS.md` §5). `--root` names the `model-bench` directory (default: the script's own). Exit `0` imported, `1` refused, `2` bad arguments. |
| `scripts/consolidate_sweep_reports.py --reports <files...> --out <path>` | Stitches already-rendered `rank` reports into one index-plus-placeholders document by reading each report's `rank-report` marker (`FORMATS.md` §8.2). Zero arithmetic; exit `2` for a missing report file, `1` when no marker is found. |
| `scripts/s6_walkthrough.py [--pack <dir>] [--script <id>]` | Read-only review aid: prints each `tool-caller` script's turns, its `expect` block, and for a `toolRequired` turn the real dispatch result from the pack's simulated environment. Writes nothing and never fills a `verifiedBy` field. |

## 3. Run lifecycle

`runner.run_pack` is the one entry point of `run`. It raises `RunRefused` (carrying the exit code,
§2.2) for every refusal and writes nothing in that case; otherwise it returns the assembled
`RunResult` and any dispatch-failure disclosures.

### 3.1 Arms and call surfaces

A run's arm is **exactly one** of a `modelKey` (a chat or embeddings arm) or a `candidateId` (a
decision-model candidate, §7); `RunConfig` refuses both or neither. The arm's identity (`armKey`)
is the candidate id on a candidate run, else the model key, and the run id is
`<packId>-<armKey>-<startedAt UTC, YYYY-MM-DDTHH:MM:SSZ>` with `/` replaced by `_`.

The **call surface** says which LM Studio route scores an item:

| Surface | Arm | Route | Fingerprint profile |
|---|---|---|---|
| `chat` | `--model`, pack requires `lmstudio-chat` | `POST /api/v0/chat/completions` | `model:chat` |
| `embeddings` | `--model`, pack requires `lmstudio-embeddings` | `POST /api/v0/embeddings` | `model:embeddings` |
| `decisions` | `--candidate` without a hand-off model | `POST /v1/decisions` | `model:decisions` |
| `handoff` | `--candidate` with a hand-off model | `/v1/decisions`, then chat | `model:handoff` |

A chat or embeddings arm takes its surface from the pack's `environment.requires`: exactly one of
`lmstudio-chat` / `lmstudio-embeddings` must be declared, else exit `4`. The surface must agree
with the model's catalog `type` (`chat` needs `llm` or `vlm`; `embeddings` needs `embeddings`),
else exit `4`. A `tool-caller` pack additionally requires tool-calling eligibility: catalog type
`llm` or `vlm`, and `capabilities` either absent or containing `tool_use`, else exit `4`. The
`deterministic` profile belongs to the reference arm that a scorer builds without any LM Studio
call (§6.2).

### 3.2 Capture order

`run_pack` takes these steps in order; each refusal happens before any later step.

1. **Request budget.** The chat-call budget is resolved first (§3.6).
2. **`host.json`** is read and validated; absent or invalid refuses with exit `5`.
3. **Probe.** `LMStudio.probe()` must answer `api-v0`; `v1-only` and `unreachable` refuse with
   exit `3` and distinct messages.
4. **Residency at start** (`residentModelsAtStart`) and the **catalog** are read. Then the arm is
   prepared: a model arm's refusals (§3.1) or a candidate arm's pre-flight refusals (§7.4).
5. **Explicit load, decision-point packs only.** Every model of the arm absent from
   `residentModelsAtStart` gets one `POST /api/v1/models/load` at `--first-call-timeout`; a
   model resident at start is used as found. A failed load refuses with exit `3`. A model the
   runner loaded **stays resident after the run** (an explicit load has no TTL).
6. **Mandatory warm-up** of each model of the arm at `--first-call-timeout`, then `--warmup <n>`
   additive extras whose results are discarded. A failed mandatory warm-up refuses with exit `3`.
   The decision model is warmed up first; a `handoff` arm then warms up the hand-off model on the
   chat route. The warm-up is never scored, never stored as an item.
7. **Attestation trip-wire** (§3.9); a stale attestation refuses with exit `5`.
8. **Baseline residency** is read for the first item's load check, never reusing step 4's. On a
   candidate arm every model of the arm must be resident, else exit `3` before any item.
9. **Catalog after load** supplies `loadedContextLength`. On a candidate arm the decision
   model's instance config is read from `GET /api/v1/models` (`loaded_instances[0].config`);
   any path that does not yield a config object refuses with exit `3`.
10. **Items** are driven (§3.3–§3.5, §7). A candidate arm's `basis` is then decided (§7.6).
11. **`residentModelsAtEnd`** is read, with three retries one second apart; still failing refuses
    with exit `5`.

The fingerprint, latency block and `RunResult` are then assembled (`FORMATS.md` §2–§3). `cli` stores
the record through `results.store`, which validates the fingerprint and refuses to write an
incomplete one; there is no bypass (§10).

### 3.3 Single-call surfaces (`chat`, `embeddings`)

Items come from `items.jsonl` in file order. **One call per item**; the scorer module resolved
from `pack.json`'s `scorer` (`modelbench.scoring.<name>`; an absent name or unresolvable module
is exit `4`) shapes it:

- Chat: `scorer.build_messages` when the scorer defines it, else a generic message list (the
  pack's system prompt, then the item row JSON-encoded), sent at the pack's `prompt.temperature`
  and `prompt.maxTokens`.
- Embeddings: `scorer.embed_text` (required); a scorer's `prime` hook, when defined, runs once
  before the loop. A failed `prime` call (the embedder's corpus embed) refuses with exit `3`.
- Before each item a **between-item residency snapshot** is taken (§3.5), never during a timed
  call.
- A call timeout or failure does not stop the run: the item is scored with no result and
  `withheldFor` `timeout` or `no_response` (§3.5).

`designEffect` is `1.0` and `basis` is `by-construction` on every single-call run: the sampling
unit is the item.

### 3.4 The `tool-caller` conversation surface

`tool-caller` packs are driven as scripted conversations (`modelbench/convo.py`). For each script
a **fresh** simulated tool environment is built (`tools.entrypoint` of the pack's `tools.module`,
satisfying `modelbench.tooling.ToolEnvironment`), the scored scripts run first, then the pack's
`sampling.determinismProbeScripts` are re-run.

- **Turn loop.** Per scripted turn the model is called; while a response carries native
  `tool_calls` and the cap `prompt.maxIterationsPerTurn` is not reached, each call is dispatched
  against the environment, the assistant message and one `tool` message per call are appended to
  the in-turn list, and the model is called again. The turn ends on a response with no tool
  calls (`replied`, its text is `finalReplyText`) or at the cap (`cap-hit`). A call that raises
  ends the turn as `timed-out` (timeout), `server-rejected` (a failure carrying an HTTP status) or
  `no-response` (a failure with no status); the exception is tested **before** the cap. Every scripted turn runs, in
  order, whatever happened before.
- **Prompt assembly.** The harness resends the conversation from scratch each turn:
  one merged system message (the pack's system prompt, then the tool-schema text when it applies,
  on turn 0 or every turn under `representToolSchemasEachTurn`), the replayed history per
  `prompt.historyReplay`, then the turn's user message last. History is replayed **from what the
  model actually produced in this run**, never from the script's `expect`. Modes: `structured`
  (the whole exchange in native roles), `structured-replies-only` (user and final reply per turn),
  `plaintext` (a flattened transcript in one user message), `none`. `historyTurns > 0` windows the
  replayed tail. A turn with no final reply replays as an empty assistant reply, never omitted.
  The native `tools` parameter is sent on every turn when the pack declares tool schemas.
- **Dispatch rules.** A `function.arguments` value that is not a JSON object is not dispatched
  and its `tool` message names the failure; every `tool_calls` entry gets exactly one `tool`
  message. A pack environment that raises in `dispatch` ends the run's conversation with a
  `ToolDispatchFailed` (a pack defect, never mistaken for a server outage); the runner stores
  the conversation **censored** at that turn (turns before it kept, it and later turns absent),
  continues with the next script on a fresh environment, and the CLI exits `4` after storing
  (§2.9). A pack environment whose trace shrinks, is rewritten, or grows by other than one
  record per dispatch raises `TraceContractViolated` and aborts the run.
- **Basis.** `designEffect` is `1.0`; `basis` is `by-construction` only when
  `sampling.replicatesPerScript == 1`, every determinism-probe script ran and
  `determinismProbe.identical` holds, else `assumed`.

### 3.5 Timing and the residency probe

Each item carries an `ItemTiming` (`FORMATS.md` §2.2): the item's wall clock, one `CallTiming` per
**completed** call, and `withheldFor`, the single reason the item contributes no latency.

- `timeout` and `no_response` mark a call that never completed (a censored and a missing
  observation respectively); both are counted in one latency counter.
- `load` marks an item taken to include a model load. It is set when the **preceding**
  residency snapshot did not show the model (every model of the arm on a `handoff` arm)
  resident, or, on a chat or embeddings call, when the summed per-call gap
  `wallClockMs − (ttftMs + generationMs)` exceeds 1000 ms. A turn that failed takes its failure
  disposition and no load rule runs on it.
- **Between-item snapshot.** It is taken by `runner._between_item_residency`, never a bare
  `residency()`. A failed probe (no response, a non-200 status or malformed body, or a socket
  timeout) re-probes the server: anything but `api-v0` refuses the run with exit `3`; otherwise
  the snapshot reads **empty** — "not shown resident", never the previous snapshot, which would
  let a reloaded run pass as one load — and one stderr line names the probe's place and the
  exception class. What an empty snapshot does is each surface's own load rule above.
- On a `tool-caller` run the snapshot granularity is per conversation, taken before it and again
  after it for the next one.
- The wall clock is measured around the HTTP call to the last byte of the body. On the
  `decisions` and `handoff` surfaces the item clock (a monotonic clock around the whole item) is
  used instead (§7.3).

`LatencyBlock` (`FORMATS.md` §2.3) is built in one pass over a run's items. A percentile is
reported only above its coverage floor: `latencyMsP50` when the median's rank is at least 45% of
the item count, the tail figure when its rank is at least 90%; when the tail rank equals the
number of timed items the figure is stored as `latencyMsMax` and `latencyMsP95` is `None`. The
TTFT, prefill-per-1k and tokens-per-second medians come only from calls with usable `stats` **and**
a positive prompt-token count, are `None` below the same 45% coverage floor (the median's rank over
those calls against the run's attempted calls), and are `None`, never `0`, on a surface with no
`stats` (embeddings, decisions, handoff); `statsCoveredCount` is `None` there too.

### 3.6 Request and first-call budgets

`--first-call-timeout` (default 300 s) bounds each warm-up and each explicit load. The scored-call
budget is `RunResult.requestTimeoutSeconds`, recorded on every run: the explicit
`--request-timeout`, else 120 s. **On a decision-point pack it is pinned to 180 s** and an
explicit value other than 180 refuses with exit `4` before any call; the pin applies to every
arm, including the hand-off and incumbent chat calls. A decision call's own budget is the
candidate's `decisionTimeoutSeconds` (§7.3). A decision-point `compare` over arms whose
`requestTimeoutSeconds` differ is an exit-`2` refusal (§2.3); a report states a shared non-180
value (§9.4).

### 3.7 Scorer seam

A role's scorer is a **module** under `modelbench/scoring/`, structurally satisfying
`runner.ItemScorer` (single-call roles: `score_item`, `aggregate`) or `runner.ConversationScorer`
(`tool-caller`: `score_conversations`). Optional hooks, each called only if defined: `prime`,
`embed_text`, `build_messages`, `deterministic_arm`. A scorer used on a decision surface also needs
`decision_input`, `score_answer` and `aggregate`, and on `handoff` `answer_from_chat`; a candidate
run whose scorer lacks one refuses with exit `4` naming it. Every scorer owes two things:
a `counts` entry for every metric it declares scoreable, and `aggregates` derived from the same
items in one pass (§10).

### 3.8 Storage

A record is written to `results/runs/<runId>.json` (`FORMATS.md` §2), UTF-8 JSON with sorted keys.
`results/runs/` is committed; `results/transcripts/` is gitignored and no run writes raw model
replies there or anywhere. A stored record never contains a model's reply text, so auditing a
scorer against a real reply needs a fresh live call.

### 3.9 Attestation trip-wire

Four fingerprint fields have no programmatic source and are attested by the operator
(`lmStudioAppVersion`, `kvCacheSetting`, `hostRamGb`, `otherResidentWorkloads`); they live in
the gitignored `host.json` and are copied into every run record. After the warm-up the trip-wire
compares what the chat warm-up observed against `host.json`'s `observedAtAttestation`:

- `unavailable` — an embeddings or decisions warm-up (no `runtime` object to compare); not
  stale. A `handoff` arm is fed by its **hand-off chat** warm-up, so it is `first-observation`
  or `compared`.
- `first-observation` — no runtime recorded yet; the runtime name, version and observation time
  are back-filled into `host.json` and the rest is untouched; not stale.
- `compared` — the runtime name, runtime version or `residencySource` differ from the recorded
  ones: stale. The run refuses with exit `5` and the message tells the operator to re-check the
  app version and KV-cache setting, then re-run `attest`.

The outcome is stored on the run (`attestationTripWire`); it is `None` exactly on a
deterministic arm.

## 4. LM Studio contract

`modelbench/lmstudio.py` is the only code that talks to the network.
`LMStudio(base_url, opener=urllib.request.urlopen)`; an injected `opener` is how offline tests
replace the transport.

### 4.1 Routes

| Operation | Method and route | Used for |
|---|---|---|
| `probe()` | `GET /api/v0/models`, then `GET /v1/models` | `api-v0` (catalog answers with a `data` list), `v1-only`, `unreachable` |
| `catalog()` | `GET /api/v0/models` | model identity fields; 10 s socket budget |
| `residency()` | the catalog filtered on `state != "not-loaded"` | `[{id, state}]`; `[]` on a clean box is a real answer |
| `loaded_instances(key)` | `GET /api/v1/models` (`models[].key`, `loaded_instances[]`) | a decision model's instance config |
| `chat(...)` | `POST /api/v0/chat/completions` | `stats`, `model_info`, `runtime` exist only here |
| `embed(...)` | `POST /api/v0/embeddings` | batched; `dimension` from the first vector |
| `decide(...)` | `POST /v1/decisions` | one typed decision; the response has no `stats` or `runtime` |
| `load_model(...)` | `POST /api/v1/models/load` with `echo_load_config` | the runner's explicit load; the response must name an `instance_id` and report `status == "loaded"` |

There is no unload operation: the harness cannot force a cold state, by design. `timeout_s` is
required with no default on `chat`, `embed`, `decide`, `load_model` and `warm_up`.

### 4.2 Error taxonomy

- `LMStudioCallTimeout` — the budget was exhausted at the connect or the read phase.
- `LMStudioCallFailed` — any other failed call, always carrying `status` (the HTTP status when the
  server answered and refused, `None` when the call never completed or the 2xx body was unusable)
  and `body` (the full HTTP error body on an HTTP error, else `None`). A refusal status is a
  `server-rejected` turn; `None` is `no-response`.
- `LMStudioUnreachable` — no response from a catalog or instance read.
- A malformed catalog entry (a required key missing) is a `LMStudioCallFailed` with `status=None`.

### 4.3 Unit boundary and derived timing

`stats.time_to_first_token` and `stats.generation_time` are seconds and are converted **once**, in
`ChatResult`, to `ttftMs` and `generationMs`; `tokensPerSecond` is type-coerced but not
converted. Each is `None` when its source is absent, a boolean, or non-finite, never `0`;
`ChatResult` construction never raises whatever `stats` holds, and the raw `stats` is kept only for
audit. `toolCallForm` is `native` when `message.tool_calls` is non-empty, else `prose`. The
catalog's `capabilities` is `None` when the key is absent and a (possibly empty) tuple when
present.

### 4.4 Decisions route

`decide(input, questions, model, timeout_s)` posts one input and its question specs and requires
the response to answer **every** question exactly once with an answer of its own type: a predicate
answer carries a `probability` in `[0, 1]`; a choice answer a `choice` among its declared values,
`probabilities` over declared values and a `confidence` in `[0, 1]`. A body that breaks this is a
`LMStudioCallFailed` with `status=None`. The decisions warm-up is one `decide` call with the
fixed input `warm-up` and the candidate's own question. A single input above the decision model's
physical batch is an HTTP `500` whose body contains `too large to process` (§7.3), not a
truncation.

## 5. Pack loading and validation

The manifest format is in `FORMATS.md` §1. This section states what loading and validation do.

### 5.1 Loading

`load_pack(root)` reads `<root>/pack.json`, hashes the pack directory and returns an immutable
`Pack`. It raises `PackConfigError` only when the manifest cannot be identified: unreadable JSON,
or a missing `packId`, `packVersion` or `role`. Every deeper contract is `validate_pack`'s, run
separately so that a caller sees every problem at once. `pack_ref_from_manifest` is the
manifest-only read that `compare` and `rank` use (no hashing, no data check); `Pack.ref()` is the
same record with a non-null content hash.

The **pack content hash** is the SHA-256 over every file under the pack directory in sorted
relative-path order, each entry contributing its POSIX relative path and its bytes, both
NUL-delimited, **excluding `PROVENANCE.md` and `__pycache__`**. Any byte change to a golden item,
prompt, tool schema or pack module is a new, distinguishable pack version in effect: stored runs
record the hash, and a comparison across two hashes is banner-labelled unpaired (§9.2).

### 5.2 `validate_pack`

Returns the list of problems; `[]` is valid. It has **no model catalog and no LM Studio
connection**; the catalog-`type` cross-check and the tool-calling eligibility gate are `run`'s
(§3.1). The axes, each independent:

1. **Sampling contract.** `sampling.pairingKey` is a non-empty list; `sampling.analysisUnit`
   equals `pairingKey[0]`, and that component is the role's own analysis-unit field (§1.4);
   `sampling.seed` is present (the continuous bootstrap's seed); the `metrics` block has a
   non-empty `verdictMetrics`, a `headlineMetric` key that is present (`null` is a decision,
   omission is not) and, if non-null, a member of `verdictMetrics`. `sampling.replicatesPerScript`
   above `1` is rejected. On a pack declaring `sampling.scripts` the **row-count identity**
   holds: `data.conversations` has `scripts × replicatesPerScript` rows, `scripts` distinct
   analysis-unit values, each appearing `replicatesPerScript` times, and every identity key has
   its declared type; an item-level pack (no `sampling.scripts`) is the one sanctioned skip.
2. **Call surface.** `environment.requires` declares exactly one of `lmstudio-chat` or
   `lmstudio-embeddings`.
3. **`tools.module` path.** A string ending in `.py` that resolves inside the pack root.
4. **Import allowlist.** Every `.py` file in the pack imports the standard library and
   `modelbench.tooling` only (an AST walk; relative imports within the pack are allowed). This is
   a coupling rule over declared imports, **not a sandbox**: `__import__` or `importlib` inside a
   pack module reaches anything importable, so a validated pack is one whose declared imports are
   in scope, never one whose code is safe to run untrusted.
5. **`prompt` block.** `prompt.historyReplay` and `prompt.maxIterationsPerTurn` are role-scoped:
   `historyReplay` must be one of the four modes and `maxIterationsPerTurn` must be present **iff**
   the role has a multi-call turn (`tool-caller`), forbidden otherwise. An absent `prompt` block
   is not a problem.
6. **`scorer` resolution.** A declared `scorer` resolves to `modelbench.scoring.<name>`
   (skipped for `tool-caller`, whose scorer is a different kind). An absent `scorer` is not a
   validation problem; `run` refuses it with exit `4`.
7. **Answerability stamp** (`nlq-generator` only): every `items.jsonl` row has an `answerable` key.
8. **`cleanThroughTurnH`.** `metrics.cleanThroughTurnH.H` does not exceed the shortest script's
   turn count.
9. **Decision-point contract** (§7.1), keyed on the presence of a `decisionPoint` key.

`validate_pack` does **not** check per-item `provenance`, `PROVENANCE.md`, or the content of
`data` keys beyond what the axes above read.

### 5.3 Pack data readers

`Pack.iter_items()` yields each non-blank `items.jsonl` row (`data.items`) in file order, parsed;
`iter_scripts()` yields each `data.conversations` row as a `Conversation` of `Turn`s;
`iter_prose_calibration()` yields `(text, isPseudoCall)` pairs; `find_script(id)` looks a script
up by `scriptId`; `load_tool_module()` imports the pack's `tools.module` by path. A scorer, not
the loader, owns the typed reading of a role's rows.

## 6. Scoring

Each role's scorer is a module under `modelbench/scoring/` (§1.4). This section states what each
declares and reports; the record shapes are in `FORMATS.md` §2.

### 6.1 Common rules

- **Outcome vocabulary.** An item's `outcome` is one of `pass`, `fail`, `parse_failure`,
  `unrunnable`. A call that returned no result is scored `fail` when it timed out and
  `unrunnable` when it produced no response, with `scoreable = {}`: such an item leaves every
  metric's denominator and is counted, never charged to the model. A `pass` outcome means "the
  call happened and was readable", never "the answer was right"; correctness lives in `counts`.
- **Declared, never inferred.** For each metric an item makes one of three statements: absent from
  `scoreable` (no statement, no outcome), `scoreable[m] = false` (a declared precondition
  failure, counted as an asymmetry in a paired comparison), or `scoreable[m] = true`, which
  **must** carry a `counts` entry (binary) or a `measures` entry (continuous). A scoreable metric
  with no entry is refused (`IncompleteItemRecord`), never read as zero. A name lives in `counts`
  **or** `measures`, never both, and every `measures` value is finite.
- **Aggregates** are a closed union of typed dataclasses, one per role family (`FORMATS.md`
  §2.4); there is no field that could hold a pooled cross-class accuracy. Every `BinaryMetric`
  states the **unit** of its denominator (`item`, `query`, `conversation`, `turn`, `call`).
- **Verdict versus exploratory.** The pack's `verdictMetrics` are the pre-registered family;
  every other metric a scorer reports is exploratory and printed without a significance claim.
- **Polarity.** `falseAdvanceRate`, `falseSuspendRate` and the four per-kind guard rates are
  undesired-event rates: lower is better in `rank` ordering and reference-anchored verdicts.
- **Stored aggregates must agree with stored items.** A scorer derives its aggregates from the same
  items in one pass; a run whose verdict-family denominators disagree with its items is excluded
  from a comparison and named (§9.2).

### 6.2 `embedder` — `scoring/retrieval.py`

Exact brute-force cosine retrieval, no ANN index and no database: the object of measurement is
the embedding model, not a retrieval pipeline. `prime` embeds the pack's corpus (121 documents in
the shipped pack) once through the model under test, or reuses `corpus.embeddings.json` when its
cache key (model, quantization, `documentPrefix`, corpus SHA-256) matches the run's; a live embed
is not persisted. Per query (`embedding.queryPrefix` + the query) every corpus document is ranked
by cosine. Per item it declares `recallAt<k>` for each `metrics.recallAtK.ks`, `precisionAt1` (in
`counts`) and `mrr`, `separationRaw`, `separationZ` (in `measures`). Aggregates: `recallAt<k>` and
`precisionAt1` as binary metrics over `query`s (an item counts as a recall success when at least
one relevant document is in the top *k*), `mrr` a `ContinuousMetric` on `[0, 1]`, and
`separationRaw`/`separationZ` `DistributionSummary` median-and-p10 figures, no mean. A query with
no result is excluded from all five. A model that wins here has **not** been shown to win through
falkor-chat's ANN retrieval path.

`deterministic_arm` builds a **BM25 reference arm** with no LM Studio call, through the same
scoring path (always-positive IDF, Unicode word tokenization with case folding, the pack's `k1`,
`b` and stopwords file). It is stored as a second record with fingerprint profile `deterministic`,
`armId` `bm25`, no timings and `latency` `None`, `designEffect` `1.0`, `basis` `by-construction`.
`run` stores it after the model arm's record.

### 6.3 `guard-judge` — `scoring/classification.py`

One judge call per item (the judge prompt is `prompts/judge.md`; the user message is rendered
from the item's `condition`, `understanding` and normalized `turns`, never from `expected`). A
reply is read by the conservative own-line JSON extractor requiring a `decision` key; no object is
a `parse_failure` and reads as no advance.

- **Non-decision-point pack** (`guard-judge-understanding`): the item's `tier` fixes its one
  metric: `falseAdvanceRate` (`clear_suspend`), `falseSuspendRate` (`clear_advance`),
  `falseAdvanceRateBoundary` (`boundary`, exploratory). The reply parse is `bool(decision)`.
  `perClass` carries the two verdict rates, `advanceRecall` (the complement of
  `falseSuspendRate`), the boundary rate and four path-split diagnostics
  (`falseAdvanceRateByUnderstanding|ByTurns`, `falseSuspendRateByUnderstanding|ByTurns`). No metric
  pools tiers; every `n` is its own subset.
- **Decision-point pack** (`guard-judge-synthetic`, §7.1): an `eval` item declares its per-kind
  verdict metric, from its label and `path` (a `false` label's error is a false advance, a `true`
  label's a false suspend), plus the pooled `falseAdvanceRate`/`falseSuspendRate` as exploratory
  figures; a `side-check` item declares only its `sideCheck*` metric by tier. The four per-kind
  rates are the verdict family. The reply parse is falkor-chat's verdict rule: only a JSON
  `true` whose rationale carries no un-negated cue (`not enough`, `still missing`, …) advances, so
  `"decision": "true"` suspends. A guard pack with `decisionPoint` therefore keeps a stricter parse
  than the `guard-judge-understanding` pack's looser `bool(decision)`.
- On a decision surface the scorer's `detail` carries `decidedBy`, `failedCall`, `reason`,
  `probability` and `inputTokens` (§7.3). Every illegal combination of these raises, so a runner
  defect cannot reach a stored record.

### 6.4 `nlq-generator` — `scoring/extraction.py`

One structured-completion call per item; execution is in-process against the pack's `tables.json`
through the pack's own `tools/exec.py` (no FalkorDB). Three gates, each with its own recorded
failure class: no parseable JSON object with a `matches` key (`parse_failure`, `no_json`), a spec
that fails structural validation (`malformed_spec`), a spec that fails schema validation
(`schema_violation`); the last two are `fail`. None of them declares any metric scoreable.
A spec whose `order_by` property is null on a row being sorted raises an uncaught `TypeError`
past all three gates and out of the scorer, which ends the run; no shipped item reaches it
(open: `docs/BACKLOG.md`). A spec that executes is compared to `expected` by `score_pair` (scalar, set, `not_found`; the
`conflicting-facts` shape uses subset containment, every other set shape exact set equality).

An **answerable** item declares `layer1ExactMatchRate` (the verdict and headline metric) and its
shape's exploratory `exactMatchBy<Shape>`. An **unanswerable** item (stamped `answerable: false`)
declares `unanswerableAbstainRate` (`1` iff the executed result is empty) and also its shape's
`exactMatchBy<Shape>`; if that exploratory spec nonetheless scores correct, `detail.luckyPass` is
set and `luckyPassCount` counts it, additively. The headline denominator is the answerable items
only (34 of the shipped pack's 40). Aggregates also carry `parseFailures`, `malformedSpecCount`,
`schemaViolationCount` and `luckyPassCount` as separate counts.

### 6.5 `chat-responder` — `scoring/grounding.py`

The deterministic layer only: grounded-reply containment against the supplied context, plus
format compliance. **It never measures reply quality**; the report states this in words. Per item
the scorer builds the prompt from the pack's system prompt, one deterministic sentence naming the
item's resolved format constraints, the numbered context passages and the question, and never
reads `mustContain`, `mustNotContain`, `mustAbstain` or `provenance`. It declares four metrics on
every item: `groundingRate` (the verdict and headline metric: all `mustContain` strings present,
no `mustNotContain` present, and the reply abstains iff `mustAbstain`, by canonicalized substring
match) and the independent `formatMaxWords`, `formatSingleParagraph`,
`formatNoForbiddenPatterns` (pack-level `format`, overridable per item key by key). Abstention is
recognized by fixed phrasings and the "don't mention" idiom, which a same-sentence contrastive
clause containing a digit turns back into an answer; sentence scoping uses one boundary rule
(`.`, `!`, `?`, except a `.` between two digits) that any new same-sentence heuristic must reuse.
`parseFailures` is always `0`.

### 6.6 `tool-caller` — `scoring/toolcalls.py`

A `ConversationScorer`. Its `items` hold **exactly one `ItemResult` per scored conversation**
(`itemId` = `scriptId`, `pairingKey` = `(scriptId, replicate, H−1)` with `H` =
`metrics.cleanThroughTurnH.H`), never one per turn; probe scripts contribute nothing to `items`.
The verdict and headline metric `cleanThroughTurnH` is declared on every conversation: `pass`
(clean through turn `H`), `fail`, or `unrunnable` with `scoreable` `false` (a conversation
censored before turn `H`, or with a `no-response` or `server-rejected` turn among its first `H`).
Aggregates:

- `cleanThroughTurn` (`cleanThroughTurnH`, unit `conversation`) and `restraint` (unit `turn`);
- `funnel`: `iterationCapHitRate`, `native`, `rightToolChosen`, `allArgsCorrect` (unit `call`),
  `spurious`, `duplicateWithinTurn`, `duplicateCrossTurn`, `stoppingWhenDone`, `replyMatchesTool`,
  pooled over turns or calls, each stating its unit, plus `funnelCounts` (the structural counts the
  funnel table prints, including unrunnable model-channel and tool-channel turns);
- `perTurnPosition` and `hazard` (per-position `f_t`, `r_t` and newly censored counts; never a
  rate);
- `determinismProbe` (`scriptIds`, `ran`, `identical`, `differingTurns`), `iterationSummary` and
  `prosePseudoCallDetector` (`n`, `precision`, `recall` against the pack's calibration corpus).

Turn scoring reads each turn's `expect` block as an oracle (`toolRequired`, `tool`, `args`,
`finalReplyMustContain`, `finalReplyMustNotContain`); the model's reply text is never ground truth
for state, which comes from the simulated environment's trace and state. A turn disposition
`no-response` or `server-rejected` leaves every scoring denominator (counted, printed in the
funnel head); `timed-out` is scored as a failure.

### 6.7 `join-in-judge`, `message-type-classifier` — `scoring/agreement.py`

The new-ability roles' scorer (§7). Each `eval` item declares **one** metric, its true answer's
class — `agreeWhenYes` or `agreeWhenNo` on the predicate point, `agree[<label>]` on the choice
point — counted `1` when the **final answer** equals `expected`. The final answer is the decided
answer; on a kept failed call it is the role's fail-closed answer (`false` for join-in, no label
for message type); an over-limit hand-off is scored on the hand-off model's answer. `perClass`
holds one metric per class in label order, every one present (`n = 0` when no item declared it).
Reply parsing: join-in reads an own-line JSON object with a `reply` key and only a real `true`
replies; message type reads a `type` key whose value must be one of the pack's labels exactly,
else no label. `parseFailures` counts `parse_failure` outcomes. The decision input and the
hand-off user message are the same rendering (`render_join_in_input`,
`render_message_type_input`; per-turn caps 400 characters, message text 2000); the hand-off system
prompt is pack data (`prompts/<point>-handoff.md`). As a chat arm (no candidate) the scorer scores
the hand-off model alone; a no-result item is withheld as on every chat arm.

## 7. Typed decision points

A **decision point** is a role whose requirement bar is stated by the requirements document
(`docs/requirements/typed-decision-models.md`, whose `FR-1`, `FR-2` and `FR-10` this document cites) and measured against a **decision-model
candidate**: a small model on LM Studio's decisions route that answers a typed question, with a
policy that decides when to answer itself and when to hand the item to a chat model. The
requirement constants live in `modelbench/decisions.py`, never in pack data: a pack can neither
pick nor lower its bar.

| Constant | Value | Meaning |
|---|---|---|
| `MIN_EVAL_ITEMS` | 200 | scored `eval` items below which a bar is `not assessable` |
| `FR2_MIN_SPEEDUP` | 2.0 | the guard's speed-up bar (ratio of totals, at least) |
| `FR10_PER_ANSWER_AGREEMENT` | 0.90 | each class's agreement bar on a new ability (at least) |
| `FR10_MEAN_LATENCY_MS_MAX` | 350 | a new ability's mean-latency bar (at most) |
| `DECISION_POINT_CHAT_TIMEOUT_SECONDS` | 180 | the pinned chat budget (§3.6) |

`BAR_KIND_BY_ROLE` maps each decision role to its bar kind: `guard-judge` is a **speed-up** point
(the candidate replaces a chat judge, the `incumbent`, and must match its quality and be faster),
`join-in-judge` and `message-type-classifier` are **new-ability** points (no incumbent). Role
tables (`SCORER_BY_ROLE`, `HANDOFF_FAILURE_BY_ROLE`, `INPUT_FIELDS_BY_ROLE`,
`ANSWER_SPACE_BY_ROLE`) fix a role's scorer, what a failed hand-off call does, which item fields
form its input, and its answer space (`predicate` for guard and join-in; `choice` over the labels
`order`, `question`, `complaint`, `escalation`, `other` for message type).

### 7.1 The decision-point pack

A pack is a decision-point pack iff its `pack.json` has a `decisionPoint` key, whatever its value.
`validate_pack` then also requires (each refusal also fires when the key it reads is deleted;
nothing is defaulted):

- a `labels` block with `source` `llm-unreviewed`, `inputs` `synthetic`, and `labeler` and
  `generator` blocks each with a non-empty string `provider` and `model`; the labeler's model is
  never the `decisionPoint.incumbent` (the incumbent is not graded against itself);
- a role in `BAR_KIND_BY_ROLE`; `decisionPoint.incumbent` present — a model id on a speed-up
  point, `null` on a new-ability point; `decisionPoint.answer` equal to the role's answer space
  (a choice's labels as a set, no repeats);
- `scorer` equal to the role's (`SCORER_BY_ROLE`): a pack cannot pick its own scorer;
- a `stratum` on every item, `eval` or `side-check`, never defaulted; at least 200 `eval` items;
  each `eval` item with `origin` `synthetic` and a non-empty string `genRunId`;
- a `side-check` stratum on `guard-judge` and only there (the human-curated items, exempt from
  `origin` and `genRunId`);
- `metrics.verdictMetrics` exactly the bar kind's set, each once, in any order: a speed-up
  pack's four per-kind rates, a new-ability pack's per-class agreement metrics derived from its
  answer space;
- on a pack scored by `classification`: each `eval` item a bool `expected` and a `path` of
  `understanding` or `turns`; each `side-check` item a `path` and a known `tier`;
- on a new-ability pack: every item carries **every** input field of its role (an empty value is
  written, `[]` or `null`, never omitted — `input_hash` omits an absent field, so omission would
  split one input into two) and each `eval` item's `expected` lies in the answer space.

The environment must declare the `lmstudio-decisions` token for a candidate to run on it, and
`lmstudio-chat` as well for a candidate with a hand-off model (§7.4).

### 7.2 The candidate

`candidates/<candidateId>.json` (format in `FORMATS.md` §5) names a decision model, a typed
question, a **policy** and optionally a hand-off model. `Candidate.load` refuses, as
`CandidateError` (exit `4` from `run`): an unreadable or non-object file; a missing or ill-typed
field; a `candidateId` that is not a plain file name or does not match its file name; a role
outside `BAR_KIND_BY_ROLE`; a `score` question (its answer shape is unprobed); a policy out of
its bounds or of the wrong kind for the question; **a policy that can hand off with no
hand-off model**; a `fit.outcome` or `fit.qualityTarget` outside its closed set. The converse is
allowed: an empty band with a hand-off model hands off only over-limit items.

**Policy** (`apply_policy`, pure and pinned by golden vectors):

- `predicate-band {noBelow: l, yesAtOrAbove: h}` with `0 ≤ l ≤ h ≤ 1`: *yes* iff the probability is
  at least `h`, *no* iff below `l`, hand off iff `l ≤ p < h`; `l == h` never hands off.
- `choice-confidence {minConfidence: c}` with `0 ≤ c ≤ 1`: the chosen label iff `confidence ≥ c`,
  else hand off; `c == 0` never hands off.

The candidate's **content hash** is the SHA-256 of the file's bytes, so an in-place edit is a new
candidate for the reuse count (§7.5).

### 7.3 The two surfaces

A `run --candidate` arm runs on the **`decisions`** surface (no hand-off model) or the
**`handoff`** surface (a hand-off model). Per item, in both: the between-item residency snapshot
(§3.5); the scorer's `decision_input`; the item clock starts; one `decide` call with the
candidate's question at its `decisionTimeoutSeconds`; the policy; then:

- **Decided** — the policy answers: `decidedBy` `decision-model`, the item's `calls` the one
  decision call, `wallClockMs` the item clock.
- **Policy hand-off** (`handoff` surface) — the hand-off chat model scores the item with the
  scorer's `build_messages` at the run's 180 s chat budget (never `decisionTimeoutSeconds`), and
  the scorer's `answer_from_chat` parses the reply; `decidedBy` `handoff`; `calls` holds both
  calls; the clock runs end to end.
- **Over-limit** — the decision call fails with HTTP `500` **and** an error body containing the
  case-sensitive text `too large to process` (`is_over_limit`; any other `500`, or a non-HTTP
  failure, is an ordinary failure). With a hand-off model the item is handed off as a policy
  hand-off with `reason` `over-limit` (no decision call completed, so no `probability` and no
  `inputTokens`); without one it is a kept failure with `reason` `over-limit`.
- **Any other decision-call failure** (timeout, non-marker error) — the item is **kept**, not
  handed off: outcome `fail`, `failedCall` `decision`, `wallClockMs` the clock at the raise.

`withheldFor` is `load` iff the preceding snapshot did not show the decision model resident
(on `handoff`, either model). No gap check runs on these surfaces: a decisions call has no
`stats`.

**A failed hand-off chat call** follows the role (`HANDOFF_FAILURE_BY_ROLE`):

- `withhold` (the guard): withheld exactly as a failed chat call always is — `withheldFor`
  `timeout`/`no_response`, no wall clock, no calls; the item leaves the bar's pairing (§7.5).
- `fail-closed` (the new abilities): kept with `failedCall` `handoff` and scored on the role's
  fail-closed answer, counted in `handoffCallFailures`.

**Metrics the runner adds** to the scorer's aggregates, always exploratory and never in
`verdictMetrics`: on both surfaces `decisionCallFailures` (kept decision failures that are not
over-limit) and `overLimit`, over every item; on `handoff` also `handoffShare` (handed-off items
over every item not withheld by a failed hand-off call), `handoffCallFailures` on fail-closed
roles, and each verdict metric split into `<metric>DecidedByModel` and `<metric>HandedOff`.
`designEffect` is `1.0` on both surfaces. A guard's withheld hand-off record is shaped by the
scorer, not patched by the runner.

### 7.4 Pre-flight refusals (exit `4`, before any call)

For `run --candidate`: the candidate does not load (§7.2) or is for another role; the pack has no
`decisionPoint` or its `environment.requires` lacks `lmstudio-decisions`; the candidate's question
is outside the pack's answer space (the same type and, on a choice, the same label set); the
pack's items give no `evalSetHash` (an item without `stratum`); the fit manifest cannot be read
or does not match its declared `sha256` and entry count; **leakage** (§7.8); the scorer lacks a
decision-surface hook (§3.7); the decision model is absent from the catalog or its catalog `type`
is not `decision`. A candidate with a hand-off model adds: the pack does not also declare
`lmstudio-chat`; the hand-off model is absent from the catalog or its `type` is not `llm`/`vlm`;
on a speed-up point the hand-off model is not the pack's `incumbent`. These checks run for every
candidate, with or without a hand-off model.

After the explicit loads and warm-ups (§3.2), every model of the arm must still be resident
(exit `3`), so a chat warm-up that evicts the decision model refuses the run before any item.
The record carries the decision model's instance `config` verbatim as `decisionInstanceConfig`;
a config that cannot be captured refuses the run, never records `None`.

### 7.5 Bars, pairing and replicates

The report (§9.4) states each bar; nothing here gates an exit code.

**Speed-up point (guard).** The candidate replicate is paired with the **one** incumbent run by
`itemId`:

1. `load` in either arm drops the item pairwise; a chat call withheld (`timeout`, `no_response`)
   in the incumbent run or in a candidate hand-off drops it pairwise, over-limit hand-offs
   included.
2. A guard **decision failure** (a kept `failedCall` `decision`, or a `decisions`-surface
   over-limit item) is a **concordant pair**: on every metric the incumbent declares, the
   candidate's error is the incumbent's, and its paired latency is its elapsed time plus the
   incumbent's. On a `side-check` item it is dropped and counted instead.
3. Any other item, a handed-off over-limit item included, pairs on its own answer.

Drops are counted per arm and reason; eval drops print beside the bars and side-check drops in the
side-check section only. Both arms must hold the same item ids, each once.

- **FR-1 parity**, per verdict cell (the four per-kind rates), is a one-sided exact McNemar test of
  "the candidate's error rate is higher" at α = 0.05, unadjusted; `p ≤ α` makes the cell `worse`.
  The test alone decides; the printed upper bound `U`, the detectable difference and the design
  effect are descriptive and reach limits only. The state is `not assessable (scored n < 200)`
  when the paired scored eval n is below 200, else `not met` if any cell is `worse`, else `met`.
- **FR-2 speed-up** is `R = Σx / Σy` over the paired eval items (`x` the incumbent's wall clock,
  `y` the candidate's), with a seeded paired percentile bootstrap interval printed; the bar holds
  iff the **measured** `R ≥ 2.0` (the interval never decides).
- The pooled false-advance/false-suspend rates and the `side-check` figures are printed as
  exploratory and do not affect pass or fail.

**New-ability point.** One arm, nothing pairs; drops are counted as above (only `load` drops an
item; a kept failed call is already scored fail-closed). **FR-10** needs every class's measured
agreement `k/n ≥ 0.90`; a class with no scored item is `not assessable` and never `met`. The
**mean-latency** bar holds iff the measured mean `ȳ ≤ 350` ms over the scored items. A scored eval
n below 200 is `not assessable` whatever the classes say. A chat arm on a new-ability comparison
is a reference whose results no bar reads, though its timeout and pack content still apply.

**Replicates.** Every run of one candidate in a session is a replicate, ordered by `startedAt`.
Each replicate has a **kind**: `assumed` (its `basis` is `assumed`; it is never counted as a
load), `load` (the group's first non-assumed run, or one whose decision model was not resident at
start), or `same-load` (a later run on a model already resident). The overall state of a bar over
its replicates (`combine_replicates`): replicates marked load-changed are left out; if every
remaining replicate is `met` and **at least two are loads**, `met`; some `met` and some `not met`,
`unstable across replicates — not met`; otherwise any `not met` is `not met`; otherwise
`not assessable`, naming each replicate's reason, or `only N load replicate(s)` when none.

**Reuse count K.** The report prints how many candidates already ran on this evaluation set: the
distinct `candidateContentHash` values among the pack's valid candidate records (every pack
version) with the group's earliest run's `evalSetHash` and a strictly earlier `startedAt`,
excluding the group's own hash. K is computed once per group, and a nonzero K adds a
reuse-adjusted detectable-difference limit. K reads valid records only, so with invalid records
excluded it is a lower bound and the report says so.

**Design effect.** Each family (the four guard cells, or a point's classes) prints one pooled
intra-class correlation over the pack's `genRunId` groups, `design effect: genRunId ICC <±x.xxx>
pooled over <k> cells`, or `not estimable (<reason>) — DE 1.00`; each cell or class takes
`max(1, 1 + (m0 − 1)·ICC)`. The design effect widens limits and `n_effective` only; it never moves
a decision.

### 7.6 The load rule and `basis`

A decision-point result is only comparable if the decision model stayed on the load the
warm-up made. After the last item and before `residentModelsAtEnd`, `runner._within_load_basis`
decides the record's `basis`:

- It probes residency first. If the decision model is absent, any item was `load`-withheld, or
  any between-item snapshot lacked a model of the arm, it makes **no** call and the run is
  **`assumed`** (not shown to be one load).
- Otherwise it **repeats the mandatory warm-up's call** (the fixed warm-up input and the same
  question at `decisionTimeoutSeconds`) and the run is **`by-construction`** iff the answer equals
  the mandatory warm-up's exactly (a predicate's probability; a choice and its confidence).
- A raise, an unequal answer, a probe that could not run (any of its three failure modes) or no
  warm-up answer to compare against is `assumed`. The value is fail-safe: it only ever weakens
  what the run may claim.

A report counts only `load` replicates toward the two loads a `met` needs (§7.5); an `assumed`
replicate's bar state is `not assessable (load changed mid-run)`. Because the runner loads the
arm's absent models itself (§3.2), no scored item is a JIT load.

### 7.7 Input and evaluation-set identity

`input_hash(item, role)` is the SHA-256 of the canonical JSON (keys sorted at every level, no
whitespace, non-ASCII kept raw) of the item's **input fields only** (`INPUT_FIELDS_BY_ROLE`); a
field the item lacks is omitted, never `null`. `eval_set_hash` is the SHA-256 of the sorted lines
`input_hash + ":" + json(expected)` of the `eval`-stratum items, each ending in a newline: the
evaluation set's identity is its content, so item ids, order and the `side-check` stratum do not
enter it and a changed label does. `question_spec_hash` is the same canonical-JSON hash over the
candidate's whole question. These three, `apply_policy` and `is_over_limit` are the **decision
contract**: pure functions transcribed into falkor-chat and decision-models and held equal by
golden vectors (`FORMATS.md` §7), never by a shared library.

### 7.8 Leakage

A candidate's fit manifest lists the `input_hash` of every item used to train, calibrate or tune
it. `run --candidate` refuses (exit `4`, before any call) when **any** pack item, `eval` or
`side-check`, has its `input_hash` in the manifest. The manifest is verified first: it must be
UTF-8, one lowercase-hex SHA-256 per line each ending in `\n` (never `\r\n`), strictly ascending,
its bytes' SHA-256 equal to the candidate's declared `sha256` and its entry count equal to the
declared `size`. The checks are independent: neither the hash nor the count stands in for the
other. A manifest file name is a plain name inside `candidates/`; nothing outside the component is
read.

## 8. Verdict statistics

`modelbench/stats.py` implements `docs/plans/small-model-benchmarking-ml.md` and no other source;
`modelbench/decision_stats.py` implements `docs/plans/typed-decision-models-ml.md` and no other
source. This section states which instrument decides what; the formulas, constants, thresholds and
verdict strings are those notes'.

### 8.1 Instruments

| Instrument | Role |
|---|---|
| Wilson score interval | Descriptive interval of one arm's own rate over the analysis unit; never the comparison instrument. Printed only where the metric's `unit` equals the role's unit kind (§10). |
| Exact two-sided McNemar test | The paired binary decision, on the discordant counts `b` and `c`. |
| MOVER-D (Newcombe) interval | The paired-difference interval on the McNemar path. |
| Exact paired quantiles (closed form) | The paired binary bootstrap's quantiles, with no resampling and no seed. |
| Conservative envelope | The MOVER-D arm and the exact-paired-bootstrap arm, each widened by `√DEFF` about the same point estimate and composed bound by bound: the interval printed whenever the design effect is above 1 or the basis is not `by-construction` (§8.2). |
| Seeded paired cluster bootstrap | Continuous per-unit differences (`mrr`, `separationZ`), responsive to the declared design effect; a single arm's mean interval (`mean_bootstrap_interval`) is descriptive only. The seed is `sampling.seed`. |
| `stats.percentile` | The **only** percentile estimator in the package (Hyndman–Fan type 1); `results.py` imports it rather than defining another. |

### 8.2 The paired binary verdict

Two arms are paired by `pairingKey`; the **analysis unit** is `pairingKey[0]` of the pack and is
never a call-site choice. Units scored by both arms enter the table; the report prints beside
every rate the pairing tally — units scored by one arm only (`asymmetry`), unscoreable in both,
and present in one arm only — so a shrunken `n` is never silent. A unit whose precondition was not
met in either arm is dropped from the pair, never counted as a failure.

`stats.verdict` **refuses rather than warns** unless its preconditions hold (the metric belongs to
the pre-registered family; resolving power describes this table; the MDD is stated at the
family-adjusted α; the design effect is `≥ 1.0`, a NaN included; the Holm step lies between the two
αs). **McNemar decides, and MOVER-D quantifies, only at `designEffect == 1.0` and
`basis == "by-construction"`.** Otherwise McNemar is anti-conservative and may not decide alone:
the printed interval is the conservative envelope (§8.1), a verdict is `distinguishable` only when
that interval excludes zero **and** McNemar's p is at most the step's α (the exact test is a
necessary condition, so the substitute only ever removes rejections), and the `- decided by:` line
names which arm bound each bound. A
comparison's design effect is the larger of its two arms' and its basis the weaker (`assumed` <
`measured` < `by-construction`).

**Rule 7 is enforced inside `verdict()`**: no path returns `distinguishable` for a difference
below the observable floor. On the McNemar path that is a theorem and a violation raises; on the
substitute path the verdict is demoted and the demotion is named.

### 8.3 Family-wise control and resolving power

The pack's `verdictMetrics` are the pre-registered family, `k = len(verdictMetrics)`, and
`ALPHA_FAMILY` (`0.05`, `stats.py`) is the α of the paired verdicts; the decision-point FR-1
cells' unadjusted test has its own constant, `decision_stats.FR1_ALPHA` (`0.05`, §8.5). Those two are
the only numeric `0.05` definitions in `modelbench/`. A homogeneous **binary** family
takes the **two-pass Holm–Bonferroni** flow: every paired table and its McNemar p-value is
computed first, `holm_steps` runs over the family (zipped `strict=True`, so a short ladder cannot
silently drop a metric), then each verdict is decided at its own step; a metric ranked below a
failed step is not tested at all. A homogeneous **continuous** family takes its correction in each
interval at `α/k` and prints no ladder. A family **mixing** binary and continuous members is
refused whole — no member is verdicted and every member prints exploratory — because dropping the
minority kind after the results exist would shrink `k` and under-correct the rest. A metric with
no paired data gets no verdict and stays in the family.

Every printed bound takes the rounding direction, α and denominator that keep **its own** claim
true. `ResolvingPower` carries both αs: the **observable floor** is truncated at the unadjusted
`alpha_family` over the unfloored effective `n`; the **MDD₈₀** is rounded up at `alpha_mdd = α/k`
over the floored one. `stats.format_floor_pp` is the only place the floor's direction lives.
`n_effective` is a `float`, so a raw observation count raises.

### 8.4 Continuous verdicts

`continuous_verdict` decides one continuous metric from per-unit differences: the bootstrap
interval **is** the test (`distinguishable` iff the interval excludes zero), taken at the
family-adjusted level, clamped to the support of the difference (`mrr`: `[-1, 1]`; an unbounded
metric is not clamped), with no resolving power, no Holm step and no McNemar p. One paired unit
yields no verdict (a one-unit interval is a point).

### 8.5 Decision-point statistics

`decision_stats` supplies the bar statistics of §7.5: `p_worse` (the one-sided exact McNemar
p-value), `mdd` (the detectable worsening at 80% power on a 0.1 pp grid), `parity_cell`,
`speedup_ratio`, `mean_latency`, `family_design_effect`, `combine_replicates`, `pair_items` and
`agreement_items`. Its constants (α `0.05`, power `0.80`, 10 000 bootstrap resamples, method
minimums of 50 for a guard cell and 52 for a new-ability class) are printed beside the figures
they bound, and a cell or class below its minimum is **printed, never refused**. It may call
`stats` primitives and **never calls `stats.verdict` or `stats.holm_steps`**: Holm divides α and
Rule 7's floor demotion turns a worsening into a pass, so either would make a bar easier to meet.
Every argument that shapes a decision is required, with no default.

## 9. Reports

A report is markdown; `compare` and `rank` write it to `reports/` and stdout (§2.3, §2.4). A
decision-point `compare` and a plain `compare` differ in what they open with and in the bar
section; every other block is shared.

### 9.1 `compare` blocks, in order

1. `# Comparison — <packId>@<packVersion> (<role>)`; on a decision-point pack, the **labels
   caveat** (labeler and generator, "unreviewed", "agreement with this labeler on synthetic
   inputs, not verified correctness") and then the `## Requirement bar` section (§9.4).
2. `chat-responder` only: the blockquote stating reply quality is not measured.
3. The negative-control banner when requested (and a "requested, not run" banner when fewer than
   two arms reached it).
4. Banners (§9.2).
5. The `tool-caller` funnel table, one per arm.
6. `## Arms`: per arm and metric `k/n`, rate and Wilson interval (a pooled count prints
   without an interval and says why; a zero denominator prints `0/0`, never dropped),
   `DistributionSummary` as median and p10, `ContinuousMetric` as mean; then the descriptive
   note, the lucky-pass markers for `nlq-generator`, and `## Scorer counts`.
7. `## Speed` (`latencyMsP50`, the tail figure, timed/n, withheld counts, medians; descriptive
   only), then, for `tool-caller`, the per-turn-position table and the hazard curve.
8. `## Verdicts`: `Comparison kind: paired, same session | paired, cross-session | unpaired
   (different pack version) | unpaired (same pack version, different content hash)`; each verdict
   metric in `verdictMetrics` order with its text, tally, overlap line and resolving-power line;
   `### Family-wise error control`; the headline (only when the pack declares one — otherwise a
   statement that the metrics are co-equal, no summary line, no arithmetic combining them);
   `### Exploratory metrics`; the overlap footnote. With fewer than two arms, or two deterministic
   arms, `## Verdicts` states why there is none.

### 9.2 Banners and exclusions

- `PACK VERSION MISMATCH` (the runs span versions), `PACK CONTENT HASH MISMATCH` (same version,
  different bytes) and `SCHEMA VERSIONS IN THIS COMPARISON` are printed and **never** drop a
  record.
- `INVALID RESULTS EXCLUDED` names every record that did not enter: an unreadable record, one
  with fingerprint field problems, and any run whose stored aggregates disagree with its own
  items for a verdict-family metric. An unreadable record has no field list and prints its reason
  alone.
- A run excluded for aggregate/item disagreement is removed **before** any banner reads the
  arms; if fewer than two remain, the verdict section says the cause is the records' defect.

### 9.3 `rank`

One ranked table per pack for the headline metric (one per `verdictMetrics` member when the pack
has no headline): rank, model, `k/n` or `n`, rate or mean, 95% interval (Wilson for binary,
percentile bootstrap of the arm's own mean for continuous), latency p95, footprint. Order is
best-first honouring polarity (§6.1); a run with no aggregate for the metric is absent, never
ranked last, and a run whose aggregate has `n = 0` is named in an exclusion banner. The table is
followed by a descriptive note stating the intervals are not a comparison, and a
`<!-- rank-report: ... -->` marker (`FORMATS.md` §8.2). Pack-specific ceiling caveats are
blockquoted above the table. With **`--reference <key>`** the report also renders a
pre-registered Holm–Bonferroni family of verdicts of every other arm against that one stored model
(binary metrics) or the continuous family's interval correction, with the same refusals as §8.3
and a banner when an arm disagrees about a metric's kind; without it, no p-value appears anywhere.
`rank` takes the **newest** run per `armKey`, and refuses to render a duplicate `modelKey` as a
backstop.

### 9.4 The decision-point bar section

`## Requirement bar` begins with a note that the bar is stated and never acted on. It names the
comparison's chat arm (the incumbent on a speed-up point; a reference on a new-ability point) and
then, per candidate:

- `### Candidate <id>` with its fit-manifest line (the input count; zero overlap with this pack's
  items, checked at run time), the reuse count K with the hashes it counted (§7.5), and the fit
  outcome and quality target;
- per replicate `#### replicate <i> — run <runId>, basis <basis>` (a same-load replicate says it
  is not counted as a load), the exploratory counts (`handoffShare`, `decisionCallFailures`,
  `handoffCallFailures` on a new ability, `overLimit`), the scored eval n beside 200 with every
  drop counted, the design-effect line, then the bar's lines: on a speed-up point each FR-1 cell
  (`e1`, `e2`, `b`, `c`, `p_worse`, `U`, the detectable difference, the design effect, method
  minimum and `overLimit`), the FR-2 ratio with its interval, the pooled exploratory rates and the
  side check; on a new-ability point each class's agreement `k/n` with its per-class Wilson
  interval at that class's design effect, method minimum, `overLimit`, and the mean latency
  against 350 ms;
- `#### Across the replicates`: the disagreement lines (final answer and `decidedBy` differing,
  with failed calls and one-sided withholds counted beside them; on a choice point also the
  items with no label in both) and one overall state per bar (§7.5).

A comparison-wide `not assessable (…)` state applies to every bar of every replicate and is
decided before the bar kind: no incumbent on a speed-up point; an incumbent whose aggregates
disagree with its items (speed-up only); a stored `requestTimeoutSeconds` other than 180 on any
arm; or the incumbent's, every replicate's and the loaded pack's content hashes not all equal. The
last wins. A replicate whose own aggregates disagree with its items is not comparison-wide: that
replicate's bars take `not assessable`, the group's overall line (`#### Across the replicates`)
names it, and other replicates are unaffected. An `assumed` replicate keeps
`not assessable (load changed mid-run)` (§7.6) in place of the comparison-wide reason, unless its
own aggregates disagree. These are properties of stored records, so the report **states** them.
They never change the exit code, except that differing timeouts across the arms of a
decision-point `compare` are refused before any report (§2.3). The generic tables of §9.1 cover
the chat arm alone.

## 10. Honesty rules the code enforces

Each is a property a change must preserve; the test suite pins them.

- **Write refuses, read quarantines.** `store()` validates the fingerprint and raises; there is no
  `force` parameter, and its absence is asserted against the API surface. `load_history()`
  re-validates every record against **its own** `benchSchemaVersion` and returns
  `(valid, invalid)`, so a hand-edited record is excluded on read and named.
- **Nothing that shapes a decision carries a default**: `RunResult.designEffect` and `basis`,
  `BinaryMetric.unit`, `PackRef.seed`, `holm_steps`' α, both bootstraps' levels and clamp, and
  `resolving_power`'s inputs are required. A design effect that is not `≥ 1.0` is refused at every
  site that uses it (a NaN included).
- **A Wilson interval prints only over the analysis unit.** A metric whose `unit` is not the
  role's unit kind (a turn- or call-pooled count) prints without one.
- **Holm needs two passes** (§8.3); `ALPHA_FAMILY` is the paired verdicts' α (§8.3); `FR1_ALPHA` is the decision-point cells' (§8.5).
- **Declared outcomes** (§6.1) and the scorer's two duties (§3.7).
- **The analysis unit is pack data**, resolved from `PackRef.analysisUnit`; no call site chooses it.
- **A bar is requirement constants**, never pack data (§7).
- **A `null` fingerprint field fails both tiers**: a field the runner cannot capture refuses the
  run, never records `None` (`FORMATS.md` §3).
- **`ARM_KINDS` is never derived from the forbidden-field mapping.**
- **A public name starting with `test` is collected by pytest as a test**, which is why the
  stored-results query is `models_with_stored_results`.
- **One sentence boundary** for scoring heuristics (§6.5).
- **`sessionId` is a top-level field** of a stored record, never inside the fingerprint.
- **Stored records hold no reply text.**

## 11. Surface index

Every shipped file and where its contract lives. A new shipped file is added here in the change
that adds it.

| File | Contract |
|---|---|
| `modelbench/__init__.py` | `__version__` is stamped as `benchVersion` (`FORMATS.md` §3) |
| `modelbench/__main__.py`, `modelbench/cli.py` | §2 |
| `modelbench/runner.py` | §3, §7.3, §7.4, §7.6 |
| `modelbench/convo.py`, `modelbench/tooling.py` | §3.4; `ToolEnvironment` and `DispatchRecord` are the pack-module seam (`FORMATS.md` §1.4) |
| `modelbench/lmstudio.py` | §4 |
| `modelbench/hostinfo.py` | §2.7, §3.9; `FORMATS.md` §6 |
| `modelbench/packs.py` | §5; `FORMATS.md` §1 |
| `modelbench/roles.py` | §1.4 |
| `modelbench/fingerprint.py` | `FORMATS.md` §3 |
| `modelbench/results.py` | `FORMATS.md` §2, §4; §10 |
| `modelbench/decisions.py` | §7; `FORMATS.md` §5, §7 |
| `modelbench/decision_stats.py` | §7.5, §8.5 |
| `modelbench/stats.py` | §8 |
| `modelbench/report.py` | §9 |
| `modelbench/scoring/__init__.py` (the package marker) and `scoring/retrieval.py`, `classification.py`, `extraction.py`, `grounding.py`, `toolcalls.py`, `agreement.py` | §6.2 to §6.7 |
| `packs/<pack-id>/` (eight shipped packs) | `FORMATS.md` §1 |
| `candidates/*.json`, `candidates/*.fit.txt` | `FORMATS.md` §5 |
| `host.json` (gitignored) | `FORMATS.md` §6 |
| `footprints.json` | `FORMATS.md` §8.1 |
| `results/runs/`, `results/index.csv` | `FORMATS.md` §2, §4; §2.3, §2.4 |
| `setup.sh`, `run.sh`, `run-sweep.sh`, `scripts/*.py`, `pyproject.toml` | §2.10; §1.3 |
| `tests/fixtures/decision-contract/*.jsonl` | `FORMATS.md` §7 |
| `tests/` (suite, `tests/fixtures/{classification,lmstudio,packs}`, `tests/fixtures/metrics_agreement.json`) | test inputs, not a contract; the suite pins §10 |
| `reports/*.md` | generated comparison, rank and consolidated reports (§9; `FORMATS.md` §8.3) |
| `sweep-log-*.log` | a local `run-sweep.sh` log (`FORMATS.md` §8.3), not tracked |
| `.gitignore` | ignores `.venv/`, `host.json`, `results/transcripts/` |
| `REVIEW.md` | the component's review standing instructions, read by reviewers, not a contract |
| `README.md`, `AGENTS.md` | entry and always-loaded context files: pointers and invariants, not a contract (root `AGENTS.md`, context-file convention) |
| `docs/DESIGN.md`, `docs/FORMATS.md` | this specification |
| `docs/BACKLOG.md`, `docs/HISTORY.md` | forward-looking items and the delivery record; not a contract |
| `docs/manuals/` | the end-user manual, not a contract (root `AGENTS.md`, module documentation convention) |
| `docs/requirements/`, `docs/plans/`, `docs/reviews/`, `docs/test-plans/`, `docs/test-reports/` | engineering-process documents (root `AGENTS.md`, module documentation convention); not a contract |
