# Small-LLM benchmarking tool (`model-bench/`) — implementation plan

> **Status:** active · **Owner:** `architect` · **Tracks:** — · **Version:** 1.1

2026-09-02 — revised for the accepted amendments to FR-8(d), FR-15/AC-4, FR-21a and FR-22a
(requirements commit `afe4aef`); nothing in this plan is blocked on a stakeholder answer.

Design for [`../requirements/small-model-benchmarking.md`](../requirements/small-model-benchmarking.md)
(Status: *Ready for design*, 23 FRs / 5 ACs). This plan covers the whole feature: the new top-level
component `model-bench/`, its harness, its task-pack format, and all five roles FR-21 requires at
first delivery — role five, `chat-responder`, honestly partial per FR-21a.

The statistical instruments are **not** designed here. They are settled once in the companion
method note, [`small-model-benchmarking-ml.md`](./small-model-benchmarking-ml.md) (`data-scientist`),
and cited by section from §3.9 below.

**CPG:** used `cpg_falkorchat` — confirmed the blast radius of the "extract falkor-chat's eval
helpers" option before rejecting it (§3.1 D1): every caller of `recall_at_k`, `mrr`,
`check_regression`, `score_pair`, `wilson_interval`, `layer2_contains`, `judge_triple`,
`build_judge_prompt` lives inside `falkor-chat/server/tests/eval/` — 8 callees, 16 caller/callee
pairs, 0 outside that directory. The graph covers `tests/` as well as `falkorchat/` (2 862 vs
1 222 `METHOD` nodes), so that is a real absence, not an unindexed one.

---

## 1. Goal & scope

Build `model-bench/` — a standalone, human-started harness that measures one local model at a time
against one named task pack, stores the result with a full environment fingerprint, and compares it
against previously stored results **for the same role**, with confidence intervals and visible
flags whenever a comparison is not apples-to-apples.

**In scope (first delivery):**

- The component skeleton: `model-bench/` at the repo root, sibling of `falkor-chat/`, with its own
  `docs/` per root `AGENTS.md`'s module-documentation convention.
- The harness core: task-pack loading + versioning, run orchestration, fingerprint capture and
  enforcement, result store, comparison report.
- The LM Studio adapter (inference + embeddings + model catalog + load control).
- **Five task packs**, one per role (FR-21): `tool-caller`, `guard-judge`, `nlq-generator`,
  `chat-responder`, `embedder`.
- Golden data **copied into the tool and versioned by it** (§3.1), including the one genuinely new
  asset: the tool-caller's fixed multi-turn conversation scripts (FR-22).

**Explicitly out of scope** (from the requirements' own out-of-scope section — restated because
each one is a thing an implementer might otherwise build):

- No CI hook, no scheduler, no timer. `model-bench` only ever runs because a person typed a command.
- No pass/fail gate, no non-zero exit on a "bad" score. The only non-zero exits are operational
  (bad arguments, unreachable LM Studio, invalid pack, missing fingerprint).
- No writing to any other component's configuration, ever, and **no run-time code path that reads
  anything outside `model-bench/`** — the one-way `refresh_golden.py` importer (§3.1) is a
  human-invoked maintenance script and is never reachable from a run.
- No global leaderboard and no cross-role aggregate number (FR-20) — enforced structurally in §3.5.
- No reproduction of published benchmark scores.

**Also out of scope for this plan, deliberately:** cloud/hosted model support, and **any LLM judge
at all** — FR-21a defers judged reply quality, so no pack in this delivery contains one (§3.8.5
records the deferred design rather than building it). Both are "deliberately not ruled out" in the
requirements; the pack format in §3.3 leaves room for the first (a pack names a *provider*, and
LM Studio is one) without building it.

---

## 2. Context & findings

### 2.1 What `falkor-chat/server/tests/eval/` actually is

Read in full. 32 files, ~2 400 lines of Python plus the data. Structure:

| Artifact | Size | Nature |
|---|---|---|
| `metrics.py` | 96 lines | **Pure**, zero imports: `recall_at_k`, `mrr`, `check_regression`. |
| `nlq_scoring.py` | 267 lines | **Pure** (stdlib only): `score_pair`, `layer2_contains`, `wilson_interval`. |
| `guard_calibration.py` | 327 lines | Imports `falkorchat.guards.evaluate_guard`. |
| `judge.py` | 159 lines | Imports `falkorchat.llm`. LLM-as-judge, faithfulness/relevance binary. |
| `conftest.py` | 137 lines | Probes `ws:eval` in FalkorDB, reads `config/models.json` for the expected dim. |
| `run_nlq_golden_set_eval.py` | 278 lines | Drives `falkorchat.tools.QueryGraphDataTool` against live FalkorDB. |
| `golden_retrieval.jsonl` | 38 items | `{id, query, relevant_msgIds, topic, target_text, rationale}`. |
| `retrieval_baseline.json` | — | Pinned: recall@10 = 0.9737, recall@5 = 0.8947, MRR = 0.6259, n = 38. |
| `golden_guards.jsonl` | 85 items | tiers: 30 clear_advance / 40 clear_suspend / 15 boundary; 30 `expected:true` / 55 `false`; evidence path 51 understanding / 34 turns. |
| `golden_judge_calibration.jsonl` | 10 items | `{question, context[], answer, expected_faithfulness, expected_relevance}`. |
| `nlq_golden_set.jsonl` | 40 items | 21 catalog / 19 knowledge_base; shapes: 9 single-fact, 8 aggregation, 7 filter-list, 6 not-found, 4 compound-filter, 4 relationship-traversal, 2 conflicting-facts. |
| `corpus_provenance.json` | — | 121 messages, 12 threads, `text-embedding-qwen3-embedding-0.6b`, dim 1024. |
| `judge_calibration.json` | — | n=10, faithfulness agreement 0.90 (κ = 0.83), relevance agreement 0.70 (**κ = 0.21** — recomputed, `-ml` §6.1), `sameModelAsAgentUnderTest: true`. |

Three things this inventory settles:

1. **The golden *data* is portable; the golden *mechanisms* are not.** Every scored item is text
   plus a label. But the retrieval items key relevance on `msgId`s that only exist in the seeded
   `ws:eval` graph, the guard items are scored through `falkorchat.guards`, and the NLQ items are
   scored by executing `QueryGraphDataTool` against a seeded `reference`/`ws:nlq-eval`.
2. **Two of the three source corpora are inline literals; the third is derived.**
   `falkor-chat/scripts/seed_eval_corpus.py` carries the whole 121-message retrieval corpus as a
   `_CORPUS` list (line 80 ff.) and `falkor-chat/scripts/seed_catalog.sh` carries the 15-product
   catalog as a `CATALOG` list of `(name, category, price)` tuples (line ~78 ff.) — both copy out to
   JSON fixtures directly. `falkor-chat/scripts/seed_nlq_eval_corpus.py`'s `_CORPUS` (line 127 ff.)
   is only the *input documents*: the queryable form is produced by falkor-chat's ingestion +
   entity-extraction pipeline, and currently lives in the `ws:nlq-eval` graph (62 `Entity`, 12
   `Document`, 12 `Chunk` — read-only count, 2026-09-02). So that one is copied as a **snapshot of
   the derived rows**, not re-derived (§3.8.3).
3. **The NL-query mechanism does not ask a model for Cypher.** `falkorchat/querygen.py`'s
   `DatasetSchema`/`CATALOG_SCHEMA` plus the prompt in `falkorchat/tools.py` (~line 882) have the
   model emit a **structured JSON query spec** (`[{"property": "name", "op": "=", "value": …}]`)
   which falkor-chat then compiles. That is what makes an in-process, database-free `nlq-generator`
   pack possible (§3.8.3) — worth knowing before assuming a FalkorDB dependency.

### 2.2 The prior tool-calling experiment (FR-22's starting point)

`falkor-chat/docs/reviews/salesperson-tool-reliability-ml.md` §8 is the empirical basis for FR-9.
Its §8.1 documents three fixed conversation scripts — **Condition A** (9 turns, read-only catalog
lookups: exact-name lookup, category filter, price-range filter and repeat, an abstention pair,
then a repeated/rephrased exact-name lookup), **Condition B** (7 turns, write-mutating: add 2×
Wireless Mouse Pro, add 1× Portable SSD 1TB, view cart, remove 1×, remove the SSD entirely, view
cart, add 1× Bluetooth Speaker Mini), **Condition C** (4 turns, read-only, a length-sensitivity
probe) — run at n=40 conversations / 280 turns.

Two consequences:

- **The scripts were never committed** (§8.6: "two throwaway scripts … left in this session's
  scratchpad"), so FR-22's asset genuinely has to be built. But §8.1's prose specifies A and B
  turn by turn, so it is a **reconstruction from a written spec**, not a blank page.
- **It hands the new harness a known-answer validation target.** §8.2 records that
  `qwen/qwen3-4b-2507` does not degrade gradually but collapses near-deterministically at a fixed
  turn position, while `mistralai/ministral-3-3b` behaves differently on the same scripts. A
  freshly built harness that cannot reproduce that contrast is broken. §5 makes this an explicit
  acceptance step rather than a hope.

### 2.3 The LM Studio surface — live-probed on this box, 2026-09-02

Verified by direct calls, not from documentation:

- `GET http://localhost:1234/api/v0/models` returns, per model: `id`, `type`
  (`llm`/`vlm`/`embeddings`), `publisher`, `arch`, `compatibility_type`, `quantization`, `state`,
  `max_context_length`, `capabilities` (e.g. `["tool_use"]`), and `loaded_context_length` for a
  loaded model. 19 models are installed on this box.
- **The requirements' "two catalog ids for the same weights" is visible in that list**:
  `mistralai/ministral-3-3b` and `mistralai_ministral-3-3b-instruct-2512` are separate entries with
  the same `Q8_0` quantization and different `publisher` fields. The fingerprint must record the
  literal key used, never a normalized alias.
- `POST /api/v0/chat/completions` (the LM Studio-specific route, **not** `/v1/`) returns, beside
  the OpenAI-shaped body: `stats{tokens_per_second, time_to_first_token, generation_time,
  stop_reason}`, `model_info{arch, quant, format, context_length}`, `runtime{name, version,
  supported_formats}`. That is FR-11's TTFT and FR-7's runtime identity, for free, per call.
- `lms.exe` **is** reachable from WSL at `/mnt/c/Users/<user>/.lmstudio/bin/lms.exe` (LM Studio runs
  on Windows; the memory note about mirrored networking applies). `lms ps --json` works and returns
  `[]` when nothing is loaded — that is FR-7's "what else was resident". `lms server status --json`
  returns `{"running":true,"port":1234}`. `lms load` exposes `--context-length`, `--gpu`, `--ttl`,
  `--identifier`, `-y`, and `--estimate-only` (resource estimate **without** loading).
- **Two FR-7 fields have no programmatic source** (§6 R-1): `lms version` prints only a *CLI commit*
  (`07b7252`), not the LM Studio application version; and no `lms load` flag or API field exposes
  the **KV-cache setting**, which is a GUI-side load option on this build.

### 2.4 Repo grain the component must follow

- `mcp-monitor/` is the structural template for a small standalone Python component: `pyproject.toml`
  (`requires-python = ">=3.12"`, ruff `select = ["E","F","W","I"]`, `line-length = 100`, pytest
  `testpaths`), an idempotent `setup.sh` creating a component-local `.venv`, a thin `run.sh`,
  `AGENTS.md` + `README.md`, and `docs/{BACKLOG,HISTORY,requirements,plans,reviews,test-plans,test-reports}`.
- `falkor-chat/server/pyproject.toml` supplies the **live-test convention** to copy verbatim:
  `addopts = '-ra -m "not live"'` plus a `live` marker, so the default suite is network-free and
  real-model tests are opt-in via `pytest -m live`.
- `falkor-chat/server/falkorchat/transport.py` reaches the OpenAI-compatible API with **stdlib
  `urllib.request`**, no `httpx` at runtime. `numpy` appears nowhere in the repo.
- `cypher-mcp/` establishes the **content-hash-as-identity** idea (its image tag is a hash of build
  inputs, "so a stale image is unrepresentable"). §3.3 applies the same idea to pack versions.

---

## 3. Design & rationale

### 3.1 D1 — Relationship to `falkor-chat/server/tests/eval/`: **copy the data, clean-build the code**

This is the design constraint the requirements deliberately left open. The decision:

> **Copy the golden *data* into `model-bench`, under `model-bench`'s own versioning and with a
> provenance record naming its origin. Re-implement the *code* — metrics, scoring, judge — inside
> `model-bench` from scratch. Change nothing in `falkor-chat`. Zero imports in either direction.**

**Why not extract-and-generalize** (move `metrics.py`/`nlq_scoring.py` into `model-bench` and have
falkor-chat import them):

- The CPG says the mechanical blast radius is small and contained (§ preamble: 16 call sites, all
  inside `tests/eval/`) — so the objection is not "it's too big to do". It is that the *result* is
  bad: falkor-chat's regression gate would acquire a runtime dependency on a brand-new sibling
  component, meaning `falkor-chat/server/.venv` must have `model-bench` installed for
  `pytest -q` to collect. Root `AGENTS.md` opens with "a monorepo of **independent, self-contained
  components**"; this would make the oldest, most locked-down component depend on the newest and
  least stable, on exactly the axis (statistics) where the new one will churn most.
- The genuinely shared surface is **~40 lines of textbook formulas** (`recall_at_k`, `mrr`,
  `wilson_interval`). Everything else does not transfer: `check_regression` implements a
  zero-tolerance *gate*, which the requirements put explicitly out of scope for this tool;
  `score_pair`/`layer2_contains` are shaped around `QueryGraphDataTool`'s return payload;
  `guard_calibration.py` calls `falkorchat.guards`. Extracting 40 lines is not worth a
  cross-component dependency.
- FR-23 permits the falkor-chat → model-bench direction. It permits it; it does not recommend it.

**Why not a clean build that also re-drafts the data:** the expensive, irreplaceable part of a
golden set is not the text, it is **the human verification of every label** (FR-19). 38 + 85 + 10 +
40 = 173 human-verified items already exist. Re-drafting them would be destroying value to avoid a
duplication that §"the honest answer" below shows is not a duplication at all.

**The duplication risk, answered head-on.** The stakeholder's concern is "two golden sets and two
metric implementations to keep honest". Three points:

1. **The two golden sets *should* diverge, and copying is what makes that safe.** falkor-chat's set
   is a regression gate pinned to *its* corpus and *its* embedding model; it must be updated
   whenever that corpus changes. `model-bench`'s set must **freeze**, because the entire stated
   value of the tool is that "a new model's result lines up against models I tested months ago".
   A shared set would mean any edit made for falkor-chat's benefit silently invalidates every
   stored `model-bench` result. Divergence is the correct behavior, not drift to be prevented.
2. **The metric duplication is discharged by a cross-check test, not by shared code.**
   `model-bench/tests/test_metrics_agreement.py` asserts that `model-bench`'s `recall_at_k`/`mrr`
   reproduce, to 1e-12, the values in the copied `retrieval_baseline.json` when fed the same
   ranked lists. If either implementation drifts, that test fails. That is a stronger honesty
   guarantee than a shared import (which guarantees only that both are wrong together).
3. **Copying is one-way and deliberate.** `model-bench/scripts/refresh_golden.py` (§4 S3) is a
   human-invoked, read-only importer that re-copies from a given `falkor-chat` path, rewrites the
   pack's `PROVENANCE.md` and **forces a pack-version bump**. It is never run automatically, and
   `model-bench` never reads `falkor-chat` at run time — the importer is a maintenance script, not
   a code path of any run.

**What gets copied, and to where:** see the per-pack table in §3.8.

### 3.2 D2 — Component shape and layout

```
model-bench/
  AGENTS.md  README.md  pyproject.toml  setup.sh  run.sh  .gitignore
  docs/{BACKLOG.md,HISTORY.md,requirements/,plans/,reviews/,test-plans/,test-reports/}
  modelbench/
    __init__.py  __main__.py  cli.py
    lmstudio.py        # LM Studio adapter: catalog, chat, embeddings, load/unload, per-call stats
    hostinfo.py        # lms.exe ps/version, host RSS sampling, attestation file
    fingerprint.py     # Fingerprint dataclass, capture(), validate() — FR-7/AC-2
    packs.py           # manifest load, content hash, version compare — FR-5/FR-6/FR-9a
    convo.py           # multi-turn driver + pack-configured prompt assembly — FR-9a
    tooling.py         # simulated-tool protocol, dispatch trace — FR-10
    runner.py          # one run = one model × one pack × n items
    results.py         # RunResult/ItemResult schemas, store, index, quarantine — FR-2/AC-2
    stats.py           # intervals + paired comparison — FR-15/FR-16, per the -ml note
    report.py          # markdown comparison output — FR-3/FR-6/AC-3/AC-4
    scoring/{toolcalls,retrieval,classification,extraction,grounding}.py
  packs/<pack-id>/     # one directory per pack, see §3.3
  results/runs/<runId>.json        # committed
  results/index.csv                # committed, derived, regenerable
  results/transcripts/<runId>.jsonl  # gitignored — raw model output, not needed for comparison
  reports/<pack-id>-<date>.md      # committed
  tests/
```

**Dependencies: none at runtime.** stdlib `urllib.request` for HTTP (falkor-chat's own precedent),
stdlib `json`/`math`/`statistics`/`hashlib`/`subprocess`. Dev extras: `pytest`, `ruff` only. The
largest numeric job is 38 queries × 121 documents × 1024 dims of cosine (~4.7 M multiply-adds),
which is a few seconds of pure Python — acceptable, and the reversal trigger is explicit: **add
`numpy` if a pack's corpus exceeds ~1 000 documents or scoring exceeds ~5 s**. A benchmarking tool
whose own dependency tree can rot is a benchmarking tool whose old results stop being reproducible;
zero runtime dependencies is worth a few seconds.

**Python 3.12**, matching every other component.

### 3.3 D3 — The task pack: a directory, a manifest, and a content hash

A pack is a directory under `model-bench/packs/`. FR-5 requires that adding a scenario not change
the harness; FR-9a requires prompt assembly to be *pack* configuration; FR-6/AC-3 require version
mismatches to be detectable after the fact.

`packs/<pack-id>/pack.json`:

```json
{
  "packId": "tool-caller-shop-assistant",
  "packVersion": "1.0.0",
  "role": "tool-caller",
  "schemaVersion": 1,
  "description": "Multi-turn catalog + cart tool-calling against a simulated storefront.",
  "scorer": "toolcalls",
  "environment": {"requires": ["lmstudio-chat"]},
  "prompt": {
    "systemPrompt": "prompts/system.md",
    "toolSchemas": "tools/schemas.json",
    "historyReplay": "structured",
    "representToolSchemasEachTurn": true,
    "historyTurns": 0,
    "temperature": 0.0,
    "maxTokens": 1024
  },
  "data": {"conversations": "conversations.jsonl", "catalog": "catalog.json"},
  "tools": {"module": "tools/sim.py", "entrypoint": "build_environment"},
  "sampling": {"repeats": 15, "seed": 20260902},
  "provenance": "PROVENANCE.md"
}
```

Key decisions:

- **`prompt` is the FR-9a carrier.** `historyReplay` ∈ `{structured, plaintext, none}` selects how
  prior turns are re-presented (native `assistant`/`tool` message scaffolding vs a flattened text
  transcript); `representToolSchemasEachTurn` and `historyTurns` (0 = unbounded) are separate
  knobs. `convo.py` implements exactly these three axes and reads them from the manifest — nothing
  about prompt shape is hardcoded, and nothing is imported from any product. Reproducing
  falkor-chat's executor shape is then *one pack's settings*, and the settings themselves become a
  testable variable (two packs differing only in `historyReplay` answer "is the replay style what
  breaks at turn 4?").
- **Identity is `(packId, packVersion, contentHash)`.** `contentHash` = SHA-256 over the sorted
  relative paths and bytes of every file in the pack directory except `PROVENANCE.md`. Declared
  versions get forgotten; a hash cannot. `compare` flags a mismatch in **either** (AC-3), which
  also catches the nastier case: same declared version, different bytes. This mirrors
  `cypher-mcp`'s content-hash image tag — a stale pack should be unrepresentable.
- **A pack may ship executable Python** (`tools/sim.py`), loaded via `importlib` from the pack
  directory. That is a deliberate plugin seam, not an accident: FR-10 requires ground truth from a
  dispatched-call trace and resulting state, which needs a real (simulated) tool implementation.
  The alternative — a declarative mini-language for tool behavior — would be a worse, buggier
  Python. Constraint: pack modules import only from stdlib and `modelbench.tooling`; a `ruff` check
  and a review rule enforce it. Pack code is part of the content hash, so a behavior change to a
  simulated tool is a version change like any other.
- **A pack declares its environment.** `environment.requires` lists capability tokens
  (`lmstudio-chat`, `lmstudio-embeddings`). A run against a pack whose requirements are unmet
  fails fast with a named reason. No pack in this delivery requires FalkorDB (§3.8).
- **Packs are versioned in place.** Bump `packVersion`, edit files, hash changes; git history holds
  the old bytes. Recovering an old pack version to re-run it is a `git checkout` of that path —
  documented in `README.md`, not a feature.

### 3.4 D4 — The environment fingerprint, made enforceable (FR-7 / AC-2)

FR-7 says a result missing any part of its fingerprint is invalid and not merged into history. To
make that mechanical rather than aspirational, the fingerprint is a **closed set of required
fields, checked on write and again on read**, split by how each is obtained:

**Auto-captured** (`fingerprint.capture()`, no human input, cannot be wrong without the tool being
wrong): `modelKey` (the literal LM Studio id, never normalized), `modelPublisher`, `arch`,
`quantization`, `compatibilityType`, `maxContextLength`, `loadedContextLength`, `runtimeName`,
`runtimeVersion`, `lmsCliCommit`, `residentModelsAtStart[]`, `residentModelsAtEnd[]` (both from
`lms ps --json`), `temperature`, `maxTokens`, `packId`, `packVersion`, `packContentHash`,
`benchVersion`, `benchSchemaVersion`, `pythonVersion`, `hostOs`, `startedAt`, `endedAt`.

**Operator-attested** (§6 R-1 — no programmatic source exists): `lmStudioAppVersion`,
`kvCacheSetting`, `hostRamGb`, `otherResidentWorkloads`. These live in a local, gitignored
`model-bench/host.json` and are **copied into every run record**, so a record is self-contained and
history stays readable when the box changes.

Three enforcement points, all cheap:

1. **Write refuses.** `results.store(run)` calls `fingerprint.validate()` and raises on any missing,
   empty, or `null` required field. There is no "save anyway" flag.
2. **Read quarantines.** `results.load_history()` re-validates every record and returns
   `(valid, invalid)`. `compare` merges only `valid` and prints an `INVALID RESULTS EXCLUDED` block
   naming each excluded `runId` and the missing field. This is AC-2's actual test surface — a
   hand-edited or older-schema record must be excluded on *read*, not merely rejected on write.
3. **Attestation staleness is detected, not trusted.** `host.json` records the `runtimeVersion` and
   `lmsCliCommit` observed when it was last attested. If either differs at run time, the run stops
   with "LM Studio changed since you last attested `host.json` — re-check the app version and KV
   cache setting, then `model-bench attest`." That converts the weakest link (a human typing a
   value once and never revisiting it) into a loud failure at the moment it goes stale.

### 3.5 D5 — Storage: JSON per run is the truth, CSV and markdown are derived

FR-2 wants durable, human-readable, comparable; the stakeholder's stated preference is "CSV or
markdown, either is fine". A single flat file cannot hold per-turn × per-failure-kind breakdowns
without becoming unreadable, so:

- **`results/runs/<runId>.json` — the record of truth.** One file per run: fingerprint, pack
  identity, per-item and per-turn scored records, aggregate counts, timings. Append-only in
  practice: a run file is never rewritten. `runId` = `<packId>-<modelSlug>-<UTC-timestamp>`.
- **`results/index.csv` — one row per run**, the human-openable summary: runId, date, role, packId,
  packVersion, packContentHash(8), modelKey, quantization, n, headline metric(s), p50/p95 latency,
  valid/invalid. **Derived and fully regenerable** (`model-bench index rebuild`), so it is never a
  second source of truth to keep honest — the same argument §3.1 makes about golden sets, applied
  internally.
- **`reports/<pack-id>-<date>.md` — generated comparisons**, committed because they are the artifact
  a human actually reads six months later. Never hand-edited.
- **`results/transcripts/<runId>.jsonl` — raw model output**, gitignored. Useful for a post-hoc
  "why did it fail", not needed for any comparison, and large.

**FR-20 is enforced structurally, not by convention.** `results.load_history()` takes a `packId`
and there is no API to load across packs; `report.py` has no code path that aggregates two roles;
`compare` requires `--pack`. There is deliberately no `model-bench leaderboard` command, and
`README.md` says why.

### 3.6 D6 — Model access: LM Studio adapter

`lmstudio.py`, stdlib `urllib.request`, four operations:

| Operation | Endpoint / command | Notes |
|---|---|---|
| `catalog()` | `GET /api/v0/models` | Full per-model metadata (§2.3) — the fingerprint's auto half. |
| `chat(messages, tools=…, temperature=…)` | `POST /api/v0/chat/completions` | **`/api/v0`, not `/v1`** — the `stats`/`model_info`/`runtime` objects only exist on the v0 route, and they are FR-11's TTFT and FR-7's runtime identity. |
| `embed(texts)` | `POST /api/v0/embeddings` | Batched; records dimension from the first vector (AC-5). |
| `load` / `unload` / `ps` | `lms.exe` subprocess | `--context-length`, `--gpu`, `--ttl`, `-y`; `ps --json` for residency. |

- **`lms.exe` path resolution** is explicit config (`host.json: lmsPath`), defaulting to a glob of
  `/mnt/c/Users/*/.lmstudio/bin/lms.exe`, with a clear error naming the setting when unresolved.
  Never silently degrade to "no residency data" — that would put a hole in the fingerprint.
- **Cold-load time (FR-11) is measured, not inferred**: `runner` optionally unloads, then times
  `lms load` for the model under test, records `coldLoadSeconds` **once per run**, and excludes the
  first `warmupTurns` requests from the steady-state percentiles. Cold-load and steady-state are
  never averaged together.
- **Peak RAM (FR-11)** — see §6 R-2 for the honesty caveat. Captured as two separate, named fields:
  `modelSizeBytes` (from `lms ps --json`, the loaded weights) and `peakHostRssBytes` (sampled from
  `powershell.exe Get-Process` on an interval during the run, best-effort). Neither is presented as
  "the model's RAM cost" without its method label.
- **`tool_use` capability is checked before a tool-caller run**, from the catalog's `capabilities`
  list, and a model lacking it is refused with a named reason rather than scored 0 — "this model
  cannot do tool calls at all" is not the same measurement as "this model got them wrong".

### 3.7 D7 — Runs, sessions, pairing, and the reference arm (FR-1 / FR-16 / FR-17)

FR-1 (one model per run) and FR-16 (paired, same session) read as a tension. They are reconciled by
separating three ideas that the requirements use in one breath:

- **A run** is one model × one pack × `n` repeats. Always exactly one model (FR-1).
- **A session** (`--session <id>`, recorded in every run record) groups runs executed back to back
  on an undisturbed box. It is a *label with a meaning the tool enforces*: `runner` refuses to
  reuse a session id if `lms ps` shows a different residency profile than the session's first run
  saw, or if more than a configurable gap has elapsed.
- **Pairing is item-level and comes from the pack, not from the clock.** Both models see the same
  items in the same order with the same seed, because the pack pins them. So a paired comparison is
  always *computable*; what a shared session adds is the guarantee that box conditions did not
  change between the arms. `compare` therefore reports **which kind of comparison it is doing** —
  `paired, same session`, `paired, cross-session`, or `unpaired (different pack version)` — and
  never silently mixes them.
- **FR-17's reference arm** is `model-bench run --pack P --model CANDIDATE --reference INCUMBENT`,
  which runs both models in one session, sequentially (**block-level**, not item-interleaved: a
  16 GB box cannot hold two models, and per-item load/unload would dominate runtime). The residual
  is named rather than hidden: within-session drift is bounded and recorded (both arms' timestamps
  and residency snapshots are in the records), not eliminated.
- **FR-17a**: the candidate list offered by `model-bench models --tested` comes from
  `results/index.csv` — the tool's own record — intersected with what LM Studio currently has
  installed. No other component's configuration is read, ever.

### 3.8 D8 — The five packs

Common to all: golden items are copied in as JSONL with a per-item `provenance` object
(`{origin, originPath, originGitSha, copiedAt, draftedBy, verifiedBy, corpusVersion}` — the git SHA
is required, not optional: without it FR-6/FR-19 provenance dead-ends at the copy and no diff
against the origin is ever possible again, `-ml` §5.5) per FR-19, and each pack
carries `PROVENANCE.md` naming the origin file, the origin commit, the copy date, and what was
changed on copy. `model-bench validate` re-checks: unique ids, required fields present, and — for
retrieval-style packs — the FR-19 **paraphrase rule** (a query must not be a verbatim substring of
its own target text), re-implementing what
`falkor-chat/server/tests/eval/test_golden_set_integrity.py::test_query_is_not_verbatim_self_retrieval`
checks today.

#### 3.8.1 `embedder` — pack `embedder-graphrag-retrieval`

- **Data copied:** `golden_retrieval.jsonl` (38 queries → `queries.jsonl`) and the 121-message
  corpus extracted from `falkor-chat/scripts/seed_eval_corpus.py`'s `_CORPUS` literal →
  `corpus.jsonl` (`{docId, text, topic}`; `docId` keeps the original `msgId` so
  `relevant_msgIds` resolve inside the pack). Also copied: `retrieval_baseline.json` and
  `golden_retrieval.embeddings.json` — the latter because it isolates "is my ranking code right"
  from "is my embedding call right" at zero cost (`-ml` §5.4).
- **Mechanism:** embed the 121 documents and 38 queries through `/api/v0/embeddings`, applying the
  pack's per-model `queryPrefix`/`documentPrefix` (FR-14 — a configuration field, never an
  assumption), **L2-normalize every vector and record the raw norm distribution** (`-ml` §5.2 — an
  endpoint returning unnormalized vectors corrupts separation without touching ranking, so it is
  invisible unless measured), then **brute-force exact cosine** in-process. No ANN index, no
  FalkorDB. Rationale: the object of measurement is the *model*, and an approximate index injects
  pipeline noise into a model comparison; exact search also gives the irrelevant-document scores
  that FR-12's score-separation metric needs. The scope boundary is stated in `README.md`: a model
  that wins here has not thereby been shown to win *through* falkor-chat's hybrid ANN pipeline.
- **The embedding cache key is `(model, quantization, docPrefix, corpusVersion)`** and the corpus is
  re-embedded whenever any part changes. Getting this wrong produces a plausible, invalid
  comparison with no visible trace (`-ml` §5.5) — so the cache key is asserted in a unit test, not
  just implemented.
- **Keyword arm (FR-13/AC-5):** BM25 over the same corpus, in-process, on every embedding run —
  reported as a **full paired arm with a confidence interval**, labelled `reference arm
  (deterministic given pack version)`. Tokenization, stopwords, `k1`/`b`, and the always-positive
  IDF variant required at N=121 are specified in `-ml` §5.3 and are pack configuration, not
  hardcoded.
- **Reported (FR-12/FR-14):** recall@k, MRR, **P@1** plus precision@k, and score separation as both
  `sep_raw` (within-model, actionable) and the corpus-sd-normalized `sep_z` (the cross-model
  comparable one), aggregated as median + p10 + fraction > 0 (`-ml` §5.2); plus output dimension,
  RAM cost, embedding throughput (texts/s and tokens/s), max input length and observed truncation
  behavior, and the prefix convention used.
- **`primaryMetric` = MRR**, and the report carries two honesty lines the note requires
  (`-ml` §5.1, §7.4): precision@k is an exact rescaling of recall@k on this set (|R| = 1 for 36 of
  38 items) and is printed for FR-12 compliance with that footnote; and **recall@10 = 37/38 leaves
  one winnable item**, so this pack can detect a materially *worse* embedder but can never certify
  a *better* one on recall. Both are printed in the report, not just documented here.
- **Cost:** low. Existing verified data, no new golden asset. The follow-up that would remove the
  recall ceiling (+22 harder queries, several with |R| ≥ 3) goes to `docs/BACKLOG.md`, not into
  first delivery.

#### 3.8.2 `guard-judge` — pack `guard-judge-understanding`

- **Data copied:** `golden_guards.jsonl` (85 items, distribution in §2.1) → `items.jsonl`.
- **Mechanism:** the pack carries the judge prompt as **text** (`prompts/judge.md`), transcribed
  from `falkorchat/app.py::_LlmGuardJudge`'s prompt construction — data, not an import. One
  single-turn call per item; the reply is parsed with the pack's declared parse mode
  (`ownLineJsonObject`, re-implemented in `modelbench/scoring/classification.py` — the conservative
  reading: an unparseable reply is a **parse failure counted in the denominator**, never a
  fabricated verdict).
- **Reported — class-conditional, and deliberately *not* a pooled 85-item accuracy figure**
  (`-ml` §7.3): false-advance rate on the 40 `clear_suspend` items (`primaryMetric` — it is the
  costly error for a bias-to-suspend judge), false-suspend rate on the 30 `clear_advance` items,
  and the 15 `boundary` items **descriptive only** (n=15 resolves ~53 pp, which is not a
  comparison). Split by evidence path (`understanding` / `turns`) as a diagnostic, plus a
  parse-failure count in the denominator. `report.py` has no code path that emits a pooled
  85-item accuracy number.
- **Cost:** low.

#### 3.8.3 `nlq-generator` — pack `nlq-structured-query`

- **Data copied:** `nlq_golden_set.jsonl` (40 items) → `items.jsonl`; the 15-product catalog from
  `falkor-chat/scripts/seed_catalog.sh`'s `CATALOG` literal, plus a **read-only snapshot** of
  `ws:nlq-eval`'s `Entity`/`Document`/`Chunk` rows (§2.1 finding 2) → `tables.json`; the dataset
  schema from `falkorchat/querygen.py`'s `CATALOG_SCHEMA`/`KNOWLEDGE_BASE_SCHEMA` → `schema.json`;
  the generation prompt from `falkorchat/tools.py` → `prompts/querygen.md`. The snapshot is taken
  once by `refresh_golden.py` via a read-only Cypher read, recorded in `PROVENANCE.md` with its
  date and row counts, and never read again at run time.
- **Mechanism, and why no database is needed:** the model emits a **structured JSON query spec**
  (§2.1 finding 3), not Cypher. The pack ships a small in-process executor (`tools/exec.py`) that
  applies a spec to the in-memory tables, implementing exactly the surface
  `querygen.DatasetSchema` declares — a label, property filters with the declared types and the
  same string→number coercion rule, returns, and the `count`/`avg`/`min`/`max` aggregates. Scoring
  is exact-match after canonicalization against the item's `expected`, with the same numeric
  epsilon and the same scalar/set shape rules `falkor-chat/server/tests/eval/nlq_scoring.py`
  documents (re-implemented, cross-checked in tests against a copied sample of
  `nlq_eval_results.json` records).
- **Answerability must be established at pack-build time, not assumed.** The declared KB schema
  exposes node *properties* only — no relationship types — so the golden set's 4
  `relationship-traversal` items ("Who did Marlowe Robotics acquire?") are not answerable by any
  valid spec against these tables. That is not speculation: falkor-chat's own stored
  `nlq_eval_results.json` scores `nlq-34` incorrect with `{"items": [], "finding": "no matching
  data found"}`. An item no model can get right is a floor, not a discriminator, and silently
  including it deflates every arm equally while making the headline number mean less. So
  `refresh_golden.py` runs a **hand-written reference spec per item** through the executor at copy
  time and stamps each item `answerable: true|false`; `validate` fails a pack that has unstamped
  items; and the report puts unanswerable items in a **separate, named bucket** excluded from the
  accuracy denominator. Keeping them is still worthwhile — they measure whether a model correctly
  abstains instead of fabricating — but as their own count, not as accuracy.
- **Reported:** correct/incorrect split by `shape` (single-fact, filter-list, compound-filter,
  not-found, aggregation, relationship-traversal, conflicting-facts) — the shape split is the
  informative part at n=40; plus malformed-spec and schema-violation counts separately from wrong
  answers, and the abstain-vs-fabricate count on the unanswerable bucket.
- **Cost:** medium — the in-process executor and the reference specs are new work, but both are
  small, pure, and unit-testable against the golden set's own expected values.
- **Scope note:** this measures *structured query generation against a declared schema*. It does
  not measure raw Cypher authorship; nothing in the requirements asks for that, and adding it would
  need a database and a much larger golden set.

#### 3.8.4 `tool-caller` — pack `tool-caller-shop-assistant` (the long pole, FR-22)

- **Data: new, and the main cost of this feature.** `conversations.jsonl` — fixed, versioned,
  multi-turn scripts. Each row:

  ```json
  {"conversationId": "A", "description": "read-only catalog lookups",
   "turns": [{"seq": 1, "user": "...",
              "expect": {"toolRequired": true, "tool": "lookup_product_fact",
                         "args": {"name": "Wireless Charging Pad"},
                         "argChecks": [{"kind": "boundary", "arg": "maxPrice", "value": 50}],
                         "terminal": false,
                         "finalReplyMustContain": ["24.99"]}}],
   "provenance": {"draftedBy": "...", "verifiedBy": "...", "basedOn": "..."}}
  ```

  Three conversation **shapes**, reconstructed from
  `falkor-chat/docs/reviews/salesperson-tool-reliability-ml.md` §8.1 (§2.2): **A** 9 turns
  read-only, **B** 7 turns write-mutating, **C** 4 turns short. Then extended — §8.1's set was
  designed to characterize one defect, and this pack needs coverage of all seven FR-8 failure
  kinds, including "stopping when done" and "final reply matches what the tool returned".
  **Sizing: 4 distinct scripts per shape × 4 replicates = 48 conversations** (`-ml` §7.2), which
  satisfies **FR-22a**. The distinct scripts are not padding: replicating *one* script 15 times, as
  the prior experiment did, produces a confidence interval that describes "this script again", not
  "a script of this kind" — a narrow interval that could move tens of points on a fourth script.
- **Simulated tool environment (`tools/sim.py`)**, deterministic and stateful within a
  conversation: `lookup_product_fact`, `filter_products`, `view_cart`, `add_to_cart`,
  `remove_from_cart`, `clear_cart`, `place_order` — schemas re-declared in the pack as JSON Schema
  (transcribed from `falkorchat/tools.py`, which is where the boundary/unit-translation wording
  that FR-8(d) targets lives). The environment records a **dispatch trace** (every call: name,
  raw arguments, parsed arguments, return value, timestamp) and exposes its **final state** (cart
  contents, orders). FR-10's "system ground truth" is exactly these two — never the model's reply
  text. Reply text is used only for FR-8(g), and only as a containment check against what the tool
  actually returned.
- **Scoring (`scoring/toolcalls.py`)** produces the FR-8 counts **per turn position**, with the
  exact denominators in `-ml` §4.2. Never a single blended percentage; `report.py` has no code path
  that produces one (AC-1 enforced structurally, not by discipline). Four shape changes the note
  requires, all of which the implementer must build rather than infer:
  - FR-8(a) and (b) **collapse into one three-way partition** over the same denominator —
    `native` / `prose_pseudo_call` / `no_attempt` — which satisfies both sub-requirements without
    the harness having to guess whether the model "intended" a call. The prose detector is a
    heuristic, so the pack ships ~20 labelled replies and the report prints **the detector's own
    precision and recall** (`-ml` §4.2).
  - FR-8(d)'s unit is the **call**, not the turn (`n_calls` printed explicitly), and
    `boundary_unit` is a **named subset of `wrong_value`**, not a sibling — as amended FR-8(d) now
    requires, with the **per-argument boundary rule supplied by the pack's tool schemas**
    (`boundaryRule`), never inferred by a regex in the scorer.
  - An **eighth count the requirements omit: restraint** — the rate of correctly *not* calling a
    tool on turns where none was required. Without it a trigger-happy model scores perfectly
    (`-ml` §4.2).
  - **Precondition failures are never silently excluded.** Every rate prints `k/n` inline, every
    count carries an `n/a` tally, and the report **opens with a funnel table**. This is the
    specific mechanism that stops a model which collapses early from scoring *better* on every
    conditional count downstream — the note names it as the harness's most likely way of lying
    (`-ml` §4.3). Paired *n* is the intersection of both arms' scoreable items, printed, with a
    one-model-only `asymmetry` count.
- **Headline and diagnostic (`-ml` §4.6).** `primaryMetric` = **`cleanThroughTurnH`**, the fraction
  of conversations with no failure through turn *H* (*H* = the shortest script length, 4 here):
  one observation per conversation, length-independent by construction, and the statistic that
  would have caught the incumbent model. The **per-turn hazard** — P(first failure at *t* | clean
  through *t*−1) — is the required diagnostic, because it is the only statistic that separates
  "gradual degradation" from "deterministic collapse", which is the entire reason FR-9 exists.
  Turn-pooled rates are never the headline.
- **Prompt assembly** is `convo.py` reading the pack's `prompt` block (§3.3). Turn *n* is built by
  replaying turns 1..*n*−1 in the configured shape; the harness never carries hidden state between
  turns beyond what the configuration says it carries.
- **Validation target (§2.2):** running this pack against `qwen/qwen3-4b-2507` and
  `mistralai/ministral-3-3b` must reproduce the documented per-turn contrast. If it does not, the
  harness — not the models — is what has been measured.

#### 3.8.5 `chat-responder` — pack `chat-responder-grounded-answers`

**Scope: deterministic layer only, per FR-21a** (amended 2026-09-02). Judged reply quality is
deferred, not cancelled — the deferred design is preserved below and in `-ml` §6.2 so it can be
picked up without re-deriving it.

- **Data: new. 30 items** derived from the copied 121-message corpus:
  `{question, context[], referenceAnswer, mustContain[], mustNotContain[], mustAbstain}`, LLM-drafted
  and **verified by a human, item by item** (FR-19). 30, not 20, because the paired floor
  (`-ml` §7.1) means n=20 resolves only ~30 pp. **The 10-item `golden_judge_calibration.jsonl` is
  not copied at first delivery** — it exists only to gate a judge, and there is no judge to gate.
- **Scoring — three deterministic families, no judge anywhere in the pack.** Mapping FR-21a's
  "latency, format, faithfulness to what the tool actually returned" onto a role that calls no
  tools: **latency** is the standard FR-11 block; **format** is reply well-formedness and length
  discipline as the pack declares them; and **grounding** — FR-21a's "faithfulness to what was
  actually returned", here the retrieved context — is `mustContain` / `mustNotContain` /
  `mustAbstain` containment, using the same canonicalization as the `nlq-generator` scorer.
  Grounding is the `primaryMetric`.
- **Cost: moderate, and it is now the *smallest* of the three data-bearing new assets** — 30
  human-verified items, with the expensive half (30 further calibration items plus a judge harness)
  deferred by FR-21a.
- **Deferred design, recorded so it is not re-derived** (backlog, `-ml` §6.1–§6.2): if judged
  quality is funded later, it is **faithfulness only** — the copied calibration record puts
  relevance agreement at κ = 0.21 (raw 0.70, inflated by skewed marginals: the judge called 2 of 3
  gold-irrelevant answers relevant) against κ = 0.83 on faithfulness, and that figure is what
  carried FR-21a. Two rules that must not be softened when it is built: the harness **errors out
  when `judgeModel == candidateModel`** and suppresses every judge-mediated number rather than
  caveating it; and the judge is gated on **class-conditional rates, not κ** —
  `falsePassRate ≤ 2/20` on a **40-item** calibration set, `parseFailureRate ≤ 0.05`. A judge that
  fails the gate produces no numbers.

### 3.9 D9 — Measurement and statistics

Owned by [`small-model-benchmarking-ml.md`](./small-model-benchmarking-ml.md). Formulas, worked
examples, regression fixtures and rationale live **only** there; `modelbench/stats.py` implements
exactly that note and cites it in its module docstring. The five conclusions the rest of this plan
is built on, stated once here so an implementer knows what they are building before opening the
note:

1. **The comparison instrument is the paired difference — now what FR-15/AC-4 require** (amended
   2026-09-02; the amendment is §6 R-9). Decision: **McNemar exact** (conditional binomial-sign) on
   the discordant pairs. Effect size: a **Newcombe MOVER-D** confidence interval on the paired
   difference, derived from the same Wilson function used for per-arm reporting; **AC-4's
   "not distinguishable at this sample size" fires exactly when that interval includes zero.**
   Continuous metrics (MRR, separation, latency) use a seeded **paired percentile bootstrap**,
   B = 10 000. Per-arm Wilson intervals are still printed, labelled *descriptive, not the comparison
   instrument*, and the superseded marginal-overlap check is retained as a **diagnostic line** with
   a footnote saying why it is not the verdict. (`-ml` §3.2; why the old rule could not fire is
   §3.1.)
2. **Every report prints its own resolving power**, computed from its own *n*, never quoted from
   the requirements: `This pack resolves differences of >= X pp with 80% power at n=N (paired).
   Differences below Y pp (= 6/N) cannot reach significance at any observed outcome.`
   (`-ml` §7.1.) `stats.min_detectable_difference` is the one function that must never be
   hardcoded to 15.
3. **Each pack pre-registers exactly one `primaryMetric`** in its manifest; every other number in
   its report is printed with an `exploratory — no significance claim` label. (`-ml` §3.3.)
4. **Per-turn-position rates are computed over conversations, never over turns**; anything pooled
   across turn positions uses a **cluster bootstrap over conversations** and prints its **design
   effect** beside the naive interval. (`-ml` §4.4.)
5. **The negative control is a first-class feature, not a test fixture**: `model-bench compare` run
   with the same model in both arms must report *not distinguishable*, with discordant counts
   roughly equal and the difference interval centred on zero. It is wired as a CLI-reachable mode
   (`--negative-control`) because it is the cheapest way to catch a whole class of harness bugs
   that otherwise present as plausible model differences. (`-ml` §9.)

### 3.10 Alternatives considered and rejected

| Option | Rejected because |
|---|---|
| Extract `metrics.py`/`nlq_scoring.py` into `model-bench`, falkor-chat imports them | Couples a locked regression gate to a new component for ~40 lines of textbook formulas; the two golden sets must diverge anyway (§3.1). |
| Share one golden set between the two components | Any edit for falkor-chat's benefit silently invalidates every stored `model-bench` result — destroys the tool's stated purpose (§3.1). |
| Clean build, re-draft the golden data | Discards 173 human-verified labels, which are the expensive part (FR-19). |
| Score the embedder through FalkorDB's ANN index | Injects index approximation into a *model* comparison, and does not expose the irrelevant-document scores FR-12's score separation needs (§3.8.1). |
| Seed a FalkorDB graph for the `nlq-generator` pack | Unnecessary: the model emits a structured spec, not Cypher, so an in-process executor is exact and dependency-free (§3.8.3). |
| Item-level interleaving of paired arms | A 16 GB box cannot hold two models; per-item load/unload would dominate runtime (§3.7). |
| Store results only as CSV (stakeholder preference read literally) | Cannot represent per-turn × per-failure-kind breakdowns without becoming unreadable; JSON is the truth, CSV is the derived human view (§3.5). |
| The original FR-15 marginal-CI-overlap rule as the decision instrument | Cannot fire at all at n ≤ 40 with baseline ≥ 0.90 — this lab's actual regime — and discards FR-16's pairing (`-ml` §3.1). **FR-15/AC-4 amended 2026-09-02** to the paired-difference interval (§6 R-9); the overlap check survives as a printed diagnostic. |
| Pooling turn-level outcomes into one accuracy rate | Turns within a conversation are not independent; the prior experiment's 280 turns (§2.2) carried roughly the information of its 40 conversations (`-ml` §4.4). Per-turn slices over conversations, cluster bootstrap for anything pooled. |
| `numpy`/`scipy` for the numerics | Zero runtime dependencies keeps old results reproducible years later; the workload is seconds of pure Python. Reversal trigger stated in §3.2. |
| A declarative mini-language for simulated tools instead of pack Python | Would become a worse Python; pack code is content-hashed, so it is versioned data like everything else (§3.3). |

---

## 4. Step-by-step implementation

Eight stages. Each leaves the tree buildable and the suite green, and each has a done-condition that
does not depend on the next. Stages S1–S2 are the harness; S3–S7 are packs, ordered so the cheapest
end-to-end proof lands first and the long pole starts as early as its prerequisites allow.

### S0 — Component skeleton

**Create:** `model-bench/{pyproject.toml,setup.sh,run.sh,README.md,AGENTS.md,.gitignore}`,
`model-bench/docs/{BACKLOG.md,HISTORY.md}` + empty `requirements/ plans/ reviews/ test-plans/ test-reports/`,
`model-bench/modelbench/__init__.py`, `model-bench/tests/`.

- `pyproject.toml`: copy `mcp-monitor/pyproject.toml`'s shape — `requires-python = ">=3.12"`, no
  runtime dependencies, `dev = ["pytest>=9.1,<10", "ruff>=0.14,<0.15"]`, ruff
  `select = ["E","F","W","I"]`, `line-length = 100`; pytest `testpaths = ["tests"]`,
  `addopts = '-ra -m "not live"'` and a `live` marker (falkor-chat's convention, §2.4).
- `setup.sh`: adapt `mcp-monitor/setup.sh` (idempotent, `--recreate`, ends with an import smoke test).
- `.gitignore`: `.venv`, `host.json`, `results/transcripts/`.
- `README.md` states the three non-features up front: no CI, no gate, no leaderboard.
- Root `AGENTS.md` gains a `model-bench/` bullet in **Structure** and a row in **Component docs**;
  `docs/requirements/small-model-benchmarking.md` is left where it is (its own footnote says so).

**Done when:** `model-bench/setup.sh && model-bench/.venv/bin/python -m pytest -q` runs and passes
with zero tests collected, and `ruff check` is clean.

### S1 — Core: fingerprint, results, stats, report (no model calls at all)

**Create:** `modelbench/{fingerprint,results,stats,report,roles}.py`, `modelbench/cli.py`,
`modelbench/__main__.py`.

Key signatures:

```python
# fingerprint.py
REQUIRED_AUTO: tuple[str, ...]; REQUIRED_ATTESTED: tuple[str, ...]
@dataclass(frozen=True)
class Fingerprint:
    ...  # every field in §3.4
    def validate(self) -> list[str]: ...        # names of missing/empty required fields
    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Fingerprint": ...
    def to_dict(self) -> dict[str, Any]: ...

# results.py
@dataclass(frozen=True) class ItemResult: ...
@dataclass(frozen=True) class RunResult:
    runId: str; sessionId: str | None; role: str
    fingerprint: Fingerprint; items: list[ItemResult]; aggregates: dict[str, Any]
def store(run: RunResult, root: Path) -> Path: ...        # raises on invalid fingerprint
def load_history(root: Path, *, packId: str) -> tuple[list[RunResult], list[InvalidRecord]]: ...
def rebuild_index(root: Path) -> Path: ...

# stats.py  — implements docs/plans/small-model-benchmarking-ml.md §3, §4.4, §7.1; no other source
def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]: ...
def mcnemar_exact(b: int, c: int) -> float: ...              # conditional binomial, math.comb
def mover_d_interval(a: int, b: int, c: int, d: int) -> tuple[float, float]:  ...  # Newcombe
def paired_bootstrap(diffs: Sequence[float], *, B: int = 10_000, seed: int) -> tuple[float, float]: ...
def cluster_bootstrap(clusters: Sequence[Sequence[bool]], *, B: int, seed: int) -> BootstrapResult: ...
def design_effect(bootstrap_width: float, naive_width: float) -> float: ...
def min_detectable_difference(n: int) -> float: ...          # ~8/n; never hardcoded to 15
def observable_floor(n: int) -> float: ...                   # 6/n
def verdict(paired: PairedResult) -> str: ...                # the three exact strings, -ml §3.2e

# report.py
def compare_report(runs: Sequence[RunResult], *, pack: PackRef) -> str: ...  # markdown
```

`compare_report` is where AC-2/AC-3/AC-4 become visible output: an excluded-invalid block, a pack
version/hash mismatch banner, the resolving-power line (§3.9 point 2), and the literal phrase
**"not distinguishable at this sample size"** wherever the decision rule says so. It also carries
the disagreement case the note calls out (`-ml` §3.2e): when MOVER-D excludes zero but McNemar does
not, the verdict is *not distinguishable* **and both component outcomes are printed in prose** —
one instrument decides, one quantifies, and they are never AND-ed into a single bloc.

**Done when:** unit tests over hand-built `RunResult` fixtures cover AC-2 (a record with a blanked
attested field is excluded on read and named in the report), AC-3 (two runs differing in
`packVersion`, and separately in `packContentHash` only, both produce the mismatch banner and still
render the comparison), and AC-4 (a paired-difference interval that includes zero renders the "not
distinguishable at this sample size" wording — including the 40/40 vs 34/40 case, where the *old*
marginal-overlap rule would have produced the opposite verdict);
`stats` reproduces the five `(a,b,c,d)` regression fixtures in `-ml` §3.2c exactly; and
`--negative-control` on two copies of the same run reports not-distinguishable with b ≈ c. No LM
Studio involved. This stage encodes the **amended** FR-15/AC-4 decision rule (§3.9 point 1) — the
paired-difference interval, not marginal overlap.

### S2 — Packs, LM Studio adapter, host info, runner

**Create:** `modelbench/{packs,lmstudio,hostinfo,runner,convo,tooling}.py`.

```python
# packs.py
@dataclass(frozen=True) class Pack:
    packId: str; packVersion: str; role: str; contentHash: str
    manifest: Mapping[str, Any]; root: Path
    def data_path(self, key: str) -> Path: ...
    def load_tool_module(self) -> ModuleType: ...   # importlib from pack root
def load_pack(root: Path) -> Pack: ...
def content_hash(root: Path) -> str: ...            # SHA-256, sorted paths, excludes PROVENANCE.md
def validate_pack(pack: Pack) -> list[str]: ...

# lmstudio.py
class LMStudio:
    def catalog(self) -> list[ModelInfo]: ...
    def chat(self, messages, *, model, tools=None, temperature, max_tokens) -> ChatResult: ...
    def embed(self, texts: Sequence[str], *, model) -> EmbedResult: ...
    def load(self, model: str, *, context_length: int | None, gpu: str | None) -> LoadResult: ...
    def unload(self, model: str) -> None: ...
    def ps(self) -> list[ResidentModel: ...]

# tooling.py
class ToolEnvironment(Protocol):
    def schemas(self) -> list[dict]: ...
    def dispatch(self, name: str, arguments: Mapping[str, Any]) -> Any: ...
    def trace(self) -> list[DispatchRecord]: ...
    def state(self) -> dict[str, Any]: ...

# convo.py
def assemble(turn_index: int, history: Sequence[Turn], cfg: PromptConfig) -> list[ChatMessage]: ...
def drive(env: ToolEnvironment, script: Conversation, llm, cfg: PromptConfig) -> ConversationTrace: ...
```

`ChatResult` carries `stats` (ttft, generation_time, tokens_per_second, stop_reason),
`model_info`, `runtime`, `usage`, and the parsed native `tool_calls` — plus a
`toolCallForm` field distinguishing **native tool-call** from **prose that looks like a call**,
which is FR-8(b) and must be decided at the transport boundary where the evidence is, not later.

**Done when:** `packs`/`convo`/`tooling` are unit-tested offline against a stub LLM and a fixture
pack; `lmstudio` is unit-tested against recorded JSON payloads; and **one** `-m live` test confirms
`catalog()` returns the real installed models and that `chat()` surfaces `stats.time_to_first_token`.

### S3 — `embedder` pack + `refresh_golden.py` (first end-to-end result)

**Create:** `model-bench/scripts/refresh_golden.py` (one-way, human-invoked importer),
`packs/embedder-graphrag-retrieval/` (§3.8.1), `modelbench/scoring/retrieval.py`.

Sequenced third on purpose: it is the cheapest path to a complete real run — the golden data
exists, the mechanism is pure arithmetic, and there is a pinned prior figure to sanity-check
against. It proves the S1/S2 core end to end before the expensive packs are built on it.

**Done when:** a real run against `text-embedding-qwen3-embedding-0.6b` produces a stored result
with a complete fingerprint, `compare` renders it against the BM25 arm, and
`test_metrics_agreement.py` passes (§3.1 point 2 — the note's stronger form: run
`falkor-chat`'s own `test_metrics.py` fixtures through the copied implementation and require
byte-identical output). The copied baseline is used as a **harness self-check, never a gate**: the
same model on the same corpus and queries should land ≥ ~0.85 recall@10, and anything below that
means a wrong prefix, unnormalized vectors, or a truncated corpus (`-ml` §5.4). It is not a metric
target — hybrid-ANN vs exact-vector-only differ in two directions at once, so a *disagreement* in
either direction is uninterpretable and must not be "explained".

### S4 — `guard-judge` and `nlq-generator` packs

**Create:** `packs/guard-judge-understanding/`, `packs/nlq-structured-query/`,
`modelbench/scoring/{classification,extraction}.py`, the pack-local `tools/exec.py` spec executor.

Both are single-call-per-item roles that reuse the S1 statistics unchanged; grouping them keeps the
new surface to two scorers plus one small executor. Do `guard-judge` first (no executor at all).

**Done when:** both packs run end to end against one model; **every item is stamped
`answerable: true|false`** by running its hand-written reference spec through the executor
(§3.8.3 — this tests the executor, not the model, and is the step that keeps unanswerable items out
of the accuracy denominator); `validate` passes on both packs; and the guard-judge report shows the
three class-conditional rates with no pooled 85-item figure anywhere in it.

### S5 — `tool-caller` pack, part 1: environment and scoring

**Create:** `packs/tool-caller-shop-assistant/{pack.json,catalog.json,tools/sim.py,tools/schemas.json,prompts/system.md}`,
`modelbench/scoring/toolcalls.py`.

Build the simulated storefront and the per-turn scorer **before** the conversation scripts, and test
both against hand-built synthetic traces (a trace where the model called nothing, emitted a
prose pseudo-call, called the wrong tool, omitted a required argument, mis-translated a boundary,
duplicated a call within a turn, re-issued one across turns, kept going after done, contradicted
the tool result, and called a tool when none was required). This is the piece where a scoring bug is
invisible and expensive, and synthetic traces are the only way to test it deterministically.

**Done when:** each FR-8 count (including the restraint count and the three-way call-form partition)
has at least one synthetic trace that moves it and one that does not; denominators and `n/a` tallies
behave per `-ml` §4.2–§4.3, verified by test, including the laundering case — a model that collapses
at turn 2 must not score *better* than one that reaches turn 8; the funnel table renders; and
`report.py` renders both the per-turn-position table and the hazard curve. Per amended FR-8(d),
one of those synthetic traces asserts that a `boundary_unit` error increments **both**
`wrong_value` and its `boundary_unit` subset, driven by the pack's declared `boundaryRule` — never
double-counted as two sibling failures, never classified by scorer heuristics.

### S6 — `tool-caller` pack, part 2: the conversation scripts (FR-22)

**Create:** `packs/tool-caller-shop-assistant/conversations.jsonl` + `PROVENANCE.md`.

1. Reconstruct shapes A / B / C from
   `falkor-chat/docs/reviews/salesperson-tool-reliability-ml.md` §8.1 (§2.2), turn by turn.
2. Extend to **4 distinct scripts per shape** — FR-22a's floor, sized in §3.8.4 — and cover the FR-8 failure
   kinds §8.1's set was not designed to exercise — in particular "stopping when done", "final
   reply matches the tool result", and turns where **no** tool is required (the restraint count).
   Also draft the ~20 labelled replies the prose-vs-native detector is scored against.
3. **Human verification of every turn's expectations** (FR-19): each `expect` block is checked by a
   person against the simulated environment's actual behavior, not against a model's output.
   `provenance.verifiedBy` is filled per conversation.
4. Run the known-answer validation: `qwen/qwen3-4b-2507` vs `mistralai/ministral-3-3b`, and compare
   the per-turn profile to §8.2's recorded finding.

**Done when:** step 4 reproduces the documented contrast, AC-1 holds on real output (per-failure-kind
**and** per-turn-position, with no blended headline anywhere in the report), and the pack validates.

**This is the long pole.** It is last among the data-bearing stages not because it is least
important but because it is the only one whose *scoring* correctness cannot be checked against
anything except itself — S5's synthetic traces have to exist first, or a script defect and a scorer
defect are indistinguishable.

### S7 — `chat-responder` pack

**Create:** `packs/chat-responder-grounded-answers/`, `modelbench/scoring/grounding.py`.

**Deterministic layer only (FR-21a).** No judge, no calibration set, no `judged.py` — the deferred
design stays in §3.8.5 and `-ml` §6.2 until it is funded. Ungated: the amendment removed the
decision this stage used to wait on, so S7 could now run any time after S1, and stays last only
because it is the least valuable of the five packs until the judged half exists.

**Done when:** the 30 items are human-verified and stamped with provenance; the grounding scorer
covers containment, exclusion and correct abstention; the report prints grounding as
`primaryMetric` alongside the standard latency block; and the report says in words that reply
*quality* is not measured by this pack.

### S8 — Documentation and close

`model-bench/README.md` (how to run, what a pack is, how to add one, the three non-features, the
exact-cosine scope note from §3.8.1), `model-bench/AGENTS.md` (working context: layout, live-test
convention, the FR-23 rule stated as a hard rule for future agents, the operator-attested
fingerprint fields and why), `model-bench/docs/HISTORY.md` first entry,
`model-bench/docs/BACKLOG.md` seeded with the two deferred items named in §7 — the judged
reply-quality layer (FR-21a) and the +22 harder retrieval queries — plus anything R-1's S2 probe
left open. Root `AGENTS.md` rows added in
S0 are re-checked against what actually shipped.

---

## 5. Test strategy

The harness is a measuring instrument, so the test strategy has an unusual centre of gravity: **the
tests that matter most are the ones that prove the instrument reports honestly when the data is
bad**, not the ones that prove it reports a number when the data is good.

**Unit (default suite, network-free, `pytest -q`)**

1. `fingerprint.validate()` — one test per required field: blank it, assert it is named.
2. `results.store()` refuses an invalid fingerprint; there is no bypass flag (assert the absence by
   API surface, not by comment).
3. `results.load_history()` quarantines: a hand-edited record with a missing attested field, a
   record with an unknown `schemaVersion`, a truncated JSON file → all three appear in `invalid`,
   none in `valid`. **(AC-2)**
4. `packs.content_hash()` — stable across path order, changes when any byte in any pack file
   changes, unchanged when `PROVENANCE.md` changes.
5. `compare_report` — version mismatch banner on differing `packVersion`; **also** on identical
   `packVersion` with differing `contentHash`; comparison still rendered in both cases. **(AC-3)**
6. `stats` — Wilson against published worked examples; `mcnemar_exact` + `mover_d_interval` against
   the five `(a,b,c,d)` regression fixtures in `-ml` §3.2c; the McNemar floor (no result at
   c=0, b=5 reaches α=0.05; b=6 does); a paired-difference interval containing zero produces the
   "not distinguishable at this sample size" wording, and the 40/40 vs 34/40 case does **not** —
   the regression test that pins the amended rule; the MOVER-D-excludes-zero-but-McNemar-does-not case renders the
   both-components prose. `min_detectable_difference` and `observable_floor` are computed from *n*,
   asserted never to return a constant. **(AC-4)**
7. `scoring/toolcalls` — the synthetic-trace matrix from S5, one per failure kind, plus denominator
   edge cases (turn where no tool was required — the restraint count; turn where the model emitted
   nothing at all; conversation that ended early) and the **laundering test**: a trace that
   collapses at turn 2 must not out-score one that reaches turn 8 on any conditional count.
7b. `cluster_bootstrap` / `design_effect` — a fixture where all conversations within a script are
   identical must produce a design effect far above 1, and independent observations one near 1.
8. `scoring/retrieval` — recall@k/MRR/precision@k/score-separation on hand-built ranked lists with
   known answers, including multi-relevant items, fewer than *k* results, and zero relevant found.
9. `scoring/extraction` — the scalar/set shape rules and numeric epsilon.
10. `convo.assemble` — the three `historyReplay` modes produce the documented message sequences;
    `representToolSchemasEachTurn=false` really does drop the schemas after turn 1.
11. `test_metrics_agreement.py` — §3.1 point 2.
12. **Pack integrity, per pack:** unique ids, required fields, per-item provenance present,
    paraphrase rule for retrieval-style packs, and every `expect` block referring to a tool the
    pack's own `schemas.json` declares.

**Integration (`-m live`, opt-in, real LM Studio)**

13. `catalog()` returns installed models with `quantization`/`max_context_length` populated.
14. `chat()` surfaces `stats.time_to_first_token` and a native `tool_calls` array on a
    tool-capable model, and `toolCallForm` correctly distinguishes native from prose.
15. `load`/`ps`/`unload` round-trip: after `load`, `ps --json` names the model; `coldLoadSeconds` is
    recorded; after `unload`, `ps --json` no longer names it.
16. One full run per pack, end to end, producing a stored, valid result.

**Acceptance (human-run, once, recorded in `docs/test-reports/`)**

17. **AC-1** on real output: a tool-caller report shows per-failure-kind **and** per-turn-position,
    and no blended "tool-calling accuracy" figure appears anywhere in it.
18. **AC-5**: an embedding report shows the keyword-only arm and the model's output dimension.
19. **The instrument-validation pair.** (a) **Negative control:** the same model in both arms, on
    the tool-caller pack, must report *not distinguishable*, with b ≈ c and the difference interval
    centred on zero — the note names this the highest-value single test in the harness, because it
    catches a whole class of bugs that would otherwise present as plausible model differences
    (`-ml` §9). (b) **Known-answer:** `qwen/qwen3-4b-2507` vs `mistralai/ministral-3-3b` on the same
    pack reproduces the per-turn contrast recorded in
    `falkor-chat/docs/reviews/salesperson-tool-reliability-ml.md` §8.2. Together these are the only
    tests that distinguish "the harness works" from "the harness produces plausible-looking
    numbers"; (a) must pass before (b) is interpreted at all.
20. **The FR-23 audit:** `grep` the shipped tree for any path reference outside `model-bench/` in
    runtime code (`scripts/refresh_golden.py` is the one permitted exception and is not on any run
    path), and confirm `model-bench` runs correctly with `falkor-chat/` renamed away.

If this plan is executed by `tdd-engineer`, tests 1–12 are the red→green sequence, in that order;
13–16 follow the implementation they cover; 17–20 are `qa-engineer`'s acceptance pass.

---

## 6. Risks, open questions, and requirement frictions

**R-1 — Two FR-7 fingerprint fields have no programmatic source (medium; resolved in design, worth
the stakeholder knowing).** `lms version` yields only a CLI commit hash, and neither the REST API
nor `lms load` exposes the **KV-cache setting** on this build (§2.3, live-probed). `lmStudioAppVersion`
and `kvCacheSetting` are therefore **operator-attested** in `host.json` (§3.4), not measured. FR-7's
invalidity rule still holds mechanically — a missing value is refused — but a *wrong* attested value
is undetectable. Mitigations built in: the staleness trip-wire (§3.4 point 3) catches the common
failure of attesting once and never revisiting; `runtime.version` is auto-captured from every call
and is arguably the more reproducibility-relevant version anyway. **Verify during S2:** if
`lms ps --json` on a *loaded* model turns out to expose the load configuration (untestable without
loading a model on a shared box), move `kvCacheSetting` from attested to auto-captured.

**R-2 — FR-11's "peak RAM at the measured settings" is best-effort, not exact (medium).** There is
no RAM endpoint. What is available: `lms ps --json`'s loaded-weights size, `lms load --estimate-only`'s
pre-load estimate, and Windows-side process working set sampled through `powershell.exe`. The design
records them as three separately named fields rather than one authoritative "peak RAM" (§3.6), and
`README.md` states the method. A cross-WSL `powershell.exe` sample is also the most fragile thing in
the tool; it degrades to "not captured" without failing the run — **but note that makes it the one
FR-7-adjacent field that is not enforceable**, which is why it lives in FR-11's speed result rather
than in the fingerprint.

**R-3 — Reconstructed conversation scripts are a reconstruction (medium).** FR-22's asset is being
rebuilt from a prose description (§2.2) because the originals were never committed. The reconstruction
will not be turn-for-turn identical to what produced §8.2's numbers, so the known-answer validation
(§5 test 19) is a **qualitative** check — "does the same contrast appear" — not a numeric
reproduction. If the contrast fails to appear, the ambiguity between "the harness is wrong" and "the
scripts differ" is real; the fallback is to bisect by making the pack's prompt-assembly settings
match falkor-chat's executor shape as closely as the manifest allows (which is exactly the FR-9a
side benefit the requirements anticipate).

**R-4 — `chat-responder` measures grounding, not quality (resolved by FR-21a; residual is a
reader-expectation risk).** Raised as a funding decision — 30 golden items plus 30 calibration
items and a judge harness, against a judge whose measured relevance agreement was κ = 0.21
(§2.1, `-ml` §6.1) — and **settled: FR-21a ships the deterministic layer only** and defers judged
reply quality. §3.8.5 builds that and records the deferred design. The residual risk is that a
reader six months from now sees a `chat-responder` score and reads it as "reply quality", which it
is not; the mitigation is in the pack's own report text (S7's done-condition) and in
`docs/BACKLOG.md`, which carries the deferred judged layer as an open item rather than letting it
disappear.

**R-5 — Copied golden data can silently go stale relative to its origin (low, accepted).** By
design (§3.1 point 1) the copies freeze. The accepted consequence is that a fix made to
falkor-chat's golden set — a mislabelled item, say — does not reach `model-bench` until someone runs
`refresh_golden.py`, which forces a pack version bump and thus flags every prior result as
version-mismatched (AC-3). That is the correct trade: comparability over time is the tool's purpose,
and a version bump is loud. `PROVENANCE.md` records the origin commit so a diff is always possible.

**R-6 — Pack Python is a plugin seam (low).** Packs execute code (`tools/sim.py`, `tools/exec.py`).
This tool runs locally, on packs written in this repo, started by a person — so the threat model is
"a mistake", not "an attacker". Mitigations: pack modules may import only stdlib and
`modelbench.tooling`; pack code is inside the content hash; `validate` reports a pack whose module
imports outside that set. A stricter sandbox would be disproportionate.

**R-7 — Item-level pairing across sessions is weaker than it looks (low, disclosed in output).** The
pack pins items and order, so a cross-session paired comparison is arithmetically valid, but the box
is shared and its state is not controlled between sessions. §3.7's report label
(`paired, same session` / `paired, cross-session`) is the disclosure; FR-17's reference arm is the
remedy the tool *offers*, and the report says so wherever it prints a cross-session comparison.

**R-8 — Two LM Studio catalog ids for the same weights (low, designed for).** Confirmed on this box
(§2.3). The fingerprint records the literal key and never normalizes, so two runs of "the same
model" under different ids compare as two models — visibly, with both ids printed. That is the
correct conservative behavior; a `notes` field lets a human record that they are the same weights.

### Requirement defects raised by this design pass — all amended and closed

Three requirements were unbuildable as written. All three amendments were accepted on 2026-09-02
and are in `docs/requirements/small-model-benchmarking.md` (commit `afe4aef`). They are kept here
because the *reason* each was wrong is what an implementer needs when they meet the code — the
requirement now states the rule, not why the earlier one failed.

- **R-9 — FR-15/AC-4's marginal-overlap rule could never fire. Amended.** At n ≤ 40 with a baseline
  ≥ 0.90, **no result whatsoever separates two marginal Wilson intervals** (`-ml` §3.1), and the
  rule discarded exactly the covariance FR-16's paired design pays for. Worked case: 40/40 vs
  34/40, perfectly nested, marginal intervals overlap → old rule prints "not distinguishable", while
  McNemar exact gives p = 0.031 and the paired difference is +15.0 pp, 95% CI [3.2, 29.1]. FR-15 and
  AC-4 now read the interval as the one on the **paired difference**; §3.9 point 1 is the build.
  The superseded overlap check remains a printed diagnostic.
- **R-10 — FR-8(d)'s three-way split was not a partition. Amended.** *boundary/unit translation* is
  a **subset** of *wrong value*, not a sibling; coding them disjoint forces double-counting or an
  arbitrary priority rule, and two runs under different priority rules stop being comparable
  (`-ml` §8). FR-8(d) now states the nesting, and the per-argument boundary rule comes from pack
  data (`boundaryRule` in the pack's tool schemas) rather than a regex in the scorer — §3.8.4.
- **R-11 — FR-9/FR-22 did not require distinct scripts per shape. Added as FR-22a.** Replicates of
  one script yield an interval describing "this script again", not "a script of this kind"
  (`-ml` §4.5). §3.8.4's 4 scripts × 4 replicates × 3 shapes already satisfies it.

**Nothing in this plan is now blocked on a stakeholder answer.** The remaining open item is a
verify-during-implementation check flagged in place: R-1's S2 probe of whether `lms ps --json`
exposes the KV-cache setting on a loaded model.

---

## 7. Ready to implement

**Plan:** `docs/plans/small-model-benchmarking.md` (this document).
**Method note:** `docs/plans/small-model-benchmarking-ml.md` (`data-scientist`) — statistics,
denominators, metric definitions, sample sizes, and the deferred judge design.

New standalone component `model-bench/`, zero runtime dependencies, Python 3.12, eight stages:
S0 skeleton → S1 core (fingerprint/results/stats/report, no model calls; delivers AC-2/AC-3/AC-4) →
S2 LM Studio adapter + pack loader + conversation driver → S3 `embedder` pack (first real
end-to-end run; AC-5) → S4 `guard-judge` + `nlq-generator` packs → S5 tool-caller simulated
environment + per-turn scorer (synthetic traces) → S6 the FR-22 conversation scripts and the
known-answer validation (AC-1) → S7 `chat-responder` (deterministic layer only, FR-21a) → S8 docs.

The eval-harness question is resolved as **copy the golden data, clean-build the code, change
nothing in falkor-chat** (§3.1), with the duplication risk discharged by a numeric agreement test
rather than by shared code — and on the observation that the two golden sets *must* diverge, because
one tracks a live corpus and the other must freeze for results to stay comparable.

**Nothing is blocked.** The four items this design pass raised were all settled on 2026-09-02
(requirements commit `afe4aef`): FR-15/AC-4 amended to the paired-difference interval, FR-8(d)'s
`boundary/unit` restated as a labelled subset of `wrong_value` with the rule supplied by pack data,
FR-21a scoping `chat-responder` to its deterministic layer with judged quality deferred, and FR-22a
requiring several distinct scripts per conversation shape. §6 keeps the reasoning; the plan is built
to the amended text throughout.

Two items carry forward into `model-bench/docs/BACKLOG.md` at S8 rather than into this delivery:
the **deferred judged-quality layer** for `chat-responder` (design preserved in §3.8.5 and
`-ml` §6.2) and the **+22 harder retrieval queries** that would lift the embedder pack's recall
ceiling (§3.8.1).
