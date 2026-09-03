# Small-LLM benchmarking tool (`model-bench/`) — implementation plan

> **Status:** active · **Owner:** `architect` · **Tracks:** — · **Version:** 1.5 · **Reviews:** `docs/reviews/small-model-benchmarking.md` · `docs/reviews/small-model-benchmarking-impl.md`

2026-09-03 — v1.5: the S1 implementation gate's plan-side corrections (`docs/reviews/small-model-benchmarking-impl.md` §4) — §3.4.1's forbidden-field list becomes a derivation over §3.4.2's now-complete field set, Appendix A's `PackRef` and `FieldProblem` catch up with §3.3 and the shipped code, §4 S1 records `RunResult.designEffect`/`basis` as required-with-no-default, §6 opens R-13 (`_percentile`'s definition), and §7 gains the intra-document half of its staleness rule plus S1's delivery status. (v1.4 closed Pass 2's N-1/N-2/N-4 and narrowed §7's precedence rule; v1.3 aligned the vocabulary to the `-ml` note; v1.2 closed the Pass 1 gate.)

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
1 222 `METHOD` nodes), so that is a real absence, not an unindexed one. **The graph is stale**
(built 2026-09-02T12:38:21Z at `4bb96e1`, `SOURCE_DIRTY = true`, several `falkor-chat/server`
commits since), but the claim is directory-scoped and the `analyst` gate independently re-ran it
against the current tree and reproduced it exactly. No CPG exists for `model-bench/` — it is a new
component — so every structural claim about `model-bench` in this plan is a claim about what is to
be built, not a graph answer.

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
2. **The metric duplication is discharged by a transcribed behavioural fixture, and the residual
   is stated rather than papered over.** *(Revised in v1.2. v1.1 claimed a cross-check "against the
   values in `retrieval_baseline.json` when fed the same ranked lists"; the gate established that
   **the ranked lists do not exist as an artifact** — `retrieval_baseline.json` holds four aggregate
   numbers, and the ranked lists behind them are produced live by `services.hybrid_search` against
   the seeded `ws:eval` graph, which FR-23 forbids `model-bench` from touching and which is
   ANN-approximate anyway. That mechanism was not constructible. This is its replacement.)*

   **(a) The fixture.** `model-bench/tests/fixtures/metrics_agreement.json` is a one-time,
   hand-transcribed capture of **every assertion in
   `falkor-chat/server/tests/eval/test_metrics.py` that exercises `recall_at_k` or `mrr`** — read
   from the file, counted: **20 cases (18 value assertions + 2 `ValueError` cases)**, 13 for
   `recall_at_k` and 7 for `mrr`. The count is stated here so an implementer who ships three cannot
   call it done. Per case:

   ```json
   {"function": "recall_at_k", "case": "hit_outside_top_k_window",
    "args": {"retrieved": ["x","y","z","w","v","a"], "relevant": ["a"], "k": 5},
    "expected": 0.0}
   ```

   with `"expectedError": "ValueError"` in place of `expected` for the two raise cases. File header:
   `sourcePath` (repo-root-relative), `sourceGitSha`
   (`9650a3858b9d5c4e7e934f977839fc1a61c84b1b` at the time of writing), `sourceSha256` of the origin
   file's bytes, `copiedAt`, `transcribedBy`, `verifiedBy`, and an `excluded` list. Transcription is
   **manual, not extracted**: only 6 of the 20 cases live in `@pytest.mark.parametrize` tables; the
   other 14 are literals inside individual test bodies, so a mechanical extractor would silently
   capture a third of the surface and pass. `check_regression`'s 6 cases go in `excluded` with their
   reason — `model-bench` deliberately does not implement a regression gate (the requirements put
   zero-tolerance gating out of scope), so there is nothing to agree with.

   **(b) The test.** `model-bench/tests/test_metrics_agreement.py` runs `model-bench`'s
   implementation over every non-excluded case and requires equality to within `1e-12` absolute
   (the values are small exact binary fractions plus `1/2` and `1/3`, which both implementations
   reach by the same division), and `pytest.raises` for the two error cases. It reads only
   `model-bench/tests/fixtures/`, so the default suite still passes with `falkor-chat/` renamed away
   (§5 test 20).

   **(c) The drift detector lives on the maintenance path, not in the suite.**
   `scripts/refresh_golden.py --check-origins` re-reads and re-hashes every recorded origin file,
   including `test_metrics.py`, and reports each one as unchanged or drifted against its
   `sourceSha256`. It is the one component that may read `falkor-chat` (§1 out-of-scope, third
   bullet), it is human-invoked, and it is never on a run path.

   **The honest residual.** This proves behavioural agreement **on the cases falkor-chat itself
   tests, as of the transcription** — not on untested inputs, and not automatically when the origin
   moves. That is a weaker guarantee than v1.1 claimed, and it is stated here rather than implied
   away. What carries the rest of the weight is **(i)** the shared numeric surface is deliberately
   tiny — `recall_at_k`, `mrr`, `wilson_interval`, three textbook formulas — and each is *separately*
   pinned to a reference outside falkor-chat: `wilson_interval` and the paired instruments to the
   `-ml` note's own worked regression fixtures (§5 test 6), `recall_at_k`/`mrr` to the transcribed
   cases plus their textbook definitions; and **(ii)** the two implementations' outputs are **never
   compared to each other as numbers, by design** — `retrieval_baseline.json`'s pinned figures come
   from a hybrid-ANN pipeline and `-ml` §5.4 already forbids reading a difference against them in
   either direction (§3.8.1, S3's done-condition). So the duplication cost is bounded at "two places
   to fix a formula bug", which a 20-case fixture detects, and never at "two numbers that silently
   disagree in a report".
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
  reports/<pack-id>-<date>-<n>.md  # committed
  tests/
```

**`results/` and `reports/` do not exist at S0.** They are created on first write by S1–S3, so the
S0 tree is the top three lines plus `modelbench/` and `tests/`. S0's `.gitignore` nevertheless names
`results/transcripts/` from the start, deliberately: a `.gitignore` entry for a path that does not
exist yet is inert and free, whereas adding it only when the first transcript lands is one commit
away from committing a megabyte of raw model output by accident.

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
  "sampling": {"scripts": 12, "replicatesPerScript": 1, "seed": 20260902,
               "pairingKey": ["scriptId", "replicate", "turnIndex"],
               "analysisUnit": "scriptId",
               "determinismProbeScripts": ["A-01", "B-03"]},
  "metrics": {
    "verdictMetrics": ["cleanThroughTurnH"],
    "headlineMetric": "cleanThroughTurnH",
    "cleanThroughTurnH": {"H": 4}
  },
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
- **`metrics` pre-registers the verdict family, and a pack may legitimately have no headline.**
  Two separable fields, because they control different things — `-ml` §3.3 states the split and
  this plan owns the names:
  - **`verdictMetrics`** — the closed, pre-registered list (1..k) of metrics that may receive a
    better / not-distinguishable **verdict**. It controls **inference**: its length *is* the
    multiplicity correction's `k`.
  - **`headlineMetric`** — either exactly one member of that list, or an **explicit `null`**. It
    controls **presentation**: what a reader is entitled to read as "the" number.

  Everything not in `verdictMetrics` prints with `exploratory — no significance claim`.
  `validate_pack` fails when `verdictMetrics` is absent or empty, when the `headlineMetric` **key**
  is absent (omission is not the same statement as `null`, and only the latter is a decision), or
  when a non-null `headlineMetric` is not a member of `verdictMetrics`. Three consequences the
  implementer must build rather than infer:

  **(i)** When `headlineMetric` is `null`, `report.py` has **no code path that synthesises a
  headline** from `verdictMetrics` — the same structural refusal FR-20 gets in §3.5 — and the report
  prints the verdict metrics side by side **in the manifest's declared order**, with no summary line
  above them and no arithmetic combining them.

  **(ii)** When `len(verdictMetrics) > 1`, family-wise error control is **mandatory, not optional**
  (`-ml` §3.3): Holm–Bonferroni across the declared family, with the adjusted threshold printed
  beside each p-value. **And the resolving-power line's α changes with it** — a k-member family
  computes at `α/k`, which is materially worse than at α=0.05 and is the honest price of a second
  co-equal verdict. `stats.verdict()` asserts `alpha == 0.05 / len(family)` and refuses otherwise
  (`-ml` §3.4 Rule 4), so a report cannot print a family verdict at the wrong threshold.

  **(iii)** Pre-registration in a content-hashed manifest is what stops a headline being chosen
  after the results exist.

  *(Naming, settled in v1.3 and aligned with the note. The retired field is the singular
  `primaryMetric`, and it is retired rather than redefined: re-pointing an
  established name at new semantics — "may now be `null`" — is its own trap, and pluralising it to
  distinguish the list would leave two fields one character apart in every manifest and every diff.
  `verdictMetrics` names what the field controls and collides with nothing; `headlineMetric` does
  the same for the presentation half.)*
- **`sampling` declares the analysis unit, and the harness never chooses one.** *(New in v1.4,
  closing gate finding N-1.)* `-ml` §3.4 Rule 1 makes `PairedOutcomes.from_units` raise on a repeated
  analysis-unit id — but **that guard only fires if the id passed in is the *cluster* key.** 48
  conversations drawn from 12 scripts have 48 *distinct conversation ids*, so a caller who passes
  conversation ids sees no error and gets an anti-conservative verdict from correlated rows. Rule 1
  is a backstop; the contract that actually closes it is here, in the pack:
  - **`sampling.pairingKey`** — the ordered component names of `ItemResult.pairingKey`
    (`["scriptId", "replicate", "turnIndex"]` for the tool-caller; `["itemId"]` for the four
    item-level packs). The pairing key stops being a scorer convention and becomes pack data.
  - **`sampling.analysisUnit`** — the independent unit of analysis, and it is fixed by a **rule**,
    not chosen per pack: **the analysis unit is the *outermost* component of `pairingKey` — the one
    whose *repetition* is what would indicate correlated rows.** For the tool-caller that is
    **`scriptId`, never a conversation id**; for the four item-level packs it is `itemId`. A rule
    rather than a per-pack list because a list goes stale the moment a pack is added, and because
    the rule is what a reviewer can check by reading. `report.py` resolves the unit id from this
    field and passes it to `from_units`; **no call site chooses it, and there is no parameter
    through which a caller could.** It is one declaration per pack rather than per metric because
    the rule is structural: every verdict metric in a pack shares the same independent unit.
  - **`validate_pack` enforces the rule twice, by two independent routes.**
    **(i) Structural:** `analysisUnit` must equal `pairingKey[0]`, and `pairingKey` is ordered
    outermost → innermost. This catches the pack that declares a correct unit against a
    wrongly-ordered key.
    **(ii) By the data — the check that catches a *consistently* wrong choice, which (i) cannot.**
    The row-count identity, computed over `analysisUnit`'s own values: the data file holds exactly
    `scripts × replicatesPerScript` rows, the number of distinct `analysisUnit` values is exactly
    `scripts`, and **each appears exactly `replicatesPerScript` times.** A pack that declared
    `analysisUnit: "conversationId"` under `scripts: 12, replicatesPerScript: 4` fails immediately —
    48 distinct values where 12 are required — which is precisely the N-1 shortcut, caught by
    arithmetic rather than by intent. It also catches the pack that declares
    `replicatesPerScript: 1` and ships four conversations per script, the case that slips past
    Rule 6's declaration check *and* past Rule 1 at once.
- **Identity is `(packId, packVersion, contentHash)`.** `contentHash` = SHA-256 over the sorted
  relative paths and bytes of every file in the pack directory except `PROVENANCE.md`. Declared
  versions get forgotten; a hash cannot. `compare` flags a mismatch in **either** (AC-3), which
  also catches the nastier case: same declared version, different bytes. This mirrors
  `cypher-mcp`'s content-hash image tag — a stale pack should be unrepresentable.
- **A pack may ship executable Python** (`tools/sim.py`), loaded via `importlib` from the pack
  directory. That is a deliberate plugin seam, not an accident: FR-10 requires ground truth from a
  dispatched-call trace and resulting state, which needs a real (simulated) tool implementation.
  The alternative — a declarative mini-language for tool behavior — would be a worse, buggier
  Python. Constraint: pack modules import only from stdlib and `modelbench.tooling`, enforced by
  **one mechanism, `validate_pack`'s AST walk** over every `Import`/`ImportFrom` node in the pack's
  Python files, checked against an allowlist. *(v1.1 also claimed "a `ruff` check … enforces it";
  that is not constructible under this component's own `select = ["E","F","W","I"]` — banned-import
  rules live in `TID`, which is unselected, and are a denylist rather than an allowlist. The claim
  is withdrawn.)* Two rules that make the AST check real rather than decorative: **`run` calls
  `validate_pack` first and fails closed**, so the check fires on a normal run and not only when
  someone remembers to type `model-bench validate`; and every subprocess launch in the tool
  (`lms.exe`, `powershell.exe`) is `subprocess.run([...], shell=False)` with an argv list — model
  keys reach the argv verbatim (`mistralai/ministral-3-3b`) and a `shell=True` slip is the only
  injection surface in the tool. Pack code is part of the content hash, so a behavior change to a
  simulated tool is a version change like any other.
- **A pack declares its environment.** `environment.requires` lists capability tokens
  (`lmstudio-chat`, `lmstudio-embeddings`). A run against a pack whose requirements are unmet
  fails fast with a named reason. No pack in this delivery requires FalkorDB (§3.8).
- **Packs are versioned in place.** Bump `packVersion`, edit files, hash changes; git history holds
  the old bytes. Recovering an old pack version to re-run it is a `git checkout` of that path —
  documented in `README.md`, not a feature.

### 3.4 D4 — The environment fingerprint, made enforceable (FR-7 / AC-2)

FR-7 says a result missing any part of its fingerprint is invalid and not merged into history. To
make that mechanical rather than aspirational, the fingerprint is a **required-field set that is a
function of `(benchSchemaVersion, armKind)`, checked on write and again on read**. Both keys are
new in v1.2 and each closes a gate finding; take them in order.

#### 3.4.1 `armKind` — a run is not always a model run (B-3)

FR-13/AC-5 require **every** embedding run to carry a keyword-only reference arm, reported as a full
paired arm with an interval (§3.8.1). BM25 has no model, no quantization and no runtime, so as a
`RunResult` it fails a model-shaped validation on write — and §3.4.2's rule is that there is no
"save anyway" flag. The resolution is a **discriminator, not an exemption**:

- `Fingerprint.armKind: "model" | "deterministic"`, required on every record, and
  the required and forbidden sets are both keyed by it — `REQUIRED_BY_SCHEMA[schema][armKind]`
  (§3.4.3 adds the outer key) and `FORBIDDEN_BY_ARM_KIND[armKind]`. `validate()` branches on
  `armKind` and never on field presence.
- **`model`** requires the full auto-captured set below plus the four operator-attested fields.
- **`deterministic`** requires exactly eleven fields: `armId` (`"bm25"`), `armParametersHash`
  (SHA-256 over the pack's declared arm parameters — tokenizer, stopword list, `k1`, `b`, IDF
  variant), `packId`, `packVersion`, `packContentHash`, `benchVersion`, `benchSchemaVersion`,
  `pythonVersion`, `hostOs`, `startedAt`, `endedAt`. (`armKind` is the discriminator itself: present
  on every record and checked before either mapping is consulted, so it is a member of **neither**
  required set and can never appear in a derived forbidden set.)
- **The forbidden set is a derivation, never a list.** *(v1.5 — see the note below.)* For each arm
  kind, at each schema version, the rule is:

  > **`FORBIDDEN_BY_ARM_KIND[k] = required(other kind at that schema) − required(k at that
  > schema)`** — a record may not carry a field that only the *other* arm kind is required to have.

  For `deterministic` that resolves to every model field, the sampling settings, and all four
  operator-attested LM Studio fields — i.e. everything in §3.4.2 that is not one of the eleven
  above; for `model` it resolves to exactly `{armId, armParametersHash}`. The implementer writes the
  set difference (`frozenset(REQUIRED_BY_SCHEMA[v]["model"]) - frozenset(REQUIRED_BY_SCHEMA[v]["deterministic"])`),
  not the resulting names, so adding a model field at schema 2 forbids it on deterministic arms for
  free. The **test** pins the same sets against an independently written literal, so a set that
  *shrinks* fails loudly rather than silently shrinking the suite (S1 impl-gate M-4).

  The forbid half is the point of the whole discriminator: it is what makes
  `{"modelKey": "bm25", "quantization": "n/a"}` — the shortcut a time-pressed implementer reaches
  for, and exactly the "invalid result reaching a report unlabelled" this design exists to prevent —
  fail loudly on write instead of quietly becoming a sixth model in the history.

  *(Why a derivation. Through v1.4 this bullet carried a hand-typed list of fourteen model fields
  beside the words "forbids every model field". The two never agreed: §3.6's eligibility-gate
  decision put `modelType`, `modelCapabilities` and `modelCapabilitiesPresent` into the model set
  (§3.4.2) and nobody swept the list here. The S1 implementer forbade all three anyway, following
  the stated intent, and the implementation gate confirmed the plan was the stale side (§4 item 3).
  Two review passes had certified the list before that — both read it against itself rather than
  against the set it claims to complement, which is what a hand-maintained enumeration invites. A
  set difference cannot drift from its own intent, so the intent is now the only thing written
  down.)*
- A `deterministic` arm is **reproducible from `(packContentHash, armParametersHash, benchVersion)`
  alone**, which is why host state is not merely optional for it but forbidden: recording a KV-cache
  setting beside a BM25 score would imply the score depends on it.
- Consequences downstream, all decided here in S1 because S3 consumes them: `runId` for a
  deterministic arm is `<packId>-<armId>-<UTC-timestamp>`; the arm is computed and stored on **every**
  embedding run (FR-13 says "every"), sharing the model run's `sessionId` so the pairing label in
  §3.7 still applies; `compare_report` accepts model and deterministic runs in one
  `Sequence[RunResult]` and labels the deterministic one `reference arm (deterministic given pack
  version)`; two deterministic arms are **never** the subject of a verdict against each other; and
  `models --tested` (FR-17a) filters to `armKind == "model"`, so BM25 can never be offered as a
  reference model.

#### 3.4.2 The fields, and *absent* versus *empty* (M-3)

**Auto-captured** (`fingerprint.capture()`, no human input, cannot be wrong without the tool being
wrong): `modelKey` (the literal LM Studio id, never normalized), `modelPublisher`, `arch`,
`quantization`, `compatibilityType`, `maxContextLength`, `loadedContextLength`, `runtimeName`,
`runtimeVersion`, `lmsCliCommit`, `residentModelsAtStart[]`, `residentModelsAtEnd[]` (both from
`lms ps --json`), `modelType`, `modelCapabilities` and `modelCapabilitiesPresent` (the catalog's raw
`type` and `capabilities` verbatim, plus the absent-vs-empty bit below — §3.6's gate decision must
be auditable after the fact), `temperature`, `maxTokens`,
`packId`, `packVersion`, `packContentHash`, `benchVersion`, `benchSchemaVersion`, `pythonVersion`,
`hostOs`, `startedAt`, `endedAt`.

**This section owns the model field set.** The 26 names above plus the four below **are**
`REQUIRED_BY_SCHEMA[1]["model"]`, 30 in total, and §3.4.1's forbidden set is derived from them — so
a field added here is forbidden on a deterministic arm without a second edit anywhere, and no list
elsewhere in this plan needs sweeping to keep up.

**Operator-attested** (§6 R-1 — no programmatic source exists): `lmStudioAppVersion`,
`kvCacheSetting`, `hostRamGb`, `otherResidentWorkloads`. These live in a local, gitignored
`model-bench/host.json` (§3.4.4) and are **copied into every run record**, so a record is
self-contained and history stays readable when the box changes.

**Validation distinguishes three states per field, not two.** v1.1 said `validate()` raises on
"missing, empty, or `null`", which rejects legitimate values: `lms ps --json` returns `[]` when
nothing is loaded, so `residentModelsAtStart: []` is the correct and informative value on a clean
box; and the catalog omits the `capabilities` key entirely for several models rather than sending an
empty list. So each required field is declared in one of two tiers:

- **`REQUIRED_NONEMPTY`** — the key must be present *and* the value truthy: `modelKey`,
  `quantization`, `runtimeName`, `runtimeVersion`, `packId`, `packVersion`, `packContentHash`,
  `benchVersion`, `lmStudioAppVersion`, `kvCacheSetting`, …
- **`REQUIRED_PRESENT`** — the key must be present; `[]`, `0`, `false` and `""` are all valid
  values: `residentModelsAtStart`, `residentModelsAtEnd`, `modelCapabilities`,
  `modelCapabilitiesPresent` (`false` is a real answer, and the whole reason the field exists),
  `otherResidentWorkloads`, `temperature` (0.0 is the pinned value for four of the five packs), …

`null` is invalid in **both** tiers — it is the shape of "we did not capture this", which is the one
thing FR-7 refuses. A field the catalog genuinely omits is captured as `[]` or `""` by `capture()`,
never passed through as `null`, and `modelCapabilities` records the distinction it needs in
`modelCapabilitiesPresent: bool` beside it.

**Capture ordering is part of the contract, because two fields only exist at one point in time.**
`loadedContextLength` appears in the catalog only once a model is loaded, and residency is the thing
that changes across a run. So: `capture()` snapshots `residentModelsAtStart` **before** `lms load`,
reads the **catalog after the load completes**, and snapshots `residentModelsAtEnd` after the last
scored call. A `capture()` that reads the catalog first produces a record with a null
`loadedContextLength` and fails its own validation — which is correct behaviour, and is why the
ordering is stated rather than left to the implementer.

#### 3.4.3 `benchSchemaVersion`, `benchVersion`, and not deleting history (M-4)

**`benchVersion` = `modelbench.__version__`**, which S0 pinned to the installed distribution's
metadata (`project.version` in `pyproject.toml`, asserted by `tests/test_package.py`). That
assertion is load-bearing, not filler: a skew between the two would silently mislabel `benchVersion`
in every stored record.

**`benchSchemaVersion` is a separate integer constant in `modelbench/results.py`, starting at `1`**,
never derived from `benchVersion` and never bumped by a release. It increments only when the
required-field set or the on-disk record shape changes in a way a *reader* must branch on.

v1.1 said an "older-schema" record is excluded on read. That is wrong as stated, and against FR-3
directly: the first time a required field is added, every result stored before it would be
quarantined out of every comparison, and the tool's entire value is that "a new model's result lines
up against models I tested months ago". The rule instead:

- `REQUIRED_BY_SCHEMA: Mapping[int, Mapping[str, FieldSpec]]` — an older record is validated against
  **the contract it was written under**, and passes if it satisfied it.
- `invalid` is reserved for a record that fails **its own** schema (hand-edited, truncated,
  blanked field) or declares a `benchSchemaVersion` this build does not know — a record from the
  *future*, which is the genuinely uninterpretable case.
- A report comparing records across schema versions prints a `SCHEMA VERSIONS IN THIS COMPARISON`
  line naming them, in the same spirit as AC-3's pack-version banner: visible, never silent, and
  never a reason to drop the record.
- `model-bench migrate` exists for a genuinely breaking change: it rewrites stored records forward,
  in place, recording `migratedFrom` on each. `README.md` states that a schema bump is a deliberate
  act — a new `REQUIRED_BY_SCHEMA` entry, a `HISTORY.md` line, and a decision about whether a
  migration is needed — not a side effect of adding a field.

#### 3.4.4 `host.json` — the operator-attested file (M-8)

Local, gitignored, written by `model-bench attest` (§3.6a), read at the start of every `model` run:

```json
{
  "schemaVersion": 1,
  "lmsPath": "/mnt/c/Users/<user>/.lmstudio/bin/lms.exe",
  "attested": {
    "lmStudioAppVersion": "0.3.31",
    "kvCacheSetting": "f16",
    "hostRamGb": 16,
    "otherResidentWorkloads": ["docker: falkordb-dev", "windows desktop session"]
  },
  "attestedAt": "2026-09-02T14:05:00Z",
  "observedAtAttestation": {"runtimeName": "llama.cpp", "runtimeVersion": "…", "lmsCliCommit": "07b7252"}
}
```

`attested` is exactly the four FR-7 fields with no programmatic source; `observedAtAttestation` is
what the staleness trip-wire compares against; `lmsPath` is §3.6's explicit path setting.

#### 3.4.5 Three enforcement points, all cheap

1. **Write refuses.** `results.store(run)` calls `fingerprint.validate()` and raises on any field
   that fails its tier, on any field forbidden for its `armKind`, and on a `null` anywhere in the
   required set. There is no "save anyway" flag.
2. **Read quarantines.** `results.load_history()` re-validates every record against its own
   `benchSchemaVersion` and returns `(valid, invalid)`. `compare` merges only `valid` and prints an
   `INVALID RESULTS EXCLUDED` block naming each excluded record and why. This is AC-2's actual test
   surface — a hand-edited record must be excluded on *read*, not merely rejected on write.
3. **Attestation staleness is detected, not trusted.** `host.json` records the `runtimeVersion` and
   `lmsCliCommit` observed when it was last attested. If either differs at run time, the run stops
   with "LM Studio changed since you last attested `host.json` — re-check the app version and KV
   cache setting, then `model-bench attest`." That converts the weakest link (a human typing a
   value once and never revisiting it) into a loud failure at the moment it goes stale. A
   `deterministic` arm skips this check entirely — it attests nothing.

### 3.5 D5 — Storage: JSON per run is the truth, CSV and markdown are derived

FR-2 wants durable, human-readable, comparable; the stakeholder's stated preference is "CSV or
markdown, either is fine". A single flat file cannot hold per-turn × per-failure-kind breakdowns
without becoming unreadable, so:

- **`results/runs/<runId>.json` — the record of truth.** One file per run: fingerprint, pack
  identity, per-item and per-turn scored records, aggregate counts, timings. Append-only in
  practice: a run file is never rewritten (`model-bench migrate` is the one exception, §3.4.3).
  `runId` = `<packId>-<modelSlug>-<UTC-timestamp>`, where **`modelSlug` is the `modelKey` with every
  character outside `[A-Za-z0-9._-]` replaced by `-`** — real keys contain `/`
  (`qwen/qwen3-4b-2507` → `qwen-qwen3-4b-2507`) and a raw key would create a directory. The slug is
  a filename convenience and is never the identity: `fingerprint.modelKey` keeps the literal id and
  is what every comparison reads (§2.3, R-8). Timestamp is `YYYYMMDDTHHMMSSZ`. For a deterministic
  arm the middle segment is `armId` (§3.4.1).
- **`results/index.csv` — one row per run**, the human-openable summary: runId, date, role, packId,
  packVersion, packContentHash(8), modelKey, quantization, n, headline metric(s), p50/p95 latency,
  valid/invalid. **Derived and fully regenerable** (`model-bench index rebuild`), so it is never a
  second source of truth to keep honest — the same argument §3.1 makes about golden sets, applied
  internally.
- **`reports/<pack-id>-<date>-<n>.md` — generated comparisons**, committed because they are the
  artifact a human actually reads six months later. Never hand-edited. `<n>` is a two-digit
  same-day sequence starting at `01`, chosen by scanning existing files — a same-day re-run is the
  normal case while a pack is being developed, and silently overwriting the earlier comparison is
  the one behaviour a tool built around durable history must not have.
- **`results/transcripts/<runId>.jsonl` — raw model output**, gitignored. Useful for a post-hoc
  "why did it fail", not needed for any comparison, and large.

**FR-20 is enforced structurally, not by convention.** `results.load_history()` takes a `packId`
and there is no API to load across packs; `report.py` has no code path that aggregates two roles;
`compare` requires `--pack`. There is deliberately no `model-bench leaderboard` command, and
`README.md` says why. The same shape of refusal carries the two other "the report must not be able
to say this" rules: no path emits a pooled 85-item guard accuracy (§3.8.2), no path emits a blended
tool-calling percentage (§3.8.4), and no path synthesises a headline for a pack whose
`headlineMetric` is `null` (§3.3). Each is a **missing function**, not a guarded one — §3.5's
argument about `index.csv` applied to the report: a rule you cannot express is a rule you cannot
break under deadline pressure.

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
  `powershell.exe Get-Process`, best-effort). Neither is presented as "the model's RAM cost" without
  its method label. **Sampling is suspended for the duration of every timed call**: the runner
  samples *between* turns, never during one, and each sample carries its timestamp so a reader can
  confirm no sample overlaps a timed window. This is not fastidiousness — a `powershell.exe`
  round-trip across the WSL boundary was measured at **0.18–0.54 s** (gate Appendix A.3) against
  ~1.3 s per turn in the prior experiment, so an interval sampler running beside the call would
  perturb the very latency it sits next to by a double-digit percentage.
- **The full FR-11 surface, with each field's source named.** FR-11 lists five things and v1.1
  accounted for three; all five, plus the one thing FR-11 demotes:

  | Field | Source | Status |
  |---|---|---|
  | `turnLatencyMsP50` / `P95` | **client wall clock**, measured around the HTTP call from just before the request to the last byte of the body | **headline** |
  | `ttftMs` | `stats.time_to_first_token` (LM-Studio-reported) | reported |
  | `prefillMsPer1kPromptTokens` | `1000 × stats.time_to_first_token ÷ (usage.prompt_tokens ÷ 1000)`, per call, aggregated as median | reported (**new in v1.2 — FR-11 names it and v1.1 omitted it**) |
  | `coldLoadSeconds` | timed `lms load`, once per run | reported separately, never averaged into steady state |
  | `tokensPerSecond` | `stats.tokens_per_second` | **diagnostic only** — FR-11 says so in words, and the report labels it so |
  | `modelSizeBytes`, `peakHostRssBytes` | `lms ps --json`; sampled `Get-Process` | best-effort, method-labelled (R-2) |

  The headline being **client wall clock** and not `stats.generation_time` is a decision, not an
  oversight: FR-11 asks for *end-to-end turn latency*, which includes request assembly, transport
  and the server's own queueing — everything the LM-Studio-side number excludes. Both are stored, so
  the difference between them is itself readable, but only one is the headline.
- **Tool-calling eligibility is gated before a tool-caller run**, so that "this model cannot do
  tool calls at all" is never recorded as "this model got them wrong". The gate is **not** a
  `"tool_use" in capabilities` test — that was v1.1's rule and it is wrong in both directions on
  this box's actual catalog, which the gate re-probed: `text-embedding-qwen3-embedding-0.6b`, a
  `"type": "embeddings"` model, advertises `"capabilities": ["tool_use"]` and would pass; while
  `google/gemma-3-4b` and `gemma-3-4b-vl-it-…` have **no `capabilities` key at all** and would be
  refused, though nothing establishes they cannot emit tool calls. The rule:

  > eligible ⟺ `type ∈ {"llm", "vlm"}` **and** (`capabilities` is absent **or** contains
  > `"tool_use"`).

  A refusal names which half failed. The raw `type` and `capabilities` go into the fingerprint
  (`modelType`, `modelCapabilities`, `modelCapabilitiesPresent` — §3.4.2), so the gate's decision on
  any past run is auditable from the stored record rather than re-derived from a catalog that has
  since moved.

### 3.6a D6a — The CLI surface, consolidated (M-8)

The CLI is the tool's entire user surface. v1.1 mentioned six commands across five sections and
specified none of them, which left `model-bench attest` — required before S3's done-condition can be
met — owned by no stage. One table, and each command is assigned to the stage that must ship it:

| Command | Flags | Effect | Stage |
|---|---|---|---|
| `compare --pack <id>` | `--models a,b` · `--session <id>` · `--negative-control` · `--out <path>` | Reads `results/runs/`, renders the markdown comparison to `reports/` and stdout | **S1** |
| `index rebuild` | — | Regenerates `results/index.csv` from `results/runs/` | **S1** |
| `models --tested` | `--pack <id>` · `--role <role>` | Lists models with stored results (`armKind == "model"`); from S2 also intersects with the installed catalog | **S1**, catalog half **S2** |
| `attest` | `--lms-path <path>` · non-interactive `--set k=v` | Prompts for the four operator-attested fields, probes LM Studio, writes `host.json` (§3.4.4) | **S2** |
| `validate --pack <path>` | `--strict` | Runs `validate_pack`: manifest schema, `metrics` block, `sampling` contract (§3.3 — `analysisUnit == pairingKey[0]`, row-count identity, `replicatesPerScript`), ids, provenance, paraphrase rule, pack-module import allowlist, `H ≤ min(script length)` | **S2** |
| `run --pack <id> --model <key>` | `--session <id>` · `--reference <key>` · `--warmup <n>` · `--no-cold-load` | One model × one pack; calls `validate` first and fails closed | **S2** plumbing, first usable **S3** |

**Exit codes.** `0` whenever the tool ran and reported, *whatever the scores* — the requirements rule
out pass/fail gating and §1 states that the only non-zero exits are operational. The closed set:
`2` bad arguments or usage · `3` LM Studio unreachable or `lms.exe` unresolvable · `4` invalid pack
(`validate` failure, load error, unmet `environment.requires`) · `5` fingerprint incomplete or
`host.json` stale/absent. Nothing else. A `compare` that finds every stored record invalid still
exits `0` and prints the `INVALID RESULTS EXCLUDED` block — that is a report, not an operational
failure.

**Output.** Every command writes its human-readable output to stdout; `run` and `compare` also write
their artifacts (`results/runs/<runId>.json`, `reports/<pack-id>-<date>-<n>.md`) and print the paths.
No command writes outside `model-bench/`, ever (FR-23, §5 test 20).

### 3.7 D7 — Runs, sessions, pairing, and the reference arm (FR-1 / FR-16 / FR-17)

FR-1 (one model per run) and FR-16 (paired, same session) read as a tension. They are reconciled by
separating three ideas that the requirements use in one breath:

- **A run** is one model × one pack × the pack's declared sampling (`sampling.scripts ×
  sampling.replicatesPerScript` for a conversation pack, one call per item otherwise). Always
  exactly one model (FR-1).
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
  `golden_retrieval.embeddings.json` — the latter to isolate "is my ranking code right" from "is my
  embedding call right" (`-ml` §5.4). **On its own it does not isolate that**, and v1.1 claimed it
  did: inspected, the file holds **38 query vectors only** (`{gr-NN: {model, vector}}`), so the 121
  corpus vectors are still computed live and a wrong `documentPrefix` or a truncated corpus
  contaminates the self-test anyway. So S3 additionally writes the **121 corpus vectors** into the
  pack once, from the same deterministic embed pass, as `corpus.embeddings.json` — giving the
  ranking path a fully fixed input on both sides. Both files are inside the content hash, so a
  re-embed is a pack version bump.
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
  hardcoded. **It is stored as its own `RunResult` with `armKind: "deterministic"`** (§3.4.1) —
  that is the data shape that carries it to `compare_report`, and the reason `armKind` is decided in
  S1 rather than improvised in S3. The pack's declared BM25 parameters are hashed into
  `armParametersHash`, so changing `k1` invalidates the arm's comparability the same way changing a
  golden item invalidates the pack's.
- **Reported (FR-12/FR-14):** recall@k, MRR, **P@1** plus precision@k, and score separation as both
  `sep_raw` (within-model, actionable) and the corpus-sd-normalized `sep_z` (the cross-model
  comparable one), aggregated as median + p10 + fraction > 0 (`-ml` §5.2); plus output dimension,
  RAM cost, embedding throughput (texts/s and tokens/s), max input length and observed truncation
  behavior, and the prefix convention used.
- **`verdictMetrics = ["mrr"]`, `headlineMetric = "mrr"`**, and the report carries two honesty lines
  the note requires
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
  (`-ml` §7.3): **`falseAdvanceRate`** = P(judge advances | gold says suspend) on the 40
  `clear_suspend` items, **`falseSuspendRate`** = P(judge suspends | gold says advance) on the 30
  `clear_advance` items, and the 15 `boundary` items **descriptive only** at that n and explicitly
  not a verdict metric. Split by evidence path (`understanding` / `turns`) as a diagnostic, plus a
  parse-failure count in the denominator. `report.py` has no code path that emits a pooled 85-item
  accuracy number.
- **This pack has no headline number, deliberately** *(stakeholder decision, 2026-09-02)*:
  `verdictMetrics = ["falseAdvanceRate", "falseSuspendRate"]`, `headlineMetric = null`. Both
  class-conditional rates get a verdict, with equal weight and no ranking between them — the
  stakeholder declined to declare which error is costlier in the product, which is precisely the
  judgement `-ml` §10's open question 2 said could not be made by the analyst.
- **Both co-primaries are error rates pointing the same way, and that is a requirement rather than a
  style choice** (`-ml` §7.3, authoritative). v1.2 paired `falseAdvanceRate` with **advance-recall**,
  which is `1 − falseSuspendRate` — the same quantity read backwards. Two co-equal verdict metrics
  where one reads "better is lower" and the other "better is higher" makes "worse on one, better on
  the other" unreadable at a glance, in the one pack where the stakeholder deliberately declined to
  rank the two errors. So both verdicts render on error rates; **`advanceRecall` stays printed as
  the labelled complement** of `falseSuspendRate`, so a reader looking for recall still finds it,
  but it carries no verdict.
- **The two-member family costs resolving power, and the report says so.** Holm–Bonferroni is
  mandatory here (§3.3(ii)), so the resolving-power line for each class computes at **α/2**, not
  α=0.05; `-ml` §7.3 carries the recomputed figures for both slices and the plan does not restate
  them. `-ml` §7.3 also records a costed **reversal trigger**: if the stakeholder ever ranks the two
  errors, the loser moves out of `verdictMetrics` into the exploratory block, the family collapses
  to k=1 at α=0.05, and resolving power improves measurably. That trade is the stakeholder's to
  make later, not the implementer's to make quietly.
- This is the pack that makes §3.3's `headlineMetric: null` a first-class case rather than a special
  case; a future pack that wants no headline needs no further harness work.
- **Cost:** low.

#### 3.8.3 `nlq-generator` — pack `nlq-structured-query`

- **Data copied:** `nlq_golden_set.jsonl` (40 items) → `items.jsonl`; the 15-product catalog from
  `falkor-chat/scripts/seed_catalog.sh`'s `CATALOG` literal — which is a **Python list of
  `(name, category, price)` tuples inside a `<<'PY'` heredoc**, not a shell array, so copying it is
  a Python parse rather than a shell one — plus a **read-only snapshot** of
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
  is **Layer 1** — exact match after canonicalization against the item's `expected`, with the same
  numeric epsilon (`_NUMERIC_EPSILON = 0.01`) and the same scalar/set shape rules
  `falkor-chat/server/tests/eval/nlq_scoring.py` documents (re-implemented, cross-checked in tests
  against a copied sample of `nlq_eval_results.json` records). **Layer 1 is not uniformly
  exact-match, and the exception must be built rather than discovered:**
  `nlq_scoring.score_pair` scores `shape == "conflicting-facts"` by **subset containment**
  (`expected_set.issubset(actual_set)`, `nlq_scoring.py:190`), not set equality — it affects 2 of
  the 40 items, and a scorer that applies equality uniformly marks both wrong for every model,
  turning them into a second silent floor beside the unanswerable bucket. `nlq_scoring`'s **Layer
  2** (`layer2_contains`, containment against the *rendered* natural-language answer) is a
  non-gating sanity signal in falkor-chat and is **not** copied: this pack scores the structured
  result, which is what the model actually produced.
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
- **Reported:** `verdictMetrics = ["layer1ExactMatchRate"]`, `headlineMetric =
  "layer1ExactMatchRate"` — the note's choice for this role (`-ml` §3.3, §7.2), and v1.1 declared
  none at all, which under §3.3's own rule would have labelled every number in this pack's report
  `exploratory` and made a verdict impossible. Beside it, exploratory: correct/incorrect split by
  `shape` (single-fact, filter-list, compound-filter, not-found, aggregation,
  relationship-traversal, conflicting-facts) — the shape split is the informative part at n=40;
  plus malformed-spec and schema-violation counts separately from wrong answers, and the
  abstain-vs-fabricate count on the unanswerable bucket.
- **Cost: medium-high, and v1.1 under-costed it.** Execution *is* small — `QueryRequest.matches` is
  `min_length=1, max_length=1`, so there are no joins — but "malformed-spec and schema-violation
  counts **separately** from wrong answers" means re-implementing `querygen`'s **validation**
  surface too, in **stdlib only** (§3.3 forbids a pack module from importing pydantic): the
  `QueryFilter`/`QueryMatch`/`QueryRequest` constraints (`extra="forbid"`, the six-operator
  whitelist, `filters` max 4, `returns` 1–6 against the projection/aggregate regexes, the `order_by`
  shape, `limit` 1–50) plus `compile()`'s allowlist and string→number coercion. That is the larger
  half of `tools/exec.py` and it is what makes the three failure classes distinguishable rather
  than pooled into "wrong". Still pure and unit-testable against the golden set's own expected
  values — but sized as its own piece of work in S4, not as a rounding error.
- **Scope note:** this measures *structured query generation against a declared schema*. It does
  not measure raw Cypher authorship; nothing in the requirements asks for that, and adding it would
  need a database and a much larger golden set.

#### 3.8.4 `tool-caller` — pack `tool-caller-shop-assistant` (the long pole, FR-22)

- **Data: new, and the main cost of this feature.** `conversations.jsonl` — fixed, versioned,
  multi-turn scripts. Each row:

  ```json
  {"scriptId": "A-02", "shape": "A", "replicate": 1,
   "description": "read-only catalog lookups",
   "turns": [{"seq": 1, "user": "...",
              "expect": {"toolRequired": true, "tool": "lookup_product_fact",
                         "args": {"name": "Wireless Charging Pad"},
                         "argChecks": [{"kind": "boundary", "arg": "maxPrice", "value": 50}],
                         "terminal": false,
                         "finalReplyMustContain": ["24.99"]}}],
   "provenance": {"draftedBy": "...", "verifiedBy": "...", "basedOn": "..."}}
  ```

  `scriptId` is the **sampling unit** and the first component of `ItemResult.pairingKey` (S1);
  `shape` ∈ `{A, B, C}` is the reporting stratum; `replicate` is `1` throughout under the current
  sizing and exists so that raising `replicatesPerScript` later does not change the record shape.

  Three conversation **shapes**, reconstructed from
  `falkor-chat/docs/reviews/salesperson-tool-reliability-ml.md` §8.1 (§2.2): **A** 9 turns
  read-only, **B** 7 turns write-mutating, **C** 4 turns short. Then extended — §8.1's set was
  designed to characterize one defect, and this pack needs coverage of all seven FR-8 failure
  kinds, including "stopping when done" and "final reply matches what the tool returned".
- **Sizing: 12 distinct scripts (4 per shape) × 1 run each = 12 conversations, at
  `temperature: 0.0`** — *stakeholder decision, 2026-09-02, replacing v1.1's 4 scripts × 4
  replicates × 3 shapes.* The authoring cost lands in S6. **Correction, v1.3:** this decision was
  put to the stakeholder as "the same total run budget", and that was wrong — 12 × 1 is roughly a
  **quarter** of the previous design's inference budget, not a re-allocation of it (`-ml` §4.5.2
  carries the measured basis, §4.5.3 the honest consequence, and a costed reversal trigger). The
  correction is being relayed to the stakeholder separately; the design is **not** revisited on it
  here, because the resolving power it buys is unchanged — the old nominal 48 never supported more,
  which is the whole reason the decision was taken. It satisfies **FR-22a**, and it is taken on
  `-ml` §4.5's
  own argument: at temperature 0, replicates of an identical prompt are near-duplicates that add
  almost no information, so 48 conversations clustered in 12 scripts carried an effective *n* the
  note put "closer to 12 than 48". Buying the 12 outright makes the honest number the real one
  instead of a design effect the bootstrap has to discover. Two consequences that must be built,
  not assumed:
  - **The sampling unit is the script and there is exactly one observation per script**, so the
    two-level cluster resample `-ml` v1.1 §4.5 specified no longer has an inner level. **The
    clustering treatment under this design — what `stats.py` must implement for
    `min_detectable_difference`, `verdict()` and any cluster-aware interval — is settled in the
    `-ml` note, not here** (see §3.9). This plan does not restate it, and `stats.py` implements the
    note's signatures verbatim.
  - **The pack declares `replicatesPerScript: 1` in its manifest and the report prints it beside
    every conversation-level *n***, with `temperature`, as `-ml` §4.5 requires. **`validate` fails
    any pack declaring `replicatesPerScript > 1`** while `stats.py` carries only the one-level
    `cluster_bootstrap` (`-ml` §3.4 Rule 6): a replicated pack needs the two-level resample, and its
    absence must be an error rather than an approximation nobody notices. So the field cannot be
    raised invisibly *or* silently mis-analysed — raising it is a deliberate act that first requires
    the note's two-level function to exist.
- **The determinism probe — the evidence behind `basis: "by-construction"`** *(new in v1.4, gate
  finding N-2)*. `-ml` §3.4 Rule 4 lets McNemar decide **only** when `design_effect == 1.0` **and**
  `basis == "by-construction"`, and §4.5.1 grants that basis to 12 × 1 on the grounds that each
  script contributes one observation. But §4.5.1(iii) also states that this design makes run-to-run
  variability *unmeasurable*, that LM Studio at temperature 0 is "near-deterministic but not
  guaranteed bit-deterministic", and prescribes the evidence: **re-run 2 of the 12 scripts a second
  time, once per model, and report whether the outcome vector is identical.** v1.3 asserted the
  basis and built no probe, so the one input deciding whether McNemar may decide the flagship metric
  was an attestation. Now:
  - **The pack names the two scripts** (`sampling.determinismProbeScripts`, one shape-A and one
    shape-B script — the long and the write-mutating shapes, where non-determinism is most likely to
    bite), so which two is pack data and cannot be chosen after seeing a result.
  - **Budget: 2 extra conversations per model** (~14 turns, one shape-A + one shape-B), stated here
    because it is a real cost that must appear in the run plan rather than surprise S6. It runs in
    the same session, after the 12 scored conversations.
  - **Diagnostic, outside `n`, never pooled into it** (`-ml` §4.5.1(iii)). The probe's conversations
    are excluded from every denominator and appear only in the report's own probe line.
  - **The outcome is wired to `basis`, and absence of evidence degrades it.** `runner` sets the
    `basis` passed to `resolving_power()` to `"by-construction"` **only if** all three hold:
    `replicatesPerScript == 1`, the probe **ran**, and both probe scripts produced outcome vectors
    identical to their scored runs. Otherwise `basis = "assumed"` — including the case where the
    probe simply was not run. Via `-ml` §3.4 Rule 4 that automatically moves McNemar out of the
    decision seat and the cluster-bootstrap CI into it, with McNemar's p still printed and labelled
    `anti-conservative under clustering — not the decision`. **The fail-safe default is the point**:
    an unrun probe can never silently buy the stronger instrument, so N-2 cannot recur by omission
    the way it arose.
  - **Carried in the record:** `ToolCallAggregates.determinismProbe =
    {scriptIds, ran: bool, identical: bool, differingTurns: [...]}`, so a stored run's `basis` is
    re-derivable years later rather than trusted.
- **Honest consequence, printed not buried:** twelve conversations is a small *n*, and the
  resolving-power line computed from it (§3.9 point 2) will be a large number. That is the true
  precision of the previous design too — it was simply hidden inside a design effect. The effects
  this pack exists to catch are the near-total ones (`-ml` §7.1's reassurance: the qwen3-4b turn-4
  collapse, the ministral duplicate-instruction defect), and the report says in words which
  magnitudes it can and cannot resolve rather than leaving a reader to assume.
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
- **Headline and diagnostic (`-ml` §4.6).** `verdictMetrics = ["cleanThroughTurnH"]`,
  `headlineMetric = "cleanThroughTurnH"` — the fraction of conversations with no failure through
  turn *H*: one observation per conversation, length-independent by construction, and the statistic
  that would have caught the incumbent model. ***H* is declared in `pack.json`
  (`metrics.cleanThroughTurnH.H`, `4` for the A/B/C set), never derived from the data.** v1.1
  derived it as the shortest script length, which means adding one 3-turn script to a future pack
  version silently redefines the headline from `cleanThroughTurn4` to `cleanThroughTurn3` — the
  primary metric changing meaning under a fixed name, which is the exact failure this tool exists
  to prevent, and one AC-3's version banner would report as "the data changed" rather than "the
  headline now measures something else". So `H` is pack data, and three consequences follow:
  - **`validate` *fails* a pack where `H > min(script length)` — it never clamps `H`, and never
    warns.** The methodological reason, and it is stronger than tidiness (`-ml` §4.6, v1.4): a
    headline computed at `H > min` is **selection-conditioned**, because short scripts cannot reach
    turn `H` at all, so the metric quietly becomes a rate over long conversations only. That is
    `-ml` §4.3's laundering failure landing in the one number the report calls its headline —
    precisely the thing §3.8.4's funnel table exists to prevent everywhere else. A clamp would
    silently produce a *different, valid* metric under the declared name, which is M-11 again by
    another route; a warning would be ignored. Fail closed.
  - **`H` is printed beside the metric name in every report** — `cleanThroughTurn4`, never
    `cleanThroughTurnH`.
  - **`H` strictly below `min(script length)` is legitimate, and the gap is reported.** Declaring
    `H < min` keeps the headline's meaning stable across pack versions, at the price that turns
    `H+1 … min` are scored and then excluded from the headline — the metric becomes **coarser, not
    wrong** (`-ml` §4.6, v1.4). So when the gap is non-zero the report carries a line naming it as
    discriminating information deliberately left on the table. It is **zero today** (`H = 4 =
    min(script length)`), which is why the line renders only when it is not. The **per-turn hazard** — P(first failure at *t* | clean
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

- **Data: new. 30 items** derived from the copied 121-message corpus, LLM-drafted and **verified by
  a human, item by item** (FR-19). 30, not 20, because of the paired floor (`-ml` §7.1). **The
  10-item `golden_judge_calibration.jsonl` is not copied at first delivery** — it exists only to
  gate a judge, and there is no judge to gate. Item shape, aligned to `-ml` §6.2's **checklist
  ground truth** (v1.1 carried a `referenceAnswer` field that the note's design does not have and
  that nothing in this pack scores — an unscored free-text field beside a deterministic scorer is
  an invitation to a future judge, so it is **dropped**):

  ```json
  {"itemId": "cr-07", "question": "...", "context": ["..."],
   "mustContain": ["24.99"], "mustNotContain": ["19.99"], "mustAbstain": false,
   "format": {"maxWords": 120, "mustBeSingleParagraph": true,
              "forbiddenPatterns": ["^\\s*[-*]\\s", "```"]},
   "provenance": {"draftedBy": "...", "verifiedBy": "...", "corpusVersion": "..."}}
  ```

- **Scoring — three deterministic families, no judge anywhere in the pack.** Mapping FR-21a's
  "latency, format, faithfulness to what the tool actually returned" onto a role that calls no
  tools — **all three ship, where v1.1 shipped two**:
  - **latency** — the standard FR-11 block (§3.6).
  - **format** — checked against the item's own `format` block, with pack-level defaults in
    `pack.json` that an item may override. Three declared constraints, each a separate count and
    never pooled with grounding: word count within `maxWords`, single-paragraph discipline when
    `mustBeSingleParagraph`, and zero matches of `forbiddenPatterns`. The constraints are **pack
    data**, not scorer heuristics — the same rule FR-8(d)'s `boundaryRule` follows, and for the
    same reason: a format rule inferred by the harness is a rule two packs cannot agree on.
  - **grounding** — FR-21a's "faithfulness to what was actually returned", here the retrieved
    context: `mustContain` / `mustNotContain` / `mustAbstain` containment, using the same
    canonicalization as the `nlq-generator` scorer.
  - `verdictMetrics = ["groundingRate"]`, `headlineMetric = "groundingRate"`; the format counts and
    the latency block are exploratory beside it.
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

Owned by [`small-model-benchmarking-ml.md`](./small-model-benchmarking-ml.md). **This section cites
that note; it no longer restates it.** *(Changed in v1.2. v1.1 paraphrased the note's formulas,
constants, thresholds and sample sizes here, and the gate found the two documents had already
silently diverged in three places — the `z` constant, the fixture tolerance, and
`min_detectable_difference`'s coefficient. Two copies of a formula is one copy and one bug. So:
every formula, constant, threshold, denominator, tolerance, bootstrap parameter, sample size and
verdict string lives **only** in the note; `modelbench/stats.py` implements the note's signatures
verbatim and cites it by section in its module docstring; and where this plan needs to name a
number, it names the function that computes it instead.)*

The five conclusions the rest of this plan is built on, as **statements of what is being built** —
each one's arithmetic is at the cited section:

1. **The comparison instrument is the paired difference — now what FR-15/AC-4 require** (amended
   2026-09-02; the amendment is §6 R-9). One instrument **decides** (an exact paired test on the
   discordant pairs) and a second **quantifies** (a confidence interval on the paired difference,
   derived from the same Wilson function used for per-arm reporting). They are never AND-ed into a
   single bloc, and when they disagree both component outcomes are printed in prose. **AC-4's
   "not distinguishable at this sample size" fires exactly when the paired-difference interval
   includes zero.** Continuous metrics use a seeded paired bootstrap. Per-arm Wilson intervals are
   still printed, labelled *descriptive, not the comparison instrument*, and the superseded
   marginal-overlap check is retained as a **diagnostic line** with a footnote saying why it is not
   the verdict. Test names, the exact constants, the bootstrap parameters and the three verdict
   strings: `-ml` §3.2. Why the old rule could not fire: `-ml` §3.1.
2. **Every report prints its own resolving power**, computed by `stats` from its own **effective**
   *n*, its analysis unit, its design effect and its family-adjusted α — never quoted from the
   requirements, never a literal in the codebase, and never derivable from a bare observation count.
   The template, the mandatory sentences and every number in them are `-ml` §7.1/§7.2/§7.3's; the
   note carries the **tool-caller pack's rendered line verbatim** (`-ml` §7.2), and that string is
   what S1's report test asserts against. `min_detectable_difference` is the one function that must
   never be hardcoded, and S1's done-condition tests exactly that.
3. **Each pack pre-registers its verdict family** — `verdictMetrics` (1..k) plus an explicit
   `headlineMetric` that may be `null` (§3.3). Only a `verdictMetrics` member can receive a verdict;
   everything else prints `exploratory — no significance claim`. A family with k > 1 takes
   Holm–Bonferroni **and** computes its resolving power at α/k. (`-ml` §3.3. v1.1's "exactly one
   `primaryMetric`" is superseded by the 2026-09-02 stakeholder decision on
   `guard-judge`, and v1.2's field names are superseded by §3.3's.)
4. **Per-turn-position rates are computed over conversations, never over turns**, and no interval
   is ever printed over a turn-pooled count. **Under the 12 × 1 sampling design (§3.8.4) the
   clustering treatment is materially different from what `-ml` v1.1 assumed, and the note owns the
   difference**: what `stats.py` must implement for `min_detectable_difference`, `verdict()` and any
   cluster-aware interval — including whether a design-effect input is required and what
   `report.py` must refuse to print without it — is specified there and is not restated, guessed at
   or pinned by signature here. `stats.py` implements the note's surface as written.
5. **The negative control is a first-class feature, not a test fixture**: `compare` run with the
   same model in both arms — **two independent runs, not two copies of one record** — must report
   *not distinguishable*, with discordant counts roughly equal and the difference interval centred
   on zero. It is wired as a CLI-reachable mode (`--negative-control`) because it is the cheapest
   way to catch a whole class of harness bugs that otherwise present as plausible model differences.
   (`-ml` §9; the acceptance run is §5 test 19a.)

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
| Pooling turn-level outcomes into one accuracy rate | Turns within a conversation are not independent; the prior experiment's 280 turns (§2.2) carried roughly the information of its 40 conversations (`-ml` §4.4). Per-turn slices over conversations. |
| 4 distinct scripts × 4 replicates × 3 shapes for the tool-caller pack (v1.1's sizing) | At `temperature: 0` the replicates are near-duplicates, so 48 conversations carried an effective *n* the note put closer to 12 (`-ml` §4.5) — a real precision hidden inside a design effect. **Stakeholder decision, 2026-09-02:** 12 distinct scripts × 1 run (§3.8.4). |
| Replicates at `temperature > 0` instead of more scripts | Informative about run-to-run variance, but it adds a second variance source to a comparison whose subject is the *model*, and it weakens what FR-18's temperature pinning means. Declined with the same decision. |
| `numpy`/`scipy` for the numerics | Zero runtime dependencies keeps old results reproducible years later; the workload is seconds of pure Python. Reversal trigger stated in §3.2. |
| A declarative mini-language for simulated tools instead of pack Python | Would become a worse Python; pack code is content-hashed, so it is versioned data like everything else (§3.3). |

---

## 4. Step-by-step implementation

Eight stages. Each leaves the tree buildable and the suite green, and each has a done-condition that
does not depend on the next. Stages S1–S2 are the harness; S3–S7 are packs, ordered so the cheapest
end-to-end proof lands first and the long pole starts as early as its prerequisites allow.

**Every done-condition below runs with `model-bench/` as the working directory**, written here once
so no stage restates it. This is not a formality: the repo has **no root `pyproject.toml`,
`pytest.ini`, `setup.cfg` or `tox.ini`**, so invoking `model-bench/.venv/bin/python -m pytest -q`
from the repo root makes `rootdir` the monorepo, ignores this component's `testpaths`, and walks
into other components' suites — measured during S0 at 9 collected, 8 collection errors, exit 2. The
canonical form, used verbatim in every stage:

```bash
cd model-bench && ./setup.sh && .venv/bin/python -m pytest -q && .venv/bin/ruff check .
```

`ruff check` takes an explicit `.` for the same reason: without a target it resolves from the
current directory and its config discovery is not the same walk as pytest's.

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

**Done when:** the canonical command above passes from `model-bench/`, with **at least one test
collected**.

**S0 is delivered; this is what shipped** (commit `0522ffd`, `model-bench/docs/HISTORY.md`), and two
points differ from v1.1's text:

- **v1.1 asked for "zero tests collected", which is not a passing state.** pytest exits `5`
  (`EXIT_NOTESTSCOLLECTED`) when nothing is collected, so the `&&` chain could never return 0. S0
  ships **one real test** — `tests/test_package.py`, asserting `modelbench.__version__` equals the
  installed distribution's metadata version — rather than configuring the exit code away. That was
  the right call and this plan adopts it: a permanent "no tests ran is fine" setting would still be
  in place at S5, where it would hide a collection breakage; and the assertion is load-bearing
  rather than filler, because that version string is what stamps `benchVersion` into every run
  record (§3.4.3).
- **`.gitignore` names `results/transcripts/` before `results/` exists.** Deliberate — see §3.2.
- `docs/BACKLOG.md` was **seeded at S0** with the two items §7 carries forward (FR-21a's judged
  layer; the +22 harder retrieval queries). **Confirmed, not reversed:** a deferred item is recorded
  when it is decided, not when the milestone closes, or it is one forgotten commit from
  disappearing. S8 therefore *re-checks and extends* that file rather than creating it.

### S1 — Core: fingerprint, results, stats, report (no model calls at all)

**Create:** `modelbench/{fingerprint,results,stats,report,roles}.py`, `modelbench/cli.py`,
`modelbench/__main__.py`.

Key signatures:

```python
# fingerprint.py
class FieldSpec(NamedTuple):
    tier: Literal["nonempty", "present"]          # §3.4.2 — absent != empty
REQUIRED_BY_SCHEMA: Mapping[int, Mapping[str, Mapping[str, FieldSpec]]]   # {schemaVersion: {armKind: {field: spec}}}
FORBIDDEN_BY_ARM_KIND: Mapping[str, frozenset[str]]                       # §3.4.1
BENCH_SCHEMA_VERSION: int = 1                                             # §3.4.3, lives in results.py
@dataclass(frozen=True)
class Fingerprint:
    armKind: Literal["model", "deterministic"]
    ...  # every field in §3.4.2, per arm kind
    def validate(self) -> list[FieldProblem]: ...   # [] means valid; each problem names field + reason
    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Fingerprint": ...
    def to_dict(self) -> dict[str, Any]: ...

# results.py
@dataclass(frozen=True)
class ItemResult:
    itemId: str                     # the pack's stable id — the pairing key's first component
    pairingKey: tuple[str, ...]     # components named by the pack's sampling.pairingKey (§3.3)
                                    # the analysis-unit id is pairingKey[pack.analysisUnitIndex],
                                    # resolved from sampling.analysisUnit — never chosen by a caller
    outcome: Literal["pass", "fail", "n_a", "parse_failure"]
    scoreable: Mapping[str, bool]   # per conditional count: was its precondition met? (-ml §4.3)
    counts: Mapping[str, int]       # per-count numerator contributions
    latencyMs: float | None
    detail: Mapping[str, Any]       # scorer-specific, never read by report.py

@dataclass(frozen=True) class RetrievalAggregates: ...      # recall@k, mrr, p@1, sep_raw, sep_z, …
@dataclass(frozen=True) class ToolCallAggregates: ...       # per-turn table, hazard, funnel, restraint, …
@dataclass(frozen=True) class ClassificationAggregates: ... # per-class rates + parse failures
@dataclass(frozen=True) class ExtractionAggregates: ...
@dataclass(frozen=True) class GroundingAggregates: ...
Aggregates = RetrievalAggregates | ToolCallAggregates | ClassificationAggregates | ExtractionAggregates | GroundingAggregates

Basis = Literal["by-construction", "measured", "assumed"]   # -ml §3.4 Rule 4's input

@dataclass(frozen=True) class RunResult:
    runId: str; sessionId: str | None; role: str; armKind: Literal["model", "deterministic"]
    fingerprint: Fingerprint; items: tuple[ItemResult, ...]; aggregates: Aggregates
    designEffect: float; basis: Basis           # required, no defaults — see below

@dataclass(frozen=True) class InvalidRecord:
    path: Path; runId: str | None; benchSchemaVersion: int | None
    problems: list[FieldProblem]; reason: Literal["field", "unknown_schema", "unparseable"]

def store(run: RunResult, root: Path) -> Path: ...        # raises on invalid fingerprint
def load_history(root: Path, *, packId: str) -> tuple[list[RunResult], list[InvalidRecord]]: ...
def rebuild_index(root: Path) -> Path: ...

# stats.py  — implements docs/plans/small-model-benchmarking-ml.md; no other source
_Z_95: float          # pinned to the note's constant (nlq_scoring.py's `_Z_95`), module-level
def wilson_interval(successes: int, n: int, *, z: float = _Z_95) -> tuple[float, float]: ...
def mcnemar_exact(b: int, c: int) -> float: ...              # conditional binomial, math.comb
def mover_d_interval(a: int, b: int, c: int, d: int) -> tuple[float, float]: ...   # Newcombe
def paired_bootstrap(diffs: Sequence[float], *, B: int, seed: int) -> tuple[float, float]: ...
# PairedOutcomes, ResolvingPower, resolving_power(), min_detectable_difference(),
# observable_floor(), verdict(), cluster_bootstrap(), design_effect(), effective_n():
# signatures and semantics are -ml §3.4's six binding rules. Not restated here — see below.

# report.py
def compare_report(runs: Sequence[RunResult], *, pack: PackRef,
                   invalid: Sequence[InvalidRecord] = ()) -> str: ...   # markdown
```

**Three deliberate changes from v1.1's signatures, each closing a gate finding:**

- **`z` is keyword-only and defaults to a module-level `_Z_95` taken from the note**, not to a
  literal `1.96`. v1.1 wrote `1.96` while the note mandates `1.959963984540054`
  (`falkor-chat/server/tests/eval/nlq_scoring.py:59` defines exactly that, and the note's paired
  interval is derived from Wilson) — so v1.1's default would have shifted every bound and failed
  the note's own regression fixtures. The lab is genuinely split on this constant, which is why it
  is pinned in one place and named here rather than left to inference.
- **`ItemResult` and the aggregates are specified, not `…`.** They are the load-bearing shapes of
  two guarantees. Pairing (FR-16) needs a stable item identity across runs, which is `pairingKey`;
  `-ml` §4.3's paired-*n* intersection and `asymmetry` count are computed from `pairingKey` plus
  `scoreable`. And "`report.py` has no code path that produces a blended figure" (§3.5) is only
  *structural* if the aggregate is typed: with `dict[str, Any]` the report renders whatever a pack
  put in it and the enforcement is back to convention. The per-role variants are a **closed union**,
  so "no path emits a pooled figure" is a type fact a reviewer can check without running anything.
  S1 is built before any pack exists to constrain these shapes, which is exactly why they are fixed
  here.
- **`compare_report` takes `invalid`.** `load_history` returns two lists and v1.1's signature had a
  parameter for one of them, while AC-2 requires the excluded record to be *named in the report*.

**Two record fields added in v1.5, after S1 shipped them and the gate confirmed they were
required.** `RunResult` carries `designEffect: float` and `basis: Basis`, and done-condition 5b is
unsatisfiable without them: `-ml` §3.4 Rule 4 decides *which instrument may decide* from exactly
these two, only the runner sees the determinism probe that sets `basis` (§5 test 12b, gate finding
N-2), and `report.py` can recompute neither after the fact. Three rules go with them:

- **Neither carries a default on the dataclass.** `designEffect = 1.0` is the anti-conservative
  value, so a default rebuilds **plan-review B-1**'s "default by omission" at the record seam — the
  caller who forgets clustering is exactly the caller that gate found — and it makes DC-5's clause "`report.py`
  refuses to render one when the required input is absent" true only vacuously, because with a
  default the input can never *be* absent. S2's runner must state both.
- **`from_dict` is the one place a fallback belongs.** `d.get("designEffect", 1.0)` /
  `d.get("basis", "assumed")` there mean "a record written before these fields existed" — a
  *reader's* compatibility rule under §3.4.3, not a constructor's default.
- **`basis` degrades fail-safe, in one direction only.** A probe that did not run, or that
  disagreed, yields `"assumed"`, which moves the decision off McNemar and onto the cluster
  bootstrap; a comparison takes the *weaker* of its two arms (`by-construction` only when both arms
  are, and `max()` of the two design effects). That propagation is the mechanism N-2 asked for, so
  it is a test target rather than an incidental line (impl-gate M-3).

`items` is a `tuple`, not a `list`: `RunResult` is frozen, and a frozen record holding a mutable
sequence is frozen in name only.

**The clustering-aware and decision-making surface of `stats.py` is deliberately absent from this
block.** `-ml` §3.4 is now a **binding six-rule contract** for exactly that surface — written, in its
own words, so that "the anti-conservative version does not typecheck, and the honest one is the only
one that runs" — and it is the single source of truth for `PairedOutcomes` (whose `from_units` is the
*only* constructor and raises on a repeated analysis-unit id), `ResolvingPower`/`resolving_power()`
(whose `design_effect`, `basis`, `unit_kind` and `alpha` are keyword-only **with no defaults**),
`min_detectable_difference` (which takes `n_effective: float`, never `n: int`), `verdict()` (which
asserts four preconditions and refuses rather than warns), `design_effect` (a **variance** ratio —
the width ratio *squared*), and `cluster_bootstrap` (one level only). Pinning any of these here is
what produced the v1.1 divergence — `min_detectable_difference(n)` against a note that requires the
sampling structure as an input — and this plan does not repeat it. What this plan *does* own is that
S1's done-condition can detect a non-conforming implementation; see below.

`compare_report` is where AC-2/AC-3/AC-4 become visible output: an excluded-invalid block naming
each record and its problems, a pack version/hash mismatch banner, a `SCHEMA VERSIONS IN THIS
COMPARISON` line when records span schema versions (§3.4.3), the resolving-power line (§3.9 point
2), and the literal phrase **"not distinguishable at this sample size"** wherever the decision rule
says so. It also carries the two-instrument disagreement case in prose (§3.9 point 1), and it
renders a `deterministic` arm (§3.4.1) beside model arms without ever ranking two of them against
each other.

**CLI:** S1 ships `compare` (including `--negative-control`), `index rebuild`, and the
stored-records half of `models --tested` (§3.6a). `attest`, `validate` and `run` are S2's.

**Done when**, with unit tests over hand-built `RunResult` fixtures and no LM Studio involved:

1. **AC-2** — a record with a blanked attested field is excluded on read and named, with its
   problem, in the report. Plus the three states of §3.4.2: `residentModelsAtStart: []` is **valid**
   (`REQUIRED_PRESENT`), an empty `modelKey` is **invalid** (`REQUIRED_NONEMPTY`), and a `null` in
   either tier is invalid.
2. **AC-3** — two runs differing in `packVersion`, and separately in `packContentHash` only, both
   produce the mismatch banner and still render the comparison.
3. **AC-4** — a paired-difference interval that includes zero renders the "not distinguishable at
   this sample size" wording, **and the 40/40 vs 34/40 case does not** — that case's paired
   difference excludes zero and the correct verdict is *distinguishable* (`-ml` §3.2; it is the
   worked example that carried the FR-15 amendment, and the one the *old* marginal-overlap rule got
   backwards). §5 test 6 states the same thing; these two must not diverge again.
4. **`stats` reproduces the note's `(a,b,c,d)` regression fixtures to the tolerance the note
   states** — the tolerance is the note's to set, and this plan does not restate it. Plus the two
   contract assertions `-ml` §3.4 calls out by name: `PairedOutcomes.from_units` **raises** on a
   repeated analysis-unit id (Rule 1), and the ρ = 1 identity holds (Rule 5 — when within-cluster
   correlation is 1, effective *n* must equal the cluster count; it is the one assertion that
   catches a squaring error in either direction). **Rule 1 is a backstop, not the mechanism**, and
   v1.3 overstated it: `from_units` only raises if the id it is handed is the *cluster* key, so 48
   conversation ids drawn from 12 scripts are unique and would be accepted (gate finding N-1). What
   closes the clustered-design case is Rule 6 plus §3.3's `sampling` contract, at pack validation —
   which is why DC-5(c) below tests *which key is used*, not merely that something raised.
5. **`min_detectable_difference` is verifiably not a constant, and verifiably not naive.** Two
   tests: it returns different values for different *n*; and — the B-1 detector — **calling it with
   only an item count, for a pack whose sampling unit is clustered, must fail rather than return a
   number.** Whatever shape the note gives that input, the assertion is the same: a resolving-power
   line for a clustered pack cannot be produced from a bare *n*, and `report.py` refuses to render
   one when the required input is absent. This is the test that would have caught v1.1's
   `min_detectable_difference(48)` printing a materially over-optimistic figure for the tool-caller
   pack. **It needs a synthetic clustered fixture**, because under §3.8.4's sizing no shipped pack
   is clustered any more — which is exactly why the guard must be tested rather than assumed
   unreachable: the next conversation pack that raises `replicatesPerScript` reintroduces the case,
   and by then S1 is long closed.

   **DC-5(c) — the fixture must assert *which key* is used as the unit id, not merely that something
   raised.** *(Gate finding N-1. Written to be lifted verbatim into the S1 brief.)* The fixture is an
   in-memory `Sequence[ItemResult]` plus a `PackRef` — S1 has no pack loader, that is S2 — built as
   **12 clusters × 4 rows = 48 rows**, each row's `pairingKey` being a **unique** `(scriptId,
   replicate)` pair, with `PackRef` declaring `pairingKey = ["scriptId", "replicate"]` and
   `analysisUnit = "scriptId"`. The test asserts all three of:

   1. **Identity of the unit ids.** Capture the `unit_ids` argument actually passed to
      `PairedOutcomes.from_units` and assert it is the rows' **`scriptId`** values — the *outermost*
      component of `pairingKey`, 12 distinct values each appearing 4 times — resolved from
      `PackRef.analysisUnit`. Assert the captured argument itself, not a property inferred from the
      outcome.
   2. **Consequence.** `from_units` therefore raises on the repeated unit id.
   3. **A negative control on the guard itself.** Passing the 48 *unique conversation ids*
      (`f"{scriptId}-{replicate}"`) to `from_units` directly is **accepted without error** — which
      proves `from_units` alone does not close this, and that assertion 1 is what does.

   **A test that asserts only 2 passes while testing nothing**: 48 conversation ids are unique, so
   the wrong unit-id choice raises nothing and the fixture goes green on a harness that would
   silently produce an anti-conservative verdict. Assertions 1 and 3 are what make the test real.
5b. **The rendered resolving-power line matches the note's verbatim string.** `-ml` §7.2 carries the
   tool-caller pack's line as four mandatory sentences; the report test asserts against that string,
   parameterised only by `<packId>@<packVersion>`. This is the acceptance surface for §3.9 point 2:
   a line missing the unit, the design effect, the best-case caveat or the conditionality clause
   fails, whatever number it prints.
6. **`armKind` (B-3)** — a `deterministic` fingerprint with no model fields **validates**; the same
   record with `modelKey: "bm25"` added **fails on write** (forbidden field), as does a `model`
   record missing `runtimeName`; and `compare_report` renders a model arm and a deterministic arm
   in one report without ranking two deterministic arms against each other.
7. **Schema versioning (M-4)** — a record written under `benchSchemaVersion: 1` still validates
   after a hypothetical field is added at version 2, and a record declaring version 99 lands in
   `invalid` with `reason == "unknown_schema"`.
8. **`headlineMetric: null` (§3.3)** — a pack fixture with two `verdictMetrics` and a null
   `headlineMetric` renders both verdicts and **no headline**; a fixture omitting the
   `headlineMetric` key entirely fails validation.
9. **`--negative-control` smoke check** — two copies of the same stored run report
   not-distinguishable. Labelled a smoke check in the test's own docstring, because with two copies
   `b = c = 0` by construction and it **cannot fail**: it proves the mode is wired, not that the
   harness is sound. The real negative control is two *independent* runs of the same model and is
   §5 test 19a, an acceptance step.

This stage encodes the **amended** FR-15/AC-4 decision rule (§3.9 point 1) — the paired-difference
interval, not marginal overlap.

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
    def ps(self) -> list[ResidentModel]: ...

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

**CLI:** S2 ships `attest`, `validate`, `run`'s plumbing, and the installed-catalog half of
`models --tested` (§3.6a). `attest` is assigned here and not later because S3's done-condition —
"a stored result with a **complete** fingerprint" — is unreachable until `host.json` exists.

**Done when:** `packs`/`convo`/`tooling` are unit-tested offline against a stub LLM and a fixture
pack; **`validate_pack` enforces the §3.3 `sampling` contract** — a fixture pack declaring
`analysisUnit: "conversationId"` under `scripts: 12, replicatesPerScript: 4` is rejected on the
row-count identity (48 distinct values where 12 are required), one declaring
`analysisUnit` outside its own `pairingKey[0]` is rejected structurally, and one declaring
`replicatesPerScript > 1` is rejected per `-ml` §3.4 Rule 6; `validate_pack`'s AST import check
rejects a fixture pack module that imports outside the
allowlist, and `run` is shown to call it and fail closed (§3.3); `lmstudio` is unit-tested against
recorded JSON payloads, including the §3.6 eligibility gate on the three real catalog entries that
break the naive rule (an `embeddings` model advertising `tool_use` → refused; an entry with **no**
`capabilities` key → admitted; an `llm` with `tool_use` → admitted); `attest` writes a `host.json`
matching §3.4.4's schema and the staleness trip-wire fires when `runtimeVersion` changes; and
**one** `-m live` test confirms `catalog()` returns the real installed models and that `chat()`
surfaces `stats.time_to_first_token`.

**R-1's probe is part of this done-condition, not a note** (§6 R-1 promised it "during S2" and v1.1
gated nothing on it): with a model actually loaded, run `lms ps --json` and record in
`model-bench/docs/HISTORY.md` whether it exposes the load configuration. **If it does,
`kvCacheSetting` moves from operator-attested to auto-captured** and §3.4.2/§3.4.4 are updated in
the same change; if it does not, the recorded negative result is what closes R-1. Either outcome
satisfies the condition; silence does not.

### S3 — `embedder` pack + `refresh_golden.py` (first end-to-end result)

**Create:** `model-bench/scripts/refresh_golden.py` (one-way, human-invoked importer),
`packs/embedder-graphrag-retrieval/` (§3.8.1), `modelbench/scoring/retrieval.py`.

Sequenced third on purpose: it is the cheapest path to a complete real run — the golden data
exists, the mechanism is pure arithmetic, and there is a pinned prior figure to sanity-check
against. It proves the S1/S2 core end to end before the expensive packs are built on it.

**Done when:**

1. A real run against `text-embedding-qwen3-embedding-0.6b` produces a **stored result with a
   complete `model` fingerprint** (which requires `attest` from S2 to have written `host.json`).
2. The BM25 arm is stored as its own `armKind: "deterministic"` record sharing that run's
   `sessionId`, and `compare` renders the two arms in one report with the deterministic one
   labelled (§3.4.1, §3.8.1).
3. `test_metrics_agreement.py` passes over all 20 transcribed cases (§3.1 point 2). *(v1.1's
   done-condition asked for "byte-identical output" from running falkor-chat's own `test_metrics.py`
   fixtures through the copied implementation — that mechanism does not exist; §3.1 point 2 explains
   what replaced it and why.)* `scripts/refresh_golden.py --check-origins` runs clean, and
   `metrics_agreement.json` carries its `sourceGitSha` and `sourceSha256`.
4. `corpus.embeddings.json` (the 121 corpus vectors) is written into the pack and the ranking path
   is shown to reproduce identical rankings from the two fixed vector files with no live embedding
   call at all (§3.8.1) — that is what actually isolates "is my ranking code right".
5. **The harness self-check is run and its outcome recorded — it is a diagnostic, never a gate**
   *(stakeholder decision, 2026-09-02)*. The same model on the same corpus and queries is expected
   to land near the copied baseline; **a below-expectation result does not block S3.** What is
   required is that the deviation and the investigation of it are written into
   `model-bench/docs/test-reports/`, naming which of the known causes (wrong prefix, unnormalized
   vectors, truncated corpus — `-ml` §5.4) were checked and what was found. It is not a metric
   target in either direction: hybrid-ANN vs exact-vector-only differ in two directions at once, so
   a *disagreement* of either sign is uninterpretable and must not be "explained". v1.1 left this
   silent, which is a done-condition that can be argued either way at the moment it matters most.

### S4 — `guard-judge` and `nlq-generator` packs

**Create:** `packs/guard-judge-understanding/`, `packs/nlq-structured-query/`,
`modelbench/scoring/{classification,extraction}.py`, the pack-local `tools/exec.py` spec executor.

Both are single-call-per-item roles that reuse the S1 statistics unchanged; grouping them keeps the
new surface to two scorers plus one small executor. Do `guard-judge` first (no executor at all).

**Done when:** both packs run end to end against one model; **every item is stamped
`answerable: true|false`** by running its hand-written reference spec through the executor
(§3.8.3 — this tests the executor, not the model, and is the step that keeps unanswerable items out
of the accuracy denominator); the executor's **validation** half is covered too — a malformed spec,
a schema violation and a wrong answer each land in their own count, in stdlib only (§3.8.3's cost
note); the two `conflicting-facts` items score by subset containment, not set equality; `validate`
passes on both packs; and the guard-judge report shows the two class-conditional verdict metrics
side by side with **no headline number and no pooled 85-item figure anywhere in it** (§3.8.2).

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
has at least one synthetic trace that moves it and one that does not; the **outcome-vector
comparison the determinism probe needs** exists as a pure function over two `ConversationTrace`s and
is unit-tested on identical and one-turn-differing pairs (§3.8.4 — no new statistics, and S6 must
not be the first place it runs); denominators and `n/a` tallies
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
2. Extend to **4 distinct scripts per shape, 12 in total, each run once** (§3.8.4's sizing, revised
   by the 2026-09-02 stakeholder decision) — and cover the FR-8 failure kinds §8.1's set was not
   designed to exercise: in particular "stopping when done", "final reply matches the tool result",
   and turns where **no** tool is required (the restraint count). **This is where the sampling
   decision's cost lands** — 12 scripts to author and human-verify instead of 12 replicated runs of
   4 — and it is why S6 remains the long pole. Also draft the ~20 labelled replies the
   prose-vs-native detector is scored against.
3. **Human verification of every turn's expectations** (FR-19): each `expect` block is checked by a
   person against the simulated environment's actual behavior, not against a model's output.
   `provenance.verifiedBy` is filled per conversation.
4. Declare `sampling.determinismProbeScripts` (one shape-A, one shape-B) and **run the determinism
   probe** — those two scripts a second time, once per model, in the same session, outside `n`
   (§3.8.4). Record `determinismProbe` in each run and let it set `basis`.
5. Run the known-answer validation: `qwen/qwen3-4b-2507` vs `mistralai/ministral-3-3b`, and compare
   the per-turn profile to §8.2's recorded finding.

**Done when:** step 5 is **run and its outcome recorded** in `model-bench/docs/test-reports/`;
**step 4's determinism probe has run for both models and its result is recorded, with `basis` set
from it** — an identical outcome vector on both probe scripts leaves `basis: "by-construction"` and
McNemar deciding; anything else records `basis: "assumed"`, and the report must then show the
cluster-bootstrap CI as the decision with McNemar labelled anti-conservative (`-ml` §3.4 Rule 4).
Both outcomes satisfy the condition — what does not is the probe not having run. AC-1 holds on real
output (per-failure-kind **and** per-turn-position, with no blended headline anywhere in the
report); and the pack validates, including `H ≤ min(script length)`, the `sampling` row-count
identity and `analysisUnit` membership (§3.3).

**If the contrast does not appear**, the stage is still completable: R-3's bisect is executed and
its result recorded, and the pack ships flagged `known-answer validation: not reproduced` in every
report it generates until it is. *(v1.1 made the done-condition "step 4 reproduces the documented
contrast" while R-3 itself states the reconstruction "will not be turn-for-turn identical" and the
contrast may not appear — a gate on an empirical outcome the plan says may not occur, which under
deadline pressure is resolved by editing the reconstruction until the contrast appears. That is
fitting the instrument to the expected answer, and it is the one thing a measuring instrument must
not be built by.)*

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
covers containment, exclusion and correct abstention; **the format scorer covers all three declared
constraints** — `maxWords`, `mustBeSingleParagraph`, `forbiddenPatterns` — read from pack data with
per-item override, each a separate count never pooled with grounding (§3.8.5; v1.1 shipped two of
FR-21a's three layers and this closes the third); the report prints `groundingRate` as
`headlineMetric` alongside the format counts and the standard latency block; and the report says in
words that reply *quality* is not measured by this pack.

### S8 — Documentation and close

`model-bench/README.md` (how to run, what a pack is, how to add one, the three non-features, the
exact-cosine scope note from §3.8.1), `model-bench/AGENTS.md` (working context: layout, live-test
convention, the FR-23 rule stated as a hard rule for future agents, the operator-attested
fingerprint fields and why), `model-bench/docs/HISTORY.md` (an entry per stage, first written at S0),
`model-bench/docs/BACKLOG.md` **re-checked and extended** — it was already seeded at S0 with the two
deferred items named in §7 (the judged reply-quality layer, FR-21a; the +22 harder retrieval
queries), so S8 adds whatever R-1's S2 probe and the S3/S6 test reports left open and removes
anything since delivered. Root `AGENTS.md` rows added in S0 are re-checked against what actually
shipped. `README.md` additionally states the two things §3.4.3 and §3.6a make into user-visible
contracts: that a `benchSchemaVersion` bump is a deliberate act with a migration decision attached,
and the closed exit-code set.

---

## 5. Test strategy

The harness is a measuring instrument, so the test strategy has an unusual centre of gravity: **the
tests that matter most are the ones that prove the instrument reports honestly when the data is
bad**, not the ones that prove it reports a number when the data is good.

**Unit (default suite, network-free, `pytest -q`)**

1. `fingerprint.validate()` — one test per required field: blank it, assert it is named. Plus the
   three tier cases (§3.4.2): `residentModelsAtStart: []` valid, `modelKey: ""` invalid, `null`
   invalid in either tier; and the two `armKind` cases (§3.4.1): a `deterministic` record with no
   model fields validates, the same record with `modelKey` added fails on a *forbidden* field.
2. `results.store()` refuses an invalid fingerprint; there is no bypass flag (assert the absence by
   API surface, not by comment).
3. `results.load_history()` quarantines: a hand-edited record with a missing attested field, a
   record declaring a **future** `benchSchemaVersion`, a truncated JSON file → all three appear in
   `invalid`, none in `valid`; **and a record written under an older, known `benchSchemaVersion`
   validates against its own contract and appears in `valid`** (§3.4.3 — the FR-3 case v1.1's
   "older-schema records are excluded" would have silently deleted). **(AC-2)**
4. `packs.content_hash()` — stable across path order, changes when any byte in any pack file
   changes, unchanged when `PROVENANCE.md` changes.
5. `compare_report` — version mismatch banner on differing `packVersion`; **also** on identical
   `packVersion` with differing `contentHash`; comparison still rendered in both cases. **(AC-3)**
6. `stats` — Wilson against published worked examples, at the note's pinned `_Z_95` and **not**
   `1.96`; the paired instruments against the note's `(a,b,c,d)` regression fixtures, to the
   tolerance the note states; the exact test's small-sample floor as the note tabulates it; a
   paired-difference interval containing zero produces the "not distinguishable at this sample size"
   wording, and the 40/40 vs 34/40 case does **not** — the regression test that pins the amended
   rule; the instruments-disagree case renders the both-components prose. The resolving-power
   functions are computed from the run's own sampling structure and asserted never to return a
   constant, **and asserted to refuse a bare item count for a clustered pack** (S1 done-condition 5,
   the B-1 detector), and the rendered line is asserted against `-ml` §7.2's verbatim four-sentence
   string. `PairedOutcomes.from_units` raises on a duplicate unit id, and the ρ = 1 / effective-n
   identity holds (`-ml` §3.4 Rules 1 and 5). Every number in this test comes from the `-ml` note;
   none is a literal in this plan. **(AC-4)**
7. `scoring/toolcalls` — the synthetic-trace matrix from S5, one per failure kind, plus denominator
   edge cases (turn where no tool was required — the restraint count; turn where the model emitted
   nothing at all; conversation that ended early) and the **laundering test**: a trace that
   collapses at turn 2 must not out-score one that reaches turn 8 on any conditional count.
7b. The note's cluster-aware surface — a fixture where all observations within a cluster are
   identical must resolve to a far smaller effective sample than the raw count, and independent
   observations to roughly the raw count. The exact functions and their expected values are the
   note's; this test exists to prove `stats.py` implements them rather than a naive substitute.
8. `scoring/retrieval` — recall@k/MRR/precision@k/score-separation on hand-built ranked lists with
   known answers, including multi-relevant items, fewer than *k* results, and zero relevant found.
9. `scoring/extraction` — the scalar/set shape rules and numeric epsilon.
10. `convo.assemble` — the three `historyReplay` modes produce the documented message sequences;
    `representToolSchemasEachTurn=false` really does drop the schemas after turn 1.
11. `test_metrics_agreement.py` — all 20 transcribed cases from §3.1 point 2, reading only
    `model-bench/tests/fixtures/`, including the two `ValueError` cases. A test that skips or
    xfails any case is a failing test: the case count is the guarantee.
11b. `report.py` structural refusals — a pack fixture with `headlineMetric: null` renders both
    verdict metrics and no headline; a manifest omitting the `headlineMetric` key fails
    `validate_pack`; a metric outside `verdictMetrics` always renders with the `exploratory` label.
12. **Pack integrity, per pack:** unique ids, required fields, per-item provenance present (with
    `originGitSha`), paraphrase rule for retrieval-style packs, every `expect` block referring to a
    tool the pack's own `schemas.json` declares, the `metrics` block well-formed (§3.3 — including
    a non-null `headlineMetric` that is not a `verdictMetrics` member being rejected), `H ≤ min(
    script length)` for the tool-caller pack, **`replicatesPerScript > 1` rejected while only the
    one-level `cluster_bootstrap` exists** (`-ml` §3.4 Rule 6), the **`sampling` contract** (§3.3 —
    `analysisUnit == pairingKey[0]`; the row-count identity over `analysisUnit`'s values, with a
    fixture declaring `analysisUnit: "conversationId"` under `scripts: 12` rejected for having 48
    distinct values where 12 are required), and `validate_pack`'s AST import check rejecting a pack
    module that imports outside stdlib + `modelbench.tooling`.
12b. **The determinism probe's wiring** (§3.8.4) — the outcome-vector comparison is exact on
    identical traces and localises the first differing turn; and `runner` sets `basis` correctly in
    all four cases: probe ran and identical → `"by-construction"`; probe ran and differed →
    `"assumed"`; **probe did not run → `"assumed"`** (the fail-safe, asserted explicitly); and
    `replicatesPerScript > 1` → `"assumed"` regardless. A `basis` of `"assumed"` must be shown to
    move the decision off McNemar and onto the cluster-bootstrap CI (`-ml` §3.4 Rule 4), with
    McNemar's p still printed under its anti-conservative label.

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
19. **The instrument-validation pair.** (a) **Negative control — two *independent* runs of the same
    model** in the two arms, on the tool-caller pack (not two copies of one record: that is S1's
    smoke check and cannot fail). It must report *not distinguishable*, with discordant counts
    roughly equal and the difference interval centred on zero — the note names this the highest-value single test in the harness, because it
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
and is arguably the more reproducibility-relevant version anyway. **The probe is in S2's
done-condition**, not left as prose here: run `lms ps --json` against a *loaded* model, record the
outcome in `model-bench/docs/HISTORY.md`, and if the load configuration is exposed, move
`kvCacheSetting` from attested to auto-captured in the same change. A mitigation nobody is gated on
executing is a mitigation nobody executes.

**R-2 — FR-11's "peak RAM at the measured settings" is best-effort, not exact (medium).** There is
no RAM endpoint. What is available: `lms ps --json`'s loaded-weights size, `lms load --estimate-only`'s
pre-load estimate, and Windows-side process working set sampled through `powershell.exe`. The design
records them as three separately named fields rather than one authoritative "peak RAM" (§3.6), and
`README.md` states the method. A cross-WSL `powershell.exe` sample is also the most fragile thing in
the tool; it degrades to "not captured" without failing the run — **but note that makes it the one
FR-7-adjacent field that is not enforceable**, which is why it lives in FR-11's speed result rather
than in the fingerprint. It is also the most *invasive*: measured at 0.18–0.54 s per invocation
across the WSL boundary against ~1.3 s per turn, so §3.6 confines sampling to the gaps between
turns and stamps every sample with a timestamp. An implementer who "simplifies" this into an
interval timer silently inflates the headline latency of every run.

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

**R-12 — The tool-caller pack's *n* is small, and the report must say so rather than imply
otherwise (medium, accepted by decision).** The 2026-09-02 sampling decision buys 12 independent
conversations where v1.1 bought 48 clustered ones (§3.8.4). The precision is not worse — `-ml` §4.5
put the old design's effective *n* near 12 anyway — but it is now *visible*, and the
resolving-power line will print a large number. Accepted deliberately: an honest wide interval is
worth more than a narrow one that a design effect would have had to walk back. The mitigations are
all in output rather than in design — the resolving-power line computed from the real structure
(§3.9 point 2), `replicatesPerScript` and `temperature` printed beside every conversation-level *n*,
and the report naming the magnitudes it can and cannot resolve. The reversal trigger is explicit:
if a real comparison lands inside the unresolvable band and the answer actually matters, the fix is
**more distinct scripts** (a pack version bump, S6's authoring cost again) — never more replicates
at temperature 0, which is what produced the problem.

**R-13 — `_percentile`'s definition is undecided, and latency comparisons will be built on it
(low, open — `data-scientist`'s call).** *(New in v1.5, from the S1 gate's open question 2.)* S1
ships two copies of a nearest-rank `_percentile` (`stats.py`, `results.py`). For the bootstrap's
B = 10 000 draws the choice of definition is invisible; for `latencyMsP95` over a handful of items
it is not, and FR-11's p50/p95 are compared across runs stored months apart, so the definition is
effectively frozen the moment the first latency figure is stored. Nothing in this plan or the `-ml`
note pins it. **S2 must not store a p95 until the definition is chosen and written into the note**
(nearest-rank vs linear interpolation, and whether the two call sites may differ); the plan does not
choose it here because a percentile definition is method, and §7 rule 2 gives method to the note.

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
  (`-ml` §4.5). §3.8.4's sizing satisfies it — **12 distinct scripts, 4 per shape, one run each**,
  per the 2026-09-02 stakeholder decision. *(FR-22a's own text still cites the plan's earlier
  4 × 4 × 3 sizing as an illustration; the requirement it states — several distinct scripts per
  shape — is satisfied a fortiori, and the illustrative clause is `tico`'s to refresh, not this
  plan's to contradict silently. Flagged, not amended.)*

**Nothing in this plan is blocked on a stakeholder answer.** The gate's three open questions were
all closed on 2026-09-02 — tool-caller sampling (§3.8.4), `guard-judge`'s absent headline (§3.8.2),
and what S3 counts as done when the self-check fires (S3 done-condition 5). Two items remain, both
verify-during-implementation and both now gated in a done-condition rather than left as prose:
R-1's S2 probe of whether `lms ps --json` exposes the KV-cache setting on a loaded model, and S6's
known-answer validation, which is recorded either way (S6, M-13's exit).

**One thing this plan no longer claims.** §3.1 point 2's duplication mitigation is weaker than v1.1
stated it, and it says so in its own text rather than here. D1 — copy the data, clean-build the
code — still stands, and on a stronger argument than the mitigation: the two golden sets *must*
diverge, and the two implementations' numbers are never compared to each other by design.

---

## 7. Ready to implement

**Plan:** `docs/plans/small-model-benchmarking.md` (this document).
**Method note:** `docs/plans/small-model-benchmarking-ml.md` (`data-scientist`) — **the single
source of truth for every formula, constant, threshold, tolerance, denominator, sample size and
verdict string in this feature**, plus the deferred judge design. It is not optional background:
`modelbench/stats.py` implements it and this plan cites it. Read it before starting S1 — in
particular **§3.4, six binding rules that are `stats.py`'s contract**, and **§7.2's verbatim
resolving-power string**, which is a test target.

**Version pairing:** this plan **v1.5** is aligned to the note **v1.5**. They share one vocabulary
(`verdictMetrics` + `headlineMetric`, the plan's names), one `guard-judge` metric pair
(`falseAdvanceRate` + `falseSuspendRate`, the note's), one `H` contract (pack-declared and validated
`H ≤ min(script length)`, the plan's), and one analysis-unit rule (the outermost component of
`pairingKey`, the note's). A future revision of either that changes a name, a number, a signature or
a contract must sweep the other in the same pass — the Pass 1 gate's finding was silent divergence
between exactly these two documents, and the fix is not a one-time correction but a standing
obligation.

**How to resolve an apparent disagreement — and why this is not a simple precedence rule.** v1.3
said "where the two ever appear to disagree the note is right". That sentence caused a defect. When
the plan moved `H` from derived to declared-and-bounded (M-11) and the note's §4.6 still carried the
old derived definition, the precedence rule did not resolve a conflict — it **propagated the stale
clause**, converting a fixed finding back into a live one (Pass 2, N-3). A blanket precedence rule
launders staleness with exactly the authority it was given to settle disputes, and the more
trustworthy the senior document, the more efficiently it does so. So:

1. **A disagreement is presumed to be staleness, not a conflict.** Find which document changed last
   on that point and reconcile both; do not apply precedence to a clause one side has already
   superseded. Both documents carry version numbers and dated revision lines for this purpose.
2. **Precedence applies only when both sides are current, and it is split by ownership, not by
   seniority.** **The note owns method** — formulas, constants, thresholds, tolerances,
   denominators, sample sizes, instrument selection, `stats.py`'s signatures. **This plan owns the
   pack-manifest contract and the harness surface** — field names, validation rules, `H`'s
   declaration semantics, the CLI, the result schema. `H` is the worked example: the *statistics* of
   `cleanThroughTurnH` are the note's, the *rule that `H` is declared and bounded* is the plan's.
3. **Neither document may resolve a disagreement by editing the other.** Raise it; the owner fixes
   it in their own file. That is what kept N-3 a one-clause fix rather than a merge conflict between
   two agents editing in parallel.
4. **The same discipline applies *inside* each document — v1.4's three rules did not reach there,
   and both of v1.5's defects lived in that gap.** *(New in v1.5.)* Appendices, recap tables and
   enumerations are **derived surfaces**: where one disagrees with the section that owns the
   contract, the owning section is right by construction and the derived surface is stale — never a
   second opinion to be weighed, and never a reason for an implementer to build the weaker of the
   two. §3.3 owns the `sampling` contract, so Appendix A's `PackRef` was simply behind it; §3.4.2
   owns the model field set, so §3.4.1's hand-typed forbidden list was behind it the moment §3.6's
   audit fields joined that set. Two consequences, both applied in v1.5:
   - **A change to an owning section sweeps its derived surfaces in the same pass** — rule 1's
     standing obligation, turned inward. The sweep is cheap and the omission is invisible: both
     defects passed two review passes, because a reader checks an enumeration against its own prose
     rather than against the set it claims to complement.
   - **Where a derived surface can be a *derivation* rather than a transcription, it must be.**
     §3.4.1's forbidden set is now a set difference over §3.4.2's names, and `PackRef`'s
     `analysisUnitIndex` is a property over `pairingKey`, precisely so there is nothing left to keep
     in sync. A list maintained by hand beside its own stated intent has already drifted once here;
     state the rule and let the list follow from it.

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

**Nothing is blocked.** The four requirement defects this design pass raised were settled on
2026-09-02 (requirements commit `afe4aef`) and the plan is built to the amended text throughout
(§6). The gate's three open questions were settled the same day and are folded in: **12 distinct
tool-caller scripts × 1 run at temperature 0** (§3.8.4), **`guard-judge` has no `headlineMetric`**
(§3.8.2, with §3.3 making a headline-less pack a first-class case), and **S3's self-check is a
diagnostic, never a gate** (S3 done-condition 5).

**S0 is delivered** (commit `0522ffd`) and **S1 is delivered** (commit `ab91419`, gated at `d6c4997`
— verdict *needs changes*: one blocker, six majors, all dispatched as fixes against the shipped
code, not as design changes). Read S0's and S1's own text before starting — they record what
shipped rather than what v1.1 asked for, including the one real smoke test and the working-directory
rule that every done-condition in §4 depends on. **S2 is the next stage**, and it inherits three
things from the S1 gate that this plan states rather than leaves in the review: `RunResult`'s two
new fields must be set by the runner with no default (§4 S1), `validate_pack` calls the existing
`packs.check_sampling_contract` rather than re-implementing the rule, and `_percentile`'s definition
(nearest-rank vs interpolated) must be decided and written down before latency figures start being
compared across runs — that last one is a `data-scientist` call, open as §6 R-13.

Two items are already recorded in `model-bench/docs/BACKLOG.md` (seeded at S0, re-checked at S8):
the **deferred judged-quality layer** for `chat-responder` (design preserved in §3.8.5 and
`-ml` §6.2) and the **+22 harder retrieval queries** that would lift the embedder pack's recall
ceiling (§3.8.1).

---

## Appendix A — Types named in this plan

Named above and defined here so an implementer is not inferring them from usage. Everything in
`stats.py` beyond this list — in particular the cluster-aware surface — is the `-ml` note's.

**This appendix is a derived surface, not a second contract** (§7 rule 4). Where a row disagrees
with the section that owns the type, the section is right and the row is stale — that is exactly how
v1.4's five-field `PackRef` survived §3.3's `sampling` block. A change to an owning section sweeps
its rows here in the same pass.

| Type | Module | Shape |
|---|---|---|
| `FieldSpec` | `fingerprint` | `NamedTuple(tier: Literal["nonempty","present"])` — §3.4.2 |
| `FieldProblem` | `fingerprint` | `NamedTuple(field: str, reason: Literal["absent","empty","null","forbidden","unknown"])`. **`unknown` is v1.5's fifth value**, for a *discriminator this build cannot interpret* — an unrecognised `armKind`, or a `benchSchemaVersion` from the future (§3.4.3). Neither is absent, empty, null or forbidden, so the four-value set would have forced a mislabel; it is the field-level counterpart of `InvalidRecord.reason == "unknown_schema"`. |
| `Fingerprint` | `fingerprint` | frozen dataclass, §3.4.1–§3.4.2; `armKind` discriminates |
| `ItemResult`, `RunResult`, `InvalidRecord`, `Aggregates` | `results` | as given in S1 |
| `PairedOutcomes`, `ResolvingPower`, `Verdict`, `BootstrapResult` | `stats` | **`-ml` §3.4's, verbatim** — not restated here. `PairedOutcomes.from_units` is the only constructor and raises on a repeated analysis-unit id; `resolving_power()`'s inputs are keyword-only with no defaults. (v1.2's `PairedResult` was this plan's own invention and is withdrawn.) |
| `PackRef` | `packs` | `NamedTuple(packId, packVersion, contentHash, role, metrics, pairingKey: tuple[str, ...], analysisUnit: str)` — the identity triple, what the report must print, and §3.3's two `sampling` declarations. **The last two are v1.5's**: `report.py` resolves the analysis-unit id from `analysisUnit` and **no call site chooses it** (§3.3, DC-5(c)), so without them the resolution has nowhere to come from; the five-field form predated §3.3's v1.4 `sampling` block. Derived, not stored: `analysisUnitIndex = pairingKey.index(analysisUnit)`, which is `0` whenever `check_sampling_contract` has passed. |
| `Pack` | `packs` | frozen dataclass, S2 |
| `ModelInfo` | `lmstudio` | one catalog entry, verbatim: `id, type, publisher, arch, compatibility_type, quantization, state, max_context_length, capabilities?, loaded_context_length?` |
| `ChatResult` | `lmstudio` | `message, tool_calls, toolCallForm, stats, model_info, runtime, usage, wallClockMs` |
| `EmbedResult`, `LoadResult`, `ResidentModel` | `lmstudio` | vectors + dimension; timed load outcome; one `lms ps --json` row |
| `PromptConfig` | `convo` | the manifest's `prompt` block, parsed: `systemPrompt, toolSchemas, historyReplay, representToolSchemasEachTurn, historyTurns, temperature, maxTokens` |
| `Turn`, `Conversation` | `convo` | one scripted turn (`seq, user, expect`); one row of `conversations.jsonl` |
| `ConversationTrace` | `convo` | per-turn `(messages_sent, ChatResult, dispatches, env_state, wallClockMs)` |
| `DispatchRecord` | `tooling` | `(name, rawArguments, parsedArguments, returnValue, timestamp)` — FR-10's ground truth |
