# model-bench — Formats

> **Status:** active · **Owner:** `coder` · **Tracks:** —

> **How to read this document.** It is part of model-bench's live spec, read by lookup: the shapes
> of every file the harness reads or writes — the pack, the stored run record, the fingerprint,
> the candidate, `host.json` and the small operator files — and the rules that make a stored
> record valid. Behavior (what a command or surface does with these) is in
> `model-bench/docs/DESIGN.md`. Section numbers are stable and cited from the code and the
> context files.

## Index

| § | Topic |
|---|---|
| 1 | Pack directory, manifest, per-role data, the pack-module seam, the decision-point manifest |
| 2 | The stored run record: `RunResult`, `ItemResult`, timing, latency, aggregates, metrics |
| 3 | The fingerprint: discriminators, profiles, field tiers, forbidden fields |
| 4 | Schema versioning, the store and read rules, `index.csv` |
| 5 | The candidate file and the fit manifest |
| 6 | `host.json` |
| 7 | The decision contract: golden vectors |
| 8 | Operator files: `footprints.json`, the rank-report marker, report names, the sweep plan and log |

## 1. Packs

### 1.1 Directory and identity

A pack is the directory `packs/<pack-id>/`, declared by `pack.json`. Beside it: the data files the
manifest names, optional `prompts/` and `tools/`, and `PROVENANCE.md`. By convention the directory
name is the `packId`; `compare` and `rank` read history by the manifest's `packId`, not by the
directory. The pack's identity is its `packId`, `packVersion` and the **content hash** (`DESIGN.md`
§5.1: every byte of the directory except `PROVENANCE.md` and `__pycache__`). `PROVENANCE.md` names,
per copied file, the origin path, the origin commit, the copy date and what changed on copy; it is
documentation: `validate_pack` and every run path ignore it, and only the importers' version gate
reads it.

### 1.2 `pack.json`

| Key | Required | Meaning |
|---|---|---|
| `packId`, `packVersion`, `role` | yes (the loader refuses without them) | identity and one of `roles.ROLES` |
| `scorer` | to run | module name under `modelbench/scoring/`; fixed by the role on a decision-point pack |
| `environment.requires` | to run | list of tokens: exactly one of `lmstudio-chat` / `lmstudio-embeddings`; a decision-point pack adds `lmstudio-decisions` |
| `prompt` | chat-surface roles | `systemPrompt` (a path to a file under the pack), `toolSchemas` (a path to a JSON array of function schemas, or `null`), `temperature`, `maxTokens`; on `tool-caller` also `historyReplay`, `historyTurns`, `representToolSchemasEachTurn`, `maxIterationsPerTurn` (required there and forbidden elsewhere) |
| `data` | yes | map of data-file keys to paths under the pack: `items` (single-call roles), `conversations` and `prosePseudoCallCalibration` (`tool-caller`), `corpus`, `corpusEmbeddings`, `queryEmbeddingsFixture`, `baseline` (`embedder`), `tables`, `schema` (`nlq-generator`), `catalog` (`tool-caller`) |
| `sampling` | yes | `seed` (int, required), `pairingKey` (non-empty list), `analysisUnit` (equal to `pairingKey[0]`); on `tool-caller` also `scripts`, `replicatesPerScript`, `determinismProbeScripts` |
| `metrics` | yes | `verdictMetrics` (non-empty list), `headlineMetric` (key required; `null` or a member of `verdictMetrics`); role extras: `recallAtK.ks` (`embedder`), `cleanThroughTurnH.H` (`tool-caller`) |
| `tools` | tool packs | `module` (a `.py` path inside the pack), `entrypoint`, `schemas` |
| `embedding` | `embedder` | `queryPrefix`, `documentPrefix`, `bm25.k1`, `bm25.b`, `bm25.stopwordsFile` |
| `format` | `chat-responder` | pack-level `maxWords`, `mustBeSingleParagraph`, `forbiddenPatterns` (regexes, matched per line) |
| `decisionPoint`, `labels` | decision-point packs | §1.5 |
| `schemaVersion`, `description`, `provenance` | carried | read by no module |

### 1.3 Per-role data

Every JSONL file is one JSON object per non-blank line, in file order.

| Pack role | File | Row shape |
|---|---|---|
| `embedder` | `queries.jsonl` | `itemId`, `query`, `relevantDocIds`, `topic`, `targetText`, `rationale` |
| | `corpus.jsonl` | `docId`, `text`, `topic` |
| | `corpus.embeddings.json` | `cacheKey` {`model`, `quantization`, `documentPrefix`, `corpusSha256`}, `generatedAt`, `normalized`, `vectors` {`docId`: float list} (raw vectors, reused only on a cache-key match) |
| | `bm25_stopwords.txt` | one lowercase token per line, `#` comments |
| | `retrieval_baseline.json`, `golden_retrieval.embeddings.json` | copied reference data (`recall_at_5`, `recall_at_10`, `mrr`, `n`; per-query vectors); read by tests and `refresh_golden.py`, not by a run |
| `guard-judge` (`guard-judge-understanding`) | `items.jsonl` | `itemId`, `tier` (`clear_suspend`, `clear_advance`, `boundary`), `path` (`understanding` or `turns`), `condition`, `understanding` (object), `turns` (list), `expected` (bool), plus audit fields `r1_probe`, `label_rationale` the prompt never reads |
| `nlq-generator` | `items.jsonl` | `itemId`, `dataset` (`catalog` or `knowledge_base`), `question`, `shape`, `expected` (`{type: scalar, value}`, `{type: set, values}` or `{type: not_found}`), `rationale`, `answerable` (bool, required) |
| | `tables.json`, `schema.json` | per dataset, the in-memory rows by label and the declared property types (`str`, `int`, `float`) |
| | `reference_specs.json` | per `itemId`, the reference spec `--stamp-answerability` executes |
| `chat-responder` | `items.jsonl` | `itemId`, `question`, `context` (list of passages), `mustContain`, `mustNotContain`, `mustAbstain`, optional `format` override, `provenance` |
| `tool-caller` | `conversations.jsonl` | `scriptId`, `shape`, `replicate`, `description`, `turns` [{`seq`, `user`, `expect`}], `provenance`; `expect` = {`toolRequired`, `tool`, `args`, `finalReplyMustContain`, `finalReplyMustNotContain`} |
| | `prose_calibration.jsonl` | `text`, `isPseudoCall` |
| | `catalog.json` | the simulated storefront's products |
| decision-point packs | `items.jsonl` | §1.5 |

An item's optional `provenance` is an object carried in the loaded item and read by no scorer, and
it is excluded from every prompt. Its keys differ per pack: a `chat-responder` item's are
`draftedBy`, `verifiedBy` and `corpusVersion`; a `tool-caller` script's are `basedOn`, `draftedBy`
and `verifiedBy`. No other shipped pack's items carry one. `validate` ignores it (`DESIGN.md` §5.2).

### 1.4 The pack-module seam

A pack may ship executable Python (`tools.module`), constrained to the standard library plus
`modelbench.tooling` (`DESIGN.md` §5.2 axis 4).

- A **`tool-caller` pack's** `tools.entrypoint` (`build_environment`) returns a fresh object
  satisfying `ToolEnvironment`: `schemas()` (the function-schema list), `dispatch(name, arguments)`
  (appends **exactly one** `DispatchRecord` to its own trace per call), `trace()` (every dispatched
  call so far, stable order, never dropping or reordering an earlier entry) and `state()`. A
  `DispatchRecord` is `name`, `rawArguments` (what the model sent, JSON-parsed), `parsedArguments`
  (what the environment made of them after its own unit and boundary handling), `returnValue` and a
  UTC ISO-8601 `timestamp`. Tool schemas may carry a `boundaryRule` on a parameter
  (`confusedWith`, `inclusive`), which the scorer reads for argument decomposition.
- An **`nlq-generator` pack's** `tools/exec.py` exposes `validate_structure`,
  `validate_against_schema`, `compile_and_execute`, `MalformedSpecError` and `SchemaViolationError`.
  A spec is an object with keys `matches` (list of `{var, label, filters}`, a filter being
  `{property, op, value}` with `op` in `=`, `<>`, `<`, `<=`, `>`, `>=`), `returns`, `order_by`,
  `order_dir` (`ASC`/`DESC`) and `limit`. Identifiers follow a fixed grammar; unknown keys are
  refused.

### 1.5 The decision-point manifest

Additional keys on a decision-point pack (`DESIGN.md` §7.1):

- `decisionPoint` = {`answer`, `incumbent`}: `answer` is `{type: predicate}` or
  `{type: choice, labels: [...]}`; `incumbent` is a model id (speed-up) or `null` (new ability).
- `labels` = {`source: llm-unreviewed`, `inputs: synthetic`, `labeler` {`provider`, `model`,
  `surface`, `effort`}, `generator` {same}}. `provider` and `model` are required strings; the
  other two keys are carried.

An item row carries, beyond the role's input fields: `itemId`, `expected`, `origin` (`synthetic`
on `eval`), `genRunId`, `genRound`, `seedId`, `labelVotes` (counts summing to 3, keyed by exactly
the answer space written as strings — `"true"`/`"false"` lowercase on a predicate point, each label
on a choice point, zeros included), `stratum` (`eval` or `side-check`), and `threadPosition` on
message type. The role's input fields (`INPUT_FIELDS_BY_ROLE`): guard `condition`,
`understanding`, `turns`, `path`; join-in `agentName`, `priorTurns` (list of thread rows),
`newMessage` (a row); message type `text`, `priorTurn` (`{speaker, text}` or `null`).
A guard pack's `side-check` rows are the 85 rows of the `guard-judge-understanding` pack, with `tier` and `path`.
The hand-off prompt is `prompts/<point>-handoff.md` (the guard's is `prompts/judge.md`, a
byte-copy of the `guard-judge-understanding` judge prompt).

### 1.6 What the dataset importer writes

`scripts/import_decision_pack.py --role <r> --dataset <dir> --pack-version <v>` reads
decision-models' `generation.json` (the `labeler` and `generator` blocks, `labelBatchCap`),
`splits.json` (`rngSeed`, `eval`, `train`, `calibration`; **only `eval` is read**, each `itemId`
once) and the items file, and writes into `packs/<r>-synthetic/` exactly `pack.json`, `items.jsonl`
and `PROVENANCE.md` by plain overwrite. Each eval item keeps `itemId`, the role's input fields,
`expected`, `origin`, `genRunId`, `genRound`, `seedId`, `labelVotes` (and `threadPosition`), with
`stratum` written as `eval`; any other source field is dropped. `prompts/` is committed pack data
that the importer never writes, and a missing pack directory or prompt is refused. The pack must
pass `validate_pack` in a staging copy first, the source files must be committed (their git SHAs go
into `PROVENANCE.md`), and an unchanged `--pack-version` is refused. Exit `0` imported, `1` refused,
`2` bad arguments.

## 2. The stored run record

`results/runs/<runId>.json` is one `RunResult`, UTF-8 JSON, sorted keys, two-space indent, a
trailing newline.

### 2.1 `RunResult`

| Field | Meaning |
|---|---|
| `runId` | a bare file name: `<packId>-<armKey>-<startedAt>`, `/` replaced by `_`; the deterministic arm's is `<packId>-bm25-<startedAt>` |
| `sessionId` | the `--session` value or `null`; **top level, never inside `fingerprint`** |
| `role`, `armKind` | the pack's role; `model` or `deterministic` |
| `fingerprint` | §3 |
| `items` | list of `ItemResult` |
| `aggregates` | one aggregate object (§2.4), discriminated by `kind` |
| `designEffect` | float, at least 1.0; `1.0` on every run the harness produces (`DESIGN.md` §3.3, §3.4, §7.3) |
| `basis` | `by-construction`, `measured` or `assumed` |
| `attestationTripWire` | `compared`, `first-observation`, `unavailable`, or `null` **iff** `armKind` is `deterministic` (a mismatch is refused at construction) |
| `latency` | `LatencyBlock` or `null` |
| `requestTimeoutSeconds` | the chat-call budget the run used, or `null` on a record written before it existed |

Derived, not stored: `modelKey` (the model id, or the deterministic `armId`) and `armKey` (the
`candidateId` when the fingerprint has one, else `modelKey`).

### 2.2 `ItemResult`, `ItemTiming`, `CallTiming`

`ItemResult`: `itemId`; `pairingKey` (list of strings, outermost first; the analysis-unit id is its
first element); `outcome`; `scoreable` (metric to bool); `counts` (metric to int); `measures`
(metric to finite float); `timing`; `latencyMs` (emitted for readability, **never read back**:
re-derived as `timing.wallClockMs` when `withheldFor` is `null`); `detail` (scorer-defined,
never a reply text). `counts` and `measures` are disjoint.

`ItemTiming`: `wallClockMs` (`null` when the item is incomplete, with the kept-failure exception of
`DESIGN.md` §7.3), `calls` (one `CallTiming` per completed call), `withheldFor` (`load`, `timeout`,
`no_response` or `null`). `CallTiming`: `wallClockMs`, `ttftMs`, `generationMs`, `promptTokens`,
`tokensPerSecond`, each `null` when absent, never `0`. A deterministic arm's items have
`timing: null`.

### 2.3 `LatencyBlock`

`latencyMsP50`, `latencyMsP95`, `latencyMsMax` (nullable; the tail figure is one of the last two),
`latencyTimedCount`, `latencyItemCount`, `latencyWithheldForLoad`, `latencyWithheldForNoResponse`
(every withheld item lands in exactly one), `statsCoveredCount` (`null` on a surface with no
`stats`), `callCount`, `ttftMsMedian`, `prefillMsPer1kMedian`, `tokensPerSecondMedian`,
`unexplainedMsMax`. Derived: `callAttemptedCount` = `callCount` + `latencyWithheldForNoResponse`.

### 2.4 Aggregates (a closed union)

| `kind` | Role family | Fields |
|---|---|---|
| `retrieval` | `embedder` | `recallAtK` (binary list), `mrr` (continuous), `precisionAt1` (binary), `separationRaw`, `separationZ` (distribution summaries) |
| `classification` | `guard-judge`, `join-in-judge`, `message-type-classifier` | `perClass` (binary list), `parseFailures`, `n` |
| `extraction` | `nlq-generator` | `exactMatch`, `byShape` (binary), `parseFailures`, `malformedSpecCount`, `schemaViolationCount`, `luckyPassCount` |
| `grounding` | `chat-responder` | `checklistPass`, `perCheck` (binary), `parseFailures` (always `0`) |
| `toolcalls` | `tool-caller` | `cleanThroughTurn`, `perTurnPosition`, `funnel` (binary), `funnelCounts`, `restraint`, `hazard`, `determinismProbe`, `iterationSummary`, `prosePseudoCallDetector` |

A scorer that has no field for a figure cannot report it: there is no pooled-accuracy field on
`classification` and no blended percentage on `toolcalls`. `funnelCounts` fields: `turnsDriven`,
`unrunnableModelChannel`, `unrunnableToolChannel`, `turnsScoredAfterUnrunnable`, `restraintTurns`,
`requiredCallTurns`, `nativeCallEmitted`, `prosePseudoCall`, `noAttempt`, `turnsWithAnyCall`,
`dispatchedCalls`, `factBearingReturns`, `unscoreableReturns`, `argsOmittedRequired`,
`argsWrongValue`, `argsBoundaryUnit` (a named subset of `argsWrongValue`, never a sibling bucket).
`hazard[i]` is `{turnIndex, metric, censored}`; `perTurnPosition[i]` is `{turnIndex, metric}`.

### 2.5 Metric encodings

Inside an aggregate, a metric is one of three objects, discriminated by `type`:

- `binary`: `name`, `successes`, `n`, `unit` — `unit` (`item`, `query`, `conversation`, `turn`,
  `call`) is **required**: it says what the denominator counts and so whether a Wilson interval may
  print;
- `continuous`: `name`, `mean`, `n`, `support` (`[lo, hi]` or `null`);
- `distribution`: `name`, `median`, `p10`, `n`, `unit`, `support`.

## 3. The fingerprint

A fingerprint is `{armKind, callSurface?, ...fields}`: two **discriminators** and the remaining
fields held in a plain mapping, so that a missing key stays distinct from a key whose value is
`null`.

- `armKind`: `model` or `deterministic` (two-valued).
- `callSurface`: `chat`, `embeddings`, `decisions` or `handoff` on a model arm; **required with no
  default**; `null` and **omitted from the record** exactly when the arm is deterministic (a
  deterministic arm carrying one is forbidden). On read, a model record missing it is `""` and a
  deterministic one `null`; the two must not be conflated.
- The derived **`armProfile`** is `<armKind>:<callSurface>` or `deterministic`: `model:chat`,
  `model:embeddings`, `model:decisions`, `model:handoff`, `deterministic`. Both discriminators are
  checked **before any mapping is consulted**, and they belong to no required set.

### 3.1 Field tiers and forbidden fields

Each required field has a tier: `nonempty` (present **and** truthy) or `present` (present; `[]`,
`0`, `false`, `""` are valid). `null` fails both tiers with no exception: a field the runner
cannot capture refuses the run. Fields forbidden on a profile are exactly the set difference
`(union of every other profile's required fields) − (this profile's required fields)`, never a
list. A residency element is exactly `{id, state}`, both non-empty strings; an extra or missing key
is a problem.

### 3.2 Required fields per profile (schema 1)

`present`-tier fields are marked; every other field is `nonempty`.

**`model:chat`** (30 fields): `modelKey`, `modelPublisher`, `arch`, `quantization`,
`compatibilityType`, `maxContextLength`, `loadedContextLength`, `modelType`, `modelCapabilities`
(present), `modelCapabilitiesPresent` (present), `runtimeName`, `runtimeVersion`,
`residencySource`, `residentModelsAtStart` (present), `residentModelsAtEnd` (present),
`temperature` (present), `maxTokens`, `packId`, `packVersion`, `packContentHash`, `benchVersion`,
`benchSchemaVersion`, `pythonVersion`, `hostOs`, `startedAt`, `endedAt`, `lmStudioAppVersion`,
`kvCacheSetting`, `hostRamGb`, `otherResidentWorkloads` (present).

**`model:embeddings`** (26): the chat set **minus** `runtimeName`, `runtimeVersion`, `temperature`,
`maxTokens`, which an embeddings call cannot have measured and which are therefore forbidden.

**`model:decisions`** (36): the embeddings set plus `candidateId`, `candidateContentHash`,
`questionSpecHash`, `decisionPolicy`, `fitManifestHash`, `fitManifestSize` (present),
`evalSetHash`, `fitOutcome`, `fitQualityTarget`, `decisionInstanceConfig` (present). There is no
`runtimeName`/`runtimeVersion`: the decisions route returns no `runtime`.

**`model:handoff`** (48): the decisions set plus the hand-off model under flat `handoff` names
(`handoffModelKey`, `handoffModelPublisher`, `handoffArch`, `handoffQuantization`,
`handoffCompatibilityType`, `handoffMaxContextLength`, `handoffLoadedContextLength`,
`handoffModelType`, `handoffRuntimeName`, `handoffRuntimeVersion`, `handoffTemperature` (present),
`handoffMaxTokens`): its runtime comes from its own chat warm-up and its sampling settings are the
pack's, which only the hand-off call obeys.

**`deterministic`** (11): `armId`, `armParametersHash` (the arm's own knobs; for BM25 `k1`, `b`,
tokenization, IDF variant and the stopwords file's SHA-256), `packId`, `packVersion`,
`packContentHash`, `benchVersion`, `benchSchemaVersion`, `pythonVersion`, `hostOs`, `startedAt`,
`endedAt`.

Field sources: model identity fields come from the LM Studio catalog entry of the model the arm
ran (the decision model on a candidate arm), never from a request string; `loadedContextLength`
from the catalog read after the load; `lmStudioAppVersion`, `kvCacheSetting`, `hostRamGb` and
`otherResidentWorkloads` from `host.json`; `benchVersion` from `modelbench.__version__`;
`startedAt` and `endedAt` are UTC `YYYY-MM-DDTHH:MM:SSZ`; `decisionPolicy` is the candidate's
policy object; `fitManifestHash` and `fitManifestSize` are measured from the verified manifest
(`size` is the entry count); `evalSetHash` is `DESIGN.md` §7.7's; `decisionInstanceConfig` is the
decision model's `loaded_instances[0].config` verbatim. A report reads the candidate's identity,
fit outcome and quality target from the run record, never from the candidate file.

## 4. Schema versioning, store and read rules

### 4.1 `benchSchemaVersion`

An integer, `1`, kept separate from `benchVersion` (the installed `modelbench` release). It
increments **only** when the required-field set or the on-disk record shape changes in a way a
reader must branch on, and never as a side effect of adding a field. A bump is a deliberate act:
a new `REQUIRED_BY_SCHEMA` entry and an explicit decision about migrating stored records. The two
decision profiles are additive to schema 1. An older-schema record is validated against the
contract it was written under and is never excluded on that basis alone; a comparison spanning more
than one schema prints the `SCHEMA VERSIONS IN THIS COMPARISON` banner (`DESIGN.md` §9.2).

### 4.2 Write and read

- **Write (`store`).** Validates the fingerprint and raises `InvalidFingerprint` on any field
  problem; there is no flag to bypass it. `runId` must be a bare file name.
- **Read (`load_history(root, packId)`).** Reads every `results/runs/*.json` in file-name order and
  classifies each from its **envelope** (`runId`, the fingerprint's `packId` and
  `benchSchemaVersion`) before decoding the body: a record that does not parse is `unparseable`;
  one of another pack is skipped silently; one whose schema is unknown is `unknown_schema`; one
  whose body does not decode is `unparseable`; one with fingerprint problems, or an item that
  declares a metric scoreable with no `counts`/`measures` entry, is `field` (the problem list
  names the field and the reason: `absent`, `empty`, `null`, `forbidden`, `unknown`). Valid and
  invalid are returned separately and invalid records are named in a report.

### 4.3 `index.csv`

`results/index.csv` is derived from `results/runs/` by `index rebuild` and never a source of truth.
Columns, in order: `runId`, `date` (the first 10 characters of `startedAt`), `role`, `packId`,
`packVersion`, `packContentHash8`, `modelKey`, `quantization`, `armKind`, `n` (the item count),
`headlineMetrics` (every named aggregate metric of the run, not only headline ones: `name=k/n`,
`name=p50 x.xxxx` or `name=x.xxxx`, joined by `; `), `latencyMsP50`, `latencyMsP95` (the median
and the 95th percentile of the items' admitted wall clocks, **not** gated by either coverage floor
and never renamed to a maximum, so they can differ from the run's `LatencyBlock`, §2.3), `valid` (`yes`
or `no`), `candidateId`, `handoffModelKey` (empty when not applicable). On a candidate record
`modelKey` is the decision model and `candidateId` the arm.

## 5. The candidate and the fit manifest

### 5.1 `candidates/<candidateId>.json`

The file name (without `.json`) must equal `candidateId` and match
`[A-Za-z0-9][A-Za-z0-9._-]*`.

| Key | Type | Rule |
|---|---|---|
| `candidateId` | string | equals the file name |
| `role` | string | a decision role (`BAR_KIND_BY_ROLE`) |
| `decisionModel` | string | an LM Studio catalog id whose catalog `type` is `decision` |
| `handoffModel` | string or `null` | catalog `type` `llm` or `vlm` |
| `decisionTimeoutSeconds` | number | finite and greater than 0 |
| `question` | object | `type` (`predicate` or `choice`; `score` is refused), `name` (non-empty), `instructions` (string); a choice also has `choices`: a non-empty list of `{value, instructions}` with non-empty unique `value`s; a predicate has no `choices`; its type must match the pack's answer space and a choice's value set its labels |
| `policy` | object | `{kind: predicate-band, noBelow, yesAtOrAbove}` for a predicate, `{kind: choice-confidence, minConfidence}` for a choice; probabilities in `[0, 1]`, `noBelow ≤ yesAtOrAbove`; a policy that can hand off needs a `handoffModel` |
| `fit` | object | `outcome` in `within-margin`, `below-margin`, `fit-infeasible: ranking`, `fit-infeasible: agreement`, `fit-infeasible: latency`; `qualityTarget` in `D_c<=0`, `mcnemar-nonreject` (speed-up) or `T=0.94`, `T=0.93`, `T=0.92`, `T=0.91`, `T=0.90`, `max-min-agreement` (new ability) |
| `fitManifest` | object | `file` (a plain name in `candidates/`), `sha256` (64 lowercase hex), `size` (non-negative integer, the entry count), `purposes` (list of non-empty strings) |
| `provenance` | object | `builtBy`, `datasetVersion`, `sourceSha`, each a non-empty string |

The content hash is the SHA-256 of the file's bytes (`DESIGN.md` §7.2). Shipped candidates sit in
`candidates/` as `<point>-<version>-only-f<n>.json` (no hand-off model) and
`<point>-<version>-handoff-f<n>.json`, each with its fit manifest `<point>-<version>-f<n>.fit.txt`.

### 5.2 The fit manifest

A UTF-8 text file: one lowercase-hex SHA-256 (an item's `input_hash`) per line, each line ending in
`\n` (the last included, never `\r\n`), **strictly ascending** (sorted, no duplicate). An empty
file is a manifest of zero inputs. The entry count is `fitManifest.size`.

### 5.3 What the candidate importer writes

`scripts/import_decision_pack.py --candidate <file>` copies the candidate and the fit manifest its
`fitManifest.file` names, from the candidate's own directory, into `candidates/`, after both load
in a staging copy. The fit report is never copied. A destination file with other bytes is refused;
identical bytes are a no-op.

## 6. `host.json`

The operator-attested half of the fingerprint, local and gitignored, written by `attest` and
read by every `run`.

```
{ "schemaVersion": 1,
  "apiBaseUrl": "<non-empty string>",
  "attested": { "lmStudioAppVersion", "kvCacheSetting", "hostRamGb", "otherResidentWorkloads" },
  "attestedAt": "<UTC timestamp>",
  "observedAtAttestation": { "residencySource": "lmstudio-api-v0",
                             "runtimeName", "runtimeVersion", "runtimeObservedAt" } }
```

`attested` has exactly the four named keys and no other. `lmStudioAppVersion` and `kvCacheSetting`
are non-empty strings, `hostRamGb` a number, `otherResidentWorkloads` a list (possibly empty);
none is `null`. `observedAtAttestation.residencySource` is required and non-empty. `runtimeName`,
`runtimeVersion` and `runtimeObservedAt` are absent until the first chat-surface run back-fills the
three together (`DESIGN.md` §3.9); `attest` never writes them, as no probed endpoint exposes a
runtime. A `host.json` that is absent, not JSON, or fails this schema refuses a run with exit `5`.

## 7. The decision contract: golden vectors

The pure functions that falkor-chat and decision-models transcribe are pinned by JSONL vectors in
`tests/fixtures/decision-contract/`, **model-bench's canonical copies**, whose expected values were
written independently of the code. Those two components' suites byte-copy the files they list and
assert them equal to these unless `model-bench/` is absent.

| Vector file | Pins |
|---|---|
| `apply-policy.jsonl` | `apply_policy` |
| `input-hash.jsonl` | `input_hash`, `INPUT_FIELDS_BY_ROLE` |
| `over-limit.jsonl` | `is_over_limit`, `OVER_LIMIT_MARKER` |
| `fit-manifest.jsonl` | the fit-manifest file format (§5.2) |
| `dataset-shape.jsonl` | the shapes of decision-models' `generation.json`, `splits.json` and items file per role (§1.6) |
| `render-guard.jsonl`, `normalize-turns.jsonl`, `handoff-parse-guard.jsonl` | the guard's judge rendering, turn normalization and hand-off parse |
| `render-join-in.jsonl`, `handoff-parse-join-in.jsonl` | join-in's input rendering and hand-off parse |
| `render-message-type.jsonl`, `handoff-parse-message-type.jsonl` | message type's input rendering and hand-off parse |

## 8. Operator files

### 8.1 `footprints.json`

A checked-in flat object `{"<modelKey>": "<display string>"}` of approximate memory footprints,
passed to `rank --footprints` and shown verbatim in the footprint column (never parsed as a number).
No run reads it implicitly.

### 8.2 The rank-report marker

`rank` prints, immediately after each ranked table, one HTML comment line:

```
<!-- rank-report: pack=<packId> metric=<metricName> top=<modelKey> value=<v> ci=[<lo>,<hi>] -->
```

with `value` and the interval bounds at four decimals. A pack with no headline metric prints one
per `verdictMetrics` member. `scripts/consolidate_sweep_reports.py` reads these markers and copies
their fields verbatim, never parsing them to numbers.

### 8.3 Report and sweep files

- `compare` reports: `reports/<packId>-<YYYYMMDD>-<NN>.md`; `rank` reports:
  `reports/<packId>-rank-<YYYYMMDD>-<NN>.md` (`DESIGN.md` §2.3, §2.4).
- The sweep plan file read by `run-sweep.sh`: one `model|pack` per line, blank lines skipped. The
  log it appends: per attempt a `START model=… pack=…` line and an `OK` or `FAIL … rc=<code>`
  line, a failure's captured output indented under it, and a final `==== SWEEP DONE: <ok> ok,
  <fail> fail, <total> total ====` or `SWEEP HALTED` line.
