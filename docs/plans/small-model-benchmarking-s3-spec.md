# `model-bench` S3 — the `embedder` pack and `refresh_golden.py` — implementation spec

> **Status:** active · **Owner:** `architect` · **Tracks:** — · **Extends:** `docs/plans/small-model-benchmarking.md` (S3)

## 1. Goal & scope

`docs/plans/small-model-benchmarking.md` (the "plan") assigns the `embedder` pack,
`modelbench/scoring/retrieval.py` and `scripts/refresh_golden.py` to stage S3 (§4 S3,
`:5654-5698`) and gives the pack's design in detailed prose (§3.8.1, `:1988-2075`) — data files, the
embedding mechanism, the cache key, the BM25 reference arm, the reported metrics — but, unlike
`packs.py`/`lmstudio.py`'s S1/S2 code-skeleton blocks or even the runner (whose own missing skeleton
`docs/plans/small-model-benchmarking-runner-spec.md`, "the runner spec", already closed), §3.8.1
gives **no concrete function signature, module layout, or step sequence**. This document is that
missing skeleton, synthesized from the plan, `docs/plans/small-model-benchmarking-ml.md` ("`-ml`")
§5.1-§5.5, and the shipped S1/S2 code S3 builds on top of — **not** a revision of the plan (the plan
gate is closed and this document does not touch it) and **not** an implementation (no source is
written here).

**In scope:** the `embedder-graphrag-retrieval` pack's directory layout and `pack.json` manifest;
`scripts/refresh_golden.py`'s two operating modes (data import, `--check-origins`) and its
`--embed-corpus` live sub-step; `modelbench/scoring/retrieval.py`'s full signature set (recall@k,
MRR, precision@k, score separation, the BM25 arm, `ItemScorer.score_item`/`aggregate`); the small,
concretely-specified extension this stage makes to the `ItemScorer` Protocol and to
`runner.py`/`cli.py`'s already-shipped S2 code, without which the embedder role cannot run at all
(§2.2); the `tests/fixtures/metrics_agreement.json` transcription and its test; the harness
self-check's diagnostic write-up.

**Out of scope:** `guard-judge`/`nlq-generator`/`chat-responder`/`tool-caller`'s scorer modules (S4-
S7); `report.py`'s `sep_z` exploratory paired-bootstrap comparison — the plan states explicitly that
this is **not** S3's done-condition (§4 S3 done-condition 2, `:5676-5680`: *"this report renders no
`sep_z` interval at all"* — `report.py` already renders each arm's own median/p10 generically for
any `DistributionSummary`, confirmed by reading `report.py` directly, §2.1); `packs.py`'s
`validate_pack` gaining a check that a pack's declared `"scorer"` name resolves to an importable
module (a `validate_pack` enhancement, not this stage's — noted as a possible follow-up in §9, not a
blocker); the +22-harder-queries golden-set extension the plan defers to `docs/BACKLOG.md` (§3.8.1).

**CPG:** considered, not relevant — `model-bench` is a Python component with no loaded CPG
(queried this session: no `cpg_model-bench` graph exists among the graphs loaded on this FalkorDB
instance); this is a code-level task, so the applicable case is "considered, not relevant" rather
than "not applicable".

## 2. Context & findings

### 2.1 What already exists (read directly from `model-bench/modelbench/*.py`, not inferred)

- **`packs.py`** — `Pack.iter_items()` (reads `data.<key>`'s JSONL rows in file order — generic,
  role-agnostic, "named for readability... has no cross-cutting design stakes", its own docstring)
  and `Pack.data_path(key)` (resolves any `data.<key>` manifest entry to a real path) are shipped
  and exactly what this stage needs for `queries.jsonl`/`corpus.jsonl`/the two fixture embedding
  files. `validate_pack`'s row-count-identity route (`_row_count_identity_problems`) is **skipped
  entirely** when `sampling.scripts` is absent from the manifest — confirmed the item-level pack
  shape (`guard-judge`'s S2 fixtures: no `scripts` key, `pairingKey: ["itemId"]`,
  `analysisUnit: "itemId"`, no `tools`/`prompt` block) already validates clean with exactly that
  shape, so the embedder pack's manifest (§4 below) follows the same item-level shape rather than
  inventing one. `roles.py`'s `ANALYSIS_UNIT_FIELD_BY_ROLE["embedder"] == "itemId"` and
  `UNIT_KIND_BY_ROLE["embedder"] == "query"` are shipped and fix `pairingKey`/`BinaryMetric.unit`.
- **`results.py`** — `ItemResult.measures`, `ContinuousMetric.support`, `DistributionSummary` (six
  fields: `name, median, p10, n, unit, support`) and `RetrievalAggregates`
  (`recallAtK: tuple[BinaryMetric,...]`, `mrr: ContinuousMetric | None`,
  `precisionAt1: BinaryMetric | None`, `separationRaw`/`separationZ: DistributionSummary | None`,
  `named_metrics()`) are **all shipped** — confirmed by reading `results.py:115-152` and `:483-506`
  directly, not by trusting the plan's §4 S1e Table F narrative (which the plan itself calls
  "already shipped" in several v1.2x revision notes). This is the one precondition §4 S3 done-
  condition 1 names by name (*"§4 S1e Table F must have landed... without it there is nowhere on
  the record for [MRR]"*) and it has. Nothing in this document adds a field to any of these types.
- **`lmstudio.py`** — `LMStudio.embed(texts: Sequence[str], *, model: str, timeout_s: float) ->
  EmbedResult` is shipped, batched (one call embeds all 121 corpus documents). `EmbedResult` carries
  `vectors: tuple[tuple[float,...],...]`, `dimension`, `model`, `usage`, `wallClockMs` — **no
  `stats`** (an embeddings call returns none, confirmed at `lmstudio.py:210-220`; this is why
  `CallTiming`'s three `stats`-derived fields are `None` on every embedder item, matching
  `LatencyBlock` rule (iv-a)). `ModelInfo.quantization: str` is shipped and is the second of the
  cache key's four components.
- **`runner.py`** — `_drive_single_call_items` (§4 S2's item-level driving loop) and
  `run_pack`'s wiring (`runner.py:790-809`) are shipped and **already call `_load_item_scorer` and
  `scorer.aggregate(items, pack=pack)`** for every non-`tool-caller` role. `_load_item_scorer`/
  `_load_conversation_scorer` **still unconditionally `raise NotImplementedError`** — expected,
  their own docstrings say the first concrete `ItemScorer` is S3's (`runner.py:190-205`). This is
  not a plan-vs-shipped-code contradiction; it is unbuilt scope explicitly deferred to this stage,
  and closing it (§6.1 below) is squarely this document's job.
- **`report.py`** — a `DistributionSummary` already renders generically (median + p10, no interval,
  confirmed at `report.py:796-797`) and any metric outside `pack.metrics.verdictMetrics` already
  renders `"exploratory — no significance claim"` generically (`report.py:1067,1096-1103`); a
  deterministic arm already renders its label generically (`report.py:560-561`) and the
  both-deterministic no-verdict case is already handled (`report.py:600-601`). **No call site of
  `stats.paired_cluster_bootstrap` exists anywhere in `report.py`** (grepped this session) — the
  `sep_z` exploratory *comparison* (as opposed to each arm's own median/p10, which already renders)
  genuinely does not exist yet, confirming §1's scope line: the plan's own §4 S3 done-condition 2
  says this is not S3's to build.
- **`fingerprint.py`** — the `deterministic` arm profile's eleven required fields (`_DETERMINISTIC_
  SCHEMA_1`, `fingerprint.py:131-143`): `armId`, `armParametersHash`, `packId`, `packVersion`,
  `packContentHash`, `benchVersion`, `benchSchemaVersion`, `pythonVersion`, `hostOs`, `startedAt`,
  `endedAt` — no host-state field at all (*"host state is not merely optional for it but
  forbidden"*, the module's own docstring). This is the exact field set §6.3 below builds a
  `Fingerprint` from for the BM25 arm.
- **`cli.py`** — `compare --pack <id> --session <id>` already filters stored records by
  `sessionId` (`cli.py:166`) with no role-specific logic — confirmed the precedent
  `--negative-control` already established (two records sharing one `sessionId` render as paired
  arms) is exactly the mechanism a BM25 sibling record needs, with zero changes to `compare`/
  `report.py`.

### 2.2 Two real gaps this stage cannot avoid, both closed by a small, concretely-specified extension

Neither is a plan-vs-shipped-code contradiction the way the runner spec's §2.2 found two (S1 code
disagreeing with S1's own owning section) — both are places where the plan's prose (§3.8.1) commits
this pack to a mechanism that the S2 runner's **generic, role-agnostic** driving loop and scorer
seam have no way to carry out, because no item-level role before the embedder ever needed either.
Both are required for done-condition 1 to be reachable at all, so both are their own implementation
step (§8, step 1) rather than left for the coder to improvise.

1. **The embeddings call branch hardcodes the wrong input text.** `_drive_single_call_items`'s
   embeddings branch (`runner.py:312-317`) issues
   `lmstudio.embed([json.dumps(item_input, sort_keys=True)], model=model_info.id, timeout_s=...)` —
   it embeds a JSON-serialized dump of the **whole item row** (`{"itemId": "gr-01", "query": "...",
   "relevantDocIds": [...], ...}`), not the query text, and applies no `queryPrefix` at all. §3.8.1
   requires *"applying the pack's per-model `queryPrefix`/`documentPrefix` (FR-14 — a configuration
   field, never an assumption)"* — the shipped branch cannot satisfy this, and no other role has
   ever reached this branch (the embedder is the only role whose `environment.requires` can declare
   `lmstudio-embeddings` at all, per `roles.py`'s table), so nothing before this stage needed to.
2. **No seam exists for embedding the 121-document corpus at all.** `ItemScorer.score_item`
   (`runner.py:165-172`) receives one already-computed `result: ChatResult | EmbedResult | None`
   per item and `ItemScorer.aggregate` (`runner.py:174`) receives only the finished `items` —
   neither is handed an `LMStudio` handle, so **no existing scorer method can issue the corpus's
   own embedding call**, and the corpus is not an "item" `iter_items()` would ever yield it as (it
   is reference data scored *against*, not a scored unit of work). §3.8.1 requires the corpus be
   "re-embedded whenever any part changes" and `-ml` §5.5 rules it must be re-embedded **per model,
   with that model's own document prefix** — a live capability that must run once per `run`
   invocation, before the per-item loop, using the SAME live model under test.

**Resolution (§6.2, marked as this document's own synthesis, the same way the runner spec marked
its scorer-seam design in its own §3.2):** `ItemScorer` gains two **optional**, duck-typed methods —
`prime(...)`, called once before the per-item loop if present, and `embed_text(item_input, *,
pack)`, called instead of the hardcoded `json.dumps` on the embeddings branch. Both are additive and
`getattr`-guarded at the call site, so every existing S2 test (`tests/test_runner.py`'s
`FakeItemScorer`, which defines neither) is untouched — confirmed by reading
`tests/test_runner.py:772-794`, where `FakeItemScorer.score_item`/`.aggregate` take exactly the
signatures already shipped, with no `context` parameter anywhere; this document does **not** widen
`score_item`/`aggregate`'s signature for this reason (§6.2 explains why a `context`-threading
alternative was rejected).

### 2.3 A third gap: no existing mechanism stores a second, model-less `RunResult` from one `run` invocation

§3.8.1 requires the BM25 arm be *"stored as its own `RunResult` with `armKind: "deterministic"`...
sharing that run's `sessionId`"*, and §4 S3 done-condition 2 requires `compare` to render both arms
from one report. Nothing in shipped `cli.py`/`runner.py` produces a second `RunResult` from one
`run` invocation — `armKind` is hardcoded to `"model"` at `run_pack`'s one construction site
(`runner.py:847`). This is not a contradiction (S2 never needed a deterministic arm — no other pack
has one), but it is unbuilt plumbing this stage needs. §6.3/§7.2 below specify the resolution: a
third optional `ItemScorer` method, `deterministic_arm`, called from `_cmd_run` after the model
arm is stored.

### 2.4 What the plan/`-ml` note give verbatim, and where

| Topic | Citation |
|---|---|
| Data files copied, corpus/query shapes, the embedding mechanism, cache key, BM25 arm, reported metrics, the two honesty lines, cost | Plan §3.8.1 `:1988-2075` |
| D1 — copy the data, clean-build the code; the 20-case transcribed fixture; the honest residual | Plan §3.1 `:278-380` |
| S3's five done-conditions | Plan §4 S3 `:5654-5698` |
| recall@k/MRR/precision@k exact definitions, the precision@k-is-a-rescaling finding | `-ml` §5.1 `:2656-2675` |
| Score separation: `sep_raw`/`sep_z` formulas, aggregation (median/p10/fraction), the no-`sep_raw`-difference rule | `-ml` §5.2 `:2677-2730` |
| BM25 tokenization/stopwords/parameters/IDF variant, arm-vs-reference-line ruling | `-ml` §5.3 `:2732-2755` |
| `retrieval_baseline.json`'s two narrow uses, the harness self-check ruling | `-ml` §5.4 `:2757-2802` |
| The embedding cache-key trap, all four components | `-ml` §5.5 `:2804-2811` |
| `RetrievalAggregates`/`ContinuousMetric`/`DistributionSummary` shapes | Plan §4 S1 `:3099-3137` (already shipped, §2.1) |
| The `metrics` manifest block's four consequences | Plan §3.3 `:548-565` |
| Deterministic-arm fingerprint schema | `fingerprint.py:131-143` (already shipped, §2.1) |

## 3. Design & rationale

### 3.1 One live corpus-embedding pass per run, cached opportunistically against one committed artifact — not a new cache directory

`-ml` §5.5's cache key is `(model id, quantization, docPrefix, corpus version)`. The plan already
commits one persistent artifact carrying exactly this — `corpus.embeddings.json`, written once by
`refresh_golden.py --embed-corpus` for the reference model
(`text-embedding-qwen3-embedding-0.6b`) and version-controlled inside the pack's content hash
(§3.8.1: *"Both files are inside the content hash, so a re-embed is a pack version bump"*). This
document's design: `corpus.embeddings.json` carries its own cache-key header (`§6.1`'s shape below);
`ItemScorer.prime()` computes the *live* run's cache key from `(model_info.id, model_info.
quantization, pack.manifest["embedding"]["documentPrefix"], sha256(corpus.jsonl's bytes))` and
compares it against the committed file's header — a match (true for done-condition 1's own run,
which uses the exact model the file was generated from) reuses the committed vectors with **no live
embedding call for the corpus**; a mismatch (any other model tested later) embeds the corpus live,
in-process, for that run only, and does **not** persist the result anywhere.

**Alternative considered and rejected: a separate, gitignored, cross-run embedding cache
directory** (e.g. `model-bench/.cache/embeddings/`), keyed the same way, to avoid re-embedding the
corpus on every run of a *non-reference* model. Rejected: nothing in the plan or `-ml` note asks for
cross-run caching beyond the one committed artifact, it introduces a new undocumented convention
this codebase has no precedent for (unlike `results/transcripts/`, which the plan itself names as
gitignored), and the cost is explicitly stated as low — *"Cost: low. Existing verified data, no new
golden asset"* (§3.8.1) — 121 embed calls in one batched request is cheap enough that repeating it
per run of a non-reference model is not worth the complexity of a second persistence layer. If a
future stage benchmarks many models against this pack repeatedly, a cross-run cache is a reversible
addition; nothing here forecloses it.

### 3.2 The `ItemScorer` extension — `prime`/`embed_text`, both optional and duck-typed, no `context` threading

Rejected alternative: widen `score_item(item_input, result, timing, *, pack, context)` and
`aggregate(items, *, pack, context)` to carry whatever `prime()` returns. This was considered
first, and rejected on backward-compatibility grounds alone: `tests/test_runner.py`'s
`FakeItemScorer.score_item`/`.aggregate` (§2.2) are called positionally/by-keyword with exactly
today's parameter set across at least six S2 tests (`grep`-counted, `tests/test_runner.py:772-927`);
adding a required `context` keyword to the Protocol would force every one of those call sites (and
every future scorer module for the four other item-level roles, none of which needs a corpus to
embed) to accept and ignore a parameter that means nothing to them. The chosen design instead keeps
`score_item`/`aggregate`'s signatures **byte-identical to what S2 shipped**, and gives
`scoring/retrieval.py` its own private module-level state, written by `prime()` and read by
`score_item`/`aggregate`/`embed_text` — safe because one `run` process drives exactly one model ×
one pack, never concurrently and never re-entrantly (confirmed: `cli.py`'s `run` command is a
single, synchronous CLI invocation with no threading anywhere in this codebase). A test that drives
`retrieval.py` directly (§8) calls `prime()` itself before `score_item()`, exactly the way a
scorer-level unit test would for any stateful adapter.

`_drive_single_call_items`'s two required edits (`runner.py:283-335`, §2.2):

```python
# runner.py — edits to the already-shipped S2 function, both additive and getattr-guarded so no
# existing S2 test needs to change (confirmed against tests/test_runner.py's FakeItemScorer, §2.2).

def _drive_single_call_items(
    pack: Pack, cfg: RunConfig, *, lmstudio: LMStudio, model_info: ModelInfo,
    call_surface: Literal["chat", "embeddings"], baseline_residency: list[ResidentModel],
) -> tuple[tuple[ItemResult, ...], float, Basis]:
    scorer = _load_item_scorer(pack)
    prime = getattr(scorer, "prime", None)
    if prime is not None:                                              # NEW — §3.2
        prime(
            lmstudio=lmstudio, model_info=model_info, call_surface=call_surface,
            timeout_s=cfg.requestTimeoutSeconds, pack=pack,
        )
    prompt_config = pack.prompt_config()
    ...
    for item_input in pack.iter_items():
        resident = lmstudio.residency()
        try:
            if call_surface == "chat":
                call = lmstudio.chat(_item_chat_messages(pack, item_input), ...)   # unchanged
            else:
                embed_text = scorer.embed_text                          # NEW — §3.2, required
                call = lmstudio.embed(                                  # on this branch only,
                    [embed_text(item_input, pack=pack)],                # since no other role
                    model=model_info.id, timeout_s=cfg.requestTimeoutSeconds,  # reaches it (§2.2)
                )
        except LMStudioCallTimeout:
            ...  # unchanged
```

`embed_text` is called directly (no `getattr` guard) because it is reached **only** on the
embeddings branch, and the embedder is — today and for the foreseeable set of packs the plan
names — the only role that ever declares `environment.requires: ["lmstudio-embeddings"]`
(confirmed: `roles.py` names five roles total, and §3.4-§3.8 name exactly one, `embedder`, on the
embeddings surface). A future embeddings-surface role's scorer that forgets `embed_text` gets a
loud `AttributeError` at the first item — a programming-contract failure, not a pack-config one, so
it does not need a `RunRefused`/exit-code path (the `ItemScorer.embed_text` docstring below states
this).

`ItemScorer`'s Protocol gains both as **optional** members (Python's structural `Protocol` does not
enforce a method's presence at the type level when the caller only ever reaches it through
`getattr`/a role-conditional branch, so no `runtime_checkable` change is needed):

```python
class ItemScorer(Protocol):
    def score_item(self, item_input, result, timing, *, pack) -> ItemResult: ...   # unchanged
    def aggregate(self, items, *, pack) -> Aggregates: ...                          # unchanged

    def prime(                                                          # NEW, optional (§3.2)
        self, *, lmstudio: LMStudio, model_info: ModelInfo,
        call_surface: Literal["chat", "embeddings"], timeout_s: float, pack: Pack,
    ) -> None:
        """Called once, before the per-item loop, iff the scorer defines it. The one hook that
        gives a scorer model access outside a scored item — the embedder's own use is embedding
        and caching the reference corpus (§3.1); no other role needs one yet. This document's own
        extension (§2.2/§3.2), not specified anywhere in the plan."""

    def embed_text(self, item_input: Mapping[str, Any], *, pack: Pack) -> str:      # NEW, optional
        """Called instead of the hardcoded `json.dumps(item_input, sort_keys=True)` on the
        embeddings call branch. Required in practice for any embeddings-surface role's scorer —
        this document's own extension (§2.2/§3.2)."""
```

### 3.3 The deterministic BM25 arm — a CLI-level, opt-in hook, not a `run_pack` signature change

`run_pack`'s return type (`tuple[RunResult, tuple[DispatchFailureDisclosure, ...]]`) is S2's own,
exercised by the runner spec's own test 15b and E1-E5 fixtures (`docs/plans/small-model-benchmarking-
runner-spec.md` §8) — widening it to carry a second, optional `RunResult` would touch every existing
caller of `run_pack` for zero benefit to four of five roles. Instead, `ItemScorer` gains a third
optional method, `deterministic_arm(*, pack, session_id) -> RunResult | None` — pure, no `LMStudio`
handle, computed entirely from pack data (BM25 needs no live call, §3.8.1) — and `_cmd_run` (§7.2)
calls it, `getattr`-guarded exactly like `prime`, **after** the model arm is stored, storing its
result too under the same `sessionId`. This mirrors `compare`'s existing `--negative-control`
precedent (two stored records sharing one `sessionId` render as paired arms, §2.1) with zero changes
to `compare`/`report.py`. **Alternative considered and rejected:** a `run --pack <id> --arm bm25`
mode that re-invokes the whole capture-order sequence for a synthetic "model" — rejected because
capture order (`hostinfo`, catalog probes, warm-up, residency) exists to fingerprint a *live model
call*, and running it for an arm that makes none would either fabricate fields the deterministic
schema explicitly forbids (`fingerprint.py:128-130`: *"host state is not merely optional for it but
forbidden"*) or require `run_pack` to special-case `armKind` throughout — more invasive than one
small, additive CLI hook.

### 3.4 `refresh_golden.py` reads `_CORPUS` by parsing the source file's AST, never by importing it

D1 (§2.4) settles "copy the data, clean-build the code" but does not say *how* a one-way importer
reads a Python literal (`_CORPUS` in `falkor-chat/scripts/seed_eval_corpus.py`) without importing
falkor-chat code. Importing that module directly would execute its top-level `from falkorchat import
config, db, modelconfig` and `from falkorchat.embedding import EmbeddingWorker` (confirmed by
reading `seed_eval_corpus.py:60-64`) — heavy, falkor-chat-`.venv`-dependent imports with real side
effects (`EmbeddingWorker`, `db` connections), exactly what FR-23/"standalone" and D1's "clean-build
the code" rule exist to keep `model-bench` away from. **This document's own design:** read the
source file's text and extract `_CORPUS` via `ast.parse` + a literal-eval walk over the one
`ast.Assign` node whose target is `_CORPUS` — the same AST-based, execute-nothing technique
`packs.py`'s own import allowlist already uses on every pack `.py` file (`packs.py:744-777`,
confirmed shipped precedent), applied here to extract one data literal instead of auditing imports.
`retrieval_baseline.json` and `golden_retrieval.embeddings.json`/`golden_retrieval.jsonl` are already
JSON/JSONL and need no AST step — a straight read.

### 3.5 One new manifest block, `"embedding"` — a genuine synthesis decision, since the plan names the fields but never the JSON key

§3.3's `pack.json` schema names `prompt` (chat-only config) and `metrics` (verdict family) as the
manifest's two role-config blocks; neither fits `queryPrefix`/`documentPrefix`/BM25's parameters.
`Pack.prompt_config()` is chat-specific (its own docstring: raises on a role outside
`MULTI_CALL_TURN_BY_ROLE`'s chat-shaped branch) and is not reused here. This document introduces
`manifest["embedding"]`, read directly by `scoring/retrieval.py` (no `packs.py` parser — the same
"pure data, generic loader, role-specific parsing is the scorer's job" split `Pack.iter_items()`'s
own docstring already states for item rows):

```json
"embedding": {
  "queryPrefix": "",
  "documentPrefix": "",
  "bm25": {"k1": 1.2, "b": 0.75, "stopwordsFile": "bm25_stopwords.txt"}
}
```

## 4. File/module layout

```
model-bench/
  scripts/
    refresh_golden.py              # one-way importer + --check-origins + --embed-corpus (§6.1)
  modelbench/
    scoring/
      __init__.py
      retrieval.py                 # embedder's ItemScorer (§6.2-§6.3)
  packs/
    embedder-graphrag-retrieval/
      pack.json                    # §4.1 below
      queries.jsonl                # 38 rows, from golden_retrieval.jsonl (§6.1)
      corpus.jsonl                 # 121 rows, {docId, text, topic}, from seed_eval_corpus.py's _CORPUS
      corpus.embeddings.json       # written by --embed-corpus; cache-key header + 121 raw vectors
      golden_retrieval.embeddings.json   # copied verbatim — 38 query vectors, ranking self-test fixture
      retrieval_baseline.json      # copied verbatim — the harness self-check's reference figures
      bm25_stopwords.txt           # small committed English stopword list (-ml §5.3)
      PROVENANCE.md                # sourceGitSha/sourceSha256/copiedAt per copied file (§6.1)
  tests/
    fixtures/
      metrics_agreement.json       # 20 hand-transcribed cases (§3.1 point 2; §6.1)
    test_scoring_retrieval.py      # test 8 + prime/embed_text/deterministic_arm unit tests
    test_metrics_agreement.py      # the 20-case agreement test
    test_refresh_golden.py         # --check-origins / cache-key assertion (§3.8.1: "asserted in a
                                    # unit test, not just implemented")
```

### 4.1 `packs/embedder-graphrag-retrieval/pack.json`

```json
{
  "packId": "embedder-graphrag-retrieval",
  "packVersion": "1.0.0",
  "role": "embedder",
  "schemaVersion": 1,
  "description": "Exact brute-force cosine retrieval over the 121-message falkor-chat eval corpus (K-026), plus a BM25 reference arm.",
  "scorer": "retrieval",
  "environment": {"requires": ["lmstudio-embeddings"]},
  "data": {
    "items": "queries.jsonl",
    "corpus": "corpus.jsonl",
    "corpusEmbeddings": "corpus.embeddings.json",
    "queryEmbeddingsFixture": "golden_retrieval.embeddings.json",
    "baseline": "retrieval_baseline.json"
  },
  "embedding": {
    "queryPrefix": "",
    "documentPrefix": "",
    "bm25": {"k1": 1.2, "b": 0.75, "stopwordsFile": "bm25_stopwords.txt"}
  },
  "sampling": {
    "seed": 20260902,
    "pairingKey": ["itemId"],
    "analysisUnit": "itemId"
  },
  "metrics": {
    "verdictMetrics": ["mrr"],
    "headlineMetric": "mrr",
    "recallAtK": {"ks": [5, 10]}
  },
  "provenance": "PROVENANCE.md"
}
```

Matches the shipped item-level pack shape exactly (§2.1): no `sampling.scripts` (item-level,
`_row_count_identity_problems` skips it by design), no `tools`/`prompt` block (embedder is
single-call, `MULTI_CALL_TURN_BY_ROLE["embedder"] is False`, and `Pack.prompt_config()`'s own
docstring already states an absent `prompt` block is never this role's problem), `pairingKey[0] ==
analysisUnit == "itemId"` matching `roles.ANALYSIS_UNIT_FIELD_BY_ROLE["embedder"]`.
`queryPrefix`/`documentPrefix` default to `""` here (the reference model's own convention; a future
pack version may set them non-empty — the field exists precisely so that is a config change, not a
code change, per FR-14).

`queries.jsonl` row shape (one per golden query, `refresh_golden.py`-written): `{"itemId": "gr-01",
"query": "...", "relevantDocIds": ["eval-payment-timeout-incident-008"], "topic":
"payment-timeout-incident", "targetText": "...", "rationale": "..."}` — `id`→`itemId` and
`relevant_msgIds`→`relevantDocIds` renamed for this pack's own naming convention (`itemId` matches
`ANALYSIS_UNIT_FIELD_BY_ROLE`; `docId` is `corpus.jsonl`'s own field name, §3.8.1), `targetText`/
`rationale` carried through unused by scoring but kept for a human reading the pack.

## 5. `scripts/refresh_golden.py`

```python
# scripts/refresh_golden.py — the one-way, human-invoked importer (plan §3.1 point 3, §4 S3).
# Never imported by any run-path module (FR-23) and never invoked automatically. Reads
# falkor-chat source files as TEXT/AST only (§3.4) — never imports falkor-chat code.

_TRACKED_ORIGINS: tuple[OriginSpec, ...] = (
    # (originPathFromRepoRoot, packRelativeDestination, kind)
    OriginSpec("falkor-chat/server/tests/eval/golden_retrieval.jsonl", "queries.jsonl", "jsonl-transform"),
    OriginSpec("falkor-chat/scripts/seed_eval_corpus.py", "corpus.jsonl", "ast-literal"),
    OriginSpec("falkor-chat/server/tests/eval/retrieval_baseline.json", "retrieval_baseline.json", "copy"),
    OriginSpec("falkor-chat/server/tests/eval/golden_retrieval.embeddings.json", "golden_retrieval.embeddings.json", "copy"),
    # Tracked for --check-origins drift detection only — refresh_golden.py never writes this file's
    # CONTENT (§3.1 point 2: the 20 cases are hand-transcribed, not extracted); it reads the file's
    # own recorded sourceSha256 and reports drift against a fresh hash of test_metrics.py.
    OriginSpec("falkor-chat/server/tests/eval/test_metrics.py", "../../tests/fixtures/metrics_agreement.json", "check-only"),
)

@dataclass(frozen=True)
class OriginSpec:
    originPath: str            # repo-root-relative
    destPath: str               # pack-root-relative (or, for check-only, relative to this pack dir)
    kind: Literal["copy", "jsonl-transform", "ast-literal", "check-only"]


def main(argv: Sequence[str] | None = None) -> int:
    """Three pairwise-exclusive modes on one CLI:

    `refresh_golden.py --repo-root <path> --pack packs/embedder-graphrag-retrieval` — the default
    data-import mode: re-copies/re-transforms every `OriginSpec` except the `check-only` one,
    rewrites `PROVENANCE.md`, and REQUIRES the caller to have already bumped `pack.json`'s
    `packVersion` by hand first (refuses otherwise — a content-hash-changing edit under an unchanged
    version number is exactly AC-3's own violation, and this script is the one place that could
    silently commit it). Never touches `corpus.embeddings.json` (the separate live step below).

    `--check-origins` — read-only, no writes: re-hashes every tracked origin (all five, incl.
    `test_metrics.py`) and prints `unchanged`/`DRIFTED` per file, exit 0 iff none drifted.

    `--embed-corpus --model <key> --api-base-url <url>` — the one live-LM-Studio mode: embeds
    `corpus.jsonl` through the given model, L2-normalizes, writes `corpus.embeddings.json` with its
    cache-key header. Exclusive with the other two modes (refuses `--check-origins --embed-corpus`
    together — two unrelated operations, one flag each, never combined in one invocation).
    """


def _read_corpus_literal(seed_script_text: str) -> list[dict[str, object]]:
    """AST-parses `seed_eval_corpus.py`'s text and literal-evals the one `_CORPUS = [...]` Assign
    node (§3.4) — never `exec`/`import`. Raises `RefreshGoldenError` if no such assignment exists
    (the origin's own shape changed) rather than silently returning `[]`."""


def _corpus_rows_from_literal(corpus: list[dict[str, object]]) -> list[dict[str, str]]:
    """One row per message: `{"docId": f"eval-{topic['slug']}-{n:03d}", "text": msg_text, "topic":
    topic['slug']}` — reproduces `seed_eval_corpus.py`'s own `msg_id = f"eval-{slug}-{n:03d}"`
    convention (confirmed at `seed_eval_corpus.py:629`) so `docId` values match `golden_retrieval.
    jsonl`'s `relevant_msgIds` verbatim, per §3.8.1's own stated requirement."""


def _write_provenance(pack_root: Path, records: Sequence[ProvenanceRecord]) -> None:
    """Rewrites PROVENANCE.md in full (never appended to — this file's job is "what does this pack's
    data trace back to right now", a living, wholly-regenerated document, not a history log)."""


@dataclass(frozen=True)
class ProvenanceRecord:
    originPath: str
    destPath: str
    sourceGitSha: str           # `git rev-parse HEAD` in the falkor-chat working tree, at copy time
    sourceSha256: str           # sha256 of the ORIGIN file's bytes at copy time
    copiedAt: str                # UTC ISO-8601


def _check_origins(repo_root: Path, pack_root: Path) -> list[OriginCheckResult]:
    """Re-hashes every `_TRACKED_ORIGINS` entry's current `originPath` bytes and compares against
    the matching `ProvenanceRecord.sourceSha256` (read from `PROVENANCE.md` for the four copied/
    transformed files; from `metrics_agreement.json`'s own header for `test_metrics.py`, per §3.1
    point 2(c): *"the drift detector... re-reads and re-hashes every recorded origin file, including
    test_metrics.py"*). Read-only; never mutates either side."""


# --- §5.4: the one live mode ---

def embed_corpus(
    corpus_rows: Sequence[Mapping[str, str]], *, lmstudio: LMStudio, model_key: str,
    document_prefix: str, timeout_s: float,
) -> CorpusEmbeddingsFile:
    """One batched `lmstudio.embed()` call over all 121 (prefixed) documents — RAW vectors, never
    pre-normalized at write time (§6.1: the ranking function L2-normalizes at read time and
    the whole point of `corpus.embeddings.json` existing is to let the offline self-test observe the
    raw norm distribution too, so normalizing before writing would throw away the one thing §5.2
    asks to be measured)."""


@dataclass(frozen=True)
class CorpusEmbeddingsFile:
    cacheKey: Mapping[str, str]     # {"model": ..., "quantization": ..., "documentPrefix": ...,
                                     #  "corpusSha256": ...} — -ml §5.5's four components, named
    generatedAt: str
    vectors: dict[str, list[float]]  # docId -> raw vector, RAW (not normalized)

    def to_dict(self) -> dict[str, Any]: ...
    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "CorpusEmbeddingsFile": ...
```

## 6. `modelbench/scoring/retrieval.py`

### 6.1 Pure arithmetic — offline-testable against hand-built ranked lists (test 8) and the transcribed fixture

```python
# modelbench/scoring/retrieval.py

# --- recall@k / MRR — verbatim ports of falkor-chat/server/tests/eval/metrics.py's recall_at_k/mrr
# (D1: re-implemented, never imported). Parameter names (`retrieved`, `relevant`, `k`) match the
# plan's own transcribed-fixture example verbatim (plan §3.1 point 2(a), the recall_at_k JSON
# example: {"args": {"retrieved": [...], "relevant": [...], "k": 5}}) so test_metrics_agreement.py
# can call `getattr(retrieval, case["function"])(**case["args"])` with no key translation.

def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """|top-k ∩ relevant| / |relevant|. Raises ValueError if `relevant` is empty (`-ml` §5.1,
    metrics.py's own discipline — a golden pair with no relevant ids is a fixture defect)."""

def mrr(retrieved: list[str], relevant: set[str]) -> float:
    """Reciprocal rank (1-indexed) of the first relevant id; 0.0 if none found. Raises ValueError
    if `relevant` is empty."""

def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """|top-k ∩ relevant| / k (`-ml` §5.1 — not in metrics.py; -ml's own definition). k > 0 always
    (pack config, never 0); no ValueError branch of its own since recall_at_k's already covers the
    shared `relevant`-empty precondition and both are always called on the same item."""

# --- L2 normalization + cosine (`-ml` §5.1/§5.2: "L2-normalize every vector... record the raw norm
# distribution" — normalization happens HERE, inside the ranking function, not at the call site, so
# the raw norm stays observable to the caller before this function discards scale) ---

def l2_norm(vector: Sequence[float]) -> float: ...
def l2_normalize(vector: Sequence[float]) -> tuple[float, ...]: ...  # returns `vector` unchanged,
                                                                       # never divides, iff norm==0.0
                                                                       # (a genuinely zero embedding
                                                                       # is a harness defect to
                                                                       # surface downstream, not a
                                                                       # ZeroDivisionError here)

@dataclass(frozen=True)
class RankedList:
    docIds: tuple[str, ...]          # every corpus doc, sorted by score descending — no truncation
                                      # before scoring (§3.8.1: exact search needs every score for
                                      # sep_raw/sep_z, brute force gives them for free)
    scores: dict[str, float]          # docId -> cosine(normalized query, normalized doc)

def rank_by_cosine(
    query_vector: Sequence[float], corpus_vectors: Mapping[str, Sequence[float]],
) -> RankedList:
    """Brute-force exact cosine over the WHOLE corpus (§3.8.1: no ANN, no FalkorDB, no top-k
    truncation). L2-normalizes `query_vector` and every corpus vector internally."""

# --- score separation (`-ml` §5.2) ---

def score_separation_raw(scores: Mapping[str, float], relevant: set[str]) -> float:
    """max_{d in relevant} score(d) - max_{d not in relevant} score(d). Raises ValueError if
    `relevant` is empty or if `scores` names no non-relevant doc (both are golden-set/corpus
    defects, never a legitimate scoring outcome — mirrors recall_at_k's discipline)."""

def score_separation_z(sep_raw_value: float, scores: Mapping[str, float]) -> float:
    """sep_raw_value / stdev({scores[d] for d in scores}) (`-ml` §5.2's `sd(...)`).

    **This document's own synthesis decision, flagged in §9:** `-ml` §5.2 does not state population
    vs. sample standard deviation. This uses `statistics.pstdev` (population) — the 121-document
    corpus IS the full population each query is scored against, not a sample drawn from a larger
    one, so the population estimator is the more defensible reading; the `-ml` note was not asked to
    disambiguate this session and the coder should confirm rather than trust this choice silently."""

# --- BM25 (`-ml` §5.3, all parameters from pack.json's "embedding.bm25" block, never hardcoded) ---

_BM25_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

def bm25_tokenize(text: str) -> list[str]:
    """Unicode-aware `re.findall(r"\\w+", text.casefold())` — `-ml` §5.3 verbatim. No stemming."""

@dataclass(frozen=True)
class Bm25Index:
    docIds: tuple[str, ...]
    docTermFreqs: tuple[Mapping[str, int], ...]   # per doc, post-stopword-removal
    docLengths: tuple[int, ...]
    avgDocLength: float
    documentFrequency: Mapping[str, int]           # term -> number of docs containing it
    n: int
    k1: float
    b: float

def build_bm25_index(
    corpus_rows: Sequence[Mapping[str, str]], *, k1: float, b: float, stopwords: frozenset[str],
) -> Bm25Index: ...

def bm25_score(index: Bm25Index, query: str) -> dict[str, float]:
    """Every doc's score for `query`. Always-positive IDF (`-ml` §5.3 verbatim):
    idf(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))."""

def load_stopwords(pack: Pack) -> frozenset[str]:
    """Reads `pack.manifest["embedding"]["bm25"]["stopwordsFile"]` via `pack.root / <name>` — a
    plain-text, one-word-per-line file, `#`-prefixed comment lines skipped."""
```

**Test 8** (plan §5 item 8, `:6040-6041`) drives `recall_at_k`/`mrr`/`precision_at_k`/
`score_separation_raw`/`score_separation_z`/`rank_by_cosine` entirely against hand-built ranked
lists and synthetic vectors — no `corpus.embeddings.json`, no pack, no live call. This is what makes
step 1 (§8) fully offline-testable before `corpus.embeddings.json` exists at all.

### 6.2 `prime`/`embed_text` — the corpus-embedding hook

```python
# Process-local state, written once per run by prime(), read by score_item/aggregate/embed_text.
# Safe under §3.2's "one run, one process, never concurrent" argument; a direct-unit-test caller
# resets it by calling prime() again before each case (§8).
_state: dict[str, Any] = {}


def prime(
    *, lmstudio: LMStudio, model_info: ModelInfo,
    call_surface: Literal["chat", "embeddings"], timeout_s: float, pack: Pack,
) -> None:
    """Embeds/caches the 121-document corpus once, via the SAME live model under test (§3.1). Reads
    `pack.data_path("corpus")` and `pack.data_path("corpusEmbeddings")` (the committed
    `corpus.embeddings.json`, if present); computes this run's cache key
    (`model_info.id`, `model_info.quantization`, `pack.manifest["embedding"]["documentPrefix"]`,
    sha256 of `corpus.jsonl`'s bytes) and compares it against the committed file's own `cacheKey`
    (§3.1). On a match: reuses the committed RAW vectors, no live call for the corpus. On a
    mismatch or an absent file: embeds `corpus.jsonl` live (documentPrefix-applied, one batched
    `lmstudio.embed()` call) — the result is NOT persisted (§3.1's rejected-alternative note).
    Also builds the BM25 index (§6.1) and records the raw norm distribution (below, `score_item`'s
    `detail` field)."""


def embed_text(item_input: Mapping[str, Any], *, pack: Pack) -> str:
    """`pack.manifest["embedding"]["queryPrefix"] + item_input["query"]` — the exact string the
    runner embeds for this item (§3.2, closing the gap §2.2 names)."""


def score_item(
    item_input: Mapping[str, Any], result: EmbedResult | None, timing: ItemTiming, *, pack: Pack,
) -> ItemResult:
    """One query. `result is None` (timeout/no-response, runner spec §3.2) scores `outcome=
    "fail"`/`"unrunnable"` per `timing.withheldFor`, `scoreable={}` — no `measures`/`counts` at all
    (a query the model never answered contributes nothing to recall@k/MRR, matching `-ml` §4.3's
    "an item's outcome is declared, never inferred": there IS no ranking to score). Otherwise: reads
    `_state["corpusVectors"]`/`_state["bm25Index"]` (written by `prime()` — raises `RuntimeError` if
    `prime()` was never called, a programming-contract check, not a pack-config one), computes
    `rank_by_cosine(result.vectors[0], corpus_vectors)`, and fills:

    - `counts`: one int per configured k (`"recallAt5"`, `"recallAt10"`, from
      `pack.manifest["metrics"]["recallAtK"]["ks"]`) plus `"precisionAt1"` — `RetrievalAggregates.
      recallAtK` is shipped as `tuple[BinaryMetric, ...]` (§2.1, non-negotiable), so each item's
      contribution must be a single binary success flag even though `recall_at_k(...)` itself
      returns a continuous fraction on the 2-of-38 multi-relevant items (`-ml` §5.1). **This
      document uses `1` iff `recall_at_k(...) > 0` (at least one relevant doc retrieved in top k)
      else `0`** — flagged as its own synthesis decision in §9, since neither the plan nor `-ml`
      §5.1 states how a fractional `recall_at_k` binarizes into `BinaryMetric`'s successes/n, and
      the alternative reading (`== 1.0`, "found every relevant doc") is equally defensible and
      disagrees with this one only on those 2 items. `counts[metric] > 0` is exactly what
      `ItemResult.scored_outcome` reads (`results.py:416`, confirmed) — each count carries a
      matching `scoreable[metric] = True` entry.
    - `measures`: `{"mrr": <float>, "separationRaw": <float>, "separationZ": <float>}` — all three
      continuous (§4 S1e Table F's split, §2.1), never boolean-counted.
    - `detail`: `{"queryNorm": l2_norm(result.vectors[0])}` — the per-query raw-norm reading, the
      concrete form §5.2's "record the raw norm distribution" takes for a query (the corpus side is
      `prime()`'s own diagnostic, logged rather than stored on any `ItemResult` — no corpus doc is a
      scored item).
    """


def aggregate(items: Sequence[ItemResult], *, pack: Pack) -> RetrievalAggregates:
    """One pass over `items` (mirrors `LatencyBlock`'s own "computed from run.items in one pass"
    discipline, runner spec §5): `recallAtK` as `tuple[BinaryMetric,...]` (one per configured k,
    `unit="query"` — `roles.UNIT_KIND_BY_ROLE["embedder"] == "query"`, §2.1), `precisionAt1` as
    `BinaryMetric(unit="query")`, `mrr` as `ContinuousMetric(support=(0.0, 1.0))`, `separationRaw`/
    `separationZ` as `DistributionSummary(median=..., p10=..., n=..., unit="query",
    support=(-1.0,1.0)|None)` — `-ml` §5.2's median/p10/fraction-above-zero aggregation (the
    fraction-above-zero figure is NOT stored, per §3.8.1's identity: it IS `precisionAt1`, so
    storing it a second time would be a second home for one number). Every item lacking a `mrr`
    entry in `measures` (a timed-out/unrunnable query) is excluded from all five aggregates' inputs,
    never coerced to 0 — the same "declared, never inferred" discipline `score_item` observes."""
```

### 6.3 The BM25 deterministic arm

```python
def deterministic_arm(*, pack: Pack, session_id: str | None) -> RunResult:
    """§3.3's resolution: builds the BM25 reference-arm RunResult with NO live LM Studio call.
    Reads `corpus.jsonl`/`queries.jsonl` directly via `pack.data_path`, `build_bm25_index` +
    `bm25_score` per query, the SAME `recall_at_k`/`mrr`/`precision_at_k`/`RetrievalAggregates`
    machinery §6.1/§6.2 use for the model arm (one scoring path, two arms — never a second,
    BM25-specific metric implementation). `ItemResult.timing` is `None` on every item (§4 S1's
    "None iff this arm produces no timings at all", results.py:344 — BM25 makes no model call, so
    there is nothing to time); `RunResult.latency` is therefore `None` too (`LatencyBlock`'s own
    rule (v), runner-spec §5). `Fingerprint` is built by `_build_deterministic_fingerprint` (below).
    `runId = f"{pack.packId}-bm25-{utc_stamp}"` — distinct from the model arm's runId, same
    `sessionId`."""


def _build_deterministic_fingerprint(*, pack: Pack, started_at, ended_at) -> Fingerprint:
    """The eleven required `deterministic`-profile fields (`fingerprint.py:131-143`, §2.1):
    `armId="bm25"`, `armParametersHash` (sha256 hex of
    `json.dumps({"k1":..., "b":..., "tokenization":"unicode-word-casefold",
    "idfVariant":"always-positive", "stopwordsSha256":...}, sort_keys=True)` — the arm's OWN knobs,
    deliberately not `packContentHash` again, which is a separate required field already covering
    the whole pack), `packId`, `packVersion`, `packContentHash` (`pack.contentHash`), `benchVersion`
    (`modelbench.__version__`), `benchSchemaVersion`, `pythonVersion`, `hostOs`, `startedAt`,
    `endedAt`. `Fingerprint(armKind="deterministic", callSurface=None, fields={...})`."""
```

## 7. Runner/CLI wiring — the required edits to already-shipped S2 code

### 7.1 `runner.py` — `_load_item_scorer`, the real dynamic import

```python
def _load_item_scorer(pack: Pack) -> ItemScorer:
    """§3.2's own naming choice, now wired: resolves `pack.manifest["scorer"]` to
    `modelbench.scoring.<name>` and returns the module itself — a module satisfies `ItemScorer`
    structurally (its `score_item`/`aggregate` module-level functions ARE the Protocol's methods;
    Python's structural typing does not require a class instance, and this is never `isinstance`-
    checked). Raises `RunRefused(exitCode=4)` on an absent `"scorer"` key or an unresolvable module
    name — a pack-config defect, not a programming-contract one, so it DOES get the CLI's exit-4
    path (unlike `embed_text`'s AttributeError, §3.2)."""
    name = pack.manifest.get("scorer")
    if not name:
        raise RunRefused(f"pack {pack.packId!r} declares no \"scorer\"", exitCode=4)
    try:
        return importlib.import_module(f"modelbench.scoring.{name}")
    except ImportError as exc:
        raise RunRefused(
            f"pack {pack.packId!r} declares scorer {name!r}, which does not resolve to "
            f"modelbench.scoring.{name}: {exc}", exitCode=4,
        ) from exc
```

`_load_conversation_scorer` is untouched — `tool-caller`'s own first scorer is S5's, out of this
document's scope (§1).

### 7.2 `cli.py` — `_cmd_run`, the deterministic-arm hook

```python
def _cmd_run(args: argparse.Namespace) -> int:
    ...
    try:
        run, disclosures = run_pack(pack, cfg, lmstudio=lmstudio, root=Path(args.root))
    except RunRefused as exc:
        print(f"model-bench: {exc}", file=sys.stderr)
        return exc.exitCode

    path = store(run, Path(args.root))
    print(f"stored: {path}")

    if pack.role != "tool-caller":                                       # NEW — §3.3
        scorer = _load_item_scorer(pack)
        deterministic_arm = getattr(scorer, "deterministic_arm", None)
        if deterministic_arm is not None:
            arm_run = deterministic_arm(pack=pack, session_id=cfg.sessionId)
            arm_path = store(arm_run, Path(args.root))
            print(f"stored (deterministic arm): {arm_path}")

    if disclosures:
        ...  # unchanged, dispatch-failure note's funnel-head block
    return EXIT_OK
```

`getattr`-guarded exactly like `prime`/`embed_text` (§3.2) — every role but `embedder` has no
`deterministic_arm`, so this is a no-op for `guard-judge`/`nlq-generator`/`chat-responder` and is
skipped outright for `tool-caller` (which resolves its scorer through `_load_conversation_scorer`,
a different function this hook does not touch).

## 8. Step sequence for implementation

Three steps, matching this coordination's own split boundary (3+ steps or 5+ files) and the "offline
before live" ordering the runner spec's own step 1 established (§7 there): everything that can be
red→green without a real LM Studio connection ships before the one step that needs one.

### Step 0 — Static pack data, no live call

Creates: `packs/embedder-graphrag-retrieval/{pack.json, queries.jsonl, corpus.jsonl,
golden_retrieval.embeddings.json, retrieval_baseline.json, bm25_stopwords.txt, PROVENANCE.md}`,
`scripts/refresh_golden.py`'s default (`copy`/`jsonl-transform`/`ast-literal`) and `--check-origins`
modes (§5, all four `OriginSpec` kinds except `--embed-corpus`), `tests/fixtures/
metrics_agreement.json` (the 20 hand-transcribed cases, §3.1 point 2(a) — **manual transcription
from `falkor-chat/server/tests/eval/test_metrics.py`, not mechanical extraction**, per the plan's
own warning that 14 of 20 cases live outside `@pytest.mark.parametrize` tables and a mechanical
extractor would silently miss them). `tests/test_refresh_golden.py` asserts the cache-key shape
(§3.8.1: *"the cache key is asserted in a unit test, not just implemented"*) against a synthetic
corpus/model-info fixture — no real corpus yet. Done when: `refresh_golden.py --check-origins`
against the real `falkor-chat` tree (still present on this machine, confirmed §1) reports all five
origins unchanged; `docId` values in `corpus.jsonl` match `queries.jsonl`'s `relevantDocIds` exactly
(a direct set-equality check, not eyeballed).

### Step 1 — `scoring/retrieval.py`'s full implementation + the runner/CLI wiring, offline-testable throughout

Ships §6's whole module (§6.1's pure arithmetic, §6.2's `prime`/`embed_text`/`score_item`/
`aggregate`, §6.3's `deterministic_arm`) and §7's two wiring edits (`runner.py`'s `_load_item_scorer`
real import + `_drive_single_call_items`'s two-line change, `cli.py`'s `_cmd_run` hook). Every piece
is testable with a `StubLMStudio`/`FakePack` exactly like `tests/test_runner.py`'s own S2 fixtures
(§2.2) — `prime()`'s corpus-embedding path is driven with a synthetic 3-document corpus and a stub
`lmstudio.embed()` returning known vectors, never a real one. Order within the step, red→green:

1. Test 8's hand-built-ranked-list cases (§6.1) — `recall_at_k`/`mrr`/`precision_at_k`/
   `score_separation_raw`/`score_separation_z`/`rank_by_cosine`, including multi-relevant items,
   fewer than *k* results, and zero relevant found (plan §5 item 8, verbatim).
2. `test_metrics_agreement.py` — all 20 transcribed cases (plan §5 item 11, `:6096-6098`), reading
   only `tests/fixtures/metrics_agreement.json` — no `falkor-chat` path touched by this test.
3. BM25 unit tests — `bm25_tokenize`, `build_bm25_index`, `bm25_score` against a small synthetic
   corpus with a hand-computed expected score (the always-positive IDF variant, `-ml` §5.3).
4. `prime()`/`embed_text()` unit tests — the cache-key match/mismatch branches (committed-file reuse
   vs. live embed), both with a `StubLMStudio`; `embed_text` applies `queryPrefix` correctly.
5. `score_item`/`aggregate` unit tests — the timeout/no-response no-`measures` branch; a clean item
   producing `mrr`/`separationRaw`/`separationZ` in `measures` and `recallAtK`/`precisionAt1` in
   `counts`; `aggregate`'s exclusion of a `measures`-absent item from every aggregate's input.
6. `deterministic_arm()` unit test — a fixed 3-query/5-document synthetic pack producing a
   `RunResult` with `armKind="deterministic"`, `timing is None` on every item, `latency is None`.
7. `runner.py`/`cli.py` wiring tests — `_drive_single_call_items` with `call_surface="embeddings"`
   and a `FakeItemScorer` that DOES define `prime`/`embed_text`, asserting both are called with the
   right arguments and that the embed call's text is `embed_text`'s return value, not
   `json.dumps(item_input)`; `_load_item_scorer` resolving `"scorer": "retrieval"` to the real
   module (an integration-shaped test, still no live LM Studio); `_cmd_run`'s deterministic-arm hook
   storing a second record under the same `sessionId`, asserted via `store()`'s own file output.

### Step 2 — The live end-to-end run (done-conditions 1, 2, 4, 5)

`-m live`, needs a reachable LM Studio with `text-embedding-qwen3-embedding-0.6b` loadable (the
standing authorization for an agent to trigger a model load autonomously for testing is already in
force — not re-litigated here). In order:

1. `refresh_golden.py --embed-corpus --model text-embedding-qwen3-embedding-0.6b` — writes
   `corpus.embeddings.json` (§5's `embed_corpus`). This is the pack's own committed artifact from here on; a
   content-hash-changing write, so `packVersion` must already be bumped (§5's `main()` refusal).
2. **Done-condition 4** — with `corpus.embeddings.json` and `golden_retrieval.embeddings.json` now
   both real, feed them straight into `rank_by_cosine`/`score_item`'s ranking path with **no live
   embedding call** and assert the rankings/metrics reproduce (plan: *"the ranking path is shown to
   reproduce identical rankings from the two fixed vector files with no live embedding call at
   all"*) — this specific assertion CAN run offline once step 1 ships, but is sequenced here because
   it needs the real `corpus.embeddings.json` this step's first action just wrote.
3. **The harness self-check** (`-ml` §5.4, plan done-condition 5) — run recall@10 against the same
   121-doc corpus/38 queries with the live model and compare to `retrieval_baseline.json`'s pinned
   0.974. **Diagnostic, never a gate** (stakeholder decision, 2026-09-02, plan `:5689-5691`): whatever
   the result, S3 proceeds. Write the outcome into `model-bench/docs/test-reports/embedder-self-
   check-report.md` (topic slug `embedder-self-check`, `-report` role — a new topic, not a family
   member of this spec document, which stays at the repo root per the footnote convention `model-
   bench/AGENTS.md`'s Documentation map states; root `AGENTS.md`'s filename grammar governs the
   name) — **required contents, if the result lands below ~0.85** (`-ml`
   §5.4): which of the three known causes were checked (wrong prefix, unnormalized vectors,
   truncated corpus) and what was found for each; **required either way**: the raw self-check number
   itself, printed and read, never silently passed over. The write-up must NOT frame either
   direction as "explained" by exact-vs-ANN or vector-only-vs-hybrid (`-ml` §5.4: *"a disagreement
   of either sign is uninterpretable... and must not be 'explained'"*).
4. `run --pack embedder-graphrag-retrieval --model text-embedding-qwen3-embedding-0.6b` —
   **done-condition 1**: the real end-to-end run, `prime()` hits a cache HIT against the just-written
   `corpus.embeddings.json` (same model → no live corpus re-embed on this specific run), 38 live
   query embed calls, a stored `RunResult` with a complete `model` fingerprint.
5. The `_cmd_run` hook (§7.2) stores the BM25 deterministic arm under the same `sessionId` —
   **done-condition 2**. `compare --pack embedder-graphrag-retrieval --session <id>` renders both
   arms in one report, the deterministic one labelled (already-generic rendering, §2.1 — no
   `report.py` change needed).
6. **Done-condition 3** is step 0/1's own (the 20-case agreement test + `--check-origins`), not
   re-run here — listed in §4 S3's done-condition table but already closed by step 1.

## 9. Risks & open questions

- **How a fractional `recall_at_k`/`precision_at_k` binarizes into `BinaryMetric`'s per-item
  successes/n is this document's own synthesis decision** (§6.2), because `RetrievalAggregates.
  recallAtK` is shipped as `tuple[BinaryMetric, ...]` (a strict binary-outcome carrier) while
  `recall_at_k(...)` itself is a continuous fraction on the golden set's 2 multi-relevant items
  (`-ml` §5.1). This document counts a success as "at least one relevant doc in the top k"
  (`recall_at_k(...) > 0`); "all relevant docs found" (`== 1.0`) is an equally defensible reading
  that disagrees only on those 2 items out of 38. Both readings reproduce `retrieval_baseline.json`'s
  pinned `recall_at_10 = 37/38` figure only if the two multi-relevant items' actual per-query
  fractions happen to coincide at 0 or 1 either way — not verified this session (the raw ranked
  lists behind that pinned figure are not an artifact, per §3.1 point 2's own finding). **This needs
  a `data-scientist`/`tico` ruling before the report ships a number under the label "recall@k"**,
  since the two readings can diverge on real data even though the illustrative case above cannot
  distinguish them.
- **`score_separation_z`'s standard-deviation estimator (population vs. sample) is this document's
  own synthesis decision, not verified against `-ml` §5.2's own text**, which states `sd({cos(q,d) :
  d in corpus})` without disambiguating (§6.1). Recommendation: population (`statistics.pstdev`,
  the choice made above) on the "121-doc corpus is the whole population being scored, not a sample"
  argument, but this needs `data-scientist` confirmation before it ships as a formula the report
  publishes — it is exactly the kind of "numeric threshold this document could not verify" the
  runner spec's own §9 flagged a precedent for, and this document is doing the same rather than
  guessing silently.
- **The `ItemScorer.prime`/`embed_text`/`deterministic_arm` extension (§3.2, §3.3) is this
  document's own design, unreviewed.** It is additive, backward-compatible with every shipped S2
  test (confirmed against `tests/test_runner.py`'s `FakeItemScorer`, §2.2), and resolves two real
  gaps the shipped runner cannot otherwise close for this pack — but the runner spec's own §3.2 flagged
  the scorer seam generally as "a reasonable, minimal seam... but it has never been reviewed", and
  this document's extension of it inherits that same caveat. **Recommendation:** an independent
  review pass (the repo's own default-independent-review convention) before S4's `guard-judge`/
  `nlq-generator` scorers are built on top of the same seam, since a second role's needs may reveal
  the extension is shaped wrong for anything beyond the embedder.
- **`packs.py`'s `validate_pack` does not check that a pack's declared `"scorer"` name resolves to an
  importable module** — `run` will therefore fail at `_load_item_scorer`'s import (a `RunRefused`,
  exit 4) rather than at `validate`, for a pack with a typo'd scorer name. Named here as a possible
  `validate_pack` follow-up, not this stage's to fix (§1's explicit out-of-scope line) — low risk,
  since the failure is still loud and still the correct exit code, just one step later than it could
  be.
- **`corpus.embeddings.json` stores RAW (non-normalized) vectors** (§5's `embed_corpus`,
  §6.2's `prime`) so the offline self-test can observe the raw norm distribution the same way a live
  run does. If a future reader assumes the committed file's vectors are pre-normalized and feeds
  them into a cosine computation that skips normalization, scores silently corrupt without an error
  — worth a one-line comment in the file itself (a `"normalized": false` header key) so a future
  reader cannot get this wrong by omission; not written into §5's signature above because it is
  small enough for the implementer to add without a design decision, but flagged here so it is not
  forgotten.
- **`refresh_golden.py --embed-corpus`'s exact CLI flag surface (`--model`, `--api-base-url`) is
  this document's own naming, not cited from the plan** — the plan never gives this script a full
  flag table the way it does `run`'s (§3.6a). Low risk: it is a maintenance script with no stored
  contract depending on its flag names, unlike `run`'s CLI surface.
- **Whether a future embeddings-surface role (none named yet) should get `embed_text` as a required
  rather than optional Protocol member is left open** (§3.2) — revisit when a second such role is
  designed; nothing here forecloses tightening it later.

## Ready to implement

Document: `docs/plans/small-model-benchmarking-s3-spec.md` (this file). Three steps (§8): **step 0**
(static pack data + `refresh_golden.py`'s no-live-call modes + the hand-transcribed 20-case
agreement fixture), **step 1** (`scoring/retrieval.py`'s full implementation + the required
`runner.py`/`cli.py` wiring edits — §3.2's `ItemScorer.prime`/`embed_text` extension and §3.3's
`deterministic_arm` CLI hook — all offline-testable with stubs), **step 2** (the live end-to-end run:
`--embed-corpus`, the offline-from-fixed-vectors self-test, the harness self-check diagnostic
write-up, the real `run` invocation, the BM25 arm's storage). Two genuine plan-vs-shipped-code gaps
found and resolved as part of this design, both flagged the way the runner spec's own §2.2 flagged
its two (§2.2-§2.3 above): the embeddings call branch's hardcoded, unprefixed input text, and the
complete absence of any seam for embedding a reference corpus or storing a second, deterministic
`RunResult` from one `run` invocation. Three genuine synthesis decisions are marked and need
attention before they ship as settled fact: two need a `data-scientist`/`tico` ruling before the
report publishes a number under their label (`score_separation_z`'s stdev estimator, and how a
fractional `recall_at_k` binarizes into `BinaryMetric`'s successes/n, §9), and the `ItemScorer`
extension itself needs an independent review pass before a second role builds on the same seam (§9).
