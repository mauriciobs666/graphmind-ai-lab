# `model-bench` S4 — the `guard-judge` and `nlq-generator` packs — implementation spec

> **Status:** active · **Owner:** `architect` · **Tracks:** — · **Extends:** `docs/plans/small-model-benchmarking.md` (S4)

## 1. Goal & scope

`docs/plans/small-model-benchmarking.md` (the "plan") assigns `guard-judge-understanding` and
`nlq-structured-query` — two single-call-per-item chat-surface packs — to stage S4 (§4 S4,
`:5699-5715`), and gives each pack's mechanism, scoring and reporting rules in detailed prose
(§3.8.2 `:2076-2115`, §3.8.3 `:2117-2177`). Unlike S3's `embedder` pack, whose own precursor
(`docs/plans/small-model-benchmarking-runner-spec.md`) had already exercised the `ItemScorer` seam
end to end, S4 is the **first stage to drive the chat branch's per-item prompt construction with a
real prompt** — and `model-bench/docs/reviews/itemscorer-extension.md` (the "review"), an
independent seam-readiness pass written specifically to inform this planning, found the seam is not
ready: two real gaps, one of them a live correctness defect waiting to fire on this stage's first
run. This document resolves both gaps as an explicit early step, then specifies both packs at file
level.

**In scope:** the two gaps the review found (a fourth `ItemScorer` Protocol method for chat-branch
prompt construction; `Pack.prompt_config()`'s unresolved `systemPrompt`/`toolSchemas` paths) and
their resolution, both design questions the review left open (resolved in §4.1/§4.2 below);
`packs/guard-judge-understanding/` and `packs/nlq-structured-query/` at file level; the two golden
sets' import (extending `scripts/refresh_golden.py`, which today only knows the embedder pack);
`modelbench/scoring/classification.py` and `modelbench/scoring/extraction.py`; `nlq-structured-
query`'s pack-local `tools/exec.py` (validation + execution surface, stdlib only); the answerability
stamping step; a small, additive extension to the already-shipped `ExtractionAggregates` type.

**Out of scope:** `tool-caller`/`chat-responder`'s scorer modules (S5-S7); any change to
`report.py` (§3.1 confirms none is needed — see below); any change to the already-shipped
`ClassificationAggregates`/`ExtractionAggregates` shapes beyond the two new integer fields §4.3
adds; a `validate_pack` check that a chat-surface item-level pack's scorer defines `build_messages`
(considered in §4.1, deferred as a named follow-up, not built here).

**CPG:** considered, not relevant — `model-bench` is a Python component with no `cpg_model-bench`
graph loaded on this FalkorDB instance (checked live this session); this is a code-level task, so
"considered, not relevant" applies rather than "not applicable".

## 2. Context & findings

### 2.1 What already exists and needs no S4 work — confirmed by reading the shipped code directly

- **`ClassificationAggregates`/`ExtractionAggregates` are already shipped types** (`results.py:531-
  553`, S1's, per §4 S1e Table F), constructed nowhere yet (`grep` confirms no scorer module builds
  one today) but already exercised by S1's own report tests
  (`tests/test_report.py:1196-1430` builds `ClassificationAggregates` fixtures using the exact
  metric name `falseAdvanceRate` this stage needs). `ClassificationAggregates.perClass: tuple[
  BinaryMetric, ...]` is a flat, unordered bag with **no field that could hold a pooled 85-item
  accuracy** — `report.py`'s own module docstring states this as a structural refusal, not a
  convention: *"there is no path that pools a per-class table into one accuracy figure, because
  `ClassificationAggregates` has no field to hold one"* (`report.py:11-12`). S4 does not need to
  avoid computing a pooled figure through discipline; the type cannot carry one.
- **The Holm-Bonferroni family machinery, the `headlineMetric: null` render path, and the
  exploratory-label fallback are all already shipped and already tested against exactly this
  pack's shape.** `PackMetrics.alpha_family`/`alpha_mdd` (`packs.py:76-96`) already compute the
  unadjusted floor α and the `α/k` MDD α from `verdictMetrics`'s length alone — for a two-member
  family this is precisely `-ml` §7.3's `0.05`/`0.025` split, with no S4-side arithmetic. `report.py`
  already special-cases a homogeneous **binary** family with `k > 1` through the unchanged two-pass
  Holm ladder (`report.py:846-849`), already renders a metric outside `verdictMetrics` as
  `"exploratory — no significance claim"` generically (confirmed at the S3-spec citation, re-
  confirmed here against `report.py` directly), and `metrics_from_manifest` (`packs.py:141-157`)
  already accepts and requires an explicit `"headlineMetric": null` key. **Conclusion: S4 does not
  touch `report.py` at all.** Its whole job is to make the scorer populate `ClassificationAggregates`
  correctly; every rendering rule §3.8.2 asks for is a consequence of the pack's own manifest
  declaration plus already-shipped, already-tested machinery.
- **`roles.py` already carries both roles' unit-kind and analysis-unit-field rows**
  (`UNIT_KIND_BY_ROLE["guard-judge"] == UNIT_KIND_BY_ROLE["nlq-generator"] == "item"`,
  `ANALYSIS_UNIT_FIELD_BY_ROLE[...] == "itemId"` for both, `MULTI_CALL_TURN_BY_ROLE[...] is False`
  for both) — confirmed by reading `roles.py` directly. Both packs' `sampling` block is therefore
  `{"pairingKey": ["itemId"], "analysisUnit": "itemId"}` with no `"scripts"` key, the same item-level
  shape already validated clean by S2's own guard-judge-role fixtures
  (`tests/fixtures/packs/scorer_unresolvable/pack.json` and siblings).
- **`Outcome = Literal["pass", "fail", "n_a", "parse_failure"]`** (`results.py:38`) is the declared
  type — but shipped `scoring/retrieval.py:421` already assigns `outcome = "fail" if
  timing.withheldFor == "timeout" else "unrunnable"`, and `"unrunnable"` is asserted directly in
  `tests/test_runner.py:879`/`tests/test_scoring_retrieval.py:582`. This is a **pre-existing type/
  usage drift in already-accepted S3 code**, not something S4 introduces — flagged in §9, not fixed
  here (S4 replicates the established, tested convention for consistency rather than inventing a
  third one).

### 2.2 The `ItemScorer` extension review — both findings, and this document's rulings

The review (read in full) is a completed, independent seam-readiness pass written specifically to
inform this plan. Its scope and evidence are not repeated here; this section states the two open
design questions it deliberately left to whoever planned S4, and rules on both.

**Finding 1 (major) — no chat-branch hook for per-item prompt construction.**
`_drive_single_call_items`'s chat branch calls `_item_chat_messages(pack, item_input)` unconditionally
(`runner.py:300-314`, `347-354`) — a generic `json.dumps(item_input, sort_keys=True)` dump, by that
function's own docstring "only what lets the runner issue a real call for a stub or fixture item
today." Neither `guard-judge`'s labelled multi-block render nor `nlq-generator`'s schema-filled
structured-completion prompt can be produced this way (confirmed independently in §2.4/§2.5 below,
reading the real falkor-chat mechanisms both packs transcribe).

*Open question the review left: should the fourth method be **required** for any item-level role
declaring `lmstudio-chat`, or stay optional?*

**Ruling: optional, `getattr`-guarded — exactly like `prime`/`embed_text`/`deterministic_arm`,
never a no-guard call the way `embed_text` is reached on the embeddings branch.** The two designs
are not symmetric, and treating them the same would be wrong. `embed_text` is reached with no guard
because the embeddings branch is, today, reached by exactly one role (the embedder) and no S2 test
drives that branch with a scorer lacking it (S3 spec §3.2, re-confirmed: `grep` finds no
`call_surface="embeddings"` fixture without `embed_text` anywhere in `tests/`). The **chat** branch
is the opposite case: it is the branch every S1/S2 test already drives (`grep -c
'call_surface="chat"' tests/test_runner.py` → 19 call sites), overwhelmingly through
`FakeItemScorer`/`FakeItemScorerWithHooks` (`tests/test_runner.py:773`, `966`), **neither of which
defines `build_messages`**. A no-guard call would raise `AttributeError` on every one of those
already-accepted, already-passing S1/S2 tests — a real regression, not a hypothetical one. Keeping
it `getattr`-guarded, falling back to `_item_chat_messages` exactly as the review's own suggested
fix describes, is fully backward compatible (confirmed: no shipped test asserts `build_messages` is
called, so none can regress) and is the only design that does not reopen already-accepted S1/S2
work. **What makes the omission loud in practice, without a new enforcement mechanism:** both S4
scorers define it (§5 below is unbuildable otherwise — neither prompt is reachable from a JSON
dump), and this document's own step sequence (§7) ships a runner-level wiring test asserting the
chat branch calls `scorer.build_messages` in preference to the fallback whenever the scorer defines
it — the same pattern S3's own step 1.7 used for `embed_text`/`prime`. A `validate_pack`-level
requirement (a chat-surface item-level pack's declared scorer module must define `build_messages`)
is named here as a **follow-up**, not built: it would need `validate_pack` to first resolve
`pack.manifest["scorer"]` to an importable module at all, which it does not do today (S3 spec §9's
own flagged, still-open gap) — reopening that gap is a larger unit than S4's own scope and is not
blocking, since the failure mode without it (a future pack's `_item_chat_messages` JSON-dump
fallback firing silently) is loud in the reviewed sense that it produces a *visibly wrong* prompt on
first live inspection, not a crash — the same category of risk S3 already accepted for
`load_stopwords`'s untyped read.

**Finding 2 (major) — `prompt.systemPrompt`/`toolSchemas` are declared as paths, consumed as
content, everywhere.** `Pack.prompt_config()`'s own docstring states both fields are "carried
through as the manifest's own declared values — paths, not resolved content" (`packs.py:366-370`),
while `convo.PromptConfig`'s docstring — the **same class**, `packs.py` imports it from `convo.py`
(confirmed: one `PromptConfig`, not two) — asserts the opposite: `systemPrompt`/`toolSchemas` "are
the **resolved content**" (`convo.py:148-153`). Every real consumer already reads the unresolved
value as if it were resolved: `_item_chat_messages` puts `cfg.systemPrompt` straight into a chat
message's `content` (`runner.py:311-312`); `run_pack`'s two `warm_up` calls pass it as
`system_prompt=` verbatim (`runner.py:794-798`, `810-814`); `convo.assemble` does the same
(`convo.py:376-378`, not read directly this session but cited identically by the review and
consistent with every other `PromptConfig` consumer). The one real fixture that declares a real
path, `tests/fixtures/packs/valid/pack.json`'s `"systemPrompt": "prompts/system.md"`, is never
asserted against by `test_prompt_config_resolves_the_valid_packs_prompt_block`
(`tests/test_packs.py:723-735`, read directly — it asserts `historyReplay`/`representToolSchemas
EachTurn`/`historyTurns`/`maxIterationsPerTurn`/`temperature`/`maxTokens` only, never
`systemPrompt`/`toolSchemas`), so the gap is real and currently untested.

*Open question the review left: precursor unit ahead of S4, or folded into S4 Step 0/1?*

**Ruling: folded into S4 Step 0, not a separate precursor unit.** Three reasons, all specific to
this fix rather than generic preference for fewer units: (1) it is small — one new `Pack` method
mirroring `data_path`'s existing resolution pattern, plus reconciling two docstrings that already
agree on the *intent*, just not on which function implements it; (2) it is **currently inert** (no
chat-surface pack ships live yet, confirmed by the review) and gains a real first caller only when
`build_messages` is built, in this same stage — a precursor unit would ship a resolver with no real
consumer, the exact "unbuilt, no real caller yet" state that let the original gap go unexercised for
two stages; building the resolver and its first real caller together in one step guarantees it is
exercised against a real pack (`guard-judge-understanding`'s `prompts/judge.md`) rather than only a
synthetic fixture; (3) both land in `packs.py`, and `runner.py`'s chat-branch edit (finding 1's fix)
sits directly beside `run_pack`'s two `warm_up` call sites finding 2 also touches — one implementer,
one pass, one review, per the review's own stated efficiency argument.
**Verified compatible with the one real existing fixture:** `tests/fixtures/packs/valid/
prompts/system.md` exists on disk and is readable (confirmed: `cat` returns its placeholder text);
`tests/fixtures/packs/valid/tools/schemas.json` is `[]`, a valid JSON array. Neither existing
assertion reads `cfg.systemPrompt`/`cfg.toolSchemas`, so resolving both to real content changes no
shipped assertion's outcome — confirmed directly, not inferred.

### 2.3 Two more real gaps, found independently this session, not in the review

The review scoped itself to the `ItemScorer` seam; two further gaps sit beside it, both required to
build S4's scorers and neither previously named:

1. **`ExtractionAggregates.parseFailures: int` is one scalar, but §3.8.3's cost note requires three
   distinguishable failure counts** ("a malformed spec, a schema violation and a wrong answer each
   land in their own count"). "Wrong answer" already falls out of `exactMatch.n − exactMatch.
   successes`; "malformed spec" (fails the hand-rolled `QueryRequest`-equivalent structural
   validation) and "schema violation" (structurally valid, fails `compile()`-equivalent checks —
   unknown label/property, duplicate `returns`) have no field to land in. §4.3 below adds two new,
   defaulted integer fields — confirmed backward compatible by reading every existing construction
   site (`grep -rn "parseFailures" tests/ modelbench/"` → all six are keyword-only; a frozen
   dataclass's new defaulted field breaks no keyword construction).
2. **`golden_guards.jsonl`'s `turns` rows carry the raw `repository.read_thread` message shape**
   (`msgId, text, role, createdAt, authorId, displayName, authorType` — confirmed by reading a
   `path: "turns"` row directly), not the `{speaker, role, text}` shape `_render_judge_user`
   actually reads (`app.py:642`: `t.get('speaker', 'member')`/`t.get('text', '')`). Production
   normalizes between the two with `guards._recent_turns` (`guards.py:538-567`) **before**
   `_render_judge_user` ever sees a turn. A transcription that renders `_render_judge_user` alone,
   skipping `_recent_turns`, would read `t.get('speaker', 'member')` against a row that has no
   `speaker` key at all — every turn silently rendering as `"member"` — a correctness defect, not a
   cosmetic one, since the judge's own real system genuinely distinguishes speakers in its prompt.
   §5.1 below transcribes both functions, not one.

### 2.4 `guard-judge`'s real mechanism, transcribed — verified against `falkor-chat/server/falkorchat/app.py` and `guards.py` directly

- **System prompt** (`app.py:602-612`, `_JUDGE_SYSTEM_PROMPT`) — one fixed string: *"You are a
  strict gate deciding whether a workflow may advance. ... Reply with a single JSON object and
  nothing else ... {"decision": <true|false>, "rationale": "<one short sentence>"}. Answer true ONLY
  when the condition is clearly satisfied; when in doubt answer false. ..."* — transcribed verbatim
  into `packs/guard-judge-understanding/prompts/judge.md`, resolved as `prompt.systemPrompt`'s
  content via §4.2's fix.
- **User message** (`app.py:619-654`, `_render_judge_user`) — `"CONDITION: {condition}"`, then, iff
  `understanding` is non-empty, `"CURRENT STATE:\n{json.dumps(understanding, indent=2,
  sort_keys=True, default=str)}"`, then, iff there are turns to show, `"RECENT TURNS (context
  only):\n{...}"` with the newest-last, oldest-dropped-first truncation to `JUDGE_USER_MAX_CHARS =
  6000` (`app.py:616`). Each golden item's own `path` field (`"understanding"` or `"turns"`) already
  encodes which evidence block is non-empty — the golden set never exercises both blocks on the
  same item (confirmed: every `path: "understanding"` row has `turns: []`; every `path: "turns"`
  row has `understanding: {}` — checked directly, all 85 rows).
- **Turn normalization** (`guards.py:538-567`, `_recent_turns`) — filters rows lacking a non-empty
  string `text`, computes `speaker = row.get("displayName") or row.get("authorId") or "member"`,
  `role = row.get("role") or "user"`, truncates `text` to `TURN_TEXT_MAX = 400` chars, keeps the
  last `RECENT_TURNS_N = 6`. **Must run before `_render_judge_user`'s render, on every item whose
  `path == "turns"`** — this is §2.3 finding 2's fix.
- **Parse** (`app.py:679-682`, `726-729`) — `llm.extract_own_line_json_object(text,
  require_key="decision")`. A reply that is entirely one JSON object (bare or fenced) or exactly one
  `decision`-carrying object that owns its lines (its `{` is a line's first non-whitespace character,
  nothing but whitespace follows the matching `}` on that line) — everything else (prose, two
  candidate objects, a quoted verdict mid-sentence) is `None`. **On `None`, the real judge falls
  back to `{"decision": False, "rationale": "unparseable judge output"}`** (`app.py:727-728`) — the
  bias-to-suspend default, not a fabricated guess. `classification.py` (§5.1) replicates this exact
  fallback, which is what makes "an unparseable reply is a parse failure counted in the denominator,
  never a fabricated verdict" (plan §3.8.2) a transcription of the real system rather than an
  invented rule: the item still contributes to its metric's denominator with `decision = False`,
  because that is what the real judge would have done with the same reply.

### 2.5 `nlq-generator`'s real mechanism, transcribed — verified against `querygen.py` and `tools.py` directly

- **The model call is a second, internal, non-agent-loop structured completion**
  (`tools.py:977-991`, `QueryGraphDataTool.run`) — system prompt =
  `_build_query_request_system_prompt(schema)` (`tools.py:902-903`), filling
  `_QUERY_REQUEST_INSTRUCTIONS` (`tools.py:854-899`) — a fixed instructions block plus four worked
  examples plus `"This dataset's schema:\n{dataset_schema}"`, where `{dataset_schema}` is
  `_describe_dataset_schema(schema)` (`tools.py:825-836`): `"{label} (properties: {sorted, comma-
  joined property names})"` per label, joined with `"; "`. User message = the item's own `question`
  text, verbatim. **The instructions template and the four examples are transcribed verbatim** into
  `packs/nlq-structured-query/prompts/querygen.md` (a `{dataset_schema}`-templated text file, filled
  per item at prompt-build time from `schema.json`); the dataset routing the production tool call
  performs is **not** reproduced — each golden item already declares its own `dataset` (`catalog` or
  `knowledge_base`), so `build_messages` resolves the schema for that one dataset directly, never
  asking a model to choose one.
- **Parse** (`tools.py:989`) — `extract_own_line_json_object(reply, require_key="matches")`, the
  **same** conservative parser guard-judge uses (`llm.py:540-590`, already transcribed for
  `classification.py` — `extraction.py` reuses the identical logic, re-implemented once and shared,
  §5.2).
- **Validate + compile** (`tools.py:993-999`) — `querygen.QueryRequest.model_validate(...)` then
  `querygen.compile(request, schema)`; **any** `ValidationError`/`ValueError` from either step
  collapses to the same abstention shape as a parse failure in production
  (`{"items": [], "finding": "no matching data found"}`) — but §3.8.3's own cost note requires
  `model-bench`'s harness to keep the two apart (malformed spec vs. schema violation), which
  production does not need to (it only needs to abstain either way). `querygen.py`'s two validation
  layers, transcribed into `tools/exec.py` in stdlib (no pydantic, per the plan's explicit
  prohibition, §3.3):
  - **Layer A — structural (mirrors `QueryFilter`/`QueryMatch`/`QueryRequest`, `querygen.py:74-
    138`):** `extra="forbid"` (no unknown top-level or nested keys) on all three shapes; `filters`
    max length 4; the six-operator whitelist `{"=", "<>", "<", "<=", ">", ">="}`; `matches` exactly
    one entry; `var` matches `^[a-z][a-z0-9]{0,7}$`; `property` matches `^[a-z][a-zA-Z0-9]{0,31}$`;
    `returns` length 1–6, each entry matching the projection regex `^([a-z][a-z0-9]{0,7})\.([a-z]
    [a-zA-Z0-9]{0,31})$` or the aggregate regex `^(count|avg|min|max)\(([a-z][a-z0-9]{0,7})(?:\.
    ([a-z][a-zA-Z0-9]{0,31}))?\)$`; `order_by`, if present, matching the projection regex only;
    `order_dir` in `{"ASC", "DESC"}`; `limit` an integer in `[1, 50]`. A violation here is a
    **malformed spec**.
  - **Layer B — schema-bound (mirrors `compile()`, `querygen.py:275-449`):** `match.label` must be a
    key of the resolved dataset's `schema.json["labels"]`; every filter's/return's/order_by's
    `property` must be a key of that label's declared properties; every `returns` entry must be
    unique (FalkorDB itself rejects duplicate result-column names, `querygen.py:365-378`); a string
    filter value against a declared `int`/`float` property is coerced via `float(value)`/`int(value)`,
    raising on a genuine parse failure; a string filter value against a `*Normalized`-suffixed
    property with `op` in `{"=", "<>"}` is case-folded/whitespace-collapsed the same way
    `extraction.normalize_name` does (`extraction.py:67-78`, transcribed as a two-line helper — no
    import). A violation here is a **schema violation**.
  - **Execution** — no FalkorDB call. `tools/exec.py`'s `execute(compiled_request, tables)` filters
    `tables[label]`'s rows by the validated filters, projects/aggregates the validated `returns`
    (`count`/`avg`/`min`/`max` over the filtered rows, or a bare property projection), applies
    `order_by`/`order_dir`/`limit`, and returns `{"items": [rows]}` — the same shape
    `QueryGraphDataTool.run()` returns, which is what makes `nlq_scoring.score_pair`'s
    re-implementation (`extraction.py`, §5.2) applicable unchanged.
- **Answerability is not assumed — the declared schema exposes properties only, no relationship
  types**, so the golden set's 4 `relationship-traversal` items ("Who did Marlowe Robotics
  acquire?") have no valid spec against `tables.json`/`schema.json` at all — confirmed independently
  this session: `querygen.KNOWLEDGE_BASE_SCHEMA.labels` (`querygen.py:213-220`) declares `Entity`/
  `Document`/`Chunk` node properties only, and falkor-chat's own stored `nlq_eval_results.json`
  scores `nlq-34` (a relationship-traversal item) `{"items": [], "finding": "no matching data
  found"}` — an empirical confirmation, not a derivation from this component's own code. §6 below
  specifies the stamping step that makes this a stated fact per item rather than an assumption.

### 2.6 The `tables.json` snapshot — resolved without adding model-bench's first external dependency

§3.8.3 requires a **read-only snapshot of `ws:nlq-eval`'s `Entity`/`Document`/`Chunk` rows**, taken
once via a live Cypher read. This needs a *live* FalkorDB client — genuinely different from every
other `refresh_golden.py` origin, which is a text/AST read of a committed falkor-chat file. `model-
bench/AGENTS.md`'s "Hard rules" section states, without qualification, **"Zero runtime dependencies
... Dev extras are `pytest` and `ruff`, nothing else"** — the one named reversal trigger is `numpy`,
for corpus size/perf, not a database client. Reaching for `redis`/`falkordb` (the pinned client
`cypher-mcp/requirements.txt` already uses for this exact database, confirmed: `falkordb>=1.6,<1.7`)
would be `model-bench`'s first pip dependency anywhere in the component, even if scoped to a
maintenance script — a direct conflict with an explicit hard rule, not a grey area.

**Resolution: the snapshot is a human-in-the-loop step, not a mode `refresh_golden.py` performs
itself.** Whoever executes S4 Step 0 (an agent or a human with access to the `cypher` MCP tool this
very document was researched with, or any other read-only FalkorDB client already on hand) runs
three read-only `MATCH` queries against `ws:nlq-eval` — `MATCH (e:Entity) RETURN e.entityId AS
entityId, e.name AS name, e.nameNormalized AS nameNormalized, e.type AS type`, the equivalent for
`Document`/`Chunk` per `KNOWLEDGE_BASE_SCHEMA`'s declared property allowlist (`querygen.py:213-220`)
— and saves the three row lists as `tables.json`'s `"knowledge_base"` half by hand (or via a five-
line ad hoc script the operator writes and discards, not committed). `refresh_golden.py` gains a
**local-only** `--check-tables-shape` mode (§6 below) that validates an *already-produced*
`tables.json` (every `knowledge_base` row carries exactly the declared properties, nothing else;
every `catalog` row likewise) and writes its `PROVENANCE.md` entry from a caller-supplied
`--source-git-sha`/row counts — it never opens a socket. **This is reversible and low-risk**: if a
later stage needs repeated re-snapshotting, automating it with the pinned `falkordb` client is a
one-line addition at that point, with an explicit reversal-trigger decision the way `numpy`'s
already is — nothing here forecloses it. The catalog half of `tables.json` needs no live read at
all: `falkor-chat/scripts/seed_catalog.sh`'s `CATALOG` heredoc (line ~80) is the same
Python-list-inside-a-heredoc shape S3 already AST-parses for `seed_eval_corpus.py`'s `_CORPUS`
(`refresh_golden.py`'s `_read_corpus_literal`, S3 spec §3.4) — reused unchanged, extended only to
also compute each row's `nameNormalized`/`categoryNormalized` via a transcribed `normalize_name`
(`extraction.py:67-78`, two lines, no import).

### 2.7 What the plan/`-ml` note give verbatim, and where

| Topic | Citation |
|---|---|
| `guard-judge` data, mechanism, reported metrics, the two-verdict-no-headline rule, the cost note | Plan §3.8.2 `:2076-2115` |
| `nlq-generator` data, mechanism, Layer 1 scoring, the answerability stamp, the cost note | Plan §3.8.3 `:2117-2177` |
| S4's two done-conditions | Plan §4 S4 `:5699-5715` |
| Test items 9 and 16, S4's row in the stage-ownership table | Plan §5 `:5979`, `6042`, `6293` |
| Guard-judge's class-conditional MDD/floor table, the reversal trigger, the naming rule | `-ml` §7.3 `:3153-3210` |
| The per-role sample-size table (nlq-generator's row) | `-ml` §7.2 `:3118-3129` |
| `ItemScorer`'s shipped shape, `_load_item_scorer`, the `getattr`-guard convention | `runner.py:154-239` (read directly, §2.2 above) |
| The two review findings and both open design questions | `model-bench/docs/reviews/itemscorer-extension.md` (read in full, §2.2 above) |
| `querygen`'s DSL, schema registry, `compile()`'s two validation layers | `falkor-chat/server/falkorchat/querygen.py` (read directly, §2.5 above) |
| `nlq_scoring`'s Layer 1 exact-match rules, the `conflicting-facts` exception, Wilson interval | `falkor-chat/server/tests/eval/nlq_scoring.py` (read directly, §2.5/§5.2 below) |
| `_LlmGuardJudge`'s prompt construction, parse, fallback | `falkor-chat/server/falkorchat/app.py:602-746` (read directly, §2.4 above) |
| Turn normalization | `falkor-chat/server/falkorchat/guards.py:538-567` (read directly, §2.3/§2.4 above) |

## 3. File/module layout

```
model-bench/
  scripts/
    refresh_golden.py                    # extended — §6 below
  modelbench/
    packs.py                             # extended — §4.2's Pack.prompt_config() fix
    runner.py                            # extended — §4.1's build_messages hook
    results.py                           # extended — §4.3's two new ExtractionAggregates fields
    scoring/
      classification.py                  # NEW — guard-judge's ItemScorer (§5.1)
      extraction.py                      # NEW — nlq-generator's ItemScorer (§5.2)
  packs/
    guard-judge-understanding/
      pack.json                          # §5.1.1
      items.jsonl                        # 85 rows, from golden_guards.jsonl (§6)
      prompts/judge.md                   # _JUDGE_SYSTEM_PROMPT, verbatim (§2.4)
      PROVENANCE.md
    nlq-structured-query/
      pack.json                          # §5.2.1
      items.jsonl                        # 40 rows + stamped `answerable`, from nlq_golden_set.jsonl (§6)
      tables.json                        # catalog (AST-parsed) + knowledge_base (snapshot, §2.6)
      schema.json                        # from querygen.CATALOG_SCHEMA/KNOWLEDGE_BASE_SCHEMA (§5.2.3)
      reference_specs.json               # 40 hand-authored QueryRequest-shaped specs (§6.2)
      prompts/querygen.md                # _QUERY_REQUEST_INSTRUCTIONS, verbatim, {dataset_schema}-templated (§2.5)
      tools/
        exec.py                          # validation (Layer A/B) + execution (§5.2.2)
      PROVENANCE.md
  tests/
    test_scoring_classification.py       # test 9's guard-judge sibling (unnumbered — §8) + build_messages
    test_scoring_extraction.py           # test 9, the extraction shape/epsilon cases (§8)
    test_tools_exec.py                   # Layer A/B validation matrix + execution (§8)
    test_refresh_golden.py               # extended — the new pack-keyed origins, --check-tables-shape
    test_packs.py                        # extended — prompt_config's resolution (§4.2)
    test_runner.py                       # extended — build_messages wiring (§4.1)
    test_results.py                      # extended — the two new ExtractionAggregates fields round-trip
```

## 4. Design & rationale — the seam fix (S4 Step 0)

### 4.1 `ItemScorer.build_messages` — optional, `getattr`-guarded, chat-branch-only

Ruled in §2.2. The Protocol addition and call-site edit, both additive:

```python
# runner.py — ItemScorer gains a fourth optional method, mirroring prime/embed_text/
# deterministic_arm's own shape and guard discipline (S3 spec §3.2/§3.3, review finding 1).

class ItemScorer(Protocol):
    ...  # score_item, aggregate, prime, embed_text unchanged

    def build_messages(
        self, item_input: Mapping[str, Any], *, pack: Pack,
    ) -> list[ChatMessage]:
        """Called instead of the generic, scorer-blind `_item_chat_messages` on the chat branch,
        iff the scorer defines it (optional, `getattr`-guarded — S4 spec §4.1, closing
        itemscorer-extension.md finding 1). `guard-judge`'s real prompt is a labelled multi-block
        render, `nlq-generator`'s is a schema-filled structured-completion instruction set — neither
        is a JSON dump of the item row. Falls back to `_item_chat_messages` when absent, so every
        existing S1/S2 test (`FakeItemScorer`, which defines neither) is untouched."""
        ...
```

```python
# runner.py — _drive_single_call_items's chat branch, the one-line dispatch change.
if call_surface == "chat":
    build_messages = getattr(scorer, "build_messages", None)          # NEW — §4.1
    messages = (
        build_messages(item_input, pack=pack) if build_messages is not None
        else _item_chat_messages(pack, item_input)
    )
    call: ChatResult | EmbedResult = lmstudio.chat(
        messages, model=model_info.id, temperature=prompt_config.temperature,
        max_tokens=prompt_config.maxTokens, timeout_s=cfg.requestTimeoutSeconds,
    )
```

### 4.2 `Pack.prompt_config()` resolves `systemPrompt`/`toolSchemas` to content — closing finding 2

```python
# packs.py — two new private helpers, called from prompt_config() only.

def _resolve_prompt_text(self, rel_path: str | None) -> str | None:
    """Resolve a prompt.<key> manifest path to its file's TEXT content, read relative to
    pack.root — mirrors data_path's resolution pattern but returns content, not a Path, because
    PromptConfig's own contract (convo.py:148-153) is content (S4 spec §4.2, closing
    itemscorer-extension.md finding 2). `None` in, `None` out: an absent systemPrompt stays absent,
    never becomes an empty-file read. Wraps a missing/unreadable file into PackConfigError — a
    pack-config defect, not a bare traceback."""
    if rel_path is None:
        return None
    try:
        return (self.root / rel_path).read_text(encoding="utf-8")
    except OSError as exc:
        raise PackConfigError(
            f"{self.packId}: prompt.systemPrompt at {rel_path!r} could not be read: {exc}"
        ) from exc

def _resolve_tool_schemas(self, rel_path: str | None) -> tuple[Mapping[str, Any], ...]:
    """Resolve prompt.toolSchemas — a single path to a JSON file holding an ARRAY of JSON Schema
    objects — to the tuple PromptConfig.toolSchemas actually declares (convo.py:165). `None`/absent
    -> (). Raises PackConfigError on a non-array file or an unreadable/unparseable one."""
    if rel_path is None:
        return ()
    try:
        data = json.loads((self.root / rel_path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise PackConfigError(
            f"{self.packId}: prompt.toolSchemas at {rel_path!r} could not be read: {exc}"
        ) from exc
    if not isinstance(data, list):
        raise PackConfigError(
            f"{self.packId}: prompt.toolSchemas at {rel_path!r} is not a JSON array"
        )
    return tuple(data)
```

`prompt_config()`'s return statement changes from `prompt.get("systemPrompt")`/`prompt.get(
"toolSchemas") or ()` to `self._resolve_prompt_text(prompt.get("systemPrompt"))`/
`self._resolve_tool_schemas(prompt.get("toolSchemas"))` — the only two lines of that function's body
that change. **Both docstrings are reconciled in the same edit**: `Pack.prompt_config()`'s own
docstring drops the "paths, not resolved content" paragraph (`packs.py:366-370`), replaced with a
one-line note that `systemPrompt`/`toolSchemas` are now resolved via `_resolve_prompt_text`/
`_resolve_tool_schemas`, matching `convo.PromptConfig`'s docstring, which needs no change (it already
states the true contract). **`run_pack`'s two `warm_up` call sites and `convo.assemble` need zero
further edits** — both already consume `prompt_config().systemPrompt`/`.toolSchemas` as-is; once
`prompt_config()` returns resolved content, they receive it automatically. This also retroactively
fixes `tool-caller`'s own currently-inert instance of the same defect, at no extra cost — noted, not
claimed as an S4 done-condition (S4 owns no `tool-caller` behavior).

### 4.3 `ExtractionAggregates`'s small, additive extension

```python
# results.py — two new defaulted fields; confirmed backward compatible against every existing
# keyword-only construction site (grep -rn "parseFailures" tests/ modelbench/ → six hits, all kwargs).

@dataclass(frozen=True)
class ExtractionAggregates:
    kind: Literal["extraction"] = "extraction"
    exactMatch: BinaryMetric | None = None
    byShape: tuple[BinaryMetric, ...] = ()
    parseFailures: int = 0               # unchanged — "no JSON object found at all" (outcome="parse_failure")
    malformedSpecCount: int = 0          # NEW — Layer A structural-validation failures (§2.5)
    schemaViolationCount: int = 0        # NEW — Layer B schema-bound-validation failures (§2.5)

    def named_metrics(self) -> tuple[MetricValue, ...]:
        found = [self.exactMatch] if self.exactMatch is not None else []
        return tuple([*found, *self.byShape])   # unchanged — the two new fields are scalar counts,
                                                  # never metric-carrying, matching parseFailures's
                                                  # own existing shape (never in named_metrics either)
```

### 4.4 Alternatives considered and rejected

- **Widening `score_item`/`aggregate`'s own signature to carry a `messages` parameter instead of a
  fourth method.** Rejected for the same reason the S3 spec rejected a `context`-threading
  `ItemScorer` widening (§3.2 there): it would force every existing scorer and every future
  item-level scorer to accept and thread a parameter most of them never use, and it breaks every
  `FakeItemScorer`-based S1/S2 test's positional/keyword call shape. A fourth optional method costs
  nothing to the three roles that don't need it.
- **A `validate_pack`-enforced requirement that a chat-surface item-level pack's scorer defines
  `build_messages`.** Rejected for S4 specifically (not rejected in general — named as a follow-up,
  §9): it needs `validate_pack` to first resolve the declared `"scorer"` module, a gap S3's own §9
  already flagged and explicitly deferred; building that machinery now would silently widen S4's
  scope into a S3-flagged, pre-existing gap this stage does not need closed to ship its own two
  packs correctly.
- **Storing `reference_specs.json`'s content inside `items.jsonl` itself** (a `referenceSpec` field
  per row). Rejected: `build_messages` must never see the answer-adjacent fields (`expected`,
  `rationale`, and a reference spec would be the same category of leak) — keeping the reference
  specs in a wholly separate file that `extraction.py`'s runtime path never opens is a structural
  guarantee against a prompt-leak defect, not a naming preference. `items.jsonl` gains only the
  stamped `answerable: bool` (needed at run time, for the accuracy-denominator exclusion), never the
  spec that produced it.
- **Automating the `ws:nlq-eval` snapshot with the pinned `falkordb` client inside
  `refresh_golden.py` itself.** Considered and set aside, not rejected outright (§2.6) — it is the
  natural next step if repeated re-snapshotting is ever needed, but adding model-bench's first pip
  dependency is exactly the kind of decision this document should not make silently against an
  explicit "stdlib only ... nothing else" hard rule; the human-in-the-loop design costs one manual
  step per pack version bump (rare — `tables.json` is versioned inside the pack's content hash) and
  forecloses nothing.

## 5. The two packs

### 5.1 `guard-judge-understanding`

#### 5.1.1 `packs/guard-judge-understanding/pack.json`

```json
{
  "packId": "guard-judge-understanding",
  "packVersion": "1.0.0",
  "role": "guard-judge",
  "schemaVersion": 1,
  "description": "The production fuzzy-guard judge's understanding/turns evidence tiers, scored class-conditionally (falseAdvanceRate / falseSuspendRate), no headline.",
  "scorer": "classification",
  "environment": {"requires": ["lmstudio-chat"]},
  "prompt": {
    "systemPrompt": "prompts/judge.md",
    "toolSchemas": null,
    "temperature": 0.0,
    "maxTokens": 256
  },
  "data": {"items": "items.jsonl"},
  "sampling": {
    "seed": 20260911,
    "pairingKey": ["itemId"],
    "analysisUnit": "itemId"
  },
  "metrics": {
    "verdictMetrics": ["falseAdvanceRate", "falseSuspendRate"],
    "headlineMetric": null
  },
  "provenance": "PROVENANCE.md"
}
```

No `maxIterationsPerTurn`/`historyReplay` — `MULTI_CALL_TURN_BY_ROLE["guard-judge"] is False`, and
`prompt_config()` neither requires nor forbids `historyReplay` for a non-multi-call role (`packs.py:
353-364`, confirmed by reading directly), so it is simply omitted, matching the item-level fixture
convention already established (`tests/fixtures/packs/scorer_unresolvable/pack.json`).

#### 5.1.2 `items.jsonl` row shape (post-copy, unchanged from the golden set except `id`→`itemId`)

`{"itemId": "ca-01", "tier": "clear_advance"|"clear_suspend"|"boundary", "path":
"understanding"|"turns", "r1_probe": bool, "condition": str, "understanding": {...}, "turns": [...],
"expected": bool, "label_rationale": str}` — carried through verbatim (D1: data, not derived);
`expected`/`label_rationale`/`r1_probe` are **never read by `build_messages`** (they would leak the
gold label into the prompt) — read only by `score_item` for scoring.

#### 5.1.3 `modelbench/scoring/classification.py`

```python
"""guard-judge's ItemScorer — a module, not a class, satisfying the Protocol structurally
(retrieval.py's own precedent). Transcribes falkorchat.app._LlmGuardJudge's prompt construction and
falkorchat.guards._recent_turns's turn normalization as DATA + CODE, never an import (D1). No live
call, no database — one chat call per item.
"""

# --- turn normalization (transcribed from guards.py:538-567, verbatim rule set) -----------------

_RECENT_TURNS_N = 6
_TURN_TEXT_MAX = 400

def _normalize_turns(raw_turns: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    """`guards._recent_turns`'s exact rule: drop rows with no non-empty string `text`; speaker =
    displayName or authorId or "member"; role = role or "user"; truncate text to 400 chars; keep
    the last 6, oldest first (input is already chronological)."""
    turns: list[dict[str, str]] = []
    for row in raw_turns:
        text = row.get("text")
        if not isinstance(text, str) or not text:
            continue
        speaker = row.get("displayName") or row.get("authorId") or "member"
        role = row.get("role") or "user"
        turns.append({"speaker": str(speaker), "role": str(role), "text": text[:_TURN_TEXT_MAX]})
    return turns[-_RECENT_TURNS_N:]


# --- prompt rendering (transcribed from app.py:602-654) ------------------------------------------

_JUDGE_USER_MAX_CHARS = 6000

def _render_judge_user(condition: str, understanding: Mapping[str, Any], turns: Sequence[Mapping[str, str]]) -> str:
    """Verbatim port of `_render_judge_user` (app.py:619-654): CONDITION always present, CURRENT
    STATE iff `understanding` is non-empty (json.dumps(..., indent=2, sort_keys=True, default=str)),
    RECENT TURNS iff `turns` is non-empty, oldest-dropped-first truncation to 6000 chars."""
    ...  # port unchanged — same eviction-by-suffix arithmetic as app.py, same signature shape


def build_messages(item_input: Mapping[str, Any], *, pack: Pack) -> list[ChatMessage]:
    """Never reads `item_input["expected"]`/`["label_rationale"]`/`["r1_probe"]` — those would leak
    the gold label. `pack.prompt_config().systemPrompt` is `prompts/judge.md`'s resolved content
    (§4.2)."""
    system_prompt = pack.prompt_config().systemPrompt
    turns = _normalize_turns(item_input.get("turns") or [])
    user = _render_judge_user(item_input["condition"], item_input.get("understanding") or {}, turns)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user},
    ]


# --- parse (transcribed from llm.py:540-590 — same conservative rule guard-judge's production
# parser uses; re-implemented once here, not imported) -------------------------------------------

def extract_own_line_json_object(content: Any, *, require_key: str | None = None) -> dict[str, Any] | None:
    """Verbatim port of `llm.extract_own_line_json_object` — fence-strip, whole-reply-is-one-object
    fast path, else exactly-one line-owning object carrying `require_key`, else None. `-ml`/plan
    reference this as the pack's declared parse mode `"ownLineJsonObject"` (plan §3.8.2)."""
    ...


# --- per-item metric assignment -------------------------------------------------------------------

#: tier -> the metric name that tier's items contribute to. `boundary`'s metric is exploratory
#: (never in verdictMetrics — plan §3.8.2/-ml §7.3: "not a verdict metric" at n=15).
_METRIC_BY_TIER: Mapping[str, str] = {
    "clear_suspend": "falseAdvanceRate",   # expected=False; a False->True flip is a false advance
    "clear_advance": "falseSuspendRate",   # expected=True; a True->False flip is a false suspend
    "boundary": "falseAdvanceRateBoundary",  # all expected=False (verified §2.4); descriptive only
}


def score_item(
    item_input: Mapping[str, Any], result: ChatResult | None, timing: ItemTiming, *, pack: Pack,
) -> ItemResult:
    """One judge call. `result is None` -> outcome "fail" (timeout) / "unrunnable" (no_response),
    per timing.withheldFor, mirroring retrieval.py's own precedent (§2.1) — `scoreable={}`, no
    contribution to any metric's denominator (a call that never happened cannot be "the judge
    advanced/suspended").

    Otherwise: parses `result.message.get("content")` via `extract_own_line_json_object(...,
    require_key="decision")`. `None` -> `advanced = False`, `outcome = "parse_failure"` — the SAME
    fallback the real judge applies (app.py:727-728), never a fabricated guess. Parsed -> `advanced
    = bool(parsed.get("decision"))`, `outcome = "pass"`.

    Exactly one metric name is scoreable per item (`_METRIC_BY_TIER[item_input["tier"]]`), with
    `counts[metric] = int(advanced)` — for `clear_suspend`/`boundary` (expected=False, verified),
    `advanced=True` IS the false-advance event; for `clear_advance` (expected=True), `advanced=
    False` IS the false-suspend event, so `counts["falseSuspendRate"] = int(not advanced)`.
    `detail` carries `{"path": item_input["path"], "tier": item_input["tier"]}` for the aggregate's
    path-split diagnostic — never anything answer-adjacent."""
    ...


def aggregate(items: Sequence[ItemResult], *, pack: Pack) -> ClassificationAggregates:
    """One pass. `perClass` carries, in order: `falseAdvanceRate` (n=40, the clear_suspend items),
    `falseSuspendRate` (n=30, the clear_advance items) — the two verdictMetrics, both BinaryMetric
    unit="item"; `advanceRecall` (the labelled complement, `1 - falseSuspendRate`'s successes/n
    inverted — successes = falseSuspendRate.n - falseSuspendRate.successes, same n=30) — printed,
    never in verdictMetrics, so report.py renders it exploratory automatically (§2.1); `false
    AdvanceRateBoundary` (n=15, the boundary tier) — exploratory, descriptive only (§2.1's same
    mechanism); `falseAdvanceRateByUnderstanding`/`falseAdvanceRateByTurns`,
    `falseSuspendRateByUnderstanding`/`falseSuspendRateByTurns` — the path-split diagnostic (plan
    §3.8.2: "split by evidence path... as a diagnostic"), four more exploratory BinaryMetrics, each
    filtered by `detail["path"]` within its own tier's item set. `parseFailures` = count of items
    with `outcome == "parse_failure"`. `n` = `len(items)` (85). **No metric here is a pooled
    figure across tiers or paths — every BinaryMetric's `n` is its own named subset's count**,
    never 85 (plan §3.8.2: "no pooled 85-item figure anywhere")."""
    ...
```

### 5.2 `nlq-structured-query`

#### 5.2.1 `packs/nlq-structured-query/pack.json`

```json
{
  "packId": "nlq-structured-query",
  "packVersion": "1.0.0",
  "role": "nlq-generator",
  "schemaVersion": 1,
  "description": "Structured NL-to-query generation over the catalog/knowledge_base datasets, scored by Layer-1 exact match after canonicalization.",
  "scorer": "extraction",
  "environment": {"requires": ["lmstudio-chat"]},
  "prompt": {
    "systemPrompt": null,
    "toolSchemas": null,
    "temperature": 0.0,
    "maxTokens": 512
  },
  "data": {"items": "items.jsonl", "tables": "tables.json", "schema": "schema.json"},
  "tools": {"module": "tools/exec.py", "entrypoint": "compile_and_execute"},
  "sampling": {
    "seed": 20260911,
    "pairingKey": ["itemId"],
    "analysisUnit": "itemId"
  },
  "metrics": {
    "verdictMetrics": ["layer1ExactMatchRate"],
    "headlineMetric": "layer1ExactMatchRate"
  },
  "provenance": "PROVENANCE.md"
}
```

`prompt.systemPrompt` is `null` at the manifest level — the pack's real system prompt is built
**per item** (it depends on the item's own `dataset`), so it cannot be one fixed resolved path the
way `guard-judge`'s is. `build_messages` (§5.2.4) reads `prompts/querygen.md`'s template text via
`pack.data_path("...")`-style resolution directly (not through `prompt_config().systemPrompt`, which
stays `None` for this pack), fills `{dataset_schema}` per item, and returns it as the system message.
This is a deliberate, stated asymmetry with `guard-judge`, not an oversight — flagged in §9.

#### 5.2.2 `packs/nlq-structured-query/tools/exec.py`

```python
"""The pack-local, stdlib-only validation + execution surface (plan §3.8.3's cost note: this is the
larger half of the module, and it is what makes malformed-spec/schema-violation/wrong-answer
distinguishable rather than pooled into "wrong"). Loaded via Pack.load_tool_module() — a module, not
a class, entrypoint `compile_and_execute` (pack.json's own `tools.entrypoint`).

Re-implements querygen.py's two validation layers (§2.5) by hand — no pydantic import anywhere in
this file (plan §3.3's explicit prohibition on a pack module importing outside stdlib +
modelbench.tooling).
"""

# --- Layer A: structural validation (mirrors QueryFilter/QueryMatch/QueryRequest) ----------------

_VAR_RE = re.compile(r"^[a-z][a-z0-9]{0,7}$")
_PROP_RE = re.compile(r"^[a-z][a-zA-Z0-9]{0,31}$")
_PROJECTION_RE = re.compile(r"^([a-z][a-z0-9]{0,7})\.([a-z][a-zA-Z0-9]{0,31})$")
_AGGREGATE_RE = re.compile(r"^(count|avg|min|max)\(([a-z][a-z0-9]{0,7})(?:\.([a-z][a-zA-Z0-9]{0,31}))?\)$")
_OPS = frozenset({"=", "<>", "<", "<=", ">", ">="})


class MalformedSpecError(ValueError):
    """Layer A: the reply's JSON does not have QueryRequest's declared shape at all — wrong types,
    an unknown key (`extra="forbid"`'s hand-rolled equivalent), a regex-failing var/property/
    returns/order_by entry, `filters` over 4 long, `returns` outside 1-6, `limit` outside [1, 50],
    `op` outside the six-member whitelist, `matches` not exactly one entry."""


class SchemaViolationError(ValueError):
    """Layer B: the spec has QueryRequest's shape but fails against THIS dataset's schema —
    unregistered label/property, a duplicate `returns` entry, a filter value that does not coerce
    to its property's declared type."""


def validate_structure(spec: Mapping[str, Any]) -> None:
    """Layer A — raises MalformedSpecError on the first violation. Every check below has a direct
    counterpart in querygen.py (cited inline); none softens or widens the original rule."""
    ...  # extra-keys check at each of the three levels; matches length == 1; var/property regex;
         # six-op whitelist; filters len <= 4; returns len in [1,6] each matching a projection or
         # aggregate shape; order_by (if present) matching only the projection shape; order_dir in
         # {"ASC","DESC"}; limit int in [1,50]


def validate_against_schema(spec: Mapping[str, Any], schema: Mapping[str, Any]) -> None:
    """Layer B — raises SchemaViolationError. label registered; every filter/return/order_by
    property registered for that label; no duplicate returns entries; each string filter value
    against a declared int/float property parses (raises SchemaViolationError, not
    MalformedSpecError, on failure — the TYPE was legal, the VALUE didn't parse for it)."""
    ...


def _normalize_name(value: str) -> str:
    """Two-line transcription of extraction.normalize_name (extraction.py:67-78) — whitespace-
    collapse + casefold. No import: the plan forbids this module reaching outside stdlib +
    modelbench.tooling, and falkorchat.extraction is neither."""
    return re.sub(r"\s+", " ", value.strip()).casefold()


def compile_and_execute(
    spec: Mapping[str, Any], *, tables: Mapping[str, list[dict[str, Any]]], schema: Mapping[str, Any],
) -> dict[str, Any]:
    """Runs Layer A then Layer B (in order — a structurally-invalid spec never reaches the
    schema check, matching querygen.py's own validate-then-compile ordering) then executes:
    filters `tables[label]` by the validated filters (coercing a string value against a
    *Normalized property via `_normalize_name`, matching querygen.compile's own rule, §2.5),
    projects/aggregates `returns`, applies `order_by`/`order_dir`/`limit`. Returns `{"items":
    [...]}, never raises past this point — MalformedSpecError/SchemaViolationError propagate to the
    caller (extraction.py's score_item), which is what lets the three failure classes stay
    distinguishable at the call site rather than being caught and pooled here."""
    ...
```

#### 5.2.3 `schema.json` (from `querygen.CATALOG_SCHEMA`/`KNOWLEDGE_BASE_SCHEMA`, transcribed by `refresh_golden.py`)

```json
{
  "catalog": {"labels": {"Product": {"name": "str", "nameNormalized": "str", "category": "str", "price": "float"}}},
  "knowledge_base": {"labels": {
    "Entity": {"entityId": "str", "name": "str", "nameNormalized": "str", "type": "str"},
    "Document": {"documentId": "str", "title": "str", "sourceFormat": "str"},
    "Chunk": {"chunkId": "str", "text": "str", "seq": "int", "documentId": "str"}
  }}}
```

Property type names are the four JSON-representable tokens `exec.py`'s Layer B coercion switches on
(`"str"`/`"int"`/`"float"` — no `"bool"` property exists in either schema, matching `querygen.py`'s
own known, documented gap, §2.5, not reopened here).

#### 5.2.4 `modelbench/scoring/extraction.py`

```python
"""nlq-generator's ItemScorer. Re-implements nlq_scoring.py's Layer 1 comparison rules (§2.5) and
builds the per-item structured-completion prompt (§2.5's transcription of tools.py). One chat call
per item; execution is entirely in-process against tables.json via tools/exec.py — no FalkorDB."""

_NUMERIC_EPSILON = 0.01


def _canon_str(value: Any) -> str:
    """Verbatim port of nlq_scoring._canon_str (nlq_scoring.py:87-90)."""
    return re.sub(r"\s+", " ", str(value).strip().casefold())


def _scalar_equal(expected: Any, actual: Any) -> bool:
    """Verbatim port of nlq_scoring._scalar_equal — numeric epsilon (0.01 + 1e-9 slop) for two
    numbers, canonical string equality otherwise, never coerced across the two."""
    ...


def score_pair(expected: Mapping[str, Any], shape: str, tool_result: Mapping[str, Any]) -> tuple[bool, str]:
    """Verbatim port of nlq_scoring.score_pair's three etype branches (scalar/set/not_found), INCLUDING
    the conflicting-facts subset-containment exception (`shape == "conflicting-facts"` ->
    `expected_set.issubset(actual_set)`, nlq_scoring.py:186-195 — NOT set equality; this is the one
    exception §3.8.3 calls out by name and it must be built, not discovered). Returns (correct,
    reason)."""
    ...


def _describe_dataset_schema(schema: Mapping[str, Any]) -> str:
    """Verbatim port of tools._describe_dataset_schema (tools.py:825-836): "{label} (properties:
    {sorted, comma-joined property names})" per label, joined with "; "."""
    ...


def build_messages(item_input: Mapping[str, Any], *, pack: Pack) -> list[ChatMessage]:
    """Never reads item_input["expected"]/["rationale"]/["answerable"] — those would leak the gold
    answer. Reads pack.data_path("schema") for THIS item's dataset's schema block, fills
    prompts/querygen.md's {dataset_schema} placeholder, returns
    [{"role": "system", "content": filled}, {"role": "user", "content": item_input["question"]}] —
    the same two-message shape QueryGraphDataTool.run() sends (tools.py:985-988)."""
    ...


def score_item(
    item_input: Mapping[str, Any], result: ChatResult | None, timing: ItemTiming, *, pack: Pack,
) -> ItemResult:
    """One structured-completion call.

    `result is None` -> outcome "fail"/"unrunnable" per timing.withheldFor, scoreable={} (mirrors
    classification.py/retrieval.py's own precedent, §2.1).

    Otherwise, three gates in order, each producing a DISTINCT outcome/detail so the three failure
    classes stay separable at aggregate() (§2.3/§4.3):
    1. `extract_own_line_json_object(result.message.get("content"), require_key="matches")` is
       `None` -> outcome="parse_failure", detail={"failureClass": "no_json"}.
    2. `tools_exec.validate_structure(parsed)` raises MalformedSpecError -> outcome="fail",
       detail={"failureClass": "malformed_spec"}.
    3. `tools_exec.validate_against_schema(parsed, schema_for(item_input["dataset"]))` raises
       SchemaViolationError -> outcome="fail", detail={"failureClass": "schema_violation"}.
    Otherwise `tools_exec.compile_and_execute(parsed, tables=tables_for(dataset), schema=...)`
    executes (no further raise expected once both layers pass — a defensive catch-and-classify-as-
    schema_violation guards the boundary anyway, never silently swallowed).

    **The unanswerable bucket (`item_input["answerable"] is False`) never contributes to
    `layer1ExactMatchRate`'s `scoreable`/`counts`** — instead, a SEPARATE, always-scoreable
    diagnostic metric `unanswerableAbstainRate` is set: `counts["unanswerableAbstainRate"] = 1` iff
    the executed result is empty (`tool_result["items"] == []`, correctly abstained), else `0`
    (fabricated an answer to an unanswerable question) — plan §3.8.3: "they measure whether a model
    correctly abstains instead of fabricating — but as their own count, not as accuracy." An
    answerable item that reaches execution is scored via `score_pair(item_input["expected"],
    item_input["shape"], tool_result)` into `layer1ExactMatchRate` (scoreable=True always, even on
    a wrong answer) AND into a per-shape exploratory metric `f"exactMatchBy{shape.title()...}"`
    (one of the seven declared shapes)."""
    ...


def aggregate(items: Sequence[ItemResult], *, pack: Pack) -> ExtractionAggregates:
    """`exactMatch` = BinaryMetric("layer1ExactMatchRate", ..., unit="item") over ONLY the
    answerable items reaching execution (n = 36, the 40 minus the 4 stamped unanswerable — plan
    §3.8.3: "the report puts unanswerable items in a separate, named bucket excluded from the
    accuracy denominator"). `byShape` carries: one BinaryMetric per shape among the answerable items
    (single-fact, filter-list, compound-filter, not-found, aggregation, conflicting-facts —
    relationship-traversal is entirely unanswerable, so it has no answerable-bucket entry, only the
    unanswerable one below) plus `unanswerableAbstainRate` (n = 4, the relationship-traversal
    items). `parseFailures` / `malformedSpecCount` / `schemaViolationCount` are each a `sum(1 for
    it in items if it.detail.get("failureClass") == ...)` over the three named classes (§2.3/§4.3) —
    the three counts §3.8.3's cost note requires, never pooled into `parseFailures` alone."""
    ...
```

## 6. `scripts/refresh_golden.py` extension

**Generalize `_TRACKED_ORIGINS` from one module-level constant to a per-pack mapping**, keyed by
`packId` (read from the target `--pack`'s own `pack.json`) — the shipped single constant is
embedder-only despite `--pack` already being a required, generic-looking flag; running today's
script against a non-embedder `--pack` path silently writes the wrong origins to the wrong
destination, a latent bug this stage's second real pack would otherwise trigger for the first time.

```python
# refresh_golden.py

_TRACKED_ORIGINS_BY_PACK_ID: dict[str, tuple[OriginSpec, ...]] = {
    "embedder-graphrag-retrieval": (
        # unchanged — the four OriginSpecs + the check-only test_metrics.py entry, moved verbatim
    ),
    "guard-judge-understanding": (
        OriginSpec(
            "falkor-chat/server/tests/eval/golden_guards.jsonl", "items.jsonl", "jsonl-transform",
        ),  # id -> itemId rename only; every other field carried through unchanged (§5.1.2)
    ),
    "nlq-structured-query": (
        OriginSpec(
            "falkor-chat/server/tests/eval/nlq_golden_set.jsonl", "items.jsonl", "jsonl-transform",
        ),  # id -> itemId rename; `answerable` is NOT written here — --stamp-answerability does that
        OriginSpec("falkor-chat/scripts/seed_catalog.sh", "tables.json#catalog", "ast-literal"),
        # AST-parses the CATALOG heredoc (same technique as seed_eval_corpus.py's _CORPUS, S3
        # spec §3.4), computes nameNormalized/categoryNormalized via the transcribed normalize_name
        # (§2.6), and writes/merges into tables.json's "catalog" key — never touches "knowledge_base"
        OriginSpec(
            "falkor-chat/server/falkorchat/querygen.py", "schema.json", "schema-literal",
        ),  # a NEW OriginSpec.kind — AST-extracts CATALOG_SCHEMA/KNOWLEDGE_BASE_SCHEMA's `labels`
            # dict literals the same execute-nothing way _read_corpus_literal does, mapping each
            # Python type name (str/int/float) to its JSON token
    ),
}
```

**A new, local-only mode, `--check-tables-shape`** (§2.6's resolution — never opens a network
socket): reads an already-present `tables.json`, asserts every `"knowledge_base"` row under each of
`Entity`/`Document`/`Chunk` carries exactly that label's `schema.json`-declared properties (no more,
no fewer — the "curated allowlist" discipline `querygen.py`'s own `DatasetSchema` docstring states),
and writes `PROVENANCE.md`'s entry for the `ws:nlq-eval` snapshot from `--source-git-sha <sha>`
(the falkor-chat commit the operator read the live graph at) and the row counts it just counted.
Refuses under an unchanged `packVersion`, via the same `_pack_version_gate` the other write paths
share (content-hash-changing, so it needs the same guard `--embed-corpus` already has).

**A new mode, `--stamp-answerability`** (nlq-structured-query only, gated: refuses on any other
`--pack`, mirroring `--embed-corpus`'s implicit embedder-only assumption, §S3): reads
`reference_specs.json` (§4.4's rejected-alternative note — a separate, hand-authored file, never
folded into `items.jsonl`), runs each item's own reference spec through
`tools.exec.compile_and_execute` against `tables.json`/`schema.json`, and stamps `items.jsonl`'s
matching row `"answerable": true` iff the reference spec compiles, executes, and its result is
non-empty when `expected.type != "not_found"` (a `not_found`-shaped item is answerable by
construction — the correct answer IS an empty result); `false` otherwise (the reference spec itself
fails Layer A/B, or executes to empty against a non-`not_found` expectation — the relationship-
traversal items' actual failure mode, confirmed §2.5). Refuses under an unchanged `packVersion`, same
gate. `validate_pack` (existing, `packs.py`, needs one new check) fails a `nlq-structured-query`
pack whose `items.jsonl` has any row missing the `answerable` key — plan §3.8.3's own stated rule
("`validate` fails a pack that has unstamped items"), a small, additive, role-scoped
`_answerability_stamp_problems` check mirroring `_prompt_problems`'s existing role-scoping shape.

## 7. Step sequence for implementation

Four steps — larger than S3's three because this stage carries a real seam fix ahead of two
concrete packs, and the two packs are independent enough to build/test in either order once the
seam is fixed (plan: "Do `guard-judge` first (no executor at all)" — a sequencing preference inside
Step 2, not a hard dependency between the two packs).

### Step 0 — The seam fix (§4), offline, before either pack exists

`ItemScorer.build_messages` (§4.1), `Pack.prompt_config()`'s resolution fix (§4.2),
`ExtractionAggregates`'s two new fields (§4.3). Every piece is testable against the SHIPPED S1/S2/S3
fixtures with no new pack data: a `FakeItemScorer` subclass (`FakeItemScorerWithBuildMessages`)
added to `tests/test_runner.py` asserting the chat branch calls `build_messages` in preference to
`_item_chat_messages` when present, and falls back correctly when absent (the full existing
`FakeItemScorer`/`FakeItemScorerWithHooks` suite stays green, unmodified — the regression check for
finding 1's fix); `tests/test_packs.py` gains two new assertions on the existing `"valid"` fixture
(`cfg.systemPrompt == (fixture root / "prompts/system.md").read_text()`, `cfg.toolSchemas == ()`)
plus a new fixture pack whose `prompt.toolSchemas` points at a non-empty schema array, asserting the
parsed tuple — the regression check for finding 2's fix; `tests/test_results.py` gains a round-trip
test for `ExtractionAggregates(malformedSpecCount=..., schemaViolationCount=...)` through
`to_dict`/`from_dict`, confirming the generic `_aggregates_from_dict` dispatch (already shipped,
`results.py:805+`) needs no change to carry the two new fields. **Done when:** the full existing
suite (`pytest -q`) is green with zero new fixtures beyond the ones just listed, and the two new
wiring tests are red-then-green against the actual edits (not written after the fact).

### Step 1 — `guard-judge-understanding`, offline half

Creates `packs/guard-judge-understanding/{pack.json, prompts/judge.md, PROVENANCE.md}`,
`modelbench/scoring/classification.py` in full, `refresh_golden.py`'s new
`_TRACKED_ORIGINS_BY_PACK_ID["guard-judge-understanding"]` entry. `items.jsonl` is produced by
running `refresh_golden.py --pack packs/guard-judge-understanding` once against the real
`falkor-chat` tree (still present on this box, confirmed §2.6's own live-verification precedent).
Order, red→green:

1. `_normalize_turns`/`_render_judge_user` unit tests against hand-built rows, including the
   eviction-by-suffix truncation case at a synthetic char budget (mirrors `_render_judge_user`'s own
   `while kept > 0 and total > JUDGE_USER_MAX_CHARS` loop) and the raw-message-shape input (`msgId`,
   `displayName`, no `speaker` key) — the case §2.3 finding 2 exists to catch.
2. `extract_own_line_json_object` unit tests — reused test vectors from `llm.py`'s own docstring
   examples (bare object, fenced object, quoted-mid-sentence rejection, two-candidate-objects
   rejection), confirming the port matches the source's documented behaviour.
3. `build_messages` unit test against one real `guard-judge-understanding` item (post-copy),
   asserting the system message equals `prompts/judge.md`'s file content exactly and the user
   message matches a hand-computed `_render_judge_user` expectation for that item; a second
   assertion that `build_messages` never reads `expected`/`label_rationale`/`r1_probe` (constructed
   by deleting those three keys from the item dict before calling it — must not raise `KeyError`).
4. `score_item` unit tests: the parse-failure fallback (`outcome="parse_failure"`, `advanced=False`,
   contributes to the tier's metric denominator with a "did not advance/suspend" count — the
   transcription-fidelity assertion, §2.4); the `result is None` timeout/no-response branches; one
   clean case per tier (`clear_suspend` advancing → `falseAdvanceRate` count 1; `clear_advance`
   suspending → `falseSuspendRate` count 1; `boundary` advancing → `falseAdvanceRateBoundary` count
   1).
5. `aggregate` unit tests over a small synthetic 85-item-shaped fixture (10/8/4 per tier, scaled
   down): `perClass` carries `falseAdvanceRate`/`falseSuspendRate` at their own tier's `n`, never
   85; `advanceRecall`'s `successes`/`n` is `falseSuspendRate`'s exact complement; the four path-
   split metrics sum back to their parent tier's own successes/n (a positive cross-check, not just a
   shape check); `parseFailures` counts only `outcome == "parse_failure"` items.
6. `validate --pack packs/guard-judge-understanding --strict` passes clean.

**Done when:** every item above is green offline; `-m live` is not required for this step (guard-
judge needs no `--embed-corpus`/`--stamp-answerability`-shaped live step — a plain `refresh_golden.py
--pack packs/guard-judge-understanding` data-import is the only live-adjacent action, and it touches
only falkor-chat's committed tree, not LM Studio).

### Step 2 — `guard-judge-understanding`, live half

`-m live`, needs a reachable LM Studio. `run --pack guard-judge-understanding --model <key>` end to
end — S4's own first "Done when" clause (plan: "both packs run end to end against one model").
`compare`/report inspection confirms: the two verdict metrics render side by side; `advanceRecall`
prints, labelled, carrying no verdict; the boundary tier prints, labelled `no significance claim`;
the path-split diagnostics print as exploratory; **no pooled 85-item figure appears anywhere** (a
`grep -c '85'` sanity check against the rendered markdown, beyond the item's own manifest-declared
n=85 count line if `report.py` prints one generically elsewhere — confirm by reading the actual
output, not by a blind grep).

### Step 3 — `nlq-structured-query`, offline half, in three passes

Larger than guard-judge's, matching the plan's own "Cost: medium-high... sized as its own piece of
work" note. Creates `packs/nlq-structured-query/{pack.json, tools/exec.py, schema.json,
prompts/querygen.md, PROVENANCE.md}`, `modelbench/scoring/extraction.py`, `refresh_golden.py`'s three
new origin entries plus `--check-tables-shape`/`--stamp-answerability`.

**Pass A — `tools/exec.py`, pure, no pack loader:**
1. Layer A (`validate_structure`) unit tests — one case per rule named in §2.5 (`extra` key at each
   of the three levels, `matches` length ≠ 1, bad `var`/`property` regex, an `op` outside the
   six-member whitelist, `filters` at length 5, `returns` at length 0 and at length 7, a `returns`
   entry matching neither the projection nor aggregate regex, a non-projection `order_by`, an
   `order_dir` outside `{"ASC","DESC"}`, `limit` at 0 and at 51) — each asserted to raise
   `MalformedSpecError`, and one full valid spec asserted to raise nothing.
2. Layer B (`validate_against_schema`) unit tests — an unregistered label, an unregistered property
   on a filter/return/order_by, a duplicate `returns` entry, a numeric-typed property given a
   non-parsing string value (`"fifty"` against `price: float`) — each `SchemaViolationError`; a
   numeric-typed property given a parsing string value (`"50"` against `price: float`) accepted and
   coerced; a `*Normalized` property given un-normalized text, asserted matched against a
   pre-normalized stored row.
3. `compile_and_execute` unit tests against a small synthetic 4-row `tables.json` fixture: a filter
   query, a `count`/`avg`/`min`/`max` aggregate each, an `order_by` + `limit` superlative query, a
   query whose `returns` includes a `*Normalized` property (never itself normalized on output,
   only on the filter side — matching `querygen.compile`'s own asymmetry).

**Pass B — `extraction.py`'s pure scoring half, no live call:**
4. `_canon_str`/`_scalar_equal`/`score_pair` unit tests — transcribed from
   `nlq_scoring.py`'s own `test_nlq_scoring.py` cases where reachable (read that file if present;
   otherwise hand-built from `nlq_scoring.py`'s docstring rules), **explicitly including the
   `conflicting-facts` subset-containment case**: `expected_set = {"a","b"}`,
   `actual_set = {"a","b","c"}`, `shape="conflicting-facts"` → correct; the same actual set against
   any other set-shaped `shape` → incorrect (set equality) — the one exception §3.8.3 calls out by
   name, and the test that would fail silently if a scorer applied equality uniformly (plan's own
   stated risk).
5. `_describe_dataset_schema`/`build_messages` unit tests against `schema.json`'s real catalog/
   knowledge_base blocks — asserting the filled system message matches a hand-computed expectation
   for one item of each dataset, and that `expected`/`rationale`/`answerable` are never read
   (same never-`KeyError`-on-deletion technique as guard-judge's step 1.3).

**Pass C — `score_item`/`aggregate`, wiring the two above together:**
6. The three-gate `score_item` unit tests — no-JSON reply (`outcome="parse_failure"`), a
   Layer-A-failing reply (`outcome="fail"`, `detail.failureClass=="malformed_spec"`), a
   Layer-B-failing reply (`outcome="fail"`, `detail.failureClass=="schema_violation"`), a clean
   reply scoring correct and one scoring incorrect (`outcome="pass"` both), and the unanswerable-
   item branch (`item_input["answerable"] is False`) scoring `unanswerableAbstainRate` correctly on
   both an abstained (`items: []`) and a fabricated (`items: [...]`) synthetic reply.
7. `aggregate` unit tests over a small synthetic fixture spanning all seven shapes plus the
   unanswerable bucket: `exactMatch.n` excludes the unanswerable items; `byShape` sums back
   consistently; `parseFailures`/`malformedSpecCount`/`schemaViolationCount` each count only their
   own `detail.failureClass`, never each other's.
8. `validate --pack packs/nlq-structured-query --strict` **fails** on a deliberately-unstamped
   fixture (one `items.jsonl` row missing `answerable`) and **passes** once every row is stamped —
   the plan's own explicit done-condition (§6's `_answerability_stamp_problems`).

**Done when:** every item above is green offline. `--check-tables-shape`/`--stamp-answerability` are
exercised against the real, human-snapshotted `tables.json` in Step 4 below, not simulated here —
`reference_specs.json`'s 40 entries are hand-authored during this step (real, non-mechanical work:
each spec must be a valid `QueryRequest`-shape a human writes by reading the catalog/knowledge-base
data directly, not derived from `expected`, which states the answer shape, not the query that
produces it).

### Step 4 — `nlq-structured-query`, live half

Two live-adjacent actions, neither needing LM Studio:
1. The `ws:nlq-eval` snapshot (§2.6) — a human/agent with `mcp__cypher__query`-equivalent access
   runs the three `MATCH` reads against `ws:nlq-eval`, writes `tables.json`'s `"knowledge_base"`
   half, then `refresh_golden.py --pack packs/nlq-structured-query --check-tables-shape
   --source-git-sha <sha>` validates the shape and writes `PROVENANCE.md`.
2. `refresh_golden.py --pack packs/nlq-structured-query --stamp-answerability` — runs
   `reference_specs.json`'s 40 specs through `tools.exec.compile_and_execute`, stamps `answerable`
   per item, confirms the 4 `relationship-traversal` items stamp `false` and every other item stamps
   `true` (a positive expectation from §2.5's independent confirmation against
   `nlq_eval_results.json` — if any *other* item also stamps `false`, that is new information about
   this pack's own tables/schema, not a bug to silently accept, and must be investigated before Step
   5).

Then, `-m live`: `run --pack nlq-structured-query --model <key>` end to end — S4's second half of its
"both packs run end to end" done-condition. `compare`/report inspection confirms `layer1ExactMatchRate`
renders as the headline, `byShape` prints per-shape, `unanswerableAbstainRate` prints separately, and
the two new failure counts print distinguishably (never pooled).

## 8. Test strategy

Plan §5's S4 row names items **9** and **16** (`docs/plans/small-model-benchmarking.md:5979`).

- **Item 9** — `scoring/extraction` — "the scalar/set shape rules and numeric epsilon"
  (`:6042`). This is `extraction.py`'s `_scalar_equal`/`score_pair` (§7 Pass B, step 4 above) — the
  numeric-epsilon case (`_NUMERIC_EPSILON = 0.01` plus the `1e-9` boundary slop) and the scalar/set
  shape rules (exactly one row/column for a scalar; flattened, canonicalized values for a set;
  the `conflicting-facts` subset-containment exception). **Note: item 9's own text names
  `scoring/extraction` only — it does not name a `scoring/classification` sibling test, and the
  numbered list has no other item that does either.** This is not an omission to route around: S4
  still needs `classification.py`'s own correctness tests (turn normalization, prompt rendering, the
  parse-failure fallback, the per-tier metric assignment), specified as unnumbered unit tests in §7
  Step 1 items 1-5, landing in `tests/test_scoring_classification.py`. They are real, required,
  red-then-green tests — simply not pre-assigned a number by the plan's own stable numbering
  (§5's own text: the numbering is stable and cited by number from review documents, so it is not
  renumbered here; a document that invented item "9b" would create a citation nobody else's review
  passes reference).
- **Item 16** — "One full run per pack, end to end, producing a stored, valid result." (`:6293`) —
  S4's live end-to-end runs, §7 Steps 2 and 4 above (one `run` invocation per pack, `-m live`,
  against a real LM Studio model). Both packs owe this independently — the S3/S4/S5/S6/S7 stage-
  ownership table (`:5978-5982`) states item 16 is "one arm of a per-pack obligation: each of S3-S7
  owes the end-to-end run for the pack it builds," so S4 owes it **twice**, once per pack, not once
  for the stage.

**Everything else in this document's own test list (§7's numbered steps) is new unit-test surface
this stage introduces and owns outright** — `tools/exec.py`'s Layer A/B validation matrix, the
`refresh_golden.py` extension's new origins and new modes, the `ExtractionAggregates`/`Pack.
prompt_config()`/`ItemScorer.build_messages` regression tests from Step 0 — none of it is a numbered
item from plan §5's list because none of it existed as a concept before S4 was planned (the plan's
numbered list was written before this stage's own seam-fix gap was found). This mirrors how S3's own
`refresh_golden.py`/cache-key tests were never numbered plan items either (S3 spec §8, step 0/1's own
"asserted in a unit test, not just implemented" language, with no item number attached) — a stage's
own scorer-module correctness tests are that stage's to specify, not a gap in the plan's numbering.

**Acceptance (human-run, once, recorded in `model-bench/docs/test-reports/`):** neither AC item in
plan §5's acceptance block (17-20) names S4 specifically, and S4's own done-conditions (§4 S4,
already quoted in §1 above) are fully covered by items 9/16 plus this document's own unit tests —
no new acceptance-tier test is owed by this stage. The relevant piece of item 20 (the FR-23 audit)
is explicitly S8's, not S4's (`:5983`).

## 9. Risks & open questions

- **`reference_specs.json`'s 40 entries are real, non-mechanical authoring work**, not something an
  implementer derives from `expected` (which states the *answer* shape, not the *query* that
  produces it) — flagged prominently in §7 Step 3's own "Done when" clause. **Possible
  acceleration, not verified this session:** falkor-chat's own `nlq_eval_results.json` may already
  carry a per-item structured spec some model produced during falkor-chat's own eval runs; if so it
  could seed a first draft, but it is the *output* of a model's own (possibly wrong) generation, not
  a certified-correct reference, and must not be trusted without a human review pass per item —
  worth a `data-scientist`/`tico` sanity check before relying on it, not a silent shortcut.
- **The `ws:nlq-eval` live snapshot is deliberately kept human-in-the-loop (§2.6)** rather than
  automated inside `refresh_golden.py`, specifically to avoid introducing `model-bench`'s first pip
  dependency against an explicit "stdlib only ... nothing else" hard rule. This is this document's
  own synthesis, not stated by the plan at this level of detail — low risk (the operation is a
  one-time, versioned-inside-the-pack-content-hash snapshot, not a repeated one), but worth a
  `tico`/stakeholder nod if a future stage needs frequent re-snapshotting, since automating it later
  is the reversal trigger this document explicitly does not pull.
- **`build_messages` staying optional (§4.1/§2.2) means a future chat-surface item-level pack
  (`chat-responder`, S7) could still ship with the generic JSON-dump fallback silently firing** if
  its own scorer forgets to define it — the same risk class finding 1 originally surfaced, not fully
  closed by this stage's ruling, only contained by convention (both S4 scorers define it, and S4's
  own wiring test proves the call-site preference works) rather than by enforcement. Named here as a
  `validate_pack` follow-up (§4.4), not built — worth revisiting at S7 planning if a third role's
  needs make the case for enforcement stronger, the same way S3's own `ItemScorer` extension was
  revisited for S4.
- **`Outcome`'s declared `Literal["pass", "fail", "n_a", "parse_failure"]` is already narrower than
  shipped usage** (`"unrunnable"` is asserted in `tests/test_runner.py`/`tests/
  test_scoring_retrieval.py` today, §2.1) — S4's own scorers replicate the established, tested
  convention (`"unrunnable"` on a no-response branch) for consistency rather than introducing a
  fourth spelling; the type/usage drift itself is pre-existing, not introduced here, and is a
  trivial one-line `Literal` widening whenever someone picks it up — not blocking, named so it is
  not mistaken for an S4-introduced defect during review.
- **The `nlq-structured-query` pack's system prompt is built per item, not resolved once via
  `prompt_config().systemPrompt`** (§5.2.1) — a deliberate, stated asymmetry with `guard-judge`
  rather than an inconsistency: the schema differs by the item's own `dataset`, so no single
  resolved string could serve every item. Confirm this reads cleanly to whoever reviews the pack
  pair side by side; if it reads as surprising, a one-line note in `pack.json`'s own `description`
  field is a cheap mitigation, not a design change.
- **`_pack_version_gate`'s reuse across four write modes now** (`--embed-corpus`,
  `--check-tables-shape`, `--stamp-answerability`, and the default import) means a single
  `packVersion` bump must cover potentially several content-hash-changing writes in sequence for
  `nlq-structured-query` specifically (import, then table-snapshot, then answerability-stamp, each
  itself content-hash-changing) — the gate as shipped checks "has `packVersion` changed since
  `PROVENANCE.md` was last written," which is satisfied by the FIRST write in the sequence and then
  fails the SECOND unless `PROVENANCE.md` is not rewritten until the whole sequence completes, or the
  operator bumps `packVersion` again between steps. **Verify `_pack_version_gate`'s exact semantics
  directly against `refresh_golden.py`'s shipped code before Step 4** (not fully re-derived in this
  session beyond the S3-spec-cited call sites) — if it gates per-write rather than per-sequence, Step
  4's two actions may need an intermediate `packVersion` bump between them, which this document does
  not currently script.

## Ready to implement

Document: `docs/plans/small-model-benchmarking-s4-spec.md` (this file). Four steps (§7): **Step 0**
(the seam fix — `ItemScorer.build_messages`, `Pack.prompt_config()`'s path-to-content resolution,
`ExtractionAggregates`'s two new fields, all offline, all backward-compatible with the full existing
S1/S2/S3 suite); **Step 1-2** (`guard-judge-understanding` — offline scorer/prompt build, then one
live end-to-end run); **Step 3-4** (`nlq-structured-query` — offline `tools/exec.py` + `extraction.py`
in three passes, then the human-in-the-loop `ws:nlq-eval` snapshot, the answerability stamp, and one
live end-to-end run). Both review findings are resolved as design rulings, not left open:
`build_messages` stays optional/`getattr`-guarded (§4.1 — required-in-the-no-guard sense would break
19 existing chat-branch call sites across S1/S2's own tests; optional-with-a-wiring-test is the only
backward-compatible design); the `prompt.systemPrompt`/`toolSchemas` path-resolution fix is folded
into S4 Step 0 alongside the new Protocol method, not a separate precursor unit (§4.2 — it is small,
currently inert with no real caller until `build_messages` exists, and both land in the same area of
`packs.py`/`runner.py` one implementer already has open). One further design gap closed beyond the
review's own two: the `ws:nlq-eval` live snapshot is resolved as a human-in-the-loop step rather than
a new `falkordb` pip dependency (§2.6), preserving the component's stated "stdlib only" hard rule.
Five items in §9 are flagged for attention before/during implementation, the sharpest being
`reference_specs.json`'s 40 hand-authored entries (real work, not mechanically derivable) and
`_pack_version_gate`'s exact multi-write semantics (verify directly before scripting Step 4's two
sequential live-adjacent actions).
