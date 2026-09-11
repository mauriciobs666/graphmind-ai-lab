# Change History — model-bench

> Dated log of actual changes to the `model-bench` component. Most recent first.

## 2026-09-11 — S4 Step 1: `guard-judge-understanding`, offline half

**What:** `docs/plans/small-model-benchmarking-s4-spec.md` §5.1/§6/§7 Step 1 — the first of S4's
two concrete packs, built entirely offline on top of Step 0's already-merged seam fix
(`ItemScorer.build_messages`, `Pack.prompt_config()`'s content resolution).

1. **`packs/guard-judge-understanding/{pack.json, prompts/judge.md, PROVENANCE.md, items.jsonl}`**
   — the pack itself. `prompts/judge.md` is `falkorchat.app._JUDGE_SYSTEM_PROMPT` transcribed
   verbatim (confirmed byte-identical via an independent AST literal-eval of the live source, not
   hand-typed and eyeballed). `items.jsonl` is the real 85-row `golden_guards.jsonl`, `id`→`itemId`
   renamed with every other field carried through unchanged (confirmed byte-identical to the
   renamed original, all 85 rows, by direct comparison) — produced by actually running
   `scripts/refresh_golden.py --pack packs/guard-judge-understanding` against the real,
   still-present `falkor-chat` tree, not hand-written.
2. **`modelbench/scoring/classification.py`** (new) — guard-judge's `ItemScorer`, a module per
   `scoring/retrieval.py`'s own precedent: `_normalize_turns` (transcribed from
   `guards._recent_turns`, filter-non-empty-text-before-slicing-to-last-6, verified against the
   live source directly), `_render_judge_user` (the CONDITION/CURRENT STATE/RECENT TURNS
   three-block render with the 6000-char oldest-evicted-first truncation), `extract_own_line_json_
   object` (the conservative own-line JSON parser, `llm.py`'s), `build_messages` (never reads
   `expected`/`label_rationale`/`r1_probe`), `score_item` (the same `{"decision": False, ...}`
   fallback the real judge applies on an unparseable reply; `result is None` → `"fail"`/
   `"unrunnable"` per `timing.withheldFor`, mirroring `scoring/retrieval.py`'s own precedent), and
   `aggregate` (builds a `ClassificationAggregates` whose `perClass` carries `falseAdvanceRate`
   (n=40)/`falseSuspendRate` (n=30) — the two verdict metrics — plus `advanceRecall`,
   `falseAdvanceRateBoundary` (n=15) and the four path-split diagnostics, every one at its own
   named subset's `n`, never the run's 85).
3. **`scripts/refresh_golden.py`** — generalized `_TRACKED_ORIGINS` (a single module-level
   constant, embedder-only) into `_TRACKED_ORIGINS_BY_PACK_ID`, keyed by `packId`, with two new
   resolvers (`_read_pack_id`, `_origins_for_pack_id`) and a `_JSONL_TRANSFORM_BY_PACK_ID` dispatch
   for the per-pack `jsonl-transform` row function. `_run_import`/`_check_origins` now take an
   already-resolved `origins` tuple as an explicit parameter rather than reading the module mapping
   themselves, so both stay testable against a synthetic origin list with no `pack.json` on disk —
   the same shape their own pre-existing tests used, now passing `origins=` explicitly instead of
   monkeypatching a module constant. Added `guard-judge-understanding`'s one `jsonl-transform`
   origin (`golden_guards.jsonl` → `items.jsonl`) and `_items_rows_from_golden_guards` (the
   `id`→`itemId`-only rename). `nlq-structured-query`'s three origins, `--check-tables-shape` and
   `--stamp-answerability` are S4 Step 3/4's, not built here.

**Tests, red before green, in the spec's own order (§7 Step 1 items 1-6):**
`tests/test_scoring_classification.py` (new, 57 tests) — `_normalize_turns`/`_render_judge_user`
against hand-built rows, including the eviction-by-suffix truncation at a synthetic char budget and
the raw-message-shape input (`msgId`/`displayName`, no `speaker` key — the exact defect §2.3
finding 2 exists to catch, re-exercised at `build_messages`'s own seam against the real `tn-01`
item); `extract_own_line_json_object` against vectors reused from falkor-chat's own
`test_app.py`/`eval/test_judge.py` (bare/fenced object, quoted-mid-sentence rejection, two-
candidate-objects rejection, the own-line-vs-inline array-wrapped asymmetry); `build_messages`
against real `ca-01`/`tn-01` items (system message equals `prompts/judge.md`'s file content
exactly; user message matches a hand-computed `_render_judge_user` expectation) plus the
delete-three-keys-first "must not raise `KeyError`" gold-field-leak guard; `score_item`'s
parse-failure fallback, timeout/no-response branches, and one clean case per tier; `aggregate` over
a synthetic 22-item (10/8/4-per-tier) fixture asserting the path-split metrics sum back to their
parent tier and `advanceRecall` is `falseSuspendRate`'s exact complement, plus a real-85-item
sanity check confirming the spec's own cited n=40/30/15; `packs.validate_pack` on the real, shipped
pack. `tests/test_refresh_golden.py` gains 6 new tests (`_items_rows_from_golden_guards`'s rename
rule and its own "never leaves a bare `id` key" mutation-test target, `_TRACKED_ORIGINS_BY_PACK_ID`
shrink/widen guard, `_read_pack_id`/`_origins_for_pack_id`) and adapts its three pre-existing
`_check_origins` tests from monkeypatching `_TRACKED_ORIGINS` to passing `origins=` explicitly.

**Verification:** baseline reproduced before any edit — `1261 passed, 1 failed (the pre-existing
S5 tripwire), 3 deselected`. After this change: `1324 passed`, the same one pre-existing failure,
`3 deselected` — 63 new tests, zero regressions. `ruff check .` clean. `./run.sh validate --pack
packs/guard-judge-understanding` prints `guard-judge-understanding 1.0.0 (guard-judge): valid`,
exit 0; `packs.validate_pack` on the loaded pack returns `[]`. `refresh_golden.py --pack
packs/guard-judge-understanding --check-origins` reports `unchanged` and exit 0, and re-running it
against `embedder-graphrag-retrieval` afterward reports all five origins `unchanged` too — the
generalization touched nothing under that pack.

**`validate --strict` discrepancy, noted and not closed here:** the spec's own §7 Step 1 item 6 and
Step 3 Pass C item 8 both name `validate --pack <id> --strict`, but `cli.py`'s `_cmd_validate`
raises `NotImplementedError` unconditionally on `--strict` — a pre-existing, deliberate S2-era
deferral (runner-spec §9: "`--strict`'s semantics are never given anywhere in the plan"),
unaffected by this pack and out of this step's own scope (§1's "in scope" list names neither
`cli.py` nor `--strict`). This step's own "passes clean" done-condition is satisfied against
`packs.validate_pack` directly instead — the check `--strict` would still have to call underneath,
once someone builds it. Flagged for whoever executes S4 Step 3, whose own item 8 needs `--strict`
to actually run (fail on an unstamped `nlq-structured-query` fixture, pass once stamped) and cannot
be satisfied without resolving this gap first.

**Mutation-tested three consequential branches** (each: copy the file aside, mutate, confirm the
targeted test(s) redden, restore by copy, never batched): the parse-failure bias-to-suspend
fallback (`advanced = False` → `True` on an unparseable reply) reddened the targeted fallback test
plus two downstream `aggregate` tests; the `_METRIC_BY_TIER` tier→metric mapping (swapped
`clear_suspend`/`clear_advance`'s metric assignments) reddened 8 tests across `score_item` and
`aggregate`, including the real-85-item sanity check; the "never reads the gold label" guarantee
(added a `item_input["expected"]` read inside `build_messages`) reddened the targeted delete-keys
guard with a `KeyError`, exactly as designed. Each mutant's diff against the restored file was
confirmed `IDENTICAL` before continuing.

**CPG:** considered, not relevant — no `cpg_model-bench` graph loaded on this FalkorDB instance
(checked live this session), and this is greenfield code-level work with no CPG to consult.

## 2026-09-11 — S4 Step 0: the seam fix (`build_messages`, `prompt_config()` content resolution, `ExtractionAggregates`'s two new counts)

**What:** `docs/plans/small-model-benchmarking-s4-spec.md` §4/§7 Step 0 — the seam fix ahead of
S4's two packs, closing both gaps `docs/reviews/itemscorer-extension.md` found:

1. **`ItemScorer.build_messages`** — a fourth optional method on the `runner.ItemScorer` Protocol,
   `getattr`-guarded exactly like `prime`/`embed_text`. `_drive_single_call_items`'s chat branch now
   calls it in preference to the generic `_item_chat_messages(pack, item_input)` when the scorer
   defines it, falling back unchanged when it does not.
2. **`Pack.prompt_config()` now resolves `prompt.systemPrompt`/`prompt.toolSchemas` to content**,
   via two new private helpers, `_resolve_prompt_text`/`_resolve_tool_schemas` (`packs.py`) —
   mirroring `data_path`'s resolution pattern. `systemPrompt` becomes the named file's text (`None`
   stays `None`); `toolSchemas` becomes the parsed JSON array as a tuple (`None`/absent → `()`).
   Both wrap a missing/unreadable file into `PackConfigError`; `_resolve_tool_schemas` additionally
   rejects a non-array file. `prompt_config()`'s own docstring paragraph ("paths, not resolved
   content... no such caller exists in this tree yet") is replaced with a note pointing at the two
   helpers, reconciling it with `convo.PromptConfig`'s docstring, which already stated the true
   contract and needed no change.
3. **`ExtractionAggregates` gains two new defaulted integer fields**, `malformedSpecCount` and
   `schemaViolationCount` (`results.py`) — scalar counts, never added to `named_metrics()`, matching
   `parseFailures`'s own existing shape. The generic `_aggregates_to_dict`/`_aggregates_from_dict`
   dispatch (already shipped, reflective over `vars()`/kwargs) needed no change to carry them.

**Tests, red before green:** `tests/test_runner.py` gains `FakeItemScorerWithBuildMessages` plus
two tests — the chat branch prefers `build_messages` when defined, and falls back to
`_item_chat_messages` unchanged when absent (the regression check: all 19 pre-existing
`call_surface="chat"` sites use `FakeItemScorer`/`FakeItemScorerWithHooks`, neither of which defines
`build_messages`, and stayed green throughout). `tests/test_packs.py` gains two new assertions on
the existing `"valid"` fixture (`cfg.systemPrompt` equals `prompts/system.md`'s real text,
`cfg.toolSchemas == ()`), a new fixture pack `tests/fixtures/packs/prompt_tool_schemas/` asserting a
non-empty `toolSchemas` array resolves to the matching tuple, and three additional `tmp_path`-built
negative tests for the two new `PackConfigError` paths (missing `systemPrompt` file, missing
`toolSchemas` file, `toolSchemas` file that parses but is not a JSON array). `tests/test_results.py`
gains a `store()`/`load_history()` round trip for `ExtractionAggregates(malformedSpecCount=3,
schemaViolationCount=5)` with non-default, mutually distinct values.

**Verification:** baseline reproduced before any edit — `1254 passed, 1 failed (the pre-existing
S5 tripwire, `test_convo.py::test_the_third_leg_of_the_disposition_probe_is_still_owed_by_s5`),
3 deselected`. After this change: `1261 passed`, the same one pre-existing failure, `3 deselected`
— 7 new tests, zero regressions. `ruff check .` clean. Mutation-tested the `build_messages` guard
(removed it — 8 pre-existing tests reddened with `AttributeError`, as expected, since none of their
fakes define `build_messages`) and both new `PackConfigError` paths in `_resolve_prompt_text`/
`_resolve_tool_schemas` (stripped the `except OSError` wrap from each, and the non-array check from
`_resolve_tool_schemas` — each mutant reddened its own targeted negative test); each mutation copied
the file aside first and was restored immediately after, never batched with another.

**Grep discrepancy, noted and closed:** the spec's own supporting comment claims `grep -rn
"parseFailures" tests/ modelbench/` finds "six hits, all kwargs"; the actual count, reproduced
independently before landing this, is 8 (5 keyword-construction sites under `tests/`, 3 field
declarations in `results.py` — `ClassificationAggregates`, `ExtractionAggregates`,
`GroundingAggregates` each declare one). Non-load-bearing: every construction site is keyword-only
either way, so the two new defaulted fields are backward compatible regardless of the exact count.

## 2026-09-11 — `validate_pack`'s sixth axis: the `"scorer"` key's own resolution

**What:** `packs.py`'s `validate_pack` did not check that a pack's declared `"scorer"` value
resolves to an importable `modelbench.scoring.<name>` module — a typo'd or nonexistent scorer
validated clean and only failed one step later, at `run`, via `runner._load_item_scorer`'s
`RunRefused(exitCode=4)`. New `_scorer_problems(pack)` mirrors that function's own
`importlib.import_module` call, returning a problem string on `ImportError` instead of raising.
Follows the same "absent key is not a problem" convention as `_tool_module_problems` (`tools.module`)
and `_prompt_problems` (`prompt`): a pack with no `"scorer"` key validates clean on this axis, since
most fixtures under `tests/fixtures/packs/*/pack.json` declare none.

**Scoped by role, not just by key presence:** `runner.run_pack`'s own branch (`if pack.role ==
"tool-caller"`) never reaches `_load_item_scorer` for that role — it calls `_drive_conversations`
-> `_load_conversation_scorer` instead, which resolves a different, unbuilt `ConversationScorer`
kind (S5's) and unconditionally raises `NotImplementedError` regardless of what `"scorer"` names.
The first pass over this check mirrored `_load_item_scorer` without the role gate and broke the
shared `tests/fixtures/packs/valid/` fixture (`"scorer": "toolcalls"`, tool-caller role, no such
module ships yet) — a false positive against a role this axis was never meant to reach. Fixed by
returning `[]` immediately for `pack.role == "tool-caller"`, same shape as the other "not this
function's problem" early-outs.

New fixture `tests/fixtures/packs/scorer_unresolvable/` (otherwise-valid `guard-judge` pack,
`"scorer": "not_a_real_scorer_module"`) plus four new tests in `tests/test_packs.py`: the rejection,
a no-`"scorer"`-key fixture staying clean, the real shipped `packs/embedder-graphrag-retrieval/`
pack (`"scorer": "retrieval"`, which genuinely resolves) staying clean end to end, and the `valid`
fixture's tool-caller `"scorer"` staying out of this axis's reach. `validate_pack`'s docstring
updated from five to six axes.

**Verification:** full suite `1254 passed` (up from `1250`), plus one pre-existing failure
unrelated to this change (`test_convo.py::test_the_third_leg_of_the_disposition_probe_is_still_
owed_by_s5`, a self-documented tripwire for S5's still-unbuilt scoring wiring, present before and
after this change). `ruff check .` clean. Mutation-tested both new branches of `_scorer_problems`
(neutralizing the `ImportError` handler; removing the `tool-caller` early return) — each killed its
target test, then was restored.

## 2026-09-11 — S3, Step 2: the live end-to-end run (`--embed-corpus`, the harness self-check, `run`/`compare`) — S3 closed

**What:** `docs/plans/small-model-benchmarking-s3-spec.md` §8 Step 2, all five done-conditions.
`scripts/refresh_golden.py` gained its real `--embed-corpus` mode: `main()`'s branch resolves a live
`ModelInfo` via `LMStudio.catalog()` + a new `_find_model_in_catalog` (mirrors `runner._find_model`'s
convention), warms the model up (mirrors `run_pack`'s own step 4) before one batched `embed_corpus()`
call over the pack's 121 prefixed documents, and writes RAW vectors plus a `compute_cache_key` header
to `corpus.embeddings.json`. Reuses `_pack_version_gate` verbatim for this write path — refuses
under an unchanged `packVersion`, checked before any LM Studio contact. `pack.json`'s `packVersion`
bumped `1.0.0` -> `1.1.0` (a new committed, content-hash-changing artifact).

**The live run, in the spec's own order:**
1. `--embed-corpus --model text-embedding-qwen3-embedding-0.6b` wrote the real
   `corpus.embeddings.json` (121 x 1024 raw vectors, `normalized: false`).
2. Done-condition 4 — the offline ranking self-test (`tests/test_scoring_retrieval.py`,
   `TestOfflineRankingSelfTestAgainstTheRealPackData`): `prime()` cache-HITs against the just-written
   file with an exploding stub `LMStudio` (no live call at all), scores all 38 golden queries against
   `golden_retrieval.embeddings.json`'s fixed vectors, and asserts determinism (byte-identical
   `RetrievalAggregates` across two runs) plus well-formed bounds — deliberately not asserted against
   `retrieval_baseline.json`'s pinned figures (`-ml` §5.4: exact-vs-ANN and vector-only-vs-hybrid are
   uninterpretable as a quality signal in either direction).
3. Done-condition 5 — the harness self-check (live, 38 fresh query embed calls): **recall@10 =
   0.9736842105263158 (37/38)**, identical to `retrieval_baseline.json`'s pinned figure; recall@5 and
   MRR also close. Well above the `-ml` §5.4 ~0.85 floor, so the below-0.85 investigation (wrong
   prefix, unnormalized vectors, truncated corpus) was not triggered. Write-up:
   `docs/test-reports/embedder-self-check-report.md` — diagnostic only, never a gate, per the
   2026-09-02 stakeholder ruling.
4. `attest` (real `host.json`) then done-condition 1 — `run --pack embedder-graphrag-retrieval
   --model text-embedding-qwen3-embedding-0.6b --session s3-step2-live-2026-09-11`: a real stored
   `RunResult` (38 items, recall@10 37/38, MRR 0.6278).
5. Done-condition 2 (storage half) — the already-wired `_cmd_run` deterministic-arm hook stored the
   BM25 reference arm under the same `sessionId` in the same invocation.

**A real, pre-existing defect surfaced mid-run, not caused by this unit's own code:** `compare
--pack embedder-graphrag-retrieval --session s3-step2-live-2026-09-11` excluded both stored arms as
`INVALID` — `results.py`'s `_item_problems` bug, dispatched separately and fixed (see the entry
immediately below this one). Re-running `compare` against the same session after that fix now
renders both arms with no exclusion, closing done-condition 2 in full.

**S3 is now fully closed — all five done-conditions met.** Files touched (in-scope only; `results.py`
was a separate unit): `scripts/refresh_golden.py`, `tests/test_refresh_golden.py`,
`tests/test_scoring_retrieval.py`, `packs/embedder-graphrag-retrieval/pack.json` (`packVersion`),
`packs/embedder-graphrag-retrieval/corpus.embeddings.json` (new),
`docs/test-reports/embedder-self-check-report.md` (new), this entry, `README.md`, `AGENTS.md`.

## 2026-09-11 — Bug fix: `results.py`'s `_item_problems` quarantined every valid continuous-metric record

**What was wrong:** `_item_problems` (the read-time half of `ItemResult.scored_outcome`'s
contract) flagged a declared-`scoreable` metric `absent` whenever it was missing from
`item.counts` — full stop, never checking `item.measures`. But `scored_outcome`/`scored_value`
already establish, and enforce, a legitimate split: a metric's instrument lives in **either**
`counts` (a boolean/binary metric) **or** `measures` (a continuous one, e.g. `mrr`,
`separationRaw`, `separationZ`) — never both, and which one is a property of the metric's own
kind. A record that correctly recorded a continuous metric's score in `measures` was still read
back as if that metric's count were missing, so `load_history` classified the whole record
`INVALID` and `compare` excluded it.

**Why it was never caught before:** no pre-S3 role/pack declared a continuous `scoreable` metric
through the real `store()` -> `load_history()` -> `compare_report()` round trip. S3's new
`embedder` pack was the first — its `mrr`/`separationRaw`/`separationZ` all live in `measures` —
and it surfaced the defect via a sibling unit's live end-to-end run against real stored data
(`results/runs/embedder-graphrag-retrieval-*-2026-09-11T11:27:30Z.json`), not via this unit's own
testing: both of that run's arms were classified `INVALID` (~114 "absent" field problems each,
all naming `mrr`/`separationRaw`/`separationZ`) and `reports/embedder-graphrag-retrieval-20260911-01.md`
rendered the `INVALID RESULTS EXCLUDED` block with no comparison.

**The fix:** `_item_problems` now accepts either home — `metric not in item.counts and metric not
in item.measures` — mirroring the split `scored_outcome`/`scored_value` already enforce, rather
than checking `counts` alone. Re-running `compare` against the same stored session
(`s3-step2-live-2026-09-11`) now renders both arms with no `INVALID` block
(`reports/embedder-graphrag-retrieval-20260911-02.md`).

## 2026-09-11 — S3, Step 1: `scoring/retrieval.py` + the `runner.py`/`cli.py` wiring, offline throughout

**What:** `docs/plans/small-model-benchmarking-s3-spec.md` §8 Step 1 — the embedder's `ItemScorer`
in full (`modelbench/scoring/retrieval.py`, new): recall@k/MRR/precision@k, L2-normalize + brute-
force cosine, score separation (raw and z, population `statistics.pstdev` per `-ml` v1.24), the
BM25 reference-arm machinery (always-positive IDF, `-ml` §5.3), `prime`/`embed_text`/`score_item`/
`aggregate` (§6.2), and `deterministic_arm` (§6.3). Plus the two required, additive edits to
already-shipped S2 code the spec's §2.2/§3.2/§3.3 name: `runner.py`'s `_load_item_scorer` (a real
`importlib.import_module("modelbench.scoring.<name>")`, `RunRefused(exitCode=4)` on an absent/
unresolvable `"scorer"` key) and `_drive_single_call_items`'s two `getattr`-guarded lines
(`prime()` once before the per-item loop, `embed_text()` replacing the old hardcoded
`json.dumps(item_input)` on the embeddings branch); `ItemScorer` gains `prime`/`embed_text` as
optional Protocol members. `cli.py`'s `_cmd_run` gains the deterministic-arm hook (§7.2): after
`store(run, ...)`, for a non-`tool-caller` pack, resolves the scorer and calls its optional
`deterministic_arm` if present, storing a second `RunResult` under the same `sessionId`.

**The spec's own flagged ambiguity, resolved per the coordinating session's ruling (not
re-litigated here):** `prime()`'s corpus raw-norm diagnostic runs unconditionally, on a cache HIT
as well as a cache miss, and is stored durably in `_state["corpusNormDiagnostic"]` (not only
logged) so a later Step-2 self-check report can read it back.

**Two deviations found from the spec's own "no existing S2 test needs to change" claim (§2.2),
both small and within this unit's own test-file scope:**
1. `tests/test_runner.py`'s pre-existing `FakeItemScorer` (no `embed_text`) is used by an existing
   S2 test with `call_surface="embeddings"`; since `embed_text` is reached with **no** `getattr`
   guard (by design, §3.2), that test broke until `FakeItemScorer` gained an `embed_text` method
   mirroring the old hardcoded text. No test's assertions changed.
2. `cli.py`'s new deterministic-arm hook calls the real `_load_item_scorer(pack)` a second time,
   after `store()`. In production this can never raise (a non-`tool-caller` pack's scorer is
   already resolved once inside `run_pack` -> `_drive_single_call_items`, which would have raised
   `RunRefused` earlier if it could not), but `tests/test_cli.py`'s existing S2 CLI tests fake
   `run_pack` entirely, bypassing that internal resolution — exposing the real, scorer-less
   `guard-judge` fixture pack to the hook's real call. Wrapped in `try/except RunRefused: scorer =
   None` so this optional, additive step degrades to a no-op rather than crashing an
   already-successfully-stored `run`, rather than editing the pre-existing fixture manifest.

**`tests/test_convo.py::test_the_third_leg_of_the_disposition_probe_is_still_owed_by_s5` is now
expected-red — working exactly as designed, not a Step 1 defect.** It is a pre-existing,
deliberate tripwire keyed on the bare existence of `modelbench/scoring/`, which this unit is the
first to create. Its own docstring anticipated and rejected narrowing the guard to a specific S5
module path ("If S5's scorer lands somewhere other than that package, this tripwire will not fire
and the leg is still owed") — the broad existence check is deliberate coarseness, not an oversight.
Flagged to the coordinating session as a fork (touch `convo.py`/`test_convo.py`, outside this
unit's fenced scope, vs. leave red); ruled: leave it red. The leg it names — wiring S5's
`TurnDisposition` branch set into `convo.py`'s three-way disposition probe — stays blocked on S5
(`tool-caller`'s scorer, out of scope here) until that stage lands.

**Tests:** `tests/test_scoring_retrieval.py` (new, 60 cases: pure arithmetic against hand-built
ranked lists; BM25 against a synthetic corpus with a hand-computed expected score; `prime`/
`embed_text` cache-key match/mismatch with a stub `LMStudio`; `score_item`/`aggregate`, including
the `-ml` v1.24 binarization ruling; `deterministic_arm`); `tests/test_metrics_agreement.py` (new,
the 20 hand-transcribed cases plus 2 meta-tests); `tests/test_runner.py` and `tests/test_cli.py`
gain the S3 wiring tests (§8 Step 1 item 7). Mutation-tested (restore-by-copy, `diff -q`-verified,
`PYTHONDONTWRITEBYTECODE=1`, one mutation at a time): the `recall_at_k` binarization's `> 0` vs a
stricter reading; the cache-key-mismatch branch actually re-embeds live rather than reusing stale
committed vectors; `prime`/`embed_text` silently no-opping when a scorer defines them. All four
mutants killed by name.

## 2026-09-10 — Bug fix: `Pack.prompt_config()` crashed every `validate`-clean, prompt-less
item-level pack

**What:** found live building the CLI `run` command (previous entry, same day). `Pack.
prompt_config()` (`modelbench/packs.py`) parsed the manifest's `prompt` block, defaulted an absent
block to `{}`, and then unconditionally raised `PackConfigError` when `historyReplay` was outside
`convo._HISTORY_REPLAY_MODES` — `None` whenever `prompt` was absent, **regardless of role**. But
`_prompt_problems`/`validate_pack` (same module) already treats an absent `prompt` block as not a
problem, on any role, so `validate` reported a prompt-less item-level pack (`guard-judge`,
`nlq-generator`, `chat-responder`, `embedder`) clean while `runner.run_pack`,
`_drive_single_call_items`, and `_item_chat_messages` — all four call `pack.prompt_config()`
unconditionally and uncaught — crashed on the same pack with an uncaught `PackConfigError`,
contradicting `validate`'s own verdict.

**Fix:** `historyReplay`'s check is now scoped by `roles.MULTI_CALL_TURN_BY_ROLE[role]`, the same
role table `maxIterationsPerTurn` was already scoped by (v1.26) — required *iff* `True` (currently
`tool-caller` only), skipped otherwise. Chosen over introducing a narrower runner-side accessor:
`convo.assemble`/`drive` (`historyReplay`'s only reader) is reached only from `runner.
_drive_conversations`, itself `tool-caller`-only, so no item-level role has ever consumed the
field, and reusing the existing column is a one-line, single-module fix rather than a signature
change across four runner call sites. The plan doesn't state a role-scoping rule for `historyReplay`
by name the way it does for `maxIterationsPerTurn`; the two conditions coincide exactly on the
pack's current role table, so no plan-semantics question was raised.

**Tests:** `tests/test_packs.py` gains a reproduction test (`Pack.prompt_config()` on the real,
on-disk, `validate`-clean `tooling_import_allowed` fixture no longer raises, returns `historyReplay
is None`) and a guard test (a prompt-less `tool-caller` fixture still raises `PackConfigError`
mentioning `historyReplay` — the role that genuinely needs the field is unaffected).
`tests/test_runner.py` gains a reproduction test driving the same real `Pack` (not `FakePack`,
which duck-types `prompt_config()` and never touches the bug) through `_item_chat_messages`, the
exact runner call site that used to crash. Mutation-tested: reverting the role guard to the old
unconditional check reddens exactly these two new item-level tests, no more, no fewer.

## 2026-09-10 — S2, Step 2: CLI `validate`/`run` wiring, exit codes — closes S2

**What:** `docs/plans/small-model-benchmarking-runner-spec.md` §7 Step 2, the last of the three
dispatchable steps and the last unit of S2. `modelbench/cli.py` gains `_cmd_validate` and
`_cmd_run`, wired into `_build_parser()`/`main()` following the existing `attest` pattern:
`validate --pack <path> [--strict]` wraps `packs.load_pack`/`packs.validate_pack` (structural only,
no LM Studio, no model catalog); `run --pack <id> --model <key> [--session] [--reference]
[--warmup] [--first-call-timeout] [--request-timeout]` resolves the pack under `--root/packs/<id>`
(`_pack_root`, the `run`-side counterpart to `_cmd_compare`'s inline manifest lookup), validates it,
builds `LMStudio` from `host.json`'s own `apiBaseUrl` (read directly — `run` takes no
`--api-base-url` of its own), and calls `runner.run_pack`, mapping `RunRefused.exitCode` and the
dispatch-failure funnel onto §3.6a's exit codes. The module docstring's exit-4 line and its
"`validate` and `run` ... deliberately still absent" sentence are both corrected — S2 now ships all
three of `attest`/`validate`/`run`.

**`--strict`'s semantics**, per the spec's own explicit routing (§6.1/§9: "a genuine gap, not merely
scattered" — the plan names the flag with no elaboration anywhere): `_cmd_validate` raises
`NotImplementedError` the moment `--strict` is set, with a message and docstring citing the spec
section to resolve it against. Deliberate and documented, not a silently-accepted no-op — the one
option the spec rules out — and matches `runner.py`'s own established precedent for an inert seam
(`_load_item_scorer`/`_load_conversation_scorer`, Step 1).

**Store-before-exit-4 ordering** (dispatch-failure note §4(d)): `_cmd_run` calls `store()` and
prints `"stored: <path>"` *before* checking `disclosures`, so a `tool-caller` conversation censored
by a dispatch raise still lands `results/runs/<runId>.json` on disk before the process reports exit
`4` — never an aborted run.

**Tests:** `tests/test_cli.py` gains 18 tests (net +17 after `test_s2s_remaining_commands_are_not_
shipped_yet` — which pinned `validate`/`run` as *unrecognized*, `main([cmd]) == 2` — was rewritten,
not deleted, since its own premise went false: `test_validate_and_run_are_now_recognized_commands`
reads `--help`'s subcommand list instead of a bare exit code, since a recognized command missing
its own required `--pack` exits `2` too — the same code an unrecognized command exits with).
`validate`: exit 0 on a structurally valid pack, exit 4 on a load error and on a `validate_pack`
problem, `--strict`'s deferral, and that omitting `--strict` runs normally. `run`: exit 4 on a pack
load error and a `validate_pack` failure; exit 5 on `host.json` absent and on schema-invalid;
exit 3 on each of `probe()`'s two distinct negative outcomes (`unreachable`/`v1-only`) and on a
warm-up timeout; exit 4 on the `callSurface`-versus-catalog-`type` contradiction and, separately, on
the tool-calling eligibility gate (`tool-caller` role only); exit 5 on a stale attestation
trip-wire; the clean exit-0 path stores and prints the path; and the store-before-exit-4 ordering
test itself. The refusal-path tests drive the real `run_pack` against a minimal on-disk pack (no
`data`/`tools`/`prompt` content needed — every refusal exercised fires before `run_pack` ever reads
one) and a hand-built `_StubLMStudioForRun` (`cli.LMStudio` monkeypatched, mirroring `attest`'s own
`_patch_lmstudio`); the exit-0 and store-before-exit-4 tests fake `cli.run_pack` directly, since
past that point it is `_cmd_run`'s own sequencing under test, not `run_pack`'s (already
`tests/test_runner.py`'s, Step 1) — matching spec §7 Step 2's own scope statement ("this step's job
is the `run`-level wiring *around* them, not re-testing the adapter itself"), entirely offline, no
`-m live` marker needed.

**Mutation-tested** (source restored by copy after each, `diff -q` verified clean,
`PYTHONDONTWRITEBYTECODE=1`): (1) the store-before-exit-4 ordering — moving the disclosure check
and its early `return EXIT_BAD_PACK` ahead of `store()` reddened exactly
`test_run_stores_the_record_before_returning_exit_four_on_dispatch_failure_disclosures` (1 failed,
5 passed among the dispatch/exit-zero tests) and no other; (2) the `RunRefused` exit-code
passthrough — hardcoding `return EXIT_LMSTUDIO_UNREACHABLE` in place of `return exc.exitCode`
reddened exactly the three `run`-exit-code tests whose refusal is genuinely a `RunRefused` carrying
a *different* code (`callSurface`/type contradiction and the tool-calling gate, both exit 4; the
stale attestation, exit 5 — 3 failed, 8 passed among the `test_run_exits_*` tests) while leaving the
two genuine exit-3 tests (`unreachable`/`v1-only`, which already expect 3) and the exit-4/5 tests
that route through a different code path entirely (pack load/validate failure, `host.json`
absent/invalid, which never reach this line) green — proving the tests discriminate the exact code
`_cmd_run` forwards, not merely "non-zero".

**Verified:** full suite `1126 passed, 3 deselected` (up from `1109 passed, 3 deselected`),
`ruff check .` clean.

CPG: considered, not relevant — `cpg_model-bench` is not a loaded graph on this FalkorDB instance;
a code-level task in a component with no CPG built, not a task without a code-level component.

## 2026-09-10 — S2, Step 1: `runner.py`'s core — capture order, `LatencyBlock` accumulation, both driving loops

**What:** `docs/plans/small-model-benchmarking-runner-spec.md` §7 Step 1, built directly on Step
0's `results.py`/`convo.py` landing (U115, previous entry). New `modelbench/runner.py`:
`RunConfig`, `RunRefused`, `DispatchFailureDisclosure`, `ItemScorer`/`ConversationScorer`
Protocols, `run_pack` (the ten-step capture-order sequence, §3.4.4a), `_drive_single_call_items`
(the four item-level roles), `_drive_conversations`/`_turn_timings` (`tool-caller`), the two load
producers `_load_withheld_for`/`_gap_withheld_for` plus `_gap_ms`, and `latency_block` — the
accumulation pass building `LatencyBlock` from a run's `items`, satisfying all nine invariants in
spec §5. `modelbench/packs.py` gains `Pack.iter_items()`/`iter_scripts()`/`find_script()` (spec §9's
own flagged, unspecified helpers — pure data-iteration, no design stakes). New
`tests/test_runner.py` (55 tests), entirely offline: hand-built stub `LMStudio`/`ToolEnvironment`/
pack objects, no real pack, no scorer, no network — driven in the spec's own order: test 15b's
case list first (`latency_block`'s nine invariants, the multi-call (a)-(f) cases via
`_turn_timings`), then v1.31's per-conversation `ToolEnvironment` tests, then a confirmation pass
on `_drive_conversations`'s reading of `TurnTrace`/`ConversationTrace`, then the dispatch-failure
note's E2-E4, then 12/12b's `basis` wiring, then `run_pack`'s own capture-order refusals and one
offline happy path.

**Three genuine gaps in the spec's own pseudocode, resolved and documented at their own sites in
`runner.py`'s module docstring** (not silently patched):

1. `_drive_conversations`'s `design_effect = stats.design_effect(...)` is called with its
   arguments elided in the spec. Reading `docs/plans/small-model-benchmarking-ml.md` §4.5.1/R1
   directly (as spec §7 Step 1 instructs) settles it differently from a bootstrap-width call: the
   note states "DEFF 1.00 by construction" for the 12×1 tool-caller design outright — the sampling
   unit (script) and the analysis unit (script) coincide, so `_drive_conversations` returns `1.0`
   unconditionally, exactly like the item-level loop, and never calls `stats.design_effect`
   (a general Rule 5 utility for a design that *does* cluster its sampling unit).
2. `latency_block`'s spec signature (`items` only) cannot implement its own rule (iv-a):
   `statsCoveredCount` must read `None` on a surface with no `stats` at all (embeddings) and a real
   `0` on a chat surface where every call happened to lack `stats` — two states structurally
   identical in `items` alone. `latency_block` therefore takes a required keyword-only
   `call_surface`, threaded from `run_pack`'s own already-computed value.
3. `run_pack`'s pseudocode constructs `RunResult(aggregates=aggregates, ...)` from a variable the
   item-level driving loop never produces — `ItemScorer.score_item` (spec §3.2) yields one
   `ItemResult` per item with no run-level aggregation method, unlike
   `ConversationScorer.score_conversations`. `ItemScorer` gains one method beyond the spec's own
   two, `aggregate(items, *, pack) -> Aggregates`. Inert today: `_load_item_scorer`/
   `_load_conversation_scorer` raise `NotImplementedError` unconditionally (no scorer ships before
   S3), so this is a seam for S3 to confirm or revise, not a load-bearing decision anything already
   depends on.

**One finding, not fixed here:** `ToolCallAggregates` (`results.py`) has no `determinismProbe`
field, though the plan (§3.8.4) and the runner spec both describe `basis` as read from it.
`_drive_conversations` reads it defensively (`getattr(aggregates, "determinismProbe", None)`,
fail-safe to `"assumed"` when absent) so a real `ToolCallAggregates` cannot crash the run; the
12/12b basis-wiring tests use a stub `ConversationScorer` returning an object that *does* carry the
attribute, per the spec's own §8 test-strategy guidance. The field itself is presumably S5's to add
alongside the first real `ConversationScorer`.

**Mutation-tested** (source restored by copy after each, `diff -q` verified, `PYTHONDONTWRITEBYTECODE=1`):
the disposition-over-load precedence rule in both `_turn_timings` and
`_drive_single_call_items` (P15-3/P16-1 — a failing call/turn must not also be marked `"load"` by
the residency guard); `_drive_conversations`'s one-`ToolEnvironment`-per-conversation obligation
(v1.31); the censoring-vs-scored-failure discrimination E2/E4 exist to catch (dropping the
disclosure record while still censoring the trace). All four mutations reddened the intended
tests and no others; all four restorations diffed clean against the pre-mutation source.

**Verified:** full suite `1109 passed, 3 deselected` (up from `1054 passed, 3 deselected`),
`ruff check .` clean.

CPG: considered, not relevant — `cpg_model-bench` is not a loaded graph on this FalkorDB instance;
a code-level task in a component with no CPG built, not a task without a code-level component.

## 2026-09-10 — `tests/test_report.py`'s embedder fixtures corrected to `itemId` (route (iii) fallout)

**What:** U114's sampling-contract route (iii) enforcement (`check_sampling_contract`, previous
entry) is also called directly from `modelbench/report.py`'s `compare_report`, a second call site
neither U113's synthesis, U114 nor U115 scoped. Three `role="embedder"` `PackRef` fixtures in
`tests/test_report.py` (`_embedder_pack`, `_mixed_pack`, and the inline `PackRef` in
`test_a_continuous_units_value_is_the_mean_over_its_items_not_a_flattened_pool`) had always used
`pairingKey=("queryId", ...)`/`analysisUnit="queryId"` — self-consistent with routes (i)/(ii), so
never caught before, but not the embedder role's own field (`itemId`, plan §3.3's table). Once
U114 landed, these 11 tests failed for real via `report.py:687`'s `check_sampling_contract(pack)`
call — exactly route (iii)'s job, catching a live instance in the suite itself.

Fixed by renaming the three literal `"queryId"` occurrences to `"itemId"` — confirmed by grep to
be the only three sites in the file, no assertion elsewhere checks for the string `"queryId"`.
Test-fixture-only; no production code touched. Verified: full suite `1054 passed, 3 deselected`
(up from 1043 passed / 11 failed), `ruff check .` clean.

## 2026-09-10 — S2 U115: `results.py`'s timing carriers, `RunResult.latency`/`attestationTripWire`, and `ToolDispatchFailed`'s censoring payload

**What:** Step 0 of `docs/plans/small-model-benchmarking-runner-spec.md` §7 — the two genuine
plan-vs-shipped-code gaps the spec's §2.2 flagged, both required before `runner.py` (Step 1) can
be built at all.

`modelbench/results.py` gains `CallTiming` (one model call — `wallClockMs`/`ttftMs`/
`generationMs`/`promptTokens`/`tokensPerSecond`, plan §4 S1 `:2987-2996`), `ItemTiming`
(`wallClockMs`/`calls`/`withheldFor`, with `callCount` and `unexplainedMs` as **derived
properties** over `calls`, never stored — plan `:2999-3025`) and `LatencyBlock` (the nine-invariant
shape in the spec's §5, with `callAttemptedCount` derived as `callCount +
latencyWithheldForNoResponse`, v1.28's rule). All three carry `to_dict`/`from_dict`. Shapes only —
the accumulation pass that builds a `LatencyBlock` from a run's `items` (`latency_block()`) is
Step 1's, in `runner.py`, which does not exist yet.

`ItemResult.latencyMs` (a stored constructor field since S1) is now `timing: ItemTiming | None`
plus a derived `latencyMs` `@property` reading `None if timing is None or timing.withheldFor is
not None else timing.wallClockMs` — plan v1.10, §4 S1 `:3087-3097`, Appendix A `:7825`. `to_dict`
still emits both `timing` and a redundant `latencyMs` for a stored record's own readability;
`from_dict` ignores any `latencyMs` key and re-derives from `timing`. Every existing
`ItemResult(..., latencyMs=X, ...)` call site (`tests/conftest.py`, `tests/test_results.py`,
`tests/test_report.py`, `tests/test_cli.py`) converts mechanically to `timing=ItemTiming
(wallClockMs=X, calls=(), withheldFor=None)` (or `timing=None` for a `None` `latencyMs`) — no
test's *assertions* changed, only fixture construction.

`RunResult` gains `attestationTripWire: Literal["compared","first-observation","unavailable"] |
None`, required with no default, and `latency: LatencyBlock | None = None` (not required — Step 0
adds no producer for it, so every pre-`runner.py` fixture is entitled to omit it). A new
`__post_init__` refuses construction (`AttestationTripWireMismatch`) unless `attestationTripWire
is None` is exactly `armKind == "deterministic"` (plan `:3145-3147`, §3.4.4, §3.4.5 point 3). The
one direct `RunResult(...)` fixture builder (`tests/conftest.py`'s `run()`) gains an
`attestation_trip_wire` parameter, forced to `None` alongside `call_surface` on a deterministic
arm exactly as that function already forces `call_surface` — so no existing caller can build an
inconsistent fixture by leaving the default in place.

`modelbench/convo.py`'s `ToolDispatchFailed` gains two keyword-only constructor parameters,
`completedTurns: tuple[TurnTrace, ...]` and `parsedArguments: Mapping[str, Any] | None`, per the
dispatch-failure note's §4(b) payload; `_drive_turn`'s one raise site now passes `observed` (the
turns already completed before this one, already in scope there) as `completedTurns` and the
already-parsed `arguments` dict as `parsedArguments`. The class docstring's "deliberately not
decided" paragraph is rewritten to state the note's §4(b)-(c) ruling as decided: the runner (not
`drive`) catches the raise, stores the conversation censored at that turn, and routes it to `-ml`
§4.1's `unrunnable` funnel category — and why there is still no sixth `TurnDisposition` member (a
dispatch raise is conversation-scoped, not turn-scoped). `drive()`'s own docstring is updated to
match; `drive` itself is otherwise unchanged — it still lets `ToolDispatchFailed` propagate, only
carrying more now.

**Tests:** `tests/test_results.py` adds direct-construction coverage for the new invariant
(`test_attestation_trip_wire_is_required_with_no_default`,
`test_attestation_trip_wire_must_be_none_iff_arm_kind_is_deterministic`, both directions plus the
two matching-pair controls) and for the new shapes (`ItemTiming`/`CallTiming` round-tripping
through JSON with a populated `calls` tuple, `unexplainedMs`'s sum-of-gaps and no-calls-at-all
cases, `LatencyBlock.callAttemptedCount`, and `RunResult.latency` round-tripping through storage).
`tests/test_convo.py` adds two cases for `ToolDispatchFailed`'s new fields: the existing single-
call-site test now also asserts `completedTurns == ()` and `parsedArguments == {"name": "Ghost"}`
on a first-call raise, and a new test drives a raise on turn 2 of a three-turn script to confirm
`completedTurns` carries exactly the one `TurnTrace` that finished cleanly before it.

**Mutation-tested both invariants named in the brief**, each in isolation, restored via `cp` from
a pre-mutation backup and `diff -q`-verified byte-identical, under `PYTHONDONTWRITEBYTECODE=1`:
disabling `RunResult.__post_init__`'s check (`if False:` in place of the real condition) reddened
`test_attestation_trip_wire_must_be_none_iff_arm_kind_is_deterministic` (`DID NOT RAISE
AttestationTripWireMismatch`); dropping the `+ self.latencyWithheldForNoResponse` term from
`LatencyBlock.callAttemptedCount` reddened
`test_latency_block_call_attempted_count_adds_no_response_withholds` (`5 == 7` failed).

**Concurrent with U114** (`roles.py`/`packs.py`'s sampling-contract route (iii), disjoint files,
dispatched in parallel): the full suite shows 11 pre-existing failures in `tests/test_report.py`,
all `PackConfigError: pairingKey[0] 'queryId' is not role 'embedder''s own analysis-unit field
'itemId'` — confirmed unrelated to this unit before touching anything (the offending fixture,
`_embedder_pack()`'s `pairingKey=("queryId",)`, is untouched here; the failure is
`check_sampling_contract`'s new route (iii) rejecting a `test_report.py` fixture U114 did not
update, entirely orthogonal to `timing`/`latencyMs`). Left as-is, per this unit's fence against
`packs.py`/`roles.py`.

**Verification:** `.venv/bin/python -m pytest -q` → **1043 passed, 11 failed (pre-existing, U114),
3 deselected**, 1057 collected total (starting-point baseline before either concurrent unit: 1030
passed, 3 deselected, 1033 total; this unit's own contribution is 8 new tests — 7 in
`tests/test_results.py`, 1 in `tests/test_convo.py` — all passing; the remaining delta is U114's).
`grep -rn 'latencyMs=' modelbench tests` → 29 hits before this unit (all `ItemResult(...,
latencyMs=X, ...)` constructor keywords, one `from_dict` read), 0 after (the read is gone;
`to_dict`'s own emission uses `"latencyMs":`, which the pattern does not match). `.venv/bin/ruff
check .` → `All checks passed!`. `modelbench/runner.py`, `modelbench/cli.py`, `modelbench/packs.py`
and `modelbench/roles.py` were not touched.

## 2026-09-10 — S2 U114: sampling-contract route (iii) — `roles.ANALYSIS_UNIT_FIELD_BY_ROLE` and `check_sampling_contract`'s role check

**What:** the gap U113's runner-spec synthesis flagged (`docs/plans/small-model-benchmarking-
runner-spec.md` §2.1) and plan §4 S2's own "Done when" names (`:5078-5082`, v1.25): `sampling`
contract route (iii), `pairingKey[0] == roles.analysis_unit_field(role)`, was missing from both
`roles.py` and `packs.py`. Routes (i) (`analysisUnit == pairingKey[0]`) and (ii) (the row-count
identity) are both satisfied by any self-consistent naming — P12-7's own shipped positive control
(`pairingKey: ["conversationId", ...]`, `analysisUnit: "conversationId"`, on a `tool-caller` pack)
passed both while naming the wrong field for its role; route (iii) is the only one that catches
that class of defect.

`modelbench/roles.py` gains `ANALYSIS_UNIT_FIELD_BY_ROLE` (a second closed-role-table column,
mirroring `UNIT_KIND_BY_ROLE`'s shape exactly) and `analysis_unit_field(role)`, both read from
plan §3.3's table verbatim:

| role | `analysis_unit_field` |
|---|---|
| `tool-caller` | `scriptId` |
| `guard-judge` | `itemId` |
| `nlq-generator` | `itemId` |
| `chat-responder` | `itemId` |
| `embedder` | `itemId` |

`modelbench/packs.py`'s `check_sampling_contract` now runs route (iii) right after route (i):
`pairingKey[0]` must equal `analysis_unit_field(ref.role)`, raising `PackConfigError` otherwise.
Wired at this one call site — `_ref_from_manifest_fields` already calls `check_sampling_contract`
for both `pack_ref_from_manifest` and `Pack.ref()`, so `validate_pack`'s existing
`_sampling_problems` → `pack.ref()` path picks up route (iii) with no further change.

**Tests, driven by execution, not by reading the constant back** (the plan's own emphasis,
`:5081`): `tests/test_roles.py` pins `set(ANALYSIS_UNIT_FIELD_BY_ROLE) == set(ROLES)` and the
table's exact values, and drives `analysis_unit_field` over all five roles plus the
`UnknownRole` refusal. `tests/test_packs.py` adds a `PackRef`-level sweep — `pytest.mark.
parametrize("role", ROLES)` — that calls `check_sampling_contract` directly with a `pairingKey[0]`
that is self-consistent with route (i) (`analysisUnit == pairingKey[0]`) but wrong for route
(iii): `"conversationId"` for `tool-caller` (P12-7's own historical fixture shape, named
explicitly in the plan and asserted as its own isolated case, not folded anonymously into the
sweep) and `"scriptId"` (a real row's value, borrowed from `tool-caller`) for the four item-level
roles. A parallel positive-control sweep asserts each role's own `analysis_unit_field(role)` as
`pairingKey[0]` returns `None` (no rejection).

**Mutation-tested route (iii) specifically:** disabled just its `if` condition (`if False and
...`), ran the suite scoped to files outside U115's concurrent `results.py`/`convo.py` edits — the
6 targeted tests (the 5-role sweep plus the isolated `tool-caller` case) reddened, all 818 others
stayed green. Restored via `cp` from a pre-mutation backup, `diff -q` confirmed byte-identical,
under `PYTHONDONTWRITEBYTECODE=1`.

**Fenced correctly around U115's concurrent work:** `modelbench/results.py` changed underneath
this unit mid-session (U115, disjoint files, dispatched in parallel per the coordination doc) —
`tests/test_results.py`, `tests/test_report.py` and `tests/test_cli.py` (all importing
`results.py`) were red for a reason confirmed unrelated to this change (`ItemResult.__init__()`
rejecting `latencyMs=`, U115's in-flight `timing=` conversion) before this unit touched anything,
and stayed exactly as red after. Neither `results.py` nor `convo.py` was touched here, per this
unit's fence.

**Verification:** `.venv/bin/python -m pytest -q tests/test_roles.py tests/test_packs.py` → **72
passed**. `.venv/bin/python -m pytest -q --ignore=tests/test_results.py --ignore=tests/test_
report.py --ignore=tests/test_cli.py` (the scope unaffected by U115's concurrent edit) → **824
passed, 3 deselected** (up from 813 before this unit, +11 — the new `test_packs.py` cases; the
`test_roles.py` cases are already inside that baseline). `.venv/bin/ruff check modelbench/roles.py
modelbench/packs.py tests/test_roles.py tests/test_packs.py` → `All checks passed!` (the full
`ruff check .` reports 9 pre-existing `E501`s, all in U115's in-flight `tests/test_results.py`,
none in this unit's files).

## 2026-09-10 — two `-m live` test-authoring defects fixed, both tests now pass for real

**What:** the prior entry's two authoring defects in the `-m live` suite, both test-file-only:

- `tests/test_hostinfo.py::test_live_loaded_catalog_entry_reveals_kv_cache_or_load_configuration`
  called `client.chat()` with the literal placeholder string `"<a model already loaded by the
  operator>"` as `model`, left over from when a human was expected to hand-edit it. Fixed by
  filtering a live `client.catalog()` read to chat-capable `type`s (`llm`/`vlm`) and taking the
  first match's `id`, with an assertion that fails clearly (not a raw 400) if the catalog holds no
  such entry.
- `tests/test_lmstudio.py::test_live_catalog_and_chat_stats_against_a_real_lm_studio` took
  `catalog()[0]` as the chat target with no `type` filter — fragile, since LM Studio sorts a
  just-loaded model first and that model can be non-chat-capable (e.g. `embeddings`-type). Fixed
  with the same filter-then-assert-then-take-first pattern as above.

Considered hardcoding a specific model id (the coordinating brief's suggested
`google/gemma-4-e2b`, confirmed present) instead, but the catalog-filter approach is more robust
to the local pack changing over time and reuses one pattern across both tests, so that's what both
fixes use. No production code (`modelbench/*.py`) changed — the existing `ModelInfo.type` field
already carries exactly this information; no new capability was needed.

**How:** `.venv/bin/python -m pytest -m live -v` from `model-bench/`, against the real, locally
reachable LM Studio (`http://localhost:1234`) — confirmed reachable via a direct
`GET /api/v0/models` first, catalog showing `google/gemma-4-e2b` (`vlm`, already `loaded`) sorted
first, so both fixed tests picked it as their chat target. All three `-m live` tests, run
together:

```
tests/test_hostinfo.py::test_live_loaded_catalog_entry_reveals_kv_cache_or_load_configuration PASSED
tests/test_hostinfo.py::test_live_loaded_context_length_on_a_loaded_embeddings_model PASSED
tests/test_lmstudio.py::test_live_catalog_and_chat_stats_against_a_real_lm_studio PASSED

3 passed, 1027 deselected in 11.89s
```

(The middle test was already verified live in the prior entry; included here to show the whole
`-m live` suite is green together, not just the two fixed tests in isolation.)

**Verification:** `.venv/bin/python -m pytest -q` (non-live) → **1027 passed, 3 deselected**,
unchanged from the prior baseline — expected, since both fixes are inside `@pytest.mark.live`
tests deselected by default. `.venv/bin/ruff check .` → `All checks passed!`.

## 2026-09-10 — `-m live` run against a real LM Studio: R-1 resolved (no), `loadedContextLength` confirmed on a loaded embeddings model

**What:** the stakeholder's 2026-09-10 authorization to trigger a model load (coordination doc,
"2026-09-10 — reversed") unblocked `tests/test_hostinfo.py`'s two `-m live` probes, written by U72
and never run. Both ran against a real, locally reachable LM Studio (`http://localhost:1234`) and
resolved their open questions:

- **R-1 (plan §4 S2, §6 R-1): no.** With a real model resident — checked on two, an
  `embeddings`-type load (`text-embedding-qwen3-embedding-0.6b`, via `client.embed()`) and
  separately a `vlm`/chat-type load (`google/gemma-4-e2b`, via `client.chat()`) — the re-read
  `GET /api/v0/models` entry exposes exactly the same field set both times: the plan's known ten
  (§2.5: `id`, `object`, `type`, `publisher`, `arch`, `compatibility_type`, `quantization`,
  `state`, `max_context_length`, `capabilities`) plus the already-known eleventh,
  `loaded_context_length` (§2.3). No KV-cache or load-configuration key appears on either loaded
  entry. Verbatim loaded entry (embeddings arm):
  ```json
  {"id": "text-embedding-qwen3-embedding-0.6b", "object": "model", "type": "embeddings",
   "publisher": "Qwen", "arch": "qwen3", "compatibility_type": "gguf", "quantization": "Q8_0",
   "state": "loaded", "max_context_length": 32768, "loaded_context_length": 2048,
   "capabilities": ["tool_use"]}
  ```
  and (chat/vlm arm):
  ```json
  {"id": "google/gemma-4-e2b", "object": "model", "type": "vlm", "publisher": "google",
   "arch": "gemma4", "compatibility_type": "gguf", "quantization": "Q4_K_M", "state": "loaded",
   "max_context_length": 131072, "loaded_context_length": 8192, "capabilities": ["tool_use"]}
  ```
  `kvCacheSetting` stays operator-attested; no `fingerprint.py`/`AGENTS.md` change follows from
  this finding (out of this unit's fences — routes to `architect` only if a future probe on a
  different LM Studio build disagrees).
- **§3.4.4a's open question: yes.** `loaded_context_length` does appear on a loaded *embeddings*
  model, not only a loaded chat model — `tests/test_hostinfo.py::test_live_loaded_context_length_on_a_loaded_embeddings_model`
  passed against the real server (`entries["text-embedding-qwen3-embedding-0.6b"].loaded_context_length == 2048`).
  §3.4.4a's placement of `loadedContextLength` outside the required set stands as a design choice
  now made with the embeddings case actually observed, not left open; no `fingerprint.py` change
  follows.

**How:** no code changed. `.venv/bin/python -m pytest -m live -v` from `model-bench/`, plus a
one-off `client.chat(model="google/gemma-4-e2b", ...)` script and raw `GET /api/v0/models` reads
run by hand outside the test file to get the second (chat-type) loaded entry. Both remaining `-m
live` tests failed on test-authoring defects unrelated to either finding above, left unfixed per
this unit's fences (reported to the dispatching session, not detailed here):
`test_live_loaded_catalog_entry_reveals_kv_cache_or_load_configuration` calls `client.chat()` with
the literal placeholder string `"<a model already loaded by the operator>"` as the `model`
argument rather than a real catalog id, so it never actually triggers a load and fails on LM
Studio's 400; `test_live_catalog_and_chat_stats_against_a_real_lm_studio` takes `catalog()[0]` as
the chat target, which — after the embed call above left the embeddings model resident and
sorted first by LM Studio — resolved to that non-chat-capable model and failed the same way. Three
live model interactions total: one embed call (the test's own), one chat call (manual,
`google/gemma-4-e2b`), one `GET /api/v0/models` catalog read.

**Verification:** `test_live_loaded_context_length_on_a_loaded_embeddings_model` — **passed**
against the real server. The other two `-m live` tests failed on the authoring defects above, not
on either finding; the manual reproduction here answers R-1's question on two independent
loaded-model observations (embeddings-type and chat-type), and a manual `client.chat()` call
against a real chat-capable model (`google/gemma-4-e2b`) separately confirmed usable `stats` with
`time_to_first_token` (0.093439 s), which is what the third test exists to check.

## 2026-09-10 — P18-1 (minor): trace-contract test-fixture coupling documentation

**What:** Impl review Pass 18 (P18-1) identified that the test fixture for the per-iteration
trace-contract check has a known coupling with the per-call check, via the fixture's read-count
behavior: deleting the per-call check alone does not redden the per-iteration test, because the
fixture's drop timing depends on when that read occurs. The production guard logic is unaffected
and all detectable defects are still caught. This is a test-fixture issue only, not a production
defect.

**How:** Narrowed the independence claim in `TraceContractViolated`'s docstring and `HISTORY.md`
entry for the P17-2 closure (this unit's trace-contract work) from "all three layers are
independent" to "per-iteration and per-turn layers are independent; per-call and per-iteration
have a known test-fixture coupling". No production code changed.

## 2026-09-10 — P17-5: the manifest→`PromptConfig` route, closing the gap Pass 17 left to this unit

**What:** the one Pass 17 finding explicitly left to `packs.py` — **P17-5**, that `validate_pack`
had no route for the `prompt` block, so a manifest declaring a bad `historyReplay` value or
violating `maxIterationsPerTurn`'s role-scoping rule (`-ml` §3.3 v1.26: required *iff*
`roles.MULTI_CALL_TURN_BY_ROLE[role]`, forbidden otherwise) failed only inside `convo.assemble` /
`convo.drive` at first use, never at validate time like every other pack defect. Three files:

- `modelbench/roles.py` — the missing third role-table column, `MULTI_CALL_TURN_BY_ROLE: Mapping[str,
  bool]`, `True` only for `tool-caller`; domain-pinned against `ROLES` the same way
  `UNIT_KIND_BY_ROLE` already is (P14-2's shape).
- `modelbench/packs.py` — `Pack.prompt_config() -> PromptConfig`, parsing the manifest's `prompt`
  block and raising `PackConfigError` on a bad `historyReplay` (outside `convo._HISTORY_REPLAY_MODES`)
  or a `maxIterationsPerTurn` that violates the role-scoping rule in either direction; wired into
  `validate_pack` as its fifth axis via `_prompt_problems`, which treats an absent `prompt` block as
  not its problem — same convention `_tool_module_problems` already uses for absent `tools.module`,
  which is what keeps every pre-existing fixture (none of which carry a `prompt` block but `valid`)
  green. `systemPrompt`/`toolSchemas` are carried through as the manifest's own declared paths, not
  resolved content — resolution is explicitly left to whichever caller eventually drives a real turn
  (the runner unit, not yet built), matching `convo.PromptConfig`'s own docstring.
- `tests/test_roles.py`, `tests/test_packs.py` — the domain pin plus `tool-caller`-only assertion for
  `MULTI_CALL_TURN_BY_ROLE`; three new fixture packs under `tests/fixtures/packs/` (a bad
  `historyReplay`, a `tool-caller` missing the required cap, a `guard-judge` carrying the forbidden
  one) each asserted by exact `validate_pack(pack) == [...]`; a happy-path test asserting the `valid`
  fixture's `prompt_config()` result; and a no-`prompt`-block fixture asserting no spurious problem.

**Every new guard was mutation-tested individually under `PYTHONDONTWRITEBYTECODE=1`, restored by
file copy and verified with `diff -q` after each — never batched.** All seven mutations (the
`MULTI_CALL_TURN_BY_ROLE` domain pin and its `tool-caller`-only value, the `historyReplay` membership
check, both directions of the `maxIterationsPerTurn` role-scoping check, two happy-path field
mappings, and the absent-`prompt`-block skip) were killed by the test written for that behavior; the
absent-block skip mutation additionally reddened eleven pre-existing fixture tests, confirming that
convention is what keeps them green today.

**Verification:** `.venv/bin/python -m pytest -q` from `model-bench/` → **1027 passed, 3 deselected**
(baseline before this unit: 1020 passed, 3 deselected). `.venv/bin/ruff check .` → `All checks
passed!`. `AGENTS.md`'s "Current state" section had its `packs.py` gap clause removed now that the
route is built; nothing else in the file was touched.

## 2026-09-10 — Impl review Pass 17 closed: the per-call trace contract, a name for a raising `dispatch`, and five coverage gaps whose names claimed the coverage

**What:** the eight Pass 17 findings assigned to this unit — **P17-1, P17-2, P17-4** (majors),
**P17-3's unconditional half**, and **P17-6, P17-7, P17-8, P17-9**. Two files:
`modelbench/convo.py` and `tests/test_convo.py`. `modelbench/tooling.py` is **unchanged and did not
need to change** — P17-2 named its *"must append **exactly one** `DispatchRecord` … `drive`
**enforces** that count"* as a claim exceeding its mechanism, and the closure widens the mechanism
to meet the claim rather than narrowing the prose. **P17-5 is not this unit's** (`packs.py`) and
neither is P17-3's record-versus-refuse design decision.

**Every change was mutation-tested one at a time under `PYTHONDONTWRITEBYTECODE=1`, restored by
file copy and verified with `diff -q`.** Baseline **1009 passed, 3 deselected**; final **1020
passed, 3 deselected**, `ruff check .` clean.

**1. The trace-count contract is now enforced per *call*, not per iteration (P17-2, major).**
`_drive_turn` reads `env.trace()` immediately before and after **each** `env.dispatch` and calls
`_check_trace_contract(..., dispatched=1, ...)`, naming the offending tool in the message. The
per-iteration aggregate it replaces is a strictly weaker claim than the prose it enforced: an
environment recording **0** entries for one call and **2** for the next balances it exactly, and
the turn then completed clean with the first call's replayed `tool` message carrying the second
call's return value — the substitution `TraceContractViolated`'s own docstring says is a checked
fact. `SkewEnvironment` is that environment; before the fix it drove the turn to completion with
no refusal. The per-iteration and per-turn re-takes of the **prefix** check are kept and are not
redundant: a `trace()` that mutates the environment on *read* moves the record **between** two
calls, where a per-call check reading its own `before` afterwards cannot see it, and on a turn that
dispatches nothing at all only the turn-level re-take is left. The **per-iteration and per-turn
layers are independently reachable** (each dies to its own deletion), while the **per-call and
per-iteration layers have a test-fixture coupling** (impl review Pass 18, P18-1): the
`ReadMutatingEnvironment` fixture's drop timing depends on whether the per-call check's extra
`trace()` reads occur, so deleting the per-call check does not cause the per-iteration test to
redden — a test-fixture issue only, the production guard logic is unaffected and all detectable
defects are still caught.

**2. A pack's `ToolEnvironment.dispatch` that raises has a name: `ToolDispatchFailed` (P17-3, the
unconditional half).** It propagated bare, and `drive`'s own docstring rules *"anything else"* to
be §3.6 clause (iv)'s **server went away** — so a `KeyError` out of a pack's `tools/sim.py` bought
a re-probe and an exit `3` under a **false cause**, a live server diagnosed as dead. The new class
sits beside `TraceContractViolated` as a sibling, carries `toolName`/`turnIndex` and preserves the
pack's own exception as `__cause__`. **The record-versus-refuse question is deliberately left
open** and is stated as open at the class: today a raising `dispatch` fails closed and abandons the
conversation, which is `TraceContractViolated`'s precedent, but §4 S2's replay contract has a
category for *"the dispatch raised"* and the trigger is partly model-chosen, so the decision is
owed to whoever writes `tools/sim.py` (S5). Settling it the other way is a
`try`/`except ToolDispatchFailed` at this exception's one raise site plus an
`_undispatchable_tool_content` reason — the naming is unaffected either way, and the current
behaviour has its own named test so the ruling is one visible test to change rather than an
implication to find.

**3. `TurnTrace.wallClockMs` is `float`, never `float | None` (P17-9).** `_drive_turn` always
computes a figure, including on a turn whose first call raised — correctly, since §4 S2 puts the
*withholding* on `ItemTiming.withheldFor`, which is the runner's. The optional annotation was the
only statement to the contrary and invited the runner to key withholding on a value that never
arrives.

**4. Five coverage gaps whose test names claimed the coverage.** Each was green under the defect
before and reddens under it now.

- **P17-1** — `_replay_structured` threads one dispatch cursor across a turn's iterations, and
  **no fixture had two tool-calling iterations**, so the only property a multi-iteration turn can
  exercise was the one no test reached, under a test named *"every iteration"*. That test is
  renamed to what its one-iteration fixture pins, and a new one replays a two-iteration turn with
  two **distinct** return values — distinct deliberately, since equal ones make the pairing
  assertion vacuous.
- **P17-4** — the propagate axis drove three classes that propagate under *any* narrowing
  (`RuntimeError` is `LMStudioError`'s **parent**), while the docstring named *"two further
  subclasses carrying no `.status`"* as the whole reason the catch is narrow and included
  **neither**. `LMStudioUnreachable` and `ToolCallingIneligible` are now driven — they are the only
  members that can see the widening — and the *"two further subclasses"* **reach** claim is bound
  to `lmstudio.py`'s own exception tree, so a third `.status`-less subclass added there reddens
  instead of quietly joining the set the docstring counts.
- **P17-6** — the turn-level trace check and `_iteration_exchange`'s `no-dispatch-record` branch
  were reachable by no test, and the per-iteration aggregate became hard to reach as a side effect
  of finding 1. Three fixtures, one per layer: `ReadMutatingEnvironment` on an iteration with a
  dispatch, the same environment on a turn that dispatches nothing, and a hand-built `TurnTrace`
  with more dispatchable calls than `DispatchRecord`s.
- **P17-7** — `plaintext`'s **role ownership** was asserted nowhere, and §3.3 (P13-8) makes it the
  axis on which `plaintext` and `structured-replies-only` differ *alone*. The existing four-mode
  test separates the modes by JSON inequality, which is blind to *which* role differs. A new test
  pins the contrast: the same two replies, `user`-owned in one mode and `assistant`-owned in the
  other.
- **P17-8** — the wall-clock fixture modelled the calls and the dispatches but not `assemble`,
  which §5 test 10b names as half of the difference it measures, so the exact `==` could not see
  the stopwatch starting one statement late. `convo.assemble` is now on the fixture's stub clock.

**No residual is deferred by choice, and none is blocked on unbuilt work.** The one thing this unit
does **not** decide is P17-3's record-versus-refuse question, which is a design decision routed to
a named owner (S5's `tools/sim.py`), not a deferral: the behaviour that ships is stated in code,
pinned by its own test, and reversible by that owner in one call site.

## 2026-09-10 — `convo.drive` becomes a bounded per-turn loop, and the replay stops being textbook

**What:** the `convo.py` rework plan v1.29 §3.8.4 and §4 S2 specify. Two rulings land together
because neither is buildable without the other: a turn is a **bounded iteration loop** rather than
one call, and a prior turn is replayed from **what the model actually produced this run**, never
from the script's `expect`. `historyReplay`'s fourth value, `structured-replies-only`, is built in
the same pass — the shipped module declared three modes and refused everything else while §3.3 has
declared four since v1.26, and `tool-caller-shop-assistant` declares the fourth.

**The two runnable gates, measured before and after.**
`grep -c 'turn\.expect\|expect\.get\|expect = ' modelbench/convo.py` → **9 → 0**;
`Turn.expect`'s field declaration and its docstring are the only survivors, and the docstring now
says outright that `scoring/toolcalls.py` is its only reader. The `historyReplay` block's own pins,
scoped to `modelbench tests`: `'"structured", "plaintext", "none"'` **3 → 0**,
`'"structured" | "plaintext" | "none"'` **1 → 0**, and the positive `'structured-replies-only'`
**0 → 12** against a stated lower bound of 2. The disposition block's four residuals and its
positive pin were re-run and were **already at their target** at `3c795cc` — the precursor round
that widened the vocabulary had swept them — so nothing there was re-authored.

**1. `assemble(turn_index, script, observed, cfg)`.** `script` supplies user text only; `observed`
is this run's own `TurnTrace`s for turns `0 .. turn_index-1`. Two preconditions raise, and the
second is the ruling's structural guarantee: `len(observed) == turn_index`, so turn *n* cannot be
assembled from anything but *n* observations and no caller can reintroduce a textbook prefix.
`_expected_exchange` and `_flatten_turn` — the module's only readers of `expect` — are gone;
`structured` now replays each tool-calling iteration's own assistant message with its `tool_calls`
verbatim plus one `tool` message per entry carrying the environment's real `returnValue`, and the
terminating response is replayed **once**, as the trailing reply, rather than a second time as an
iteration. `historyTurns` windows the `(script[i], observed[i])` **pair**: the window now spans two
sequences and a drift between them is silent, so the pin asserts the alternation
`u3, a3, u4, a4` directly rather than through absence.

**2. A reply-less prior turn is replayed in every mode and never omitted** — `content: ""` in the
two reply-text modes, its iterations with no trailing assistant message under `structured`.
Omitting it shortens the visible history, which is the covariate this pack measures against. The
temptation the plan names is substituting `expect` for the missing reply; a fixture whose oracle
carries a string appearing nowhere else asserts that string is absent from what is assembled.

**3. `drive`'s loop, and it never abandons a script.** `LMStudioCallTimeout` and
`LMStudioCallFailed` are caught, recorded as the turn's disposition, and the script continues —
`-ml` §4.1's hard rule, and the blocker the plan gate's Pass 15 raised, where one failed turn
destroyed the whole record. The catch is narrowed to those two classes because `LMStudioError` has
two further subclasses carrying no `.status`, so a handler reading `exc.status` off the base would
raise `AttributeError` *inside the handler*; a coverage probe drives both sides of that axis,
including an exception class the module has never seen. The exception path is tested **before** the
cap, so a call raising at the cap-th iteration is never `cap-hit`.

**4. Four findings the previous gate routed here, each closed rather than carried.**
`_parse_tool_arguments` returns `dict | None`: an unparseable, array-shaped or scalar `arguments`
value is **no longer dispatched**, so `{}` recovers its single meaning and a harness-side parse
failure no longer mutates FR-10 ground-truth state or feeds a return value back into the model's
next iteration. `ConversationTrace` carries `shape` and `replicate` beside `scriptId`, without
which §3.3's `pairingKey` is not derivable from a trace. `PromptConfig` gained
`maxIterationsPerTurn` and an Appendix-A field-order pin, in the shape of the `DispatchRecord` one
that already worked — these names are manifest keys, so a drift is an unreadable `pack.json`, and
`tests/fixtures/packs/valid/pack.json`, the only fixture carrying a `prompt` block, gains the
field §3.3 requires of every `tool-caller` manifest (inert until `validate_pack` grows its
`prompt` route, and named by §3.3 as this unit's to land rather than to be discovered later).
`messagesSent` is defined as the in-turn working list **as sent at the turn's last model call**,
which is total over all five dispositions and has iteration 1's list as a prefix; both halves are
asserted rather than documented.

**5. One divergence from a finding's literal prescription, and it is a strengthening.** Pass 15
§5's trace guard was specified as *refuse when `len(after) < trace_before`*. Executed, that check
cannot see the fixture the finding itself describes: an environment that clears its trace inside
`dispatch` and re-appends leaves the length exactly where it was, and on the conversation's very
first call there is no prefix to have shrunk at all. So `drive` takes **two** checks — the entries
already reported are still there in order, and the trace grew by exactly the number of calls
dispatched — raising `TraceContractViolated`. The second is what fires on the finding's own
fixture, and it also makes `assemble`'s positional pairing of `tool_calls` onto `DispatchRecord`s a
checked fact rather than an assumption. Both are **pack** defects and pack defects fail closed
(§3.3), unlike the model failures `-ml` §4.1 requires to be driven past. `tooling.py`'s Protocol
docstrings state the enforced contract, and its `rawArguments` gloss — which claimed to hold *what
the model's tool call actually carried* while the harness was substituting `{}` — is now true of
the mechanism.

**Verification:** `.venv/bin/python -m pytest -q` from `model-bench/` → **1009 passed, 3
deselected** (0 failed / 0 skipped; up from 941 passed / 3 deselected at `3c795cc`, `test_convo.py`
going 32 → 100 tests). `.venv/bin/python -m ruff check .` → `All checks passed!`. **26 mutants,
one at a time, each restored by file copy and verified with `diff -q`, all run with
`PYTHONDONTWRITEBYTECODE=1` — all 26 killed.** The one that initially **survived** is the one worth
recording: replacing the turn's `wallClockMs` with the last call's left the suite green, because
the plan's own prescribed assertion (`turn.wallClockMs >= Σᵢ wallClockMsᵢ`) is satisfied by any
single call's figure when the fixture's per-call figures are zero — which every stub `ChatResult`
in this file had been. The fixture now runs on a **stub clock** the stubs advance themselves, with
three distinct non-zero per-call figures and a per-dispatch tick, so the turn's figure is an exact
arithmetic fact; the last-call, first-call and sum-of-calls substitutions all redden.

**Two residuals, both blocked on unbuilt work and neither deferred.** The disposition probe's third
leg is still S5's, held by the live tripwire that reddens when `modelbench/scoring/` appears. And
the `prompt`-block half of `validate_pack` — the manifest→`PromptConfig` constructor, the
`historyReplay` and `maxIterationsPerTurn` refusals — is `packs.py`'s and belongs to the unit that
owns the manifest surface, not to this one; until it exists a pack declaring a bad `historyReplay`
still fails at run time rather than at validation.

## 2026-09-10 — `turnDisposition` widened to five, and the prose the widen falsified

**What:** plan v1.28 (`docs/plans/small-model-benchmarking.md` §3.8.4, §4 S2) splits `timed-out`
out of `no-response`, taking the turn-disposition vocabulary from four members to five. Both
halves of that landed here: the widen itself, and the shipped comments and docstrings the widen
made false.

**1. The widen, and the probe reddening on it is the whole return on the precursor unit.**
`convo.TurnDisposition`, `convo.TURN_DISPOSITIONS` and `tests/test_convo.py`'s
`_DISPOSITIONS_PER_PLAN_3_8_4` each gained `"timed-out"`. The transcript moved first: legs 1 and 2
of §4 S2's probe both went red naming the missing member, then each module declaration was widened
separately — leg 1 green with leg 2 still red — which is the independence the two written-out
declarations exist to have. Ten mutants, one at a time, restored by file copy: shrink, widen and
member-rename on each of the three declarations (a set takes no move-a-value-to-another-key
mutation, having no keys; the rename is its analogue), plus `TURN_DISPOSITIONS` demoted to a plain
`set`. All killed, and the two module mutations kill only their own leg.

**2. The scoring ruling reversed, so five doc blocks were rewritten whole.** Under `-ml` §4.3
rule 4 only a **timeout** scores `fail`; `no-response` and `server-rejected` are channel failures
the harness cannot attribute and are both `unrunnable`. `LMStudioCallFailed`'s class docstring
stated the old rule in three places and cited a four-row table; `convo.TurnDisposition`'s block
claimed to be *the only home of the mapping from mechanism to what scores it* — which was the plan
gate's P14-1, two homes for one mapping — and folded `LMStudioCallTimeout` into `no-response`;
`TURN_DISPOSITIONS`' block carried P14-3's over-claim (a probe authored beside its transcript
cannot redden in round 1, so what the precursor buys is **cross-unit** protection, not
same-unit); the test transcript's gloss and `tests/test_lmstudio.py`'s `status`-partition banner
carried the same two. Each block was rewritten whole rather than edited token by token, and the
mechanism vocabulary now states no scored outcome at all — it cites rule 4 for that.

**3. Why whole blocks, and the defect class it comes from.** The review that raised this proposed
pinning the sole-ownership claim with `grep -rnF 'only home'`. That command **matched nothing**:
the sentence wrapped across `convo.py:79/80` at exactly *"…which is the only / home of the
mapping…"*, and `grep` is line-based. A reviewer's pin was silently vacuous — the same class of
defect as the prose it was aimed at. The plan's replacement pins are five **symbols** whose blocks
are rewritten, plus five residual counts scoped to `modelbench tests`, all of which moved as
specified: ``scores `fail`|score it `fail``` 3 → 0, `four-row` 5 → 0, `home of the mapping` 1 → 0,
`no-response.*LMStudioCallTimeout` 2 → 0, and `"timed-out"` 0 → 3. Each of the four zero-target
commands was mutation-tested against the **rewritten** text by reintroducing the falsified claim
into the new block, and each fired.

**Not in scope, and why.** The `convo` module docstring's textbook-replay claim (v1.25 forbids
replaying a script's `expect`) is false at this tree and is assigned by §4 S2 to the `drive`
rework unit, which owns the body that would make it true. The disposition probe's **third leg** —
the set S5's scorer branches on — is still owed and still blocked on unbuilt work, held by
`test_the_third_leg_of_the_disposition_probe_is_still_owed_by_s5`, which was left untouched.

Suite: 941 passed, 3 deselected. `ruff check .` clean.

## 2026-09-10 — The constant-pin convention's directional half, restored and applied

**What:** impl review Pass 16 (`docs/reviews/small-model-benchmarking-impl.md`), whose subject is
Pass 14's own audit. Pass 15 fixed a tautology in `AGENTS.md`'s "A guard's reach lives in an
asserted constant" and deleted the directional clause in the same edit; the forms then said what
a good pin looks like without saying how to tell whether the pin in front of you is one. Nine of
the fourteen constants Pass 14 recorded as held were named by no test at all — they reddened on a
shrink only because some other fixture happened to use the deleted member — and **seven of twenty
went green on a widen**. Both halves are now in the line, and all seven are pinned in both
directions.

**1. `AGENTS.md`, the convention line, third revision.** Pass 16 §4's proposal was applied with
two changes the pins themselves forced. It says *"the two standing exceptions"*; after §P16-3's
paired-declaration bindings land there is exactly **one** (`packs._STDLIB_MODULE_NAMES`, measured
below) — the "inert widen" class was an artefact of the missing binding, not an exception to the
rule. And it earns a third clause: *both directions is necessary and not sufficient for a
**table***. Three mutations landed here that no key-set assertion can see — two
`_ROW_COUNT_IDENTITY_KEY_HINTS` hints swapped between keys, two `INDEX_COLUMNS` reordered, one
`_EXEMPT_CELLS` member made both exercised and exempt — so keys and contents are two pins, and a
value the test reads back out of the table under test is asserted against itself.

**2. P16-1 — `Basis` was declared three times with nothing binding any pair.** `stats.py:67`,
`results.py:39` and `report._BASIS_STRENGTH`. Python does not enforce a `Literal` at runtime, so
**deleting `"measured"` from `stats.Basis` alone left the suite at 920 passed** (reproduced before
changing anything). `basis` is what `-ml` §3.4 Rule 4 turns on — which instrument may decide a
verdict — and a value absent from `_BASIS_STRENGTH` is a `KeyError` in the report path. Closed two
ways, because the two duplications differ in kind: `results.Basis` is now an **import** of
`stats.Basis` (one home, on an import edge that already carries `percentile` for the same reason),
and `_BASIS_STRENGTH` is bound to a literal transcribed from `-ml` §7.1 — never to the other
declaration, since two sets authored in one unit agree by construction. The ranking's *values* are
pinned too, through `min(..., key=...)` rather than against the integers. `CallSurface` had the
same shape (`fingerprint.py`, `lmstudio.py`, and `CALL_SURFACES` derived from a third source), as
did `ArmKind`; those are bound by test rather than collapsed, because `lmstudio.py` deliberately
imports nothing from `modelbench` and the two `Literal`s are static declarations of a set the
other module derives at runtime.

**3. P16-2/P16-3 — the seven widen-green constants, each pinned in both directions.**
`INDEX_COLUMNS` against a hand-transcribed fourteen **in order** (a positional CSV reader is
broken by a reordering a set comparison calls identical) plus the header actually written to disk
and the keys `_index_row` actually emits — `csv.DictWriter` fills a declared-but-unemitted column
with a blank, so the widen was silent. `_METRIC_DECODERS` against the tags `_metric_to_dict`
emits, with a per-kind round trip that kills a decoder wired to the wrong constructor.
`_ROW_COUNT_IDENTITY_KEY_HINTS` against `ROW_COUNT_IDENTITY_KEYS`, plus each hint against the type
`_row_count_identity_field_valid` actually enforces. `_DISCRIMINATORS` against `Fingerprint`'s own
non-`fields` attributes. `_RESIDENCY_FIELDS` against the schema's `residentModelsAt*` fields.
`_NO_VERDICT_REASON` against the causes `_comparison_pair` returns over an exhaustive grid of the
two dimensions it branches on.

**4. P16-4 — the grid-coverage assertion is a partition, not a subtraction.** Pass 16's named fix
(`all_cells == exercised | _EXEMPT_CELLS`) is a trade rather than a repair: measured, it closes
the undeclared-cell direction and **opens** another, going green on a cell that is exercised and
exempt at once, which the subtraction refused (2 kills down to 1). Both lines are now asserted —
union *and* disjointness.

**5. One rendering nit, from a unit that reported rather than fixed it.** An `unparseable` record
carries `problems=[]` by construction, and the detail fell back to `record.reason`, which is
already the first half of the line: every such record printed as `unparseable: unparseable`. The
colon now introduces the fields that failed, and with none to introduce the reason stands alone.
Pinned from both sides, so removing the suffix outright reddens too.

**6. P16-5 withdrawn — the last standing exception was not one, and it was the allowlist.** Pass
16 ruled `packs._STDLIB_MODULE_NAMES` a constant the convention cannot govern: form (i) has no
independent declaration to bind to, since binding it to `sys.stdlib_module_names` is the
definition rather than a check, and form (ii) would be ~300 assertions. The first half is false,
and measuring it is what showed that. **Binding a derived constant to its own source is circular
against a *re-derivation* and not against an *augmentation*** — and the augmentation is the whole
hazard: `frozenset(sys.stdlib_module_names) | {"requests"}` widens what every pack module is
permitted to import and left the suite at 940 passed, refusing `not_an_allowed_package` in the
two behavioural fixture tests exactly as before. One equality in `tests/test_packs.py` kills it.
Measured on the delivered pin: the augmentation → 1 failed, that test alone; removing `"json"` →
3 failed; and the re-derivation as an equivalent comprehension → 941 passed, green, which is
correct and is the pin's stated bound. Of the two forms only the ~300-assertion half survives,
and the convention is an *or*, so there is **no** standing exception — `AGENTS.md`'s clause says
that positively rather than preserving an exemption for symmetry.

Recorded because the shape recurs: the exception clause of a rule about stated reach exceeding
implemented reach had itself stated a reach (*"nothing independent to bind to"*) that its own
reasoning did not support, and it was found the way all thirteen before it were — by running it,
not by reading it.

**Verification.** Every pin mutated in both directions, one at a time, restored by file copy and
`diff -q` after each: 39 distinct mutations, each re-run after a test it targets changed. Every
previously-green widen now reddens, each named by the test written for it: `_DISCRIMINATORS`
and `_RESIDENCY_FIELDS` widened by a bogus name are killed by exactly one test each, the new
one. The coordinator's own mutation (`stats.Basis` − `"measured"`) reddens
`test_the_basis_literal_is_exactly_the_ml_notes_vocabulary`. Suite `941 passed, 3 deselected`
(from 920), `ruff check .` clean, `AGENTS.md` 2 303 words with no line over 700 characters.

## 2026-09-09 — S2 precursor to the `drive` loop rework: `LMStudioCallFailed.status` and `convo.TURN_DISPOSITIONS`

**What:** the two pieces plan v1.26 §3.8.4/§4 S2 require to exist *before* the unit that rewrites
`convo.drive` into a bounded per-turn iteration loop — the adapter distinction that loop
partitions on, and the disposition set it will consume. Deliberately its own round: a coverage
probe authored in the step that introduces the values it must reject contains those values from
birth, can never redden, and leaves the guard's name standing over no guard.

**1. `LMStudioCallFailed` gains a required keyword-only `status: int | None`.** Verified at
`2ec3026` before changing anything: the class had twelve raise sites and no way to tell them
apart. An HTTP 400 (`-ml` §4.1's own example, and what cost §8.4 six `gpt-oss-20b` conversations)
and a dropped connection both raised a bare `LMStudioCallFailed` — `_raw_post` rung 1 and rung 3,
differing only in a message string — and the class docstring folded "a non-2xx status, a dropped
connection, or an unparseable body" into one disposition. §3.8.4's four-row table splits exactly
that: `server-rejected` (the server answered and refused) scores `-ml` §4.1's `unrunnable`,
`no-response` (the call never completed) scores §3.6's `fail`. Every raise site was audited and
now states its side:

- **A status** — `catalog()`'s non-2xx check, and `_raw_post` rung 1 (`exc.code`, read from the
  response, never hardcoded).
- **`None`** — rungs 3/4/5 (URLError, `http.client.HTTPException`, bare `OSError`), and **every
  2xx whose body is unusable**: unparseable JSON, a missing `choices`/`data` list, a malformed
  catalog entry. The server did not *refuse* — §3.8.4 files a body error under `no-response` — so
  the field is the refusal status, not "the last status seen". Reporting `200` there would make
  `status is not None` read as a refusal that never happened, which is the laundering direction
  P13-1 exists to block.
- **Required, not defaulted.** A default would let a new raise site inherit `None` — i.e. score
  `fail` — without anyone deciding. The keyword forces the decision; the completeness pin below
  forces it to be tested.

**2. `convo.TurnDisposition` and `convo.TURN_DISPOSITIONS`,** the four mechanisms as a `Literal`
and as `frozenset[str]`, written out **separately rather than one derived from the other**: the
probe binds each to a constant transcribed by hand from §3.8.4's table, so a member added to
either declaration alone reddens. Nothing consumes them yet, which is the point — the rework unit
consumes a constant it did not write.

**The probe is two of §4 S2's three legs, and the third is named rather than omitted.** The S5
scorer's branch set has no declaration to bind while `modelbench/scoring/` does not exist
(**blocked on unbuilt work**, not deferred). `test_the_third_leg_of_the_disposition_probe_is_still_owed_by_s5`
is the tripwire: it asserts that package's absence and fails, with instructions, the moment it
appears. Its own limit is written into its docstring — a scorer landing outside that package
would not fire it, and the leg would still be owed.

**Verification.** Baseline before the change: `893 passed, 3 deselected`. After: `920 passed, 3
deselected` (three other units were live in the tree concurrently and added the difference beyond
this unit's 19 new tests); `ruff check .` clean. Mutation evidence, each restored by file copy and
`diff -q`-verified: `_raw_post` rung 1 `status=exc.code` → `None` killed both rung-1 scenarios;
a 2xx-body-shape site `None` → `200` killed its scenario; `catalog()`'s `status=status` → `None`
killed its scenario; a thirteenth raise site added with no scenario killed the completeness pin;
a fifth member added to the `Literal` alone, and to `TURN_DISPOSITIONS` alone, each killed exactly
its own leg, as did removing `server-rejected` from each alone; re-deriving `TURN_DISPOSITIONS`
from `get_args(TurnDisposition)` as a mutable `set` killed the immutability pin; creating
`modelbench/scoring/` killed the tripwire.

**One test-harness fact worth recording:** `urllib.error.HTTPError` is **single-use** as a stub
outcome. `_raw_post` calls `exc.close()` on it, so raising one prebuilt instance a second time
fails inside `tempfile` with `ValueError: I/O operation on closed file` rather than in the
adapter — invisible until a test re-runs a scenario, which the completeness pin does. The new
scenario table therefore takes outcome *factories*, matching `_CONNECT_KINDS`/`_READ_KINDS`.

**Not done, deliberately:** no iteration loop, no `turnDisposition` field on `TurnTrace`, no
change to `drive`, and no edit to `convo.py`'s module docstring (which still describes v1.24's
one-call-per-turn design) — all of that is the rework unit's, and touching it here would
reintroduce exactly the ordering problem this round exists to prevent.

## 2026-09-09 — S2 U83: `load_history` classifies a record's envelope before decoding its body (P14-4's live consequence)

**What:** Pass 14's **P14-4** named a consequence of `_AGGREGATE_BY_KIND` losing a kind — a
structurally valid record quarantined as `"unparseable"`, indistinguishable from a corrupt file.
U82 pinned the constant and argued the consequence was reachable only through the drift the pin
now blocks. **That premise was wrong, and the coordinator's reproduction settled it: the
consequence needs no drift at all.**

**The defect was an ordering, not a table.** `load_history` decoded the whole record with
`RunResult.from_dict` inside the same `try` that read the file, so *any* body this build cannot
decode landed on `unparseable` with `runId=None` and `benchSchemaVersion=None` — before the pack
filter or the schema branch below it ever ran. Both of those branches were therefore reachable
only for a future record whose *body shape had not changed*, which is precisely the future record
that would not have needed the version bump: `BENCH_SCHEMA_VERSION`'s own docstring says the
integer increments when the on-disk shape changes in a way a reader must branch on. The guard was
tested only on the case it was not written for.

Two measured consequences, neither involving any drift of `_AGGREGATE_BY_KIND` (a well-formed
record, a complete fingerprint block, the constant exactly as shipped):

- **The comment above the schema branch made a claim that was false of the very record it
  describes.** Review m-1 fixed "another pack's future-schema record is surfaced as this pack's
  exclusion" on the grounds that "its `packId` is right there and readable". It was just as
  readable in a record whose body had also changed — and that record was still reported in a
  `tool-caller` comparison's `INVALID RESULTS EXCLUDED` block.
- **A record left by a later build was reported as file corruption.** "Upgrade the tool" and
  "restore the file" are different operator actions, and the AC-2 line printed the second for
  both, under a filename rather than a run id.

Reached through two independent decoders — an unknown aggregate kind (`_AGGREGATE_BY_KIND`'s
`KeyError`) and an unknown metric tag (`_metric_from_dict`'s `ValueError`) — so P14-4 named one
instance of a general defect. The `from_dict` and `_metric_from_dict` comments that documented
`unparseable` as the intended landing place were half right: raising is theirs to decide, the
diagnosis is the reader's, and both now say so.

**The fix:** the **envelope** — `runId`, and the fingerprint's `packId` and `benchSchemaVersion` —
is parsed first and separately, through `Fingerprint.from_dict` rather than off the raw dict so
that how a stored fingerprint is read keeps one home. The pack filter and the schema branch then
run on it, and the body is decoded afterwards. Three outcomes, all behavioural:

| the file | before | after |
|---|---|---|
| envelope unreadable (truncated, no `runId`, no `fingerprint` block) | `unparseable`, `None`/`None` | unchanged — still surfaced even under another pack's id (m-1's boundary) |
| envelope readable, declares another pack | `unparseable`, this pack's finding | dropped, as m-1 already promised for its readable sibling |
| envelope readable, declares a schema this build does not know | `unparseable`, `None`/`None` | `unknown_schema`, naming the run and the schema |
| envelope readable, declares a **known** schema, body still will not decode | `unparseable`, `None`/`None` | `unparseable`, **named** — the record claims a contract it does not meet |

**`InvalidRecord.reason` gained no fourth value, though the unit had authority to add one.** The
last row is what makes that right: a record claiming a schema this build knows and still failing
to decode is damaged or non-conforming, not from the future, and answering `unknown_schema` would
launder it into a tooling-version excuse. Reusing the three existing values also makes the two
future-schema populations — body decodable or not — indistinguishable in the report, which is the
point: whether this build happens to choke on the body is an accident of which fields the bump
changed and must not change the diagnosis. `report.py` needed no change; its block reads `reason`
generically, verified by rendering both new cases end to end.

**Verification:** eight new tests in `tests/test_results.py`, six red before the change (the two
envelope-reach tests are green either way by design — they guard against the fix *widening* the
silent drop). Five mutations, each applied alone and restored by file copy with `diff -q`:
restoring the pre-fix ordering kills 6; answering `unknown_schema` for a known-schema decode
failure kills 4 (including two tests that predate this unit, which is independent evidence the
classification matches what the suite already demanded); a tolerant envelope read
(`raw.get(...)`) kills 2; dropping `runId` or `benchSchemaVersion` from the named-`unparseable`
record kills 2 each. `.venv/bin/python -m pytest -q` from `model-bench/` → **920 passed, 3
deselected**, exit 0 — `tests/test_results.py` alone went 68 → 76; the rest of the rise over the
893 this unit started from is three concurrent units' work landing in the same tree, not this
one's. `.venv/bin/python -m ruff check .` → `All checks passed!`.

**No new constant, set, table or `Literal`** — so `AGENTS.md`'s guard-reach convention has nothing
to bind here; the reach that *is* new (which fields must be legible to classify a record) is
pinned behaviourally, one consequence per member, by
`test_a_record_whose_envelope_is_unreadable_is_never_pack_filtered`.

## 2026-09-09 — S2 U82: closing impl review Pass 14's five guard-reach gaps (P14-1..5), and Pass 15's correction to the convention that closes them

**What:** `docs/reviews/small-model-benchmarking-impl.md` Pass 14 audited every guard-reach
constant in the S2 diff and found five (of 25 judged) where a one-member shrink left the full
suite green — the pre-committed threshold (Pass 13 §6) for treating the recurrence as a
convention gap rather than five separate defects. Added one line to `AGENTS.md`'s Conventions
list and closed all five with a pin test plus, where the review found a worse consequence, the
production fix.

**The convention line was itself defective as first written, and a concurrent Pass 15 caught it
by execution before this unit finished.** The first wording ("asserts the computed set equals the
constant") is a tautology whenever the guard it describes is a pure membership test: driving the
guard over its own constant's members and asserting it accepts them is true *for any value* of
the constant, so it reddens on nothing. `AGENTS.md` now reads "binds it to the other declaration
of the same set, or asserts a distinct behavioural consequence for every member — never merely
that the guard accepts what the guard's own constant contains." Every pin below was re-verified
against this: P14-2/3/4/5 bind two independently-written declarations (form (i)) and, for P14-3
and P14-4, also assert a distinct rendered/round-tripped output per member (form (ii)). P14-1's
first-written pin escaped the tautology only by accident — it distinguishes a *typed*
`LMStudioCallFailed` from a bare `KeyError`, which happens to redden on the review's named
*shrink* — but a companion mutation found during this unit (marking an already-`.get()`-optional
field, e.g. `"capabilities"`, required in the constant) stayed green under that pin alone, because
both sides of the comparison were computed from the same mutated constant. Added a second, form
(i) test (`test_required_model_info_keys_partitions_every_catalog_field`) that binds
`_REQUIRED_MODEL_INFO_KEYS` to an independently-named partition of the full field set, closing
that direction too.

**P14-1** (`modelbench/lmstudio.py` `_REQUIRED_MODEL_INFO_KEYS`) — pin only, no production change:
the bare `KeyError` the review found escaping `catalog()` is reachable only through the constant
drifting out of sync with the `raw[key]` accesses `_model_info_from_raw` makes, which the two new
tests in `tests/test_lmstudio.py` now hold in both directions; no currently-reachable malformed
catalog response can still trigger it. **P14-2** (`modelbench/roles.py` `ROLES` /
`UNIT_KIND_BY_ROLE`) — new `tests/test_roles.py`, binding the two declarations directly.
**P14-3** (`modelbench/report.py` `_SAMPLE_NOUN`) — bound to `UNIT_KIND_BY_ROLE.values()` and
given a render assertion per noun, including `"query"` (embedder), which no existing fixture had
exercised. **P14-5** (`tests/test_lmstudio.py` §4B grid) — the three per-phase outcome registries
bound to `_PHASE_KINDS` minus `_EXEMPT_CELLS`. **P14-6** (nit, decided rather than merely pinned):
`("connect", "unparseable_body")` held the exact reasoning U79 used to drop
`("read", "non_2xx")` from `_PHASE_KINDS["read"]` rather than exempt it — no body exists before a
response object does, in either case — so it gets the same treatment now: removed from
`_PHASE_KINDS["connect"]`'s domain, and `_EXEMPT_CELLS` is `frozenset()`, kept as a named constant
(not deleted) so a future genuinely-unreachable cell is still a recorded decision.

**P14-4** (`modelbench/results.py` `_AGGREGATE_BY_KIND`) — pinned in `tests/test_results.py`
(domain-equality against each `Aggregates` subclass's own `kind` default, plus a real
`store`/`load_history` round trip per kind). **The worse consequence the review named — a
genuinely valid record of a dropped kind is quarantined as `"unparseable"`, indistinguishable
from a corrupt file — is pinned but not fixed**: closing it for real would add a value to
`InvalidRecord.reason`'s closed `Literal`, a shape `report.py`'s exclusion block already renders
downstream. That reaches a stored-record shape this unit's brief named as a stop-and-ask
boundary; routed back to the coordinator rather than decided here.

**Tests:** `tests/test_roles.py` (new), plus additions to `tests/test_lmstudio.py`,
`tests/test_report.py`, `tests/test_results.py`. Every pin verified by the mutation discipline the
review itself used — the named one-member shrink (and, for P14-1, the widen this unit found)
applied by file copy, run isolated, confirmed red, restored byte-identical, confirmed green again.

## 2026-09-09 — S2 U78: `modelbench/tooling.py` and `modelbench/convo.py`, new

**What:** the two S2 units named in `docs/plans/small-model-benchmarking.md` §4 S2's code sketch
that neither `packs.py`/`lmstudio.py`/`hostinfo.py` nor any concurrent unit this wave owns:
`modelbench.tooling` (the pack-importable simulated-tool plugin seam — `ToolEnvironment`,
`DispatchRecord`) and `modelbench.convo` (prompt assembly and the scripted-conversation driver —
`PromptConfig`, `Turn`, `Conversation`, `assemble`, `drive`, `ConversationTrace`, `TurnTrace`). Both
new. Added: `modelbench/tooling.py`, `modelbench/convo.py`, `tests/test_tooling.py`,
`tests/test_convo.py`. No existing file touched — `packs.py`, `lmstudio.py`, `hostinfo.py`,
`cli.py`, `conftest.py` and every existing test file are untouched, per this unit's fences.

**Scope carved out, not built here, because it is the runner unit's:** the timing discipline and
`LatencyBlock` (§4 S2's nine invariants), and the tool-caller scoring rules ((i)-(vi) including
(iv-a/b/c)). `drive()` therefore issues exactly one LLM call per scripted turn, never catches an
LLM error, and has no timeout parameter to size one with — all stated in its own docstring rather
than left to be discovered by a caller.

**The one design decision worth a reviewer's attention, because nothing in the plan states it in so
many words: `assemble`'s replay of a prior turn (`historyReplay="structured"`/`"plaintext"`) is
built from that turn's *scripted* `expect` block, never from what the model under test actually
said or did this run.** Reasoned from three converging clues rather than assumed: (1) §3.8.4's "the
harness never carries hidden state between turns beyond what the configuration says it carries"
reads most literally as `assemble` being a pure function of `(turn_index, history, cfg)`; (2) the
determinism probe (§3.8.4) re-runs a script and diffs outcome vectors turn by turn, which is only a
clean comparison if every run sees an identical context regardless of what the model actually
produced; (3) `expect` itself has no field for "what the model actually replied" — only what a
correct agent should do — so replaying real output would need a second, undocumented type. Pinned
by `test_drive_replays_history_from_the_script_never_from_this_runs_own_model_output`
(`tests/test_convo.py`), which feeds the stub model *different* tool-call arguments than the
script's `expect` and asserts turn 2's assembled messages carry the scripted ones, never the
model's real ones — and by a mutation (below) that makes the drive loop replay from an
`observed_turns` accumulator instead, confirming that exact test reddens and nothing else does.
Also documented: the native `tools=` parameter is sent to the LLM on **every** turn regardless of
`representToolSchemasEachTurn` (that knob governs only the *textual* schema block `assemble`
embeds) — withholding it after turn 1 would make native tool calls structurally impossible from
turn 2 on every pack that sets the flag `false`.

**Both recurring defect classes checked against this unit's own work, per the brief.** Class 1 (a
test's name/docstring asserting more than its assertions pin) found three instances before review,
all fixed: `test_assemble_represent_tool_schemas_each_turn_true_keeps_schemas_every_turn` checked
only 2 of 3 turns against a 3-turn fixture (widened to all 3, matching its `False` counterpart's
own rigor); `test_assemble_no_tool_schemas_means_no_schema_message_regardless_of_the_flag` checked
only one flag value despite "regardless" (parametrized over both);
`test_assemble_history_turns_windows_to_only_the_last_n_prior_turns` and
`test_drive_records_a_nonnegative_wall_clock_per_turn` each checked only one N / one turn despite an
"N"/"per turn" in the name implying generality (parametrized to N=1 and N=2; widened to all three
turns of a 3-turn script). Class 2 (a guard/docstring whose declared reach exceeds its mechanism)
found one instance: `tooling.py`'s `ToolEnvironment` docstring, written first, stated
"`@runtime_checkable` exists so the harness side (`modelbench.convo.drive`) can assert conformance
with a plain `isinstance` check" — but `drive()` never actually called `isinstance`. Fixed by adding
the guard for real (`drive` now raises `TypeError` before any LLM call when `env` does not
structurally satisfy `ToolEnvironment`), not by softening the docstring, since the guard is cheap,
correct, and exactly what a pack-facing contract should do rather than merely claim. New test
`test_drive_raises_type_error_before_any_llm_call_when_env_is_not_a_tool_environment` pins it, and
also asserts `llm.calls == []` so "before any LLM call" is checked, not just the raise.
**Plan-literal check-input:** `test_dispatch_record_fields_match_the_plans_appendix_a_literal_in_
order` transcribes Appendix A's own tuple `(name, rawArguments, parsedArguments, returnValue,
timestamp)` and asserts `DispatchRecord`'s dataclass fields match it in that exact order;
`test_assemble_transcribed_from_the_plans_own_conversation_row_literal` transcribes §3.8.4's own
`conversations.jsonl` row JSON (the `A-02`/`lookup_product_fact`/`Wireless Charging Pad`/`24.99`
example) verbatim and drives it through `assemble`'s structured-mode replay. **No exemption
constant was needed** — the one type-tolerance fallback in this unit (`_parse_tool_arguments`
degrading a malformed/unparseable tool-call-argument value to `{}`) is a coercion helper in the
shape of `lmstudio.py`'s own `_seconds_to_ms`/`_as_float`, not a validator with silent-skip cells,
so it is tested per malformed-input kind directly rather than framed as an exempt set.

**Mutation-tested, `cp`-aside once per file / mutate / run the matching test file / `cp`-back
restore, `diff -q` byte-identical after every single one — 14 mutations across both files, all
caught for the stated reason, none batched:** (1) `tooling.py`: `@runtime_checkable` removed —
reddened all 6 `isinstance`-based `ToolEnvironment` tests on `TypeError: … can only be used with
@runtime_checkable protocols`; (2) `DispatchRecord`'s `rawArguments`/`parsedArguments` field order
swapped — reddened exactly the Appendix A field-order test; (3) `frozen=True` dropped from
`DispatchRecord` — reddened exactly the frozen test with "DID NOT RAISE"; (4) the Protocol's
`dispatch` method renamed to `invoke` — reddened the conforming-object isinstance test and the
declared-method-set coverage test. (5) `convo.py`: `assemble`'s `representToolSchemasEachTurn`
gate removed — reddened the flag-off drop test and (as a bonus catch) the plaintext test, which
happens to share a schema-bearing fixture; (6) `historyTurns` windowing block deleted — reddened
exactly the windowing test; (7) the current-turn message moved from `append` to `insert(0, …)` —
reddened 8 tests, every one that asserts message order or content-by-position; (8) `drive`'s
per-turn dispatch slice (`env.trace()[trace_before:]`) widened to the whole trace — reddened
exactly the per-turn-isolation test, on `2 == 1`; (9) `_expected_exchange`'s `toolRequired` branch
inverted — reddened both structured-replay tests and the plan-literal test; (10)
`_parse_tool_arguments` shorted to always return `{}` — reddened the dispatched-arguments test and
the per-turn-isolation test; (11) the malformed-tool-call `if not name: continue` guard removed —
reddened exactly the malformed-call test, on a `DispatchRecord(name=None, …)` appearing where none
should; (12) `drive`'s history source changed from the script's own `script.turns` to an
`observed_turns` accumulator poisoned with each turn's real dispatched arguments — reddened exactly
`test_drive_replays_history_from_the_script_never_from_this_runs_own_model_output`, the central
design-decision test, and nothing else; (13) the native `tools=` parameter withheld after turn 1
when `representToolSchemasEachTurn=False` — reddened exactly the "tools sent every turn" test;
(14) the new `isinstance(env, ToolEnvironment)` guard removed — reddened the new guard test with an
`AttributeError` on the first `env.trace()` call instead of the expected `TypeError`, i.e. the
guard's *absence* surfaces as a worse, later failure, which is the point of having it first.

**Observed, this run.** `model-bench/` as working directory. This unit's own two test files alone,
throughout: **39 passed** (`tests/test_tooling.py`: 8 `def test_` functions, 11 collected cases;
`tests/test_convo.py`: 26 `def test_` functions, 28 collected cases — parametrization accounts for
the gap in both). `.venv/bin/ruff check modelbench/tooling.py modelbench/convo.py
tests/test_tooling.py tests/test_convo.py`: `All checks passed!`. Full suite at the start of this
unit: **815 passed, 3 deselected**. A concurrent session was mid-edit on `modelbench/hostinfo.py`
(`M`, uncommitted) partway through this run — one full-suite pass showed 13 reds in
`tests/test_cli.py`/`tests/test_hostinfo.py`/`tests/test_packs.py`, none of them in a file this
unit touches, and a `NameError` for an undefined name in `hostinfo.py` itself confirmed it as that
session's own in-progress state rather than anything this unit caused; re-run after it moved on,
full suite: **873 passed, 3 deselected**, of which this unit's own new files account for
**39** attributable, new collected cases (both new files are untracked, `git status`-confirmed, so
the whole 39 is this unit's delta on top of whatever the concurrent session landed independently).

**Files:** `modelbench/tooling.py`, `modelbench/convo.py`, `tests/test_tooling.py`,
`tests/test_convo.py` (all new). Left uncommitted for review; a concurrent session commits to this
repository continuously and appends its own `HISTORY.md` entries.

## 2026-09-09 — S2 U81: closing impl review Pass 13's `hostinfo.py`/`attest` findings

**What:** review Pass 13 findings scoped to `modelbench/hostinfo.py` and the `attest` CLI command
— P13-4 (major), P13-5, P13-6, P13-8 (minors), P13-10, P13-12 (nits) — against U74 as committed
at `dd40ede`. P13-7 (minor) and P13-9 (a plan-text sweep) are **not fixed here**; see below.
Changed: `modelbench/hostinfo.py`, `modelbench/cli.py` (the `attest` command and its wiring only),
`tests/test_hostinfo.py`, `tests/test_cli.py`.

**P13-4 (major) — the fix, and the coupling question the coordinator asked me to answer.**
`lmStudioAppVersion`, `kvCacheSetting` and `hostRamGb` are `_NONEMPTY` in `fingerprint.py`;
`validate_host_info` checked only presence and non-`null`, so `{"lmStudioAppVersion": "",
"kvCacheSetting": "", "hostRamGb": 0, "otherResidentWorkloads": []}` returned `[]` — clean — and
`attest` could write a `host.json` that `store()` refuses only after a whole run (§3.4.5 point 1),
twenty minutes later. **Chose the coupled fix over three hand-written guards.** `hostinfo.py` now
imports `modelbench.fingerprint` (read-only — no edit to that file) and derives
`ATTESTED_NONEMPTY_FIELD_NAMES` directly from `fingerprint.REQUIRED_BY_SCHEMA[1]["model:chat"]`'s
own tiers, rather than retyping "these three are non-empty" a second time — exactly the shape that
drifted the first time, per the coordinator's framing that a shared, derived constant is worth
more than hand-written checks a fourth field can silently outrun. `validate_host_info`'s attested-
block loop now refuses any of the three an empty value (`not value`, the identical predicate
`Fingerprint.validate()` itself uses for its `_NONEMPTY` tier) while leaving
`otherResidentWorkloads` (`_PRESENT`) untouched. **The required assertion is behavioral, not a
second set comparison**, per the coordinator's warning that a re-derivation test only proves two
sets match, never that the shared reason is true:
`test_validate_host_info_agrees_with_the_fingerprints_own_attested_field_tiering`
(`tests/test_hostinfo.py`) drives `validate_host_info` against each attested field with an empty
value and asserts the refusal *matches* `fingerprint.py`'s own live tier for that field — computed
independently of `ATTESTED_NONEMPTY_FIELD_NAMES`, so it exercises the actual behavior rather than
re-running the same derivation. A second test,
`test_attested_field_tiers_agree_between_chat_and_embeddings_profiles`, is the checkable claim
that reading tiers from the single `"model:chat"` profile is safe (none of the four attested names
is in `model:embeddings`'s forbidden set, so both profiles must agree) rather than an unstated
assumption. A third, `test_validate_host_info_rejects_the_reviews_own_m4a_repro`, is the review's
own Appendix M.4/A input verified rejected. `hostRamGb`'s `> 0` half of the review's suggested fix
needed no separate comparison: `Fingerprint.validate()`'s own `_NONEMPTY` check is `not value`, and
`0` is already falsy under it — the identical predicate covers both without a second rule.

**P13-8 (minor).** `check_attestation_staleness` now raises `HostInfoError` at entry, before either
branch, when `host` carries no non-empty `observedAtAttestation.residencySource` — the precondition
`read_host_info` alone guarantees. Pre-fix, a `host` missing `observedAtAttestation` entirely
back-filled straight into an `updated_host` that itself failed `validate_host_info`, a file that,
once written, would make every later run exit `5` until re-attested. Three tests, each red before
the fix for the exact reason claimed: no `observedAtAttestation` key at all; the key present but
`residencySource` empty; and the same missing-block case on the `"embeddings"` surface, confirming
the guard runs before the `call_surface` branch rather than only inside the path that would
otherwise corrupt the file.

**P13-5 / P13-6 (minors), both closed exit-code escapes.** `_cmd_attest` now also catches
`hostinfo.HostInfoError` (P13-5's repro: `attest --api-base-url ""` previously escaped as an
uncaught traceback, exit `1`, outside §3.6a's closed set — now exit `2`).
`_gather_attested_fields` now catches `EOFError` per missing field instead of leaving it uncaught
(P13-6: `--set` with a field left unset and no stdin to prompt with previously raised `EOFError`
uncaught; `--set` is §3.6a's own non-interactive route, so this is normal usage), collects every
still-unset field name, and raises the existing `AttestUsageError` (exit `2`) naming all of them in
one message rather than failing prompt by prompt.

**P13-10 (nit).** `_residency_source_for_probe(probe_result)` ignored its only parameter and
returned a constant, promising a decision the body never made. Renamed
`_residency_source_after_a_successful_probe()` with no parameter — the only value reachable at its
one call site, since `attest()` already returns on any probe outcome but `"api-v0"`.

**P13-12 (nit).** `validate_host_info` now refuses an `attested` block carrying any key outside the
four §3.4.4 names (`attested.{key}` reported as `unexpected key(s)`). The CLI's own `--set` already
closed this route (`_parse_set_flags` rejects an unrecognized key); this closes the hand-edited
`host.json` route too.

**Not fixed — P13-7, left open on purpose.** The review's own §5 OQ-1 routes this to `architect`:
whether §3.4.5's "the check degenerates to `residencySource` alone" on a `model:embeddings` arm
means *compare just that field* or *give up entirely* is a genuine two-reading ambiguity in the
plan text itself, and the two readings produce different stored `attestationTripWire` values on a
surface `run` (not yet built) will eventually branch on. Nothing consumes `check_attestation_
staleness` yet, so guessing wrong here costs a rename-sized fix later, not a stored-record
migration — but it is still a plan-semantics call, not an implementation one, and the review
already routed it correctly. Left as shipped (`"unavailable"` unconditionally on the embeddings
surface, no comparison attempted) pending that one-clause answer. **Not fixed — P13-9** is a
`docs/plans/` text sweep (the plan's own `warm_up` code block needs a parameter added), entirely
outside this unit's fenced files and `architect`'s per the review.

**Mutations, seven, all caught, each `cp`-aside / mutate / run (`tests/test_hostinfo.py
tests/test_cli.py` only) / `cp`-back, `diff -q` byte-identical against the pre-mutation file after
every single one:** in `hostinfo.py` — `ATTESTED_NONEMPTY_FIELD_NAMES` forced to `frozenset()`
(2 of the new P13-4 tests reddened; the per-field agreement test and the M.4/A repro); the
`attested`-unexpected-key check disabled (exactly its one test reddened); the `check_attestation_
staleness` precondition removed (all three P13-8 tests reddened, one via an uncaught `TypeError`
rather than the expected `HostInfoError` — still red for the missing-guard reason, just a
different exception shape); `_residency_source_after_a_successful_probe` changed to return a wrong
literal (caught by the pre-existing U74 regression tests, confirming the P13-10 rename carried no
behavior change). In `cli.py` — the `EOFError` handling removed (the one P13-6 test reddened, via
an uncaught `EOFError`, the exact pre-fix failure mode); the `HostInfoError` catch removed from
`_cmd_attest` (the one P13-5 test reddened, via an uncaught `HostInfoError`, the exact pre-fix
failure mode).

**Observed, this run.** `model-bench/` as working directory. This unit's own attributable delta:
`git diff -- tests/test_hostinfo.py tests/test_cli.py | grep -c '^+def test_'` → **9** new test
functions (7 in `test_hostinfo.py`, 2 in `test_cli.py`). Scoped run,
`.venv/bin/python -m pytest -q tests/test_hostinfo.py tests/test_cli.py`: **84 passed, 2
deselected**. `.venv/bin/ruff check modelbench/hostinfo.py modelbench/cli.py
tests/test_hostinfo.py tests/test_cli.py`: **All checks passed!**

**Full suite, run once at the end, per the coordinator's concurrency note** (three sibling units
were live on `modelbench/lmstudio.py`, `tests/test_packs.py`, and new `modelbench/tooling.py` +
`convo.py`, all disjoint from this unit's files): `.venv/bin/python -m pytest -q` →
**870 passed, 3 deselected**, no failures — nothing needed isolating and re-running.
`.venv/bin/ruff check .` → **All checks passed!**. Baseline at `d013436` was 815 passed, 3
deselected, ruff clean; the difference is this unit's 9 new tests plus whatever the three
concurrent sibling units added, not decomposed here since only the total was directly observed.

**Files:** `modelbench/hostinfo.py`, `modelbench/cli.py`, `tests/test_hostinfo.py`,
`tests/test_cli.py`. Left uncommitted for review; `modelbench/lmstudio.py`, `tests/test_lmstudio.py`,
`tests/test_packs.py`, `modelbench/tooling.py`, `modelbench/convo.py`, `tests/test_tooling.py` and
`tests/test_convo.py` were modified/added by concurrent sibling units, not touched here.

## 2026-09-09 — S2 U79: the error-body read closed, and the unit boundary rejects non-finite/boolean sources

**What:** `docs/reviews/small-model-benchmarking-impl.md` Pass 13's blocker (P13-1) and one major
(P13-3), landing beside three concurrent units on disjoint files (`modelbench/tooling.py` +
`modelbench/convo.py`; `tests/test_packs.py`; `modelbench/hostinfo.py`, none touched here).
Changed: `modelbench/lmstudio.py`, `tests/test_lmstudio.py`. No fixtures added.

**P13-1 (blocker), fixed.** U75 (Pass 12, P12-1) moved the *success* body read
(`resp.read()`) inside `_raw_get`/`_raw_post`'s try/except ladder and left the *error* body read
(`exc.read()`, inside `except urllib.error.HTTPError`) outside any guard of its own — a response
whose error body failed to read (`http.client.IncompleteRead`, `ConnectionResetError`, a
read-phase `TimeoutError`) escaped `catalog()`, `residency()`, `probe()`, `chat()`, `embed()` and
both `warm_up()` surfaces untyped, all six operations, all three exception kinds (reproduced: 21
of 21 escaping). `§4B`'s coverage probe (U75) did not catch this because its own
`_EXEMPT_CELLS` declared `("read", "non_2xx")` structurally unreachable on the grounds that
`urlopen()` raises `HTTPError` *before* handing back a response object — true of the *success*
read, false of `HTTPError` itself: it **is** a response object, with its own status and `.read()`,
and `probe()`'s own `v1-only` diagnosis calls `exc.read()` on every 404 a non-LM-Studio server
returns, making this normal-path code rather than an edge case.

Fixed per the reviewer's own suggested shape: wrapped `exc.read()` in its own inner
`try`/`except (TimeoutError, http.client.HTTPException, OSError)` in both `_raw_get` and
`_raw_post`. `_raw_get` degrades to `None` on a failed error-body read — the same "no usable
response" fold it already applies to a failed *success*-body read — so `catalog()`/`residency()`
raise `LMStudioUnreachable`; `_raw_post` already has the status in hand (`exc.code`) before the
read is attempted, so it degrades the *message* (empty body) and still raises
`LMStudioCallFailed`, identical to every other non-2xx POST outcome.

**The probe itself was the deeper fix, not the code change.** Per the reviewer's instruction ("the
assertion that catches me being wrong goes in the grid, not in the fixed code"), `§4B`'s
`(phase, kind)` axis gained a third phase, `error-body`, with `non_2xx` as its only applicable
kind — the phase the original two-phase grid had no way to express at all, not a corrected cell
within an existing phase. Each phase now owns its own domain of applicable kinds
(`_PHASE_KINDS`), rather than one flat `phase x kind` cross product with two ad hoc exemptions:
`(connect, unparseable_body)` stays exempt (no body exists before a response object, with a
status, is obtained); `(read, non_2xx)` is **not** re-added as an exemption — it was never a real
cell, since `non_2xx` was never in the `read` phase's domain to begin with, which is the lesson
P13-1 draws out explicitly. `_EXEMPT_CELLS` is now a single-entry constant whose one remaining
reason was re-checked, not assumed. A new fake, `_RaisingFp` (a file-like object whose `.read()`
raises, used as a hand-built `urllib.error.HTTPError`'s own `fp`), expresses the new phase; seven
new parametrize cases (one new cell x seven existing per-operation tests) exercise it for real,
and the existing re-derivation test (`test_probe_cell_exemptions_are_exactly_the_structurally_
unreachable_ones`) was rewritten to compute the grid from `_PHASE_KINDS`'s per-phase domains
instead of a flat cross product, so a fifth axis value or a wrongly-scoped kind fails loudly
rather than silently passing.

**P13-3 (major), fixed.** `_seconds_to_ms`/`_as_float` accepted anything `float()` accepts,
including `bool` (`float(True) == 1.0`, so a stray boolean landed as `ttftMs=1000.0`,
`tokensPerSecond=1.0`) and non-finite values (`nan`/`inf`/`-inf` — `float("nan")` does not raise,
and `json.loads` parses the bare tokens `NaN`/`Infinity`/`-Infinity` without error by default, so
no malformed transport is needed, only a server serialising e.g. a 0/0 rate). Both landed as an
actual number rather than degrading to `None`, contrary to `ChatResult`'s docstring. Decision
(stated in both the module and class docstrings, per the brief's instruction to say exactly which
boundary was chosen): reject **at the coercion boundary** (`_seconds_to_ms`/`_as_float`, now
sharing one `_coerce_finite_float` helper), **not** at `_parse_json`/`json.loads` — `_parse_json`
parses every body this adapter reads and the raw `stats` mapping is kept verbatim beside the
derived fields "for auditability"; narrowing the fix to the two functions that actually produce a
typed timing figure keeps that verbatim guarantee intact and leaves every other body's parsing
unchanged. `bool` is excluded before coercion (an `isinstance` check), non-finite after
(`math.isfinite`).

**Mutation-tested, each `cp`-aside / mutate / run `tests/test_lmstudio.py` alone / `cp`-back
restore, `diff -q` byte-identical after every one — four mutations against `modelbench/lmstudio.py`,
all caught for the stated reason:** (1) `_raw_get`'s new inner error-body guard reverted —
reddened exactly the 3 new GET-side `error-body`/`non_2xx` cells (catalog, residency, probe); (2)
`_raw_post`'s new inner error-body guard reverted — reddened exactly the 4 new POST-side cells
(chat, embed, both `warm_up` surfaces); (3) `math.isfinite` check dropped from
`_coerce_finite_float` — reddened both new non-finite tests; (4) the `bool` exclusion dropped —
reddened the new bool test. One additional check against the test-side guard itself (not the
formal `cp`-aside process, since it is test machinery rather than production code): widening
`_EXEMPT_CELLS` with a spurious extra entry reddened `test_probe_cell_exemptions_are_exactly_
the_structurally_unreachable_ones`, confirming the re-derivation test still catches a stale
exemption under the new per-phase-domain shape. Also verified directly (outside the grid, which
samples one representative exception per kind): all three exception kinds the reviewer named
(`IncompleteRead`, `ConnectionResetError`, `TimeoutError`) land in the correct typed outcome on
both `catalog()` and `chat()`, not just the one the grid parametrizes.

**Observed, this run.** `model-bench/` as working directory. `tests/test_lmstudio.py` alone: entry
82 passed / exit **92 passed**, 1 deselected (+10: 3 new `def test_` functions for P13-3, plus 7
new parametrize cases across the 7 existing per-operation grid tests for the new `error-body`
phase — confirmed via `git diff | grep -c '^+def test_'` = 3). `ruff check modelbench/lmstudio.py
tests/test_lmstudio.py`: `All checks passed!`. Full suite (before U80's own entry below landed):
two reds appeared once in `tests/test_cli.py`/`tests/test_hostinfo.py` (a concurrent unit's
`hostinfo.py`/`cli.py` work, fenced off from this one) and were gone, both in isolation and on a
full re-run — consistent with a sibling's live edit, not this unit's work. Full suite, this run:
**870 passed, 3 deselected**.

**Not done — a decision, not an oversight.** P13-4 (a `validate_host_info` gap in `hostinfo.py`)
and every other Pass 13 minor/nit land outside this unit's fences (`hostinfo.py`, `cli.py`,
`packs.py`, `docs/reviews/`) and are untouched.

**Files:** `modelbench/lmstudio.py`, `tests/test_lmstudio.py`. Left uncommitted for review;
concurrent units append their own entries to this same `HISTORY.md`.

## 2026-09-09 — S2 U80: the row-count identity's coverage probe corrected to measure its own route, not `validate_pack`

**What:** `docs/reviews/small-model-benchmarking-impl.md` Pass 13's P13-2, landing beside three
concurrent units on disjoint files (`modelbench/lmstudio.py`; `modelbench/tooling.py` +
`convo.py`; `modelbench/hostinfo.py`, none touched here). Changed: `tests/test_packs.py` only —
`modelbench/packs.py` is unchanged (every edit to it was a mutation, restored byte-identical
before this entry was written; `git status` shows it clean). No fixtures added.

**The finding, and what I checked before touching anything.** Pass 12 §4A's coverage probe
(`test_row_count_identity_coverage_over_its_own_keys_and_value_kinds`) was written to guarantee
that `_row_count_identity_problems` — the row-count identity route — never goes silent on a cell
it doesn't own an exemption for. Its silence criterion was `validate_pack(pack) == []`
(`tests/test_packs.py:340`, pre-fix), the *whole validator's* output, not the route's own return
value. `ROW_COUNT_IDENTITY_EXEMPT_CELLS` is documented as "the one case **this route** is
sanctioned to skip silently" (`packs.py:412`) — a claim about the route, checked by a probe over a
different function. I reproduced the masking directly rather than trusting the review's appendix
description: with a P12-6-shaped silent branch (`if "analysisUnit" not in sampling: return []`)
re-inserted at the head of `_row_count_identity_problems`, the pre-fix probe stayed at **35
passed** — unchanged. Calling `_row_count_identity_problems` and `validate_pack` on the exact
manifest that mutation makes silent shows why: the route itself returns `[]` on that cell, but
`validate_pack(pack)` still returns `['…: sampling.analysisUnit is absent']` — not from the route,
but from `_sampling_problems`'s other call, `pack.ref()`, whose own `_ref_from_manifest_fields`
independently refuses a manifest with no `sampling.analysisUnit` at all, before
`check_sampling_contract` ever runs. The old criterion `validate_pack(pack) == []` was simply never
true on this cell, so it was never counted "silent" — masking the route's own silence underneath
an unrelated problem that happened to cover the same manifest. (My first written draft of this
finding's regression-test docstring asserted the opposite — that `validate_pack` *does* stay `[]`
on the masked cell — which is backwards; I caught it by executing the claim before committing to
it, not by inspection, and corrected the docstring to the verified mechanism above.)

**Fix.** `test_row_count_identity_coverage_over_its_own_keys_and_value_kinds` now calls
`_row_count_identity_problems(pack, sampling)` directly for each of the 12 cells (imported as a
private name from `modelbench.packs`, the same pattern `tests/test_stats.py` already uses for
`_family_ci_levels` etc.) instead of `validate_pack(pack)`; the assertion shape
(`silent_cells == set(ROW_COUNT_IDENTITY_EXEMPT_CELLS)`) and the all-valid control are unchanged.
The control gained one extra assertion (`_row_count_identity_problems(...) == []` alongside the
existing `validate_pack(...) == []`) so the route's own silence on the genuinely-valid case is
pinned too, not only inferred from the whole validator.

**Test-first, run both ways.** With the P12-6-shaped mutant re-inserted: pre-fix probe green (35
passed, unchanged — the masking); corrected probe **red**, `AssertionError`, `silent_cells` holding
the extra `("sampling.analysisUnit", "absent")` member the exemption constant does not name — the
mutation caught for the exact reason the finding names. Restored `packs.py` (`cp`-back, `diff -q`
byte-identical): corrected probe **green**, 35 passed, tip unchanged.

**Mutation-tested.** `cp`-aside `modelbench/packs.py` before the first mutation; one mutation
(the P12-6-shaped `analysisUnit`-absent branch), run against `tests/test_packs.py` alone, `cp`-back
and `diff -q` byte-identical restore after. No second mutation was needed: this unit's job was the
probe itself, not a new route defect, and the one mutation that reproduces P13-2 is also the one
that proves the fix.

**The other exemption mechanisms, checked.** The brief asked whether any exemption constant in my
fences shares this guard-versus-mechanism gap. `modelbench/packs.py` carries exactly one exemption
constant, `ROW_COUNT_IDENTITY_EXEMPT_CELLS` (grepped for `EXEMPT`/`exempt`/`sanctioned`; nothing
else in the module declares a sanctioned-silent set), and it is the one just fixed. The other two
exemption mechanisms the coordination has been bitten by — the adapter's `_EXEMPT_CELLS`
(`modelbench/lmstudio.py`, P13-1) and the attested-field tiers `validate_host_info` checks against
(`modelbench/hostinfo.py`, P13-4) — live in files this unit's fences exclude (both were modified,
mid-flight, by the concurrent units named above); neither is addressed here.

**Observed, this run.** `model-bench/` as working directory. `tests/test_packs.py` alone: **35
passed** before and after (test count unchanged — this was an assertion-target fix, not a new
test). `.venv/bin/ruff check modelbench/packs.py tests/test_packs.py`: `All checks passed!`. Full
suite once at the end: **867 passed, 3 failed, 3 deselected** — the 3 failures are
`tests/test_lmstudio.py::test_{catalog,residency,probe}_..._[error-body-non_2xx]`, entirely inside
`modelbench/lmstudio.py` / `tests/test_lmstudio.py` (both showing modified in `git status`, owned
by the concurrent unit chasing P13-1), untouched by and unrelated to this change — `packs.py` and
`test_packs.py` are the only fenced files this unit modified, and `packs.py` ends this run
byte-identical to how it started.

**CPG:** considered, not relevant — no Code Property Graph is loaded for `model-bench` (only
`cpg_falkorchat` and `cpg_deprecated_salesperson` exist on this instance), and this task is a
test-only regression fix within one already-read, already-open file, with no call-graph or
data-flow question to put to one.

**Files:** `tests/test_packs.py`. Left uncommitted for review.

## 2026-09-09 — S2 U77: `ChatResult`'s "never raises" promise closed for any `stats` shape, not just the caller-guarded one

**What:** `docs/reviews/small-model-benchmarking-impl.md` Pass 12's P12-11, the one finding U76 left
open on `packs.py`'s sibling thread and flagged rather than fixing unilaterally. Changed:
`modelbench/lmstudio.py`, `tests/test_lmstudio.py`.

**The finding.** `ChatResult`'s class docstring states construction "never raises" on a missing or
partial `stats` object, unconditionally in its wording. The mechanism that actually held that
promise lived one level up, in `chat()`'s own `isinstance(..., Mapping)` guard before it ever
constructs a `ChatResult` — `__post_init__` itself did `stats = self.stats or {}`, so a direct
construction with a malformed, non-`Mapping` `stats` (e.g. `ChatResult(stats=[1], ...)`) raised
`AttributeError` from `stats.get(...)`, bypassing the guard entirely. Separately,
`tokensPerSecond = stats.get("tokens_per_second")` was read verbatim with no coercion, so a
string-valued source survived into a field declared `float | None` as a `str`.

**Closure chosen: (a), make the mechanism total** — over the two closures the brief offered, and
against its own steer of (a) only in the sense that (a) was also the steer taken. Reasoning: (1)
`docs/plans/small-model-benchmarking.md` §3.6's own unit-boundary text says the same thing the class
docstring does — "construction never raises on a missing or partial `stats` object" — with no
caller-side qualifier anywhere in that section, so narrowing the docstring to name `chat()`'s guard
as load-bearing would contradict the plan, not just the docstring, and the plan is the more
authoritative of the two. (2) `ChatResult` is a public, directly-constructible frozen dataclass —
nothing about its `__init__` signature suggests "only ever construct this through `chat()`" — and
the brief's own framing (the class's whole job is to be where malformed payloads are made safe) is
consistent with §3.6. (3) Widening costs nothing observable on the real call path: `chat()`'s guard
still runs first and nothing changes for it; the only behavior added is tolerance for a construction
route the guard was never protecting anyway. No genuine programming error is swallowed by this
widening — a caller handing `ChatResult` a list where `stats` belongs is exactly the malformed-input
case the class's stated job is to absorb, not a caller's own logic bug being hidden from it.

**Fix.** `__post_init__` now derives the timing trio from `self.stats if isinstance(self.stats,
Mapping) else {}` rather than `self.stats or {}` — a non-`Mapping` `stats` degrades the same way an
absent one does, and `stats` itself is still kept verbatim, unmodified, for auditability regardless
of its shape. `tokensPerSecond`'s source now goes through a new `_as_float` helper — `None` when
absent, `float(value)` when coercible, `None` on `TypeError`/`ValueError` otherwise — the same
tolerance `_seconds_to_ms` already applies to `ttftMs`/`generationMs`, minus the ×1000 conversion
`tokensPerSecond` does not need (it is already a per-second rate). Both the module docstring and
`ChatResult`'s class docstring were rewritten to state the widened, exact contract — including that
`tokensPerSecond` is now type-coerced even though it is not unit-converted — rather than the
narrower claim the old mechanism actually kept.

**Test-first.** Three new tests, all written and confirmed red before the production change, each
for the stated reason: `test_chat_result_construction_never_raises_when_stats_is_not_a_mapping`
(`ChatResult(stats=[1], ...)`) red with `AttributeError: 'list' object has no attribute 'get'`;
`test_chat_result_tokens_per_second_coerces_a_numeric_string_source_to_float`
(`stats={"tokens_per_second": "51.4"}`) red on `AssertionError: assert '51.4' == 51.4`;
`test_chat_result_tokens_per_second_is_none_when_source_is_not_numeric`
(`stats={"tokens_per_second": "fast"}`) red on `AssertionError: assert 'fast' is None`. Checked
against the existing suite for duplication first: no prior test constructs a non-`Mapping` `stats`
or a string-valued `tokens_per_second` — the closest, `test_chat_result_construction_never_raises_
on_a_partial_stats_object`, only ever passes a `Mapping` missing keys, never a wrong-typed `stats`.

**Mutation-tested, `cp`-aside / mutate / run `tests/test_lmstudio.py` / `cp`-back restore,
`diff -q` byte-identical after each restore, two mutations, both caught for the stated reason:**
(1) `isinstance(self.stats, Mapping)` reverted to `self.stats or {}` — reddened exactly
`test_chat_result_construction_never_raises_when_stats_is_not_a_mapping`, on the same
`AttributeError` the finding names; (2) `_as_float(stats.get("tokens_per_second"))` reverted to
`stats.get("tokens_per_second")` verbatim — reddened both `tokensPerSecond` coercion tests, on the
`str` value surviving uncoerced in each.

**Observed, this run.** `model-bench/` as working directory. Baseline before this unit: **812
passed, 3 deselected**. After: **815 passed, 3 deselected** (the three new tests; no other test's
outcome changed). `.venv/bin/ruff check .`: `All checks passed!`.

**Files:** `modelbench/lmstudio.py`, `tests/test_lmstudio.py`. Left uncommitted for review.

## 2026-09-09 — S2 U75: the LM Studio adapter's read-phase exception taxonomy, closed

**What:** `docs/reviews/small-model-benchmarking-impl.md` Pass 12's adapter thread (P12-1, the
blocker; P12-4; P12-5), landing beside two concurrent units on disjoint files
(`modelbench/hostinfo.py`+`modelbench/cli.py`; `modelbench/packs.py`+`tests/test_packs.py`, not
touched here). Changed: `modelbench/lmstudio.py`, `tests/test_lmstudio.py`. No fixtures added.

**P12-1 (blocker), fixed.** `_raw_get` and `_raw_post` each wrapped only the connect call
(`self._opener(...)`) in their try/except ladder; `resp.read()` sat outside it, so a response
whose body-read failed — `http.client.IncompleteRead`, `ConnectionResetError`, a read-phase
`TimeoutError` — escaped `catalog()`, `residency()`, `probe()`, `chat()`, `embed()` and `warm_up()`
untyped. Fixed by moving the read inside the same try in both methods and adding
`http.client.HTTPException` as an explicit rung (`IncompleteRead` is not an `OSError`, so the
existing socket catch did not reach it); a read-phase `TimeoutError`/`ConnectionResetError` are now
covered by the ladder's existing timeout/socket rungs, unchanged, now that they're in scope.

**P12-4, fixed.** `_raw_post` stopped its wall clock right after the connect call, before
`resp.read()`, so `ChatResult.wallClockMs` excluded the entire body-read time — systematic and
always short. The clock now stops after a successful read, matching §3.6 FR-11's definition
("to the last byte of the body").

**P12-10 (minor, in the same two methods so fixed alongside P12-1), fixed.** `catalog()` parsed
the response body before checking its HTTP status, so a non-2xx response with a non-JSON body (an
HTML error page) reported "not valid JSON" instead of the real status. Status is now checked
first.

**P12-5, decided and fixed.** `warm_up`'s docstring justified re-probing `residency()` internally,
immediately before the timed call, by claiming `residentModelsAtStart` "would misreport every
cold warm-up as already resident" if used instead — backwards: `model in set()` on an empty
cold-start set is correctly `False` ("not resident"). §3.6 itself names `residentModelsAtStart` as
`coldLoadSeconds`'s source in so many words. Decided on the merits (the brief delegated this call
explicitly, and nothing outside this unit's tests calls `warm_up` yet — `runner.py` is a later
unit — so this is the cheapest point to fix the interface rather than carry the false docstring
and an unlisted extra catalog GET forward): `warm_up` now takes `was_resident_before: bool` as a
required, no-default keyword argument, supplied by the caller from `residentModelsAtStart`, and no
longer probes `residency()` itself. This is a public signature change (`LMStudio.warm_up`); safe
now because nothing outside `tests/test_lmstudio.py` calls it yet.

**§4B's coverage probe, built as the regression net (per Pass 12 §4B, "done when it reads 0").** A
parametrized test grid over the six public operations `{catalog, residency, probe, chat, embed,
warm_up}` x two failure phases `{connect/headers, body-read}` x four failure kinds `{timeout,
non-2xx, connection drop, unparseable body}`. Two of the eight (phase, kind) cells per operation
are not reachable through `urllib.request`'s own contract — `(connect, unparseable_body)`: no body
exists before a response object is obtained; `(read, non_2xx)`: `urlopen()` raises `HTTPError` for
any status >= 400 *before* ever handing back a response object, so non-2xx cannot be observed once
`.read()` is reachable — declared as a module-level `_EXEMPT_CELLS` constant with those reasons,
and a dedicated test re-derives the full 8-cell grid and asserts the exempted set equals that
constant exactly (asserted, not implied by absence — required by the brief, and this is where a
class-2 guard-reach defect would reappear if the exemption silently grew or shrank). The remaining
six cells per operation are exercised for real, for all six operations: `catalog()`/`residency()`
must land in `LMStudioUnreachable`/`LMStudioCallFailed`; `probe()` must never raise and must return
one of its three literals; `chat()`/`embed()`/`warm_up()` (both call surfaces) must land in
`LMStudioCallTimeout`/`LMStudioCallFailed`. **Run today: 0 cells escape** (down from the reviewer's
measured 9 of 9 on the body-read row).

**One test caught not pinning what it claimed — by my own mutation, before review.** The first
version of `test_warm_up_passes_was_resident_before_through_verbatim` exercised only the chat call
surface. Mutating `warm_up`'s embeddings-surface `return LoadResult(...)` to invert
`wasResidentBefore` left the full suite green — the same class of defect Pass 12 (P12-8) and prior
units have hit five times before. Parametrized over both call surfaces; the embeddings-branch
inversion now reddens exactly that parametrize case, and only that one.

**A discrepancy in the review document, not acted on.** Pass 12 §4's "the adapter thread" names
P12-1, P12-3, P12-4 and P12-5 as this thread's findings, but P12-3's content (`tools.module`'s AST
allowlist gaps) is entirely `modelbench/packs.py` — the concurrent unit's file, explicitly fenced
off from this one. P12-3 was not touched; the fences govern over the thread label. (Independently
confirmed: the concurrent U76 entry below lists P12-1/P12-4/P12-5/P12-8/P12-10/P12-11 as "this
unit's" — i.e. mine — matching this read.)

**Mutation-tested, each `cp`-aside / mutate / run `tests/test_lmstudio.py` / `cp`-back restore,
`diff -q` byte-identical after every single one — nine mutations, all caught for the stated
reason:** (1) full P12-1 revert in `_raw_get` (read moved back outside the try, `HTTPException`
dropped from the catch) — reddened exactly the 6 GET-side read-phase probe cells (catalog,
residency, probe x timeout/connection_drop); (2) `http.client.HTTPException` rung alone removed
from `_raw_post` — reddened exactly the 4 `IncompleteRead` probe cells (chat, embed, both
`warm_up` surfaces x read-connection_drop); (3) wall clock computed before `resp.read()` instead
of after — reddened exactly the new P12-4 slow-body test; (4) `catalog()`'s status-check/parse
order reverted — reddened exactly the new P12-10 test; (5) `warm_up` reverted to self-probing
`residency()` — reddened all 16 `warm_up`-touching tests, `test_warm_up_never_probes_residency_
itself` among them, each on "no stubbed route for .../api/v0/models"; (6) `_EXEMPT_CELLS` widened
to include a genuinely-reachable cell — reddened the exemption-derivation test; (7) the same cell
dropped from `_EXPECTED_FOR_GET` alone (an undeclared third exemption) — reddened the same test,
plus silently lost 3 parametrize cases, which the test's own re-derivation still caught; (8) a
default (`= False`) added to `warm_up`'s `was_resident_before` — reddened
`test_warm_up_requires_was_resident_before_with_no_default`; (9) `wasResidentBefore` inverted on
the `chat`-surface branch of `warm_up`'s return — reddened both the chat-surface warm-up test and
the (then-unparametrized) passthrough test, which is what surfaced the "one test caught not
pinning what it claimed" finding above; re-run after parametrizing, the embeddings-branch
companion mutation reddened exactly the new `[embeddings]` case.

**Observed, this run.** `model-bench/` as working directory. `tests/test_lmstudio.py` alone,
throughout: stable at **79 passed, 1 deselected** (up from 32 passed, 1 deselected before this
unit — 33 `def test_` functions before, 44 after; the difference between 44 functions and 79
collected cases is the probe's parametrization). Two other units were mutation-testing
`modelbench/packs.py`/`tests/test_packs.py` and `modelbench/hostinfo.py`/`modelbench/cli.py`
concurrently during this session; the full suite showed transient, unrelated reds in
`tests/test_packs.py`, `tests/test_hostinfo.py` and `tests/test_cli.py` at various points, each
time gone on the next run — consistent with a sibling's live mutation, not this unit's work. Final
full suite, this run: **812 passed, 3 deselected**. `.venv/bin/ruff check modelbench/lmstudio.py
tests/test_lmstudio.py`: `All checks passed!`.

**Files:** `modelbench/lmstudio.py`, `tests/test_lmstudio.py`. Left uncommitted for review;
concurrent units append their own entries to this same `HISTORY.md`.

## 2026-09-09 — S2 U76: impl-review Pass 12 fix round — the row-count identity's coverage probe, `tools.module`'s own path, and the `valid` fixture's role shape

**What:** `docs/reviews/small-model-benchmarking-impl.md` Pass 12's `packs.py`-scoped findings —
P12-6, P12-2, P12-3(i)/(ii)/(iii), and P12-7. `modelbench/lmstudio.py`'s findings (P12-1, P12-4,
P12-5, P12-8, P12-10, P12-11) and the two open questions are a concurrent unit's / the plan
owner's, not touched here.

**P12-6 — round three of the row-count identity's exemption being wider than declared, closed
with a falsifiable coverage property instead of a fourth narrowing.** U73's `declares_scripts`
predicate (`isinstance(scripts, int)`) silently exempted a `scripts`-declaring pack whose `scripts`
was the *wrong type* (`"12"`), and separately `replicatesPerScript` absent or wrong-typed always
returned `[]` unconditionally. Closed per the reviewer's §4A design (accepted rather than
overruled — it reads well and gives a property that survives a fourth attempt): `packs.py` gained
`ROW_COUNT_IDENTITY_KEYS` (the four manifest keys the route reads, generated once and consulted by
both the route and the new test) and `ROW_COUNT_IDENTITY_EXEMPT_CELLS` (a `dict[(key, kind), str]`
naming the **one** sanctioned silent cell — `sampling.scripts` truly absent — with its reason).
`_row_count_identity_problems` now skips only on true absence of `sampling.scripts`; every other
absent-or-wrong-type field on a `scripts`-declaring pack is a reported problem, table-driven via
`_row_count_identity_field`/`_row_count_identity_field_valid` rather than four hand-written `or`
clauses. New test `test_row_count_identity_coverage_over_its_own_keys_and_value_kinds`
(`tests/test_packs.py`) builds the 4-key × 3-value-kind grid (plus an all-valid control) against a
rows file that violates the identity, computes the actual silent-cell set by execution, and asserts
it equals `ROW_COUNT_IDENTITY_EXEMPT_CELLS` exactly.

**P12-2 / P12-3(i) — `tools.module`'s own path, unconstrained.** A manifest declaring
`"module": "../outside.py"` validated clean and `load_tool_module` executed code outside the pack
root that `content_hash` never covers (falsifying §3.3's "pack code is part of the content hash");
a `tools/sim.pyc` module validated clean and ran unscanned since the AST walk globs `*.py` only.
New `_tool_module_problems`, called from `validate_pack`, refuses both: a `tools.module` that
resolves outside `pack.root` (via `Path.is_relative_to`), or that does not end in `.py`. Two new
fixtures, `tests/fixtures/packs/tools_module_outside_root/` (plus a stray
`tests/fixtures/packs/outside.py` it points at) and `tests/fixtures/packs/tools_module_not_py/`
(a dummy `tools/sim.pyc`), each with one new test.

**P12-3(iii) — the overclaiming docstring, narrowed rather than the mechanism changed.**
`Pack.load_tool_module`'s docstring said the AST check exists "to make safe … before this ever
runs against an untrusted pack"; P12-3(ii) showed `__import__`/`importlib.import_module` inside a
pack module reach any importable name the walk never sees. Rewritten to state the AST check is a
coupling rule over each file's own `import` statements, not a sandbox — P12-3(ii) itself needed no
code change, since it is the mechanism working as documented once the claim is corrected.

**P12-7 — the `valid` fixture's positive control violated the plan's own role rule.**
`tests/fixtures/packs/valid/pack.json` declared `role: "tool-caller"` with
`pairingKey: ["conversationId", "turnIndex"]` / `analysisUnit: "conversationId"` — satisfying the
*mechanised* half of §3.3's rule (`analysisUnit == pairingKey[0]`) while violating the *stated*
half ("for the tool-caller that is `scriptId`, never a conversation id"). Re-keyed to
`["scriptId", "turnIndex"]` / `"scriptId"`, with `conversations.jsonl` re-keyed to match. New test
`test_the_valid_fixtures_analysis_unit_is_scriptId_not_a_conversation_id` pins the shape.

**Test-first.** All six new tests were written before their corresponding production/fixture
change and confirmed failing for the stated reason first: the coverage-probe and
`ROW_COUNT_IDENTITY_EXEMPT_CELLS`-dependent tests via `ImportError` (the constants did not exist
yet); the two `tools.module` tests via `PackConfigError: pack.json is absent` (the fixtures did
not exist yet); the P12-7 test via `AssertionError: assert 'conversationId' == 'scriptId'`.

**Six mutations, all caught, each `cp`-aside / mutate / run `tests/test_packs.py` / `cp`-back,
`diff -q` confirmed byte-identical after every restore:** (1) the whole file reverted to the
pre-U76 version — `ImportError` on collection, the same failure observed pre-fix; (2) the
`sampling.scripts` exemption widened back to U73's type-based predicate — the coverage probe's
`silent_cells == set(ROW_COUNT_IDENTITY_EXEMPT_CELLS)` assertion fails, naming the reopened
`("sampling.scripts", "wrong-type")` cell exactly; (3) `_row_count_identity_field_valid` patched
to always accept `sampling.replicatesPerScript` — the probe reddens with a `TypeError` inside the
row math (an absent `replicatesPerScript` reaching `scripts * replicates` unguarded), a louder
failure than a silent-cell mismatch but still a failure the mutation earns; (4) the `tools.module`
containment check removed — `test_validate_pack_rejects_a_tool_module_outside_the_pack_root`
reddens with `[] == [...]`; (5) the `.py`-suffix check removed —
`test_validate_pack_rejects_a_non_py_tool_module` reddens the same way; (6) fixture-level: the
`valid` fixture's `pack.json` reverted to `analysisUnit: "conversationId"` (`conversations.jsonl`
left re-keyed, deliberately, to isolate this from a rows-file change) —
`test_the_valid_fixtures_analysis_unit_is_scriptId_not_a_conversation_id` reddens on the exact
`'conversationId' == 'scriptId'` assertion; `test_validate_pack_accepts_a_fully_valid_pack` also
reddens, incidentally, because the row-count identity notices the mismatched key name against the
still-`scriptId`-keyed rows file — both restored together.

**Observed, this run.** `model-bench/` as working directory. `tests/test_packs.py` alone throughout
the mutation loop: **35 passed** (29 pre-U76 + 6 new) at every restore point.
`.venv/bin/ruff check .`: `All checks passed!`. Full suite, run once at the end per the
coordinator's concurrency instructions (two other units were mutation-testing
`modelbench/lmstudio.py` and `modelbench/hostinfo.py`/`modelbench/cli.py` concurrently; a
transient full-suite red in `tests/test_lmstudio.py` mid-session, outside these fences, was
confirmed to be that concurrent activity by re-running `tests/test_packs.py` alone per instruction,
which stayed green throughout): **811 passed, 3 deselected**.

**Files:** `modelbench/packs.py`; `tests/test_packs.py`;
`tests/fixtures/packs/tools_module_outside_root/pack.json` (new),
`tests/fixtures/packs/outside.py` (new), `tests/fixtures/packs/tools_module_not_py/pack.json` and
`tools/sim.pyc` (new); `tests/fixtures/packs/valid/pack.json` and `conversations.jsonl` (re-keyed).
Left uncommitted for review; other units continue to append their own sections to this same
`HISTORY.md`.

## 2026-09-09 — S2 U74: `host.json`, the attestation trip-wire, and the `attest` CLI command

**What:** `docs/plans/small-model-benchmarking.md` §3.4.4 (`host.json`'s schema), §3.4.4a
(capture order, the source-of-truth table), §3.4.5 point 3 (the attestation staleness trip-wire)
and §3.6a's `attest` row, landing beside a concurrent unit mid-edit on `modelbench/packs.py` and
`modelbench/lmstudio.py` (not touched here). New: `modelbench/hostinfo.py`,
`tests/test_hostinfo.py`. Changed: `modelbench/cli.py` (the `attest` subcommand and its wiring
only), `tests/test_cli.py` (the S2-boundary test narrowed to name only `validate`/`run`, plus new
`attest` tests).

**Delivered.** `hostinfo.py` owns three things. (1) `validate_host_info(d) -> list[str]`
(`Fingerprint.validate()`/`validate_pack`'s own `[]`-means-valid shape), `write_host_info` and
`read_host_info` — the latter raises `HostInfoError` on an absent or schema-invalid file, which is
`run`'s (a later unit's) cue to exit `5` at capture-order step 1; not wired into `run` here, since
`run` is out of this unit's fences. (2) `attest(root, *, api_base_url, attested, client, now=None)`
— the `attest` command's actual work: probes an injected `LMStudio`-like `client`, and on a
successful `"api-v0"` probe writes `host.json` with `observedAtAttestation` carrying
**`residencySource` only** (`"lmstudio-api-v0"`, this repo's own established literal, reused from
`tests/conftest.py`'s `MODEL_FIELDS` rather than invented) — never `runtimeName`/`runtimeVersion`,
since neither probed endpoint exposes a `runtime` object and `attest` has no model to call one
against (plan-gate P4-6). A `"v1-only"` or `"unreachable"` probe raises `AttestProbeFailed` with
§3.4.4a's two distinguishing messages and writes nothing. (3)
`check_attestation_staleness(host, *, call_surface, residency_source, runtime_name,
runtime_version, now=None) -> AttestationCheck` — the trip-wire's pure decision function (plan
§3.4.5 point 3), covering its three named outcomes plus the mismatch case that lives inside
`"compared"` as `stale=True` rather than as a fourth outcome string (a `RunResult` is never built
on a mismatch, so that case is never stored at all): `"unavailable"` unconditionally on
`call_surface == "embeddings"` (no comparison is attempted, whatever `host` holds); a first chat
run with no `runtimeName` key yet observed backfills `runtimeName`/`runtimeVersion`/
`runtimeObservedAt` into a copy of `host` (`attested` and `attestedAt` untouched) and returns
`"first-observation"`, `stale=False`; every later chat run returns `"compared"`, comparing
`runtimeName`, `runtimeVersion` **and** `residencySource` against what was last observed — a
difference in any one of the three sets `stale=True` and `message` to the plan's verbatim string.
**Not wired into a `run` command** — `run`'s own capture-order sequence (the warm-up call that
would supply `runtime_name`/`runtime_version`) is a later unit's; this ships the mechanism, tested
directly against hand-built `host.json` states, one test per outcome (four: unavailable,
first-observation, compared-and-clean, compared-and-stale — the last parametrized over each of the
three comparands independently).

`cli.py` gains the `attest` subcommand (`--api-base-url`, default `http://localhost:1234`;
repeatable `--set key=value` for the four attested fields, with interactive `input()` prompting
for whatever `--set` leaves unset) and `EXIT_LMSTUDIO_UNREACHABLE = 3`. `hostRamGb` is parsed as
an integer and `otherResidentWorkloads` split on commas (both CLI-encoding choices this unit made,
not named by the plan beyond "`--set k=v`" — reversible by construction, since no stored record
depends on the encoding, only on the JSON `host.json` ends up holding). An unrecognized `--set`
key, a malformed `key=value` pair, or a non-integer `hostRamGb` all exit `2` before any network
call. `test_s2_commands_are_not_shipped_yet` is renamed
`test_s2s_remaining_commands_are_not_shipped_yet` and narrowed to `("validate", "run")` — `attest`
now legitimately exits `0`/`3`, and the boundary test still reddens if either of the two remaining
commands silently starts working.

**Guard built from the plan's own literal, per this coordination's standing requirement.**
`tests/test_hostinfo.py`'s `PLAN_LITERAL_HOST_JSON` is §3.4.4's example JSON block transcribed
verbatim (not a fixture this implementation wrote for itself);
`test_the_plans_own_literal_host_json_validates` asserts `validate_host_info(...) == []` against
it, and every rejection test in the file mutates one field off a copy of that same literal.

**Mutations, nine, all caught, each `cp`-aside / mutate / run / `cp`-back, `diff -q`
byte-identical against the pre-mutation file after every single one:** in `hostinfo.py` —
(1) the embeddings branch of `check_attestation_staleness` changed to fire on `"chat"` instead
(7 tests reddened, including the round-trip test, since the derived `"unavailable"` case is what
the second half of that test depends on); (2) the `"runtimeName" not in observed` guard forced to
always take the first-observation branch (5 tests reddened, every `"compared"` case); (3) the
`residencySource` comparand dropped from the staleness predicate (exactly the one parametrized
case that changes only `residency_source` reddened — the other two comparands' cases stayed
green, confirming they exercise different code); (4) `attest`'s two `AttestProbeFailed` message
branches swapped (4 tests reddened, 2 in each file); (5) `validate_host_info`'s
`observedAtAttestation.residencySource` check deleted (exactly the two tests that exist to pin it
reddened, nothing else — confirming the guard's reach matches its declared one, not less). In
`cli.py` — (6) `hostRamGb`'s int-coercion branch disabled by renaming its guard condition (6 tests
reddened via an uncaught `HostInfoError` from `attest`'s own defensive
`validate_host_info` check — red for the right reason, not a silent pass); (7) the unrecognized-
`--set`-key check deleted (the one test written for it reddened, via a stdin-read `OSError` once
the bad key fell through to interactive prompting rather than by coincidence); (8) `.strip()`
dropped from the `otherResidentWorkloads` comma-split (the one test asserting the split's exact
output reddened on a stray leading space, nothing else); (9) the malformed-`key=value` check
deleted (the one test for it reddened, again via the stdin-read `OSError`, confirming the
fallback-to-prompting path is what the check exists to prevent).

**Observed, this run.** `model-bench/` as working directory. This unit's own attributable delta:
`tests/test_hostinfo.py` is new and collects **42 tests (40 selected, 2 deselected — see below)**;
`tests/test_cli.py` gained **11 new test functions** (10 new `attest` tests plus the renamed
boundary test) per `git diff -- tests/test_cli.py | grep -c '^+def test_'`. Scoped run,
`.venv/bin/python -m pytest -q tests/test_hostinfo.py tests/test_cli.py`: **75 passed, 2
deselected**. `.venv/bin/ruff check modelbench/hostinfo.py modelbench/cli.py
tests/test_hostinfo.py tests/test_cli.py`: **All checks passed!**

**A full-suite run was not a stable baseline while this unit worked** — `git status` and
`git diff --stat` at the time showed a concurrent session with `modelbench/packs.py`,
`modelbench/lmstudio.py`, `tests/test_packs.py` and `tests/test_lmstudio.py` all mid-edit
(uncommitted, outside this unit's fences and never touched by it), and a full-suite count taken at
one moment (`807 passed, 3 deselected, 4 failed`, all four failures inside `tests/test_packs.py`)
had already moved by the next run (`764 passed`, then `801 passed`, then `807 passed`, the failure
set itself changing between runs, at one point including two `test_lmstudio.py` cases). None of
those failures are in this unit's files or attributable to this change — the scoped run above,
which exercises exactly what this unit shipped, is the number this entry stands behind.

**Done-conditions in this unit's scope that could not be executed — blocked on a live LM Studio
session, per this task's constraint that agents are not authorised to load a model.** Both are
R-1's (plan §4 S2, §6 R-1): (1) *"with a model actually loaded ... re-read `GET /api/v0/models`
and record ... whether the loaded entry exposes the KV-cache or load configuration"* — the
2026-09-03 probe saw only `not-loaded` catalog entries, and only a live session can re-probe a
loaded one; (2) *"does `loadedContextLength` appear on a loaded embeddings model"* — §2.3's
evidence is from a chat model only. Both are written as `@pytest.mark.live` tests in
`tests/test_hostinfo.py` (`test_live_loaded_catalog_entry_reveals_kv_cache_or_load_configuration`,
`test_live_loaded_context_length_on_a_loaded_embeddings_model`), deselected by default exactly
like `tests/test_lmstudio.py`'s existing live test, and never run by this unit. Per the plan,
"either outcome satisfies the condition; silence does not" — neither outcome is recorded here, and
whichever future live session runs them should record the finding in this file; a positive KV-
cache-on-load finding is a `fingerprint.py`/`AGENTS.md` change and a positive/negative
`loadedContextLength` finding is a `fingerprint.py` change, both out of this unit's fences either
way. **Not blocked, and not attempted for a different reason:** R-1's third question — whether
LM-Studio-reported `time_to_first_token` includes the JIT load — belongs to the runner's own
timing-budget instrumentation (`modelbench/runner.py`), a different not-yet-built unit, and is not
named anywhere in this unit's fenced files.

**Files:** `modelbench/hostinfo.py` (new), `tests/test_hostinfo.py` (new), `modelbench/cli.py`,
`tests/test_cli.py`. Left uncommitted for review; a concurrent session is editing
`modelbench/packs.py`/`modelbench/lmstudio.py` and their tests and may append its own entry to
this same `HISTORY.md`.

## 2026-09-09 — S2 U73: the row-count identity's own exemption widened to match its stated reach

**What:** `_row_count_identity_problems` (`modelbench/packs.py`) skipped silently — returned `[]`
— for *any* pack missing `data.conversations`, including a `scripts`-declaring (conversation-
shaped) pack that simply omitted the key; only an item-level pack (no `scripts` declared at all)
was the shape its own docstring named as the intended exemption. Verified by execution before
changing anything: a pack copied from `tests/fixtures/packs/valid` with `data.conversations`
deleted from the manifest but `scripts`/`replicatesPerScript`/`analysisUnit` intact passed
`validate_pack` with no problems. Closed per option (a) — widened the mechanism rather than the
docstring — because §3.3's own manifest literal and the "conversation pack" run-shape rule
(`docs/plans/small-model-benchmarking.md` §3.9 point 2) both tie `scripts` to conversation shape,
so a `scripts`-declaring pack without a rows file is a real problem, not "nothing to check."

**Change:** `_row_count_identity_problems` now skips only when `scripts` itself is absent or not a
plain int (the item-level exemption, unchanged). A pack that declares `scripts` but has no
`data.conversations` now gets exactly one reported problem naming the gap; `replicatesPerScript`
and `analysisUnit` validity are unchanged and still skip silently when malformed (left out of this
fix's scope). One existing fixture, `tests/fixtures/packs/replicates_per_script_violation`, had
declared `scripts: 6, replicatesPerScript: 2` with no `data` block at all — a pre-existing instance
of the same gap, invisible only because the route was exempt — so it gained a `data.conversations`
key and a matching 12-row `conversations.jsonl` (6 distinct `scriptId` values × 2) to keep its
Rule‑6‑only test isolated to the one violation it's meant to pin.

**Test-first, confirmed failing for the stated reason.** New fixture
`tests/fixtures/packs/missing_data_conversations/` (declares `scripts`/`replicatesPerScript`/
`analysisUnit`, no `data` block) and
`test_validate_pack_rejects_a_scripts_declaring_pack_missing_data_conversations` in
`tests/test_packs.py`, run alone against the pre-fix code: `assert [] == ['fixture-mis...']` —
red for the right reason before any production change.

**Mutations, both caught, each `cp`-aside / mutate / run / `cp`-back, `diff -q` byte-identical
after each restore:** (1) the whole file reverted to the pre-fix version — the new test alone goes
red with the exact same `[] == [...]` failure observed pre-fix, the sibling
`test_validate_pack_rejects_replicates_per_script_greater_than_one` still passes; (2) with the fix
in place, `declares_scripts` forced to the constant `True` — 5 of 29 `test_packs.py` tests redden
(`test_validate_pack_rejects_analysis_unit_outside_pairing_key_structurally`, both call-surface
rejection tests, the bad-import test, and `test_validate_pack_accepts_a_module_importing_
modelbench_tooling`), proving the item-level fixtures' green results depend on the exemption guard
rather than passing by accident.

**Observed, this run:** `model-bench/` as working directory. Baseline before any change:
`708 passed, 1 deselected`. After the fix, full suite: `709 passed, 1 deselected` (the one new
test; no other count moved). `.venv/bin/ruff check .`: `All checks passed!`.

**Files:** `modelbench/packs.py`, `tests/test_packs.py`,
`tests/fixtures/packs/missing_data_conversations/pack.json` (new),
`tests/fixtures/packs/replicates_per_script_violation/pack.json` and `conversations.jsonl` (new
data file). Left uncommitted for review; a separate concurrent unit appends its own entry to this
same `HISTORY.md`.

## 2026-09-09 — S2 U72: the LM Studio adapter, offline against stubbed HTTP

**What:** `docs/plans/small-model-benchmarking.md` §3.4.4a/§3.6's LM Studio adapter, landing beside
a concurrent unit building `modelbench/packs.py` (not touched here). `modelbench/lmstudio.py`
(new), `tests/test_lmstudio.py` (new), `tests/fixtures/lmstudio/` (new). `tests/test_lmstudio.py`
contributes **32 selected tests plus 1 deselected** (`-m live`) to the suite — this unit's own
attributable delta; the suite-wide total is reported separately, in the concurrent pack-loader
unit's entry above. `.venv/bin/ruff check .` clean on this unit's files. **Twelve mutations, all
caught**, each `cp`-aside / mutate / run / `cp`-back restore, diffed byte-identical against the
pre-mutation file after every single one: dropping the seconds→ms conversion; wrongly converting
`tokensPerSecond` (the one figure that must stay a raw rate); `_seconds_to_ms` returning `0.0`
instead of `None` on a missing key; removing the `stats or {}` guard so construction raises on a
missing `stats`; inverting `toolCallForm`'s native/prose branch; widening the eligibility gate's
`type` predicate to admit `"embeddings"`; changing the gate's scope constant from `"tool-caller"`
to `"embedder"`; flipping `residency()`'s filter operator; inverting `warm_up`'s residency
membership test; moving `warm_up`'s residency probe to *after* the chat call (an ordering defect,
not a branch); computing `EmbedResult.dimension` from `len(vectors)` instead of the first vector;
and swapping which raw `stats` key feeds `ttftMs` versus `generationMs`.

**One test caught not pinning what it claimed — by mutation, before review, not after.** The first
version of `test_warm_up_checks_residency_before_issuing_the_call_not_after` stubbed the residency
catalog to change on the *second* call to `/api/v0/models`, but `warm_up` only ever calls that
endpoint once regardless of where the call sits relative to the chat request — so the ordering
mutation above (residency probed after the timed call instead of before) left the test green. Its
name and docstring claimed to pin the ordering; its assertions did not. Rewritten to key the stub's
answer on whether the chat call has actually fired yet, re-confirmed green against the correct
implementation, then re-confirmed it fails under the same mutation that previously slipped past it.
Recorded because this is the first time on this coordination the *test-whose-name-outruns-its-
assertions* class was caught by the implementer during mutation testing rather than at review.

**Delivered.** `LMStudio(base_url, *, opener=urllib.request.urlopen)` — `base_url` a constructor
parameter (the not-yet-built `hostinfo` unit supplies it from `host.json`); `opener` injectable the
way `falkorchat/transport.py`'s HTTP transport is, so every test but the one `-m live` test stays
offline. `catalog()`, `residency()`, `probe()` (`GET /api/v0/models`, filtered, and the two-step
reachability probe — `"api-v0"` / `"v1-only"` / `"unreachable"`, all three tested against stubbed
HTTP); `chat()`, `embed()`, `warm_up()` with `timeout_s` required and **no default** on all three
(§3.6's two budgets belong to the runner, a later unit). Deliberately **no `load`/`unload`/`ps`**
— the CLI is gone and nothing on either HTTP surface can unload a model.

`ChatResult` normalises LM Studio's seconds-valued `stats.time_to_first_token`/`generation_time`
into `ttftMs`/`generationMs` on construction (§3.6's unit boundary, plan-gate P4-1); `tokensPerSecond`
is the one figure left unconverted. Each of the three is `None` — never `0` — when its source key
is absent, and construction never raises on a missing or partial `stats` object (plan-gate P5-8),
mutation-tested three separate ways above. `ChatResult.toolCallForm` (`"native"` / `"prose"`,
FR-8(b)) is decided at the transport boundary, on the one fact only this layer can observe
directly — whether the response used LM Studio's native `tool_calls` mechanism — rather than
deferred to a later prose-heuristic scorer. Transport failures raise one of two distinguishable
exceptions, `LMStudioCallTimeout` versus `LMStudioCallFailed`, matching §3.6's "timeout" versus
"no_response" dispositions.

`tool_calling_eligible(model_info)` / `check_tool_calling_eligibility(role, model_info)` implement
§3.6's eligibility gate. `role` is a plain `str`, **never a `Pack` object** — `packs.py` was a
concurrent unit this wave and its shape was not final; the wiring unit is expected to call
`check_tool_calling_eligibility(pack.role, model_info)`. Tested against the three real catalog
entries that break the naive `"tool_use" in capabilities` rule (an `embeddings` model advertising
`tool_use` → refused; an entry with no `capabilities` key → admitted; an `llm` with `tool_use` →
admitted), plus the v1.11/plan-gate-P5-1 negative case: the same `embeddings`-advertising-`tool_use`
entry is admitted, un-gated, on an `embedder` pack.

**Fixture note.** No literal captured `GET /api/v0/models` 19-model payload exists anywhere in this
repo's docs (checked: plan §2.5, review Pass 1 Appendix A.2, review Pass 4 Appendix D.3 — all
narrative descriptions of a live probe, never a saved response). `tests/fixtures/lmstudio/catalog.json`
holds the 7 entries the docs record a field for, each cited in a `_provenance` block, rather than a
fixture padded to 19 with invented models. The "llm with `tool_use`" entry (`qwen/qwen3-4b-2507`)
reuses `tests/conftest.py`'s own established S1 fixture precedent for that model (`modelType: "llm"`,
`modelCapabilities: ["tool_use"]`) rather than an independent capture — worth closing properly in a
future pass over the plan's fixture framing, not fixed here.

**Not this unit's:** `run`'s capture-order sequence, the two timing budgets in anger,
`coldLoadSeconds`, the withholding dispositions, `LatencyBlock`, and everything in `hostinfo.py` are
the runner/host-info units'. One `-m live` test is written
(`test_live_catalog_and_chat_stats_against_a_real_lm_studio`) and deselected by default — not run;
no model may be loaded by an agent.

## 2026-09-09 — S2 U71: the real pack loader (`load_pack`, `content_hash`, `validate_pack`)

**What:** `docs/plans/small-model-benchmarking.md` §4 S2's pack-loader portion — the first S2
unit, landing beside a concurrent unit building `modelbench/lmstudio.py` (not touched here).
`modelbench/packs.py`, `tests/test_packs.py` (new), `tests/conftest.py` (`pack_fixture()` +
`PACKS_DIR`), nine fixture packs under `tests/fixtures/packs/`. **648 → 708 tests** (this unit
added 28, all in `test_packs.py`; the remainder of the combined 708 is the concurrent `lmstudio`
unit's, filed separately), `.venv/bin/ruff check .` clean, **13 mutations, all caught**, each
`cp`-aside / mutate / run / `cp`-back restore, diffed byte-identical against the pre-mutation file
after every single one.

**Delivered.** `Pack` (frozen dataclass: `packId`, `packVersion`, `role`, `contentHash: str`
total, `manifest`, `root`) with `data_path`, `load_tool_module` (`importlib.util.spec_from_file_
location`, never `sys.path`) and `ref()`; `load_pack(root)`; `content_hash(root)` (SHA-256 over
sorted, NUL-delimited relative paths and bytes, excluding `PROVENANCE.md` **and** any
`__pycache__` a prior `load_tool_module` call left behind — not named by the plan, added because a
bytecode cache is a loader side effect, never pack content, and would otherwise make identity
depend on whether some earlier process happened to import the pack); `validate_pack(pack) ->
list[str]` (`[]` means valid, `Fingerprint.validate()`'s own shape), covering three independent
axes: the `sampling` contract — structural, by calling `Pack.ref()` (which runs the existing
`check_sampling_contract` rather than re-implementing it, per impl review Pass 1 §4 item 6), the
row-count identity (reading `data.conversations` — the plan's own manifest key, see the correction
below), and `-ml` §3.4 Rule 6's `replicatesPerScript > 1` rejection; `callSurface` derivation from
`environment.requires` (factored into a standalone `derive_call_surface` for direct unit testing),
rejecting a pack declaring neither or both of `lmstudio-chat` / `lmstudio-embeddings`; and an AST
import allowlist (stdlib, via `sys.stdlib_module_names`, plus `modelbench.tooling` by name only —
that module does not exist yet, and the check is a syntactic `ast.parse`/`ast.walk`, never an
import, so it needs no dependency on it). `pack_ref_from_manifest` and `Pack.ref()` now share one
manifest-parsing routine (`_ref_from_manifest_fields`) so the two routes cannot silently diverge on
what a valid `sampling` block is; `pack_ref_from_manifest`'s externally observed messages are
unchanged. The §3.3 totality boundary is asserted directly: `load_pack(...).ref().contentHash` is
not `None` and equals `content_hash(root)`, while `pack_ref_from_manifest(...).contentHash` is
`None` — both halves mutation-tested independently.

**Correction (coordinator finding, same day, before acceptance): the row-count identity was dead
on every plan-conformant manifest.** The first pass keyed the check off `sampling.dataFile`, a key
this module invented — no manifest the plan specifies carries it (the check's own docstring claim
that "the structural route already covers a pack that omits it" was false for that specific key,
since nothing but this module's own code knew it existed), so the route silently returned `[]` on
every real pack shape, including the `row_count_violation` fixture built to exercise it. The
plan's tool-caller manifest literal (`docs/plans/small-model-benchmarking.md` line 435) already
names the key: `"data": {"conversations": "conversations.jsonl", ...}`, beside the matching
`sampling` block. Fixed to read `data.conversations` instead; `sampling.dataFile` is gone from
every fixture. Added `tests/fixtures/packs/undeclared_replication/` and a test for §3.3's own
worked example of why this route exists — `replicatesPerScript: 1` declared, four conversations
per script shipped, "the case that slips past Rule 6's declaration check and past Rule 1 at once"
— which Rule 6 does not catch (declared value is 1, not `> 1`) and the structural route does not
catch (`analysisUnit == pairingKey[0]` holds); only the fixed row-count route does. Mutation:
disabling the row-count call site entirely (`_sampling_problems` returning before
`_row_count_identity_problems`) now fails two tests built on plan-conformant manifests (no invented
key) — `test_validate_pack_rejects_the_row_count_identity_specifically` and
`test_validate_pack_rejects_undeclared_replication_row_count_only` — where under the pre-fix code
that same mutation was survivable, which was the whole finding.

**Not this unit's:** `run` cross-checking a pack's derived `callSurface` against a model's catalog
`type`, and `run` calling `validate_pack`'s AST check and failing closed, are the runner/CLI unit's
(§3.3, §3.4.4a) — `packs.py` only builds the check and makes it callable.
`modelbench/lmstudio.py` / `tests/test_lmstudio.py` (a concurrent S2 unit) and
`modelbench/tooling.py` (not yet built by anyone) were neither read nor depended on.

**Verification:** `.venv/bin/python -m pytest -q` from `model-bench/` → **708 passed, 1
deselected** (the concurrent unit's one `-m live` test), exit 0. `.venv/bin/ruff check .` → `All
checks passed!`.

## 2026-09-09 — the support clamp moves off the envelope's arms, onto the printed interval

**What:** `docs/plans/small-model-benchmarking.md` §4 S1e Table H, implementing `-ml` v1.19 §3.4
Rule 4a — S1's last table. `modelbench/stats.py`, `modelbench/report.py`, `tests/test_stats.py`,
`tests/test_report.py`. **635 → 648 tests** (13 added: 12 in `test_stats.py`, 1 in
`test_report.py`), `.venv/bin/ruff check modelbench tests` clean, **9 mutations, all caught**,
each `cp`-aside / mutate / run / `cp`-back / `diff -q` byte-identical restore.

**Delivered.** A support is a property of the *estimand*, so it is applied once, to the printed
interval, never to a composition's input. `envelope_arms` now widens both arms with `clamp=None`
and returns them unclamped; the private composer `_compose` clamps its own composed result to the
new `SUPPORT_DIFF_PROPORTIONS` constant and returns `(interval, bound_by)`; `bound_by` is now the
three-token `BoundBy` alias (`"MOVER-D"`, `"exact paired bootstrap"`, `"support bound"`), computed
from the composed *unclamped* value against the support on a strict comparison — expressed as
`lo != u_lo` against the already-clamped bound rather than re-spelling the support subscript a
second time, which is what keeps the plan's residuals 2 and 3 at their stated count of one each.
`conservative_envelope` returns `_compose(...)`'s first element; `verdict()` takes both halves of
`_compose`'s return rather than recomputing the attribution inline, closing impl-gate P8-5's
finding as collateral. `report.py`'s `- decided by:` renderer gives a `support bound` token the
support's own boundary value and no `p=` clause: `support bound (-1)`, pinned verbatim by the
note's assertion 10. Confirmed bit-identical against the note's ten assertions and its two
exhaustive sweeps (commutation and verdict-invariance, both at n=12) before writing any test.

**The table's own line pins were stale** (`stats.py` +5, `report.py` +~155, against the table's
`e162ba9` baseline) from three units landed since — verified by content-match, not by line number,
per this coordination's standing discipline. All six residuals reached their stated target
(2→0, 0→1, 0→1, 1→0, 1→0, 2→0); `test_neither_printed_bound_is_ever_tighter_than_either_arm` was
the one shipped test the edit falsifies, and its comparison moved to the support-clamped arms per
the table's own row. `report.py`'s edit landed in the same `_decided_by_line` renderer region as
`c926308`'s continuous-verdict branch but does not touch it — confirmed by reading `c926308`'s
diff, which never reaches `_decided_by_line`.

**Mutation table (9, all caught):** `envelope_arms` reinstating `clamp=(-1.0, 1.0)` on both arms
(3 tests); `_compose`'s strict comparison weakened to `<=`/`>=` (2 tests); the tie-break's own
`<=`/`>=` narrowed to `<`/`>` (1 test, Pass 8's mutation 6 restored); `_compose` dropping its own
clamp on return (2 tests); `conservative_envelope` returning `_compose(...)`'s second element
instead of the first (8 tests); `verdict()` passing `_compose(exact_arm, mover_arm)` — arguments
swapped — instead of `(mover_arm, exact_arm)` (3 tests, the data-plumbing shape the coordination
flagged); `_decided_by_line` dropping its `support bound` branch so it falls into the `p=` clause
(1 test); `_decided_by_line`'s `zip` over `SUPPORT_DIFF_PROPORTIONS` reversed (1 test); `_compose`'s
two support subscripts transposed (12 tests). No equivalent mutants.

**Not delivered, and not owed.** No number in any §3.8 pack moves — the note's headline is that
arms-versus-composed is immaterial to the statistics, verified over 173,472 combinations, and
nothing in this change disturbs that; `bound_by` is computed at report time and was never stored,
so no `migrate` step is owed (`grep -rFn bound_by modelbench/results.py` stayed at 0 throughout).

## 2026-09-09 — the report-side seam: `compare_report` routes a continuous verdict metric

**What:** `docs/plans/small-model-benchmarking.md` §4 S1's `compare_report` block and §3.3 (iv),
against `cf54f5b`. `modelbench/report.py`, `tests/test_report.py`. **628 → 634 tests** (7 added,
1 rewritten test removed — the placeholder it replaced pinned a raise that is no longer reached),
`.venv/bin/ruff check modelbench tests` clean, **10 mutations, all caught**, each `cp`-aside /
mutate / run / `cp`-back / `diff -q` byte-identical restore.

**Delivered — Table F's `report.py:623-789` row, its stated exception now closed.** Pass 1
resolves each pre-registered verdict metric's kind from its own arm aggregate (`_metric_kind`,
falling back to `b`'s aggregate when `a` declares none, and to `"binary"` when neither does — DC-10's
existing cross-check has already reconciled an arm's own aggregate against its own items by this
point, so the aggregate type is a type fact rather than a guess). Three branches follow: a
**homogeneous binary** family is the unchanged two-pass Holm flow; an **all-continuous** family
takes one difference per analysis unit (`_paired_diffs`, the continuous sibling of `_paired_rows`
— joins at the unit id rather than at item `pairingKey`, folding a unit's items into one value by
averaging, so a unit spanning more than one item gets the mean §3.2d asks for and a unit ≡ item
gets the identity) and hands `diffs` and the metric's own `ContinuousMetric`/`DistributionSummary`
`.support` to `stats.continuous_verdict()` — no `holm_steps`, no `mcnemar_exact`, no
`resolving_power` on this path; a **mixed** family is refused *whole* per §3.3 (iv): no member
verdicted, nothing excluded, each member's own block names its resolved kind, the Exploratory
section widens to include the whole family, the headline (if one of the refused members) prints
the same exploratory label instead of the false `_NO_PAIRED_DATA` fallback, and the family-wise
section states the refusal instead of a Holm claim that never happened. The same replacement
mechanism serves the all-continuous `k > 1` case, which states its correction was taken in the
interval instead. `scored_outcome`'s `MetricKindError` raise (Table F's placeholder) is untouched
as a contract but is no longer reached by a well-formed continuous member — the family loop now
resolves kind before choosing an extractor rather than always calling `scored_outcome`.

**Two guards the continuous branch owns that the binary branch's `rp is None` parallel doesn't
quite cover.** Zero paired units renders the existing `_NO_PAIRED_DATA` message (parity with
binary). Exactly **one** paired unit is a case `_NO_PAIRED_DATA` cannot state truthfully — one
unit *is* paired — and `continuous_verdict` refuses a one-unit interval outright (`-ml` §3.4 Rule
8, refusal 4), so a new message (`_ONE_PAIRED_UNIT`) names it rather than letting the `ValueError`
escape uncaught.

**One design call made and not escalated, per the coordination's own precedent for routine
ambiguity:** which arm's aggregate to prefer when both declare one for the same metric (`a`, with
`b` as fallback) and what a metric with no aggregate on either arm resolves to (`"binary"`,
matching the pre-Table-F assumption). Neither is pinned by the plan or the note; both are cheap to
reverse and untested by any fixture that would distinguish them from the alternative.

**Mutation table (10, all caught):** `_metric_kind` forced to always return `"binary"` (7 tests
caught — everything continuous- or mixed-family-shaped); `mixed_kinds` forced `False` (2 mixed
tests — falls through to the binary branch, which raises `MetricKindError` on the continuous
member); `continuous_family` forced `False` (5 continuous tests, same failure mode); `_paired_diffs`
mutated to take a unit's first item instead of the mean (the averaging test only — required
redesigning the test's fixture values first, since the original values happened to make first-item
and mean coincide); the zero-diffs and one-diff guards each disabled in turn (each caught by its
own test, surfacing as an uncaught `ValueError` from `stats.py` instead of a rendered message); the
Exploratory filter's `or mixed_kinds` removed (`IndexError`, section absent); the headline's
`mixed_kinds` branch removed (falls back to the old `_NO_PAIRED_DATA`-vs-`.text` logic and raises
`StopIteration` looking up a verdict that was never computed); both family-wise replacement
branches removed (all three of the mixed/continuous/mixed-headline tests — one via a direct
assertion, two via an `AttributeError` reading `.mcnemar_p` off a `ContinuousVerdict`); the mixed
per-member label forced to always read `"continuous"` (caught by the binary member's assertion);
`_BOOTSTRAP_B` changed from `10_000` (caught once a `B=10000` provenance assertion was added to
the routing test — the constant had no witness before that).

**Whole-diff cross-check:** re-read as one change. Checked and found no issue: `computed`'s
3-tuple shape holds a `None` `HolmStep` for every mixed/continuous entry, but the only loop that
reads the third field is the Holm table, itself gated to the branch that never appends such an
entry; the widened Exploratory filter (`m.name not in family or mixed_kinds`) only ever *adds*
family members for a genuinely refused family, never suppresses an already-true case; the
`tables`/`p_values`/`steps` locals are now scoped inside the binary `else:` branch with no use
outside it. Nothing else found.

**Line-pin drift, reported and not fixed** (routes to `architect`). The insertions before
`compare_report` (`_BOOTSTRAP_B`, `_metric_aggregate`/`_metric_kind`, `PairedDiffs`/
`_paired_diffs`, four message constants — about 155 lines) push every citation below them down by
that much, and the family loop itself grew from 167 to about 280 lines. Table F's own site table
(`docs/plans/small-model-benchmarking.md:3845-3847`) cites three now-stale locations against the
pre-this-unit tree: `report.py:211` (DC-10's kind-cross-check selector) is now `:342`;
`report.py:581-601` (the Arms table's `else`-split) is now approximately `:746-786`; and
`report.py:623-789` (`compare_report`'s Table F row itself — "the family loop and the two
renderers downstream of it") is now approximately `:810-1088`, `compare_report`'s def itself now
at `:664`. §3.3 (iv)'s own citations (`:719`, `:738`, `:751`, `:763-777`, etc.) are pinned to a
named historical commit (`5878014`) rather than the live tree, per that section's own v1.23
discipline, and do not drift.

**Amendment (U67) — a mutation-testing gap found during integration review, not by the suite.**
The delivered `stats.continuous_verdict(family=family, ...)` call at `report.py:901` survives
being mutated to `family=[metric]`: all 634 tests above still pass. That argument is not
cosmetic — `continuous_verdict()` derives its quantile levels from `k = len(family)`
(`_family_ci_levels`), so collapsing it to the metric's own singleton family sets `k = 1` and
silently skips the `k > 1` Bonferroni-in-the-interval correction §3.3 (iv) commits to: a `k = 3`
family would render at `k = 1` levels, publishing an interval too narrow and a verdict too
confident, with no visible symptom.
`tests/test_report.py::test_an_all_continuous_family_takes_its_correction_in_the_interval_not_a_ladder`
names that correction in its docstring, but its four assertions only check the explanatory
sentence's presence, the absence of a Holm claim, that both members print the bootstrap-decided
text, and that no arm is excluded — none of which observes interval width, so all four hold
unchanged under the mutation. Its docstring is corrected to say only what it actually proves (no
Holm ladder, the explanatory section, both members verdicted, no exclusion) and to point at the
new test for the correction itself. Closed by
`tests/test_report.py::test_continuous_verdict_receives_the_whole_family_not_just_the_metric`:
renders identical `mrr` data and seed once as a `k = 1` family and once as a `k = 2` family
alongside a second all-continuous metric, and asserts the `k = 2` interval comes out strictly
wider — same seed and identical per-unit differences mean the two runs bootstrap-resample
identically, so only the quantile levels can account for a width difference; `family=[metric]`
renders both at the `k = 1` levels and the assertion fails with a message naming the two
intervals and widths. Mutation-confirmed: `report.py:901`'s `family=family,` changed to
`family=[metric],` now fails exactly the new test (`1 failed, 634 passed`); `cp`-aside / mutate /
run / `cp`-back / `diff -q` byte-identical restore. **634 → 635 tests**,
`.venv/bin/ruff check modelbench tests` clean. No production code changed by this amendment —
`modelbench/report.py` and `modelbench/stats.py` are exactly as this entry originally left them.

## 2026-09-09 — `-ml` §3.4 Rule 8: `continuous_verdict()`, the continuous producer

**What:** `docs/plans/small-model-benchmarking-ml.md` §3.4 Rule 8 (v1.16-v1.19), against `2d23482`.
`modelbench/stats.py`, `tests/test_stats.py`. **600 → 628 tests**, `.venv/bin/ruff check modelbench
tests` clean, **14 mutations** (2 caught real gaps, both closed in this entry — see below), each
`cp`-aside / mutate / run / `cp`-back / `diff -q` byte-identical.

**Delivered:** `ContinuousVerdict` (13 fields — a sibling type to `Verdict`, never one with six
fields left `None`) and `continuous_verdict()`, the entry point for every continuous metric
(MRR, score separation). Two new module-private helpers it composes: `_family_ci_levels(alpha,
k)`, the exact-rational quantile pair `alpha/(2k)`, `1 - alpha/(2k)` recovered as
`Fraction(str(alpha_family))` (never `Fraction(alpha_family)` — the double's own binary value);
and `_support_clamp(support)`, the clamp of the *difference's* support `(lo - hi, hi - lo)`,
`None` for an unbounded metric. `continuous_verdict()` takes no `resolving`, `alpha_step`,
percentile levels or `clamp` — all four are derived internally or unrepresentable at the call
site — and refuses on five conditions: an empty or non-finite `diffs` and `design_effect < 1.0`
are inherited unchanged from `paired_bootstrap`/`paired_cluster_bootstrap` rather than
re-checked; a metric not in its own `family` and a degenerate `support` (`lo >= hi`) are this
function's own. It calls `paired_cluster_bootstrap` — Rule 8's stated entry point — with the
derived levels and clamp, so the pack's declared `design_effect` and the metric's `support` both
reach the one interval that is printed. `diff`, both CI bounds and the half-width print at three
decimal places (never "pp" — this path has no percentage-point convention to inherit from
`verdict()`), and `alpha_used` is `alpha_family / k`, the two-sided alpha the printed interval
was actually taken at.

**State recovered, not started over.** `tests/test_stats.py`'s 348-line red phase (22 test
definitions, 26 test instances with parametrization) predated this unit, landed by an agent
killed twice by platform failures; this unit added the green phase plus one test the mutation
pass found the red phase had missed (below), and one unrelated one-line fix (`import re`, used
by two of the inherited tests but never added — a `NameError` waiting under the first `re.search`
call, not an assertion).

**A gap the red phase's own mutation table did not close, found and closed in this unit.**
`test_continuous_verdict_mrr_worked_case_from_the_note` (`diffs = [1.0] * 10`) survived a mutation
that dropped the derived clamp entirely (`clamp=None` unconditionally): with every difference
identical, the bootstrap interval is a zero-width point at 1.0, and widening a zero half-width by
any `sqrt(DEFF)` is still zero — so the clamp never has anything to clamp in that construction,
and `v.ci == (1.0, 1.0)` holds with or without it. The test's own docstring claim ("the unclamped
widened interval would run off `[-1, 1]`") was false of the construction it describes. Not a wrong
assertion — it passes, correctly — just not evidence the clamp ran. Added
`test_continuous_verdict_clamp_actually_binds_when_diffs_have_variance` (diffs with real spread,
`design_effect=9.0`, where the unclamped widened upper bound is measured at `1.08` against a
clamped `1.0`) as a second, load-bearing witness; confirmed it kills the mutation the worked case
did not. **`teco` reviewed and accepted this finding, then asked for the false claim itself to be
corrected** — a green test with a false docstring is how the next reader concludes the clamp is
covered when it is not. `test_continuous_verdict_mrr_worked_case_from_the_note`'s docstring is
rewritten to say only what the construction proves (the worked case runs and returns the note's
published `(1.0, 1.0)`) and points at the clamp-binding test above for actual coverage; its
assertion is untouched.

**Mutations, and what each targeted:** the metric-not-in-family refusal · the degenerate-support
refusal (both `_support_clamp` and `continuous_verdict`'s own path to it) · the single-analysis-
unit refusal, plus a variant merging it with the empty-`diffs` check (`len(diffs) < 2` instead of
`== 1`) to confirm the two refusals' messages would otherwise collide — the empty case must reach
`paired_bootstrap`'s own "at least one difference" text, not the single-unit refusal's — the
inherited empty-`diffs`, non-finite-`diffs` and `design_effect < 1.0` refusals (each disabled at
its source in `paired_bootstrap`/`paired_cluster_bootstrap`) · `_family_ci_levels`'s
`Fraction(str(...))` recovery and its `2 * k` factor · `_support_clamp`'s difference conversion ·
`alpha_used`'s `/ k` (against `/ (2 * k)`) · the family size `k` itself (off-by-one) · the clamp
call-through (the gap above) · `distinguishable`'s strict `>`/`<` (against `>=`/`<=`).

**The strict-comparison mutation initially survived** — both existing worked tests place the CI
bounds strictly away from zero, so the mutation went unwitnessed at the boundary — **and this
entry first reported it as a deferred, low-priority gap on the (wrong) assumption that an exact
zero bound needed Monte-Carlo luck to construct.** `teco` verified it against the committed code
and pushed back: an **all-zero** `diffs` sample makes every bootstrap percentile exactly `0.0`
deterministically (the resample distribution is a single atom at zero, for any `B`/`seed`), which
is neither Monte-Carlo-dependent nor blocked on unbuilt work — so the deferral was not earned.
Added `test_continuous_verdict_is_not_distinguishable_at_an_exact_zero_boundary`
(`diffs=[0.0]*8`, `ci == (0.0, 0.0)`, `distinguishable is False`); confirmed it now kills the
`>=`/`<=` mutation and that the mutation survives the rest of the 627-test suite (628th being the
new test itself) — i.e. this test is the sole witness. `verdict()`, the pre-existing binary-path
producer, carries the identical strict `ci[0] > 0 or ci[1] < 0` convention with the identical
absence of a boundary test; that predates this unit and is out of its scope, and is recorded as a
follow-up for whichever unit next touches `verdict()` rather than fixed here.

**Whole-diff cross-check, done before declaring done: nothing found beyond the one gap above.**
Walked `stats.py` and `test_stats.py` together as one change — the ordering of
`continuous_verdict`'s own checks (family membership, then `_support_clamp`, then the
single-unit refusal, then the call into `paired_cluster_bootstrap`) against every refusal test's
combination of valid/invalid arguments (no two refusal conditions are ever true in the same test,
so ordering never changes which message a test sees); the module docstring's rule count against
the new `continuous_verdict()` addition (updated "seven" to "eight", added item 8); `_plural`,
`Basis` and `Literal` reuse against their existing single definitions (no duplicate homes).

**Line-pin drift — reported, not fixed (routes to `architect`).** The module docstring's 5-line
Rule 8 addition sits above every function in the file, so every `stats.py:<line>` citation in
`docs/plans/small-model-benchmarking-ml.md` shifts by **+5**: `paired_bootstrap` `149` → `154`,
`paired_cluster_bootstrap` `202` → `207`, `_widen` `246` → `251`, and §11.2.2's `stats.py:159` →
`164`. New code was appended at the file's end, so nothing else moves.

## 2026-09-08 — §4 S1e Table F: the continuous carrier lands, scoped to its own proof surface

**What:** `docs/plans/small-model-benchmarking.md` §4 S1e **Table F** (v1.23, plan-gate P6-1), the
last of the eight S1e tables, against `e162ba9`. `modelbench/results.py`, `modelbench/report.py`,
`tests/test_results.py`, `tests/test_report.py`. **577 → 600 tests**, `.venv/bin/ruff check .`
clean, **6 mutations**, each `cp`-aside / mutate / run / `cp`-back / `diff -q` byte-identical.

**Delivered — the carrier and its two report-side readers, all 11 site rows** except one row's
continuous-verdict branch (below): `ItemResult.measures: Mapping[str, float]` beside `counts`, with
`__post_init__` refusing a metric name present in both maps (`MetricKindError`) and a non-finite
measure (`NonFiniteMeasure`); `scored_value(metric) -> float | None`, `scored_outcome`'s sibling
over `measures` with the same three states; `scored_outcome` itself now **raising**
`MetricKindError` on a `measures`-resident metric instead of booleanising it; `ContinuousMetric`
gains `support: tuple[float, float] | None`, required with no default; the new
`DistributionSummary(name, median, p10, n, unit, support)`, frozen; `RetrievalAggregates`'s
`separationRaw`/`separationZ` retype `float | None` → `DistributionSummary | None` and
`named_metrics()` returns both — `sep_z` reaches a table at all for the first time; the metric
(de)serialisers gain a **module-level tag→decoder mapping** (`_METRIC_DECODERS`) that
`_metric_from_dict` dispatches on and `_decode` gates on, so an unrecognised `"type"` **raises**
where it used to fall through as a raw `dict`; `_index_row`'s metrics cell renders a
`DistributionSummary` as `{name}=p50 {median:.4f}` rather than reading `.mean` (`AttributeError`
before this table). On the `report.py` side: DC-10's selector (`_aggregate_item_mismatches`)
widens past `isinstance(metric, BinaryMetric)` to a third arithmetic over `scored_value` for a
continuous member, with **no unit filter** (neither `ContinuousMetric` nor `DistributionSummary`
carries a denominator noun) — the same check is where a kind disagreement between an arm's
aggregate and its own items surfaces, in either direction, and both `IncompleteItemRecord` and
`MetricKindError` are caught there so neither escapes as a traceback; and the Arms table's bare
`else` splits so a `DistributionSummary` renders its median and p10, never `.mean`.

**Deliberately not delivered, per an explicit scope decision (Option A) put to the stakeholder and
confirmed before implementation.** Table F's own two-file scope statement ("the continuous carrier:
`modelbench/results.py` and `modelbench/report.py`") and its 8 enumerating commands / 3 residuals
never touch `modelbench/stats.py` — but the `report.py:623-789` row's own text ("pass 1 resolves
each member's kind and branches, per §4 S1's `compare_report` block") cites a spec that requires
calling `-ml` §3.4 Rule 8's `continuous_verdict()` and rendering its `ContinuousVerdict` sibling
type, neither of which exists anywhere in the tree (confirmed: only five docstring/comment
citations in `stats.py`/`test_stats.py`, no implementation, no test). Building it means authoring a
new statistical producer from a separate ~3,600-line note this unit was not scoped to implement, and
DC-13(e) / §3.3(iv)'s mixed-kind-family refusal depends on it too. **That gap is not silently
absorbed:** the still-binary family loop's `_paired_rows` calls `scored_outcome` on every family
member unconditionally, and the moment a pack declares a continuous `verdictMetrics` member,
`scored_outcome`'s new raise fires immediately and uncaught — converting "silently wrong when a
pack finally arrives" into "refuses loudly right now", which is the property this table exists to
guarantee ahead of that pack existing. `tests/test_report.py::test_a_continuous_verdict_member_refuses_loudly_rather_than_booleanising`
pins exactly this and doubles as the seam description for the follow-up unit: replace that raise
with a resolved-kind branch that calls `continuous_verdict()` instead of `_paired_rows` for a
continuous member. **A separate, properly-sized unit builds `continuous_verdict()`/`ContinuousVerdict`
from `-ml` §3.4 Rule 8, the family-loop continuous branch, and §3.3(iv)'s mixed-kind refusal, landing
before S1 closes.**

**Residuals — all three, before → after, re-run at `e162ba9` and again after the edit:**

| # | Command | Before | After |
|---|---|---|---|
| 1 | `grep -nF 'separationRaw: float \| None' modelbench/results.py` | 1 | 0 |
| 2 | `grep -nF 'separationZ: float \| None' modelbench/results.py` | 1 | 0 |
| 3 | `grep -rFn '{"binary", "continuous"}' modelbench --include='*.py'` | 1 | 0 |

**Whole-diff cross-check, done before declaring done: nothing found.** Walked every pair of edits
across both files for interaction — `_decode`'s widened `"type" in value` gate against every
`Aggregates` field's possible encoded shape (no field anywhere holds a bare dict with an unrelated
`"type"` key; `TurnPositionRate`'s `{"turnIndex", "metric"}` shape carries no `"type"` at its own
level, so the two branches stay mutually exclusive); `__post_init__`'s new refusal against all 14
existing `ItemResult(` fixture sites (none pass `measures`, so the overlap check is vacuously
satisfied everywhere unchanged — confirmed by the unchanged 577 continuing to pass unmodified); DC-10's
widened `continuous` selector against `_arm_label`/pooled-count rendering (untouched, since neither
reads the new flag); and `scored_outcome`'s ordering (`measures` check before the `counts` check) is
safe only because `__post_init__` already guarantees the two maps are disjoint — checked explicitly
rather than assumed. No self-created defect found.

**Line-pin drift — reported, not fixed; re-pinning is the coordinator's to route.** My insertions
in `results.py` and `report.py` are additive-only (no deletions), so every pin below moved by a
constant positive offset within its file, confirmed by locating each pinned line's exact text
rather than by arithmetic:

| Owner (unaffected by this unit) | Pin (old, `e162ba9`) | New | What's there |
|---|---|---|---|
| Appendix A | `results.py:466` | `results.py:628` | `FieldProblem(field=f"items[{item.itemId}].counts.{metric}", …)` |
| Table F's own prose (self-citation, now inside a landed table's body) | `results.py:327` | `results.py:427` | "A `KeyError` here surfaces as `unparseable`…" comment |
| Table F's own prose (self-citation) | `results.py:365-366` | `results.py:487-488` | the `BinaryMetric.unit` no-`.get`-fallback comment, cited by analogy for `support` |
| Table B (**landed**, `8fc2341`) | `test_results.py:62`, `:71`, `:239` | `:69`, `:78`, `:246` | the `Fingerprint(…)`/`_run(…)` construction sites the row names |
| Table B (**landed**, `8fc2341`) | `test_report.py:471`, `:486` | `:479`, `:494` | the two `arm_kind="deterministic",` fixture sites the row names |

No pin inside a table still describing an **instruction** (an un-landed row) was found stale by
this edit — Table H's `report.py:338`/`:927`/`:1199` sites are in `stats.py`/`report.py`'s
`_decided_by` region, which this table's `report.py` edits (`:211`, Arms table) do not overlap, per
Table F's own "meets Table H on `report.py`, neither order constrained" note.

**Mutation table** — each mutation `cp`-aside first, then a targeted disable (`if False and …` or a
silent-fallthrough rewrite), the pinned test(s) re-run alone, then restored by `cp` from the
untouched copy and `diff -q` confirmed byte-identical before the next mutation:

| # | Mutation | Result |
|---|---|---|
| M1 | `scored_outcome`'s `measures` check disabled | killed both `test_scored_outcome_raises_on_a_measures_resident_metric` and `test_a_continuous_verdict_member_refuses_loudly_rather_than_booleanising` |
| M2 | `__post_init__`'s both-maps overlap check disabled | killed `test_a_metric_name_present_in_both_maps_is_refused_at_construction` |
| M3 | `__post_init__`'s `math.isfinite` check disabled | killed `test_a_non_finite_measure_is_refused_at_construction` |
| M4 | `_metric_from_dict`'s unrecognised-tag raise replaced with a silent `return d` | killed `test_an_unrecognised_metric_type_tag_raises_rather_than_returning_a_raw_dict` and `test_a_record_with_an_unrecognised_metric_type_is_quarantined_as_unparseable` |
| M5 | DC-10's `continuous` selector forced to `False` | killed `test_a_continuous_member_declaring_a_measure_it_does_not_carry_is_a_mismatch` and `test_a_kind_disagreement_continuous_aggregate_binary_items_is_the_same_mismatch_class`; correctly left `test_a_continuous_verdict_member_refuses_loudly_rather_than_booleanising` passing (a different site) |
| M6 | Arms table's `DistributionSummary` branch disabled | killed `test_arms_table_renders_a_distribution_summary_without_reading_mean` (reproduces the exact `AttributeError` the site row describes) |

**Verification:** `.venv/bin/pytest -q` from `model-bench/` → **600 passed** (577 baseline + 23 new:
18 in `test_results.py`, 5 in `test_report.py`), 0 failed / 0 skipped / 0 deselected.
`.venv/bin/ruff check modelbench tests` → `All checks passed!`.

**What:** the last pass of `docs/reviews/small-model-benchmarking-impl.md` (gated as `## Pass 11`,
being renumbered to `## Pass 10`) against `93b0e42` — **N4** (major) and **N5** (minor), both in
scope and both closed. `modelbench/stats.py`, `tests/test_stats.py`. **560 → 577 tests**,
`.venv/bin/ruff check .` clean, **6 mutations plus one forward-compatibility probe**, run one at a
time. P8-1 remains held, blocked on `-ml` §3.4 Rule 4a's own unit.

**N4 (major) — `+inf`, or a non-finite difference, passed every guard and printed as a real
interval.** Rule 4's four precondition-4 guards close the *below* side of `>= 1.0` by construction.
Nothing closed the *above* side or the data. `not inf >= 1.0` is `False`, so `+inf` satisfies the
rule the note states; `sqrt(inf)` is `inf`; `_widen` returns `(-inf, +inf)`; and the clamp returns
the support bounds. All reproduced against the untouched tree before any fix:
`envelope_arms((4,5,3,0), design_effect=inf)` → `((-1.0, 1.0), (-1.0, 1.0))`;
`verdict(...)` at `deff=inf` → **`ci = (-1.0, 1.0)`, `bound_by = ('MOVER-D', 'MOVER-D')`**, a
full-support interval attributed to a named instrument; and one `nan` among `diffs` →
`paired_cluster_bootstrap(...)` → `(-1.0, 1.0)`, on a surface with **no guard on its data at all**.

**The mechanism is the clamp, not the predicate.** Every comparison with a NaN is `False`, so
`max(-1.0, nan)` is `-1.0` and `min(1.0, nan)` is `1.0`: the clamp converts *no number* into *the
widest honest number*, silently and in the direction that prints. Unclamped, the same input returns
`(nan, nan)` — visibly wrong. That contrast is what identifies the launderer, and both clamp
settings are swept in the test for exactly that reason.

**Fixed with a result guard in `_widen`, and this was the round's design call.** The alternative was
an input guard: extend the four precondition-4 predicates to reject `+inf`. Rejected, for two
reasons. **(1) Completeness.** The harm is one transformation — a non-number becoming a plausible
bound — and it happens at one place. A guard there closes every upstream cause, including ones
nobody has enumerated; an input guard closes the causes we happened to think of, and the whole
finding is that the previous round enumerated three sites and missed the fourth path entirely.
The module's stated shape is *"written so the anti-conservative version does not typecheck"*, and
the bad **state** is a laundered bound, not a particular bad input. **(2) Cost.** Extending four
predicates means also rewriting four messages — each says "must be >= 1.0", which is false for
`inf` — at four sites, for strictly less coverage: it would not catch a non-finite *difference* at
all. The guard is on the **result** rather than on `_widen`'s inputs for the same completeness
reason; it also catches overflow, which no input check does.

**But one guard was not enough, and this is where the fix goes past the reviewed candidate.**
`paired_bootstrap` does **not** go through `_widen` — it is public, is `-ml` §3.2d's own quantile
surface, and returns before any widening — so the result guard cannot protect it. Measured, it was
the worst-looking of them: `paired_bootstrap([nan, 1.0, 0.0, -1.0], ...)` returned **`(-0.6, nan)`**,
not a pair of `nan`s but a plausible lower bound beside a `nan` upper, which is the shape most
likely to be read as a rendering glitch over a real interval. The data precondition therefore sits
on the function that reads the data, where it names itself. Two guards, each owning the precondition
it actually has.

**It survives Rule 4a — verified, not argued.** Under Rule 4a `envelope_arms` widens with
`clamp=None` and the clamp moves into the composer. A guard placed *on the clamp* would move with it
or vanish; this one is on `_widen`'s **result, before the clamp branch**, so it fires on the
unclamped path too. Simulated the edit — both `clamp=(-1.0, 1.0)` in `envelope_arms` set to
`clamp=None` — and re-ran the N4 tests: **17 passed**, guard still firing, source restored
byte-identical. Guarding before the branch is also why the unclamped `sep_z` path is covered, which
it was not before.

**N5 (minor) — the rejection domain was written out three times.** `[0.5, 0.25, 0.999999, 0.0,
-1.0, nan]` appeared at three `parametrize` marks; a seventh failure class meant three edits and
missing one is silent — the sweep still passes, one surface just stops being swept. Now one
module constant, `_SUB_ONE_DESIGN_EFFECTS`, whose docstring records *why* each of the six is there
(they partition the rejection domain by predicate-failure mode) and why `-inf` is deliberately
absent (`not -inf >= 1.0` is `True`, so it is behaviourally the `-1.0` row). Plan §3.9's rule on the
test side, third application this arc after `_percentile` and `_compose`. Verified by collecting:
the three sweeps now produce identical id lists.

**The accepting side is under test now.** Both existing sweeps looked only at the rejection side,
which is precisely where N4 hid — no sweep of *refused* values could ever have reached a value that
is **accepted**. `test_no_design_effect_ever_yields_a_bound_that_is_not_a_number` asserts the
invariant across both sides over a mixed domain: for any design effect, a surface either refuses it
or returns bounds that are numbers. Nothing in between — and "in between" is exactly where the
laundered `(-1.0, 1.0)` sat, accepted and not a number.

**Line pins — these moved, and there was no line-count-neutral option.** A new refusal is new lines.
Both guards sit **before** every pinned line, so all ten shift by the same **+31** (31 insertions,
0 deletions; `stats.py` 1418 → 1449 lines):

| Table | Pin (old) | New | Table | Pin (old) | New |
|---|---|---|---|---|---|
| E | `:261` | `:292` | — | `:896` | `:927` |
| E | `:382` | `:413` | H | `:1168` | `:1199` |
| E | `:387` | `:418` | H | `:1184-1187` | `:1215-1218` |
| G | `:413` | `:444` | | | |

Verified by locating each pinned line's exact text in the new file, not by adding 31 by hand; the
`:1184-1187` block was confirmed intact as a unit. **Re-pinning is the coordinator's to route.**

**Mutation table** — each `cp` aside, mutated, run alone, `cp` back, `diff -q` byte-identical before
the next. Both guards were verified test-first: 13 of the 17 new ids fail on the shipped source
before `stats.py` is touched.

| # | Mutation | Result |
|---|---|---|
| N4-M1 | `_widen`'s finiteness guard deleted | killed |
| N4-M2 | that guard moved **after** the clamp | killed — placement is load-bearing, not just presence |
| N4-M3 | `paired_bootstrap`'s data guard deleted | killed — the path `_widen` cannot see |
| N4-M4 | `isfinite` weakened to a NaN-only test (`b == b`) | killed — the `inf` half survives a half-fix |
| N5-M1 | one row dropped from `_SUB_ONE_DESIGN_EFFECTS` | survives by design — collection falls 28 → 24, i.e. **one edit reached all three sweeps**, which is the property N5 asked for and which no assertion can carry |
| N-M6 | **control** — P8-1's tie-break `<=,>=` → `<,>` | **SURVIVES, as intended** (577 passed) |
| probe | Rule 4a simulated (`clamp=None` in `envelope_arms`) | 17 N4 ids still pass — the guard survives the in-flight edit |

N4-M2 is the one worth keeping: it proves the guard's **placement** carries the fix. Moved one line
later, after the clamp, it inspects the laundered value and is blind — green on the defect it exists
to catch, which is the exact failure mode this arc has been finding all week.

## 2026-09-08 — Implementation-review Pass 9 fix round: the same respelling at the two remaining sites

**What:** the `## Pass 9` findings of `docs/reviews/small-model-benchmarking-impl.md` against
`7f865e2`, all three in scope — **N1**, **N2** (majors) and **N3** (nit). `modelbench/stats.py`,
`tests/test_stats.py`, `AGENTS.md`. **550 → 560 tests**, `.venv/bin/ruff check .` clean,
**5 mutations run one at a time**, 4 killed and 1 surviving **by design** (P8-1's control again).
P8-1 remains held, blocked on `-ml` v1.19 Rule 4a's separate unit.

**One sentence for both majors: the Pass 8 round diagnosed the predicate correctly and respelled it
at one of the three sites that carried it.** `resolving_power` was already NaN-safe before that
round — which is why the "two spellings now coexist" worry it was briefed against turned out
backwards, the gate having checked `cc28d48` and found the safe spelling already there, comment and
all. `envelope_arms` joined it. `verdict()` and `paired_cluster_bootstrap` did not, and both were
live NaN holes.

**N1 (major) — `verdict()`'s precondition 4 was NaN-blind, so the layer-ordering property the
previous round restored was false at exactly one value.** `stats.py:1126` spelled
`resolving.design_effect < 1.0`, which is `False` for a NaN, so the value fell through and
`envelope_arms` raised **its** message one layer down. Reproduced before fixing: at `deff=0.5`
`verdict()` raises `verdict() precondition 4: …`, at `deff=nan` it raises
`design_effect must be >= 1.0 (-ml §3.4 Rule 4, precondition 4)` — the inner layer's sentence. That
is precisely the trap review P3-11 installed this check to prevent, alive again at one input.
`test_the_two_envelope_refusals_name_which_layer_raised` could not see it because it shipped with
`deff: float = 0.5` as a **default argument** rather than a parametrization — twelve lines below a
sibling already swept over `nan` for this exact reason. Now parametrized over the same six values;
the `nan` row fails on the old predicate and passes on the new one.

**N2 (major) — `paired_cluster_bootstrap` returned the full support as an interval for a NaN design
effect.** `stats.py:228`, same `< 1.0`. Reproduced verbatim before fixing:
`paired_cluster_bootstrap([1.0, 0.0, -1.0, 1.0], design_effect=nan, B=50, seed=1,
clamp=(-1.0, 1.0), levels=(LEVEL_CI95_LO, LEVEL_CI95_HI))` returned **`(-1.0, 1.0)`** — `sqrt(nan)`
widens both bounds to `nan`, and the clamp's `max(-1.0, nan)`/`min(1.0, nan)` return the clamp's own
endpoints, so a number nobody supplied prints as a maximally wide real interval. This is the same
symptom that justified the previous round's deviation, and it needs **no `dataclasses.replace`
bypass to reach**: `design_effect` is a bare float parameter with no `resolving_power` in the path,
on `-ml` §3.2d's **continuous** entry point that Rule 8's `continuous_verdict()` is specified to
call with a value arriving from a pack manifest. `test_paired_cluster_bootstrap_refuses_a_design_
effect_below_one` gains the `parametrize` its envelope sibling twenty lines away already carried,
plus a `match="precondition 4"` it lacked.

**N3 (nit) — `AGENTS.md` described the NaN-safe guard using the NaN-unsafe spelling.** In the
always-loaded file, in the round whose whole finding is that `< 1.0` is the wrong predicate. Fixed
wider than the three words asked for, because N1 and N2 make the wider version the live constraint:
the bullet now names **all four sites** that spell it `not … >= 1.0`, says why (a NaN widens to
`nan`, which the clamp turns into full support), and says plainly *do not simplify any of them*.

**Line pins — none moved.** Plan §4 S1e Tables E, G and H pin `stats.py:261`, `:382`, `:387`,
`:413`, `:896`, `:1168` and `:1184-1187`. Both source edits were made **in place**, one line for
one line (`git diff --numstat` reads `2 2`), the file is 1418 lines before and after, and all ten
pinned lines are byte-identical — verified by diffing the extracted lines, not by inspection. **No
re-pinning is required.** The NaN rationale went into an end-of-line comment at each site rather
than a comment block precisely to keep the edits line-count-neutral; both lines are 93 and 95
characters against the project's `line-length = 100`.

**Attributed delta, +10**, entirely from turning two single-value tests into sweeps:
`test_the_two_envelope_refusals_name_which_layer_raised` 1 → 6 ids and
`test_paired_cluster_bootstrap_refuses_a_design_effect_below_one` 1 → 6 ids. No test was retired.

**Mutation table** — each `cp` aside, mutated, run alone, `cp` back, `diff -q` byte-identical before
the next. Both majors were verified test-first: the new test fails on the shipped predicate before
`stats.py` is touched.

| # | Mutation | Result |
|---|---|---|
| N1-M1 | `verdict()`'s guard reverted to the NaN-blind `< 1.0` | killed — `…which_layer_raised[nan]` |
| N1-M2 | `verdict()`'s message stops naming itself (layers indistinguishable) | killed — `…which_layer_raised[0.5]` |
| N2-M1 | `paired_cluster_bootstrap`'s guard reverted to `< 1.0` | killed — `…below_one[nan]` |
| N2-M2 | that guard deleted outright | killed — `…below_one[0.5]` |
| N-M5 | **control** — P8-1's tie-break `<=,>=` → `<,>` | **SURVIVES, as intended** (560 passed) |

N1-M1 and N2-M1 are the review's own acceptance test — break the guard back and confirm the new
assertion fires — and each is caught by the `nan` row alone, which is the evidence that the sweep
rather than a second example was the necessary shape. N1-M2 pins the property that makes the
ordering checkable at all: the two layers' messages must stay distinguishable.

## 2026-09-08 — Implementation-review Pass 8 fix round: two restored refusals, one composition, five new assertions

**What:** the `## Pass 8` findings of `docs/reviews/small-model-benchmarking-impl.md` against
`cc28d48`, scope **P8-2 … P8-7** — **P8-1 is excluded and untouched**, being adjudicated by a
`data-scientist` as a methodology question (whether the `(-1, 1)` support clamp belongs on the
envelope's *arms* or on the *composed* interval — ruled mid-round by `-ml` v1.19's new §3.4
Rule 4a, `a707d09`, and implemented in a separate unit). `modelbench/stats.py`,
`tests/{test_stats,test_report}.py`, `AGENTS.md`; `report.py` is unchanged — P8-2's loss was an
absent assertion, not a wrong renderer. **519 → 550 tests**, `.venv/bin/ruff check .` clean,
**14 mutations run one at a time**, 13 killed and 1 surviving **by design** (the control below).

**P8-2 (major) — the `mcnemar-exact` bullet was asserted nowhere.** `_decided_by_line` has exactly
three renderings over its domain and only two were pinned; mutating the `bound_by is None` branch
to a constant string left the whole suite green. The positive assertion had lived in
`test_the_seed_is_not_printed_where_no_bootstrap_decided_anything` and moved to neither of the two
tests that replaced it — the one genuine coverage loss in that round's seven-test churn, and on
the branch **every** `by-construction` comparison at DEFF 1.00 takes. Closed by the third
rendering's own named test rather than by a line appended to an unrelated one, plus the invariant
nothing held: `(bound_by is None) == (decided_by == "mcnemar-exact")`, asserted on both constructed
verdicts. `bound_by` is a discriminated union over `decided_by`, not a field that is `None` until
it is not, and `_decided_by_line` transcribes that discriminator a second time by branching on the
wrong one of the two — so they could only ever disagree silently.

**P8-3 (major) — `envelope_arms`/`conservative_envelope` had stopped refusing `design_effect < 1.0`.**
The refusal shipped at `c19f875` inside `paired_cluster_bootstrap`, the call v1.11's closed form
deleted, so it **retired by accident**: Table D retires only the `n != len(diffs)` guard. What got
through was not an exception but a narrower interval — `(34, 6, 0, 0)` at DEFF 0.5 returned
`[6.6, 25.0] pp` against `[3.2, 29.1]` at 1.00, anti-conservative in the direction that prints.
`verdict()` still guarded its own entry, so nothing that ran was wrong; both functions are public
and `envelope_arms` is new, so this was a defence-in-depth loss on a surface the plan exposes.
Restored with `paired_cluster_bootstrap`'s exact message and the **NaN-safe** predicate
(`not design_effect >= 1.0`) that `resolving_power` already documents: `< 1.0` is False for a NaN,
and `sqrt(nan)` then clamps both arms to full `(-1, 1)` support — a maximally wide interval
conjured out of a missing number. **The ordering half of
`test_verdict_refuses_a_design_effect_below_one` comes back with it**: a second raise on the same
precondition exists again, so that test's `match` is re-anchored on `verdict()`'s own prefix rather
than on the bare `precondition 4` both layers now share — mutation M13 confirms deleting
`verdict()`'s check is caught rather than masked by the layer below.

**P8-4 (minor) — the atom-boundary fixture.** The `>=` → `>` mutant on the exact quantile's
selector was reported as equivalent; it is not, only unreachable at `LEVEL_CI95_LO`/`LEVEL_CI95_HI`
(no tie exists there — swept). `(0, 1, 1, 0)` has CDF exactly `1/4, 3/4, 1` over `s/n ∈ {-1, 0, 1}`,
so a level can land **on** an atom boundary — the one place `inf{ v : F(v) >= p }` and
`inf{ v : F(v) > p }` differ, and the property the docstring's "the two agree by construction"
claim rests on. Parametrized on the boundary, below it, above it, and at `Fraction(1)`, which under
`>` raises `IndexError` instead of returning the largest atom.

**P8-5 (minor) — one composition, not two.** `conservative_envelope` and `verdict()` each spelled
`min(exact[0], mover[0]), max(exact[1], mover[1])` out for itself, while `envelope_arms`' docstring
claimed "neither recomputes the other's arithmetic". Both copies were independently pinned, so this
was never a live bug — it was plan §3.9's rule, the one that retired the two private percentiles,
violated at four lines instead of forty. Extracted to `_compose(mover, exact)`, called from both;
the docstring sentence is now true. The arithmetic is **unchanged** and the arms are still composed
bound by bound at the same place.

`-ml` v1.19's Rule 4a (`a707d09`) landed mid-round and closes P8-5 as collateral by the same
reasoning, so `_compose` is the seam its clamp pass builds on rather than a second composer to
reconcile. What that pass still has to add to it, and what this round deliberately did **not** do
(P8-1 being out of scope): clamp `_compose`'s **result** to `(-1.0, 1.0)` once `envelope_arms`
widens with `clamp=None`, widen its return to `(interval, bound_by)`, and move `verdict()`'s inline
`bound_by` computation inside it as Rule 4a's three-token closed set. `conservative_envelope` then
returns the first element. Rule 4a measures compose-and-clamp as exactly commuting over 173 472
combinations, so nothing built here on the current arithmetic moves: the printed numbers are
bit-identical under either placement.

**P8-6 (minor) — `exact_paired_quantiles`' refusals were a weaker second copy.** Its transposed-pair
guard was `paired_bootstrap`'s twin and untested — deleting the six lines was green. It was also
*weaker* than `percentile` on two refusals it did not carry: a `float` level reached `.numerator`
and died with `AttributeError` where `-ml` §11.2.2 publishes a `TypeError`, and a level above 1
fell off the end of the atom loop and died with `IndexError` on `bounds[1]` where the note publishes
a `ValueError`. Both are reachable from the signature, which advertises `levels` as the caller's.
Closed with a shared `_check_level(level, *, caller)` — the two are the same estimator over a sample
and over a known distribution (`-ml` §11.2 reason 2), so they owe a caller the same errors — and the
three refusal tests now **parametrize over the estimators** rather than naming one, so a third has
to opt out rather than be forgotten. The `(0, 1]` bound is also what makes the atom loop total:
at `level <= 1` the final cumulative always satisfies the selector, so `bounds[1]` always exists.

**P8-7 (nit) — `AGENTS.md`.** The F2 clause's disposition has since been **ruled on** by plan v1.17
(`b6f578c`), which closes F2 by restating the residual's target as two survivors named and says
explicitly that renaming is not the answer, the residual's stated virtue being that it survives a
rename. So the clause is ratified rather than stale, and what it gains is the live constraint an
editor would otherwise revert: *do not rename `exact_paired_quantiles` to make the grep read 1*.
The 131-character line 81 is rewrapped and now carries the two-layer design-effect refusal.

**Also closed, from Pass 8's unnumbered nit:** the `n <= 0` refusal was written in `envelope_arms`
and `exact_paired_quantiles` and `grep -rn 'describes no rows' tests/` returned nothing — no test
reached either. It is the surviving half of the intent behind the retired
`…refuses_a_table_that_does_not_describe_its_rows`, whose own guard was correctly retired as
unrepresentable; this one is still representable, the table being the caller's. Now pinned on all
three functions that take one.

**A visible figure moved at `cc28d48` and was not recorded there.** Table C's estimator swap changes
`index.csv`, not only the code: over `1..100`, `latencyMsP50` goes **51 → 50**, because the retired
`int(round(pct/100 · (X−1)))` and Hyndman-Fan type 1 disagree by one rank at even sample sizes.
Anyone comparing runs across that commit sees the shift; `latencyMsP95` is unmoved at that size.

**Mutation table** — each `cp` aside, mutated, run alone, `cp` back, `diff -q` byte-identical
before the next.

| # | Mutation | Result |
|---|---|---|
| M1 | `_decided_by_line`'s `bound_by is None` branch → a constant string | killed (**was surviving**) |
| M2 | `envelope_arms`' restored precondition-4 refusal deleted | killed |
| M3 | that refusal weakened to the NaN-admitting `< 1.0` spelling | killed (only the `nan` case) |
| M4 | atom selector `>=` → `>` | killed (**was surviving**) |
| M5 | `_compose`'s `min`/`max` swapped | killed |
| M6 | `verdict` re-spells the composition and it drifts | killed |
| M7 | `exact_paired_quantiles`' transposed-pair guard deleted | killed (**was surviving**) |
| M8 | its `_check_level` calls deleted | killed |
| M9 | `_check_level`'s `0 < level` opened to `0 <= level` | killed |
| M10 | the `bound_by` iff broken — mcnemar path names MOVER-D twice | killed |
| M11 | M10 again, scoped to the new stats invariant assertion | killed |
| M12 | `envelope_arms`' `n <= 0` refusal deleted | killed |
| M13 | `verdict()`'s **own** precondition-4 check deleted | killed — *not* masked by M2's layer |
| M14 | **control** — P8-1's tie-break `<=,>=` → `<,>` | **SURVIVES, as intended** |

M14 is the control, not a gap: P8-1 is out of this round's scope and its mutant surviving is the
evidence that the tie-break and the clamp placement were left for the `data-scientist`'s ruling.

## 2026-09-08 — S1e Tables C, D, E and G: one percentile, the closed-form paired interval, and two required parameters

**What:** `docs/plans/small-model-benchmarking.md` **§4 S1e Tables C, D, E and G** (plan v1.16),
against `-ml` v1.18. `modelbench/{stats,results,report,packs}.py` and
`tests/{test_stats,test_results,test_report}.py`. **487 → 519 tests**, `.venv/bin/ruff check .`
clean, **13 mutations run, 11 killed, 2 surviving and both reported**. Tables C and G were applied
in the mandated order — C first, because Table G's `Fraction` level handed to the shipped
`_percentile(ordered, pct: float)` reads `Fraction(1, 40)` as `0.025` where that signature means
`2.5`, a unit error one substitution away from a plausible number.

**Table C — one percentile.** `stats.percentile(values, *, level: Fraction)` replaces both shipped
`int(round(pct/100 · (X−1)))` copies: Hyndman-Fan type 1, the rank taken as one integer expression
over the level's numerator and denominator, sorting a copy of its input, refusing an empty sample,
a `float` level and a level outside `(0, 1]`. Four `LEVEL_*` constants are the whole literal level
space. `results.py` imports *that object* — asserted as identity, not as equal behaviour, which is
what kills a second copy that happens to agree. `_index_row` keeps its own emptiness test, because
whether a latency figure exists at all is a decision about the run; **both index cells still bypass
`-ml` §11's two floors and closing that is S2's**, when the runner builds the `LatencyBlock` §3.5
requires every latency cell to be copied from.

**Table G — the bootstrap's levels.** `levels: tuple[Fraction, Fraction]`, keyword-only and
required, on `paired_bootstrap` and `paired_cluster_bootstrap`, with a transposed pair raising —
the one error that otherwise returns a plausible *inverted* interval. An all-continuous family with
`k > 1` takes its Bonferroni correction in the interval and had nowhere to put it.

**Table E — the clamp.** `_widen` and `paired_cluster_bootstrap` take `clamp`, required with no
default, `None` meaning do not clamp. `[-1, 1]` is correct for a difference of proportions and
false for `sep_z`; left as literals it printed an upper bound of 1.0 beside a point estimate of
1.48, the point estimate outside its own interval.

**Table D — the seed retires from the paired binary path.** `conservative_envelope(table, *,
design_effect)`: no `diffs`, no `B`, no seed, and the `n != len(diffs)` guard retires by being made
unrepresentable. Its bootstrap arm is now `exact_paired_quantiles`, the exact multinomial resample
quantile in integer arithmetic — no float tie-break, no Monte-Carlo estimate of an atomic quantile.
Both published anchors reproduce: `(0, 6, 0, 34)` at DEFF 1.00 renders `[3.2, 29.1] pp` and
`(4, 5, 3, 0)` renders `[−27.1, 58.3] pp`. `DecidedBy`'s `cluster-bootstrap` becomes
`conservative-envelope` and every string that named one arm of a two-arm interval is swept; the
`- decided by:` bullet loses its seed parenthetical and gains the audit that replaces it — which
arm bound each bound, carried on `Verdict.bound_by`. `PackRef.seed` stays, its consumer moved to
`-ml` §3.2d's continuous bootstrap.

**Three residual findings, all handed to `architect` and none fixed here.** (1) Table E's two
residuals go **blind** after a faithful edit: they match the shipped text `max(-1.0, point …)` /
`min(1.0, point …)`, which the edit necessarily rewrites, so a `clamp[1]` left as the literal `1.0`
is caught by neither — measured, that half-application passed the whole suite. A test asserting
both clamp components against an arbitrary `(0.9, 1.5)` stands in its place. (2) Table C's third
residual is at **2, not its stated 1**, because Table D's mandated closed form is a second `def
…quantiles`; renaming it to reach 1 would defeat the very property that residual exists to have,
so the name stands. (3) Table D's first residual is at **1, not 0**: the surviving line is
`def test_cluster_bootstrap_seed_is_keyword_only_with_no_default`, a substring collision on a test
for `cluster_bootstrap` — a function Table D explicitly *keeps*.

**Two surviving mutations, both reported rather than papered over.** `>=` → `>` in the atom
selector is an equivalent mutant on the reachable domain (an exact tie needs `40·cum == n**n`).
Not widening the exact arm by `sqrt(DEFF)` survived the suite as delivered, because the envelope's
own widening test sits on `(34, 6, 0, 0)` where MOVER-D binds both bounds; a new test asserts each
arm's widening separately on the table where the exact arm binds, and kills it.

## 2026-09-07 — P6-1: the stored `callSurface` shape, made total over invalid records

**What:** `docs/reviews/small-model-benchmarking-impl.md` **`## Pass 6`** finding **P6-1** (minor,
the pass's only one — Pass 5's four majors are dispositioned *fixed* and P5-7 *withdrawn*).
`modelbench/fingerprint.py` and `tests/test_fingerprint.py` only. **475 → 477 tests**,
`.venv/bin/ruff check .` clean, **3 mutations run, 3 killed**.

**The defect.** `to_dict` keyed its omission on the *value* — `{} if self.callSurface is None` —
so it fired on a **model** record whose surface was `None` too. Such a record validates as `null`
("something had the value and lost it"), serialises with no key, and reads back as `absent`
("never written"): the information loss the previous round closed in `validate()`, reintroduced one
method over. Unreachable through this package's writers, because `store()` validates before
`RunResult.to_dict()` — and reachable through **`model-bench migrate`** (§3.4.3), which reads with
`from_dict` and writes with `to_dict`, and is the one writer that serialises records `store()`
never validated. *(Corrected at Pass 7: an earlier phrasing here said a migration "by definition
walks records that did not validate". False, and it originated in the Pass 6 finding — §3.4.3
validates a record against **its own** schema entry, so most migrated records are valid. What is
unguarded is the write side, not the read, and that is enough to make the defect reachable.)*

**The fix is the conjunction, not either half.** The gate suggested keying on the arm instead
(`{} if self.armKind == "deterministic"`), which is a strictly worse trade on the same path:
executed over all eight `(armKind, callSurface)` shapes, it drops a **forbidden** surface off a
deterministic record, which then reads back **valid** — laundering an invalid reference arm into a
clean one and deleting the evidence of the claim §3.4.1 exists to refuse. So the key is omitted for
**exactly one** record, `armKind == "deterministic" and callSurface is None`, and written for every
other, whatever it holds. The round trip is then total over **all twelve** shapes —
`armKind` over `model`, `deterministic`, the `""` sentinel and an unrecognised value, times
`callSurface` over `"chat"`, `""` and `null` — where the value-keyed condition is asymmetric on
three of them and the arm-keyed one on two. *(The count is the suite's, not a transcript's, since
Pass 7: `ROUND_TRIP_SHAPES` in `tests/test_fingerprint.py` is that product, and this paragraph
states what it covers. The round originally claimed eight, enumerated by hand, and an independent
enumeration found nine — neither was in the suite, which was the finding.)*

**The mutations**, each applied to a file copied aside and restored by copy immediately, the tree
verified byte-identical after every one:

| # | The wrong implementation | Result |
|---|---|---|
| 1 | omission keyed on the value alone — the shipped `c523a35` condition | 1 failed — killed (`model-lost-its-surface`) |
| 2 | omission keyed on the arm alone — **the Pass 6 gate's suggested one-liner** | 1 failed — killed (`reference-arm-claiming-a-surface`) |
| 3 | the key always written — Pass 5's M2 / Pass 6's N1, re-run against the fix | 1 failed — killed, so P5-2's pin still holds |
| 4 | `callSurface is None and armKind != "model"` — omits on an unrecognised arm kind, the same laundering one arm-kind over | 2 failed — killed |
| 5 | `armKind == "deterministic" and not callSurface` — omits on `("deterministic", "")`, so a `forbidden` record migrates in clean | 1 failed — killed |
| 6 | the key never written | 40 failed — killed |

Mutations 4 and 5 are **Pass 7's** (P7-1): they passed the two-case version of the round-trip test,
which pinned the two narrowings its author had tried rather than the rule itself. The test is now a
`parametrize` over the product, so a wrong `omit` fails at the cell that names the record it
mishandles.

**One thing recorded rather than fixed, as a trigger.** §3.4.1's "`None` **iff** deterministic" is
now transcribed in two places — `validate()`'s deterministic branch and `to_dict`'s `omit` — with
nothing tying them together, and *which* value a profile pins is a **schema** fact that `to_dict`
hard-codes. At schema 1 there is exactly one such pin and the duplication is cheap, so it stays.
**A third transcription, or a schema-2 profile pinning another discriminator, is the moment to lift
the rule into `REQUIRED_BY_SCHEMA` rather than the moment to write it again.**

## 2026-09-07 — S1e Tables A and B, fix round: three states for the second discriminator, and two decisions that were held by comments

**What:** `docs/reviews/small-model-benchmarking-impl.md` **`## Pass 5`** findings **P5-1**, **P5-2**
and **P5-4**, plus **P5-3**'s code half once plan **v1.16** (`1ed8599`) made it writable.
`modelbench/fingerprint.py` and `tests/test_fingerprint.py` only — no other file in the package
moved, and neither remaining S1e unit opens either. **472 → 475 tests**, `.venv/bin/ruff check .`
clean, **7 mutations run, 7 killed**.

**P5-1 — `callSurface` reported *absent*, *empty* and *null* as one failure, under a test comment
claiming it did not.** The collapse was in `from_dict`: `d.get("callSurface")` maps a missing key
and a stored `null` to the same `None`, so a record written by something that *had* the surface and
lost it was reported to the operator as one that never carried it. The fix is `armKind`'s own
mechanism, applied one field over: the missing-key sentinel is `""`, and `validate()` answers `null`
before it answers `absent`. **Two reasons for three stored shapes, and the suite now says which
two** — `empty` still collapses into `absent`, because `""` *is* the sentinel and the two states are
indistinguishable on this field by construction, where `null` is not. That is the honest half of the
old comment, which claimed a three-state discipline directly above an assertion pinning the
collapse.

**The one subtlety, and it is why the sentinel is chosen per record rather than globally.** `None`
is not an absence marker on a **deterministic** arm — it is that arm's *value* (§3.4.1, "`None` iff
deterministic"), and it is what `to_dict` omits. So `from_dict` reconstructs a missing key as `None`
on a deterministic record and as `""` on a model one; a sentinel applied to both would report a
correct reference-arm record as carrying a forbidden surface, and fails the round trip.

**P5-2 — `to_dict` omits `callSurface` on a deterministic arm rather than writing `null`, and
nothing held it there.** The decision is right — that arm calls no surface, which is a different
fact from "we did not capture this", the one thing `null` means in this record (§3.4.2) — but the
round-trip test cannot see it, because `from_dict` reads a stored `null` back as the same `None` the
omission restores. Writing `"callSurface": self.callSurface` unconditionally passed all 472 tests.
It is now pinned by an assertion on the stored shape, with its positive twin on both model profiles
so "omit it" cannot be over-applied into "never write it". This matters beyond tidiness: S2's runner
and S3's `load_history` read that shape off disk.

**P5-3 — the retired residency element is now asserted by value, both keys named.** Plan v1.16
restates S1 done-condition 1 over the element's **key set** and re-scopes §4 S1e Table A's second
residual to `modelbench` plus `tests/conftest.py`, which makes `tests/test_fingerprint.py` the one
place the retired `lms ps --json` literal may live — and therefore makes the assertion writable at
all. Before it, an extra-key loop carrying a one-name tolerance for either retired key passed the
entire suite: the rule was right in the code and held by nothing. The new test constructs the
retired element on **both** snapshots and names both keys' `forbidden` problems, plus the two keys
it lacks. The residual stays at its stated target: `grep -rFn sizeBytes modelbench tests/conftest.py
--include='*.py'` → **0**.

**P5-4 — the half-swap test did not detect the half-application it is named for.** It asserted
`!= []`, and each of its two cases is *also* missing one of `{id, state}`, which produces an
`absent` problem on its own — so the assertion was satisfied whether or not the extra-key rule
existed. Deleting that rule entirely left both of its cases green while failing six other
parameters. It now asserts the surviving retired key's own `forbidden` problem, and its size case
uses the real retired spelling rather than the stand-in v1.15 forced on it.

**The mutations.** Each was applied to a file copied aside and restored by copy immediately, never
by `git restore`; the tree was verified byte-identical after every one.

| # | The wrong implementation | Result |
|---|---|---|
| 1 | `validate()` collapses `null` back into `absent` | 2 failed — killed |
| 2 | `from_dict` drops the missing-key sentinel (`d.get("callSurface")`) | 2 failed — killed |
| 3 | `to_dict` writes `"callSurface"` unconditionally — the review's M2, which had survived | 1 failed — killed |
| 4 | extra-key loop tolerates the retired size key — the review's M1, which had survived | 3 failed — killed |
| 5 | extra-key loop tolerates the retired identity key | 5 failed — killed |
| 6 | the extra-key `forbidden` loop deleted — the review's M5 | **10** failed — killed, where before this round it was 6 and **both** half-swap cases passed |
| 7 | the missing-key sentinel reaches a deterministic record too | 2 failed — killed |

**Also swept:** `ProblemReason`'s comment named two of `unknown`'s three families and omitted the
non-list snapshot; plan v1.16's Appendix A now writes all three out, so the module docstring says
what the plan says.

**Not in this round.** P5-5 and P5-6 are plan-side and closed at v1.16. Nothing from Pass 5 is
carried.

## 2026-09-07 — S1e Tables A and B: the residency source, the third arm profile, and the element shape

**What:** Plan **v1.15** §4 S1e **Table A** (`lmsCliCommit` → `residencySource`) and **Table B**
(`armKind` → `armProfile`, and `ARM_KINDS` decoupled), the first of the S1 fix round's three
implementation units. `modelbench/fingerprint.py` plus its three test surfaces; nothing else in
the package moved. **389 → 472 tests**, `.venv/bin/ruff check .` clean, all six of the two tables'
residuals at their stated target of zero, and **11 mutations run, 11 killed**.

**Table A — one field swapped, one element shape enforced.** `lmsCliCommit` recorded which `lms`
build produced the residency snapshot; after plan §3.4.4a nothing in the harness runs that CLI, so
the field had no source and could only have been kept by defaulting it to `""` — the
silently-defaulted fingerprint field FR-7 exists to refuse. `residencySource` replaces it
one-for-one on both model profiles: a `nonempty` token naming the surface the residency and catalog
data actually came from. The count is unchanged at 26 + 4 = 30.

**The fourth site carried no token, and it is the one that mattered.** `tests/conftest.py`'s
`residentModelsAtEnd` fixture declared the retired `lms ps --json` element where §3.4.4a's shape is
`{id, state}`. That field's tier is `present`, which checks presence and **never element shape**, so
the stale element validated, shipped green, and would have travelled into S2 — where `residency()`
emits `{id, state}` and the two disagree with nothing to catch them. The structural fix is
therefore not the fixture edit but the missing assertion (plan S1 done-condition 1):
`validate()` now checks each residency element's **whole key set** — exactly `{id, state}`, both
non-empty strings — so any key outside the pair is `forbidden`, a missing one `absent`, a `None`
`null`, and a non-string `unknown`. Problems name the element and key they came from
(`residentModelsAtEnd[0].modelKey`), which is what lets AC-2's block print them. Reverting the
fixture to the retired element now fails **63** tests where it previously failed none.

**Why that check has no residual over `modelKey`.** The retired element carried two keys and
Table A states a residual over only one of them, because `modelKey` keeps its meaning as a required
field of the same record — 90 lines in the tree — so a residual over it would fail on a *faithful*
edit, which is the trap §7 rule 5(b) forbids. The element-shape assertion is rule 5(b)'s named
alternative and covers that half: a half-application that swaps one retired key and keeps the other
fails there rather than in a count.

**Table B — the mapping key becomes a profile, the arm kind does not.** `REQUIRED_BY_SCHEMA[1]` is
re-keyed to the three profiles `model:chat` (30 fields), `model:embeddings` (26 — the chat set minus
`runtimeName`, `runtimeVersion`, `temperature`, `maxTokens`) and `deterministic` (11).
`FORBIDDEN_BY_ARM_KIND` becomes `FORBIDDEN_BY_ARM_PROFILE`, still the union-minus-mine **set
operation** and never a list, now over three profiles instead of two kinds. It resolves to exactly
what §3.4.1's table says, checked against independently written literals in the suite:
`{armId, armParametersHash}` on `model:chat`, those two plus the four chat-only fields on
`model:embeddings`, and 21 fields on `deterministic`. That last row is the derivation earning its
keep — nobody wrote the four embeddings names down, and forbidding them is exactly right, because
an embeddings call has no `runtime` object to observe and no sampling parameters to obey, so a
record carrying either is claiming something it cannot have measured.

**`ARM_KINDS` had to stop being derived from the forbidden mapping in the same edit.** It was
`frozenset(FORBIDDEN_BY_ARM_KIND)`; re-keying that mapping and leaving the derivation makes its
members the three *profiles*, so `armKind == "model"` fails the membership test in `validate()` and
**every model record returns `FieldProblem("armKind", "unknown")` and refuses on write** — a green
mapping and a dead harness. It is now derived from the profile keys' prefixes
(`p.split(":", 1)[0]`), which is decoupled from the forbidden mapping while staying a derivation
rather than a second hand-maintained list, and pinned by value in the suite. `CALL_SURFACES` comes
from the same split. Every `armKind == "model"` / `== "deterministic"` filter in `results.py` and
`report.py` is unchanged by design and was re-read to confirm each still means the two-valued
discriminator; `models --tested` is asserted to still return an embeddings arm.

**`callSurface` is a second discriminator, required with no default.** `Fingerprint` takes
`armKind` and `callSurface` (`None` **iff** deterministic) and derives `armProfile`; both are
members of no required set and are checked **before any mapping is consulted**, because without a
surface there is no profile and so no contract to report the fields against — answering a
surface-less model record with thirty `absent` problems would bury the one that is true. A
deterministic record carrying a surface is `forbidden`; a surface this build has never seen is
`unknown` rather than resolved to a profile key no mapping carries. `from_dict` strips both
discriminators (one left in `fields` lands in every profile's forbidden set) and `to_dict` omits
`callSurface` rather than writing `null` on a deterministic arm — that arm calls no surface, which
is a different fact from "we did not capture this". Identity (`__eq__`/`__hash__`) includes it.
Being required with no default is what made the type system enumerate the ten real `Fingerprint(`
construction sites for us, which is §7 rule 5's *adds rather than retires* half.

**Suite.** Both model profiles now get the per-required-field treatment — 30 + 26 cases for the
absent loop and the same for the null loop, where only `model` had them before — and the M-4
hand-transcribed literal gains a `model:embeddings` sibling, transcribed independently and
deliberately **not** derived from the chat one. Write-side acceptance covers all three profiles: a
clean embeddings arm stores, and one carrying `runtimeName` is refused on write.

**A plan friction worth recording.** S1 done-condition 1 names both retired residency keys, which
invites a test that spells the second one — and Table A's second residual requires that token to
reach **zero** across `modelbench` and `tests`, comments included. Written literally, the two
cannot both hold. Resolved by making the *implementation* rule key-set-exact, so it refuses either
retired key by construction, and asserting it through the rule and through `modelKey` (which has no
residual) rather than by naming the retired token. The tree therefore carries the token nowhere and
the behaviour is still pinned.

## 2026-09-03 — S1 fourth gate round: the nets that catch the first scorer's first mistake

**What:** Closed every major and minor from the fourth gate round on S1 —
`docs/reviews/small-model-benchmarking-impl.md` `## Pass 4` (**P4-1**…**P4-10**, plus nits
**P4-11**…**P4-13**) and `docs/reviews/small-model-benchmarking-ml.md` `## Pass 4` (**M-ML-8**,
**m-ML-9**…**m-ML-12**, nit **n-ML-8**) — against method note **v1.8** and plan **v1.8**.
Test-first throughout; **353 → 389 tests**, and **49 mutations run against the fixes, 49 killed,
no survivors**. `.venv/bin/ruff check .` clean.

**The round's shape.** Six of the fifteen findings are *"a mutation survives the suite"* — code
that was already right with nothing pinning it — so for those the test **is** the deliverable and
the implementation did not move. The rest divide into one real statistical correction (M-ML-8),
three note-published strings that were false on the path every comparison currently takes, and two
report-level nets that had been deferred to S2 and were pulled back to S1 because a net added after
the thing it protects has shipped is how all four of these rounds began.

**M-ML-8 (major) — the fail-safe path quantified with the narrower of its two instruments.**
B-ML-2's veto fixed what the non-`by-construction` path *decides* with two rounds ago and left what
it *quantifies* with on the bare percentile bootstrap. Measured exactly, that interval covers 0.939
at n=40 against MOVER-D's 0.976 while printing narrower bounds on **100%** of the probability mass,
and it degenerates at the sparse discordant counts `-ml` §3.2b says this lab will see: at n=30 with
`b=4, c=0` it returned `[3.3, 26.7] pp`, **excluding zero**, against an exact p of 0.125, because
four non-zero rows make `P(no +1 drawn) = (26/30)³⁰ = 1.4% < 2.5%` and a 2.5th percentile of zero
unreachable. Note v1.8 §3.4 Rule 4 replaces it with the **conservative envelope**
(`stats.conservative_envelope`): the wider of the `√DEFF`-widened bootstrap and the `√DEFF`-widened
MOVER-D, half-widths scaled about the same point estimate.

*Read bound by bound, not by picking the wider interval whole* — Rule 4's own stated property is
"uniformly at least as conservative as either alone", which choosing one interval does not deliver:
at `(4, 5, 3, 0)`, the tool-caller pack's own n, the bootstrap is the wider interval while MOVER-D's
lower bound is the more conservative one, so picking it whole would print a bound tighter than an
instrument the rule says it dominates. Confirmation that the reading is right came from the
rendered output: at DEFF 1.00 the guard-judge shape now prints
`+15.0 pp (95% CI [3.2, 29.1] pp)`, which is `-ml` §3.2e verdict 1's published string exactly —
Rule 4's "reduces to MOVER-D exactly at DEFF 1.00", observed rather than asserted.

**Not done, and deliberately: v1.8's second half, the closed-form percentile.** The note also says
the percentile "should" be computed in closed form rather than resampled, which removes the
seed/row-order sensitivity of an atomic quantile. The envelope hides that wherever MOVER-D is the
binding arm — measured, the overwhelming majority of tables — but not where the resample escapes it:
at n=12 the rendered lower bound still moves between `-27.1` and `-33.3 pp` across seeds (11 of 19
against 8 of 19, measured). Landing the closed form retires `-ml` §3.2d's seed from this decision
and with it review **P3-5**'s delivered contract (`PackRef.seed`, `verdict`'s `bootstrap_seed`, and
the report's `decided by: … (seed N)` line), which is a scope decision rather than a formula.
Carried to the coordinator, not taken here.

**Three note-published strings, all false on the default path** (m-ML-9, m-ML-10, m-ML-11). The
equality boundary now reads *"is at or above that"* — one comparative true across the whole branch,
and equality is reachable at this component's own n (`n_units=85, k=2, DEFF=1.9` gives
`mdd80 = 20.0 pp`, and `|b−c| = 17` over 85 rows is exactly that). The cluster-path label is
published verbatim in two variants keyed on the **design effect**, because the shipped sentence
asserted *"under clustering McNemar rejects too readily"* on a comparison that declares no
clustering — P3-3's defect surviving in the half P3-3 did not touch. The floor sentence gains an
effective-unit qualifier wherever `design_effect > 1.0`, because the floor is `b_min/n_eff` while
the McNemar p three clauses away is over the raw rows, and without it the line reads as a flat
contradiction of the number beside it (30.0 pp of floor beside `p=0.008`).

**P4-4 (major) → S1 done-condition 10 — the `aggregates`-versus-`items` cross-check.** An arm
declaring `BinaryMetric(m, successes=0, n=10)` for a metric no item declares scoreable printed
`0/10 = 0.000` — a claim that ten items were scored — in the same document as *"No verdict: no
paired data"*. On mismatch the arm is **excluded and named** in the `INVALID RESULTS EXCLUDED`
block: raising reproduces P4-5's shape, and suppressing one metric's row leaves a partly-trusted
arm in the comparison. **Two defects in the done-condition's own text were caught by the plan gate
mid-implementation and are recorded here because the code deviates from the written spec:**
**G3-6** — DC-10's selector names two disjoint vocabularies (`BinaryMetric.unit` is a denominator
noun, `PackRef.analysisUnit` a `pairingKey` component name), so the literal predicate is never true
and would have checked nothing; the code uses `metric.unit == roles.unit_kind(pack.role)`.
**G3-7** — DC-10 counts with `scored_outcome`, which *raises* for the sibling malformation; the
check treats that as a mismatch and names the offending item instead.

**P4-5 (major) — one bad item took the whole comparison down.** `IncompleteItemRecord` escaped
`compare_report` as a traceback at **exit 1**, outside §3.6a's closed `{0,2,3,4,5}`, with no report
written and the valid arms lost with it. `load_history` now quarantines such a record on read as an
ordinary `field` failure naming the item and the metric, which is AC-2's actual mechanism.

**The rest.** **P4-1** — a `--negative-control` run with nothing to duplicate wrote a durable report
claiming *"both arms are the same stored record … cannot fail"* ten lines above *"fewer than two
arms were selected"*; the banner is now decided after the arms are known and replaced, not merely
suppressed, so the artifact still says the mode was requested and did not run. **P4-2**, **P4-3**,
**P4-7**, **P4-8**, **P4-9**, **m-ML-12** — six surviving mutations pinned, of which m-ML-12's and
P4-2's each **flip a printed verdict** (the veto tested at `alpha_family` instead of the Holm step;
`holm_tested` hardcoded `True`, which reprints Pass 1's blocker verbatim — a significance claim
beside its own *"not tested (Holm stops here)"* row). **P4-6** — a zero-denominator aggregate is
rendered as `0/0 — no observations` rather than silently dropped. **P4-10** — two arms of one model,
which is plan §5 test 19a, are told apart by session (falling back to `runId` where the session does
not distinguish, and adding nothing where the model key already does). **n-ML-8** — Rule 7's raise
is no longer gated on `holm_tested`, which is not one of the theorem's premises.

**One defect found by rendering the output rather than by an assertion, as in every round so far.**
The P4-4 exclusion path left two individually-true sentences contradicting each other: the block at
the top said the arms were excluded, and the verdict line below still said *"fewer than two arms
were **selected** … Check `--models` and `--session`"* — the wrong remedy, sending a scorer author
to their command line when the defect is in their record. Split into its own reason.

**Verification:** from `model-bench/`, `.venv/bin/python -m pytest -q -m "" -rsx` → `389 passed`,
nothing skipped, xfailed or deselected; `.venv/bin/python -m pytest --collect-only -q -m ""` →
`389 tests collected`, so collected equals run; `.venv/bin/ruff check .` → `All checks passed!`.
Mutation testing was run by copying each source file aside and restoring from the copy, never
through git.

## 2026-09-03 — S1 third gate round: absence is not an outcome, and five sentences that were false

**What:** Closed the third round of gate findings on S1 —
`docs/reviews/small-model-benchmarking-impl.md` `## Pass 3` (**P3-1**…**P3-15**) and
`docs/reviews/small-model-benchmarking-ml.md` `## Pass 3` (**M-ML-7**, **m-ML-7**, **m-ML-8**,
**n-ML-4**…**n-ML-6**) — against method note **v1.7** and plan **v1.7**. Test-first throughout;
314 → 353 tests, and 28 mutations run against the fixes with **27 killed and one equivalent by
construction**. This round is dominated by a single failure mode: **the blocker and four of the six
majors are about what the report *says* rather than what it computes** — two false clauses, one
disclosure the report omitted entirely, and one number it never printed — and every one was found
by rendering a report and reading it, not by an assertion. The remaining two majors are tests that
could not fail.

**P3-1 (blocker) — an arm holding no data for a metric was scored as failing every item.**
`report.py` carried two defaults of its own: a missing `scoreable` entry read as *scoreable*, and a
missing `counts` entry then scored the row a **loss**. Together they rendered
*"cand is better than incumbent: +100.0 pp (95% CI [60.8, 100.0] pp) … p=0.002"* against an arm
whose ten items carried `counts={}` and `scoreable={}` — while the §4.3 tally, whose entire job is
to make dropped rows visible, printed `0 unscoreable in both`. Absence was laundered into the
denominator's complement, the mirror image of the laundering `-ml` §4.3 forbids. Which state a row
is in is now **`ItemResult.scored_outcome`'s call and nothing else's**, with three declared answers:
absent or `False` in `scoreable` is *no outcome* and routes through the tally; `True` **must** carry
a count, and one that does not is refused (`IncompleteItemRecord`) rather than read as a zero — a
scorer that declares an item scored must supply its score. A metric whose paired intersection is
empty now renders an explicit refusal with the tally beneath it and gets **no** `ResolvingPower`
(`n_effective` of zero is not a small sample); its Holm row prints `—` and `no verdict — no paired
data` rather than `mcnemar_exact(0, 0)`'s misleading `1.000`. **This is a contract on S2's scorers**
and is recorded in `AGENTS.md`.

**M-ML-7 / P3-2 (found independently by both gates) — the "not distinguishable" verdict asserted
the observed difference was below the MDD without checking.** `"; the observed X pp is below that."`
was fixed prose, false whenever `|diff| >= mdd80` — which is §7.1's *normal case for a model swap*,
a candidate that wins more than it loses without strictly dominating. Measured by the statistics
gate: 268 of the 1 580 by-construction tables that print the clause printed it falsely. Note v1.7
§3.2e mandates a conditional clause and publishes the alternate wording verbatim, discordance
counts included; it is implemented as published. The comparison is **strict**, and that boundary is
reachable rather than theoretical — swept over `6 <= n <= 120`, `|diff| == mdd80` occurs at k=2 for
n = 90, 100 and 120, where "below that" is false a second way — so it has its own test.

**P3-3 — the fail-safe path claimed a widening that never happened.** The cluster-path label read
*"widened by sqrt(DEFF)=1.00 **for the declared clustering**"* at `design_effect == 1.0`: nothing
was widened and no clustering was declared, on the path **every** comparison carries until S2's
determinism probe lands. What actually displaces McNemar there is Rule 4's other half, the
`basis` — and the sentence never named it. The clause is now conditional: it names the design
effect where one was applied, and the basis where none was. **The sentence itself is not note-owned
prose** — `-ml` §3.4 Rule 4 requires "the design effect and its basis printed" on this path but
publishes no string for it, which is recorded as an open item for `data-scientist`.

**P3-4 — `--negative-control` wrote a durable report indistinguishable from a real comparison.**
The mode puts two copies of one record in both arms, so `b = c = 0` is arithmetic; the report said
nothing about that (`grep -ic negative` returned 0) and was filed beside real comparisons under a
filename differing only in its sequence number. A reader got a plausible validated null. The report
now opens with a banner naming it a wiring smoke check that **cannot fail**, and pointing at the
real negative control (two independent runs, an acceptance step). The code comment that claimed the
report already said this is corrected.

**P3-5 — the bootstrap seed was a literal in the renderer and was never printed.** `report.py`
passed `bootstrap_seed=20260902`, duplicating the manifest's `sampling.seed` in a type that had no
field for it, so the pack's own declaration could not reach the decision — and on the fail-safe path
the seeded bootstrap is what decides. `PackRef` now carries `seed` with **no default**, read from
the manifest (a manifest omitting it is refused by name), and the `decided by:` line prints it where
a resample actually ran. The test asserting the printed line was not enough on its own: it left the
literal alive, printing one seed over an interval resampled at another. It now asserts that two
packs differing only in `seed` render **different intervals** — possible only at a fixture coarse
enough for the percentile to move (n=12, b=5, c=3; at n=40 the rendered bounds are identical at
every seed tried). The fingerprint half stays S2's.

**P3-6, P3-7 — two tests that could not fail.** The suite's only k=2 α assertion was
`"alpha=0.025" in md`, satisfied by the family-wise paragraph rather than by the MDD sentence it was
placed to guard, so `provenance` naming the wrong α survived; it now asserts the whole provenance
parenthetical, and the floor's own α beside it. The exploratory-label test asserted two strings'
presence and not their pairing, so **inverting** the filter — labelling the pre-registered verdict
metrics "exploratory" and hiding the genuinely exploratory ones — was green; it now asserts the
rendered line whole plus the negative. Both inversions now fail.

**The minors and nits.** `compare --session` had no test at all (P3-8) and now has two, from both
directions. `index.csv`'s `valid` column (P3-9), the `armKind` absent-vs-null discriminator
(P3-10), `verdict()`'s and `resolving_power`'s design-effect guards (P3-11), `compare_report`'s
headline-membership guard (P3-12, whose failure mode was a bare `StopIteration`), `wilson_interval`'s
probability clamps and `pack_ref_from_manifest`'s `analysisUnit` check (P3-15) each gained the one
assertion that kills their surviving mutation. `holm_steps`' `alpha` default — the second literal
`0.05` in the module that declares there is only one (P3-13, n-ML-4) — is **removed** rather than
re-pointed at `ALPHA_FAMILY`, matching plan v1.7's signature block. `resolving_power` now refuses
`design_effect < 1.0` at construction instead of `<= 0` (n-ML-5): below 1 a design effect *inflates*
effective *n* and shrinks both printed bounds, and the refusal used to arrive a layer later. The
report filename is now the manifest's `packId`, not the pack directory name (P3-14) — the half
Pass 1's m-6 left behind. m-ML-7's `floor_clause` boundary, m-ML-8's duplicated MDD stem and
n-ML-6's hard-coded `"80% power"` were closed by the same round.

**One correction to a test, not to the code.** `verdict()`'s design-effect precondition raised the
*same sentence* as `paired_cluster_bootstrap`'s identical bound one layer down, so the obvious test
passed with the precondition deleted. What it asserts now is the **ordering** Rule 4 states — every
precondition checked before any instrument is selected — which is visible only when the two orders
produce different errors.

**Verification.** `.venv/bin/python -m pytest -q -m "" -rsx` → **353 passed**, nothing skipped,
deselected or xfailed; `--collect-only` collects 353, so the run count equals the collected count.
`.venv/bin/ruff check .` → clean. 28 distinct mutations run, **27 killed**. The survivor is
**equivalent by construction**: restoring `report.py`'s duplicate of the MDD stem with *identical*
text renders identically, so no test can distinguish it. The mutation that matters — the same stem
edited in `stats.py` while the duplicate stays stale, which is exactly the drift m-ML-8 predicted
M-ML-7's fix would cause — **is** killed, by the test asserting the report renders
`stats.mdd_clause`'s own string.

## 2026-09-03 — S1 second gate round: the floor's α, McNemar as a veto, Rule 7 by path

**What:** Closed the second round of gate findings on S1 —
`docs/reviews/small-model-benchmarking-ml.md` `## Pass 2` (**B-ML-2**, **M-ML-6**, **m-ML-6**) and
`docs/reviews/small-model-benchmarking-impl.md` `## Pass 2` (**P2-1**…**P2-5**) — against method
note **v1.6**, which landed this morning and changed Rules 3, 4, 6 and 7, §3.3, §7.1 and §7.3.
Test-first throughout; 296 → 314 tests, and 22 mutations run with **one survivor, equivalent by
construction** (see below).

**M-ML-6 — the observable floor moves to the unadjusted α, and `ResolvingPower` carries two αs.**
The floor claims *"below Y nothing can reach significance at any observed outcome"*, which is true
only at the **loosest** Holm step a member can face. Printed at α/k it is `7/n` and **false**: at
n=40 a rank-2 member with b=6, c=0 reaches p=0.031, clears its own 0.05 step, and its 15.0 pp sits
below the 17.5 pp the old floor printed. `resolving_power` now takes `alpha_family` **and**
`alpha_mdd`, both keyword-only with no default, and each bound is computed at its own — the floor
at `alpha_family`, the MDD unchanged at `α/k`. §7.1's mandatory sentence names both. The third α,
Holm's data-dependent `alpha_step`, stays `verdict()`'s parameter rather than becoming a field:
it is known only after the family is ranked, so a field would be `None` until it was not, and the
number would have two homes. The sweep went past `stats.py`: `PackMetrics.alpha` — a code-side
restatement of exactly this α — is now `alpha_family` / `alpha_mdd`, and `report.py` reads them
instead of recomputing `0.05 / len(family)` inline. `stats.ALPHA_FAMILY` is the single home of the
unadjusted 0.05.

*Consequence, taken deliberately:* the α/k floor was reducing Holm to Bonferroni for every
difference in `[6/n, 7/n)` — precisely the band §7.3 already prices as the cost of a second verdict
metric — so the build was charging that price twice. It no longer is, and the rendered family table
for the review's own case now reads `distinguishable` where it read `not distinguishable — below
the observable floor`.

**B-ML-2 — the substitute path is a conjunction, not an interval.** At `design_effect == 1.0` with
`basis == "assumed"` — the fail-safe **every** comparison carries until S2 lands the determinism
probe — the decision moves off McNemar and `sqrt(1.0)` widens nothing, so a bare percentile interval
was deciding. Reproduced at n=40: `(b=7, c=1)`, `(9, 2)` and `(11, 3)` all rendered
*distinguishable* at p = 0.057–0.070, where the exact test refuses, and Rule 7 does not catch them
because 15.0, 17.5 and 20.0 pp are all **at or above** the floor. The non-`by-construction` decision
is now *"the widened CI excludes zero **and** `mcnemar_exact <= alpha_step`"*. Note v1.6's Rule 4
permits this explicitly: the objection to McNemar under clustering is that it *rejects* too readily,
and a necessary condition only ever removes rejections, so the pair is uniformly at least as
conservative as either instrument alone. The verdict strings say which instrument played which
role — one sentence for both paths would have contradicted one of them.

**m-ML-6 — Rule 7 splits by path.** On `mcnemar-exact` the invariant is a **theorem** (re-verified
here by binary search over every `b + c <= 400` at both αs, zero violations), so a fire is a module
bug and now raises `Rule7Violation`; silently demoting discarded exactly the detector property the
rule exists for. On `cluster-bootstrap` it stays demote-and-name, because a widened interval and a
shrunken effective *n* legitimately disagree there. Sequenced **after** M-ML-6, as the review
required: at α/k the McNemar branch is reachable and the raise would have fired on correct data. A
fifth precondition makes the theorem's premise checkable rather than assumed — `alpha_step` must
lie in `[alpha_mdd, alpha_family]`, which Holm's own steps do by construction.

**The bin-edge truncation guard stays; its justification was corrected.** It was load-bearing on the
17.5 pp cell at n=40 — **the cell M-ML-6 deleted**. Swept to n ≤ 2000: with `b_min = 7` naive
truncation misfires at n = 5, 10, 20, 40; with `b_min = 6`, which the floor now always uses, never.
So the guard is **defensive**, kept because `b_min` is a function of α and any future α reopens the
hazard, and its test is no longer a regression pin on a published figure. Both the code comment and
the test say so. The `floor(x/precision)` expression is still pinned by the code's own form — a
test now also pins the `precision` parameter, since `floor(x*1000)` agrees at the default and
nowhere else, which is how the original sweep missed the hazard.

**Engineering findings.** **P2-1** — `"measured"` at DEFF 1.0 had no test at Rule 4's branch, and
widening `mcnemar_may_decide` to admit it survived all 296 tests; two mirror tests (unit and report)
now close it. **P2-2** — `validate()` accepted `benchSchemaVersion: true` (`True == 1`) while
`load_history` quarantined it, so `store()` wrote a record the reader refused; the bool guard now
lives at both enforcement points, and a quarantined bool no longer lands in a field typed
`int | None`. **P2-3** — `holm_steps` builds its list without a `None`-filter and `report.py` zips
`strict=True`, so a short ladder raises instead of silently dropping a pre-registered verdict
metric. **P2-4** — accepted in part, with a correction: `Path(".").name` is `""` so `"."` is
redundant, but **`Path("..").name` is `".."`**, so `".."` is *not* already caught and stays; one
third of the guard was unreachable, not two thirds. **P2-5** — `packs.py`'s `contentHash` docstring
now says `None`, matching the code.

**Verification.** `.venv/bin/python -m pytest -q` → `314 passed`; `.venv/bin/ruff check .` → clean.
22 mutations run against the fixes, **21 killed**. The survivor — restoring `holm_steps`'
`None`-filter — is **equivalent by construction**: every index is assigned, so the filter changes
no output on its own. Compounding it with a ladder that actually returns short is killed twice over
(the strict zip and the length invariant), which is the honest statement of what P2-3's fix buys.

## 2026-09-03 — S1 gate remediation: both blockers, all ten majors, and Rule 7

**What:** Fixed the findings of the two independent S1 gates —
`docs/reviews/small-model-benchmarking-impl.md` (`analyst`: 1 blocker, 6 majors, 7 minors, 4 nits)
and `docs/reviews/small-model-benchmarking-ml.md` (`data-scientist`: 1 blocker, 4 majors, 5 minors,
3 nits) — against plan v1.5 and method note v1.5. Test-first throughout; every fix was
mutation-tested and the reviewer's **ten surviving mutations are now all killed**.

**The two blockers.**

- **B-ML-1 — the clustered decision path did not cluster.** `verdict()`'s substitute for McNemar
  was `paired_bootstrap` over the *rows* of the paired table: an i.i.d. resample of observations
  the declared design effect says are correlated, so the interval was identical at DEFF 2, 4 and 7
  and *narrower* than the MOVER-D it replaced. It changed the instrument's name, not its interval.
  New primitive `paired_cluster_bootstrap` inflates the percentile half-widths about the point
  estimate by `sqrt(design_effect)` — the Kish variance ratio is exactly the quantity that converts
  (`-ml` §3.4 Rule 5). **This is the note's "smallest honest version", taken deliberately:** the
  structurally right fix resamples clusters of paired differences, and `PairedOutcomes` carries one
  row per analysis unit with no grouping, which could only come from a pack declaring
  `replicatesPerScript > 1` — something Rule 6 makes a validation error while only the one-level
  `cluster_bootstrap` exists. Building it now would have had no data to consume and no seam to
  reach it.
- **B-1 / M-ML-2 — Holm–Bonferroni was printed and never applied.** `report.py` called `verdict()`
  without `alpha_step`, so every metric was decided at plain Bonferroni α/k, and `holm_thresholds`
  had no step-down stop. `compare_report` now runs **two passes** — Holm is a property of the
  family, so no verdict can be decided until every p-value exists — and `holm_thresholds` is
  replaced by `holm_steps`, returning a `HolmStep` per member with its rank, threshold, `tested`
  and `rejected`. `verdict()` gained `holm_tested`, which is the stop.

**Rule 7 (`-ml` v1.5 §3.4), enforced in `verdict()` rather than left to a test.** No verdict path
returns `distinguishable` when `|diff|` is below `resolving.observable_floor`. Three decisions in
it, each with a reason:

- **It demotes and says so; it does not raise.** The note's contrast is code-versus-test, not
  raise-versus-demote, and a raise would be unreachable in practice: the √DEFF-widened bootstrap
  and McNemar's exact rejection region are different instruments that do not align by construction
  (measured — at DEFF 2 on the `(34, 6, 0, 0)` table the widened interval still excludes zero while
  15.0 pp sits below the 30.0 pp floor). The demotion renders the contradiction it resolved, which
  surfaces the defect more loudly than a traceback the report never prints.
- **It compares against the exact float, never `format_floor_pp`'s truncation** — otherwise the
  invariant inherits the presentation layer's rounding and can fire, or fail to fire, by 0.05 pp.
- **The converse is not asserted.** `|diff| >= floor` does not imply distinguishable; §3.2c's row 4
  `(20, 8, 2, 10)` is the counterexample already in the suite — 15.0 pp exactly on the α=0.05
  floor, p = 7/64, not distinguishable.

It never fires on the McNemar branch: `test_the_mcnemar_path_satisfies_rule_7_by_construction`
checks every `(b, c)` split at n ∈ {12, 20, 30, 38, 40, 48, 85} and α ∈ {0.05, 0.025}. That
asymmetry is what makes it a detector rather than a formality.

**The floor's rounding direction, per the adjudication: the floor truncates, the MDD ceilings.**
`stats.format_floor_pp` is the one place the direction lives, and it is where the report and the
verdict strings both print from — the tests assert **through the formatter**, because re-rounding
inside a test (`round(observable_floor(...) * 100, 1)`) asserts the presentation layer's arithmetic
against itself. `ResolvingPower.observable_floor` stays exact, so Rule 7's guard is not weakened.
Truncation is guarded (`math.floor(x / precision + 1e-12)`), mirroring the MDD's `- 1e-12`, and
**the guard is load-bearing rather than defensive**: `7/40 = 0.175` is `174.99999999999997` bins in
IEEE doubles, so naive truncation prints `17.4` for the α=0.025, n=38–40 row the note publishes as
**17.5**. Corrected cells: 15.8→15.7 (n=38), 7.1→7.0 (n=85), 46.7→46.6 (n=15, α=0.025); 58.3 and
23.3 were already truncations.

**The other majors.**

- `load_history` validated *after* the pack filter, so a record whose `packId` was blanked or
  deleted on disk landed in **neither** returned list — the comparison quietly lost an arm (M-1).
  The filter now drops only a record that *says* it belongs to another pack; it also applies to an
  unknown schema, whose `packId` is readable, and stays **off** `unparseable`, which cannot declare
  one (m-1).
- `RunResult.designEffect`/`basis` lost their dataclass defaults (M-2, m-ML-3, plan v1.5 §3.5). The
  legacy fallback stays in `from_dict`, where it is a reader's §3.4.3 compatibility rule.
- `BinaryMetric` gained a required `unit`, and the Arms table prints a Wilson interval only over
  the analysis unit (M-ML-3). §4.4's first mandatory consequence is verbatim *"Never print a Wilson
  interval over a turn-pooled count"*, and a turn-pooled 142/320 was printing ±5 pp where the
  honest bound is ~48.7 pp. The count is never suppressed; only the precision claim is.
- `_paired_rows` returns a `PairedRows` tally and every verdict prints it — the `asymmetry` count
  §4.3's paired corollary requires, plus rows present in one arm only and unscoreable in both
  (M-5, M-ML-4). It is printed even when nothing was dropped, because otherwise a reader cannot
  tell a shrunken `n` from a full one.
- `min_detectable_difference` raises `UnattainablePower` below `b_min(alpha)` units instead of
  converging on its bisection bracket and returning `1.0`; `ResolvingPower.mdd80` is then `None`
  and the line reads *"No difference is resolvable…"* (M-ML-1). The delivered build printed
  *"resolves differences of >=100.0 pp with 80% power"* where power is identically **zero**.
- A comparison with fewer than two arms has its own reason, and `--models` naming a key with no
  stored run exits **2** rather than silently rendering a one-arm report (M-6).
- The basis/design-effect propagation is now tested at report level (M-3), and prints the **weaker
  of the two actual bases** rather than collapsing to `assumed` — false provenance in the one
  sentence whose job is auditability (m-ML-4). The decision rule is unchanged.
- `REQUIRED_BY_SCHEMA` and `FORBIDDEN_BY_ARM_KIND` are pinned against **independently transcribed
  literals**, by name and by tier (M-4). Parametrizing over them meant deleting an entry deleted
  its test case rather than failing one.

**Minors and nits:** the unpaired label distinguishes a content-hash divergence from a version one
(m-2); `_unit_ids` is called by `_paired_rows` instead of being dead code the docstring names
(m-3); `--role` and the index's `latencyMsP95` gained tests (m-4); `PackRef.contentHash` is
`str | None` so "not yet computed" is expressible (m-5); `compare` filters by the manifest's
`packId`, not the directory name (m-6); `store()` refuses a `runId` that is not a bare filename
(m-7); the tautological assertion is gone (n-1); `Fingerprint` copies its mapping behind a
`MappingProxyType` and hashes its **values** (n-2); an absent `aggregates` block is reported as
`unparseable` rather than repaired into an empty one (n-3); the conditionality clause names the
pack's own sample noun (n-4, m-ML-5); the `-ml` §3.2c fixtures are republished at 10 dp and
asserted at the mandated **1e-9 on the proportion**, with the docstring's margin claim corrected
from four orders to three (m-ML-1); `test_z_95_matches_the_inverse_normal_cdf` records that the
pinned literal is one ULP from `NormalDist().inv_cdf(0.975)` and must not be tightened to `==`
(n-ML-3).

**One finding declined, with its reason.** n-ML-1 asked for the floor and the MDD to share a
denominator (`observable_floor` divides by the unfloored `n_effective`; `min_detectable_difference`
floors first). Unifying them would make one of the two anti-conservative: Rule 3's principle is to
round each printed bound in the direction that keeps its own claim true, and the two claims point
opposite ways — a **larger** MDD is the safe error, a **smaller** floor is. The asymmetry is now
documented at `observable_floor`, which is the one line the finding asked for.

**Two defects found by reading rendered output, not assertions** — the same discipline that caught
the CI-orientation bug at S1. The "Best case — assumes the candidate wins every…" caveat was still
printing where no MDD exists, qualifying a figure that is not on the page; and the clustered label
was appended to two of the five verdict strings rather than all of them, so a reader seeing only a
demoted verdict was never told which instrument produced it.

**Verification, from `model-bench/`:** `.venv/bin/python -m pytest -q` → **296 passed** in 2.12s,
exit 0 (0 failed, 0 skipped, 0 deselected — the `live` marker still deselects nothing because no
live test exists until S2). `.venv/bin/ruff check .` → `All checks passed!`. **34 source mutations
against a scratch copy — 34 killed, 0 survivors**, including all ten the `analyst` gate reported as
surviving and 24 new ones aimed at this change's own fixes. Two of the new ones initially survived,
both because a test asserted a passthrough field instead of the behaviour it gates; both tests were
rewritten onto cases where the mutation changes a verdict.

## 2026-09-03 — S1: fingerprint, results, stats, report, CLI (no model calls)

**What:** Built the harness core per stage S1 of `docs/plans/small-model-benchmarking.md` §4 —
everything that decides whether a number may be printed, and nothing that produces one. No model
calls, no network, no LM Studio, no pack loader: the whole S1 suite runs offline.

- `modelbench/fingerprint.py` — `Fingerprint` (frozen, `armKind`-discriminated), `FieldSpec`,
  `FieldProblem`, `REQUIRED_BY_SCHEMA` (`{schemaVersion: {armKind: {field: spec}}}`) and
  `FORBIDDEN_BY_ARM_KIND`. Fields are held in a **mapping, not dataclass attributes**, because a
  dataclass with `None` defaults collapses *absent* into *null* — the two states plan §3.4.2 exists
  to separate. `validate()` returns problems and never raises; the `deterministic` arm kind
  forbids every model field, so `{"modelKey": "bm25"}` fails loudly on write (plan §3.4.1, gate B-3).
- `modelbench/results.py` — `ItemResult`, `RunResult`, `InvalidRecord`, `BENCH_SCHEMA_VERSION = 1`,
  a **closed union** of five typed aggregate dataclasses, `store()` (raises, no bypass parameter),
  `load_history()` (returns `(valid, invalid)`, re-validating each record against **its own**
  `benchSchemaVersion`), `rebuild_index()` and `models_with_stored_results()`.
- `modelbench/stats.py` — implements `docs/plans/small-model-benchmarking-ml.md` §3.4's six binding
  rules and nothing else: `wilson_interval` (`z` keyword-only, defaulting to the pinned
  `_Z_95 = 1.959963984540054`), `mcnemar_exact`, `mover_d_interval`, `paired_bootstrap`,
  `cluster_bootstrap`, `PairedOutcomes` (duplicate-unit guard in `__post_init__`, so it holds on
  every construction route), `resolving_power`/`ResolvingPower`, `min_detectable_difference`
  (exact bisection over the McNemar rejection region, ceilinged to the printed precision, and
  taking `n_effective: float` so a raw `int` count raises `TypeError`), `observable_floor`,
  `design_effect`/`effective_n`/`width_inflation`, `verdict()` and `holm_thresholds`.
- `modelbench/report.py` — `compare_report()`: the excluded-invalid block (AC-2), the pack
  version/content-hash banners (AC-3), the `SCHEMA VERSIONS IN THIS COMPARISON` line (§3.4.3), the
  comparison-kind line (§3.7), per-arm Wilson intervals labelled *descriptive, not the comparison
  instrument*, the resolving-power line, the three verdict strings (AC-4), Holm–Bonferroni for a
  k>1 family, and the marginal-overlap diagnostic with its footnote.
- `modelbench/packs.py` — `PackRef`, `PackMetrics`, `metrics_from_manifest`,
  `check_sampling_contract`, `pack_ref_from_manifest`. **Not** S2's pack loader: no content hash,
  no AST import walk, no data-file row-count identity. `PackRef` extends Appendix A's five fields
  with `pairingKey` and `analysisUnit`, without which §3.3's analysis-unit resolution has no source.
- `modelbench/roles.py` — FR-21's five roles and `-ml` §3.3's unit-kind column.
- `modelbench/cli.py` + `modelbench/__main__.py` — `compare` (with `--negative-control`),
  `index rebuild`, `models --tested`; §3.6a's closed exit-code set. `attest`, `validate` and `run`
  are S2's and their absence is asserted by a test.
- `run.sh` — the S0 guard block deleted, as S0's own entry said S1 would.

**Two decisions taken here that the plan does not state, both additive and both flagged to
`architect`:**

- **`RunResult` gains `designEffect: float` and `basis`.** §5 test 12b requires `runner` to *set*
  `basis`, and `-ml` §3.4 Rule 4 decides which instrument may decide from it — but the plan's
  `RunResult` shape carries neither, and a report cannot recompute either after the fact. Without
  them S1 done-condition 5b is unsatisfiable. The degradation is fail-safe: any arm not
  `by-construction` drops the comparison to `assumed`, which moves the decision off McNemar.
- **`FieldProblem.reason` gains `"unknown"`** beside Appendix A's four, for a discriminator this
  build cannot interpret — an unrecognized `armKind`, or a `benchSchemaVersion` from the future.
  Forcing either into `absent`/`empty` would mislabel it.

**One defect found and fixed by reading the rendered output rather than the assertions:** when arm
B won, `verdict()` re-oriented the difference to the winner (`+66.7 pp`) but left the confidence
interval in A-minus-B orientation (`[-86.2, -29.9]`) — a positive effect printed beside a wholly
negative interval. Nothing raised; it is a plausible-looking, internally contradictory line, which
is the exact failure mode a measuring instrument must not have. The non-significant strings now
keep the signed A-minus-B difference for the same reason.

**Verification:** `.venv/bin/python -m pytest -q` from `model-bench/` → **233 passed**, exit 0
(0 failed / 0 skipped; the `live` marker deselects nothing at S1 because no live test exists yet).
`.venv/bin/ruff check .` → `All checks passed!`. `./run.sh --help` and `./run.sh models --tested`
both exit 0. Every done-condition test was mutation-tested; the load-bearing one is S1
done-condition 5(c), where pairing on the conversation id instead of the pack-declared
`analysisUnit` is caught by the captured-argument assertion independently of the raise.

## 2026-09-02 — S0: component skeleton

**What:** Created the `model-bench/` component per stage S0 of
`docs/plans/small-model-benchmarking.md` §4 — packaging, scripts, docs skeleton and an empty
package/suite. No harness code: S0's done-condition is deliberately an empty test suite, so that
S1–S8 land against a tree that already builds and lints.

- `pyproject.toml` — `requires-python = ">=3.12"`, **no runtime dependencies** (plan §3.2, a hard
  design constraint), dev extras `pytest>=9.1,<10` + `ruff>=0.14,<0.15`, ruff `select = ["E","F","W","I"]`
  / `line-length = 100` (mcp-monitor's shape), pytest `testpaths = ["tests"]` plus falkor-chat's
  live-test convention verbatim: `addopts = '-ra -m "not live"'` and a `live` marker.
- `setup.sh` — adapted from `mcp-monitor/setup.sh`: idempotent, `--recreate`, resolves paths from the
  script's own location, ends with an import smoke test.
- `run.sh` — the mcp-monitor shape (venv check, then `exec .venv/bin/python -m modelbench "$@"`) with
  an **S0 guard**: `modelbench/__main__.py` does not exist until S1, so the script reports that in
  words and exits 1 rather than `exec`-ing into a `No module named` traceback. S1 deletes the guard.
- `.gitignore` — `.venv/`, `host.json` (the operator-attested fingerprint fields, plan §3.4),
  `results/transcripts/` (raw model output: large, and not needed for any comparison, plan §3.5).
- `README.md` — what the tool is, and the three non-features stated up front: no CI/scheduler, no
  pass/fail gate, no leaderboard or cross-role aggregate.
- `AGENTS.md` — working context: current state, the hard rules (zero runtime deps, FR-23 standalone,
  no cross-role aggregate), the `live` marker, the attested fingerprint fields, and the note that
  an empty suite exits 5.
- `docs/{BACKLOG.md,HISTORY.md}` plus empty `requirements/ plans/ reviews/ test-plans/ test-reports/`
  held by `.gitkeep` files. `BACKLOG.md` is seeded with the two items plan §7 carries forward.
- `modelbench/__init__.py` (`__version__`) and `tests/test_package.py` — one install smoke test,
  asserting `modelbench.__version__` equals the installed distribution's metadata version. The plan
  called for an empty suite at S0, but pytest exits 5 (`EXIT_NOTESTSCOLLECTED`) when nothing is
  collected, so "runs and passes with zero tests collected" cannot return 0 (plan gate finding m1).
  Resolved with this one real test rather than by configuring the exit code away: a permanent
  "no tests ran is fine" setting would still be in place at S5 and would hide a collection
  breakage. The assertion is not filler — that version string is what stamps `benchVersion` into
  every run record (plan §3.4), so a skew between `pyproject.toml` and `__init__.py` fails here.
- Root `AGENTS.md` — a `model-bench/` bullet in **Structure** and a row in **Component docs**. The
  feature's requirements and plan stay at the repo root, where they were written (plan §4 S0).

**One defect found and fixed by reading the rendered output rather than the assertions:** when arm
B won, `verdict()` re-oriented the difference to the winner (`+66.7 pp`) but left the confidence
interval in A-minus-B orientation (`[-86.2, -29.9]`) — a positive effect printed beside a wholly
negative interval. Nothing raised; it is a plausible-looking, internally contradictory line, which
is the exact failure mode a measuring instrument must not have. The non-significant strings now
keep the signed A-minus-B difference for the same reason.

**Verification:** `model-bench/setup.sh` → venv created with Python 3.12.3, `model-bench[dev]`
installed (pytest 9.1.1, ruff 0.14.14), smoke import printed `model-bench 0.1.0`; re-run to confirm
idempotence. `.venv/bin/python -m pytest -q` from `model-bench/` → `1 passed in 0.01s`, exit 0
(0 failed / 0 skipped / 0 deselected). `.venv/bin/ruff check .` → `All checks passed!`.
`./run.sh --help` → the S0 guard's message, exit 1. Note that the test command must be run with
`model-bench/` as the working directory: the repo has no root pytest configuration, so from the repo
root pytest ignores this component's `testpaths` and walks the whole monorepo (measured: 9 collected,
8 collection errors, exit 2).
