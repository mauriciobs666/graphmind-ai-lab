# model-bench — Backlog

Living backlog, per the module documentation convention (root `AGENTS.md`) — forward-looking only:
delivered work leaves this file and is recorded in `HISTORY.md`.

## Open items

### GAL-F0013 — Judged reply quality for the `chat-responder` pack (deferred, not cancelled)

- **Status:** 🔵 proposed — defaulted; owner confirms status

FR-21a scopes
first delivery to the deterministic layer (latency, format, grounding-by-containment); no pack in
this delivery contains an LLM judge. The design is preserved so it need not be re-derived:
`docs/plans/small-model-benchmarking.md` §3.8.5 and `docs/plans/small-model-benchmarking-ml.md`
§6.1–§6.2. Two rules that must not be softened if it is funded: **faithfulness only** (the copied
calibration record puts relevance agreement at κ = 0.21 against κ = 0.83 on faithfulness), and the
judge is gated on class-conditional rates — `falsePassRate ≤ 2/20` on a 40-item calibration set,
`parseFailureRate ≤ 0.05` — with the harness erroring out when `judgeModel == candidateModel`.

### GAL-F0014 — +22 harder retrieval queries for the `embedder` pack

- **Status:** 🔵 proposed — defaulted; owner confirms status

The copied 38-item golden set puts
recall@10 at 37/38, so the pack can detect a materially *worse* embedder but can never certify a
*better* one on recall (plan §3.8.1; the report prints this as a standing honesty line). Lifting
the ceiling means new queries, several with |R| ≥ 3, each human-verified per FR-19 — new golden
data, which is why it is not in first delivery.

### GAL-F0015 — `tools/exec.py`'s `order_by` sort raises an uncaught `TypeError` on a `None`-valued sort key

- **Status:** 🔵 proposed — defaulted; owner confirms status

(DESIGN §6.4). No shipped golden item reaches it (every `order_by` item sorts on a never-null
`Product.price`), but a sparse property in `tables.json`'s catalog half would. The fix is deferred
to the next deliberate `packVersion` bump of `nlq-structured-query`: the file sits inside the
content-hashed pack, so any edit changes `contentHash` (`8f975842`) and unpairs future runs from
the 18 recorded `1.2.0` runs. It orders nulls the way FalkorDB does (last ASC, first DESC;
live-verified, including the `WITH DISTINCT … ORDER BY` shape `querygen.compile` emits) and never
raises `SchemaViolationError` — `compile_and_execute` does not raise past validation. The spec
is `tests/test_tools_exec.py::test_order_by_a_sparse_property_places_nulls_the_way_falkordb_does`,
`xfail(strict=True, raises=TypeError)`; the marker comes off with the fix. A test with tied keys
or several nulls compares each tie group as a set: FalkorDB's order among ties is not mirrored
between ASC and DESC (`docs/reviews/s4-gate-followups.md`).

### GAL-F0016 — `report.py`'s lucky-pass labelling (†/‡) covers the descriptive k/n cells only

- **Status:** 🔵 proposed — defaulted; owner confirms status

If a pack
ever makes an `exactMatchBy*` metric a verdict metric, everything downstream goes unlabelled:
`compare_report`'s verdict text and Holm row, `rank_report`'s reference-anchored family, and the
`<!-- rank-report -->` marker that `scripts/consolidate_sweep_reports.py` carries into a sweep
consolidation. No shipped pack reaches it (`nlq-structured-query`'s only verdict and headline
metric is `layer1ExactMatchRate`). Whoever ships such a pack labels those sections (one line under
the heading reusing the † footnote) or accepts the limit knowingly. Two polish items in the same
file: a headline-only `rank_report` never names a stored-vs-placed `luckyPassCount` disagreement
(suffix the value in `_render_scorer_counts` with `‡`), and
`_LUCKY_PASS_DISAGREEMENT_FOOTNOTE` is worded for one arm though `{arms}` can list several
(`docs/reviews/s4-gate-followups.md`).

### GAL-F0017 — `results.Outcome` (`results.py:38`) omits `"unrunnable"`

- **Status:** 🔵 proposed — defaulted; owner confirms status

Every scorer builds
`ItemResult(outcome="unrunnable", …)` on a no-response call (`classification.py`, `extraction.py`,
`grounding.py`, `retrieval.py`, `agreement.py`, `toolcalls.py`, `runner.py`), and DESIGN §6.1 lists
it as an outcome, but the `Literal` admits only `pass`/`fail`/`n_a`/`parse_failure`. No type checker
runs in this repo (no `mypy`/`pyright` config), so nothing flags it. Widening is one line, but first
sweep every `match`/`if` chain over `Outcome` (at least `report.py` and `results.py`'s aggregation
helpers) for one that silently assumes four members.


### GAL-F0018 — `checklist_pass` substring matching fails ordinary morphological paraphrase

- **Status:** 🔵 proposed — defaulted; owner confirms status

(`mustContain`/`mustNotContain` are plain, canonicalized (case/whitespace) substring checks) (`scoring/grounding.py`;
DESIGN §6.5). Live on `cr-11`/`cr-17` of `packs/chat-responder-grounded-answers/items.jsonl`:
`"4 retries" in "4 retry attempts"` and `"30 minutes" in "30-minute"` are both `False` (different
stems; a hyphen splitting one token). Word-boundary tokenization alone fixes neither; a real fix
needs stemming or hand-rolled normalization, with its own held-out fixture set to bound the new
false positives (as the `_ABSTENTION_MARKERS` widening needed its hedge-then-answer guard). A
`data-scientist`-scoped design task, not a one-line change, and scope creep for a
zero-runtime-dependency component if folded into a smaller fix
(`docs/test-reports/small-model-benchmarking-s7-report.md`, mechanism 2).


### GAL-F0019 — `report.py` prints no per-script breakdown for `tool-caller`

- **Status:** 🔵 proposed — defaulted; owner confirms status

Only pooled turn-level exploratory
metrics and the one per-conversation verdict metric are printed. Named in the S6 QA pass's feedback
(`docs/test-reports/small-model-benchmarking-s6-report.md`) as a testability gap, not a blocker: a
future pass wanting to confirm one specific script's specific turn behavior needs either a richer
stored trace or a standalone tool like `scripts/s6_walkthrough.py` run against a live model.

### GAL-F0020 — `ministral-3-3b`'s rung-2 tool-call censoring under `plaintext` replay is not root-caused

- **Status:** 🔵 proposed — defaulted; owner confirms status

`mistralai/ministral-3-3b` under `historyReplay: "plaintext"` (69/81 turns unrunnable) was observed only. Recommended as a follow-up, not
actioned, by the S6 QA pass (`docs/test-reports/small-model-benchmarking-s6-report.md`) — worth a
dedicated root-cause pass only if `plaintext` `historyReplay` is ever considered for real use with
this model.

### GAL-F0021 — No paired significance test exists for a turn-pooled exploratory metric

- **Status:** 🔵 proposed — defaulted; owner confirms status

(e.g.
`duplicateWithinTurn`/`duplicateCrossTurn`): `report.py` has no interval for a metric whose analysis
unit is a turn rather than the role's conversation/item unit (DESIGN §10 prints it without one), so
such a metric can be reasoned about but not measured
(`docs/test-reports/small-model-benchmarking-s6-report.md`).


### GAL-F0022 — `validate_pack` raises instead of reporting when a declared data file is not yet authored

- **Status:** 🔵 proposed — defaulted; owner confirms status

`_answerability_stamp_problems` (`packs.py`, the `nlq-generator`-only `"answerable"` check) iterates
`pack.iter_items()` unguarded, so a manifest declaring `data.items` before `items.jsonl` exists
makes `validate_pack` raise `FileNotFoundError` out of both `cli.py` callers (`validate` and
`run`'s pre-flight), breaking its own contract that `[]` means valid and anything else is a problem
string. Reproduced on a copy of `packs/nlq-structured-query` with `items.jsonl` removed. The two
sibling axes that read a data file are guarded (`_row_count_identity_problems`,
`_clean_through_turn_h_problems`: `try/except (OSError, json.JSONDecodeError)` → one problem
string, pinned in `tests/test_packs.py`); the fix is the same shape plus its mirror test. Low risk:
only a pack mid-authoring hits it.


### GAL-F0023 — `stats.verdict()`'s "is better than" wording is wrong for `falseAdvanceRate`/`falseSuspendRate`

- **Status:** 🔵 proposed — defaulted; owner confirms status

It assumes a higher raw rate is the better outcome, which is untrue for these metrics. `classification.score_item` stores the error
itself as the metric's success (`counts["falseAdvanceRate"] = int(advanced)` on a `clear_suspend`
item; `counts["falseSuspendRate"] = int(not advanced)` on a `clear_advance` one, documented at
`classification.py:509-512`), so lower is better, while `stats.verdict`'s winner selection
(`stats.py:1400-1405`, `diff >= 0`) is polarity-blind and `compare_report` prints its text
unmodified (`report.py:3134`). A distinguishable guard-judge verdict on either metric would print
"X is better than Y" when X has the higher error rate. `rank_report` already corrects for it
(`_LOWER_IS_BETTER`, `_polarity_corrected`). Read-verified, never observed live: no shipped
`reports/guard-judge-*.md` has a distinguishable verdict on either metric. Mechanism and a scoped
fix: `docs/plans/small-model-catalog-sweep.md` §2.4 and §6.


### GAL-F0024 — `report.rank_report()`'s reference-not-found `ValueError` appends `" and session"` whenever any other run is stored

- **Status:** 🔵 proposed — defaulted; owner confirms status

(`report.py:1843-1846`, `+ (" and session" if runs else "")`), not only when
a `--session` filter hid the reference. Accurate but misleading wording, not a live bug. A
one-line fix needs the CLI's `session` threaded into `rank_report` (it has no such parameter;
`cli._select_rank_arms` applies the filter) or the suffix dropped; do it the next time this
function is touched.


### GAL-F0025 — `rank_report()`'s zero-`n` exclusion banner double-counts items for shared-run metrics

- **Status:** 🔵 proposed — defaulted; owner confirms status

This happens when two verdict metrics share a run and both have `scoreable={}` items. `_attempted_for_metric` (`report.py`) counts an item
toward `metric` when it declares `metric` **or** declares nothing (`scoreable == {}`; the second
clause is what lets it read `nlq-structured-query`'s real parse-failure shape). But `{}` is also
every scorer's **no-response** shape (`classification.py:586-596` returns before the tier→metric
map is consulted). For guard-judge, the only pack with two verdict metrics on one `items` list, a
run whose items from both tiers time out prints "70 item(s) attempted" under **both**
`falseAdvanceRate`'s and `falseSuspendRate`'s banners where the true split is 40/30. Reproduced
with a fixture built from `classification.py`'s own no-response `ItemResult`; never observed live
(none of the 19 stored `guard-judge-understanding-*` runs has an empty-`scoreable` item).
The fix is not to guess which tier an empty item belonged to (that is not in the stored record,
and inventing it breaks "declared, never inferred", `AGENTS.md`): once more than one verdict
metric shares a run's items and an ambiguous count exists, the banner prints the items that
explicitly declared `metric` plus a separate, named count of items that recorded no response for
any metric and cannot be attributed to it
(`docs/reviews/small-model-catalog-sweep-impl.md`, Defect 1 fix, verification round 2).


### GAL-F0026 — The `nlq-structured-query` pack no longer mirrors falkor-chat's production parse

- **Status:** 🔵 proposed — defaulted; owner confirms status

(filed from
`falkor-chat/docs/reviews/salesperson-single-model.md` P5-4). Since falkor-chat `7f3ad0e5`,
`QueryGraphDataTool` merges repeated list keys (conflicting scalars rejected) via
`querygen.merge_duplicate_keys` and fills an absent `returns` via `querygen.default_returns`. Here,
`tools/exec.py` still rejects a missing `returns` and the model JSON is parsed last-wins by
`scoring/extraction.py` through `classification.extract_own_line_json_object` — **outside the
content-hashed pack**, so the repeated-key half needs a scorer edit as well as a pack bump. The
pack under-scores a model that repeats `filters` or omits `returns` (ministral on `nlq-12`).
Decide between a pack version bump (plus that scorer change) that copies both normalizations, or
a `PROVENANCE.md` note that the pack scores the stricter pre-K-066 contract. Copy, don't import:
nothing under `modelbench/` or the pack may read from `falkor-chat/` at runtime.

### GAL-F0027 — An LM Studio outage met by an item's own call is not re-probed

- **Status:** 🔵 proposed — defaulted; owner confirms status

(typed-decision-models plan4
R-21). A server that dies mid-call surfaces as `LMStudioCallTimeout` or `LMStudioCallFailed`
(`LMStudio._raw_post` maps a refused connection to the latter, `status=None`), so the item is
scored as `timeout`/`no_response`, or kept failed on the decision surfaces. **For chat and
embeddings items SMB plan §3.6 already rules it**: a timeout or a connection failure re-probes
and exits 3 unless `probe()` answers `"api-v0"`; an error response does not. No driver
implements that. `status is None` alone does not mark a connection failure: `chat()` raises it
on a malformed 200 body too, though re-probing that case is harmless. **Unruled, for
`architect`:** the decisions/handoff kept failures and the multi-turn `tool-caller` turn.


### GAL-F0028 — Residual limits of `compare`'s arm selection and `RunConfig.referenceKey`

- **Status:** 🔵 proposed — defaulted; owner confirms status

- "Newest run" (`compare --models` and plain `compare`) is the last run in file-name order, not the
  greatest `fingerprint.startedAt`, so a hand-named run id can select an older run and change the
  verdict silently (analyst R2, `docs/reviews/model-bench-defects.md`). A fix picks the greatest
  `startedAt` per key (ties: file-name order) in both `_select_arms` branches (`--models` and plain).
- `RunConfig.referenceKey` is dead data: `run` has no `--reference`, `cli` always passes `None`,
  and about eight test sites still construct it (analyst F6); removing it touches those tests.
- `compare --negative-control` over more than two arms takes the first record without a refusal.
