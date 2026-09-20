# model-bench — Backlog

Living backlog, per the module documentation convention (root `AGENTS.md`) — forward-looking only:
delivered work leaves this file and is recorded in `HISTORY.md`.

## Open items

- **Judged reply quality for the `chat-responder` pack (deferred, not cancelled).** FR-21a scopes
  first delivery to the deterministic layer (latency, format, grounding-by-containment); no pack in
  this delivery contains an LLM judge. The design is preserved so it need not be re-derived:
  `docs/plans/small-model-benchmarking.md` §3.8.5 and `docs/plans/small-model-benchmarking-ml.md`
  §6.1–§6.2. Two rules that must not be softened if it is funded: **faithfulness only** (the copied
  calibration record puts relevance agreement at κ = 0.21 against κ = 0.83 on faithfulness), and the
  judge is gated on class-conditional rates — `falsePassRate ≤ 2/20` on a 40-item calibration set,
  `parseFailureRate ≤ 0.05` — with the harness erroring out when `judgeModel == candidateModel`.
- **+22 harder retrieval queries for the `embedder` pack.** The copied 38-item golden set puts
  recall@10 at 37/38, so the pack can detect a materially *worse* embedder but can never certify a
  *better* one on recall (plan §3.8.1; the report prints this as a standing honesty line). Lifting
  the ceiling means new queries, several with |R| ≥ 3, each human-verified per FR-19 — new golden
  data, which is why it is not in first delivery.
- **`report.py` never renders `luckyPassCount`/`parseFailures`/`malformedSpecCount`/`schemaViolationCount`.**
  Flagged in the S4 code gate (`docs/reviews/small-model-benchmarking-s4.md`) — currently harmless
  (`luckyPassCount=0` on the only live run), but a future non-zero `luckyPassCount` would surface
  in a shape's raw `k/n` with no visual distinction from a genuine pass. Whoever wires this
  rendering must add a footnote/asterisk on a shape's `k/n` line whenever that shape's
  `luckyPassCount` contribution is non-zero, per `extraction.py:325-328`'s own docstring obligation.
- **`tools/exec.py`'s `order_by` sort has no defensive handling for a `None`-valued sort key.**
  Flagged in the S4 code gate (`docs/reviews/small-model-benchmarking-s4.md`) — not reachable by
  any shipped golden item today (every `order_by` item sorts on a never-null `Product.price`), but
  a future sparse property in `tables.json`'s catalog half would raise an uncaught `TypeError`
  instead of a graceful `SchemaViolationError`. Needs a one-line guard plus a regression test the
  next time that file is touched.

- **`validate_pack` doesn't check that a pack's declared `"scorer"` name resolves to an importable
  module** (named as a possible follow-up in `docs/plans/small-model-benchmarking-s3-spec.md` §9,
  carried open since S3's close, still nobody's). A typo'd scorer name fails at `run`'s
  `_load_item_scorer` import instead of at `validate` — a `RunRefused` (exit 4), the correct exit
  code, just one step later than a pre-flight check would catch it. Low risk, not blocking any
  stage.
- **`results.Outcome`'s `Literal` (`results.py:38`, `["pass", "fail", "n_a", "parse_failure"]`) has
  been missing `"unrunnable"` since S4.** `classification.score_item`/`extraction.score_item` both
  already construct `ItemResult(outcome="unrunnable", ...)` on a `no_response` withholding
  (`classification.py:218-221`, `extraction.py:216-221`) — a real, shipped, already-gated value the
  type annotation does not admit. Uncaught because this repo runs no type checker in CI (no
  `mypy`/`pyright` config anywhere in the tree). Found independently while verifying the S7 spec
  (`docs/plans/small-model-benchmarking-s7-spec.md`), whose own `grounding.py` design correctly
  mirrors this same precedent rather than inventing a fresh one — not this spec's defect to fix.
  Widening the `Literal` is a one-line change but not a no-judgment one: it needs a sweep of every
  `match`/`if`-chain over `Outcome` (at least `report.py`, `results.py`'s own aggregation helpers)
  to confirm none silently assumes only four members before the fifth is added to the type.

- **`checklist_pass`'s `mustContain`/`mustNotContain` containment is plain, canonicalized (case/
  whitespace-only) substring matching, which fails on ordinary morphological paraphrase** — split
  out of the S7 live-run defect (`docs/test-reports/small-model-benchmarking-s7-report.md`, "Defect
  — `_ABSTENTION_MARKERS` does not recognize...", mechanism 2) as explicitly out of scope for the
  `_ABSTENTION_MARKERS` widening fix, per a `data-scientist` consult that scoped the two mechanisms
  separately. Confirmed live on `cr-11`/`cr-17` (`packs/chat-responder-grounded-answers/
  items.jsonl`): `"4 retries" in "4 retry attempts"` is `False` (different stems, not a whitespace
  issue) and `"30 minutes" in "30-minute"`/`"8 hours" in "8-hour"` are both `False` (a tokenization-
  boundary artifact — the hyphen splits what plain substring containment treats as one token).
  Word-boundary tokenization alone does not fix either cited example — a real fix needs stemming or
  hand-rolled normalization, itself a real design task, not a one-line change: it needs its own
  held-out fixture set to bound how many new false positives a widened containment check
  introduces, the same way the `_ABSTENTION_MARKERS` widening needed the hedge-then-answer
  adversarial guard. Scope creep for this zero-runtime-dependency component if folded into a
  smaller fix; a `data-scientist`-scoped design task on its own.

- **`report.py` prints no per-script breakdown for `tool-caller`, only pooled turn-level exploratory
  metrics and the one per-conversation verdict metric.** Named in the S6 QA pass's feedback
  (`docs/test-reports/small-model-benchmarking-s6-report.md`) as a testability gap, not a blocker: a
  future pass wanting to confirm one specific script's specific turn behavior needs either a richer
  stored trace or a standalone tool like `scripts/s6_walkthrough.py` run against a live model.
- **`mistralai/ministral-3-3b`'s rung-2 tool-call censoring pattern under `historyReplay: "plaintext"`
  (69/81 turns unrunnable) was observed but not root-caused.** Recommended as a follow-up, not
  actioned, by the S6 QA pass (`docs/test-reports/small-model-benchmarking-s6-report.md`) — worth a
  dedicated root-cause pass only if `plaintext` `historyReplay` is ever considered for real use with
  this model.
- **No paired significance test exists for a turn-pooled exploratory metric** (e.g. the
  duplicate-instruction rate) — `report.py` has no interval for a metric whose analysis unit is a
  turn rather than the role's own conversation/item unit, so such a metric can only be reasoned about
  conceptually, never measured directly. Named as a coverage gap by the S6 QA pass
  (`docs/test-reports/small-model-benchmarking-s6-report.md`), not actioned there.

- **`validate_pack` still has one axis that crashes instead of reporting when a declared data file is
  not yet authored.** `_answerability_stamp_problems` (`packs.py`, the `nlq-generator`-only
  `"answerable"` check) iterates `pack.iter_items()` unguarded, so a manifest that declares
  `data.items` before `items.jsonl` exists makes `validate_pack` raise `FileNotFoundError` — out of
  both `cli.py` callers, `validate` and `run`'s pre-flight — instead of returning a problem string,
  breaking the shape its own docstring states (`[]` means valid, matching `Fingerprint.validate()`). Reproduced 2026-09-18 on a copy of the shipped
  `packs/nlq-structured-query` with `items.jsonl` removed. The two sibling axes that read a
  data file are guarded — `_row_count_identity_problems` from the start, `_clean_through_turn_h_problems`
  since S6, when it had this identical defect (`try/except (OSError, json.JSONDecodeError)` → one
  problem string, pinned by a `tests/test_packs.py` case); the fix here is the same shape, plus its
  mirror test. Low risk: no shipped pack hits it, only a pack
  mid-authoring.

- **`stats.verdict()`'s "is better than" wording assumes every verdict metric's own raw rate is the
  desired outcome — untrue for `falseAdvanceRate`/`falseSuspendRate`.** `modelbench/scoring/
  classification.py:209-210,259` stores the error itself as the metric's own "success": for a
  `clear_suspend` item, `advanced=True` IS the false-advance event, and
  `counts["falseAdvanceRate"] = int(advanced)`; symmetrically, `counts["falseSuspendRate"] =
  int(not advanced)` on a `clear_advance` item IS the false-suspend event. So a model's own
  `BinaryMetric.rate` for either metric is the rate of the *undesired* event — lower is better —
  while `stats.verdict()`'s generic winner selection (`stats.py:1320`, `winner, loser = (a_label,
  b_label) if diff >= 0 else (b_label, a_label)`) and the `report.py` text built from it are
  polarity-blind: a distinguishable guard-judge verdict on either metric would print "X is better
  than Y" when X in fact has the *higher* error rate. **Read-verified by tracing both files; never
  yet observed live** — every shipped `reports/guard-judge-*.md` was checked and none has reached a
  distinguishable verdict on either metric, so this has never surfaced in a rendered report. Full
  mechanism, and a scoped design that avoids the defect in new code without fixing it here:
  `docs/plans/small-model-catalog-sweep.md` §2.4 and §6.

- **`report.rank_report()`'s reference-not-found error message is oddly worded, not wrong.**
  `report.py:1362-1368`: when `--reference` names a model key with no stored run, the raised
  `ValueError` reads `"reference model {reference!r} has no stored, consistent run for this
  pack"`, with `" and session"` appended — but that suffix's condition is `if runs`, i.e. it
  appends whenever the pack has *any* other stored run at all, not when a `--session` filter was
  actually the reason the reference wasn't found. Reads backwards: a reader would expect the
  session clause to appear only when a `--session` argument was actually passed and is plausibly
  why the reference is missing, not merely because *some* runs exist. **Confirmed by reading the
  code; not a live bug** — the message is still factually accurate either way, just confusingly
  phrased. Flagged during Unit B's code-gate review of `docs/plans/small-model-catalog-sweep.md`
  (`docs/reviews/small-model-catalog-sweep-impl.md`, "Unit B Implementation Review"). Worth a
  one-line reword (tie the suffix to whether `session` was actually passed to `rank_report`, not
  to whether `runs` is non-empty) the next time this function is touched.

- **`report.rank_report()`'s zero-`n` exclusion banner double-counts a run's items when two
  verdict metrics on the same run both have `scoreable={}` items — the same overcount the banner
  exists to prevent, triggered by a different scorer path than the one that motivated it.**
  `_attempted_for_metric` (`report.py`, Defect-1 fix) counts an item toward `metric`'s attempted
  total when it either declares `metric` in `scoreable` or declares nothing at all
  (`scoreable == {}`) — the second clause is what lets it correctly read `nlq-structured-query`'s
  real parse-failure shape (a received-but-unparseable response, whose `scoreable` is genuinely
  `{}`, not `{metric: False}`) without regressing to "0 items attempted." But `scoreable == {}` is
  also exactly what every scorer in the package emits on a **no-response** item — `classification.py:
  239-244`'s `score_item`: `if result is None: return ItemResult(..., scoreable={}, counts={},
  ...)`, returned *before* the tier→metric mapping (`_METRIC_BY_TIER`) is ever consulted, so a
  timed-out/unrunnable item carries no record of which verdict metric its tier would have scored.
  `extraction.py`, `retrieval.py`, and `grounding.py` all emit the identical `scoreable={}` shape
  on their own no-response path. For guard-judge — the only shipped pack with two verdict metrics
  sharing one run's `items` list — a run whose items from **both** tiers time out (a total or
  partial outage, not a parse failure) makes `_attempted_for_metric` count every one of those
  empty items toward **both** `falseAdvanceRate`'s and `falseSuspendRate`'s own banners: reproduced
  directly (40 `falseAdvanceRate`-tier + 30 `falseSuspendRate`-tier items, all `scoreable={}`,
  mirroring `classification.py`'s own no-response `ItemResult` exactly) — both banners read "70
  item(s) attempted" where the true split is 40/30, the identical overcount shape Defect 1's fix
  exists to close. **Read-verified by tracing the real scorer code and reproducing against a
  fixture built from its own no-response `ItemResult` shape; never yet observed live** — every
  stored guard-judge run in the current sweep (`results/runs/guard-judge-understanding-*.json`,
  17 files) was checked and none carries a single `scoreable == {}` item, so this has not produced
  a wrong number in any rendered report to date.

  The fix is not "guess which tier an empty item belonged to" — that information is not in the
  stored record (a no-response `ItemResult` carries no `detail`/tier field, per
  `classification.py:241-248`), and inventing a split would itself violate the "declared, never
  inferred" rule this component holds everywhere else (`AGENTS.md`'s load-bearing invariants) —
  the exact rule the empty-scoreable-item design already leans on for every other case. The honest
  shape is to make the banner say what it actually knows once more than one verdict metric shares
  a run's items and an ambiguous item count exists: the number of items that explicitly declared
  `metric`, plus a separate, named count of items that recorded no response for *any* metric in
  this run and cannot be attributed to `metric` specifically — never a single confident total that
  silently assumes an even (or any particular) split. Full mechanism, reproduction, and the
  precedent this follows: `docs/reviews/small-model-catalog-sweep-impl.md`, "Verification round 2
  — 2026-09-20" under the Defect 1 fix section.
