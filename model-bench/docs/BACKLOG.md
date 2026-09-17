# model-bench — Backlog

Living backlog, per the module documentation convention (root `AGENTS.md`) — forward-looking only:
delivered work leaves this file and is recorded in `HISTORY.md`.

## Open items

- **The default (network-free) test suite is not actually independent of `falkor-chat/` being
  present on disk, contrary to `model-bench/AGENTS.md`'s FR-23 standalone rule.** Found at S8 close
  (item 20, the FR-23 audit) and confirmed directly: `tests/test_scoring_extraction.py`'s
  `test_prompts_querygen_md_matches_the_live_falkorchat_source_byte_for_byte` and
  `tests/test_refresh_golden.py`'s `test_read_catalog_literal_handles_the_real_seed_catalog_script`
  and `test_read_schema_literal_handles_the_real_querygen_module` each `Path(...).read_text()` a
  real file under `../falkor-chat/` with no `@pytest.mark.live` and no skip guard. Runtime/CLI code
  (`validate`/`run`/`compare`/`attest`) is clean — every other `falkor-chat` mention in
  `modelbench/`/`scripts/`/`packs/` is either `scripts/refresh_golden.py` (the plan's named
  permitted exception, unreferenced by any `modelbench/` import) or doc-only (docstring/comment/
  `PROVENANCE.md` table) — and `model-bench validate` itself runs clean with `falkor-chat/` renamed
  away. Only the default `pytest -q` suite is affected: with `falkor-chat/` renamed away it reports
  3 failed instead of the true baseline, an environment-dependent false failure a future agent could
  easily misread as a real regression. `tests/test_refresh_golden.py`'s own module docstring
  (lines 9-12) already asserts the suite passes with `falkor-chat/` renamed away — that claim is
  false as shipped and needs correcting alongside the fix. Fix is either an `@pytest.mark.live` (if
  reachability, not content, is what should gate them) or a `pytest.importorskip`/explicit skip
  keyed on the sibling directory's presence — a design call, not a one-line default.
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
