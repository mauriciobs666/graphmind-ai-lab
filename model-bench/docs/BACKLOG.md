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

## Note

Stage S8 of `docs/plans/small-model-benchmarking.md` re-checks this list at close and adds whatever
the R-1 probe (does `lms ps --json` expose the KV-cache setting on a loaded model?) leaves open.
