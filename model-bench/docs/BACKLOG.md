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

## Note

Stage S8 of `docs/plans/small-model-benchmarking.md` re-checks this list at close and adds whatever
the R-1 probe (does `lms ps --json` expose the KV-cache setting on a loaded model?) leaves open.
