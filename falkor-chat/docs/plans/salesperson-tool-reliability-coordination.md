# Salesperson tool-orchestration reliability — K-057 + K-058 — Coordination

> **Status:** active · **Owner:** `teco` · **Tracks:** K-057, K-058 (post-M6, not a milestone gate)

Coordinating the investigation and (where warranted) fix of the two live, unresolved
`salesperson@v4` tool-orchestration defects on `mistralai/ministral-3-3b` surfaced during M6's
close-out: **K-057** (intermittent self-contradictory/incomplete answer on a compound
category+price filter question, `docs/BACKLOG.md` K-057) and **K-058** (silent duplicate
write-tool re-fire on a follow-up instruction, `docs/BACKLOG.md` K-058). Both are in the same root
class — this model's tool-orchestration reliability under this scaffold — but are mechanistically
distinct defects with different evidence bases and different next steps; they are coordinated
together because they share a live-testing environment and a natural stopping point (a combined
regression pass once both are addressed), not because they're expected to share one fix.

**Prior art (read before dispatching or picking this up cold):**
- `docs/reviews/salesperson-tool-reliability-ml.md` — the `data-scientist` diagnostic note for the
  whole model-reliability thread. §1-§7 diagnosed and resolved K-056 (qwen3-4b skip-and-fabricate,
  now `HISTORY.md`-recorded as resolved-by-model-swap). §8 is the controlled alternative-model eval
  that picked `ministral-3-3b` and, as a byproduct, first surfaced K-058's duplicate-instruction
  defect (§8.4, n=10, 30%). §9 is the scoped K-058 follow-up eval that narrowed the estimate (n=32,
  6 conditions, 4.2% pooled) and proposed but did not implement a candidate fix (§9.4: a
  dispatch-time check that a write tool's resolved target appears in the current turn's own text).
  §10 is an unrelated model-landscape survey, not needed for this thread.
- `docs/reviews/salesperson-tool-reliability-impl.md` — the `analyst` diff review of K-056's two
  falsified/reverted scaffold-level mitigation attempts. Relevant here only as precedent for what
  NOT to do (a naive dedup-by-signature for K-058 is already ruled out in §9.4 for the same
  reason one of these attempts failed) — not a live blocker for this thread.
- `docs/test-reports/workflow-nl-query-generation2-report.md` (DEF-01) — the live QA finding behind
  K-057; K-057's own BACKLOG entry has the full reasoning and candidate angles.
- `docs/BACKLOG.md` K-057/K-058 — both items carry their own "Owner"/"Test strategy" fields; treat
  those as the brief, not a re-derivation.

**Sequencing.** Two independent tracks, dispatched in parallel (different files/code paths; both
use the same already-loaded `ministral-3-3b`, so no LM-Studio JIT-reload thrash risk from running
together — the documented hazard is specifically *concurrent requests to different models*):
- **Track A (K-057):** `data-scientist` diagnoses first — root cause is not yet conclusively
  determined (orchestration-layer suspected, not confirmed). Its output determines whether Track A
  needs a follow-up implementation unit (e.g. a `systemPrompt` steering change) or closes as
  "diagnosed, accepted low-rate risk, no fix warranted."
- **Track B (K-058):** the candidate fix is already fully specified (§9.4) — goes straight to
  `tdd-engineer` for TDD implementation + mutation testing, no separate diagnosis unit needed.

Both tracks close with `analyst` review of any code diff and a combined live regression pass
(`qa-engineer` or a `data-scientist` eval, sized to the unit) before this coordination closes.

## Ledger

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict | Cost |
|---|---|---|---|---|---|---|
| U1 (Track A) | `data-scientist` (fresh) | `a6686c8ef000c2ec1` | in-flight | K-057 root-cause diagnosis + rate estimate → new dated section in `docs/reviews/salesperson-tool-reliability-ml.md` | self → — | — |
| U2 (Track B) | `tdd-engineer` (fresh) | `a057dc6683287232c` | in-flight | K-058 fix: dispatch-time text-presence guard on write-mutating tool calls, TDD + mutation-tested + live regression check | `analyst` → — | — |
