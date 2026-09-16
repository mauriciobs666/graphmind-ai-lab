# Provenance — tool-caller-shop-assistant

> **Pack version:** 0.2.0 · **Generated:** 2026-09-16T19:56:49Z

`conversations.jsonl` and `prose_calibration.jsonl` are **authored content, not copied**: no
single origin file in this monorepo is their source. The 12 conversation scripts reconstruct the
turn-count/turn-type patterns of `falkor-chat/docs/reviews/salesperson-tool-reliability-ml.md`
§8.1's three conditions (A/B/C), re-expressed against this pack's own `catalog.json`/
`tools/schemas.json` — never a verbatim transcription of §8.1's own falkor-chat-specific product
names or prices, which do not exist here. See `docs/plans/small-model-benchmarking-s6-spec.md`
for the reconstruction method and the coverage matrix.

| scriptId | shape | draftedBy | basedOn | verifiedBy | verifiedAt |
|---|---|---|---|---|---|
| A-01 | A | tdd-engineer | §8.1 condition A, turns 1-9 (determinism-probe fidelity) | | |
| A-02 | A | tdd-engineer | §8.1 condition A pattern, extended with restraint turns per plan step 2's own named coverage gap | | |
| A-03 | A | tdd-engineer | §8.1 condition A pattern, extended for boundary-argument and fabrication-guard coverage | | |
| A-04 | A | tdd-engineer | §8.1 condition A pattern, extended for restraint-phrasing diversity and cross-turn-duplicate coverage | | |
| B-01 | B | tdd-engineer | §8.1 condition B, turns 1-7 (determinism-probe fidelity) | | |
| B-02 | B | tdd-engineer | §8.4 ministral duplicate-instruction defect, turns 1-4 pattern; §8.1 condition B's add/view/order shape for turns 5-7 | | |
| B-03 | B | tdd-engineer | §8.1 condition B pattern, extended with a customer-initiated removal error and clear_cart coverage | | |
| B-04 | B | tdd-engineer | §8.1 condition B pattern, extended with a read-only opening turn and a closing-then-reopening restraint/new-request pair | | |
| C-01 | C | tdd-engineer | §8.1 condition C, turns 1-4 (direct replication) | | |
| C-02 | C | tdd-engineer | §8.2's position-not-content finding, adapted to a 4-turn script with a non-tool-bearing opening turn | | |
| C-03 | C | tdd-engineer | the pack's own boundary-argument pairing pattern (S6 spec §3.2), adapted to a 4-turn short script | | |
| C-04 | C | tdd-engineer | §8.2's turn-4 onset finding, adapted to an abstention-heavy 4-turn script | | |

`prose_calibration.jsonl`'s 20 labelled replies are hand-authored calibration data for
`scoring/toolcalls.py`'s `detect_prose_pseudo_call` heuristic — not derived from any live model
run, and not subject to FR-19's `verifiedBy` step (FR-19 governs the 12 conversation scripts'
`expect` blocks specifically; the calibration corpus's correctness is that its own labels are
true, checked by the same agent pre-check step that reviews the conversation scripts).
