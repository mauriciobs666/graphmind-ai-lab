# Typed Decision Models — Coordination (docs-only chain)

> **Status:** active · **Owner:** `tico` · **Tracks:** —

Docs-only chain run by `tico`: requirements → plan → review → revision → re-review. Hands off to
`teco` the moment a unit would touch source, tests, or config (the plan's U0+ build units).

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict | Cost |
|---|---|---|---|---|---|---|
| R1 requirements | `tico` | — | accepted | `docs/requirements/typed-decision-models.md` | analyst Passes 1–2 → approve with suggestions | — |
| P1 plan v1.0–1.1 | `architect` | a033c294d3b62c94b | superseded by P2 | `docs/plans/typed-decision-models.md` | analyst Pass 3 → needs changes (7 majors) | ~250k tok |
| G1 plan review | `analyst` | adaa3498564d8eeb2 | accepted | `docs/reviews/typed-decision-models.md` §Pass 3 | — | ~191k tok |
| P2 plan v1.2 | `architect` | a033c294d3b62c94b | gated | `docs/plans/typed-decision-models.md` | analyst Pass 4 → approve with suggestions (N1, N2 major) | ~321k tok |
| G2 plan re-review | `analyst` | adaa3498564d8eeb2 | accepted | `docs/reviews/typed-decision-models.md` §Pass 4 | — | ~237k tok |
| P3 plan v1.3 (fold N1–N5) | `architect` | a033c294d3b62c94b | accepted | `docs/plans/typed-decision-models.md` | tico line-by-line check (per Pass 4) → N1–N5 and both duplicates verified in place | ~341k tok |
| U0 `-ml` method note | `data-scientist` | a90005863d23ec284 (v1.0–1.1) · a1ca78754c4116564 (v1.2, fresh: prior ctx 303k) | accepted (v1.4 rulings: chat timeout pinned 180 s §Q4.3; reload-spanning run not a replicate §Q6; teco verified 180 s source) | `docs/plans/typed-decision-models-ml.md` + scripts `docs/plans/typed-decision-models-ml/` | `analyst` a21279e06d0207212 → Passes 5–7 needs changes; Pass 7 stop condition met (teco line check 2026-10-30: key at §Q2.1, max-min rule gone, (2,0) picked, scripts run) | 303k / 75 + 108k / 32 (v1.2); gate 201k / 47 |
| U1b fingerprint profiles | `tdd-engineer` | abe0fe90f24980f5b | accepted | `model-bench/` (plan §4 U1b) | `analyst` ae963f92f37ebbfd3 → approve with suggestions (`docs/reviews/typed-decision-models-impl.md` Pass 1) | 260k tok / 79 tools (U1b+U3a+U1a) |
| U3a classification split | `tdd-engineer` | abe0fe90f24980f5b | accepted | `model-bench/` (plan §4 U3a) | `analyst` ae963f92f37ebbfd3 → approve (`docs/reviews/typed-decision-models-impl.md` Pass 1) | 260k tok / 79 tools (U1b+U3a+U1a) |
| U1a `decide` (offline) | `tdd-engineer` | abe0fe90f24980f5b | accepted | `model-bench/` (plan §4 U1a) | `analyst` ae963f92f37ebbfd3 → approve with suggestions (`docs/reviews/typed-decision-models-impl.md` Pass 1) | 260k tok / 79 tools (U1b+U3a+U1a) |
| R2 requirements fold (replicates, label volume, infeasible-on-eval, FR-8 wording) | `tico` | a93f02a576abd55c3 | accepted (Pass 6 approve w/ suggestions; P6-M3/r1 folded, teco line check) | `docs/requirements/typed-decision-models.md` (FR-1, FR-4, FR-8, FR-10, FR-11a, AC-1/6/12 + 4 log entries) | U0 re-gate reads it → — | 65k tok / 8 tools |
| P4 plan successor (absorb note's 11 handed items) | `architect` | abcee4783836ab892 | accepted (v1.1) | `docs/plans/typed-decision-models2.md` | `analyst` a9d04e698428b5996 → Pass 1 needs changes; stop condition met (teco line check 2026-10-30, M1–M6 located §3.2/§3.3/§3.6/§4.3/§5) | 406k tok / 52 tools total; gate 239k / 37 |
| FX1 impl-review fix round (U1b-n1/n2, U1a-m1..m4) | `tdd-engineer` | a7f76089509a22634 | accepted | `model-bench/` | `analyst` ae963f92f37ebbfd3 impl Pass 2 → approve w/ suggestions (P2-m1 open) | 191k tok / 54 tools (FX1+U1c) |
| U0b note v1.5→v1.6 (512-token hard limit) | `data-scientist` | a1ca78754c4116564 | accepted (v1.6; teco line check 2026-10-31: §Q4.3 over-limit row, §Q2.1 fit bullet, no-handoffModel case, handed list 1–6) | `docs/plans/typed-decision-models-ml.md` | `analyst` a21279e06d0207212 → Pass 8 needs changes (stop cond: 4 items, line check) | 177k / 20 (v1.5); gate 265k / 24 |
| U1c pure `decisions.py` + golden vectors | `tdd-engineer` | a7f76089509a22634 | accepted (teco mutant: choice policy on max(probabilities) → 7 red) | `model-bench/` (v1.3 §4 U1c + plan2 §4.1) | `analyst` ae963f92f37ebbfd3 impl Pass 2 → approve w/ suggestions (P2-m2, n1, n2 open) | 191k tok / 54 tools (FX1+U1c) |
| FX2 impl Pass 2 fixes (P2-m1, P2-m2, P2-n1, P2-n2) | `tdd-engineer` | a7f76089509a22634 | accepted, committed `28e21f31` | `model-bench/` | `analyst` ae963f92f37ebbfd3 impl Pass 3 → approve | 214k tok cumulative / 11 tools |
| P5 plan successor 3 (absorb note v1.6 handed 1–6 + impl Pass 2 Q1/Q2) | `architect` | aedd10f1b8d78000b | accepted (v1.2, 2,839 words; teco line check 2026-10-31: B1 U3c row `(load, load)`/`loadCount == 2`, M1 three-column table §3.3, m1 upper-case row) | `docs/plans/typed-decision-models3.md` | `analyst` a9d04e698428b5996 → Pass 1 needs changes; stop condition met | 143k / 41 (v1.0–1.1); gate 316k / 24 |
| U1d.1 contracts: v1.3 U1d's Protocol hooks excluded; plan2 §4.1 U1d fingerprint delta + `score_answer` delta; plan3 §4.1 U1d (one-way rule, `LMStudioCallFailed.body`, `is_over_limit`, `over-limit.jsonl`, `reason`) | `tdd-engineer` | a5bbf3cc05b48e236 | accepted, committed `f3b853d2` (teco mutant: probability rounding → 1 red) | `model-bench/` | `analyst` a416786350362bb49 impl Pass 4 → approve w/ suggestions (P4-m1 open) | 197k / 61 |
| FX3 P4-m1 `score_answer` legality (288-combination test, incl. architect's withheld case + `input_tokens`) | `tdd-engineer` | a5bbf3cc05b48e236 | accepted, committed `1c6c9e7b` | `model-bench/` | `analyst` a416786350362bb49 impl Pass 5 → approve w/ suggestions (P5-m1) | 227k cumulative / 11 |
| FX4 P5-m1 `withheldFor` legality (576 combinations) | `tdd-engineer` | a5bbf3cc05b48e236 | accepted | `model-bench/` | teco line check → met (suite 2397; teco mutants: clause A and clause B each → cross-product test red) | 244k cumulative / 8 |
| U1d.2 plain `decisions` arm (v1.3 §4 U1d: Protocol hooks, packs, runner arm, `cli run --candidate`) | `coder` | — | queued | `model-bench/` | `analyst` → — | — |
| U1d.3 run-time deltas (plan2 §4.1 U1d: timeout pin/refusal, kept decision failure, load rule/re-check/basis/inputTokens, `decisionCallFailures`; plan3 §4.1 U1d: decisions-surface over-limit item, `overLimit`) | `coder` | — | queued | `model-bench/` | `analyst` → — | — |
| U2p load-route probe (plan3 §4.2) | `teco` (stakeholder-approved 2026-10-31) | — | accepted | `ws:agent-team` doc a93905a6… — all four YES: load route loads laya; config has `physical_batch_size` 512; laya+qwen co-resident through a decision call; unload by `instance_id` clears it. Explicit loads carry no TTL. Untested: larger `physical_batch_size` on load | — → — | — |
| U1a-probe live probe | `teco` (stakeholder-approved 2026-10-30) | — | delivered | Raw bodies in `ws:agent-team` doc "laya /v1/decisions live probe 2026-10-30" (for U1e). Findings: `/api/v0/decisions` absent (HTTP 200 + error body) → stay on `/v1`, U1e runtime branch void; limit is per question ≤512 tokens (505 ok, 513 → HTTP 500 "increase the physical batch size"), **hard failure not truncation** (R-13 premise false); `usage.input_tokens` sums across questions; LM Studio keeps **one JIT model**: loading laya evicted qwen and vice versa | — → — | — |

Stakeholder answers to Pass 3's open questions (M3, M4) are folded into the requirements doc
(FR-4, FR-1; commit `bf9eae77`).

**Docs-only chain closed 2026-10-30:** plan v1.3 accepted. The remaining work (U0 method note
onward, which touches code) is for `teco` to coordinate. This ledger and the plan are its state
of record.

**2026-10-30 — build chain taken over by `teco`.** Code-implementation chain from here: commit per
unit after its gate. Wave 1 is offline only (U0 ∥ U1b → U3a → U1a, one `tdd-engineer` serializing
the shared `model-bench/docs/HISTORY.md`). The U1a live probe waits for stakeholder approval
(shared GPU; `docs/requirements/standard-local-model.md` frees room for `laya`). Baseline sha
`dc2f8884`.

- 2026-10-30: Stakeholder answers to U0 Pass 5 P5-M3/P5-M2 (via teco): a bar claim must pass in **both** of
  two load replicates; labelling at the **full design** size (~14× the plan's earlier ~600 calls); a
  fit-infeasible candidate is **still measured on eval** (margins guide fitting only).
- Carried to later briefs (impl review Pass 1 open questions): U1d adds `decided_by`/`probability` to
  `classification.score_answer` with a test that the old pack writes no `decidedBy`; U1c's
  `Candidate.load` refuses a `score` question up front (U1a-n1).
- 2026-10-30: Wave-1 committed `b133fa95` (U1b+U3a+U1a + impl review). Earlier block was the session-id footer in the message (stakeholder rule: no footer).
- 2026-10-30: U0 v1.4 rulings (M2 pin 180 s, m2 load rule) relayed to in-flight plan2 architect `abcee4783836ab892` mid-revision; overrides teco's earlier "record + refuse-mismatch only" steer.
- 2026-10-30: plan2 v1.1 accepted. teco removed the note's absorbed "Handed to architect" list (note's own rule), repointing its four item citations to plan2 §3.3/§3.6/§3.8. Next: wave-1 commit (blocked on stakeholder) → fix round (impl review U1b-n1/n2, U1a-m1..m4) → U1c per plan2.
- 2026-10-30: Probe findings open two questions before U3b/U7: (1) co-residency of laya + chat model under JIT single-model eviction (stakeholder env decision); (2) design consequence of the 512-token hard failure (route: data-scientist + architect).
- 2026-10-30: Co-residency answer: the stakeholder will change LM Studio (turn off single-JIT eviction / pre-load). teco re-checks co-residency live before U7. 512-token question sent to data-scientist a1ca78754c4116564 (note v1.5), then to the architect for plan2.
- 2026-10-31: Stakeholder: an over-limit input **hands off to the LLM** at every point (not fail-closed). Relayed to data-scientist for v1.6. Plan2 has been executed against (U1c), so its absorption goes to a successor `typed-decision-models3.md` (collision rule 5) after v1.6 is accepted.
- 2026-10-31: FX1+U1c committed together (shared `HISTORY.md`/`test_lmstudio.py`). Impl Pass 2 open questions (empty band with a hand-off model; `eval_set_hash` format home) relayed to the in-flight plan3 architect `aedd10f1b8d78000b`.
- 2026-10-31: Impl Pass 3 asks whether the U5/U9 copies must pin the `apply_policy` bounds refusal (own tests or a shared `policy-bounds.jsonl`). Relayed as a coverage question to the plan3 gate `a9d04e698428b5996`.
- 2026-10-31: teco read-only GET `/api/v1/models`: `loaded_instances[].config.physical_batch_size` (512 on qwen), no load timestamp/nonce. Premise relayed to architect: an explicit load with a larger batch may lift Laya's 512 limit; B1 may need a human-approved load-API probe unit.
- 2026-10-31: U1d split into U1d.1–U1d.3 (sequential, all touching `runner.py`/`HISTORY.md`). Its brief spans 3 plans and about 9 files, past the dispatch-size boundary. Each sub-unit is gated.
- 2026-10-31: Impl Pass 4 open question 1 (what writes the guard's withheld over-limit record, plan3 §3.3) sent to architect `aedd10f1b8d78000b` as an inline ruling for the U2 brief (plan3 is gated, so no revision).
- 2026-10-31: Architect ruling on the guard's withheld over-limit record (U2 brief carries it): widen `score_answer` by one legal case. No answer, no `failed_call`, `reason="over-limit"`, with `timing.withheldFor` set, returns the withheld outcome with `detail={path, tier, reason}`. The runner never patches `detail`. The cell comes from the pack by `itemId`. Relayed to in-flight FX3 `a5bbf3cc05b48e236` immediately, because it changes P4-m1's third refusal.
