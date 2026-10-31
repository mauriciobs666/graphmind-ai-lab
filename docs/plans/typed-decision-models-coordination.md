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
| U1a-probe live probe | human-approved | — | queued | Ask stakeholder first (GPU). Done = `/api/v0/decisions` route question settled **and** a raw `/v1` (or `/api/v0`) body captured into `model-bench/tests/fixtures/lmstudio/decisions_response.json`, `_provenance` removed (impl review U1a-m5); also check Laya's 512-token context (U0 note §Q7) | — → — | — |

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
