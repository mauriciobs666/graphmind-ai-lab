# Salesperson demo on one LLM (`ministral` for agent + `query_graph_data` step) — Coordination

> **Status:** archived · **Owner:** `teco` · **Tracks:** K-066 (demo residency; not a milestone)

## Why this exists

The demo currently needs three resident LM Studio models (`ministral-3-3b` agent,
`qwen3-4b-2507` for the `query_graph_data` tool's `llm("step")` call, the 0.6B embedder) — about
7.6 GB of weights against a 6 GB RTX 4050 Laptop GPU, so layers spill to CPU/RAM. Stakeholder
chose (2026-10-21) to drop `qwen` and run **`ministral` for both roles**, so the demo fits on the
GPU with two resident models (~5.1 GB weights).

`qwen`-for-both is ruled out by prior evidence: the turn-4 tool-call collapse
(`falkor-chat/docs/reviews/salesperson-tool-reliability-ml.md` §8.2) and the K-056 model swap
(`falkor-chat/docs/HISTORY.md`). Evidence for `ministral` on the NLQ role:
`model-bench/reports/catalog-sweep-2026-09-19-consolidated.md` (32/32 vs `qwen` 34/34; 2 items
unscored — to be explained).

**Scoping constraint:** `falkor-chat/config/models.json` defaults are shared with the
`ws:agent-team` server (:8200) — the switch must be scoped to the salesperson demo, not a global
default flip.

## Definition of done

- The salesperson demo's `query_graph_data` step resolves to `ministral`, with no other consumer's
  model changed; tests + docs updated; two `analyst` gates passed.
- Live QA with only `ministral` + embedder loaded: language-salience probe no worse than
  `falkor-chat/docs/test-reports/salesperson-language-salience2-report.md`; NLQ correctness
  **gated**: no catalog-question regression vs the tracked qwen baseline (any regression goes to
  the stakeholder before acceptance, not silently accepted); dead-turn rate and GPU fit measured.
- **Fit check first (U0):** ministral + embedder at 8192 ctx must load fully on the GPU before U2
  is dispatched; failure pauses to the stakeholder (teco ruling 2026-10-21 on review open questions).
- K-066's preload guidance updated to the two-model set.

## Ledger

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict | Cost |
|---|---|---|---|---|---|---|
| U0 fit check | `qa-engineer` | `ab6fa3f58c13a8954` | delivered — fits only with WDDM paging (~1.6 GiB, ≤1.4 s on first embed after chat); 3 concurrent reqs → context exceeded; stakeholder 2026-10-21: build it, paging cost accepted | `falkor-chat/docs/test-reports/salesperson-single-model-fit-report.md` | — | 94k tok / 31 tools |
| U1 plan | `architect` | `ab2626a6195c6dc05` | accepted | `falkor-chat/docs/plans/salesperson-single-model.md` | `analyst` (`a686db21f34087f02`) → pass 1 needs changes (§7, §2.3; `falkor-chat/docs/reviews/salesperson-single-model.md`) → v2 → pass 2 needs changes (§7 only; code half S1–S9 cleared) → v3 → pass 3 approve w/ suggestions (P3-1 scratchpad baseline/instruments lost in session restart — U3 rebuilds them durably; P3-2/nits folded into U3 brief; P3-3 applied by teco) | 170k tok / 69 tools |
| U2 implement | `coder` | `a9d95df64f7cd7175` | accepted | per U1 step table (suite 3161/14/1; teco argument mutant red) | `analyst` (`a686db21f34087f02`) pass 4 → approve; committed `ad01aee7` | 160k tok |
| U3 live QA | `qa-engineer` | `a917311eb4da9b33b` | accepted — Q3 0/1,125 dead, 0 wrong-language; p95 14.1 s; Q5(d) 0 embeds; Q2 gate: stakeholder 2026-10-22 → fix before closing (U4) | `falkor-chat/docs/test-reports/salesperson-single-model-report.md` | — | 433k tok / 133 tools |
| U4 fix `nlq-12` duplicate `filters` key | `tdd-engineer` | `a2d3f370c9533bca0` | accepted — `querygen.merge_duplicate_keys` (identical kept once, lists merged per element, conflicting scalar abstains) + `querygen.default_returns` (absent `returns` → `<var>.name`, skipped on an `=` name lookup); 216 targeted green, full suite 3184/14/1 xfail, `reference` re-seeded + verified; teco mutants red ×3 | `llm.py`, `querygen.py`, `tools.py`, `test_tools.py`, `test_extraction.py` | `analyst` (`a686db21f34087f02`) pass 5 → needs changes (P5-1..3; P5-4 → model-bench backlog) → `analyst` fresh (`a65defe4a1fb74eab`) pass 6 → approve w/ suggestions (P6-1..3 landed, teco-verified) → live Q2 on final code: **catalog 21/21**, only diff vs qwen is KB `nlq-27` (→ K-081) | 161k tok / 57 tools |
| U5 manual §4 sentence (step model reaches `query_graph_data`) | `tico` | `adb374954fadb83af` | accepted | `falkor-chat/docs/manuals/llm-provider-config.md` §4 (38-word note) | teco fact-check vs `SERVER.md` §1.8 L623 + `tools.py` `run["stepModel"]`/`modelOverrides` → holds | 49k tok / 5 tools |

## Documentation-impact scan

`falkor-chat/AGENTS.md` / `salesperson/AGENTS.md` (demo model set), `falkor-chat/docs/BACKLOG.md`
K-066 (preload list), `falkor-chat/docs/HISTORY.md` (entry at close), `falkor-chat/docs/SERVER.md`
if it documents the model config.

## Close (2026-10-22)

Definition of done met: the demo runs on `ministral` + the embedder (U2 `ad01aee7`), live QA passed
(U3), the `nlq-12` gate was fixed (U4 `7f3ad0e5`, catalog 21/21 live), the manual note landed (U5
`d042611b`), and the K-066 preload guidance names the two-model set. K-066 stays open in
`falkor-chat/docs/BACKLOG.md` for the residency preflight only. Follow-ups filed: K-078–K-081, plus
a `model-bench/docs/BACKLOG.md` item for the NLQ pack's parse (P5-4). Pass 4's carry-overs (the
results-path parent check and the static-gateway nit) are minor and not filed.
