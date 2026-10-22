# Salesperson demo on one LLM (`ministral` for agent + `query_graph_data` step) — Coordination

> **Status:** active · **Owner:** `teco` · **Tracks:** K-066 (demo residency; not a milestone)

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
| U2 implement | `coder` | `a9d95df64f7cd7175` | gated | per U1 step table (suite 3161/14/1; teco argument mutant red) | `analyst` (`a686db21f34087f02`) pass 4 → in-flight | 160k tok |
| U3 live QA | `qa-engineer` | — | queued | `falkor-chat/docs/test-reports/salesperson-single-model-report.md` | — | — |

## Documentation-impact scan

`falkor-chat/AGENTS.md` / `salesperson/AGENTS.md` (demo model set), `falkor-chat/docs/BACKLOG.md`
K-066 (preload list), `falkor-chat/docs/HISTORY.md` (entry at close), `falkor-chat/docs/SERVER.md`
if it documents the model config.
