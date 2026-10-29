# `query_graph_data` row identity and category-miss recovery — Stage C test plan

> **Status:** archived · **Owner:** `qa-engineer` · **Tracks:** K-079 (F2), K-086

## Scope & objective

Live verification (Stage C) of `docs/plans/salesperson-query-graph-data.md` (v3): F2 (D1–D5,
`ensure_row_identity`) and K-086 (D6 category listing in the querygen prompt, D7 empty-result
recovery). The spec is that plan's §8.2 (F2 done-condition: no Layer 1 regression on ministral)
and §8.3 (DEF-C done-condition: translated category browse), plus review
`docs/reviews/salesperson-query-graph-data-impl.md` Pass 2 m1 (`returns` presence on pt-BR
misses). Offline §8.1 is the implementer's gate and is not re-run here.

Code under test: Stage A `6286de44`, tests `d5ba5728`, Stage B `9ba0d788`. Run 0 baseline:
`e8798351` (ledger `docs/plans/salesperson-query-graph-data-coordination.md`).

## References

- Plan §3, §4 (design), §8.2/§8.3 (method, targets), §9 R7 (pre-existing execution failures —
  not regressions).
- DEF-C origin: `docs/test-reports/salesperson-cart-product-id2-report.md` (browse table, id2).
- K-066 setup precedent: `docs/test-reports/salesperson-single-model-report.md` E2/Q2.
- Environment: `falkor-chat/AGENTS.md` (demo residency, `:8200` idle, models).

## Risk assessment

| Risk | Where it shows | Covered by |
|---|---|---|
| D6's category listing perturbs a catalog golden pair (plan R1; `nlq-14` sharpest) | Layer 1 catalog, not-found pairs | TP-003, TP-005, TP-008 |
| F2 prepend changes a golden pair's verdict | Replay with/without F2 | TP-006, TP-007 |
| D7 hint not followed; pt-BR `name = "periféricos"` without `returns` never reaches D7 (m1) | Browse replies, mechanism | TP-010, TP-011, TP-012 |
| Product-type word ("mouse") over-mapped onto Peripherals | `browse-type` | TP-013 |
| Recovery path adds latency | Turn p95 | TP-014 |
| Live run disturbs shared state (`:8200`, `falkordb-dev`, tracked eval baseline) | Environment | TP-000, TP-015, TP-016 |

## Environment & data setup

- LM Studio at `localhost:1234`: exactly `mistralai/ministral-3-3b` (ctx 8192) +
  `text-embedding-qwen3-embedding-0.6b`, loaded via `scripts/ensure_demo_models.sh` (never unloads).
- `reference` seeded with the catalog; `ws:nlq-eval` seeded. `verify_salesperson.sh demo` and
  `verify_catalog.sh` rc 0 before and after.
- `:8200` agent-team server: up, left untouched, idle (log watched).
- Scratch: `/tmp/claude-1000/-home-mauricio-prg-graphmind-ai-lab/c6bf0232-c1ae-4f33-8aa9-de25f4a9b94a/scratchpad/stageC/`
  (models json copy, Run 0 export, results, probe JSONL, proxy log).
- Mechanism capture for §8.3: an observe-only proxy (`:1235 → :1234`, request + response bodies
  per line, as id2 did), with the demo pointed at it through a scratch copy of
  `opencode.local.json`. Required for B6 and m1 (the storefront probe does not record the
  internal querygen completion).
- Harness change sanctioned by plan §7: one `SCRIPTS` entry `browse-type`
  ("Vocês têm algum mouse?") in `salesperson/scripts/refire_probe.py`.

## Test items

| ID | Title | Steps | Expected (§8 target) | Pri | Type |
|---|---|---|---|---|---|
| TP-000 | Preconditions | models loaded as above; `:8200` idle; verify scripts rc 0 | as stated | P0 | env |
| TP-001 | Run 0 diff guard | `git -C <top> diff --quiet e8798351 6286de44^ -- falkor-chat/server` | rc 0 | P0 | integrity |
| TP-002 | Run 0 export + import check | `git archive e8798351 falkor-chat/server falkor-chat/config` → scratch; print `falkorchat.__file__` with runner's `sys.path` order | path under scratch | P0 | integrity |
| TP-003 | Run 0 / Run 1 eval | §8.2 command on both trees; must print `step model: lmstudio/mistralai/ministral-3-3b` | both complete | P0 | e2e |
| TP-004 | L1 catalog Layer 1, Run 1 | from Run 1 JSON | **21/21** | P0 | acceptance |
| TP-005 | L2 KB Layer 1, Run 1 | `nlq_compare.py` Run 0 vs Run 1 | equal pair-by-pair except replay-attributed F2 diffs; floor ≥ 12/19 | P0 | acceptance |
| TP-006 | L3 not-found false answers | Run 1 | **0/6** | P0 | acceptance |
| TP-007 | L4/L5 replay | scratch script: parse → compile with/without `ensure_row_identity` → execute → `score_pair` | L4 **0/40**; L5 reported, each non-zero listed | P0 | contract |
| TP-008 | L6 per-pair flips | `nlq_compare.py`; list every catalog pair whose completion changed; watch list | **0** catalog correct→incorrect | P0 | acceptance |
| TP-009 | Demo up + def drift | `start_demo.sh` (via proxy config); `verify_salesperson.sh demo` | rc 0 | P0 | env |
| TP-010 | `browse` re-measure B1–B4 | `refire_probe.py run --languages pt-BR,es --schedule browse --repeat 12`; hand-classify | B1 0/24, B2 ≥ 20/24, B3 ≤ 2/24, B4 0/0 | P0 | acceptance |
| TP-011 | B6 mechanism | proxy log per turn: first querygen filter, D7 fired, `filter_products(category="Peripherals")` followed | reported per language | P1 | exploratory |
| TP-012 | m1 `returns` tally | for each pt-BR browse miss: did the querygen completion carry `returns` | reported | P1 | exploratory |
| TP-013 | B5 `browse-type` | `--languages pt-BR --schedule browse-type --repeat 12` | reported, ungated | P2 | exploratory |
| TP-014 | B7 turn p95 by path | probe `turn_ms` split by path | reported next to id2 (3,604 ms p95-class figure from id2 table is slot-based; id2 turn p95 cited as recorded there) | P2 | non-functional |
| TP-015 | Tracked eval baseline untouched | `git status --short tests/eval/` from `falkor-chat/server` | prints nothing | P0 | integrity |
| TP-016 | Post-run verify | `verify_salesperson.sh demo`, `verify_catalog.sh` | rc 0, reported | P0 | env |

## Entry / exit criteria

- **Entry:** TP-000 and TP-001 pass. If `:8200` stays busy, the live items are blocked.
- **Exit:** every item has an observed outcome or a stated block. F2 verdict = L1–L4, L6; DEF-C
  verdict = B1–B4.

## Out of scope

§8.1 offline suite and probe (implementer gate, already recorded); C1/C2 (K-085); the RC-1 en
price-list incidence; an English browse control (plan §8.3 reasoning); R7 execution failures
(pre-existing — noted if seen, not counted as regressions); Layer 2 (non-gating) beyond noting it.
