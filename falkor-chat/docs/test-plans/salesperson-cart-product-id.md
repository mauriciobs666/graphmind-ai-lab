# Cart writes keyed by product ID — live acceptance check (S7)

> **Status:** archived · **Owner:** `qa-engineer` · **Tracks:** K-078 (+ K-079 F3) · **Extended by:** `falkor-chat/docs/test-plans/salesperson-cart-product-id2.md`

## Scope & objective

This is the live check for `salesperson@v9`. That version has three parts:
- `add_to_cart` and `remove_from_cart` take `productId`;
- the K-058 guard grounds on where the ID came from;
- a missed `lookup_product_fact` returns the catalog list.

The check runs on the storefront demo (`start_demo.sh`, ministral-3-3b at 8192, PARALLEL 4, plus
the embedder), against the acceptance bar in `falkor-chat/docs/plans/salesperson-cart-product-id.md`
§6.3 (v3), with the stakeholder's bar from 2026-10-22. C1 is gated. C2 and the code-review additions
are reported, not gated.

## References

- Plan: `falkor-chat/docs/plans/salesperson-cart-product-id.md`, §2 (baseline), §3 (D1–D7), §6.3, §8.
- Plan gate: `falkor-chat/docs/reviews/salesperson-cart-product-id.md` (Appendix A, baseline views).
- Code review: `falkor-chat/docs/reviews/salesperson-cart-product-id-impl.md`. This plan uses
  Pass 1 F4 (the miss→`query_graph_data` tag) and the Pass 3 open questions (first-lookup miss rate,
  confirmation-turn add).
- Baseline: `falkor-chat/docs/test-reports/salesperson-single-model-report.md` Q3 (n = 75, `v8`).
- Sweep method: `falkor-chat/docs/reviews/salesperson-fabricated-prices-rca.md` §2 step 5.
- Code: `4dac501e` (S1–S3) and `8a72f08d` (S4–S6, `v9`).

## Risk assessment

| Risk | Why it matters | Covered by |
|---|---|---|
| pt/es models still fail to reach the ID (translated lookup, no pick from the list) | the item's whole purpose | TP-001, TP-008 |
| en regresses (extra iterations, guessed slugs held) | the demo's main language | TP-001 |
| R3: the guard is one decision deep, so a later turn re-adds | doubled purchases | TP-003, TP-010 |
| R4: more iterations and a bigger prompt raise latency and KV/context pressure under 3-way concurrency | dead turns, the 29 s line | TP-004, TP-005 |
| the miss list invites price misjoins (impl F4) | wrong prices stated | TP-009 |
| the v9 prompt's "current message only" refuses a confirmation add | silent no-op for a real request | TP-011 |

## Test items

| ID | Title | Steps | Expected (gate or report) | Pri | Type |
|---|---|---|---|---|---|
| TP-001 | M1 per language | C1: probe, 75 trials. `cart_outcome_analysis.py` reads `ws:demo` with RO queries | pt-BR ≥ 60/75, es ≥ 60/75, en ≥ 65/75 | P1 | e2e |
| TP-002 | M2, other-product lines | same | ≤ 2/225, each one inspected by hand | P1 | e2e |
| TP-003 | M3, mouse quantity > 1 (cart + order) | same | ≤ 2/225, each one inspected by hand | P1 | e2e |
| TP-004 | Dead turns | probe records (`dead_turn`), with the demo log for attribution | ≤ 1.2% of 1,125 | P1 | non-functional |
| TP-005 | Turn p95 (all / turn 2, pooled) and turn-2 p95 per language | probe `turn_ms`, nearest-rank, replied turns | ≤ 29 s each. The per-language figure is reported | P1 | non-functional |
| TP-006 | Wrong-language replies | `language_salience_probe.py aggregate` (v2 classifier), with every flagged or unknown reply read by hand | ≤ 3/375 per language | P1 | functional |
| TP-007 | K-058/K-061 holds, recovery rate, turn-2 tool mix | demo log plus the LLM request log, rebuilt per turn | reported | P2 | integration |
| TP-008 | First-lookup miss rate per language | LLM request log: the first `lookup_product_fact` of each add turn, judged by its result and by whether its argument is an exact catalog name | reported | P2 | integration |
| TP-009 | Name→price sweep and non-`$` currency | every reply: catalog name paired with the stated price. Each wrong pair is tagged miss→`query_graph_data` yes/no | reported. Baseline: 31 wrong of 79 (RCA); € in 7 replies | P2 | functional |
| TP-010 | C2 re-fire | `refire_probe.py`: same-category and distinct scripts, 12 each. en and pt-BR at n = 24, es optional. Read graph M3 | reported against the pre-guard 1/24. **≥ 3/24 in either language goes to the stakeholder** | P1 | e2e |
| TP-011 | Confirmation-turn add | `refire_probe.py` `confirm` script ("Do you have…?" → "Yes, add it to my cart."), plus any add on a non-add turn in C1 and C2 | reported: did the add happen, and was it held? | P3 | exploratory |

## Environment & data

- **Bring-up:** `start_demo.sh`, which seeds and verifies `salesperson@v9`. `verify_salesperson.sh demo`
  and `verify_catalog.sh` must both exit 0.
- **LM Studio:** exactly the demo pair loaded. A log-only watcher records any foreign model.
- **Instrumentation (approved by `teco`):** an observe-only HTTP proxy (`:1235` → `:1234`).
  - It logs every `/v1/chat/completions` request and response body. That is the only source for
    per-turn tool arguments and results (`trace` is off, and LM Studio does not log requests).
  - It reaches the server through a scratch copy of `opencode.local.json`, kept outside the repo,
    whose baseURL is `:1235`.
  - Its overhead is measured and reported.
- **Data:** participant IDs come from the probe JSONL files. The graph is read through
  `GRAPH.RO_QUERY` only, filtered by participant.

## Entry / exit criteria

- **Entry:** both verify scripts are OK, only the demo models are loaded, `:8200` is idle, and the
  probe harness hash matches the baseline's (`5949774a…`).
- **Exit:** C1 has 75 complete trials; C2 has n = 24 per language; the report is written.

## Out of scope

- Offline unit tests: done at the code gate.
- The full pytest suite: it wipes `reference`.
- K-079 F2 (`query_graph_data` identity column), which has not shipped.
- Any code or prompt fix.
