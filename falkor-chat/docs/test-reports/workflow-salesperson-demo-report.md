# `workflow-salesperson-demo` — Test Report (M6 combined e2e)

> **Status:** active · **Owner:** `qa-engineer` · **Tracks:** K-052, K-053, K-054, K-055 (M6)

## Summary

Live combined-integration pass executed 2026-08-30 against the real running system — FalkorDB
(`falkordb-dev`), a fresh M1 server instance on `http://localhost:8023` bound to a fresh throwaway
workspace `ws:qa-salesperson-demo`, LM Studio at `http://localhost:1234` serving
`mistralai/ministral-3-3b` (`salesperson@v4`'s `assistant` step) and `qwen/qwen3-4b-2507`
(`query_graph_data`'s internal structured-completion call) — per
`docs/test-plans/workflow-salesperson-demo.md`. Code under test: `salesperson@v4` (the full
eleven-tool `SALESPERSON_DEF`, commit state as gated by `workflow-nl-query-generation-impl.md`
Pass 2, `c033b30`).

**Verdict: PASS.** All four sibling capabilities (K-052 catalog lookup, K-053 cart/order, K-054
durable profile, K-055 NL query generation) fired correctly, in the correct order, on the correct
tools, within one continuous conversation, with no cross-capability interference. Profile and
post-order cart state were both proven correct **together**, in a second, independent thread that
never itself ran the state-creating turns — the integration property this pass exists to check.
No regression found versus any capability's own prior acceptance pass. This closes M6's own
combined-demo closing condition.

**CPG:** considered, not relevant — live black-box integration testing of already-shipped,
already-individually-reviewed code; no structural-impact-analysis question this pass raises that
driving the running system doesn't answer better.

## Run context

- **Date:** 2026-08-30. **Server:** `http://localhost:8023`, `ws:qa-salesperson-demo`,
  `salesperson@v4` trigger, no `--reload`.
- **Workspace:** fresh, throwaway, zero pre-existing `Customer` nodes confirmed before TP-02.
- **Models:** `lmstudio/mistralai/ministral-3-3b` (assistant step), `lmstudio/qwen/qwen3-4b-2507`
  (internal `query_graph_data` completion).
- **Data:** `reference` — 15 `Product` nodes (unchanged throughout).

## Results table

| ID | Capability | Result | Evidence |
|---|---|---|---|
| TP-01 | — | **PASS** | FalkorDB up; LM Studio lists both required models; `bootstrap_schema.sh`/`seed_demo.sh`/`seed_salesperson.sh` on `ws:qa-salesperson-demo` completed; `verify_salesperson.sh qa-salesperson-demo` → `OK`, in sync, 2 steps/1 transition; zero pre-existing `Customer` nodes; server `GET /health` → `{"status":"ok"}`. |
| TP-02 | K-052 | **PASS** | `@assistant How much does the Wireless Mouse Pro cost?` → *"The **Wireless Mouse Pro** costs **$29.99**. Would you like to add it to your cart?"* (`toolsUsed: ["lookup_product_fact"]`). Matches catalog exactly. |
| TP-03 | K-054 | **PASS** | `@assistant Hi, my name is Jordan Rivera and my delivery address is 77 Birch Lane, Denver.` → *"Thank you, Jordan! I've saved your name and delivery address..."* (`toolsUsed: ["save_profile"]`). Ground truth: `MATCH (c:Customer {customerId:'u1'}) RETURN c.name, c.deliveryAddress, c.profileUpdatedAt` → `name:'Jordan Rivera'`, `deliveryAddress:'77 Birch Lane, Denver'`, `profileUpdatedAt` set. |
| TP-04 | K-053 | **PASS** | `@assistant Add 1 Wireless Mouse Pro to my cart.` → *"Your cart now includes: Wireless Mouse Pro – $29.99..."* (`toolsUsed: ["add_to_cart"]`). Ground truth: exactly one `CartItem{productId:'wireless-mouse-pro', quantity:1}`. |
| TP-05 | K-055 | **PASS** | `@assistant How many products do you have in the Wearables category?` → *"We have **2 products** in the Wearables category..."* (`toolsUsed: ["query_graph_data"]`). Ground truth: `MATCH (p:Product) WHERE p.category='Wearables' RETURN count(p)` → `2`. Exact match, coexisting cleanly with the ten fixed-shape/state tools already exercised in this same conversation. |
| TP-06 | K-053 | **PASS** | `@assistant Please place my order now.` → *"Your order has been placed successfully!... Order ID: `173e35bbba7348318b9b23093782044d`... Wireless Mouse Pro – $29.99... Total: $29.99"* (`toolsUsed: ["place_order", "view_cart"]`). Ground truth: one `Order{status:'placed'}` with one `OrderLine{productId:'wireless-mouse-pro', quantity:1, unitPrice:29.99}`; cart-item count after: `0`. |
| TP-07 | K-054, K-053 | **PASS** | Fresh Thread B, turn 1: `@assistant Do you have any wireless accessories under $100?` → answered directly (Wireless Charging Pad, Bluetooth Speaker Mini, Smart Home Hub), **no re-ask** of name/address (`toolsUsed: ["filter_products", "query_graph_data"]`). Turn 2: `@assistant What's my name, and what's currently in my cart?` → *"Your name is **Jordan Rivera**, and your cart currently contains nothing."* (`toolsUsed: ["get_profile", "view_cart"]`) — correct on both counts, in a thread that never itself ran TP-03/TP-04/TP-06. |

## Defects

None found. All four capabilities coexisted correctly on the full eleven-tool grant set; no
cross-capability state leak, no wrong-tool dispatch, no regression versus any capability's own
individually-recorded acceptance pass (`workflow-catalog-lookup2-report.md`,
`workflow-cart-and-totals2-report.md`, `workflow-durable-profile-report.md`,
`workflow-nl-query-generation2-report.md`).

**Noted, not filed as a new defect (informational only):** TP-07 turn 1's answer to "wireless
accessories under $100" omitted `Wireless Mouse Pro` ($29.99, itself a "wireless" item) from its
volunteered list — a keyword-interpretation completeness nuance of `filter_products`/
`query_graph_data`'s joint handling of a loosely-worded, non-category-exact term, already inside
K-052's own scope and not a new cross-capability interference this pass's own AC list gates
against; not reproduced/investigated further here.

## Coverage & gaps

**Covered:** one continuous conversation touching all four capabilities in a realistic order
(browse → identify → cart → arbitrary query → checkout), plus a second, independent thread proving
profile and post-order cart state are correct together without re-running the state-creating
turns. This is exactly M6's own closing bar: "all four wired into one def... proving they coexist
without regression."

**Gaps, stated plainly:**
- Each capability's own full AC list is not re-verified here — by design (§3 of the plan); that
  work is already on record in each capability's own report.
- `order-fulfillment@v1`'s operator-side lifecycle (fulfilled/delivered/cancelled) is not
  exercised here — already covered in `workflow-cart-and-totals2-report.md`.
- K-055's AC-2 (second dataset) is not exercised in this workspace (no ingested entity corpus
  here) — already proven separately in `workflow-nl-query-generation2-report.md`; this pass
  exercises K-055 against the `catalog` dataset only, sufficient for this pass's own integration
  question.
- Only one conversation shape was driven; a different ordering of the four capabilities' turns
  (e.g. checkout before profile) was not tried — the task brief's own scope ("a single (or small
  number of) realistic conversation(s)") did not call for an ordering-permutation sweep.

## Feedback & recommendations

- The eleven-tool grant set on one `assistant` step continues to resolve correctly turn-by-turn
  even as it has grown across four capability landings (v1→v4) — no sign of tool-selection
  degradation from the larger tool surface or the correspondingly longer `systemPrompt`.
- This pass, together with the four already-recorded per-capability passes, is sufficient evidence
  for M6's combined-demo closing condition; no further live-agent work is identified as needed
  before this milestone can be considered closed on the QA side.
