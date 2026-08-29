# `workflow-cart-and-totals` — Test Plan (Ministral re-verification)

> **Status:** active · **Owner:** `qa-engineer` · **Tracks:** K-053, K-056 (M6) · **Extends:** `docs/test-plans/workflow-cart-and-totals.md`

## 1. Scope & objective

Regression re-verification of K-053's full acceptance-test suite against `salesperson@v2.1`, now
re-pointed from `qwen/qwen3-4b-2507` to `mistralai/ministral-3-3b` (K-056, commit `03a3c8c`). Same
relationship to its parent plan as `workflow-catalog-lookup2.md` has to its own: the feature under
test (cart/order tools, `order-fulfillment@v1`) is unchanged; only the model driving the
`assistant` step differs. Methodology, conversation shapes, and the full AC-1..AC-10 list are
reused from `docs/test-plans/workflow-cart-and-totals.md`; this document states deltas and gives
a self-contained test-item list.

**This pass carries the more consequential regression check of the two K-052/K-053 re-runs.** The
parent report's D-1 (`docs/test-reports/workflow-cart-and-totals-report.md`) was the first live
confirmation that qwen's skip-and-fabricate mechanism (K-056) reaches a **write-mutating** tool
(`remove_from_cart`) — the specific escalation the `data-scientist`'s ml-note warned would happen,
and the reason K-056 was prioritized ahead of K-054/K-055. Confirming or refuting that this does
not reproduce on Ministral is this pass's central question.

## 2. References

- Parent plan: `docs/test-plans/workflow-cart-and-totals.md` (full risk assessment, schema/tool
  references, workspace-choice rationale — not repeated here).
- Parent report: `docs/test-reports/workflow-cart-and-totals-report.md` — qwen-era baseline;
  **D-1 (MAJOR, write-mutating fabrication on `remove_from_cart`)** and **D-2 (MINOR,
  `add_to_cart`-onto-non-empty-cart total-display quirk)**, both must be re-checked here.
- Requirements (unchanged): `docs/requirements/workflow-cart-and-totals.md` — AC-1..AC-10.
- Re-point unit: `docs/plans/workflow-salesperson-demo-coordination.md` U43 (commit `03a3c8c`) —
  `SALESPERSON_DEF` `v2.1`, `config.model` on the `assistant` step only.
- Ministral evidence to date: `docs/reviews/salesperson-tool-reliability-ml.md` §8.4 (0/176
  skip-and-fabricate; **3/10 (30%) duplicate-instruction re-trigger on cart-add sequences**),
  §9 (dedicated follow-up eval, pooled 1/32 (3.1%), honestly-grounded/self-disclosing, judged
  non-blocking, go-recommendation for piloting). **This is the pass's real watch item** — K-053's
  own tool surface (`add_to_cart`/`remove_from_cart`/`clear_cart`/`place_order`) is exactly the
  surface the ml-note's eval probed, unlike K-052's read-only surface.
- Code under test (unchanged from K-053 except `proof_defs.py`'s `config.model` field):
  `server/falkorchat/pricing.py`, `repository.py` §16 methods, `services.py` cart/order methods,
  `tools.py` cart tools, `proof_defs.py` (`SALESPERSON_DEF` v2.1, `ORDER_FULFILLMENT_DEF`),
  `scripts/{seed,verify}_salesperson.sh`.

**CPG:** considered, not relevant — live acceptance re-run against a changed model dependency, no
code-shape change to analyze structurally.

## 3. What's different from the parent plan

- **Model:** `mistralai/ministral-3-3b`, reached via the same session-scratch
  `FALKORCHAT_OPENCODE_CONFIG` override (corrected `baseURL: http://localhost:1234`) as the
  catalog-lookup2 pass and every prior unit in this coordination.
- **Def version:** `salesperson@v2.1` (was `v2` at the original pass) — topology unchanged.
- **Fresh workspace:** `ws:qa-cart-totals2` (not the original `ws:qa-cart-totals`, not `ws:acme`)
  — same durable-write-isolation rationale the parent plan gives (§3 there), reused verbatim.
- **D-1 gets a dedicated, verbatim repro item (TP-04), not folded into a general "does AC-2 hold"
  check** — TP-04 below reruns the *exact* 4-turn sequence the parent report's D-1 reproduced on,
  because a regression check for a specific reported defect is strongest when it repeats the
  precise repro steps rather than a fresh, potentially-easier-to-pass variant.
- **New watch item (informational, not a gate): TP-16**, below — a dedicated 2-turn
  add-then-add-different-category conversation mirroring the ml-note §9's own
  `same-category-immediate`/`distinct-immediate` conditions (the axis with the one confirmed
  ministral duplicate in the pilot eval), plus opportunistic review of every other turn's
  `toolsUsed`/trace for an uninstructed repeat call. This is watched, not gated, per the brief:
  already judged non-blocking at pilot time.
- **D-2 re-check (TP-08) is retained but not re-hunted** — per the brief, D-2 is judged
  model-independent (display formatting); TP-08 reuses the same add-onto-non-empty-cart step the
  parent report's TP-08 used and simply records whether the same wording quirk appears, without
  extra investigation either way.

Everything else — the per-scenario short-conversation discipline, the ground-truth-Cypher-in-the-
same-item cross-check, the Services-layer direct driving of `order-fulfillment`'s no-REST-route
half, the seed-idempotence check — is reused unchanged.

## 4. Environment & data setup

- FalkorDB: `falkordb-dev`, already up.
- LM Studio: up at `http://localhost:1234`; `mistralai/ministral-3-3b` confirmed listed.
- Server venv: `server/.venv`, already present.
- Fresh throwaway workspace: `ws:qa-cart-totals2` — bootstrapped
  (`EMBEDDING_DIM=1024 ./scripts/bootstrap_schema.sh qa-cart-totals2`), demo-seeded, catalog
  present (global `reference`), `salesperson@v2.1`/`order-fulfillment@v1` materialized
  (`seed_salesperson.sh qa-cart-totals2`) — confirmed `OK`/in-sync via `verify_catalog.sh`/
  `verify_salesperson.sh qa-cart-totals2` before any test item runs.
- Server bound to this workspace: `FALKORCHAT_WS_ID=qa-cart-totals2 FALKORCHAT_ENABLE_AGENT=1
  FALKORCHAT_WORKFLOW_ENABLED=1 FALKORCHAT_TRIGGER_DEF_KEY=salesperson
  FALKORCHAT_TRIGGER_DEF_VERSION=v2.1 FALKORCHAT_EMBEDDING_DIM=1024
  FALKORCHAT_OPENCODE_CONFIG=<scratch copy>`, a fresh port, no `--reload`.
- Same two fixture products as the parent plan: Wireless Mouse Pro ($29.99, Peripherals),
  Portable SSD 1TB ($109.99, Storage); Mechanical Keyboard K200 (Peripherals, same category as
  the mouse) additionally used for TP-16's same-category watch condition.

## 5. Test items

| ID | Title | Priority | Type |
|---|---|---|---|
| TP-01 | Environment pre-flight + fresh-workspace provisioning | P0 | environment/setup |
| TP-02 | AC-1 — add an item to an empty cart, correct total (brand-new-customer path) | P0 | e2e/acceptance |
| TP-03 | AC-1/AC-2 — add a second product, view cart, correct running total | P0 | e2e/acceptance |
| TP-04 | AC-2 — remove item, updated total — **verbatim D-1 repro sequence** | P0 | e2e/acceptance |
| TP-05 | AC-2 — clear the cart entirely | P0 | e2e/acceptance |
| TP-06 | AC-3 — cross-conversation cart persistence (second thread) | P0 | e2e/acceptance |
| TP-07 | AC-4 — deterministic total, hand-calculated cross-check | P0 | functional |
| TP-08 | AC-5/AC-6 — place an order, frozen snapshot survives a later price change (D-2 observation slot) | P0 | e2e/acceptance |
| TP-09 | AC-9 — no LLM call solely for the computation (structural + outcome) | P1 | integration |
| TP-10 | AC-7 — order-fulfillment lifecycle: placed → fulfilled → delivered | P0 | e2e/acceptance |
| TP-11 | AC-8 — cancellation only possible before fulfillment | P0 | e2e/acceptance |
| TP-12 | AC-10 — seed/verify idempotence (second run) | P0 | acceptance |
| TP-13 | Graph-level ground truth cross-check for TP-02..TP-08 | P1 | integration |
| TP-14 | Exploratory — partial vanished-catalog-product checkout | P2 | exploratory |
| TP-15 | Final regression confirmation (offline suite + `test_queries.sh`), shared with K-052's re-run | P1 | regression |
| TP-16 | Informational watch — Ministral duplicate-instruction re-trigger (ml-note §8.4/§9) | P1 | exploratory |

### TP-02 — AC-1: add to an empty cart (brand-new customer)
**Preconditions:** TP-01. Fresh thread. `ws:qa-cart-totals2` has zero prior `Customer`/`Cart`
nodes (confirmed via Cypher).
**Steps:** `@assistant I'd like to add 2 Wireless Mouse Pro to my cart`.
**Expected:** reply confirms 2×$29.99; ground-truth Cypher shows exactly one `Customer`, one
`Cart`, one `CartItem{quantity:2}`.
**Priority:** P0. **Type:** e2e/acceptance.

### TP-03 — AC-1/AC-2: second product, view cart, running total
**Preconditions:** TP-02, same thread.
**Steps:** `@assistant also add 1 Portable SSD 1TB` then `@assistant what's in my cart?`.
**Expected:** total `2×29.99+109.99=169.97`.
**Priority:** P0. **Type:** e2e/acceptance.

### TP-04 — AC-2: remove/decrease — verbatim D-1 repro
**Preconditions:** TP-03, same thread (this is turn 4 of the same continuous conversation the
parent report's D-1 reproduced on: add mouse×2 → add SSD×1 → view cart → remove mouse×1).
**Steps:** `@assistant remove 1 Wireless Mouse Pro from my cart`.
**Expected:** `remove_from_cart` genuinely dispatched (`Message.toolsUsed` non-empty); ground-truth
Cypher shows `wireless-mouse-pro` quantity 2→1 immediately after; reply matches (1 Wireless Mouse
Pro + 1 Portable SSD 1TB, total $139.98). **This is the exact turn/sequence position the parent
report's D-1 fabricated on for qwen** — a clean pass here, cross-checked against
`Message.toolsUsed` and ground-truth Cypher (not reply text alone), is the direct disconfirmation
this pass exists to obtain. Any fabrication (non-empty confirming reply, empty/absent
`toolsUsed`, unchanged ground truth) is D-1's regression and gets one same-thread rerun, then a
fresh-thread isolation check, mirroring the parent methodology.
**Priority:** P0. **Type:** e2e/acceptance.

### TP-05 — AC-2: clear the cart
**Steps:** `@assistant clear my cart`. **Expected:** empty, total 0, ground-truth confirms zero
`CartItem` rows. **Priority:** P0. **Type:** e2e/acceptance.

### TP-06 — AC-3: cross-conversation persistence
Two independent threads, same actor; second thread's `view_cart` shows the first thread's added
item. **Priority:** P0. **Type:** e2e/acceptance.

### TP-07 — AC-4: deterministic total, hand-calculated cross-check
Compare live total against hand calculation; call `pricing.compute_line_total` twice via the venv,
assert byte-identical (model-independent, structural). **Priority:** P0. **Type:** functional.

### TP-08 — AC-5/AC-6: place an order, frozen snapshot survives a later price change
**Steps:** build a known cart (1 Wireless Mouse Pro + 1 Portable SSD 1TB — note whether the
`add_to_cart` reply onto the already-non-empty cart shows the parent report's D-2 total-omission
wording or not, recorded but not chased further); `@assistant place my order`; mutate
`Product.price` for the mouse directly in `reference` (revert after); re-read via `get_order`.
**Expected:** order confirms 2 lines, total $139.98; snapshot unchanged after the price mutation.
**Priority:** P0. **Type:** e2e/acceptance.

### TP-09 — AC-9: no LLM call solely for the computation
Structural code read (unchanged code, re-confirmed) + outcome-level evidence (exact, repeatable
totals across TP-02..TP-08). **Priority:** P1. **Type:** integration.

### TP-10 — AC-7: order-fulfillment lifecycle, happy path
`POST /workflow-runs` against TP-08's placed order; `submit_workflow_input`
`fulfill`→`deliver`, each paired with a direct `services.advance_order` call.
**Priority:** P0. **Type:** e2e/acceptance.

### TP-11 — AC-8: cancellation only possible before fulfillment
A second live-placed order; cancel-before-fulfillment succeeds; deliver-after-cancel and
cancel-after-fulfilled both no-op. **Priority:** P0. **Type:** e2e/acceptance.

### TP-12 — AC-10: seed/verify idempotence
Re-run `seed_salesperson.sh qa-cart-totals2`; re-run `verify_salesperson.sh qa-cart-totals2`.
**Priority:** P0. **Type:** acceptance.

### TP-13 — Graph-level ground truth cross-check
Cypher cross-check inline after every mutating turn in TP-02..TP-08, not deferred.
**Priority:** P1. **Type:** integration.

### TP-14 — Exploratory: partial vanished-catalog-product checkout
Reuses the parent report's own TP-14 method (Services-layer direct driving, disposable product) —
confirmatory only, per the implementation review's accepted disposition, model-independent.
**Priority:** P2. **Type:** exploratory.

### TP-15 — Final regression confirmation
Run last, shared with `workflow-catalog-lookup2.md`'s own TP-15/final step (both plans converge on
one shared final regression pass, not duplicated): `server/.venv/bin/python -m pytest -q` and
`./scripts/test_queries.sh`; both wipe `reference`; reseed (`bootstrap_schema.sh`→`seed_demo.sh`→
`seed_catalog.sh`→`seed_workflows.sh`→`seed_salesperson.sh`) and re-verify
(`verify_workflows.sh`/`verify_catalog.sh`/`verify_salesperson.sh acme`) before handing back.
**Priority:** P1. **Type:** regression.

### TP-16 — Informational watch: Ministral duplicate-instruction re-trigger
**Preconditions:** none additional — draws on TP-02..TP-08's own turns, plus one dedicated
2-turn probe mirroring the ml-note's own highest-signal condition.
**Steps:** (a) review `Message.toolsUsed`/trace for every mutating turn in TP-02..TP-08 for any
tool call whose target product is not named in that turn's own trigger text (the ml-note §9.4
mechanism signature); (b) run one dedicated fresh-thread probe: `@assistant add 1 Wireless Mouse
Pro to my cart` then `@assistant also add 1 Mechanical Keyboard K200` (same-category-immediate,
the ml-note's one confirmed-duplicate condition), reading ground-truth `CartItem` quantities
afterward.
**Expected:** not gated — report whatever is observed. A duplicate would show as a `CartItem`
quantity exceeding what was ever explicitly requested (e.g. the mouse ending up at quantity 2
when only 1 was ever asked for).
**Priority:** P1. **Type:** exploratory.

## 6. Entry / exit criteria

**Entry:** TP-01 passes.

**Exit / verdict rule (unchanged from the parent plan):**
- **Pass** — TP-02..TP-13 (all P0/P1) pass on first attempt or after at most one reproducibility
  rerun, TP-15 confirms no regression. Verdict: "K-053 continues to meet its acceptance criteria
  against the live running system on `mistralai/ministral-3-3b`."
- **Pass with defects** — one or more non-blocking issues found — reported with severity, verdict
  still ships.
- **Fail** — any of AC-1..AC-10 reproducibly fails, **or** the qwen-era D-1 write-mutating
  fabrication mechanism reproduces on Ministral.

## 7. Out of scope

- Everything the parent plan already scoped out (§7 there): re-deriving unit/integration
  coverage, multi-tenant cart isolation (still gated on K-016), re-litigating the implementation
  review's already-disposed findings beyond TP-14's confirmatory check, K-054/K-055, browser/UI
  automation.
- A full statistical re-characterization of Ministral's duplicate-instruction rate — already done
  at n=32+10 by the ml-note's dedicated eval; TP-16 here is an opportunistic watch during AC
  testing plus one confirmatory probe of the pilot's own highest-signal condition, not a fresh
  eval campaign.
