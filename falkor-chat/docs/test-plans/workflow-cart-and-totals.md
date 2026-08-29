# `workflow-cart-and-totals` — Test Plan

> **Status:** active · **Owner:** `qa-engineer` · **Tracks:** K-053 (M6)

## 1. Scope & objective

Live acceptance pass for K-053 (durable cart/order state, deterministic totals, order-fulfillment
lifecycle) — the second business-entities capability layered onto the shared `salesperson` demo
agent, on top of the already QA-passed K-052 (catalog lookup, `docs/test-reports/
workflow-catalog-lookup-report.md`, PASS WITH DEFECTS/D-1, unrelated to K-053's own code). Both
prior K-053 gates (`analyst`'s plan-gate review, `docs/reviews/workflow-cart-and-totals.md`
Pass 2 approve, and its implementation review, `docs/reviews/workflow-cart-and-totals-impl.md`,
approve with suggestions) are static/fixture-level or offline-`pytest`-level. This plan exercises
the real `salesperson@v2` demo agent (real local LLM, LM Studio `qwen/qwen3-4b-2507`), the real
`order-fulfillment@v1` process def, and the real FalkorDB `reference`/`ws:<id>` graphs, against
**AC-1..AC-10** of `falkor-chat/docs/requirements/workflow-cart-and-totals.md`.

## 2. References

- Requirements: `falkor-chat/docs/requirements/workflow-cart-and-totals.md` — FR-1..FR-9,
  AC-1..AC-10 (the acceptance bar for this pass).
- Gated plan: `falkor-chat/docs/plans/workflow-cart-and-totals.md` v2 (`architect`, approved) —
  FR-8 mechanism decision (§3.1), schema integration (§3.2), cart/order tools (§3.3),
  order-fulfillment process def (§3.4).
- Graph design: `falkor-chat/docs/plans/workflow-cart-and-totals-graph.md` v2 (`graph-dba`,
  approved) — schema, DDL, live-verified Cypher (§0/§9).
- Plan-gate review: `falkor-chat/docs/reviews/workflow-cart-and-totals.md` — Pass 2 verdict
  **approve** (both Pass-1 findings, the `ensure_customer`/`ensure_cart` MAJOR and the
  price-change-race MINOR, closed).
- Implementation review: `falkor-chat/docs/reviews/workflow-cart-and-totals-impl.md`
  (`analyst`, **approve with suggestions**, 0 BLOCKER/MAJOR, 2 MINOR, 1 nit) — both plan-gate
  findings independently confirmed closed in code; new MINOR (partial vanished-catalog-product
  checkout silently drops the line, untested, out of scope per graph note §8); new MINOR
  (commit-message test-count typo, no code action); nit (`AddToCartTool` treats `quantity: 0` as
  omitted). **Explicitly flags for this pass:** `services.advance_order` has no REST route this
  milestone — drive that half through the Services layer directly, not HTTP.
- Canonical Cypher: `falkor-chat/docs/QUERIES.md` §16 (Cart/Order).
- Code under test: `server/falkorchat/pricing.py`, `repository.py` §16 methods, `services.py`
  cart/order methods (`add_cart_item`/`get_cart`/`remove_cart_item`/`clear_cart`/`place_order`/
  `get_order_status`/`advance_order`), `tools.py` (`ViewCartTool`/`AddToCartTool`/
  `RemoveFromCartTool`/`ClearCartTool`/`PlaceOrderTool`), `proof_defs.py` (`SALESPERSON_DEF` v2,
  `ORDER_FULFILLMENT_DEF`), `scripts/{seed,verify}_salesperson.sh`.
- Precedent for live-driving a `conversation`-kind def via REST + `@mention`, and for
  Services-layer direct driving of a `process`-kind def: `docs/test-plans/
  workflow-catalog-lookup.md` (`@mention` technique) and `server/tests/test_order_fulfillment.py`
  (Services-layer driving pattern — mirrored here against the real, non-test def and a real,
  live-placed order rather than a `v1-test` fixture def).

**CPG:** considered, not relevant — per the coordinator's brief, `cpg_falkorchat` is stale (built
2026-08-26T22:27:22Z, several commits behind including all of K-053's own) and this is
acceptance testing of already-implemented, reviewed code via live execution, not a
structural-impact-analysis question. Read `services.py`/`tools.py`/`repository.py`/
`proof_defs.py` directly instead (done, above, before writing this plan).

## 3. Risk assessment

**What's already covered, and not re-tested here:** unit/integration correctness of every
repository method against a fixture graph, `pricing.compute_line_total`'s 14 unit tests (empty/
one/many lines, every malformed-input shape, non-mutation, identical-inputs-identical-output),
`Tool.run` dispatch against a fake `Services`, the `ensure_customer`/`ensure_cart` call-ordering
regression test, the AC-9 trace-count test through the real `AddToCartTool`/`_run_agent_node`
path, and `test_order_fulfillment.py`'s full offline drive of the fulfillment lifecycle against a
`v1-test` def — all independently re-verified by `analyst` (U21, implementation review) by close
reading, call-ordering assertions, node-count probes, and `GRAPH.PROFILE` index-scan checks.
Re-deriving that here would be duplicate unit-layer coverage.

**What genuinely hasn't been checked yet, and is this pass's real risk surface:**
- **Does the real LLM actually drive the five cart tools correctly from natural language,
  end-to-end, across a multi-turn shopping session?** No unit test proves a real model calls
  `add_to_cart`/`remove_from_cart`/`place_order` with correct arguments from a customer's
  free-text request, or that the customer-visible reply correctly reflects what the tool actually
  returned (not a fabrication) — exactly the class of live-only risk K-052's own D-1 defect
  (`docs/test-reports/workflow-catalog-lookup-report.md`) already demonstrated is real for this
  exact model/harness. **Known, disclosed, open risk (K-056, unresolved):** the model can skip
  tool invocation entirely and fabricate a reply on a later turn of an extended conversation
  (collapse point ~6-8 short messages per the `data-scientist` root-cause note,
  `docs/reviews/salesperson-tool-reliability-ml.md`). **Mitigation for this pass:** keep each
  scenario's own conversation short (well under the documented collapse point) and — per the
  lesson U17 already banked — never trust the model's natural-language reply alone for a
  mutating action; cross-check every cart/order mutation against ground-truth Cypher
  (`redis-cli GRAPH.QUERY ws:qa-cart-totals ...`) in the same test item, not a separate one.
- **`add_cart_item`'s brand-new-customer path (the closed MAJOR) has never been proven against a
  real conversational turn** — only against `FakeRepo` call-order assertions and direct
  Services-layer calls (`test_order_fulfillment.py` seeds the `Customer`/`Order` directly, never
  going through `add_cart_item`). This pass's very first cart-mutating turn, against a workspace
  with **zero** prior `Customer`/`Cart` nodes, is the first live proof this doesn't silently
  no-op.
- **AC-3 (cross-conversation persistence)** is inherently a live, multi-thread question — no
  offline test starts two separate `WorkflowRun`s in two separate `Thread`s and confirms the
  second sees the first's cart contents through the real tool-calling path (`test_services.py`
  proves the repository/service layer directly; this pass proves the tool-calling harness doesn't
  lose or scope the identity differently across runs).
- **AC-6 (price-change-proof snapshot)** needs a real catalog mutation between cart-build and
  placement, then a real re-read of the placed order — an integration-level claim the offline
  suite covers at the repository/service layer but not through the live tool surface a customer
  actually drives.
- **AC-7/AC-8 (order-fulfillment lifecycle) have never been driven against a real, live-placed
  order** — `test_order_fulfillment.py` seeds its own synthetic `Order` directly via
  `repository.place_order`, bypassing the cart/checkout path and the real `order-fulfillment@v1`
  def (it publishes a `v1-test` version). This pass starts a real `order-fulfillment@v1` process
  run (`POST /workflow-runs`, no chat trigger, per plan §3.4) against an order this same pass just
  placed live through the chat surface, and drives it with real `submit_workflow_input` calls,
  pairing each with the real `services.advance_order` CAS (driven directly, since — per the
  implementation review's own flagged gap — no REST route exists for it this milestone).
- **AC-9 (no LLM call solely for the computation)** — the plan's own verification method
  (`docs/plans/workflow-cart-and-totals.md` §3.1) requires `trace: true`, which (per K-052's own
  test plan §3, still true here) is off by default on the `@mention` trigger path. Same disclosed
  substitution as K-052's plan: proven by observable outcome (an exact, correct, repeatable total
  across independent turns) rather than a literal trace diff, **plus** a direct read of
  `executor.py`'s dispatch path (already done in §2 above, and confirmed again during execution)
  as the structural half of the argument. Recorded as a coverage gap, not silently substituted.
- **AC-10 (seed/verify)** — already exercised once in this pass's own environment setup
  (`bootstrap_schema.sh`/`seed_demo.sh`/`seed_salesperson.sh` against a fresh workspace); this
  plan additionally re-runs `seed_salesperson.sh`/`verify_salesperson.sh` a second time to prove
  idempotence, mirroring K-052's TP-14.

**Workspace choice — a fresh, dedicated `ws:qa-cart-totals`, not the shared `ws:acme`.** Unlike
K-052 (catalog lookup — read-only), K-053 writes durable, workspace-scoped `Customer`/`Cart`/
`Order` state under the hardcoded actor `u1`. Testing against `ws:acme` would leave test cart/
order data mixed into the shared demo workspace future work reads, and — since M1's auth seam
resolves every call to the single actor `u1` (plan §6's single-hardcoded-actor caveat) — a prior
demo session's cart state in `ws:acme` could also confound this pass's own "given an empty cart"
preconditions. A fresh workspace gives a known-clean starting state and leaves no residue in
`ws:acme`. This is the same judgment K-052's own test plan made (`ws:qa-catalog-lookup`), extended
here for a stronger reason (K-053 mutates, K-052 only read).

**Regression baseline — deliberately not re-run.** `analyst` (U21) already ran the two read-only
live checks (`verify_salesperson.sh`, `db.constraints()`) on this exact commit set today and
explicitly declined to run the offline `pytest`/`test_queries.sh` suites to avoid disturbing the
live `ws:acme` state it had just confirmed in sync — this pass follows the same discipline through
its own execution (see §7's final regression-confirmation step, run only *after* all live items
complete, as a clearly separate, final step per the coordinator's brief).

## 4. Environment & data setup

- FalkorDB: `falkordb-dev` container, already up (confirmed `PONG` at 127.0.0.1:6379).
- LM Studio: up at `http://localhost:1234`, `qwen/qwen3-4b-2507` listed — reached via a
  session-local `FALKORCHAT_OPENCODE_CONFIG` override pointing `baseURL` at `localhost:1234`
  (mirrored WSL2 networking works this session; the shared `opencode.json`'s gateway-IP baseURL is
  left untouched, same workaround U17 banked).
- Server venv: `falkor-chat/server/.venv` already present.
- **Fresh throwaway workspace: `ws:qa-cart-totals`** — bootstrapped
  (`EMBEDDING_DIM=1024 ./scripts/bootstrap_schema.sh qa-cart-totals`), demo-seeded
  (`seed_demo.sh qa-cart-totals`), and `salesperson@v2`/`order-fulfillment@v1` materialized into
  it (`seed_salesperson.sh qa-cart-totals`) — all confirmed `OK`/in-sync via `verify_catalog.sh`/
  `verify_salesperson.sh qa-cart-totals` before any test item runs.
- Server started bound to this workspace: `FALKORCHAT_WS_ID=qa-cart-totals
  FALKORCHAT_ENABLE_AGENT=1 FALKORCHAT_WORKFLOW_ENABLED=1
  FALKORCHAT_TRIGGER_DEF_KEY=salesperson FALKORCHAT_TRIGGER_DEF_VERSION=v2
  FALKORCHAT_EMBEDDING_DIM=1024`, port 8099, no `--reload` (avoids the documented mid-run worker
  restart hazard).
- Products used as fixtures (from `seed_catalog.sh`'s literal, `reference` graph, global):
  Wireless Mouse Pro ($29.99, Peripherals), Portable SSD 1TB ($109.99, Storage).

## 5. Test items

| ID | Title | Priority | Type |
|---|---|---|---|
| TP-01 | Environment pre-flight + fresh-workspace provisioning | P0 | environment/setup |
| TP-02 | AC-1 — add an item to an empty cart, correct total (brand-new-customer path) | P0 | e2e/acceptance |
| TP-03 | AC-1/AC-2 — add a second product, view cart, correct running total | P0 | e2e/acceptance |
| TP-04 | AC-2 — remove/decrease an item, updated total | P0 | e2e/acceptance |
| TP-05 | AC-2 — clear the cart entirely | P0 | e2e/acceptance |
| TP-06 | AC-3 — cross-conversation cart persistence (second thread) | P0 | e2e/acceptance |
| TP-07 | AC-4 — deterministic total, hand-calculated cross-check | P0 | functional |
| TP-08 | AC-5/AC-6 — place an order, frozen snapshot survives a later price change | P0 | e2e/acceptance |
| TP-09 | AC-9 — no LLM call solely for the computation (trace-off substitution) | P1 | integration |
| TP-10 | AC-7 — order-fulfillment lifecycle: placed → fulfilled → delivered | P0 | e2e/acceptance |
| TP-11 | AC-8 — cancellation only possible before fulfillment | P0 | e2e/acceptance |
| TP-12 | AC-10 — seed/verify idempotence (second run) | P0 | acceptance |
| TP-13 | Graph-level ground truth cross-check for TP-02..TP-08 | P1 | integration |
| TP-14 | Exploratory — partial vanished-catalog-product checkout (impl review's flagged gap) | P2 | exploratory |
| TP-15 | Final regression confirmation (offline suite + `test_queries.sh`), separate final step | P1 | regression |

### TP-01 — Environment pre-flight + fresh-workspace provisioning
**Preconditions:** none.
**Steps:** `redis-cli PING`; `GET http://localhost:1234/v1/models`; bootstrap/seed
`ws:qa-cart-totals` per §4; `verify_catalog.sh`/`verify_salesperson.sh qa-cart-totals`; start
server; `GET /health`.
**Expected:** FalkorDB `PONG`; model listed; both verify scripts `OK`/in-sync; health `{"status":
"ok"}`.
**Priority:** P0. **Type:** environment/setup.

### TP-02 — AC-1: add to an empty cart (brand-new customer)
**Preconditions:** TP-01. Fresh thread in `demo-general`. `ws:qa-cart-totals` has zero prior
`Customer`/`Cart` nodes (confirmed via Cypher before this turn).
**Steps:** `@assistant I'd like to add 2 Wireless Mouse Pro to my cart` in a new thread; poll for
the reply.
**Expected:** reply confirms 2 Wireless Mouse Pro added, unit price $29.99; ground-truth Cypher
(TP-13) shows exactly one `Customer`, one `Cart`, one `CartItem {productId, quantity:2}` —
proving `ensure_customer`/`ensure_cart` actually ran, not a silent no-op (the closed MAJOR
finding, now proven live for the first time).
**Priority:** P0. **Type:** e2e/acceptance.

### TP-03 — AC-1/AC-2: second product, view cart, running total
**Preconditions:** TP-02, same thread.
**Steps:** `@assistant also add 1 Portable SSD 1TB` then `@assistant what's in my cart?`.
**Expected:** view-cart reply lists both items with current prices and a total of
`2×29.99 + 1×109.99 = 169.97`.
**Priority:** P0. **Type:** e2e/acceptance.

### TP-04 — AC-2: remove/decrease
**Preconditions:** TP-03, same thread.
**Steps:** `@assistant remove 1 Wireless Mouse Pro from my cart`.
**Expected:** reply confirms; cart now shows 1 Wireless Mouse Pro + 1 Portable SSD 1TB, total
`29.99 + 109.99 = 139.98`.
**Priority:** P0. **Type:** e2e/acceptance.

### TP-05 — AC-2: clear the cart
**Preconditions:** TP-04, same thread.
**Steps:** `@assistant clear my cart`.
**Expected:** reply confirms; a follow-up `@assistant what's in my cart?` reports empty,
total 0.
**Priority:** P0. **Type:** e2e/acceptance.

### TP-06 — AC-3: cross-conversation persistence
**Preconditions:** TP-01. A **second, independent** thread (different `threadId`, same
workspace, same hardcoded actor `u1` per M1's auth seam).
**Steps:** in the first thread, add 1 Portable SSD 1TB (fresh short conversation, not reusing
TP-02..TP-05's now-cleared cart — start this item's own thread instead so the precondition is
unambiguous); in a **new**, second thread, ask `@assistant what's in my cart?`.
**Expected:** the second thread's reply shows the same item added from the first thread — proving
the cart is anchored to the customer, not the thread.
**Priority:** P0. **Type:** e2e/acceptance.

### TP-07 — AC-4: deterministic total, hand-calculated cross-check
**Preconditions:** none (can reuse TP-03's known cart state, or a fresh one).
**Steps:** compute `2×29.99 + 1×109.99` by hand (`169.97`); compare against the live view-cart
total from TP-03; separately, call `pricing.compute_line_total` twice with the identical input
list via the server's own venv (`python -c` one-liner) and assert byte-identical output.
**Expected:** live total matches the hand calculation exactly; repeated calls are identical.
**Priority:** P0. **Type:** functional.

### TP-08 — AC-5/AC-6: place an order, frozen snapshot survives a later price change
**Preconditions:** a fresh thread with a known cart (e.g. 1 Wireless Mouse Pro + 1 Portable SSD
1TB).
**Steps:** `@assistant place my order`; capture the confirmed order total/lines from the reply and
from ground-truth Cypher (`get_order`); then **mutate** `Product.price` for Wireless Mouse Pro
directly in `reference` (e.g. `29.99 → 39.99`, reverted after this item); re-read the same order
via `get_order`.
**Expected:** order confirms 2 lines, correct snapshotted prices, total `29.99 + 109.99 = 139.98`;
cart is empty afterward; after the price mutation, the order's `OrderLine.unitPrice`/`lineTotal`
for the mouse are **unchanged** at the original $29.99 — the snapshot does not re-derive from the
now-changed catalog price.
**Priority:** P0. **Type:** e2e/acceptance.

### TP-09 — AC-9: no LLM call solely for the computation
**Preconditions:** none (structural + outcome-based, per §3's disclosed substitution).
**Steps:** (a) re-confirm by direct code read that `pricing.compute_line_total` and
`services._priced_cart_lines`/`place_order` never touch `self._models`/an LLM client anywhere in
the call chain (already done in §2, re-checked here); (b) observe that TP-02..TP-08's totals are
exact and repeatable across independently-issued turns (no drift, no approximation) — the
outcome-level evidence the plan's own AC-9 test-strategy row accepts when `trace` is off.
**Expected:** code read confirms zero LLM-reachable calls in the arithmetic path; every total
observed live matches its hand-calculated value exactly, every time.
**Priority:** P1. **Type:** integration. **Known gap:** literal trace-diff evidence is
unavailable on the `@mention` path (same as K-052's own AC-9-adjacent gap) — recorded, not
silently substituted.

### TP-10 — AC-7: order-fulfillment lifecycle, happy path
**Preconditions:** TP-08's placed order (a real, live-placed `Order`, not a synthetic fixture).
**Steps:** `POST /workflow-runs {"defKey":"order-fulfillment","version":"v1","ctx":
{"orderId":"<TP-08's orderId>"}}` (no chat trigger, per plan §3.4); confirm the run parks at
`placed`; via a Python one-liner against the server's own venv, construct a live `Repository`/
`Services` pointed at `ws:qa-cart-totals` and call `POST /workflow-runs/{id}/input
{"action":"fulfill"}` **followed by** `services.advance_order(ctx, order_id=..., transition=
"fulfill")` (the "two-step, accepted" pairing plan §3.4 describes, driven manually since no REST
route exists for the `Order`-side half); repeat with `"deliver"`.
**Expected:** each `submit_workflow_input` call succeeds (run advances `placed→fulfilled→
delivered`, confirmed via `GET /workflow-runs/{id}`); each paired `advance_order` call succeeds
and `get_order_status` reflects `fulfilled` then `delivered` in turn; the order never advances
between explicit calls (no auto-progression).
**Priority:** P0. **Type:** e2e/acceptance.

### TP-11 — AC-8: cancellation only possible before fulfillment
**Preconditions:** a second, fresh order (place one more live via chat, mirroring TP-08).
**Steps:** start its own `order-fulfillment@v1` run; call `advance_order(transition="cancel")`
while status is still `placed` — expect success; then attempt `advance_order(transition=
"deliver")` on the now-cancelled order — expect the guarded-CAS zero-rows no-op (`None`
return), status still `cancelled`. Separately, on TP-10's already-`fulfilled`→`delivered` order,
attempt `advance_order(transition="cancel")` — expect the same zero-rows no-op, confirming
"cannot cancel once fulfilled."
**Expected:** cancel-before-fulfillment succeeds; a `deliver` after `cancel`, and a `cancel` after
`fulfilled`, both no-op (status unchanged) rather than silently succeeding.
**Priority:** P0. **Type:** e2e/acceptance.

### TP-12 — AC-10: seed/verify idempotence
**Preconditions:** TP-01's first seed already ran.
**Steps:** re-run `./scripts/seed_salesperson.sh qa-cart-totals`; re-run
`./scripts/verify_salesperson.sh qa-cart-totals`.
**Expected:** re-run reports "already present — no-op" for both `reference` defs and the
`ws:qa-cart-totals` snapshots; verify still reports `OK`/in-sync, same topology as TP-01.
**Priority:** P0. **Type:** acceptance.

### TP-13 — Graph-level ground truth cross-check
**Preconditions:** TP-02..TP-08 executed.
**Steps:** direct read-only Cypher against `ws:qa-cart-totals` (`Customer`/`Cart`/`CartItem`/
`Order`/`OrderLine` reads per `docs/QUERIES.md` §16) after each mutating turn.
**Expected:** graph state matches the customer-visible reply at every step — defense against a
reply-text bug masking (or hiding) a write-path problem, same technique K-052's TP-13 used.
**Priority:** P1. **Type:** integration.

### TP-14 — Exploratory: partial vanished-catalog-product checkout
**Preconditions:** a fresh thread with 2 items in the cart.
**Steps:** delete one of the two products directly from `reference` (a disposable, seed-restored
product — not one of the two canonical fixtures above, to avoid disturbing `seed_catalog.sh`'s
idempotent slug precondition for any other concurrent work); place the order via chat.
**Expected (per the implementation review's own documented, accepted gap):** the order is created
with only the surviving line; the vanished line's `CartItem` is also removed from the cart (not
left stranded); no error, but also no explicit signal in the tool's JSON response that a line was
silently dropped — confirming the review's description, not hunting for a new defect.
**Priority:** P2. **Type:** exploratory. **Not a blocker either way** — explicitly out of AC scope
per graph note §8 and the review's own disposition; run to confirm current behavior matches what
was reported, not to re-litigate the fix.

### TP-15 — Final regression confirmation (separate final step)
**Preconditions:** all live items (TP-01..TP-14) complete and their findings recorded.
**Steps:** run the destructive offline suites last: `server/.venv/bin/python -m pytest -q` and
`./scripts/test_queries.sh` — both `GRAPH.DELETE` the shared `reference` graph at teardown. Then
restore shared state: `bootstrap_schema.sh` → `seed_demo.sh` → `seed_catalog.sh` →
`seed_workflows.sh` → `seed_salesperson.sh` for `acme` (and any other workspace this pass is
expected to leave intact), then re-verify (`verify_workflows.sh`/`verify_catalog.sh`/
`verify_salesperson.sh acme`) all report `OK` before handing back.
**Expected:** offline suite green (no regression from K-053's own code, independently confirming
`analyst`'s and `teco`'s prior same-commit runs); `test_queries.sh` green; `ws:acme` restored and
verified in sync afterward.
**Priority:** P1. **Type:** regression.

## 6. Entry / exit criteria

**Entry:** TP-01 passes.

**Exit / verdict rule:**
- **Pass** — TP-02..TP-13 (all P0/P1) pass on first attempt or after at most one reproducibility
  rerun (to distinguish a reproducible defect from nondeterministic live-model noise, same rule
  K-052's plan used), TP-15 confirms no regression. Verdict: "K-053 meets its acceptance criteria
  against the live running system."
- **Pass with defects** — one or more non-blocking issues found that don't invalidate the core
  ACs — reported as defects with severity, verdict still ships.
- **Fail** — any of AC-1..AC-10 reproducibly fails to hold against the live system (wrong/
  fabricated cart or order state, a snapshot that drifts with catalog price changes, an order
  advancing without an explicit step, or seed/verify non-idempotence).

## 7. Out of scope

- Re-deriving unit/integration-level correctness already covered by `test_{pricing,repository,
  services,tools,order_fulfillment}.py` and independently re-verified by `analyst` (U21) — see
  §3.
- Re-running the destructive offline `pytest`/`test_queries.sh` suites as part of the live pass —
  deferred to TP-15, a clearly separate, final step, per the coordinator's brief and the same
  testing-hazard discipline `falkor-chat/AGENTS.md` documents.
- Literal tool-call-argument tracing for AC-9 — `trace` is off by default on the `@mention` path;
  see §3's disclosed substitution.
- Multi-tenant cart isolation (a *second, distinct* customer's cart never leaking into the
  first's) — not independently demonstrable until real per-user auth (K-016) lands, per plan §6's
  own caveat; this pass only proves the same-customer case AC-3 actually asks for.
- Re-litigating the implementation review's already-disposed findings (the partial-vanish MINOR,
  the commit-message typo, the `quantity: 0` nit) beyond TP-14's confirmatory exploratory check.
- K-054/K-055 (the two remaining sibling capabilities) — not yet implemented, not in scope.
- Browser/UI automation — driven via direct REST calls + a Services-layer script for the
  no-REST-route half, same substitution K-052's plan used.
