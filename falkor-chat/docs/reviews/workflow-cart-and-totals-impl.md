# Cart, orders, and deterministic totals — Implementation Review

> **Status:** archived · **Owner:** `analyst` · **Tracks:** K-053 (M6) · **Extends:** `docs/plans/workflow-cart-and-totals.md`, `docs/plans/workflow-cart-and-totals-graph.md`, `docs/reviews/workflow-cart-and-totals.md`

## Scope & verdict

Diff-scoped review of the three committed K-053 clusters (all on `main`):

- `f020f90` — `server/falkorchat/pricing.py` (new), `repository.py` Cart/Order methods,
  `bootstrap_schema.sh` DDL.
- `bcd2dcc` — `services.py` cart/order methods, `tools.py` (5 new tools), `repository.py`
  additions (`lookup_products_by_id`, `lookup_product`'s `productId`).
- `4b4d807` — `proof_defs.py` (`SALESPERSON_DEF` v1→v2, new `ORDER_FULFILLMENT_DEF`),
  `seed_salesperson.sh`/`verify_salesperson.sh` (two-def), `docs/QUERIES.md` §16,
  `scripts/test_queries.sh` (+41), `AGENTS.md`, new/changed tests.

Reviewed against the two gated design artifacts (`docs/plans/workflow-cart-and-totals.md` v2,
`docs/plans/workflow-cart-and-totals-graph.md` v2) and their combined plan-gate review
(`docs/reviews/workflow-cart-and-totals.md`, verdict: approve). Read every changed production
file in full against the diff (not just the commit messages), read all new/changed tests against
the implementation they exercise, and independently re-ran the two **read-only** live checks the
brief pointed at: `./scripts/verify_salesperson.sh` (both `salesperson@v2` and
`order-fulfillment@v1` report `in sync`/topology-OK against the live `ws:acme` — confirms the
commit message's "Live seed/verify confirmed against ws:acme" claim directly, not just by
narration) and `redis-cli GRAPH.QUERY ws:acme "CALL db.constraints()"` (confirms the
`Customer`/`Cart`/`CartItem`/`Order` constraints from `bootstrap_schema.sh`'s new DDL are actually
`OPERATIONAL` on the live instance). **Did not** run the offline `pytest` suite or
`scripts/test_queries.sh` — both wipe the shared `reference` graph (`server/tests/conftest.py`'s
`wf_repo` fixture; `test_queries.sh`'s own teardown) as a side effect, which would discard the
`salesperson@v2`/`order-fulfillment@v1`/catalog state this run just confirmed live and that
`qa-engineer`'s upcoming live acceptance pass likely depends on; re-seeding afterward is a
mechanical fix but not one this review needed to make to reach its verdict. Verified test quality
and the mutation-testing claims by close reading (call-ordering assertions, node-count probes,
`GRAPH.PROFILE` index-scan assertions, idempotent-replay checks) rather than an independent
mutation run, per the brief.

**Verdict: approve with suggestions.**

**CPG:** considered, not relevant — `cpg_falkorchat` is confirmed stale (built
`2026-08-26T22:27:22Z`, predates all three of K-053's own commits, per the brief) and this is a
diff-scoped static review of new/changed code; every file discussed below was read directly, not
inferred from graph structure.

## Findings

### MAJOR (plan-gate) — verified closed

The plan-gate review's one MAJOR — `ensure_customer`/`ensure_cart` ownership never assigned to a
service method, risking a silent no-op on a brand-new customer's first `add_to_cart` — is closed
exactly as the plan's revision and the cluster-2 commit message claim.
**Evidence:** `services.py`'s `add_cart_item` calls `ensure_customer` → `ensure_cart` →
`add_to_cart` in that literal order (`server/falkorchat/services.py`, the `add_cart_item` method
added in `bcd2dcc`); `place_order` calls `ensure_customer` defensively before the cart read.
**Verified, not just read:** `test_add_cart_item_brand_new_customer_ensures_customer_and_cart_first`
(`server/tests/test_services.py`) asserts the call order directly (`kinds.index("ensure_customer")
< kinds.index("ensure_cart") < kinds.index("add_to_cart")`) against a `FakeRepo` that returns
`None`/no-writes exactly as the real repository does when the anchors are missing — a mutation
that reordered or dropped either call would fail this test, not just look wrong on inspection. The
MINOR (in-flight catalog-price-change race) is likewise closed as designed: `place_order` resolves
prices via one batched `lookup_products_by_id` call immediately before the snapshot write, and the
plan's §6 acceptance of either-outcome-is-fine for a mid-checkout race is unaffected by this diff.

### MINOR — a partial vanished-catalog-product checkout silently drops the line and clears the whole cart, untested

**Evidence:** `services._priced_cart_lines` (`services.py`) drops any cart line whose `productId`
no longer resolves via `lookup_products_by_id` — documented and correct per graph note §8. But
`services.place_order` passes only the *survivors* to `repository.place_order`
(`ctx.ws, ..., lines=order_lines`), and `repository.place_order`'s own Cypher
(`repository.py`, added in `f020f90`) clears **every** `CartItem` under the customer's `Cart`
unconditionally (`OPTIONAL MATCH (cart)...HAS_ITEM...FOREACH (_ IN CASE WHEN created THEN [1]
ELSE [] END | DETACH DELETE item)`) — not filtered to the lines that made it into the order. So a
cart with 2 items, one of which references a since-deleted product, places an order for the 1
surviving item and **silently discards the other from the cart entirely**, with no signal in the
tool's JSON response (`PlaceOrderTool` returns only `result["lines"]`, i.e. the survivors) that
anything was dropped.

**Why it matters:** this is the *partial*-vanish case, and it is untested — the two existing
tests (`test_get_cart_drops_a_line_whose_product_vanished_from_the_catalog`,
`test_place_order_every_line_products_vanished_is_treated_as_empty`) cover "one line, dropped, cart
still empty-treated" and "every line, dropped, whole cart empty-treated" but not "some lines
survive, checkout proceeds, and the discarded item quietly disappears from the cart without the
customer ever being told." Not an AC violation (the requirements doc and graph note §8 both
explicitly leave this case unresolved), and low-probability in this milestone's static ~15-product
catalog with no product-delete surface — but it is a real, reachable behavior with no test
asserting it's the intended one, and no observability into it for whoever hits it live.

**Suggested improvement:** add one test asserting the partial-vanish outcome explicitly (order
created with only the surviving line; the vanished line's `CartItem` is also gone afterward, not
left stranded) so the current behavior is a documented contract rather than an untested side
effect of two independently-correct pieces of code composing this way. If the silent-drop-with-
no-signal shape turns out to be wrong once `qa-engineer`'s live pass or a future capability meets
it, that's a `services.place_order`/`PlaceOrderTool` response-shape change, not a graph-schema
one.

### MINOR — cluster-3 commit message's test-delta prose doesn't match its own before/after numbers

**Evidence:** `4b4d807`'s message says "19 new/changed tests (1910 -> 1919 passed offline)". The
stated before/after delta is 9, and `git show 4b4d807 -- server/tests | grep -c '^+def test_'`
independently counts exactly 9 new test functions across the three touched test files
(`test_executor_agent.py` +1, `test_order_fulfillment.py` +7, `test_salesperson_scaffold.py` +1) —
matching the 1910→1919 delta exactly, not the "19" in the prose. Almost certainly a typo (perhaps
carried over from cluster 1/2's "41"/"42" phrasing), not a real discrepancy in what was tested; the
numeric before/after count is the one that's actually checkable and it's internally consistent.
**Suggested improvement:** none needed in code — flagging only so it doesn't get transcribed
verbatim into `HISTORY.md` when `teco` does milestone bookkeeping; the "9" is the accurate figure.

### nit — `AddToCartTool` silently upgrades an explicit `quantity: 0` to `1`

**Evidence:** `tools.py`'s `AddToCartTool.run`: `quantity=arguments.get("quantity") or 1` treats a
model-supplied `0` the same as "omitted," defaulting it to `1` rather than adding zero (or
rejecting it). The tool's own JSON schema declares `"minimum": 1`, so a compliant model should
never send `0`, but the server doesn't enforce that itself — a model that ignores the schema hint
gets a silent behavior change (add 1) rather than a no-op or an error. Low-stakes (same trust
boundary every other tool argument in this codebase already accepts from the model), worth a
one-line comment or an explicit `is None` check if it's ever revisited, not worth a change on its
own.

## What's solid

- **Faithful, byte-for-byte transcription of the graph note's `[verified]` Cypher.** Every
  repository method in `f020f90` (`ensure_customer`, `ensure_cart`, `add_to_cart`,
  `adjust_cart_item`, `read_cart`, `clear_cart`, `place_order`, `get_order`, the three lifecycle
  CAS writes) matches `workflow-cart-and-totals-graph.md` §1.3/§2.1-§2.5/§3.1-§3.4 exactly —
  parameterized throughout, no interpolation anywhere.
- **`pricing.compute_line_total` is genuinely pure and well-tested**: 14 unit tests cover empty,
  one/many lines, every malformed-input shape (missing keys, non-numeric, `None`, `bool`-as-int
  rejection), non-mutation of the input, and identical-inputs-identical-output (AC-4) — no I/O, no
  LLM reachable from it, exactly the FR-8 argument the plan makes.
- **The `ORDER_FULFILLMENT_DEF` design (a `decision`-typed step declaring
  `config.waitsForHuman: true` to park) is a subtle, correctly-reasoned mechanism, and I verified
  the two claims it rests on directly against `executor.py`/`services.py` rather than trusting the
  inline comment**: `_drive_loop`'s OUTCOME B keys on `config.get("waitsForHuman")` alone,
  independent of `step.get("type")` (`executor.py:551`); and `_validate_against_parked_step`'s
  `expects` check (`services.py:2216-2227`) runs unconditionally after the type-gated
  `fields`/`signal` check, so a `decision`-typed parked step's `expects` is enforced exactly like a
  `human` step's would be. `test_order_fulfillment.py` then proves the whole mechanism live (real
  `publish_workflow_def`/`materialize_def`/`submit_workflow_input`, no LLM), including the
  AC-8 "cannot cancel once fulfilled" guarded-CAS zero-rows path and a genuine `WorkflowInputRejectedError`
  free-typo test.
- **Tests are mutation-resistant, not just execution-covering.** Repeated pattern across all three
  clusters: assert exact node/edge counts via a raw `_probe` query after two idempotent calls (not
  just "didn't raise"), assert exact call ordering via a `FakeRepo.calls` list, assert
  `GRAPH.PROFILE` output contains `Node By Index Scan` and not a label scan
  (`test_queries.sh` §16.3/§16.10), and assert the AC-9 no-extra-LLM-call property via actual
  trace-kind counts through the real `AddToCartTool`/`_run_agent_node` path
  (`test_executor_agent.py`), not a stub that couldn't demonstrate the property either way.
- **Live-verified, not just plan-verified.** `verify_salesperson.sh`'s "Live seed/verify confirmed
  against ws:acme" claim in the cluster-3 commit message reproduces cleanly when re-run
  independently (both defs present, in sync, correct topology); the new DDL's constraints are
  confirmed `OPERATIONAL` on the same live instance.
- **Documentation obligations met**: `AGENTS.md` schema-conventions and script-table rows updated
  in the same commit as the code they describe; `docs/QUERIES.md` §16 transcribes (doesn't
  re-derive) the graph note's Cypher with correct cross-references; `HISTORY.md`/`BACKLOG.md` are
  correctly untouched, as the brief expected (milestone bookkeeping is `teco`'s job once both
  K-053 gates close).

## Open questions

- **`services.advance_order` has no REST route in this milestone** (explicitly stated in the
  plan §3.4 and the graph note §4, and confirmed in code — no `app.py`/`api.py` caller exists).
  `qa-engineer`'s live acceptance pass for FR-6/FR-7/AC-7/AC-8 will need to drive the
  `Order.status` half of the fulfillment lifecycle through the Services layer directly (as
  `test_order_fulfillment.py` does), not through an HTTP call a live operator could actually make
  — worth confirming with `teco`/`qa-engineer` that this is understood going into that pass, since
  it's a real gap between "provably correct" and "operable" for the fulfillment half of this
  capability, not a defect in this diff.
