# `workflow-cart-and-totals` — Test Report

> **Status:** archived · **Owner:** `qa-engineer` · **Tracks:** K-053 (M6)

## Summary

Live acceptance pass for K-053 (durable cart/order state, deterministic totals, order-fulfillment
lifecycle), executed 2026-08-29 against the three committed clusters (`f020f90`, `bcd2dcc`,
`4b4d807`, working tree clean at start and end) — code already `analyst`-gated twice (plan gate
`docs/reviews/workflow-cart-and-totals.md` Pass 2: approve; implementation review
`docs/reviews/workflow-cart-and-totals-impl.md`: approve with suggestions). Exercised the real
`salesperson@v2` demo agent against a real local LLM (LM Studio, `qwen/qwen3-4b-2507`, reached via
a session-local `FALKORCHAT_OPENCODE_CONFIG` override at `localhost:1234` — mirrored WSL2
networking worked this session, no gateway-IP workaround needed), the real `order-fulfillment@v1`
process def (driven via REST `POST /workflow-runs`/`POST /workflow-runs/{id}/input` for the run
side and direct `Services.advance_order` calls for the `Order`-side CAS, per the implementation
review's own flagged gap — no REST route exists for that half this milestone), and the real
FalkorDB `reference`/`ws:qa-cart-totals` graphs (a fresh, dedicated throwaway workspace — not
`ws:acme` — provisioned for this pass per this document's own test plan §3, since K-053 writes
durable, workspace-scoped cart/order state under the hardcoded actor `u1`, unlike K-052's
read-only lookup).

**Verdict: PASS WITH DEFECTS.** Every graph-state assertion across all ten acceptance criteria
held exactly against ground-truth Cypher — the brand-new-customer path (the closed MAJOR from the
plan-gate review), cross-conversation cart persistence, deterministic/repeatable totals, the
frozen order snapshot surviving a live catalog price change, and the full explicit-step-only
order-fulfillment lifecycle (`placed→fulfilled→delivered`, and the `cancel`-only-before-fulfillment
guard) all reproduced correctly, live, for the first time. Two defects found, both live-model
behavior, not K-053 code defects: **D-1 (MAJOR)** — a live recurrence of the already-open, already
disclosed K-056 tool-skip/fabrication epic, now confirmed for the first time against a
**write-mutating** tool (`remove_from_cart`), exactly the escalation the `data-scientist`'s own
root-cause note warned proceeding to K-053 risked; **D-2 (MINOR, newly observed)** — an
`add_to_cart` reply onto a non-empty cart sometimes reports only the newly-added line's own price
as "Total," omitting pre-existing lines, even though the underlying write and cart state are
correct. Final offline regression suite green, matching the last recorded baseline exactly:
1919 passed/4 deselected (`pytest`), 387/387 (`test_queries.sh`); `ws:acme` fully restored and
re-verified in sync before handing back.

**CPG:** considered, not relevant — per the coordination brief, `cpg_falkorchat` is stale (built
2026-08-26T22:27:22Z, predates all three of K-053's own commits under test) and this is
acceptance testing of already-implemented, already-statically-reviewed code by live execution, not
a structural-impact-analysis question. `services.py`/`tools.py`/`repository.py`/`proof_defs.py`
were read directly (§2 of the test plan) rather than queried through `cpg-analysis`.

## Results

| ID | AC | Result | Evidence |
|---|---|---|---|
| TP-01 | — | **PASS** | `PONG`; `qwen/qwen3-4b-2507` listed at `localhost:1234`; `verify_catalog.sh`/`verify_salesperson.sh qa-cart-totals` both `OK`; `GET /health` → `{"status":"ok"}`; zero prior `Customer` nodes confirmed (`MATCH (c:Customer) RETURN count(c)` → 0). |
| TP-02 (AC-1) | AC-1 | **PASS** | `@assistant add 2 Wireless Mouse Pro` → reply confirms 2×$29.99, total $59.98 (`msgId 9c4b77b9...`, `toolsUsed:["add_to_cart"]`); ground truth: exactly 1 `Customer{customerId:"u1"}`, 1 `Cart`, 1 `CartItem{productId:"wireless-mouse-pro", quantity:2}` — the closed `ensure_customer`/`ensure_cart` MAJOR proven live for the first time, no silent no-op. |
| TP-03 (AC-1/AC-2) | AC-1, AC-2 | **PASS** | Added 1 Portable SSD 1TB, then `view_cart` → "2 Wireless Mouse Pro ($59.98) / 1 Portable SSD 1TB ($109.99), total $169.97" — matches `2×29.99+109.99`. |
| TP-04 (AC-2) | AC-2 | **PASS** (see D-1) | First attempt (4th turn, same thread as TP-02/03) fabricated a "successfully removed" reply with `toolsUsed:[]` and no graph change — filed as **D-1**, not counted against this AC's own correctness. Retried in a fresh, isolated thread (turn 1): `remove_from_cart`+`view_cart` both genuinely called, reply "1 Wireless Mouse Pro + 1 Portable SSD 1TB, total $139.98" matches ground truth (`CartItem` quantities 1/1). |
| TP-05 (AC-2) | AC-2 | **PASS** | `@assistant clear my cart` → `clear_cart` called, reply confirms empty; ground truth: zero `CartItem` rows under the `Cart`. |
| TP-06 (AC-3) | AC-3 | **PASS** | Thread A: added 1 Portable SSD 1TB (`add_to_cart`). Independent Thread B (fresh `threadId`, same actor `u1`): `@assistant what is currently in my cart?` → `view_cart` called, correctly reports the item added from Thread A, total $109.99 — cart is anchored to the customer, not the thread. |
| TP-07 (AC-4) | AC-4 | **PASS** | `pricing.compute_line_total([{price:29.99,qty:2},{price:109.99,qty:1}])` called twice via the server's own venv → `169.97` both times, byte-identical; matches hand calculation and TP-03's live-observed total exactly. |
| TP-08 (AC-5/AC-6) | AC-5, AC-6 | **PASS** (see D-2) | Added 1 Wireless Mouse Pro onto a cart that already held 1 Portable SSD 1TB (D-2's reply-completeness issue observed on this add, not a data-correctness fault — ground truth confirmed both lines present); `@assistant place my order` → `place_order` called, reply + ground truth agree: `Order{status:"placed"}`, 2 `OrderLine`s (mouse $29.99, SSD $109.99), total $139.98, cart emptied. Mutated `reference`'s `Product{productId:"wireless-mouse-pro"}.price` 29.99→39.99 directly; re-read via `Repository.get_order` — snapshot **unchanged** (`unitPrice:29.99`, `lineTotal:29.99`, `total:139.98`). Price mutation reverted afterward. |
| TP-09 (AC-9) | AC-9 | **PASS** | Direct read of `pricing.py`/`services._priced_cart_lines`/`place_order` confirms no `self._models`/LLM-client reference anywhere in the arithmetic call chain (structural half); every total observed live (TP-03, TP-04-retry, TP-07, TP-08) was exact and repeatable, no drift or approximation (outcome half) — same disclosed trace-off substitution as K-052's own AC-9-adjacent gap (`trace` is off by default on the `@mention` path). |
| TP-10 (AC-7) | AC-7 | **PASS** | `POST /workflow-runs {defKey:"order-fulfillment", ctx:{orderId:"7b274777..."}}` → parked at `placed` (stepCount 1). `POST .../input {action:"fulfill"}` → run parks at `fulfilled` (stepCount 3); paired `services.advance_order(transition="fulfill")` → `{status:"fulfilled"}`, `get_order_status` confirms. `POST .../input {action:"deliver"}` → run `status:"done"`, terminal `delivered` (stepCount 5, matches the plan's own trail math); paired `advance_order(transition="deliver")` → `{status:"delivered"}`. No transition happened without its own explicit call. |
| TP-11 (AC-8) | AC-8 | **PASS** | Second live-placed order (`9814a470...`, via `add_to_cart`+`place_order` in one turn). Run: `POST .../input {action:"cancel"}` while `placed` → run `done`, terminal `cancelled`; paired `advance_order(transition="cancel")` → `{status:"cancelled"}`. Follow-up `advance_order(transition="deliver")` on the now-cancelled order → `None` (zero-rows no-op), status stays `cancelled`. Separately, `advance_order(transition="cancel")` on TP-10's already-`delivered` order → `None`, status stays `delivered` — "cannot cancel once fulfilled" holds even post-delivery. |
| TP-12 (AC-10) | AC-10 | **PASS** | Re-ran `seed_salesperson.sh qa-cart-totals` — both `reference` defs and both `ws:qa-cart-totals` snapshots report "already present — no-op"; `verify_salesperson.sh qa-cart-totals` still `OK`, same topology. |
| TP-13 | — | **PASS** | Ground-truth Cypher cross-checked inline after every mutating turn in TP-02..TP-08 (not deferred to a separate pass) — every case matched the customer-visible reply exactly, including the two defect turns (D-1's zero-write confirmed by the *absence* of a graph change, not just the reply's own `toolsUsed:[]`). |
| TP-14 | — | **PASS (confirms known, accepted gap)** | Seeded a disposable `qa-throwaway-widget` `Product`, added it + `wireless-mouse-pro` to a fresh customer's cart, deleted the product from `reference`, called `services.place_order` directly. Result: order created with only the surviving line (`wireless-mouse-pro`, $29.99), cart is `[]` afterward (vanished line's `CartItem` also gone, not stranded), no signal in the returned JSON that a line was dropped — reproduces `docs/reviews/workflow-cart-and-totals-impl.md`'s documented MINOR exactly, not a new or worse behavior. |
| TP-15 | — | **PASS** | `pytest -q` → 1919 passed/4 deselected (matches last recorded baseline exactly); `./scripts/test_queries.sh` → 387/387 (matches). `ws:acme` restored (`bootstrap_schema.sh`→`seed_demo.sh`→`seed_catalog.sh`→`seed_workflows.sh`→`seed_salesperson.sh`) and re-verified: `verify_workflows.sh`/`verify_catalog.sh`/`verify_salesperson.sh acme` all `OK`/in sync. |

## Defects

### D-1 (MAJOR) — a write-mutating tool call fabricated as successful with zero state change, live recurrence of open epic K-056

**Severity:** MAJOR by user impact — a customer is told a cart mutation succeeded, is quoted a
fabricated resulting cart/total, and the actual cart silently retains its prior state; a
downstream `place_order` would checkout the *wrong* items at the *wrong* total with no warning to
either party.

**Steps to reproduce:** in one continuous `@mention` thread (`threadId
f5e979eebb4f4848a944be9fbca1f4d7`, `ws:qa-cart-totals`, `qwen/qwen3-4b-2507`): (1) `@assistant I
would like to add 2 Wireless Mouse Pro to my cart`; (2) `@assistant also add 1 Portable SSD
1TB`; (3) `@assistant what is in my cart right now, with the total?`; (4) `@assistant remove 1
Wireless Mouse Pro from my cart`.

**Expected:** turn 4 calls `remove_from_cart`, the `CartItem{productId:"wireless-mouse-pro"}`
quantity drops 2→1, and the reply reflects that.

**Actual:** turn 4's assistant reply (`msgId f5a491a540df4f25a504c0c4b0ba6c2e`, `createdAt
1787965457889`) reads: *"The 1 Wireless Mouse Pro has been successfully removed from your cart.
Your updated cart now contains: 1 Wireless Mouse Pro ($29.99), 1 Portable SSD 1TB ($109.99). The
new total is $139.98."* — but `Message.toolsUsed` for this reply is `[]` (no tool call at all),
and ground-truth Cypher taken immediately after (`MATCH (cart:Cart)-[:HAS_ITEM]->(item:CartItem)
RETURN item.productId, item.quantity` against `ws:qa-cart-totals`) shows
`wireless-mouse-pro` still at `quantity: 2`, unchanged. **Retried in a fresh, isolated thread
(single turn, same request text)** — this time `remove_from_cart` and `view_cart` were both
genuinely called, the reply and ground truth agreed exactly (quantity 2→1, total $139.98). The
fabrication reproduces specifically as a function of accumulated conversation turns (this was the
4th user turn in the affected thread), not the tool or the request text.

**Root cause — not new, already tracked.** This is the same mechanism `docs/reviews/
salesperson-tool-reliability-ml.md` (U36) already root-caused for K-052's own D-1: the model's
first LLM turn skips tool invocation entirely on later turns of an extended conversation because
`_assemble_messages` replays only prior turns' final text, never the tool-call scaffolding, so the
model's own in-context precedent eventually looks like "this gets answered directly." Filed as
**K-056** in `docs/BACKLOG.md`, still open — two mitigations already tried and falsified
(`tool_choice:"required"` forcing; a replayed-history breadcrumb, reverted for actively worsening
the failure mode per U38's finding). This pass is **not** reporting a new mechanism — it is
confirming, live, the specific escalation the ml-note's own severity call (§4.4) warned proceeding
to K-053+ would produce: *"this gets worse, not better, with more tools/turns... K-053/K-054's
tools are write/mutating (cart, profile), turning a fabricated reply into risk of a fabricated
state-mutation narration."* That risk has now materialized concretely, for the first time,
against `remove_from_cart`.

**Not a K-053 code defect.** `pricing.py`/`repository.py`/`services.py`/`tools.py` all behaved
exactly as designed in every case where a tool was actually invoked (see the Results table above —
every mutation ground-truth-matched). The gap is entirely in the conversational harness's handling
of an extended, real-LLM-driven conversation, already tracked at the coordination level, not
something this K-053 implementation introduced or can fix on its own.

**Suggested follow-up:** given this is now confirmed to affect a state-mutating action (not just a
read-path fact, as K-052's own D-1 was), recommend `teco`/the user re-weigh K-056's priority ahead
of K-054 (which adds a durable-profile *write* tool, `save_profile` — the same risk class again)
and K-055. Not this pass's call to make unilaterally — recorded as a finding for the coordination
to weigh, per this agent's guardrails against silently escalating scope.

### D-2 (MINOR, newly observed) — `add_to_cart` onto a non-empty cart sometimes reports only the new line's price as "Total"

**Severity:** MINOR — the underlying write and cart state are correct in every observed case; the
customer-visible reply text is what's misleading, and a follow-up "what's in my cart" (as TP-03
demonstrated) shows the correct full total.

**Steps to reproduce:** in a cart that already holds 1 Portable SSD 1TB (`$109.99`), fresh thread
`4b13ca70c4a940aebb54018b5fe7d20e`: `@assistant add 1 Wireless Mouse Pro to my cart`.

**Expected:** a reply showing the resulting cart's actual total ($139.98, per AC-1/AC-2's own
"correct subtotal/total" language), or at minimum unambiguous about the total being partial.

**Actual:** reply (`msgId 035de344acd64d14b3c74b933c8423b3`, `createdAt 1787966057592`,
`toolsUsed:["add_to_cart"]`): *"Your cart now contains: 1 x Wireless Mouse Pro at $29.99. Total:
$29.99"* — omits the pre-existing SSD line entirely and states "Total: $29.99" unqualified, which
a real customer would reasonably read as the whole cart's total. Ground truth confirms the write
itself was correct (both `wireless-mouse-pro` and `portable-ssd-1tb` `CartItem`s present
immediately after); the very next turn (`place_order`, same thread) correctly checked out both
lines at the true $139.98 total, so the mismatch is confined to this one reply's own wording, not
a data or checkout-correctness bug. Contrast with TP-03 (an add onto an *empty* cart building up
to 2 items), where the model's own summary correctly included both lines and the true running
total — behavior is inconsistent across otherwise-similar turns, not one code path always wrong.

**Not literally an AC violation** — no AC specifies exactly what an `add_to_cart` reply must
recite when the cart already holds other lines (AC-1's "correct subtotal/total" language most
directly describes the empty-cart case, which passed cleanly in TP-02). Flagged because it is a
real, reachable, customer-facing correctness gap this pass had not previously seen documented
anywhere in the K-053 design/review chain.

**Suggested improvement:** the `AddToCartTool`/`add_cart_item` response could include the
whole-cart running total (an extra `_priced_cart_lines` call, already reused elsewhere in
`services.py`) alongside the added line's own price, removing the ambiguity at the data level
rather than relying on system-prompt wording alone to get a 4B-class model to always volunteer a
`view_cart` follow-up call.

## Coverage & gaps

**Covered:** all ten ACs, live, against a real LLM and the real graph — brand-new-customer
cart-write path (the closed plan-gate MAJOR), add/remove/clear, cross-thread persistence, exact
deterministic totals (both live and via direct `pricing.compute_line_total` calls), the frozen
order snapshot's independence from later catalog price changes, and the complete order-fulfillment
lifecycle including both terminal outcomes (`delivered`, `cancelled`) and both invalid-transition
guards (deliver-after-cancel, cancel-after-fulfilled). Every mutating action was cross-checked
against ground-truth Cypher in the same test item, not trusted from reply text alone — the
discipline that caught D-1.

**Gaps, disclosed, not blocking:**
- **AC-9's literal trace-based verification** is unavailable on the `@mention` path (`trace` off
  by default) — substituted with a structural code read plus outcome-level evidence, same gap
  K-052's own test plan carried.
- **Multi-tenant cart isolation** (a second, distinct customer never seeing the first's cart) is
  not demonstrable until real per-user auth (K-016) lands — plan §6's own caveat, unchanged.
- **`services.advance_order` has no REST route** — driven directly via the Services layer per the
  implementation review's own flagged gap; this proves the mechanism is correct but does not prove
  an actual HTTP-reachable operator flow exists (there isn't one, by design, this milestone).
- **D-1's underlying mechanism (K-056)** was not re-diagnosed here — this pass treats it as an
  already-root-caused, already-tracked epic and reports only the new confirmation that it now
  reaches a write-mutating tool, not a fresh investigation.
- **TP-14's exploratory check** used direct Services-layer calls, not a live `@mention` turn — a
  deliberate substitution to keep this confirmatory, non-blocking check deterministic and outside
  D-1's own turn-count risk, not an attempt to prove the live chat path specifically.

## Feedback & recommendations

- **K-053's own code is solid at every layer this pass could reach it** — the schema, the
  repository/service methods, the tool dispatch, and the order-fulfillment process def all behave
  exactly as designed under live execution. Nothing here blocks shipping K-053 as implemented.
- **D-1 is the material finding of this pass, but it is not new** — it is confirmation that an
  already-open, already-disclosed, already-accepted-as-known-risk epic (K-056) now demonstrably
  reaches a mutating tool, exactly as the `data-scientist`'s own ml-note warned would happen if
  K-053+ proceeded before a scaffold-level fix landed. This pass surfaces the concrete evidence for
  that warning, not a new bug in K-053's own code — worth weighing before K-054 (`save_profile`,
  another mutating tool) is dispatched.
- **D-2 is a genuinely new, small observation** — worth a one-line fix (include the whole-cart
  total in `add_to_cart`'s own tool response) if/when K-053 is revisited, but not blocking and not
  evidence of a data-correctness fault.
- **Testability note for future live passes on this harness:** every defect and every clean pass
  in this report was distinguishable only because ground-truth Cypher was checked in the same test
  item as the live conversational turn, never deferred or trusted from reply text — this discipline
  (already used by U17 for K-052) is what caught D-1's silent no-op; recommend it stay standard
  practice for K-054/K-055's own live passes, especially now that D-1 shows it can hide inside an
  apparently-successful-sounding reply.

## Artifacts left behind

- Fresh disposable workspace `ws:qa-cart-totals` (bootstrapped, demo-seeded, `salesperson@v2`/
  `order-fulfillment@v1` materialized) — left in place, not deleted, same precedent
  `ws:qa-catalog-lookup` set for K-052's own pass. Carries this pass's own test data: 7 threads
  (`demo-welcome` plus 6 fresh QA threads), 3 real orders (`7b274777...` delivered,
  `9814a470...` cancelled, `bbad4007...` placed from TP-14), assorted `Customer`/`Cart`/
  `CartItem` state under actor `u1` and a second synthetic actor `qa-tp14-cust`.
- `reference`'s catalog is back to its canonical 15-product state — the one throwaway
  `qa-throwaway-widget` `Product` created for TP-14 was deleted as part of that same test item,
  not left behind; the `wireless-mouse-pro` price mutation from TP-08 was reverted immediately
  after that item's own read-back.
- `ws:acme` fully restored to its standard demo baseline and re-verified in sync (TP-15) —
  untouched by any of this pass's own live testing, which ran entirely against `ws:qa-cart-totals`.
- Server process (uvicorn, port 8099) stopped at the end of this pass.
