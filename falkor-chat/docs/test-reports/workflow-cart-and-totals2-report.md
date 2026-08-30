# `workflow-cart-and-totals` — Test Report (Ministral re-verification)

> **Status:** archived · **Owner:** `qa-engineer` · **Tracks:** K-053, K-056 (M6)

## Summary

Regression re-verification of K-053's full acceptance-test suite, executed 2026-08-29 against the
real running system — FalkorDB (`falkordb-dev`), the M1 server on `http://localhost:8011`, LM
Studio on `http://localhost:1234` now serving `mistralai/ministral-3-3b` (`salesperson@v2.1`, K-056
commit `03a3c8c`) — per `docs/test-plans/workflow-cart-and-totals2.md`. Workspace: a fresh
throwaway `ws:qa-cart-totals2`, provisioned and verified in sync before any test item ran. LM
Studio reached the same corrected-`baseURL` way as the catalog-lookup2 pass and every prior unit in
this coordination.

**Verdict: PASS WITH DEFECTS.** All ten acceptance criteria (AC-1..AC-10) hold against the live
system on `mistralai/ministral-3-3b`. **This pass's central question — does the qwen-era D-1
write-mutating fabrication (`docs/test-reports/workflow-cart-and-totals-report.md`, zero tool call,
zero graph change, `remove_from_cart`) reproduce on Ministral — is answered no.** The exact 4-turn
conversation the original D-1 reproduced on (add 2 mouse → add 1 SSD → view cart → remove 1 mouse)
was replayed verbatim at TP-04: `remove_from_cart` was genuinely dispatched
(`Message.toolsUsed:["remove_from_cart"]`), and ground-truth Cypher taken immediately after showed
the quantity actually dropped 2→1 — a real write, not a fabricated confirmation. Every one of the
~15 real cart/order-mutating live turns across this entire pass showed the same pattern: a genuine
tool dispatch, confirmed against ground truth, never a hollow "success" claim over an unchanged
graph.

**Two non-blocking defects found, both distinct from the qwen-era D-1 mechanism:**
- **D-1 (MINOR, new)** — a cart-mutation reply's own summary text intermittently omits a
  still-present line and understates the total, even though the underlying write is always
  correct — observed on both `remove_from_cart` (TP-04's main-thread turn) and `add_to_cart`
  (TP-08's order-building turn), confirming the original report's own D-2 finding is
  model-independent, exactly as the brief predicted, and extending it to a second tool.
- **D-2 (MINOR, informational, per the brief's own instruction not a gate)** — Ministral's own
  known duplicate-instruction defect (`docs/reviews/salesperson-tool-reliability-ml.md` §8.4/§9)
  reproduced live, once, in TP-16's dedicated same-category-immediate probe: an uninstructed
  second `add_to_cart(Wireless Mouse Pro)` fired when the customer's own turn-2 message named only
  "Mechanical Keyboard K200." Honestly-grounded (real write, real reply reflecting the resulting —
  wrong — state) and self-disclosing (the doubled quantity is stated plainly in the reply), exactly
  as the ml-note characterizes it. Recorded per the brief, does not affect the verdict.

Final offline regression suite green, matching the last recorded baseline exactly: 1920 passed/4
deselected (`pytest`), 387/387 (`test_queries.sh`); `ws:acme` fully restored and re-verified in sync
before handing back (shared final step for both this re-run and `workflow-catalog-lookup2.md`'s).

**CPG:** considered, not relevant — live acceptance re-run against a changed model dependency; no
code under test changed shape. `services.py`/`tools.py`/`repository.py`/`pricing.py` were read
directly where relevant.

## Results

| ID | AC | Result | Evidence |
|---|---|---|---|
| TP-01 | — | PASS | `PONG`; `mistralai/ministral-3-3b` listed at `localhost:1234` (confirmed fresh); `verify_catalog.sh`/`verify_salesperson.sh qa-cart-totals2` both `OK`; `GET /health` → `{"status":"ok"}`; zero prior `Customer` nodes confirmed before TP-02. |
| TP-02 (AC-1) | AC-1 | PASS | `@assistant I'd like to add 2 Wireless Mouse Pro to my cart` → reply confirms 2×$29.99=$59.98, `toolsUsed:["add_to_cart"]`; ground truth: exactly 1 `Customer{customerId:"u1"}`, 1 `Cart`, 1 `CartItem{productId:"wireless-mouse-pro", quantity:2}` — the `ensure_customer`/`ensure_cart` path proven live again on the new model, no silent no-op. |
| TP-03 (AC-1/AC-2) | AC-1, AC-2 | PASS | Added 1 Portable SSD 1TB (`toolsUsed` included `add_to_cart`+`view_cart`), then explicit `view_cart` turn → "Wireless Mouse Pro 2×$29.99=$59.98 / Portable SSD 1TB 1×$109.99=$109.99, Total: $169.97" — matches `2×29.99+109.99` exactly; ground truth confirms both `CartItem`s. |
| TP-04 (AC-2 — **verbatim D-1 repro**) | AC-2 | PASS (see D-1) | Turn 4 of the same continuous conversation: `@assistant remove 1 Wireless Mouse Pro from my cart` → `toolsUsed:["remove_from_cart"]` (genuine dispatch, unlike the qwen-era zero-call fabrication); ground truth immediately after: `wireless-mouse-pro` quantity 2→1, `portable-ssd-1tb` unchanged at 1 — the write is correct. The reply text itself, however, read "Updated Cart: Portable SSD 1TB $109.99, Total: $109.99" — omitting the still-present mouse line and understating the true total ($139.98) — filed as **D-1** (a reply-completeness gap, not a fabrication: the underlying state is right, only the summary is incomplete). |
| TP-05 (AC-2) | AC-2 | PASS | `@assistant clear my cart` → `toolsUsed:["clear_cart"]`; reply confirms empty; ground truth: `count(item)` → `0`. |
| TP-06 (AC-3) | AC-3 | PASS | Thread A: added 1 Portable SSD 1TB. Independent Thread B (fresh `threadId`, same actor `u1`): `@assistant what is currently in my cart?` → `["view_cart"]`, correctly reports the item added from Thread A — cart anchored to the customer, not the thread. |
| TP-07 (AC-4) | AC-4 | PASS | `pricing.compute_line_total([{price:29.99,qty:2},{price:109.99,qty:1}])` called twice via the server's own venv → `169.97` both times, byte-identical; matches hand calculation and TP-03's live total. |
| TP-08 (AC-5/AC-6) | AC-5, AC-6 | PASS (D-1 recurs) | Added 1 Wireless Mouse Pro onto a cart already holding 1 Portable SSD 1TB — reply read only "Added 1 × Wireless Mouse Pro ($29.99) to your cart," omitting the total/other line entirely (same D-1 class, this time on `add_to_cart`, matching the *original* report's own D-2 finding and confirming it is model-independent per the brief); ground truth confirmed both lines present regardless. `@assistant place my order` → `place_order` genuinely called, reply + ground truth agree: 2 `OrderLine`s (mouse $29.99, SSD $109.99), total $139.98, cart emptied afterward. Mutated `reference`'s `wireless-mouse-pro.price` 29.99→39.99 directly; re-read `OrderLine` — snapshot **unchanged** (`unitPrice:29.99`, `lineTotal:29.99`). Price mutation reverted. |
| TP-09 (AC-9) | AC-9 | PASS | Direct read of `pricing.py`/`services._priced_cart_lines`/`place_order` (unchanged code) confirms no `self._models`/LLM-client reference anywhere in the arithmetic chain; every total observed live (TP-03, TP-06, TP-07, TP-08) was exact and repeatable. |
| TP-10 (AC-7) | AC-7 | PASS | `POST /workflow-runs {defKey:"order-fulfillment", ctx:{orderId:"f5c09d65..."}}` → parked at `placed` (stepCount 1). `POST .../input {"input":{"action":"fulfill"}}` → run parks at `fulfilled` (stepCount 3); paired `services.advance_order(transition="fulfill")` → `{status:"fulfilled"}`. `POST .../input {"input":{"action":"deliver"}}` → run `status:"done"` (stepCount 5); paired `advance_order(transition="deliver")` → `{status:"delivered"}`. No transition happened without its own explicit call. |
| TP-11 (AC-8) | AC-8 | PASS | Second live-placed order (add mouse + place order in one turn, `toolsUsed:["add_to_cart","place_order"]`). Its own run: `input{"action":"cancel"}` while `placed` → run `done`; paired `advance_order(transition="cancel")` → `{status:"cancelled"}`. Follow-up `advance_order(transition="deliver")` on the now-cancelled order → `None` (zero-rows no-op), status stays `cancelled`. Separately, `advance_order(transition="cancel")` on TP-10's already-`delivered` order → `None`, status stays `delivered`. |
| TP-12 (AC-10) | AC-10 | PASS | Re-ran `seed_salesperson.sh qa-cart-totals2` — both `reference` defs and both `ws:qa-cart-totals2` snapshots report "already present — no-op"; `verify_salesperson.sh qa-cart-totals2` still `OK`, same topology. |
| TP-13 | — | PASS | Ground-truth Cypher cross-checked inline after every mutating turn in TP-02..TP-08 (not deferred) — matched the customer-visible reply's *state* claims every time; D-1's reply-completeness gap was itself only detectable *because* of this discipline (the write was correct even when the reply text wasn't). |
| TP-14 | — | PASS (confirms known, accepted gap) | Seeded a disposable `qa-throwaway-widget2` `Product`, added it + `wireless-mouse-pro` to a fresh synthetic customer's (`qa-tp14-cust2`) cart via direct `Services.add_cart_item` calls, deleted the product from `reference`, called `services.place_order` directly. Result: order created with only the surviving line (`wireless-mouse-pro`, $29.99), cart `[]` afterward, no signal in the returned JSON that a line was dropped — reproduces the implementation review's documented MINOR exactly, model-independent as expected. |
| TP-15 | — | PASS | `pytest -q` → **1920 passed/4 deselected** (matches U43's recorded baseline exactly); `./scripts/test_queries.sh` → **387/387** (matches). `ws:acme` restored (`bootstrap_schema.sh`→`seed_demo.sh`→`seed_catalog.sh`→`seed_workflows.sh`→`seed_salesperson.sh`) and re-verified: `verify_workflows.sh`/`verify_catalog.sh`/`verify_salesperson.sh acme` all `OK`/in sync. Shared final step for both this re-run and `workflow-catalog-lookup2.md`'s. |
| TP-16 | — | **Confirmed 1 instance (informational, not a gate)** | See D-2 below. |

## Defects

### D-1 (MINOR, new) — cart-mutation reply text intermittently omits a still-present line and understates the total

**Severity:** MINOR — the underlying write and cart state are correct in every single case
observed in this pass (confirmed via ground-truth Cypher every time); only the customer-visible
reply's own summary is sometimes incomplete. This is the same class of defect the original K-053
report filed as D-2 (there, only on `add_to_cart`) — this pass confirms it (a) is model-independent
(reproduces on Ministral, per the brief's own prediction) and (b) is not confined to `add_to_cart`
— it now also reproduces on `remove_from_cart`.

**Steps to reproduce (verbatim D-1 repro, `remove_from_cart` manifestation):** in one continuous
`@mention` thread (`ws:qa-cart-totals2`, `mistralai/ministral-3-3b`): (1) `@assistant I'd like to
add 2 Wireless Mouse Pro to my cart`; (2) `@assistant also add 1 Portable SSD 1TB`; (3) `@assistant
what's in my cart?`; (4) `@assistant remove 1 Wireless Mouse Pro from my cart`.

**Expected:** turn 4's reply reflects the cart's actual resulting contents and total.

**Actual:** turn 4's reply (`toolsUsed:["remove_from_cart"]`, genuinely dispatched) read: *"Removed
1 × Wireless Mouse Pro from your cart. --- Updated Cart: Portable SSD 1TB – 1 × $109.99 → $109.99.
Total: $109.99"* — omitting the still-present `wireless-mouse-pro` line (now at quantity 1) and
understating the total ($109.99 instead of the correct $139.98). Ground-truth Cypher taken
immediately after confirms the *write* was correct (`wireless-mouse-pro` quantity 1,
`portable-ssd-1tb` quantity 1 both present) — the gap is confined to the reply's own wording.

**Reproducibility check — not deterministic.** Re-ran the identical shape (add mouse+SSD onto an
existing cart, then remove 1 mouse) in a fresh, isolated thread: this time `remove_from_cart` was
followed by a genuine `view_cart` call in the same turn, and the reply correctly listed both
remaining lines and the correct total ($249.97 for that thread's own cart state). **A second
manifestation was independently observed at TP-08**, this time on `add_to_cart`: adding 1 Wireless
Mouse Pro onto a cart already holding 1 Portable SSD 1TB produced a reply reading only "Added 1 ×
Wireless Mouse Pro ($29.99) to your cart" — no total shown at all, not even a wrong one. Ground
truth again confirmed both lines present. This intermittency (sometimes complete, sometimes not,
across otherwise-similar turns) matches the original report's own D-2 characterization exactly.

**Not a K-053 code defect, not a data-correctness fault, not the qwen-era K-056 mechanism.** Every
mutation in this pass had a genuine tool dispatch and a correct resulting graph state; this is
purely a reply-wording completeness gap, present on both mutating cart tools, confirmed
model-independent as the coordination brief anticipated.

**Suggested improvement (unchanged from the original report's own D-2 recommendation, now doubly
motivated):** have `remove_from_cart`'s (and `add_to_cart`'s) own tool response include the
whole-cart running total directly (an extra `_priced_cart_lines` call, already used elsewhere in
`services.py`), removing the dependency on the model reliably volunteering a follow-up `view_cart`
call or correctly summarizing state it already has.

### D-2 (MINOR, informational — explicitly not a gate per this pass's brief) — Ministral's own duplicate-instruction defect reproduced live once

**Severity:** informational, not gating. Already characterized and judged non-blocking at pilot
scale by `docs/reviews/salesperson-tool-reliability-ml.md` §8.4/§9 (pooled rate 3.1%-30% depending
on conditioning, wide CIs, categorically less severe than K-056's own fabrication). This pass's
brief explicitly asks to watch for, not gate on, this pattern.

**Steps to reproduce:** fresh thread, fresh cart (confirmed empty via ground-truth Cypher
beforehand): (1) `@assistant add 1 Wireless Mouse Pro to my cart` → correctly added, quantity 1;
(2) `@assistant also add 1 Mechanical Keyboard K200` (same category as the mouse — Peripherals —
mirroring the ml-note's own `same-category-immediate` condition, its one confirmed-duplicate
condition at pilot scale).

**Expected:** turn 2 adds only the keyboard; the mouse stays at quantity 1.

**Actual:** turn 2's reply read *"Your cart now contains: Wireless Mouse Pro × 2: $59.98,
Mechanical Keyboard K200: $89.99. Cart total: $149.97"* — `toolsUsed:["add_to_cart","view_cart"]`.
Ground-truth Cypher confirms the mouse is genuinely at quantity 2 (`wireless-mouse-pro` qty 2,
`mechanical-keyboard-k200` qty 1) even though the customer's turn-2 text never mentioned the mouse
again. This is the exact mechanism and axis the ml-note's own §9.2 confirmed instance describes: an
uninstructed re-dispatch of an already-completed `add_to_cart` call, honestly grounded (a real
write, a reply that accurately reflects the — wrong — resulting state), self-disclosing (the
doubled quantity is stated plainly, an attentive customer or a `view_cart`-review step has a real
chance to catch it before checkout).

**Not a K-053 code defect** — this is Ministral's own characterized behavior, already tracked as an
accepted, disclosed risk conditional on Ministral's adoption (ml-note §9.5). Recorded here as this
pass's own live confirmation the pattern is real and reachable through the actual production
`@mention` path (the ml-note's own eval used a throwaway in-process harness script, not this path)
— not a new finding, and explicitly not weighed against the verdict per the brief.

## Coverage & gaps

**Covered:** all ten ACs, live, against the new model and the real graph — brand-new-customer
cart-write path, add/remove/clear, cross-thread persistence, exact deterministic totals (live and
via direct `pricing.compute_line_total`), the frozen order snapshot's independence from a later
catalog price change, the complete order-fulfillment lifecycle including both terminal outcomes and
both invalid-transition guards, and — the pass's central question — a verbatim replay of the exact
4-turn sequence that produced the qwen-era D-1, confirmed clean. Every mutating action was
cross-checked against ground-truth Cypher in the same test item, the discipline that caught D-1's
reply-completeness gap and confirmed D-2's real graph mutation.

**Gaps, disclosed, consistent with the parent plan's scope:**
- **AC-9's literal trace-based verification** unavailable on the `@mention` path — substituted with
  structural code read plus outcome-level evidence, same gap the original pass carried.
- **Multi-tenant cart isolation** not demonstrable until K-016 lands — unchanged caveat.
- **D-2's rate was not re-measured at scale here** — one dedicated probe plus opportunistic review
  of every other mutating turn (no further instances found in ~15 additional mutating turns across
  TP-02..TP-11); the ml-note's own dedicated eval (n=32+10) is the authoritative rate estimate, not
  this pass.
- **TP-14 used direct Services-layer calls, not a live `@mention` turn** — same deliberate
  substitution the original pass used, to keep this confirmatory check deterministic.

## Feedback & recommendations

1. **Ship the Ministral re-point for K-053's own scope too — the write-mutating escalation of
   K-056 does not reproduce.** This was the more consequential of the two re-verifications in this
   coordination (K-052's surface is read-only; K-053's is not), and the verbatim-replay methodology
   gave a direct, apples-to-apples answer: every `remove_from_cart`/`add_to_cart`/`place_order`
   call in this pass was genuinely dispatched and ground-truth-correct.
2. **D-1 (reply-completeness) is worth the same one-line fix recommended in the original report**
   — now doubly motivated since it demonstrably recurs across models and across both mutating cart
   tools. Not blocking.
3. **D-2 (Ministral's duplicate-instruction defect) is confirmed live and reachable through the
   real production path, not just the ml-note's own harness.** Per the ml-note's own §9.4, an
   untested but well-reasoned mitigation (a dispatch-time sanity check: does the resolved
   write-tool target appear in the current turn's own text) exists as a named follow-up candidate.
   This pass adds no new evidence for or against that mitigation's effectiveness — just a second,
   independent confirmation the underlying pattern is real, on the actual `@mention` path. Not
   this pass's call to weigh against K-054/K-055 sequencing — recorded for the coordination.
4. **Testability note, reused and reaffirmed from the original report's own feedback:** ground-truth
   Cypher checked in the same test item as the live turn (never deferred, never trusted from reply
   text alone) is what distinguished "the write is right but the reply is wrong" (D-1) from "the
   write is also wrong" (which is what the qwen-era D-1 was, and what this pass confirms does not
   happen here) — this discipline remains the load-bearing technique for any live pass on this
   harness, doubly so now that two distinct defect *shapes* (silent zero-effect fabrication vs.
   honestly-grounded-but-misreported state) can otherwise look identical from reply text alone.
5. **`docs/test-plans/workflow-cart-and-totals2.md` and this report `Extend`
   `docs/test-plans/workflow-cart-and-totals.md`** per collision rule 5 — the original stays
   intact as the qwen-era historical record.

## Artifacts left in the live demo (disclosed, not cleaned up)

- Fresh disposable workspace `ws:qa-cart-totals2` — left in place, not deleted, same precedent the
  original `ws:qa-cart-totals` pass set. Carries this pass's own test data: 7 threads (`demo-welcome`
  plus 6 fresh QA threads), 4 real orders under actor `u1` (one delivered, one cancelled, one
  placed via the combined add+place turn, one from the D-2 probe's cart state) plus a synthetic
  `qa-tp14-cust2` actor's own order from TP-14.
- `reference`'s catalog is back to its canonical 15-product state — the disposable
  `qa-throwaway-widget2` `Product` created for TP-14 was deleted as part of that same test item;
  the `wireless-mouse-pro` price mutation from TP-08 was reverted immediately after that item's own
  read-back.
- `ws:acme` fully restored to its standard demo baseline and re-verified in sync (TP-15, the shared
  final step for both re-runs in this coordination) — untouched by any of this pass's own live
  testing.
- Server process (uvicorn, port 8011) stopped at the end of this pass. Scratch
  `FALKORCHAT_OPENCODE_CONFIG` copy left only in this session's own scratchpad, not committed, not
  written to any shared file.
