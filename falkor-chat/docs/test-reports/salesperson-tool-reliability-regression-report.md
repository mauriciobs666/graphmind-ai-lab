# `salesperson-tool-reliability-regression` — Test Report

> **Status:** active · **Owner:** `qa-engineer` · **Tracks:** K-057, K-058

## Summary

Live combined-integration pass executed 2026-08-31 against the real running system — FalkorDB
(`falkordb-dev`), a fresh throwaway workspace `ws:qa-k057-k058-regression` (torn down after this
pass, `GRAPH.DELETE`), LM Studio at `http://localhost:1234` serving `mistralai/ministral-3-3b`
(`salesperson@v5`'s `assistant` step) and `qwen/qwen3-4b-2507` (`query_graph_data`'s internal
structured-completion call) — per `docs/test-plans/salesperson-tool-reliability-regression.md`.
Code under test: `salesperson@v5` (K-057's shipped wording fix) + `server/falkorchat/executor.py`'s
K-058 dispatch-time write guard, both already individually live-regression-tested and
`analyst`-approved (`docs/HISTORY.md` 2026-08-31/2026-08-30). This is their first pass **together**,
inside one realistic conversation, alongside two more of the shipped eleven-tool surface (durable
profile, NL query generation).

**Driving method note (read before the results below):** the REST-facing `WorkflowTrigger`
`app.py` constructs never sets `trace=True` (`app.py:396-399`), so a REST-driven run writes zero
`TraceEvent`s and cannot ground-truth raw tool-call arguments — only `Message.toolsUsed` (tool
*names*) survives. This pass instead reused `server/tests/test_workflow_live.py`'s own in-process
live-harness pattern (byte-identical production wiring — real `ModelGateway.from_env()`, real
`build_builtin_registry`, real `WorkflowExecutor` — driven directly instead of over HTTP,
`WorkflowTrigger(..., trace=True)`), so every run's full `TraceEvent` chain is recoverable via
read-only Cypher against `ws:qa-k057-k058-regression`. Driver script (not part of the shipped
suite, not committed, session-scratchpad only):
`k057_k058_regression.py`.

**Verdict: PASS on both fixes' own narrow claims, WITH ONE NEW, GENUINE DEFECT FOUND** — exactly
the kind of interference this closing gate exists to catch, and exactly why "each fix passed in
isolation" was never sufficient on its own. K-057's boundary-price translation was correct in
6/6 combined conversations (matching its own n=20 isolated result). K-058's guard held every one
of 12 live off-turn re-fire attempts across the 6 conversations (100%) and never false-positive-
blocked the legitimate second add. **But in 2/6 (33%) of those same conversations, the model
itself — not the guard, and not blocked by it — silently duplicated its own already-successful,
*legitimately-mentioned* `add_to_cart` call later in the same turn's own tool loop, inflating a
cart line beyond what the customer asked for**, a mechanism K-058's guard is explicitly, correctly
scoped to *not* catch (`ml.md` §9.4 rules out blocking a repeat of a legitimately-mentioned
target, since a customer's own later "add another" request must not be blocked). Durable profile
and NL query generation were correct in 6/6 conversations each. Full details and repro below.

**CPG:** considered, not relevant — this is a live, black-box combined-integration drive of
already-shipped, already-individually-reviewed code; no structural-impact-analysis question this
pass raises that driving the running system doesn't answer better. `cpg_falkorchat` was confirmed
stale at coordination-dispatch time (2026-08-26) regardless.

## Run context

- **Date:** 2026-08-31 (harness driven 2026-08-30 22:xx-23:xx UTC-equivalent per `WorkflowRun`
  timestamps; see below — the calendar date rolled over mid-session).
- **Workspace:** `ws:qa-k057-k058-regression`, fresh, throwaway. `verify_salesperson.sh
  qa-k057-k058-regression` → `OK`, in sync, `salesperson@v5`, 2 steps/1 transition, before driving.
  `verify_catalog.sh` → `OK`, 15 products. Torn down (`GRAPH.DELETE`) after this pass;
  `reference`/`ws:acme` independently re-verified in sync afterward (`verify_workflows.sh acme`,
  `verify_catalog.sh`, `verify_salesperson.sh acme`, all `OK`) — this pass never wrote to either.
- **Models:** `lmstudio/mistralai/ministral-3-3b` (assistant step, confirmed loaded via `GET
  /v1/models` before driving), `lmstudio/qwen/qwen3-4b-2507` (internal `query_graph_data`
  completion, confirmed loaded).
- **6 independent conversations** (`rep-1`..`rep-6`), each a fresh `customerId`
  (`qa-k057k058-cust-<n>`) and fresh channel/thread, same def/version, same workspace — isolated
  per §4 of the test plan (cart/profile state keys off `ctx.actor`).

## Results table

| ID | Item | Result | Evidence |
|---|---|---|---|
| TP-01 | Environment pre-flight | **PASS** | FalkorDB up; both LM Studio models listed; workspace bootstrapped/seeded; `verify_salesperson.sh`/`verify_catalog.sh` both `OK`; zero pre-existing `Customer` nodes. |
| TP-02 | K-057 boundary-price correctness (6 reps) | **PASS** | `filter_products` used `maxPrice: 59.99` in every rep (6/6, 100%, Wilson 95% CI 61.0-100%) — 4 reps in one call, 2 reps (`rep-1`, `rep-2`) additionally self-refined to `{"category":"Peripherals","maxPrice":59.99}` on a second call, same correct bound. Every rep's reply listed all 3 ground-truth items (Gaming Mouse Pad XL $19.99, Wireless Mouse Pro $29.99, Webcam HD 1080p $59.99 — re-verified live: `MATCH (p:Product) WHERE p.category='Peripherals' AND p.price<60 RETURN p.name,p.price`), no self-contradiction in any rep. Matches K-057's own n=20 100% result (`docs/HISTORY.md` 2026-08-31). |
| TP-03 | K-058 write-guard correctness (6 reps) | **PASS** | Every rep's turn 3 shows the model attempting an off-turn re-fire of `add_to_cart(Wireless Mouse Pro)` — **twice each**, 12/12 attempts across 6 reps, all held (`HELD add_to_cart(...) — productName not mentioned in this turn's own text (K-058)`, `off_turn_write_held` trace entries). `wireless-mouse-pro` cart quantity stayed 1 in all 6 reps (ground truth: `MATCH (cart:Cart)-[:HAS_ITEM]->(i:CartItem) ... RETURN i.productId,i.quantity`). Zero false-positive blocks — `mechanical-keyboard-k200` was added in every rep (never rejected). |
| TP-04 | Durable profile (6 reps) | **PASS** | `save_profile` dispatched with the exact rep-specific name/address in every rep; `Customer` node ground truth matches exactly in all 6 (e.g. `qa-k057k058-cust-4` → `Riley Chen` / `9 Pine Ct, Reno`, `profileUpdatedAt` set). |
| TP-05 | NL query generation (6 reps) | **PASS** | `query_graph_data({"dataset":"catalog","question":"How many products are there in the Wearables category?"})` dispatched in every rep; every reply states the correct count. Ground truth: `MATCH (p:Product) WHERE p.category='Wearables' RETURN p.name,p.price` → Smartwatch Series 5, Fitness Tracker Band → count 2. |
| TP-06 | Cross-capability interference check | **FAIL — 1 new defect found** | See Defects below. `rep-4` and `rep-6` (2/6, 33%, Wilson 95% CI 9.7-70.0%) show `mechanical-keyboard-k200` quantity **2** in the cart though the customer asked for 1 — a same-turn self-duplicate of the model's own legitimate `add_to_cart` call, not caught by K-058's guard because its target *is* mentioned in the turn's own text (out of the guard's designed scope). `rep-2` (1/6, 17%, Wilson 95% CI 3.0-56.4%) shows a related but distinct synthesis defect: a false "I couldn't find a product named Mechanical Keyboard K200" reply despite a fully successful, correctly-recorded add (no quantity error there — data was right, the customer-facing text was wrong). |

## Defects

### Defect 1 (new) — Same-turn self-duplicate `add_to_cart` on a legitimately-mentioned target, immediately following the K-058 guard holding an unrelated off-turn call in the same turn

**Severity: MAJOR.** Silently inflates a cart line beyond the customer's actual request; not
prevented by K-058's guard (correctly out of its scope — see below); not consistently
self-disclosed in the reply the customer reads.

**Not K-057, not K-058's own confirmed mechanism (`ml.md` §9.2 — an *off-turn* re-fire of a
*previous turn's already-completed* action), and not K-060 (the `filter_products`
mixed-category synthesis omission).** This is a third, distinct pattern: a *same-turn* duplicate
of the model's *own current-turn* successful write, occurring right after the model saw two
consecutive `{"held": true, ...}` rejections for an unrelated tool call earlier in the same turn's
own multi-iteration loop.

**Steps to reproduce** (exact, live-confirmed twice — `rep-4` runId `37238ba184a147d7a614714ed035079f`,
`rep-6` runId `232488d5190e43cbba7764395712c59b`, both against `ws:qa-k057-k058-regression`,
`salesperson@v5`):
1. Fresh conversation. Turn 1: `Which peripherals cost less than $60?` (any correct K-057 answer).
2. Turn 2: `Add 1 Wireless Mouse Pro to my cart.` — dispatches cleanly, cart: mouse ×1.
3. Turn 3: `Also add 1 Mechanical Keyboard K200.`

**What happens inside turn 3's own tool loop (both occurrences, identical shape):**

```
iter 2: tool_calls: add_to_cart, add_to_cart
  → HELD add_to_cart({"productName": "Wireless Mouse Pro"})            [correct — K-058]
  → add_to_cart({"productName": "Mechanical Keyboard K200"})            [dispatched — quantity: 1]
iter 3: tool_calls: add_to_cart
  → HELD add_to_cart({"productName": "Wireless Mouse Pro", "quantity": 1})  [correct — K-058]
iter 4: tool_calls: add_to_cart
  → add_to_cart({"productName": "Mechanical Keyboard K200", "quantity": 1}) [dispatched AGAIN — quantity: 2]
```

**Expected:** exactly one `mechanical-keyboard-k200` `CartItem` at quantity 1 (the customer asked
for 1, once).

**Actual (ground truth, `MATCH (cart:Cart {customerId:$id})-[:HAS_ITEM]->(i:CartItem) RETURN
i.productId, i.quantity`):**
- `rep-4` (`qa-k057k058-cust-4`): `mechanical-keyboard-k200` → **2**, `wireless-mouse-pro` → 1.
  Reply text (`post_message` payload): *"Your cart now includes:\n- **Wireless Mouse Pro** –
  $29.99\n- **Mechanical Keyboard K200** – $89.99\n\nWould you like to add anything else or
  proceed to checkout?"* — **no quantity marker at all; the duplicate is completely
  undisclosed** to the customer.
- `rep-6` (`qa-k057k058-cust-6`): `mechanical-keyboard-k200` → **2**, `wireless-mouse-pro` → 1.
  Reply text: *"I couldn't add the **Wireless Mouse Pro**—it wasn't explicitly requested in this
  message. I *did* add **one** Mechanical Keyboard K200 to your cart..."* — accurately explains
  the held mouse call, but **explicitly, falsely claims "one"** when two were actually dispatched.

**Why K-058's guard cannot and should not catch this as written.** The guard
(`_target_mentioned_in_turn_text`, `executor.py:327-337`) checks whether a write call's resolved
target is mentioned anywhere in the turn's own text — "Mechanical Keyboard K200" genuinely *is*
mentioned in turn 3's text, on both the iter-2 and the iter-4 call. Blocking a second call whose
target is legitimately mentioned is exactly the "blind cross-turn dedup-by-signature" fix
`ml.md` §9.4 already ruled out (it would incorrectly block a customer's own later "add another
one" request) — this defect sits in the gap that ruling deliberately leaves open, now
ground-truth-confirmed as a live, non-hypothetical occurrence rather than a theoretical risk.

**Contributing observation, not itself a defect but material context:** in this pass's combined
conversation shape, **all 6/6 reps** produced the exact two-consecutive-held-rejection pattern on
the mouse (12/12 attempts held, 100%) — a markedly higher off-turn-re-fire *attempt* rate than
K-058's own isolated eval saw for a similar shape (1/20, `docs/HISTORY.md` 2026-08-30). Both
duplicate-keyboard occurrences (`rep-4`, `rep-6`) happened in turns that hit this exact
two-held-rejections shape; the 4 reps that did *not* duplicate the keyboard (`rep-1`, `rep-2`,
`rep-3`, `rep-5`) hit the identical two-held-rejections shape too, so this pass's own n=6 cannot
establish two-held-rejections as a deterministic trigger — only that every observed occurrence of
the new defect co-occurred with it. Worth a larger follow-up sample if this is prioritized (see
Feedback below); not claimed as proven causation here.

### Defect 2 (new, related but distinct) — Model states a false "product not found" failure after a fully successful `add_to_cart`

**Severity: MAJOR** (customer-facing false statement about order state; data integrity itself
unaffected — cart quantity was correct in this occurrence).

**Steps to reproduce:** identical to Defect 1's steps 1-3. `rep-2` (runId
`449e712581544efab9299d021d3bb810`, `ws:qa-k057-k058-regression`).

**Trace (turn 3, full):**
```
iter 1: lookup_product_fact({"name": "Mechanical Keyboard K200"}) → found, $89.99
iter 2: HELD add_to_cart({"productName": "Wireless Mouse Pro"})                    [correct]
        add_to_cart({"productName": "Mechanical Keyboard K200"}) → found, quantity: 1  [SUCCESS]
iter 3: HELD add_to_cart({"productName": "Wireless Mouse Pro", "quantity": 1})     [correct]
iter 4: final text: "I couldn't find a product named **Mechanical Keyboard K200** in our
        catalog. Did you mean another model? Let me know what you'd like to add instead!"
```
**Expected:** a reply confirming the keyboard was added (it was, cleanly, at iter 2).
**Actual:** the customer is told the product doesn't exist — the opposite of what happened.
**Ground truth (unaffected by this defect):** `MATCH (cart:Cart {customerId:'qa-k057k058-cust-2'})
-[:HAS_ITEM]->(i:CartItem) RETURN i.productId, i.quantity` → `mechanical-keyboard-k200` qty 1,
`wireless-mouse-pro` qty 1 — both correct. Only the reply text is wrong.

**Practical risk chain worth naming:** a real customer reading this false reply has no reason to
suspect their cart is actually correct — a plausible next action is retrying ("let me try
again"), which (being a fresh, current-turn instruction) K-058's guard would *not* hold, since the
target would then genuinely be mentioned in that later turn's own text — potentially producing a
real, customer-induced duplicate on top of an already-silent one. Not observed in this pass (no
rep retried), named as a risk the defect creates, not a separately confirmed occurrence.

## Coverage & gaps

**Covered:** 6 independent, realistic combined conversations, each exercising K-057's boundary-
price trigger, K-058's exact confirmed off-turn-repro shape (`ml.md` §9.2), durable profile, and
NL query generation, all in one continuous thread per rep — the same combined-integration pattern
`docs/test-reports/workflow-salesperson-demo-report.md` applied to M6's four capabilities.
Ground truth throughout is the raw `TraceEvent` tool-call chain and direct Cypher state reads,
never the model's own rendered reply text, per this investigation's established discipline.

**Gaps, stated plainly (restated from the plan's §3/§7, plus what this pass's findings newly
motivate):**
- **K-059** (`place_order`'s own off-turn-duplicate exposure) was not exercised — this pass's
  6/6 finding that the off-turn re-fire *attempt* rate can run much higher than K-058's own
  isolated 1/20 in a realistic multi-turn conversation is directly relevant context for whoever
  picks up K-059 next, since `place_order` has no equivalent guard at all.
- K-060 was not re-investigated and did not resurface in this pass's own 6 reps (turn 1's
  `filter_products` calls in every rep either passed `category` on a refinement call or produced
  a correct reply on the mixed-category call alone — no dropped-match instance observed here,
  consistent with it being a separate, lower-rate mechanism, `docs/HISTORY.md` 2026-08-31).
- Full per-capability AC re-verification, `order-fulfillment@v1`'s lifecycle, multi-tenant
  isolation — out of scope per the plan, unchanged.
- This pass's own n=6 is sized for a closing regression gate, not a rate re-estimation of either
  new defect above — both Wilson CIs are wide (9.7-70.0% and 3.0-56.4%) and should not be read as
  precise point estimates if this work continues.

## Feedback & recommendations

- **The headline finding: K-057 and K-058 each hold up perfectly on their own stated claims
  together, but their combination with the rest of a realistic conversation surfaces a genuine,
  new, previously-undetected defect neither fix's own isolated eval was positioned to catch.**
  This is precisely the value this closing-gate pattern (`docs/test-reports/
  workflow-salesperson-demo-report.md`'s own precedent, applied here) exists to provide, and the
  coordination should not close on "both fixes individually verified" alone.
- **This pass's finding does not fault K-058's implementation** — the guard did exactly what it
  was specified and reviewed to do (`docs/reviews/salesperson-tool-reliability-impl2.md`, approved),
  100% of the time, on 12 real attempts. The new defect sits in a gap `ml.md` §9.4 already named
  and deliberately left open (no safe way to block a legitimately-mentioned repeat) — this pass
  turns that named theoretical gap into a ground-truth-confirmed, twice-reproduced live
  occurrence, which is new information worth having even though it doesn't implicate the shipped
  code as wrong.
- **Recommend routing both defects above back through `teco`** for a backlog decision (new K-item
  vs. folding into K-059's own upcoming `place_order` guard-design work, since the underlying
  model tendency — re-issuing/duplicating a write after seeing a nearby rejection — looks like the
  same family) — not something this pass fixes itself, per this agent's own guardrails and the
  task brief's own instruction.
- **Worth a larger follow-up sample (n≈20-30) specifically on the two-consecutive-held-rejections
  conversation shape** if the team wants to move past "found twice in six" toward a rate estimate
  — this pass was sized for a regression gate, not for that.
- No regression found versus any of the four M6 capabilities' own prior acceptance passes for the
  two capabilities this pass touched (profile, NL query) — both fired correctly in every rep,
  unaffected by either fix's presence.
