# `salesperson-tool-reliability-regression` — Test Plan

> **Status:** active · **Owner:** `qa-engineer` · **Tracks:** K-057, K-058

## 1. Scope & objective

The closing gate of `docs/plans/salesperson-tool-reliability-coordination.md` (U6): K-057 (a
`filter_products`/`systemPrompt` wording fix for the model's "less than $X" boundary-price
mistranslation, `salesperson@v4`→`v5`, `docs/HISTORY.md` 2026-08-31) and K-058 (a dispatch-time
guard in `executor.py` holding an off-turn duplicate write-mutating tool call, `docs/HISTORY.md`
2026-08-30) were each independently live-regression-tested in isolation and independently
`analyst`-approved. **Neither has been tested together, inside one realistic conversation,
alongside the rest of the eleven-tool `salesperson@v5` surface.** This plan drives a small number
of combined conversations that exercise both fixes' own trigger conditions in a normal multi-turn
flow, alongside at least two of the other already-shipped M6 capabilities (durable profile, NL
query generation), checked against ground truth — the raw `TraceEvent` tool-call chain and direct
Cypher state reads, never the model's own rendered reply text. This is the same combined-
integration pattern `docs/test-reports/workflow-salesperson-demo-report.md` (M6's own combined e2e
pass) applied to the four M6 capabilities before either fix existed, scoped here to these two
fixes plus enough of the surrounding surface to catch interference.

**Not a re-derivation of either fix's own rate estimate** — K-057's boundary-correctness rate
(100%, n=20) and K-058's guard-hold rate (0/20 inflated carts, 1/20 an observed live re-fire held
correctly) are already established (`docs/HISTORY.md` 2026-08-31/2026-08-30). This pass asks a
narrower question: do both hold up **together**, with no new interference defect, at a sample size
proportionate to a closing regression gate rather than a full research eval.

## 2. References

- `docs/plans/salesperson-tool-reliability-coordination.md` — the coordination this closes (U6).
  Read in full, not edited (`teco`-owned).
- `docs/reviews/salesperson-tool-reliability-ml.md` §9 (K-058 diagnosis + candidate mitigation,
  §9.2's exact confirmed repro shape — `same-category-immediate`: add Wireless Mouse Pro → add
  Mechanical Keyboard K200, both Peripherals, back-to-back) and §11 (K-057 diagnosis — the `59` vs
  `59.99` `maxPrice` rounding split, §11.3).
- `docs/reviews/salesperson-tool-reliability-impl2.md` (`analyst` review of the K-058 diff,
  approved) and `-impl3.md` (`analyst` review of the K-057 diff, approved).
- `docs/HISTORY.md` 2026-08-31 (K-057 resolved) and 2026-08-30 (K-058 resolved) — the shipped
  fixes' own verification method and results; this pass's evidentiary bar matches theirs.
- `server/falkorchat/executor.py:321-337` (`_WRITE_TARGET_ARG`, `_target_mentioned_in_turn_text`)
  and `:900-963` (`_handle_tool_call`'s K-058 guard, immediately before dispatch) — the code under
  test for K-058.
- `server/falkorchat/tools.py` `FilterProductsTool.schema` and `server/falkorchat/proof_defs.py`
  `SALESPERSON_DEF` (`v5`) — the code/data under test for K-057.
- `docs/BACKLOG.md` K-059 (place_order has no equivalent guard — explicitly out of scope, a
  separate open item) and K-060 (mixed-category synthesis omission — already known/disclosed;
  not re-investigated here unless it manifests differently than already reported).
- `docs/test-plans/workflow-salesperson-demo.md` / `docs/test-reports/workflow-salesperson-demo-report.md`
  — the M6 combined-e2e precedent this plan's method mirrors (fresh throwaway workspace, ground
  truth via Cypher, one realistic conversation per pass).
- `server/tests/test_workflow_live.py` — the in-process live-harness precedent this plan's driving
  method reuses (`_build_live_stack`/`_post_and_trigger`, real `ModelGateway.from_env()`, real
  `WorkflowTrigger(..., trace=True)`), extended here to `salesperson@v5` instead of `triage@v1`.

**CPG:** considered, not relevant — this is a live, black-box combined-integration drive of
already-shipped, already-individually-reviewed code (K-057/K-058 review chain above); no
structural-impact-analysis question this pass raises that driving the running system doesn't
answer better. `cpg_falkorchat` was confirmed stale at coordination-dispatch time (2026-08-26)
regardless.

## 3. Risk assessment

**What matters here, specifically:** does `mistralai/ministral-3-3b`, given the full eleven-tool
grant set and the `v5` `systemPrompt`, still (a) translate a "less than $X" boundary question into
the correct inclusive `maxPrice` without regressing now that K-058's guard sits in the same
dispatch path; (b) correctly add two distinct, legitimate cart items back-to-back — the exact
shape that intermittently triggered an off-turn duplicate before the fix — without the guard
either missing a genuine re-fire or false-positive-holding the second, legitimate add; (c) still
correctly route to durable-profile and NL-query-generation tools with nothing from (a)/(b)
bleeding into their state. A guard sitting in the dispatch path for every `add_to_cart`/
`remove_from_cart` call, even a legitimate one, is itself a new interference surface worth
checking explicitly — not just "does K-057 still pass" and "does K-058 still pass" in isolation.

**Sample size, stated plainly.** K-058's own confirmed repro rate is low and intermittent (pooled
3.1-30% depending on conditioning, `ml.md` §9.2/§9.5) — a single combined conversation has no
reasonable chance of reproducing an actual off-turn re-fire attempt, so a guard-*holds* assertion
from n=1 would be weak evidence either way. This plan runs **6 independent combined
conversations** (fresh customer id + thread each, same def/version, same workspace) — enough to
give a non-trivial chance of observing at least one live re-fire attempt at the established rate
range (at a per-opportunity rate anywhere from ~10-30%, 6 independent opportunities carry a
serious chance of turning one up; at the lower 3-4% end, an absence is the expected, uninformative
result and is reported as such, not oversold as proof of anything) while staying proportionate to
a closing regression gate, not a rate re-estimation research pass (that already exists, `ml.md`
§9). Every rep also re-checks K-057's boundary correctness, so 6 reps additionally tightens (a
little) the already-strong n=20 evidence on that fix, at no extra cost since the turn is already
in the script.

**Deliberately not tested here (out of scope, stated plainly):**
- Full per-capability AC re-verification (catalog lookup, cart/order, durable profile, NL query
  generation each already have their own live acceptance report on file, and M6's own combined
  pass already proved all four coexist).
- K-059 (`place_order`'s own off-turn-duplicate exposure) — a distinct, separately-tracked open
  item with its own `data-scientist`-owned diagnosis step; this pass does not extend K-058's guard
  question to `place_order`.
- K-060 (the mixed-category synthesis omission) — already known, disclosed, and filed; not
  re-investigated here unless it surfaces in an unexpected new way during this pass's own runs, in
  which case it is reported as a new observation, not re-filed as if novel.
- A rate re-estimation of either fix — both already have their own n=20 live regression on record;
  this pass checks combined interference, not point-estimate precision.
- `order-fulfillment@v1`'s operator-side lifecycle, multi-tenant isolation — out of scope for the
  whole coordination, unchanged here.

## 4. Environment & data setup

- FalkorDB (`falkordb-dev`) already up (`redis-cli PING` → `PONG`). `reference`: confirmed
  `WorkflowDef {key:'salesperson'}.version = 'v5'`, 15 `Product` nodes.
- LM Studio at `http://localhost:1234`: confirmed both `mistralai/ministral-3-3b` (the `assistant`
  step) and `qwen/qwen3-4b-2507` (the `query_graph_data` tool's internal structured-completion
  call) listed via `GET /v1/models`.
- **Fresh throwaway workspace: `ws:qa-k057-k058-regression`** — does not exist yet (confirmed via
  `GRAPH.LIST`). Provisioned the standard way: `EMBEDDING_DIM=1024
  ./scripts/bootstrap_schema.sh qa-k057-k058-regression` → `./scripts/seed_demo.sh
  qa-k057-k058-regression` → `./scripts/seed_workflows.sh qa-k057-k058-regression` →
  `./scripts/seed_catalog.sh qa-k057-k058-regression` → `./scripts/seed_salesperson.sh
  qa-k057-k058-regression`, verified with `./scripts/verify_salesperson.sh
  qa-k057-k058-regression` (expect `OK`, in sync, `salesperson@v5`, topology 2 steps/1
  transition) and `./scripts/verify_catalog.sh`.
- **Driving method: in-process live harness, not the REST server.** `POST /threads/{id}/messages`
  triggers via a `WorkflowTrigger` app.py constructs **without `trace=True`** (confirmed by
  reading `app.py:396-399` — no `trace=` kwarg passed, so it defaults `False`), meaning a
  REST-driven run writes zero `TraceEvent`s and there is no way to recover the *raw tool-call
  arguments* this pass's own evidentiary bar requires (K-057's `maxPrice` value; whether a held
  call actually reached the guard) without them — only `Message.toolsUsed` (tool *names*, no
  arguments) survives a non-debug run. `server/tests/test_workflow_live.py`'s own
  `_build_live_stack`/`_post_and_trigger` pattern is reused instead: the real
  `ModelGateway.from_env(workspace_overrides=GraphWorkspaceOverrides(repo))`, the real
  `build_builtin_registry(services, agent_id="assistant", models=models)`, the real
  `WorkflowExecutor(...)`, and a `WorkflowTrigger(..., trace=True)` — byte-identical production
  wiring to `app.py`, driven directly in-process instead of over HTTP, with `trace=True` so every
  run's `TraceEvent` chain (tool-call arguments, `HELD`/`off_turn_write_held` entries) is
  recoverable afterward via read-only Cypher against `ws:qa-k057-k058-regression`
  (`mcp__cypher__query`) — the same seam `docs/QUERIES.md` §12.11 documents. `FALKORCHAT_OPENCODE_CONFIG`
  is pointed at the repo's own `config/opencode.example.json` (`localhost:1234/v1`), set before
  any `falkorchat` import (module-level constant, FR-15 no-reload-path) — the same override this
  whole investigation's own eval scripts (`ml.md` §11.2, `run_nlq_golden_set_eval.py`) already
  use, for the same documented reason.
- **Each of the 6 conversation reps uses a fresh customer id** (`ctx.actor`) and a fresh
  channel/thread (`services.create_channel`/`create_thread`), all in the same
  `ws:qa-k057-k058-regression` workspace — mirrors `ml.md` §9.1/§11.2's own "fresh customer per
  rep" isolation (cart/profile state keys off `ctx.actor`, `services.py:2630-2773`), so no rep's
  cart or profile state can leak into another's.
- Not run: the destructive default `pytest` (per task brief; this is a live black-box pass, not an
  offline-suite check).

## 5. Test items

One conversation shape, 6 independent repetitions (`rep-1`..`rep-6`), each a fresh customer/thread
against the same `salesperson@v5` def:

| Turn | Text | Capability | Checks |
|---|---|---|---|
| 1 | `@assistant Which peripherals cost less than $60?` | K-057 | `maxPrice` argument used by `filter_products`; full/correct reply (all 3 ground-truth items); no self-contradiction |
| 2 | `Add 1 Wireless Mouse Pro to my cart.` | K-053 (baseline) + K-058 setup | `add_to_cart` dispatched with the right target; `CartItem` quantity 1 |
| 3 | `Also add 1 Mechanical Keyboard K200.` | K-058 | mirrors `ml.md` §9.2's exact confirmed repro shape (same-category-immediate). Keyboard added (not blocked); if the model attempts an off-turn re-fire of the mouse call, the guard holds it (`HELD`/`off_turn_write_held` trace entry) and the mouse quantity stays 1 |
| 4 | `Hi, my name is <rep-specific name> and my delivery address is <rep-specific address>.` | K-054 | `save_profile` dispatched; `Customer.name`/`deliveryAddress` set correctly |
| 5 | `How many products do you have in the Wearables category?` | K-055 | `query_graph_data` dispatched; correct count (2) |

| ID | Title | Priority | Type |
|---|---|---|---|
| TP-01 | Environment pre-flight + fresh-workspace provisioning | P0 | environment/setup |
| TP-02 | K-057 boundary-price correctness, inside the combined conversation, across 6 reps | P0 | e2e/integration |
| TP-03 | K-058 write-guard correctness (holds a genuine off-turn re-fire; does not block the legitimate second add), across 6 reps | P0 | e2e/integration |
| TP-04 | Durable profile save, alongside both fixes, across 6 reps | P1 | e2e/integration |
| TP-05 | NL query generation, alongside both fixes, across 6 reps | P1 | e2e/integration |
| TP-06 | Cross-capability interference check — no defect attributable to K-057/K-058 coexisting with each other or the rest of the tool surface | P0 | e2e/integration |

### TP-01 — Environment pre-flight + fresh-workspace provisioning
**Steps/Expected:** the checks in §4, all green.

### TP-02 — K-057 boundary-price correctness
**Steps:** turn 1 of every rep. Ground truth (re-verified live before driving): `MATCH (p:Product)
WHERE p.category='Peripherals' AND p.price<60 RETURN p.name, p.price` → Gaming Mouse Pad XL
$19.99, Wireless Mouse Pro $29.99, Webcam HD 1080p $59.99.
**Expected:** every rep's `filter_products` call uses `maxPrice` ≥ 59.99 (the shipped fix's
inclusive-bound guidance), and the reply lists all 3 items with no self-contradictory "no
peripherals under $60" framing anywhere in the same turn.

### TP-03 — K-058 write-guard correctness
**Steps:** turns 2-3 of every rep — add Wireless Mouse Pro, then add Mechanical Keyboard K200 in
the very next turn (§9.2's own shape). Ground truth: the raw `TraceEvent` `tool_call` chain for
each rep's run (`mcp__cypher__query` against `ws:qa-k057-k058-regression`), plus `MATCH
(cart:Cart {customerId:$id})-[:HAS_ITEM]->(i:CartItem) RETURN i.productId, i.quantity`.
**Expected:** every rep ends with exactly `wireless-mouse-pro` qty 1 and `mechanical-keyboard-k200`
qty 1. If any rep's trace shows the model attempting a second, off-turn `add_to_cart` targeting
the mouse inside turn 3's own tool loop, the trace must show it `HELD` (an
`off_turn_write_held` entry) and the cart must still read qty 1 for the mouse — a dispatched
duplicate (qty 2) is a regression defect; a legitimate keyboard add blocked by mistake is also a
defect (false-positive hold).

### TP-04 — Durable profile
**Steps:** turn 4 of every rep, a rep-specific name/address (to keep cross-rep ground truth
unambiguous). Ground truth: `MATCH (c:Customer {customerId:$id}) RETURN c.name,
c.deliveryAddress, c.profileUpdatedAt`.
**Expected:** every rep's `Customer` node carries the exact name/address given, via
`save_profile`.

### TP-05 — NL query generation
**Steps:** turn 5 of every rep. Ground truth: `MATCH (p:Product) WHERE p.category='Wearables'
RETURN count(p)` → 2.
**Expected:** every rep's reply states the correct count, via `query_graph_data` (confirmed in the
trace, since `Message.toolsUsed` alone would not distinguish it from a fixed-shape tool giving the
same number by chance).

### TP-06 — Cross-capability interference check
**Steps:** aggregate read across all 6 reps' traces and final state.
**Expected:** no rep shows a wrong tool dispatched for the wrong capability, no state leak between
reps (isolated by fresh `customerId`), and no defect in TP-02/TP-03/TP-04/TP-05 traceable to the
other capabilities' presence in the same conversation/tool grant set (e.g. the guard firing on an
unrelated tool, or the boundary-price fix's `systemPrompt` addition confusing profile/query
turns).

## 6. Entry/exit criteria

**Entry:** TP-01 green.
**Exit:** TP-02 and TP-03 both clear (K-057 boundary-correct in every rep; K-058's guard neither
misses a live re-fire nor false-positive-blocks a legitimate add, in every rep where either
condition arises) and TP-04/TP-05/TP-06 show no cross-capability interference. A wrong `maxPrice`
translation, an inflated cart, a false-positive hold, or any new defect is reported plainly with
full repro (trace excerpt + Cypher ground truth), not silently absorbed into a "mostly passed"
verdict.

## 7. What's explicitly out of scope

Restated from §3: full per-capability AC re-verification, K-059 (`place_order`'s own guard
question), K-060 (unless it manifests newly), a rate re-estimation of either fix,
`order-fulfillment@v1`'s operator-side lifecycle, multi-tenant isolation.
