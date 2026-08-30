# `workflow-salesperson-demo` — Test Plan (M6 combined e2e)

> **Status:** archived · **Owner:** `qa-engineer` · **Tracks:** K-052, K-053, K-054, K-055 (M6)

## 1. Scope & objective

M6's own closing condition: prove the combined `salesperson@v4` demo agent's four sibling
capabilities — catalog lookup (K-052), cart/order (K-053), durable profile (K-054), NL query
generation (K-055) — **coexist on one `WorkflowDef`, one `assistant` step, one `systemPrompt`,
one eleven-tool grant set**, without one capability's tools/state confusing another or regressing
a sibling now that all four are wired in together. This is **integration**, not re-litigation: each
capability already has its own live acceptance pass on record
(`docs/test-reports/workflow-catalog-lookup2-report.md`,
`docs/test-reports/workflow-cart-and-totals2-report.md`,
`docs/test-reports/workflow-durable-profile-report.md`,
`docs/test-reports/workflow-nl-query-generation2-report.md`) — this plan does not re-run each
capability's full AC list; it drives one realistic, continuous conversation (plus one
cross-thread follow-up) exercising all four together and watches specifically for cross-capability
interference.

## 2. References

- `docs/plans/workflow-salesperson-demo-coordination.md` — the coordination log for all four
  capabilities' build sequence into the one shared `salesperson` def; no standalone
  `plans/workflow-salesperson-demo.md` exists (this coordination doc has always been
  coordination-only for a 4-capability design phase, not itself an implementation plan for a
  fifth thing).
- `server/falkorchat/proof_defs.py` — `SALESPERSON_DEF` at `v4`: `config.tools` = the five K-052
  lookup/filter tools + five K-053 cart/order tools + two K-054 profile tools + K-055's
  `query_graph_data` (11 tools total on one `assistant` step); `config.model` =
  `lmstudio/mistralai/ministral-3-3b` (carried forward unchanged since K-056's re-point).
  Topology byte-identical across v1..v4 (2 steps, 1 transition) — only `config`/`systemPrompt`
  changed at each bump.
- The four requirements docs' AC lists (read, not re-verified in full here):
  `docs/requirements/workflow-catalog-lookup.md`, `workflow-cart-and-totals.md`,
  `workflow-durable-profile.md`, `workflow-nl-query-generation.md`.
- Each capability's own live acceptance report (cited above) — this plan's baseline for "already
  proven, don't re-prove."
- Graph schema for ground truth: `docs/QUERIES.md` §15 (catalog, global `reference`), §16
  (`Customer`/`Cart`/`CartItem`/`Order`/`OrderLine`, per-workspace), `docs/DESIGN.md` §5.1
  (`Entity`, per-workspace, K-055's second dataset — not exercised in this pass's own workspace,
  since this workspace has no ingested entity data; the `query_graph_data` catalog dataset is
  what this pass exercises, matching K-055's own AC-5 live-demo bar).

**CPG:** considered, not relevant — this pass is a live, black-box integration drive of
already-shipped, already-reviewed code; no structural-impact-analysis question it raises that
driving the running system doesn't answer better (same posture as every prior document in this
coordination).

## 3. Risk assessment

**What matters here, specifically:** does the outer model, given eleven tools and a
correspondingly longer `systemPrompt`, still pick the *right* tool for each of the four
capabilities' own question shapes within one conversation; does adding `query_graph_data`
(K-055, the newest, least fixed-shape tool) create any new confusion for the fixed-shape tools it
sits alongside (`filter_products`, `lookup_product_fact`); does profile state and cart/order state
survive correctly *together* in the same conversation and across a fresh thread (proving each
capability's own persistence claim holds when the other three are live in the same session, not
just in isolation); does nothing that worked in a capability's own single-capability QA pass stop
working now that the full tool set is present.

**Deliberately not tested here (out of scope, stated plainly):**
- Full re-verification of any single capability's own AC list — each has its own report on file.
- `order-fulfillment@v1`'s operator-side lifecycle advancement (fulfilled/delivered/cancelled) —
  K-053's own `workflow-cart-and-totals2-report.md` already covers this; this pass stops at
  `place_order` (cart→order transition), which is the point where all four capabilities'
  state has been exercised together.
- K-055's AC-2 (second dataset) — already proven in `workflow-nl-query-generation2-report.md`;
  this pass's fresh workspace has no ingested entity corpus, so it exercises K-055 against the
  `catalog` dataset only, which is sufficient for this pass's own integration question (does
  `query_graph_data` coexist correctly with the other ten tools) without requiring a fresh
  ingestion pass solely for this integration check.
- Multi-tenant/distinct-customer isolation — out of scope for every sibling capability's own
  requirements doc, unchanged here.

## 4. Environment & data setup

- FalkorDB: `falkordb-dev`, already up. `reference`: 15 `Product` nodes (global, unchanged
  baseline shared with every prior pass in this coordination).
- LM Studio: `http://localhost:1234`, `mistralai/ministral-3-3b` (assistant step) and
  `qwen/qwen3-4b-2507` (the `query_graph_data` tool's internal structured-completion step),
  confirmed listed.
- **Fresh throwaway workspace: `ws:qa-salesperson-demo`** — not reused from any prior pass in this
  coordination (`ws:qa-durable-profile`, `ws:qa-catalog-lookup2`, `ws:qa-cart-totals2`,
  `ws:nlq-eval` are each a different capability's own artifact). Provisioned:
  `EMBEDDING_DIM=1024 ./scripts/bootstrap_schema.sh qa-salesperson-demo` (indexes/constraints),
  `./scripts/seed_demo.sh qa-salesperson-demo` (standard `u1`/`assistant`/`demo-general` actors),
  `./scripts/seed_salesperson.sh qa-salesperson-demo` (materializes `salesperson@v4` +
  `order-fulfillment@v1`) — catalog is global `reference`, already seeded, no per-workspace step
  needed. Verified: `verify_salesperson.sh qa-salesperson-demo` → `OK`, in sync, topology 2
  steps/1 transition. Zero pre-existing `Customer` nodes confirmed
  (`MATCH (c:Customer) RETURN count(c)` → `0`).
- Server bound to this workspace: `FALKORCHAT_WS_ID=qa-salesperson-demo FALKORCHAT_USER_ID=u1
  FALKORCHAT_ENABLE_AGENT=1 FALKORCHAT_WORKFLOW_ENABLED=1 FALKORCHAT_TRIGGER_DEF_KEY=salesperson
  FALKORCHAT_TRIGGER_DEF_VERSION=v4 EMBEDDING_DIM=1024 FALKORCHAT_OPENCODE_CONFIG=<the same
  session-scratch lmstudio-only config used for deliverable 1 — localhost:1234/v1, registering
  both required models>`, `UVICORN_ARGS="--port 8023"` (no `--reload`). `GET /health` →
  `{"status":"ok"}`. The prior deliverable's server (port `8022`, `ws:nlq-eval`) was stopped first
  — this box's single-model-at-a-time LM Studio JIT loading means only one server/conversation is
  ever active at a time regardless, but running two `uvicorn` processes bound to different
  workspaces has no reason to coexist here.

## 5. Test items

One continuous conversation (Thread A) exercises all four capabilities in a realistic order
(browse → identify yourself → add to cart → ask an arbitrary question → check out); a second,
independent thread (Thread B, same actor) proves cross-conversation state survives correctly
*together*.

| ID | Title | Capability | Priority | Type |
|---|---|---|---|---|
| TP-01 | Environment pre-flight + fresh-workspace provisioning | — | P0 | environment/setup |
| TP-02 | Catalog lookup — exact-name fact question | K-052 | P0 | e2e/integration |
| TP-03 | Durable profile — give name + delivery address | K-054 | P0 | e2e/integration |
| TP-04 | Cart — add an item | K-053 | P0 | e2e/integration |
| TP-05 | NL query generation — arbitrary-phrased aggregate question | K-055 | P0 | e2e/integration |
| TP-06 | Order — place the order; cart clears | K-053 | P0 | e2e/integration |
| TP-07 | Cross-thread integration check — profile recalled, cart correctly empty, no re-ask | K-054, K-053 | P0 | e2e/integration |

### TP-01 — Environment pre-flight + fresh-workspace provisioning
**Steps/Expected:** the checks in §4, all green.

### TP-02 — Catalog lookup: exact-name fact question
**Steps:** fresh Thread A, `@assistant How much does the Wireless Mouse Pro cost?`. Ground truth:
`MATCH (p:Product {name:'Wireless Mouse Pro'}) RETURN p.price`.
**Expected:** correct price ($29.99), via `lookup_product_fact` — proves the oldest, most
fixed-shape tool in the eleven-tool set still resolves correctly first.

### TP-03 — Durable profile: give name + delivery address
**Steps:** same Thread A, next turn: `@assistant Hi, my name is Jordan Rivera and my delivery
address is 77 Birch Lane, Denver.`. Ground truth immediately after: `MATCH (c:Customer
{customerId:'u1'}) RETURN c.name, c.deliveryAddress, c.profileUpdatedAt`.
**Expected:** reply confirms the save; ground truth shows exactly one `Customer` node with both
fields set, via `save_profile`.

### TP-04 — Cart: add an item
**Steps:** same Thread A, next turn: `@assistant Add 1 Wireless Mouse Pro to my cart.`. Ground
truth: `MATCH (:Customer{customerId:'u1'})-[:HAS_CART]->(:Cart)-[:HAS_ITEM]->(i:CartItem) RETURN
i.productId, i.quantity`.
**Expected:** reply confirms; ground truth shows exactly one `CartItem{productId:'wireless-mouse-pro',
quantity:1}`, via `add_to_cart` — proves cart-write dispatch is unaffected by the profile turn
immediately preceding it (both tools reachable from the same, now-longer, `systemPrompt`).

### TP-05 — NL query generation: arbitrary-phrased aggregate question
**Steps:** same Thread A, next turn: `@assistant How many products do you have in the Wearables
category?` — an aggregate (count), not one of K-052's fixed lookup/filter shapes. Ground truth:
`MATCH (p:Product) WHERE p.category='Wearables' RETURN count(p)`.
**Expected:** correct count (2: Fitness Tracker Band, Smartwatch Series 5), via
`query_graph_data` — proves the newest, most general tool coexists correctly with the ten
fixed-shape/state tools already exercised in this same conversation, and that the model reaches
for it rather than misapplying `filter_products`/`lookup_product_fact` to a shape they cannot
express (count).

### TP-06 — Order: place the order; cart clears
**Steps:** same Thread A, next turn: `@assistant Please place my order now.`. Ground truth
immediately after: `MATCH (:Customer{customerId:'u1'})-[:PLACED]->(o:Order)-[:HAS_LINE]->(l:OrderLine)
RETURN o.orderId, o.status, l.productId, l.quantity, l.unitPrice`, and cart-item count
(`MATCH (:Customer{customerId:'u1'})-[:HAS_CART]->(:Cart)-[:HAS_ITEM]->(i) RETURN count(i)`).
**Expected:** reply confirms the order (item, total); ground truth shows exactly one `Order`
(`status:'placed'`) with one `OrderLine` matching TP-04's cart line at the catalog's current
price, and the cart item count is now `0` — via `place_order`. Proves the order-placement
mechanism (already deterministic/LLM-free per K-053's FR-8/AC-9) fires correctly with a durable
profile and a prior arbitrary NL-query turn both already in the same conversation's context.

### TP-07 — Cross-thread integration check
**Steps:** a fresh, independent Thread B (same actor `u1`, same workspace). Turn 1:
`@assistant Do you have any wireless accessories under $100?` (an unrelated catalog question —
watches for an unwanted re-ask of name/address). Turn 2: `@assistant What's my name, and what's
currently in my cart?`.
**Expected:** turn 1 does not re-ask for name/address (already known) and answers the actual
question. Turn 2's reply states the correct name (`Jordan Rivera`, via `get_profile`) and
correctly reports the cart as empty (via `view_cart` — the order in TP-06 already cleared it) —
proving profile persistence and post-order cart state are both correct **together**, in a thread
that never itself ran any of TP-02..TP-06's turns.

## 6. Entry/exit criteria

**Entry:** TP-01 all green.
**Exit:** TP-02..TP-07 all demonstrate correct, non-interfering behavior across all four
capabilities, backed by ground-truth Cypher after every state-changing turn. A wrong tool choice,
a state leak between capabilities, or a regression versus any capability's own already-recorded
acceptance pass is a reportable defect.

## 7. What's explicitly out of scope

Restated from §3: full per-capability AC re-verification, `order-fulfillment@v1`'s operator-side
lifecycle, K-055's AC-2 second dataset, multi-tenant isolation.
