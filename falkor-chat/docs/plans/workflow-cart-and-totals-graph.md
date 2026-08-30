# Cart, orders, and deterministic totals — Graph Design

> **Status:** archived · **Owner:** `graph-dba` · **Tracks:** — (M<n> TBD) · **Version:** 2

Graph-side design note for `docs/requirements/workflow-cart-and-totals.md` (FR-1..FR-9,
AC-1..AC-10). Schema, indexes/constraints, and write/read-path Cypher for a durable,
workspace-scoped shopping cart and an immutable placed-Order record with a lifecycle. An
`architect` agent is producing the step/tool integration plan for the same requirements
document in parallel; this note owns the graph shape and the exact Cypher, and is meant to be
pointed at by path, not re-derived.

*Revision note — 2026-08-26 (v1 → v2).* `docs/plans/workflow-catalog-lookup.md` §3.1 landed
(the sibling `architect` plan) with the catalog's `Product` schema keyed on `productId`, not
the `sku` this note assumed in v1 pending that landing. Every `sku` occurrence renamed to
`productId` throughout (schema, Cypher, prose); §8 item 1 records the resolution; §0 records
the re-verification of the one write shape the rename materially touched.

**This document also defines the `Customer` anchor node** — the workspace-local identity both
this capability and `docs/requirements/workflow-durable-profile.md` need. §1 is written once,
here, because carts need it first; `docs/plans/workflow-durable-profile-graph.md` references
this section rather than re-deriving it and adds only the two profile properties.

---

## 0. Verification standing

Every write shape below marked **[verified]** was executed on **2026-08-26** against the live
pinned instance (`falkordb/falkordb:v4.18.11`, module `41811`) via `redis-cli GRAPH.QUERY`
against a disposable graph key (`ws_cartprobe_test`, `ws_cartprobe_test2` — plain non-`ws:`
names, never touching `reference` or any real `ws:<id>`), created fresh and `GRAPH.DELETE`d at
the end of the session. The `mcp__cypher__query` tool was not used for this verification — it
authorizes writes only in six `kaizen_team`-specific shapes (see its own tool instructions), so
it cannot exercise a new write path against a throwaway workspace graph; direct `redis-cli`
against the `falkordb-dev` container (already running, confirmed via `docker ps`) is the
verification channel this note actually used, following the spirit of `falkor-chat/AGENTS.md`'s
"Probing shared graph state without mutating it" (same throwaway-graph discipline, DDL-only-safe
guidance extended here to disposable *data* since no `reference`/`ws:<id>` graph was touched).

**v1→v2 rename re-check (2026-08-26, same day, separate disposable graph
`ws_cartprobe_rename_check`, deleted at the end):** the `sku`→`productId` rename is a pure
property-name swap with no behavioral difference, but the one write shape it touches most
directly — the 2-property composite `UNIQUE` constraint on `CartItem` and the `MERGE`-and-
increment that relies on it — was re-run under the new name rather than assumed. `CREATE INDEX
FOR (n:CartItem) ON (n.productId)` + `GRAPH.CONSTRAINT CREATE ... UNIQUE NODE CartItem
PROPERTIES 2 customerId productId` came back `OPERATIONAL`; two `MERGE`-and-increment calls
with `qty=2` against the same `(customerId, productId)` produced `quantity=4` and exactly one
node, identical to the original `sku`-named result in §9. Every other renamed occurrence (the
`add_to_cart`/`adjust_cart_item`/`place_order`/`get_order` query bodies, the DDL block in §7) is
a straight identifier substitution with no new Cypher shape introduced — not independently
re-run.

---

## 1. The `Customer` anchor — workspace-local identity for cart, order, and profile

### 1.1 The problem

FR-2 requires the cart to be "anchored to the customer within that workspace, not to any one
thread." Nothing in the existing schema names a durable "customer" — the closest precedent,
`access-request`'s `assignee` field (`server/falkorchat/proof_defs.py:77,97`, values
`"requester"`/`"manager"`), is a **free-text role label baked into `Step.config`**, never a
graph reference to any node. There is no existing "who is this run for" node to reuse.

The nearest *real* identity concept already in the schema is `(:User {userId})` — the
workspace-local membership projection of `identity` (DESIGN §1.2, §5.1), stable across
conversations/threads, exactly the durability FR-2 needs. But `User` is deliberately
create-once: `ensure_user` (`repository.py:304-325`) is a guarded `CREATE`, never updated after
first write, and exists to answer "who authored this message," not "what does this customer
want to buy" — piling mutable commercial fields (cart pointer, profile) onto it would couple an
auth-flavored, create-once projection to state that must be freely mutable (FR-3's live cart,
FR-3-of-the-profile-doc's updatable address) and would put every future identity-sync concern
in this capability's blast radius for no reason.

### 1.2 Decision

**A new `Customer` node, one per workspace, keyed on `customerId`, where `customerId` is set
equal to the anchoring `User.userId`** — same value, distinct label:

```
(:Customer {customerId, createdAt})
```

- `customerId = the userId of the User who is chatting` — resolved once, by the run/tool
  wiring, from `(:WorkflowRun)-[:TRIGGERED_BY]->(:Message)-[:POSTED_BY]->(:User)` (the same
  traversal `find_runs_for_thread`/message reads already do) or from whatever the demo's
  tool-calling convention passes as the acting member id. **This is the one call in this note
  that isn't purely a graph-design decision** — it fixes the identity source the tool layer must
  supply — but it follows directly from "which workspace-local identity concept already exists"
  (there is exactly one: `User.userId`), so I am making the call rather than escalating it: it
  is reversible (a later migration could re-key `Customer` off something else) and does not
  block or contradict anything in the sibling requirements docs.
- **Not the `User` node itself** — for the reasons in §1.1: keeps `ensure_user`'s create-once
  contract untouched, and gives cart/order/profile a node that is genuinely, unremarkably
  mutable without any auth-adjacent baggage.
- **Not scoped by `workspaceId` property** — the graph key (`ws:{workspaceId}`) already is the
  scope, per rule 7.

### 1.3 `ensure_customer` — get-or-create, idempotent

**[verified]**
```cypher
// $customerId, $now
MERGE (c:Customer {customerId: $customerId})
ON CREATE SET c.createdAt = $now
RETURN c.customerId AS customerId, c.createdAt AS createdAt
```
Confirmed live: two calls with different `$now` values leave `createdAt` at the **first** call's
value (`ON CREATE` not re-applied on the second `MERGE` hit), exactly one `Customer` node.
Backed by the `Customer.customerId` uniqueness constraint (§7) — this `MERGE` is safe per rule
"every `MERGE` must be backed by a uniqueness constraint."

### 1.4 Where the durable-profile fields live

`docs/plans/workflow-durable-profile-graph.md` adds `name`/`deliveryAddress`/
`profileUpdatedAt` as two more properties on this **same** `Customer` node — no separate
`Profile`/`Contact` node. Rationale (stated once, here, referenced from there): the profile is
two scalar fields with no independent identity, no edges of its own, and no lifecycle apart from
"the customer's current name/address" — reifying it as its own node would add a hop for zero
traversal benefit. See that document's §1 for the write/read Cypher.

---

## 2. Cart / CartItem — durable, mutable, live-priced

```
(:Customer)-[:HAS_CART]->(:Cart {customerId, createdAt, updatedAt})
(:Cart)-[:HAS_ITEM]->(:CartItem {customerId, productId, quantity, addedAt, updatedAt})
```

- **`Cart` is keyed on `customerId`, not a separate `cartId`.** One cart per customer, forever
  (no multi-cart requirement anywhere in the FRs) — reusing `customerId` as `Cart`'s own natural
  key avoids a pointless surrogate id and mirrors the precedent `WorkflowDefSnapshot` already
  sets for a composite/derived natural key instead of a synthetic `{label}Id` (DESIGN §7.1).
- **`CartItem` is keyed on the composite `(customerId, productId)`, no separate `cartItemId`.**
  Same precedent, same reasoning: a cart line's real identity *is* "this customer's line for this
  product" — a synthetic id would be one more thing to plumb through the tool-calling surface for
  no benefit, since every cart tool operates in terms of `productId` (add/remove/adjust by item),
  never an opaque line id.
- **No `price`/`name` on `CartItem`.** FR-3 is explicit: a cart's displayed price is the
  catalog's *current* price, not a snapshot. Storing a price on `CartItem` would go stale exactly
  when FR-3 says it must not — see §6 for how the live total is actually computed (a two-graph
  read, not a single query, because `reference` and `ws:{id}` cannot be joined by an edge).
- **`productId`** is `CartItem`'s join key into the catalog — confirmed against
  `docs/plans/workflow-catalog-lookup.md` §3.1 (the architect's graph design for that sibling
  requirements doc, landed after this note's first pass): `(:Product {productId, name,
  nameNormalized, category, price})` in `reference`, `productId` a server-minted uuid with its
  own range index + UNIQUE constraint. This was originally written against an assumed `sku`
  join key (flagged in §8 of the first pass) — renamed throughout once the real schema landed;
  §0 records the re-verification.

### 2.1 `add_to_cart` — MERGE-and-increment, idempotent-by-design

**[verified]** — two calls with `qty=2` against the same `(customerId, productId)` leave exactly one
`CartItem` with `quantity=4`, never two rows:
```cypher
// $customerId, $productId, $qty, $now
MATCH (cart:Cart {customerId: $customerId})
MERGE (cart)-[:HAS_ITEM]->(item:CartItem {customerId: $customerId, productId: $productId})
ON CREATE SET item.quantity = $qty, item.addedAt = $now, item.updatedAt = $now
ON MATCH  SET item.quantity = item.quantity + $qty, item.updatedAt = $now
SET cart.updatedAt = $now
RETURN item.productId AS productId, item.quantity AS quantity
```
Zero rows ⇒ no `Cart` for that `customerId` yet — the caller runs `ensure_customer` +
`ensure_cart` (§2.3) first, exactly the `ensure_user`-before-write convention this codebase
already uses everywhere else. **"Add" means increment, not replace** — calling this twice with
the same `productId` models "the customer added two more," matching the FR-1 user story's
shopping-cart mental model; a workflow author who wants "set quantity to exactly N" computes the
delta app-side before calling this (no separate primitive needed).

### 2.2 `adjust_cart_item` — guarded decrement-or-remove, single query

**[verified]** — decrementing 4→3 updates in place (no delete); decrementing the remaining 3→0
deletes the node and its edge in the same call:
```cypher
// $customerId, $productId, $qty (positive = amount to remove), $now
MATCH (cart:Cart {customerId: $customerId})-[:HAS_ITEM]->(item:CartItem {customerId: $customerId, productId: $productId})
WITH cart, item, (item.quantity - $qty) AS newQty
FOREACH (_ IN CASE WHEN newQty > 0  THEN [1] ELSE [] END | SET item.quantity = newQty, item.updatedAt = $now)
FOREACH (_ IN CASE WHEN newQty <= 0 THEN [1] ELSE [] END | DETACH DELETE item)
SET cart.updatedAt = $now
RETURN newQty AS quantity, (newQty <= 0) AS removed
```
Zero rows ⇒ no such line (already removed, or never added — a workflow author's "remove an item
not in the cart" is a no-op the app reports as such, not an error). This is the same
`FOREACH (_ IN CASE WHEN … THEN [1] ELSE [] END | …)` guarded-write idiom `QUERIES.md` §4 uses for
message writes — chosen deliberately over two round trips (check-then-act) precisely because
FalkorDB serializes writes per graph and folding a check-then-act into one query is race-free for
free (`falkordb-quirks.md`, "Concurrency & atomicity") — moot here since concurrent cart edits are
explicitly out of scope (requirements doc, "Out of scope"), but free correctness at zero extra
cost.

### 2.3 `ensure_cart` — get-or-create, mirrors §1.3

**[verified]** — MERGE across the whole `(Customer)-[:HAS_CART]->(Cart)` pattern is safe here
specifically *because* `Cart`'s matched property (`customerId`) is fully specified and
constraint-backed on both ends, unlike the K-034 `START`-edge gotcha (`falkordb-quirks.md`, "a
`MERGE` with a changed endpoint creates a second edge") where the *target* of the relationship
could legitimately differ between calls. Confirmed live: two calls produce exactly one `Cart`
node and exactly one `HAS_CART` edge.
```cypher
// $customerId, $now — call after ensure_customer
MATCH (cust:Customer {customerId: $customerId})
MERGE (cust)-[:HAS_CART]->(cart:Cart {customerId: $customerId})
ON CREATE SET cart.createdAt = $now, cart.updatedAt = $now
RETURN cart.customerId AS customerId, cart.createdAt AS createdAt
```

### 2.4 `read_cart` — current lines (prices resolved separately, §6)

```cypher
// $customerId
MATCH (cart:Cart {customerId: $customerId})-[:HAS_ITEM]->(item:CartItem)
RETURN item.productId AS productId, item.quantity AS quantity, item.addedAt AS addedAt
ORDER BY item.addedAt
```
Zero rows is a legitimate "empty cart" (or no cart yet) — both look the same, which is fine:
FR-1's "view the cart" and "clear it entirely" don't need to distinguish "never touched" from
"emptied."

### 2.5 `clear_cart` — used standalone (FR-1) and inside `place_order` (§3.2)

```cypher
// $customerId
MATCH (cart:Cart {customerId: $customerId})-[:HAS_ITEM]->(item:CartItem)
DETACH DELETE item
```
A `MATCH` against zero items is a plain no-op — no `UNWIND`-collapse risk here (that quirk is
about an empty **list parameter** fed through `UNWIND`, not a `MATCH` finding no rows; those are
different mechanisms and only the former needs the `CASE WHEN … = [] THEN [null] …` guard).

---

## 3. Order / OrderLine — immutable snapshot, explicit lifecycle

```
(:Customer)-[:PLACED]->(:Order {orderId, status, placedAt, updatedAt})
(:Order)-[:HAS_LINE]->(:OrderLine {productId, name, unitPrice, quantity, lineTotal})
```
`status ∈ {'placed', 'fulfilled', 'delivered', 'cancelled'}` — a **property**, not a label,
mirroring `WorkflowRun.status`/`StepRun.status` (DESIGN §1.2, "avoids re-labeling churn").

- **`OrderLine` has no independent identity/index/constraint.** It is a pure value-object,
  always reached via `HAS_LINE` from the constraint-anchored `Order` — never looked up on its
  own by any AC. This follows the same "index the anchor, not every hop" rule that leaves
  `Step.key` index-only in `reference` (DESIGN §7.2): giving it a synthetic `orderLineId` would
  be schema weight nothing reads.
- **`unitPrice`/`name` are snapshotted at placement** (unlike `CartItem`, which never stores
  them) — this is precisely AC-6's requirement: a later catalog price change must never
  retroactively alter a placed order.
- **`Order.total` is deliberately NOT stored.** It's `sum(OrderLine.lineTotal)` computed on
  read (§3.3) — per-order line count is tiny (bounded by how many distinct products one order
  contains, never a supernode), so the aggregate is free, and computing it avoids a
  write-time-only derived field that could ever drift from its own source lines. If a future
  "list orders with totals" listing needs to avoid the aggregate per row at scale, that's a
  reason to reconsider — not needed by any AC today.

### 3.1 The idempotency shape: guarded `CREATE`, not `MERGE` — mirrors `Message`

Placing an order is a **one-time durable-record creation event**, the same shape as posting a
chat message (DESIGN §5.3/§9: "no `MERGE` on `Message`… guarded `CREATE`… retry replay is a
no-op"), not a "find or update" concept like `Cart`/`Customer`. So it uses the identical
guarded-`CREATE`-with-status-row idiom, keyed on a caller-minted `orderId` (the idempotency key,
supplied by whichever tool/service call drives checkout — analogous to `Message.msgId` being
server/caller-minted, never re-derived): a retried call with the same `orderId` is a true no-op,
not a duplicate order and not an error.

### 3.2 `place_order` — snapshot + clear cart, one atomic query

**[verified live, including idempotent replay]** — `$lines` is pre-resolved **app-side** before
this query runs: the tool/service reads the cart (§2.4), resolves current name/price per
`productId` from `reference` (§6), computes `lineTotal = unitPrice × quantity` per line, and
passes the finished list in. This query only persists what was already computed — no arithmetic happens in
Cypher, consistent with FR-4/FR-8 wanting the arithmetic itself to be a plain, auditable,
non-LLM step (§6 elaborates why this can't be one cross-graph query).
```cypher
// $customerId, $orderId, $now, $lines: [{productId, name, unitPrice, quantity, lineTotal}, ...]
MATCH (cust:Customer {customerId: $customerId})
OPTIONAL MATCH (dup:Order {orderId: $orderId})
WITH cust, (dup IS NULL) AS created
FOREACH (_ IN CASE WHEN created THEN [1] ELSE [] END |
  CREATE (cust)-[:PLACED]->(:Order {orderId: $orderId, status: 'placed',
                                     placedAt: $now, updatedAt: $now})
)
WITH cust, created
UNWIND (CASE WHEN $lines = [] THEN [null] ELSE $lines END) AS line
OPTIONAL MATCH (o:Order {orderId: $orderId})
FOREACH (_ IN CASE WHEN created AND line IS NOT NULL THEN [1] ELSE [] END |
  CREATE (o)-[:HAS_LINE]->(:OrderLine {productId: line.productId, name: line.name,
                                        unitPrice: line.unitPrice, quantity: line.quantity,
                                        lineTotal: line.lineTotal})
)
WITH cust, created, count(CASE WHEN line IS NOT NULL THEN 1 END) AS lineCount
OPTIONAL MATCH (cart:Cart {customerId: cust.customerId})-[:HAS_ITEM]->(item:CartItem)
FOREACH (_ IN CASE WHEN created THEN [1] ELSE [] END | DETACH DELETE item)
RETURN created, lineCount
```
Live-verified end to end against a 2-item cart: first call → `created=true, lineCount=2`, the
`Order` + 2 `OrderLine`s exist with the right snapshotted values, and the cart's 2 `CartItem`s
are gone; an immediate **retry with the same `$orderId`** → `created=false`, still exactly one
`Order` and exactly 2 `OrderLine`s (no duplicates) — the guarded-`CREATE` idempotency holds.
The `Cart` node itself is left in place, empty, ready for the next shopping session (matches
`Thread` staying in place after every message — the container never gets deleted, only its
contents).

> A `CartItem`'s own `productId` disappearing from the catalog between add-to-cart and checkout
> (a real possibility once catalog data can change) isn't addressed by any AC in the requirements
> doc — flagged as an open item in §8, not resolved here.

### 3.3 `get_order` — status + snapshot + computed total

**[verified]** — a `collect()` and a `sum()` together, both anchored on the same
uniqueness-constrained `Order`, so the "constant scalar beside an aggregate fan-out" concern in
`falkordb-quirks.md` doesn't apply (that entry's caveat is about a *non-unique* grouping key
producing more than one row per distinct value; `orderId` is unique by constraint, so there is
exactly one `o` value for the whole fan-out):
```cypher
// $orderId
MATCH (o:Order {orderId: $orderId})
OPTIONAL MATCH (o)-[:HAS_LINE]->(l:OrderLine)
RETURN o.orderId AS orderId, o.status AS status, o.placedAt AS placedAt, o.updatedAt AS updatedAt,
       collect({productId: l.productId, name: l.name, unitPrice: l.unitPrice,
                 quantity: l.quantity, lineTotal: l.lineTotal}) AS lines,
       sum(l.lineTotal) AS total
```

### 3.4 Order lifecycle — guarded CAS, one transition per call (FR-6/FR-7, AC-7/AC-8)

Same compare-and-set idiom as `resume_run`/`suspend_run` (`QUERIES.md` §12.3/§12.4): the write
commits only if the order is currently in the expected prior state; a stale/duplicate/
out-of-order transition attempt matches zero rows and writes nothing.
```cypher
// fulfill: $orderId, $now
MATCH (o:Order {orderId: $orderId})
WHERE o.status = 'placed'
SET o.status = 'fulfilled', o.updatedAt = $now
RETURN o.orderId AS orderId, o.status AS status
```
```cypher
// deliver: $orderId, $now
MATCH (o:Order {orderId: $orderId})
WHERE o.status = 'fulfilled'
SET o.status = 'delivered', o.updatedAt = $now
RETURN o.orderId AS orderId, o.status AS status
```
```cypher
// cancel: $orderId, $now — only reachable from 'placed', enforcing AC-8 ("cannot cancel once fulfilled")
MATCH (o:Order {orderId: $orderId})
WHERE o.status = 'placed'
SET o.status = 'cancelled', o.updatedAt = $now
RETURN o.orderId AS orderId, o.status AS status
```
**[verified] live plan check, and a nuance worth stating precisely.** `falkordb-quirks.md`'s
"guarded-CAS `WHERE` folds into the index scan, no residual `Filter`" entry was recorded for
`WorkflowRun`, which has **two** indexed properties (`runId` and `status`). `Order` here has only
`orderId` indexed (no index on `status` — see §7, deliberately, cardinality is tiny and nothing
scans "all orders by status"). `GRAPH.PROFILE` on the `fulfill` shape above shows:
```
Node By Index Scan | (o:Order)   ← anchors on orderId
    Filter                        ← evaluates status = 'placed'
        Update / Project
```
A residual `Filter` sits above the index scan — it does **not** fold in the way the two-indexed-
property case does. Functionally identical guarantee either way (a mismatch produces **zero
rows, nothing written** — verified: attempting `cancel` immediately after `fulfill` returns zero
rows and the order stays `fulfilled`), and at this cardinality (orders per customer, never a
scan target) the extra `Filter` costs nothing measurable. Only add a `status` index here if a
future requirement needs to scan/list orders **by** status across a workspace — no AC asks for
that today.

---

## 4. Order lifecycle as a workflow — mirroring `access-request`

FR-6/FR-9 ask the fulfillment lifecycle to be "a separate, process-kind workflow, mirroring
`access-request`'s own conversation/process split." Graph-design-relevant claim I can confirm:
**the existing `cmp` guard language is already expressive enough for this — no new primitive is
needed.** A fulfillment `WorkflowDef` (`kind: 'process'`) can use the exact same
`human`/`wait`/`decision` shapes as `ACCESS_REQUEST_DEF`, e.g. a step parked with
`waitsForHuman: true` whose outgoing transitions guard on `{kind:'cmp', path:'ctx.action',
op:'eq', value:'fulfilled'}` / `'delivered'` / `'cancelled'`.

**What this note adds to that picture — the link from the run to the Order it manages:**
```
(:WorkflowRun)-[:FULFILLS]->(:Order)
```
Created once, when the fulfillment run starts — via whichever of `start_run`
(chat-triggered) or `start_run_untriggered` (`QUERIES.md` §12.12, no chat message — likely the
right shape for an operator-initiated fulfillment run) the architect's plan settles on. New
edge type, same `UPPER_SNAKE` convention, same shape as the existing `TRIGGERED_BY`
(run→business-context edge).

**What this note deliberately does *not* decide: how a `ctx` input that advances the run also
drives the `Order.status` CAS in §3.4.** `human`/`decision`/`wait` steps have **no side effect at
all** on this engine (DESIGN §6.1) — the guard language can only read `ctx.`/`output.`, never
write to an arbitrary domain node like `Order`. So the pairing is necessarily a **service-layer**
concern: whatever endpoint accepts the operator's fulfillment input calls both
`resume_run_with_ctx` (advance the run) **and** the matching §3.4 CAS (advance the `Order`) as
two graph writes from one request — the same "two-step, accepted" shape `link_step_emission`
already uses for `StepRun`→`Message` (`QUERIES.md` §12.6: "run AFTER the §4 chat write… two-step,
accepted"). The exact trigger (a new endpoint, or the existing `POST /workflow-runs/{id}/input`
with an app-level side effect layered on) is the architect's call — I'm only confirming the
graph-side pieces (the CAS, the guard shape) compose the way FR-6 needs.

---

## 5. RAM implications (rule 6)

All five new labels (`Customer`, `Cart`, `CartItem`, `Order`, `OrderLine`) plus one relationship
type (`FULFILLS`) are **low-cardinality per workspace**: bounded by distinct chatting customers ×
distinct products ever carted/ordered, nowhere near message/thread volume. No vector properties, no
full-text index, no supernode risk — `Customer`'s fan-out is one `Cart` + a handful of `Order`s,
`Cart`'s fan-out is bounded by catalog size, `Order`'s fan-out is bounded by one checkout's line
count. Four new range indexes (`Customer.customerId`, `Cart.customerId`,
`CartItem.customerId`+`CartItem.productId`, `Order.orderId`) plus their constraints — negligible next
to the vector index RAM line that already dominates per-workspace sizing (DESIGN §11).

---

## 6. The two-graph computation (FR-4/FR-8) — why it can't be one query

**This is the load-bearing consequence of DESIGN §2's constraint #3 ("relationships cannot cross
graphs") for this capability**, worth stating plainly since it directly shapes the tool the
architect wires: `CartItem`/`Order` live in `ws:{workspaceId}`; the catalog
(`docs/requirements/workflow-catalog-lookup.md`, FR-6: "single, shared, global dataset") lives in
`reference`. There is no single Cypher query that can join a cart line to its current catalog
price — not because of a missing edge, but because **`GRAPH.QUERY` operates on one named graph
per call**, full stop. The deterministic-total mechanism is necessarily:

1. **Read** (`ws:{workspaceId}`, `GRAPH.RO_QUERY`): §2.4's `read_cart` →
   `[{productId, quantity}, ...]`.
2. **Read** (`reference`, `GRAPH.RO_QUERY`): resolve current price/name for those products —
   `UNWIND $productIds AS pid MATCH (p:Product {productId: pid}) RETURN p.productId AS productId,
   p.name AS name, p.price AS price` (label/property names confirmed against
   `docs/plans/workflow-catalog-lookup.md` §3.1's `Product` schema — `productId` is `Product`'s
   own UNIQUE-constrained key, not an assumption anymore).
3. **Compute** (app/tool code, not Cypher, not an LLM call): `total = Σ price[productId] ×
   quantity` — plain arithmetic in whatever language the tool runtime is, satisfying FR-8 ("no
   LLM/model call … solely to perform it") by construction, since nothing in this path invokes a
   model.

This is also exactly the shape `place_order` (§3.2) already uses: the app performs steps 1–3
*before* calling the write, then passes the finished `$lines` (already carrying `lineTotal`) into
one atomic write. **The graph-design claim I'm making here is narrow but firm**: whatever step
type or tool the architect designs for "compute the total," it cannot be a single graph
traversal — it is at minimum two `GRAPH.RO_QUERY` calls plus app-side arithmetic, and that's a
property of the topology (§1.2's chosen deployment shape), not an implementation shortcut.

---

## 7. Proposed DDL (consolidated, index-before-constraint)

To fold into `scripts/bootstrap_schema.sh`'s `bootstrap_workspace()`, alongside the existing
identity-anchor block:

```bash
# ── cart / order anchors (workflow-cart-and-totals-graph.md) ──
echo "[index] Customer.customerId"
gquery "$g" "CREATE INDEX FOR (n:Customer) ON (n.customerId)"

echo "[index] Cart.customerId"
gquery "$g" "CREATE INDEX FOR (n:Cart) ON (n.customerId)"

echo "[index] CartItem.customerId"
gquery "$g" "CREATE INDEX FOR (n:CartItem) ON (n.customerId)"

echo "[index] CartItem.productId"
gquery "$g" "CREATE INDEX FOR (n:CartItem) ON (n.productId)"

echo "[index] Order.orderId"
gquery "$g" "CREATE INDEX FOR (n:Order) ON (n.orderId)"

# ── constraints (after ALL indexes, same file-wide ordering rule) ──
echo "[constraint] Customer unique {customerId}"
gconstraint "$g" UNIQUE NODE Customer PROPERTIES 1 customerId

echo "[constraint] Cart unique {customerId}"
gconstraint "$g" UNIQUE NODE Cart PROPERTIES 1 customerId

echo "[constraint] CartItem unique {customerId, productId}"
gconstraint "$g" UNIQUE NODE CartItem PROPERTIES 2 customerId productId

echo "[constraint] Order unique {orderId}"
gconstraint "$g" UNIQUE NODE Order PROPERTIES 1 orderId
```
All four constraints **[verified] OPERATIONAL** (`CALL db.constraints()`, polled immediately
after creation — this build's constraints go `PENDING`→`OPERATIONAL` asynchronously, per the
standing ordering rule; at probe scale they were already `OPERATIONAL` on the very next query).
No index/constraint proposed for `OrderLine` or `Order.status` — see §3/§3.4 for why.

---

## 8. Open questions, assumptions, and things this note deliberately leaves to others

1. ~~The catalog's own key property name is an assumption~~ **Resolved 2026-08-26.**
   `docs/plans/workflow-catalog-lookup.md` §3.1 landed with `(:Product {productId, name,
   nameNormalized, category, price})` in `reference`, `productId` server-minted with a range
   index + UNIQUE constraint. This note originally assumed a `sku` join key (flagged here as
   open); every occurrence has been renamed to `productId` throughout (§0 records the
   re-verification of the one write shape — the `CartItem(customerId, productId)` composite
   constraint + its `MERGE` — that the rename touched).
2. **A cart referencing a since-deleted/renamed catalog `productId` at checkout time** — not
   addressed by any AC. §3.2 flags it inline; resolving it (skip the line? fail the checkout?
   snapshot a "no longer available" marker?) is a product/architect decision, not a graph-schema
   one.
3. **The exact trigger that pairs a fulfillment-run `ctx` resume with the `Order.status` CAS**
   (§4) is explicitly the architect's call — this note only confirms the two graph writes
   compose correctly as a "two-step, accepted" pair.
4. **`customerId = User.userId` resolution** (§1.2) assumes every chat participant in the demo is
   a real, `ensure_user`-projected workspace `User` — true for every existing falkor-chat
   surface (REST/MCP/web UI all route through message authorship), but stated explicitly since
   it's the one assumption this whole capability's durability rests on.

---

## 9. Live verification log (2026-08-26)

Disposable graph `ws_cartprobe_test` (deleted at end of session), `redis-cli` against the running
`falkordb-dev` container (`docker ps` confirmed up, port 6379 open, `PING` → `PONG`):

- DDL: 5 indexes + 4 constraints (incl. the 2-property composite on `CartItem`) created;
  `CALL db.constraints()` showed all four `OPERATIONAL`.
- `ensure_customer` (§1.3): 2 calls, `createdAt` stable at the first value, `count(Customer) = 1`.
- `ensure_cart` (§2.3): 2 calls, `count(Cart) = 1`, `count(HAS_CART) = 1`.
- `add_to_cart` (§2.1): 2 calls with `qty=2` → `quantity = 4`, `count(CartItem) = 1`; a second
  product added cleanly alongside it.
- `adjust_cart_item` (§2.2): 4→3 (update, no delete), then 3→0 (delete node + edge, other product
  untouched).
- `place_order` (§3.2): first call → `created=true, lineCount=2`, `Order`+2 `OrderLine`s with
  correct snapshotted `unitPrice`/`lineTotal`, cart's 2 items deleted; **retry with the same
  `orderId`** → `created=false`, still exactly 1 `Order` and 2 `OrderLine`s (no duplicates).
- Order lifecycle CAS (§3.4): `placed→fulfilled` succeeds; `fulfilled`-state `cancel` attempt →
  zero rows, status unchanged (AC-8's "cannot cancel once fulfilled" holds); `fulfilled→delivered`
  succeeds. `GRAPH.PROFILE` on the CAS showed `Node By Index Scan | (o:Order)` → `Filter` →
  `Update` (residual `Filter` because only `orderId` is indexed here, not `status` — see §3.4's
  discussion of why this differs from the fully-folded `WorkflowRun` CAS precedent).
- `get_order` (§3.3): `collect()` + `sum()` together over a 2-line order returned the correct
  lines array and `total = 35` (10×2 + 5×3) — separate probe graph `ws_cartprobe_test2`.
- Durable-profile `MERGE`+`SET` pattern (§1.4 / the sibling note's §1): verified in the same
  session against the same `Customer` node created by §1.3/§2's tests — see
  `docs/plans/workflow-durable-profile-graph.md` §3 for that log; recorded here too since it
  confirms cart and profile writes land on the *same* node without collision.

Both probe graphs `GRAPH.DELETE`d at the end of the session; confirmed absent from `GRAPH.LIST`.
