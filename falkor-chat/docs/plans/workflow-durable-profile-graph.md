# Durable user-profile data for workflows — Graph Design

> **Status:** active · **Owner:** `graph-dba` · **Tracks:** — (M<n> TBD) · **Version:** 2

Graph-side design note for `docs/requirements/workflow-durable-profile.md` (FR-1..FR-4,
AC-1..AC-3). Schema and write/read-path Cypher for a durable, workspace-scoped, **updatable**
customer name/delivery-address record. An `architect` agent is producing the step/tool
integration plan for the same requirements document in parallel; this note owns the graph shape
and the exact Cypher.

*Revision note — 2026-08-27 (v1 → v2).* `analyst`'s plan gate raised a BLOCKER: v1's
`write_profile` used an unconditional `SET` (mirroring `write_model_overrides`'s NULL-clears
semantics), which is wrong for `SaveProfileTool`'s calling convention where arguments are
genuinely optional — a partial update would have silently nulled an already-stored field,
defeating AC-2. §3 now uses `coalesce($field, c.field)` per field instead; the omitted-field-
survives case is live-verified (§0). No schema change, no change to §1/§2/§4/§6/§7.

**This is not a second identity mechanism.** It reuses, unchanged, the `Customer` anchor node
defined in `docs/plans/workflow-cart-and-totals-graph.md` §1 — same node, same `customerId =
User.userId` resolution, same reasoning for why it isn't `User` itself and isn't scoped by a
`workspaceId` property. Read that section before this one; it is not re-derived here. This
document's only original content is the two profile properties and their write/read path.

---

## 0. Verification standing

The write/read pattern below was executed live on **2026-08-26** against the disposable graph
`ws_cartprobe_test` (`falkordb/falkordb:v4.18.11`, module `41811`, via `redis-cli` against the
running `falkordb-dev` container) — the **same** `Customer` node the cart-and-totals note's tests
created, specifically to confirm both capabilities' writes land on one node without collision,
not two competing ones. Graph deleted at the end of the session. See
`workflow-cart-and-totals-graph.md` §9 for the shared verification log; §3 below repeats only the
profile-specific evidence.

**v2 re-verification (2026-08-27)**, against a fresh disposable graph
`ws_cartprobe_profile_check2` (module `41811`, `redis-cli` against the same running
`falkordb-dev`, deleted at the end): `CREATE INDEX FOR (n:Customer) ON (n.customerId)` +
`GRAPH.CONSTRAINT CREATE ... UNIQUE NODE Customer PROPERTIES 1 customerId` (same DDL as
`workflow-cart-and-totals-graph.md` §7), then the exact §3 sequence:
1. Full write → `name='Alice', deliveryAddress='123 Main St', profileUpdatedAt=1000`.
2. Corrected `write_profile` called with `$name=NULL, $deliveryAddress='456 New Ave', $now=2000`
   → returned row shows `name='Alice'` (preserved), `deliveryAddress='456 New Ave'` (updated).
   Read-back confirms the same, and `count(Customer) = 1`.
3. Corrected `write_profile` called with `$name='Bob', $deliveryAddress=NULL, $now=3000` (the
   symmetric case) → `name='Bob'` (updated), `deliveryAddress='456 New Ave'` (preserved,
   **not** nulled). `count(Customer) = 1` held throughout both partial calls — no duplicate node
   from either `MERGE`.

This is the exact failure `analyst` flagged (an omitted field surviving a partial update,
in both field directions), now closed and live-proven rather than asserted.

---

## 1. Why one node, not two — and why not `identity`

`docs/requirements/workflow-durable-profile.md`'s own decision log flags the real fork: whether
`identity` (the global, read-mostly auth graph) should start accepting writes, versus a
workspace-local shortcut. The requirements doc **deliberately does not resolve that** and takes
the shortcut — same workspace-scoping `workflow-cart-and-totals.md` uses for the cart. This note
follows that instruction to the letter: **nothing here touches `identity`**, and nothing here
adds a write path to the identity graph's future. `Customer` lives entirely inside
`ws:{workspaceId}`, exactly like `Cart`/`Order`.

Given that shortcut, using the **same** `Customer` node the cart already anchors to (rather than
a second `Profile`/`Contact` node) is not just convenient — it's the more defensible model:
- Both capabilities want the identical answer to "which durable record is this customer's, in
  this workspace" — inventing a second node with its own key would mean either duplicating
  `customerId` resolution logic or introducing a join between two workspace-local nodes that
  represent the same real-world entity, for no modeling benefit.
- The requirements doc's own out-of-scope list rules out exactly the case that would justify
  splitting them: "an extensible/open-ended profile schema" is explicitly not wanted. Two fixed
  scalar fields (`name`, `deliveryAddress`) with no edges and no independent lifecycle are a
  textbook case for "property on the existing anchor," not a new label — per the general
  modeling principle: *if a property is a shared join target, question whether it's a node; if a
  relationship carries no traversal meaning, question whether it's a property.* Neither `name`
  nor `deliveryAddress` is ever a traversal target here.
- A future direction the stakeholder named but explicitly deferred — "automatically attaching a
  saved profile to a placed order" — is *cheaper*, not harder, with one node: `Order` already
  has `(:Customer)-[:PLACED]->(:Order)`, so a future feature reads `name`/`deliveryAddress`
  straight off the same `Customer` an `Order` is already anchored to, no new edge required.

## 2. Schema — two more properties on the existing node

```
(:Customer {customerId, createdAt,
             name, deliveryAddress, profileUpdatedAt})   // name/deliveryAddress: NULLable until first write
```
No new label, no new relationship, no new index, no new constraint — `Customer.customerId`'s
existing uniqueness constraint (`workflow-cart-and-totals-graph.md` §7) already backs every
`MERGE` this note performs. `profileUpdatedAt` is separate from `Customer.createdAt` (customer
identity can exist — e.g. from cart activity — before any profile fact is ever captured; FR-1's
"first time they're given" is about the profile fields specifically, not the `Customer` node's
own lifetime).

## 3. `write_profile` — MERGE + coalesce()-guarded SET, update-in-place (not create-once)

**[verified]** — this is deliberately **not** the `ensure_user`/`ensure_customer` "guarded
create, never updates" shape (`workflow-cart-and-totals-graph.md` §1.3): FR-3 requires the
opposite — "if a customer provides updated name/address information later, the stored profile is
updated, not frozen after the first write."

**v2 correction (2026-08-27): not an unconditional `SET`.** v1 mirrored `write_model_overrides`
(`QUERIES.md` §13.1) literally — `MERGE` + unconditional `SET`, where a `NULL` parameter clears
the property. That precedent is correct *for that caller* (`write_model_overrides`'s own calling
convention treats `NULL` as "clear this override," an explicit, deliberate signal). It is
**wrong here**: `analyst`'s plan gate caught that `SaveProfileTool`'s calling convention is
different — its arguments are genuinely *optional*, so a partial update (the customer only gives
an updated address) would pass `$name = NULL` meaning "not provided," and an unconditional `SET`
would silently null out the already-stored name — exactly the failure AC-2 exists to rule out.
The fix is `coalesce()` per field, so a `NULL` argument means "leave this field as it is," never
"clear it":
```cypher
// $customerId, $name, $deliveryAddress, $now
MERGE (c:Customer {customerId: $customerId})
ON CREATE SET c.createdAt = $now
SET c.name            = coalesce($name, c.name),
    c.deliveryAddress = coalesce($deliveryAddress, c.deliveryAddress),
    c.profileUpdatedAt = $now
RETURN c.customerId AS customerId, c.name AS name, c.deliveryAddress AS deliveryAddress
```
`profileUpdatedAt` stays an unconditional `SET` — "this profile was touched" is true on *any*
call that reaches this query, partial or full, so it isn't a field this note is trying to
preserve.

**[verified, 2026-08-27]** — the case that matters is a partial update *after* an initial full
write: does the omitted field survive unchanged, not get overwritten with the stale value the
caller happened to still be holding, and not get nulled. Run against a disposable probe graph
(`ws_cartprobe_profile_check2`, deleted after) that reproduces the earlier session's node state
one property write at a time:
1. `MERGE (c:Customer {customerId:'cust-1'}) SET c.name='Alice', c.deliveryAddress='123 Main St', c.profileUpdatedAt=1000` — full initial write.
2. The corrected query above, called with `$name = NULL`, `$deliveryAddress = '456 New Ave'`,
   `$now = 2000` — a partial update supplying only the address, exactly `SaveProfileTool`'s
   "customer only gave an updated address" case.
3. Read back: `name = 'Alice'` (**unchanged**, not nulled — this is the bug the fix closes),
   `deliveryAddress = '456 New Ave'` (updated), `profileUpdatedAt = 2000` (bumped on the partial
   call too), `count(Customer) = 1` throughout (still one node, no duplicate from the `MERGE`).
4. A second partial call, `$name = 'Bob'`, `$deliveryAddress = NULL`, `$now = 3000`, confirms the
   symmetric case: `name = 'Bob'` (updated), `deliveryAddress = '456 New Ave'` (**unchanged**,
   confirming the fix isn't one-field-only), `profileUpdatedAt = 3000`.

Live output for both partial calls and the final read is in §0's verification log.

**Two caller-facing notes, revised for the `coalesce()` shape (superseding v1's, which assumed
the `write_model_overrides` clear-on-`NULL` semantics that no longer apply here):**
- **`NULL` now means "not provided, leave unchanged" — there is no way to *clear* a field
  through this query.** That's deliberate: nothing in FR-1..FR-4 asks for clearing a
  previously-captured name/address, only for updating one, and the requirements doc's own
  out-of-scope list rules out anything more elaborate than the two fixed fields. If a future
  requirement needs "the customer wants their stored address removed," that's a new, explicit
  write shape (e.g. a sentinel value, or a separate `clear_profile_field` query) — not something
  this note's `write_profile` should grow a silent second meaning for.
- **Never write `''` to mean "no value."** An empty string is a value ("a name literally
  empty"), not "not provided" — `NULL`/argument-omission is the only representation of "the
  customer didn't say." Not enforced in Cypher — a caller discipline for whichever tool wires
  `SaveProfileTool`'s arguments into `$name`/`$deliveryAddress`.

## 4. `read_profile` — the one code path for "unset"

```cypher
// $customerId
MATCH (c:Customer {customerId: $customerId})
RETURN c.name AS name, c.deliveryAddress AS deliveryAddress, c.profileUpdatedAt AS profileUpdatedAt
```
Zero rows ⇒ no `Customer` node at all yet for this id (nobody has ever interacted with this
workspace as this customer — cart, order, or profile). A `Customer` row with `name`/
`deliveryAddress` both `NULL` ⇒ the customer exists (e.g. from cart activity) but has never given
a name/address. Both cases mean "ask them" to the calling agent — the distinction only matters if
a future feature needs to tell "brand-new customer" from "returning customer, no profile yet"
apart, which nothing in this requirements doc asks for.

## 5. AC coverage, stated plainly

- **AC-1** (FR-1/FR-2 — ask once, don't ask again across conversations): `read_profile` keyed on
  `customerId` (not thread/run) returns the same row regardless of which conversation calls it,
  because `Customer` was never thread-scoped to begin with (§1's whole point).
- **AC-2** (FR-3 — update, don't freeze): §3's `coalesce()`-per-field `SET` applies every call —
  a value takes effect when provided, an omitted (`NULL`) field falls through to its current
  stored value — live-verified in both field directions (§0) to update what's given and leave
  everything else exactly as it was, never nulling an unrelated field on a partial update.
- **AC-3** (FR-4 — proven inside the combined demo agent): no graph-side implication — this note
  puts no constraint on which workflow/tool calls §3/§4; that wiring is the architect's plan.

## 6. RAM implications

Zero new labels, zero new relationships, zero new indexes/constraints — two nullable scalar
properties on a node `workflow-cart-and-totals-graph.md` §5 already priced in as negligible. This
capability adds no measurable RAM line of its own.

## 7. Open questions / assumptions

None beyond what `workflow-cart-and-totals-graph.md` §8 already states about `customerId`
resolution — this note inherits that assumption wholesale (same node, same resolution rule) and
does not add a new one.
