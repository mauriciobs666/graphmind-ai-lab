# Cart, orders, and deterministic totals — Plan Review

> **Status:** active · **Owner:** `analyst` · **Tracks:** K-053 (M6) · **Extended by:** `docs/reviews/workflow-cart-and-totals-impl.md`

## Scope & verdict

Reviewed `docs/plans/workflow-cart-and-totals.md` (`architect`) together with its companion
`docs/plans/workflow-cart-and-totals-graph.md` (`graph-dba`, `Version: 2`), against
`docs/requirements/workflow-cart-and-totals.md` (FR-1..FR-9, AC-1..AC-10), as part of the combined
M6 four-document gate. Verified the FR-8 decision's engine-fact claims directly against
`executor.py`/`guards.py`/`services.py`; did not independently re-verify `graph-dba`'s live Cypher
probes (already recorded as `[verified]` with a stated methodology and a session log in §0/§9 of
that document) but checked them for internal consistency and fit against the access-pattern
contract in the architect plan's §3.2.

**Verdict: approve with suggestions — confirmed, see Pass 2 below: approve.**

**CPG:** considered, not relevant — new-code design over the current tree; `cpg_falkorchat` is
stale (coordinator's brief) and the plan correctly reads `executor.py`/`services.py`/`guards.py`
directly instead of leaning on it.

## Findings

### MAJOR — the access-pattern contract's write ownership for `ensure_customer`/`ensure_cart` is never explicitly assigned to a service method

**Evidence:** `workflow-cart-and-totals-graph.md` §2.1 states plainly: "Zero rows ⇒ no `Cart` for
that `customerId` yet — the caller runs `ensure_customer` + `ensure_cart` (§2.3) first." The
architect plan's §3.3 lists `services.py` additions (`get_cart`, `add_cart_item`,
`remove_cart_item`, `clear_cart`, `place_order`, `get_order_status`, `advance_order`) and §4's
step-by-step, but none of the prose states which of these calls `ensure_customer`/`ensure_cart`
before the first cart mutation for a brand-new customer, nor does it appear as its own listed
method.

**Why it matters:** `add_to_cart`'s own `[verified]` Cypher (`workflow-cart-and-totals-graph.md`
§2.1) is a `MATCH (cart:Cart {customerId: $customerId})` — it returns **zero rows** and silently
does nothing if no `Cart` exists yet, which is exactly the very-first-add case for every new
customer in the demo. If an implementer builds `add_cart_item` as a thin `repository.add_to_cart`
pass-through (as §3.3's "thin, repository-delegating" description literally suggests) without
independently reading graph-dba's §2.1 caller note, the first "add to cart" a customer ever makes
in the live demo silently no-ops — a `{"found": true, ...}`-shaped success path with nothing
actually written, likely surfacing as a confusing "empty cart" bug well downstream of where the
mistake was made.

**Suggested improvement:** add one line to §3.3 or §4 assigning `ensure_customer`+`ensure_cart`
(and, by the same reasoning, `ensure_customer` alone ahead of `place_order`/order-status reads
that assume a `Customer` already exists from prior cart activity) explicitly to `add_cart_item`'s
service-layer body — e.g. "`services.add_cart_item` calls `repository.ensure_customer` +
`repository.ensure_cart` before `repository.add_to_cart`, mirroring the `ensure_user`-before-write
convention `falkor-chat/AGENTS.md` already documents elsewhere." This is a one-sentence fix to the
plan, not a design change.

### MINOR — `place_order`'s multi-step, non-atomic read-then-snapshot-then-clear sequence is flagged but left genuinely open

§6 already flags this precisely and correctly (a real race between catalog price reads, cart
reads, and the atomic snapshot write) and defers it to `graph-dba` as a possible follow-up. Given
"handling concurrent edits to the same cart" is explicitly out of scope in the requirements doc,
I agree this is acceptable to ship as-is — noting only that the *catalog price changing between
cart-read and order-write* is a narrower, real risk than the out-of-scope "two open tabs" case
(the source data itself moving mid-checkout, not a second editor), and is not explicitly named as
out of scope anywhere. Not blocking — the requirements doc's AC-6 only requires that a price
change *after* placement doesn't retroactively alter the order, which this design satisfies; a
price change *during* placement producing a snapshot from either the old or new price is
acceptable either way for a single-actor demo. Worth one sentence in §6 acknowledging this
distinction explicitly, so a future reader doesn't conflate "concurrent cart edits, out of scope"
with "a concurrent catalog price change during checkout, in scope but low-risk here."

## Cross-cutting checks (per the coordinator's brief)

- **FR-8 resolution (plain Python function, not a new step type) — verified sound and correctly
  scoped.** All three engine facts (`_execute_step`'s LLM-free dispatch for every typed handler;
  the routing-call-is-unavoidable argument; the single-`agent`-step topology making a new step
  type structurally unreachable this milestone) were independently confirmed against
  `executor.py`. The §3.1 "Propagation to the sibling plans" paragraph correctly confirms neither
  `workflow-catalog-lookup.md` nor `workflow-durable-profile.md` has an analogous determinism
  need, and correctly distinguishes `workflow-nl-query-generation.md`'s FR-3 (structural
  non-mutation) as an orthogonal axis rather than a candidate for the same mechanism — I checked
  the other three plans and confirm none of them reaches for a different mechanism for an
  analogous need. No drift found.
- **`Customer` anchor — shared correctly, not duplicated.** `workflow-cart-and-totals-graph.md`
  §1 defines the one `Customer` node (`customerId = User.userId`, workspace-local, not `identity`);
  `workflow-durable-profile-graph.md` explicitly reuses it verbatim ("This is not a second identity
  mechanism... same node, same `customerId = User.userId` resolution") and adds only two
  properties. Verified by direct read of both documents — this is the single shared mechanism the
  brief asked me to confirm, and it is.
- **`productId` naming — resolved and re-verified, not merely asserted.** The v1→v2 revision note
  and §0/§8 item 1 show the `sku`→`productId` rename was actually re-run live against a disposable
  probe graph (the composite `CartItem(customerId, productId)` constraint + its `MERGE`), not just
  edited in text. `docs/plans/workflow-catalog-lookup.md` §3.1's `Product` schema uses `productId`
  as its own UNIQUE-constrained key. Consistent end to end.

## What's solid

- The FR-8 decision (§3.1) is the strongest single piece of reasoning across the whole four-plan
  set — it correctly separates "the arithmetic is LLM-free" (true regardless of mechanism) from
  "no *additional* LLM call is made solely to compute it" (the actual AC-9 property), with a
  concrete, falsifiable trace-based verification method.
- The `Cart`/`Order` schema (graph note) makes defensible, well-justified modeling calls
  throughout: natural keys over surrogate ids for `Cart`/`CartItem` (no multi-cart requirement
  anywhere), no price/name on `CartItem` (correctly derived from FR-3's live-pricing requirement),
  full frozen snapshot on `OrderLine` (correctly derived from AC-6), guarded-`CREATE`-not-`MERGE`
  for `Order` (correctly modeled as a one-time event, mirroring `Message`'s own precedent) with a
  live-verified idempotent-replay test.
- The order-fulfillment process-def split (FR-6/FR-9) correctly mirrors `access-request@v1`'s
  existing conversation/process split rather than inventing a new pattern, and the graph note is
  appropriately explicit about what it does and does not decide (the CAS/guard shapes vs. the
  exact resume-trigger wiring, left to the architect).
- Both documents are transparent about open risk rather than hiding it (§6 in each) — the
  single-hardcoded-actor caveat is stated once and correctly not restated by the sibling profile
  plan, avoiding duplicated risk registers that could drift.

## Pass 2 — 2026-08-27 (re-gate against `workflow-cart-and-totals.md` `Version: 2`)

**Verdict: approve.**

- **MAJOR (`ensure_customer`/`ensure_cart` ownership unassigned) — fixed, and extended beyond what
  was asked.** §3.3 now explicitly assigns `services.add_cart_item` to call, in order,
  `repository.ensure_customer` → `repository.ensure_cart` → `repository.add_to_cart` — exactly
  closing the brand-new-customer silent-no-op risk. It goes one step further than my finding asked
  for: `services.place_order` also now calls `ensure_customer` defensively before
  `read_cart`/`place_order`, removing an implicit ordering assumption (that `place_order` is never
  reached without a prior `add_cart_item` having already ensured the `Customer`) that my review
  didn't independently flag but is a real, related instance of the same class of gap — good catch
  by the fix, not scope creep. §3.3 also explicitly and correctly declines to add the same calls to
  `get_cart`/`remove_cart_item`/`clear_cart`, citing graph-dba's own documented "empty cart is a
  legitimate, indistinguishable-from-emptied state" (`workflow-cart-and-totals-graph.md` §2.4,
  quoted verbatim and accurately) — I agree this is the correct boundary, not an inconsistent
  half-fix. §5's test-strategy row for AC-1/AC-2 now names a concrete regression test ("a brand-new
  `customerId` with no prior `Customer`/`Cart` node calling `add_cart_item` succeeds and the item
  is actually persisted... not a silent no-op"), and §4 step 4 flags this as needing code-review
  verification, not just a plan-text claim — appropriately cautious.
- **MINOR (`place_order` in-flight price-change race left unscoped) — fixed.** §6 now states
  explicitly: this race is "accepted, in scope, and not mitigated," is distinct from the
  requirements doc's out-of-scope concurrent-cart-edit case (source data moving mid-checkout vs.
  two editors racing the same cart), and does not violate AC-6 regardless of which price a
  mid-checkout race resolves to (AC-6 only constrains *post-placement* price changes). This is
  precisely the clarifying sentence I asked for, with correct reasoning attached.

No new finding from re-reading the full `Version: 2` document. The schema itself
(`workflow-cart-and-totals-graph.md`) is unchanged from Pass 1 (still the same `Version: 2` I
already reviewed) — this revision only changed how the architect plan names and assigns its
methods, not the schema's own modeling calls, which I already found solid.

## Open questions

None. Both findings are closed with live/code-level specificity, not just restated intent.
