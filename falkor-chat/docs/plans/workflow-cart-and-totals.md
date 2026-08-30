# Cart, orders, and deterministic totals — Implementation Plan

> **Status:** archived · **Owner:** `architect` · **Tracks:** K-053 (M6) · **Version:** 2

> **Revision note (2026-08-27).** `analyst`'s plan-gate review
> (`docs/reviews/workflow-cart-and-totals.md`, verdict: approve with suggestions) found one MAJOR
> — the access-pattern contract in §3.2 never assigned `ensure_customer`/`ensure_cart` ownership
> to a specific service method, risking a silent no-op on a brand-new customer's first
> `add_to_cart` — and one MINOR (the in-flight catalog-price-change race during `place_order` was
> flagged but not explicitly scoped). Both closed below (§3.2, §3.3, §6); no design change. This
> revision also folds in `graph-dba`'s now-landed `workflow-cart-and-totals-graph.md` (Version 2)
> concretely — §3.2/§3.3/§3.4/§4 now name its actual schema/method shapes rather than describing
> an access-pattern contract for a note that didn't exist yet when this plan was first written.

Turns `docs/requirements/workflow-cart-and-totals.md` (FR-1..FR-9, AC-1..AC-10) into an ordered,
staged build. **This document owns the FR-8 step-type-vs-tool decision** (the coordinator's brief
calls this the keystone call for the whole four-document effort) and is the canonical statement
of it — the other three plans cite this section rather than re-deciding it. The exact `Cart`/
`Order` graph schema was delegated to, and has now landed in, `graph-dba`'s
`docs/plans/workflow-cart-and-totals-graph.md` (Version 2) — this plan describes *what* must
persist and *how* the demo agent's tools use it, and (as of this revision) names that note's
actual schema/method shapes directly rather than only the access-pattern contract it was
originally handed.

**Read `docs/plans/workflow-catalog-lookup.md` first** — it is the canonical owner of the shared
`salesperson` `WorkflowDef` scaffold (the single `agent` step, the `ended`/`ctx.endConversation`
transition workaround for the pre-K-030 "≥1 transition" publish rule, and the version-bump
discipline this plan's tool additions depend on). This plan does not re-describe that scaffold.

## 1. Goal & scope

**Goal.** Give a workflow durable, workspace-scoped cart/order state (surviving across
conversations) and an exact, LLM-free computation for line-item totals and order snapshots, and
prove both inside `salesperson@v2` (this plan's version bump of the shared demo def).

**In scope:** FR-1..FR-9, AC-1..AC-10; the FR-8 mechanism decision; the cart/order tools; the
process-kind order-fulfillment def (FR-6/FR-9's split); the landed `Cart`/`Order` schema
(`graph-dba`'s note, integrated concretely as of this revision — §3.2).

**Out of scope** (per the requirements doc): cross-workspace cart persistence; payment/returns/
refunds; concurrent-edit handling; a cart/order admin CRUD API; timer-driven lifecycle
transitions; discounts/tax/rounding; cross-order aggregation/analytics; a general expression
language; rebuilding `salesperson/`.

**CPG:** considered, not relevant — same reasoning as `docs/plans/workflow-catalog-lookup.md`
§1: new-code design over the current tree, with a stale, uninvolved `cpg_falkorchat` (the
coordinator's brief already flags this; `executor.py`/`services.py`/`guards.py` were read
directly).

## 2. Context & findings

### 2.1 The comparison case (`salesperson/cart.py`)

`salesperson/cart.py` keeps cart state in an in-process `Dict[str, Tuple[CartState, float]]` TTL
store (`cart.py:34-35`) — lost on restart, never durable, exactly the gap FR-1/FR-2 close. More
relevantly for FR-8: `add_to_cart_tool`/`remove_from_cart_tool` each make a **separate LLM call**
purely to extract structured quantity/removal intent from free text
(`_extract_quantity_from_flavor`, `cart.py:112-130`, calling `llm.invoke([...])`) *before* the
arithmetic (`item["price"] * item["quantity"]`, plain Python, `cart.py:248-263`) ever runs. The
arithmetic itself was already LLM-free in `salesperson`; the extra, avoidable cost is the
*routing* LLM call this plan's tool-calling design eliminates by construction (§3.1).

### 2.2 The engine facts that decide FR-8 (`executor.py`, `guards.py`, `services.py`)

Three facts, read directly off the current tree, that the FR-8 decision below rests on:

1. **A step's arithmetic never needs a second LLM call regardless of step type.** Every typed
   step handler (`_run_decision_node`, `_run_human_node`, `_run_wait_node`, `executor.py:578-636`)
   is a plain Python `@staticmethod` with zero LLM access — `_execute_step` (`executor.py:534-574`)
   dispatches to them *before* ever touching `self._models`. A `Tool.run()` implementation
   (`tools.py`'s `Protocol`, `tools.py:72-85`) is exactly as LLM-free: it receives already-parsed
   `arguments`/`ctx`/`run` and returns a string — nothing in that call signature reaches an LLM
   client. **The engine gives no special "no LLM" guarantee to a step type that a tool's own
   Python body doesn't already have** — both are, at the point the arithmetic runs, ordinary
   Python.
2. **What a tool call *does* cost is the routing turn that decides to call it.** `_run_agent_node`
   (`executor.py:638-808`) calls `llm.chat(messages, offered)` once per iteration
   (`executor.py:737`); the model's response is either a tool call or a text answer. Getting a
   number added up via a tool therefore always costs at least the one LLM completion that decided
   "call `add_to_cart`" — this is the "still involves an LLM turn to decide to call it" fact the
   requirements doc's FR-8 prose names directly. That routing call is unavoidable for *any*
   tool-calling design (including catalog lookup, profile save, everything in this whole
   four-document effort) because natural-language intent has to be routed to *some* tool
   somehow — it is not specific to arithmetic, and eliminating it would mean not using the
   `agent`-step/tool-calling shape at all, which FR-9/all four requirements docs fix as the demo's
   shape.
3. **The demo's topology is fixed to one `agent` step** (`docs/plans/workflow-catalog-lookup.md`
   §2.4) — a new engine **step type** for deterministic computation (the other horn of the FR-8
   fork) would be **structurally unreachable** in this milestone's own def: `salesperson`'s
   topology is one `agent` step + one `decision` terminal step, and adding a third, computation-
   typed step would need its own transition into/out of the single conversational loop, which
   nothing in this demo's turn-by-turn interaction model calls for (the agent decides *when* to
   compute a total, not a fixed graph position). Building an unexercised step type here would be
   exactly the "left as an untested primitive" the requirements doc's own Intent section says this
   work must *not* be (`workflow-cart-and-totals.md` Intent: "proven via a runnable demo... rather
   than left as an untested primitive").

### 2.3 What FR-8 is actually protecting against

Given fact 2 above, "no LLM call is made solely to perform it" (AC-9) cannot mean "zero LLM calls
anywhere in the causal chain of adding an item" (that would rule out tool-calling entirely, which
contradicts FR-9). Read literally against AC-9's own wording — "no LLM/model call is made
**solely** to perform it" — the property being protected is narrower and concrete: **the
computation itself must never be the reason for, or the mechanism of, a model completion.** The
routing call that decides to invoke `add_to_cart` is not made *solely* to compute a total — it is
made to interpret the customer's request and select/parameterize a tool; the arithmetic that runs
once inside that tool costs nothing extra. What FR-8 rules out is exactly `salesperson/cart.py`'s
own second pattern from §2.1: a *dedicated, additional* LLM call whose only job is to parse or
compute something a deterministic function could do directly — whether that is free-text quantity
extraction (as in `salesperson`) or, worse, asking the model to do the multiplication itself.

## 3. Design & rationale

### 3.1 FR-8 decision: a pure computation function, called directly from tool bodies — not a
new step type

**Decision.** Add one pure, deterministic function:

```python
# server/falkorchat/pricing.py — new module, no I/O, trivially unit-testable
# (mirrors chunking.split_into_chunks's "pure function" precedent, document-ingestion.md §3.2)

def compute_line_total(items: list[dict[str, Any]]) -> float:
    """Sum price * quantity across line items. Deterministic: same inputs, same output,
    always. Malformed lines (non-numeric price/quantity, missing keys) are skipped, never
    raised on — mirrors the guard family's totality/bias-to-decline discipline
    (guards.py's `_order` wrapper) rather than failing a whole cart/order on one bad row."""
```

Every cart/order tool (`ViewCartTool`, `AddToCartTool`, `RemoveFromCartTool`, `ClearCartTool`,
`PlaceOrderTool`, §3.3) calls `pricing.compute_line_total` **directly in its own `Tool.run()`
body** — plain Python, no LLM client reachable from that call, no second completion. This is *not*
a new engine step type. Rationale, weighing the fork the requirements doc explicitly leaves open:

- **A new step type would be unreachable in this milestone** (§2.2 fact 3) — the topology is
  fixed to one `agent` step; there is no transition-graph position to put a computation step at
  that the interactive, turn-by-turn demo would ever actually drive through. Building one anyway
  would violate this exact document's own "not left as an untested primitive" framing.
- **A step type buys no LLM-freedom a tool doesn't already have** (§2.2 fact 1) — both are plain
  Python at the point the arithmetic runs. The only place the "no LLM" property could be lost is
  in a *second, avoidable* LLM call layered on top of the arithmetic (§2.3) — a discipline, not a
  step-type property, and one a tool's `run()` body enforces exactly as well as a step handler's
  would (neither has an LLM client in scope unless the implementer deliberately adds one).
- **The routing call is orthogonal to FR-8 and unavoidable either way** (§2.2 fact 2) — a future
  `kind:'process'` def with no `agent` step at all *would* need a real step-type primitive to
  reach `pricing.compute_line_total` with literally zero LLM turns in the run (no routing call to
  make, because nothing needs routing in a process flow). That need does not exist in this
  milestone (FR-9 keeps totals/snapshots inside the interactive agent, moving only *fulfillment*
  — status transitions, no arithmetic — to the process side, §3.4). If a future capability needs
  computation from inside a `kind:'process'` def, the `tool`-typed step already reserved and
  whitelisted in `services.STEP_TYPES` (`services.py:79-81`) but still an `_execute_step`
  `NotImplementedError` (`executor.py:571-574`, the documented D-E typed-handler seam) is exactly
  where it belongs — narrowly implemented for a single, closed op name (mirroring this plan's own
  closed-whitelist discipline, not a generic arbitrary-tool-from-config facility), reusing
  `pricing.compute_line_total` as its handler body. **Not built here** — flagged as the natural
  extension point if/when a process-kind def needs it, consistent with `docs/DESIGN.md` §6.1's own
  framing of `tool` as a deliberately still-open seam.
- **This is a strict improvement over `salesperson/cart.py`'s own pattern** (§2.1): removing the
  free-text quantity/removal-extraction LLM call entirely, because the tool-calling model already
  gives the tool structured `{item, quantity}` function-calling arguments directly — there is no
  free text to extract from in the first place. This is a concrete, checkable reduction in LLM
  calls per cart operation relative to the comparison case, not just a restatement of it.

**AC-9 verification.** On a debug (`trace: true`) run, the executor emits one `tool_call`/
`tool_result` trace pair per dispatched tool (`executor.py:843`/`878`) and one `llm_prompt`/
`llm_response` pair per model turn (`executor.py:733-738`). Proving AC-9 is: for a
`add_to_cart`/`place_order` turn, exactly the turn's own routing `llm_prompt`/`llm_response`
pair appears, with **no additional** `llm_prompt` entry between the `tool_call` and its
`tool_result` — the arithmetic produces no trace event of its own because it never touches
`self._models`. This is qualitatively different from — and directly falsifiable against —
`salesperson/cart.py`'s two-LLM-call shape, which the QA/architect equivalent-check AC-9 asks for
can point at as the negative example.

**Propagation to the sibling plans.** `docs/plans/workflow-catalog-lookup.md` and
`docs/plans/workflow-durable-profile.md` have no arithmetic/computation need at all (a Cypher
lookup and a two-field read/write, respectively, are already exact by construction — no
"deterministic vs. LLM" fork exists for either). `docs/plans/workflow-nl-query-generation.md`'s
own "exactness" concern is structural *non-mutation* (FR-3), a different axis than determinism-
of-arithmetic — that plan's mechanism is decided independently (its own §3). The general principle
this decision sets — *a tool's Python body does exact work directly; it never delegates exactness
to a second, avoidable LLM call* — is the one thing that does generalize, and each sibling plan's
own design already follows it without needing to import anything concrete from this section.

### 3.2 What persists — `graph-dba`'s landed schema (`workflow-cart-and-totals-graph.md`, v2)

Both shapes are workspace-scoped and keyed off a new `(:Customer {customerId, createdAt})` node
(`workflow-cart-and-totals-graph.md` §1) — `customerId = User.userId`, the same member-id
namespace `MEMBER_OF`/message authorship already use (`docs/DESIGN.md` §1.2), resolved once by
the tool-calling layer from `CallContext.actor` (`config.py:125-133`). This is the "workspace-local
shortcut" both this document's own decision log and `workflow-durable-profile.md` deliberately
reuse instead of resolving the open `identity`-graph-write question — **not** a new identity
mechanism, and (per graph-dba's §1.4) the same `Customer` node `workflow-durable-profile.md`'s two
profile fields land on.

```
(:Customer {customerId, createdAt})
(:Customer)-[:HAS_CART]->(:Cart {customerId, createdAt, updatedAt})
(:Cart)-[:HAS_ITEM]->(:CartItem {customerId, productId, quantity, addedAt, updatedAt})
(:Customer)-[:PLACED]->(:Order {orderId, status, placedAt, updatedAt})
(:Order)-[:HAS_LINE]->(:OrderLine {productId, name, unitPrice, quantity, lineTotal})
```

- **`Cart`/`CartItem` carry no price** — keyed on `customerId` and the composite
  `(customerId, productId)` respectively (natural keys, no synthetic `cartId`/`cartItemId` —
  §2 of the graph note). FR-3's live-pricing requirement is why: a stored cart-line price would
  go stale exactly when FR-3 says it must not, so the current price is always resolved fresh from
  `reference` at read/checkout time (graph note §6 — the two-graph read is structurally
  unavoidable there, not an implementation shortcut).
- **`Order`/`OrderLine` carry a full frozen snapshot** (`name`, `unitPrice`, `quantity`,
  `lineTotal`) — AC-6's requirement, satisfied because `place_order` (below) computes and persists
  the snapshot once, atomically, and nothing ever re-derives it from a live catalog read again.
  `Order.total` is deliberately **not** stored (computed as `sum(OrderLine.lineTotal)` on read,
  graph note §3) — per-order line count is bounded and tiny, so this avoids a write-time-only
  derived field that could drift from its own source lines.
- **The repository methods this schema backs** (graph note §1.3, §2.1-§2.5, §3.1-§3.4), named
  here because §3.3 below assigns each to a specific service-layer caller: `ensure_customer`
  (idempotent get-or-create), `ensure_cart` (idempotent get-or-create, mirrors `ensure_customer`),
  `add_to_cart` (MERGE-and-increment), `adjust_cart_item` (guarded decrement-or-remove),
  `read_cart`, `clear_cart`, `place_order` (one atomic snapshot-and-clear write, idempotent on a
  caller-minted `orderId`), `get_order` (status + snapshot + computed total), and the three
  lifecycle CAS writes (`fulfill`/`deliver`/`cancel`, guarded exactly like `resume_run`/
  `suspend_run`).
- **`(:WorkflowRun)-[:FULFILLS]->(:Order)`** (graph note §4) — the edge the order-fulfillment
  process def (§3.4) uses to find the `Order` it's managing; created once when that run starts.

### 3.3 Cart/order tools (FR-1..FR-7)

New `Tool` classes in `tools.py` (peers of the catalog-lookup tools, same registration
mechanism): `ViewCartTool` (`view_cart`, no args — returns items + a live-computed total),
`AddToCartTool` (`add_to_cart`, `{productName, quantity}` — resolves the product via
`services.lookup_product` — the *same* catalog lookup `workflow-catalog-lookup.md` §3.5 already
built, reused rather than duplicated — then writes the cart line), `RemoveFromCartTool`
(`remove_from_cart`, `{productName, quantity?}` — omitted quantity removes the line entirely),
`ClearCartTool` (`clear_cart`, no args), `PlaceOrderTool` (`place_order`, no args — reads the
cart, resolves each line's *current* price via `lookup_product`, calls
`pricing.compute_line_total` for the frozen total, writes the `Order` snapshot, clears the cart —
one `services.place_order(ctx)` call covering all of it, so the tool body itself stays a thin
dispatcher like every other tool here).

Every tool's abstention/error shape follows the established idiom: `add_to_cart` on an unknown
product name returns `{"found": false}` (mirroring `LookupProductFactTool`, since it calls the
same underlying lookup); `place_order` on an empty cart returns an explanatory string rather than
creating a zero-line order.

**`services.py` additions, with explicit `ensure_customer`/`ensure_cart` ownership (closing
`analyst`'s MAJOR finding — `docs/reviews/workflow-cart-and-totals.md`).**
`graph-dba`'s `add_to_cart` Cypher (`workflow-cart-and-totals-graph.md` §2.1) is a
`MATCH (cart:Cart {customerId: $customerId})` that returns **zero rows and writes nothing** if no
`Cart` exists yet for that customer — exactly the very-first-add case for every brand-new customer
in the live demo. So:

- **`services.add_cart_item(ctx, *, product_name, quantity) -> dict`** calls, in order,
  `repository.ensure_customer(customer_id)` → `repository.ensure_cart(customer_id)` →
  `repository.add_to_cart(...)`. Both `ensure_*` calls are idempotent `MERGE`s (graph note §1.3/
  §2.3), so this costs two cheap extra round trips **only** the first time a given customer ever
  adds anything, and is a correctness no-op (but not a performance concern) on every later call —
  this is the mirror of the `ensure_user`-before-write convention `falkor-chat/AGENTS.md` already
  documents elsewhere, applied to `Customer`/`Cart` the same way.
- **`services.place_order(ctx) -> dict`** also calls `repository.ensure_customer(customer_id)`
  before `repository.read_cart`/`repository.place_order` — defensively, not because any real path
  reaches `place_order` without a prior `add_cart_item` having already ensured the `Customer` (an
  order can only be non-empty because items were added first), but because `place_order`'s own
  Cypher likewise `MATCH`es `(cust:Customer {customerId: $customerId})` and would otherwise depend
  on an implicit ordering assumption the analyst review specifically flagged as fragile. Cheap
  (one idempotent `MERGE`) and removes that assumption entirely.
- **`services.get_cart(ctx) -> dict`, `remove_cart_item(ctx, *, product_name, quantity) -> dict`,
  `clear_cart(ctx) -> None` do *not* call `ensure_customer`/`ensure_cart`.** A read or a removal
  against a `Customer`/`Cart` that doesn't exist yet is graph-dba's own documented, legitimate
  "empty cart" case (`workflow-cart-and-totals-graph.md` §2.4: "Zero rows is a legitimate 'empty
  cart' [state]... FR-1's 'view the cart' and 'clear it entirely' don't need to distinguish 'never
  touched' from 'emptied'") — calling `ensure_*` here would silently create `Customer`/`Cart` nodes
  for a customer who has never added anything, which is unnecessary write traffic for a read path
  and no more correct than the zero-rows answer these methods already return. Stated explicitly so
  a future implementer doesn't over-generalize the fix above onto these three methods.
- `get_order_status(ctx, *, order_id) -> str | None` and
  `advance_order(ctx, *, order_id, transition) -> dict` operate on an already-placed `Order` by
  `orderId`, never on `Customer`, so no `ensure_*` call applies to either (used by the fulfillment
  process def, §3.4, not by any agent-step tool — no tool grants order-fulfillment actions, since
  FR-9 keeps that on the human/operator process side).

### 3.4 Order fulfillment as a separate process-kind def (FR-6, FR-9)

Mirrors `access-request@v1` exactly (`server/falkorchat/proof_defs.py:59-154`) — a `kind:'process'`
def of `human`/`decision`/`wait` steps, no `agent` step, no LLM, no network. New constant
`ORDER_FULFILLMENT_DEF` in `proof_defs.py`:

```
placed (start, decision — no side effect, entered once an Order node exists)
  -> [cmp: ctx.action == "fulfill"]  -> fulfilled (human, waitsForHuman, assignee: "operator")
  -> [cmp: ctx.action == "cancel"]   -> cancelled (decision, terminal)
fulfilled
  -> [cmp: ctx.action == "deliver"]  -> delivered (decision, terminal)
```

This flow does not itself compute anything (no FR-4 mechanism involved) — it only advances
`Order.status` via `services.advance_order` (each transition backed by one of graph-dba's three
guarded-CAS writes, §3.2 — `fulfill`/`deliver`/`cancel`, each matching only from the correct prior
`status` so a stale/duplicate/out-of-order attempt matches zero rows and writes nothing), called
from a `human` step's resume path exactly as `access-request@v1`'s `approval`/`provision` steps
already do. It is triggered with `trigger_msg_id=None` (`POST /workflow-runs`, `docs/SERVER.md`
§1.4 — "start a run from a snapshot with no chat trigger") once an order is placed, carrying
`ctx.orderId`, and creates the `(:WorkflowRun)-[:FULFILLS]->(:Order)` edge (graph note §4) at
start — mirroring how `access-request`'s own trigger has no chat message behind it. Per the graph
note's own explicit non-decision (§4): pairing a `ctx` input that advances the run with the
matching `Order.status` CAS is a service-layer concern (two graph writes from one request, the
same "two-step, accepted" shape `link_step_emission` already uses) — `services.advance_order` is
where this plan places that pairing, called from whichever endpoint accepts the operator's
fulfillment input alongside `resume_run_with_ctx`. Seeded via the **same**
`scripts/seed_salesperson.sh` this plan extends (§4), alongside the `salesperson@v2` def bump —
one script publishing/materializing two defs, exactly as `seed_workflows.sh` already does for
`triage`+`access-request`.

## 4. Step-by-step implementation

Builds on `docs/plans/workflow-catalog-lookup.md`'s already-landed scaffold (§4 there, steps 1-9).

1. **`graph-dba`'s `workflow-cart-and-totals-graph.md` (Version 2) has landed** — DDL + verified
   Cypher for `Customer`/`Cart`/`CartItem`/`Order`/`OrderLine` (§3.2). Nothing further to wait on;
   proceed directly to step 3.
2. **`server/falkorchat/pricing.py`** — `compute_line_total` (§3.1), pure, unit-tested first
   (no dependency on anything else in this list — buildable/testable immediately).
3. **`server/falkorchat/repository.py`** — `ensure_customer`, `ensure_cart`, `add_to_cart`,
   `adjust_cart_item`, `read_cart`, `clear_cart`, `place_order`, `get_order`, and the three
   lifecycle CAS writes (`fulfill`/`deliver`/`cancel`) — one method per Cypher shape in
   `workflow-cart-and-totals-graph.md` §1.3, §2.1-§2.5, §3.1-§3.4 (already live-verified there;
   this step is a direct transcription, not new query design).
4. **`server/falkorchat/services.py`** — the methods listed in §3.3, each thin over step 3 +
   `lookup_product` (reused from `workflow-catalog-lookup.md`) + `pricing.compute_line_total` —
   **with `add_cart_item` and `place_order` each calling `ensure_customer`/`ensure_cart` first**,
   exactly as §3.3 now specifies (this is the MAJOR fix's concrete landing point — verify it's
   actually wired this way in code review, not only stated in this plan).
5. **`server/falkorchat/tools.py`** — the five cart/order tools (§3.3); register into the same
   salesperson tool registry `workflow-catalog-lookup.md` built.
6. **`server/falkorchat/proof_defs.py`** — bump `SALESPERSON_DEF["version"]` to `"v2"`, extend
   `config.tools` with `["view_cart", "add_to_cart", "remove_from_cart", "clear_cart",
   "place_order"]`, extend `systemPrompt` with cart/checkout guidance; add
   `ORDER_FULFILLMENT_DEF` (§3.4).
7. **`scripts/seed_salesperson.sh`** — publish/materialize `salesperson@v2` **and**
   `order-fulfillment@v1` (both, one script run, mirroring `seed_workflows.sh`'s two-def
   pattern); **`scripts/verify_salesperson.sh`** — extend to check both defs, mirroring
   `verify_workflows.sh`'s two-def check.
8. **`docs/QUERIES.md`** / **`scripts/test_queries.sh`** — new Cart/Order query entries + baseline
   bump, per `graph-dba`'s Cypher.
9. **`falkor-chat/AGENTS.md`** — no new scripts this stage (existing ones extended in place); note
   the `Cart`/`Order` labels in the schema-conventions section if `graph-dba`'s note doesn't
   already land that update itself.

**Done (this plan):** `salesperson@v2` proves FR-1..FR-5 live (add/remove/view/clear a cart with
a correct running total, catalog-current prices, placing an order snapshots correctly); a
separately-triggered `order-fulfillment@v1` run proves FR-6/FR-7 (explicit-step-only lifecycle
advance, queryable status, no automatic progression) — both seeded/verified the same way
`triage`/`access-request` are today (AC-10).

## 5. Test strategy

| AC | What proves it | Altitude |
|---|---|---|
| AC-1, AC-2 (add/remove/clear, correct total) | `pricing.compute_line_total` pure unit tests (empty, one line, many lines, a malformed line skipped not raised); `AddToCartTool`/`RemoveFromCartTool`/`ClearCartTool` against a fake `Services`; live `@mention` add→view→remove→view sequence; **regression test for `analyst`'s MAJOR finding: a brand-new `customerId` with no prior `Customer`/`Cart` node calling `add_cart_item` succeeds and the item is actually persisted (`get_cart` reflects it), not a silent no-op** | pure unit + tool unit + live e2e |
| AC-3 (cross-conversation persistence) | Two separate `Thread`s, same `ctx.actor`, same `ws`; second thread's `view_cart` sees the first thread's items | service/repository integration (mirrors AC-3's own wording almost exactly) — note the single-hardcoded-actor caveat in §6 |
| AC-4 (exact, repeatable total) | `compute_line_total` called twice with identical inputs asserts identical output; a hand-calculated fixture cross-check | pure unit |
| AC-5, AC-6 (order snapshot, price-change-proof) | `place_order` then mutate the source `Product.price`; re-read the `Order` and assert its snapshot price is unchanged; assert the source cart is empty afterward | service/repository integration |
| AC-7, AC-8 (lifecycle, no auto-advance) | `order-fulfillment@v1` driven through `fulfilled`→`delivered` via explicit `human` resumes, mirroring `test_process_flow.py`'s own pattern; a parked run's status never changes absent an explicit resume; a `cancel` before fulfillment blocks a later `deliver` | offline acceptance test, mirroring `test_process_flow.py` |
| AC-9 (no LLM call solely for the computation) | Debug-trace inspection per §3.1's verification method: exactly one `llm_prompt`/`llm_response` pair around each cart-mutating `tool_call`, no extra pair between `tool_call` and `tool_result` | trace-inspection test (new, `test_executor_agent.py`-adjacent) |
| AC-10 (seed/verify) | `seed_catalog.sh`/`seed_salesperson.sh`/`verify_salesperson.sh` all green | script-level |

## 6. Risks & open questions

- **Resolved: `Cart`/`Order` schema landed and reviewed** (`docs/plans/workflow-cart-and-totals-graph.md`,
  Version 2) — §3.2/§3.3/§3.4 above now name its actual shapes. `analyst`'s plan-gate review
  (`docs/reviews/workflow-cart-and-totals.md`) confirmed the schema's modeling calls (natural keys
  for `Cart`/`CartItem`, no price on `CartItem`, full frozen snapshot on `OrderLine`, guarded-
  `CREATE`-not-`MERGE` for `Order`) and found one MAJOR gap in this plan's own write-path
  specification (not in the schema itself): the access-pattern contract never assigned
  `ensure_customer`/`ensure_cart` ownership to a service method, risking a silent no-op on a
  brand-new customer's first `add_to_cart`. Closed in §3.3 (`services.add_cart_item`/`place_order`
  now explicitly call `ensure_customer`/`ensure_cart` first) and tested for in §5.
- **`place_order`'s in-flight catalog-price-change race — scoped explicitly (`analyst`'s MINOR
  finding).** §3.2's two-graph read (cart lines from `ws:{id}`, current prices from `reference`,
  §6 of the graph note) is not atomic with the snapshot write: a catalog price change landing
  *between* the price read and the `place_order` write could produce a snapshot from either the
  old or the new price. **This is accepted, in scope, and not mitigated in this plan** — distinct
  from, and narrower than, the "handling concurrent edits to the same cart" bullet the requirements
  doc explicitly puts out of scope (that bullet is about two editors racing the same cart; this is
  the *source catalog data* moving mid-checkout, a different and much rarer real-world case for a
  single-actor demo). AC-6 only requires that a price change *after* placement never retroactively
  alters the order, which this design satisfies regardless of which price a mid-checkout race
  picks — so accepting either outcome of the race (old price or new price snapshotted) does not
  violate any AC. Not conflated with the out-of-scope concurrent-cart-edit case; if a future need
  requires closing this race too, folding the price read into `place_order`'s own atomic write
  (impossible today only because `reference`/`ws:{id}` can't be joined by one query, graph note §6)
  is the natural next design question, not resolved here.
- **Single-hardcoded-actor caveat (M1's real limitation, not this plan's).** `docs/SERVER.md`
  §1.3: M1's auth seam resolves every call to one hardcoded actor (`u1`). AC-3 ("the *same*
  customer's cart persists across conversations") is fully provable today with two `Thread`s under
  that one actor — but a *second, distinct* customer's isolation (their cart never leaking into
  the first customer's) is **not independently demonstrable** until real per-user auth (K-016,
  M2.5, still deferred) lands. This mirrors exactly the same caveat `workflow-durable-profile.md`
  would need and is stated once here as the shared fact — not a blocker to this milestone (the
  requirements doc's own AC-3 only asks for the same-customer case), but worth recording so nobody
  later reads "demo-proven" as "multi-tenant-proven."
- **The `tool`-step-type extension point (§3.1) is explicitly not built** — flagged for a future
  capability, not this one. If a future process-kind def needs LLM-free computation with zero
  routing calls at all, implement it narrowly against the closed op-name whitelist this plan
  establishes (starting with `"sum_line_items"` calling `pricing.compute_line_total`), not as a
  generic arbitrary-tool-from-config facility.
- **Resolved/superseded by the bullet above:** this plan's first draft flagged `place_order`'s
  per-line price re-resolution as an open question for `graph-dba`'s note to settle. The note has
  since landed (§3.2/§6 there): the two-graph read is confirmed structurally unavoidable (not an
  implementation shortcut — `GRAPH.QUERY` operates on one named graph per call, and `Cart`/`Order`
  live in `ws:{id}` while the catalog lives in `reference`), and the resulting race is scoped
  explicitly two bullets up, not left open.
