# Durable user-profile data for workflows — Implementation Plan

> **Status:** archived · **Owner:** `architect` · **Tracks:** K-054 (M6)

Turns `docs/requirements/workflow-durable-profile.md` (FR-1..FR-4, AC-1..AC-3) into an ordered,
staged build. The exact `Profile` graph schema is delegated to `graph-dba`'s
`docs/plans/workflow-durable-profile-graph.md`; this plan describes what must persist and how the
demo agent's tools use it. **Read `docs/plans/workflow-catalog-lookup.md` first** (the shared
`salesperson` `WorkflowDef` scaffold) and `docs/plans/workflow-cart-and-totals.md` §3.1/§3.2 (the
FR-8 computation decision and the identity-anchoring convention this plan reuses verbatim) —
neither is re-described here.

## 1. Goal & scope

**Goal.** Durably record a customer's name and delivery address, scoped to one workspace, so a
later, separate conversation with the same customer in the same workspace doesn't ask again — and
prove it inside `salesperson@v3` (this plan's version bump of the shared demo def).

**In scope:** FR-1..FR-4, AC-1..AC-3; the profile read/write tools; the access-pattern contract
for `graph-dba`'s schema note.

**Out of scope** (per the requirements doc): cross-workspace profile persistence; writing to the
`identity` graph; auto-attaching a saved profile to a placed order; an extensible/open-ended
profile schema; a standalone demo workflow.

**CPG:** considered, not relevant — same reasoning as the two prior plans (new-code design, stale
and uninvolved `cpg_falkorchat`, `services.py`/`repository.py` read directly).

## 2. Context & findings

### 2.1 The comparison case (`salesperson/customer_profile.py`)

`salesperson/customer_profile.py` keeps profile state in the same in-process TTL-store pattern as
`cart.py` (`_profile_store`, `customer_profile.py:26-27`) — lost on restart, never durable, and
explicitly session-keyed rather than customer-identity-keyed (a session is a browser/chat session,
not a durable person). This is precisely the gap FR-1/FR-2 close. Nothing in `customer_profile.py`
does computation or LLM extraction (unlike `cart.py`'s quantity-parsing pattern) — its `info_stage`
state-machine logic (`_create_default_profile`, `handle_cart_changed`) is a UI/flow concern
specific to `salesperson`'s own chatbot loop and does not carry over; this plan only needs the two
durable fields (`customer_name`, `delivery_address`) and the persistence gap around them.

### 2.2 Identity anchoring — reuses `workflow-cart-and-totals.md` §3.2 verbatim

Same workspace-local shortcut, same reason: "the harder version of this problem — whether
`identity`... should start accepting writes, or whether per-workspace snapshots are the right
answer instead — was flagged by both the architect and graph-dba as a genuinely unresolved design
question. This document deliberately does not resolve it" (`workflow-durable-profile.md`,
Problem & current state). A `Profile` is keyed off `(ws, CallContext.actor)`, the exact same
member-id namespace the `Cart`/`Order` shapes use — one identity-anchoring decision, stated once
in `workflow-cart-and-totals.md` §3.2, reused here without restatement. The same single-
hardcoded-actor caveat (`docs/SERVER.md` §1.3, M1's auth seam) applies identically and is not
restated (see that plan's §6).

### 2.3 The write-path idiom to reuse: `ensure_user`/`ensure_agent`, not a bare `MERGE`

`docs/DESIGN.md` §1.2 documents the existing convention for "does this identity-keyed row already
exist, and if so update it" writes: `ensure_user`/`ensure_agent` "are v2 guarded-CREATE queries
returning `(created, existed, collided)`" (`docs/QUERIES.md` §2/§7). A `Profile` write
(FR-3: "if a customer provides updated information later, the stored profile is updated, not
frozen after the first write") is the same shape — find-or-create-or-update, single atomic query,
never a read-then-write pair (which would race two profile updates in flight the same way a naive
cart read-then-write would, `workflow-cart-and-totals.md` §6's flagged risk). This plan recommends
`graph-dba` model `upsert_profile` on this exact idiom rather than inventing a new one — a
`MERGE (p:Profile {profileId: ...}) ON CREATE SET ... ON MATCH SET ...`-shaped guarded write,
returning a status row the same way the member-ensure queries already do, is the natural fit and
requires no new mechanism this codebase doesn't already have proven.

## 3. Design & rationale

### 3.1 What must persist (the access-pattern contract for `graph-dba`)

One durable shape: a `Profile`, one per `(workspace, customerId)`, with exactly two data fields —
`name` (string), `deliveryAddress` (string) — both nullable independently (FR-1's "durably write
**and** later read a customer's name and delivery address" does not specify both must arrive
together; a workflow author may capture one before the other). Needs: point lookup by
`(workspaceId-implicit-via-graph-key, customerId)` returning both fields (or "no profile yet");
an upsert that sets whichever of the two fields the caller supplies, leaving the other unchanged
if omitted (so a later "just update the address" call doesn't blank out an already-known name) —
this is the one access-pattern nuance beyond a plain two-column upsert, and this plan flags it
explicitly for `graph-dba`'s Cypher to get right (a naive `SET p.name = $name, p.address =
$address` would null out an omitted field on a partial update; the query must distinguish "not
supplied" from "explicitly cleared," e.g. via `coalesce($name, p.name)`-style per-field handling,
or by only ever setting the fields the service layer actually passes).

Unlike `Cart`/`Order` (`workflow-cart-and-totals.md` §3.2), a `Profile`'s two fields are plain
scalar strings — no list-of-line-items shape, no "flat JSON string vs. real node" fork to weigh.
This plan expects a straightforward `(:Profile {profileId, name, deliveryAddress, updatedAt})`
node with a standard `{label}Id` index+constraint pair, but leaves the exact property names/DDL to
`graph-dba`'s note per the coordinator's delegation.

### 3.2 Profile tools (FR-1..FR-3)

Two new `Tool` classes in `tools.py`: `GetProfileTool` (`get_profile`, no args — returns
`{name, deliveryAddress}` or nulls for unset fields; **not `{"found": false}`** — unlike a catalog
lookup, "no profile yet" is not an error/abstention case, it's the ordinary first-conversation
state, so the tool always returns a (possibly partially-null) object rather than the
catalog-lookup abstention shape) and `SaveProfileTool` (`save_profile`, `{name?, deliveryAddress?}`
— both optional, at least one expected but not structurally enforced; calls
`services.save_profile`, which performs the upsert per §3.1's contract).

`services.py` additions: `get_profile(ctx) -> dict` (always returns a shape, defaulting absent
fields to `None`), `save_profile(ctx, *, name=None, delivery_address=None) -> dict` (thin over
`repository.upsert_profile`).

The system prompt guidance (bundled into the `salesperson@v3` version bump, §4) tells the model:
ask for name/delivery address once, only if `get_profile` shows either missing, and never ask
again once both are known for this customer in this workspace — this is prompt-level behavior
(AC-1's "does not ask again"), not an engine-enforced constraint; the tool layer only guarantees
the *data* persists and round-trips correctly, not that the model never re-asks. This mirrors how
`triage`'s own `intake` step's "ask one question at a time" behavior is a system-prompt discipline
(`scripts/seed_workflows.sh:190-196`), not something the engine structurally enforces — consistent
with the existing precedent for how this codebase separates "what the tool layer guarantees" from
"what the prompt asks the model to do."

## 4. Step-by-step implementation

Builds on `docs/plans/workflow-catalog-lookup.md`'s scaffold and
`docs/plans/workflow-cart-and-totals.md`'s `salesperson@v2` (this plan bumps to `v3`) — sequence
this plan **after** cart-and-totals lands, or independently if the coordinator parallelizes (the
two touch disjoint tool sets and disjoint schema; only the shared `SALESPERSON_DEF["version"]`
bump and `config.tools` list are a shared-file edit — flag to the coordinator if both are
in flight at once, per the standing "serialize shared-file units" practice).

1. **Wait for `graph-dba`'s `workflow-durable-profile-graph.md`** — DDL + the per-field-upsert
   Cypher (§3.1).
2. **`server/falkorchat/repository.py`** — `get_profile`, `upsert_profile` per that note.
3. **`server/falkorchat/services.py`** — `get_profile`, `save_profile` (§3.2).
4. **`server/falkorchat/tools.py`** — `GetProfileTool`, `SaveProfileTool` (§3.2); register.
5. **`server/falkorchat/proof_defs.py`** — bump `SALESPERSON_DEF["version"]` to `"v3"`, extend
   `config.tools` with `["get_profile", "save_profile"]`, extend `systemPrompt` per §3.2.
6. **`scripts/seed_salesperson.sh`** / **`scripts/verify_salesperson.sh`** — publish/materialize/
   verify `salesperson@v3`.
7. **`docs/QUERIES.md`** / **`scripts/test_queries.sh`** — new `Profile` query entries + baseline
   bump.

**Done:** `salesperson@v3` proves FR-1..FR-3 live — a customer's name/address, given once, is
retrievable in a fresh conversation and updatable later; AC-3's proof is that this all happens
inside the same combined demo agent, not a separate workflow (trivially true here since nothing
in this plan introduces a second def).

## 5. Test strategy

| AC | What proves it | Altitude |
|---|---|---|
| AC-1 (persists, not re-asked across conversations) | `repository.upsert_profile` then `get_profile` from a second `Thread`, same `ctx.actor`/`ws`, returns the stored values; live `@mention` sequence across two threads | repository/service integration + live e2e |
| AC-2 (update, not frozen) | `upsert_profile` called twice with a changed `deliveryAddress` and an omitted `name`; assert the address updated and the name is unchanged (the §3.1 partial-update nuance, explicitly tested — this is the one case a naive implementation would get wrong) | repository/service integration |
| AC-3 (same combined agent) | `salesperson@v3`'s def is a version bump of the same `key: "salesperson"`, not a new def — trivially checkable from the publish call itself | service/publish-contract check |

## 6. Risks & open questions

- **Profile schema not yet designed** — this plan's step-by-step is gated on `graph-dba`'s note,
  same posture as `workflow-cart-and-totals.md`.
- **Single-hardcoded-actor caveat** — identical to `workflow-cart-and-totals.md` §6's note, not
  restated in full here: AC-1's "same customer, new conversation" is provable today; distinct-
  customer isolation is not independently demonstrable until K-016 (M2.5) lands.
- **"Does not ask again" is a prompt discipline, not an engine guarantee** (§3.2) — if QA finds the
  model re-asking despite a populated profile, that is a prompt-tuning fix, not evidence the
  persistence layer is broken; keep the two failure modes distinct when triaging.
- **Shared-file edit risk with `workflow-cart-and-totals.md`** (§4) — both plans edit
  `proof_defs.py`'s `SALESPERSON_DEF` and `scripts/seed_salesperson.sh`. If a coordinator runs
  both implementation stages concurrently, serialize the edits to those two files specifically
  (the rest of each plan's file list is disjoint).
