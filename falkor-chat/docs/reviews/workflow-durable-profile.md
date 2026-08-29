# Durable user-profile data for workflows — Plan Review

> **Status:** active · **Owner:** `analyst` · **Tracks:** K-054 (M6) · **Extended by:** `docs/reviews/workflow-durable-profile-impl.md`

## Scope & verdict

Reviewed `docs/plans/workflow-durable-profile.md` (`architect`) together with its companion
`docs/plans/workflow-durable-profile-graph.md` (`graph-dba`), against
`docs/requirements/workflow-durable-profile.md` (FR-1..FR-4, AC-1..AC-3), as part of the combined
M6 four-document gate. Also checked `docs/BACKLOG.md`'s K-054 entry against what these two
documents actually specify, per the coordinator's brief item 5.

**Verdict: needs changes** (one blocker) — **superseded, see Pass 2 below: approve with suggestions.**

**CPG:** considered, not relevant — new-code design over the current tree; `cpg_falkorchat` is
stale (coordinator's brief) and both documents correctly read `services.py`/`repository.py`/
`QUERIES.md` §13.1 directly instead of leaning on it.

## Findings

### BLOCKER — the partial-update contract the architect plan explicitly asked for is not actually implemented by graph-dba's `write_profile` Cypher, and the gap would silently erase previously captured customer data under FR-3's own literal acceptance scenario

**Evidence:**

- `workflow-durable-profile.md` §3.1 states the requirement precisely: "an upsert that sets
  whichever of the two fields the caller supplies, leaving the other unchanged if omitted... the
  query must distinguish 'not supplied' from 'explicitly cleared,' e.g. via
  `coalesce($name, p.name)`-style per-field handling, or by only ever setting the fields the
  service layer actually passes."
- `workflow-durable-profile-graph.md` §3's actual `write_profile` Cypher does neither:
  ```cypher
  MERGE (c:Customer {customerId: $customerId})
  ON CREATE SET c.createdAt = $now
  SET c.name = $name, c.deliveryAddress = $deliveryAddress, c.profileUpdatedAt = $now
  ```
  Both fields are set unconditionally, every call, from whatever `$name`/`$deliveryAddress`
  values are passed in. The same section explicitly documents that "passing `$name`/
  `$deliveryAddress` as `NULL` in `SET` clears the property" and warns that "a tool... must be
  careful never to pass `NULL` for a field it isn't actually trying to change" — but assigns
  responsibility for avoiding that to "whichever tool authors this," without either document
  actually specifying the mechanism that prevents it.
- The tool-layer signatures that would have to enforce this discipline don't: `architect`'s own
  §3.2 sketches `SaveProfileTool(save_profile, {name?, deliveryAddress?})` — both optional — and
  `services.save_profile(ctx, *, name=None, delivery_address=None)`. If the model calls
  `save_profile` with only `deliveryAddress` (an entirely ordinary partial update — e.g. FR-3's
  own scenario, "a customer provides updated address information later"), `arguments.get("name")`
  returns `None`, indistinguishable in plain Python from "the caller explicitly wants to clear the
  name." That `None` flows straight through `services.save_profile` to
  `repository.upsert_profile` to `$name` in the Cypher above, which then executes
  `SET c.name = NULL` — erasing a previously captured name the customer never asked to remove.

**Why it matters:** this is not a hypothetical edge case — it is **exactly** the requirements
doc's own FR-3 ("if a customer provides updated name/address information later, the stored
profile is updated, not frozen") and AC-2 ("a customer already has a stored name/address... they
provide an updated address... the stored profile reflects the update" — i.e. the name must
survive). The architect plan's own §5 test-strategy row even names this precisely as "the one case
a naive implementation would get wrong" and calls for a test asserting the name is unchanged when
only the address is supplied — but neither document actually specifies an implementation that
passes that test. As currently specified, an implementer following both documents literally ships
code that silently corrupts customer data on the very acceptance path the plan set out to prove.

The `write_model_overrides` precedent graph-dba cites (`QUERIES.md` §13.1) as the model to mirror
is a poor fit here: that endpoint's docstring explicitly says `NULL` means "leave unset / CLEAR" by
design, because its caller is an admin config-write endpoint that always supplies the full
four-field state explicitly (there is no "argument omitted vs. explicitly null" ambiguity at that
call site). `SaveProfileTool`'s access pattern is the opposite — a conversational, genuinely
partial call — so copying that precedent's *NULL-clears* semantics onto a *NULL-may-mean-omitted*
caller is the specific mismatch that produces the bug. Nothing in FR-1..FR-4 or the out-of-scope
list requires the ability to explicitly clear a field once set, so there is no competing
requirement this fix would break.

**Suggested improvement:** change `write_profile`'s `SET` clause to coalesce against the existing
value instead of overwriting unconditionally:
```cypher
MERGE (c:Customer {customerId: $customerId})
ON CREATE SET c.createdAt = $now
SET c.name = coalesce($name, c.name),
    c.deliveryAddress = coalesce($deliveryAddress, c.deliveryAddress),
    c.profileUpdatedAt = $now
```
This makes `NULL`/omitted mean "leave unchanged" — the semantics FR-3 actually needs — while still
being a single atomic `MERGE`+`SET` with no read-then-write race, satisfying the architect plan's
"never a read-then-write pair" constraint. `$now`/`profileUpdatedAt` should still update
unconditionally (a call that supplies nothing meaningful is not expected, but even a no-op call
updating just the timestamp is harmless). This is a one-line Cypher change to graph-dba's note, not
a redesign — recommend it be resolved before this note is treated as implementation-ready, since
it is exactly the kind of gap that would otherwise be discovered mid-implementation or, worse, in
production data loss.

### MINOR — `docs/BACKLOG.md`'s K-054 entry describes a "`Profile` schema" that does not exist in the design as specified

**Evidence:** `docs/BACKLOG.md` line 61-64: "Depends on `graph-dba`'s
`workflow-durable-profile-graph.md` for the `Profile` schema." But
`workflow-durable-profile-graph.md` explicitly and deliberately does **not** create a `Profile`
node or label — its whole §1 argument is "not a second identity mechanism... two more properties
on the existing [`Customer`] node... No new label." The two documents under review are internally
consistent with each other; the drift is in `BACKLOG.md`'s own summary of them.

**Why it matters:** low-stakes on its own (a future reader who trusts the plan documents over the
backlog blurb won't be misled), but it is exactly the kind of imprecision that could cause a future
implementer skimming only the backlog to go looking for a `Profile` label/constraint that was never
designed, or to assume a schema decision is still open when it's actually settled.

**Suggested improvement:** when `teco` next touches this backlog entry, reword to "for the two
profile properties added to the shared `Customer` node" (or similar) rather than "the `Profile`
schema." Not blocking this plan gate.

## Cross-cutting checks (per the coordinator's brief)

- **`Customer`-anchoring — genuinely shared, not duplicated.** Confirmed by direct read: this
  document's graph note explicitly reuses `workflow-cart-and-totals-graph.md` §1's `Customer` node
  verbatim, with the identical `customerId = User.userId` resolution and the identical "not
  `identity`, not scoped by a `workspaceId` property" reasoning. No competing identity concept was
  independently invented here.
- **FR-8-style determinism fork — not applicable, and the plan correctly says so.** A two-field
  read/write has no "deterministic vs. LLM" question; the plan states this explicitly rather than
  silently omitting the discussion.

## What's solid

- The decision to add two properties to the existing `Customer` node rather than a new `Profile`
  node is well-argued (§1 of the graph note) and consistent with the requirements doc's own
  explicit rejection of "an extensible/open-ended profile schema."
- The single-hardcoded-actor caveat and the shared-file (`proof_defs.py`/`seed_salesperson.sh`)
  edit-risk with `workflow-cart-and-totals.md` are both correctly identified and correctly not
  restated in full where already stated once elsewhere — good cross-document discipline.
- "Does not ask again" is correctly and explicitly scoped as a prompt-level discipline, not an
  engine guarantee — the plan is honest about the boundary between what the tool layer proves and
  what the system prompt merely asks for.

## Pass 2 — 2026-08-27 (re-gate against `workflow-durable-profile-graph.md` `Version: 2`)

**Verdict: approve with suggestions.**

- **BLOCKER (partial-update `NULL`-clears bug) — fixed, live-reproduced in both directions.**
  `workflow-durable-profile-graph.md` §3's `write_profile` now reads
  `SET c.name = coalesce($name, c.name), c.deliveryAddress = coalesce($deliveryAddress,
  c.deliveryAddress), c.profileUpdatedAt = $now` — exactly the fix requested. §0/§3's fresh
  2026-08-27 verification log (disposable graph `ws_cartprobe_profile_check2`) shows both partial
  directions: a name-only-omitted call (`$name=NULL, $deliveryAddress='456 New Ave'`) leaves
  `name='Alice'` untouched; the symmetric address-only-omitted call
  (`$name='Bob', $deliveryAddress=NULL`) leaves `deliveryAddress='456 New Ave'` untouched — both
  with `count(Customer) = 1` throughout, ruling out a duplicate-node side effect from the `MERGE`.
  This also dissolves the deeper concern in my original finding (Python's `None` default cannot
  distinguish "argument omitted" from "argument explicitly null" at the tool/service boundary):
  with `coalesce()`, both cases now correctly resolve to "leave the field as it is," so the
  ambiguity at the Python layer stops mattering — no sentinel-object mechanism is needed, and none
  was added. The trade-off this closes with (there is now no way to *explicitly clear* a
  previously-set field through this query) is correctly named as deliberate in the revised §3, and
  I agree it's the right call — no FR/AC in the requirements doc asks for clearing.
- **MINOR (`docs/BACKLOG.md` K-054 "`Profile` schema" wording) — not addressed, unaffected by this
  re-gate.** Out of scope for `architect`/`graph-dba` to fix in these two documents; still routes
  to `teco`'s next touch of the backlog entry. Not a blocker to this plan.

No new finding from re-reading the full `Version: 2` document (the schema, §1/§2/§4/§6/§7, is
unchanged per the revision note; only §0/§3 changed, and both were read in full above, not
sampled).

## Open questions

None. The blocker is closed and live-verified; the one remaining MINOR is a `docs/BACKLOG.md`
wording fix outside this plan's scope, not a gate on implementation.
