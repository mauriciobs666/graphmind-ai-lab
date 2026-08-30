# Structured catalog/reference lookup for workflows — Plan Review

> **Status:** archived · **Owner:** `analyst` · **Tracks:** K-052 (M6) · **Extended by:** `docs/reviews/workflow-catalog-lookup-impl.md`

## Scope & verdict

Reviewed `docs/plans/workflow-catalog-lookup.md` (`architect`, the shared M6 demo scaffold)
against `docs/requirements/workflow-catalog-lookup.md` (FR-1..FR-7, AC-1..AC-5), as part of a
combined gate across all four M6 sibling plans (`workflow-cart-and-totals.md`,
`workflow-durable-profile.md`, `workflow-nl-query-generation.md`). Verified directly against
`server/falkorchat/executor.py`, `services.py`, `guards.py`, `tools.py`, `extraction.py` (per the
coordinator's brief — `cpg_falkorchat` is stale and was not used) rather than trusting the plan's
own characterizations. Did not re-litigate the security lens (not applicable to this document —
no adversarial/mutation surface here).

**Verdict: approve.**

**CPG:** considered, not relevant — this is new-code design review for `falkor-chat/server`
(no CPG-backed impact-analysis question arises), and `cpg_falkorchat` is independently confirmed
stale (uncommitted K-048 `executor.py` changes noted in the coordinator's brief). All structural
claims below were verified by reading the actual source files.

## Findings

### MINOR — `ensure_customer`/`ensure_cart` ownership is a downstream (cart-and-totals) concern, not this plan's, but this plan's scaffold makes it easy to overlook

Not a defect in this document — flagged here only because this is the plan a future reader
consults for the scaffold. No action needed on this document itself; see the corresponding
finding in `docs/reviews/workflow-cart-and-totals.md`.

### NIT — `maxIterations: 8`/`SALESPERSON_MAX_STEPS = 40` are explicitly acknowledged as calibration seeds

§3.3 already flags both constants as "calibration seed, not load-bearing" and assigns
calibration to coder/QA on real runs. No change requested — noting only that this is exactly the
right way to flag an unverified numeric constant in a plan (see "What's solid").

## What's solid

- **Grounding is excellent and independently verified.** Every load-bearing claim was checked
  against the real tree and held: `_validate_def_spec`'s "≥ 1 transition" invariant
  (`services.py:1376-1379`, confirmed by direct read), the unconditional-guard-always-fires
  behavior (`guards.py:223-224`, `if not guard: return GuardVerdict(decision=True, ...)`, confirmed
  verbatim), the OUTCOME A-before-B ordering in `_drive_loop` (`executor.py:495-524`, confirmed —
  a firing transition is checked before the `waitsForHuman` park path), the `tool`-typed step's
  `NotImplementedError` seam (`executor.py:534-575`, `STEP_TYPES` at `services.py:79-81`), the
  `truthy` guard operator actually existing in `guards.py:157`, and `ToolRegistry`'s
  `schema`/`dispatch` contract (`tools.py`) all match the plan's characterization exactly.
- **The `ended`/`ctx.endConversation` scaffold decision is sound and well-precedented.** It
  satisfies the ≥1-transition invariant without ever firing in the demo's acceptance path, and the
  plan correctly cites the existing `human_handoff`-present-not-exercised pattern
  (`tools.py:338-353`) as direct precedent for "structurally required, deliberately unexercised" —
  this is not dead code smuggled in, and the plan says so clearly enough that a future reviewer
  won't mistake it for such.
- **`Product` schema placement in `reference` is correctly reasoned against `docs/DESIGN.md` §3/§4**
  (global, read-mostly catalog data, looked up not traversed from workspace nodes) and the
  `productId` identity/index/constraint choices follow the codebase's established `{label}Id`
  convention precisely.
- **`extraction.normalize_name` reuse** (rather than a second, independently-written normalizer)
  is exactly the right call and is verified correct: `extraction.py:67-78`'s docstring itself
  states it is "the ONE shared normalization helper." The plan's own §6 transparently flags the
  minor cross-feature-module-naming smell this reuse introduces — appropriately non-blocking, and
  the right way to disclose a known, deliberate trade-off rather than hide it.
- **The version-bump discipline (§2.5)** is correctly derived from the create-only property
  semantics documented in `docs/DESIGN.md` §4/K-034, and correctly anticipates the exact hazard
  (`salesperson@v1` re-published with new tools silently no-oping) that the three sibling plans
  each depend on this document to have gotten right.
- **Test strategy is proportionate**: pure/unit + tool-unit + live e2e altitudes are assigned
  sensibly per AC, and the additional coverage list (publish-time validation of the one
  conditional transition, a no-op-republish regression test, an "endConversation never fires"
  regression guard) closes exactly the risk this plan's own scaffold decision introduces.
- **Honest self-disclosure of residual risk** (§6): the DDL-not-independently-DBA-gated call is
  reasoned rather than assumed, and explicitly offered as cheap to add if a reviewer wants it — I
  don't think it's needed here (three scalar properties, structurally identical to a dozen existing
  `bootstrap_reference` labels), so I'm not asking for it.

## Open questions

None — this document is ready to build against as written.
