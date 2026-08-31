# Salesperson tool-orchestration reliability — round 3 (K-059) — Coordination

> **Status:** active · **Owner:** `teco` · **Tracks:** K-059 (post-M6, not a milestone gate)

Successor to `docs/plans/salesperson-tool-reliability2-coordination.md` (still `active`, K-061
shipped/reviewed, live regression left open by design) — ordinal-bumped per root `AGENTS.md`'s
collision rule 5 (same kind/topic/role, `-coordination`, round 2 already executed against).
Picking up **K-059** (`docs/BACKLOG.md`), chosen over the other open follow-up, **K-062**, on
value/risk: K-059 concerns a possible live *duplicate order* (real financial/data-correctness
impact if confirmed), while K-062 is a text-only misstatement explicitly filed as
low-severity/opportunistic (`docs/BACKLOG.md` K-062's own `Owner` line: "pick up opportunistically
... rather than as its own dedicated pass") — not worth a dedicated coordination round on its own.

**Why K-059 now, specifically:** K-061's own diagnosis (`docs/reviews/
salesperson-tool-reliability-ml.md` §12.8) found K-061's genuine same-turn `add_to_cart` duplicates
all co-occurred with a nearby HELD rejection on an unrelated target in the same turn, and
explicitly recommended K-059's own next diagnosis pass **deliberately test that same
held-rejection-adjacent condition** (not just the original, too-small n=4
`place-order-retrigger` condition `ml.md` §9 already tried) — this is a concrete, already-designed
next step, not a cold start.

**Prior art (read before dispatching or picking this up cold):**
- `docs/BACKLOG.md` K-059 — the filed item: why `place_order`'s zero-argument shape makes K-058's
  existing per-argument guard structurally inapplicable, the owner/test-strategy lines.
- `docs/reviews/salesperson-tool-reliability-ml.md` §9 (the original `place-order-retrigger`
  condition, n=4, too small) and §12.7-§12.8 (K-061's held-rejection-adjacent finding and its
  explicit recommendation for this next K-059 pass).
- `server/falkorchat/executor.py` — `_WRITE_TARGET_ARG` (only `add_to_cart`/`remove_from_cart`
  have a resolved-target argument to key on; `place_order`/`clear_cart` take zero arguments, so
  K-058's and K-061's existing guards structurally cannot cover them) and `place_order`'s own
  idempotency behavior (mints a fresh `order_id` per call — does not protect against two
  independently-decided dispatches).

**Scope discipline (carried from rounds 1-2):** diagnosis first, larger n, before any fix
attempt — this thread has repeatedly found that a fix shape guessed ahead of a proper rate
estimate either doesn't hold (K-057's reverted second iteration) or isn't warranted (K-060 still
awaiting root-cause). K-059's own BACKLOG entry explicitly asks for a rate estimate before a fix
shape is chosen; if a fix does turn out to be warranted, its shape must also account for
`place_order` having no target argument to key on, structurally unlike K-058/K-061's dedup guards.

## Ledger

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict | Cost |
|---|---|---|---|---|---|---|
| U1 | `data-scientist` | `a8ebe52269f4f11b7` | in-flight | `docs/reviews/salesperson-tool-reliability-ml.md` §13 | — → — | — |

## Notes

- Single-unit start, same shape as rounds 1-2's own first units — `teco` verifies the diagnosis
  directly (re-checkable stats/ground-truth, no code change) rather than dispatching a separate
  `analyst` gate for a diagnosis-only deliverable. A fix unit, if warranted, follows normal
  implementer + `analyst` review gating.
- No parallel dispatch risk (single unit; round 2's coordination has no unit currently in flight).
