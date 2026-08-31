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
| U1 | `data-scientist` | `a8ebe52269f4f11b7` | delivered | `docs/reviews/salesperson-tool-reliability-ml.md` §13 | — → — | 185.0k tok / 85 tools |
| U2 | `tdd-engineer` | `a450d3f4aadd7fa5b` | in-flight | deterministic `services.place_order` double-dispatch test | — → — | — |

## Notes

- Single-unit start, same shape as rounds 1-2's own first units — `teco` verifies the diagnosis
  directly (re-checkable stats/ground-truth, no code change) rather than dispatching a separate
  `analyst` gate for a diagnosis-only deliverable. A fix unit, if warranted, follows normal
  implementer + `analyst` review gating.
- No parallel dispatch risk (single unit; round 2's coordination has no unit currently in flight).
- **U1 delivered and independently re-verified by `teco` 2026-08-31** — after two premature
  "completed" notifications whose `<result>` were mid-task status lines (checked via `SendMessage`
  rather than accepted), the real §13 landed: 0/28 (0.0%, Wilson CI 0.0-12.1%) on both
  ground-truth signals, no fix warranted, structural argument independent of the sample. All
  Wilson CIs recomputed from scratch and matched exactly (0/28, 1/28, 14/24, 5/30 all reproduced
  to one decimal). Every cited code range re-read against source and confirmed:
  `repository.place_order` (`repository.py:2913-2970`, guarded `CREATE` keyed on caller-minted
  `order_id`, cart cleared only on the creating call), `services.place_order`/`_priced_cart_lines`
  (`services.py`, `None`/`[]` on empty cart), `tools.py`'s empty-cart message, `_WRITE_TARGET_ARG`
  (`executor.py:321-324`) confirmed to exclude `place_order` entirely (dedup key never computed
  for it — K-061's guard structurally can't and doesn't touch it). `ws:ds-k059` confirmed
  `GRAPH.DELETE`d (absent from the live graph list); `reference` catalog confirmed intact (15
  products). Diff confirmed scoped to exactly this section (208 insertions, one file);
  `docs/BACKLOG.md` confirmed untouched by the delegate, folded in by `teco` directly (K-059
  rewritten in place: 🟡 in-progress, no fix warranted, one cheap deterministic test left as the
  closing step; K-061's stale "open, relevant to K-059 too" cross-reference also rewritten to
  reflect the now-resolved answer). Two early-session findings preserved in this note earlier
  turned out accurate and are restated here for the permanent record:
  (1) the original 3-turn script (mirroring §12.1's wording) never reached `place_order` at all —
  `salesperson@v5` asks for profile/address first when none is on file; fixed by seeding profile
  in turn 2 before the turn-3 target action (a script-design correction, same class as §12.2's).
  (2) a structural read of `repository.place_order` (`server/falkorchat/repository.py:2913-2970`)
  and `services.place_order`/`_priced_cart_lines` (`server/falkorchat/services.py:2613-2788`)
  found `place_order` **destructively clears the cart on the call that creates the `Order`**, and
  a call against an already-empty cart returns `None` (no `Order`, no line-item write) rather than
  an idempotent no-op — so a literal second sequential `place_order` dispatch right after a
  successful first one cannot, by this mechanism alone, produce a second `Order` node the way
  K-061's `add_to_cart` duplicate could; a second `Order` needs the cart repopulated in between.
  Delegate will report dispatch-count and `Order`-count anomalies **separately** in §13 rather than
  collapsing them, given this. Not yet independently re-verified by `teco` — pending §13's arrival.
- **U2 dispatched 2026-08-31** (`tdd-engineer`, fresh agent, per §13.5 point 4's own recommendation
  — a deterministic unit test calling `services.place_order` twice back-to-back asserting exactly
  one `Order` results, mutation-tested, cheaper and more certain than a larger live re-run). Brief
  instructs reading §13/`services.py`/`repository.py` directly, a safe copy-aside mutation method,
  and a stop-and-ask escalation if the first honest run (pre-mutation) contradicts the diagnosis by
  showing a real duplicate `Order` — that would be a genuine blocker, not a green test.
