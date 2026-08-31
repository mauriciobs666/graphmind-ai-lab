# Salesperson tool-orchestration reliability — round 3 (K-059) — Coordination

> **Status:** archived · **Owner:** `teco` · **Tracks:** K-059 (post-M6, not a milestone gate)

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
| U2 | `tdd-engineer` | `a450d3f4aadd7fa5b` | delivered | `server/tests/test_services.py` + `docs/HISTORY.md` entry | `teco` (direct) → verified | 96.3k tok / 31 tools |

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
- **U2 delivered and independently re-verified by `teco` 2026-08-31** — no stop-and-ask fired
  (first honest run was green, corroborating the diagnosis). Diff read in full: new test matches
  `FakeRepo`'s real `place_order` contract (fresh `order_id` per call, cart cleared only on the
  creating call — confirmed against the actual `FakeRepo.place_order` implementation, not just the
  delegate's claim). Test re-run in isolation (green). Mutation independently reproduced from a
  safe copy-aside backup (`if not priced: return None` → `if False: return None` in
  `services.py`): failed red for the predicted reason (second call minted a real second `Order`),
  restored via `cp`, `diff` confirmed byte-identical, re-run green. Full offline suite re-run
  personally: **2307 passed, 14 deselected** — matches exactly. Shared `reference`/`ws:acme` state
  re-verified `OK` (`verify_workflows.sh`, `verify_salesperson.sh`, `verify_catalog.sh`) after
  re-seeding — **note:** the first reseed attempt used a malformed `bootstrap_schema.sh` invocation
  (`teco`'s own typo — a positional `EMBEDDING_DIM=1024` arg instead of a leading env var),
  creating a stray empty `ws:EMBEDDING_DIM=1024` graph key; caught immediately via a probe query
  against the loaded-graphs list, `GRAPH.DELETE`d, confirmed gone before the correct
  `EMBEDDING_DIM=1024 ./scripts/bootstrap_schema.sh acme` reseed. No data loss, no unrelated graph
  touched. `HISTORY.md` entry independently confirmed accurate against everything re-verified
  above. **K-059 closed**: removed from `BACKLOG.md` entirely (delivered item, not kept even as an
  index row, per root `AGENTS.md`) — its record is the `HISTORY.md` entry above.

## Closed 2026-08-31 — K-059 fully resolved: no fix warranted, invariant pinned by a deterministic test

Diagnosis (U1, n=28, §13) found no live duplicate-dispatch defect and a structural argument
explaining why `place_order` doesn't share K-061's harm mechanism. Per the diagnosis's own
recommendation, a deterministic unit test (U2) now pins that invariant with certainty rather than
resting on a live sample alone — closing the residual power gap the diagnosis was honest about
(§13.2's suppressed held-rejection stratum) more cheaply than a corrected live re-run would have.
No production code change was needed or made. This coordination doc is now `archived` — both units
delivered and independently verified, K-059 fully closed out of `BACKLOG.md`.
