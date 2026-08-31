# Backlog — falkor-chat

> **Status:** active · **Owner:** `teco` · **Tracks:** K-016…K-062

> **How to read this.** Forward-looking only — what is proposed but unbuilt. *When* something
> changed and *what* it involved live in [`HISTORY.md`](./HISTORY.md), one dated entry per
> delivered item — **a delivered item is not kept here at all, not even as an index row**
> (root `AGENTS.md`). Design lives in [`DESIGN.md`](./DESIGN.md), whose §1 is the standing
> decision register; the query library in [`QUERIES.md`](./QUERIES.md). Completed plan
> documents stay where they are, marked `Status: archived` (root `AGENTS.md`); `archive/` holds
> frozen documents from the previous convention.
>
> **Status of an *open* milestone is authoritative here.** A closed one's record is
> `HISTORY.md`; DESIGN §12's roadmap is the shape of the work, not a status board.
>
> Status markers: 🔵 proposed · 🟡 in-progress · ✅ delivered (→ `HISTORY.md`) · ⚪ rejected/deferred.
> Item IDs keep the `K-` prefix from the former `kaizen/plan.md`.

## Active

Nothing currently active — **M6 (business entities in workflows) closed 2026-08-30**: all four
sibling capabilities (K-052 catalog lookup, K-053 cart/orders, K-054 durable profile, K-055
natural-language query generation) delivered and proven live together inside the combined
`salesperson@v4` demo agent; see `HISTORY.md` for the five closing entries and
`docs/plans/workflow-salesperson-demo-coordination.md` (archived) for the full multi-session trail.

Everything open is a follow-up filed out of a closed milestone (`## Open follow-ups`) or the
deferred M2.5 hardening track.

## Milestones still open

| Milestone | Reaches ✅ when | Items |
|---|---|---|
| **M2.5 — Hardening** ⚪ *(deferred)* | Real auth, transport-level agent path, real-time push | K-016 → K-017, K-018 |

Follow-ups filed out of a closed milestone are **not** green-gates for it; they are listed under
`## Open follow-ups`.

## Open follow-ups

Each was filed out of a closed milestone's gates or a later investigation; none gates M5.

### K-059 — `place_order` has no protection against a live off-turn duplicate dispatch (🔵 proposed — flagged by `analyst`'s K-058 review, `docs/reviews/salesperson-tool-reliability-impl2.md`, 2026-08-30)

> **Why it exists.** K-058's dispatch-time write guard (`docs/reviews/
> salesperson-tool-reliability-ml.md` §9.4, shipped `HISTORY.md` 2026-08-30) only covers
> write-mutating tools with a single resolved string target argument (`add_to_cart`/
> `remove_from_cart`'s `productName`). `place_order`/`clear_cart` take zero arguments, so the
> guard structurally cannot check them — this is correct-and-faithful-to-spec, not an oversight
> (`executor.py:42-46`'s own comment states it plainly), but it means the underlying model
> tendency the guard targets is entirely unaddressed for `place_order` specifically.
> `place_order`'s own idempotency guard mints a fresh `order_id` per call, so it does not protect
> against two independently-decided dispatches the way it protects against a literal retried
> call. The ml note's own §9's `place-order-retrigger` condition found 0/4 — too small a sample to
> support any rate claim either way.
- **Owner:** `data-scientist` for a rate estimate (mirror §9's condition-isolation method, larger
  n) before deciding whether a fix is warranted; if confirmed at a non-trivial rate, likely fix
  shape is a confirmation step or a different structural guard than K-058's (no single target
  argument to check presence of).
- **Risks/RAM:** none — diagnosis first, no graph/schema surface.
- **Test strategy:** scripted, repeated live-conversation probe isolating a place-order-after-
  place-order opportunity, ground-truth-checked against `Order` node count, same harness pattern
  as K-057/K-058's own eval scripts.

### K-060 — `salesperson@v5` sometimes silently drops a genuine match when `filter_products` returns a mixed-category result (🔵 proposed — disclosed during K-057's own fix verification, two wording attempts failed to move it, `docs/HISTORY.md` 2026-08-31)

> **Why it exists.** Live-verifying K-057's wording fix (`docs/HISTORY.md` 2026-08-31,
> `docs/reviews/salesperson-tool-reliability-ml.md` §11) at n=20 found every one of the 4/20
> remaining wrong replies traced to a third, previously unobserved mechanism, distinct from both
> mechanisms §11 diagnosed: when `filter_products` is called with no `category` argument (a
> mixed-category result set), the model sometimes silently drops a genuinely-matching item from
> its synthesized reply instead of including it — not a rounding error, not a self-contradiction,
> a synthesis-time omission on an unfiltered result set.
- **Two independent, reasonable-looking wording attempts already failed to fix this — worth
  weighing before a third.** A `category`-parameter nudge (encouraging the model to always pass
  `category` when the customer names one) plus a `systemPrompt` synthesis-time "check every
  returned item, never drop a match" safety net were live-tested together at n=20: the model
  still never passed `category` (0/20) and net wrong-reply rate went **up** (30% vs. the shipped
  fix's own 20%), while also suppressing a multi-call self-correction pattern that had rescued
  several replies under the plain wording. Reverted, never shipped. Two failed wording attempts
  is itself evidence this may not be fixable by wording alone.
- **Owner:** `data-scientist` for a proper root-cause pass before any further wording guess — same
  discipline this whole investigation thread has applied throughout (never ship/reattempt an
  unverified mitigation, ml note §4.2/§8.4/§9.4). Worth checking whether `filter_products`'s own
  synthesis step (not `systemPrompt` wording) is the more reliable lever — e.g. whether the tool
  could return an explicit per-item flag or a restructured payload less prone to synthesis-time
  omission — before assuming another wording iteration is the only path.
- **Risks/RAM:** none — diagnosis first, no graph/schema surface.
- **Test strategy:** re-run §11's own harness pattern (throwaway workspace, ground-truth
  instrumented tool registry) isolating no-`category` `filter_products` calls specifically, at a
  larger n than the n=20 (or n=16 single-call-only) samples so far, to get a tighter rate estimate
  before root-causing. **Also fold in** (flagged by `analyst`'s K-057 diff review, `docs/reviews/
  salesperson-tool-reliability-impl3.md`): a "more than $X" probe against `minPrice` — the
  shipped K-057 fix added symmetric inclusive-bound guidance to `minPrice` by analogy, but only
  the `maxPrice`/"less than" direction was actually live-regression-tested; whenever this item's
  own harness is next run is the cheapest place to close that gap.

### K-061 — `salesperson@v5` sometimes silently duplicates its own current-turn, legitimately-mentioned `add_to_cart` call, inflating a cart line the customer never asked to double (🟡 in-progress — unit-level fix shipped, `analyst`-approved with suggestions and both findings closed; live regression confirmation still the only open item, 2026-08-31)

> **Why it exists.** Diagnosed at n=24 (`ml.md` §12, pooled 5/30, 16.7% Wilson CI 7.3-33.6%): the
> model's own current-turn tool-call loop re-dispatches `add_to_cart` a second time for a target it
> already successfully added earlier in that same loop, doubling the cart quantity. Distinct from
> K-058 by design (K-058 blocks an *off-turn* re-fire of a *previous* turn's write; this is a
> *same-turn* re-fire of the model's own *current*-turn write on a target genuinely mentioned in
> that turn's text — exactly the case K-058's guard deliberately doesn't touch).
- **Shipped** (`server/falkorchat/executor.py`, commit `381c9fc`, `docs/HISTORY.md` 2026-08-31): a
  new dispatch-time guard keyed on `(tool name, full argument set)` holds an exact same-turn
  repeat of an already-succeeded write, without blocking a legitimate different-argument call
  (e.g. "add 1, then make that 2"). Reproduction test first, mutation-tested.
- **`analyst`-reviewed** (`docs/reviews/salesperson-tool-reliability2-impl.md`): approve with
  suggestions, no blocker — placement, keying, and scope all verified correct. Both findings now
  closed (commit `381fdb8`): a genuine mutation-testing coverage gap on the "must not poison the
  dedup set on a failed dispatch" invariant is now pinned down by a dedicated test, and the
  misleadingly-named test was renamed to match what it covers. Full offline suite green
  (2306 passed, 14 deselected).
- **Still open — this item stays in BACKLOG until closed:** the live n≈20-30 regression pass
  K-061's own test strategy calls for (same ground-truth method as the diagnosis: `Cart`/
  `CartItem` Cypher + raw `TraceEvent`, never reply text) to confirm the fix actually drops the
  rate in a full live conversation, not just at the unit level. Not run yet — genuinely open, not
  silently skipped.
- **The co-occurring false-negative-reply symptom originally filed alongside this did not
  reproduce at n=24** (pooled 1/30, 3.3% CI 0.6-16.7%) — too rare for its own track; re-screen
  opportunistically whenever this conversation shape is next live-tested, no standalone follow-up.
- **The two-consecutive-held-rejections co-occurrence is a real candidate contributing factor,
  still underpowered to confirm**: all 3 of the n=24 pass's occurrences had ≥1 nearby HELD
  rejection on an unrelated target in the same turn; 0/10 reps with zero HELD rejections
  duplicated (CIs still overlap at n=24, not yet distinguishable).
- **Also found, filed separately as K-062:** a distinct text defect where the model misstates why
  an unrelated `add_to_cart` call was correctly held — not touched by this fix, confirmed
  byte-unchanged in the shipped diff.
- **Open, not yet resolved — relevant to K-059 too:** whether K-061 and K-059 (`place_order`
  off-turn duplicate dispatch, still `🔵 proposed`/unstarted) should eventually share one guard
  design. `ml.md` §12.8 found suggestive-but-unproven evidence (same "re-fire after a nearby held
  rejection" shape); this shipped fix is K-061-only (confirmed no shared design was in flight at
  dispatch time) — K-059's own next diagnosis pass should test the held-rejection-adjacent
  condition before its own fix shape is chosen.
- **Owner:** `data-scientist` (or `qa-engineer`) for the live regression pass; no further
  design/implementation owner needed unless that pass finds the rate didn't move.
- **Risks/RAM:** none.
- **Test strategy:** live n≈20-30 conversation pass reusing `ml.md` §12.1's exact 3-turn script,
  ground-truth via `Cart`/`CartItem` Cypher + raw `TraceEvent` chain, comparing the post-fix rate
  against the pooled pre-fix 16.7% (CI 7.3-33.6%).

### K-062 — `salesperson@v5` sometimes states the wrong reason for a correctly-held write call, even though the correct reason was handed back to it (🔵 proposed — found opportunistically during K-061's n=24 diagnosis pass, `docs/reviews/salesperson-tool-reliability-ml.md` §12.5, 2026-08-31)

> **Why it exists.** 2/24 (8.3%, Wilson CI 2.3-25.8%) reps in K-061's diagnosis pass show the
> model telling the customer an unrelated `add_to_cart` call was held because the product "was not
> recognized as a product" / "not recognized in the catalog" — factually wrong; the product is
> real and correctly catalogued, and K-058's guard's own held-call result
> (`executor.py:955-961`) states the *actual* reason verbatim back to the model ("was not
> mentioned anywhere in this turn's own message"). The model had the accurate explanation
> available and substituted a plausible-sounding wrong one. Distinct from K-061 (which is about
> the cart *state* being wrong); here the cart state is correct throughout and only the
> customer-facing explanation is wrong.
- **Owner:** unassigned — low severity (doesn't affect cart correctness, only an explanation the
  customer has no way to act on incorrectly), pick up opportunistically alongside K-061 or K-058
  follow-up work rather than as its own dedicated pass.
- **Risks/RAM:** none.
- **Test strategy:** whoever next runs `ml.md` §12.1's 3-turn script (K-061's fix verification is
  the natural occasion) should screen replies for this pattern too and fold a count into that
  pass's report rather than standing up a separate probe.

### K-029 — Converge the seed def sources into `proof_defs.py` (+ the symmetric `decision` publish invariant) (🔵 proposed — filed out of K-024, open item O-5 / gate m-9 / nit n-3)

> **Why it exists.** The two seeded defs use **two different source conventions**, deliberately for
> the K-024 slice: `access-request@v1`'s spec is imported from `server/falkorchat/proof_defs.py`
> (so the seed script and the offline acceptance test provably cannot drift), while **`triage@v1`'s
> literal is still inline in `scripts/seed_workflows.sh`**. Moving `triage`'s def *during* K-024 was
> declined with a reason, since corrected by **K-034**: at the time, published defs were believed
> **create-only** (`MERGE … ON CREATE SET`) end to end, so a byte-diff introduced while relocating a
> **live** def was assumed silently swallowed. As of K-034 that is only true for a **property**-only
> byte-diff (e.g. a `config` field reformatted during the move) — it still silently no-ops, so K-029's
> planned before/after equality check remains load-bearing for that half. A **topology**-changing
> byte-diff (e.g. a retargeted transition introduced while relocating the literal) is now **rejected**
> (`409 WorkflowDefConflictError`, nothing written) rather than swallowed — safer, but a `409` mid-
> deploy is still worse than catching it in a pre-flight check, so the equality check stays this item's
> load-bearing safeguard either way. `reference`/`ws:<id>` can still go stale independently whenever
> one side was never re-published/re-materialized at all. That is a split-brain risk to take on its
> own, with its own verification, not as a rider on a feature slice.
- **Owner:** **`coder`**, with an explicit before/after equality check on the published def subgraph
  (not just "the script ran").
- **Scope:** (1) move `triage@v1`'s inline literal into `proof_defs.py` beside `ACCESS_REQUEST_DEF`,
  leaving `seed_workflows.sh` a pure driver over the service layer for **both** defs; (2) prove the
  move is byte-identical *in the graph*, which given create-only semantics means either verifying
  against a freshly published `reference` or bumping `triage`'s version in lockstep with
  `config.TRIGGER_DEF_KEY`/`TRIGGER_DEF_VERSION` (note `start_server.sh` neither forwards nor exports
  those two vars today — a version bump also needs a script change); (3) fold the `n-A` warning
  (`ACCESS_REQUEST_DEF`'s key set **is** `publish_workflow_def`'s keyword signature) into whatever
  shape both defs end up sharing.
- **Also carries nit n-3 — the symmetric `decision` publish invariant.** K-024 enforces
  "a `human`/`wait` step must declare `config.waitsForHuman: true`" at publish, but **not** its
  mirror: **a `decision` step whose outgoing transitions are *all* conditional and which does not
  declare `waitsForHuman` self-loops until the step budget fails the run.** It is documented as a
  warning in `falkor-chat/AGENTS.md` and deliberately left unenforced because the symmetric check
  would **retro-reject existing test fixtures** (`server/tests/test_services.py`) — the same
  blast-radius problem B-2 caused in K-024, which is precisely why it belongs in an item that can
  budget for the fixture edits.
- **Risks:** touching a live published def is the risk; there is no new graph surface and no RAM cost.
- **Test strategy:** a test that both defs come from importable constants; a publish-equality check
  over the def subgraph; if n-3 is implemented, one ordering pin (it must run **last**, like the other
  three invariants) plus the fixture edits it forces.

### K-030 — Allow zero-transition (single-step) workflow defs; guard the `UNWIND` instead of rejecting (🔵 proposed — filed out of K-024 re-gate findings r-1/r-2)

> **Why it exists.** `repository._PUBLISH_CYPHER` ends in a bare `UNWIND $transitions AS tr …
> RETURN …`. With `$transitions = []` the row stream **collapses** — after the `WorkflowDef`, its
> `Step`s and the `START` edge have already been written — so the caller's `res.result_set[0]` raises
> `IndexError` on a **partially written** def. Because publish is `MERGE … ON CREATE SET`, retrying
> the corrected spec on the same `(key, version)` is a **silent no-op on the half-written def**: the
> version is permanently wrong and cannot be repaired by re-publishing. This is the same empty-`UNWIND`
> class that `AGENTS.md` documents as *guarded* for the §4 mention write-block; this path was not
> guarded. K-024 U4b **closed the reachable route** with a `_validate_def_spec` rule (running last)
> that rejects a transition-less spec **before any repository call** — prevention, not a nicer
> exception.
- **What is still open (re-gate r-1):** the fix is **publish-only**. `services.materialize_def` →
  `repository.materialize_snapshot` (`repository.py:1397`) **reuses the same query shape** and performs
  **no** spec validation, so a def poisoned before U4b — or any zero-transition subgraph read back by
  `read_def_subgraph`, which returns `transitions: []` rather than `None` — is still an unguarded
  `IndexError`/500 on materialize. Low likelihood (materialize is fed by publish, now guarded), but the
  guard is **asymmetric**, and the docs/tests currently imply it isn't:
  `server/tests/test_services.py:916` seeds a `FakeRepo` def with `"transitions": []` and asserts
  materialize **succeeds** — true of the fake, and exactly the shape the real query rejects.
- **Accepted limitation to remove (re-gate r-2):** the U4b rule also **rejects a legitimate shape** —
  a genuine single-step def. All four doc sites state the workaround ("a terminal outcome is a step
  with no *outgoing* transition, never a def with none") but none records it as **debt**. Without a
  `K-` number the next person needing a one-step def will either fight the rule or bypass validation.
- **The known cheap remedy:** guard the trailing `UNWIND` in `_PUBLISH_CYPHER` (and therefore
  `materialize_snapshot`, which reuses it) with the **§4 empty-`UNWIND` `CASE` pattern** this codebase
  already relies on and documents as load-bearing — `UNWIND (CASE WHEN $transitions = [] THEN [null]
  ELSE $transitions END) AS tr` with a `FOREACH` that never filters — then **relax**
  `_validate_def_spec`'s rule, and drop the `transitions=[]` mitigation comments in `proof_defs.py`
  and `tests/test_process_input.py`.
- **Owner:** **`graph-dba`** (the query change needs a gate + a re-PROFILE: the guard must not turn the
  index-anchored publish plan into a scan) → **`coder`** for the service-layer relaxation and the
  fixture/doc cleanup.
- **Risks/RAM:** none — no new node, index or property; a query-shape change only. The risk is plan
  regression, which is what the re-PROFILE is for.
- **Test strategy:** a publish and a materialize of a genuine single-step, zero-transition def, both
  asserted to succeed *and* to leave a complete subgraph (steps + `START` + the returned row); the
  existing ordering pins for the other publish invariants must stay green.

### K-032 — Materialize the workflow def's **data-dependence overlay** (CPG-style READS/WRITES) for publish-time static analysis (🔵 proposed — from a design conversation, 2026-07-22)

> **The framing (Code Property Graph lens).** The def graph is already a control-flow graph:
> `(:Step)-[:TRANSITION {guard, order}]->(:Step)` is a guarded CFG, `HAS_STEP` is one-level AST
> containment, and `(:StepRun)-[:NEXT]->` is an executed CFG path. **What's missing is the
> data-dependence layer (DDG):** which `ctx`/`output` keys each step *reads* (via its `cmp`/`llm`
> guards and, for a `decision`/`human`, the keys it branches on) and which it *writes* (a `human`/
> `wait` step's `config.expects`, a step's declared outputs). That information is **not missing —
> it's trapped inside the opaque `guard`/`config` strings**, and `services._validate_def_spec` +
> `guards.validate_cmp` already walk the `cmp` guard tree at publish, so ~90% of the extraction pass
> exists and is currently thrown away. Materialize it as real edges and three otherwise-impossible
> checks become one-hop Cypher.
- **Why it's worth doing (the payoff).** Publish-time (not live-run) detection of:
  1. **Dangling read** — a guard reads a `ctx`/`output` key no upstream step writes. This is exactly
     the **un-enforced n-3 hazard** AGENTS.md documents (a `decision` step with all-conditional
     outgoing transitions and no `waitsForHuman` **self-loops to budget exhaustion**) — today a live
     discovery, turned into a `WorkflowDefSpecError` at seed time. Overlaps K-029's symmetric-invariant
     proposal; this is the graph-shaped way to get there.
  2. **Unreachable step / dead branch** — plain CFG reachability from `START`.
  3. **Change-impact / blast radius** — "I changed `submit`'s output shape; which downstream guards
     read it?" This matters **specifically because published defs are topology-immutable (K-034) and
     property-create-only**: a def edit costs a version bump + snapshot republish + a
     `reference`↔`ws:{id}` split-brain risk, so knowing the blast radius *before* the bump has real
     value here.
- **Hard constraints (fall out of locked decisions — non-negotiable in any plan):**
  - **Derive at publish, never parse in Cypher.** Rule 8 (`ctx`/`config`/`guard` opaque, never
    filtered in Cypher) holds *iff* publish is treated as a compile step — `joern-parse` builds
    overlays once, queries traverse edges; same contract. Extraction runs app-side in the publish
    validator; only the resulting edges hit the graph.
  - **Overlay edges built inside `_PUBLISH_CYPHER` and `materialize_snapshot`, same query.** A
    separately-written overlay on a `MERGE … ON CREATE SET` def is a **new split-brain axis** on top
    of the `reference`-vs-snapshot one — and per the K-030 note the materialize path still `IndexError`s
    after a partial write, so the overlay must ride the existing atomic publish, not a follow-up write.
  - **Static-only, on the def — never on `StepRun`.** The def graph is tens of nodes (overlay edges
    are single-digit multiples → RAM non-issue, the inverse of a repo CPG where AST/CFG/REACHING_DEF
    fan-out dominates). The run graph is thousands of nodes and RAM-bound; "why did *this run* branch
    here" is a join through `RAN`, not a second copy of the layer.
  - **Honest `READS_UNKNOWN` for what can't be derived statically** — an `agent`/`llm`-guard node
    whose reads aren't extractable gets an explicit marker (Joern marks indirect calls the same way).
    A **feature**: it says precisely which parts of a flow are analyzable vs. trust-the-model. Do
    **not** attempt a probabilistic DDG — an unsound dependence edge produces confident-wrong impact
    answers, worse than none.
- **Owner:** **`graph-dba`** gates the FalkorDB model first (the overlay labels/edge types, whether a
  `CtxKey`-style node is per-def or shared, indexes) → **`architect`** designs the publish-time
  extraction pass + the three validations → **`coder`**/**`tdd-engineer`**. A CPG-model design note
  would land at `falkor-chat/docs/plans/<slug>-graph.md` (graph-dba convention).
- **Scope sketch (to be designed, not decided here):** first slice = extract read/write sets from
  `cmp` guard paths (`ctx.`/`output.` roots) + `config.expects` at publish → materialize
  `(:Step)-[:READS]->` / `(:Step)-[:WRITES]->` a key node → add the **dangling-read** and
  **unreachable-step** publish validations (closes n-3 the graph way). `llm`-guard reads and the
  change-impact query are follow-on slices. No DDL beyond one node label + its index; no rule-8
  violation; no run-side cost.
- **Relationship to neighbours:** complements **K-031** (that exposes def *structure* for reading;
  this *analyzes* it), and overlaps **K-029**'s symmetric-`decision` invariant (K-029 proposes the
  rule; K-032 proposes the graph mechanism that could enforce it). Not an M3-green gate — M3 is ✅.
- **Risks/RAM:** negligible on the def side (see the static-only constraint). Real risk is *scope
  creep* toward a general expression/data-flow engine — the `expr` seam stays a `NotImplementedError`;
  this rides the existing closed `cmp` family only.
- **Test strategy:** offline contract tests — a published def reads back with the exact READS/WRITES
  overlay for its guards/`expects`; a def with a guard reading an unwritten key is **rejected at
  publish**; an unreachable step is **rejected at publish**; a step whose reads can't be derived
  carries `READS_UNKNOWN` rather than silently claiming zero reads.

### K-033 — Make `maxSteps` an exact cap (`>` → `>=` in `_drive_loop`) (🔵 proposed — filed out of K-031, stakeholder decision OQ-1 "document now, fix later", 2026-07-24)

> **Why it exists.** `maxSteps` does not mean what its name says. `executor._drive_loop` records a
> step and *then* checks `rec["stepCount"] > max_steps` — at `executor.py:410` (OUTCOME A, a guard
> fired) and `:427` (OUTCOME C, a legitimate self-loop). With `maxSteps: 2`: step 1 → `1 > 2` false;
> step 2 → `2 > 2` false; **step 3 runs**, `3 > 2` → fail. So the budget means *"at least N, then one
> more"*, and a run executes at most **`maxSteps + 1`** steps. Confirmed by reading and **pinned by a
> passing test** — `tests/test_executor.py:158`, `assert len(trail) == 4  # maxSteps=3 → the 4th
> advance trips the guard`. Harmless for the two proof defs (8/6/6 steps against `maxSteps: 24`), but
> it makes `maxSteps` unusable as an SLA or a cost budget, which is exactly what a caller reaches for
> it for. K-031 shipped the **documentation** of the real semantics at six sites (DESIGN §6,
> QUERIES §12.5 + the two `$maxSteps` comments, `schemas.py`, `AGENTS.md`'s executor-invariants
> block) per the binding stakeholder decision; this item is the **fix**.
- **The change is two characters, in two places** — `>` → `>=` at `executor.py:410` and `:427`.
  Everything else about this item is ceremony, and the ceremony is the reason it was deferred out of
  an observability slice rather than the difficulty of the edit.
- **Both sites are *inside* the SHA-locked `_drive_loop`** (`71055f756280`, still live on the
  tree — recompute with the DESIGN §6.2 recipe). Landing it therefore costs:
  - a lock break + **re-lock ceremony**: recompute the SHA, then re-quote it everywhere it is
    asserted. Scope it with `grep -rn 71055f756280` rather than a count here — the number only
    grows, and it is now well past forty sites across plans, reviews and test plans;
  - **records that assert the SHA was unchanged during their own work must not be rewritten** —
    every `archived` plan/review that quotes it is a historical claim about its own delivery, true
    when written. The re-lock has to be expressed as *"as of K-033 the lock is `<new>`; earlier
    records quote the pre-K-033 value"* — i.e. the lock stops being a single grep-able constant.
    **Decide that framing before editing**; it is the item's only real design question.
  - **test edits**: `tests/test_executor.py:142-158` (the pinned count 4 → 3, and its explanatory
    comment), plus a sweep of `tests/test_process_flow.py`'s step accounting and the
    `access-request@v1` `maxSteps: 24` headroom;
  - **behavioural blast radius**: every existing run's effective budget shrinks by one.
- **There is no bundling opportunity left — settled 2026-08-25.** This item was filed hoping to
  ride the next change that legitimately broke the `_drive_loop` lock, plausibly K-027 item 2 (the
  terminal-node-must-post engine contract). K-027 closed 2026-08-21 and **item 2 shipped without
  breaking the lock** — it was implemented at `_run_agent_node`, outside it, exactly as this item's
  original scepticism predicted. K-033 now pays the full re-lock ceremony on its own, or the honest
  `maxSteps + 1` prose stays in six documents indefinitely. That is the decision to make; nothing is
  waiting on anything else.
- **Also decide (part of the item, not a separate one):** whether the *park* path (OUTCOME B,
  `executor.py:415-421`) and the terminal path stay deliberately unchecked. They are unchecked today
  by design — a parked run cannot self-drive — and K-031's documentation says so explicitly, so
  changing that is a second semantic decision, not a consequence of `>=`.
- **Owner:** **`tdd-engineer`** (a behaviour change pinned by an existing passing test is the
  test-first shape: flip the assertion red, then the operator) — with an **`architect`** call first
  **only** on the re-lock framing above, if the coordinator wants the archive-document question
  settled before code moves. No `graph-dba` gate: no Cypher, no DDL, no index.
- **Risks/RAM:** **none** — no new node type, label, property, index or vector dimension; no query
  changes. The risk is purely behavioural (every run's budget shrinks by one) and procedural (the
  lock/archive framing).
- **Test strategy:** flip `tests/test_executor.py:142-158` to assert exactly `maxSteps` advances
  (3, not 4) and watch it go red before the fix; add the boundary case at `maxSteps = 1`; assert the
  park path is still **not** budget-checked (a parked run at `stepCount == maxSteps` must stay
  `waiting`, not fail); re-run `tests/test_process_flow.py` and the `access-request@v1` acceptance
  flow to confirm the `maxSteps: 24` headroom still covers it.

### K-035 — An argument key named `name`/`action`/`tool` shadows the bare call's own name (🔵 proposed — filed out of the K-027 slice A analyst gate, finding M-2, 2026-07-24)

> **Why it exists.** In `llm._parse_content_tool_calls` the **JSON probe runs before** the bare-call
> probe, and `_normalize_tool_call` maps `name` / `action` / `tool` loosely. So for a bare call whose
> *argument object* happens to carry one of those keys, the argument is mistaken for the call
> envelope and the real call name is never seen:
>
> | model emits | parsed as |
> |---|---|
> | `create_user({"name": "bob"})` | `ToolCall(name='bob', arguments={})` |
> | `run_tool({"action": "delete"})` | `ToolCall(name='delete', arguments={})` |
> | `x({"tool": "y", "args": {"a": 1}})` | `ToolCall(name='y', arguments={"a": 1})` |
>
> Reproduced by `analyst` at the K-027 slice A gate; the three rows above are verbatim from
> [`docs/reviews/k027-parse-robustness.md`](reviews/k027-parse-robustness.md) **M-2**.
- **Not currently reachable, and that is the whole risk.** No registered tool declares such a
  parameter — `post_message` takes `text`/`mentions`, `graphrag_retrieve` takes `query`,
  `human_handoff` takes `reason` (`server/falkorchat/tools.py`). The premise was verified at the
  gate and **deferring the fix was judged correct** for a slice whose point was not to widen
  precedence. It is filed as its own item because the *shape* of the failure is bad, not because the
  likelihood is high: it manufactures a tool call **named after a user-supplied value**, so it fails
  **silently and misleadingly** — `executor._handle_tool_call`'s AC-6 check rejects `'bob'` as an
  ungranted tool, burns a re-prompt iteration, and tells the model its own argument is not a tool.
  That trace is close to undebuggable, and a bullet under "candidate follow-ups, small" would not
  have surfaced at the moment someone registers `create_user(name)`.
- **Tripwire in place (do not remove).** `server/falkorchat/llm.py` carries a comment at the
  probe-order site naming this item, because that is where the next author will be looking. **Read
  this item before registering any tool with a `name`, `action` or `tool` parameter.**
- **Owner:** **`architect`** to pick the remedy (it is a precedence decision, not a bug fix), then
  **`tdd-engineer`**. Candidates, cheapest first: (1) in `_normalize_tool_call`, skip the loose
  `name`/`action`/`tool` mapping when the surrounding content also matches `_BARE_CALL_OPEN` — a
  partial hardening available today; (2) run the bare-call probe **first**, which reorders the
  content fallback and needs its own regression pass over the JSON shapes; (3) pass the granted tool
  names down as a **recognition filter** — this also closes most of M-1's residual and the
  `Summary ({"a": 1})` class, but it is a real layering decision (`llm.py`'s note that name
  validation belongs to the agent loop is deliberate, and worth keeping unless consciously reversed).
  Open question 3 of the K-027 gate review is exactly this choice.
- **Risks/RAM (rule 6):** none — parse layer only, no node type, index, property or vector dimension.
  No Cypher, no schema, no script.
- **Test strategy:** offline pins in `server/tests/test_llm.py` driving the public `llm.chat(...)`
  seam (never the private probes). Pin all three rows in the table above to the **call's own**
  identifier; pin that a genuine `{"name": …, "arguments": …}` envelope with no surrounding call
  expression still parses as a call (the shape this must not regress); and pin the negative
  direction — an argument object carrying `name` must not resurrect a call the M-1 final-content rule
  rejected.

### K-038 — `refreshRunPanel` has no mutex against overlapping poll-tick/submit-response invocations (🔵 proposed — filed out of K-036's Wave 3+4 analyst re-review gate, `docs/reviews/web-api-coverage-impl.md` Pass 3, findings m6/m7, 2026-07-29)

> **Why it exists.** Pass 3 fixed M1 (the destructive every-tick `renderWaitingForm` rebuild) and,
> while deep-tracing the fix per the task's own request, found two narrower, non-blocking races in
> the same run-panel poll/submit machinery — recorded there rather than chased at gate time, and
> filed here as this item per the K-036 close-out plan.
- **m6 — unordered concurrent `refreshRunPanel` calls; a stale response can transiently overwrite
  a fresher one.** `refreshRunPanel` (`web/app.js`) is not mutex-protected against overlapping
  invocations, and both `startRunPolling`'s periodic tick and `submitRunInput`'s post-submit call
  invoke it independently. If a tick's `refreshRunPanel` is still in flight
  (`Promise.all([GET run, GET step-runs])` awaiting) at the moment a submit resolves and triggers
  its own `refreshRunPanel`, both fetches race against the same `runId`, and whichever's
  `Promise.all` resolves *last* wins the render — regardless of which is fresher. A stale
  (pre-submit) response landing after the fresh one briefly shows an already-superseded step.
- **m7 — the same-step-key rebuild guard can't distinguish "still the same wait" from "revisited
  after an externally-driven round-trip this session's poll never observed."** The
  `state.runWaitingKey` guard (M1's fix) only resets when *this session's own* render observes a
  non-`waiting` status. If some other actor completes a full `waiting → running → waiting`
  round-trip on the *same* `atStepKey` entirely between two of this session's poll ticks, the key
  still matches on the next tick and the box stays hidden behind the early-return guard even
  though it's a genuinely new visit to that step.
- **Non-blocking, self-healing — carry this framing accurately.** Both require timing conditions
  well outside normal single-operator, human-typing-speed form use (a sub-`POLL_MS`, i.e. <3s,
  window for m6; an external actor plus a round-trip faster than one `POLL_MS` for m7), and both
  self-heal within one more `POLL_MS` (≤3s) tick — the next poll fetches true current state again
  and forces a correct rebuild either way. Neither loses user-entered input (unlike M1); neither is
  a permanent inconsistency.
- **Owner:** **`frontend-engineer`**, `web/app.js`.
- **Scope sketch (to be designed, not decided here):** m6 — a request-sequence token (stamp each
  `refreshRunPanel` call with an incrementing counter, ignore a response whose stamp is behind the
  latest issued) would close it; m7 — needs a server-supplied, monotonically-changing identifier
  per wait-instance (not just `atStepKey`) to distinguish "unchanged wait" from "revisited wait",
  which is a server-side surface change, not JS-only.
- **Risks/RAM:** none — web-only, no server/graph surface touched by m6; m7's full fix would need a
  new/changed field on the run-detail response, scoped by whoever picks this up, not decided here.
- **Test strategy:** a DOM-stub harness (the same shape Pass 2's fix verification used) driving two
  overlapping `refreshRunPanel` promises resolving out of order, asserting the later-issued one
  always wins regardless of resolution order (m6); an injected-clock or mocked-response test
  simulating an external round-trip between two ticks, asserting the panel still rebuilds (m7).

### K-040 — `POST /workflow-runs`'s request field is `version`, not `defVersion` — decide whether to rename for consistency (🔵 proposed — found during a `tico` manual-verification pass, 2026-07-31)

> **Why it exists.** `StartWorkflowRunIn` (`server/falkorchat/schemas.py:198-202`) declares the
> field as `version`, while the rest of the def/snapshot vocabulary — `WorkflowRun.defVersion`
> (DESIGN §6.2), the `GET /workflow-defs/{key}/versions/{version}` path segment name notwithstanding,
> and the general "def key + version" phrasing throughout `DESIGN.md` §14.4/§12 — uses `defVersion`.
> The mismatch is easy to get wrong by pattern-matching the graph model rather than the actual
> schema: it produced a real `422 Unprocessable Entity` in a first draft of
> `falkor-chat/docs/manuals/workflows.md`'s API walkthrough, caught only because a `qa-engineer`
> verification pass actually called the endpoint rather than composing the example from the schema
> next to the design doc. Already fixed in the manual (`docs/manuals/workflows.md`, 2026-07-31); this
> item is about the underlying inconsistency, not the doc.
- **The actual decision, not pre-judged here:** either (a) rename `StartWorkflowRunIn.version` →
  `defVersion` for consistency with the rest of the surface — a live API contract change, so it
  needs an assessment of what currently depends on `version` (the web UI does not call this route
  today per `docs/manuals/workflows.md`'s Walkthrough 4 note that it's API-only; MCP tools don't
  cover workflow runs either per DESIGN §15.2 — so the blast radius may be small, but that needs
  confirming, not assuming) — or (b) leave the field as-is and treat this as a documented,
  intentional naming divergence (a one-line callout at `DESIGN.md` §14.4 / `QUERIES.md` §12.12 would
  suffice). Both are legitimate; this item exists so the choice is made deliberately rather than by
  the next person guessing again.
- **Owner:** **`architect`** — assess the rename's blast radius and decide (a) vs (b); if (a),
  **`coder`**/`tdd-engineer` implements + updates every call site and doc reference.
- **Scope:** grep every caller of `POST /workflow-runs` (tests, scripts, docs, the manual) before
  deciding; if renaming, land the schema change, the `test_process_flow.py`/`test_services.py` call
  sites, and the `docs/manuals/workflows.md` example in the same change.
- **Risks/RAM:** none — no graph/DDL surface; purely a request-schema field name. The only risk is a
  silently-missed caller if the rename is chosen without the grep above.
- **Test strategy:** if renamed, flip the existing `test_process_flow.py` start-run assertions to the
  new field name (should go red first, confirming the old name is actually gone) plus a negative test
  that the old `version` name is rejected, not silently ignored.

### K-043 — `compose.yaml`/`Dockerfile` never verified against a real `docker build`/`docker compose` (🔵 proposed — filed out of K-042 close, 2026-08-11)

> **Why it exists.** K-042 Landing 1's L1-5 unit updated `compose.yaml` (the two config-file paths,
> a read-only bind mount of the shared overlay file, `host.docker.internal:host-gateway`) and
> `Dockerfile`-adjacent run instructions on the strength of static review only — no Docker toolchain
> was available anywhere in the coordination pipeline (agents, gates, or the QA acceptance pass), so
> the change was never exercised by an actual `docker build` / `docker compose up`. The risk is
> narrow (a bind-mount path typo or a missing `host.docker.internal` extra_hosts entry would only
> surface in a real container run) but real, and it's the one surface in K-042 that shipped unverified.
- **Owner:** **`devops`** — build the image, bring the stack up via `compose.yaml`, and confirm the
  server inside the container can resolve both config-file paths and reach LM Studio on the host via
  `host.docker.internal`.
- **Scope:** `docker build` against `falkor-chat/Dockerfile`; `docker compose up` against
  `falkor-chat/compose.yaml`; confirm the bind-mounted shared overlay file is readable at the path
  the container expects, and that a chat request round-trips to the host LM Studio instance.
- **Risks/RAM:** none — verification only, no design change expected unless the run surfaces a defect.
- **Test strategy:** a live manual run is the test; if it surfaces a defect, file a fix as its own
  follow-up rather than folding it into this item.

## Deferred — M2.5 hardening track

Auth + real-time. Not on the M5 path; no scheduled start.

### K-016 — Real auth/tenancy replacing the hardcoded `get_context` seam (🔵 proposed — M2.5, deferred)

- **Owner:** **`architect`** (design pass — designs the auth mechanism *per* the authoritative-identity decision, now
  resolved: the `identity` graph is authoritative/standalone, DESIGN §1.2) → **`tdd-engineer`** (implement the resolved `get_context`).
- **Inputs/prereqs:** the identity source-of-truth is **decided** (identity graph authoritative/standalone; DESIGN §1.2) —
  K-016 no longer needs the user for that axis; it implements per that decision. Localized by design — only
  `config.get_context` changes (`config.py:43`); everything below already parameterized on `ws`/`actor`.
- **Scope:** token → (user, workspace claim) resolution replacing hardcoded `ws=acme/user=u1`; wire the `identity`
  graph per the §1.2 authoritative-identity decision; keep or replace MCP's `frm`-ignoring rule with authenticated agent identity.
- **Done-condition:** `get_context` resolves a real principal from a credential; multi-tenant isolation test; pytest green.
- **Risks/RAM:** `identity` graph nodes (small). First real trust boundary — MCP endpoint is currently unauthenticated (§15.3).
- **Test strategy:** service/api tests with injected auth contexts; a cross-tenant isolation test.

### K-017 — Transport-level agent-actor path (K-007 QA carry-over) (🔵 proposed — M2.5, deferred · depends on K-016)

- **Owner:** `qa-engineer` (+ small `tdd-engineer`/`coder` fold-in if MCP must express an authenticated agent actor).
- **Scope:** with auth able to express an *agent* principal, drive an external agent authoring over MCP/REST (the M1
  hardcoded seam couldn't) and verify authorship/role/provenance end-to-end.
- **Done-condition:** the K-007 QA carry-over closed — a report showing an externally-authenticated agent authoring
  first-class over the transport.
- **Risks/RAM:** none new. **Test strategy:** black-box over MCP with an agent credential.

### K-018 — Real-time push (Redis Pub/Sub → WebSocket/SSE) (🔵 proposed — M2.5, deferred)

- **Owner:** **`architect`** (design: Pub/Sub fan-out topology; resolve the DESIGN §13 Bolt-vs-RESP gateway question
  here since it touches the transport) → **`coder`/`tdd-engineer`**.
- **Inputs/prereqs:** K-012/K-014 web client (swap polling → push).
- **Scope:** Redis Pub/Sub on message write → WebSocket/SSE endpoint on the same FastAPI process (§14.1: "slots onto
  the same service layer, no schema change") → web client subscribes instead of polling.
- **Done-condition:** a posted message appears in another client without a poll; graceful fallback to polling.
- **Risks/RAM:** no graph RAM; Pub/Sub is transient. Publish *after* the guarded §4 write commits, never inside it (atomicity rule).
- **Test strategy:** integration test of publish-on-write + a WebSocket client receiving it.
- **Related work (client-side polling alternative):** `mcp-monitor/` (`mcp-monitor/docs/requirements/mcp-monitor.md`) has shipped as a separate, polling-based watcher that detects MCP tool-result changes and launches commands — a distinct, complementary approach to K-018's server-side push. K-018 remains its own open item.

## Plan docs still to author

Every other plan named by a delivered item now exists; this is what is left.

| Path | For | Scope |
|---|---|---|
| `docs/plans/auth-tenancy.md` | K-016 (deferred) | Real auth replacing the `get_context` seam, per the §1.2 identity-authoritative decision. |
| `docs/plans/realtime-push.md` | K-018 (deferred) | Pub/Sub → WebSocket/SSE, resolving DESIGN §13's Bolt-vs-RESP question. |

> Both slugs were previously listed as `m2-auth-tenancy.md` / `m2-realtime.md`. Renamed here before
> creation: root `AGENTS.md`'s filename grammar forbids a new document's basename beginning with
> `m<digit>`, and neither topic *is* a milestone.

## Parking lot / ideas

- Verify the K-009 GitHub Action goes green on first push (path-filtered `.github/workflows/falkor-chat.yml`; FalkorDB
  service container). Note the CI baseline echoes in its comments (75/92) predate K-007/K-010's 110/126 — the suites
  themselves are the source of truth. (K-019 fixes the README/DESIGN body numbers; the CI comments are separate.)
- File upstream FalkorDB issues (K-007 OQ6, recommended to the user): `GRAPH.MEMORY USAGE` under-reports vector-index
  memory; one-shot instant-timeout anomaly after a long override run.
- Per-endpoint response schemas (QA, recommended three times now): full-thread / since-reads / search each carry a
  different field subset (all documented/intentional) — a declared schema per endpoint would make the contract testable
  and stop accretion. **The repo is deliberately on a mixed convention today.** The three §11 structure/diff routes
  declare `response_model=` (`WorkflowDefStructureOut` / `WorkflowDiffOut`, `schemas.py`) with exact-key-set contract
  tests; no pre-existing route does. The non-retrofit is deliberate, not pending — FastAPI's `response_model` *filters*
  undeclared fields, so a wrong model silently drops a field the web client reads, which makes a bulk retrofit riskier
  than the accretion it would stop. The §11 routes are the worked precedent for an eventual per-route retrofit.
- **Opportunistic nit — re-slug the K-031 implementation review** (recorded, **not** scheduled work).
  It is filed under the slug `k031-structure-read-impl`, while the rest of its family — the plan and
  the plan review — uses `workflow-def-structure-read`. The filename grammar's family rule (*the same
  slug across several kinds **is** the family; a downstream document inventing a new slug is a
  defect*) is therefore broken by one member. Correcting it to `workflow-def-structure-read-impl`
  spans **four files** as of 2026-08-25 (this backlog, `docs/HISTORY.md`,
  `docs/plans/m3-followups-coordination.md`, `docs/plans/workflow-republish-semantics.md`) — check
  with `grep -rln k031-structure-read-impl docs/` rather than trusting that count. Fold it into a
  change that already opens them; it does not earn a change of its own, and renames in this repo are
  forward-only by ruling.
- DESIGN §13 remaining open questions — resolve as their milestones arrive: real auth (K-016),
  message/embedding retention, cross-workspace analytics, Bolt vs RESP for the gateway (K-018).
  (The workflow guard expression language is **no longer** among them — resolved at DESIGN §6.1,
  and §13 records it struck through.)
- **WSL2 memory cap for the 16GB host** (parked, not applied per user 2026-07-18) — WSL2 runs uncapped at its 8GB
  default (50% of the 16GB host) with `autoMemoryReclaim` off, overcommitting host RAM alongside Windows-side LM Studio;
  likely root cause of the recent memory-overload crashes. Parked fix: set `memory=6GB` + `swap=4GB` +
  `autoMemoryReclaim=gradual` in `C:\Users\mauri\.wslconfig` (keep `networkingMode=mirrored`), then `wsl --shutdown`.
  Full diagnostic + apply procedure: `docs/plans/wsl2-memory-diagnostic.md`. Un-park (apply) if the crashes recur —
  verdict was confirmed-by-defaults, not reproduced live (FalkorDB was down during the diagnostic).
