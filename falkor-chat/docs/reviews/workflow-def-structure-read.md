# Plan review — K-031 def/snapshot **structure** read surface

> **Reviewer:** `analyst` · **Date:** 2026-07-24 · **Artifact:**
> `docs/plans/workflow-def-structure-read.md` (architect, design complete, awaiting this gate).
> **Baseline tree:** `2ee6eba`, working tree carrying the concurrent K-027 slice A churn
> (`server/tests/test_app.py`, `server/tests/test_llm.py`, `claude/architect/kaizen/inbox.md`) —
> disregarded per the brief.
> **Gate:** U2-G1, pre-implementation plan gate (`docs/plans/m3-followups-coordination.md`).
> Static review — nothing executed that mutates state.

## 1. Scope & verdict

**Reviewed:** the plan end to end against the live tree —
`server/falkorchat/{repository,services,api,schemas,executor,config,db,guards}.py`,
`server/tests/{conftest,test_api,test_executor,test_app}.py`,
`scripts/{bootstrap_schema,seed_workflows,test_queries}.sh`,
`docs/{DESIGN,QUERIES,BACKLOG,HISTORY}.md`, `docs/archive/test-reports/m3-workflow-engine-report.md`
(§5 DEF-1, §7 feedback 1 and 5), `falkor-chat/AGENTS.md`, and the coordination doc.
**Every file:line citation in plan §2 and §5 was checked individually** — all of them resolve exactly
(`repository.py` 937 / 961 / 968 / 976 / 1025 / 1035 / 1470 / 1498; `services.py` 663 / 687;
`api.py` 228 / 239 / 304; `app.py:86`; `executor.py` 410 and 427; `tests/test_executor.py` 142–158;
`schemas.py` 84–86; `BACKLOG.md:793`; `AGENTS.md:256`; `HISTORY.md` 144 / 291). This is the
best-grounded plan I have reviewed in this component.

**Not verified / not verifiable statically:** the `server` pytest and `test_queries.sh` counts
(both suites wipe the global `reference` graph — running them is a shared-state mutation, so I did
not); V-1's live outcome (the plan schedules it deliberately); the actual live `triage@v1` /
`access-request@v1` def-vs-snapshot state (R-1). `pytest --collect-only -q` (non-mutating) reports
**551/552 collected, 1 deselected** on the current working tree — consistent with the 533 baseline
plus the ~18 tests the concurrent U1 has added.

**Verdict: needs changes.**

Counts: **2 blocker · 3 major · 6 minor · 4 nit.**

The design is right and I would approve it on substance. Both blockers are **scoping/handoff
corrections that cost a paragraph each**, not a design rethink:

- The plan's *first scheduled action* (V-1) is unexecutable as written — `publish_def` has no graph
  seam, so "an isolated throwaway workspace" cannot exist for a def publish (**B-1**).
- The plan discovers a genuine, previously-undocumented **live defect** (finding 2, which I
  confirmed), correctly declines to fix it, and then **files nothing** — while filing K-033 for a
  strictly less dangerous off-by-one. Its own §1.2 rule ("if the read exposes something ugly, the
  deliverable is a backlog entry") is not applied to its own finding, and §8's doc list leaves ten
  shipped assertions of the now-falsified claim in place (**B-2**).

The plan's three load-bearing claims — "no new Cypher", the additive `MERGE`, and the consequent
`start.key` ambiguity — are all **correct**; see §3.

---

## 2. Findings

### Blockers

#### B-1 · V-1 cannot be run as specified — a def publish has no workspace seam

**Evidence.** Plan §7 V-1: *"In an isolated throwaway workspace (**never** `reference`/`ws:acme`),
publish a two-step def, then re-publish the same `(key, version)` with a different start step."*

`Repository.publish_def` (`server/falkorchat/repository.py:1011`) writes to `self._reference()`
(`:132-134`) → `db.reference_graph()` (`server/falkorchat/db.py:87-94`), which is
`db.select_graph("reference")` — a **hardcoded literal, with no parameter, env var or config
override** anywhere in `config.py` or `db.py`. There is no such thing as a per-workspace def
publish. An implementer following V-1 literally will discover this in minute one and improvise;
the obvious improvisation is *"just publish a probe key into `reference`"*, which is precisely what
the plan forbids two lines earlier and what R-1/§7 step 20 (*"leave the environment seeded"*) is
guarding against.

**Why it matters.** V-1 gates U1 (§10: *"Run V-1 first"*) and de-risks R-2 and finding 3. A gating
step whose stated preconditions are impossible either gets skipped (the design ships on an
unverified premise) or gets improvised against the live global graph.

**Suggested improvement.** Run V-1 on the **snapshot side**, which is byte-identical Cypher and is
genuinely workspace-scoped. `materialize_snapshot` (`repository.py:1483`) formats *the same*
`_PUBLISH_CYPHER` constant with `label="WorkflowDefSnapshot"` against `self._graph(ws)`, and
`_READ_META_CYPHER` is the same constant formatted with the same label. Concretely:

1. `./scripts/bootstrap_schema.sh k031probe` (the `Step.stepUid` index + constraint must exist —
   AGENTS.md: every `MERGE` is constraint-backed);
2. two `repo.materialize_snapshot("k031probe", key="probe", version="v1", …)` calls differing only
   in `start_key`;
3. read the raw `result_set` of `_READ_META_CYPHER.format(label="WorkflowDefSnapshot")` — count the
   rows;
4. `GRAPH.DELETE ws:k031probe`.

Because the query text is identical, the answer transfers to the `WorkflowDef` side unchanged.
State that transfer explicitly in the plan so the implementer does not re-litigate it.

#### B-2 · Finding 2 is confirmed, invalidates ten shipped assertions, and is filed nowhere

**Evidence — the finding is true.** I traced `_PUBLISH_CYPHER` (`repository.py:937-956`):

- `MERGE (st:Step {stepUid: $key+':'+$version+':'+s.key}) … MERGE (d)-[:HAS_STEP]->(st)` — a new
  step key mints a new `Step` + edge.
- `MATCH (start:Step {stepUid: … + $startKey}) MERGE (d)-[:START]->(start)` — the `MERGE` pattern
  includes the endpoint, so a **changed start key creates a second `START` edge** beside the old one.
- `MERGE (from)-[rel:TRANSITION {on: tr.on, order: tr.order}]->(to)` — a changed `to`, `on` or
  `order` mints a **parallel `TRANSITION` edge**; only `guard` is a true no-op
  (`ON CREATE SET rel.guard`).

`materialize_snapshot` reuses the same constant (`repository.py:1483`), so the snapshot side is
additive too.

**Evidence — the consequences are worse than "the executor drives both edges" (plan §0.2).**

1. **Nondeterministic branch selection.** `executor._select_transition` sorts by
   `(guard == "", order)` and takes the first firing. Two `TRANSITION` edges with the same `on` and
   `order` but different `to` sort equal; Python's sort is stable, so the winner is *the order
   FalkorDB returned the edges in*. A re-published edge silently makes the def's routing
   nondeterministic.
2. **A two-`START` snapshot breaks run start outright.** `repository.start_run`
   (`repository.py`, §12.1) is
   `MATCH (snap:WorkflowDefSnapshot {…})-[:START]->(start:Step) MATCH (trigger:Message {…})
   CREATE (r:WorkflowRun {runId: $runId, …}) …`. Two `START` edges ⇒ two rows ⇒ the `CREATE` runs
   twice against `UNIQUE NODE WorkflowRun PROPERTIES 1 runId`
   (`scripts/bootstrap_schema.sh:180`) — the start errors, or (without the constraint) mints a
   duplicate run with two `AT_STEP` edges.

Both are reachable by the operation AGENTS.md actively encourages (*"re-run `seed_workflows.sh`
after `test_queries.sh` or a pytest run"*) combined with a def edit — exactly the K-029 scenario.

**Evidence — the doc surface now lies, in ten places, and §8 lists none of them.** R-8 claims
*"§7's list is exhaustive"*. It is not:

| Site | Assertion falsified by finding 2 |
|---|---|
| `docs/QUERIES.md` §11 preamble | "re-running the same `key@version` is a structural no-op (0 nodes/rels created). **Immutability per version comes for free from `MERGE`**" |
| `docs/QUERIES.md` §11.1 footnote | "run 2 → 0 created (idempotent), same row" |
| `docs/QUERIES.md` §11.4 footnote | "Snapshots are **immutable** per `(workspace, key, version)`; re-materialize is a no-op" |
| `docs/DESIGN.md:544` | "Publish workflow def … **Immutable per version**; bump version, never mutate in place" |
| `docs/DESIGN.md:102` / `:144` / `:147-149` | "versioned, immutable" / "the snapshot is immutable so it never…" |
| `repository.publish_def` docstring (`:1003-1009`) | "re-publishing the same `key@version` is a **structural no-op** (immutability per version)" |
| `repository.materialize_snapshot` docstring (`:1481`) | "Re-materialize is a no-op" |
| `services.materialize_def` docstring (`:659`) | "Idempotent (the workspace `MERGE` no-ops on re-materialize)" |
| `falkor-chat/AGENTS.md`, `seed_workflows.sh` row | "⚠️ Published defs are effectively IMMUTABLE … re-running changes nothing live" |
| `docs/BACKLOG.md` K-029 premise | "a byte-diff introduced while relocating a **live** def is **silently swallowed**" |

The K-029 row is the dangerous one: K-029's entire risk framing — and therefore its chosen
mitigation — assumes a relocation typo is *swallowed*. Under finding 2 a typo'd step key or
transition endpoint is **additive**, i.e. K-029's proposed move is more dangerous than K-029
believes, and its "before/after equality check on the published def subgraph" acceptance criterion
becomes load-bearing in a way nobody has told its owner.

**Why it matters.** Once U3 ships and this run closes, the finding lives in a plan doc and a test
comment. The plan files **K-033** for a two-character off-by-one with no live impact, and files
**nothing** for a defect that can non-deterministically re-route a live workflow and break run
start. That asymmetry is the finding.

**Suggested improvement (all plan edits, no scope change to K-031's build):**

1. Add to §8's `docs/BACKLOG.md` row: **file a new proposed item — "create-only re-publish is
   *additive*, not a no-op"** — carrying the three `MERGE` shapes, the two executor consequences
   above, and the note that the cheap remedy (`ON MATCH SET` / a pre-publish structural equality
   check / rejecting a re-publish that differs) is a **publish-semantics** change and therefore
   belongs there, not in K-031.
2. Add a `docs/BACKLOG.md` line under **K-029** correcting its "silently swallowed" premise and
   pointing at the new item — K-029 is the item most likely to trip over this.
3. Extend §8 with the ten sites above (the three `QUERIES.md` §11 notes, three `DESIGN.md` lines,
   three docstrings, the AGENTS.md row). Correcting a docstring is not a behaviour change and does
   not touch `_PUBLISH_CYPHER`; §3.6's "left byte-identical" constraint is about the query, not the
   prose above it. Suggested wording: *"create-only on **properties**, additive on **structure** —
   a re-publish never updates, but it does add."*
4. Amend R-8 to drop the "exhaustive" claim, or make exhaustiveness a grep-verified done-condition
   (`grep -rn -i "immutab\|no-op" docs/ server/falkorchat/` over the §11 surface).

**Not** requested: any change to publish behaviour, or to `_PUBLISH_CYPHER`. K-031 stays read-only.

### Majors

#### M-1 · The publish receipt's counts are the *submitted* spec, not the stored def — §3.2's comparability claim inverts

**Evidence.** `_PUBLISH_CYPHER` computes `WITH d, count(st) AS stepCount` immediately after
`UNWIND $steps` and `count(rel)` after `UNWIND $transitions` — one row per **unwound input
element**, so the returned `stepCount`/`transitionCount` equal `len($steps)`/`len($transitions)`,
never the stored subgraph's size. Plan §3.2: *"`stepCount`/`transitionCount` … named to match the
existing publish/materialize response fields, **so an operator can compare a publish receipt
against a read without a mental translation**."*

**Why it matters.** In the *only* case that matters — the additive re-publish of finding 2 — the
receipt says `6` and the structure read says `7`. An operator told the two are directly comparable
will read the mismatch as an endpoint bug rather than as the split-brain signal it is. The plan
gets the framing exactly backwards on its own headline scenario.

**Suggested improvement.** Keep the names (they are right), and replace the rationale with the
truth, in §3.2 and in the DESIGN §14.4 paragraph: *"the publish/materialize receipt counts what was
**submitted**; the structure read counts what is **stored**. A divergence between them is the
additive-re-publish tripwire, not a bug."* That single sentence is arguably the most useful
operator-facing line in the whole deliverable.

#### M-2 · The K-033 bundling argument rests on an unverified assumption about an unscheduled item

**Evidence.** Plan §5 / OQ-1 defers the `>` → `>=` fix to *"the next item that legitimately breaks
the `_drive_loop` lock, i.e. **K-027 item 2** (the terminal-node-must-post engine contract), **which
cannot avoid touching the loop anyway**."*

Two problems. First, the "cannot avoid" is asserted, not shown: the lock covers `_drive_loop` only —
`_execute_step`, `_select_transition`, `_trace_step` and `resume` are explicitly **outside** it
(`AGENTS.md:256-257`). Terminality (`outgoing.get(current_key)` falsy) is *known* in the loop, but a
terminal-post guarantee could plausibly be implemented by passing that fact into `_execute_step` or
by a post-step seam — i.e. the premise may not hold. Second, `docs/BACKLOG.md:315` shows K-027 is
**🔵 proposed**, six items wide, with items 2–5 explicitly out of scope for this run
(`m3-followups-coordination.md:7-8`) and no scheduled date. If item 2 slips or lands without
breaking the lock, K-033 orphans indefinitely — while the plan has meanwhile written the honest
`maxSteps + 1` semantics into six documents, which is the *cheap* half of a permanent divergence.

**Why it matters.** The recommendation (document now) is right; the *justification* is the weak
part, and it is the justification the stakeholder is being asked to approve in OQ-1.

**Suggested improvement.** File K-033 as **self-standing** ("make `maxSteps` an exact cap"), with the
bundling stated as a *preference* rather than a precondition: *"prefer to land alongside the next
item that breaks the `_drive_loop` lock; if none arrives, K-033 breaks it on its own — the
re-lock ceremony is the cost either way."* And soften §5's claim about K-027 item 2 to "expected to
need a `_drive_loop` change (unverified)".

#### M-3 · The `533 → 533 + N` gate is already stale by construction

**Evidence.** Plan §1.3 and §7 make *"`server` pytest **533 → 533 + N**"* a done-condition. U1
(K-027 slice A) is running **in parallel** on the same tree (`m3-followups-coordination.md:30-32`)
and adds tests to `tests/test_llm.py` + `tests/test_app.py`. `pytest --collect-only -q` on the
current working tree already reports **551/552 collected**, i.e. +18 from U1's uncommitted churn.
U3 starts *after* U1-G, so the coder's entry baseline will be ~551, not 533.

**Why it matters.** A stated numeric done-condition that is wrong at the moment of use gets either
mechanically "corrected" (fine) or treated as evidence something regressed (not fine) — and the plan
elsewhere leans hard on exact counts as tripwires (`256/256`).

**Suggested improvement.** Change §1.3/§7 to *"re-derive the entry count at U3 start (`pytest
--collect-only -q`, non-mutating) and report `entry → entry + N`; the 533 figure is the pre-U1
baseline."* Leave the `test_queries.sh` **256/256** assertion exactly as it is — that one is
genuinely invariant and is the plan's best tripwire.

### Minors

#### m-1 · The diff is version-qualified, so it cannot answer "is the workspace on a stale *version*"

`GET /workspaces/{ws}/snapshots/{key}/versions/{version}/diff` requires the caller to already know
the version. The AGENTS.md-documented staleness shape it *does* catch is "same version, `reference`
republished after a wipe, snapshot older" — good. It does **not** catch "`reference` now has `v2`,
the workspace only ever materialized `v1`", which is the shape a `key`/`version` bump produces (the
documented way to land a def edit). `verify_workflows.sh` covers this for the two seeded defs
because it reads the expected version from `config`/`proof_defs`, but an ad-hoc operator gets no
signal. **Suggested improvement:** one sentence in the DESIGN §14.4 paragraph — *"the diff is
version-qualified; to detect a stale **version**, compare `GET /workflow-defs` against
`GET /workspaces/{ws}/snapshots` first"* — and the same line in the endpoint's route comment.

#### m-2 · §7 test 16's first alternative is not reachable through the service layer

Plan §7 test 16 offers *"publishing a second, edited def under a different key and materializing it
under the first key's identity"*. `services.materialize_def(ctx, key, version)`
(`services.py:653-671`) reads `reference` at `key@version` and writes the snapshot at the **same**
`key@version` — there is no seam to materialize under another identity short of calling
`repo.materialize_snapshot` with mismatched arguments. The plan's own second option ("materialize
`A@1`, then re-publish `A@1` with an added step") is correct and simpler. **Suggested improvement:**
delete the first alternative; it will cost the implementer a detour.

#### m-3 · DESIGN §14.4's table deliberately excludes the §11 routes — adding only the new three leaves it half-populated

`docs/DESIGN.md` §14.4 carries an explicit parenthetical: *"(The §11 def-authoring routes
`POST/GET /workflow-defs…` and the §12 inspection routes … are also mounted; they are read/publish
paths and are described at their own sections.)"* Plan §8 adds three §11 routes to that table
without addressing the convention. **Suggested improvement:** either add **all** §11 routes in the
same edit (and delete the parenthetical), or follow the existing convention and describe the three
new routes at §11's own section with a one-line pointer from §14.4. Either is fine; silently doing
half is what produces the next doc-drift finding.

#### m-4 · V-1 is a graph-dba-domain question being answered by `coder` with no escalation rule

"No `graph-dba` gate" is the right call for the **query text** (§3, claim 1 — confirmed). But V-1
asks a pure engine-semantics question: *does a non-aggregated grouping key that takes two values
across an `OPTIONAL MATCH` fan-out yield two rows on this build?* That is exactly the class of fact
that lives in `claude/graph-dba/falkordb-quirks.md`, and the plan already routes the answer there.
What is missing is what happens if the answer is **surprising** — one row with an arbitrary
`start.key`, or an error. **Suggested improvement:** add to §7 V-1: *"any outcome other than
'N rows for N distinct start keys' is a stop-and-escalate to teco/graph-dba, not a design
adjustment absorbed by the implementer"* — the same shape as R-4's tripwire.

#### m-5 · `api.py` does not currently import `MAX_KEY_LEN`

U3 step 2 specifies `Path(..., min_length=1, max_length=MAX_KEY_LEN)`. `api.py:18-26` imports only
`MAX_ID_LEN` from `.schemas`. Trivial, but the plan is otherwise precise enough that an implementer
will trust its import-level detail. **Suggested improvement:** note the import addition in U3.

#### m-6 · `_read_structure` returns a shape QUERIES.md §11.2 does not document

The plan has the repository return `start_keys: list[str]` where §11.2's documented contract is a
scalar `startKey`. That is a (small) stretch of "`repository.py` is a 1:1 mirror of QUERIES.md"
(DESIGN §14.2, AGENTS.md layering). §8's QUERIES §11.2 note is the right mitigation and I would not
move the logic. **Suggested improvement:** make the `_read_structure` docstring cite QUERIES.md
§11.2's new note by name, so the divergence is discoverable from the code side too — the plan
already mandates a *why-it-duplicates* comment; extend it to *why-the-shape-differs*.

### Nits

- **n-1 · OQ-2 is not a stakeholder call.** Diff path placement is an internal API-shape decision
  the architect has already made with a sound reason (§3.3, §9). Presenting it at a gate invites a
  round-trip on something the plan itself calls "purely cosmetic". Fold it into §3.3 as a decision
  with the alternative noted.
- **n-2 · §7 calls V-1 "read-only-ish".** It publishes twice. Name it a write, in a throwaway graph,
  torn down — that framing is what makes B-1's fix obvious.
- **n-3 · Front-matter cites the QA report as "§5 / §7.1 / §7.5".** The report has no §7.1/§7.5;
  the referents are §5 (DEF-1) and §7 numbered items 1 and 5. Cite them as "§7 items 1 and 5".
- **n-4 · §7 test 11 re-publishes with a changed `kind`.** Ensure the substituted value is a member
  of the allowed kind set enforced by `services._validate_def_spec`, or the test asserts 400 rather
  than the intended "201 + unchanged read".

---

## 3. Answers to the gate's specific questions

**1. "No new Cypher" — CONFIRMED.** `_READ_META_CYPHER` (`repository.py:961`) and
`_READ_TRANSITIONS_CYPHER` (`:968`) are `{label}`-templated and already formatted with both
`"WorkflowDef"` and `"WorkflowDefSnapshot"` by the single shared `_read_subgraph` (`:976-997`), used
by `read_def_subgraph` (`:1025`) and `get_snapshot` (`:1498`). The plan's `_read_structure` reads
*more rows of the same result set* — Python, not Cypher. No DDL, index, property or query-string
change is implied anywhere in U1–U5. **No `graph-dba` gate is needed and `test_queries.sh` should
stay at exactly 256/256**; R-4's "a delta means new Cypher slipped in" is a correct and valuable
tripwire. The one caveat is m-4: V-1 is an engine-semantics question, and it needs an escalation
rule even though the query text is frozen.

**2. The additive-`MERGE` finding — CONFIRMED, and it is worse than the plan says.** All three
shapes hold (new step key ⇒ new `Step` + `HAS_STEP`; changed `to`/`on`/`order` ⇒ parallel
`TRANSITION`; changed start ⇒ second `START`), traced through `_PUBLISH_CYPHER:937-956` and shared
verbatim by `materialize_snapshot:1483`. Only `guard`, `d.name`, `d.kind` and `st.type`/`st.config`
are genuinely create-only. **A test is not proportionate.** Two live consequences the plan does not
name — nondeterministic branch selection in `_select_transition`, and `start_run`'s duplicate
`CREATE` against the `WorkflowRun.runId` unique constraint — plus ten shipped doc/docstring
assertions that the finding falsifies (including K-029's core risk premise), make this a
**backlog-item-and-doc-correction** finding. See **B-2**. K-031 should still not *fix* it.

**3. The consequent ambiguity claim & V-1's adequacy — claim sound, V-1 not adequately specified.**
The reasoning is correct: `start.key` is a non-aggregated grouping key beside
`collect(DISTINCT …)`, and QUERIES.md §11.2's own footnote states the collapse holds *"because
`start.key` is constant across the fan-out"* — a premise a two-`START` def breaks, at which point
`_read_subgraph`'s `result_set[0]` is arbitrary. Scheduling a live check rather than asserting is
the right call. But V-1 is **not** non-mutating and **cannot** be isolated as written — see **B-1**.
With the snapshot-side reformulation it becomes genuinely isolated and torn down.

**4. Scope discipline — held.** §1.2 is explicit and the six units honour it: nothing touches
`_PUBLISH_CYPHER`, `_validate_def_spec`, `executor.py`, `bootstrap_schema.sh` or `web/`; K-029 and
K-030 are named as out of scope and never re-litigated; §7 test 11 pins create-only as a *decision*
rather than treating it as a bug. The **one** discipline slip is the inverse of drift: knowing about
a live defect and declining to *file* it (B-2), which the plan's own §1.2 rule requires and which it
does correctly for `maxSteps`.

**5. `maxSteps` — recommendation sound, justification weak.** The defect is real and confirmed:
`rec["stepCount"] > max_steps` at `executor.py:410` (OUTCOME A) and `:427` (OUTCOME C), pinned by a
**passing** test asserting the overshoot (`tests/test_executor.py:158`, `assert len(trail) == 4
# maxSteps=3`). Document-only in an observability slice is the right disposition, and the six named
doc sites all resolve and are the right ones (`schemas.py:84-86`; `DESIGN.md:350-351` and `:390`;
`QUERIES.md:1141` §12.5 plus the `$maxSteps` comments at `:1034` and `:1254`; the `AGENTS.md`
executor-invariants block). Two gaps: the bundling premise is unverified and K-027 is unscheduled
(**M-2**), and the proposed wording (*"a run executes at most `maxSteps + 1` steps"*) should note
that the budget is **not** checked on the park (OUTCOME B) or terminal paths — deliberately, per the
comment at `executor.py:415-421` — otherwise the "+1" reads as the whole story.

**6. Executability by a `coder` — yes, apart from B-1.** Units are ordered with explicit
dependencies and per-unit done-conditions; interfaces are named down to method signatures and
key names; the response contracts are concrete JSON; the 20-item test list maps onto specific files
and fixtures; the `from_`/`alias="from"` trap is pre-empted (`schemas.py:58` — correct). §8's list
is complete for *what the change adds* and incomplete for *what the change's own finding
invalidates* (**B-2**), plus m-3 and m-5.

**7. Response size, layering, RAM — all three check out.** `MAX_STEPS = 200` × `MAX_CONFIG_LEN =
8000` + `MAX_TRANSITIONS = 500` × `8000` = 5.6 MB, matching §3.7's figure exactly
(`schemas.py:43-45`); the "publishers below Pydantic bypass the caps" residual is honestly stated
and correctly declined. The "no `?limit=`" argument (a truncated subgraph is a *wrong* answer, not a
partial one) is right, and the `MAX_RUN_STEPS = 50` precedent for unpaginated §12 reads is real
(`schemas.py:86`). Layering is honoured — Cypher stays in `repository.py`, ordering/normalization/
diffing in `services.py`, HTTP mapping in `api.py`, matching DESIGN §14.2 (m-6 is the only wrinkle,
and it is mitigated). RAM: no node type, label, property, index or vector dimension is added, both
reads are `ro_query` against verified index-anchored plans on unchanged query text — **zero graph
RAM impact**, correctly stated as AGENTS.md rule 6 requires.

**On the three open questions (§9).** **OQ-1** is a genuine stakeholder call and the recommendation
is right (see M-2 for the justification fix). **OQ-2** is not a stakeholder call (n-1). **OQ-3** is
genuinely one, and its recommendation is correct and well-argued — deleting a snapshot breaks live
`WorkflowRun`s via `OF_DEF`/`AT_STEP`, exactly as AGENTS.md warns.

---

## 4. What's solid

- **Grounding.** Every one of ~25 file:line citations resolves exactly. The plan was written against
  the real tree, not a remembered one.
- **The three headline findings are all correct**, and finding 2 is a genuine discovery about a
  shipped, documented behaviour that four separate documents got wrong. Surfacing it at design time
  rather than at QA time is the plan doing its job.
- **The endpoint-shape decision.** The `?expand=steps` rejection is the strongest reasoning in the
  document: it would put two shapes on one route (the standing `BACKLOG.md:793` feedback) *and*
  `{key}` with no version resolves latest, so an expand on a version-less request would structure-read
  a version the caller never named — in a tool whose entire purpose is confirming *which* version is
  live. Correct, and correctly argued from the code.
- **Transition identity `(from, to, on, order)`** is derived from `_PUBLISH_CYPHER`'s actual `MERGE`
  key rather than guessed — and the §3.3 argument that a client would plausibly key on `(from, to)`
  and mis-report an added parallel edge as a modified one is exactly right, and is the strongest
  argument for the server-side diff.
- **`response_model` on new routes only**, with the FastAPI field-filtering hazard named and the
  mixed convention pushed onto the parking-lot item instead of declared closed. Honest.
- **Byte-fidelity of `config`/`guard`** (never parsed, never re-serialized) — the right call under
  rule 8, and the "a diff that round-trips JSON would hide a whitespace-only divergence" rationale is
  the correct one.
- **One-side-missing is a 200, not a 404.** That is the post-`pytest` trap made into a first-class
  reportable state; a 404 there would push the operator straight back to raw Cypher, defeating the
  item.
- **`verify_workflows.sh` driving the service layer, not HTTP**, with the reason stated (it is most
  needed when uvicorn is down) and read-only as a hard constraint with its own risk row (R-5).

---

## 5. Open questions for the caller

1. **B-2's filing:** a new `K-034`-style item for the additive-`MERGE` defect, or fold it into
   **K-029** (which already owns "touching a live published def is the risk" and whose premise the
   finding corrects)? My recommendation: **a new item**, cross-referenced from K-029 — K-029 is a
   refactor with its own acceptance criteria, and burying an engine-semantics defect inside it is
   how it gets deferred with the refactor.
2. **OQ-1 stands and is yours** (document `maxSteps` now vs. fix now). My recommendation matches the
   plan's — document now — with M-2's correction: file K-033 self-standing rather than conditional
   on K-027 item 2.
3. **R-1 is still unknown.** I did not read the live `reference` / `ws:acme` graphs. If you want the
   actual current def-vs-snapshot state of `triage@v1` and `access-request@v1` before approving the
   plan (it would tell you whether the additive defect has *already* fired in production), that is a
   read-only Cypher pass and I can do it — but it is a live-environment inspection, not a static
   review, so I did not do it unasked.

---
---

# Re-gate — plan **v2** (U2-G1, round 2)

> **Reviewer:** `analyst` · **Date:** 2026-07-24 · **Artifact:**
> `docs/plans/workflow-def-structure-read.md` **v2** (812 lines, §11 = the architect's dispositions
> record). The round-1 review above is left intact as the first gate's record.
> **Method:** every disposition verified against the **plan text and the code**, not against §11's
> summary of itself, plus a fresh end-to-end read of the plan looking for revision-induced
> contradictions.
> **Baseline tree:** working tree carrying the concurrent K-027 slice A churn
> (`server/falkorchat/{llm,app}.py`, `server/tests/{test_llm,test_app}.py`,
> `docs/{BACKLOG,HISTORY}.md`) — disregarded per the brief, but it *is* why some `docs/BACKLOG.md`
> line numbers have drifted (see RG-n4).
> **Executed (non-mutating only):** `pytest --collect-only -q` → **552/553 collected, 1 deselected**.
> Nothing was written, no suite was run, no graph was touched.

## R1. Verdict

**Approve with suggestions.**

**All 15 round-1 findings are closed.** I re-verified each one against the code rather than against
§11; none was closed on a claim that does not hold. Both blockers are genuinely resolved:

- **B-1** — V-1 is now a *write* in a throwaway `ws:k031probe`, and the "the answer transfers to the
  def side" argument is **sound** (verified below, R2.B-1 (a)–(d)). The isolation contract is
  airtight, and the one interaction with `reference` (`bootstrap_reference`) really is DDL-only.
- **B-2** — the additive-`MERGE` finding has genuinely **left** K-031: the additive re-publish test
  is deleted, §8's scope note routes the ten falsified assertions to K-034, R-8 drops the
  exhaustiveness claim for a grep-verified done-condition, and R-9 covers the not-yet-filed case.
  Critically, **no surviving unit or test depends on K-034's semantics** — the reworked divergence
  fixture is the one place that could have, and it does not (R2.B-2).

New findings from the revision: **0 blocker · 0 major · 6 minor · 5 nit.** None of them blocks
implementation; all six minors are one-paragraph or one-line clarifications, and two of them
(RG-m1, RG-m2) are the kind a coder will hit as a *failing contract test* rather than as a silent
defect — which is the good failure mode.

**Can a `coder` execute this plan as written? Yes.** See R5.

## R2. Round-1 findings — close / re-open

| # | Round-1 finding | Re-gate | Evidence checked |
|---|---|---|---|
| **B-1** | V-1 unexecutable — no workspace seam for a def publish | **CLOSED** | v2 §7 V-1 + §2 rows at plan `:124`/`:125`. Verified (a)–(d) below. |
| **B-2** | Additive-`MERGE` confirmed, filed nowhere | **CLOSED (re-scoped → K-034, per binding decision)** | §0.2, §1.2, §7 "Removed in v2" note, §8 scope note, R-8, R-9, §10 non-negotiable 5. No residual dependency found. |
| **B-2/R-8** | "§7's list is exhaustive" | **CLOSED** | R-8 now a three-way grep classification, no exhaustiveness claim. (Command has a cwd bug — RG-n1.) |
| **M-1** | Receipt counts = submitted spec, not stored def | **CLOSED** | Re-verified in Cypher: `_PUBLISH_CYPHER` (`repository.py:945`) `WITH d, count(st) AS stepCount` sits one line after `UNWIND $steps` (`:941`) ⇒ one row per unwound element ⇒ `len($steps)`; same shape for `count(rel)` at `:954`. §3.2 now states it correctly, the operator sentence is verbatim, and §8's DESIGN §14.4 row mandates it verbatim. |
| **M-2** | K-033 bundling premise unverified | **CLOSED** | §5 files K-033 self-standing; bundling is an explicit *preference*; the premise is marked **Unverified** with the `AGENTS.md:256-257` out-of-lock seams (verified: `_execute_step`/`_select_transition`/`_trace_step`/`resume` are named as outside the lock) and `BACKLOG.md:315`'s 🔵 proposed status (verified at HEAD). ⚠️ The **coordination log** was not updated to match — RG-m6. |
| **M-3** | `533 → 533 + N` stale by construction | **CLOSED** | §1.3 + §7 suite gates + front matter now say "re-derive at U3 start with `pytest --collect-only -q`". I re-ran it: **552/553 collected, 1 deselected** — the plan's quoted reference figure is accurate and the method is non-mutating. |
| **m-1** | Diff is version-qualified | **CLOSED** | §3.3 final bullet + §3.8 check 1 ("at the expected version") + §4 U4.3 route comment + §8 DESIGN row. |
| **m-2** | Test 16's first alternative unreachable | **CLOSED, and the rework is sound** | Alternative deleted with the reason recorded; reworked fixture verified below. |
| **m-3** | DESIGN §14.4 half-populated | **CLOSED (option B)** | Verified the convention exists at `docs/DESIGN.md:806-809` verbatim; §8's row follows it instead of half-filling the table. ⚠️ Diverges from the coordination log's doc-impact scan (`m3-followups-coordination.md:41`, "new endpoints belong in the table") — coordinator note, not a defect. |
| **m-4** | V-1 escalation rule missing | **CLOSED** | §7 V-1 escalation paragraph enumerates the surprising outcomes + graph-dba inbox routing; echoed at R-2 and §10.1. (One residual tension — RG-m4.) |
| **m-5** | `api.py` lacks `MAX_KEY_LEN` | **CLOSED** | Verified: `api.py:17-25` imports only `MAX_ID_LEN`; `MAX_KEY_LEN = 200` is `schemas.py:42`. §4 U3.2 makes the import an explicit step. |
| **m-6** | `start_keys` shape not in QUERIES §11.2 | **CLOSED** | §3.6 documented-divergence paragraph + §4 U1.1 ("citing the §11.2 note by name") + §8's §11.2 row. |
| **n-1** | OQ-2 not a stakeholder call | **CLOSED** | §3.3 heading + §9. |
| **n-2** | V-1 called "read-only-ish" | **CLOSED** | §7 heading: "**This is a WRITE**, in a throwaway workspace graph, torn down." |
| **n-3** | QA report §7.1/§7.5 citations | **CLOSED** | Front matter, §3.3, §3.8 all now say "§5 (DEF-1)" / "§7 item 1" / "§7 item 5". |
| **n-4** | Test 11's changed `kind` | **CLOSED, with a residual** | Verified `WORKFLOW_KINDS` at `services.py:51`, enforced `:531`; `DEF_BODY["kind"] == "process"` (`tests/test_api.py:416`), so `conversation` is correct. The residual is the *other* edit in the same sentence — RG-m5. |
| **Gate ans. 5(b)** | "+1" needs the park/terminal clause | **CLOSED** | §5's doc text names OUTCOME A (`executor.py:410`) and OUTCOME C (`:427`) as the only checked paths; verified the park path (`:415-423`) and the terminal path (`:431-432`) carry no budget check. |
| **§5 Q1** | K-034 vs fold into K-029 | **CLOSED by the stakeholder** (new item) | Coordination log `:85-98`, `:121-123` (number reservations). |
| **§5 Q3** | Live def/snapshot state unknown | **Deliberately open** | Governed by R-1/OQ-3: first observed at §7 test 19, report-and-file only. Correct disposition. |

### B-1 — the four sub-checks, verified

**(a) The "it transfers from the def side" argument holds.** `publish_def` (`repository.py:1011`) →
`self._reference()` (`:132-134`) → `db.reference_graph` (`db.py:87-94`) = `select_graph("reference")`,
a literal with no parameter/env/config override — so no throwaway def publish can exist, exactly as
the plan states. `materialize_snapshot` (`:1484-1485`) formats **the same `_PUBLISH_CYPHER` constant**
(`:937`) with `label="WorkflowDefSnapshot"` against `self._graph(ws)`, and `_READ_META_CYPHER`
(`:961`) is `{label}`-templated and read through the same `_read_subgraph` (`:976`). The probe's
argument dicts (`{"key","type","config"}` / `{"from","to","on","order","guard"}`) match exactly what
`services.publish_workflow_def` hands the repository (`services.py:633-646`) — so the probe exercises
the production shape, not an invented one. **The transfer argument is correct.**

**(b) The isolation contract is airtight.** Every step is workspace-scoped or global-DDL:
`materialize_snapshot` → `_graph("k031probe")`; the raw read → `db.workspace_graph(conn,
"k031probe")`; teardown → `GRAPH.DELETE ws:k031probe` (the same statement `scripts/test_queries.sh:1059`
uses). Nothing in the procedure can reach `ws:acme` or `ws:test`, and nothing writes data to
`reference`.

**(c) The ≥1-transition trap is correctly handled.** `_PUBLISH_CYPHER` does end in
`UNWIND $transitions` (`:949-954`) with the `RETURN` after it, and `materialize_snapshot:1492`
indexes `res.result_set[0]` — so an empty list is the `IndexError` shape the plan describes. The
probe supplies one transition `a→b@go#0`, which is the minimum. Both `MATCH (from…)`/`MATCH (to…)`
resolve because both steps are MERGEd in the same query. ✓

**(d) `bootstrap_reference` is idempotent DDL with no data effect.** Read `scripts/bootstrap_schema.sh:37-70`
— it is exclusively `CREATE INDEX` (×5) and `GRAPH.CONSTRAINT CREATE` (×3) on `reference`; no
`MERGE`, no `CREATE (n:…)`, no `DELETE`. The script header (`:8-10`) documents that duplicates are
non-fatal under `set -euo pipefail` because `redis-cli` exits 0 on Redis-level errors. It is invoked
unconditionally at **`:238`** (the plan says `:239` — RG-n4). ✓ The claim holds.

### B-2 — the handoff is genuine, and the reworked fixture is sound

*Handoff.* Verified absent: any additive-re-publish assertion (§7's "Removed in v2" note at plan
`:671-673`), any "immutable/no-op" prose correction (§8 scope note at `:717-722`, and the AGENTS.md
row's explicit *"Do not rewrite the row's create-only/immutability description"*), any exhaustiveness
claim (R-8), and any self-filing of K-034 (§8's BACKLOG row: *"Do not file K-034 here"*, consistent
with the coordination log's deliberate sequencing at `:124-126`). §10 non-negotiable 5 repeats it.
The surviving references to K-034 are all *citations* (§0.2, §3.1's multi-`START` rationale, U1's
docstring, test 11's comment) — none is a dependency on K-034 landing first except for the text of
the cross-reference itself, which R-9 handles.

*The reworked divergence fixture (test 15) is sound and introduces no new shared-state hazard.*
The sequence — publish `A@1` → materialize into `ws:test` → wipe `reference` in-test → re-publish
`A@1` edited — produces every asserted difference for a mechanical reason: after the wipe, the
re-publish is a **fresh create** into an empty graph, so `ON CREATE SET` fires for `d.name`, the new
`Step`, and `rel.guard`. It therefore depends on **create** semantics only, never on K-034's
*additive* semantics. Validation clears it: `_validate_def_spec` (`services.py:531-590`) has no
step-reachability rule, so an added step with no transitions publishes fine, and the def keeps its
one transition (the K-024 U4b ≥1 rule).
On the shared-state question: `reference` **is** global, but the `wf_repo` fixture already wipes it
at **setup** for every workflow test (`conftest.py:93`, `db.reference_graph(conn).query("MATCH (n)
DETACH DELETE n")` — the exact statement the plan reuses), pytest runs sequentially here (no xdist
in `pyproject.toml:31` addopts), and the post-`pytest` re-seed is already a documented standing
hazard. The in-test wipe adds **one more wipe inside a session that already wipes** — no new hazard
class. One guardrail worth stating in the plan: `_wipe_reference` must be a plain helper called
*inside* a test that already owns `reference` (i.e. under `wf_repo`/`wf_client`), never an autouse
or broader-scoped fixture (RG-n3's neighbour; folded into RG-n2's suggestion).

## R3. New findings (from the revision / fresh read)

### Minors

#### RG-m1 · `startKeys` "omitted unless > 1" is unspecified against `response_model=` — and collides with the exact-key-set contract test

**Evidence.** §3.2 declares `"startKeys": [...] // OMITTED unless > 1`, §3.4 requires
`response_model=WorkflowDefStructureOut` on the new routes, §4 U2.1 says "omit `startKeys` when
`len(start_keys) <= 1`", and §7 test 10 asserts the **exact key set** of the body while test 6 pins
"omitted for one / present for two". A Pydantic field declared `startKeys: list[str] | None = None`
is serialized by FastAPI as `"startKeys": null` — *present* — unless the route sets
`response_model_exclude_none=True`. The plan never names the mechanism.

**Why it matters.** The coder gets a failing exact-key-set test and must invent the fix. The obvious
one (`response_model_exclude_none=True`) has a side effect the plan would not want: `startKey` is
itself nullable (`_READ_META_CYPHER`'s `OPTIONAL MATCH (d)-[:START]->(start)` returns `null` when a
root has no `START` edge), so a def with **no** start would silently drop `startKey` from the body —
an observability endpoint hiding exactly the anomaly it exists to show.

**Suggested improvement.** Pick one in §3.2 and say so: either (a) always emit `startKeys` as a list
(length 0/1/2+) and delete the omission rule — simplest, keeps one key set, and makes test 10
trivially stable; or (b) keep the omission and specify
`response_model_exclude_none=True` *plus* a note that `startKey: null` will then also disappear, with
a contract test pinning the no-`START` case.

#### RG-m2 · `startKeys` is not covered by the canonical-ordering rule, so it can produce the exact false divergence §3.3 exists to prevent

**Evidence.** §3.5 canonicalizes *steps* (by `key`) and *transitions* (by `(from, order, to, on)`)
only. §4 U2.1 says `_canonical_structure` "sorts (§3.5), renames `start_key(s)` → `startKey`/
`startKeys`" — renaming, not sorting. `_READ_META_CYPHER` has no `ORDER BY`, so the row order (and
hence the `start_keys` list order, and hence §3.1's *"`startKey` (the first)"*) is engine-arbitrary.
§3.3 then compares `meta.startKeys` as "the two lists".

**Why it matters.** Two structurally identical two-`START` sides could report `inSync: false` purely
on list order — which is precisely argument 1 of §3.3's case for a server-side diff ("a naive
comparison of two arrays reports false divergences on ordering alone"). It also makes `startKey`
nondeterministic between two calls to the same endpoint.

**Suggested improvement.** Add one line to §3.5: *"`startKeys` sorted lexicographically; `startKey`
is `startKeys[0]` after sorting."* One `sorted()` call in `_canonical_structure`; add it to test 5's
shuffled-input assertion.

#### RG-m3 · The multi-row read path — the plan's #3 headline finding — ships with no automated coverage

**Evidence.** §7's repository tests 1–4 all read a single-`START` def. Test 6 pins the
`startKeys` shape at the **service** layer against `FakeRepo`, i.e. against a hand-built dict, never
against the engine. The only real-engine evidence for the multi-row behaviour is **V-1**, a one-shot
manual probe whose graph is deleted and whose output lands in prose (§8's HISTORY row).

**Why it matters.** The branch that justifies the whole "read all rows" design is unpinned: a future
refactor of `_read_structure` back to `result_set[0]` would pass the entire suite. The plan's R-2
framing ("a cheap tripwire") is only true if something keeps the tripwire wired.

**Suggested improvement.** Either (a) state the gap explicitly in §7 — *"the multi-row repository
path is evidenced by V-1 only; it is deliberately not pinned in the suite, because constructing a
two-`START` fixture means asserting publish-structure semantics K-034 owns"* — or (b) add one
repository test that performs V-1's exact operation in `ws:test` (two `repo.materialize_snapshot`
calls differing only in `start_key`) and asserts the **read** returns both start keys. (b) is
defensible — it asserts the reader, not the writer — but it *would* couple K-031's suite to
publish-additivity and break if K-034 later makes re-publish non-additive. My recommendation: (a)
plus a one-line comment in `_read_structure` pointing at V-1's recorded outcome; take (b) only if
the coordinator accepts the K-034 coupling.

#### RG-m4 · R-2's "the design works either way" and §7's stop-and-escalate rule are in tension, and no fallback is named

**Evidence.** R-2: *"The design works either way: if only one row ever returns, `startKeys` is simply
never emitted and the code path is a cheap tripwire."* §7 V-1: *"one row carrying an arbitrary
`startKey` … is a **stop-and-escalate**."* Both describe the same outcome, one as benign, one as
blocking.

**Why it matters.** They are reconcilable, but the reconciliation is load-bearing and unstated: if
the engine collapses to one row, the endpoint **cannot surface a two-`START` def at all**, which
defeats §3.1's stated reason for existing (*"a read surface that hides it would be worse than no read
surface"*). The natural remedy — counting `START` edges in Cypher — is **new Cypher**, which R-4
declares a hard stop. So the collapse outcome is not "the design still works"; it is "K-031's
finding-3 mitigation is impossible within K-031's scope", and that is what the escalation would have
to resolve.

**Suggested improvement.** One sentence in R-2: *"a single-row collapse does not merely make
`startKeys` dormant — it means the structure read cannot surface a multi-`START` def at all, and the
only remedy (a `count(START)` read) is new Cypher and therefore out of scope; the escalation is
about whether K-031 ships without that detection or waits for a `graph-dba` query."*

#### RG-m5 · Test 11's *other* edit — "a changed step `config`" — can 400 for the same reason `kind` could

**Evidence.** §7 test 11: *"re-publish the same `(key, version)` with a changed `name`, a changed
`kind` and a changed step `config`"*. Gate n-4 was adopted for `kind` only. But `DEF_BODY`'s start
step is `{"key": "start", "type": "human", "config": '{"waitsForHuman": true}'}`
(`tests/test_api.py:421-422`), and `_validate_def_spec` (`services.py:572-581`) rejects any
`human`/`wait` step whose normalized config lacks `waitsForHuman: true`. A coder editing the natural
target — the one step that *has* a config — gets a **400**, and the test pins the wrong thing in
exactly the way n-4 described.

**Suggested improvement.** Extend test 11's ⚠️ note: *"the edited `config` must keep
`config.waitsForHuman: true` on the `human` step (add an inert key, e.g.
`{"waitsForHuman": true, "note": "edited"}`), or edit the `done` step's config instead — dropping the
flag makes the re-publish a 400 (`services.py:572-581`)."*

#### RG-m6 · The coordination log still carries the two claims v2 corrected, and the coder reads both documents

**Evidence.** `docs/plans/m3-followups-coordination.md:76-79` records the binding OQ-1 decision as
*"file K-033 … bundled with **K-027 item 2** … **which must break the lock anyway** — one re-lock
ceremony, two fixes"* — the precise premise M-2 falsified and plan §5 now marks **unverified**.
Separately, `:51-52` lists as a hazard travelling with every brief: *"Published defs are create-only
… an edited re-publish is a **silent no-op**"* — one of the assertions the additive finding
falsifies (K-034's list covers docs/docstrings/AGENTS.md, not this log).

**Why it matters.** The implementer's brief points at both documents. Two binding artifacts giving
different instructions about K-033's filing is how the plan's careful M-2 wording gets overwritten by
the log's.

**Suggested improvement.** Coordinator-owned, not a plan edit: update the log's OQ-1 bullet to
"self-standing, bundling a preference (premise unverified — see plan §5)", and annotate the
create-only hazard bullet with "…and *additive on structure* — see K-034". This is a note to `teco`,
not a finding against the plan.

### Nits

- **RG-n1 · R-8's grep command works from no working directory.** `grep -rn -i "immutab\|no-op"
  docs/ server/falkorchat/ falkor-chat/AGENTS.md` mixes component-relative and root-relative paths:
  from `falkor-chat/` the third path does not exist; from the repo root `server/falkorchat/` does not
  exist *and* `docs/` silently resolves to the **root** `docs/` directory. Since it is a
  done-condition, make it runnable: from `falkor-chat/`, `grep -rn -i "immutab\|no-op" docs/
  server/falkorchat/ AGENTS.md`.
- **RG-n2 · U4's done-condition overstates test 15.** U4 says *"the divergence fixture (§7) shows
  every difference class"*, but the reworked test 15 asserts three (`meta.name`, step presence,
  transition `guard`); exhaustive class coverage lives in the service-level test 9. Either reword U4
  ("…shows the meta / step-presence / transition-guard classes; per-class coverage is test 9") or
  widen test 15 with a step `config` edit (cheap, and it exercises the byte-fidelity claim
  end-to-end). While there: state that `_wipe_reference` is a plain in-test helper under
  `wf_repo`/`wf_client`, never an autouse or session fixture.
- **RG-n3 · R-9 names only U6, but the first K-034 cross-reference lands in U1.** §4 U1.1 requires
  `_read_structure`'s docstring to carry "a pointer to **K-034**", and U1 runs well before U3, which
  is when R-9 expects the filing. Extend R-9's "(filing in flight)" rule to *any* unit that writes a
  K-034 reference.
- **RG-n4 · Citation drift (all trivial, none load-bearing).** `bootstrap_reference` is invoked at
  `scripts/bootstrap_schema.sh:238`, not `:239` (§7 V-1 step 1). `materialize_snapshot`'s
  `_PUBLISH_CYPHER.format` is `repository.py:1485`; `:1483` (§0.2, §3.6-adjacent) is inside the
  docstring — the §2 row's `1470-1490` range is right. `app.py:86` (§2) is a blank line; the
  `WorkflowDefNotFoundError → 404` handler is `:87-92`. `BACKLOG.md:793` was correct at HEAD but is
  `:810` on the current working tree because of the concurrent K-027 churn — **expected, disregard**;
  the coder should grep the parking-lot entry rather than trust the line.
- **RG-n5 · V-1's `START`-count probe interpolates literals.** §7 step 3's
  `MATCH (d:WorkflowDefSnapshot {key:'probe', version:'v1'})-[r:START]->() RETURN count(r)` is
  string-literal Cypher, against AGENTS.md rule 1. Harmless in a throwaway one-shot, but the plan
  leans on rule discipline elsewhere and the probe is the artifact most likely to be copy-pasted into
  a future script — parameterise it.

## R4. What's solid in v2 (beyond round 1's list)

- **The B-1 rewrite is better than the fix I suggested.** It adds the `≥ 1 transition` trap (the
  K-030/O-6 `IndexError` shape), an explicit *"Nothing is published into `reference` at any point"*
  isolation contract, and the "do not re-litigate this" instruction — all three are things an
  implementer would otherwise burn time on.
- **The m-2 rework is a genuine improvement, not a compliance edit.** Recasting the divergence
  fixture around an in-test `reference` wipe makes it (i) reachable through the real service layer,
  (ii) independent of K-034, and (iii) an exact replica of the documented live trap. That is one
  change buying three properties.
- **The B-2 handoff is disciplined in both directions.** The plan neither absorbs the finding nor
  quietly leans on it: it removes the test, refuses the doc corrections it would have been natural to
  slip in (§8's "Add the detection half only"), and declines to file K-034 itself because the
  coordinator owns the sequencing.
- **M-3's remediation is the right shape** — a *measured* baseline with the method named, plus a
  dated reference figure explicitly labelled as moving. I re-measured: 552/553, exactly as stated.
- **§11 is an honest dispositions record.** Every row I checked matched the plan text; nothing was
  claimed as adopted that was not.

## R5. Can a `coder` execute this plan as written?

**Yes.** The spine (V-1 → U1 → U2 → U3 → U4 → U5 → U6) is ordered, each unit names its files, method
signatures, return shapes and a done-condition, the response contracts are concrete JSON, the test
list maps onto real fixtures (`wf_repo`/`wf_client`, `conftest.py:84-94` — verified), and the traps
that would have cost the most time are pre-empted (`from_`/`alias="from"`, the `MAX_KEY_LEN` import,
the `≥1 transition` probe trap, the `WORKFLOW_KINDS` membership, `materialize_def`'s
same-`key@version` constraint). The two decisions a coder would otherwise have to make alone —
whether V-1 blocks and whether K-034 belongs here — are both settled by explicit non-negotiables.

The six minors are refinements a competent implementer would either resolve correctly on their own
(RG-m1, RG-m2, RG-m5 — all surface as failing tests, not as silent defects) or escalate (RG-m4). None
is a re-gate condition. **RG-m6 is the one I would fix before dispatching U3**, because it is the only
finding where the coder is handed two conflicting instructions, and it costs the coordinator one
edit.

## R6. Notes to the coordinator (not findings against the plan)

1. **The re-scoping is right.** I would not re-open either binding decision. Routing the additive-
   `MERGE` finding to K-034 keeps K-031 a genuinely read-only observability slice, and document-only
   for `maxSteps` correctly refuses to ship a live semantic change inside it. The plan's detection-
   vs-cause split is the correct seam.
2. **RG-m6** — sync `m3-followups-coordination.md` (OQ-1's bundling wording, and the create-only
   hazard bullet) before U3 is dispatched.
3. **m-3's option B diverges from the log's doc-impact scan** (`:41` says the new endpoints "belong
   in the table"; the plan follows DESIGN §14.4's existing exclusion convention instead). The plan's
   reasoning is better; update the scan row so integration does not flag it.
4. **R-9 timing** — K-034's filing is held until U1 (`tdd-engineer`) lands to avoid a `BACKLOG.md`
   collision, but the plan's *own* U1 writes a K-034 docstring reference. Confirm the ordering, or
   accept the "(filing in flight)" fallback for U1 too (RG-n3).
5. **R-1 remains unmeasured.** The live `triage@v1`/`access-request@v1` def-vs-snapshot state is
   still unknown and is first observed by §7 test 19. That is the plan's intent, not a gap — but it
   means the run may end with a live-divergence report to triage.
