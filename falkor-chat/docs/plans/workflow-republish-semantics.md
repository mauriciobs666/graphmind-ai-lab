# Workflow def/snapshot re-publish semantics — closing the additive-`MERGE` defect

> **Status:** active · **Owner:** `architect` · **Tracks:** K-034

## 0. Summary of the decision (read this first)

K-034 offers three candidate remedies. This plan picks **candidate 2 — a pre-publish/pre-materialize
structural-equality check that rejects a differing re-publish** — with one refinement forced by the
evidence: the check is **topology-only**, not full-payload equality. A re-publish/re-materialize of an
existing `(key, version)` is rejected (`409`, nothing written) **only if the step-key set, the
transition-identity set `(from,to,on,order)`, or the start key differ** from what's already stored.
A property-only edit (`name`, `kind`, a step's `type`/`config`, a transition's `guard`) stays exactly
what it is today: a silent no-op on properties, because `MERGE … ON CREATE SET` genuinely already
gives that for free and an existing, currently-green test pins it (`tests/test_api.py
::test_republish_is_create_only_on_properties_structure_read_unchanged`, added by K-031). Full-payload
equality would have broken that test and reopened ground K-031 deliberately left alone; topology-only
equality closes exactly K-034's two documented live defects and nothing more.

**The load-bearing implementation fact that shapes everything below:** K-031 already shipped the read
+ diff machinery this fix needs — `Repository.read_def_structure`/`read_snapshot_structure`
(`server/falkorchat/repository.py`) and `services._canonical_structure`/`_diff_structures`
(`server/falkorchat/services.py:214-345`). The fix is a **pure-Python gate in `services.py`**, built
by filtering `_diff_structures`' existing output to structural paths and calling it before the
existing repository write. **`_PUBLISH_CYPHER` does not change.** That means the `graph-dba` gate the
backlog item conditions on ("if and only if the query changes") **does not trigger** — see §2 for the
justification and the tripwire if implementation drifts from this plan.

---

## 1. Goal & scope

**Goal.** Close K-034's two live, reachable defects — nondeterministic branch selection
(`executor._select_transition`) and the duplicate-`START` `WorkflowRun` creation crash
(`repository.start_run`/`start_run_untriggered`) — both caused by `_PUBLISH_CYPHER`'s `MERGE`
patterns minting parallel `TRANSITION`/`START` structure on a differing re-publish, while the
component's docs assert re-publish is a no-op. Fix the semantics, then correct every doc/docstring
site that asserted the false claim, in the same delivered change.

**In scope:**
1. A topology-equality gate in `services.publish_workflow_def` and `services.materialize_def` that
   rejects a *structurally* differing re-publish/re-materialize of an existing `(key, version)`.
2. A new `WorkflowDefConflictError` (409), wired the same way `WorkflowDefSpecError`/
   `WorkflowDefNotFoundError` already are.
3. A small, isolated defense-in-depth fix to `executor._select_transition`'s tie-break (deterministic
   regardless of edge retrieval order), covering the residual case of pre-existing/bypassed corruption
   this change does not retroactively repair.
4. The full doc/docstring correction sweep (13 falsified + 5 weaker sites from K-034's table, plus
   K-029's premise paragraph, plus this plan's own new sites).
5. Test additions at the service, API and executor layers (see §6).

**Explicitly out of scope** (per K-034's own scope note, unchanged here):
- Repairing any live `reference`/`ws:acme` drift that already exists — destructive, stakeholder-gated
  (K-031 §6 R-1). §7 below flags this as a **rollout risk**, not a deliverable.
- K-031's read/diff surface itself — reused, not modified.
- K-030's remaining open item (`_PUBLISH_CYPHER`'s bare trailing `UNWIND` still crashes on a
  **first-time** materialize of an already-poisoned zero-transition def) — orthogonal, see §5.
- A new delete-and-republish REST affordance (candidate 3) — considered and rejected, see §3.4.
- Making published-def *properties* immutable/enforced — not needed to close the two defects, and
  would break the K-031-pinned property-only-republish test for no gain.

---

## 2. Context & findings

### 2.1 The defect, precisely

`Repository._PUBLISH_CYPHER` (`server/falkorchat/repository.py:960-980`, shared verbatim by
`publish_def` and `materialize_snapshot` via `.format(label=...)`) is:

```cypher
MERGE (d:{label} {key: $key, version: $version}) ON CREATE SET d.name = $name, d.kind = $kind
WITH d
UNWIND $steps AS s
  MERGE (st:Step {stepUid: $key+':'+$version+':'+s.key}) ON CREATE SET st.key=s.key, st.type=s.type, st.config=s.config
  MERGE (d)-[:HAS_STEP]->(st)
WITH d, count(st) AS stepCount
MATCH (start:Step {stepUid: $key+':'+$version+':'+$startKey})
MERGE (d)-[:START]->(start)
WITH d, stepCount
UNWIND $transitions AS tr
  MATCH (from:Step {...tr.from}) MATCH (to:Step {...tr.to})
  MERGE (from)-[rel:TRANSITION {on: tr.on, order: tr.order}]->(to) ON CREATE SET rel.guard = tr.guard
WITH d, stepCount, count(rel) AS transitionCount
RETURN d.key AS key, d.version AS version, stepCount, transitionCount
```

`ON CREATE SET` genuinely makes **properties** create-only. But three `MERGE` **patterns** include
identity that can shift between two calls with the same `(key, version)`:
- `MERGE (d)-[:START]->(start)` — the pattern's endpoint is `start`, resolved from `$startKey`. A
  changed start key ⇒ a **second `START` edge** beside the old one (the old one is never removed).
- `MERGE (from)-[rel:TRANSITION {on, order}]->(to)` — the MERGE key is `{on, order}` **plus the
  `(from)→(to)` pattern endpoints**. A changed `to` for the same `(from, on, order)` is a *different*
  relationship, not a match — a **parallel `TRANSITION` edge**.
- A new step `key` mints a new `Step` + `HAS_STEP` — additive, and (unlike the two above) genuinely
  benign in isolation, but see §3.2 for why this plan still rejects it.

Confirmed by `docs/reviews/workflow-def-structure-read.md` finding **B-2** and handed off via
`docs/plans/workflow-def-structure-read.md` §0.2/§1.2 — do not re-derive; both already trace the exact
same three MERGE patterns.

### 2.2 The two live consequences (why this is a defect, not a curiosity)

1. `executor._select_transition` (`server/falkorchat/executor.py:769-793`, current line — K-034's
   quoted `:758` has drifted): `ordered = sorted(transitions, key=lambda t: (t["guard"] == "",
   t["order"]))`. Two `TRANSITION` edges with equal `(guard=="", order)` but different `to` sort
   equal; Python's stable sort means the winner is whatever order FalkorDB returned the edges in —
   unpinned, can differ run to run.
2. `repository.start_run` / `start_run_untriggered` (`repository.py:1183-1195`, `:1236-1250` current
   lines) both `MATCH (snap)-[:START]->(start:Step) … CREATE (r:WorkflowRun {runId: …})`. Two `START`
   edges ⇒ two rows ⇒ `CREATE` runs twice ⇒ the second violates `UNIQUE NODE WorkflowRun PROPERTIES 1
   runId` (`scripts/bootstrap_schema.sh:180`) — the run **fails to start**.

Both are reachable exactly the way `AGENTS.md` (pre-correction) tells operators to work: re-run
`scripts/seed_workflows.sh` after `./scripts/test_queries.sh` or a pytest run (both wipe `reference`;
workspace snapshots survive), combined with any def edit made in between. Concretely: `reference` gets
wiped, so the next publish is a **fresh create** (no conflict — nothing to compare against); but the
surviving `ws:<id>` snapshot from *before* the wipe is untouched, and the subsequent materialize call
now writes a **second, differing** structure into it — this is where the live defect actually bites,
on the *materialize* side, which is why §4 gates `materialize_def` independently, not just `publish_def`.

### 2.3 Why no `graph-dba` gate is needed here

The backlog item conditions the gate on "`_PUBLISH_CYPHER`/`materialize_snapshot` changes." This plan's
gate lives entirely in `services.py`, calling two **already-shipped, already-profiled, index-anchored**
read-only queries (`read_def_structure` / `read_snapshot_structure`, both routed `GRAPH.RO_QUERY`,
both anchored on `Node By Index Scan | (d:WorkflowDef)` / `(snap:WorkflowDefSnapshot)` per
`docs/QUERIES.md` §11.2's live-verified `GRAPH.PROFILE`) plus a pure-Python filter over
`_diff_structures`' existing output. **`_PUBLISH_CYPHER` itself is untouched** — no new query shape,
no new `MERGE`, no new index/constraint need. Rule 6 (RAM) holds trivially: zero new node types,
labels, properties, or vector dimensions.

**Tripwire for whoever implements this:** if at any point the implementation needs to change
`_PUBLISH_CYPHER`'s text (not just add a read before it), stop and route to `graph-dba` for a
re-`GRAPH.PROFILE` before landing — that is exactly the condition the backlog item names, and this
plan's whole point is that the condition doesn't fire for the chosen design.

### 2.4 `DESIGN.md` §1's register does not actually contain an immutability row

Checked directly: §1.2 "Locked design decisions (detailed register)" (`docs/DESIGN.md:15-53`) lists 16
rows — thread-scoped `NEXT`, `Message.role` derivation, vector indexes via DDL, guarded-CREATE write
paths, member-id namespacing, etc. **None of them is "WorkflowDef versions are immutable."** The
immutability language lives in **narrative prose outside §1** — §3's topology diagram (`docs/
DESIGN.md:102`), §4's "Decision:" paragraph (`:144-149`), and §9's write-paths table (`:570`) — plus
the QUERIES.md §11 preamble and several docstrings. This matters for the candidate-1 question (§3.1):
rejecting `ON MATCH SET` is not "avoiding reopening §1's register" in the literal sense (there's
nothing there to reopen); it's declining to reopen §4's stated intent, which this plan can honor in
full without touching mutability at all — see §3.

---

## 3. Design & rationale

### 3.1 Candidate 1 (`ON MATCH SET`, mutable versions) — rejected

Would make versions genuinely mutable. Not needed: both live defects are additive-structure bugs, not
a demand for editability. Making versions mutable is a strictly bigger, harder-to-reverse change than
required, it reopens §4's explicit "immutable once published" narrative decision (real prose, even if
not in the §1.2 table — see §2.4) with no offsetting benefit for this item, and it invalidates the
newly-corrected docs the moment they're fixed. Rejected on scope-discipline grounds ("without an
unbounded surface change" — the brief's own words) as much as design-purity ones.

### 3.2 Candidate 2 (structural-equality reject) — chosen, topology-only

The naive reading of candidate 2 is *full-payload* equality (any difference at all, including
properties, rejects). That naive reading **would break a currently-green, deliberately-designed test**:
`tests/test_api.py::test_republish_is_create_only_on_properties_structure_read_unchanged` (added by
K-031) republishes the same `(key, version)` with an edited `name`, `kind`, and a step's `config`, and
asserts **`201`** with the *stored* structure unchanged — i.e. it pins "create-only on properties" as
current, intentional, K-031-approved behavior, explicitly deferring only "the structural (additive)
half" to K-034. Rejecting on any property diff would contradict a test written *specifically in
anticipation of this item*.

So the chosen equality is **topology-only**: the set of step *keys* (existence, not content), the set
of transition *identities* `(from, to, on, order)` (existence, not `guard` content), and the start key.
This maps directly onto `_PUBLISH_CYPHER`'s own `MERGE` **pattern** identity (§2.1) — which is exactly
the set of things a differing re-publish can silently *add* rather than update. Properties (`name`,
`kind`, `step.type`, `step.config`, `transition.guard`) stay exactly as create-only as they are today,
because that's what `ON CREATE SET` already gives correctly and safely — there is no defect there to
fix. This resolves K-034's own open sub-question ("a re-publish with a new step key behaves per the
chosen decision") cleanly: a **new step key is a topology change** (its presence flips
absent→present), so it is **rejected**, same as a retargeted transition or a moved start key. "Bump
the version" is the sanctioned way to add a step — consistent with how K-029 already frames def edits
("a def edit costs a version bump") and with §4's own narrative ("bump version, never mutate in
place").

**Mechanism — reuse, don't rebuild.** `services._diff_structures` (`services.py:281-345`) already
enumerates exactly this class of difference, with a documented path grammar: `meta.<field>` (property),
`steps[<key>]` (presence — structural) vs `steps[<key>].<type|config>` (property), `transitions[…]`
(presence — structural) vs `transitions[…].guard` (property). A small filter over its output,
`_structural_diffs`, keeps only the presence-shaped / `meta.startKey`/`meta.startKeys` entries:

```python
def _structural_diffs(diffs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Filter `_diff_structures`' output to topology-changing entries only (K-034).

    `meta.startKey`/`meta.startKeys` and a bare `steps[<key>]`/`transitions[...]`
    presence row are structural — exactly what `_PUBLISH_CYPHER`'s `MERGE` patterns
    can mint as *parallel* structure on a differing re-publish. `meta.name`/`meta.kind`
    and any `.type`/`.config`/`.guard`-suffixed row are property-only — `ON CREATE SET`
    already makes those safely create-only; re-publish stays a silent no-op on them
    (K-031-pinned, unchanged by this filter).
    """
    return [
        d for d in diffs
        if d["path"] in ("meta.startKey", "meta.startKeys")
        or (d["path"].startswith("steps[") and not d["path"].endswith((".type", ".config")))
        or (d["path"].startswith("transitions[") and not d["path"].endswith(".guard"))
    ]
```

A shared gate helper (new, `services.py`) reads the existing structure (if any), builds a canonical
structure for the *candidate* payload from data the caller already has in hand (no extra query for the
candidate side), diffs, filters, and raises on any survivor:

```python
def _check_no_structural_conflict(
    *, existing_raw: dict[str, Any] | None, candidate_raw: dict[str, Any],
    key: str, version: str, resource: str,
) -> None:
    """Raise `WorkflowDefConflictError` if `candidate_raw`'s topology differs from
    what's already stored at `(key, version)`. No-op when nothing is stored yet
    (`existing_raw is None`) — a first-time publish/materialize is unaffected.
    `resource` is "workflow def" or "workspace snapshot", for the message only.
    """
    if existing_raw is None:
        return
    existing = _canonical_structure(existing_raw, source="existing", key=key, version=version)
    candidate = _canonical_structure(candidate_raw, source="candidate", key=key, version=version)
    diffs = _structural_diffs(_diff_structures(existing, candidate))
    if diffs:
        paths = ", ".join(d["path"] for d in diffs)
        raise WorkflowDefConflictError(
            f"{resource} {key!r} version {version!r} is already published with a "
            f"different topology ({len(diffs)} difference(s): {paths}) — a published "
            f"version's structure is immutable; publish a new version instead of "
            f"editing this one, or inspect the mismatch with "
            f"GET /workflow-defs/{key}/versions/{version}"
        )
```

Both `_canonical_structure`'s `source=` parameter and `_diff_structures`' `"def"`/`"snapshot"` result
keys are cosmetic labels from the `diff_def_snapshot` use case; reusing them here for
(existing, candidate) is harmless — nothing asserts on those particular string values outside
`diff_def_snapshot`'s own tests (verified: `grep -rn '"source"' tests/test_services.py` only matches
that call site).

### 3.3 Wiring into the two call sites

`services.publish_workflow_def` (`services.py:834-871`) — insert the gate **after**
`_validate_def_spec` (unchanged ordering — see §5) and **before** `self._repo.publish_def(...)`:

```python
start_key = self._validate_def_spec(kind=kind, steps=steps, transitions=transitions)
repo_steps = [...]          # unchanged
repo_transitions = [...]    # unchanged
existing_raw = self._repo.read_def_structure(key=key, version=version)
_check_no_structural_conflict(
    existing_raw=existing_raw,
    candidate_raw={
        "name": name, "kind": kind, "start_keys": [start_key],
        "steps": repo_steps, "transitions": repo_transitions,
    },
    key=key, version=version, resource="workflow def",
)
return self._repo.publish_def(...)   # unchanged call
```

`services.materialize_def` (`services.py:873-894`) — insert after the existing `sub is None` check,
before `self._repo.materialize_snapshot(...)`:

```python
sub = self._repo.read_def_subgraph(key=key, version=version)
if sub is None:
    raise WorkflowDefNotFoundError(...)   # unchanged
existing_raw = self._repo.read_snapshot_structure(ctx.ws, key=key, version=version)
_check_no_structural_conflict(
    existing_raw=existing_raw,
    candidate_raw={
        "name": sub["name"], "kind": sub["kind"], "start_keys": [sub["start_key"]],
        "steps": sub["steps"], "transitions": sub["transitions"],
    },
    key=key, version=version, resource="workspace snapshot",
)
return self._repo.materialize_snapshot(...)   # unchanged call
```

**`Repository.publish_def`/`materialize_snapshot` themselves are not touched.** They remain thin,
non-validating write primitives — exactly the existing layering: `publish_def`'s own docstring already
says "the spec is validated by `services.publish_workflow_def` *before* this call" for
`_validate_def_spec`; this plan adds the topology check to that same list of service-layer
preconditions the repository method assumes. Confirmed by `grep -rn '\.publish_def(\|\.
materialize_snapshot(' server/falkorchat/` — `services.py` is the **only** caller of either method in
the shipped code (`scripts/seed_workflows.sh` goes through the service/REST layer, not the repository
directly), so gating in `services.py` covers every production path.

### 3.4 Candidate 3 (explicit delete-and-republish affordance) — considered, rejected

Not built. The repair story for "I need to fix a typo in the *content* of an unpublished-elsewhere
edit" is already the sanctioned one: bump the version (K-029's own framing: "a def edit costs a
version bump"; §4's "bump version, never mutate in place"). For "I need to fix a version that's
*already* materialized into a live workspace" — a genuinely destructive, cross-graph repair — this
plan treats it the same way K-031 already treats live `reference`/`ws:acme` drift: **a stakeholder-
gated, destructive shared-state operation, out of this item's scope** (K-031 §6 R-1 precedent), not a
new REST verb. Building a delete endpoint now would be exactly the "unbounded surface change" the
brief asks this plan to avoid, for a repair path the version-bump convention already covers in the
common case.

### 3.5 Defense-in-depth: `executor._select_transition`'s tie-break

The gate in §3.3 makes duplicate outgoing transitions **unreachable going forward** through the
sanctioned write path. It does **not** retroactively fix: (a) any `(key, version)` that already
carries K-034 damage in a live `reference`/`ws:acme` graph before this fix ships (§7), or (b) a future
caller that bypasses `services.py` and calls `Repository.materialize_snapshot`/`publish_def` directly
(the codebase already does this in tests, e.g. `tests/test_executor.py::_start_run`, precisely because
`Repository` is a thin, non-validating primitive by design — §3.3). K-034's own test strategy asks for
"an executor test pinning that a def with duplicate outgoing transitions is either impossible or
deterministic under your fix." Given (a)/(b) above, "impossible" is not fully true, so this plan adds
the cheap "deterministic" half: extend `_select_transition`'s sort key with `to` as a final tie-break.

`server/falkorchat/executor.py`, in `_select_transition` (current line ~787):

```python
# before:
ordered = sorted(transitions, key=lambda t: (t["guard"] == "", t["order"]))
# after:
ordered = sorted(transitions, key=lambda t: (t["guard"] == "", t["order"], t["to"]))
```

One-line, additive-only sort key (matches every existing ordering for the non-duplicate case — `to`
only breaks a tie that was previously undefined). `_select_transition` is **outside** the SHA-locked
`_drive_loop` (confirmed: K-033's backlog entry names `_select_transition` explicitly as one of the
methods "outside" the lock), so no lock-break/re-lock ceremony applies.

---

## 4. New error: `WorkflowDefConflictError`

Add to `server/falkorchat/repository.py`, beside `WorkflowDefSpecError`/`WorkflowDefNotFoundError`
(same reasoning as their own docstrings: defined in `repository.py` to avoid a repository→services
import cycle, re-exported by `services.py`):

```python
class WorkflowDefConflictError(Exception):
    """A re-publish or re-materialize would change the topology of an existing
    `(key, version)` (M3 §11 / K-034).

    Raised by `services.publish_workflow_def` (against `reference`) and
    `services.materialize_def` (against `ctx.ws`'s snapshot) when the incoming
    step-key set, transition-identity set `(from,to,on,order)`, or start key
    differs from what's already stored — before any repository write. A
    property-only difference (`name`, `kind`, a step's `type`/`config`, a
    transition's `guard`) is NOT a conflict; that stays create-only-on-properties
    (K-031-pinned). Nothing is written when this is raised. Re-exported by
    `services`.
    """
```

**HTTP mapping (`server/falkorchat/app.py`, `_register_error_handlers`)** — same pattern as
`WorkflowDefSpecError`/`WorkflowDefNotFoundError`, and the same status code already used for the
analogous "state conflict, nothing written" case (`WorkflowRunNotWaitingError` → 409):

```python
@app.exception_handler(WorkflowDefConflictError)
async def _handle_wf_def_conflict(_request, exc: WorkflowDefConflictError):  # noqa: ANN001
    return JSONResponse(
        status_code=409,
        content={"error": type(exc).__name__, "detail": str(exc)},
    )
```

Import `WorkflowDefConflictError` alongside the other workflow errors at the top of `app.py` and in
`services.py`'s re-export block (`from .repository import WorkflowDefConflictError as
WorkflowDefConflictError`), matching the existing three-line pattern for the sibling errors.

### 4.1 API contract — explicit before/after

`POST /workflow-defs` (`api.py:237`, `status_code=201`) and
`POST /workflow-defs/{key}/versions/{version}/materialize` (`api.py:292`, `status_code=201`):

| Scenario | Today | After this change |
|---|---|---|
| No existing `(key, version)` | `201`, written | `201`, written — **unchanged** |
| Existing, topology-identical (incl. property-only diffs: name/kind/step config/guard) | `201`, `ON CREATE SET` no-ops | `201`, `ON CREATE SET` no-ops — **unchanged** |
| Existing, topology differs (new/removed step, new/removed/retargeted transition, moved start) | `201`, **silently mints parallel structure** | **`409 {"error":"WorkflowDefConflictError","detail":"…"}`, nothing written** |

Update the `api.py` §11 section comment (currently: *"Spec/​not-found errors map to 400/404 through
the app-level exception handlers"*) to add the 409 case, mirroring the existing §12 run-section comment
style (`api.py`, just above `start_workflow_run`, which already documents its own 404/409/400/503 map).

---

## 5. Sequencing vs K-030

K-030 shipped the **service-layer** half already: `_validate_def_spec` rejects a zero-transition spec
before any repository call (`services.py:826-833`, K-024 U4b). K-030's **remaining open item** ("what
is still open (re-gate r-1)") is a **query-level** fix — CASE-guarding `_PUBLISH_CYPHER`'s trailing
`UNWIND $transitions` so a *first-time* materialize of an already-poisoned zero-transition def (reached
some way other than fresh publish) doesn't `IndexError`.

This plan's gate is **orthogonal and non-interacting**:
- It runs entirely inside `services.py`, before any repository call — same layer as K-030's shipped
  half, same ordering discipline (`_validate_def_spec` → this gate → repository call). No collision.
- It never touches `_PUBLISH_CYPHER`'s text, so it cannot affect K-030's *remaining* query-level fix
  or its interaction with the query plan.
- It does **not** close K-030's r-1 gap: a *first-time* materialize (`existing_raw is None` at the new
  gate) still falls through unguarded to `_repo.materialize_snapshot`, which still reuses the bare
  `UNWIND`. That gap is K-030's to close, not this item's — this plan's gate only fires on a
  **second** publish/materialize of the same `(key, version)`.
- If K-030's query-level CASE-guard lands (before or after this change), it is the one that needs the
  `graph-dba` re-`GRAPH.PROFILE` gate, independent of this plan's work.

---

## 6. Test strategy

### 6.1 Service layer — `tests/test_services.py` (`FakeRepo`-based, no live DB)

Follow the file's existing naming convention (`test_publish_workflow_def_<condition>_raises_nothing_
written` / `..._succeeds`). `FakeRepo.def_structures`/`snapshot_structures` (keyed `(key, version)`,
already used by the K-031 structure-read tests at `tests/test_services.py:1140+`) is the seeding
mechanism — no new fixture machinery needed.

For **both** `publish_workflow_def` (seed `repo.def_structures`) and `materialize_def` (seed
`repo.snapshot_structures`, plus `repo.defs[(key,version)]` for the `read_def_subgraph` source):

1. No existing structure → succeeds, `repo.published`/`repo.materialized` gets exactly one entry
   (regression: confirms the new read doesn't change first-publish behavior).
2. Existing, byte-identical resubmission → succeeds, `repo.published`/`repo.materialized` still gets
   the call (unchanged — the gate does not skip the write, see §3.2's "reuse, don't rebuild" note: this
   plan never special-cases "skip the write," it only ever adds a reject).
3. Existing, **name/kind/step-config-only** difference (mirrors the K-031-pinned API test's exact
   scenario) → succeeds, same as (2). This is the test that would catch an over-broad (full-equality)
   implementation of the gate.
4. Existing, **new step key** (topology grows, everything else identical) → raises
   `WorkflowDefConflictError`; `repo.published`/`repo.materialized` gets **no new entry**.
5. Existing, **changed transition `to`** for the same `(from, on, order)` → raises
   `WorkflowDefConflictError`, nothing written. (This is K-034's own named test case (a)/(d).)
6. Existing, **changed start key** → raises `WorkflowDefConflictError`, nothing written. (K-034's
   named case (b)/(d).)
7. Existing, **removed transition** (candidate has fewer transitions than stored) → raises (the
   presence-diff direction `_diff_structures` already covers symmetrically).

### 6.2 Repository layer — `tests/test_repository.py` (live FalkorDB, `wf_repo` fixture)

Keep the two existing idempotency tests unmodified
(`test_publish_def_is_idempotent_no_new_nodes_on_republish`,
`test_materialize_snapshot_is_idempotent_on_rematerialize`) — they resubmit byte-identical content, so
they stay green under the unchanged `Repository` layer.

Add one small **contract-boundary** test (recommended, not mechanically required by the fix, but
directly answers K-034's test-strategy wording at the mechanism level and prevents a future reader from
assuming `_PUBLISH_CYPHER` is already guarded): calling `wf_repo.publish_def(...)` **directly** twice
with a changed transition `to` for the same `(key, version)` **does** leave two `TRANSITION` edges —
i.e. pin that raw `Repository` calls are unsafe on their own, and the safety is a `services.py`
contract. Comment it explicitly: `# K-034: the guard lives in services.py, not here — see
services._check_no_structural_conflict. Do not "fix" this by changing _PUBLISH_CYPHER without a
graph-dba re-PROFILE gate.`

### 6.3 API layer — `tests/test_api.py` (live FalkorDB, `wf_client` fixture)

- Keep `test_republish_is_create_only_on_properties_structure_read_unchanged` **unmodified** — it must
  stay green exactly as-is; this is the test that proves the topology-only scoping decision (§3.2) was
  implemented correctly, not just decided correctly.
- New: `POST /workflow-defs` with a changed transition `to` on an already-published `(key, version)` →
  `409`, `{"error": "WorkflowDefConflictError", ...}`; follow with `GET /workflow-defs/{key}/versions/
  {version}` and assert the structure is **exactly** the original (one transition, not two) —
  end-to-end proof against the real `_PUBLISH_CYPHER`.
- New: same for a changed start key → `409`; `GET .../versions/{version}` shows exactly one `startKey`
  (not `startKeys` with two entries).
- New: materialize-side conflict. Publish a def, materialize it (clean snapshot). Directly call
  `wf_repo.materialize_snapshot(...)` for a **different** `(key, version)` pair is not useful here —
  instead simulate real drift the way the defect actually arises (§2.2): call
  `wf_repo.materialize_snapshot(ws, key=key, version=version, ..., transitions=<different from what's
  now in reference>)` directly (bypassing the service, the same pattern `tests/test_executor.py::
  _start_run` already uses) to seed an out-of-sync snapshot, then `POST /workflow-defs/{key}/versions/
  {version}/materialize` through the service/REST layer → `409`, and confirm via `GET /workspaces/
  test/snapshots/{key}/versions/{version}` that the snapshot is unchanged (still the seeded, differing
  content — proving the service-layer call truly wrote nothing).

### 6.4 Executor layer — `tests/test_executor.py`

New test: materialize (directly via `repo.materialize_snapshot`, as the file's existing fixtures
already do) a snapshot whose transitions list contains two entries sharing `(from, on, order)` but
different `to` (both unconditional, or both guarded identically so both fire) — supplied in one order,
then run the same scenario again with the list **reversed**. Assert both runs advance to the **same**
`to` step. This exercises the §3.5 tie-break directly and does not depend on FalkorDB's actual edge
retrieval order (which is what made the original bug non-reproducible on demand).

### 6.5 Doc-correction verification — the grep done-condition

From `falkor-chat/`: `grep -rn -i "immutab\|no-op" docs/ server/falkorchat/ AGENTS.md`. §8 below is
the classification for every hit found during this plan's investigation (2026-07-31); re-run the grep
at implementation time (line numbers drift) and confirm no new hit appeared uncategorized. Record the
classification as part of the delivered change (e.g. in the `docs/HISTORY.md` entry or the
implementer's final report) — it does not need its own document.

---

## 7. Risks & open questions

1. **Rollout risk — pre-existing damage blocks, it doesn't heal.** If `reference` and/or `ws:acme`
   already carry K-034 damage (a duplicate `START`/`TRANSITION` from before this fix ships), the gate
   will **permanently reject** the next publish/materialize attempt on that `(key, version)` — correct
   protective behavior, but a surprise if nobody checks first. **Before this change reaches an
   environment with live def data, run `./scripts/verify_workflows.sh <wsId>` (or the `/diff` route)
   for every published `(key, version)`.** If it reports divergence, that must be manually repaired
   (delete the corrupted def/snapshot subgraph and republish clean) **before** this fix lands there —
   this repair itself stays out of this item's scope (K-031 §6 R-1: destructive, stakeholder-gated),
   but the *sequencing* is this plan's to flag.
2. **Residual concurrent-first-publish race (not fixed, not one of K-034's two documented defects).**
   Two callers publishing **different** content for the same **brand-new** `(key, version)`
   concurrently both see `existing_raw is None` and both proceed to write — the pre-existing additive
   race, unrelated to re-publish. Out of scope; flagged for awareness, not a blocker.
3. **`_diff_structures`/`_canonical_structure` reuse outside their original call site.** Low risk
   (§3.2 confirms no test asserts on the specific `source=` string), but the implementer should
   re-check `grep -rn '"source"' tests/test_services.py` after implementation in case a new test was
   added elsewhere in parallel.
4. **Whether "bump the version to repair a typo" is an acceptable answer for the stakeholder.**
   Candidate 3 (delete-and-republish) was rejected on scope-discipline grounds (§3.4), but if an
   operator's actual workflow leans heavily on quick def iteration during development (not just
   production authoring), the version-bump-only repair story may feel heavy. This is a product
   judgment call, not something this plan can resolve unilaterally — flagged for the coordinator/
   stakeholder if it becomes a friction point in practice.

---

## 8. Doc/docstring correction sweep

Every non-archive site the K-034 backlog table names, plus this plan's own new sites. **`docs/
archive/**` is excluded per root `AGENTS.md`'s frozen-document rule** ("a document that freezes does
not move... nothing is ever moved into [archive] again, and nothing is un-archived") — those sites are
historical record, not live claims, and stay untouched. Likewise `docs/HISTORY.md` (append-only dated
log — add a new entry, never rewrite an old one), `docs/reviews/*` (point-in-time verdicts), and
`docs/plans/workflow-def-structure-read.md`/`docs/plans/m3-followups-coordination.md` (already-executed
plans documenting the finding, not asserting current behavior) are **correct-as-is / out of scope** —
do not edit them.

Line numbers below are from this plan's own investigation (2026-07-31) and will drift; grep the quoted
text as the backlog item itself instructs.

| # | Site | Current claim | Correction |
|---|---|---|---|
| 1 | `docs/QUERIES.md` §11 preamble (currently ~`:786-836`, two paragraphs: the "versioned, immutable `WorkflowDef` templates" opening sentence, and the "Idempotency & the collapse idiom" paragraph ending "...Immutability per version comes for free from `MERGE`") | Re-publish is a structural no-op; immutability comes free from `MERGE` | Replace with K-034's own suggested framing: *"create-only on **properties** — a re-publish never updates a stored `name`/`kind`/step `config`/transition `guard`. It is **not** additive on **structure** as of K-034: `services.publish_workflow_def`/`materialize_def` reject a re-publish/re-materialize whose step set, transition set, or start key differs from what's stored (`409 WorkflowDefConflictError`), before any write. `Repository.publish_def`/`materialize_snapshot` themselves remain thin, non-validating primitives — the guarantee is enforced one layer up."* Do **not** touch the immediately-following sentence about the two-`UNWIND`-block row-multiplication fix — that's a different, true, still-accurate claim. |
| 2 | `docs/QUERIES.md` §11.1 footnote ("...run 2 → 0 created (idempotent), same row.") | Re-publish is idempotent (0 created) unconditionally | Qualify: "...run 2 (same content) → 0 created (idempotent), same row. A re-publish with a **different** topology (§11 preamble) is rejected at the service layer before reaching this query — see K-034." |
| 3 | `docs/QUERIES.md` §11.4 footnote ("Snapshots are immutable per `(workspace, key, version)`; re-materialize is a no-op.") | Snapshots are immutable, unconditionally | "Re-materialize with unchanged topology is a no-op on write (properties are always create-only). A re-materialize whose topology differs from the stored snapshot is rejected (`409 WorkflowDefConflictError`) before this query runs — K-034." |
| 4 | `docs/QUERIES.md` §11.1's existing `⚠️` warning box (the K-030 zero-transition note) | — | No change needed to this box's own content; optionally add one sentence cross-referencing the new conflict gate so a reader lands on both guards from one place. |
| 5 | `docs/DESIGN.md:102` (topology diagram) | "Canonical WorkflowDef templates (versioned, immutable)" | "Canonical WorkflowDef templates (versioned; topology-immutable per version, K-034)" |
| 6 | `docs/DESIGN.md:144, :147, :149` (§4 "Decision:" paragraph) | "(immutable, versioned)" / "immutable once published" / "the snapshot is immutable so it never drifts" | Reword to state the **enforced** guarantee precisely: topology (steps, transitions, start) is immutable per version (rejected at publish/materialize, K-034); properties are create-only (silently unchanged on a differing resubmit). Drop the unqualified "immutable" claims. |
| 7 | `docs/DESIGN.md:570` (§9 write-paths table, "Publish workflow def" row) | "Immutable per version; bump version, never mutate in place" | "Topology-immutable per version (rejected `409` on a differing re-publish, K-034); properties stay create-only. Bump version to change either." |
| 8 | `server/falkorchat/repository.py`, `publish_def` docstring (~`:1090-1096`) | "re-publishing the same `key@version` is a structural no-op (immutability per version)" | "This method itself does not enforce structural immutability — a differing payload for an existing `(key, version)` mints parallel `TRANSITION`/`START` structure (K-034). The guarantee is enforced by the caller: `services.publish_workflow_def` rejects a topology-differing re-publish before calling this method." |
| 9 | `server/falkorchat/repository.py`, `materialize_snapshot` docstring | "Re-materialize is a no-op." | Same correction pattern as #8, naming `services.materialize_def` as the enforcing caller. |
| 10 | `server/falkorchat/services.py`, `materialize_def` docstring | "Idempotent (the workspace MERGE no-ops on re-materialize)." | "Topology-immutable per `(key, version)` against the workspace snapshot (K-034): a re-materialize whose step/transition/start topology differs from what's already in `ctx.ws` raises `WorkflowDefConflictError` (409) before any write. Property-only differences stay a silent no-op (unchanged `MERGE … ON CREATE SET` behavior)." |
| 11 | `falkor-chat/AGENTS.md` (`seed_workflows.sh` row, ~`:75`) | "⚠️ Create-only, not update: re-running after editing a def is a silent no-op..." | "⚠️ Topology-immutable, not update (K-034): re-running after editing a def's steps/transitions/start now **fails loudly** (`409`) on the *materialize* half if the workspace snapshot predates the edit — publish a new version instead of editing this one. Property-only edits (name, step config, guard text) still silently no-op." Keep the existing K-031-added detection-pointer sentence ("Use `verify_workflows.sh`...") — do not remove it, it's still correct and now doubly useful (confirms nothing drifted before you hit the new 409). |
| 12 | `docs/requirements/agent-import.md:81` | "Published workflow defs are effectively immutable...; a re-import of a changed def cannot update in place" | Correct the "dangerous direction" the backlog flags: as of K-034, a re-import with a changed **topology** now fails (`409`), it does not additively corrupt; a **property-only** changed re-import still silently no-ops (does not update). Re-check this document's FR-2 idempotence requirement against the corrected claim — flag to `tico`/the doc's owner if FR-2's collision handling assumed the old "silently swallowed" framing. |
| 13 | `docs/requirements/workflow-dependence-overlay.md:21, :44, :54, :160` | Various "effectively immutable" / "create-only + immutable" framings | Update each to the corrected two-part framing (topology-immutable/enforced, properties create-only) — motivational text, low-risk, mechanical find/replace once the wording above is settled. |
| 14 | `docs/BACKLOG.md`, K-032's "why it exists" premise (~`:678`) | "This matters specifically because published defs are create-only + immutable (K-031)" | Update to "...because published defs are topology-immutable (K-034) and property-create-only" — note in the item that this may shift K-032's own motivating framing slightly; do not otherwise re-scope K-032. |
| 15 | `docs/BACKLOG.md`, K-029's premise paragraph (~`:531-535`) | "...published defs are create-only (`MERGE … ON CREATE SET`), so a byte-diff introduced while relocating a live def is silently swallowed" | Per K-034's own instruction: correct this. As of this fix, a *topology*-changing byte-diff (e.g. a retargeted transition introduced while relocating `triage@v1`'s literal) is **rejected**, not swallowed — the risk K-029 describes shifts from "silent corruption" to "the relocation publish fails loudly and must be resolved before landing," which is safer but still worth K-029's planned before/after equality check as a matter of process hygiene (a `409` mid-deploy is still worse than catching it in a pre-flight check). A *property*-only byte-diff (e.g. a `config` field reformatted during the move) still silently no-ops — K-029's equality check remains load-bearing for that half. |
| 16 | `server/falkorchat/repository.py:954` region (§11 block comment, "...immutable WorkflowDef templates...") and the "Publish/materialize share the same idempotent MERGE shape" comment near `_PUBLISH_CYPHER`'s definition | Weaker framing, same class | Align with #1's replacement framing (create-only-on-properties / topology-enforced-by-services, not "idempotent" unconditionally). |
| 17 | `server/falkorchat/api.py` §11 section comment (~`:230-233`, "Spec/​not-found errors map to 400/404...") | Doesn't mention 409 | Add: "a topology-differing re-publish/re-materialize maps to 409 (`WorkflowDefConflictError`, K-034)" — mirrors the existing §12 run-section error-map comment style. |
| — | `docs/archive/**` (all hits: `m1-chat-mcp.md`, `m3-executor-coordination.md`, `m3-workflow-engine.md`, `m3-workflow-engine-coordination.md`, `m3-process-flow.md`, `m3-guard-thread-context.md`, `archive/test-plans/m3-workflow-engine.md`, `archive/test-reports/m3-workflow-engine-report.md`) | Various | **Do not touch** — frozen historical record (root `AGENTS.md`: archive is read-only, nothing moves in or out). |
| — | `docs/HISTORY.md` (all hits) | Various dated entries | **Do not rewrite** — append-only log. Add a *new* entry for this change instead. |
| — | `docs/reviews/*` (`workflow-def-structure-read.md`, `k031-structure-read-impl.md`, `web-api-coverage-impl.md`) | Various | **Do not touch** — point-in-time verdicts; `web-api-coverage-impl.md:440`'s "immutable snapshot" is additionally **unrelated** (a `WorkflowRun`'s own append-only trace, not def/snapshot structure — different, true claim). |
| — | `docs/plans/workflow-def-structure-read.md`, `docs/plans/m3-followups-coordination.md` | Various | **Do not touch** — already-executed plans documenting/handing off the K-034 finding itself. |
| — | `docs/QUERIES.md:70, :298, :301, :321, :647, :749, :1147, :1206, :1413` and similar "no-op" hits | Various | **Unrelated** — different true no-op claims (member-identity MERGE, mention-block empty-`UNWIND` guard, run/step-run `DELETE`-of-null-edge idempotence). Do not touch. |

**Verification:** re-run `grep -rn -i "immutab\|no-op" docs/ server/falkorchat/ AGENTS.md` from
`falkor-chat/` after all edits land; every hit must fall into one of: corrected (per the table above),
correct-as-is (archive/HISTORY/reviews/executed-plans/unrelated, per the table), or a newly-discovered
site not in this table — the last case is a report to the coordinator, not a silent add.

---

## 9. Step-by-step implementation (owner: `tdd-engineer`)

Sequence so the tree stays buildable and each step is independently reviewable:

1. **Error plumbing (no behavior change yet).** Add `WorkflowDefConflictError` to `repository.py`
   (§4); re-export from `services.py`; register the 409 handler in `app.py`. No caller raises it yet —
   safe, isolated commit.
2. **`_structural_diffs` + `_check_no_structural_conflict` helpers** in `services.py`, next to
   `_diff_structures`/`_canonical_structure` (§3.2). Unit-test the filter directly (parametrized over
   the same path-grammar cases `test_diff_structures_one_class_at_a_time` already covers, asserting
   which survive the structural filter and which don't) before wiring it into any call site —
   red/green on the helper in isolation.
3. **Wire into `publish_workflow_def`** (§3.3). TDD: write the §6.1 FakeRepo tests (cases 1-3, 5-7)
   red first, then the four-line change to make them green. Confirm existing
   `test_publish_workflow_def_*_raises_nothing_written` tests (spec-validation failures) stay green
   unmodified.
4. **Wire into `materialize_def`** (§3.3). Same TDD cycle, mirroring step 3's tests for the
   materialize side.
5. **Executor tie-break** (§3.5). One-line change plus the §6.4 reversed-order test, red then green.
6. **API-layer tests** (§6.3). These exercise the real `_PUBLISH_CYPHER`/`materialize_snapshot` and are
   the strongest evidence the fix works end-to-end; run `./scripts/test_queries.sh` afterward per
   `AGENTS.md` rule 5 (should be unaffected — no query/DDL change, but the rule is unconditional).
7. **Repository contract-boundary test** (§6.2, recommended).
8. **Doc/docstring sweep** (§8) as its own reviewable commit/pass, followed by the grep
   verification (§6.5) and a new `docs/HISTORY.md` entry summarizing the change (per root `AGENTS.md`'s
   "append an entry for every delivered change" convention) and recording the grep classification
   outcome.
9. **Pre-flight note for whoever deploys this**, not a code step: per §7 risk 1, check
   `verify_workflows.sh`/`/diff` against any live `reference`/`ws:*` before this ships to an
   environment with real def data.

No `graph-dba` gate in this sequence (§2.3) — if step 2 or 3 discovers the pure-Python approach doesn't
hold up (it should, per the reuse analysis in §3.2, but if implementation surfaces a reason
`_PUBLISH_CYPHER` itself must change), stop and loop in `graph-dba` before continuing past that step.
