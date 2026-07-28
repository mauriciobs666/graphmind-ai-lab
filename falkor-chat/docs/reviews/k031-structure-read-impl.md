# Review — K-031 implementation: def/snapshot structure read surface

> **Status:** archived · **Owner:** `analyst` · **Tracks:** K-031 (M3 follow-ups)
>
> **Reviewer:** `analyst` · **Date:** 2026-07-24 · **Artifact:** uncommitted working tree
> **Baseline:** `docs/plans/workflow-def-structure-read.md` **v2** + its two gates
> (`docs/reviews/workflow-def-structure-read.md`, round 1 + re-gate RG-m1…RG-m5 / RG-n1…RG-n5)
> **Verdict: APPROVE WITH SUGGESTIONS** — 0 blocker · 1 major · 3 minor · 4 nit

---

## 1. Scope

**Reviewed (the coder's own files):**
`server/falkorchat/{repository,services,api,schemas}.py` ·
`server/tests/{test_repository,test_services,test_api}.py` · `scripts/verify_workflows.sh` (new) ·
`docs/{DESIGN,QUERIES,HISTORY,BACKLOG}.md` · `falkor-chat/AGENTS.md`.

**Excluded as instructed (concurrent K-027 slice A, no findings reported against them):**
`server/falkorchat/{llm,app}.py`, `server/tests/test_{llm,app}.py`,
`docs/reviews/k027-parse-robustness.md`, the K-034/K-035 backlog sections, the K-027 HISTORY entry,
and the K-027 bullet added to `AGENTS.md`.

### What I ran (evidence, not inference)

| Check | Result |
|---|---|
| `pytest tests/test_services.py -k "structure or diff_" tests/test_api.py::test_structure_response_model_start_keys_omission_all_three_directions` | **21 passed**, 0.44 s, offline (no graph touched) |
| `pytest --collect-only -q` (non-mutating) | **596/597 collected, 1 deselected** — matches the claimed exit collection |
| `pytest --collect-only` filtered to K-031 test names across the three files | **exactly 33** — the `+33` claim is arithmetically correct (3 repository · 20 service incl. the 10-case table · 10 API) |
| `ruff check .` | **All checks passed!** |
| `./scripts/verify_workflows.sh acme` (read-only) | **exit 0** — both `triage@v1` and `access-request@v1` present on both sides, `in sync: YES`, one start key each |
| Route table dump via `build_router` | no path shadowing (see §5.5) |
| `grep -rn -i "immutab\|no-op" docs/ server/falkorchat/ AGENTS.md` (R-8 done-condition) | run; see minor **3** |
| Nested `exclude_unset` propagation probe (pydantic) | confirmed — see nit **1** |

### What I deliberately did **not** run

- **The full pytest suite** and **`./scripts/test_queries.sh`** — both wipe the global `reference`
  graph, which is in a known-good re-seeded state and shared with a concurrent `analyst` run. The
  `596 passed` and `256/256` claims are therefore **unverified by execution**; §5.2 explains why
  256/256 is nonetheless guaranteed by construction.
- **V-1** — its `ws:k031probe` graph is gone by design; I judged its recorded outcome against the
  query text instead (§5.4).
- **Nothing mutating was run. Nothing was re-seeded. No file outside this review was touched.**

---

## 2. Direct answers to the three questions asked

### Q1 — Did K-031 hold its scope line? **Yes, cleanly. Verified, not assumed.**

No publish or materialize behaviour shifted:

- `server/falkorchat/executor.py` is **absent from `git status`** — literal zero-line diff.
- `repository.py`'s diff is **three additions only** (`_read_structure`, `read_def_structure`,
  `read_snapshot_structure`, `repository.py:999-1010`, `:1094-1104`, `:1578-1590`). `_PUBLISH_CYPHER`
  (`:937-956`), `_READ_META_CYPHER` (`:961`), `_READ_TRANSITIONS_CYPHER` (`:968`), `_read_subgraph`
  (`:975-997`), `read_def_subgraph`, `get_snapshot` and `materialize_snapshot` show **no diff hunks
  at all**.
- `services.py`'s only non-additive line is the import widening
  `from .schemas import MAX_CONFIG_LEN` → `…, MAX_DIFF_PREVIEW` (`services.py:48`). `schemas.py` is a
  pydantic-only leaf, so no cycle and no layering inversion. Everything else is new module-level
  helpers plus three new methods appended after `list_snapshots`.
- `api.py`/`schemas.py` are purely additive. All three new routes are `GET`; every repository read
  they reach goes through `graph.ro_query` (`repository.py:1030`, `:1046`) — `GRAPH.RO_QUERY` is a
  **server-enforced** write barrier, not a convention, which is the strongest possible form of the
  read-only claim.
- `verify_workflows.sh` calls only `services.diff_def_snapshot` / `get_workflow_def_structure` /
  `get_snapshot_structure`, all of which bottom out in the same `ro_query`. No publish, no
  materialize, no delete anywhere in the file.
- Plan non-negotiable 4 (**do not repair**) held: `verify_workflows.sh acme` is green *and* the
  HISTORY entry records the pre-re-seed `reference def: MISSING` observation as a **report**, with no
  repair. I re-ran the script and reproduce exit 0.

The only observable behaviour change in the component is three new read-only HTTP routes and one new
read-only script. That is exactly the mandate.

### Q2 — The "no new Cypher" claim. **Holds. `QUERIES.md`'s changes are documentation only.**

- I filtered every added line under `server/falkorchat/` for Cypher keywords: the only hits are
  Python `return` statements (word-boundary false positives, and all in the out-of-scope `llm.py`).
  **No query string was added anywhere.**
- `_read_structure` formats *the existing constants*, byte-for-byte
  (`Repository._READ_META_CYPHER.format(label=label)`, `repository.py:1031`;
  `_READ_TRANSITIONS_CYPHER`, `:1047`). The novelty is Python row handling above unchanged SQL-side
  text — precisely what the plan's §0.1 qualification described.
- `docs/QUERIES.md`'s diff is **five prose blocks and three comment lines**, no query body:
  - `§11.2` — the new multi-`START` blockquote recording V-1's outcome and the deliberate
    `start_keys`-vs-`startKey` shape divergence;
  - `§11.5` — the one-line mirror;
  - `§12.5` — the `maxSteps` note;
  - `:1034` and `:1254` — text appended to the two `// $maxSteps = …` **comment** lines inside the
    §12.1 / §12.12 fences. The executable statements in both fences are untouched.
- **`scripts/test_queries.sh` is not modified** (`git status` shows only `verify_workflows.sh` as new
  under `scripts/`), and no DDL, index, constraint or query the suite exercises changed. **256/256 is
  therefore invariant by construction**, which is why I did not pay the cost of re-running a suite
  that wipes `reference`. If you want the tripwire pulled for real, it must be run in a window where
  a `seed_workflows.sh acme` re-seed is acceptable.

### Q4 — RG-m3: is documentation sufficient mitigation, or a real regression hole?

**Both, in this order:** the coder's rejection of option (b) is sound *and* gate-compliant, but the
resulting hole is real and closable for ~15 lines that neither the gate nor the coder considered.

1. **The rejection of (b) is correct on its own terms, and it is what the gate recommended.** The
   re-gate's RG-m3 (`docs/reviews/workflow-def-structure-read.md:603-611`) offered (a) and (b) and
   closed with *"My recommendation: (a) plus a one-line comment in `_read_structure` pointing at
   V-1's recorded outcome; take (b) only if the coordinator accepts the K-034 coupling."* The coder
   took (a) and delivered the comment. Building the fixture by two `materialize_snapshot` calls
   really would assert publish-additivity — semantics K-034 is chartered to change — so the test
   would ship pre-broken.
2. **The hole is real.** `_read_structure`'s union loop (`repository.py:1036-1044`) is the **only
   novel logic** in U1, and its non-degenerate branch — `if row[2] is not None and row[2] not in
   start_keys` accumulating a second key, and the step union across rows — is never executed by any
   of the 33 tests. Every test feeds it a single meta row. **A refactor back to `result_set[0]` would
   leave the entire suite green**, silently un-wiring both the `startKeys` field and
   `verify_workflows.sh`'s finding-3 tripwire (`scripts/verify_workflows.sh:142-159`), which is the
   only thing in the repo that will ever notice a two-`START` root in production.
3. **Documentation is honest but is not a guard.** The disclosure is genuinely good — the docstring
   (`repository.py:1010-1017`), `QUERIES.md` §11.2's blockquote, and HISTORY's explicit *"Deliberate
   residual: the multi-row branch is evidenced by V-1 only and is not pinned in the suite"* all say
   the same true thing. None of them fails a build.
4. **There is a third option the gate's (a)/(b) framing missed** — see major **M-1**. It closes the
   hole with **zero** coupling to publish semantics, zero Cypher, and no FalkorDB.

Net: the disposition was reasonable and well-documented; the hole should still be closed, and it is
cheap. It does not block.

---

## 3. Findings

### Major

#### M-1 · The multi-row read path is testable *without* K-034 coupling — `_read_structure` is a `@staticmethod` over an injected `graph`

**Evidence.** `repository.py:999` — `def _read_structure(graph, *, label, key, version)` is a
`@staticmethod` whose only collaborator is the `graph` object, used exactly twice
(`graph.ro_query(...)` at `:1030` and `:1045`). It needs no `Repository` instance, no connection, no
schema and no publish. A fake with a canned two-row `result_set` exercises the union loop directly:

```python
class _FakeRes:
    def __init__(self, rows): self.result_set = rows
class _FakeGraph:
    def __init__(self, meta, trs): self._r = [_FakeRes(meta), _FakeRes(trs)]
    def ro_query(self, q, p): return self._r.pop(0)

STEPS = [{"key": "a", "type": "decision", "config": ""},
         {"key": "b", "type": "decision", "config": ""}]
g = _FakeGraph(meta=[["probe", "process", "a", STEPS],
                     ["probe", "process", "b", STEPS]], trs=[[[]]])
out = Repository._read_structure(g, label="WorkflowDefSnapshot", key="probe", version="v1")
assert out["start_keys"] == ["a", "b"]          # V-1's recorded shape, pinned
assert [s["key"] for s in out["steps"]] == ["a", "b"]   # union de-dupes across rows
```

**Why it matters.** This pins the exact behaviour V-1 observed live, asserts only the *reader*, and
depends on **nothing** K-034 can change — the fake row shape is `_READ_META_CYPHER`'s `RETURN` list
(`repository.py:965-966`), which K-034 does not touch. Option (b)'s objection ("couples K-031's suite
to publish-additivity") simply does not apply, because no publish runs.

**Suggested improvement.** Add one test in `tests/test_repository.py` (or `test_services.py`, since
it needs no FalkorDB) shaped as above, with a comment citing V-1's recorded outcome and QUERIES.md
§11.2. Optionally a second row-shape asserting the null-`startKey` case (`row[2] is None`), which is
the other unexercised branch of the same loop. Owner: `tdd-engineer` or `coder`. This is the
suggested closure of RG-m3's residual, not a defect in what shipped.

### Minor

#### m-1 · `schemas.py` is claimed as a `maxSteps` documentation site in three places; the note is not there

**Evidence.**
- `docs/HISTORY.md` (K-031 entry): *"now land at six sites: DESIGN §6, QUERIES §12.5 + the two
  `$maxSteps` comments, **`schemas.py`**, and `AGENTS.md`'s executor-invariants block."*
- `docs/BACKLOG.md` K-031 delivered bullet and K-033's "Why it exists" paragraph repeat the same
  five-item list including `schemas.py`.
- `grep -n "maxSteps\|max_steps" server/falkorchat/schemas.py` returns exactly two lines:
  `MAX_RUN_STEPS = 50` (`:175`) and `maxSteps: int | None = Field(None, ge=1, le=MAX_RUN_STEPS)`
  (`:201`). **Neither carries the `maxSteps + 1` text**, and `schemas.py`'s diff contains no
  `maxSteps` change at all.

**Root cause (so the fix is not a guess).** Plan §5 says the text lands *"at the six sites listed in
§8"*, but §8 enumerates **five**: DESIGN §6 · QUERIES §12.5 note · QUERIES `:1034` comment · QUERIES
`:1254` comment · AGENTS.md executor block. All five landed and read well. The coder appears to have
reconciled the plan's own miscount by naming a sixth site that was never written.

**Why it matters.** K-033's backlog entry is a to-do list for a future implementer who must re-sync
the same six sites when the fix lands; one of them does not exist, so they will hunt for it. And
`schemas.py:201` — `StartWorkflowRunIn.maxSteps`, the **caller-facing request field** — is arguably
where the `+1` semantics matter most and is the one place a REST client author would look.

**Suggested improvement.** Pick one: (a) add a one-line comment above `schemas.py:201` —
`# A tripwire checked AFTER each recorded step ⇒ a run executes at most maxSteps + 1 (DESIGN §6).` —
making the claim true and improving the surface; or (b) correct the three "six sites" claims to five.
(a) is better.

#### m-2 · R-3's exact-key-set anti-drift assertion covers the two structure routes but not `/diff`

**Evidence.** Plan R-2/R-3 (`§6`): *"contract tests assert exact key sets on both structure routes
**and the diff**."* `tests/test_api.py` defines `_STRUCTURE_KEYS` and asserts
`set(body) == _STRUCTURE_KEYS` on both structure routes (the def route and the snapshot route). The
diff tests (`test_diff_identical_def_and_snapshot_is_in_sync`,
`test_diff_reports_divergence_after_the_documented_reseed_trap`,
`test_diff_def_missing_snapshot_present_is_200_with_presence_flags`) assert individual fields only —
there is **no** `set(body) == {...}` on `WorkflowDiffOut` and none on a `WorkflowDiffEntry`.

**Why it matters.** `response_model=` *filters* undeclared fields (the exact hazard `schemas.py`'s new
module docstring names). A future field added to the service dict but not to `WorkflowDiffOut` — or a
field silently dropped from it — would pass all ten API tests. The `def_ → "def"` alias is exercised
(the divergence test indexes `d["def"]`), so a broken alias would fail; a *missing key* would not.

**Suggested improvement.** In `test_diff_identical_def_and_snapshot_is_in_sync`, add
`assert set(body) == {"key","version","defPresent","snapshotPresent","inSync","differences","differenceCount"}`,
and in the divergence test add `assert set(body["differences"][0]) == {"path","def","snapshot"}`. Two
lines; closes R-3 for all three routes.

#### m-3 · The R-8 classification reported two un-enumerated "immutable" sites; there is at least a third

**Evidence.** Running R-8's done-condition from `falkor-chat/`
(`grep -rn -i "immutab\|no-op" docs/ server/falkorchat/ AGENTS.md`, excluding archive/plans/reviews
and HISTORY), the hits of the falsified class that appear in **neither** K-034's ten-site table nor
its "four further sites" list are:

| Site | Text | Reported? |
|---|---|---|
| `docs/requirements/agent-import.md:81` | "Published workflow defs are effectively **immutable** … a re-import of a changed def cannot update in place" | ✅ reported by the coder |
| `docs/requirements/workflow-dependence-overlay.md:21`, `:54` | "published defs are effectively immutable" / "before I edit an (immutable) published def" | ✅ reported by the coder |
| `docs/BACKLOG.md:655` (K-032's premise) | "This matters **specifically because published defs are create-only + immutable** (K-031)" | ❌ not reported |

**Why it matters.** Low stakes — K-034's own test-strategy done-condition is a `grep` over the same
`docs/` tree (`BACKLOG.md:852`), so all four will be swept when K-034 lands. But R-8 asked for a
**three-way classification of every hit**, and the delivered report enumerated two of three.

**Suggested improvement.** Add `docs/BACKLOG.md:655` to the same report line to `teco`. No doc edit —
correcting the prose is K-034's, per plan §1.2.

### Nits

1. **`response_model_exclude_unset=True` propagates into nested models.** I verified this against the
   installed pydantic: a nested model built from a dict missing a *defaulted* field serializes
   without it. Today `WorkflowStepOut` and `WorkflowTransitionOut` have only required fields, and
   `test_structure_response_model_serializes_from_not_from_underscore` pins the transition key set,
   so the trap is closed — but by accident, not by design. One sentence in
   `WorkflowDefStructureOut`'s docstring ("nested step/transition models must keep every field
   required, or `exclude_unset` will drop defaulted ones") would make it deliberate.
2. **`verify_workflows.sh` reads each side twice per def.** `diff_def_snapshot`
   (`scripts/verify_workflows.sh:108`) already canonicalizes both structures, then `:150-151` re-reads
   both purely for the `startKeys` tripwire — four extra RO queries per def. Harmless at n=2 defs;
   noted only because the alternative (surfacing `startKeys` on the diff envelope) would also make
   the tripwire visible over HTTP, which it currently is not.
3. **The two structure routes return different error-body shapes.** The def route 404s through
   `WorkflowDefNotFoundError` → `{"error": "WorkflowDefNotFoundError", …}` (`app.py:86`); the snapshot
   route raises a plain `HTTPException(404, "workflow snapshot not found")` → `{"detail": …}`. This is
   **plan-mandated** (§3.2, mirroring `get_workflow_def`'s style), so it is not a deviation — but
   DESIGN §14.4 advertises the two bodies as *"identical shape apart from `source`"*, which a client
   author will reasonably read as covering the error path too. One clause in that paragraph would
   pre-empt it.
4. **`_diff_preview` joins `startKeys` with `","`** (`services.py:262`), so a start key containing a
   comma is ambiguous in the preview. Theoretical (step keys are author-chosen identifiers) and the
   full value is one structure read away.

---

## 4. Also assessed — the two reported-but-not-acted-on assertions

**The call to report rather than correct was right, and both are genuinely falsified.**

- **`docs/requirements/agent-import.md:81`** — *"Published workflow defs are effectively **immutable**
  (`MERGE … ON CREATE SET`); a re-import of a changed def cannot update in place — it needs a version
  bump or an explicit teardown."* Under the K-034 finding this is not merely imprecise, it is
  **optimistic in the dangerous direction**: a re-import of a changed def does not fail to update, it
  **additively mutates** — a new step key mints a `Step` + `HAS_STEP`, a retargeted transition mints a
  parallel `TRANSITION`, a changed start step mints a second `START`. The line is load-bearing there
  (it is the stated collision with that document's FR-2 idempotence requirement), so an architect
  designing agent-import off it would design against the wrong hazard.
- **`docs/requirements/workflow-dependence-overlay.md:21` and `:54`** — *"published defs are
  effectively immutable"* / *"before I edit an (immutable) published def"*. Same class, weaker
  framing: these are motivational rather than load-bearing, matching K-034's own "four further sites"
  category.

**Why reporting was correct.** Plan §1.2 and §8's scope note forbid K-031 from correcting *any*
"immutable / no-op" prose — that entire deliverable class is K-034's, precisely so the corrections
land once, together, against a settled decision about what publish semantics should become. R-8's
done-condition explicitly routes a hit that is in none of its three classes to *"a report to teco"*,
which is what happened. Correcting them here would also have pre-committed the wording before K-034
decides whether publish stays additive, becomes reject-on-difference, or becomes upsert.

---

## 5. What's solid

1. **Layering is exactly as locked.** Cypher constants untouched in `repository.py`; canonical
   ordering and the comparator are pure module-level functions in `services.py` (`_canonical_structure`,
   `_diff_preview`, `_transition_path`, `_diff_structures`) with no `Repository` dependency; `api.py`
   is a three-route adapter. `_read_structure`'s docstring carries all three things U1.1 demanded
   (why it duplicates `_read_subgraph`, why it reads every row, why the shape differs from §11.2) and
   cites the §11.2 note.
2. **The tripwire held statically.** Zero new/modified Cypher, verified by keyword scan of every added
   line, and `test_queries.sh` is untouched. `executor.py` diff is literally zero lines; the
   `_drive_loop` SHA lock is intact.
3. **RG-m1's third option is correct, and I verified the mechanism rather than the reasoning.**
   `response_model_exclude_unset=True` keys off which fields the service dict actually populated:
   `_canonical_structure` always writes `"startKey"` (even as `None`) and writes `"startKeys"` only
   when `len(start_keys) > 1` (`services.py:241-249`), so absent stays absent and an explicit `null`
   survives. `test_structure_response_model_start_keys_omission_all_three_directions` pins all three
   directions — including the anomaly case (`startKey: None` present in the body) — against a
   throwaway `FastAPI` app with no graph, and it passes. The coder's argument against `exclude_none`
   is right: `startKey` is nullable, and `exclude_none` would delete the one field that reveals a root
   with no `START` edge. It interacts with no other field on either model (see nit 1 for the only
   latent edge).
4. **RG-m2 is properly closed.** `start_keys = sorted(raw.get("start_keys") or [])` then
   `startKey = start_keys[0]` **after** sorting (`services.py:240-246`), with the reason in the
   docstring; `test_def_structure_start_keys_omitted_for_one_present_for_two` feeds `("z", "a")` and
   asserts `["a", "z"]` / `"a"`.
5. **The diff is correct on the two things that decide whether it is useful.**
   - **Transition identity is the 4-tuple** `(from, to, on, order)` — `services._identity`, `:318-322`
     — taken from `_PUBLISH_CYPHER`'s actual `MERGE` key
     (`MERGE (from)-[rel:TRANSITION {on, order}]->(to)`, `repository.py:952`), not guessed. And it is
     *tested for the failure mode a `(from,to)` key would produce*:
     `test_diff_structures_changed_transition_endpoint_reads_as_two_presences` asserts a retargeted
     edge reads as two presence rows, never one "modified" row. Dict-keying by identity is safe
     against collapse because `MERGE` on `{on, order}` + endpoints cannot produce two edges with the
     same 4-tuple.
   - **One side missing is a 200 with flags** — `diff_def_snapshot` (`services.py:900-935`) raises
     `WorkflowDefNotFoundError` only when **both** sides are absent, and otherwise returns
     `defPresent`/`snapshotPresent` with `differences: []`. Covered at the service layer
     (`test_diff_def_snapshot_in_sync_and_one_sided`) and over HTTP against a real graph
     (`test_diff_def_missing_snapshot_present_is_200_with_presence_flags`), with `test_diff_both_absent_is_404`
     on the other side. `inSync = both and not differences` matches §3.3's formula exactly. And the
     live script *observed* the one-sided state in the wild before the re-seed.
   - Meta comparison covers `name`, `kind`, `startKey` **and** `startKeys`, and deliberately excludes
     `source` (which always differs) and `key`/`version` (equal by construction). `stepCount` /
     `transitionCount` are not compared, correctly — a count difference is implied by a presence
     difference, so there is no false negative and no duplicate row.
6. **Route registration is clean.** Dumping `build_router`'s routes in order gives
   `GET /workflow-defs` → `GET /workflow-defs/{key}` → `GET /workflow-defs/{key}/versions/{version}` →
   `POST …/materialize` → `GET /workspaces/{ws}/snapshots` → `…/{key}/versions/{version}` → `…/diff`.
   No shadowing (Starlette's compiled regexes are segment-anchored), and `app.py` was correctly left
   untouched per U3.3.
7. **Test quality is high, and the 10-case table is not filler.** Each parametrized case isolates one
   difference class and asserts `len(diffs) == 1` **plus the exact difference dict** — so a
   comparator that over-reports fails, not just one that under-reports. The `meta.startKeys` case is
   constructed so `startKey` is *equal* on both sides (`["s"]` vs `["s","t"]` → both `"s"`), which is
   what makes it a single-class case rather than an accidental two-row assertion. The API tests pin
   byte-identical opaque `config`/`guard` strings (rule 8) and exact key sets. The divergence fixture
   reproduces the documented trap by wiping `reference` in-test and re-publishing edited into the
   empty graph — depending on **create** semantics only, exactly as RG-m2's re-specification asked,
   with the four expected difference paths asserted individually.
   `test_republish_is_create_only_on_properties_structure_read_unchanged` correctly keeps `kind`
   inside `WORKFLOW_KINDS` (gate n-4) *and* keeps `waitsForHuman: true` on the `human` step (RG-m5),
   with a comment stating it pins a decision and that the additive half is K-034's.
8. **`verify_workflows.sh` does what its header promises.** Read-only by construction (every path is
   `ro_query`), works with no uvicorn, sources both expected `(key, version)` pairs from the *same*
   constants `seed_workflows.sh` publishes from (`config.TRIGGER_DEF_KEY/_VERSION`,
   `proof_defs.ACCESS_REQUEST_DEF["key"]/["version"]`) so it cannot drift from the seed, tolerates a
   cold graph (`ResponseError "empty key"` → treated as absent), prints the seed command on MISSING
   and explicitly refuses to suggest a re-seed on **divergence**. The `chmod +x` bit is set. Check 1
   really is the version-staleness check `/diff` cannot do: because the expected version comes from
   the config source of truth, a workspace still holding the previous version reports
   `snapshot: MISSING`.
9. **Documentation is materially good, and I read it rather than accepting the claim.**
   `AGENTS.md`'s `seed_workflows.sh` row is **byte-identical up to the appended detection sentence** —
   the create-only / "silent no-op" wording was correctly left alone for K-034; the `test_queries.sh`
   row gains a pointer; a new `verify_workflows.sh` row lands; the executor-invariants block gains the
   `maxSteps` bullet. `DESIGN.md` §14.4 followed the section's convention (m-3, option B): the
   `§11`/`§12` exclusion parenthetical was **extended**, no table rows added, with the four
   operator-facing facts and the receipt-vs-stored sentence verbatim. `QUERIES.md` §11.2's blockquote
   records V-1's actual observation and the shape divergence. `README.md` was correctly left alone —
   I confirmed it enumerates no endpoints (one `GET /health` mention at `:110`).
10. **The delivery record is honest in the places where it would have been easy not to be** — the
    unpinned multi-row branch is called a "deliberate residual"; the 596-vs-585 test-count gap is
    attributed to the concurrent K-027 unit rather than claimed; the live R-1 finding is recorded as
    a report.

---

## 6. Open questions (for the caller, not fixes)

1. **Does M-1 land now or as a K-034 rider?** The fake-graph test is ~15 lines and belongs to K-031's
   own logic, but K-034 will build real multi-`START` fixtures anyway. Landing it now is cheap
   insurance against a refactor in the window between the two items; folding it into K-034 saves a
   commit. My recommendation: now, because the window is exactly when `verify_workflows.sh`'s tripwire
   is the only detection in the repo.
2. **Should `startKeys` be surfaced on the `/diff` envelope?** Today the multi-`START` anomaly reaches
   an HTTP client only as a `meta.startKeys` **difference row** — i.e. only when the two sides
   *disagree*. A def and snapshot that both have two `START` edges are reported `inSync: true`.
   `verify_workflows.sh` catches that (it queries both structures separately), a `curl …/diff` does
   not. Not a defect against the plan — §3.3's envelope is exactly as designed — but worth a
   deliberate decision before K-034 makes the state more likely.
3. **Who owns the `maxSteps` note at `schemas.py:201`** (m-1 option (a))? It is a one-line comment,
   but it is a source file, and K-031 is closed.
