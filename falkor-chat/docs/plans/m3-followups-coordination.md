# M3 follow-ups — coordination log (option A: K-027 slice A + K-031)

> **Status:** active · **Owner:** `teco` · **Tracks:** K-027 · K-031 (M3 follow-ups)
>
> Coordinator: `teco`. Started 2026-07-24. Scope decided by the user: run **K-027 slice A**
> (parse-layer robustness) and **K-031** (def/snapshot structure read surface) **in parallel** —
> the two cheapest, highest-leverage items on the post-M3 debt path.
> Backlog items: `docs/BACKLOG.md` → K-027 (items 1 + the 2026-07-21 addendum) and K-031.
> Everything else on the debt path (K-028, K-029, K-030, K-032, and K-027 items 2–5) is **out of
> scope for this run** and must not be folded in.

## Entry baselines (from HISTORY.md 2026-07-21, to be re-confirmed by each unit)

| Gate | Baseline |
|---|---|
| `server` pytest | **533 passed / 1 deselected** (report `N passed, M skipped` — a green line with FalkorDB down means half the suite silently skipped) |
| `scripts/test_queries.sh` | **256/256** |
| `pytest -m live` | **RED 2/2** on the AC-4 answer-post assertion — a *known, filed* limitation (DEF-K027-A / D12-B), **not** a regression |
| `ruff check .` in `server/` | clean since 2026-07-24 (not a wired gate) |

## Units

| # | Unit | Owner | Depends on | Done-condition |
|---|---|---|---|---|
| **U1** | K-027 slice A — widen `_parse_content_tool_calls` to bare `name({json})` call syntax + make the guard judge's parse fence/prose-tolerant | `tdd-engineer` | — | Offline pins red→green; pytest ≥ 533 at the new enumerated count; docs updated |
| **U1-G** | Review gate on U1's delivered diff | `analyst` | U1 | Verdict approve / approve-with-suggestions → `docs/reviews/k027-parse-robustness.md` |
| **U2** | K-031 — design the def/snapshot **structure** read surface | `architect` | — | Plan at `docs/plans/workflow-def-structure-read.md` |
| **U2-G1** | Review gate on the U2 plan | `analyst` | U2 | Verdict → `docs/reviews/workflow-def-structure-read.md` |
| **U3** | K-031 implementation per the approved plan | `coder` | U2-G1 | Endpoints + tests green; docs updated |
| **U3-G** | Review gate on U3's delivered diff | `analyst` | U3 | Verdict appended to the same review doc |

U1 and U2 are **fully parallel** — U1 touches `server/falkorchat/llm.py` + `app.py` + their tests;
U2 is a design pass over the workflow-def read path (`api.py` / `services.py` / `repository.py`).
No shared file.

## Documentation-impact scan (folded into each unit's done-condition)

| Doc | U1 | U3 (K-031) |
|---|---|---|
| `docs/HISTORY.md` | entry required | entry required |
| `docs/BACKLOG.md` | mark K-027 item 1 + addendum (a)/(b) delivered; **leave items 2–5 open** | K-031 → delivered; note the budget off-by-one's disposition |
| `falkor-chat/AGENTS.md` | only if a stated invariant changes | likely — the create-only trap is documented there; add how it is now **detectable** |
| `docs/DESIGN.md` §14.4 (REST surface → service → verified query) | no | yes — new endpoints belong in the table |
| `docs/QUERIES.md` §11.3 | no | only if new Cypher lands (not expected — `read_def_subgraph` exists) |
| `README.md` | no | only if it enumerates endpoints |

## Hazards travelling with every brief

- **A plain `pytest` run wipes the global `reference` graph** (the `wf_repo` fixture, at *setup*),
  so published defs (`triage@v1`, `access-request@v1`) disappear while each `ws:<id>` snapshot
  survives. **Re-seed with `./scripts/seed_workflows.sh <wsId>` after a pytest run**; the same is
  true after `./scripts/test_queries.sh` (which deletes `reference` at *teardown*).
- **Published defs are create-only** (`MERGE … ON CREATE SET`) — an edited re-publish returns `201`
  while stored **properties** keep their old values. Never "fix" a def by re-running a seed.
  **⚠️ Corrected 2026-07-24 (analyst-confirmed, → K-034):** create-only is *not* a pure no-op — it is
  **additive**. The `MERGE` *patterns* still create structure: a changed `to`/`on`/`order` mints a
  **parallel `TRANSITION`** edge and a changed start step a **second `START`** edge. Only `guard`,
  `d.name`, `d.kind`, `st.type`/`st.config` are truly create-only. Consequences: duplicate transitions
  tie `executor._select_transition`'s sort ⇒ arbitrary branch selection; a two-`START` snapshot breaks
  `start_run` against the `runId` uniqueness constraint. This supersedes the "silent no-op" wording
  still present in `falkor-chat/AGENTS.md` and nine other doc sites — correcting them is **K-034's**
  deliverable, not K-031's.
- `_drive_loop` is **SHA-locked** (`71055f756280`). Neither unit has any business touching it.

## Log

- **2026-07-24** — user chose option A. Coordination doc created; U1 (`tdd-engineer`) and U2
  (`architect`) dispatched in parallel.
- **2026-07-24** — **U2 delivered**: plan at `docs/plans/workflow-def-structure-read.md` (638 lines,
  units U1–U6 + a pre-implementation live verification V-1). The "no new Cypher" assumption **held**
  (`_READ_META_CYPHER`/`_READ_TRANSITIONS_CYPHER` are label-parameterised and serve both sides) ⇒ no
  graph-dba gate; `test_queries.sh` must stay at **exactly 256/256** as the scope-creep tripwire.
  Two unexpected findings: **create-only re-publish is *additive*, not a pure no-op** (a changed
  `to`/`on`/`order` creates a second `TRANSITION`; a changed start step a second `START`), and the
  §11.2 single-row collapse is therefore **conditional** — `_read_subgraph`'s `result_set[0]` is
  arbitrary on a two-`START` def. `maxSteps` off-by-one: recommended **document-only**, with the real
  fix (`>` → `>=`, both inside the SHA-locked `_drive_loop`, pinned by `test_executor.py:158`)
  proposed as a new **K-033** bundled with K-027 item 2, which must break the lock anyway.
  Three stakeholder open questions (§9) raised with the user.
  U2-G1 (`analyst` plan gate) dispatched.
- **2026-07-24 — stakeholder decisions on the plan's §9 open questions (user, verbatim: "go with the
  recommendations"). These are now binding on U3 and are not to be reopened by the implementer:**
  - **OQ-1 — `maxSteps` off-by-one: DOCUMENT-ONLY.** Do **not** change `executor.py:410`/`:427`; the
    SHA lock on `_drive_loop` (`71055f756280`) stays intact and `test_executor.py:158` keeps its
    current assertion. Document the semantics ("`maxSteps` is a tripwire checked *after* each step;
    a run executes at most `maxSteps + 1` steps") at the plan's six named doc sites, and **file
    K-033** in `docs/BACKLOG.md` proposing the real fix. **Corrected 2026-07-24 (analyst RG-m6, to
    match plan v2):** K-033 is filed **self-standing**. Bundling it with **K-027 item 2** (the
    terminal-node engine contract) is a stated *preference* — one re-lock ceremony, two fixes — whose
    premise that "K-027 item 2 must break the lock anyway" is **explicitly unverified** (the relevant
    seams `_execute_step`/`_select_transition`/`_trace_step`/`resume` sit *outside* the lock,
    `AGENTS.md:256-257`, and K-027 is 🔵 proposed/unscheduled). Do not let K-033 depend on K-027.
  - **OQ-2 — the diff endpoint lives under `/workspaces/…`**, per the plan's recommendation.
  - **OQ-3 — repairing live def/snapshot divergence is OUT OF SCOPE.** If V-1 or the new read
    reveals that the live `triage@v1` / `access-request@v1` def and snapshot are already out of sync,
    **report and file it — do not repair it.** Deleting a snapshot breaks live `WorkflowRun`s via
    `OF_DEF`/`AT_STEP`: a destructive shared-state op, not a routine re-seed.
- **2026-07-24 — stakeholder decision on the additive-`MERGE` finding (user): file it as its OWN
  backlog item, not as a rider inside K-031.** The architect's claim is that create-only re-publish
  is *additive*, not a pure no-op — a changed `to`/`on`/`order` MERGEs a **second `TRANSITION`** edge
  beside the old one and a changed start step a **second `START`** edge, which the executor then
  drives. If confirmed, this contradicts what `falkor-chat/AGENTS.md` currently documents (create-only
  described as a silent *no-op*) and is a materially worse hazard than the documented one.
  **Sequencing (deliberate): the item is filed only once the `analyst` gate returns**, because that
  gate is reviewing this exact claim against `repository._PUBLISH_CYPHER` right now — filing first
  would either pre-commit an unverified defect to the backlog as fact or duplicate the review in
  flight. The filing brief will carry the analyst's evidence.
  **Consequences for K-031:** the finding leaves K-031's scope entirely (K-031 stays "make the
  current semantics observable"); the plan's §7 test for additive re-publish and any `AGENTS.md`
  correction of the create-only description move to the new item. K-031's structure read remains the
  *detection* mechanism and should cross-reference the new item, not absorb it.
- **2026-07-24 — U2-G1 (`analyst` plan gate) returned `needs changes`** — 2 blocker · 3 major ·
  6 minor · 4 nit → `docs/reviews/workflow-def-structure-read.md`. The design itself was judged right
  (all ~25 file:line citations resolve; both blockers are scoping/handoff corrections, not a rethink).
  - **Claim 1 "no new Cypher" — CONFIRMED.** `_READ_META_CYPHER`/`_READ_TRANSITIONS_CYPHER` are
    `{label}`-templated and already formatted with both `WorkflowDef` and `WorkflowDefSnapshot` by the
    shared `_read_subgraph` (`repository.py:961/968/976`). No graph-dba gate; the 256/256 tripwire stands.
  - **Claim 2 additive-`MERGE` — CONFIRMED and worse than the plan stated** (`_PUBLISH_CYPHER:937-956`,
    shared verbatim by `materialize_snapshot:1483`). Only `guard`, `d.name`, `d.kind`, `st.type`/
    `st.config` are truly create-only. **Two live consequences:** (a) duplicate transitions tie
    `executor._select_transition`'s sort on `(guard=="", order)` ⇒ the branch taken is whatever edge
    order FalkorDB returns; (b) a two-`START` snapshot runs `repository.start_run`'s `CREATE
    (r:WorkflowRun …)` twice against `UNIQUE NODE WorkflowRun … runId` (`bootstrap_schema.sh:180`) ⇒
    **run start breaks**. It falsifies **ten shipped assertions** (QUERIES §11 preamble/§11.1/§11.4,
    DESIGN:544/102/144-149, three docstrings, the AGENTS.md `seed_workflows.sh` row) and undermines
    **K-029's core risk premise** (K-029 believes an errant edit is "silently swallowed"; it is
    additive, making K-029's proposed def relocation *more* dangerous than K-029 believes).
  - **B-1:** V-1 was unexecutable — `publish_def` targets the hardcoded global `reference`
    (`db.py:87-94`), so no throwaway workspace can exist for a def publish; probe the snapshot side
    (`ws:k031probe`) instead.
  - Majors: publish receipt counts are the *submitted* spec, not the stored def (inverting §3.2's
    comparability claim exactly in the additive case); the K-033 bundling premise is **unverified** and
    K-027 is unscheduled; the `533 → 533 + N` gate is stale (tree collects **551** with U1's churn).
  - **Number reservations (assigned by the coordinator to prevent collision):** **K-033** = the
    `maxSteps` off-by-one fix (filed by K-031's implementer per OQ-1); **K-034** = the additive-`MERGE`
    defect + its ten doc corrections.
  - U2 revision re-briefed to `architect` with the review path. **K-034's filing is deliberately held
    until U1 (`tdd-engineer`) lands** — U1's done-condition edits `docs/BACKLOG.md` and
    `docs/HISTORY.md`, and a concurrent filing would collide in those two files.
- **2026-07-24 — U2 revision delivered: plan v2.** Architect reports **every** finding adopted
  (2 blocker · 3 major · 6 minor · 4 nit), **none rejected on merits**; dispositions in the plan's new
  **§11**. Highlights: V-1 rebuilt as an explicit *write* probe on the **snapshot side** in a
  throwaway `ws:k031probe` (with the architect's own added trap — the probe def needs ≥1 transition or
  `materialize_snapshot`'s `result_set[0]` `IndexError`s, the K-030/O-6 shape); the additive-`MERGE`
  finding fully handed off to **K-034** (§7 test deleted, §8 scoped, new **R-9** covering "K-034 not
  yet filed when U6 runs" ⇒ write "filing in flight", report, never self-file); §3.2's comparability
  rationale corrected to receipt-counts-**submitted** vs read-counts-**stored**; **K-033** now
  self-standing with the K-027-item-2 bundling premise explicitly marked *unverified*; the stale suite
  gate replaced by `entry → entry + N` measured at U3 start via a non-mutating
  `pytest --collect-only -q` (architect measured **552 collected / 1 deselected**; reviewer saw 551 —
  a moving number by design, while `test_queries.sh` stays pinned at **256/256**). OQ-3 promoted from
  an aside to explicit instruction **R-1** (report and file divergence, never repair).
  U2-G1 **re-gate** dispatched to `analyst`, with an explicit "did the revision introduce *new*
  problems?" mandate — a 15-finding revision is exactly where cross-section contradictions appear.
- **2026-07-24 — U2-G1 re-gate: `approve with suggestions` ⇒ the K-031 plan is APPROVED for
  implementation.** All **15** round-1 findings **closed**, each verified against the code rather than
  against the plan's §11 self-summary; **nothing re-opened**. B-1's four sub-checks all hold (the
  def→snapshot transfer argument, airtight isolation, the ≥1-transition trap, and `bootstrap_reference`
  being DDL-only at `bootstrap_schema.sh:238`); B-2's handoff is genuine and no surviving unit depends
  on K-034's semantics; the reworked divergence fixture relies on *create* semantics only.
  Reviewer's answer to "can a `coder` execute this as written?" — **yes**.
  New findings: **0 blocker · 0 major · 6 minor · 5 nit** — they surface as failing tests, not silent
  defects, and travel into the U3 brief by path. Coordinator-owned **RG-m6 is now FIXED** (both stale
  passages in this file corrected above: the create-only "silent no-op" wording, and K-033's
  bundling premise).
  **Next actions, deliberately serialized behind U1** (both would edit `docs/BACKLOG.md` /
  `docs/HISTORY.md` concurrently with it): (1) file **K-034** with the analyst's evidence; (2) dispatch
  **U3** (`coder`) on plan v2 with the re-gate's RG-m1…RG-m5 + nits attached.

## ⏸ PAUSED 2026-07-24 (user: out of credits) — RESUME ANCHOR

**This file is the resume anchor.** Nothing is committed; the working tree is dirty and intentionally so.

**Delivered and verified by the coordinator (not yet reviewed):**
- **U1 — K-027 slice A** ✅ delivered by `tdd-engineer`. `llm._parse_bare_call_syntax` (bare
  `name({json})` recovery, second probe *after* the JSON probe, precedence untouched) + public
  `llm.extract_json_object` shared with `app._build_llm_judge` (fenced/prose-wrapped verdicts now
  parse). 19 new tests, 11 red before / all green after; 8 negative-direction pins green throughout.
  Coordinator-verified: **552/553 collected, 1 deselected**, `ruff check .` clean, diff confined to
  `llm.py`/`app.py`/`test_llm.py`/`test_app.py` + `BACKLOG.md`/`HISTORY.md`. No Cypher, no scripts,
  `_drive_loop` untouched. K-027 correctly left **open** (items 2–5 + addendum (a)'s engine half).
  `seed_workflows.sh acme` re-run after the suite (defs `(created)` in `reference`, snapshots
  `already present` — the documented split-brain shape, restored).
- **U2 — K-031 plan v2** ✅ **APPROVED** (`analyst` re-gate: *approve with suggestions*, all 15
  round-1 findings closed, nothing re-opened, "a `coder` can execute this as written" = yes).

**In flight when paused (both dispatched, results unknown — check for their deliverables first on resume):**
1. `analyst` — **U1-G**, review gate on the K-027 diff → expected at `docs/reviews/k027-parse-robustness.md`.
2. `coder` — **filing K-034** (additive-`MERGE` defect) → expected as a new `### K-034` section in
   `docs/BACKLOG.md`. Doc-only, `BACKLOG.md` only.

**Next actions on resume, in order:**
1. Check whether the two in-flight deliverables landed; if either is missing or partial, re-brief that
   owner once with the gap made explicit.
2. If U1-G's verdict is *needs changes* → back to `tdd-engineer` with the review path, then re-gate.
   If approve/approve-with-suggestions → U1 is done.
3. **Dispatch U3** (`coder`) on **plan v2** (`docs/plans/workflow-def-structure-read.md`), briefed
   **by path**, carrying: the three binding OQ decisions above; the re-gate's **RG-m1…RG-m5 + 5 nits**
   (`docs/reviews/workflow-def-structure-read.md`, re-gate section) as known findings to address;
   the `entry → entry + N` suite gate measured at U3 start (`pytest --collect-only -q`, currently
   **552/553**); `test_queries.sh` pinned at **256/256** as the no-new-Cypher tripwire; **R-1** (report
   divergence, never repair) and **R-9** (K-034 filing state). U3 must also file **K-033** (self-standing,
   bundling premise marked unverified).
4. Then **U3-G** (`analyst`) on the delivered K-031 diff.
5. Commit is the user's call — nothing has been committed this session.

## ▶ RESUMED 2026-07-24 (user: "finish this")

**Both paused in-flight tasks produced nothing** — verified on resume: `docs/reviews/k027-parse-robustness.md`
does not exist and no `### K-034` section is in `docs/BACKLOG.md`. The pause killed both mid-run, so
they were re-dispatched fresh (a *deficient*-result re-brief, not a blocker — the first and only retry
for each).

**Change of assignment for the K-034 filing:** it is **folded into the U3 `coder` brief** rather than
run as a separate agent. Reason: the standalone filing and U3's own §8 doc work would both write
`docs/BACKLOG.md` concurrently. Folding gives a **single owner** for that file for the whole run,
removing the collision, and saves an agent run (the user is credit-constrained). U3 therefore files
**both** K-033 (per plan §5) and **K-034** (the additive-`MERGE` defect), and resolves plan instruction
**R-9** by citing the real item instead of "filing in flight".

**Dispatched:**
- `analyst` — **U1-G**, review gate on the K-027 slice-A diff → `docs/reviews/k027-parse-robustness.md`.
- `coder` — **U3**, K-031 implementation on **plan v2**, carrying the three binding OQ decisions, the
  re-gate's **RG-m1…RG-m5 + 5 nits** (RG-m6 already fixed by the coordinator), the `entry → entry + N`
  self-measured suite gate, the **256/256** `test_queries.sh` tripwire, **R-1** (report divergence,
  never repair) and the V-1 stop-and-escalate rule — plus the two filings.

**Remaining after these:** U3-G (`analyst` gate on the K-031 diff), then the run is complete and the
commit decision goes to the user.

### ⚠ Race: K-034 was filed TWICE-ASSIGNED — coordinator error, resolve at integration

The original standalone `coder` filing K-034 **was not dead after all** — it completed *after* my
resume check found no `### K-034` in `docs/BACKLOG.md`, and filed a high-quality item at
**`docs/BACKLOG.md:644`** (verified present, exactly **1** occurrence at the time of writing).
By then U3 had already been dispatched carrying "file K-034" in its brief, and **there is no way to
message a running subagent from here**, so the instruction cannot be withdrawn mid-run.

**Consequence to check at integration:** U3 may file a **duplicate** K-034 section. Verify with
`grep -c '^### K-034' docs/BACKLOG.md` — expected **1**. If 2, keep the standalone filing (it is the
more thorough one — see below) and delete U3's, or re-brief U3's owner to merge them. Do **not**
assume U3 noticed; its brief incorrectly told it "you own `docs/BACKLOG.md` for this run — no other
agent is editing it."

**Lesson (coordinator):** an in-flight subagent that produced no artifact is **not** necessarily dead;
a filesystem check proves only what has been *written so far*, not what is still coming. Re-dispatching
on that evidence risks double-execution of any unit whose work is not idempotent.

**What the standalone filing added beyond the brief** (all verified by it against the tree):
- `materialize_snapshot` is `repository.py:1470-1496`; the `_PUBLISH_CYPHER.format(...)` call is at
  **`:1485`**, not `:1483` (matches the re-gate's own RG-n4 citation-drift finding).
- **The two-`START` breakage is wider than the analyst stated:** `start_run_untriggered`
  (`repository.py:1145-1156`) is a *second, self-contained copy* of the same
  `MATCH …-[:START]->… CREATE (r:WorkflowRun …)` shape ⇒ **both** start paths break, including the
  REST-started `kind:'process'` flow (`access-request@v1`), which never goes through `start_run`.
- **Four further falsified doc sites** beyond the analyst's ten (`docs/QUERIES.md:744`, `:799`, `:900`;
  `repository.py:925`, `:933`).
- **One nearby statement is CORRECT and must not be swept up by a grep-driven doc pass:**
  `docs/QUERIES.md:826-827` — "the `TRANSITION` MERGE-key is `(from, on, order, to)` so distinct
  outcomes/orders between the same two steps are distinct edges" — is true, and is the very mechanism.
  An explicit do-not-correct note is recorded in the item.

### U1-G verdict: `needs changes` — K-027 slice A looped back to `tdd-engineer`

`docs/reviews/k027-parse-robustness.md` — **1 blocker · 2 major · 6 minor · 3 nit**. The gate earned
its keep: the change shipped a **false-advance** path, which is the dangerous direction.

- **B-1 (blocker)** — the tolerant judge parse reads a *quoted/hypothetical* JSON verdict out of prose
  and **advances** where the pre-change behaviour correctly suspended. `guards._coerce_verdict` biases
  to suspend precisely so an unreliable judge never falsely advances; the fix can defeat that.
- **M-1** — bare-call recovery fires on an *illustrative* call inside a multi-line message, dispatching
  a real thread write **and** discarding the model's real answer (`text=None`). Root cause: the
  docstring claims "the expression owns its lines" but the code enforces "*some* line looks like a
  call", with leading whitespace allowed. **All 6 negative pins were single-line — there was no
  multi-line negative pin at all**, which is exactly how this got through a TDD unit.
- Reviewer **verified a surgical fix**: require the last accepted call to be the final non-whitespace
  content of the fence-stripped message ⇒ all 8 positive pins survive, all 3 false positives close
  (~4 lines).
- Reviewer ran the full suite (**552 passed, 1 deselected** — reproduces the claim), re-seeded
  `reference`, and re-verified the `_drive_loop` SHA lock plus that `executor.py`/`guards.py`/
  `scripts/`/`QUERIES.md` are byte-untouched.

Fix pass dispatched to `tdd-engineer` with the review by path, a mandate for **multi-line** negative
pins, and a hard constraint that `docs/BACKLOG.md`/`docs/HISTORY.md` are **contended** by the running
`coder` (narrow exact-string edits to its own K-027 bullets only, re-read before each edit).

### ⚠ Second instance of the same race — the K-027 review ran TWICE

Same coordinator error as the K-034 filing: the pre-pause `analyst` was **not dead**, and both it and
the re-dispatched one wrote to `docs/reviews/k027-parse-robustness.md`. The **later, more thorough**
review survives at that path (**1 blocker · 2 major · 7 minor · 3 nit**, verified in-file). Both agreed
on the verdict, the blocker and both majors, so the conclusion is corroborated rather than contested —
but it was duplicated work, and the `tdd-engineer` fix pass may have read whichever draft was present
when it started. Findings are consistent; the extra minor is additive.

**Confirmed lesson, now twice:** never re-dispatch on the evidence of a missing artifact alone. There
is no way to query or cancel a running subagent from here, so a "produced nothing" judgement is
unfalsifiable until the task notification arrives. Wait for the notification, or accept double
execution.

### 🔴 ESTATE STATE — `reference` is EMPTY, re-seed required at the end of this run

**Coordinator-verified** (`GRAPH.QUERY reference "MATCH (n) RETURN count(n)"` → **0**) while `ws:acme`
still holds both snapshots. So `@mention`-to-start is **silently no-op'ing right now**. The first
review's claim that it re-seeded is **contradicted by the live graph**.

Deliberately **not** re-seeded yet: `tdd-engineer` (fix pass) will run `pytest` — which wipes
`reference` again at fixture setup — and `coder` (K-031) runs `test_queries.sh`, which deletes
`reference` at teardown. Re-seeding before both finish would just be overwritten.

**Closing action for the coordinator once BOTH agents have reported:**
```bash
cd falkor-chat && ./scripts/seed_workflows.sh acme
docker exec $(docker ps -qf name=falkordb) redis-cli GRAPH.QUERY reference "MATCH (n) RETURN count(n)"
```
Expect both defs `(created)` in `reference` and `already present — no-op` for the `ws:acme` snapshots
— the documented split-brain shape. **This is exactly the hazard K-031 exists to make detectable and
K-034 exists to fix.**
**Resolved 2026-07-24:** `tdd-engineer` re-seeded after its fix pass and `coder` re-seeded after its
final run; coordinator-verified `reference` = **11 nodes** and `verify_workflows.sh acme` → **exit 0,
2 defs in sync**. The recovery was performed *by K-031's own new script* — the deliverable detecting
the very hazard it was built for, on its first live outing.

### U1 fix pass + U3 delivered; both at their gates

- **K-027 fix pass** (`tdd-engineer`): all 13 findings dispositioned — B-1 fixed by **splitting the
  seam** (`extract_json_object` stays permissive for tool calls; new
  `extract_own_line_json_object(…, require_key="decision")` takes the judge), M-1 by the reviewer's
  final-non-whitespace rule, M-2 promoted to **K-035**. Two honest retractions: m-4's "would have
  converted the observed run" **could not be substantiated** (searched every live graph for the real
  `StepRun.output` — zero matches, the run's graph is gone) and was softened; m-3 found one of its own
  backlog bullets **false** and withdrew it. Re-gate dispatched.
- **K-031** (`coder`): exit **596 passed / 1 deselected**, own contribution **+33**;
  **`test_queries.sh` 256/256** — the no-new-Cypher tripwire **held**; ruff clean. **V-1 matched the
  plan exactly** (2 meta rows, `START` count 2, `result_set[0]` arbitrary) in a throwaway
  `ws:k031probe`, deleted after. **R-1 honoured — no repair performed.** RG-m1 solved with a *third*
  option (`response_model_exclude_unset=True`, avoiding `exclude_none`'s swallowing of a legitimately
  null `startKey`); **RG-m3 closed by documentation, not a test** — the multi-row path ships unpinned,
  which the U3 gate is specifically asked to judge. `executor.py` zero-line diff; K-034 verified
  rather than duplicated (no duplicate filing occurred — confirmed, one heading each for K-033/034/035).

### Coordinator follow-ups still owed (doc-only, outside teco's write guard ⇒ must be delegated)

1. **Two falsified "immutable" assertions outside K-034's ten-site table** —
   `docs/requirements/agent-import.md:81` and `docs/requirements/workflow-dependence-overlay.md:21`/`:54`.
   Reported by `coder`, correctly **not** corrected (correcting is K-034's deliverable class). They
   should be **added to K-034's site table** so the item's own done-condition covers them.
2. **`docs/BACKLOG.md:59`** (the M3 milestone row) still lists K-031 among "follow-ups" with no
   delivered marker.
3. **K-034's file:line citations have drifted** by K-031's own additive changes (`materialize_snapshot`
   now `repository.py:1540-1566`, format call `:1555`; `start_run` `:1158+`). The item already says
   "grep the quoted text", so this is a nicety, not a defect.

**All three delegated 2026-07-24** to `coder` in the K-031 closing pass (bundled with the gate's
M-1/m-1/m-2 to avoid a separate run), with the instruction to add the falsified sites to **K-034's
table only** — never to correct the sites themselves (plan §1.2 forbids it, and it would pre-commit
wording before K-034 decides what publish semantics become).

### U3-G verdict: `approve with suggestions` ⇒ **K-031 ACCEPTED** (0 blocker · 1 major · 3 minor · 4 nit)

`docs/reviews/k031-structure-read-impl.md`. Scope discipline **verified**: `executor.py` literal zero
diff; `repository.py` shows three added methods and **no hunks** on `_PUBLISH_CYPHER`/`_READ_*`/
`_read_subgraph`/`read_def_subgraph`/`get_snapshot`/`materialize_snapshot`; every new read bottoms out
in `graph.ro_query` — `GRAPH.RO_QUERY` is a **server-enforced** write barrier, so read-only here is
mechanical, not a convention. "No new Cypher" **holds**: `QUERIES.md`'s diff is five prose blocks plus
text appended to two `//` **comment** lines; every executable statement untouched ⇒ **256/256 invariant
by construction**. R-1 held (reviewer re-ran `verify_workflows.sh acme` → exit 0; nothing repaired).

- **M-1 (major)** — a **third option** both earlier rounds missed: `_read_structure` is a
  `@staticmethod` over an injected `graph`, so a ~15-line **fake-graph** unit test pins V-1's multi-row
  shape with no publish/Cypher/FalkorDB and **no K-034 coupling**. Why it matters:
  `repository.py:1036-1044`'s union loop is the **only novel logic in U1** and its multi-row branch is
  executed by **none** of the 33 tests — a refactor back to `result_set[0]` would leave the suite green
  while silently un-wiring `startKeys` **and** `verify_workflows.sh`'s finding-3 tripwire, the only
  thing in the repo that would ever notice a two-`START` root. (The gate's own (a)/(b) framing was
  incomplete — worth remembering: a reviewer's option list is not exhaustive.)
- **m-1** — three docs claim `schemas.py` carries the `maxSteps` note; **it was never written**. Root
  cause: plan §5 says "six sites", §8 enumerates five, and the coder reconciled the plan's own miscount
  by naming a site that does not exist. A plan miscount became a false doc claim in three places.
- **m-2** — R-3's exact-key-set anti-drift assertion misses `/diff`; `response_model` filters
  undeclared fields, so a dropped `WorkflowDiffOut` field passes all ten API tests silently.
- Reviewer confirmed the coder was **right to report rather than correct** the requirements-doc
  "immutable" sites — and that `agent-import.md`'s is optimistic in the **dangerous** direction (a
  changed re-import does not fail to update, it *additively mutates*) and load-bearing for that doc's
  FR-2 collision handling.

### U1-G re-gate: `needs changes` (0 blocker · 1 major) — **B-1 CLOSED**, fix pass 2 dispatched

`docs/reviews/k027-parse-robustness.md` → "Re-gate … (round 2)". Blocker verified closed end-to-end
(the reproduction input suspends; `'[{"decision": true, …}]'` suspends; two own-line objects suspend in
both orders). 12 of 13 dispositions land.
- **N-1 (major)** — the final-non-whitespace rule anchors only the **last** call; nothing constrains
  what sits *between* accepted calls, and `executor.py:595` dispatches all of them ⇒ an echoed snippet
  plus a real call = **two thread writes**, re-opening all six round-1 shapes whenever the model ends
  with a genuine call. **No negative pin ends with a genuine call** — the round-1 blind spot (all pins
  single-line) recurring in a new dimension. Three docs assert the guarantee as closed, so the doc half
  is mandatory. Reviewer prototyped the ~3-line fix (reject when non-whitespace sits between accepted
  calls; all 8 positive pins unchanged).
- **N-3** — original finding **m-7 was never dispositioned at all**; it silently vanished from the fix
  pass. Worth noting as a process failure mode: a disposition *table* is not evidence every finding was
  seen.
- Trend is converging (blocker → major → expected minor). **Stop rule set:** if round 3 does not close
  cleanly, pause to the user rather than loop again.

### U1-G round 3: **APPROVE — 0 findings ⇒ K-027 slice A ACCEPTED**

`docs/reviews/k027-parse-robustness.md` (round-3 section). The stop rule was not needed — it closed
cleanly. Deliberately narrow re-check (third round, small diff, credit-constrained), and it still went
beyond the claim: the reviewer drove **32 shapes** through `_parse_bare_call_syntax`.

- **N-1 closed, behaviour and docs.** The guard at `llm.py:314-318` is the prototype verbatim, and
  sits *before* `end_of_last_call` advances and *before* the dedup `continue` — so it also catches a
  repeat-with-prose-between, which the prototype was not asked to do. All four N-1 rows return text; a
  **rejected** candidate between two genuine calls also discards the whole recovery (fail-safe via the
  unchanged tail check at `:328`), so the docs' rule is literally true and not merely true of
  *accepted* calls. Every legitimate multi-call form still fires (blank line, CRLF, tab-only line,
  indented second call, three calls, multi-line args, whole-message fence, dup-then-distinct).
- **No pin weakened** — all 9 positive bare-call pins and 4 judge positives read by name *and body*
  (full `arguments` dicts, call-`id` ordering, `text is None`, rationale substrings). The +8 is pure
  insertion.
- **Docs true** — a repo-wide grep for the superseded "final non-whitespace content" wording finds only
  historical narrative, **no live assertion**. N-2…N-6 all close as claimed.

**Why it took three rounds — the durable lesson** (filed by `tdd-engineer`): *a positional accept-rule
anchored on one element of a collection is not a rule about the collection when the consumer iterates,
and a pin corpus uniform in that dimension can never reveal it.* Round 1's pins were all single-line;
round 2's all ended with the call. The blind spot was in the **test corpus's shape**, not the code —
which is exactly what an independent gate catches and a self-review cannot.

## ✅ RUN COMPLETE — coordinator's integration verification (2026-07-24)

**Run by the coordinator, not reported by a delegate:**

| Gate | Result |
|---|---|
| `server` pytest (full) | **606 passed, 1 deselected, 0 skipped** (entry baseline 533) |
| `scripts/test_queries.sh` | **256/256** — the no-new-Cypher tripwire held end-to-end |
| `ruff check .` | `All checks passed!` |
| `seed_workflows.sh acme` → `verify_workflows.sh acme` | **OK — 2 defs in sync**, `EXIT=0`; `reference` restored after the suites |

**Both units accepted, each independently reviewed:**
- **K-027 slice A** — `approve` at round 3 (blocker → major → 0 findings): 3 review rounds, 2 fix passes.
- **K-031** — `approve with suggestions`, then a closing pass fixing the gate's major + 2 minors.

**Filed this run:** **K-033** (`maxSteps` exact cap), **K-034** (additive re-publish — 13 falsified doc
sites), **K-035** (argument-key shadowing). All 🔵 proposed; none started.

**Last action:** one stale sentence inside K-033 (`docs/BACKLOG.md:748-749`, "K-027 is itself 🔵 proposed
and unscheduled" — false since K-027 flipped to 🟡) delegated to `coder` as a one-line correction.

**Not committed.** The commit decision is the user's.
