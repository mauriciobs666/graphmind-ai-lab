# Review — Workflow def/snapshot re-publish semantics plan (K-034)

> **Status:** active · **Owner:** `analyst` · **Tracks:** K-034

## Scope & verdict

Pre-implementation review of `docs/plans/workflow-republish-semantics.md` (owner `architect`,
`Status: active`) against K-034 (`docs/BACKLOG.md`), the evidence it cites
(`docs/reviews/workflow-def-structure-read.md` finding B-2 and its closure;
`docs/plans/workflow-def-structure-read.md` §0.2/§1.2), `falkor-chat/AGENTS.md`, `docs/DESIGN.md`
§1, and the live tree (`server/falkorchat/services.py`, `repository.py`, `executor.py`, `app.py`,
`api.py`, and the relevant test files). This is a design/plan review — no diff exists yet.

Verification method: every load-bearing factual claim in the plan was checked against the actual
file it cites (not just skimmed) — line-by-line comparison of `_PUBLISH_CYPHER`, the read/diff
helpers, the exception-handler registration pattern, the K-031-pinned test, and every doc site in
the §8 correction table I spot-checked. I also independently searched the test suite for any
existing caller that resubmits a topology-differing payload through the *service* layer (not the
raw `Repository`), which the plan's own analysis does not exhaustively enumerate — see Finding 1.

**Verdict: approve with suggestions.** The plan is unusually well-grounded — every claim I checked
against the live code and test suite held up exactly as stated, including several claims that are
easy to get subtly wrong (the exact test that forces topology-only scoping, the `source=` label
collision check, the SHA-lock boundary, the DESIGN §1.2 register's actual contents). No blocker.
Two minor completeness gaps and a couple of nits below; none require a loop back to `architect`,
but the minor findings should be picked up either now (cheap) or explicitly acknowledged by
`tdd-engineer` during implementation.

---

## Findings

### Minor 1 — An existing test that *does* resubmit a topology-differing payload through the service layer isn't named in the plan, even though it's exactly the class of caller §4.1 asks to check for

`server/tests/test_api.py::test_diff_reports_divergence_after_the_documented_reseed_trap`
(line 681) publishes `DEF_BODY`, materializes it, wipes `reference` (`_wipe_reference(conn)`),
then re-publishes an **edited** body — new step (`escalate`), a new transition guard, a changed
`config` — against the *same* `(key="onboarding", version="1")`, and asserts `wf_client.post(...).
status_code == 201` (line 705). This is a genuine structural re-publish that the naive reading of
K-034's remedy would reject.

I traced it and confirmed it **stays green** under this plan's gate: `_check_no_structural_conflict`
short-circuits when `existing_raw is None` (§3.2), and because the reference graph was wiped
immediately before the edited publish, `self._repo.read_def_structure(key, version)` returns
`None` — the gate never fires, and the write proceeds as an ordinary first-time create, exactly as
the test's own comment says ("this fixture depends on create semantics only, never on K-034's
additive semantics"). The plan's §2.2 narrative describes this exact mechanism (reference wipe →
next publish is a fresh create) but never connects it to this specific test by name or file:line,
and the doc-verification brief this review was given explicitly asked whether "the plan accounts
for any existing caller/test that currently expects `201` on a structural change" — this is that
test, and the plan doesn't show its work on it.

This isn't a blocker (I verified the test stays green), but it's the one place in an otherwise
very thorough plan where the implementer would have to re-derive an analysis the plan should have
handed them pre-solved — and if the reasoning had been wrong, this is exactly the kind of test that
would silently regress from `201` to `409` without any of §6's *new* tests catching it (none of the
new tests touch the reseed-wipe scenario).

**Suggested improvement:** add one sentence to §3.3 or §6.3 naming
`test_diff_reports_divergence_after_the_documented_reseed_trap` explicitly as a verified-safe case
(the wipe makes `existing_raw is None`, so the gate doesn't fire), and have `tdd-engineer` run it
first during step 6 as a canary before writing the new 409 tests.

### Minor 2 — §7 risk 2's residual-race framing only names `publish`, but the identical TOCTOU applies to `materialize_def`

§7 risk 2 ("Residual concurrent-first-publish race") describes two callers racing a **brand-new**
`(key, version)` with different content, both observing `existing_raw is None`, both proceeding to
write — correctly scoped as pre-existing and out of this item's remedy. I traced the same shape
through `materialize_def`'s wiring (§3.3): two concurrent first-time materializes of the same
never-before-materialized `(key, version)` into the same workspace, racing on
`read_snapshot_structure`, hit the identical gap. This matters more on the materialize side than
the plan's wording suggests, because §2.2 itself establishes that "this is where the live defect
actually bites, on the materialize side" — the exact scenario K-034 traces (reference wiped,
`ws:<id>` snapshot survives, a subsequent materialize races) is a materialize-side race, not a
publish-side one.

This doesn't change the verdict (the risk is correctly out-of-scope either way, and the plan's own
reasoning generalizes cleanly), but as written a reader skimming §7 could conclude the residual race
is publish-only and miss that the same caveat governs the materialize call the item's own root-cause
narrative centers on.

**Suggested improvement:** reword risk 2 to say "the same brand-new `(key, version)` — whether via
`publish_workflow_def` against `reference` or `materialize_def` against a workspace snapshot."

---

## Nits

- **§2.4 row count.** "§1.2 ... lists 16 rows" — I counted 17 rows in the live
  `docs/DESIGN.md` §1.2 table (Single store … Identity source of truth). Doesn't affect the
  substantive, correctly-verified claim (no row states "WorkflowDef versions are immutable"), but
  worth a one-character fix if the plan is touched again before implementation.
- **`_structural_diffs`'s coupling to `_diff_structures`' path-suffix grammar** (§3.2) is
  string-suffix matching (`.type`, `.config`, `.guard`) rather than a diff-entry field the producer
  tags directly (e.g. a `kind: "structural"|"property"` key on each diff dict). It's workable and
  its failure mode is safe-by-default (an unrecognized suffix falls through to "structural," i.e.
  over-rejects rather than under-rejects), and §9 step 2 already calls for a parametrized unit test
  mirroring `test_diff_structures_one_class_at_a_time` — which is the right guardrail. No action
  needed beyond what's already planned; flagging only so whoever extends `_diff_structures` later
  (a new step/transition property) knows the filter needs a matching update.

---

## What's solid

- **Grounding is exceptional.** I independently re-verified essentially every load-bearing claim
  against the live tree rather than trusting the plan's citations, and every one held:
  `_PUBLISH_CYPHER`'s three MERGE patterns and their additive failure modes (§2.1); `start_run`'s
  `MATCH (snap)-[:START]->(start)` double-row hazard (§2.2); the DESIGN §1.2 register genuinely
  containing no immutability row (§2.4); the `WorkflowDefSpecError`/`WorkflowDefNotFoundError`
  exception-handler pattern in `app.py` that the new `WorkflowDefConflictError` mirrors exactly,
  including the 409-for-"nothing written, state conflict" precedent (`WorkflowRunNotWaitingError`);
  the re-export block pattern in `services.py`; `_select_transition`'s current line/sort key and its
  explicit "outside the SHA lock" status (`docs/DESIGN.md:406-409`, confirmed verbatim); and the
  `FakeRepo.def_structures`/`snapshot_structures` fixture machinery already existing from K-031.
- **The topology-only scoping decision is genuinely forced, not a convenience.** I traced
  `test_republish_is_create_only_on_properties_structure_read_unchanged` (`test_api.py:550`) and
  confirmed it edits *only* `name`, `kind`, and the start step's `config` — steps and transitions
  are byte-identical — and asserts `201` with the stored structure unchanged. A full-payload-equality
  gate would reject this test outright. The plan's `_structural_diffs` filter maps 1:1 onto
  `_PUBLISH_CYPHER`'s own MERGE-key boundary (§2.1 ↔ §3.2), which is the right authority to draw the
  line from — not intuition about what "feels structural."
- **The "no `graph-dba` gate needed" claim checks out.** The design's two read calls
  (`read_def_structure`/`read_snapshot_structure`) are pre-existing, already-profiled K-031 queries;
  `_PUBLISH_CYPHER` is untouched by every wiring snippet in §3.3. The TOCTOU question the review
  brief specifically asked about is real but narrow and correctly scoped (see Minor 2) — for an
  *existing* `(key, version)`, a structurally-differing candidate is always rejected regardless of
  read timing, because any divergence from the stored baseline triggers the reject; the residual
  race only exists for two racing *first-time* writes, which is a pre-existing hazard this change
  doesn't worsen.
- **The REST contract table (§4.1) is precise and I could not find a regression it misses** — I
  swept every `POST /workflow-defs` / `.../materialize` call site across the whole test suite
  (`test_api.py`, `test_services.py`, `test_executor.py`) for a same-`(key,version)` resubmission
  through the service/API layer with differing content; the only two are the K-031-pinned
  property-only test (correctly identified and preserved) and the reseed-wipe test (Minor 1).
- **Doc-correction table spot-checks all matched.** I independently grepped and read the live text
  at `docs/QUERIES.md` (§11 preamble, §11.1/§11.4 footnotes), `docs/DESIGN.md` (topology diagram, §4
  decision paragraph, §9 write-paths row), `falkor-chat/AGENTS.md`'s `seed_workflows.sh` row,
  `docs/requirements/agent-import.md:81`, `docs/requirements/workflow-dependence-overlay.md`, and
  `docs/BACKLOG.md`'s K-029/K-032 premises — every "current claim" quote in §8's table matches the
  live file verbatim (mod expected line-number drift, which the plan itself flags and directs
  implementers to re-grep).
- **Sequencing vs. K-030/K-029 is correctly reasoned** — traced `_validate_def_spec`'s existing
  zero-transition rejection and confirmed the plan's gate genuinely sits at the same layer, before
  any repository call, with no query-level overlap with K-030's remaining `_PUBLISH_CYPHER`
  CASE-guard gap.
- **The step-by-step sequence (§9) is concrete and directly executable** — every step names the
  file, the insertion point, and (for the trickier ones) the actual code, which is what makes this
  reviewable as a plan rather than a sketch.

## Open questions

None that block moving to `tdd-engineer`. The two minor findings are cheap enough that `architect`
could fold them into the plan in a two-line edit before handoff, or `tdd-engineer` could simply
absorb them during step 6/§7 without a full loop back — the coordinator's call on which is cheaper
given where this is in the pipeline.
