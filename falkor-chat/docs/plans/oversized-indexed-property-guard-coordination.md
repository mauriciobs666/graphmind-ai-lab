# Oversized indexed-property guard — coordination

> **Status:** archived · **Owner:** `teco` · **Tracks:** K-049 (open follow-up, delivered 2026-08-26)

## Goal

K-049: while building K-028's ctx-merge length-bound test, `coder` found that publishing a def
with a deliberately oversized **indexed/constrained** graph property (`Step.key`) crashed the
shared dev `falkordb-dev` container outright — connection dropped mid-request, `--rm` container
vanished from `docker ps -a`, reproduced twice across independent restarts. Root cause was never
established (no logs survived the `--rm` crash). This is a shared-instance reliability risk: the
same container also hosts `cpg_falkorchat`, `kaizen_team`, and every workspace graph any agent is
using. Full item text: `falkor-chat/docs/BACKLOG.md` K-049.

**Done when:** the crash is root-caused (via a disposable, isolated container — never the shared
`falkordb-dev`), the failure mode is bounded (threshold? any oversized value? specific to
indexed/constrained properties or broader?), and an app-side guard is implemented + tested so no
future write can reproduce it against the shared instance.

## CPG freshness (checked at coordination open)

`cpg_falkorchat` exists but is **stale** relative to `falkor-chat/server`: built
`2026-08-17T00:40:42Z` from a `.git`-less scratch copy (`sourceCommit`/`sourceDirty` absent — the
known scratch-build pattern), and `git log --oneline --since=2026-08-17T00:40:42Z -- falkor-chat/server`
shows **14 commits since**, including the entire M5 ingestion pipeline (K-050 stages 1–6a) and
K-051. Any unit here that would otherwise consult the CPG for structural/impact claims should
treat it as stale and either flag the caveat or ask `graph-dba` for a rebuild before trusting a
broad structural answer from it — this task is unlikely to need it (it's a live-repro + narrow
code-guard task, not a structural sweep), but the caveat travels with the coordination regardless.

## Ledger

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict | Cost |
|---|---|---|---|---|---|---|
| U1 | `graph-dba` | `acbdfd7fd442ce8b4` | accepted | `docs/plans/oversized-indexed-property-guard-graph.md` | `analyst` → approve with suggestions | 151k tok / 53 tools |
| U2 | `tdd-engineer` | `a1d9d8a1a183e38a5` | accepted | `services.py`/`api.py`/`schemas.py` diff + new tests in `test_services.py`/`test_api.py` | `analyst` → approve | 170k tok / 83 tools |

## Close-out (2026-08-26)

**U2 gate result (Pass 3, same review doc): approve, no blockers.** `analyst` independently
verified the actual working-tree diff (not just `tdd-engineer`'s report): the guard fires before
both write call sites with no bypass (confirmed by grepping every caller of the underlying
publish/materialize Cypher), the `transition.on` exclusion still holds, the new tests are sound
(traced through what would happen if the guard were disabled, confirming they'd correctly go red —
the exact class of masking bug `tdd-engineer`'s own kaizen entry flags was genuinely fixed, not
just claimed fixed), the kaizen entry itself was cross-checked in `kaizen_team`, and the full suite
was re-run independently: **1782 passed, 4 deselected** — exact match. `analyst` also restored
`reference`/`ws:acme` after its own suite run wiped `reference` (documented, expected pytest
teardown behavior) via the standard `bootstrap_schema.sh` → `seed_demo.sh` → `seed_workflows.sh`
sequence, confirmed back in sync via `verify_workflows.sh` — `teco` independently re-ran
`verify_workflows.sh acme` before committing and confirmed `OK — 2 defs in sync` directly, rather
than trusting the self-reported recovery alone.

**Committed:** the code diff (`services.py`, `api.py`, `schemas.py`, `test_services.py`,
`test_api.py`) as one commit; this coordination doc, the design note, the three-pass review doc,
`BACKLOG.md`'s K-049 removal, and the new `HISTORY.md` entry as a second, docs-only commit. See
`git log` for the exact commit hashes.

K-049 is fully delivered and integrated. This coordination is closed.

## U2 delivery summary

`_validate_key_lengths` static helper added to `services.py`, wired into `publish_workflow_def`
(before `_validate_def_spec`) and `materialize_def` (after `read_def_subgraph`, before
`materialize_snapshot`); `api.py`'s materialize route gained the matching `Path(max_length=...)`
bound; one-line comment added at `MAX_KEY_LEN`'s definition. New tests: 5-case parametrized publish
test (key/version/step-key/transition-from/transition-to), a negative-space test proving
`transition.on` is deliberately unbounded by this guard, a `materialize_def` test writing an
oversized key directly into `FakeRepo` (simulating corrupted `reference` data), two REST 422 tests.
Mutation-tested in two rounds (first round caught a real test-construction overlap bug — two cases
accidentally passed for the wrong reason via a pre-existing unrelated check — fixed before
re-mutating; second round confirmed clean). Full offline suite: **1782 passed, 4 deselected**
(`-m live` only), observed directly. Kaizen entry confirmed logged
(`kaizen_team`, `tdd-engineer`, dated 2026-08-26, the mutation-test-overlap lesson).

## U1 gate result (Pass 2, `docs/reviews/unique-constraint-oversized-value-crash-rca.md`)

**Verdict: approve with suggestions, no blockers.** `tdd-engineer` proceeds against §5 of the
graph doc as written. Two new minor findings, both independently re-verified by `teco` before
folding into U2's brief (not just trusted):

1. Two `schemas.py` field citations in the graph doc's §4 are off by 1-2 lines:
   `WorkflowStepIn.key` is `schemas.py:106` (not `:108`); `WorkflowTransitionIn.from_/to/on` is
   `schemas.py:115-117` (not `:116-118`). Confirmed by direct read.
2. The graph doc's "zero RELATIONSHIP-type constraints" grep (used to justify not bounding
   `tr["on"]`) is now stale — a `SAME_AS` RELATIONSHIP UNIQUE constraint on `matchId` was added
   2026-08-24 (`scripts/bootstrap_schema.sh:215`), two days before the doc was written. The
   design's conclusion still holds: `matchId` is server-minted (`ingestion.py`'s `self._id()`,
   the same `uuid.uuid4().hex` pattern as every other safe ID), so no guard is needed for it — but
   the doc's own "scope check" enumeration doesn't name it, so its audit isn't as exhaustive as it
   claims. Confirmed by direct read of `ingestion.py:133`/`:200` and `bootstrap_schema.sh:215`.

Neither finding changes the guard design or blocks implementation — both are folded into U2's
brief as corrected facts rather than routed back to `graph-dba` for a doc rewrite (the design doc
itself is left as `graph-dba`'s authored artifact; the corrections travel with the implementation
brief instead).

## Correction — this coordination's premise was stale (found by U1, 2026-08-26)

U1 discovered that root-cause work for K-049 **already existed**, done and `analyst`-approved
*before* this coordination opened: `falkor-chat/docs/reviews/unique-constraint-oversized-value-crash-rca.md`
(committed `daf3ff0`, 2026-08-21, `Status: active`, `analyst` Pass 1 verdict: approve with
suggestions, no blockers — independently re-reproduced the crash in its own container). Neither
`falkor-chat/docs/BACKLOG.md`'s K-049 text nor this coordination doc's opening section reflected
that — both were written from the pre-RCA state. `graph-dba` did **not** re-run a full fresh
investigation; it independently re-confirmed the crash a third time (its own disposable container,
byte-for-byte same fault signature) before relying on the RCA for a design deliverable, then
produced `oversized-indexed-property-guard-graph.md` as a confirmation + finalization pass:
corrects two stale line citations the RCA and `falkor-chat/AGENTS.md` both carry, adds one newly
confirmed gap (`api.py`'s materialize route missing a `Path(max_length=...)` bound — a hygiene nit
`analyst`'s original Pass 1 already flagged and that's still unfixed), and gives `tdd-engineer` a
citation-accurate, finalized guard design (§5 of that document) that supersedes the RCA's §6.

**Root cause: established.** FalkorDB v4.18.11 SIGSEGVs the whole `redis-server` process (not just
the query) when a `CREATE`/`MERGE` commits a value >4096 bytes into a `UNIQUE`-constrained property
— a null-pointer dereference in `EnforceUniqueEntity`, confirmed independently three times (the
RCA, `analyst`'s gate, U1's third run). **What was never done is the harden phase** — `services.py`
never imports `MAX_KEY_LEN`, so the approved guard design was designed but never implemented. That
is what U2 is for.

`BACKLOG.md`'s K-049 entry has been rewritten in place to match this state (see that file).

## Notes

- U1's repro (and its predecessor RCA) ran in disposable, isolated FalkorDB containers, never
  `falkordb-dev` — verified up/`PONG` throughout, every time.
- Upstream FalkorDB issue: recommended (genuine engine defect, not app-design). This is the user's
  call to file, not an agent action.
- The durable engine quirk is already correctly recorded in `claude/graph-dba/falkordb-quirks.md`
  (added alongside the original RCA) — no new entry needed.
- U2 is gated on U1's design note being accepted (`analyst` verdict on U1 must be
  approve/approve-with-suggestions before U2 dispatches) — scoped narrowly to what's actually new
  (the corrected citations, the api.py finding, the finalized §5 design), not a re-derivation of
  the crash mechanism itself, which has already been independently confirmed three times.
- On close: remove K-049 from `falkor-chat/docs/BACKLOG.md` entirely (delivered items aren't kept
  there, not even as an index row) and add a `docs/HISTORY.md` entry.
