# The one salesperson UI — Coordination (continued)

> **Status:** active · **Owner:** `teco` · **Tracks:** — (M<n> TBD) · **Extends:** `docs/plans/salesperson-ui-coordination.md`

## Why this document exists

`docs/plans/salesperson-ui-coordination.md` carries the full history of this feature from the
first architecture-plan draft through the close of **S9** (all five sub-units, S9a-e, accepted and
committed) — over a dozen plan-gate passes, a dozen-plus implementation units, several with 2-4
review-gate rounds of their own, and ~110 individually-titled retrospective sections recording
specific incidents, corrections, and lessons along the way. That document **stays intact and
authoritative for everything through S9** — nothing here restates it; every fact from that phase is
cited by path, not re-narrated.

This successor exists because that predecessor crossed the point where continuing to grow it
imposed a real, rising cost on every future resume (the "reconcile the ledger before acting" step
means reading the whole thing), while a clean phase boundary exists exactly at S9's close: the
backend turn-mechanics work is done, and what remains — presenter-route relocation, the demo
bring-up script, the entire frontend, QA, and docs closeout — is a distinct, forward-looking phase
with no open units carried over. Per this repo's own doc-collision convention (`AGENTS.md`, rule 5:
once an earlier document has been "approved, gated, or executed against," a successor may be
written with an ordinal on the slug), this is licensed housekeeping, not a scope change. The
predecessor's `Status:` stays `active` — it isn't being retired, only extended — and gains an
`Extended by:` pointer to this file.

## Goal (unchanged — see the predecessor for the full statement)

Deliver the business-facing salesperson UI specified in `docs/requirements/salesperson-ui.md`
(FR-1…FR-11, AC-1…AC-11), replacing the retired standalone `salesperson/` Streamlit app.

**Definition of done:** AC-1…AC-11 verified; the old `salesperson/` app retired; documentation
(root `AGENTS.md`, component READMEs, `HISTORY.md`, a `tico` user manual) reflects the delivered
surface.

## State inherited from the predecessor (read the cited sections, don't take this list as a
## substitute for them)

- **S9 fully closed** — all five sub-units (S9a-e) accepted and committed; S9e (the last) landed at
  `dcec3f2`, review approved at Pass 28 (`docs/reviews/salesperson-ui-impl.md`). See the
  predecessor's `## Ledger` for the full S0-S9e history and its `RESUME HERE` section for the
  closing summary.
- **Both stakeholder decision points from 2026-09-09 are closed** — see the predecessor's
  `RESUME HERE` section, paragraph 2, for the disposition (nothing further routes through either).
- **The plan (`docs/plans/salesperson-ui.md`) is at v1.33, plan-gate lane closed** — no further
  Pass N re-gates on the plan document itself are expected; implementation units are gated
  individually against it.
- **`cpg_falkorchat` freshness**: check before relying on it for any S10/S11 unit — it was
  intentionally left stale through the whole S9 chain (per the predecessor's dispatch notes) to
  avoid tearing against live S9 units. That reason no longer applies now that S9 is closed; a
  rebuild is worth reconsidering before S10 dispatches if a unit leans on structural analysis.
- **Shared-tree hazard, worth carrying forward as a standing caution, not a premise**: a concurrent,
  unrelated coordination (`document-ingestion2`) was found editing the same `falkor-chat/server/`
  files S9e also touched, causing a real, recurring commit-hygiene collision (see the predecessor's
  S9e ledger row and Pass 27/28). No reason to expect this specific collision to recur on S10/S11's
  files, but the general pattern — check `git log`/`git status` for unexpected drift on shared files
  before trusting the working tree — is now a demonstrated risk in this repo, not a hypothetical.
- **Open, unanswered stakeholder question, carried forward**: whether to keep driving straight
  through S9f/S10/S11 and then stop to explicitly scope the frontend push (S12a-d/S13/S14) as its
  own decision point, or some other pacing. Not yet answered as of this document's opening.

## RESUME HERE — state as of 2026-09-11, after S10/S11's close

**Read this section first.** Reconcile it against `git log` and `git status` before acting — if
they disagree, they win. Nothing is in flight as of this writing.

**Closed this session:** S9f (was already closed pre-dating this doc's own opening — see
"Reconciliation" below), **S10** (`ec829d9`, two-pass `analyst` gate, approve) and **S11**
(`25a3219`, one-pass `analyst` gate, approve with suggestions) — both teco-verified independently,
both with `HISTORY.md` entries (`7ffbfc2`, `0dc8267`). The backend tail is done.

**Decided this session, not yet acted on:** the frontend-pacing question (see its own section
below) — **S12a dispatches alone first, gated hard, then S12b/S12c/S12d/S13/S14 fan out** — and the
visual bar for the whole frontend phase — **polished, not merely functional**, carried into every
frontend brief from here on, S12a's included even though S12a itself is mostly plumbing.

**Next units, in the order the plan implies:**
- **S12a** — shared entry files (`main.tsx`, `App.tsx`, `index.css`), routing, session/API client;
  the blocking foundation S12b/S12c/S12d/S13/S14 all wait on. Routes to `frontend-engineer`. Gate
  hard against §5.3's C1-C14 per-rule test requirement before touching anything else in the UI —
  **do not fan out the other five until this gate is green**, even once dispatched.
- **S12b, S12c, S12d, S13, S14** — mobile shell, i18n, presenter view, chat view, cart/order/
  profile/catalog — fan out once S12a is accepted. `frontend-engineer` for all five; check for
  cross-file/cross-claim collisions before parallelizing (S12b/S12c/S12d each state "edits no
  shared entry file" in their own plan rows — verify that holds before trusting it).
- **S15** — test suites & AC evidence (load harness, live-LLM run, mobile Playwright pass, versioned
  test plan/report) → `qa-engineer`.
- **S16** — docs close-out across root `AGENTS.md`, root `docs/HISTORY.md`,
  `falkor-chat/README.md`/`AGENTS.md`/`docs/SERVER.md`.

## S10 delivered — teco's independent verification, 2026-09-11

Full re-run of the suite myself (`falkor-chat/server`, `.venv/bin/python -m pytest -q`):
**2737 passed, 14 deselected, 0 failed** — matches `coder`'s reported figures exactly, not taken on
report. Diffed every file `coder` touched: `storefront.py`/`storefront_api.py`/`config.py`/
`docs/SERVER.md`/`test_storefront_api.py`/`test_storefront.py`. `_STEP_10_INTERIM` correctly
replaced by a tombstone comment (not left as dead scaffolding). The two files outside the brief's
declared scope (`test_storefront.py`, `docs/SERVER.md`) were forced by a real cross-check test
(`test_config_reads_exactly_the_documented_storefront_env_vars`) — legitimate, not overreach.
`services.py`/`repository.py` growth I'd flagged earlier as a hazard turned out to be
**document-ingestion2's own Stage B/C commits landing mid-run** (`aa1c9be`, `f61d193`) — confirmed
via `git log`, unrelated to S10, already gone from `git status` by the time I checked. The two new
`ServiceError` subclasses `coder` had to declare unreachable in `storefront_api.py`
(`DocumentNotFoundError`, `DocumentUpdateNotFoundError`) are real and necessary: document-ingestion2
explicitly excluded `storefront_api.py` from its own commits ("currently carries a different,
concurrent, uncommitted salesperson-ui S10 unit" — its own commit message), leaving the shared
family-completeness assertion for `coder` to keep green. Spot-checked 4 of the reported mutation
tests by name — all exist (`test_reset_everyone_stops_intake_before_it_drains`,
`test_a_configured_presenter_key_is_compared_once_in_constant_time`,
`test_presenter_login_never_locks_out_after_repeated_failures`,
`test_every_row_of_the_table_was_produced_by_execution`).

**Side effect I own, repaired**: my own verification suite run wiped the global `reference` graph
(documented, expected behaviour of a default offline `pytest` run — `falkor-chat/AGENTS.md`'s own
warning). Restored via `seed_workflows.sh acme`, `seed_catalog.sh`, `seed_salesperson.sh demo`, all
three re-verified `OK` — this undid nothing S11 or S10 delivered, just repaired shared state my own
check disturbed.

Dispatched `analyst` for S10's review gate.

## Reconciliation, 2026-09-11 (resuming this coordination)

- **S9f confirmed already closed**, contrary to how it reads in this doc's own "Next units" list
  above: D-1 landed as **U62** (`404c409`, "S9f — the QUIESCE_S row rewritten against measurement")
  and its third site as **U67** (`3c23992`, "S9f's other two sites — config.py rewritten, one
  already true"), both in the predecessor's ledger, both confirmed still present in
  `falkor-chat/docs/SERVER.md` §1.3 and `falkorchat/config.py`. No dispatch needed for S9f; nothing
  rides on it except the general expectation that S15's QA pass will exercise the quiesce surface it
  documents, same as any other shipped behaviour.
- **`cpg_falkorchat` freshness, measured**: `hand-backfilled`, `sourceCommit b795f4c`,
  `sourceTree 85ddeed…` vs current `HEAD:./falkor-chat/server` = `967c648…` — **different**, and
  `git log b795f4c..HEAD -- falkor-chat/server` = **15 commits**, including the entire S9a-e chain.
  Stale. **Chose not to rebuild for S10**: the plan's own row carries a source-level inventory
  (`_STEP_10_INTERIM` in `storefront_api.py`) as the explicit authority for exactly what moves,
  so S10 doesn't lean on CPG-derived call-graph analysis. Said so in S10's brief rather than silently
  proceeding.
- **New shared-tree hazard, distinct from the S9e-era one**: working tree has uncommitted changes to
  `falkor-chat/server/tests/test_storefront_api.py` (bumping a `ServiceError` family-count assertion
  10→11, `docstring` prose updated to match) and `falkor-chat/scripts/bootstrap_schema.sh` (adding
  `SUPERSEDES` relationship index/constraint, commented as "document-ingestion2 Stage B") — neither
  is this coordination's. Both belong to the concurrently-running `document-ingestion2` work
  (`falkor-chat/docs/plans/document-ingestion2-coordination.md`, new, untracked). `test_storefront_api.py`
  is a file S10 also needs to touch (new tests for the moved presenter routes) — flagged in S10's
  brief: build on top, don't revert, don't assume the current family count. Also uncommitted and
  confirmed **not ours**, per the predecessor's own note: `claude/**` and `model-bench/**` (a `cobb`
  and a separate session's work, respectively).

## Frontend-pacing question — both independent views in, put to the stakeholder (2026-09-11)

`architect` (`aa9c2ab31eda5c8a9`, 65k tok / 11 tools) gave an independent view before seeing mine:
row word-count understates S12a-d/S13/S14's effort — §5.3 (C1-C14, ~9,255 words) and §5.2
(~3,160 words) are the real content S12a must implement, comparable in density to anything in
S0-S9; no unresolved OQ blocks any of the six rows; the one true gap is **no visual/brand spec
at all** (structure and behaviour are exhaustive, presentation is silent), worth one stakeholder
sentence before S12b/S14 land. Recommendation: don't hold a big re-scoping session — the six rows
aren't under-specified — but treat **S12a as its own hard checkpoint** (dispatch alone, gate against
§5.3's per-rule test requirement, *then* fan out S12b/S12c/S12d/S13/S14), the same way S6 was
treated as blocking-foundation risk in the backend chain. Net effort: comparable to the whole S0-S9
chain, not a tail-end mop-up.

`teco`'s own (pre-read) lean: pause for an explicit checkpoint given S12a's blast radius (blocks
five units) — converges with architect's "S12a alone, gate hard" framing rather than either of the
two extremes (blind drive-through, or a full re-scoping session) originally posed.

**Stakeholder decision, 2026-09-11 (both questions answered directly, not relayed through either
independent view):**
- **Pacing: S12a alone first, gate hard, then fan out S12b/S12c/S12d/S13/S14.** Confirmed as
  recommended by both `teco` and `architect` independently. S12a is not to be batched with the
  other five at dispatch time even once its own gate is green — dispatch it alone, wait for the
  `analyst` gate against §5.3's C1-C14 per-rule test requirement, only then fan out the rest.
- **Visual bar: polished, not merely functional** — this ships as a live, audience-facing demo.
  `frontend-engineer` briefs for S12b and S14 (the two rows architect named as most affected —
  mobile shell and the cart/order/profile/catalog panels) must state this explicitly as part of
  their done-condition, not leave visual quality as an afterthought to default component-library
  styling. No design system or branding pass beyond that is implied — the plan's own framing
  (catalog re-theming out of scope) stands.

Both decisions are closed; nothing further routes through either question. S12a's brief, when
dispatched, carries the visual-bar decision forward even though S12a itself is mostly plumbing
(shared entry files, routing, session/API client) rather than visual surface, so that S12b/S14's
later briefs can cite it by pointing at S12a's own brief/deliverable rather than re-relaying it.

## Ledger

| Unit | Owner | Agent id | Status | Deliverable | Gate → verdict | Cost |
|---|---|---|---|---|---|---|
| S10 | `coder` | `a4f0f6cca1968114d` | **accepted — committed `ec829d9`** (HISTORY `7ffbfc2`) | `falkor-chat/server/falkorchat/{storefront.py,storefront_api.py,config.py}`, `.../tests/{test_storefront_api.py,test_storefront.py}`, `falkor-chat/docs/SERVER.md`, `falkor-chat/docs/reviews/salesperson-ui-s10.md` | `analyst` (`aeadb1a55a968aa63`) Pass 1 **approve w/ suggestions** → both fixed → Pass 2 **approve** (1 non-blocking nit, recorded in HISTORY) | coder 409k+9k tok/34+? tools; analyst 150k+170k tok/70+14 tools |
| S11 | `devops` | `ac275dc0e189af01a` | **accepted — committed `25a3219`** (HISTORY entry `0dc8267`, teco caught the doc gap after the fact — S11's own brief should have named it, noted for future briefs) | `falkor-chat/scripts/start_demo.sh`, `falkor-chat/AGENTS.md` | `analyst` (`abb81b4a4d5507727`) → **approve with suggestions** (2 follow-ups, both non-blocking, recorded in HISTORY, not re-dispatched) | devops 170k tok/69 tools; analyst 106k tok/28 tools |
