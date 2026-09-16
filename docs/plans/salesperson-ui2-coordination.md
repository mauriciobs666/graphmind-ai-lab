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

## Stakeholder decision, 2026-09-16 — defect-fix scope (resolves the pause below)

Put to the stakeholder with two independent views formed first (teco's own read of
`docs/test-reports/salesperson-ui-report.md`, then `architect`'s independent, source-verified
read, `a2ce6c17b00d76b08` — 89.6k tok/12 tools — dispatched before either view was shared with the
other). Both converged almost exactly: fix DEF-1/DEF-2/DEF-4/DEF-5 now (cheap, well-understood, no
design decisions); DEF-3 needs a short design pass first before an implementer touches it
(`architect` calls it arguably the most severe of the six — `_drive_or_fault` is shared by the
REST/sweep callers too, not just the chat path, so the fix isn't purely local); DEF-6 has no code
fix to route yet (root cause unconfirmed, hypothesis is LM Studio's own concurrent-request
handling) — the actionable question is a diagnostic spike, not an implementation unit.

**Stakeholder chose all three recommended tracks:**
- **DEF-1 + DEF-2 + DEF-4/DEF-5** → implementation units now.
- **DEF-3** → architect design pass, then a gated implementation unit.
- **DEF-6** → `data-scientist` diagnostic spike (investigation only, no fix) gating the *first live
  demo*, not S16 — same shape as the existing K-056→AC-10 gate precedent.

Dispatch plan (file-disjoint, claim-disjoint — four parallel first-wave units):
- **U-DEF1** (`tdd-engineer`) — `falkor-chat/server/falkorchat/app.py`, SPA-fallback route.
- **U-DEF2** (`tdd-engineer`) — `salesperson/src/views/CartPanel.tsx` + `endpoints.ts` +
  `CartPanel.test.tsx`, field-name fix + fixture fix.
- **U-DEF3-design** (`architect`) — design note (plan amendment) on the `_drive_or_fault`/
  `_run_turn` failure-isolation contract, gated by `analyst` before implementation.
- **U-DEF6-spike** (`data-scientist`) — diagnostic note at `docs/plans/salesperson-ui-ml.md`,
  investigation only.

Queued, dependent, dispatched once their upstream lands:
- **U-DEF45** (`tdd-engineer`) — batch of DEF-4 + DEF-5, sequenced after U-DEF1 is committed (DEF-4
  can only go green once the SPA-fallback route exists).
- **U-DEF3-fix** (`tdd-engineer`) — sequenced after U-DEF3-design is delivered and gated.

## RESUME HERE — state as of 2026-09-16, all six S15 defects closed for this wave

**Every defect-fix unit the 2026-09-16 stakeholder decision selected is accepted and committed**:
U-DEF1 (`97eca1c`), U-DEF2 (`f0e5719`), U-DEF3-design (`04f8922`, plan v1.40), U-DEF3-fix
(`4cebd96`), U-DEF45 (`dafec93`), U-DEF6-spike (`6ddf88b`). See the ledger for each unit's gate
verdict and teco's independent verification. `falkor-chat/server`'s full suite: 2830 passed, 14
deselected (re-confirmed after the last commit). `salesperson`'s offline suite: `tsc -b` clean,
`vitest run` 243/243. `salesperson/tests/e2e/`: 16/16 live.

**DEF-6 is diagnosed, not mitigated.** The stakeholder's 2026-09-16 decision authorized a
diagnostic spike only — `docs/plans/salesperson-ui-ml.md` confirms the failure reproduces at LM
Studio's own serving layer and recommends against accepting it as residual risk for the first live
demo, with a prioritized mitigation path (D → C → B). **No mitigation unit has been authorized or
dispatched.** This is a distinct, still-open decision gating the *first live demo* specifically —
not S16, not blocking it — per the stakeholder's own framing (mirrors the existing K-056→AC-10
precedent).

**Next step: S16 (docs close-out)**, now unblocked per plan v1.40's own updated dependency ("after
S18 and every other defect-fix unit the 2026-09-16 stakeholder decision selects that is still
open" — all closed). §5.1's S16 row: root `AGENTS.md`, root `docs/HISTORY.md`,
`falkor-chat/README.md`/`AGENTS.md`/`docs/SERVER.md`, `salesperson/{README,AGENTS}.md`. **S16 should
also record**: all six S15 defects (fixed: DEF-1/2/3/4/5; diagnosed-not-yet-mitigated: DEF-6, with
its own open pre-live-demo decision named explicitly, not silently dropped) — this coordination's
own accumulated history since S15 is the source, not a re-derivation.

**Not yet dispatched — paused here for a checkpoint, not a stakeholder question this time.** This
is a natural completion boundary (the defect-fix wave that was blocking everything is fully
closed); S16 is a large, many-file docs-closeout pass worth its own dispatch rather than folding
into this already-long session silently. Resume by dispatching S16 to `coder` per §5.1's row,
unless the stakeholder has a different next step in mind.

## RESUME HERE (superseded state, kept for history) — state as of 2026-09-14, S15 delivered &
## committed (`f4f828a`) — CONDITIONAL PASS, 4 product defects found; **paused for a stakeholder
## decision on defect-fix scope before S16** — **resolved, 2026-09-16** (see the current
## "RESUME HERE" above for the outcome and the "Stakeholder decision, 2026-09-16" section for the
## dispatch record)

**Read this section first.** Reconcile it against `git log` and `git status` before acting — if
they disagree, they win.

**Every implementation unit is accepted and committed** — S12a through S12d (see the ledger for
shas/verdicts; S12d closed `993e8b2` on its second `analyst` gate pass, not restated here). **S15
— test suites & AC evidence — is also now delivered and committed (`f4f828a`)**, `qa-engineer`
(`ad9cd71a1563ddc25`), teco-verified independently (see the S15 ledger row): a versioned test plan
(18 items), a test report driving the real running server + live LLM, and a new load/concurrency
harness (`load_demo.py` + `stub_llm_server.py`).

**S15's verdict: CONDITIONAL PASS.** Most ACs are met; AC-3 is met for reads and not met as
literally worded for agent-turn latency under heavy concurrency (anticipated, measured not
asserted); AC-5 is functionally correct but its primary access path is broken. **Four genuine,
reproducible product-level defects found** — full repro/evidence in
`docs/test-reports/salesperson-ui-report.md`, each independently re-confirmed by teco directly
against source before this doc was updated:
- **DEF-1 (High)** — `/shop/presenter` (any deep SPA route) 404s on direct navigation — no
  server-side SPA-fallback under `falkor-chat/server/falkorchat/app.py`'s `/shop` `StaticFiles`
  mount. Breaks the presenter's actual access path (link/QR/refresh).
- **DEF-2 (High)** — `CartPanel.tsx` renders `$NaN` per line item — client reads
  `item.unitPrice`, server's `get_cart` sends `price` (confirmed reading both sides directly).
- **DEF-3 (High)** — the dead-turn latch (`turn.lastTurn`) never fires when
  `services._drive_or_fault` catches a `ProviderCallError` internally and returns rather than
  re-raising — `storefront.py`'s `_run_turn` only sets the latch from its own `except Exception`,
  which this path never reaches. Proven live: 230 graph-confirmed failed `WorkflowRun`s, zero
  latched.
- **DEF-6 (Medium, attribution not established)** — `en`-configured participants sometimes get a
  fully-formed Spanish reply under concurrency (2/10 trials even at the plan's own literal
  3-way-concurrent wording); never at concurrency=1. Hypothesis is LM Studio's concurrent-request
  handling, not application code — not confirmed this pass.
- Plus two low-severity **test-only** defects (DEF-4: `presenter.spec.ts`'s URL resolves outside
  `baseURL`'s path; DEF-5: `mobile-shell.spec.ts` asserts stale S12b-era placeholder copy S14
  replaced) — both confirmed via direct grep.

**Standing process correction (gate before commit): held cleanly through S12d and applies going
forward to any defect-fix unit** — none of DEF-1/2/3/4/5/6 have been fixed yet; S15's own
guardrail (and this agent's standing rule) is report-only, no in-pass patching.

**Paused here, not proceeding autonomously**, per this session's explicit credit-conservation
request: whether/which of DEF-1/2/3/4/5/6 to route to implementers now (each would be its own
gated unit — most likely `tdd-engineer` for DEF-1/DEF-2/DEF-4/DEF-5, a design decision needed on
`services.py`/`storefront.py`'s failure-isolation contract for DEF-3, and a `data-scientist`/
`graph-dba`-informed follow-up for DEF-6) versus deferring all of them and going straight to S16
with the defects logged as known residual risk, is a stakeholder call teco has put to the user
directly (not decided here). **Do not dispatch any defect-fix unit or S16 until that answer is
in** — resume by reading the user's answer in the live conversation, or, if this doc is being
read cold in a fresh session, treat the absence of any defect-fix ledger row below S15 as proof
no such decision has been acted on yet.

**Next units, in order (blocked until the pause above resolves):**
- **Defect-fix units** — however many the stakeholder decision selects, each its own row below,
  each gated by `analyst` before commit like any other code unit (same standing correction).
- **S16** — docs close-out across root `AGENTS.md`, root `docs/HISTORY.md`,
  `falkor-chat/README.md`/`AGENTS.md`/`docs/SERVER.md`, `salesperson/README.md`/`AGENTS.md`, plus a
  new row/section recording S15's defects (fixed or accepted-as-residual, whichever the pause
  above resolves to). Dispatch only once every selected defect-fix unit is closed.
- **Outstanding housekeeping**: this coordination doc remains uncommitted between edits *except*
  this checkpoint commit (`ce76a9f`, made under the same credit-conservation request that caused
  this pause) — going forward, resume the normal practice of holding further edits uncommitted
  and batch them into the natural closeout commit after S16, by explicit path.

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
| S12a | `frontend-engineer` | `a644babc1407367d1` | **accepted — committed `bbd9eb7`** (HISTORY `489a498`) | `salesperson/src/api/**`, `salesperson/src/session/**`, `salesperson/src/routes.tsx`, `salesperson/src/{main.tsx,App.tsx,index.css}` | `analyst` (`afde52ef31943c5f1`) Pass 1 **needs changes** (2 blockers, 1 major routed separately, 1 minor) → both fixed → Pass 2 **approve**; `falkor-chat/docs/reviews/salesperson-ui-s12a.md` | frontend-engineer 300.5k+179.9k tok/116+68 tools; analyst 174k+224.4k tok/60+29 tools |
| S12a-ownership | `architect` | `a57cac5408e27eeaf` | **accepted — committed `b15d0bb`** | `docs/plans/salesperson-ui.md` (v1.34: §5.0 two new rows, S12b/S12d/S13/S14 rows swept) | teco read directly (no separate re-gate — narrow, review-triggered amendment, judged sound: distinguishes S14's zero-touch fix from S12d/S13's additive-swap fix correctly, matches actual `routes.tsx` contents; three-way baseline/HEAD/worktree check clean before commit) | architect 145k tok/28 tools |
| S12b | `frontend-engineer` | `a939bb47232a842de` | **accepted — committed `70b593b`** (HISTORY `1d14369`) | `salesperson/src/layout/**`, `salesperson/src/components/sheets/**`, `salesperson/tests/e2e/**`, `salesperson/playwright.config.ts`, `salesperson/src/App.tsx`, `salesperson/src/routes.tsx` (v1.36), + 4 seed placeholders in `salesperson/src/views/{Cart,Order,Profile,Catalog}Panel.tsx` | `analyst` (`a7851a1b1a47ab4b4`) Pass 1 **needs changes** (1 blocker, 1 major routed to `app-composition`, 2 minor) → Pass 2 **approve** (blocker + minor-1 closed; major + minor-2 correctly carried forward) → **Pass 3 approve with suggestions** (major + both minors closed via v1.36's rewrite; 1 new minor — stale error copy not cleared when a retry starts — fixed same-commit by `frontend-engineer`, independently re-verified by teco with a mutation outside both the implementer's and reviewer's own tables, not re-gated given its narrow scope and the verdict already non-blocking); `falkor-chat/docs/reviews/salesperson-ui-s12b.md` | frontend-engineer 265.9k+145.9k+235.7k+256k tok/92+41+52+13 tools; analyst 148.9k+176.9k+227k tok/51+14+27 tools |
| app-composition (→ plan v1.36) | `architect` | `ad027152d9702bd27` | **accepted — committed `0f99b63`** (agentId corrected 2026-09-13 — was swapped with S12b's implementation dispatch in an earlier edit; both corrected and confirmed against each agent's own task-notification task-id) | `docs/plans/salesperson-ui.md` (v1.36 §4.11: `LayoutShell` becomes the router's pathless layout route; `App.tsx`'s three providers reorder to wrap `RouterProvider` normally; grants S12b one narrow edit right on each of `App.tsx`/`routes.tsx`; `sessionBridge.ts`/`injectBridge.tsx` become dead code, `ResetControl` simplifies to `useResetMine()`) | teco independently verified the load-bearing technical claim myself (`RouterProviderProps` has no `children` — read directly in `node_modules/react-router/dist/development/index-react-server-client-*.d.ts`; confirmed `Outlet` is exported via the barrel; confirmed `Header.tsx`/`Shell.tsx` call no router hook today) before accepting | architect 165.9k tok/52 tools |
| S12c | `frontend-engineer` | `aec3979e889af54e0` | **accepted — committed `42686fe`** (HISTORY `6ecbb84`) | `salesperson/src/i18n/**`, `salesperson/src/locales/{en,pt-BR,es}.json`, minimal `routes.tsx` join-screen edit (language chooser), `tsconfig.app.json` (`resolveJsonModule`) | `analyst` (`a1759fb8141ba8141`) **approve with suggestions**, no blockers; 1 major (routes.tsx ownership, routed to architect below) + 3 minor/nit (non-blocking, logged); `falkor-chat/docs/reviews/salesperson-ui-s12c.md` | frontend-engineer 164.4k tok/63 tools; analyst 120.8k tok/41 tools |
| S12c-ownership | `architect` | `abc9d6a5167045deb` | **accepted — committed `753e842`** | `docs/plans/salesperson-ui.md` (v1.35), `salesperson/AGENTS.md` (teco's trivial-fix sync of the same routes.tsx row, folded into the same commit) | teco read diff directly (narrow, 4-hunk plan-text-only change, matches review's Major exactly; sweep confirmed no other row assumed a fixed edit count; three-way baseline/HEAD/worktree check clean before commit) | architect 103.6k tok/14 tools |
| S13 | `frontend-engineer` | `a97985d2364e57b4a` | **accepted — committed `acd0413`** (fix-back for the Major landed; teco-verified independently: diff read incl. `routes.tsx`'s 97-line swap confirmed clean, own `vitest run` 178/178, own `tsc -b` clean, own `build.sh` clean, read `TurnIndicator.tsx`/`ChatView.tsx` directly matching the report's claims, own mutation outside the implementer's table — `composerNotice.ts`'s `reread`-default branch gutted, both `composerNotice.test.ts` and `ChatView.test.tsx` stayed green, restored byte-identical, full suite re-confirmed) | `salesperson/src/views/Chat*`, `salesperson/src/components/message/**`, + one narrow additive `routes.tsx` swap (inline `ChatScreen` placeholder → real import) | `analyst` (`a0ef1de11457831a9`) → **approve with suggestions**, no blockers; 1 Major (`composerNotice.ts` `reread`-default branch — untested AND, unlike S14's `isOrderLine`, confirmed genuinely reachable via a live reproduction — fix requested, test-only, `SendMessage`d back to `a97985d2364e57b4a`, fix landed in `acd0413`) + 1 Minor/open question (SPA chrome not yet routed through `react-i18next` outside the join screen — cross-cutting, not S13-specific, routed to teco/architect as a scope question, became `i18n-chrome-scope`/S17); `falkor-chat/docs/reviews/salesperson-ui-s13.md` | frontend-engineer 282.2k tok/146 tools; analyst 159.8k tok/59 tools |
| S14 | `frontend-engineer` | `a7944016cc8625387` | **accepted — committed `c2943aa`** (HISTORY `751b3f9`) — teco-verified independently (diff read, own `vitest run`/`tsc -b`/`build.sh`, own mutation test outside the implementer's table — `isOrderLine` gutted to always-true, confirmed all 9 `OrderPanel.test.tsx` tests stay green, a real but low-severity coverage gap, restored byte-identical) | `salesperson/src/views/{Cart,Order,Profile,Catalog}*`, `salesperson/public/products/**`, `salesperson/README.md` (image licence note) | `analyst` (`a856c10f668d48362`) **approve with suggestions**, no blockers/majors; 2 minor (`isOrderLine` guard's reject branch untested; `cancelled` status axis untested) + 1 nit (README "fetched at build time" wording), all non-blocking, logged rather than re-dispatched (precedent: S11/S12c); `falkor-chat/docs/reviews/salesperson-ui-s14.md` | frontend-engineer 235k tok/121 tools; analyst 126.8k tok/44 tools |
| S12b-testfix | `frontend-engineer` | `ae8f6d9ea84cab01a` | **accepted — committed `1e230ef`** (HISTORY `7cc1372`) — teco-verified independently (both diffs read in full, matches the delegate's own stat exactly; `Shell.test.tsx`'s "Loading …" strings confirmed as genuine production copy via direct grep of the four panel files, not test-invented; `git status` confirmed clean restore of both mutation-test targets, `Header.tsx`/`ResetControl.tsx`; own `vitest run` 178/178, own `tsc -b` clean); narrow test-only scope + low design risk → accepted without a further `analyst` re-gate (precedent: S11/S12c non-blocking-suggestion handling) | fixes a pre-existing test-design defect S14's landing exposed: `Shell.test.tsx`'s wiring probe asserted literal placeholder text now replaced by S14's real content; `ResetControl.test.tsx`'s global URL-unscoped `fetchMock` now also intercepts `ProfilePanel`'s own `/shop/api/state` poll. 9 pre-existing failures, confirmed by teco via a direct `vitest run`, root-caused by teco (read `ProfilePanel.tsx`), not caused by or in S14's scope | teco-accepted, no re-gate | frontend-engineer 163.6k tok/67 tools |
| S13-welcome-ownership | `architect` | `a126d81f075b127c7` | **accepted — committed `4d74fcb`** (agentId corrected 2026-09-14, was recorded swapped with S13's own analyst gate) | `docs/plans/salesperson-ui.md` (v1.37: new §4.12, §5.0 two rows, S13's §5.1 row swept), `salesperson/AGENTS.md` (ownership table row) | teco read both diffs directly (narrow, review/gap-triggered plan amendment; design mirrors `pendingLanguageStep`'s existing shape exactly, both rejected alternatives sound; three-way baseline/HEAD/worktree check clean before commit) — no further re-gate, same precedent as `S12a-ownership`/`S12c-ownership` | architect 161.4k tok/41 tools |
| welcome-turn-followup | `frontend-engineer` | `a97985d2364e57b4a` | **accepted — committed `7ab7a97`**, gate `0b71aa3` (post-commit, process note below) | `salesperson/src/session/SessionContext.tsx`, `salesperson/src/api/hooks.ts` (per §4.12's grant), `salesperson/src/views/ChatView.tsx`+`.test.tsx`, new `salesperson/src/components/message/welcome.ts`+`.test.ts`, same-subtree polish to `MessageBubble.tsx`+`.test.tsx`/`Transcript.tsx`+`.test.tsx` (timestamp suppression for the synthetic welcome row) | `analyst` (`acce46b7877e06077`) → **approve**, no blockers/majors/minors; 1 non-blocking nit (the "renders once" test doesn't independently pressure the `useMemo` deps list — only the "does not reappear" test does; structural, no action needed) — reproduced `tsc -b`/`vitest run` 190/190 itself, ran one further independent mutation (deps-list drop) beyond the implementer's 3 and teco's 1, all caught; `falkor-chat/docs/reviews/salesperson-ui-welcome-turn.md`. **Process note:** this was a genuine feature addition (new render behavior), not a narrow ownership/test-only unit — should have been gated *before* commit per this coordination's own default (plans/code → `analyst`), same as S13/S14/S12b. Committed first by oversight; gated post-commit to close the loop rather than leaving it ungated — **standing correction for every unit from here on: gate before commit** | frontend-engineer 389.5k tok/64 tools; analyst 115.1k tok/52 tools |
| S17-impl | `frontend-engineer` | `a5a09bdd4169ece43` | **accepted — committed `578c713`** (gated pre-commit, corrected process) — teco independently verified before gating: own `tsc -b` clean + `vitest run` 219/219 reproduced, diffstat (35 files) matches claim, `composerNotice.ts`/locale diffs read in full against §4.13, cognate-exception + Blocker-fix keys confirmed present, plan's own residual regex re-run (2 hits, both false-positive JSX-syntax comments, zero genuine residuals), one independent mutation (`order.status.placed` garbage value) reddened 2 `OrderPanel.test.tsx` tests correctly, restored byte-identical; kaizen entry confirmed written (`i18next.t` not pre-bound) | 17 files per §4.13's table + `layout/Header.test.tsx` (new): `src/layout/**`, `src/components/sheets/**`, `src/locales/**` (new namespaces), `src/views/Chat*`+`src/components/message/**`, `src/views/{Cart,Order,Profile,Catalog}*` | `analyst` (`ae907f03e92b509c7`) → **approve**, no blockers/majors; 1 Minor (a handful of dynamic branches — `*.staleNotice`, `order.error.*` — remain untested, confirmed genuinely pre-existing via `git show HEAD:...`, not worsened by this sweep, §4.13 never commits S17 to closing them) + 1 Nit (cosmetic formatting inconsistency, no formatter gate wired); two further independent mutations beyond teco's own (cognate-exception content mutation, interpolation-key case mismatch) both caught cleanly; `falkor-chat/docs/reviews/salesperson-ui-s17-impl.md` | frontend-engineer 292.2k tok/176 tools; analyst 174.3k tok/58 tools |
| S12d | `frontend-engineer` | `a5c467b4235ee4049` | **accepted — committed `993e8b2`** — all 3 Majors from Pass 1 fixed inside already-owned files: (1) `PresenterRoster.tsx` gained the same `unhandled`(C13) branch its two siblings already had, new `presenter.roster.error.unhandled` key (28 leaf keys/locale, parity 28/28/28 reconfirmed), old bare-500 test replaced with a real C13 test + 2 separate C9 tests; (2) `PresenterKeyScreen.tsx` gated its derived error on `login.isPending`, mirroring `PresenterResetAllControl.tsx`; (3) added exact-text `count:1` cases for `success_one`(en) and `incomplete.heading_one`(en, es, pt-BR). teco independently verified before and after the fix-back (own `tsc -b`/`vitest run` reproduced at both 237/237 and 243/243, diffstat/file-ownership match, kaizen entry confirmed, one independent mutation each round — the singular-plural-forms probe pre-fix, `PresenterRoster.tsx`'s `unhandled` derivation gutted post-fix, both caught cleanly and restored byte-identical) | `salesperson/src/views/Presenter*`, `salesperson/tests/e2e/presenter.spec.ts`, + one narrow additive `routes.tsx` swap (inline `PresenterKeyScreen`/`PresenterRoster` placeholders → real import), + new `presenter.*` namespace in `locales/{en,pt-BR,es}.json` | `analyst` (`ad2e44ad2598e44aa`) Pass 1 **needs changes** (3 Majors) → all fixed → Pass 2 **approve**, no new findings, one non-blocking residual noted (symmetric `success_one` coverage in es/pt-BR, explicitly not re-opened — Pass 1's stated minimum was "at minimum en"); `falkor-chat/docs/reviews/salesperson-ui-s12d-impl.md` | frontend-engineer 212.7k+263.8k tok/66+95 tools; analyst 143.1k+173.7k tok/57+19 tools |
| i18n-chrome-scope | `architect` | `a65d93f8f1a347756` | **accepted — committed `57db7d7`** (plan v1.39; review `falkor-chat/docs/reviews/salesperson-ui-s17.md`) | plan v1.39 closes all 6 findings: `chat.notice.messageNotSent` + `chat.transcript.label` added to the key table; residual-check regex rewritten (`grep -Pzo`, real multiline, no trailing-letter requirement) with an honest disclosed limitation (cannot see JS-variable-built strings); cognate/brand-name exception added naming `cart.total`/`order.total`/`layout.header.brand`, verified by Layer 1 only; file count corrected "thirteen"→**seventeen** everywhere, §5.1's S17 row now cites §4.13 instead of restating the count; `ChatView.test.tsx`'s `useTranslation()`-import obligation named explicitly | `analyst` (`a94295a0e72f7ec8e`) Pass 1 **needs changes** (3 Blockers, 1 Major, 2 Minor) → all fixed → **Pass 2 approve with suggestions** (1 new non-blocking Minor — cognate-exception clause has no explicit review-gate instruction, logged not re-dispatched) — Pass 2 independently re-ran the corrected residual regex (35 raw/34 genuine matches + 1 disclosed false positive) and **could not reproduce teco's own quick re-run figure of 62** — teco's `tr '\0' '\n' \| grep -c` method over-counts multi-line matches; corrected in the commit message, own-instrument lesson | architect 252.5k+309.5k tok/93+35 tools; analyst 155k+225.7k tok/48+16 tools |
| S15 | `qa-engineer` | `ad9cd71a1563ddc25` | **delivered — committed `f4f828a`** — **CONDITIONAL PASS**: most ACs met; AC-3 met for reads, not met as literally worded for agent-turn latency under heavy concurrency (measured, anticipated); AC-5 functionally correct but its access path is broken (DEF-1). 4 product defects (DEF-1/2/3 High, DEF-6 Medium) + 2 test-only (DEF-4/5 Low) — full detail in the report, summarized in RESUME HERE above. teco independently re-derived DEF-1 (read `app.py`'s `/shop` `StaticFiles` mount, no fallback route), DEF-2 (read `CartPanel.tsx`/`endpoints.ts` vs. `services.py`'s `get_cart` directly — client's `unitPrice` vs. server's `price`), DEF-3 (read `_drive_or_fault`'s except clause vs. `_run_turn`'s own `except Exception` directly), DEF-4/DEF-5 (grep-confirmed against the two spec files), and both kaizen entries (`kaizen_team` query). Did **not** independently re-run the load harness's live sweep or the 2821-test pytest baseline — accepted on report, explicitly noted, given this session's stated credit constraint | `salesperson/scripts/load_demo.py` (new), `salesperson/scripts/stub_llm_server.py` (new), `docs/test-plans/salesperson-ui.md`, `docs/test-reports/salesperson-ui-report.md` | none — QA/verification deliverable, no `analyst` code-gate on the harness scripts (teco's own read raised no design concern) | qa-engineer 476.2k tok/222 tools |
| defect-scope-view | `architect` | `a2ce6c17b00d76b08` | **accepted — advisory, no commit** — independent view on defect-fix scope, converged closely with teco's own; see "Stakeholder decision, 2026-09-16" above | none (advisory only) | n/a | architect 89.6k tok/12 tools |
| U-DEF1 | `tdd-engineer` | `a50911d63a92f8558` | **accepted — committed `97eca1c`** — `analyst` (`a9243cfa77e96edfd`) **approve with suggestions** (no blocker; 1 major non-blocking — confirmed the flagged `exc.status_code != 404` gap is real via Starlette 1.3.1 source, gave a concrete parametrized-test fix, called it non-blocking; 1 minor — `404.html`-file branch bypasses the handler entirely, latent since no such file ships today; 1 nit) — logged, not re-dispatched (precedent: S11/S12c/S14 non-blocking-finding handling); residual test-coverage gap worth folding into a future touch of this file, not blocking this unit. new `_SPAStaticFiles(StaticFiles)` subclass (Starlette `Mount`s confirmed terminal, a plain added route wouldn't have worked), wired into the `/shop` mount with an `/shop/api/*`-exclusion derived from `storefront_api.API_PREFIX`/`SHOP_MOUNT` rather than hardcoded. 4 new reproduction tests, 2 own mutations (full revert; API-exclusion removed), both killed cleanly. teco independently re-verified: confirmed `API_PREFIX[len(SHOP_MOUNT):]` slicing is correct (`"/api"` → `.strip("/")` → `"api"`), re-ran `tests/test_app.py` (57/57), and ran **one further mutation outside the implementer's own table** — removed the `exc.status_code != 404` guard (so *any* `StarletteHTTPException`, not just a real 404, falls back to the SPA shell) — **this mutation survived, all 57 tests still passed**: confirmed via `starlette.staticfiles.StaticFiles.get_response`'s own source that it can raise `HTTPException(405)` (wrong HTTP method) or `401` (PermissionError), neither of which any current test exercises against the `/shop` mount — a real, narrow gap (the guard is correct code with no test proving it's load-bearing), flagged to the `analyst` gate rather than sent back for a fix pre-gate (teco's own call: narrow, non-regressing, worth the reviewer's judgment on blocking vs. suggestion) | `falkor-chat/server/falkorchat/app.py`, `falkor-chat/server/tests/test_app.py`, `falkor-chat/docs/SERVER.md` | `analyst` (`a9243cfa77e96edfd`) → **approve with suggestions** | tdd-engineer 123.7k tok/47 tools; analyst 99.1k tok/40 tools |
| U-DEF2 | `tdd-engineer` | `aebbc315ca65ea664` | **accepted — committed `f0e5719`** — chose client-side rename (`price`, not `unitPrice`) after checking blast radius: server's `get_cart`/`_priced_cart_lines` and `tools.py`'s `ViewCartTool` (LLM-agent-facing) both consistently use `price`; renaming server-side would've touched 4+ internal call sites plus the live agent's tool-output contract. teco independently verified (server docstring, `tsc -b`/`vitest run` 7/7, one mutation outside implementer's table, restored byte-identical); `analyst` re-verified independently via zero-touch scratch copies (3 `TS2353` errors on simulated fixture drift; exact `$NaN` repro on component-only revert) | `salesperson/src/views/CartPanel.tsx`, `salesperson/src/api/endpoints.ts`, `salesperson/src/views/CartPanel.test.tsx` | `analyst` (`a216481d4eb1ea2f9`) → **approve** (no blockers/majors, 2 informational notes) | tdd-engineer 101.6k tok/37 tools; analyst 130.2k tok/41 tools |
| U-DEF3-design | `architect` | `a68995fd2d252ebeb` | **accepted — committed `04f8922`** — plan v1.40, new §4.14 + step S18: `_run_turn` reads `maybe_trigger`'s own return value (`isinstance(result, dict) and result.get("status") == "failed"`) rather than re-reading the graph; closes both the `start_workflow_run`-swallowed-fault gap and a second, independently-found resume-path budget-exhaustion gap; zero changes to `services.py`/`executor.py`/`api.py`. teco independently re-traced the mechanism against current source — every claim checked out, including the `_run_turn` before-context matching the proposed diff verbatim; `analyst` independently re-derived all four grounding claims from source too (not inherited from the plan's prose) | `docs/plans/salesperson-ui.md` (amendment, v1.40) | `analyst` (`a7fa235697e26b87d`) → **approve with suggestions** (1 minor — stale §9 step count, folds into S16; 1 nit) | architect 176.7k tok/84 tools; analyst 130.1k tok/37 tools |
| U-DEF6-spike | `data-scientist` | `a88133aaf356fda45` | **accepted — committed `6ddf88b`** — reproduced DEF-6 directly against LM Studio, application code bypassed entirely; ruled out app-layer-only cause; recommends against accepting as residual risk for first live demo; mitigation D (prompt-salience) → C (classifier+retry) → B (bounded semaphore), not A (full serialize); required re-eval protocol specified. teco independently verified the quoted `SALESPERSON_DEF` v7 language instruction against `proof_defs.py` verbatim and confirmed the reported kaizen entry in `kaizen_team`; did **not** independently re-run the LM-Studio harness (lives only in that session's own scratchpad, not committed) — accepted on report, noted | `docs/plans/salesperson-ui-ml.md` (new) | teco read directly, no code gate (advisory/investigation deliverable) → accepted | data-scientist 135.6k tok/46 tools |
| U-DEF45 | `tdd-engineer` | `ab74ceb5ca85d3079` | **accepted — committed `dafec93`** — DEF-4: 3× `page.goto('/presenter')` → `page.goto('presenter')` (relative path, joins `baseURL` correctly). DEF-5: fixed (not removed) — S14's real panels are unauthenticated in this test (`enabled: Boolean(authHeader)`, no session), so each shows its own permanent, panel-distinct "Loading …" copy; assertions updated to match, same precedent as `S12b-testfix`'s `Shell.test.tsx` fix. Live e2e run (first time either spec ever passed against a real server): `presenter.spec.ts` 6/6, `mobile-shell.spec.ts` 10/10, full `tests/e2e/` 16/16; offline `tsc -b` clean, `vitest run` 243/243. teco independently verified: diff matches report exactly, locale strings (`en.json`: "Loading your cart…"/"…order…"/"…profile…"/"the catalog…") and `hooks.ts`'s `enabled:` gating confirmed directly against source, offline suite reproduced (243/243, `tsc -b` clean); accepted the live e2e run on report (literal command output provided) — narrow, test-only, low design risk, no separate `analyst` re-gate (precedent: `S12b-testfix`) | `salesperson/tests/e2e/{presenter,mobile-shell}.spec.ts` | teco-accepted, no re-gate (precedent: `S12b-testfix`) | tdd-engineer 109.5k tok/34 tools |
| U-DEF3-fix | `tdd-engineer` | `ab6861fe4dd72fc0f` | **accepted — committed `4cebd96`** — `analyst` (`af47ccc2a69197ae6`) **approve**, no blockers/majors, 2 non-blocking minors/nits (case-6 test is a narrower sibling than "extended" implies, defensibly so; a fake defined inline rather than at module scope). Independently confirmed diff matches §4.14 verbatim, zero-touch claim on `services.py`/`executor.py`/`api.py`/`trigger.py`, ran 2 further mutations of its own (status broadened to include `"waiting"`; `and`→`or` swap) — both killed cleanly, restored byte-identical, suite re-confirmed 2830/2830 — implemented §4.14 exactly (`_run_turn` captures `maybe_trigger`'s return, `isinstance(result, dict) and result.get("status") == "failed"` → `_mark_turn_failed`); 6 test cases, all red-before/green-after as specified; 3 own mutations, all killed. teco independently verified: diff matches §4.14's prescribed change verbatim, re-ran `test_storefront.py` (105/105) and full suite (2830/2830, matching report exactly), ran **one further mutation outside the implementer's own table** — removed the `isinstance(result, dict)` guard, keeping only `.get("status")` — this crashes on `_RecordingTrigger`'s ordinary `None` return (used by most normal-turn tests), correctly killed by 2 pre-existing lifecycle tests (confirms the guard is real and load-bearing, not dead code), restored byte-identical | `falkor-chat/server/falkorchat/storefront.py`, `falkor-chat/server/tests/test_storefront.py` | `analyst` (`af47ccc2a69197ae6`) → — | tdd-engineer 111k tok/38 tools |
