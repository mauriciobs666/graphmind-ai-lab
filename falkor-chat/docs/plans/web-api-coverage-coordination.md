# K-036 (Wave 2–5) — Web API Coverage · teco coordination ledger

> **Status:** archived — archived 2026-07-29, K-036 delivered across 5 waves, M3.5 reached; QA
> verdict PASS with parked/non-blocking limitations (`docs/test-reports/web-api-coverage-report.md`);
> follow-ups K-037/K-038 filed, non-gating · **Owner:** `teco` · **Tracks:** K-036 (M3.5)

Companion to the architect plan `docs/plans/web-api-coverage.md` (v3, analyst-approved Pass 3 —
`docs/reviews/web-api-coverage.md`). This is the **coordination** record: units, owners, gates,
status. The design lives in the plan; do not paraphrase it here.

## Entry state (verified by teco, 2026-07-29)

- Working tree clean at `70d0981` (`docs(agents): trim root AGENTS.md...`).
- **Wave 1 delivered and committed** (`3d2234c`, 2026-07-28): U1 (graph-dba — `find_runs_for_thread`
  + thread-participants queries, `WorkflowRun.startedAt` index, `test_queries.sh` 256→276/276),
  U2 (coder — `services.check_demo_readiness` + `GET /workspaces/{ws}/readiness`), U3
  (frontend-engineer — workflow-defs viewer in `web/`). Server suite 614 passed at that commit.
- **Doc gap noted, not chased:** `docs/HISTORY.md` carries a dated entry for U2 only (2026-07-28);
  U1 and U3 were delivered in the same commit but have no separate HISTORY entries. Pre-existing
  drift, out of this coordination's scope — flagged in the closing report, not fixed here.
- Remaining scope: **Wave 2 (U4, U5) → Wave 3+4 (U6, U7, U8, U9) → Wave 5 (U10)**, per the plan's
  §4 build sequence and §4 dependency diagram.

## Sequencing decision (teco, file-sharing over logical-independence)

The plan's wave diagram groups units by *logical* dependency only. U4/U5 both touch
`repository.py`/`services.py`/`api.py` and their three test files; U6/U7/U8/U9 all touch
`web/app.js`/`web/index.html`. Per standing practice (serialize shared-file units rather than
run parallel agents against the same files), each wave's units are delivered as **one bundled
brief to one specialist**, not fan-out to parallel agent calls — this also matches this
component's own precedent (K-022/K-024 land multi-unit "landings" as one gated diff, not
per-unit gates).

Planned chain:

1. **`coder`** — Wave 2 backend: U4 (`GET /threads/{id}/workflow-runs`) + U5
   (`GET /threads/{id}/participants`). Depends on U1 (delivered).
2. **`analyst`** — review gate on **Wave 1 + Wave 2 combined** (decision, user + teco, 2026-07-29):
   Wave 1 (`3d2234c`) shipped without an independent code-review gate — only the plan was
   analyst-gated (three passes), not the delivered diff, a deviation from this project's own
   established practice (K-022/K-024 both treat the post-implementation analyst gate as
   non-negotiable). Rather than a separate retroactive review, Gate 1 now covers **both** diffs:
   Wave 1 (U1 graph-dba queries + index, U2 readiness endpoint, U3 defs-viewer UI) and Wave 2
   (U4/U5) together, as one implementation review against the plan.
3. **`frontend-engineer`** — Wave 3+4: U6 (run cue + detail panel), U7 (participants toggle), U8
   (readiness banner), U9 (waiting/structured-input/failure). Depends on U2/U3 (delivered) + U4/U5
   (step 1).
4. **`analyst`** — review gate on the Wave 3+4 diff.
5. **`qa-engineer`** — U10 black-box acceptance pass (two-pass session per plan §5.2), against
   green baselines. Depends on everything above.

Any "needs changes"/defect verdict loops back to the producing specialist, then re-review/re-test,
before the chain advances.

## Status

| Step | Unit(s) | Owner | Status |
|---|---|---|---|
| Wave 1 | U1, U2, U3 | graph-dba, coder, frontend-engineer | ✅ delivered 2026-07-28 (`3d2234c`) |
| Wave 2 | U4, U5 | `coder` | ✅ delivered 2026-07-29 (uncommitted) — server pytest 614→**641 passed / 1 deselected** (+27), query suite unchanged 276/276, ruff clean, HISTORY.md entry added |
| Wave 2 doc fix | refresh K-036 progress bullet | `coder` (same agent, follow-up) | ✅ delivered 2026-07-29 — `docs/BACKLOG.md` K-036 progress bullet now covers Wave 1 + Wave 2 |
| Gate 1 | review Wave 1 + Wave 2 | `analyst` | ✅ **approve with suggestions** 2026-07-29 — 0 blocker/major, 2 minor (m1: HISTORY gap, pre-tracked; m2: small test-helper dup, low priority, not chased) — `docs/reviews/web-api-coverage-impl.md`; independently re-ran suites: pytest 641/1 deselected, query 276/276, ruff clean |
| Wave 3+4 | U6, U7, U8, U9 | `frontend-engineer` | ✅ delivered 2026-07-29 (uncommitted; 3rd attempt succeeded after two transient platform 500s) — `web/app.js`/`web/index.html` wired for U6-U9, `web/run-select.js` pure-function tie-break (12/12 pass), two pre-existing CSS scoping bugs found+fixed (`#run-panel` overlay rule, `.badge` scoping), server pytest unaffected (641/1 deselected), live E2E API-level walk-through against `access-request@v1` (no headless browser available in sandbox — API-level not click-level verification), HISTORY.md + BACKLOG.md updated. Surfaced (not fixed, flagged as K-034-territory): a pre-existing `reference`/`access-request@v1` drift in this dev environment |
| Gate 2 | review Wave 3+4 | `analyst` | ⛔ **needs changes** 2026-07-29 — `docs/reviews/web-api-coverage-impl.md` `## Pass 2`. 1 major (M1: poll-tick `renderWaitingForm` destructively wipes unsaved form input every ~3s while a run is parked — realistic on `access-request@v1`'s free-text field), 2 minor (m4 snapshot re-fetched every tick instead of cached; m5 readiness panel can open before first fetch lands). Independently re-verified: pytest 641/1 deselected unchanged, tie-break tests 12/12, both CSS-fix claims check out, AC-5/U9 contracts match, doc curation accurate, K-034 drift confirmed pre-existing |
| Gate 2 fix | fix M1 (+ m4/m5 at discretion) | `frontend-engineer` | ✅ delivered 2026-07-29 — M1 fixed (`state.runWaitingKey` guards the destructive rebuild + snapshot re-fetch, reset on real state change/successful submit); m4 fixed as a side effect (snapshot no longer re-fetched every tick); m5 left as-is (cosmetic, discretionary). Verified with a DOM-stub harness driven against a live server, incl. reproducing the pre-fix defect on the same harness as a sanity check; pytest 641/1 deselected unaffected; HISTORY.md addendum added |
| Gate 2 re-review | re-review the fix | `analyst` | ✅ **approve with suggestions** 2026-07-29 — `## Pass 3`. M1 confirmed genuinely fixed (the submit/reset race is structurally impossible — no `await` between the two statements, JS run-to-completion). New non-blocking minors found while deep-tracing: m6 (`refreshRunPanel` has no mutex — an overlapping poll-vs-submit response race can transiently render a stale response last), m7 (an external same-step waiting→running→waiting round-trip between ticks could leave the form incorrectly skipped). Both self-heal within one `POLL_MS` (≤3s) tick, no data loss, timing conditions well outside normal use — **filed as follow-ups for the K-036 close/BACKLOG pass, not chased now.** Re-verified: `node --check` clean ×2, tie-break 12/12, pytest 641/1 deselected unaffected, HISTORY.md accurate, no forbidden touches |
| Wave 5 | U10 | `qa-engineer` | ✅ **PASS with parked/non-blocking limitations** 2026-07-29 — all six committed ACs (AC-1..AC-6) verified per plan §5.2's two-pass session (Pass A `triage`/v1 default, restart, Pass B `access-request`/v1, restart back to default, confirmed via `/proc/<pid>/environ`). `docs/test-plans/web-api-coverage.md` + `docs/test-reports/web-api-coverage-report.md`. Genuinely new findings (not gating): a major operational bug — overriding `TRIGGER_DEF_KEY` for the plan's own sanctioned Pass-B workaround makes `start_server.sh`'s re-seed graft `triage`'s steps onto `access-request@v1` (readiness endpoint correctly caught it); a minor stale banner-text nit in the same script; an AC-3 testability caveat (handled at REST/rendering-contract level, not a defect) |
| Close-out | BACKLOG.md ✅ delivered + file follow-ups + HISTORY.md gap-fill | `coder` | ✅ done 2026-07-29 — K-036 header ✅ delivered, M3.5 row added to milestone map, **K-037** (TRIGGER_DEF_KEY graft bug + banner nit, owner devops/coder) and **K-038** (`refreshRunPanel` poll race m6/m7, owner frontend-engineer) filed; HISTORY.md gap-fill entry added for U1/U3 (honestly dated 2026-07-29, grounded in `3d2234c`) |
| Close-out | archive plan doc | `architect` | ✅ done 2026-07-29 — `docs/plans/web-api-coverage.md` `Status: archived`, verified against the impl review + test report before flipping |
| Close-out | archive both review docs | `analyst` | ✅ done 2026-07-29 — `docs/reviews/web-api-coverage.md` + `docs/reviews/web-api-coverage-impl.md` both `Status: archived` |
| Close-out | archive both test docs | `qa-engineer` | ✅ done 2026-07-29 — `docs/test-plans/web-api-coverage.md` + `docs/test-reports/web-api-coverage-report.md` both `Status: archived` |
| Close-out | close requirements doc | `tico` | ✅ done 2026-07-29 — `docs/requirements/web-api-coverage.md` `Status: archived` (flipped straight from `Interviewing`, honestly noting the skipped `Ready for design` step rather than backfilling a fictitious one); also filled `Tracks: K-036 (M3.5)` for family consistency with its siblings |

## Observation, out of scope (2026-07-29)

`git status` shows uncommitted changes to `claude/teco/kaizen/history.md`, `claude/teco/kaizen/plan.md`,
`claude/teco/teco.md`, and `opencode/agents/severino/opencode.json` that were **not** made by this
coordination run (no unit briefed here touches those paths, and teco's own write guard doesn't
reach `teco.md`/`opencode/`). Working tree was clean at this session's start. Left untouched —
flagged for the user, not investigated further here.

(Updated as steps close; this doc gets `Status: archived` at K-036 close, per root `AGENTS.md`'s
`plans/<slug>-coordination.md` → `teco` closing-flip row.)
