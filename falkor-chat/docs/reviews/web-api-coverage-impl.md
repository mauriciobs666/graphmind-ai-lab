# Web API Coverage — Implementation Review (Wave 1 + Wave 2)

> **Status:** archived · **Owner:** `analyst` · **Tracks:** K-036 (M3.5)

archived 2026-07-29 — K-036 delivered, M3.5 reached; see `docs/plans/web-api-coverage-coordination.md`.

Independent code review of the first two delivered waves of K-036 against the approved plan
(`docs/plans/web-api-coverage.md`, v3, Pass 3 = approve — `docs/reviews/web-api-coverage.md`).
Two diffs reviewed as one implementation landing:

- **Wave 1** — committed at `3d2234c`: U1 (`graph-dba` — `find_runs_for_thread` + thread-participants
  queries, `WorkflowRun.startedAt` index, `test_queries.sh` fix), U2 (`coder` —
  `services.check_demo_readiness` + `GET /workspaces/{ws}/readiness`), U3 (`frontend-engineer` —
  workflow-defs viewer).
- **Wave 2** — uncommitted at review time: U4 (`GET /threads/{id}/workflow-runs`) + U5
  (`GET /threads/{id}/participants`), plus `docs/BACKLOG.md`/`docs/HISTORY.md` curation.

Reviewed by reading both diffs in full (`git show 3d2234c -- falkor-chat/`, `git diff -- falkor-chat/`),
cross-checking every contract against plan §3.1(a)/(b)/(c) and §4, and independently re-running the
two test gates.

---

## Independent verification (re-run, not trusted from the report)

| Gate | Command | Result | Claimed |
|---|---|---|---|
| Server suite | `.venv/bin/python -m pytest -q` | **641 passed, 1 deselected** | 641 passed / 1 deselected ✓ matches |
| Query suite | `./scripts/test_queries.sh` | **276/276 passed** | 276/276 ✓ matches |
| Lint | `.venv/bin/python -m ruff check .` | All checks passed | ruff clean ✓ matches |

`test_queries.sh` wiped the `reference` graph at teardown as documented (`AGENTS.md`'s own
warning); re-ran `bootstrap_schema.sh acme` → `seed_demo.sh acme` → `seed_workflows.sh acme` →
`verify_workflows.sh acme` afterward to restore the demo environment to the state other manual
checks assume. `verify_workflows.sh` reports `RESULT: OK` post-restore. No code changes were made
during this review — restoration only touched live FalkorDB state, not the repo.

All three counts match what was reported in `docs/plans/web-api-coverage-coordination.md` and
`docs/HISTORY.md`. No discrepancy found.

---

## Findings

### Minor

**m1 — `docs/HISTORY.md` has no entry for U1 or U3 (Wave 1).** Confirmed by reading the file: the
only Wave-1 HISTORY entry is `## 2026-07-28 — K-036 U2: …`; U1 (queries + index +
`test_queries.sh` fix, `graph-dba`) and U3 (defs-viewer UI, `frontend-engineer`) — both delivered
in the same commit, `3d2234c` — have no entries of their own. `AGENTS.md` states the convention
plainly: "append an entry for every delivered change." This is not a new problem introduced by
this gate — `docs/plans/web-api-coverage-coordination.md` already flags it verbatim ("Doc gap
noted, not chased … Pre-existing drift, out of this coordination's scope — flagged in the closing
report, not fixed here") — so this finding is confirmation that the gap is real and still open,
not a new discovery. Recommend it get closed (two short entries, or one combined U1+U3 entry) at
or before K-036's milestone close, since `HISTORY.md` is meant to be a complete dated log and
right now a reader scanning it would miss that U1/U3 happened on 2026-07-28 at all.

**m2 — `_add_to_channel` test helper is duplicated verbatim between `test_repository.py` and
`test_api.py`.** Both Wave-2 additions define an identical raw-Cypher `MEMBER_OF`-write helper
(same body, same docstring content) because no `Repository` method for writing channel membership
exists yet (a pre-existing gap `seed_demo.sh` also works around with raw Cypher — not something
this feature was asked to fix). Low cost today, but if a third test file needs the same helper
it's worth promoting to a shared test fixture/util rather than a third copy.

### Nit

**n1 — none identified beyond m2's duplication**, which is already logged as m2.

---

## What was checked and found correct (no finding — recorded so the verdict isn't a leap of faith)

- **Contracts match the plan exactly.** `repository.find_runs_for_thread(ws, *, thread_id,
  limit=10)`, `services.list_workflow_runs_for_thread(ctx, *, thread_id, limit=10)`,
  `repository.list_thread_participants(ws, *, thread_id)`,
  `services.list_thread_participants(ctx, *, thread_id)` all match plan §3.1(a)/(b) signatures
  verbatim. Routes match: `GET /threads/{thread_id}/workflow-runs?limit=` (`Query(10, ge=1, le=50)`
  → 422 outside 1-50, tested), `GET /threads/{thread_id}/participants`. Neither declares a
  `response_model` — grep-confirmed only the three K-031 structure/diff routes do, matching the
  plan's explicit "matches the surface's convention" requirement.
- **`thread_exists`/`ThreadNotFoundError` guard reuse is real, not just claimed.** Both new
  service methods call `self._repo.thread_exists(...)` and raise `ThreadNotFoundError` before
  touching the repository read — byte-for-byte the same idiom `_validate_and_derive_role` uses
  (`services.py:513-514`). `ThreadNotFoundError` is confirmed wired to 404 via the generic
  `ServiceError` exception handler in `app.py` (not a duplicate handler).
- **`GET /workspaces/{ws}/readiness` (U2, Wave 1) is always-200** — no `raise`/`HTTPException` path
  in the route or `check_demo_readiness`; confirmed by reading both. Problem-string phrasing is
  verified **verbatim-identical** to `verify_workflows.sh`'s own strings (`"{label}: not published
  in \`reference\` at this version"`, `"... not materialized into ws:{ws} at this version"`,
  `"... diverge (N differences)"`, `"... has N START edges (...) — see K-034"`) — read both files
  side by side, not just trusted the docstring's claim.
- **`find_runs_for_thread`'s query matches the plan's evolved (not original) shape**, and the
  evolution is properly justified: the plan's originally-proposed query (no `WHERE` on `r`)
  turned out, on live `GRAPH.PROFILE`, to anchor on a workspace-wide `Message` label scan rather
  than the small `WorkflowRun` scan the plan expected — a genuinely new planner fact, documented
  in `QUERIES.md` §12.14, promoted to `claude/graph-dba/falkordb-quirks.md`, and the plan's own v2
  caveat about the index possibly being a no-op for an `ORDER BY`-only shape is confirmed exactly
  right (bare index alone was a no-op; the fix needed a second, functionally-vacuous
  `WHERE r.startedAt >= 0` predicate, shipped together with the index). U4's repository query
  carries this fix forward verbatim from `QUERIES.md` §12.14 — checked the two side by side, they
  match character-for-character.
- **No new Cypher, schema/DDL change, or `web/` touch leaked into Wave 2** — `git diff --stat`
  confirms only `api.py`/`repository.py`/`services.py`/three test files/`BACKLOG.md`/`HISTORY.md`
  changed; `bootstrap_schema.sh` (Wave 1's index) is untouched in Wave 2, as expected since the
  index already landed. Neither `docs/plans/web-api-coverage.md` nor
  `docs/reviews/web-api-coverage.md` was touched by either wave — `Status:` fields checked via
  `grep -n "Status:"` across the Wave-2 diff: zero hits, nothing flipped.
- **Three-layer test coverage matches plan §5.1's enumerated edge cases exactly**, at all three
  layers (repository/service/API) for both U4 and U5: thread-runs — empty list not 404, ordering
  (newest-first), limit truncation, unknown-thread 404 (service+API), limit-bounds 422 (API);
  participants — human-only, agent-only, both-kinds, zero-member channel (empty list, not error),
  unknown-thread 404. 27 new tests total (10 repository + 7 service + 10 API), counted directly
  against the diff, matching the reported figure.
- **U3's frontend work matches FR-9/AC-5 as the plan reads it (v3, post-B1/B2).** Read
  `web/index.html`/`web/app.js` directly: the "Workflow defs" header button is unconditionally
  rendered (`<button type="button" id="defs-btn">Workflow defs</button>` sits next to the
  pre-existing `tenant` span/search form, sharing the same generic `button`/`input` CSS rule —
  same visual weight by construction, not by a separate styling exception); `#defs-panel` is
  `display:none` by default and its content (`loadDefsList()`) is fetched only from
  `openDefsPanel()`, which only runs on click. No polling for defs, matching "static once
  published." This matches the plan's zero-footprint-until-clicked / always-visible-affordance
  reading exactly, not the earlier B1-rejected "collapsed by default" reading.
- **Doc curation for Wave 2 itself is accurate and complete.** `docs/HISTORY.md` gained a full,
  accurate entry dated 2026-07-29 for U4+U5 with matching test counts and suite numbers;
  `docs/BACKLOG.md`'s K-036 progress bullet was refreshed to cover both waves accurately (Wave 1
  and Wave 2 both marked delivered, "Remaining: U6..U10 (Waves 3-5) not started" — correct against
  the coordination ledger). No `Status:` field anywhere was flipped — this feature is correctly still
  `active`/in-progress, not closed.

---

## Verdict

**Approve with suggestions.**

No blocker, no major finding. Both waves match the plan's contracts precisely — including a case
(U1's query-anchor discovery) where the implementation correctly *deviated* from the plan's
originally-proposed query text because live `GRAPH.PROFILE` evidence contradicted the plan's
assumption, and that deviation was documented exactly the way the plan's own v2 caveat asked for.
Test coverage is thorough and traceable to the plan's edge-case list at all three layers. Both
independently-re-run suite gates match the reported numbers exactly.

Two minor findings, neither blocking: **m1** (the pre-existing, already-flagged HISTORY.md gap for
U1/U3 — confirmed still open, recommend closing before K-036's milestone close) and **m2** (a
small, low-cost test-helper duplication worth consolidating if a third call site appears). Neither
touches correctness, contract compliance, or test coverage of the delivered behavior.

---

## Pass 2 — 2026-07-29 (Wave 3+4)

Independent code review of the third delivered landing of K-036, scoped explicitly to Wave 3+4
(U6 inline run cue + run detail panel, U7 participants toggle, U8 readiness banner, U9
waiting-step form/failure display) against the approved plan (`docs/plans/web-api-coverage.md`,
v3, Pass 3 = approve). Gate 1's Wave 1+2 scope is not re-reviewed here. Reviewed diff: uncommitted
working-tree changes to `falkor-chat/web/index.html` + `falkor-chat/web/app.js`, plus the new
`falkor-chat/web/run-select.js` and `falkor-chat/web/tests/run-select.test.js`. Reviewed by reading
both changed files in full, cross-checking every contract against plan §3.2/§3.3/§5.2/§4 Wave 3-4,
tracing every new endpoint call against the actual FastAPI routes, and independently re-running the
claimed gates.

### Independent verification (re-run, not trusted from the report)

| Gate | Command | Result | Claimed |
|---|---|---|---|
| Server suite | `.venv/bin/python -m pytest -q` (FalkorDB already up as `falkordb-dev`) | **641 passed, 1 deselected** | 641 passed / 1 deselected ✓ matches, unchanged from Gate 1's baseline |
| `run-select.js` unit tests | `node web/tests/run-select.test.js` (no `node` on `PATH`; ran via the Windows-side install at `/mnt/c/Program Files/nodejs/node.exe`, WSL2 host) | **12/12 passing** | 12/12 ✓ matches |
| JS syntax | `node --check web/app.js`, `node --check web/run-select.js` | both clean | clean ✓ matches |

No code changes were made during this review.

### Findings

#### Major

**M1 — the run detail panel's poll tick unconditionally wipes an in-progress, unsubmitted
structured-input form.** `refreshRunPanel()` (`web/app.js:409-419`) runs on open, on every
`POLL_MS`-interval tick while the panel is open (`startRunPolling`, `app.js:382-385`), and
immediately after a successful submit. Every call unconditionally invokes `renderRunPanel()`
(`app.js:421-438`), which unconditionally calls `renderWaitingForm(run)` (`app.js:484-525`) with no
check for whether the currently-parked step is unchanged from what's already rendered.
`renderWaitingForm` always does `box.innerHTML = ...` (`app.js:522`), destroying and rebuilding the
`<form id="run-input-form">` and its `<input>`/`<select>` elements from scratch — which wipes any
text the user has typed but not yet submitted. Since `POLL_MS = 3000`, **any interaction slower
than ~3 seconds silently loses the user's in-progress input, every tick, for as long as the run
stays parked on the same step.** This is not a hypothetical edge case: `access-request@v1`'s
`submit` step (`server/falkorchat/proof_defs.py:74-84`) asks for a free-text `request` field —
composing that text plausibly takes longer than 3 seconds in any real (non-scripted) interaction,
and even the `approval` step's two-click `<select>` is at risk if the reviewer pauses to read the
prompt.

This is a genuine regression against this same codebase's own established polling idiom: the
existing message poll (`pollMessages()`/`appendMessage()`, `app.js:190-202`, unchanged by this
diff) is deliberately **incremental and non-destructive** — it only appends new messages and never
touches `#messages`' existing children or the composer's `<input>`. Plan §2.5 explicitly calls out
reusing "the polling idiom" and "the same shape" for the run panel; this landing reused the
*polling cadence* but not the *non-destructive rendering discipline* that makes polling safe next
to a live input field. The plan itself doesn't spell out "preserve in-progress form input across
polls" as an explicit requirement, so this isn't a contract violation in the strict sense — but it
is a real, demonstrable usability defect in the one piece of genuinely interactive UI this whole
feature ships (FR-6), and the fix is small and well-scoped: skip rebuilding
`#run-waiting-form` when `run.atStepKey` matches what's already rendered (only actually needed when
the parked step *changes*, or after a submit clears it), or read back and re-apply any in-flight
field values before replacing the DOM. The implementer's own manual verification log
(`docs/HISTORY.md`'s Waves-3-4 entry) drove the form via direct API calls rather than realistic
timed typing, which is why this wasn't caught. Recommend fixing before Wave 5 (U10) — a black-box
session that actually fills the form by hand at human typing speed is very likely to hit this and
would otherwise burn a QA cycle rediscovering it.

#### Minor

**m4 — the waiting-step snapshot is re-fetched on every poll tick, not just when the parked step
changes.** `renderWaitingForm` (`app.js:492-497`) issues a fresh
`GET /workspaces/{ws}/snapshots/{defKey}/versions/{defVersion}` on every call — i.e., every 3
seconds while the panel is open and the run stays `waiting` — even though the snapshot content for
a fixed `(defKey, defVersion)` cannot change during a run. Not incorrect and low-impact at demo
scale (one small extra read per tick), but avoidable; worth caching per `(defKey, defVersion)` if
this is revisited for M1's fix (it would also naturally help M1, since the cached step data could
be reused instead of re-parsed on every tick).

**m5 — `#readiness-panel` can be opened before `loadReadiness()`'s first response lands.**
`renderReadinessPanel` only populates `#readiness-content` once `loadReadiness()` resolves
(`app.js:588-591`); clicking the badge (wired for both click and Enter/Space, `app.js:715-719`) in
the brief window before that first fetch completes opens an empty panel with no explicit
"loading…" state (the badge itself does say "Checking…", so the signal exists, just not inside the
panel). Cosmetic; not worth a dedicated fix, but noted since it's a users'-first-second-on-page
scenario.

### What was checked and found correct (no finding — recorded so the verdict isn't a leap of faith)

- **Both CSS-fix claims verified, tightly scoped.** `#run-panel` is now joined into the shared
  `#results, #defs-panel, #run-panel { position:absolute; ...; display:none; ... }` overlay rule
  (`index.html`, was previously only styled with a width override and no default hidden/positioned
  state — confirmed by reading the pre-diff rule) — the join adds nothing else and doesn't touch
  `#results`/`#defs-panel`'s existing behavior. `.badge` broadened from `.msg .badge` to
  `.msg .badge, .chip .badge` — grepped both class names across `index.html`/`app.js`: `.chip` is
  used nowhere except the new `.participants-row .chip` rule and the new participant-chip markup
  (`openParticipants`, `app.js:571-580`), so the broadened selector cannot match anything
  pre-existing; no accidental restyle.
- **AC-5 "no more crowded" discipline holds for this wave's two new affordances.** Read
  `index.html`'s `<header>` and thread column directly: the only two new always-visible elements
  are `#readiness-badge` (next to `tenant`, matched in size/style to the existing badge/pill
  tokens) and `#participants-toggle` (next to `#thread-heading`, `.mini-btn`-sized, present but
  `disabled` until a thread is open). `#run-cue` is `display:none` by default and only set to
  `flex` when `selectMostRelevantRun` returns a non-null run (`renderRunCue`, `app.js:349-368`);
  `#run-panel` and `#participants-row` are `display:none`/hidden by default and gain no content
  until `openRunPanel`/`openParticipants` runs on click. No content fetch happens without
  interaction except `loadReadiness()` on page load — which is exactly what plan §3.3.5 specifies
  ("fetched once on load"), not a violation.
- **U6's tie-break (`run-select.js`) matches the plan's rule precisely and is genuinely
  exercised.** `selectMostRelevantRun` is dependency-free (no `require`/`import` beyond Node's
  built-in `assert` in the test file itself; the implementation file has zero dependencies) —
  confirmed by reading the whole ~45-line file. Non-terminal (`running`/`waiting`) beats terminal
  (`done`/`failed`) unconditionally regardless of recency (test cases "a running run beats a done
  run even if the done run is more recent" and the `waiting`-vs-`failed` case both pass); ties
  within the same terminality class break on the higher `startedAt` (both non-terminal-vs-non-terminal
  and terminal-vs-terminal cases pass, in both list orders). 12/12 assertions pass under a bare
  `node`, independently re-run.
- **U9 wiring matches the endpoint contracts exactly.** `POST /workflow-runs/{id}/input`'s body
  shape (`{"input": {...}}`, `api.py:320-326`, `SubmitWorkflowInputIn.input`) matches
  `submitRunInput`'s POST body (`app.js`, `body: JSON.stringify({ input: data })`) exactly. The
  form renders one input/select per `config.fields` entry sourced from the snapshot read
  (`GET /workspaces/{ws}/snapshots/{key}/versions/{version}`), a `<select>` when
  `config.expects[field]` is a list — checked against `proof_defs.ACCESS_REQUEST_DEF`'s actual
  `submit` (`fields: ["request"]`, no `expects` → text input) and `approval`
  (`fields: ["decision"]`, `expects: {"decision": ["approve","reject"]}` → `<select>`) steps
  directly, both render as the plan describes. A rejected submit propagates the `api()` throw
  through `guard`'s existing `showError` toast path (no new error handling invented) and, since the
  throw happens before `refreshRunPanel()`/`box.innerHTML` is touched, the filled-in-but-rejected
  form **is** left as the user typed it on that specific path (only the *poll-driven* re-render,
  M1 above, wipes it). On success, `submitRunInput` calls `refreshRunPanel()` then `updateRunCue()`
  immediately, not waiting for the next tick, matching §3.3.3/review-m2's requirement. Failure
  display reads `JSON.parse(run.ctx).error` exactly as specified (`renderRunFailure`,
  `app.js:443-457`), with a safe fallback (`"unknown reason"`) if `ctx` is unparseable — not
  required by the plan but a reasonable defensive addition, not a deviation.
- **`provision`'s `config.signal`-only shape (no `fields` array) is handled by a sensible fallback,
  not required by U9's literal done-condition but not a plan violation either.** U9's done-condition
  text names only `submit`/`approval`; `renderWaitingForm`'s `config.signal` fallback
  (`app.js:505-507`) is what lets the manually-verified walk-through also drive the `provision`
  `wait` step through to a terminal `done` run — a reasonable, harmless extension, not scope creep
  that touches any existing contract.
- **No server-side files touched by this wave.** `git diff --stat -- falkor-chat/web/` shows only
  `app.js`/`index.html` changed, plus the two new files; `falkor-chat/server/` is untouched by this
  diff (its currently-uncommitted changes are Wave 2's, already covered by Gate 1, and re-running
  the server suite here reproduces the identical 641/1-deselected count, consistent with no
  additional server-side change having landed since Gate 1).
- **Neither `docs/plans/web-api-coverage.md` nor `docs/reviews/web-api-coverage.md` was touched**
  (`git diff --stat` for both is empty) — no `Status:` field flipped anywhere in either document.
- **Doc curation is accurate.** `docs/HISTORY.md` gained a full, dated (2026-07-29) entry for
  U6+U7+U8+U9 with an accurate description of scope, the CSS fixes, and the manual-verification
  walk-through (including the K-034-territory drift disclosure, below); `docs/BACKLOG.md`'s K-036
  progress bullet now reads "Waves 1-4 of 5 landed, U10 remains," which matches the actual delivered
  scope exactly (U1-U9 landed, only U10 outstanding).
- **The reported `reference`/`access-request@v1` drift is genuinely pre-existing and out of
  scope.** This wave touches no schema, Cypher, or publish/materialize path — the drift is live
  graph state in this dev environment, not something any file in this diff could have introduced.
  It matches the shape of the already-tracked **K-034** backlog item verbatim (`docs/BACKLOG.md:772`,
  "duplicate `TRANSITION`/`START` edges" from create-only re-publish) — a real instance of an
  already-known, already-filed class of drift, correctly not fixed here. No action taken on it,
  per the task's own instruction.

### Verdict

**Needs changes.**

One major finding (**M1**), no blockers. The wiring, contracts, tie-break function, CSS fixes, and
doc curation are all solid — verified independently against the running endpoints, the actual
`access-request@v1` def content, and two re-run test gates, all matching the reported figures
exactly. But M1 is a real, reliably-reproducible-by-reading-the-code defect in the one genuinely
interactive surface this feature delivers (the structured-input form, FR-6): any user who takes
longer than the 3-second poll interval to fill in a field loses that input, repeatedly, for as long
as the run stays parked on the same step. Recommend `frontend-engineer` fix M1 (skip rebuilding
`#run-waiting-form` when the rendered step is unchanged, or otherwise make the poll-driven
re-render non-destructive to in-progress input) before Wave 5's `qa-engineer` black-box pass — U10
will very likely hit exactly this while driving AC-2 by hand, and fixing it now avoids burning a QA
cycle rediscovering it. **m4**/**m5** are low-cost, non-blocking, worth a look at the same time but
not gating.

---

## Pass 3 — 2026-07-29 (Wave 3+4 fix re-review)

Re-review of the fix for Pass 2's M1 (and its claimed incidental fix of m4), still against the same
uncommitted `falkor-chat/web/app.js` working-tree diff (Wave 3+4 has not been split into its own
commit — the file diff reviewed here is the full Wave-3+4 landing plus the fix, since there is no
separate pre-fix commit to diff against; the fix itself was traced by reading the current
`renderWaitingForm`/`refreshRunPanel`/`submitRunInput` against Pass 2's exact quoted logic and
confirming the destructive-every-tick behavior it described is gone). Reviewed by reading the full
current `web/app.js`, the `docs/HISTORY.md` addendum describing the fix, and independently
re-running every gate — no code changes made during this review.

### Independent verification (re-run, not trusted from the report)

| Gate | Command | Result | Claimed |
|---|---|---|---|
| `node --check web/app.js` | via `/mnt/c/Program Files/nodejs/node.exe` (no `node` on this session's `PATH`, same WSL2-host workaround Pass 2 used) | clean | clean ✓ matches |
| `node --check web/run-select.js` | same | clean | clean ✓ matches |
| `run-select.js` unit tests | `node web/tests/run-select.test.js` | **12/12 passing** | 12/12 ✓ matches, unaffected (file untouched by this fix) |
| Server suite | `.venv/bin/python -m pytest -q` | **641 passed, 1 deselected** | 641 passed / 1 deselected ✓ matches, unaffected — JS-only fix, no server file touched |

### M1 — verified fixed

Read `renderWaitingForm` (`app.js`, current) in full. The guard is:

```js
const key = `${run.runId}:${run.atStepKey}`;
if (key === state.runWaitingKey && !box.hidden) return;
```

placed *before* both the snapshot fetch and the `box.innerHTML = ...` rebuild, so an unchanged
parked step now skips both — M1's destructive-every-3s rebuild is gone, and m4's every-tick
snapshot re-fetch is gone as a genuine consequence of the same early return (verified directly:
the `api(/workspaces/.../snapshots/...)` call is textually after the guard, so it is unreachable
on a matching tick). `state.runWaitingKey` is set only after a real rebuild completes, and reset to
`null` on the status-leaves-`waiting` branch (top of the function) — so re-entering `waiting` on a
*different* step always has a non-matching key and always rebuilds correctly; confirmed by reading
both branches.

**Traced the re-park-on-same-step-key edge case specifically, per the task's request.**
`submitRunInput` is:

```js
await api(`/workflow-runs/${...}/input`, { method: "POST", body: ... });
state.runWaitingKey = null;
if (runId === state.runPanelId) await refreshRunPanel();
await updateRunCue();
```

The line `state.runWaitingKey = null;` and the `refreshRunPanel()` call that follows it are both
synchronous statements with no `await` between them — once the POST's promise resolves, this
continuation runs as one uninterrupted microtask. A `setInterval` poll tick is a macrotask and
cannot preempt that block, so **the specific race the task asked about (a tick landing between the
reset and the write) cannot happen** — JS's run-to-completion semantics rule it out structurally,
not just empirically. Forcing `runWaitingKey = null` before the re-poll means even a run that parks
again on the *identical* `atStepKey` after a submit is guaranteed a fresh rebuild on the very next
render, exactly per plan §3.3.3.

### New problem found: a narrow, self-healing race from concurrent in-flight `refreshRunPanel` calls (not a regression of M1, but worth recording)

`refreshRunPanel` is not mutex-protected against overlapping invocations, and the poll interval
(`startRunPolling`) and the post-submit call (`submitRunInput`) both call it independently. Normal
case: they don't overlap. But if the periodic tick's `refreshRunPanel` is already in flight (its
`Promise.all([GET run, GET step-runs])` awaiting) at the moment a submit resolves and triggers its
*own* `refreshRunPanel`, both calls' fetches are in flight concurrently against the same `runId`,
and whichever's `Promise.all` resolves *last* wins the render — regardless of which is fresher. If
the tick's (now-stale) fetch — kicked off before the submit, reflecting the pre-submit
`waiting`/oldStepKey state — happens to resolve *after* the submit's fresh fetch, it calls
`renderWaitingForm` with the stale `run` object. At that point `state.runWaitingKey` was already
set to the new (post-submit) key by the fresher render, so the stale run's `key` no longer matches
it, the early-return guard does **not** fire, and the stale data overwrites the just-correct box —
briefly showing an already-superseded step. This self-heals on the *next* poll tick (≤`POLL_MS` =
3s later: the tick will fetch the true current state again, and by then `runWaitingKey` will not
match it either, forcing another correct rebuild) — so it's not a lost-input regression like M1
(no user-entered text is discarded, since the affected render is a fresh, empty rebuild) and not a
permanent inconsistency. But it is a real ordering gap: neither `refreshRunPanel` nor
`renderWaitingForm` carries any "is this response still the latest request" token beyond the
coarse `runId !== state.runPanelId` check (which only guards against switching/closing the panel
entirely, not against two in-flight fetches for the *same* still-open run). Likelihood in practice
is low — it requires a poll tick's GET to still be in flight at the exact moment a submit's POST
resolves, a sub-3-second window that a human filling out a form at realistic speed is unlikely to
land in — and it self-corrects within one more tick, so it is not blocking.

**Second, related edge case, also newly identified:** the "reset `runWaitingKey` on leaving
`waiting`" logic only fires when *this session's own* `renderWaitingForm` call observes a
non-waiting status. If some other actor (a different browser tab, an MCP client, a script) submits
input for the same run outside this session, and the run completes a full waiting → running →
waiting round-trip back onto the *same* `atStepKey` entirely between two of this session's poll
ticks (i.e., this session's poll never observes the intermediate non-waiting state), then on the
next tick `key` still matches the never-reset `state.runWaitingKey` and the box stays hidden behind
the early-return guard — even though it's a genuinely new visit to that step. This requires an
external actor plus a round-trip faster than one `POLL_MS` (3s), which doesn't fit this app's
single-operator demo shape (`AGENTS.md`/plan context), so it's a theoretical robustness gap, not a
practical one for Wave 5's black-box pass.

Neither of these is the kind of defect M1 was — reliably reproducible by anyone who types for more
than 3 seconds. Both require tight, low-probability timing conditions external to normal
single-user form-filling, and both self-heal within one poll cycle. Recording them as new minor
observations (not gating Wave 5) rather than findings that need a fix now:

**m6 — overlapping `refreshRunPanel` calls (poll tick vs. post-submit) are unordered; a stale
response can transiently overwrite a fresher one.** Self-heals within one `POLL_MS` tick. Worth a
request-sequence token (e.g., stamp each `refreshRunPanel` call with an incrementing counter and
ignore a response whose stamp is behind the latest) if this surface sees more traffic/latency
variance than the demo environment, but not worth the complexity now.

**m7 — the same-step-key rebuild guard can't distinguish "still the same wait" from "back to the
same step after a full external round-trip the poll never observed."** Would need a
server-supplied, monotonically-changing identifier per wait-instance (not just `atStepKey`) to
close fully. Out of scope for a single-operator demo; noted for completeness only.

### Everything else checked

- **`box.hidden` half of the guard condition (`key === state.runWaitingKey && !box.hidden`)**:
  traced when this could diverge from `key` alone. Only path found is closing the panel and
  reopening the *same* run while it's still parked on the *same* step — `closeRunPanel`/
  `openRunPanel` neither reset `runWaitingKey` nor clear `box.hidden`, so a reopen in that exact
  state also early-returns and the previously-rendered (possibly draft-filled) form reappears
  as-is. Since `config`/`prompt` for a fixed `(defKey, defVersion, atStepKey)` cannot change during
  a run (immutable snapshot, confirmed in Pass 2's "everything checked" section and unchanged
  here), the reappearing content is not stale/wrong — at worst it's a preserved draft the user
  didn't ask to keep. Not a finding.
- **`renderRunPanel`'s other two sub-renders (`run-summary`/`run-steps` via direct `.innerHTML`,
  and `renderRunFailure`) are unconditionally rebuilt every tick, unaffected by the new guard** —
  confirmed by reading `renderRunPanel`: only the `guard(() => renderWaitingForm(run))` call is
  gated internally; the panel's status/timestamp/step-list and failure-reason display stay live on
  every poll exactly as before. No loss of freshness outside the form itself.
- **HISTORY.md addendum accuracy.** Read the "Addendum (same day)" block in full against the actual
  code: its description of the mechanism (track `(runId, atStepKey)` as `runWaitingKey`, skip
  rebuild when unchanged, reset on submit, m4 fixed as a side effect, m5 deliberately left) matches
  what the code does exactly — no overstatement found. Its claimed verification method (a DOM-stub
  harness driving the unmodified functions, one `innerHTML` write / one snapshot fetch across 3
  simulated ticks, reproducing the defect against the pre-fix source) is a plausible, appropriately
  rigorous check for this class of bug; not independently re-run here since the equivalent behavior
  was confirmed directly against the real DOM API contract by code inspection above.
- **No unrelated files touched.** `git diff --stat` for this fix's scope shows only
  `falkor-chat/web/app.js` and `falkor-chat/docs/HISTORY.md` changed since Pass 2 (the other
  modified/untracked files — `api.py`/`repository.py`/`services.py`/test files/`BACKLOG.md`/
  `run-select.js`/`web/tests/`/the coordination doc — are Wave 1-4's own already-reviewed content,
  unchanged by this fix). Neither `docs/plans/web-api-coverage.md` nor
  `docs/reviews/web-api-coverage.md` shows any diff — confirmed both are absent from `git status
  --porcelain`'s modified-file list — so no `Status:` field was touched anywhere.

### Verdict

**Approve with suggestions.**

M1 is genuinely fixed: the destructive every-tick rebuild is gone, the specific re-park-on-same-key
race the task asked about is structurally impossible (JS run-to-completion rules it out, not just
happens-to-not-occur), and m4 is fixed as a real consequence of the same guard, not just claimed.
m5 remains correctly unfixed per the standing agreement that it's discretionary. All four
independently re-run gates match exactly, including the server suite being provably unaffected by
this JS-only change.

Deep-tracing the guard for new problems (as asked) surfaced two narrow, non-blocking edge cases
that were not present in Pass 2's scope because Pass 2's finding was about the *unconditional*
rebuild — these are new, second-order observations about the *conditional* one: **m6** (unordered
concurrent `refreshRunPanel` calls can let a stale poll-tick response transiely overwrite a fresher
post-submit one) and **m7** (the same-step guard can't tell "unchanged wait" from "revisited after
an externally-driven round-trip this session's poll missed entirely"). Both require timing
conditions well outside normal single-operator, human-typing-speed form use, and both self-correct
within one more `POLL_MS` tick (≤3s) with no lost user input — neither is in the same class as M1's
guaranteed, silent data loss. Not gating: this UI is safe to hand to Wave 5's `qa-engineer`
black-box pass, including a tester who genuinely sits and fills out the form by hand.
