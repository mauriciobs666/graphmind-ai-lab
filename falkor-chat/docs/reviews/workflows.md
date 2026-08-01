# Review: Workflows manual (factual/architectural claims)

> **Status:** active · **Owner:** `analyst` · **Tracks:** K-022, K-024 (M3)

## Scope & verdict

Static review of `falkor-chat/docs/manuals/workflows.md` (as authored, no prior version) against
`falkor-chat/docs/DESIGN.md` §6 (workflow engine model) and §14 (M1 app architecture),
`falkor-chat/AGENTS.md`, and source: `server/falkorchat/services.py`, `server/falkorchat/trigger.py`,
`server/falkorchat/executor.py`, `server/falkorchat/repository.py`, `server/falkorchat/proof_defs.py`,
`scripts/seed_workflows.sh`, `web/app.js`, `web/index.html`. This is a static grounding review only —
no walkthrough was driven against a running app (that is `qa-engineer`'s parallel pass); every finding
below was verified by reading the cited source, not inferred.

**Verdict: needs changes.** Two must-fix findings — one in the run-status state diagram, one in the
run-detail-panel walkthrough — assert behavior the code does not implement. Everything else is
should-fix/nit.

## Findings

### Must-fix

**1. The state diagram's `waiting --> failed: ran too long while parked` transition does not exist, and contradicts the manual's own "not a timer" claim a few lines below.**

- **Manual passage** (lines 37–47):
  ```
  running --> waiting: parked, needs a reply
  waiting --> running: reply received
  running --> done: reached a finish point
  running --> failed: hit an error or ran too long
  waiting --> failed: ran too long while parked
  ```
- **Contradicting source:**
  - `docs/DESIGN.md:411-418` — "What `maxSteps` actually means (K-031)": the step-budget check
    "runs only on the two driving outcomes... It is **deliberately not applied on the park path**
    (OUTCOME B — a parked run cannot self-drive)... Treat it as a safety bound, not an SLA or a cost
    budget."
  - `server/falkorchat/executor.py:416-424` (OUTCOME B, suspend): "No budget check here by design:
    the intake loop is human-paced... a parked run cannot self-drive."
  - `server/falkorchat/repository.py:1333-1348` — `resume_run`/`resume_run_with_ctx` only ever CAS
    `waiting → running` (`WHERE r.status = 'waiting' ... SET r.status = 'running'`); there is no CAS
    or write path that flips `waiting` directly to `failed`.
  - `server/falkorchat/repository.py:1399-1415` — `fail_run` has no `WHERE status = 'running'` guard,
    but it is called only from inside `_drive_loop`'s OUTCOME A/C checks (`executor.py:411`,
    `:429`) and the exception net (`_drive`, called from `run()`/`resume()` — both post-CAS, i.e.
    after the status is already `running`). No code path calls it while status is still `waiting`.
  - The manual's own text three lines later: "**waiting** — ... This is **not a timer** — nothing in
    falkor-chat resumes a waiting run just because time passed." A parked run failing "while parked"
    (i.e. due to elapsed time) is the exact behavior that sentence denies.
- **Why it matters:** a reader following the diagram would conclude a parked run can silently expire
  on its own — the opposite of the design's explicit no-scheduler guarantee (DESIGN §6.1 D-C) and the
  manual's own FAQ ("**A process run has been `waiting` for a long time — will it time out or resume
  on its own? No.**"). The diagram directly contradicts the prose two sections later in the same
  document.
- **Suggested correction:** delete the `waiting --> failed` edge entirely. The existing two edges
  already cover the real path: a reply resumes the run (`waiting --> running`), and *that* renewed
  drive can then exceed the step budget and fail (`running --> failed`) — the diagram doesn't need a
  third edge to express this, and adding one that skips the intermediate `running` state states a
  transition that cannot happen in the code.

**2. "An optional Show trace toggle... only present for runs started with tracing on" is wrong — the toggle is always present; only its content differs.**

- **Manual passage** (Walkthrough 2, lines 105–107): "An optional **Show trace** toggle, for a
  detailed technical breakdown of what the assistant did internally at each step (only present for
  runs started with tracing on — most day-to-day runs won't have one)."
- **Contradicting source:**
  - `web/index.html:190` — `<button type="button" class="mini-btn" id="run-trace-toggle">Show
    trace</button>` is a static element in the run panel's markup, not conditionally rendered per run.
  - `web/app.js:388-398` (`openRunPanel`) resets/shows the toggle unconditionally on every panel open
    — no branch on `run.trace` or any per-run flag.
  - `web/app.js:460-466` (`loadRunTrace`): clicking the toggle for a non-debug run fetches
    `/workflow-runs/{id}/trace`, gets an empty list, and renders the text **"No trace events (not a
    debug run)."** — i.e. the toggle is clickable and present for every run; the *result* differs.
- **Why it matters:** a user following the manual on a non-debug run (the common case, per the
  manual's own "most day-to-day runs won't have one") would look for the toggle to be absent and,
  finding it present, could reasonably conclude the manual is describing a different UI.
- **Suggested correction:** rephrase to something like: "An always-present **Show trace** toggle — for
  most day-to-day runs (started without tracing) it reports 'No trace events (not a debug run)'; for a
  run started with tracing on, it shows a detailed technical breakdown of what happened at each step."

### Should-fix

**3. Walkthrough 3 claims the defs list shows "kind" — it doesn't; kind only appears after clicking into the detail view.**

- **Manual passage** (lines 111–113): "Click the **Workflow defs** button... to see every published
  definition in the workspace — its name, version, and kind. Click one to see its full flowchart..."
- **Source:** `web/app.js:287-302` (`loadDefsList`) renders only `d.key`, `d.version`, and `d.name`
  per list row — no `kind`. `web/app.js:314-338` (`renderDefDetail`) is the first place `kind` is
  rendered: `` `${escapeHtml(s.key)} ${escapeHtml(s.version)} · ${escapeHtml(s.kind)}` `` — inside the
  detail view reached only after clicking a def.
- **Why it matters:** the sentence structure implies all three facts (name/version/kind) are visible
  in the initial browse list; a user scanning the list for "kind" (e.g. to tell a conversation flow
  from a process flow before clicking in) won't find it there.
- **Suggested correction:** split the sentence — "...to see every published definition in the
  workspace by name and version. Click one to see its full flowchart as a table (including its kind,
  every step...)".

**4. The trigger's ordered rule means re-`@mention`ing while a run is already `waiting` on that thread does not start a second run — it silently feeds the mention into the still-waiting run instead. The manual doesn't say this, and a reasonable reader could expect the opposite.**

- **Source:** `server/falkorchat/trigger.py:53-87` (`maybe_trigger`) — the rule is strictly ordered:
  step 2 (resume-if-waiting) runs and unconditionally `return`s *before* step 3 (@mention-to-start) is
  even reached, and step 2's check (`find_waiting_run_for_thread`) does not look at whether the
  message mentions the agent at all. So a message that re-`@mention`s the assistant while a run is
  parked on that thread is treated purely as the reply, never as "start a new one."
- **Why it matters:** a user who wants to abandon a stuck conversation and start fresh by
  `@mention`-ing again (a natural thing to try) will instead have their message swallowed as an
  answer to whatever question the parked run was asking — potentially confusing the assistant rather
  than resetting it. This is exactly the kind of "what if I do X" case the manual's FAQ section
  otherwise anticipates well.
- **Suggested correction:** add a line near "A reply in the same thread is enough" (line 91) or to the
  FAQ, e.g.: "Re-`@mention`ing the assistant while it's still waiting on you doesn't start a new
  conversation — it's treated the same as any other reply to the pending question. To abandon a
  stuck conversation, wait for it to finish or fail, or start a new thread."

### Nit

**5. The `access-request@v1` flowchart omits the guard on `submit → route` (requires `ctx.request.role` to exist); the diagram draws it as unconditional while every other edge in the same diagram is labeled with its condition.**

- **Manual passage** (lines 127–135): `submit["submit\n(files the request)"] --> route{route}` — no
  edge label, unlike every other edge in the same diagram.
- **Source:** `server/falkorchat/proof_defs.py:120-123` — the `submit → route` transition's guard is
  `{"kind": "cmp", "path": "ctx.request.role", "op": "exists"}`, not the empty/unconditional guard.
  If a submission omits `role`, the guard never fires and the run stays parked at `submit` (it does
  not silently advance).
- **Why it matters:** low — the manual's own worked example always submits `role`, so the omission
  never bites the reader following the walkthrough as written. But the asymmetry (every other edge
  labeled, this one not) reads as if this is the one "always fires" edge, when it's actually
  conditional like the rest.
- **Suggested correction:** optional — label the edge, e.g. `submit -- "request filed" --> route`, for
  consistency with the rest of the diagram. Not required for correctness of the walkthrough as
  written.

## What's solid

- The "waiting is not a timer" claim (item 2 of the brief) is accurate for both mechanisms behind it:
  DESIGN §6.1/§6.3 (D-C, no scheduler) and the `human`/`wait` mechanical-identity claim both check out
  against `proof_defs.py` (the one `wait` step, `provision`, carries the same `waitsForHuman: true` as
  the `human` steps) and the executor (`_drive_loop`'s OUTCOME B treats them identically).
- The "`@mention` once, reply without re-mentioning" claim (item 3, minus the caveat in finding 4) is
  correctly grounded in `trigger.py`'s ordered rule.
- The "process runs started via API are invisible in the web UI" claim (item 4) is architecturally
  correct, not just an observed absence: `repository.py:1233` confirms a `POST /workflow-runs` run
  gets no `TRIGGERED_BY` edge at all, and `find_runs_for_thread` (`repository.py:1528-1537`) requires
  `MATCH (r:WorkflowRun)-[:TRIGGERED_BY]->(m:Message)` — a run with no such edge can never match,
  for any thread.
- The `access-request@v1` flowchart's topology (6 steps, 6 transitions, branch conditions on
  `route`/`approval`/`provision`) matches `proof_defs.py`'s `ACCESS_REQUEST_DEF` exactly apart from
  the one nit above.
- The run-detail-panel waiting-form walkthrough (item 7) is accurate: `renderWaitingForm`
  (`web/app.js:485-548`) is driven by `config.prompt`/`config.fields`/`config.expects` exactly as
  described.
- No internal-architecture leakage found (item 8) — no mention of node labels, Cypher, `WorkflowRun`/
  `StepRun` internals, CAS mechanics, or file layout; "runId" and API request/response shapes are
  appropriate for the one section explicitly aimed at a technical/operator audience.

## Open questions

- None — all eight numbered checks in the brief were resolved against source with a definite verdict.
