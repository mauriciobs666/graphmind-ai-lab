# Web API Coverage — Test Report (K-036, Wave 5 / U10)

> **Status:** archived · **Owner:** `qa-engineer` · **Tracks:** K-036 (M3.5)

archived 2026-07-29 — K-036 delivered, M3.5 reached; see `docs/plans/web-api-coverage-coordination.md`

Execution of `docs/test-plans/web-api-coverage.md` against the running `falkor-chat` app
(`server/` + `web/`, uncommitted working tree, Waves 1-4 delivered and reviewed —
`docs/reviews/web-api-coverage-impl.md` Pass 1-3, approve with suggestions, zero blockers).
Executed 2026-07-29.

## Tooling constraint (repeated from the test plan, load-bearing for how to read this report)

**No browser-automation tool was available in this session** (no Playwright/Puppeteer/DevTools
MCP wired into this environment; no headless browser binary reachable). Every scenario below was
driven by calling the exact REST endpoints `web/app.js` calls, in the same sequence and payload
shapes a person clicking through the UI would trigger, cross-checked against a direct reading of
the specific `app.js`/`index.html` render/open function that consumes each response. This is the
closest black-box equivalent available, not a literal browser session — call this out explicitly
rather than imply visual confirmation that didn't happen. Two consequences worth naming: (1) no
JS-console-error check was possible (no console to observe); (2) AC-5's CSS-weight matching was
confirmed by reading the stylesheet rules, not by eyeballing rendered pixels. Nothing found during
the session contradicts what `analyst`'s three independent code-review passes already verified
about the same rendering code — this pass exercised the live server responses those renderers
consume, not the renderers themselves a second time.

## Pre-flight

- `falkordb-dev` up, LM Studio reachable (`curl localhost:1234/v1/models` returned 18 models).
- `./scripts/verify_workflows.sh acme` — `FAIL`: `triage@v1` in sync; `access-request@v1`
  **MISSING** (neither def nor snapshot present in `reference`). Ran `./scripts/seed_workflows.sh
  acme` (create-only, safe) — both defs then present. Re-ran `verify_workflows.sh acme`: `triage@v1`
  in sync; `access-request@v1` **DIVERGED** (5 differences — `reference`'s def was missing three
  steps/two transitions the `ws:acme` snapshot already had: `intake`, `research`, `answer`, and
  their transitions). This matches the task brief's disclosed pre-existing, already-known
  `reference`/`access-request@v1` drift (K-034-territory) exactly — **not re-filed as new**, and
  used below as AC-6's naturally-occurring negative case rather than manufactured.

## Pass A — server started with default config (`triage`/`v1`, no override)

`./scripts/start_server.sh` with no env overrides. Confirmed `AI agent: enabled=1`,
`Workflow: enabled=1 (triage def triage@v1)`, `GET /health` → `{"status":"ok"}`.

### AC-1 — PASS

Drove the full story via REST, mirroring `app.js`'s exact call sequence:

1. `GET /workflow-defs` → both `triage`/`access-request` listed (`loadDefsList`'s shape).
2. `GET /workflow-defs/triage/versions/v1` → 3 steps / 2 transitions, `startKey: "intake"` —
   matches `renderDefDetail`'s expected fields exactly.
3. `POST /threads/demo-welcome/messages` with `{"text": "@assistant I need help understanding
   what falkor-chat is used for.", "mentions": ["assistant"]}` — the `@mention` that starts a run.
4. Polled `GET /threads/demo-welcome/workflow-runs` (mirrors `updateRunCue`'s cadence): the new
   run appeared within one poll (`running`), reached `waiting` on the next (~3s later) — matches
   FR-4's freshness bar and the cue's non-terminal-beats-terminal tie-break (this run, non-terminal,
   correctly ranked above an older `done` run also in the thread).
5. `GET /workflow-runs/{id}` confirmed `status: "waiting"`, `atStepKey: "intake"` — the agent had
   posted a real clarifying question to the thread (`"I'm not sure what 'falkor-chat' is used
   for..."`), matching FR-5's "makes visible what it is waiting for" (via the chat transcript,
   since `intake`'s config declares no `prompt`/`fields` — a plain-chat-reply park, exactly the
   FR-6 carve-out: "plain chat reply keeps working... the panel is the path for steps whose
   continuation depends on structured values").
6. Answered by **plain chat reply**, no re-`@mention`: `POST /threads/demo-welcome/messages` with
   `{"text": "It is a hybrid chat and workflow system backed by FalkorDB..."}`.
7. Polled `GET /workflow-runs/{id}` until terminal: `running` (at `research`) → `done` within two
   more poll ticks (~6s). Final step-run list showed `intake` (x2) → `research` → `answer`, all
   `status: "done"`.

No direct FalkorDB/Cypher/file intervention was used anywhere in this sequence — only the same
REST calls a browser session driving the UI would make.

**Observation, not a new finding:** the `answer` step's generated text was not posted back to the
thread as a chat message this time (the run still reached `status: "done"` — AC-1 does not require
the final answer to land in chat, only that the run reach a terminal state). This matches a
**pre-existing, already-documented** risk in `scripts/seed_workflows.sh`'s own comments ("Defect
C" — the `answer`/`intake` nodes sometimes emit the response as plain text instead of calling
`post_message`, a known Qwen3-4B tool-calling reliability gap with an already-applied prompt-level
mitigation, tracked outside K-036). Not re-filed.

### AC-3 — attempted here, deferred to Pass B

Read `executor.py`'s `_drive_loop` before attempting: `OUTCOME B` (a `waitsForHuman` park) is
**explicitly exempt from the step-budget check** ("No budget check here by design... a parked run
cannot self-drive"), and neither `triage` nor `access-request` contains a step that can re-loop
without parking (`OUTCOME C` never applies to either def — `triage`'s `research→answer` transition
is unconditional, and `access-request`'s only non-parking node (`route`) always fires via its
conditional-or-unconditional-fallback pair). Concretely: **a chat-triggered run of either seeded
def cannot exhaust its step budget through any number of ordinary replies** — parking never
counts against the budget, and there is no other way to burn steps. This is a design property
(D-C: "wait/human steps are signal-driven, not timer-driven"), not a defect, and it made forcing
AC-3 under `triage` in Pass A genuinely inconvenient (would require inducing an LLM tool-call
malfunction, non-deterministic and not "convenient" as the task's own wording anticipated) —
deferred to Pass B per the plan's own allowance.

### AC-4 — PASS

`GET /threads/demo-welcome/participants` →
`[{"memberId":"assistant","displayName":"Assistant","kind":"Agent"},
{"memberId":"u1","displayName":"Demo User","kind":"User"}]` — both member kinds present,
`kind` field exactly what `openParticipants` uses to attach the `.badge`/"AI" pill to the `Agent`
row only.

### AC-5 — PASS (static verification, see tooling constraint)

Read `index.html`'s `<header>`/thread-head markup and `app.js`'s bottom "initial load" block:
- The only always-visible additions are `#defs-btn` (shares the generic `button` CSS rule with
  the pre-existing Search button — same visual weight by construction), `#participants-toggle`
  (`.mini-btn`-sized, `disabled` until a thread opens), and `#readiness-badge` (`.readiness` pill,
  sized like the existing header `.tenant` span).
- `#defs-panel`, `#run-panel`, `#participants-row` are all `display:none`/hidden by default in the
  stylesheet, and every function that populates them (`openDefsPanel`, `openRunPanel`,
  `openParticipants`) only runs from a click handler — confirmed by reading each call site.
- `#run-cue` (`display:none` by default) is only set to `flex` inside `renderRunCue` when
  `selectMostRelevantRun` returns non-null — zero footprint with no run.
- The bottom of `app.js` fires exactly two unconditional calls on load: `guard(loadChannels)` and
  `guard(loadReadiness)` — confirmed no other module-scope call exists in the file, and
  `run-select.js` (the only other loaded script) is pure logic with no DOM/fetch side effects
  (read in full).
- Cross-checked against the live server: only `GET /channels` and `GET /workspaces/acme/readiness`
  are the two reads the app is designed to fire unprompted; both were exercised in this session's
  own pre-flight/AC-6 calls with no other endpoint firing without an explicit action first.

This matches plan §3.3 v3's AC-5 reading exactly (three small, existing-weight trigger
affordances, not "zero new elements") — consistent with, not contradicting,
`analyst`'s Pass 2 finding that already checked the same property.

### AC-6 — PASS

`GET /workspaces/acme/readiness` (pre-flight state, before any further drift):
`triage@v1` → `inSync: true`, `problems: []`; `access-request@v1` → `inSync: false`,
`problems: ["access-request@v1: reference def and ws:acme snapshot diverge (5 differences)"]`,
overall `ready: false`. Both the **synced** (`triage`) and **desynced-and-named**
(`access-request`, with a human-readable reason) cases were exercised in a single read, using the
pre-existing drift as the negative case rather than a manufactured one — matches
`renderReadinessBadge`/`renderReadinessPanel`'s consumption of the response shape exactly (badge
text driven by `report.ready`; panel content driven by `defs[].problems`).

## Restart

Server stopped cleanly (`kill` on the reloader + worker PIDs, confirmed via `ps`). Restarted with
`FALKORCHAT_TRIGGER_DEF_KEY=access-request FALKORCHAT_TRIGGER_DEF_VERSION=v1
./scripts/start_server.sh`. Confirmed via `/proc/<pid>/environ` that the running `uvicorn` process
actually carried the override (the startup banner text did not — see **New finding 1** below).

## Pass B — restarted, pointing at `access-request`/`v1`

### AC-2 — PASS, with wide margin

Chat-triggered a fresh `access-request` run (`@assistant` mention), confirmed it parked
immediately at `submit` (`fields: ["request"]`, no `expects` → free-text field in
`renderWaitingForm`). Submitted `{"input": {"request": {"role": "engineer", ...}}}` via
`POST /workflow-runs/{id}/input`, timed wall-clock from POST to a confirmed `GET` showing the
state change (`atStepKey` advanced from `submit` to `provision`, having auto-routed past
`approval` since `role: "engineer"` isn't privileged): **~88ms**, not ~3-5s — because
`access-request` is the LLM-free `kind:'process'` def, the drive is synchronous within the POST
itself, so the state change is visible essentially immediately, comfortably inside the 5s bar with
a very large margin (not just "under the wire").

Repeated against the `approval` step (a second run, routed through it via `role: "contractor"`) to
also exercise the `<select>`-rendering path (`fields: ["decision"]`, `expects:
{"decision":["approve","reject"]}`): an invalid value (`"maybe"`) was correctly rejected with
`400 WorkflowInputRejectedError` and cost nothing (`stepCount` unchanged) — matches
`app.js`'s `guard`/`showError` toast path and the def's own "a typo can never burn step budget"
design note. The valid submit (`"approve"`) advanced the run to `provision` in **~31ms**.

Both runs were driven to `done` afterward for cleanliness.

### AC-3 — PASS (forced via a directly-started run; see caveat)

`access-request`'s deterministic branch shape makes budget exhaustion forceable in one step:
`POST /workflow-runs` with `{"defKey": "access-request", "version": "v1", "maxSteps": 1, "trace":
true}` parks at `submit` (parking is budget-exempt, so starting still succeeds even at
`maxSteps: 1`). `POST /workflow-runs/{id}/input` with `{"request": {"role": "engineer"}}` then
drives two consecutive advances (`submit→route`, `route→provision`) in one loop iteration, which
exceeds `maxSteps: 1` and fails the run deterministically. `GET /workflow-runs/{id}` confirmed
`status: "failed"`, `ctx: "{\"error\":\"step budget exceeded\",...}"` — exactly the shape
`renderRunFailure` reads (`JSON.parse(run.ctx).error`), which would render `"Failed: step budget
exceeded"` in the panel.

**Caveat, honestly recorded:** this run was started directly (`POST /workflow-runs`, no
`trigger_msg_id`), so it carries no `TRIGGERED_BY` edge and does **not** appear in
`GET /threads/demo-welcome/workflow-runs` — confirmed empty of this run id by re-polling the
thread-runs endpoint after the failure. It is therefore **not reachable via the inline cue**, only
via the same `GET /workflow-runs/{id}` read the panel itself issues once opened. Investigated
whether a chat-triggered run of either def could ever exhaust budget (so a thread-linked failing
run would exist for a fully faithful AC-3 pass): confirmed it cannot, for the reason given under
Pass A's AC-3 section — no re-loop exists in either def and parking is budget-exempt by design.
**This is a genuine, if narrow, gap in what a black-box session can demonstrate about AC-3 through
the browser alone with the current two demo defs** — not a defect in the delivered rendering code
(which was independently confirmed correct against the real response shape above), but worth
recording plainly rather than silently substituting a REST-only check for what AC-1 was specific
about ("no terminal, curl, or reload"). Recommend either a third, intentionally-tiny demo def with
a real self-loop, or accepting that AC-3 is validated at the endpoint/rendering-contract level
(as done here) rather than through cue navigation, going forward.

## Restart back to default

Server stopped and restarted with no env override. Confirmed via `/proc/<pid>/environ` that
`FALKORCHAT_TRIGGER_DEF_KEY`/`_VERSION` are **absent** from the running process's environment
(i.e., the code default `triage`/`v1` is in effect) and `GET /health` → `{"status":"ok"}`. This is
the state left running at the end of this session, matching what pytest and other manual checks
assume.

## New findings (not among the already-known items listed in the task brief)

### Finding 1 — Major: `FALKORCHAT_TRIGGER_DEF_KEY=access-request` restart corrupts the
`reference` graph's `access-request@v1` def (confirmed, reproducible)

**Severity: Major** (silent, irreversible graph-data corruption; triggered by the exact,
plan-endorsed operational procedure this feature's own test session requires).

**What was found.** After the Pass-B restart (`FALKORCHAT_TRIGGER_DEF_KEY=access-request
FALKORCHAT_TRIGGER_DEF_VERSION=v1 ./scripts/start_server.sh`), `verify_workflows.sh acme` reported
a **new, different** `access-request@v1` divergence than the pre-flight one:

```
access-request@v1
  in sync : NO (2 differences)
    - meta.startKey    def='intake'          snapshot='submit'
    - meta.startKeys   def='intake,submit'   snapshot=None
  ⚠ reference def: 2 START edges ['intake', 'submit'] — expected exactly one
```

Direct Cypher against `reference` confirmed it precisely: `access-request@v1`'s `WorkflowDef` node
in `reference` now has **9** `Step`s (the correct 6 — `submit`/`route`/`approval`/`provision`/
`activate`/`rejected` — **plus 3 spurious ones**: `intake`, `research`, `answer`, all `type:
"agent"`) and **2** `START` edges (`submit` — correct — and `intake` — spurious). `intake`/
`research`/`answer` are `triage`'s step keys, not `access-request`'s.

**Root cause, confirmed by reading the Pass-B startup log.** `start_server.sh` unconditionally
re-runs `scripts/seed_workflows.sh` on every start (stage 5/6) — including a restart. The log line
makes the mechanism visible directly:

```
[5/6] Seeding the triage workflow def...
── seeding workflow defs 'access-request@v1' + 'access-request@v1' into reference + ws:acme ──
  reference def   access-request@v1  steps=3 transitions=2  (already present — no-op)
  ...
  reference def   access-request@v1  steps=6 transitions=6  (already present — no-op)
```

Both defs print as `'access-request@v1'` because `seed_workflows.sh` reads
`DEF_KEY="${FALKORCHAT_TRIGGER_DEF_KEY:-triage}"` / `DEF_VERSION="${FALKORCHAT_TRIGGER_DEF_VERSION:-v1}"`
for its **first** (inline `triage`-literal, `steps=3 transitions=2`) def entry — the exact same
env-var pair the plan's own §7 risk #1 resolution and this task instruct an operator to override
to reach `access-request` via chat-mention. With that override active, `seed_workflows.sh`
publishes the **triage-shaped step literal (`intake`/`research`/`answer`) under the key
`access-request@v1`** — the def node itself reports "already present — no-op" (it already
existed), but the per-step `MERGE (st:Step {stepUid: $key + ':' + $version + ':' + s.key})`
underneath is keyed by `stepUid`, and `"access-request:v1:intake"` had never existed before, so it
gets created and `HAS_STEP`/`START`-linked into the *existing* `access-request@v1` def — silently
grafting `triage`'s steps onto a different, unrelated, already-published def. This is create-only
by the same design `docs/QUERIES.md`/`AGENTS.md` document everywhere else — nothing here violates
that contract locally, but the *cross-def key collision* means the "safe, idempotent" property is
violated at the def-identity level, not just the usual "re-editing the same def" case K-034 names.

**Pre-existing vs. newly observed, precisely.** The task brief pre-disclosed a *different*,
already-known drift shape (the pre-flight state above: `reference` clean, `ws:acme` snapshot
already contaminated with the same three spurious steps — presumably from an earlier session that
used this same restart trick, most plausibly Wave 3-4's own manual verification walkthrough,
which the impl review documents as having exercised exactly this `FALKORCHAT_TRIGGER_DEF_KEY`
workaround). **What is new here** is that *this session's* Pass-B restart advanced the same
mechanism one step further: the `reference` graph, previously clean, is now *also* contaminated,
and the `access-request@v1` def now carries two `START` edges instead of one. In other words:
**every restart with this override further corrupts `reference`, additively and irreversibly**
(publish/materialize being append-only, there is no clean way to remove the spurious steps short
of deleting and republishing the def+snapshot subgraphs — itself flagged elsewhere as breaking any
live `WorkflowRun`s pointing at the old snapshot).

**Impact on this test session's own results: none.** The executor drives off the `ws:acme`
snapshot, not `reference` (documented, unchanged behavior), and the snapshot's spurious steps are
simply unreachable dead nodes off the correct `submit`-rooted path — every `access-request` run
exercised in Pass B (AC-2, AC-3) executed correctly. The readiness endpoint (FR-10/AC-6) correctly
detected and named the *worsened* drift after the fact (`"reference def has 2 START edges (intake,
submit) — see K-034"`) — which is a good sign for K-036's own feature (it caught exactly the class
of problem it exists to catch), even though the underlying corruption is not this feature's fault.

**Relationship to K-034.** Same *symptom* class (duplicate `START`/`TRANSITION` edges from a
create-only re-publish) that `docs/BACKLOG.md`'s K-034 already names — the readiness endpoint's
own message cites K-034 verbatim. But the **trigger mechanism is different and, as far as this
report can tell, previously undocumented**: K-034 as filed is about re-publishing an *edited*
version of the *same* def; this is a *generic env-var name collision* between an operational
override meant for the chat-trigger wiring and an unrelated def-seeding script that happens to
read the same variable name for a different def's identity. Recommend filing this as its own
backlog note (or an explicit amendment to K-034's description) and routing a fix to whichever
agent owns `scripts/seed_workflows.sh` (`devops`/`graph-dba` per `AGENTS.md`'s script ownership) —
e.g., give the triage-literal's key/version their own dedicated env var name, independent of
`FALKORCHAT_TRIGGER_DEF_KEY`/`_VERSION`, so overriding the trigger for a demo/QA session can never
feed back into what `seed_workflows.sh` publishes. Also recommend a `graph-dba`-owned cleanup of
the now-doubly-contaminated `ws:acme` (`reference` and snapshot both) before the next demo — left
as-is at the end of this session per this task's explicit scope (QA does not remediate).

### Finding 2 — Minor: `start_server.sh`'s startup banner ignores `FALKORCHAT_TRIGGER_DEF_KEY`/`_VERSION` overrides

`scripts/start_server.sh:136` prints `"Workflow:  enabled=$FALKORCHAT_WORKFLOW_ENABLED (triage def
triage@v1)"` — a hardcoded literal, not interpolated from the actual configured
`FALKORCHAT_TRIGGER_DEF_KEY`/`_VERSION`. During the Pass-B restart, the running server's actual
trigger config was confirmed correct (`access-request`/`v1`, verified directly via
`/proc/<pid>/environ` and functionally via a successful chat-triggered `access-request` run) but
the terminal banner still claimed `"triage def triage@v1"` — misleading to an operator following
the plan's own restart procedure, who has no other on-screen confirmation of which def is
actually wired without reading `config.py` or testing it live. Low severity (purely cosmetic, no
functional effect — confirmed the actual wiring is correct), but worth a one-line fix
(`"(${FALKORCHAT_TRIGGER_DEF_KEY:-triage} def ${FALKORCHAT_TRIGGER_DEF_KEY:-triage}@${FALKORCHAT_TRIGGER_DEF_VERSION:-v1})"`
or similar) given this exact restart is a named, sanctioned operational procedure (plan §7 risk
#1) that operators will repeat.

## Confirmed known/accepted items (not re-filed)

- **m6** (overlapping `refreshRunPanel` calls, poll tick vs. post-submit, can transiently render a
  stale panel state) and **m7** (external same-step waiting→running→waiting round-trip between two
  poll ticks) — not directly observed in this session (both require timing windows a
  REST-sequenced session does not naturally hit), recorded here as confirmed-still-applicable,
  non-blocking, per the review's own characterization. Not re-tested independently; nothing in
  this session contradicts the review's analysis.
- Pre-existing `reference`/`access-request@v1` drift, pre-flight shape (`reference` clean,
  `ws:acme` snapshot contaminated, 5 differences) — confirmed present exactly as the task brief
  described, used as AC-6's negative case, not re-filed. (Finding 1 above documents how this
  session's own required restart *worsened* it — that escalation is what is newly reported, not
  the pre-existing baseline.)
- `wait`/`human` steps are signal-driven, not timer-driven (D-C) — confirmed by reading
  `executor.py`, consistent with the design.
- `prompt`/`tool`/`message` step types raise `NotImplementedError` by design (D-E) — not exercised
  (neither demo def uses these step types).

## Verdict

**PASS with parked/non-blocking limitations.**

All six acceptance criteria (AC-1..AC-6) were satisfied against the delivered `web/` +
`server/` code, exercised through the closest black-box equivalent available in this environment
(REST-level driving of the exact calls the UI makes, cross-checked against the actual render
logic). No defect was found in the K-036 feature's own delivered code — every response shape
matched exactly what the already-reviewed renderers expect, and FR-10/AC-6's readiness check
correctly caught the live drift it exists to catch, including the newly-worsened case (Finding 1).

Two genuinely new observations were made, neither blocking K-036 itself:

- **Finding 1 (Major, operational/tooling, not K-036's diff):** the plan's own sanctioned
  `FALKORCHAT_TRIGGER_DEF_KEY=access-request` restart procedure — required by this exact test
  session and endorsed by plan §7 risk #1 — silently and irreversibly grafts `triage`'s steps
  onto `access-request@v1` in the `reference` graph via `scripts/seed_workflows.sh`'s reuse of the
  same env-var pair for both defs' identity. Recommend routing a fix (`devops`/`graph-dba`) and a
  cleanup of `ws:acme` before the next demo session that uses this restart trick.
- **Finding 2 (Minor, cosmetic):** `start_server.sh`'s startup banner does not reflect a
  `FALKORCHAT_TRIGGER_DEF_KEY`/`_VERSION` override — the actual wiring is correct, only the
  printed text is stale.
- **AC-3 caveat:** forcing a *chat-triggered, thread-linked* failing run is not achievable with
  either current demo def (by design — parked steps are budget-exempt and neither def contains a
  self-loop); AC-3 was validated at the REST/rendering-contract level via a directly-started run
  instead, which is not reachable through the UI's cue navigation. Recorded as a testability gap,
  not a defect.

Neither finding, nor the AC-3 caveat, reflects a defect in the code this feature's five waves
delivered — Finding 1 is in a demo-environment shell script outside the plan's build units, and
the AC-3 gap is a property of the two proof workflow defs' shapes, not the run-panel/failure-
display code (independently confirmed correct here and, previously, by `analyst`).
