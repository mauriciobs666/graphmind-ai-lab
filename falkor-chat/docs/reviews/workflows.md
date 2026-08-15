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

## Pass 2 — 2026-08-14 — §5 (definition authoring) + three new/updated FAQ entries

### Scope & verdict

Static review of the content `tico` added since Pass 1: `docs/manuals/workflows.md` §5 "Creating and
configuring a workflow definition" (lines ~167–367) and the three new/modified FAQ entries near the
bottom of the file ("I published a new definition…", "I re-published a definition with an
extra/removed step…", and the updated "Can I edit a workflow definition?"). Pass 1's §1–4/Overview/
older-FAQ findings are out of scope here and untouched. Verified against
`server/falkorchat/api.py`, `services.py`, `schemas.py`, `config.py`, `trigger.py`, `executor.py`,
`guards.py`, `tools.py`, `proof_defs.py`, and `server/tests/test_api.py`, plus one live check (ran
`schemas.WorkflowStepIn`/`WorkflowTransitionIn` directly against the installed `.venv`'s pydantic
2.13.4 to confirm a validation failure, not inferred from reading).

**Verdict: needs changes.** One blocker: the wire-format shape of `config`/`guard` shown in every
JSON example in §5 is wrong and would produce a 422 if copy-pasted. One major: the `agent` step's
`model` config key and the publish-time model-resolvability check are real, load-bearing, and
entirely undocumented. Two minor findings round it out. Everything else checked out — see "What's
solid" below; the two-step publish/materialize lifecycle, the `@mention` trigger-key mechanism, the
publish-time validation rules, size limits, the step-type-to-behavior table, all `human`/`wait`
config semantics, guard kinds/ops/caps/priority ordering, `ctx` provenance, and the
structural-vs-property-only republish semantics are all accurately and precisely described.

### Findings

**1. [BLOCKER] Every `config`/`guard` JSON example in §5 shows a nested JSON object, but the REST
wire format requires a JSON-*encoded string* — following any of these examples literally produces a
422.**

- **Manual passages affected:** the `agent` config block (lines 246–253), `human` config block
  (272–280), `wait` config block (293), `decision` config block (302–304), the transition/guard
  worked example (314–318), and the guard-kinds table's "Looks like" column (328–330). None of these
  are captioned as "the parsed value" or "JSON-stringify this before sending" — they're presented as
  literal JSON, and the transition example is explicitly framed as one element you'd put in the
  `transitions` array from "The shape of a definition" just above it.
- **Contradicting source, verified two ways:**
  - Static: `schemas.py:57-62` (`WorkflowStepIn.config: str | None`) and `:65-70`
    (`WorkflowTransitionIn.guard: str | None`) — both plain `str` fields, no `field_validator`
    anywhere in the file that coerces a dict to a string (only `ctx`/`input` on
    `StartWorkflowRunIn`/`SubmitWorkflowInputIn` get a `@field_validator`, and those really are
    dict-typed fields on the wire, a different endpoint).
  - Live: ran directly against the repo's `.venv` (pydantic 2.13.4):
    `WorkflowStepIn(key='a', type='agent', config={'systemPrompt': 'x'}, start=True)` and
    `WorkflowTransitionIn(from_='a', to='b', on='x', order=0, guard={'kind':'cmp', ...})` both raise
    `1 validation error … Input should be a valid string [type=string_type, input_value={...},
    input_type=dict]`.
  - Confirmed by the test suite's actual usage: `server/tests/test_api.py:421,543,569,638,668,674,
    739,803,809` — every one sends `config`/`guard` as a Python string literal containing escaped
    JSON, e.g. `'{"waitsForHuman": true}'`, never a nested dict.
  - `services.py:191-207` (`_normalize_opaque` docstring) states the split explicitly: "the REST
    front door types both as `str` (`schemas.py`), while service-layer and MCP callers hand over
    dicts" — this is a documented, deliberate API-vs-service-layer asymmetry, not an edge case. The
    manual's §5 is exclusively about the REST front door (`POST /workflow-defs`), so it lands on the
    wrong side of that split throughout.
  - `proof_defs.py:20-22`'s own module docstring flags the same trap from the other side: "`config`/
    `guard` are **plain dicts here**... so this module never hand-rolls JSON" — i.e. even the
    reference example def (`ACCESS_REQUEST_DEF`, whose `route→approval` transition is the literal
    source of the manual's transition guard example) is service-layer Python, not what goes over
    HTTP.
- **Why it matters:** this is the walkthrough's core payload — an operator following §5 verbatim to
  author a first definition would get a 422 on the very first `POST /workflow-defs` call that
  includes a non-empty `config` or `guard`, with no clue from the manual about why (the error message
  itself, `"Input should be a valid string"`, gives no hint that the fix is "stringify it").
- **Suggested correction:** two changes. (a) Add one sentence in "The shape of a definition" (before
  or after the size-limits paragraph, ~line 219) stating plainly: "`config` and `guard` are sent as
  JSON-encoded **strings**, not nested objects — `"config": "{\"waitsForHuman\":true}"`, not
  `"config": {"waitsForHuman":true}`." (b) Rewrite the standalone JSON blocks so they read as the
  *parsed meaning* of the string (keep them as-is for readability) but show at least one full,
  correctly-shaped request body nearby — e.g. turn the transition example into:
  ```json
  { "from": "route", "to": "approval", "on": "needs_approval", "order": 0,
    "guard": "{\"kind\":\"cmp\",\"path\":\"ctx.request.role\",\"op\":\"in\",\"value\":[\"contractor\",\"exec\"]}" }
  ```
  so a reader who copies it gets a request that actually publishes.

**2. [MAJOR] The `agent` step config's `model` key — and the publish-time model-resolvability check
it triggers — are real and load-bearing, but entirely undocumented in §5.**

- **Source:** `executor.py:639` — `requested_model = config.get("model")`, read once per node
  execution and fed to `self._models.resolve_llm("step", requested=requested_model, ...)`
  (`:640-643`); this is the K-042 per-step model-pinning mechanism the component's own
  `falkor-chat/AGENTS.md` calls out as a headline behavior ("a ref with no `/` resolves as a role...
  a per-workspace override is a hard cap"). `services.py:409-439` (`_declared_model_refs`) shows the
  same applies to an `{"kind":"llm"}` guard's own optional `model` key (kind `'guard'`), not just
  `agent` steps.
  `services.py:443-483` (`_check_models_resolvable`), called from `publish_workflow_def` at
  `services.py:1044` (immediately before the repository write), rejects a publish (400) that names an
  unresolvable model or role — exactly the behavior `falkor-chat/AGENTS.md` documents as landing in
  K-042 Landing 2: "publishing a workflow def that names an unresolvable model or role now fails at
  publish time (400) instead of first use."
- **Manual passage:** §5's "What each type's `config` understands" for `agent` (lines 245–269) lists
  only `systemPrompt`, `tools`, `maxIterations`, `waitsForHuman` — no `model`. The guard-kinds table's
  `{"kind":"llm", ...}` row (328–330) likewise shows only `text`, no `model`. And the "rules the
  server checks before writing anything" bullet list (207–218) — five bullets, framed as the
  complete set of publish-time checks — omits the model-resolvability check entirely.
- **Why it matters:** an operator who wants a specific step to use a particular model (or who hits
  the FR-9 400 because they typo'd a role/model name) has no manual passage that explains the
  mechanism, the config key, or the error. Given §5 explicitly bills its rules list and config tables
  as what the server checks / what each type's config understands, this reads as an authoritative,
  exhaustive reference — the omission of a real, currently-enforced check is the kind of gap that
  sends an operator straight to source-reading or a support ping.
- **Suggested correction:** add a `model` bullet to the `agent` config list (e.g. "`model` — optional;
  pins this step to a specific model or role instead of the kind default — see
  `falkor-chat/AGENTS.md`'s model-config note. An unresolvable model/role name is rejected at publish
  time, not at first run."), the matching note on the `{"kind":"llm"}` guard row, and a sixth bullet
  in the publish-time rules list for the resolvability check.

**3. [MINOR] The FAQ "I re-published a definition with an extra/removed step and nothing happened
(or I got an error)" leads with a symptom ("nothing happened") that cannot actually occur for that
specific scenario — an added/removed step is always structural and always raises a 409, never a
silent no-op.**

- **Source:** `services.py:360-375` (`_structural_diffs`) — a bare `steps[<key>]` presence-row
  (a step that exists on one side and not the other) always passes the structural filter (no
  `.type`/`.config` suffix to exempt it); `services.py:378-400` (`_check_no_structural_conflict`)
  unconditionally raises `WorkflowDefConflictError` whenever `diffs` is non-empty. There is no code
  path where adding or removing a step silently no-ops.
- **Why it matters:** low-to-moderate — the answer text itself is accurate ("locked... rejected...
  publish a new version"), so a reader isn't misled about the *mechanism*, only about which of the
  two symptoms in the heading is the one that actually happens for *this* trigger (extra/removed
  step). "Nothing happened" is the real symptom only for a **property-only** edit (e.g. editing just
  a `systemPrompt`) — but that case *does* take effect, it doesn't no-op either, so as written the
  heading's first clause doesn't match any real scenario for a step-set change.
- **Suggested correction:** reorder/tighten to lead with the actual behavior: "**I re-published a
  definition with an extra/removed step and got an error.**" (drop "nothing happened," or move it to
  a separate FAQ about property-only edits appearing to silently no-op, which *is* real and already
  covered by the "Changing a definition later" prose).

**4. [MINOR] The `agent` config's `tools` bullet doesn't mention that an unregistered/typo'd tool
name is not caught at publish time — only at drive time, as a run failure.**

- **Source:** `services._validate_def_spec` (services.py:884-984) never inspects `config.tools`
  contents — no whitelist-membership check runs at publish. The name is first resolved at drive time
  in `_run_agent_node` (`executor.py:645-651`, `offered = [self._tools.schema(name) for name in
  granted]`), which raises `UnknownToolError` (`tools.py:86-94`, `ToolRegistry.schema`,
  `tools.py:122-126`) — an exception that reaches the executor's M-1 fault net and fails the *run*,
  not the publish.
- **Why it matters:** low — the manual's `tools` bullet is correct about what the whitelist does
  (gates which tools the model may call); it just doesn't warn that a typo in that list surfaces much
  later (a failed run, not a rejected publish), which is the opposite of most of §5's other
  invariants (all caught at publish time).
- **Suggested correction:** optional one-clause addition: "...(a name the registry doesn't recognize
  isn't caught here — it fails the run the first time that step executes, not the publish)."

### What's solid

- The publish→materialize two-step lifecycle (§5 opening + Mermaid flowchart, lines 173-191) matches
  `api.py`'s `publish_workflow_def`/`materialize_def` routes and `services.py`'s corresponding methods
  exactly, including the "published but not materialized here can't run in this workspace" claim.
- The `@mention`-liveness claim — one conversation def per deployment, keyed by
  `FALKORCHAT_TRIGGER_DEF_KEY`/`_VERSION` (default `triage`/`v1`), with no `kind`-based filtering
  anywhere in `trigger.py` — is exactly right (`config.py:112-115`, `trigger.py:33-87` read in full).
- The five publish-time validation rules (unique step keys, exactly one `start`, transition endpoints
  must resolve, ≥1 transition required, `human`/`wait` must declare `waitsForHuman`) match
  `services._validate_def_spec` (884-984) precisely, including the documented "last, deliberately"
  ordering.
- Size limits (`MAX_STEPS=200`, `MAX_TRANSITIONS=500`, `MAX_CONFIG_LEN=8000` for both `config` and
  `guard`) match `schemas.py:51-54` exactly, current as of this pass.
- The step-type table (`agent`/`human`/`wait`/`decision` implemented; `prompt`/`tool`/`message`
  accepted at publish but `NotImplementedError` at drive time) matches `services.STEP_TYPES`
  (services.py:61-63) and `executor._execute_step` (489-529) exactly, word for word against the
  code's own docstring characterization ("typed-handler seam").
- The `waitsForHuman` field is checked generically for *any* step type post-guard-evaluation
  (`executor.py:467`), matching the manual's "optional for `agent`" framing — confirmed live against
  the shipped `triage` def's `intake` step (`scripts/seed_workflows.sh:168-199`), which is exactly
  the case the manual cites.
- `human`/`wait` config semantics (`fields` whitelist + permissive omit-fallback, `expects` per-field
  allow-list, `signal` as a single key) match `services._validate_against_parked_step`
  (1501-1561) precisely, down to "an explicitly empty `fields` list still counts as a declaration."
- The three built-in tools (`post_message`, `graphrag_retrieve`, `human_handoff` "present, not
  exercised") match `tools.py` exactly, including the AC-6 dispatch-time enforcement
  (`executor.py:749-750`) that backs the manual's "not a permission the model can talk its way
  around" claim, and `human_handoff`'s own docstring ("Present, not exercised," `tools.py:330`)
  literally matches "shipped, but not used by any published definition yet."
- `mode`/`permissions` are confirmed absent from every config-consuming site in the server (`grep`
  across `falkorchat/*.py` found zero occurrences of either key) — the manual's deliberate omission
  of both is correct, not a gap.
- Guard kinds/ops/priority: the `(guard == "", order, to)` sort (`_select_transition`,
  executor.py:891-893), all ten `cmp` ops, the `not`-takes-exactly-one-child rule, bare-`ctx`
  rejection, and the three caps (`MAX_GUARD_DEPTH=5`, `MAX_GUARD_NODES=32`, `MAX_GUARD_WIDTH=8`) all
  match `guards.py` exactly (98-417).
- `ctx` provenance: the only writers of `WorkflowRun.ctx` in `repository.py` are run creation
  (`start_run`), the resume flat-merge (`:1464`, matching `submit_workflow_input`'s
  `services.py:1451-1453`), and `fail_run`'s diagnostic `error` stamp (`:1501`, an engine-owned
  reserved key, not caller/step data) — no step output is ever auto-promoted into `ctx`, confirming
  the manual's claim with no exceptions found.
- The structural-vs-property-only republish distinction (§5 "Changing a definition later," lines
  359-367, and the matching FAQ) is precisely right, including the non-obvious detail that a step's
  `type` (not just `config`) counts as property-only, not structural — `services._structural_diffs`
  (360-375) explicitly exempts `.type`/`.config`-suffixed step diffs and `.guard`-suffixed transition
  diffs, and treats transition identity as the `(from, to, on, order)` 4-tuple, exactly as described.
- Both Mermaid diagrams in §5 are factually accurate: the publish→materialize flowchart's edge labels
  are the literal route paths, and the step-type decision flowchart's branches correctly map to the
  four implemented types with no misdescribed behavior.
- The `access-request` worked example's guard shown in the transition example (`ctx.request.role in
  [contractor, exec]`) is a byte-accurate copy of `proof_defs.py:126-129`'s real
  `route→approval` transition — content-correct, independent of finding 1's wire-format issue.

### Open questions

- None — all fourteen numbered claims plus the two Mermaid diagrams were resolved against source
  with a definite verdict; no question needs the caller's input to close this pass.
