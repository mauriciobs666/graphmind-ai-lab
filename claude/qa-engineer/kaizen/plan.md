# Kaizen — Improvement Plan: qa-engineer

> Forward-looking backlog for the `qa-engineer` agent.
> Status: 🔵 proposed · 🟡 in-progress · ✅ done (then moved to history.md) · ⚪ rejected/deferred
> Last reviewed: 2026-09-07

## Active

| ID | Added | Priority | Status | Summary |
|------|------------|----------|--------|---------|
| K-001 | 2026-07-01 | med | 🔵 | Ship a reusable test-plan + test-report markdown template pair (as skill or in-repo doc) so structure is consistent across runs |
| K-003 | 2026-07-01 | low | 🔵 | Consider a handoff protocol: qa-engineer files defects → coder/tdd-engineer fix → qa-engineer re-runs (regression loop) |
| K-004 | 2026-07-01 | low | 🔵 | Capture a first-run smoke-eval as a repeatable check; document the "new subagent isn't routable until a new session" registry-reload gotcha where users will see it |
| K-007 | 2026-09-07 | med | 🔵 | Three live-verified `falkor-chat` QA gotchas need a home in `falkor-chat/docs/SERVER.md` §1.7 (+ one fix to `falkor-chat/config/opencode.example.json`) — outside `cobb`'s write remit, so carried here ready to paste |

### K-001 — Reusable plan/report templates
- **Status:** 🔵 proposed
- **Priority:** medium
- **Rationale:** the prompt describes the plan/report structure prose-only; a concrete template (skill or doc) would make output consistent and speed each run.
- **Proposed change:** author a small `qa-templates` skill (or a `docs/_templates/` pair) with the test-plan and test-report skeletons the agent fills in.
- **Notes:** keep it lean; progressive-disclosure skill is the natural home if it grows.

### K-003 — Defect → fix → re-run handoff
- **Status:** 🔵 proposed
- **Priority:** low
- **Rationale:** QA is most valuable in a loop with implementation. A light protocol (report format that `coder`/`tdd-engineer` consume, plus a re-run pass) closes it.
- **Notes:** The `teco` side is shipped — its roster entry carries the path-handoff convention and its integrate-and-verify step carries defect→re-brief→re-run. What remains is an assessment of the loop **in practice**, and the vehicle designated for it has already passed unassessed: falkor-chat K-022→K-025 ran to a QA acceptance PASS on 2026-07-21 and did return findings (K-027 was filed out of that pass), but how the handoff itself held up was never written down. Close on a deliberate look at one completed defect→fix→re-run cycle — that one retrospectively, or the next one live.

### K-004 — First-run smoke-eval + document the registry-reload gotcha
- **Status:** 🔵 proposed
- **Priority:** low
- **Rationale:** the first-spin (2026-07-01) confirmed the agent works but had to be **proxy-run** because a freshly-created subagent isn't in the session's registry until a new session starts. Users will hit this; it belongs in the deploy/testing notes, not tribal memory.
- **Proposed change:** add a one-line "restart the session to route to a newly added agent" note to `claude/README.md` deployment section (or `cobb/TESTING.md`), and keep the M1 pass as a lightweight smoke reference.

### K-007 — Three falkor-chat QA gotchas awaiting a `falkor-chat/` home
- **Status:** 🔵 proposed
- **Priority:** medium
- **Rationale:** the 2026-09-07 `kaizen_team` distillation pass (unit U4) re-derived three facts
  from `qa-engineer`'s raw capture that are all still true and all belong in `falkor-chat/docs/`,
  where every agent sees them — not in this agent's private knowledge base. `cobb`'s write remit
  stops at `claude/`, `skills/` and a short list of MCP docs, so they are parked here in
  ready-to-paste form rather than written by stretching the remit. Source entries (for a future
  distillation pass's dedup grep): `7f3d2a1c-9b4e-4a6f-8c2d-1e5f7a9b3c6d`,
  `b2f6a3e1-7c4d-4e2a-9f0b-1d8c6a5e3f42`, `a3f0f6c2-8e1a-4b7a-9d3e-6c1f2b7a9e01`.
- **Proposed change:** whoever next works `falkor-chat/docs/` adds these three bullets to
  `docs/SERVER.md` §1.7's "QA/acceptance-testing gotchas" list, and fixes the example config:
  1. **`POST /workflow-runs/{runId}/input` takes the action nested, and rejects the flat shape
     with a misleading error.** The body schema is `{"input": {...}}`
     (`schemas.SubmitWorkflowInputIn.input: dict[str, Any]`, `default_factory=dict`); the model
     does not forbid extra keys, so a flat `{"action": "fulfill"}` parses cleanly to `input={}`
     and then 400s with `WorkflowInputRejectedError: no input submitted — an empty input cannot
     advance a parked run` (`services.py:2149`). That error reads like a workflow-state defect
     rather than a request-shape mistake — check the nesting first. (Re-verified 2026-09-07.)
  2. **`config/opencode.example.json` cannot be used verbatim as `FALKORCHAT_OPENCODE_CONFIG` on a
     box with no `OPENAI_API_KEY`.** It declares an `openai` provider whose `apiKey` is
     `{env:OPENAI_API_KEY}`, and `modelconfig._build_providers` builds a `ProviderSpec` for
     **every** provider in the catalog eagerly at `ModelGateway.from_env()`, substituting each
     one's `{env:…}` refs whether or not that provider is ever resolved — so uvicorn dies at
     startup with `ModelConfigError: … environment variable 'OPENAI_API_KEY' … is not set`. The
     durable fix is to the **example file**, not a doc bullet: an `example` config that no fresh
     box can run is the defect. Drop the `openai` block (or move it to a separate
     `opencode.cloud.example.json`), leaving `lmstudio` — and re-**resolve** the result once via
     `ModelGateway.resolve`, per the `options.baseURL` gotcha already in that section.
     (Re-verified 2026-09-07: file and code both unchanged.)
  3. **The chat/`@mention` trigger path is hardwired untraced — but a REST-started run is not.**
     `app.py:545` constructs `WorkflowTrigger(...)` with no `trace=` kwarg (default `False`,
     `trigger.py:44`) and `config.py` exposes no trace setting, so a run started by an `@mention`
     writes zero `TraceEvent`s and leaves only `Message.toolsUsed` (tool names, no arguments) as
     evidence. This is **not** a property of REST as such: `POST /workflow-runs` accepts
     `trace: true` (`schemas.StartWorkflowRunIn.trace`, shipped with K-024 U3) and
     `GET /workflow-runs/{runId}/trace` reads the events back black-box. So to ground-truth raw
     tool-call **arguments**, start the run over REST with `trace: true`; reach for
     `server/tests/test_workflow_live.py`'s in-process harness only when the pass must exercise
     the `@mention` trigger path itself.
- **Notes:** item 3 corrects the raw capture, which claimed REST-driven runs write zero traces in
  general and prescribed the in-process harness unconditionally.

## Parking lot / ideas

- **Judged and kept, do not re-litigate (2026-08-24, C5 lint).** Four restatements in this file
  will read as class-7 duplicates to any future dedup sweep. All four are keeps, and the reasons
  differ:
  - **The doc-convention override, stated twice with its exception** — once in phase 2 ("Detect the
    convention first"), once in the "Match the project" principle. This one is not a judgment call:
    `docs/reviews/doc-reference-convention.md` **m17** found the second clause and ruled that it
    must carry the exception too, *"or the rewritten `:28` is contradicted from 26 lines below."*
    Removing either is a regression against a completed review.
  - **The environment-mutation rule + its subagent fallback, in phase 3 and in Guardrails** — the
    strongest-looking candidate in the file (22 w) and still a keep: phase 3's instance fires inside
    the baseline procedure, Guardrails' is the standing rule read at a different moment. Also
    protected structurally — the `agent-maintenance` skill §4 check 3 requires *every* "ask" phrasing
    to carry its own delegated-subagent carve-out, so thinning one is a certification regression.
  - **"Never report a pass you didn't observe" (Principles) vs. "never invent a passing run"
    (Guardrails)** — the one genuine "same moment, same actor" instance in the file, i.e. the only
    one that fails the finding-5 test. Kept anyway: anti-fabrication is the last category to thin
    for a 7-word gain.
  - **"Match the component's existing framework, layout, naming…" (phase 3) vs. "Match the project"
    (Principles)** — authoring a test file vs. general orientation.
- **Two accuracy nits, noted not fixed (2026-08-24, C5 lint).** (1) The second hook bullet's
  "nothing else" is very slightly wrong: the shared core `claude/scripts/guard-doc-writes.sh`
  appends `/tmp/*` to every wrapper's allowlist, so three path classes are allowed, not two.
  Behaviorally immaterial under `on_mismatch=pass`; recorded so a future enforcement-parity pass
  doesn't score it as drift. (2) The retained "This hook never escalates" is arguably 4 words of
  waste — both branches are already exhaustively enumerated by the two sentences before it, and its
  intended payload (contrast with every other agent's guard) is a contrast this agent cannot see,
  since it never reads another agent's prompt. Left alone in C5 rather than re-editing a bullet
  whose lint had already run; available to a future unit under its own gate.
- **Corpus non-conformance to reconcile elsewhere (2026-08-24).** `falkor-chat/docs/test-reports/`
  holds five filenames that don't match the repo-wide grammar — `graphrag-eval-2026-08-15.md`,
  `graphrag-eval-2026-08-16.md`, `guard-judge-calibration-2026-08-17.md`,
  `guard-judge-calibration-2026-08-21.md` (dated, no `-report` role) and
  `docs/test-reports/kaizen-agent-ontology.md` (no `-report`). C5's edit removed the prompt's only
  textual license for them, so the prompt is now correct and the corpus is not — while the agent is
  separately told to learn from the corpus. **Not a `qa-engineer` prompt item**: the case those
  files represent (a second run of the same test plan) is already answered by root `AGENTS.md`
  collision rule 5, which is auto-loaded. It belongs to whoever next reconciles `falkor-chat/docs/`.
- Optional non-functional playbooks (perf smoke via `GRAPH.PROFILE`, basic security/permission probes) as an on-demand skill rather than resident prompt weight.
- A `qa-engineer` ↔ `saul`/`dra-claudia`-style workdir option if the user later wants reports kept out of version control (currently in-repo `docs/` per user's choice).
