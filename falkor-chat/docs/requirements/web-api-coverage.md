# Web API Coverage — Feature Requirements

> **Status:** Interviewing · **Owner:** `tico` · **Tracks:** — · **Last updated:** 2026-07-27

## Intent

Every capability the falkor-chat server offers over its REST API should be exercisable from the
web page, so that the stakeholder can (a) drive and eyeball the whole system by hand instead of
reaching for curl or pytest, and (b) show the system end to end — chat *and* workflows actually
running — without touching a terminal. The page must stay **minimalist**: coverage grows, visual
weight does not.

## Problem & current state

The web UI (`falkor-chat/web/index.html` + `web/app.js`) reaches four capabilities:

| Reachable today | Endpoint(s) |
|---|---|
| List / create channels | `GET`/`POST /channels` |
| List / create threads | `GET`/`POST /channels/{id}/threads` |
| Read / post messages | `GET`/`POST /threads/{id}/messages` |
| Search messages | `GET /search` |

Unreachable from the browser today:

- **Workflow definitions** — `POST /workflow-defs` (publish), `GET /workflow-defs`,
  `GET /workflow-defs/{key}`, `GET …/structure`, `POST …/versions/{version}/materialize`
- **Workflow runs** — `POST /workflow-runs` (start), `POST /workflow-runs/{id}/input`,
  `GET /workflow-runs/{id}`, `GET …/step-runs`, `GET …/trace`
- **Snapshots** — `GET /workspaces/{ws}/snapshots`, `GET …/structure`, def↔snapshot `diff`
- **Misc** — `GET /health`, `GET /messages/{id}`

Exercising any of those means curl or a script, which makes both hand-verification and demos
slow and terminal-bound.

The specific pain that triggered this: the stakeholder wants to **demonstrate the chat with
agents running workflows** — the M3 story (a human `@mentions` the agent, a def starts, the run
parks on a human step, a reply resumes it, the run completes). Every part of that story except
posting and reading messages is invisible in the browser today.

Also missing: there is no way to see **who is in a thread** (humans + agents). Note for the
architect: the server has no read path for thread members at all — `resolve_member_kinds` only
validates IDs supplied by a caller — so this need is not pure UI wiring.

## User stories

- As the demo driver, I want to `@mention` an agent in a thread and **watch the resulting
  workflow run progress** in the browser, so that the audience sees the chat and the workflow
  engine as one system.
- As the demo driver, I want to see **which workflow definitions exist** and what shape they
  have, so that I can explain what the agent is about to run before running it.
- As the demo driver, I want to **answer a run's human step and see the run resume**, so that
  the human-in-the-loop part of the story lands.
- As the stakeholder verifying by hand, I want to **exercise every API capability from the
  page**, so that I can smoke-test the running server without curl or pytest.
- As a chat participant, I want to see the **list of a thread's participants** (humans and
  agents), so that I know who is present and who I can mention.

## Functional requirements

**Committed — the demo path**

- **FR-1** The page lists the workflow definitions available to the workspace (key, version) and
  lets the user view a chosen def's shape (its steps and the transitions between them) *before*
  any run is started.
- **FR-2** When a posted message triggers a workflow run in a thread, that thread shows an
  **inline cue** naming the definition that started and giving access to the run's detail.
- **FR-3** The inline cue opens a **run detail panel** showing, for the selected run: its current
  status, the steps executed so far with their outcome and time, and the run's trace.
- **FR-4** The run detail reflects a run's progress **without the user reloading the page**.
- **FR-5** When a run parks waiting on a human step, the page makes visible that it is waiting
  and *what* it is waiting for (the step's prompt).
- **FR-6** The user can supply a parked run's awaited human input from the page and observe the
  run resume.
- **FR-7** A run that fails surfaces the failure and its reason in the run detail.
- **FR-8** The page shows a thread's **participants** — humans and agents — with the two kinds
  visually distinguishable.
- **FR-9** The chat surface keeps its current minimalist visual language: the new surfaces are
  secondary (collapsed / opened on demand), and a user who ignores workflows sees a page no
  busier than today's.

**Listed, not committed — remaining API coverage** *(nice-to-have; a later iteration)*

- **FR-10** Publish a workflow definition and materialize a version into a workspace.
- **FR-11** Snapshots: list a workspace's snapshots, view a snapshot's structure, and diff a
  definition against its workspace snapshot (the drift check that `verify_workflows.sh` does).
- **FR-12** Start a workflow run explicitly (without going through a chat mention).
- **FR-13** Server health and fetch-a-single-message-by-id.

## Out of scope

- End-user-facing product UX for workflows — this surface serves hand-verification and demoing,
  not a polished end-user feature.
- Parity with the **MCP** tool surface; MCP clients are exercised through their own channel.
- Any redesign of the chat layout, theming, or a component/framework migration.
- Authentication, per-user identity, or permissions on the new surfaces.
- Editing or authoring workflow definitions in the browser (viewing only).

## Acceptance criteria

- **AC-1** With the server running and the demo workspace seeded, a person can drive the whole
  M3 story in the browser alone — see the available defs, post the `@mention` that starts a run,
  watch the run appear and progress, answer the human step, and see the run reach a terminal
  state — **without a terminal, curl, or a page reload**.
- **AC-2** Given a run parked on a human step, When the awaited input is supplied from the page,
  Then the run's state visibly changes from waiting to running/next-step.
- **AC-3** Given a failing run, When the user opens its detail, Then the failure and its reason
  are readable in the page.
- **AC-4** Given an open thread, When the user looks at it, Then the thread's participants are
  listed and agents are distinguishable from humans.
- **AC-5** A user who never opens a workflow surface sees a chat page whose default layout is no
  more crowded than today's.

## Open questions

1. How does the demo human answer a step that needs **structured** input (e.g. `access-request`'s
   `request` object with a `role`) — a form in the page, or a plain chat reply?
2. How fresh must run progress be (FR-4) — what delay between a step advancing and the page
   showing it is still acceptable for a demo?
3. Is the def↔snapshot **drift check** (FR-11) actually part of the demo prep, i.e. should it be
   promoted into the committed set?

## Decision log

2026-07-27 — Who uses the fuller page and when? → Both a manual verification console for the
stakeholder *and* a presentable demo surface; not end-user-facing product UX.
2026-07-27 — Keep the minimalist look? → Yes, explicitly: coverage grows, visual weight does not.
2026-07-27 — What triggered this now? → Wanting to demonstrate the chat with agents running
workflows, end to end, in the browser.
2026-07-27 — Anything beyond API coverage? → Yes: show a thread's participants list (humans +
agents). No server read path exists for it today.
2026-07-27 — Coverage bar: every endpoint, or every capability? → **Demo path first**: only what
the agent+workflow demo needs is committed; the rest is listed as nice-to-have (FR-10..13).
2026-07-27 — Where is a running workflow visible? → **Both** — a light inline cue in the thread
plus a detail panel (steps, trace) opened on demand.
