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

_(to be filled during the interview)_

## Out of scope

_(to be filled during the interview)_

## Acceptance criteria

_(to be filled during the interview)_

## Open questions

1. What triggered this now — is there a specific capability that was painful to verify?
2. How should the workflow/snapshot surfaces relate to the chat surface visually?
3. Is MCP-surface parity in or out?

## Decision log

2026-07-27 — Who uses the fuller page and when? → Both a manual verification console for the
stakeholder *and* a presentable demo surface; not end-user-facing product UX.
2026-07-27 — Keep the minimalist look? → Yes, explicitly: coverage grows, visual weight does not.
2026-07-27 — What triggered this now? → Wanting to demonstrate the chat with agents running
workflows, end to end, in the browser.
2026-07-27 — Anything beyond API coverage? → Yes: show a thread's participants list (humans +
agents). No server read path exists for it today.
