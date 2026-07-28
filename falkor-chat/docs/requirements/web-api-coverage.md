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

## User stories

_(to be filled during the interview)_

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
