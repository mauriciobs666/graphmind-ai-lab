# Kaizen — Improvement Plan: falkor-chat

> Forward-looking backlog for the `falkor-chat` component.
> Status: 🔵 proposed · 🟡 in-progress · ✅ done (then moved to history.md) · ⚪ rejected/deferred
> Last reviewed: 2026-07-02 (K-005 done — M1-final cleanup: search + create_channel MCP tools,
> flat `GET /messages/{mid}` route, web mention polish; 70 server tests + query suite 92/92)

## Active

_(none in flight — K-002, K-003, K-004, K-005 completed and logged in `kaizen/history.md`)_

## Parking lot / ideas

From the 2026-07-02 full-project review (defects fixed under K-004; these remain):

- **Agent authorship write path** — the §4 author MATCH is `{userId: …}` only and `services`
  hardcodes `role="user"`: an `Agent` cannot author a message yet, despite the hybrid-timeline
  premise. Needs a label-specific author resolution (per the AGENTS.md member-resolution rule)
  + role from the actor type. Natural M2 item (arrives with real per-client auth).
- **`threadId` in room-wide reads (§9.2)** — rows omit the thread, so a client can't bucket a
  workspace-wide catch-up (QA addendum A2). `Message` has no `threadId` property and reverse
  HEAD/NEXT traversal is O(thread length) — likely a denormalised `threadId` on `Message`
  (schema change; RAM rule 6 applies).
- **Write retry idempotency + first-post race** — `MERGE (m:Message)` is idempotent but the
  surrounding structural `CREATE`s are not (a same-`msgId` retry duplicates NEXT/TAIL edges,
  live-verified), and first-vs-subsequent dispatch is a two-query check-then-write (two
  concurrent first posts → two HEADs). Needs an idempotency guard and/or single-query dispatch
  that respects the locked "two write paths" decision.
- **Millisecond `createdAt` ties** — same-ms messages have unstable read order and a cursor
  advanced to a tied `createdAt` can skip a same-ms sibling at a page boundary; the `NEXT` chain
  knows the true order. Low risk single-process; revisit before concurrent writers (M2 agents).
- **Pin dependency bounds** — `fastapi`/`mcp`/`falkordb` are unbounded in `pyproject.toml`; the
  code already depends on version-specific `mcp` behavior (`lifespan_context`, python-sdk #1367).
- **Lint/type/CI** — no ruff/mypy config and nothing runs the suites automatically; a minimal
  ruff + GitHub Action with a FalkorDB service container would catch drift the QA passes catch
  manually.

- The DESIGN §13 open questions are the larger backlog seeds — resolve as their milestones arrive:
  embedding model & dimension (M2), workflow guard expression language (M3), `identity` source of
  truth + real auth (replaces the M1 hardcoded-tenant seam, §14.3), message/embedding retention,
  cross-workspace analytics, Bolt vs RESP for the gateway.
