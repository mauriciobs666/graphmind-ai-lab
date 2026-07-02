# Kaizen — Improvement Plan: falkor-chat

> Forward-looking backlog for the `falkor-chat` component.
> Status: 🔵 proposed · 🟡 in-progress · ✅ done (then moved to history.md) · ⚪ rejected/deferred
> Last reviewed: 2026-07-02 (K-004 done — M1 hardening: five live-verified defects fixed + DEF-1;
> 68 server tests + query suite 92/92)

## Active

_(none in flight — K-002, K-003, K-004 completed and logged in `kaizen/history.md`)_

## Parking lot / ideas

- **`search` over MCP** — expose the new `search_messages` service as a fourth MCP tool so agents
  can keyword-search too (REST has it; MCP still lists 3 tools). Small, additive.
- **`create_channel` over MCP** (Q#4) — deferred from K-002; agents create threads but not channels.

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
- **Web UI mention polish** — (a) `renderMessages` checks `m.isMention` but `GET
  /threads/{id}/messages` (`read_thread` §4) never returns it — highlight is dead code; (b)
  `parseMentions` treats every `@token` as a mention, so an unknown one 400s the whole send.
- **`GET /threads/{tid}/messages/{msg_id}` ignores `tid`** — any message resolves under any
  thread path; validate membership or drop the nested route.
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
