# Kaizen — Improvement Plan: falkor-chat

> Forward-looking backlog for the `falkor-chat` component.
> Status: 🔵 proposed · 🟡 in-progress · ✅ done (then moved to history.md) · ⚪ rejected/deferred
> Last reviewed: 2026-07-02 (K-004 done — M1 hardening: five live-verified defects fixed + DEF-1;
> 68 server tests + query suite 92/92)

## Active

- ✅ **K-004 — M1 hardening: five live-verified defects + DEF-1 — COMPLETE 2026-07-02.** TDD
  (11 red tests → green). (1) Silent no-op writes: unknown author made §4 writes no-op while REST
  returned 201 — repository now raises on zero-row writes, service validates the actor
  (`UnknownActorError`), and `create_app`'s lifespan runs `services.ensure_actor()` (out-of-box
  fix). (2) Cursor-vs-limit message loss: cursor advanced to server clock past `LIMIT`-truncated
  rows — since-reads are now chronological (mention carried by `isMention` flag, not resorted;
  QUERIES.md §9 ordering note) and the cursor advances to the newest **delivered** `createdAt`
  (empty page advances nothing). (3) `advance_cursor` IndexError on missing member → returns
  `None`. (4) QA DEF-1: `POST /mcp` 405 → ASGI path-alias middleware serves both spellings.
  (5) RediSearch syntax errors 500 → `InvalidSearchQueryError` → 400. Full detail in
  `kaizen/history.md`. **68 server tests green; query suite 92/92.**
- ✅ **K-003 — M1 chat core finish: full-text search + web UI — COMPLETE 2026-07-01.** TDD,
  search-first. (a) `GET /search?q=` end-to-end (`repository.search_messages` → `services` → REST,
  `QUERIES.md` §5 workspace-wide) — 5 new tests. (b) minimal `web/{index.html,app.js}` (channels ·
  threads · messages · @mention-parse · search), served as static files by `app.py` at `/`
  (mounted last; catch-all behind REST + `/mcp`; same-origin ⇒ no CORS) — mount seam unit-tested,
  UI verified manually. Docs (`DESIGN.md` §12/§14.5–14.6, `README.md`, `AGENTS.md`) in the same
  change. **57 server tests green; query suite 92/92.** M1 chat core is code-complete.
- ✅ **K-002 — M1 Chat MCP transport (mentions + read-cursors) — COMPLETE 2026-07-01.** Both steps
  landed; full detail in `kaizen/history.md` (Step 1 gate + Step 2 server). Step 2 built the
  greenfield `server/` tree (repository → services → `mcp.py` + `api.py` mounted by `app.py`), with
  REST `mentions[]` parity and docs (`DESIGN.md` §14–§15, `README.md`, `AGENTS.md`) in the same
  change. **51 server tests green; query suite 92/92.** Locked: MCP actor = `get_context()` (Q#1),
  per-thread cursors only (Q#3), member-match strategy (Q#2). **Deferred:** `create_channel` over MCP
  (Q#4).

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
