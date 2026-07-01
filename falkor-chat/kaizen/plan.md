# Kaizen — Improvement Plan: falkor-chat

> Forward-looking backlog for the `falkor-chat` component.
> Status: 🔵 proposed · 🟡 in-progress · ✅ done (then moved to history.md) · ⚪ rejected/deferred
> Last reviewed: 2026-07-01 (K-003 done — M1 chat core code-complete: full-text search + web UI;
> 57 server tests + query suite 92/92)

## Active

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

- The DESIGN §13 open questions are the larger backlog seeds — resolve as their milestones arrive:
  embedding model & dimension (M2), workflow guard expression language (M3), `identity` source of
  truth + real auth (replaces the M1 hardcoded-tenant seam, §14.3), message/embedding retention,
  cross-workspace analytics, Bolt vs RESP for the gateway.
