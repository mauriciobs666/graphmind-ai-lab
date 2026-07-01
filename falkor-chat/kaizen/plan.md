# Kaizen — Improvement Plan: falkor-chat

> Forward-looking backlog for the `falkor-chat` component.
> Status: 🔵 proposed · 🟡 in-progress · ✅ done (then moved to history.md) · ⚪ rejected/deferred
> Last reviewed: 2026-07-01 (K-002 complete — M1 server built: repo→services→MCP+REST; 51 server
> tests + query suite 92/92)

## Active

- ✅ **K-002 — M1 Chat MCP transport (mentions + read-cursors) — COMPLETE 2026-07-01.** Both steps
  landed; full detail in `kaizen/history.md` (Step 1 gate + Step 2 server). Step 2 built the
  greenfield `server/` tree (repository → services → `mcp.py` + `api.py` mounted by `app.py`), with
  REST `mentions[]` parity and docs (`DESIGN.md` §14–§15, `README.md`, `AGENTS.md`) in the same
  change. **51 server tests green; query suite 92/92.** Locked: MCP actor = `get_context()` (Q#1),
  per-thread cursors only (Q#3), member-match strategy (Q#2). **Deferred:** `create_channel` over MCP
  (Q#4), full-text `search` REST endpoint, and the minimal **web UI** (M1 §14.5) — the next M1 step.

## Parking lot / ideas

- **M1 web UI** — minimal `web/{index.html,app.js}` (channels list + thread view) over the REST API;
  the last piece of M1 chat core (DESIGN §14.5). Server front doors are done.
- **Full-text `search` REST endpoint** — `GET /search?q=` → `QUERIES.md` §5 (deferred from K-002).

- The DESIGN §13 open questions are the larger backlog seeds — resolve as their milestones arrive:
  embedding model & dimension (M2), workflow guard expression language (M3), `identity` source of
  truth + real auth (replaces the M1 hardcoded-tenant seam, §14.3), message/embedding retention,
  cross-workspace analytics, Bolt vs RESP for the gateway.
