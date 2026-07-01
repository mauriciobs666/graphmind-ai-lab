# Kaizen — Improvement Plan: falkor-chat

> Forward-looking backlog for the `falkor-chat` component.
> Status: 🔵 proposed · 🟡 in-progress · ✅ done (then moved to history.md) · ⚪ rejected/deferred
> Last reviewed: 2026-06-11 (K-001 `list_channels` query done; suite at 67/67)

## Active

- 🔵 **K-002 — M1 Chat MCP transport (mentions + read-cursors).** Add an MCP (Streamable-HTTP)
  front door for AI agents alongside the M1 REST API, on the shared `services.py`. Folds two
  capabilities into M1: participant `@mentions` (`MENTIONS_MEMBER` edge — distinct from the existing
  GraphRAG `MENTIONS`→`Entity`) and per-agent read-cursors (`ReadCursor` node). Decisions: A
  (additive transport) · 2a (chat-only; coordination deferred to M3) · Streamable-HTTP. Full spec:
  **`docs/plans/m1-chat-mcp.md`**.
  - **Step 1 (gate — graph-dba):** `ReadCursor` index+constraint in `bootstrap_schema.sh`; author +
    `GRAPH.PROFILE` the §5 queries (mention-extended §4 write paths, new §9.1–9.4); extend
    `QUERIES.md` + `test_queries.sh`. Suite must stay green (baseline 67/67) with index scans
    confirmed before app work starts.
  - **Step 2 (coder/tdd-engineer):** repository → services → `mcp.py` + `app.py` mount → REST
    mention parity; then docs (`DESIGN.md`, `AGENTS.md`, `README.md`, kaizen) in the same change.
  - **Open:** member-match index strategy (graph-dba's call, step 1, Q#2); `create_channel` over
    MCP (deferred, Q#4). Locked: MCP actor = `get_context()` (Q#1); per-thread cursors only (Q#3).

## Parking lot / ideas

- The DESIGN §13 open questions are the larger backlog seeds — resolve as their milestones arrive:
  embedding model & dimension (M2), workflow guard expression language (M3), `identity` source of
  truth + real auth (replaces the M1 hardcoded-tenant seam, §14.3), message/embedding retention,
  cross-workspace analytics, Bolt vs RESP for the gateway.
- M1 app build (tdd-engineer) starts at the repository layer once K-001 lands — see `docs/DESIGN.md` §14.6.
