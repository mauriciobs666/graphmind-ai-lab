# Kaizen — Improvement Plan: falkor-chat

> Forward-looking backlog for the `falkor-chat` component.
> Status: 🔵 proposed · 🟡 in-progress · ✅ done (then moved to history.md) · ⚪ rejected/deferred
> Last reviewed: 2026-07-01 (K-002 Step 1 gate done — schema + queries + tests; suite at 92/92)

## Active

- 🟡 **K-002 — M1 Chat MCP transport (mentions + read-cursors).** Add an MCP (Streamable-HTTP)
  front door for AI agents alongside the M1 REST API, on the shared `services.py`. Folds two
  capabilities into M1: participant `@mentions` (`MENTIONS_MEMBER` edge — distinct from the existing
  GraphRAG `MENTIONS`→`Entity`) and per-agent read-cursors (`ReadCursor` node). Decisions: A
  (additive transport) · 2a (chat-only; coordination deferred to M3) · Streamable-HTTP. Full spec:
  **`docs/plans/m1-chat-mcp.md`**.
  - ✅ **Step 1 (gate — graph-dba) — DONE 2026-07-01.** `ReadCursor` index+constraint in
    `bootstrap_schema.sh`; §4 write paths extended with the `MENTIONS_MEMBER` block and new
    `QUERIES.md` §9 (§9.1–9.4) authored & `GRAPH.PROFILE`-verified; `test_queries.sh` extended.
    **Suite green at new baseline 92/92**, all index scans confirmed. **Q#2 resolved:** member
    resolution in the write path uses dual `OPTIONAL MATCH` + `coalesce` (two `Node By Index Scan`s);
    the `OR` form is reserved for already-bound `me`/`mem` in reads. Two live gotchas captured in
    AGENTS.md: empty-`UNWIND` row collapse (guarded via `CASE`+`FOREACH`) and the `All Node Scan`
    trap on `OR` scan-anchors.
  - 🔵 **Step 2 (coder/tdd-engineer):** repository → services → `mcp.py` + `app.py` mount → REST
    mention parity; then docs (`DESIGN.md` §14/§15 MCP surface, `README.md` roadmap, kaizen) in the
    same change. **Unblocked — the Step 1 gate has passed.**
  - **Open:** `create_channel` over MCP (deferred, Q#4). Locked: MCP actor = `get_context()` (Q#1);
    per-thread cursors only (Q#3); member-match index strategy resolved (Q#2, Step 1).

## Parking lot / ideas

- The DESIGN §13 open questions are the larger backlog seeds — resolve as their milestones arrive:
  embedding model & dimension (M2), workflow guard expression language (M3), `identity` source of
  truth + real auth (replaces the M1 hardcoded-tenant seam, §14.3), message/embedding retention,
  cross-workspace analytics, Bolt vs RESP for the gateway.
- M1 app build (tdd-engineer) starts at the repository layer once K-001 lands — see `docs/DESIGN.md` §14.6.
