# Kaizen — Improvement Plan: falkor-chat

> Forward-looking backlog for the `falkor-chat` component.
> Status: 🔵 proposed · 🟡 in-progress · ✅ done (then moved to history.md) · ⚪ rejected/deferred
> Last reviewed: 2026-06-11 (M1 client/server architecture defined)

## Active

| ID | Added | Owner | Milestone | Priority | Status | Summary |
|-------|------------|-----------|-----------|----------|--------|---------|
| K-001 | 2026-06-11 | graph-dba | M1 | High | 🔵 proposed | `list_channels` query — list a workspace's channels (unblocks `GET /channels`) |

### K-001 — `list_channels` query (list channels in a workspace)  🔵 PROPOSED
- **Owner:** graph-dba (FalkorDB query/schema work)
- **Milestone:** M1 — Chat core. Unblocks the `GET /channels` REST endpoint (`docs/DESIGN.md` §14.4).
- **Why:** the M1 REST surface needs to list a workspace's channels, but `QUERIES.md` has no such
  query yet — it covers list channel *members* (§2) and list recent *threads* (§3), not channels.
- **Do:**
  - Author + live-verify a `list channels` query in the `ws:{id}` graph, returning enough for the
    UI (`channelId`, `name`, and a recent-activity ordering if cheap).
  - Add it to `docs/QUERIES.md` (§3 Channels & threads) in the canonical-query format.
  - Add a matching assertion to `scripts/test_queries.sh`.
- **Acceptance:** query in `QUERIES.md`; `GRAPH.PROFILE` confirms an index-backed plan (no
  avoidable `NodeByLabelScan`); `./scripts/test_queries.sh` green at the new baseline (64/64 → 65/65).

## Parking lot / ideas

- The DESIGN §13 open questions are the larger backlog seeds — resolve as their milestones arrive:
  embedding model & dimension (M2), workflow guard expression language (M3), `identity` source of
  truth + real auth (replaces the M1 hardcoded-tenant seam, §14.3), message/embedding retention,
  cross-workspace analytics, Bolt vs RESP for the gateway.
- M1 app build (tdd-engineer) starts at the repository layer once K-001 lands — see `docs/DESIGN.md` §14.6.
