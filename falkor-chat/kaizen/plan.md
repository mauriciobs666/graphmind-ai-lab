# Kaizen — Improvement Plan: falkor-chat

> Forward-looking backlog for the `falkor-chat` component.
> Status: 🔵 proposed · 🟡 in-progress · ✅ done (then moved to history.md) · ⚪ rejected/deferred
> Last reviewed: 2026-06-11 (K-001 `list_channels` query done; suite at 67/67)

## Active

_None. K-001 done 2026-06-11 — see `kaizen/history.md`._

## Parking lot / ideas

- The DESIGN §13 open questions are the larger backlog seeds — resolve as their milestones arrive:
  embedding model & dimension (M2), workflow guard expression language (M3), `identity` source of
  truth + real auth (replaces the M1 hardcoded-tenant seam, §14.3), message/embedding retention,
  cross-workspace analytics, Bolt vs RESP for the gateway.
- M1 app build (tdd-engineer) starts at the repository layer once K-001 lands — see `docs/DESIGN.md` §14.6.
