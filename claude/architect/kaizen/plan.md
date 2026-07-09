# Kaizen — Improvement Plan: architect

> Forward-looking backlog for the `architect` agent.
> Status: 🔵 proposed · 🟡 in-progress · ✅ done (then moved to history.md) · ⚪ rejected/deferred
> Last reviewed: 2026-07-09

## Active

*(no active items)*

> **K-002 — end-to-end handoff validation — ✅ done 2026-07-09** (moved to history.md). Validated
> live in the teco K-001 run: the M3 slice-1 plan (`falkor-chat/docs/plans/m3-workflow-engine.md`)
> was executed cold by isolated-context implementers (graph-dba + tdd-engineer) with no
> re-investigation; one plan gap logged (missing `start_key` contract). Template unchanged.

## Parking lot / ideas
- A short self-review checklist before delivering a plan (every step concrete & file-specific, alternatives recorded, risks listed, handoff summary present).
- Optionally delegate wide codebase sweeps to the Explore agent by default for large repos.
- Extend `hooks/guard-plan-doc-writes.sh` to cover Bash write patterns (`sed -i`, `>` redirects, `git commit`, package installs) **only if** the prompt-guarded Bash ever proves leaky in practice — deliberately left out on 2026-07-08 (see history).
