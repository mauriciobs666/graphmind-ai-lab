# Kaizen — Improvement Plan: architect

> Forward-looking backlog for the `architect` agent.
> Status: 🔵 proposed · 🟡 in-progress · ✅ done (then moved to history.md) · ⚪ rejected/deferred
> Last reviewed: 2026-07-08

## Active

| ID | Added | Priority | Status | Summary |
|------|------------|----------|--------|---------|
| K-002 | 2026-06-20 | medium | 🔵 | Validate the handoff end-to-end with a real task (architect plans → coder implements) and tune the plan template from what the coder actually needed. |

### K-002 — End-to-end handoff validation
- **Status:** 🔵 proposed
- **Priority:** medium
- **Rationale:** The architect→coder split is designed but unproven. The plan template should be shaped by what the coder genuinely needs to execute without re-investigating.
- **Proposed change:** Run a real feature through architect→coder; capture friction; adjust the six-section plan structure.
- **Notes:** **Update 2026-07-08:** the transport half is now fixed — plan doc at `<component>/docs/plans/<slug>.md` is the default deliverable (K-001 ✅), handed off by path through teco to coder. What remains is the *live* validation run: does the coder execute a real architect plan cold, without re-investigating? Pairs with coder K-002 and teco K-001.

## Parking lot / ideas
- A short self-review checklist before delivering a plan (every step concrete & file-specific, alternatives recorded, risks listed, handoff summary present).
- Optionally delegate wide codebase sweeps to the Explore agent by default for large repos.
- Extend `hooks/guard-plan-doc-writes.sh` to cover Bash write patterns (`sed -i`, `>` redirects, `git commit`, package installs) **only if** the prompt-guarded Bash ever proves leaky in practice — deliberately left out on 2026-07-08 (see history).
