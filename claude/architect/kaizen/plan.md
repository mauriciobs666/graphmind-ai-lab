# Kaizen — Improvement Plan: architect

> Forward-looking backlog for the `architect` agent.
> Status: 🔵 proposed · 🟡 in-progress · ✅ done (then moved to history.md) · ⚪ rejected/deferred
> Last reviewed: 2026-06-20

## Active

| ID | Added | Priority | Status | Summary |
|------|------------|----------|--------|---------|
| K-001 | 2026-06-20 | medium | 🔵 | Define a conventional plan-document location/format so architect→coder handoff is consistent. |
| K-002 | 2026-06-20 | medium | 🔵 | Validate the handoff end-to-end with a real task (architect plans → coder implements) and tune the plan template from what the coder actually needed. |
| K-003 | 2026-06-20 | low | 🔵 | Consider a `permissionMode: plan` / stricter gating experiment if `Write` guardrail proves too loose in practice. |

### K-001 — Conventional plan-document location & format
- **Status:** 🔵 proposed
- **Priority:** medium
- **Rationale:** The agent currently proposes a path (`docs/plans/<feature>.md`) ad hoc. A fixed convention would make handoffs predictable and let the `coder` agent know where to look.
- **Proposed change:** Agree a default (e.g. `docs/plans/<slug>.md` per component, or inline-only) and bake it into both the architect and coder prompts.
- **Notes:** Tie to K-002 — let real use decide.

### K-002 — End-to-end handoff validation
- **Status:** 🔵 proposed
- **Priority:** medium
- **Rationale:** The architect→coder split is designed but unproven. The plan template should be shaped by what the coder genuinely needs to execute without re-investigating.
- **Proposed change:** Run a real feature through architect→coder; capture friction; adjust the six-section plan structure.
- **Notes:** —

### K-003 — Revisit tool gating
- **Status:** 🔵 proposed
- **Priority:** low
- **Rationale:** `Write` can overwrite any path; only a prompt guardrail keeps the agent from editing code. If that proves leaky, a stricter mechanism may be warranted.
- **Proposed change:** Evaluate `permissionMode: plan`, a `PreToolUse` hook restricting Write to a plans dir, or dropping Write entirely (inline-only plans). Verify current field semantics against `agent-standards` first.
- **Notes:** Re-verify Claude Code frontmatter before changing — perishable.

## Parking lot / ideas
- A short self-review checklist before delivering a plan (every step concrete & file-specific, alternatives recorded, risks listed, handoff summary present).
- Optionally delegate wide codebase sweeps to the Explore agent by default for large repos.
