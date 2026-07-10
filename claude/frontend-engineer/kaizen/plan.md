# Kaizen — Improvement Plan: frontend-engineer

> Forward-looking backlog for the `frontend-engineer` agent.
> Status: 🔵 proposed · 🟡 in-progress · ✅ done (then moved to history.md) · ⚪ rejected/deferred
> Last reviewed: 2026-07-09

## Active

| ID | Added | Priority | Status | Summary |
|------|------------|----------|--------|---------|
| K-001 | 2026-07-09 | medium | 🔵 | First real-run shakedown on a UI task in this repo |
| K-002 | 2026-07-09 | low | 🔵 | Visual verification tooling (screenshots/browser automation) |

### K-001 — First real-run shakedown
- **Status:** 🔵 proposed
- **Priority:** medium
- **Rationale:** the prompt is untested against a real task; first runs usually surface routing gaps (vs. coder) and missing repo specifics.
- **Proposed change:** delegate a real UI task (e.g. a salesperson Streamlit change or the future falkor-chat front-end), observe where the agent hesitates or drifts out of scope, and fold fixes back into the prompt.
- **Notes:** watch especially whether the coder↔frontend-engineer efficiency boundary routes correctly from teco.

### K-002 — Visual verification tooling
- **Status:** 🔵 proposed
- **Priority:** low
- **Rationale:** "verify in the running UI" currently relies on dev-server output and console checks; real screenshot/browser-drive tooling (Playwright, a browser MCP server) would make visual claims evidence-backed.
- **Proposed change:** once the lab standardizes a browser-automation tool, wire it into the prompt's verification step (and consider an `mcpServers` frontmatter entry).
- **Notes:** keep it optional — Streamlit apps may only need `streamlit run` + manual checks.

## Parking lot / ideas
- A perishable "framework quirks" resource file (like graph-dba's `falkordb-quirks.md`) if the lab settles on one web framework and version-specific gotchas accumulate.
- Design-system/token conventions section if the lab adopts one.
