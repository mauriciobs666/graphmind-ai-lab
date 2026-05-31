# Kaizen — Improvement Plan: dra-claudia

> Forward-looking backlog for the `dra-claudia` agent.
> Status: 🔵 proposed · 🟡 in-progress · ✅ done (then moved to history.md) · ⚪ rejected/deferred
> Last reviewed: 2026-05-31

## Active

| ID | Added | Priority | Status | Summary |
|------|------------|----------|--------|---------|
| K-001 | 2026-05-31 | low | 🔵 | Consider a stable patient-folder convention so prontuários survive across projects |

### K-001 — Stable prontuário location
- **Status:** 🔵 proposed
- **Priority:** low
- **Rationale:** Prontuários are written to `prontuarios/{slug}.md` relative to the project root, so they scatter per working directory and can be lost between sessions.
- **Proposed change:** Optionally anchor to a fixed path (e.g. `~/prontuarios/`) or make the location configurable, so a patient's record is found regardless of where the agent is invoked.
- **Notes:** Trade-off: project-relative keeps records with the relevant context; a fixed path centralizes them. Confirm preference with the user before changing.

## Parking lot / ideas
- Greeting could self-identify as "Dra. Cláudia" on first contact to reinforce the persona.
