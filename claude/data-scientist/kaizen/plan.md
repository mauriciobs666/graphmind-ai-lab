# Kaizen — Improvement Plan: data-scientist

> Forward-looking backlog for the `data-scientist` agent.
> Status: 🔵 proposed · 🟡 in-progress · ✅ done (then moved to history.md) · ⚪ rejected/deferred
> Last reviewed: 2026-07-09

## Active

| ID | Added | Priority | Status | Summary |
|------|------------|----------|--------|---------|
| K-001 | 2026-07-09 | med | 🔵 | First-run shakedown: a real method note + a real methodology review |
| K-002 | 2026-07-09 | low | 🔵 | Perishable model/embedding landscape reference (skill or resource file) |

### K-001 — First-run shakedown: a real method note + a real methodology review
- **Status:** 🔵 proposed
- **Priority:** medium
- **Rationale:** The prompt is untested against a live run. Likely weak spots: whether the method note stays at method altitude (vs. drifting into the architect's sequencing), whether every recommendation actually ships with an evaluation design, and whether the `-ml.md` naming + hook behave as intended in both doc homes (`docs/plans/`, `docs/reviews/`).
- **Proposed change:** Delegate (a) a real method question from this lab — e.g. an embedding/chunking strategy or retrieval-eval design for `falkor-chat`'s GraphRAG layer — and (b) a methodology review of an existing plan with ML content; assess deliverables against the prompt's own structures; fold findings back.

### K-002 — Perishable model/embedding landscape reference
- **Status:** 🔵 proposed
- **Priority:** low
- **Rationale:** Model/embedding capabilities and pricing are perishable; the prompt rightly forbids quoting them from memory, but repeated WebFetch verification is wasteful. A dated, `Verified:`-stamped resource file (pattern: `graph-dba/falkordb-quirks.md`) or skill could cache the current landscape.
- **Proposed change:** If model-selection questions recur, add `data-scientist/model-landscape.md` (dated entries, re-verify stamps) and point the prompt at it — kept out of the always-on prompt.

## Parking lot / ideas
- Revisit the advisory-only shape if the lab starts wanting evals *executed* rather than designed — either grant hands-on eval-execution powers (graph-dba-style) or define a standing data-scientist→qa-engineer handoff for eval execution (2026-07-09, creation decision: user chose advisory).
- A worked example of a good method note (once one exists) linked from the prompt, if note quality proves inconsistent.
