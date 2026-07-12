# Kaizen — Improvement Plan: analyst

> Forward-looking backlog for the `analyst` agent.
> Status: 🔵 proposed · 🟡 in-progress · ✅ done (then moved to history.md) · ⚪ rejected/deferred
> Last reviewed: 2026-07-12

## Active

| ID | Added | Priority | Status | Summary |
|------|------------|----------|--------|---------|
| K-001 | 2026-07-09 | med | 🟡 | First-run shakedown: plan-review half evidenced (m3-executor review); code review + RCA still open — K-022 impl review is the designated vehicle |
| K-002 | 2026-07-09 | low | 🔵 | Reciprocal mentions in producer prompts (architect/coder) |

### K-001 — First-run shakedown: review a real artifact end-to-end
- **Status:** 🟡 in-progress (plan-review half evidenced 2026-07-11)
- **Priority:** medium
- **Rationale:** The prompt is untested against a live run. The likely weak spots: verdict calibration (does it rubber-stamp or nitpick-flood?), whether it actually runs suites for evidence, and whether the review doc lands at `docs/reviews/<slug>.md` with the hook staying silent.
- **Proposed change:** Delegate a plan review (e.g. an existing `falkor-chat/docs/plans/` doc), a code review of a recent diff, and an RCA of a real (or seeded) failing test; assess deliverable quality against the prompt's own structures; fold findings back into the prompt.
- **Progress (2026-07-12):** the **plan-review** half ran for real — `falkor-chat/docs/reviews/m3-executor.md` (K-022 design review; majors M1–M4 raised and closed into the approved plan; doc landed at the right path, hook silent). Still open: a **code review of a real diff** — designated vehicle: the **K-022 executor implementation review** (`docs/reviews/m3-executor-impl.md`), now a baked-in done-condition of that item in `falkor-chat/docs/BACKLOG.md` (see teco K-003) — and an **RCA** run. Fold verdict-calibration findings back into the prompt afterwards.

### K-002 — Reciprocal mentions in producer prompts (architect/coder)
- **Status:** 🔵 proposed
- **Priority:** low
- **Rationale:** The analyst names the owners it routes findings to (coder/tdd-engineer/architect/qa-engineer), but no producer prompt mentions the analyst as an available review gate — teco's roster is currently the only router. Fine while teco mediates everything; worth revisiting if plans/code should advertise "reviewable by analyst" themselves.
- **Proposed change:** If review gates become a standing part of the pipeline, add a one-line mention in architect's handoff section (plan may be routed through analyst) — keep it minimal to avoid roster sprawl in specialist prompts.

## Parking lot / ideas
- A severity rubric calibrated on real reviews (examples of blocker vs major from this repo) once a few reviews exist — only if verdicts prove inconsistent.
- Re-review mode: given a prior review doc + a revised artifact, verify each finding was addressed and append a dated re-review section instead of writing a fresh doc.
