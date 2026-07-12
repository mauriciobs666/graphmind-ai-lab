# Kaizen — Improvement Plan: analyst

> Forward-looking backlog for the `analyst` agent.
> Status: 🔵 proposed · 🟡 in-progress · ✅ done (then moved to history.md) · ⚪ rejected/deferred
> Last reviewed: 2026-07-12

## Active

| ID | Added | Priority | Status | Summary |
|------|------------|----------|--------|---------|
| K-001 | 2026-07-09 | med | 🟡 | Shakedown — RCA mode only remaining: plan-review ✅ (2026-07-11) + code-review ✅ (K-022 impl review, 2026-07-12); RCA run still open |
| K-002 | 2026-07-09 | low | 🔵 | Reciprocal mentions in producer prompts (architect/coder) |

### K-001 — First-run shakedown: RCA mode remaining
- **Status:** 🟡 in-progress — plan-review ✅ + code-review ✅; **RCA mode only remaining**
- **Priority:** medium
- **Rationale:** The prompt is untested against a live run. The likely weak spots: verdict calibration (does it rubber-stamp or nitpick-flood?), whether it actually runs suites for evidence, and whether the review doc lands at `docs/reviews/<slug>.md` with the hook staying silent. Two of the three review modes have now cleared these on real artifacts; the RCA mode has not run.
- **Proposed change:** Run an **RCA of a real (or seeded) failing test** end-to-end — assess whether it delivers a clean causal chain + suggested fix at `docs/reviews/<slug>-rca.md`, hook silent; fold any verdict/structure findings back into the prompt. Then close K-001.
- **Progress:**
  - **Plan-review ✅ 2026-07-11** — `falkor-chat/docs/reviews/m3-executor.md` (K-022 design review; majors M1–M4 raised and closed into the approved plan; right path, hook silent).
  - **Code-review ✅ 2026-07-12** — `falkor-chat/docs/reviews/m3-executor-impl.md` (K-022 impl review; approve-with-suggestions, 0 blockers / 1 major / 3 minor / 3 nit; calibration healthy). Counterpart to teco K-003 (now closed). See history.md.
  - **RCA ⬜ open** — no RCA run yet; this is the sole remaining piece of the shakedown.

### K-002 — Reciprocal mentions in producer prompts (architect/coder)
- **Status:** 🔵 proposed
- **Priority:** low
- **Rationale:** The analyst names the owners it routes findings to (coder/tdd-engineer/architect/qa-engineer), but no producer prompt mentions the analyst as an available review gate — teco's roster is currently the only router. Fine while teco mediates everything; worth revisiting if plans/code should advertise "reviewable by analyst" themselves.
- **Proposed change:** If review gates become a standing part of the pipeline, add a one-line mention in architect's handoff section (plan may be routed through analyst) — keep it minimal to avoid roster sprawl in specialist prompts.

## Parking lot / ideas
- A severity rubric calibrated on real reviews (examples of blocker vs major from this repo) once a few reviews exist — only if verdicts prove inconsistent.
- Re-review mode: given a prior review doc + a revised artifact, verify each finding was addressed and append a dated re-review section instead of writing a fresh doc.
