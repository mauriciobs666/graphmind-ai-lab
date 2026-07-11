# Kaizen — Improvement Plan: qa-engineer

> Forward-looking backlog for the `qa-engineer` agent.
> Status: 🔵 proposed · 🟡 in-progress · ✅ done (then moved to history.md) · ⚪ rejected/deferred
> Last reviewed: 2026-07-01

## Active

| ID | Added | Priority | Status | Summary |
|------|------------|----------|--------|---------|
| K-001 | 2026-07-01 | med | 🔵 | Ship a reusable test-plan + test-report markdown template pair (as skill or in-repo doc) so structure is consistent across runs |
| K-002 | 2026-07-01 | med | ✅ | Define the `docs/test-plans/` + `docs/test-reports/` convention explicitly in `falkor-chat/AGENTS.md` (currently inferred by the agent) — done 2026-07-11 via the docs-unification pass (see history) |
| K-003 | 2026-07-01 | low | 🔵 | Consider a handoff protocol: qa-engineer files defects → coder/tdd-engineer fix → qa-engineer re-runs (regression loop) |
| K-004 | 2026-07-01 | low | 🔵 | Capture a first-run smoke-eval as a repeatable check; document the "new subagent isn't routable until a new session" registry-reload gotcha where users will see it |

### K-001 — Reusable plan/report templates
- **Status:** 🔵 proposed
- **Priority:** medium
- **Rationale:** the prompt describes the plan/report structure prose-only; a concrete template (skill or doc) would make output consistent and speed each run.
- **Proposed change:** author a small `qa-templates` skill (or a `docs/_templates/` pair) with the test-plan and test-report skeletons the agent fills in.
- **Notes:** keep it lean; progressive-disclosure skill is the natural home if it grows.

### K-002 — Pin the artifact-location convention in component docs
- **Status:** ✅ done 2026-07-11 — the repo-wide docs unification defined the module documentation convention (active `docs/test-plans/`+`docs/test-reports/` vs. frozen `docs/archive/`) in the root `AGENTS.md` and the `falkor-chat/AGENTS.md` key-docs table; the agent's PLAN bullet was updated to match (see history 2026-07-11).
- **Priority:** medium
- **Rationale:** the agent currently *detects* where to write plans/reports. Writing the convention into `falkor-chat/AGENTS.md` (and other components as they gain QA needs) removes ambiguity and drift.
- **Proposed change:** add a short "Test plans & reports live in `docs/test-plans/` and `docs/test-reports/`, kebab-case per feature" note to the relevant component `AGENTS.md`.

### K-003 — Defect → fix → re-run handoff
- **Status:** 🔵 proposed
- **Priority:** low
- **Rationale:** QA is most valuable in a loop with implementation. A light protocol (report format that `coder`/`tdd-engineer` consume, plus a re-run pass) closes it.
- **Notes:** `teco` could orchestrate; verify subagent-to-subagent handoff patterns before hardwiring. **Update 2026-07-09:** the teco side is now in teco's prompt (roster entry with path-handoff conventions + defect→re-brief→re-run in its integrate-&-verify step); remains open pending a live orchestrated cycle.

### K-004 — First-run smoke-eval + document the registry-reload gotcha
- **Status:** 🔵 proposed
- **Priority:** low
- **Rationale:** the first-spin (2026-07-01) confirmed the agent works but had to be **proxy-run** because a freshly-created subagent isn't in the session's registry until a new session starts. Users will hit this; it belongs in the deploy/testing notes, not tribal memory.
- **Proposed change:** add a one-line "restart the session to route to a newly added agent" note to `claude/README.md` deployment section (or `cobb/TESTING.md`), and keep the M1 pass as a lightweight smoke reference.

## Parking lot / ideas
- Optional non-functional playbooks (perf smoke via `GRAPH.PROFILE`, basic security/permission probes) as an on-demand skill rather than resident prompt weight.
- A `qa-engineer` ↔ `saul`/`dra-claudia`-style workdir option if the user later wants reports kept out of version control (currently in-repo `docs/` per user's choice).
