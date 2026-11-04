# Team engagement in other repos — Feature Requirements

> **Status:** Ready for design · **Owner:** `tico` · **Tracks:** — · **Last updated:** 2026-11-03

## Intent
Let the agent team (and the tooling built in this repo) engage a **green-field** (new) project that lives in a different repo. Legacy (existing-code) projects need a different engagement strategy and are deferred (see Out of scope).

## Problem & current state
The agents are installed user-level (`~/.claude/agents` symlinks to this repo's `claude/`), so they run in any repo on this machine. Much of the supporting machinery stays in this repo: the `cypher` MCP wiring, the skills, the root `AGENTS.md`/`REVIEW.md`, the `docs/` conventions, the shared `ws:agent-team` learnings pool. No distinction exists today between starting a new project and joining an existing one.

**Context for design (not decided here):** how the shared pool and the skills are reached from another repo without editing every agent prompt, and how a learning is attributed to its project (FR-5, FR-5a, FR-6).

## User stories
- As the stakeholder starting a new project in an empty repo, I want to open a session with `tico` and end it with a requirements doc and a minimal skeleton, so that the project starts from a clear intent.
- As the stakeholder, I want to then hand that requirements doc to `teco` and have it deliver the project through the usual pipeline, so that I don't coordinate the build myself.
- As the team maintainer, I want lessons from the new project to reach the shared pool, so that the whole team improves.

## Functional requirements
- FR-1: Starting a green-field project begins with a requirements interview; no scaffolding precedes it.
- FR-2: After the interview, the new repo is given a **minimal** convention set ("lighter by default"): at kickoff, a project rules file (`AGENTS.md`: what the project is, how to run and test it) and the docs layout with status headers.
- FR-2a: A test setup is established at the project's first build, before the first feature.
- FR-2b: A review checklist (`REVIEW.md`) and graph tooling (cypher MCP, code graph) are added on demand, not at kickoff.
- FR-3: The convention set can grow later as the project needs more, without re-scaffolding.
- FR-4: The team's pipeline (requirements, plan, build, review, docs) runs in the new repo through the same stages and specialists as in this repo.
- FR-5: Learnings captured while working in a new project feed the shared team pool, so lessons improve the team in every project.
- FR-5a: A learning from a new project can be told apart from this repo's own (project-specific vs. general), so `cobb` can judge which to promote; mechanism TBD (design).
- FR-5b: A project can be marked, by the stakeholder, as **not feeding** the shared pool; learnings from such a project are not written to it.
- FR-6: The team's skills are available in every new project automatically, with no per-project setup.

## Out of scope
- Stack-specific support beyond the stack already used in this repo; the first new project uses the same stack.
- **Legacy / existing-code projects** — a distinct engagement strategy, deferred by the stakeholder; revisit as its own requirements doc.

## Acceptance criteria
- AC-1 (FR-1, FR-2): Given an empty repo, when I run `claude --agent tico` and describe the project, then no `AGENTS.md` or `docs/` content exists until the interview has concluded, and the session ends with exactly: a requirements document under `docs/requirements/` carrying the status header; and an `AGENTS.md` giving what the project is and how to run it, with a test section stating that the test setup is created at first build. Other `docs/` folders are created when first used. No `REVIEW.md`, test runner or graph tooling exists yet.
- AC-2 (FR-2a, FR-4): Given that repo and the requirements doc, when I run `teco`, then it plans, builds, reviews and documents the project through the usual specialists, establishing the test setup before the first feature, pausing only for a decision the requirements doc does not settle.
- AC-3 (FR-5, FR-5a): Given a learning captured during AC-2, then it appears in the shared pool attributable to the new project, with no hand-wiring of the pool for that project.
- AC-5 (FR-2b, FR-3): Given a project after kickoff, then it has no `REVIEW.md` or graph tooling until I (or `teco`) request one; when requested, it is added without redoing or overwriting the kickoff files.
- AC-6 (FR-5b): Given a project marked as not feeding the pool, when an agent captures a learning during work, then nothing from that project is written to the shared pool.
- AC-4 (FR-6): Given an empty repo with no setup of its own, when an agent needs one of the team's skills, then it finds and uses it.

## Open questions
None.

## Decision log
2026-11-03 — Should engagement differ by project type? → Yes: green-field (new) and legacy need distinct strategies (scope of FRs TBD).
2026-11-03 — Cover legacy now? → No, out of scope; green-field only (Intent, Out of scope).
2026-11-03 — What should green-field get? → Full pipeline from day one; scaffolded conventions; up-front design decisions with the stakeholder; also "same as today" (FRs TBD).
2026-11-03 — What happens first on a new project? → Interview first, scaffold after (FR-1, FR-2).
2026-11-03 — Same conventions everywhere or adapted? → Lighter by default; minimal set that grows (FR-2, FR-3).
2026-11-03 — What is in the minimal set? → AGENTS.md + docs layout at kickoff; test setup at first build; REVIEW.md and graph tooling on demand (FR-2, FR-2a, FR-2b).
2026-11-03 — Should a new project's learnings share the team pool? → Yes, shared: lessons should improve the team everywhere (FR-5, FR-5a).
2026-11-03 — What does a successful kickoff look like? → Run tico in an empty repo to get requirements + skeleton, then run teco, which builds everything (user stories, AC-1, AC-2).
2026-11-03 — What is the first target project? → Nothing special; same stack as this repo, so no stack-specific requirements (Out of scope).
2026-11-03 — Should skills be available automatically in new projects? → Yes, automatically (FR-6, AC-4).
2026-11-03 — Closing note: the "FRs TBD" placeholders in the entries above are resolved by FR-1 to FR-6.
2026-11-03 — Consult: analyst on readiness review (doc `claude/docs/reviews/team-engagement-in-other-repos.md`) → approve with suggestions; testability gaps folded into FR-4, AC-1, AC-2, AC-3, AC-5.
2026-11-03 — May project-specific or private content reach the shared pool? → Shared by default, with a per-project opt-out (FR-5b, AC-6).
2026-11-03 — Kickoff docs layout: full skeleton or as needed? → Only as needed, as today (AC-1).
2026-11-03 — Readback confirmed by stakeholder; analyst verdict approve with suggestions, folded in → Status flipped to Ready for design.
