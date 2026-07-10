# Kaizen — Change History: analyst

> Dated log of actual changes to the `analyst` agent. Most recent first.

## 2026-07-09 — data-scientist route-away clause (boundary symmetry)
- **What:** Frontmatter `description` and the findings-routing guardrail now route the AI/ML/data-science **methodology** dimension of a plan or change — model/embedding choice, evaluation design, metric validity, statistical claims — to the new `data-scientist` agent, whose methodology review (`docs/reviews/<slug>-ml.md`, same verdict scale) complements the analyst's general static review. Pair `analyst:data-scientist` added to `claude/scripts/audit-team.sh` `BOUNDARY_PAIRS` (check 6, description symmetry).
- **Why:** The `data-scientist` agent was created 2026-07-09 to work alongside the analyst at review time; "review this ML-heavy change" plausibly matched both, so the boundary must live in both descriptions.
- **Plan items:** none.

## 2026-07-09 — Description gained the qa-engineer route-away clause (boundary symmetry)
- **What:** Frontmatter `description` now states the verification boundary explicitly: analyst judges statically — reading, reasoning, and running what already exists — and planning/executing *new* black-box/acceptance testing of the running system routes to `qa-engineer`. The prompt body already carried this split (findings-routing guardrail); the description — the routing contract every router sees — didn't. Counterpart clause added to qa-engineer in the same change; the pair is now mechanically enforced by `claude/scripts/audit-team.sh` check 6 (boundary-pair description symmetry).
- **Why:** Description-symmetry sweep after teco's roster→routing-table restructure (same day): analyst↔qa-engineer was asymmetric at the description level (analyst never named qa-engineer), leaving "test this" work plausibly matching both.
- **Plan items:** none.

## 2026-07-09 — Added root cause analysis (RCA) mode
- **What:** Extended the reviewer into a reviewer-and-diagnostician: a third artifact class ("Defects and failures — RCA") with its own method (reproduce when possible, trace the actual code path, read git history; distinguish root cause vs trigger vs contributing factors; five-whys stops at the deepest cause actionable in the codebase; record ruled-out hypotheses) and its own deliverable skeleton at `docs/reviews/<slug>-rca.md` (symptom & impact → reproduction/evidence → causal chain → root cause with confirmed/inferred confidence → suggested fix + prevention). Frontmatter description updated; guardrail clarified (diagnoses only — the fix routes to the implementer, typically `tdd-engineer` with a reproduction test first, briefed by the RCA path). No hook change needed (`docs/reviews/` already covers the RCA doc). Rosters/catalogs synced (teco, claude/AGENTS.md, claude/README.md, root AGENTS.md).
- **Why:** User: "analyst is also good with RCA" — the team had no owner for cause-unknown defects; tdd-engineer starts from a known bug, qa-engineer finds and reports defects, but nobody's job was tracing a symptom to its root cause.
- **Plan items:** none (K-001 shakedown should now cover an RCA run too).

## 2026-07-09 — Created
- **What:** Initial version of the `analyst` subagent — a systematic, experienced developer acting as a pure reviewer: reviews architect plans (grounding, completeness, soundness, proportionality, test strategy) and source code (correctness → tests → fit → clarity → security/perf, in priority order), plus plan↔code conformance when given both. Deliverable is a severity-ranked (blocker/major/minor/nit), evidence-backed review with a verdict (approve / approve with suggestions / needs changes), written to `<component>/docs/reviews/<slug>.md` by default and handed off by path. Review-only contract is harness-enforced: `hooks/guard-review-doc-writes.sh` (PreToolUse, matcher `Write|Edit`, same pattern as architect's guard) escalates any Write/Edit outside `docs/reviews/` (or `/tmp`) to the human. Subagent-aware (questions/blockers return as the deliverable). Model `opus`, tools `Read, Grep, Glob, Bash, Write, Edit, WebFetch, WebSearch, Agent` — mirrors architect. Deployed via `~/.claude/agents/analyst` symlink.
- **Why:** The team had no review gate between handoffs — architect plans went straight to implementation and implementer code straight to QA, with nobody judging design soundness or code quality statically. User requested a systematic reviewer covering both plans and source code.
- **Plan items:** — (K-001, K-002 seeded)
