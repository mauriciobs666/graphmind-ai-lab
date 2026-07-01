# Kaizen — Change History: qa-engineer

> Dated log of actual changes to the `qa-engineer` agent. Most recent first.

## 2026-07-01 — true delegated run confirmed (auto-routing works)
- **What:** after the session reloaded its subagent registry, invoked `qa-engineer` for real via the `Agent`/Task tool (`subagent_type: qa-engineer`) on a focused follow-up pass against falkor-chat M1 (residual gaps: room-wide `read_messages`, DEF-1 regression). The subagent ran its own playbook end-to-end and **appended** to the existing plan + report (didn't overwrite): TP-026 + TP-027 both PASS, baseline 57/57, DEF-1 still reproduces.
- **Why:** close the loop on the K-004 registry-reload gotcha — prove the agent is routable and behaves correctly under genuine delegation, not just as a cobb proxy.
- **Result:** ✅ auto-routing works; the agent honored the self-contained brief (subagents don't share context), respected the append-don't-overwrite instruction, obeyed the environment pre-authorization, started/stopped the server itself, and left the environment clean. Confirms the K-004 gotcha is purely a session-start registry-load timing issue.
- **Docs touched:** falkor-chat test-plan + report (appended by the subagent).

## 2026-07-01 — first spin (proxy-run) against falkor-chat M1
- **What:** exercised the agent's four-phase playbook end-to-end on the falkor-chat M1 server (REST + MCP). Produced `falkor-chat/docs/test-plans/m1-chat-mcp.md` and `.../test-reports/m1-chat-mcp-report.md`. Result: 22/22 functional+contract items passed on a 57/57 baseline; found DEF-1 (MCP endpoint 405s at `/mcp`, only `/mcp/` works — README/DESIGN mismatch).
- **Why:** validate the new agent's methodology yields a usable strategy → plan → execute → report cycle.
- **Run mode:** **proxy** — run by cobb following the qa-engineer prompt, NOT via Task delegation. Reason: Claude Code loads the subagent registry at **session start**, so the freshly-symlinked `qa-engineer` was not yet routable in the session that created it (`Agent(subagent_type='qa-engineer')` → "agent type not found"). Expected behavior; a new session picks it up.
- **Playbook validation (what worked):** the "verify before asserting" rule caught a wrong hypothesis (assumed `ServiceError`→500 because `api.py` lacks handlers; actually `app.py` maps them 404/400). Evidence-over-assertion produced a clean, reproducible defect. Doc-convention detection (`docs/test-plans/` + `docs/test-reports/`, kebab per feature) worked. Environment-approval guardrail behaved (needed cobb's explicit pre-authorization to touch the shared DB).
- **Docs touched:** falkor-chat test-plan + report (new); `falkor-chat/kaizen/history.md` note.
- **Plan items:** validated K-001 need (templates would have sped the plan/report authoring); added K-004 (first-run smoke-eval + document the registry-reload gotcha in the agent README/testing notes).

## 2026-07-01 — created
- **What:** authored the `qa-engineer` subagent — a QA / functional-testing specialist that (1) reasons about risk to build a test strategy, (2) writes it to a versioned test plan following the component's doc conventions (`docs/test-plans/<kebab>.md`), (3) executes it by authoring automated functional/acceptance tests, running existing suites, AND driving the running app black-box, and (4) writes a test report (`docs/test-reports/<kebab>-report.md`) with results, defects, and feedback. `model: opus`, inherits all tools (needs Write/Edit/Bash to author tests, run suites, and drive apps).
- **Why:** user asked for a functional-testing agent that reasons → plans → executes → reports. Fills the behavior/acceptance-altitude gap next to `tdd-engineer` (unit, test-first) and `coder` (implementation).
- **Design decisions (user-confirmed):** execution mode = "both — author, run, and drive"; artifact location = per-component `docs/` dirs (detect each component's convention). Name `qa-engineer` chosen by cobb (user went idle on the name question) to match the role-named technical specialists (`tdd-engineer`, `graph-dba`, `coder`, `architect`).
- **Boundaries drawn:** does NOT fix code under test unless asked (defers to coder/tdd-engineer); never mutates the shared FalkorDB environment without approval; evidence-over-assertion; extends past the unit layer rather than duplicating it.
- **Docs updated:** `claude/README.md` (catalog + kaizen list), `claude/AGENTS.md` (agent context), root `AGENTS.md` (repo catalog). Deployed via `~/.claude/agents/qa-engineer` → `claude/qa-engineer` symlink.
- **Plan items:** seeded K-001 (reusable plan/report templates), K-002 (pin artifact-location convention in component AGENTS.md), K-003 (defect→fix→re-run handoff).
