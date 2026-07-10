# Kaizen — Change History: qa-engineer

> Dated log of actual changes to the `qa-engineer` agent. Most recent first.

## 2026-07-09 — analyst boundary clause (description + intro)
- **What:** Frontmatter `description` and the intro's deferral paragraph now route *static* judgment — reviewing a plan, diff, or module by reading and reasoning, without executing the system — to `analyst`, mirroring analyst's new clause routing new black-box/acceptance execution here. The pair is mechanically enforced by `claude/scripts/audit-team.sh` check 6 (boundary-pair description symmetry). Catalogs synced (`claude/AGENTS.md`, `claude/README.md`, root `AGENTS.md`).
- **Why:** Description-symmetry sweep after teco's roster→routing-table restructure (same day): qa-engineer named tdd-engineer but not analyst, leaving the static-review vs. executed-verification boundary invisible to routers.
- **Plan items:** none.

## 2026-07-09 — Subagent-awareness lines (teco interface review)
- **What:** Three clauses added during the teco interface review: workflow step 1's "ask one sharp question", the EXECUTE-phase "ask before installing or mutating the environment" bullet, and the never-mutate-the-environment guardrail now all say what to do when running as a subagent (e.g. delegated by teco) — return the sharp question / approval request as the result (marking affected items blocked) instead of trying to ask mid-run, which subagents can't do. Catalog entry (`claude/AGENTS.md`) updated. In the same change, **teco itself gained the K-003 loop**: its roster now includes qa-engineer (with the `docs/test-plans/` / `docs/test-reports/` path-handoff conventions), its pipeline ends in a QA pass when warranted, and its integrate-&-verify step encodes defect → re-brief implementer with the report path → re-run failed items.
- **Why:** The agent's "ask" phrasing assumed an interactive session; under teco delegation that would stall or misfire. The teco-side change closes the orchestration half K-003 anticipated.
- **Plan items:** K-003's teco side is now in teco's prompt; K-003 stays open pending a live orchestrated defect→fix→re-run cycle.

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
