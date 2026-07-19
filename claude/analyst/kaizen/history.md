# Kaizen — Change History: analyst

> Dated log of actual changes to the `analyst` agent. Most recent first.

## 2026-07-19 — CPG capability wired into the routing description (M2 / C-207)
- **What:** Frontmatter `description` gained one clause: when a Joern CPG is loaded in FalkorDB, the analyst uses the `cpg-analysis` skill (graph-dba-owned) for impact-analysis, RCA data-flow, and code-review taint queries instead of reading files. `claude/README.md` catalog entry updated to match. No body change — the skill is progressively disclosed and self-describes; the description clause is the routing signal.
- **Why:** M2 delivered the `cpg-analysis` skill (`analyst` is a named consumer for impact/RCA/code-review recipes per FR-10/11/12). C-207 makes the consumer agents' routing contract advertise the capability. cobb wired it as part of Gate-2b (skill also passed the standards vet).
- **Plan items:** none.

## 2026-07-12 — K-001: code-review half of the shakedown proven (K-022 impl review) — RCA remains
- **What:** The **code-review** half of the first-run shakedown ran for real. On falkor-chat
  **K-022 Landing 1** (executor implementation, committed `3921f87`) the analyst reviewed the
  delivered diff and produced `falkor-chat/docs/reviews/m3-executor-impl.md`: verdict
  **approve-with-suggestions, 0 blockers / 1 major (M-1) / 3 minor / 3 nit**, doc landed at the
  right path with the write-guard hook silent. This was the designated vehicle named in K-001 and
  the counterpart to teco K-003 (the team's first fully-gated run). Verdict calibration read as
  healthy — a real major surfaced, not a nitpick flood, and the two deferred seams were ruled
  acceptable-for-Landing-1 rather than inflated to blockers.
- **Why:** teco K-003 closed 2026-07-12 with the gated run committed; that same run is the
  evidence for analyst K-001's code-review half. Recording it here so the shakedown's remaining
  scope is honest.
- **Plan items:** **K-001 narrowed** (not closed): plan-review ✅ (2026-07-11) + code-review ✅
  (this entry); the **RCA** mode is still unexercised — K-001 now tracks that remainder only.
  No prompt change: no verdict-calibration weakness surfaced across the two review runs.

## 2026-07-12 — Learning-capture loop: kaizen inbox + closing protocol + guard allowlist
- **What:** Added `kaizen/inbox.md` (append-only learnings inbox, seeded empty) and a "Learning capture" closing-protocol section to the prompt; the doc-scoped write guard's allowlist gained exactly the agent's own inbox path (`<name>/kaizen/inbox.md`), with the escalation message updated to match.
- **Why:** Team-wide self-improvement loop (agent-maintenance skill §5, added the same day): capture is cheap and unreviewed during runs, promotion is curated — cobb periodically verifies each entry and routes it to the prompt, an on-demand knowledge base, or project docs. Requested by the user.
- **Plan items:** none.

## 2026-07-11 — Description slimmed (team-wide token-cost pass)
- **What:** Frontmatter `description` compressed from 1449 to 525 chars: capability lists tightened, reciprocal boundary prose reduced to short route-away clauses that still name the counterpart agents (audit check 6 boundary symmetry preserved — full pass green), and "how I work" detail dropped from the description since the prompt body already carries it. Routing semantics unchanged; no body/catalog changes needed.
- **Why:** All 12 agents' descriptions are auto-injected into every session and into every subagent spawn that carries the `Agent` tool; team-wide they cost 12,609 chars (~3.1K tokens) per injection. The pass cut them to 7,036 chars (~44%), saving ≈1,400 tokens per session/spawn with the same routing contract.
- **Plan items:** none.

## 2026-07-11 — Guard hook refactored to a thin wrapper over a shared core
- **What:** `guard-review-doc-writes.sh` was reduced from a ~60-line standalone script to a thin wrapper that `exec`s the new shared core `claude/scripts/guard-doc-writes.sh` with two parameters — this agent's allowed-path globs (`docs/reviews/*|*/docs/reviews/*`) and its escalation-message template (`__PATH__` placeholder for the offending path). The core carries the shared machinery unchanged: jq→python3 path extraction, fail-open on unparseable input, `/tmp/*` always allowed, `permissionDecision: "ask"` JSON emit. The wrapper resolves the core via `readlink -f "$0"`, so it works when invoked through the `~/.claude/agents/<name>` deployment symlink; the frontmatter hook command is unchanged. Verified: `bash -n`, allowed/denied/scratchpad/fail-open cases through the symlink path, the no-jq python3 fallback, and `claude/scripts/audit-team.sh` all pass.
- **Why:** a repo redundancy audit (2026-07-11) found the five doc-scoped guards (analyst, architect, data-scientist, teco, tico) byte-identical except one `case` glob and one message string — ~250 duplicated lines that had to be patched five times per fix. One parameterized core removes the drift risk. (`devops/hooks/guard-destructive-ops.sh` stays standalone — it matches Bash command patterns, not write paths.)
- **Plan items:** none.

## 2026-07-10 — Hook command made machine-independent (`$HOME` symlink path)
- **What:** the frontmatter `PreToolUse` hook command was rewired from the absolute repo path (`/home/<user>/prg/graphmind-ai-lab/claude/analyst/hooks/guard-review-doc-writes.sh`) to `$HOME/.claude/agents/analyst/hooks/guard-review-doc-writes.sh`, which resolves through the user-scope deployment symlink (`~/.claude/agents/analyst` → the repo folder). Shell-form hook commands (no `args`) run via `sh -c`, so `$HOME` expands — verified 2026-07-10 against `code.claude.com/docs/en/hooks`. Resolution through the symlink confirmed (`test -x` passes).
- **Why:** the committed agent source leaked the user's personal home path into the repo; the symlink path is identical on any machine that follows the deployment convention (`~/.claude/agents/<name>` → `claude/<name>`), keeping the hook enforceable without machine-specific paths. (`${CLAUDE_PROJECT_DIR}` was rejected: the agents are user-scoped and must guard in any project, where the project dir isn't this repo.)
- **Plan items:** none.

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
