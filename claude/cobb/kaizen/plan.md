# Kaizen — Improvement Plan: cobb

> Forward-looking backlog for the `cobb` agent.
> Status: 🔵 proposed · 🟡 in-progress · ✅ done (then moved to history.md) · ⚪ rejected/deferred
> Last reviewed: 2026-06-07

## Active

| ID | Added | Priority | Status | Summary |
|-------|------------|----------|--------|---------|
| K-001 | 2026-05-31 | high | 🔵 | Periodically re-verify the documented standards against live official docs (these ecosystems change fast). |
| K-002 | 2026-05-31 | medium | 🔵 | Add a worked "port an agent across tools" reference example (Claude subagent ↔ OpenCode agent ↔ Kiro steering). |
| K-003 | 2026-05-31 | low | 🔵 | Track additional agentic tools as they mature (e.g. Codex CLI, Cursor, Gemini CLI) where they share the open AGENTS.md / Agent Skills standards. |
| K-004 | 2026-06-07 | medium | 🔵 | Capture a repeatable "audit & reconcile agent-context docs" method (drift detection via `git ls-files` vs. what the doc claims). |
| K-005 | 2026-06-07 | high | 🔵 | Automate doc-drift detection: a scheduled routine that re-fetches the canonical docs, diffs vs. stored snapshots, and files a kaizen item on change. |
| K-006 | 2026-06-07 | high | 🔵 | Slim the prompt: extract the maintenance machinery into a progressively-disclosed skill, keep a tight mandate resident, and split the doc obligation into in-scope (per-edit) vs. cross-scope (reconcile pass). |

### K-004 — Audit & reconcile context docs (drift detection)
- **Status:** 🔵 proposed
- **Priority:** medium
- **Rationale:** Distinct from the existing "sync on my own create/edit/remove" duty: a standalone pass to bring an *already-drifted* `AGENTS.md`/`CLAUDE.md`/steering doc back in line with repo reality. Surfaced 2026-06-07 updating graphmind-ai-lab's root `AGENTS.md`, which had silently fallen behind (missing the whole `falkor-chat/` component, the `graph-dba` agent, and the `severino` OpenCode agent).
- **Proposed change:** Encode the method — enumerate ground truth (`git ls-files`, per-component README/CLAUDE/SKILL headers), diff against what the doc currently lists, then add/edit/remove entries. Likely a **skill** (progressive disclosure) rather than prompt bloat, per the lean-context principle.
- **Notes:** Pairs naturally with K-002 (both candidates for a small "agent-docs" skill bundle). The DRY import-pattern note (CLAUDE.md → `@AGENTS.md`) was folded into the prompt's Documentation section on 2026-06-07.

### K-001 — Re-verify standards against live docs
- **Status:** 🔵 proposed
- **Priority:** high
- **Rationale:** Frontmatter fields, directory paths, and inclusion modes for Claude Code, Kiro, and OpenCode shift between releases. Stale specifics would make Cobb produce broken artifacts.
- **Proposed change:** On a cadence (or whenever a user reports a mismatch), fetch kiro.dev/docs, opencode.ai/docs, code.claude.com/docs, platform.claude.com/docs and reconcile the "Standards you know cold" section. Log diffs in history.md.
- **Notes:** Baseline verified 2026-05-31 at creation. Subagent context-loading + frontmatter re-verified 2026-06-07 against code.claude.com/docs/en/sub-agents (prompt updated). Other sections (Kiro, OpenCode, SDK) still on the 2026-05-31 baseline — due for a refresh.

### K-002 — Worked cross-tool porting example
- **Status:** 🔵 proposed
- **Priority:** medium
- **Rationale:** Porting between tools is a common request; a canonical example would make answers faster and more consistent.
- **Proposed change:** Add a reference walkthrough mapping a Claude Code subagent's frontmatter/body to an OpenCode markdown agent and to Kiro steering, noting what each tool drops or renames.
- **Notes:** Could live as a skill rather than bloat the agent prompt.

### K-003 — Broaden tool coverage
- **Status:** 🔵 proposed
- **Priority:** low
- **Rationale:** The open AGENTS.md and Agent Skills standards are adopted by more tools than the core three.
- **Proposed change:** Add concise coverage of Codex CLI, Cursor, Gemini CLI, Copilot where they intersect the open standards, clearly flagged as secondary.
- **Notes:** Keep the big-three depth primary; don't dilute.

### K-005 — Automate doc-drift detection
- **Status:** 🔵 proposed
- **Priority:** high
- **Rationale:** K-001 (manual re-verify) relies on someone remembering. A frozen prompt silently rots between checks. Determinism beats hope: a harness-run job that diffs the official docs and surfaces changes is the real safeguard.
- **Proposed change:** A scheduled agent (Claude Code `/schedule` cron routine, or local cron) that, per tool, fetches the canonical pages (code.claude.com/docs, opencode.ai/docs, kiro.dev/docs, platform.claude.com/docs), diffs the relevant sections against a stored `sources/` snapshot (last-verified date + section excerpt/hash), and on change appends an item to this plan + pings the user. Keeps perishable specifics out of the always-on prompt and re-checked on a cadence.
- **Notes:** Surfaced 2026-06-07 — user asked "how do we ensure the info won't drift?" Pairs with the new "Drift-resistance" principle (timestamp + verify volatile facts). Offered to build it; awaiting go-ahead.

### K-006 — Slim the prompt; extract maintenance machinery into a skill
- **Status:** 🔵 proposed
- **Priority:** high
- **Rationale:** ~95 of 159 prompt lines are reference *manual* — the kaizen templates, the file-location decision tree, the documentation-audience table — sitting in the always-loaded prompt and firing on every turn, including pure Q&A. That violates Cobb's own "lean context / push detail into progressively-disclosed skills" principle. Surfaced 2026-06-07 in a blunt self-review. Two further findings from that thread: (1) inline `OBLIGATORY` text is not enforcement — it's hopeful prompt text that already drifted, so moving the *manual* to a skill costs nothing on the compliance axis while saving context weight; the *mandate* stays resident. (2) The documentation duty conflates two different obligations with different correct mechanisms.
- **Proposed change:**
  - Extract an `agent-maintenance` skill holding the kaizen templates + procedure, the dual-audience documentation method, the file-location rules, and the K-004 audit/reconcile method. Fold the orphaned `TESTING.md` in as a supporting file (the prompt currently has no pointer to it, so a fresh session can't reach it).
  - Replace the two `OBLIGATORY` sections with a tight (~8-line) "Maintenance duties" block that states the obligations and points at the skill.
  - **Split the doc obligation:** *in-scope* (Cobb edits an agent → update its kaizen + catalog) stays a resident per-edit rule; *cross-scope* (repo-root catalog reflecting all components/agents) is reframed from a per-edit push to an on-demand **reconcile pass** — no scoped session owns the global catalog (the falkor-chat drift was scope, not negligence: that session never saw the parent catalog). Promote K-004 from idea to the documented method for this.
- **Notes:** Pairs with K-002 + K-004 as the "agent-docs / agent-maintenance" skill bundle. Distinct from K-005 (which automates *knowledge*-doc drift); K-006 is about *prompt structure* and the *maintenance process*. A nudge hook (PostToolUse on `claude/*/` source changes) was considered and deferred — it conscripts scoped sessions into global bookkeeping; the reconcile pull already works. Honesty follow-up (own knowledge): refresh the stale Kiro/OpenCode/SDK sections (still on the 2026-05-31 baseline per K-001) and add per-section "verified: YYYY-MM-DD" stamps so staleness is visible.

## Parking lot / ideas
- Consider a lightweight self-review checklist Cobb runs before delivering any agent artifact (frontmatter valid, description routing-friendly, right mechanism chosen).
- Maintain a small catalog of agents/skills Cobb has authored, cross-linking their kaizen files.
