# Kaizen — Improvement Plan: cobb

> Forward-looking backlog for the `cobb` agent.
> Status: 🔵 proposed · 🟡 in-progress · ✅ done (then moved to history.md) · ⚪ rejected/deferred
> Last reviewed: 2026-07-09

## Active

| ID | Added | Priority | Status | Summary |
|-------|------------|----------|--------|---------|
| K-001 | 2026-05-31 | high | 🔵 | Periodically re-verify the documented standards against live official docs (these ecosystems change fast). |
| K-002 | 2026-05-31 | medium | 🔵 | Add a worked "port an agent across tools" reference example (Claude subagent ↔ OpenCode agent ↔ Kiro steering) — now skill material, candidate for the `agent-maintenance` bundle. |
| K-003 | 2026-05-31 | low | 🔵 | Track additional agentic tools as they mature (e.g. Codex CLI, Cursor, Gemini CLI) where they share the open AGENTS.md / Agent Skills standards. |
| K-005 | 2026-06-07 | high | 🔵 | Automate doc-drift detection: a scheduled routine that re-fetches the canonical docs, diffs vs. stored snapshots, and files a kaizen item on change. |
| K-008 | 2026-06-07 | low | 🔵 | Dog-food the frontmatter cobb teaches: evaluate adding `memory: project` for a persistent cross-session drift/verified-date store (distinct from kaizen). |
| K-009 | 2026-06-20 | medium | 🔵 | Add a CI/script guard that every component `AGENTS.md` has a sibling `CLAUDE.md` = `@AGENTS.md` stub (so Claude Code never silently misses context — it reads `CLAUDE.md`, not `AGENTS.md`). Fold into the K-005 drift job. *(Sibling shipped 2026-07-09: `claude/scripts/audit-team.sh` covers the agent-collection invariants — the `@AGENTS.md`-stub check could join it.)* |

> **Closed:** K-004 (audit/reconcile method) — documented in the `agent-maintenance` skill (§3), done 2026-06-07, see history. K-006 (slim the prompt) — done 2026-06-07, see history. K-007 (extract standards → reference skill) — done 2026-06-07, the `agent-standards` skill, see history. K-010 (cobb subagent-awareness clause) + K-011 (destructive-ops guard parity: shared core + graph-dba/qa-engineer wrappers) — done 2026-07-11, see history.

### K-001 — Re-verify standards against live docs
- **Status:** 🔵 proposed
- **Priority:** high
- **Rationale:** Frontmatter fields, directory paths, and inclusion modes for Claude Code, Kiro, and OpenCode shift between releases. Stale specifics would make Cobb produce broken artifacts.
- **Proposed change:** On a cadence (or whenever a user reports a mismatch), fetch kiro.dev/docs, opencode.ai/docs, code.claude.com/docs, platform.claude.com/docs and reconcile the "Standards you know cold" section. Log diffs in history.md.
- **Notes:** Baseline verified 2026-05-31 at creation. Subagent context-loading + frontmatter re-verified 2026-06-07 against code.claude.com/docs/en/sub-agents. **Kiro + OpenCode re-verified 2026-06-07** (kiro.dev/docs/steering, opencode.ai/docs/agents+rules) during the K-007 skill extraction — caught real OpenCode drift (`mode: all` default, new `disable`/`color`/`top_p`/`steps` fields, granular permission keys, AGENTS.md precedence). All current specifics now live in the `agent-standards` skill with per-file `Verified:` stamps. **2026-06-20:** Claude Code **subagents** re-verified against code.claude.com/docs/en/sub-agents (tool inheritance + withheld-tools list, expanded frontmatter, discovery/scopes, agent teams + background agents) — claude-code.md stamp bumped to 2026-06-20. Claude Code **Skills/Memory/Hooks/MCP/SDK** still on the 2026-05-31 baseline — next refresh target. **Kiro** docs re-read 2026-06-20 (agents/subagents, steering, specs, hooks). NB: the old "does `inclusion: always` steering reach a subagent" dispute is **not** resolved by a doc re-read — docs affirm it, but it's field-disputed and needs a runtime test on the install. Specs/Hooks confirmed *not* reaching subagents. **OpenCode** re-verified 2026-06-20 (agents/permissions/rules — caught the `tools`→`permission` deprecation; subagent nesting documented; flagged the parent-context divergence vs Claude). **All three tools' agent/subagent surfaces now current as of 2026-06-20.** Remaining stale: only Claude Code Skills/Memory/Hooks/MCP/SDK (2026-05-31).

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

### K-008 — Dog-food the frontmatter cobb teaches
- **Status:** 🔵 proposed
- **Priority:** low
- **Rationale:** Cobb runs on `name`/`description`/`model` only, yet teaches a rich field set (`memory`, `disallowedTools`, `permissionMode`, `skills`, `isolation`, `effort`). A `memory: project` store (auto-injected `MEMORY.md`) would give cobb persistent cross-session knowledge of drift findings / verified-dates / gotchas — distinct from kaizen (human change-log, not auto-injected into the prompt).
- **Proposed change:** Evaluate adding `memory: project`. Leave the `agent-maintenance` skill on-demand (do NOT pin via `skills:` — pinning defeats leanness; the on-demand choice is deliberate).
- **Notes:** Surfaced 2026-06-07 self-review.

## Parking lot / ideas
- Consider a lightweight self-review checklist Cobb runs before delivering any agent artifact (frontmatter valid, description routing-friendly, right mechanism chosen, perishable facts dated). *(Re-flagged 2026-06-07 — candidate for promotion to an active item or a short resident checklist.)*
- Maintain a small catalog of agents/skills Cobb has authored, cross-linking their kaizen files.
