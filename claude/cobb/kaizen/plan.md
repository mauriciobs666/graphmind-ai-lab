# Kaizen — Improvement Plan: cobb

> Forward-looking backlog for the `cobb` agent.
> Status: 🔵 proposed · 🟡 in-progress · ✅ done (then moved to history.md) · ⚪ rejected/deferred
> Last reviewed: 2026-07-16

## Active

| ID | Added | Priority | Status | Summary |
|-------|------------|----------|--------|---------|
| K-001 | 2026-05-31 | high | 🔵 | Periodically re-verify the documented standards against live official docs (these ecosystems change fast). |
| K-002 | 2026-05-31 | medium | 🔵 | Add a worked "port an agent across tools" reference example (Claude subagent ↔ OpenCode agent ↔ Kiro steering) — now skill material, candidate for the `agent-maintenance` bundle. |
| K-003 | 2026-05-31 | low | 🔵 | Track additional agentic tools as they mature (e.g. Codex CLI, Cursor, Gemini CLI) where they share the open AGENTS.md / Agent Skills standards. |
| K-005 | 2026-06-07 | high | 🔵 | Automate doc-drift detection: a scheduled routine that re-fetches the canonical docs, diffs vs. stored snapshots, and files a kaizen item on change. |
| K-008 | 2026-06-07 | low | 🔵 | Dog-food the frontmatter cobb teaches: evaluate adding `memory: project` for a persistent cross-session drift/verified-date store (distinct from kaizen). |
| K-014 | 2026-07-25 | medium | 🔵 | Skills outside cobb's two have **no kaizen home**: `skills/` carries no `kaizen/` dirs and `skills/README.md` routes only `agent-maintenance`/`agent-standards` changes to cobb's history — so edits to `cpg-analysis`, `joern-cpg`, etc. land with no per-artifact log. Decide the convention (owner-agent's history vs. `skills/<name>/kaizen/`) and write it into `skills/README.md` + `claude/AGENTS.md`. **Update 2026-08-09:** a third data point — `python-web-quirks` (new skill, no natural single "owner" persona) was logged in `cobb`'s own history by the same by-example precedent as `agent-maintenance`/`agent-standards`. Three skills now, three different "owner" shapes (cobb's own machinery; `graph-dba`-driven; nobody's), all landing in an owning agent's history rather than a per-skill `kaizen/`. That's converging evidence for "owner-agent's history" as the actual convention — still not written down anywhere a new skill author would find it. |
| K-009 | 2026-06-20 | medium | 🔵 | Add a CI/script guard that every component `AGENTS.md` has a sibling `CLAUDE.md` = `@AGENTS.md` stub (so Claude Code never silently misses context — it reads `CLAUDE.md`, not `AGENTS.md`). Fold into the K-005 drift job. *(Sibling shipped 2026-07-09: `claude/scripts/audit-team.sh` covers the agent-collection invariants — the `@AGENTS.md`-stub check could join it.)* |

## Parking lot / ideas
- From the 2026-08-09 self-review during the C-308/C-312/C-319 skill-review pass
  (`docs/reviews/cpg-followups-skills-impl.md`): the C-319 promotion compressed two independently
  true, parallel "cwd-independent" facts (`.mcp.json` discovery walk-up; `${CLAUDE_PROJECT_DIR}`
  expansion) into one causal clause ("stays uniform via ...") that the source evidence never
  established. General lesson for future inbox distillations (§5): when promoting a fact that
  echoes or sits next to another fact in the doc, keep them stated as separate claims unless the
  *mechanism link* between them was itself verified — don't add connective "via"/"because" prose
  as free editorial polish. Consider adding a line to §5's procedure about this specific failure
  mode (causal-compression during promotion) if it recurs.
- From the 2026-08-09 independent review of `analyst`'s inbox-distillation pass
  (`docs/reviews/analyst-inbox-distillation.md`): consider a sub-list format for `analyst.md`'s
  "Evidence over vibes" Guardrails bullet (now 5 sub-rules in one run-on sentence after four clause
  extensions) to restore scannability without adding to the bullet count. Also: `claude-code.md`'s
  top-of-file `Verified:` stamp block could gain a one-line pointer to the new "Bash tool
  environment" section (observed-not-doc-sourced, so it doesn't fit the existing dated-doc-citation
  pattern, but a reader skimming only the header wouldn't know the section exists).

> **Closed:** K-004 (audit/reconcile method) — documented in the `agent-maintenance` skill (§3), done 2026-06-07, see history. K-006 (slim the prompt) — done 2026-06-07, see history. K-007 (extract standards → reference skill) — done 2026-06-07, the `agent-standards` skill, see history. K-010 (cobb subagent-awareness clause) + K-011 (destructive-ops guard parity: shared core + graph-dba/qa-engineer wrappers) — done 2026-07-11, see history. K-012 (single-artifact prompt-quality lint — promoted from the dormant self-review-checklist parking-lot idea) — done 2026-07-16, `agent-maintenance` skill §7 + cobb.md trigger + §4 fold-in, see history. K-013 (§7 refinements from the first-run teco smoke test: prompt-severity rubric + cross-cutting-finding attribution) — done 2026-07-16, see history.

### K-001 — Re-verify standards against live docs
- **Status:** 🔵 proposed
- **Priority:** high
- **Rationale:** Frontmatter fields, directory paths, and inclusion modes for Claude Code, Kiro, and OpenCode shift between releases. Stale specifics would make Cobb produce broken artifacts.
- **Proposed change:** On a cadence (or whenever a user reports a mismatch), fetch kiro.dev/docs, opencode.ai/docs, code.claude.com/docs, platform.claude.com/docs and reconcile the "Standards you know cold" section. Log diffs in history.md.
- **Notes:** Baseline verified 2026-05-31 at creation. Subagent context-loading + frontmatter re-verified 2026-06-07 against code.claude.com/docs/en/sub-agents. **Kiro + OpenCode re-verified 2026-06-07** (kiro.dev/docs/steering, opencode.ai/docs/agents+rules) during the K-007 skill extraction — caught real OpenCode drift (`mode: all` default, new `disable`/`color`/`top_p`/`steps` fields, granular permission keys, AGENTS.md precedence). All current specifics now live in the `agent-standards` skill with per-file `Verified:` stamps. **2026-06-20:** Claude Code **subagents** re-verified against code.claude.com/docs/en/sub-agents (tool inheritance + withheld-tools list, expanded frontmatter, discovery/scopes, agent teams + background agents) — claude-code.md stamp bumped to 2026-06-20. Claude Code **Skills/Memory/Hooks/MCP/SDK** still on the 2026-05-31 baseline — next refresh target. **Kiro** docs re-read 2026-06-20 (agents/subagents, steering, specs, hooks). NB: the old "does `inclusion: always` steering reach a subagent" dispute is **not** resolved by a doc re-read — docs affirm it, but it's field-disputed and needs a runtime test on the install. Specs/Hooks confirmed *not* reaching subagents. **OpenCode** re-verified 2026-06-20 (agents/permissions/rules — caught the `tools`→`permission` deprecation; subagent nesting documented; flagged the parent-context divergence vs Claude). **All three tools' agent/subagent surfaces now current as of 2026-06-20.** **2026-07-25:** Claude Code **MCP** fully re-verified against `code.claude.com/docs/en/mcp` and rewritten from a 3-line stub into a full section (scopes/approval + untrusted-workspace gate, `.mcp.json` shape, `${VAR}`-only expansion + the `CLAUDE_PROJECT_DIR`-in-the-server's-env trap, tool naming, the four timeout layers, **tool search on by default** + `alwaysLoad` + server `instructions`, **output limits and the persist-to-disk/file-reference behaviour**, stdio non-reconnection, and how MCP meets subagent `tools:` allowlists); **OpenCode** MCP + Skills sections added/re-verified against `opencode.ai/docs/{mcp-servers,skills}`. Remaining stale: Claude Code Skills/Memory/Hooks/SDK (2026-05-31); Kiro MCP not re-checked since 2026-06-20.

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

| K-015 | 2026-07-31 | medium | 🔵 | `analyst/kaizen/inbox.md` has a substantial backlog of already-verified, "suggested home: prompt" entries never distilled (stub-package HEAD-vs-working-tree import, review-safe pytest subset, isolatable snapshot side, byte-identity AST hash, line-number-invariance re-gate, exclude_unset nested-model gotcha, scratch-copy-reverse-patch). Run a full §5 pass: verify each still holds, promote the prompt-worthy ones into `analyst.md` (or a knowledge base for the FastAPI/FalkorDB/MCP-version-sensitive ones), log in `analyst/kaizen/history.md`, clear the inbox. |

## Parking lot / ideas
- From the 2026-08-12 corrective pass fixing `analyst`'s gate on the 2026-08-11 distillation
  (`docs/reviews/kaizen-distillation-2026-08.md`): a raw inbox entry can leak the maintainer's
  home path/username into a tracked file the moment an agent *appends* it — before any
  distillation ever runs, since `kaizen/inbox.md` is itself a tracked file and an entry's
  `**Evidence:**` line often quotes a live shell command's literal output (`ls -la ~/.claude/
  agents` prints the real symlink target). Caught here only because `audit-team.sh` check 7 was
  re-run incidentally, not because anything in the distillation-review workflow prompts for it.
  Worth deciding whether check 7 (or a lighter version of it) should run as part of *any* agent's
  closing protocol when it appends an inbox entry that quotes command output, not just at
  distillation time — candidate for `agent-maintenance` §5 or the "Learning capture" boilerplate
  itself.
- Maintain a small catalog of agents/skills Cobb has authored, cross-linking their kaizen files.
- The §7 prompt-lint is judgment-only by design; if a *deterministic* pre-check for a single artifact ever proves cheap (frontmatter valid, description non-empty, no personal identifiers), consider a small script assist — but keep the six semantic dimensions in the skill, not a grep. *(Noted 2026-07-16 during the §7 build; the composition load-set enumerator the design floated was skipped as not-cheap-enough.)*
