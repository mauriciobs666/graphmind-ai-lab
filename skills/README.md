# Skills

Unified [Agent Skills](https://agentskills.io) home for the repo — one `SKILL.md` package per
folder (with optional `references/`, scripts, and storage). This is the open, cross-tool skill
standard read by **Claude Code**, **OpenCode**, and **Kiro**.

> **Portability:** the `SKILL.md` *format* ports across all three tools, but *behavior* does not
> automatically — tool-gating frontmatter (e.g. Claude Code's `allowed-tools`) and
> activation/matching semantics differ per tool. Each tool also keeps `name` ≤ 64 chars (must
> match the folder) and `description` ≤ 1024 chars. Re-verify any skill you deploy to a new tool.

## Catalog

| Skill | What it does | When to use | Origin |
|-------|--------------|-------------|--------|
| [`agent-maintenance`](./agent-maintenance/SKILL.md) | Kaizen plan/history upkeep, dual-audience documentation, file-location conventions, the drift audit/reconcile method, the team-coherence certification pass (§4; deterministic checks scripted in `claude/scripts/audit-team.sh`), the learnings-inbox distillation procedure (§5 — the team's capture→distill self-improvement loop), and the single-artifact prompt-quality lint (§7 — semantic review of one prompt over six dimensions: contradiction, ambiguity, persona, cognitive load, coverage, composition conflict). | Creating/editing/renaming/removing/reviewing any agent, subagent, skill, steering doc, or memory file; certifying an agent team; processing learnings inboxes; linting a single prompt's quality. | cobb machinery |
| [`agent-standards`](./agent-standards/SKILL.md) | Perishable per-tool reference: exact frontmatter fields, directory paths, inclusion modes, config keys, what-loads-where tables, and **MCP wiring per tool** (scopes/approval, `.mcp.json` shape and `${VAR}` expansion, tool naming, timeouts, tool search + `alwaysLoad`, output limits, and how MCP meets subagent `tools:` allowlists and skill `allowed-tools`) for Claude Code/Kiro/OpenCode. Every fact `Verified:`-stamped. | Producing/porting/debugging a concrete artifact and needing exact field names/paths rather than mental models. | cobb machinery |
| [`joern-cpg`](./joern-cpg/SKILL.md) | Operates the **Joern** toolset to turn a source repo into a Code Property Graph and export/load it into **FalkorDB** as Cypher: scripts for parse → export (neo4jcsv) → transform → load (pinning `JOERN_HOME`/`JAVA_HOME`), the default CPG→FalkorDB model (shared `:CpgNode` label + indexed `id`), and a CPGQL cheat-sheet (`references/cpg-model.md` for the deeper schema). | Building a CPG for a codebase, running CPGQL queries, or exporting/ingesting a repo's code graph into FalkorDB / Cypher. | `graph-dba` (on demand — CPG generation is rare) |
| [`cpg-analysis`](./cpg-analysis/SKILL.md) | **Consumer** counterpart to `joern-cpg`: queries an already-loaded CPG in FalkorDB with read-only Cypher through the **`mcp__cpg__query` MCP tool** (one tool, two parameters: `graph`, `cypher`; `EXPLAIN` supported, `PROFILE` refused), with **`redis-cli GRAPH.QUERY` kept as the documented fallback**. Lean `SKILL.md` core (query surface + shared CONTAINS→CALL / `REACHING_DEF` / symbol def-ref idioms + the topology gotchas) plus four copy-adaptable `references/` recipes — impact analysis, root-cause analysis, code review (taint), test-gap. Cites `joern-cpg/references/cpg-model.md` as the one schema source; live-verified against a `pysrc2cpg` CPG. ⚠ **The MCP wiring is Claude-Code-only** (repo-root `.mcp.json`); under OpenCode/Kiro the `redis-cli` fallback is the only path — see the portability note below. | Impact / root-cause / taint / test-gap questions over a loaded CPG — when analyst, architect, or qa-engineer need call-graph or data-flow answers instead of reading files. Building/loading a CPG routes to `graph-dba`. | `graph-dba` (M2) |
| [`comparison-driver`](./comparison-driver/SKILL.md) | Systematically identifies pros/cons, finds cost-effective options, and presents comprehensive overviews with summaries. | Analyzing ideas or product models / decision support. | OpenCode |
| [`python-coding`](./python-coding/SKILL.md) | Python assistant following best practices: writing, debugging, pytest, type hints, Python-specific refactoring. | Creating/maintaining Python code. | OpenCode |
| [`skill-builder`](./skill-builder/SKILL.md) | Builds new `SKILL.md` files with proper structure, conventions, and best practices. | Authoring a new skill. | OpenCode |
| [`user-preferences`](./user-preferences/SKILL.md) | Stores, retrieves, and keyword-searches user preferences across markdown files (`storage/`). | Conversational agents that remember the user across sessions (used by the `rpg` agent). | OpenCode |
| [`write-tutorial`](./write-tutorial/SKILL.md) | Creates structured learning paths and comprehensive markdown tutorials; uses `comparison-driver` for option analysis. | Generating tutorials / learning content. | OpenCode |

## Deployment

Skills live here, version-controlled, and are surfaced to all three harnesses via a whole-dir
symlink from each tool's global config — so **every tool sees every skill in this directory** and
edits here are picked up live:

| Tool | Symlink |
|---|---|
| Claude Code | `~/.claude/skills` → `skills/` |
| OpenCode | `~/.config/opencode/skills` → `skills/` |
| Kiro | `~/.kiro/skills` → `skills/` |

Recreate on a new machine with `ln -s <repo>/skills <target>`. Skills are progressively-disclosed
(only the `description` is always-on), so exposing every skill everywhere costs ~nothing; unused
ones simply never activate. If you later want per-tool scoping, switch a tool to per-skill symlinks
(the pattern the `claude/` agents use) instead of the whole-dir link.

### Portability notes (verified 2026-07-25)

- **Claude-only frontmatter is ignored, not rejected.** `cpg-analysis` declares
  `allowed-tools: mcp__cpg__query, Bash, Read` — an MCP tool name that exists only under Claude
  Code. OpenCode still discovers and parses the skill with its description intact (`opencode debug
  skill`, 6 runs, no warning on stderr), and its docs state that unknown `SKILL.md` frontmatter
  fields are ignored (`opencode.ai/docs/skills`; its recognized set is `name`, `description`,
  `license`, `compatibility`, `metadata` — gating is done with `permission.skill` patterns, not
  `allowed-tools`). **Not** exercised: an actual OpenCode *invocation* of the skill — tracked as
  **C-310** with the OpenCode/Kiro MCP wiring.
- **⚠ OpenCode's discovery over the whole-dir symlink is non-deterministic.** Repeated
  `opencode debug skill` runs return *different subsets* (7–9 of the 9 skills here plus the
  built-in), with no error or warning — observed both before and after the change above, so it is
  a harness behaviour, not a defect in any skill. Don't read a missing skill in one run as "the
  skill is broken"; re-run. This qualifies the "every tool sees every skill" claim above for
  OpenCode. Claude Code and Kiro were not re-measured.

## Maintenance

`agent-maintenance` and `agent-standards` are cobb's machinery; changes to them are logged in
[`claude/cobb/kaizen/history.md`](../claude/cobb/kaizen/history.md). Keep this catalog and the
root [`AGENTS.md`](../AGENTS.md) in sync when adding/editing/removing a skill.
