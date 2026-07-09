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
| [`agent-maintenance`](./agent-maintenance/SKILL.md) | Kaizen plan/history upkeep, dual-audience documentation, file-location conventions, the drift audit/reconcile method, and the team-coherence certification pass (§4; deterministic checks scripted in `claude/scripts/audit-team.sh`). | Creating/editing/renaming/removing/reviewing any agent, subagent, skill, steering doc, or memory file; certifying an agent team. | cobb machinery |
| [`agent-standards`](./agent-standards/SKILL.md) | Perishable per-tool reference: exact frontmatter fields, directory paths, inclusion modes, config keys, and what-loads-where tables for Claude Code/Kiro/OpenCode. Every fact `Verified:`-stamped. | Producing/porting/debugging a concrete artifact and needing exact field names/paths rather than mental models. | cobb machinery |
| [`comparison-driver`](./comparison-driver/SKILL.md) | Systematically identifies pros/cons, finds cost-effective options, and presents comprehensive overviews with summaries. | Analyzing ideas or product models / decision support. | OpenCode |
| [`python-coding`](./python-coding/SKILL.md) | Python assistant following best practices: writing, debugging, pytest, type hints, Python-specific refactoring. | Creating/maintaining Python code. | OpenCode |
| [`skill-builder`](./skill-builder/SKILL.md) | Builds new `SKILL.md` files with proper structure, conventions, and best practices. | Authoring a new skill. | OpenCode |
| [`user-preferences`](./user-preferences/SKILL.md) | Stores, retrieves, and keyword-searches user preferences across markdown files (`storage/`). | Conversational agents that remember the user across sessions (used by the `rpg` agent). | OpenCode |
| [`write-tutorial`](./write-tutorial/SKILL.md) | Creates structured learning paths and comprehensive markdown tutorials; uses `comparison-driver` for option analysis. | Generating tutorials / learning content. | OpenCode |

## Deployment

Skills live here, version-controlled, and are surfaced to all three harnesses via a whole-dir
symlink from each tool's global config — so **every tool sees all 7 skills** and edits here are
picked up live:

| Tool | Symlink |
|---|---|
| Claude Code | `~/.claude/skills` → `skills/` |
| OpenCode | `~/.config/opencode/skills` → `skills/` |
| Kiro | `~/.kiro/skills` → `skills/` |

Recreate on a new machine with `ln -s <repo>/skills <target>`. Skills are progressively-disclosed
(only the `description` is always-on), so exposing all 7 everywhere costs ~nothing; unused ones
simply never activate. If you later want per-tool scoping, switch a tool to per-skill symlinks
(the pattern the `claude/` agents use) instead of the whole-dir link.

## Maintenance

`agent-maintenance` and `agent-standards` are cobb's machinery; changes to them are logged in
[`claude/cobb/kaizen/history.md`](../claude/cobb/kaizen/history.md). Keep this catalog and the
root [`AGENTS.md`](../AGENTS.md) in sync when adding/editing/removing a skill.
