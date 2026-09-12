# OpenCode skills (retired)

> **Retired and unmaintained.** This directory was moved to `deprecated/opencode/skills/` along
> with the OpenCode agents that used it (`deprecated/opencode/agents/`) — preserved for reference
> only, no fixes or upgrades. See `deprecated/README.md`. The catalog below is kept as it was at
> retirement.

[Agent Skills](https://agentskills.io) packages authored for and used only by this repo's
(now-retired) OpenCode agents (`deprecated/opencode/agents/`). The repo's cross-tool /
Claude-Code-oriented skills live separately in [`../../../skills/`](../../../skills/README.md);
see that directory's README for the split rationale and the shared portability notes.

## Catalog

| Skill | What it does | When to use | Used by |
|-------|--------------|-------------|---------|
| [`comparison-driver`](./comparison-driver/SKILL.md) | Systematically identifies pros/cons, finds cost-effective options, and presents comprehensive overviews with summaries. | Analyzing ideas or product models / decision support. | `write-tutorial` |
| [`python-coding`](./python-coding/SKILL.md) | Python assistant following best practices: writing, debugging, pytest, type hints, Python-specific refactoring. | Creating/maintaining Python code. | `coding-senior` |
| [`skill-builder`](./skill-builder/SKILL.md) | Builds new `SKILL.md` files with proper structure, conventions, and best practices. | Authoring a new skill. | `coding-senior` |
| [`user-preferences`](./user-preferences/SKILL.md) | Stores, retrieves, and keyword-searches user preferences across markdown files (`storage/`). | Conversational agents that remember the user across sessions. | `rpg` |
| [`write-tutorial`](./write-tutorial/SKILL.md) | Creates structured learning paths and comprehensive markdown tutorials; uses `comparison-driver` for option analysis. | Generating tutorials / learning content. | — |

## Deployment (historical)

OpenCode's global config used to symlink its whole skills directory here:

```
~/.config/opencode/skills -> opencode/skills/
```

That symlink now points at the live (currently empty) `opencode/skills/` instead — this directory
is retired, frozen content, not relocated live config. Claude Code and Kiro never saw this
directory; their symlinks point at the repo's top-level [`skills/`](../../../skills/README.md)
instead, which never included these five packages.

## Maintenance

This directory is frozen — see `deprecated/README.md`. Its retirement is recorded in
[`skills/README.md`](../../../skills/README.md) and the root
[`AGENTS.md`](../../../AGENTS.md).
