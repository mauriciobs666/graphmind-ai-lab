---
name: agent-standards
description: Reference specifics for authoring agent artifacts across Claude Code, Kiro, and OpenCode — exact frontmatter fields, directory paths, inclusion modes, config keys, and "what-loads-where" tables. Load when producing, porting, or debugging a concrete subagent, skill, steering doc, hook, slash command, or config file and you need exact field names/paths rather than mental models. Perishable: every fact is dated; re-verify against the cited official doc before relying on it.
allowed-tools: Read, WebFetch, WebSearch
---

# Agent standards — perishable reference

The exact, version-sensitive specifics for the three agent toolchains cobb knows
cold: **Claude Code / Claude Agent SDK**, **Kiro**, and **OpenCode**. The
resident prompt keeps only stable mental models + canonical URLs; the field
lists, directory paths, inclusion modes, and who-loads-what tables live here so
they can be updated in one place when the docs change.

> **This content is a cache, not the source of truth.** Every reference file is
> stamped `Verified: YYYY-MM-DD against <url>`. These ecosystems move fast.
> **Before asserting a field name or path, check the stamp** — if it's older
> than a few weeks or the user reports a mismatch, **WebFetch the cited official
> doc and reconcile** (then update the file + stamp, and log a kaizen note).
> For perishable facts, live docs beat this snapshot.

## Navigation — load the file you need

| You're working on… | Open |
|---|---|
| A Claude Code subagent, skill, hook, memory file, MCP server, or Agent SDK call | [`claude-code.md`](./claude-code.md) |
| A Kiro steering doc, spec, or hook | [`kiro.md`](./kiro.md) |
| An OpenCode agent/subagent, `opencode.json`, command, or skill | [`opencode.md`](./opencode.md) |
| A portable, cross-tool artifact | the cross-tool section below + the relevant file(s) |

## Canonical doc URLs (verify against these)

- Claude Code: `https://code.claude.com/docs` (subagents: `/en/sub-agents`)
- Claude API / Agent SDK: `https://platform.claude.com/docs`
- Kiro: `https://kiro.dev/docs` (steering: `/docs/steering`)
- OpenCode: `https://opencode.ai/docs` (agents: `/docs/agents`, rules: `/docs/rules`)

## Cross-tool open standards (stable)

Two open standards span all three tools — write to these for the broadest reach:

- **`AGENTS.md`** — portable project rules/memory. Adopted by Claude Code
  (alongside `CLAUDE.md`), OpenCode (its primary rules file, created via `/init`),
  and Kiro (loads always, no inclusion modes). When `CLAUDE.md` and `AGENTS.md`
  would carry identical content, put the content in `AGENTS.md` and make
  `CLAUDE.md` a one-line `@AGENTS.md` import (Claude Code `@`-import; tool-specific).
- **Agent Skills** (`SKILL.md` + progressive disclosure) — the same skill format
  works across Claude Code, OpenCode, and the wider adopters (Codex CLI, Cursor,
  Gemini CLI, Copilot). Frontmatter `name` + `description` always; tool-gating
  keys (`allowed-tools`) are tool-specific in effect — see each tool's file.

**Portability gotcha:** don't assume one tool's *behavior* transfers even when
the *file format* does — subagent context-loading, skill tool-gating, and which
files reach a subagent all differ per tool and per release. Verify per tool.

## Updating this skill (when docs drift)

1. WebFetch the cited official doc for the affected tool.
2. Reconcile the per-tool file; fix changed fields/paths, add new ones, flag removed ones.
3. Bump that file's `Verified:` stamp to today + the exact URL checked.
4. Append a dated note to `claude/cobb/kaizen/history.md` (what drifted, what changed).

This skill is the single update target for K-005 (the automated doc-drift job),
should it be built.
