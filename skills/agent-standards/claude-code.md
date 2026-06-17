# Claude Code / Claude Agent SDK — reference

> **Verified:** subagents/frontmatter **2026-06-07** against `code.claude.com/docs/en/sub-agents`.
> Skills / Memory / Hooks / MCP / SDK on the **2026-05-31** baseline (`code.claude.com/docs`,
> `platform.claude.com/docs`) — due for refresh. Field lists grow between releases; re-verify
> before relying on an exact key.

## Subagents

- **Location:** Markdown + YAML frontmatter in `.claude/agents/` (project) or
  `~/.claude/agents/` (personal). Discovered **recursively**; the agent is
  identified by its `name:` frontmatter, not its path — so per-agent
  subdirectories (`<name>/<name>.md`) work, as long as names are unique across the tree.
- **Frontmatter fields:**
  - `name` — unique identifier.
  - `description` — drives auto-delegation; write it to say *what it does and
    precisely when to invoke*, third person, with trigger keywords. This is the routing signal.
  - `model` — `opus` | `sonnet` | `haiku` | `inherit`.
  - `tools` — allowlist (omit to inherit all).
  - `disallowedTools` — denylist.
  - `permissionMode` — e.g. default / acceptEdits / plan / bypassPermissions.
  - `skills` — pin specific skills to the agent.
  - `memory` — persistent learning store (see below).
  - `isolation`, `effort` — context-isolation / reasoning-effort controls.
  - ⚠️ The field set grows; verify the full list against the docs.
- **Execution:** each subagent runs in its **own isolated context window**,
  launched via the Task/Agent tool — use for context isolation and parallelism.

### What loads into a subagent (verified 2026-06-07)

- The body **replaces** the default system prompt.
- The **full `CLAUDE.md`/memory hierarchy still auto-loads** via the normal
  message flow — and `@`-imports expand, so a `CLAUDE.md` of just `@AGENTS.md`
  reaches the subagent.
- It does **not** see the parent's conversation history, prior tool results, or
  already-invoked skills — **pass those in the delegation prompt.**
- **Exception:** built-in **Explore** and **Plan** skip `CLAUDE.md` + git status
  for speed (not configurable). A **fork** is the opposite — it inherits the
  entire parent conversation.

### `memory:` frontmatter ≠ `CLAUDE.md`

A **separate** persistent learning store. `memory: user|project|local` gives the
agent an `agent-memory/<name>/` dir whose `MEMORY.md` (first ~200 lines / 25 KB)
is injected into its system prompt, for cross-session knowledge. Distinct from
the always-loaded project memory (`CLAUDE.md`).

## Skills

- **Location:** a directory under `.claude/skills/<name>/` containing `SKILL.md`
  (frontmatter `name`, `description`, optional `allowed-tools`) plus any
  supporting files.
- **Progressive disclosure:** only the `description` is loaded at startup; the
  body loads when the model decides the task matches; supporting files load on
  explicit reference. Keeps always-on context lean.
- **`allowed-tools`** applies in the **CLI** but **not** through the SDK —
  control tools via `allowedTools` there.
- Follows the open **Agent Skills** standard (Claude Code, Codex CLI, Cursor,
  Gemini CLI, Copilot).

## Memory

- `CLAUDE.md` — Claude-specific project rules.
- `AGENTS.md` — universal/cross-tool project law.
- **Hierarchy:** enterprise → user → project → local (more specific wins / appends).
- DRY: when both would carry the same content, `CLAUDE.md` = `@AGENTS.md` import.

## Hooks

- Shell commands fired on lifecycle events (**PreToolUse, PostToolUse, Stop**,
  etc.), configured in `settings.json`.
- The **harness** runs them, not the model — this is how you enforce
  deterministic "always do X" behavior. Prefer a hook over hopeful prompt text
  whenever the requirement is "must always happen."

## MCP

External tools/servers exposed to the agent. Configure per project/user; servers
provide tools the model can call.

## Agent SDK

- Packages: `claude_agent_sdk` (Python) / `@anthropic-ai/sdk` (TS) — for building
  programmatic agents.
- Key options: `settingSources` / `setting_sources` (which settings files to
  load), `skills`, `allowedTools` (tool gating — note skills' `allowed-tools` is
  ignored here), prompt caching.
