# Claude Code / Claude Agent SDK — reference

> **Verified:** subagents (frontmatter, tool inheritance, discovery, what-loads,
> multi-agent primitives) **2026-06-20** against `code.claude.com/docs/en/sub-agents`.
> Skills / Memory / Hooks / MCP / SDK on the **2026-05-31** baseline (`code.claude.com/docs`,
> `platform.claude.com/docs`) — due for refresh. Field lists grow between releases; re-verify
> before relying on an exact key.

## Subagents

- **Location & scopes:** Markdown + YAML frontmatter. Scopes, highest priority
  first: **managed** (org admin, in the managed-settings dir) → **project**
  (`.claude/agents/`) → **user** (`~/.claude/agents/`). Also **plugin** subagents
  (from installed plugins) and **CLI-defined** (`--agents` JSON, session-only).
  Same `name` across scopes → higher-priority scope wins.
- **Discovery:** the agent is identified by its `name:` frontmatter, **not its
  path** — so per-agent subdirectories (`<name>/<name>.md`) work. Project
  subagents are found by **walking up from the cwd**, scanning every
  `.claude/agents/` between cwd and the repo root; **v2.1.178+** → on a `name`
  collision the definition **closest to the cwd** wins. `--add-dir` directories
  are also scanned.
- **Frontmatter fields** (full set verified 2026-06-20; `--agents` JSON uses the
  same keys plus `prompt`/`initialPrompt`):
  - `name` — unique identifier.
  - `description` — **required**; drives auto-delegation. Say *what it does and
    precisely when to invoke*, third person, with trigger keywords. The routing signal.
  - `model` — `opus` | `sonnet` | `haiku` | `inherit` (Claude can also pass a
    per-invocation model override).
  - `tools` — allowlist (omit to inherit all). · `disallowedTools` — denylist.
  - `permissionMode` — `default` | `acceptEdits` | `auto` | `dontAsk` |
    `bypassPermissions` | `plan`.
  - `skills` — **preload** skills: the *full skill content* is injected at
    startup (not just the description). The subagent can still invoke *unlisted*
    project/user/plugin skills via the Skill tool.
  - `mcpServers` — MCP servers for this subagent (name ref or inline config).
  - `hooks` — lifecycle hooks scoped to this subagent.
  - `memory` — persistent learning store (see below).
  - `isolation` — `worktree` runs it in a temporary git worktree (isolated repo
    copy, branched from the default branch; auto-cleaned if it makes no changes).
  - `effort` — reasoning-effort control. · `maxTurns` — turn cap.
  - `background` — `true` always runs it as a background task.
  - `color` — UI identifier.
  - ⚠️ **Plugin** subagents ignore `hooks`, `mcpServers`, `permissionMode`.
- **Execution:** each subagent runs in its **own isolated context window**,
  launched via the Task/Agent tool — for context isolation and parallelism. It
  starts in the parent's cwd; `cd` does **not** persist between its Bash calls.

### Tool inheritance & the withheld-tools list (verified 2026-06-20)

- Subagents **inherit the main conversation's internal + MCP tools by default**
  (gate with `tools`/`disallowedTools`).
- These tools are **withheld from subagents even if listed in `tools`** (they
  depend on the main-session UI/state): `AskUserQuestion`, `EnterPlanMode`,
  `ExitPlanMode` (allowed only if `permissionMode: plan`), `ScheduleWakeup`,
  `WaitForMcpServers`.
- **The `Agent` (Task) tool is NOT withheld** → **a subagent can delegate to
  other subagents.** Orchestrator/coordinator subagents are viable. (This
  supersedes the older "subagents can't spawn subagents" lore.) Consequence:
  an orchestrating subagent **can't `AskUserQuestion`** — design it to *return*
  to the user with the decision rather than ask mid-run.
- To stop delegation entirely, deny the `Agent` tool via `permissions.deny`;
  in headless/SDK, `CLAUDE_AGENT_SDK_DISABLE_BUILTIN_AGENTS=1` removes built-ins.

### Built-in subagents & multi-agent primitives (verified 2026-06-20)

- **Built-ins** always registered interactively: **Explore** (wide read-only
  search), **Plan** (quick implementation plan), **general-purpose**.
- **Agent teams** (`/en/agent-teams`) — multiple sessions that *communicate*;
  a teammate can reference a subagent definition (uses its `tools`/`model`, body
  appended as instructions). **Background agents** (`/en/agent-view`) — many
  independent sessions run in parallel, monitored from one place; `background:
  true` frontmatter opts a subagent into this. Prefer these over hand-rolled
  nested delegation for parallel/long-running multi-agent work.

### What loads into a subagent (verified 2026-06-20)

- The body **replaces** the default system prompt — a subagent receives **only**
  its system prompt + basic environment (cwd), **not** the full Claude Code system prompt.
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
