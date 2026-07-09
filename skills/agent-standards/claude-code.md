# Claude Code / Claude Agent SDK — reference

> **Verified:** subagents (frontmatter, tool inheritance, discovery, what-loads,
> multi-agent primitives) **2026-06-20** against `code.claude.com/docs/en/sub-agents`;
> **main-session (`--agent`) mode added 2026-07-09** against the same page.
> **Agent Teams + `SendMessage` re-verified 2026-06-21** against `code.claude.com/docs/en/agent-teams`
> (experimental, env-var-gated; see the multi-agent-primitives section).
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

### Running a definition as the MAIN session agent (verified 2026-07-09 against `/en/sub-agents`)

An agent definition is not only a delegation target — it can be the **first-order,
conversational agent**:

- **`claude --agent <name>`** (or the **`agent` setting**) starts a session where the
  **main thread itself** takes on that definition's system prompt, tool restrictions,
  and model. Plugin agents: `claude --agent my-plugin:name` (include any `agents/`
  subfolder in the scoped name).
- **`initialPrompt` frontmatter** is auto-submitted as the first *user* turn in this
  mode (commands and skills are processed; it's prepended to any user-provided prompt).
- **Frontmatter hooks fire in main-session mode too**, alongside `settings.json` hooks
  (they also fire when the agent is spawned as a subagent or @-mentioned).
- The **withheld-tools list applies to subagents only** — as the main session the agent
  can use `AskUserQuestion` etc., so live multi-turn interaction works.
- The main-thread agent can spawn subagents via `Agent`; the **`Agent(agent_type)`
  allowlist syntax** in `tools` restricts which types — but **only** in main-thread
  mode (inside a subagent definition the parenthesized type list is ignored).

### Built-in subagents & multi-agent primitives (agent-teams re-verified 2026-06-21 against `/en/agent-teams`)

- **Built-ins** always registered interactively: **Explore** (wide read-only
  search), **Plan** (quick implementation plan), **general-purpose**.
- **Two communication models — know which you're in:**
  - **Subagents (default):** workers run in their own context and **only report
    results back to the main agent — they never talk to each other.** No
    inter-agent messaging tool is exposed.
  - **Agent teams (`/en/agent-teams`):** teammates share a task list + a
    **mailbox** and **message each other by name** via the **`SendMessage`**
    tool. `SendMessage` + the task-management tools are **always available to a
    teammate even when its `tools` allowlist restricts everything else.**
- **⚠️ `SendMessage` only exists inside Agent Teams, which is EXPERIMENTAL and
  OFF by default** — gated behind the env var **`CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1`**
  (set in `settings.json` `env` or the shell). Without it, no team is formed and
  **`SendMessage` is not exposed in the session at all** — so it can't be
  conjured via an agent's `tools:` frontmatter (the allowlist filters from what
  the runtime exposes; it can't add a tool the harness isn't shipping). The lever
  to "give an agent SendMessage" is the env-var flag, not the frontmatter. (As of
  v2.1.178 the old `TeamCreate`/`TeamDelete` tools no longer exist; spawning a
  teammate needs no setup step.)
- A **subagent *definition*** can be **reused as a teammate** (mention its type
  when spawning): the teammate honors that def's `tools` allowlist + `model`, the
  body is *appended* (not replacing), and it then gets `SendMessage` + task tools
  automatically. (`skills`/`mcpServers` frontmatter is **ignored** for teammates —
  they load skills/MCP from project+user settings.)
- **Agent-teams limits:** one team per session; **no nested teams** (teammates
  can't spawn teammates); lead is fixed for the session's lifetime; `/resume` +
  `/rewind` don't restore in-process teammates.
- **Background agents** (`/en/agent-view`) — many independent sessions run in
  parallel, monitored from one place; `background: true` frontmatter opts a
  subagent into this. Prefer teams/background over hand-rolled nested delegation
  for parallel/long-running multi-agent work.

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
