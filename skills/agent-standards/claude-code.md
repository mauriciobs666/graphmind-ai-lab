# Claude Code / Claude Agent SDK — reference

> **Verified:** subagents (frontmatter, tool inheritance, discovery, what-loads,
> multi-agent primitives) **2026-06-20** against `code.claude.com/docs/en/sub-agents`;
> **main-session (`--agent`) mode added 2026-07-09** against the same page.
> **Agent Teams + `SendMessage` re-verified 2026-06-21** against `code.claude.com/docs/en/agent-teams`
> (experimental, env-var-gated; see the multi-agent-primitives section).
> **MCP re-verified 2026-07-25** against `code.claude.com/docs/en/mcp` (whole page) — see the
> `## MCP` section, which carries its own stamp.
> **`model` frontmatter field re-verified 2026-07-27** against
> `code.claude.com/docs/en/sub-agents` (accepts `fable` and full model IDs; **defaults to
> `inherit`**).
> Skills / Memory / Hooks / SDK still on the **2026-05-31** baseline (`code.claude.com/docs`,
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
  - `model` — `opus` | `sonnet` | `haiku` | `fable` | a **full model ID**
    (e.g. `claude-opus-5`) | `inherit`. **Defaults to `inherit`** — omitting the
    field is the idiomatic way to say "use whatever model the session/system
    default selects" (verified 2026-07-27). Claude can also pass a
    per-invocation model override.
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

> **Verified: 2026-07-25** against `code.claude.com/docs/en/mcp` (full page).
> Where the doc gates a behaviour on a release, the version is kept — these are
> the facts most likely to move.

### Scopes, precedence, and the approval gate

Three scopes, precedence **local → project → user** (then plugin servers, then
claude.ai connectors). When the same server name exists in two scopes the
**higher-precedence entry is used whole — fields are not merged**.

| Scope | Stored in | Reach |
|---|---|---|
| `local` (default for `claude mcp add`) | `~/.claude.json`, under `projects.<abs-path>.mcpServers` | you, this project — **untracked** |
| `project` | **`.mcp.json` at the repo root** | the team, version-controlled |
| `user` | `~/.claude.json` | you, every project |

- **Project-scoped servers require a one-time interactive approval.** Reset the
  answers with `claude mcp reset-project-choices`.
- `.claude/settings.json` → `enabledMcpjsonServers: ["<name>"]` pre-approves by
  name (`enableAllProjectMcpServers` for all) — **but both are ignored in an
  untrusted workspace** (v2.1.196+): a freshly cloned repo cannot approve its own
  servers and they stay at `⏸ Pending approval`. **Consequence: a headless
  `claude -p` run in an un-approved workspace silently has no server at all** —
  and the failure looks like "the agent ignored the tool", not like a config error.
- Don't confuse the two key pairs: `disabledMcpServers` / `enabledMcpServers`
  (in `~/.claude.json`, per project — the `/mcp` panel's on/off toggle) vs
  `enabledMcpjsonServers` / `disabledMcpjsonServers` (in settings — `.mcp.json`
  *approval*).
- **There is no per-server tool filter.** You can disable a whole server; you
  cannot subset the tools a server advertises. A permission `deny` rule blocks a
  *call*, but the tool still occupies the model's tool list — so "buy the server
  and hide the tools we don't want" is not available. Verified twice (2026-07-24
  analyst, 2026-07-25 cobb): nothing in the docs offers it.

### `.mcp.json` shape and variable expansion

```json
{
  "mcpServers": {
    "<name>": {
      "command": "…", "args": ["…"], "env": { "K": "V" },
      "timeout": 60000, "alwaysLoad": false
    }
  }
}
```

Remote servers use `type` (`http` | `sse` | `ws`) + `url` + `headers` /
`headersHelper` instead of `command`/`args`.

- **The only expansion syntax is `${VAR}` and `${VAR:-default}`** — a bare
  `$VAR` is passed through untouched. Expansion applies in `command`, `args`,
  `env`, `url` and `headers`.
- If a `${VAR}` is unset and has no default, the config **still loads**: the
  server is flagged with a missing-variable warning in `claude mcp list` and the
  literal `${VAR}` text is used — i.e. it fails at connect time, not at parse time.

### `CLAUDE_PROJECT_DIR` — set in the *server's* env, not Claude Code's

Claude Code sets `CLAUDE_PROJECT_DIR` **in the spawned server's environment** (the
stable project root; unchanged by mid-session `--add-dir`). It is **not** set in
Claude Code's own config-expansion environment, so `${CLAUDE_PROJECT_DIR}` inside
a project- or local-scoped entry needs a default (`${CLAUDE_PROJECT_DIR:-.}`) —
and that default silently resolves to **cwd**, breaking any session started in a
subdirectory. (Plugin-provided MCP configs are the exception: `${CLAUDE_PROJECT_DIR}`
substitutes directly there.)

**The idiom that works** for a repo-relative stdio launcher with no absolute path
in a tracked file — expand it in the *shell*, from the server env:

```json
{ "command": "bash", "args": ["-c", "exec \"$CLAUDE_PROJECT_DIR/path/to/run.sh\""] }
```

A server can also read it directly (`os.environ["CLAUDE_PROJECT_DIR"]`). A server
that needs the user's *granted* directories should implement MCP `roots/list`
instead — Claude Code answers with the launch dir plus every `--add-dir`
directory, and sends `notifications/roots/list_changed` when that set changes
(v2.1.203+).

### Tool naming

`mcp__<server>__<tool>` — e.g. `mcp__cpg__query`. Plugin-bundled servers take the
longer form `mcp__plugin_<plugin>_<server>__<tool>` (any character outside
`A-Za-z0-9_-` becomes `_`), and a hook matcher written against the bare server key
**never fires** for one.

That exact string is what goes in: **permission rules**, a **skill's
`allowed-tools`**, a **subagent's `tools:` field**, and **hook matchers**. MCP
prompts surface as slash commands: `/mcp__<server>__<prompt>`.

### Timeouts

- `MCP_TIMEOUT` (env) — server **startup** timeout.
- Per-server **`"timeout"`** in `.mcp.json`, in ms — a **hard wall-clock cap per
  tool call**, overriding `MCP_TOOL_TIMEOUT` for that server. Progress
  notifications do not extend it. **Values below 1000 are ignored** and fall
  through to `MCP_TOOL_TIMEOUT`, whose default is ≈28 hours — so "no timeout set"
  effectively means "no timeout". Set it.
- Idle timeout (v2.1.187+): a call that sends nothing for the idle window aborts.
  Default **30 min for stdio**, 5 min for HTTP/SSE/ws; `CLAUDE_CODE_MCP_TOOL_IDLE_TIMEOUT`
  (ms, `0` disables). A per-server `timeout ≥ 1000` acts as a floor on it (v2.1.203+).
- A main-conversation call running past **2 minutes moves to a background task**
  (v2.1.212+); the per-call limits still apply.

### Tool search is ON by default — and it changes what a cold session sees

- **MCP tool definitions are deferred, not loaded upfront.** Only **tool names and
  server instructions** load at session start; Claude calls a `ToolSearch` tool to
  pull in the schema when a task needs it. Expect a `ToolSearch` event in a
  transcript that uses an MCP tool — an assertion of "exactly one tool call" must
  account for it.
- **Server `instructions` are the routing signal under tool search.** They are what
  tells Claude when to go looking for your tools (the same role a skill's
  `description` plays). With the Python SDK: `FastMCP(name=…, instructions=…)`.
  **Claude Code truncates tool descriptions and server instructions at 2 KB each**
  — put the decisive words first.
- `"alwaysLoad": true` on a server entry (v2.1.121+, all server types) loads *all*
  its tools at session start regardless of `ENABLE_TOOL_SEARCH`. Worth it only for
  a small tool set needed most turns — each upfront tool costs context, and
  `alwaysLoad` **also blocks startup until that server connects** (capped at the
  5 s connect timeout). A server can mark a single tool instead with
  `_meta: {"anthropic/alwaysLoad": true}` in its `tools/list` entry.
- Off-switches / fallbacks: `ENABLE_TOOL_SEARCH=false` (off), `=auto`
  (threshold-based: load upfront while under 10% of the context window, defer the
  rest); `CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS` forces it off and
  `ENABLE_TOOL_SEARCH` cannot override that. Needs a model supporting
  `tool_reference` blocks (Sonnet 4.5 / Haiku 4.5 / Opus 4.5 and later) and is
  disabled by default on Google Cloud's Agent Platform and behind a non-first-party
  `ANTHROPIC_BASE_URL`. Without tool search, Claude Code uses `WaitForMcpServers`
  and (on Bedrock / Agent Platform / Foundry) does **not** report failed server
  connections to Claude.

### Output limits — the trap when a tool returns a big table

- Claude Code **warns above 10,000 tokens** of MCP tool output (threshold fixed)
  and **caps at 25,000 tokens** by default (`MAX_MCP_OUTPUT_TOKENS`).
- **Above the threshold the result is persisted to disk and replaced by a file
  reference in the conversation.** So a server that appends its own "truncated: …"
  notice as the *last* line can lose exactly that line in exactly the run where it
  matters. Design for it: put the notice first as well, or keep payloads under the
  threshold.
- Per-tool escape: `_meta["anthropic/maxResultSizeChars"]` in the tool's
  `tools/list` entry raises *that tool's* threshold, **ceiling 500,000 chars**,
  independent of `MAX_MCP_OUTPUT_TOKENS` (text content only — image content stays
  bound by the token limit).

### Lifecycle

- **Stdio servers are local processes and are NOT auto-reconnected** if they die
  mid-session. HTTP/SSE reconnect with exponential backoff (five attempts), and
  retry a failed initial connection three times on transient errors. Recovery for
  stdio is `/mcp` → reconnect, or a new session — so an stdio server must be
  crash-proof and must never write to stdout (the transport owns it).
- `list_changed` notifications refresh a server's tools/prompts/resources without a
  reconnect; a failed refresh keeps the previously discovered set (v2.1.214+).

### How MCP meets subagents and skills

- **A subagent's `tools:` is an allowlist, and MCP tools are subject to it.** An
  agent that declares `tools:` sees **no** MCP tool that isn't named there — the
  single easiest way to ship an MCP feature that is silently inert for some agents.
  Agents that omit `tools:` inherit everything.
- Subagent frontmatter also accepts `mcpServers`, but see the caveats above:
  **ignored for teammates** (`:110-111`) and **ignored for plugin subagents**
  (`:46`) — don't rely on it to scope a server to particular agents.
- A skill's `allowed-tools` **pre-approves** the listed tools for the turn that
  invokes the skill; it does not gate them, and it does not grant a tool the
  session doesn't have. Naming an MCP tool there is also harmless in other
  harnesses that read `SKILL.md` (they ignore unknown frontmatter — see
  `opencode.md`), which is what makes the field portable.
- **MCP wiring itself does not port.** `.mcp.json` is Claude Code's; OpenCode and
  Kiro configure servers in their own files (see `opencode.md` § MCP servers and
  `kiro.md`). A `SKILL.md` shared across harnesses that *depends* on an MCP tool
  must keep a non-MCP fallback path documented, or it is broken everywhere else.

## Agent SDK

- Packages: `claude_agent_sdk` (Python) / `@anthropic-ai/sdk` (TS) — for building
  programmatic agents.
- Key options: `settingSources` / `setting_sources` (which settings files to
  load), `skills`, `allowedTools` (tool gating — note skills' `allowed-tools` is
  ignored here), prompt caching.
