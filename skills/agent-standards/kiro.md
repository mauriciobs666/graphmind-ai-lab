# Kiro (spec-driven, agentic IDE) — reference

> **Verified: 2026-06-20** — Subagents/Agents, Steering, Specs, Hooks re-verified
> against `kiro.dev/docs/chat/subagents`, `kiro.dev/docs/cli/custom-agents/configuration-reference`,
> `kiro.dev/docs/steering`, plus specs/hooks doc pages. Skills section verified
> 2026-06-16 (`kiro.dev/docs/skills`, `/cli/skills`). Re-verify before relying on
> an exact key — Kiro ships fast and has **two surfaces (IDE + CLI)** that differ.

## Building blocks

**Steering Docs**, **Specs**, **Hooks**, **Agent Skills** (added 2026-02-05), and
**Custom Agents / Subagents** (IDE custom subagents added in 0.9). Kiro also reads
the open **`AGENTS.md`** standard.

> **Two surfaces, different formats.** The **IDE** defines custom agents as
> **Markdown + YAML frontmatter** (like Claude Code); the **CLI** defines them as
> **JSON**. Same concept, different file format and a richer CLI schema. Know which
> one the user is on before authoring.

## Custom Agents & Subagents (verified 2026-06-20)

A *subagent* is a focused task handed to an agent that runs in its **own isolated
context** — keeps the main conversation lean, runs independent tracks in parallel,
or chains specialized agents into a pipeline. A *custom agent* is the reusable
definition you can invoke directly or hand a subagent task to.

### IDE custom subagents — Markdown + YAML
- **Location:** `<workspace>/.kiro/agents/` (workspace) or `~/.kiro/agents/` (global).
  Prompt = file body; config = YAML frontmatter.
- **Frontmatter fields:** `name` (mandatory), `description` (used for automatic
  selection — the routing signal), `tools` (array of accessible tools), `model`,
  `includeMcpJson` + `includePowers` (booleans).
- **Invoke:** "Use the `code-reviewer` subagent to…" or slash command `/code-reviewer …`.

### CLI custom agents — JSON (`kiro.dev/docs/cli/custom-agents/configuration-reference`)
- **Location:** `.kiro/agents/*.json` (local) or `~/.kiro/agents/*.json` (global).
  **Filename (minus `.json`) = agent name.**
- **Config keys:**
  - `name` (optional; from filename), `description`, `model` (e.g. `"claude-sonnet-4"`;
    falls back to default if unavailable).
  - `prompt` — inline text or `file://` URI (`file://./prompt.md` relative, or absolute).
  - `welcomeMessage`, `keyboardShortcut` (`[ctrl|shift]+[a-z|0-9]`).
  - `tools` — array; built-ins (`read`,`write`,`shell`,`aws`,`knowledge`,…),
    MCP (`@server` / `@server/tool`), wildcards (`*` all, `@builtin` all built-ins).
  - `allowedTools` — usable **without prompting**; exact / glob patterns
    (`@server/read_*`) / server-level (`@server`).
  - `toolAliases` — remap tool names to resolve collisions.
  - `toolsSettings` — per-tool config, e.g. `{"write": {"allowedPaths": ["src/**"]}}`
    and **`toolsSettings.subagent`** (subagent permissions — see below).
  - `mcpServers` (each needs `command`), `includeMcpJson` (pull `~/.kiro/settings/mcp.json`).
  - `resources` — `file://…` globs, `skill://.kiro/skills/**/SKILL.md`, or a
    `knowledgeBase` object.
  - `hooks` — JSON map keyed by trigger: `agentSpawn`, `userPromptSubmit`,
    `preToolUse`, `postToolUse`, `stop` (entries take `command` + optional `matcher`).

### What reaches a Kiro subagent (docs re-read 2026-06-20)
- **Steering files: docs say YES** — they state steering works in subagents as in
  the main agent. ⚠️ **Caveat:** this exact claim has historically been *disputed*
  in the field — older bug reports said `inclusion: always` steering did **not**
  actually reach subagents at runtime. Re-reading the docs only confirms the
  *documentation* side of a docs-vs-reality dispute; it does **not** prove runtime
  behavior. **Verify on your install** (drop an `inclusion: always` steering file
  with a unique marker, spawn a subagent, ask it to echo the marker).
- **MCP servers: docs say YES.**
- **Specs: NO** — subagents lack access to Specs.
- **Hooks: NO** — hooks do **not** trigger in subagents.
- Default subagent toolset (when delegating to the default, not a named custom
  agent): `read, write, shell, aws, grep, glob, code, web_search, web_fetch,
  introspect, knowledge, thinking, todo, tool_search` + configured MCP tools.
- A **named custom agent** used as a subagent **inherits its own** `tools` /
  `toolsSettings` / `allowedTools`.

### Concurrency & permission gating
- The main agent can spawn **multiple subagents concurrently** (reported max **4**
  at once; monitor live with **Ctrl+G**, results combined on completion).
- Subagent permissions live under **`toolsSettings.subagent`**: restrict *which*
  agents can be spawned, which run **without prompting**, and what each may do.
- **Nesting (a subagent spawning further subagents): NOT documented** — verify per
  release before designing a multi-level orchestrator on Kiro.

> **Porting vs. Claude Code:** an IDE Kiro subagent (`.kiro/agents/<name>.md`,
> frontmatter `name`/`description`/`tools`/`model`) maps almost 1:1 to a Claude
> subagent (`.claude/agents/`). **But context differs:** Kiro docs say steering
> reaches subagents (≈ Claude's `CLAUDE.md` hierarchy — but field-disputed, verify
> on your install), while Kiro **Specs & Hooks do not** reach subagents. Tool-gating keys differ entirely (Kiro `tools`/`allowedTools`/
> `toolsSettings` vs Claude `tools`/`disallowedTools`/`permissionMode`). The CLI's
> JSON schema has no Claude equivalent — don't port it as-is.

## Steering files

- **Location:** `.kiro/steering/` (workspace) or `~/.kiro/steering/` (global).
  Workspace **overrides** global on conflict. **Team scope:** distributed via
  MDM / Group Policies into `~/.kiro/steering/`.
- **Default trio** (Kiro generates): `product.md`, `tech.md`, `structure.md` —
  loaded into **every** interaction by default.
- **Inclusion modes** via YAML frontmatter (must be first, no leading blank line):

  | Mode | Frontmatter | Behavior |
  |---|---|---|
  | Always (default) | `inclusion: always` | Every interaction |
  | File match | `inclusion: fileMatch` + `fileMatchPattern: "pattern"` | When working with matching files (**supports array syntax**) |
  | Manual | `inclusion: manual` | On demand via `#steering-file-name` in chat |
  | Auto | `inclusion: auto` + `name:` + `description:` | Auto-included when the request matches |

- **File references:** inline a live workspace file with `#[[file:<relative_path>]]`
  (e.g. `#[[file:api/openapi.yaml]]`) — points at the real artifact, not a stale copy.

## Specs

Spec-driven flow: **requirements → design → tasks**, as structured docs that drive
implementation.
- **Artifacts:** `requirements.md` (or `bugfix.md` for bugfix specs — user stories /
  acceptance criteria / bug analysis), `design.md` (architecture, sequence diagrams),
  `tasks.md` (discrete, trackable implementation plan). *(Exact paths not pinned in
  the docs; conventionally `.kiro/specs/<feature>/…`.)*
- **Spec types:** Feature Specs (Requirements-First / Design-First / Quick Plan) and
  Bugfix Specs.
- **Task execution:** real-time status tracking; Kiro builds a **dependency graph**
  and runs independent tasks **concurrently in waves** (wave 1 = no-dependency tasks).
- Specs do **not** reach subagents (see above).

## Hooks

Agent workflows triggered by events — harness-driven automation, not prompt text.
- **Triggers:** file **save / create / delete**; **user prompt submission** &
  **agent turn completion**; **before/after tool invocation**; **before/after spec
  task execution**; **manual**.
- **Defined** by natural-language generation or a manual form: **Title, Event,
  Action** (`Ask Kiro` agent-prompt | `Run Command` shell), plus **File pattern** /
  **Tool name** matchers. (Storage path/keys for IDE hooks not documented; CLI
  agents carry hooks inline under the `hooks` JSON key — see CLI config above.)
- Hooks do **not** trigger in subagents (see above).

## Agent Skills (added 2026-02-05)

Kiro implements the **open Agent Skills standard** (`agentskills.io`) — same
`SKILL.md` format as Claude Code / OpenCode. On-demand *capabilities*, distinct
from always-on *steering* (the docs draw this line deliberately).
- **Location:** `.kiro/skills/<name>/SKILL.md` (workspace) or `~/.kiro/skills/`
  (global). Workspace **overrides** global on name conflict.
- **Required frontmatter:** `name` (= folder name; lowercase/numbers/hyphens, ≤64
  chars) and `description` (≤1024 chars). **Optional:** `license`, `compatibility`,
  `metadata`.
- **No tool-restriction field documented** — unlike Claude Code's `allowed-tools`.
  A skill sandboxed via `allowed-tools` elsewhere is **not** sandboxed in Kiro;
  re-audit security-sensitive skills on port.
- **Progressive disclosure:** reference files load **only when the SKILL.md body
  explicitly directs it** — don't rely on a whole folder being auto-ingested.

## AGENTS.md

Kiro reads the `AGENTS.md` standard — **loads always, no inclusion modes** (unlike
steering, which gates via frontmatter).
