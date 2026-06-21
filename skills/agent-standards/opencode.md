# OpenCode — reference

> **Verified: 2026-06-20** against `opencode.ai/docs/agents`, `opencode.ai/docs/permissions`,
> `opencode.ai/docs/rules`. Re-verify before relying on an exact key — OpenCode's
> field set moves (note the `tools`→`permission` deprecation below).

## Agents

Defined **two ways**:

1. **Markdown files** — `~/.config/opencode/agents/` (global) or `.opencode/agents/`
   (project). The **filename becomes the agent name** (`review.md` → `review`).
2. **`opencode.json`** — under the top-level **`"agent"`** key (singular):
   ```json
   { "agent": { "review": { /* config */ } } }
   ```

### Config / frontmatter fields (verified 2026-06-20)

- `description` — **required**; agent purpose (routing signal for `@`-invocation & auto-delegation).
- `mode` — `"primary"` | `"subagent"` | `"all"` (**default `"all"`**).
- `model` — `provider/model-id` (e.g. `anthropic/claude-...`, `lmstudio/<id>`); overrides global.
- `temperature` — `0.0`–`1.0` (defaults model-specific). `top_p` — alt diversity control.
- `prompt` — system-prompt file ref, form `{file:./prompts/build.txt}`.
- `permission` — granular tool gating (see below) — **the current way to gate tools.**
- `tools` — ⚠️ **DEPRECATED** (`{tool: true/false}` enable map). **Use `permission` instead.**
- `disable` — `true` to disable the agent.
- `hidden` — `true` hides a **subagent** from `@` autocomplete (subagents only).
- `color` — hex, or a theme name (`primary`/`secondary`/`accent`/`success`/`warning`/`error`/`info`).
- `steps` — max agentic iterations before a text-only response.

### Permissions (verified 2026-06-20)

Each key resolves to `"allow"` | `"ask"` | `"deny"`. **Keys:** `read`, `edit`
(edit/write/patch), `glob`, `grep`, `list`, `bash`, `task` (**launching
subagents**), `external_directory`, `lsp`, `skill`, `todowrite`, `webfetch`,
`websearch`, `question`, `doom_loop` (≥3 identical repeated calls).

- **Glob/pattern control** (object syntax) is supported by: `read, edit, glob,
  grep, list, bash, task, external_directory, lsp, skill`. **Last matching rule
  wins** — put `"*"` first, then specific patterns:
  ```json
  "permission": { "bash": { "*": "ask", "git status *": "allow", "rm *": "deny" } }
  ```
- **Scope:** global (`"*"` wildcard / per-tool) **and** per-agent; **agent config
  overrides/merges over global**, agent winning on conflict.
- **Defaults:** most keys default to **`allow`**; `doom_loop` and
  `external_directory` default to **`ask`**; **reading `.env` defaults to `deny`.**

### Primary vs. subagent (verified 2026-06-20)

- **Primary agents** — talk to directly (cycle with **Tab** / `switch_agent`).
  Built-ins: **`build`** (all tools, default) and **`plan`** (restricted: edit/bash
  → `ask`).
- **Subagents** — invoked by a primary **automatically** (by `description`) or
  manually via **`@mention`** (`@general help me search…`). Built-ins: **`general`**
  (full access), **`explore`** (read-only codebase), **`scout`** (read-only
  external docs/deps). Inherit the invoker's model unless overridden; `hidden: true`
  removes from autocomplete.
- `mode: "all"` (default) → the agent can act as either.
- **Nesting: YES (documented).** Subagents can invoke other subagents via the
  **Task tool**, gated by **`permission.task`** glob patterns. (Contrast Kiro:
  nesting undocumented.) Subagents run as **child sessions** (navigable via
  `session_child_first` / `session_parent` keybinds).

### What reaches an OpenCode subagent (verify — notable divergence)

⚠️ The agents doc indicates a subagent **receives the parent session's conversation
history and file context** ("task continuity"). **This diverges from Claude Code**,
where a subagent does *not* see the parent conversation. The phrasing is loose and
this is consequential — **verify on your version** before relying on either passing
context implicitly *or* on isolation. The rules doc **does not state** whether
`AGENTS.md` propagates to subagents — treat that as unverified too.

## Rules / memory — `AGENTS.md`

- **Project:** `AGENTS.md` in project root. **Global:** `~/.config/opencode/AGENTS.md`.
- Created/updated via **`/init`** (scans repo for build/lint/test, architecture,
  conventions; may ask targeted questions).
- **Precedence** (first match wins per category): (1) local files traversing up
  from cwd — `AGENTS.md` then `CLAUDE.md`; (2) global `~/.config/opencode/AGENTS.md`;
  (3) Claude Code fallback `~/.claude/CLAUDE.md` (unless disabled). If both
  `AGENTS.md` and `CLAUDE.md` exist locally, **only `AGENTS.md` is used.**
- **`instructions`** field in `opencode.json` references additional rule files via
  **glob patterns or remote URLs** (remote = 5 s fetch timeout); all are **combined
  with** your `AGENTS.md` files:
  ```json
  "instructions": ["CONTRIBUTING.md", "docs/guidelines.md", ".cursor/rules/*.md"]
  ```

## Commands & Skills

OpenCode supports **commands** (custom slash commands) and **Skills** (the open
Agent Skills `SKILL.md` standard). Skill tool-gating is governed by the `skill`
permission key above. *(Skills section last verified 2026-06-07 — lighter touch
this pass.)*

## Severino gotchas (local LM-Studio agent in this repo)

- Top-level config key is **`agent`** (singular), **no `name`** field on the agent.
- Model id form `lmstudio/<model-id>`; LM Studio context **≥16K** or OpenCode's
  system prompt overflows (`n_keep >= n_ctx`).
- See `opencode/agents/severino/AGENTS.md` (its `CLAUDE.md` is a `@AGENTS.md` import stub).
