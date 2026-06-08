# OpenCode — reference

> **Verified: 2026-06-07** against `opencode.ai/docs/agents` and `opencode.ai/docs/rules`.
> Re-verify before relying on an exact key — OpenCode's field set moves.

## Agents

Defined **two ways**:

1. **Markdown files** — `~/.config/opencode/agents/` (global) or `.opencode/agents/`
   (project). The **filename becomes the agent name** (`review.md` → `review`).
2. **`opencode.json`** — under the top-level **`"agent"`** key (singular):
   ```json
   { "agent": { "review": { /* config */ } } }
   ```

### Config / frontmatter fields (verified 2026-06-07)

- `description` — **required**; agent purpose (routing signal for `@`-invocation).
- `mode` — `"primary"` | `"subagent"` | `"all"` (**default `"all"`**).
- `model` — `provider/model-id` (e.g. `anthropic/claude-...`, `lmstudio/<id>`).
- `temperature` — randomness `0.0`–`1.0`. `top_p` — alternative randomness control.
- `prompt` — path to a system-prompt file.
- `tools` — enable/disable specific tools.
- `permission` — granular tool gating (see below).
- `disable` — `true` to disable the agent.
- `hidden` — hide a subagent from the autocomplete menu.
- `color` — UI appearance.
- `steps` — max agentic iterations.

### Permission sub-keys (verified 2026-06-07)

Each takes `"allow"` | `"ask"` | `"deny"`. Keys (more granular than just
edit/bash): `read`, `edit`, `glob`, `grep`, `list`, `bash`, `task`,
`external_directory`, `todowrite`, `webfetch`, `websearch`, `lsp`, `skill`,
`question`, `doom_loop`.

### Primary vs. subagent

- **Primary agents** — the assistants you talk to directly (cycle with **Tab**),
  e.g. Build, Plan.
- **Subagents** — invoked by a primary agent automatically or via **`@mention`**
  (`@review check this`); inherit the invoking agent's model unless overridden;
  hide from autocomplete with `hidden: true`.
- `mode: "all"` (the default) means the agent can act as either.

## Rules / memory — `AGENTS.md`

- **Project:** `AGENTS.md` in the project root. **Global:** `~/.config/opencode/AGENTS.md`.
- Created/updated via the **`/init`** command (scans the repo for build/lint/test
  commands, architecture, conventions, gotchas).
- **Precedence** (first match wins per category): (1) local files, traversing up
  from cwd — `AGENTS.md` then `CLAUDE.md`; (2) global `~/.config/opencode/AGENTS.md`;
  (3) Claude Code fallback `~/.claude/CLAUDE.md` (unless disabled). If both
  `AGENTS.md` and `CLAUDE.md` exist locally, **only `AGENTS.md` is used.**
- `instructions` field in `opencode.json` references additional external rule files.

## Commands & Skills

OpenCode supports **commands** (custom slash commands) and **Skills** (the same
open Agent Skills `SKILL.md` standard). Skill tool-gating is governed by the
`skill` permission key above.

## Severino gotchas (local LM-Studio agent in this repo)

- Top-level config key is **`agent`** (singular), **no `name`** field on the agent.
- Model id form `lmstudio/<model-id>`; LM Studio context **≥16K** or OpenCode's
  system prompt overflows (`n_keep >= n_ctx`).
- See `opencode/agents/severino/CLAUDE.md`.
