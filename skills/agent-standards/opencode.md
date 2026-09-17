# OpenCode — reference

> **Verified: 2026-06-20** against `opencode.ai/docs/agents`, `opencode.ai/docs/permissions`,
> `opencode.ai/docs/rules`. **Skills + MCP servers verified 2026-07-25** against
> `opencode.ai/docs/skills` and `opencode.ai/docs/mcp-servers` (own sections below).
> **Subagent nesting re-verified 2026-09-17 two ways:** first against the live docs
> (they turned out silent on it, not confirming either the old "documented: yes" or
> a "no" — downgraded to "undocumented" at that point), **then settled empirically**
> the same day by actually running it (`opencode debug agent --tool task` + a full
> `opencode run`, v1.18.30) — see "Primary vs. subagent" below for the confirmed
> mechanism. The context-inheritance ("task continuity") claim in "What reaches an
> OpenCode subagent" is still only docs-downgraded, not empirically re-tested.
> Re-verify before relying on an exact key — OpenCode's field set moves (note the
> `tools`→`permission` deprecation below).

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
- `prompt` — system-prompt file ref, form `{file:./prompts/build.txt}`. **Verified 2026-09-12:**
  a `{file:./path}` reference and literal text can be mixed in one string
  (`{file:./persona.txt}\n\nsome literal text`) — the file's content and the literal text are
  concatenated, so a shared persona file can be live-included without a build/generation step.
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
- **`"ask" hangs a headless run forever (verified v1.18.30).** A `permission.bash`/`write`/`edit`
  rule set to `"ask"` with no human present hangs `opencode run` indefinitely (confirmed by
  timeout — no fallback, no auto-deny). `"deny"` instead lets the tool call fail gracefully: the
  run completes with a narrated refusal, exit 0. Any headless/unattended agent config must avoid
  `"ask"` on a path the run can actually reach.
- **Glob `"*"` matches across a command's own flag/argument boundaries, not just within one path
  segment (verified v1.18.30) — a security-relevant gotcha for any `bash` allow pattern gating a
  multi-flag command.** For a command with a repeatable, last-value-wins scoping flag (e.g.
  `docker compose -f`/`--project-directory`), a pattern like
  `"docker compose -f <dir>/* --project-directory <dir>/* down"` can be satisfied by a string that
  *also* contains a second, fully-formed `-f`/`--project-directory` pair inside the wildcard span —
  Compose then executes against the last (attacker-chosen) `--project-directory`, not the one the
  pattern text appears to scope to. No shell metacharacter is involved, so a metacharacter-chaining
  deny net (`&&`, `;`, `|`, `` ` ``, `$(`, `>`) does not catch it. **The fix that actually closes
  it:** never let `permission.bash` glob-match a model-authored string for this class of command at
  all — route it through a fixed wrapper script that looks up a caller-provided slug against a
  small, code-reviewed allow-list and constructs the full argv itself, and scope the agent's `bash`
  permission to only the wrapper script's own absolute path (live-reproduced and closed:
  `opencode/docs/reviews/devops-opencode-headless.md`, `opencode/docs/plans/devops-opencode-headless.md`).
- **A live-testing gotcha for `opencode debug agent --tool bash --params` probes (verified
  2026-09-13): reusing a fixture path/name across repeated probe calls in the same project cwd can
  spuriously DENY an otherwise-allowed command**, via OpenCode's internal `doom_loop`/
  `external_directory` permission layers rather than the agent config actually under test — a false
  negative that looks exactly like the permission rule under test doing its job. Use a fresh,
  never-reused fixture name for each probe to avoid the false reading; confirmed by re-running an
  identical smuggling shape against a never-reused path immediately after a reused one was denied.

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
- **Nesting: YES — empirically confirmed 2026-09-17, v1.18.30, undocumented on the
  live pages (docs gap, not a docs contradiction).** The docs never state this either
  way (see prior note in this file's history); a direct three-legged probe settled
  it:
  1. `opencode debug agent <subagent-name> --tool task --params '{"description":...,
     "prompt":...,"subagent_type":"<other-subagent>"}'` — invoking the **Task tool
     directly as a `mode: subagent` agent** (not a primary) creates a real **child
     session** for the target subagent, `parentID` correctly set to the calling
     subagent's own session. Confirmed via `opencode export <sessionID>` on the
     child. (The probe then errors — `TaskTool requires promptOps in ctx.extra` —
     but that's the bare `debug agent --tool` harness lacking plumbing the real
     agentic loop has; it doesn't touch permission/nesting at all.)
  2. A full `opencode run` (primary → subagent A → subagent B) reproduced the same
     nesting live through the real loop, independently confirming session creation
     works the same way outside the debug probe.
  3. **The actual gate is `permission.task` on the *invoked* agent's own config —
     mode (`primary` vs `subagent`) never enters into it.** A Task-spawned child
     session's permission set is the target agent's normal resolved permissions
     **plus a harness-injected override list** (`question: deny`, `plan_enter: deny`,
     `plan_exit: deny`, `todowrite: deny`, and — only when the target agent's own
     config does **not** explicitly set `permission.task` — `task: deny` too).
     Concretely: a subagent with `"permission": {"task": "allow"}` in its own
     frontmatter/config **keeps `task: allow`** once spawned as a child (its
     explicit grant survives the override layer) and can go on to invoke a further
     subagent itself; a subagent with no explicit `permission.task` gets `task:
     deny` injected the moment it's running *as a Task-spawned child* — even though
     that same agent, run standalone/top-level, resolves `task` to the ordinary
     global `"*": "allow"` default. **So nesting depth is not hardcoded-capped at
     one level** — it goes exactly as deep as each successive agent's own config
     explicitly re-grants `permission.task`; the default is a safety rail (don't
     recurse unless you opted in), not a ceiling.
  - **Reproduction recipe** (throwaway `opencode.json`, no real API keys needed —
    works against a local LM Studio provider): define agent A (`mode: subagent`,
    `permission.task: "allow"`) and agent B (`mode: subagent`, sentinel prompt), then
    run leg 1 above with A as `<subagent-name>` and B as `subagent_type`. Full
    end-to-end text round-trip (B's actual reply flowing back through A) needs a
    model context window generous enough for OpenCode's own tool-calling system
    prompt — an 8K-context local model overflowed before completing the hop
    (`exceed_context_size_error`, consistent with the ≥16K guidance already in
    `opencode/docs/manuals/local-llm.md`); that's an unrelated model-sizing limit,
    not a nesting limit — the session-creation and permission-override behavior
    above is independent of it and was confirmed without needing the round-trip to
    finish.

### What reaches an OpenCode subagent (verify — notable divergence)

⚠️ **Re-verified 2026-09-17: the "task continuity" claim below no longer has
textual support on the live docs page** — a targeted re-fetch found no sentence
describing whether a subagent inherits the parent session's conversation history
or file context; the mechanics of what the Task tool passes into a subagent's
session are simply not described. Older phrasing here read that as an implicit
"yes, subagents receive parent context" — **do not rely on that** without a live
probe. This still diverges from Claude Code, where a subagent's isolation *is*
explicitly documented (no parent conversation). The rules doc **does not state**
whether `AGENTS.md` propagates to subagents — treat that as unverified too.

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
permission key above. *(Commands last verified 2026-06-07; the skill specifics
below re-verified 2026-07-25 against `opencode.ai/docs/skills`.)*

### Skills — what OpenCode reads (verified 2026-07-25)

- **Search paths**, project first (walking up to the git worktree root) then
  global: `.opencode/skills/<name>/`, `.claude/skills/<name>/`,
  `.agents/skills/<name>/`; `~/.config/opencode/skills/<name>/`,
  `~/.claude/skills/<name>/`, `~/.agents/skills/<name>/`. So a Claude-Code skills
  tree is picked up without any porting step.
- **Recognized frontmatter:** `name` (required, 1–64, lowercase/hyphens),
  `description` (required, 1–1024), `license`, `compatibility`, `metadata`.
  **Unknown frontmatter fields are ignored, not rejected** — which is why a
  Claude-specific `allowed-tools:` (including an `mcp__*` tool name) is safe to
  leave in a shared `SKILL.md`. It simply has **no effect** here; gating is
  `permission.skill` patterns (`{"*": "allow", "internal-*": "deny"}`), global or
  per-agent.
- Invocation is an explicit native tool: `skill({ name: "<skill-name>" })`.
- ⚠ **Observed 2026-07-25 in this repo:** repeated `opencode debug skill` runs
  over a whole-directory skills symlink returned **different subsets** of the same
  9 skills, with no error or warning. Treat a skill missing from one enumeration
  as noise, not as a broken skill — and don't build a portability conclusion on a
  single run.

## MCP servers (verified 2026-07-25 against `opencode.ai/docs/mcp-servers`)

**MCP config does not port between harnesses even though `SKILL.md` does.**
OpenCode reads neither Claude Code's `.mcp.json` nor Kiro's
`~/.kiro/settings/mcp.json`; it configures servers under the top-level **`"mcp"`**
key of `opencode.json` / `opencode.jsonc`:

```json
{
  "mcp": {
    "<server-name>": {
      "type": "local",
      "command": ["npx", "-y", "some-server"],
      "cwd": "optional-working-dir",
      "environment": { "VAR": "value" },
      "enabled": true,
      "timeout": 5000
    },
    "<remote-name>": {
      "type": "remote",
      "url": "https://example.com/mcp",
      "headers": { "Authorization": "Bearer …" },
      "oauth": false,
      "enabled": true,
      "timeout": 5000
    }
  }
}
```

Note the divergences from Claude Code, each a real porting trap:

| | Claude Code | OpenCode |
|---|---|---|
| Config home | `.mcp.json` (project) / `~/.claude.json` | `opencode.json` → `"mcp"` key |
| stdio command | `"command"` string + `"args"` array | **`"command"` is a single array** (`["npx","-y","x"]`) |
| Server env | `"env"` | **`"environment"`** |
| Local vs remote | inferred / `"type": "http"\|"sse"\|"ws"` | explicit **`"type": "local"\|"remote"`** |
| Tool name | `mcp__<server>__<tool>` | **`<server>_<tool>`** — prefix is the server name; disable with patterns like `"<server>_*": false`, or enable per-agent in its tool settings |

Consequence for shared skills: a `SKILL.md` deployed to all three harnesses that
routes work through an MCP tool must keep a non-MCP fallback documented, because
the wiring reaches only the harness it was written for. In this repo the `cypher`
server is wired for Claude Code only (repo-root `.mcp.json`); `cpg-analysis` keeps
`redis-cli GRAPH.QUERY` for exactly that reason, and OpenCode/Kiro wiring is
tracked as backlog **C-310**.

## Severino gotchas (retired local LM-Studio agent — kept for the config-schema lesson)

`severino` itself is retired (`deprecated/opencode/agents/severino/`), but the schema pitfalls it
surfaced still apply to any OpenCode agent using an LM Studio provider:

- Top-level config key is **`agent`** (singular), **no `name`** field on the agent.
- Model id form `lmstudio/<model-id>`; LM Studio context **≥16K** or OpenCode's
  system prompt overflows (`n_keep >= n_ctx`).
- See `deprecated/opencode/agents/severino/AGENTS.md` (its `CLAUDE.md` is a `@AGENTS.md` import
  stub) and `opencode/docs/manuals/local-llm.md` for the live walkthrough.
