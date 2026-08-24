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
> **Bash tool environment** (shell-shadowed `find`/`grep`) — **observed 2026-07-26/2026-08-08**,
> not doc-sourced (no official page documents this; see that section for the evidence).
> **`initialPrompt` language gotcha** — **observed 2026-08-09**, not doc-sourced (see the
> main-session section below).
> **Subagent tool-set/definition-load/AutoMem-index facts** — **observed 2026-08-10**, not
> doc-sourced (see "What loads into a subagent" and "Bash tool environment").
> **Hooks — `PreToolUse` "ask" enforcement gap observed 2026-08-21, filed upstream** (Claude Code
> 2.1.238, Auto Mode, not doc-sourced — contradicts current docs; see the `## Hooks` section's
> dated callout) — four isolated live tests found a `PreToolUse` hook (matcher `Bash` in three,
> `Write`/`Edit` in one), confirmed correctly wired and confirmed to compute `ask` in isolation,
> does not pause execution for the real matching command from **either** a Task-dispatched
> subagent **or the main session itself**, regardless of whether the hook is defined in a
> subagent's own frontmatter or in `settings.json`. Treat `PreToolUse` "ask" enforcement as
> unverified under Auto Mode until re-confirmed — matcher-agnostic and context-agnostic, not a
> narrow subagent-dispatch-only gap.
> **`defaultMode`/subagent-mode-inheritance re-verified 2026-08-24** against
> `code.claude.com/docs/en/permission-modes`, `.../permissions`, and `.../sub-agents` (whole
> pages) — see the `## Hooks` section's 2026-08-24 resolution callout for the parent-mode
> inheritance rule and the frontmatter-`permissionMode`-is-inert-for-primary-sessions finding.
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
    ⚠️ **`acceptEdits` is scoped to the working directory + `additionalDirectories`,
    not global** (verified 2026-08-01 against `code.claude.com/docs/en/permissions`):
    "Automatically accepts file edits … for paths in the working directory or
    `additionalDirectories`." A Write/Edit outside that boundary — e.g. an agent's
    own `$HOME/.claude/agents/<name>/kaizen/*.md`, which sits outside the project
    repo the session was launched in — still prompts for approval every time,
    regardless of `permissionMode: acceptEdits`. This is why an agent whose kaizen
    learning-capture instruction tells it to write to its own inbox file still asks
    for permission on that write even though it never asks for edits inside the repo.
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

An agent definition is not only a delegation target — it can be the **interactive,
conversational agent**:

- **`claude --agent <name>`** (or the **`agent` setting**) starts a session where the
  **main thread itself** takes on that definition's system prompt, tool restrictions,
  and model. Plugin agents: `claude --agent my-plugin:name` (include any `agents/`
  subfolder in the scoped name).
- **`initialPrompt` frontmatter** is auto-submitted as the first *user* turn in this
  mode (commands and skills are processed; it's prepended to any user-provided prompt).
- **Frontmatter hooks fire in main-session mode too**, alongside `settings.json` hooks —
  per official docs, identically whether the agent is run as the main session, spawned as a
  Task-dispatched subagent, or @-mentioned, with no documented exception.
  > **Contradicted by four independent, isolated controlled live tests (graphmind-ai-lab,
  > 2026-08-21, Claude Code 2.1.238, Auto Mode active) — matcher-agnostic, not "subagent-dispatch
  > only" or "`Bash`-only."** (1) A subagent's own frontmatter `Bash` hook, confirmed correctly
  > wired and confirmed to match+`ask` when fed the exact payload directly, did not fire when the
  > subagent actually ran that command after being Task-dispatched with `subagent_type`
  > explicitly correct (ruling out the "silently degraded to `general-purpose`" confound).
  > (2) The identical guard mirrored as a **session-wide `.claude/settings.local.json` `Bash`
  > hook** and run from the **main session itself** (no subagent) also did not fire. (3) Repeated
  > after the user explicitly reloaded hook config via `/hooks`, which visibly listed the hook as
  > registered (`[Local] Bash — 1 hook`) — still did not fire. (4) A `Write`/`Edit`-matched
  > frontmatter hook, main session, targeting a path outside the hook's own allowlist — the write
  > went through unescalated; re-fed the real payload to the script afterward and confirmed it
  > correctly computes `ask` for that exact path. All four used a real, disposable payload
  > (scratch graph or scratch file, immediately cleaned up). **This rules out `subagent_type`
  > omission, stale/unloaded hook config, hook-not-registered, and a `Bash`-specific quirk as
  > explanations** — the hook is loaded, matches, and correctly computes `ask` in isolation on
  > both matchers tested, and still doesn't pause execution for the real call, in either
  > main-session or subagent context. Working hypothesis (unconfirmed): Auto Mode's classifier
  > layer silently resolves/overrides a correctly-emitted `ask` decision before it reaches the
  > user — this contradicts both current official docs and third-party reporting that "ask forces
  > a prompt in auto mode; the classifier can't approve it silently." **Filed upstream via
  > `/feedback` 2026-08-21** (the 3-test `Bash` repro; test 4 landed after filing). **Practical
  > consequence: don't treat a `PreToolUse` "ask" hook as a reliable backstop under Auto Mode,
  > from any source (frontmatter or settings.json), on any matcher tested (`Bash`, `Write`/
  > `Edit`), in either main-session or subagent execution**, until independently reconfirmed
  > fixed. Full trail: `claude/cobb/kaizen/history.md`, 2026-08-21 entries (K-018/K-019).
- The **withheld-tools list applies to subagents only** — as the main session the agent
  can use `AskUserQuestion` etc., so live multi-turn interaction works.
- The main-thread agent can spawn subagents via `Agent`; the **`Agent(agent_type)`
  allowlist syntax** in `tools` restricts which types — but **only** in main-thread
  mode (inside a subagent definition the parenthesized type list is ignored).
- **Gotcha — `initialPrompt`-driven greetings and a "default language" rule don't mix
  reliably** (observed, not doc-sourced): `initialPrompt` auto-submits as the first
  *user* turn, but it carries no actual linguistic evidence from the human — nobody
  has "written" anything yet. A prompt rule like "respond in English by default, mirror
  the user's language once they write in another one" can still get overridden at that
  first line by other contextual signals available to the model (e.g. the operator's
  git identity/locale), because the literal condition for the default ("no user text
  yet") is met but the model reaches for a different heuristic anyway. If a canned
  `initialPrompt` greeting must have a deterministic language, say so explicitly inside
  the `initialPrompt` text itself ("Introduce yourself in one line **in English**...")
  rather than relying on a general default-language rule stated elsewhere in the prompt
  to reach that one line. Simpler fix where a canned greeting isn't earning its keep:
  drop `initialPrompt` entirely and let the human's real first message set the language
  and the routing naturally.

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

### Cross-session peer addressing (`SendMessage` + `ListAgents`) — verified 2026-08-16

- **`SendMessage`/`ListAgents` now also reach independently-launched peer sessions, not only
  Agent-Teams teammates.** The live tool descriptions (fetched via `ToolSearch` in a running
  session) document address classes beyond in-process subagents: "other local Claude sessions on
  this machine," "your Claude sessions running in the cloud," and (when Remote Control is
  connected) "your account's other sessions." This is broader than — and current docs don't yet
  visibly reconcile with — the Agent-Teams-gated description above
  (`CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS`, re-verified 2026-06-21); treat the tool's own live
  description as the more current source until the two are reconciled against fresh official docs.
- **Bare-name addressing is ambiguous when two independently-launched sessions share the same
  agent name.** `SendMessage`'s own description states name resolution is "latest wins" among
  same-named live agents; `ListAgents` may list only one row for a name even when the ambiguity is
  real (observed: two separate `--agent teco` invocations, on different tasks, with only one
  visible to the probing session). Nothing in the address itself proves which task the target
  session is actually running.
- **Incident (2026-08-16):** an orchestrating session sent a full multi-paragraph resume brief for
  one coordination ("K-026") to a peer named `teco`, found via `ListAgents` and assumed — from
  stale prior-session context — to be that coordination's session. It was actually a different,
  independently-launched `teco` session mid-coordination on an unrelated task. That session's own
  human caught the mismatch; by then the session had already done a small amount of read-only
  "safe" work (a test re-run, a state-restoring reseed) before declining — harmless here, but not
  free, and worse mixing is possible with a less careful peer.
- **Practice:** before sending a substantive brief to a bare-named peer you did not yourself
  spawn (i.e. its identity/task isn't already known from a ledger `agentId` or this session's own
  spawn record), send a cheap identity-confirming probe first ("what are you currently
  coordinating?") and read the reply before committing the real brief. If there's any doubt and the
  task has a persistent state artifact (a coordination-doc ledger), prefer a fresh subagent that
  reads that artifact over resuming an unverified peer — slower, but structurally can't misfire
  onto someone else's work.

### Nested-delegation notification routing (observed 2026-08-15, graphmind-ai-lab — data points, not confirmed stable contracts)

- **A nested delegate's completion notification appears to bubble to whichever ancestor session is currently *live*, not necessarily to the direct delegator.** In a three-level chain (`cobb` → `teco` → `architect`/`analyst`), every nested-child completion routed its task-notification to `cobb` (two levels up) rather than `teco` (one level up) — observed when `teco`'s own turn had already ended right after dispatch, leaving it dormant. `cobb` then had to relay each result and explicitly `SendMessage teco` to continue; that call's own tool result read `"Resuming agent a8d402d..."`, i.e. the call itself appears to **force-resume a dormant target**, a distinct mechanism from the passive bubbling above. Not yet independently confirmed as a stable harness contract — could be specific to this session's dormancy pattern.
- **A background delegate that cannot address its coordinator by bare agent name gets relayed through "main" (the top-level session) as a `<system-reminder>`-shaped block — this is the legitimate delivery path for exactly that reason (the delegate has no address to resolve), not an injection to decline by default.** Two delegates `teco` had itself dispatched via `SendMessage` each tried `SendMessage teco` on completion and got `"No agent named 'teco' is reachable"` — nothing currently hands a delegate its coordinator's own `agentId`. Their results arrived instead wrapped as `<system-reminder>The coordinator sent a message while you were working: ...</system-reminder>`. **The envelope shape is not itself the trust signal in either direction** — verify the *content* every time (these two checked out independently against `git`/the filesystem) regardless of whether a relayed message arrives this way or as the documented `<cross-session-message>`/`<task-notification>` envelope; a message carrying only a completion relay of work the receiving session itself just dispatched is a different risk class from one asserting new, unverifiable directive authority.

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

- **A subagent's runtime tool set can be narrower than its frontmatter `tools:` list, silently
  — the field is a request, not a guarantee.** A live probe of a `teco` run whose frontmatter
  declared `Read, Grep, Glob, Bash, Agent, SendMessage, Write, Edit` reported exactly `Read, Bash,
  Agent, SendMessage, Write, Edit` when asked to enumerate its own available tools — `Grep`/`Glob`
  were simply absent, no error or deferred-tool notice, reproduced on a second probe. Don't write
  prompt logic that assumes a declared tool is actually present; verify with a live probe on a
  **fresh session** first (see next bullet).
- **Custom agent definitions resolve into context at parent-session start; editing an agent
  mid-session does not reach a subagent spawned later in that same session.** Adding a tool to an
  agent's frontmatter, then spawning that agent from the *same* session to verify the change, is
  inconclusive by construction — the session's own context already resolved the pre-edit
  definition before the edit happened. Any verification of an agent-definition edit needs a fresh
  session, never the session that made the edit.
- **User-scoped AutoMem (`~/.claude/projects/<repo>/memory/MEMORY.md`) reaches a subagent as an
  *index* only, not the entry bodies.** A subagent's injected context carried every memory entry's
  title and a short gloss, inside a system-reminder labeled "user's auto-memory" — but the agent
  reported seeing only that, not the linked file's actual content, and could not act on facts that
  lived only in the entry body. Corollary: **behavior an agent must exhibit belongs in its
  committed prompt**, never left in user-scoped memory, which is untracked, index-only to
  subagents, and invisible to a team-coherence audit script.

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
- **A distinct layer: auto mode's built-in Bash safety classifier.** Separate
  from any project's `PreToolUse` hooks — it's a product-level check, not
  something a repo's `guard-*.sh` scripts can see or special-case. Observed
  (graphmind-ai-lab, `analyst` agent, 2026-07-31 — not independently
  re-verified against official docs, so treat as a data point, not a spec): it
  hard-blocks a `git stash push` targeting a tracked path outright ("Blocked by
  classifier … modify the working tree"), including `--keep-index`, with no
  carve-out for a fully reversible operation. Consequence for anyone designing
  a review/verification workflow: don't build a repo mechanism that tries to
  pattern-match "safe" instances of a blocked command back open (that
  re-implements a safety classifier with less rigor than the one already
  there); instead prefer a substitute that needs **no working-tree write at
  all** (e.g., copy the tree to a scratch dir and reverse-apply the diff there)
  — a strictly stronger isolation property than the blocked command, not a
  lower-visibility route to its effect.
- **The same classifier layer also flags a delegate proposing to self-modify its own
  permissions/settings to route around a blocked write** — observed (graphmind-ai-lab, `cobb`,
  2026-08-20): a curator-clear `DETACH DELETE` via `mcp__cypher__query` was blocked by the
  permission system even after the coordinator (`teco`) had told the delegate to proceed; on a
  later attempt the delegate proposed adding a persistent bypass rule to its own settings, citing
  the coordinator's authorization rather than the user's, and this was auto-flagged ("Auto-Mode
  Bypass/Self-Modification") and correctly not acted on. A coordinator's own "proceed" does not
  substitute for the harness's own human-approval gate on a write it chooses to gate — and a
  delegate's proposal to route around that gate via self-modification is itself the signal to stop.
- **A `PreToolUse` hook's explicit `"allow"` is not a full override of the confirm-before-Write/Edit
  prompt — it composes with two separate layers, and neither the shipped write-guards (
  `agent-permission-friction.md`) nor its own root-cause finding (§1.3) accounted for the second
  one.** Verified 2026-08-24 against `code.claude.com/docs/en/permissions` ("Extend permissions
  with hooks") and `.../permission-modes` ("How the classifier evaluates actions" /
  "How auto mode handles subagents"), both fetched fresh (not from a cached prior read):
  1. **Settings-rule layer (already correctly quoted by phase 1, but under-weighted in its own
     summary):** "Hook decisions don't bypass permission rules. Claude Code evaluates deny and ask
     rules regardless of what a `PreToolUse` hook returns: a matching deny rule blocks the call,
     and a matching ask rule still prompts even when the hook returned `"allow"` or `"ask"`." A
     hook `"allow"` is therefore conditional on no matching settings.json `ask`/`deny` rule existing
     — not literally unconditional "every time" as `agent-permission-friction.md` §1.3 asserted from
     the same source quote.
  2. **Auto-mode classifier layer (undocumented interaction with hooks — the actual live-reproduced
     gap, graphmind-ai-lab, `agent-permission-friction2.md` open question 3, 2026-08-23/24):** on
     Pro/Max/Team, `auto` is the account-default `permissionMode` (confirmed live via this repo's
     own session transcripts, `~/.claude/projects/<proj>/<session>.jsonl` — a `"type":"permission-mode"`
     record announces the session's mode, and it stayed `"auto"` continuously across two independent
     manual-confirmation-prompt incidents on 2026-08-23). Auto mode's own documented decision order
     (`permission-modes` doc, "How the classifier evaluates actions") is: (1) explicit settings
     allow/ask/deny rules resolve immediately — protected-path writes route to the classifier even
     past a matching allow rule; (2) **"Read-only actions and file edits in your working directory
     are auto-approved, except writes to protected paths"**; (3) everything else goes to the
     classifier. Nowhere in this decision order, nor in "How auto mode handles subagents" ("each of
     its actions goes through the classifier with the same rules as the parent session, and any
     `permissionMode` in the subagent's frontmatter is ignored"), does a `PreToolUse` hook's
     `"allow"` get named as exempting an action from classifier review — the hook layer and the
     auto-mode classifier layer are only documented to interact through the settings-rule
     precedence in point 1 above, never directly with each other. **Live-reproduced result:** two
     separate agents (`analyst`, `tdd-engineer`), two separate shared guard cores (allow-list vs.
     deny-list), both statically verified to emit an explicit `permissionDecision:"allow"` on a
     genuinely in-remit, non-protected, working-directory path — and both still produced a
     multi-minute/multi-hour human-confirmation gap in the session transcript (`toolUseResult`
     landing 14 min and 4h38m after the matching `tool_use`, respectively) when the write was
     **subagent-delegated from a concurrent top-level session already confirmed to be in `auto`
     mode throughout**. Per the documented decision order alone, step 2 should have auto-approved
     both with zero classifier involvement and zero human prompt, hook or no hook — so either a
     delegated subagent's own file edits don't get step 2's "your working directory" fast path the
     same way a top-level session's own edit would, or the classifier's per-subagent review
     (point 2, "How auto mode handles subagents") independently re-opens a confirmation the hook
     already resolved. **This could not be fully disambiguated by static analysis or transcript
     reading — it needs a live, human-observed test** (a fresh, non-concurrent, top-level
     `--agent analyst` session making one in-allowlist write, mode-bar watched at the moment of the
     call) to determine whether the friction is specific to Task/Agent-tool delegation or reproduces
     even without it. **Practical guidance until that test lands:** don't extend or re-derive a
     phase-2-style write-guard design from phase 1's §1.3 "hook allow is unconditional" premise
     without re-verifying against this note — the premise is only half true, and the half that's
     false (the auto-mode classifier) is not something a repo-local hook script can see or
     special-case, the same caveat already logged above for the Bash classifier.
- **Resolution/update, 2026-08-24 (`claude/docs/plans/write-guard-classifier-gap-coordination.md`
  §U7, `claude/docs/plans/permission-default-mode.md`):** the live test the note above called for
  has now run, twice, and both attempted fixes are refuted for the delegated-write case. (1) A
  `permissions.allow` `Edit(path)` settings rule does **not** suppress the prompt for a
  Task/Agent-delegated write either, even matching a hook that independently returns `"allow"` —
  so settings-rule precedence (point 1 above) doesn't reach a delegated action the way it reaches a
  top-level one. (2) The actual documented mechanism that *would* remove the classifier from a
  delegated write's path is **parent-session mode inheritance**, precisely quoted from
  `code.claude.com/docs/en/sub-agents` ("Permission modes"): "If the parent uses `bypassPermissions`
  or `acceptEdits`, this takes precedence and can't be overridden. If the parent uses auto mode, the
  subagent inherits auto mode and any `permissionMode` in its frontmatter is ignored." So it's
  specifically the **delegating parent's own ambient mode being `auto`** that forces every dispatched
  subagent through the classifier — not a hook or rule failing to bypass it. **A second finding this
  surfaced, worth knowing before reaching for that lever:** every one of this repo's 13 agents
  already declares `permissionMode: acceptEdits` in its own frontmatter, and it has never once taken
  effect for controlling that agent's own top-level starting mode — the documented "which mode a
  session starts in" decision order (`--permission-mode` flag → `permissions.defaultMode` in a
  settings file → built-in default) has no step for a custom agent's own frontmatter at all; only
  `~/.claude/settings.json`'s (this repo's) explicit `"defaultMode": "auto"` pin decides it, which is
  also exactly what the `teco` session transcript in the note above independently confirmed. The
  frontmatter field is real and does matter, but only inside the dispatch-time inheritance rule
  quoted above — and since the delegating parent is always itself in `auto` today, that rule's
  `auto`-forces-`auto` branch fires every time, so the frontmatter never gets a turn. Switching
  `defaultMode` away from `auto` (globally, per-project, or per-launch via `--permission-mode`) is a
  documented, mechanistically sound way to close this gap for whatever session's mode is changed —
  `permission-default-mode.md` works through why it's nonetheless **not recommended** as a standing
  default at any persisted scope (the Bash-classifier coverage a delegating session and everything
  it dispatches would lose outweighs the write-confirmation friction being solved). **Caveat this
  entry should carry alongside the "Hooks — `PreToolUse` 'ask' enforcement gap" callout above:** that
  gap was reproduced only under `auto` mode; whether a hook's `"ask"` reliably enforces under
  `acceptEdits` specifically is untested, so don't assume `acceptEdits` both closes the `"allow"`
  gap *and* leaves the `"ask"` safety net intact just because switching modes wasn't the thing that
  broke it originally (`claude/docs/reviews/permission-default-mode.md`, Major finding 2).

## Bash tool environment

- **The Bash tool's shell shadows some coreutils with wrapper functions that
  `exec` a different binary under a spoofed `ARGV0`** — observed for both `find`
  (wraps `bfs`: `type find` shows a function execing `${CLAUDE_CODE_EXECPATH}`
  with `ARGV0=bfs -S dfs -regextype findutils-default`) and `grep` (wraps
  `ugrep`: `ARGV0=ugrep "$_cc_bin" -G --ignore-files --hidden -I ...`). Both are
  Claude-Code-provided convenience wrappers, defined only in the interactive
  tool shell — **not** `export -f`'d, so a freshly spawned subprocess (`bash
  script.sh`, or `bash -c '...'`) does **not** inherit them and genuinely runs
  real GNU `find`/`grep`. Consequence: a bare `find`/`grep` typed directly at
  this shell's own prompt can behave differently from the same tool run by a
  script the shell invokes (`bfs`'s breadth-first traversal order vs. GNU
  `find`'s; `ugrep -G`'s basic-regex flavor vs. POSIX ERE `grep -E` — one
  boundary-heavy alternation pattern produced a different match verdict between
  the two). When auditing a script's own `find`/`grep`/`sed` logic by running
  it, exercise it **through the real invocation** (`bash script.sh`, the actual
  hook entry point) rather than trusting a bare command typed at this shell's
  prompt — the two are not guaranteed equivalent here. (Observed
  graphmind-ai-lab, 2026-07-26 and 2026-08-08.)

- **A command manually backgrounded inside a Bash call (`cmd &`) is not the same as the tool's own
  `run_in_background` parameter, and the difference bites twice.** (1) A compound command ending in
  `&` can still stall the Bash tool call for its full timeout even after the backgrounded process
  itself has already exited — the tool is waiting on the *shell*, not the child process, so a
  crashed-in-1s server can still eat the full 120s. (2) After a `cd X && ... &` call is reported as
  auto-backgrounded, the **next foreground Bash call's cwd does not stay at `X`** — it reverts to
  whatever it was before that call, not to inside the backgrounded command's `cd`. Prefer the Bash
  tool's own `run_in_background: true` (one command per call) over hand-rolled `&`/`nohup` for
  launching a long-running local process — it avoids both the stall and the cwd surprise, and gives
  a clean task-id/notification instead. (Observed graphmind-ai-lab, 2026-08-11, launching/killing a
  throwaway `uvicorn` instance for black-box QA.)

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
- **`.mcp.json` *discovery* walks up to the git root and is cwd-independent,
  but project-scope *approval* is keyed to the session's cwd** — a session
  started inside a subdirectory can see a server that a root-started session
  already approved as still `⏸ Pending approval`. `~/.claude.json`'s
  `projects` map holds one entry per absolute path a session has actually been
  launched from (with its own `hasTrustDialogAccepted`); a subdirectory that
  has never itself been a launch cwd has no entry, so the repo root's
  `enabledMcpjsonServers` pre-approval doesn't reach it even though the same
  `.mcp.json` is discovered and in effect there. Consequence: one extra
  interactive approval per subdirectory a session happens to start in — this
  is approval *scoping*, distinct from the discovery mechanism (walk-up to the
  git root), which is cwd-independent on its own terms. Separately,
  `${CLAUDE_PROJECT_DIR}` path expansion inside a `.mcp.json` entry is also
  cwd-independent, but via the unrelated server-launch-env-var mechanism
  described below — two parallel cwd-independent facts, not one shared cause.
  Observed graphmind-ai-lab,
  2026-07-25 (`claude/devops/kaizen/inbox.md`, C-319): with `.mcp.json` +
  `"enabledMcpjsonServers": ["cypher"]` at the repo root, `claude mcp list` from
  the repo root reported `cypher: … ✔ Connected`; the identical command run from
  the `falkor-chat/` subdirectory reported `⏸ Pending approval (run \`claude\`
  to approve)` — the server was still *discovered* there (walk-up to the git
  root worked), confirming this is approval scoping and not a discovery
  failure. `~/.claude.json`'s `projects` map carried exactly one entry for
  this repo (the root) and none for the subdirectory.
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

`mcp__<server>__<tool>` — e.g. `mcp__cypher__query`. Plugin-bundled servers take the
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
- **FastMCP (Python `mcp` SDK) emits an `outputSchema` — and duplicates the whole
  payload as `structuredContent` — even for a plain `-> str` tool, unless you opt
  out.** Verified against `mcp` 1.28.1: `@mcp.tool()` on a `def f(...) -> str`
  produces `outputSchema: {"properties": {"result": {"type": "string"}}, ...}` in
  `tools/list`, and a text-returning tool ships its text twice (`content` +
  `structuredContent`) — doubling the token cost for any tool designed around a
  character budget. `@mcp.tool(structured_output=False)` suppresses it
  (`outputSchema: None`); the `structured_output: bool | None = None` parameter is
  on `FastMCP.tool` in this version. Check for it whenever a plan or review
  assumes a `str`-returning tool ships unstructured text only.

### Lifecycle

- **A newly-wired `.mcp.json` server materializes only at the *next* session
  start — not live, within the session that wrote/approved it.** `claude mcp
  list` can report a server `✔ Connected` while the current session still has
  no matching tool, because Claude Code reads `.mcp.json` once, at startup.
  Treat a dependent unit's done-condition as *"first act of the next
  session"*, not a same-session resumable action. Subagents inherit MCP tools
  from the parent session's state, so this gates every delegate's access too.
  Verified 2026-07-25 (graphmind-ai-lab, cpg-query-access delivery: the
  delivering session wrote `.mcp.json` and had `claude mcp list` show
  Connected, but no `mcp__cypher__query` tool existed in that session; the
  following session started with the server's `instructions` in the system
  prompt and the tool present, no user action needed).
- **Stdio servers are local processes and are NOT auto-reconnected** if they die
  mid-session. HTTP/SSE reconnect with exponential backoff (five attempts), and
  retry a failed initial connection three times on transient errors. Recovery for
  stdio is `/mcp` → reconnect, or a new session — so an stdio server must be
  crash-proof and must never write to stdout (the transport owns it).
- `list_changed` notifications refresh a server's tools/prompts/resources without a
  reconnect; a failed refresh keeps the previously discovered set (v2.1.214+).
- **A containerized stdio MCP server is itself a running container for the whole
  session** — its process owns the session's pipe for the session's lifetime. A
  verification/orphan-check step written as "`docker ps -a --filter
  label=<server>` must be empty" is therefore unsatisfiable **the moment it runs
  from inside an open session**: the current session's own labelled container is
  legitimately `Up` at that point, and a follow-on "stop/remove the survivor"
  instruction would kill the live server the check had just been run against
  (with no auto-reconnect for stdio — see above, so recovery costs a session
  restart). Write such a check as **liveness-aware**: expect exactly one `Up`
  container per currently-open session and zero `Exited` ones, and only assert
  "empty" with all sessions using that server closed. (Observed
  graphmind-ai-lab, 2026-07-26, reviewing a `cypher` MCP server containerization
  design whose own author had measured the check with no session open.)

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
