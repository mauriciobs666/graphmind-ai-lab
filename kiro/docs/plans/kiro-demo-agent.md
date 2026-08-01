# Minimal Kiro Demo Agent for falkor-chat — Implementation Plan

> **Status:** active · **Owner:** `architect` · **Tracks:** — (—)

Requirements: `docs/requirements/kiro-demo-agent.md` (Status: Ready for design — relocation to
`kiro/docs/requirements/kiro-demo-agent.md` is **recommended by this plan**, see §3.5). Related,
not in scope: `docs/requirements/kiro-vision-followups.md` item 4 (this feature is that item's
first slice). Coordination: `kiro/docs/plans/kiro-demo-agent-coordination.md` (unit 1 of 6 — this
document is that unit's deliverable).

## 1. Goal & scope

Ship a repo-checked-in Kiro CLI agent config that connects to falkor-chat's existing MCP server
(`http://localhost:8000/mcp`) as a client, exposing **only** `send_message` and `read_messages`,
hardcoded to falkor-chat's seeded demo thread (`demo-welcome` in channel `demo-general`,
workspace `ws:acme`). A person runs `kiro-cli chat --agent falkor-chat-demo` from the `kiro/`
directory, types a message that gets posted into falkor-chat (mentioning the seeded `@assistant`
responder), and reads its reply back — live, manually, for a demo. This plan is also the first
real build-out of `kiro/` as a structured component (`kiro/docs/` tree, root `AGENTS.md` rows).

**Out of scope** (mirrors the requirements' own out-of-scope list): any change to falkor-chat's
MCP server or any other falkor-chat file; discovery tools (`list_channels`/`list_threads`/
`create_thread`/`create_channel`/`search_messages`); a scripted one-command demo; a distinct Kiro
persona/identity in falkor-chat; authentication/production hardening; the broader multi-agent
turn-taking/artifact-provenance work in `kiro-vision-followups.md`. **Hard constraint verified**:
this plan's file list touches zero files under `falkor-chat/` — see §4 step list.

## 2. Context & findings

### 2.1 falkor-chat's MCP surface (ground truth, `falkor-chat/docs/DESIGN.md` §15, unchanged by
this feature)

- Mounted at `/mcp` (Streamable-HTTP) on the same FastAPI process as the REST API and web UI.
  Unauthenticated in M1 — localhost only.
- `falkor-chat/server/falkorchat/mcp.py:52-84` — tool registration is `@mcp.tool()` on
  bare-named functions, so the on-wire tool names are exactly `send_message` and `read_messages`
  (FastMCP defaults the tool name to the function name; verified by reading the module, not
  inferred from DESIGN.md).
  - `send_message(body: str, re: str, mentions: list[str] | None = None, frm: str | None = None)`
    — `frm` is **ignored**; every call is attributed to the server's single configured actor
    (§15.2, Q#1). This is why FR-6 ("no distinct Kiro persona") is a non-issue: there is no lever
    to set one even if we wanted to.
  - `read_messages(re: str | None = None, since: int | None = None, limit: int = 50, advance: bool = True)`
    — with `re` given, reads that thread since the caller's per-thread cursor (or explicit
    `since`), advancing the cursor by default.
  - The other 5 tools (`create_thread`, `create_channel`, `list_channels`, `list_threads`,
    `search_messages`) exist on the same server and must not be reachable through this agent.

### 2.2 The seeded demo target (`falkor-chat/scripts/seed_demo.sh`, defaults, unchanged)

Fixed ids the demo agent's system prompt hardcodes:

| Thing | Value | Source |
|---|---|---|
| Workspace | `acme` (graph `ws:acme`) | `seed_demo.sh` default / `start_server.sh` default |
| Channel | `demo-general` | `seed_demo.sh` `DEMO_CHANNEL_ID` default |
| Thread (the `re` argument) | `demo-welcome` | `seed_demo.sh` `DEMO_THREAD_ID` default |
| Responder to `@mention` | `assistant` | `seed_demo.sh` `FALKORCHAT_AGENT_ID` default, matches `config.AGENT_ID` |
| Human actor MCP calls are attributed to | `u1` / "Demo User" | `seed_demo.sh` `FALKORCHAT_USER_ID` default, §15.2 |
| MCP endpoint | `http://localhost:8000/mcp` | `start_server.sh` banner, DESIGN.md §15.3 |
| Web UI (for AC-1 visual check) | `http://localhost:8000/` | `start_server.sh` banner |

`start_server.sh` is the one-shot precondition script (starts FalkorDB, bootstraps schema, seeds
the agent/channel/thread, seeds the triage workflow def, starts uvicorn with the AI responder
enabled) — assumed already run per the requirements' explicit precondition, not part of this
feature.

### 2.3 `kiro-cli` — empirically verified (v2.14.1, installed at `~/.local/bin/kiro-cli`), NOT
taken from `kiro/DESIGN.md`'s illustrative JSON sketch

`kiro/DESIGN.md`'s config sample (`allowedTools: ["read","write","shell","glob","grep"]`,
`mcpServers.chat = {"command": "python3", "args": [...]}`) predates the real CLI and is a stdio,
not-remote sketch — not used here. Everything below was produced or confirmed by actually running
the CLI in a scratch directory (`/tmp/.../scratchpad/kiro-probe{,2}`), never inside the repo:

- **`kiro-cli agent create <name> [-d DIR]`** writes `<DIR>/<name>.json` (global
  `~/.kiro/agents/` if `-d` omitted) and opens `$EDITOR`. The genuine default-template shape
  (`kiro-cli agent create probe-agent -d .kiro/agents`, `$EDITOR=true`, capturing the
  pre-edit file):
  ```json
  {
    "name": "probe-agent",
    "description": "",
    "prompt": null,
    "mcpServers": {},
    "tools": ["*"],
    "toolAliases": {},
    "allowedTools": [],
    "resources": [],
    "toolsSettings": {},
    "includeMcpJson": true,
    "model": null
  }
  ```
  `~/.kiro/agents/agent_config.json.example` (pre-existing on this machine, untouched by this
  investigation) documents the built-in `tools` vocabulary and confirms the MCP-tool reference
  syntax: `"@mcp_server_name/mcp_tool_name"` (one tool) and `"@mcp_server_name"` (all tools from
  that server).
- **`tools` vs `allowedTools`** (confirmed against `kiro.dev/docs/cli/custom-agents/
  configuration-reference/`): `tools` is the **reachability allowlist** — only tools listed there
  exist in the session at all (default `["*"]` = everything, built-ins + every tool of every
  configured MCP server). `allowedTools` is a **subset of `tools`** that runs without an
  interactive approval prompt; it does not add reachability. This is the exact lever FR-2/AC-4
  need: listing only `@falkor-chat/send_message` and `@falkor-chat/read_messages` in `tools`
  (not `@falkor-chat` bare, not `"*"`) makes every other falkor-chat tool absent, and no built-in
  tool (`read`/`write`/`shell`/...) is reachable either, which is the literal reading of AC-4
  ("only send_message and read_messages are reachable").
- **`mcpServers` entry shape for a remote HTTP server has no `type` field** — confirmed two ways:
  (a) `kiro-cli mcp add --name falkor-chat --url http://localhost:8000/mcp --agent probe2
  --force` wrote `"mcpServers": {"falkor-chat": {"url": "http://localhost:8000/mcp"}}` verbatim,
  no `"type"` key; (b) `kiro.dev/docs/cli/mcp/configuration/` confirms: local servers use
  `"command"`, remote servers use `"url"` (+ optional `"headers"`/`"oauth"`) — presence of `url`
  is the discriminator. **This differs from `falkor-chat/docs/DESIGN.md` §15.3's generic MCP
  client example** (`{"type": "streamable-http", "url": "..."}`), which is a different client's
  spelling, not kiro-cli's — do not copy that shape into the Kiro config.
- **`includeMcpJson`** (confirmed via docs search): when `true` (the generated default), the
  agent additionally loads MCP servers from workspace-/user-level `mcp.json` files outside the
  agent's own `mcpServers` map. Setting it `false` scopes the agent to exactly its inline
  `mcpServers.falkor-chat` entry — belt-and-suspenders for AC-3/AC-4 against a stray personal
  `~/.kiro/mcp.json` (the `tools` allowlist already blocks new tool exposure either way, since it
  is not `"*"`, but this avoids the agent attempting to connect to unrelated servers at all).
- **Local agent discovery is exact-CWD, no upward directory walk** — the single most
  consequential finding for AC-3. `kiro-cli agent list` run from `.../kiro-probe/` (containing
  `.kiro/agents/probe-agent.json`) lists `probe-agent` as `Workspace`; run from
  `.../kiro-probe/subdir/` (one level down, no local `.kiro/agents/` there) it does **not** —
  and does not fall back to the parent's `.kiro/agents/` either. So the demo agent config must
  live under `<some-dir>/.kiro/agents/`, and `kiro-cli` must be invoked with CWD = that exact
  directory. This plan places it at `kiro/.kiro/agents/`, so the run command is `cd kiro/ &&
  kiro-cli chat --agent falkor-chat-demo` (§3.2 rationale).
- **`kiro-cli agent validate --path <file>`** is a pure static/schema check — verified by
  validating a config whose `mcpServers.falkor-chat.url` pointed at an unreachable
  `localhost:8000` (nothing was listening); it returned silently, exit `0`, no network attempt,
  no hang. Safe and fast for a pre-flight step (§5).
- **`/tools` is an in-session slash command** (`kiro.dev/docs/cli/reference/slash-commands/`,
  `.../built-in-tools/`) that lists every tool available in the current chat session, with
  permission state — this is the live AC-4 inspection mechanism (§5).
- **`@word` in the chat input is kiro-cli's file/prompt-reference syntax**
  (`kiro.dev/docs/cli/chat/file-references/`: `@path` inlines a file, `@dir/` a directory tree;
  prompts matched before files). The docs do not say what happens when `@word` matches **no**
  file or prompt (a real risk for FR-4's "@mention-ing @assistant", since "assistant" won't be a
  file in `kiro/`). Tested empirically: `kiro-cli chat --agent probe2 --no-interactive
  --trust-tools= "... My message is: @nonexistentfile123 hello world"` did **not** error
  client-side (no "file not found" block) and proceeded to a model call (non-zero credits/time
  reported) — an unmatched `@word` is not rejected. Residual, unverified-by-me risk: in the
  **interactive** TUI (the real demo mode), typing `@a` may pop a completion dropdown while
  typing; as long as the presenter keeps typing and hits Enter without Tab-selecting a spurious
  match, the literal text should go through — flagged as a live-demo QA item, not a design
  blocker (§7).
- `kiro-cli doctor` shows `Auth ✔` on this machine (already logged in) — a fresh clone on a
  different machine needs its own `kiro-cli` login, exactly like it needs its own falkor-chat
  setup; both are explicit out-of-scope preconditions, not part of this feature.

All scratch-directory investigation happened under
`/tmp/claude-.../scratchpad/kiro-probe{,2}/`, never under this repo or the user's real
`~/.kiro/agents/` (confirmed clean afterward — only the pre-existing `.example` file is there).

## 3. Design & rationale

### 3.1 The agent config — exact content (to be written by the implementer, not by this plan)

`kiro/.kiro/agents/falkor-chat-demo.json`:

```json
{
  "name": "falkor-chat-demo",
  "description": "Minimal demo agent — connects to falkor-chat's MCP server as a client, restricted to send_message/read_messages, targeting the seeded demo thread.",
  "prompt": "You are the falkor-chat demo agent, a minimal Kiro CLI agent built for a live demo of Kiro-to-falkor-chat MCP connectivity.\n\nYou have exactly two tools, both from the `falkor-chat` MCP server:\n- `send_message` — post a message into the falkor-chat demo thread.\n- `read_messages` — read messages from the falkor-chat demo thread.\n\nFixed target — always use these, never ask the user to choose a channel or thread (you have no tools to discover one):\n- Thread id (the `re` argument): `demo-welcome`\n- The demo responder's id (for `mentions`): `assistant`\n\nBehavior:\n- When asked to send/post/say something, call `send_message` with `body` set to the user's text, `re: \"demo-welcome\"`, and `mentions: [\"assistant\"]` (so falkor-chat's AI responder is triggered) unless the user explicitly says not to mention it.\n- When asked to check/read/catch up on replies, call `read_messages` with `re: \"demo-welcome\"` and show the returned messages (author, body, timestamp) plainly.\n- Never attempt to discover channels or threads — you have no tools for that; `demo-welcome` is the only valid target.",
  "mcpServers": {
    "falkor-chat": {
      "url": "http://localhost:8000/mcp"
    }
  },
  "tools": [
    "@falkor-chat/send_message",
    "@falkor-chat/read_messages"
  ],
  "toolAliases": {},
  "allowedTools": [
    "@falkor-chat/send_message",
    "@falkor-chat/read_messages"
  ],
  "resources": [],
  "toolsSettings": {},
  "includeMcpJson": false,
  "model": null
}
```

Field-by-field rationale for anything that isn't the generated default:

- **`tools` = exactly the two `@falkor-chat/...` entries, not `["*"]` and not bare
  `"@falkor-chat"`.** This is the FR-2/AC-4 mechanism — see §2.3. No built-in tool is listed
  either, matching AC-4's literal wording ("only `send_message` and `read_messages` are
  reachable").
- **`allowedTools` mirrors `tools`.** Without this, every tool call prompts for interactive
  approval — acceptable in principle, but it would stall the *live* demo (FR-4/FR-5 are
  explicitly a live, manually-typed flow) on a confirmation keypress per call. Recommendation,
  not a hard requirement: since the tool surface is already locked to exactly two read/write
  falkor-chat calls (no filesystem/shell access), auto-approving them is low-risk. **Open call for
  the presenter** (§7): if you'd rather show the approval step as part of the demo narrative,
  delete `allowedTools`'s two entries before running — one-line edit, no redesign needed.
- **`resources: []`** — kept at the generated default (§2.3's captured template already shows
  `"resources": []` for a fresh `kiro-cli agent create`; this is not an override of a populated
  default). No repo-file context is needed since this agent only calls two remote tools.
- **`includeMcpJson: false`** — see §2.3. Scopes the agent to exactly its own inline
  `mcpServers.falkor-chat`.
- **`model: null`** — inherits whatever model the user's `kiro-cli` is configured with, so the
  config doesn't go stale if the default model changes; nothing about this demo depends on a
  specific model.
- **MCP server key is `"falkor-chat"`** (matches the component name and the `@falkor-chat/...`
  tool references — must match by construction, since kiro-cli resolves `@server/tool` against
  the `mcpServers` map key).

**Design decision — FR-4/AC-1's "@mention-ing @assistant" is read as the structural `mentions`
argument, not literal `@`-typing in the message body.** The requirement text is ambiguous between
"the presenter's keystrokes contain a literal `@assistant`" and "the resulting falkor-chat message
ends up mentioning `assistant` via falkor-chat's own mention mechanism." Verified against
`falkor-chat/server/falkorchat/services.py:618-644` (`Services.post_message`): `mentions` is a
distinct, structured `list[str]` argument, never parsed out of `body` text — falkor-chat has no
concept of scanning a message for `@word` tokens. The system prompt above therefore implements the
second reading: it calls `send_message` with `mentions: ["assistant"]` whenever the user's intent
implies mentioning the responder, regardless of whether the human literally typed an `@` character.
This is also the reading that avoids the `@`-file-completion risk noted above — a deliberate,
mutually-reinforcing choice, not a coincidence.

### 3.2 Where the config lives, and why not elsewhere

**Chosen: `kiro/.kiro/agents/falkor-chat-demo.json`**, run via `cd kiro/ && kiro-cli chat --agent
falkor-chat-demo`.

Alternatives considered and rejected:
- **Global `~/.kiro/agents/`** — rejected outright: not checked into the repo, so a fresh clone
  (AC-3) would have nothing. FR-1 explicitly requires "checked into this repository."
- **Repo root `.kiro/agents/`** (so `kiro-cli chat` works from the repo root directly, no `cd`)
  — rejected. Root `AGENTS.md` already states "OpenCode and Kiro configure MCP through their own
  files" as a *contrast* to the existing Claude-Code-only root `.mcp.json`; putting Kiro's own
  wiring at the repo root would blur that boundary and conflate two independent MCP surfaces
  (exactly what the dispatching brief called out to avoid). It also breaks the repo's own
  "independent, self-contained components" framing (root `AGENTS.md` §1) — every other
  component's run instructions start with `cd <component>/`.
- **A server-side default-argument/proxy trick to pin `re="demo-welcome"` without relying on the
  prompt** — rejected: no such lever exists in the verified kiro-cli schema (no per-tool
  default-argument binding was found — `toolsSettings` is documented only for specific built-ins
  like `shell.allowedCommands`, not for injecting fixed MCP call arguments), and building a
  stdio proxy MCP server to inject it would be new infrastructure the requirements explicitly
  rule out ("Any change to falkor-chat's MCP server itself" is out of scope, and a new proxy is
  scope creep for a "minimal" demo). FR-3's own decision log already settled for "hardcoded
  thread [in the prompt] is fine" — the system prompt textually pinning `re`/`mentions`, combined
  with there being **no discovery tool available to override it**, is the correct-sized
  mechanism. Residual risk (the LLM could technically call `send_message` with a different `re`
  string) is inherent to any agent whose behavior is prompt-directed rather than code-enforced —
  flagged in §7, not solvable without the proxy alternative just rejected.

### 3.3 What "no manual MCP wiring beyond what's checked in" (AC-3) actually requires from the
runbook

Because local-agent discovery is exact-CWD (§2.3), "checked in" config alone is necessary but not
sufficient — the human still has to `cd kiro/` before invoking `kiro-cli`. This is the same shape
every other component in this repo already has (`cd server && .venv/bin/uvicorn
falkorchat.app:app`, `cd salesperson && streamlit run ...`) — a documented run command, not
"manual wiring" in the sense AC-3 means (which is about *not* having to hand-edit MCP server URLs,
tool lists, etc.). This plan adds a `kiro/README.md` (§4) carrying that one command so the
fresh-clone flow is fully documented, matching the pattern of `falkor-chat/README.md` /
`salesperson/README.md`.

### 3.4 `kiro/docs/` tree — what's created, and what's deliberately deferred

Net-new: `kiro/docs/requirements/` (via relocation, §3.5), `kiro/docs/HISTORY.md` (first entry
added at implementation, §4 step 6). `kiro/docs/plans/` already exists (the coordination doc).
`kiro/docs/reviews/`, `kiro/docs/test-plans/`, `kiro/docs/test-reports/` are **not** pre-created
with placeholder files — git doesn't track empty directories, and per the module-documentation
convention (root `AGENTS.md`) they come into existence naturally when unit 2 (plan review), unit 5
(QA test plan/report) first write into them. Creating empty dirs now would add nothing checkable.

**`kiro/docs/BACKLOG.md` — not created in this unit.** Reasoning: `BACKLOG.md` earns its keep once
a component has more than one forward-looking, independently-sequenced item to track (see
`falkor-chat/docs/BACKLOG.md`'s K-numbered items). Right now `kiro/`'s only forward work is this
single demo-agent feature (already fully captured by the requirements doc + this plan +
`kiro-vision-followups.md`'s own backlog-shaped item list) — a `BACKLOG.md` with one row would be
process theater. Recommendation: create it the first time a second `kiro/`-scoped item is
prioritized (a natural next candidate being `kiro-vision-followups.md` item 1, 2, or 3). Flagged
as a judgment call, not a hard rule — `teco`/the stakeholder can override.

**No end-user manual** — this plan agrees with the dispatching brief's framing: a live,
manually-run demo config for a presenter (not an end user of a shipped product) doesn't fit
`tico`'s `manuals/` definition ("how to *use* the shipped product"). The `kiro/README.md`'s
run-command block (§3.3, §4) covers the same ground at the right altitude for this feature's
actual audience.

### 3.5 Recommendation: relocate the two requirements docs to `kiro/docs/requirements/`

**Recommend: yes, relocate both.** Rationale:

- Root `docs/BACKLOG.md`/`docs/HISTORY.md` state their scope explicitly in their own header
  paragraphs — *"Forward-looking backlog for the repo-root **CPG / code-graph** component"* /
  *"Dated log of actual changes to the repo-root **CPG / code-graph** component"* (verified by
  reading both files). `docs/requirements/` currently holds four files: two CPG
  (`cpg-query-access.md`, `joern-cpg-pipeline.md`) and the two Kiro ones — the Kiro pair is the
  only content in root `docs/` that isn't CPG-scoped.
- The filename-grammar rule (root `AGENTS.md`) is `<component>/docs/<kind>/<topic-slug>.md`, and
  the family rule requires the *same slug across kinds* to co-locate: `kiro-demo-agent.md`
  already has a `plans/` member at `kiro/docs/plans/kiro-demo-agent.md` (this document) and a
  `plans/...-coordination.md` member at `kiro/docs/plans/kiro-demo-agent-coordination.md` — both
  already under `kiro/`. Leaving `requirements/kiro-demo-agent.md` at the repo root breaks that
  family's co-location for no remaining reason (the stopgap that justified it — `kiro/` not yet
  having a `docs/` tree — is exactly what this feature fixes).
- Both docs' own content is 100% Kiro-scoped (`kiro-vision-followups.md`'s intent line explicitly
  starts from `kiro/DESIGN.md`); nothing in either file is CPG-adjacent or otherwise root-scoped.

**Alternative considered:** leave them at the root, since `tico` (their owner) filed them there
and moving a document the owning agent didn't move itself could read as overstepping. Rejected —
the dispatching brief explicitly delegates this exact call to `architect` ("Decide and
recommend... you decide and recommend, the coordinator will not decide this unilaterally"), the
filename-grammar rule is unconditional once `kiro/docs/requirements/` exists, and relocation is a
`git mv` (content-preserving, header/`Status:`/`Owner:` untouched) — not a rewrite of `tico`'s
work.

**Execution mechanics** (for the implementer, unit 3): `git mv docs/requirements/kiro-demo-agent.md
kiro/docs/requirements/kiro-demo-agent.md` and the same for `kiro-vision-followups.md`, then fix
the inbound references found by a repo-wide grep (enumerated exhaustively, confirmed exhaustive by
`grep -rn "docs/requirements/kiro-demo-agent.md\|docs/requirements/kiro-vision-followups.md"
--include="*.md" .`):

1. `docs/requirements/kiro-demo-agent.md` itself, line 7 and line 58 — each cites the other file
   as `docs/requirements/kiro-vision-followups.md`; after the move both live in the same
   directory, so per the citation convention (backticked path from repo root) these become
   `kiro/docs/requirements/kiro-vision-followups.md`.
2. `kiro/docs/plans/kiro-demo-agent-coordination.md` lines 7 and 10 — markdown links
   (`../../../docs/requirements/....md`); after the move, `kiro/docs/plans/` →
   `kiro/docs/requirements/` is one level up, so these become `../requirements/kiro-demo-agent.md`
   and `../requirements/kiro-vision-followups.md`.
3. Same file, lines 33, 59, 62 — backticked (non-link) path citations; update to
   `kiro/docs/requirements/kiro-demo-agent.md` / `kiro/docs/requirements/kiro-vision-followups.md`.

Note on ownership: the coordination doc is `teco`-owned, but these three edits are mechanical
link-repair caused directly by executing this plan's own recommendation (not unrelated
meddling) — bundling them into unit 3's commit is appropriate; flag to `teco` in the unit-3
hand-off so it isn't surprised by the diff.

`Status:` tokens on both requirements docs are **not** touched by this relocation (still `Ready
for design` / `Interviewing`) — that stays `tico`'s call, exercised in coordination unit 6.

## 4. Step-by-step implementation (for `coder`, coordination unit 3)

1. **Relocate the two requirements docs** (§3.5): `git mv docs/requirements/kiro-demo-agent.md
   kiro/docs/requirements/kiro-demo-agent.md` and `git mv docs/requirements/kiro-vision-followups.md
   kiro/docs/requirements/kiro-vision-followups.md`. Fix the two internal cross-references inside
   the moved file, and the five references in `kiro/docs/plans/kiro-demo-agent-coordination.md`
   (§3.5's exact list). Done: `grep -rn "docs/requirements/kiro-demo-agent\|docs/requirements/kiro-vision-followups"
   --include="*.md" .` from repo root returns only hits under `kiro/docs/requirements/` itself
   (self-references, if any) — no stale `docs/requirements/...` pointer remains anywhere.
2. **Write `kiro/.kiro/agents/falkor-chat-demo.json`** with the exact content in §3.1. Prefer
   writing the file directly (not the interactive `kiro-cli agent create` wizard, which opens
   `$EDITOR` and is awkward to script) — the config is fully specified above.
3. **Validate the config statically**: `kiro-cli agent validate --path
   kiro/.kiro/agents/falkor-chat-demo.json` — exit `0`, no output. Then `cd kiro && kiro-cli agent
   list` — confirm `falkor-chat-demo` appears with scope `Workspace`.
4. **Write `kiro/README.md`** — short, quick-start-shaped (matching `falkor-chat/README.md`'s /
   `salesperson/README.md`'s pattern): one-paragraph "what this is" (points at
   `kiro/DESIGN.md` as the broader, still-Draft vision and at
   `kiro/docs/requirements/kiro-demo-agent.md` / this plan as the concrete, built slice),
   prerequisite (falkor-chat running per its own docs — link `../falkor-chat/README.md` or
   `../falkor-chat/AGENTS.md`), and the exact run block:
   ```bash
   # Terminal 1 — from repo root, start falkor-chat (if not already running):
   cd falkor-chat && ./scripts/start_server.sh

   # Terminal 2 — from repo root:
   cd kiro
   kiro-cli chat --agent falkor-chat-demo --require-mcp-startup
   ```
   plus one line each on what to type for AC-1 (send + mention) and AC-2 (read back). **Use the
   same non-`@`-prefixed phrasing §5's test recipe already uses** — e.g. "post 'hello from the
   kiro demo' and mention assistant" / "read the latest messages" — never a suggested literal
   `"@assistant, ..."` example. This isn't stylistic: §2.3/§7 flag a real, only-partially-verified
   risk that typing a literal `@` in the interactive TUI can trigger kiro-cli's own
   file/prompt-completion; §5 already sidesteps it by phrasing the example in plain English, and
   the README's own suggested wording must not reintroduce the risk it deliberately avoided.
5. **Create `kiro/docs/HISTORY.md`** with its first dated entry (this is `kiro/`'s first delivered
   change) describing: the checked-in `falkor-chat-demo` agent config, the `kiro/docs/` scaffold,
   the `kiro/README.md`, the requirements-doc relocation, and the root `AGENTS.md` rows (step 6).
   Follow `falkor-chat/docs/HISTORY.md`'s entry format (most-recent-first, dated).
6. **Update root `AGENTS.md`**:
   - Structure table: add a `kiro/` bullet, alphabetically/logically placed among the existing
     component bullets (after `claude/`, before `skills/`, or wherever reads best next to the
     existing list) — one paragraph: what `kiro/` is (a checked-in Kiro CLI agent for a
     falkor-chat MCP demo, plus the still-Draft broader vision in `kiro/DESIGN.md`), pointing at
     `kiro/README.md` and `kiro/DESIGN.md`.
   - Component-docs table: add a `kiro/` row — entry docs `kiro/README.md` ·
     `kiro/docs/requirements/kiro-demo-agent.md` · `kiro/DESIGN.md` (marked Draft/vision in the
     cell text so nobody mistakes it for the built system's spec).
   - Done: both tables have the new row, formatted identically to existing rows (same column
     widths/style — match, don't reinvent).
7. **Self-check the hard constraint**: `git status --porcelain -- falkor-chat/` (or equivalent)
   shows only the pre-existing K-034 modifications this unit did not touch, i.e. the diff this
   step produces is empty against whatever `falkor-chat/` looked like before step 1.

No step touches any file under `falkor-chat/`.

## 5. Test / verification strategy (AC-1 … AC-4 — all live/manual per the requirements)

A static pre-flight (steps 3 above) catches config-shape mistakes before anyone attempts the live
flow. The four ACs themselves require a running falkor-chat + a live `kiro-cli` session — exactly
as the requirements and coordination doc already say (unit 5, `qa-engineer`, live). Concrete
execution recipe for whoever runs that unit:

**Preconditions (once):** `cd falkor-chat && ./scripts/start_server.sh` running in a terminal
(bootstraps FalkorDB, schema, seeds `assistant`/`demo-general`/`demo-welcome`, starts uvicorn with
the AI responder enabled). Confirm with `curl -s http://localhost:8000/` (web UI reachable) before
proceeding.

- **AC-1** (typed `@mention` lands in falkor-chat): `cd kiro && kiro-cli chat --agent
  falkor-chat-demo --require-mcp-startup` (the flag fails loudly if the MCP connection didn't
  come up, rather than silently running with the tool missing). Type e.g. "post 'hello from the
  kiro demo' and mention assistant". Confirm in the CLI's tool-call UI that `send_message` was
  invoked (not a different tool). Then open `http://localhost:8000/` → channel `#general` →
  thread "Welcome" → confirm the new message appears, authored by "Demo User" (the MCP actor,
  §2.2 — **not** a "Kiro" identity, per FR-6/no-persona), body matching what was typed, mentioning
  `assistant`.
- **AC-2** (read `@assistant`'s reply back): wait for the AI responder to post its reply in the
  same thread (asynchronous, LLM-backed — allow a few seconds; visible first in the web UI as a
  sanity check). In the same `kiro-cli` session, type "read the latest messages" (or similar).
  Confirm `read_messages` was invoked with `re: "demo-welcome"` and that `assistant`'s reply text
  is shown back in the Kiro session output.
- **AC-3** (fresh clone, no manual wiring beyond checked-in): `git clone` the repo to a scratch
  path, `cd <clone>/kiro`, run `kiro-cli agent list` — confirm `falkor-chat-demo` is listed
  (`Workspace` scope) with **no** prior manual step besides having falkor-chat itself already
  running per its own docs (the explicit precondition) and `kiro-cli` already authenticated on
  that machine (also explicitly out of scope, same as any other tool needing its own login).
  Then run the AC-1/AC-2 flow from that clone to confirm the checked-in config alone is
  sufficient.
- **AC-4** (tool set is exactly the two): two complementary checks — (a) static:
  `kiro-cli agent validate --path kiro/.kiro/agents/falkor-chat-demo.json` passes, and a direct
  read of the file's `tools`/`allowedTools` arrays shows exactly `@falkor-chat/send_message` and
  `@falkor-chat/read_messages`, nothing else; (b) live: inside a `kiro-cli chat --agent
  falkor-chat-demo` session, run the `/tools` slash command — confirm the listed/reachable tool
  set is exactly those two (no built-ins, no `create_thread`/`create_channel`/`list_channels`/
  `list_threads`/`search_messages`).

Edge cases worth a QA pass, not full ACs on their own:
- Typing a message **without** the word "assistant"/`@assistant` — the agent should still be able
  to `send_message` (mentions is optional), just without triggering the responder; confirms the
  prompt isn't so rigid it breaks on a plain post.
- The `@` file-reference collision (§2.3) — actually type `@assistant` (not paraphrase it) in the
  interactive TUI and confirm no spurious file-completion selection derails the message; this is
  the one finding in this plan that could only be confirmed non-interactively (§2.3) and needs a
  true interactive pass.
- Cold start: launching `kiro-cli chat --agent falkor-chat-demo --require-mcp-startup` **before**
  falkor-chat's uvicorn is up — should fail fast with a clear error (per the flag's documented
  behavior) rather than silently degrading; confirms the precondition ordering in `kiro/README.md`
  is the right one to document.

## 6. Documentation-impact checklist (recap, cross-referenced to §4)

- [x] Root `AGENTS.md` — Structure + Component-docs rows (§4 step 6).
- [x] `kiro/docs/` tree — `requirements/` (relocation, step 1), `HISTORY.md` (step 5); `plans/`
  already exists; `reviews/`/`test-plans/`/`test-reports/` deferred to their producing units
  (§3.4); `BACKLOG.md` deliberately not created now (§3.4, reasoning stated).
- [x] Relocation of both root `docs/requirements/kiro-*.md` files — recommended (§3.5), executed
  in step 1.
- [x] `kiro-vision-followups.md` item 4 factual update ("this slice shipped") — **not** this
  plan's job; it's coordination unit 6, owned by `tico`, after QA passes.
- [x] No end-user manual — agreed with the dispatching brief's framing (§3.4); `kiro/README.md`
  (step 4) covers the operational run-command need at the right altitude.

## 7. Risks & open questions

- **The fixed-thread guarantee (FR-3) is prompt-directed, not code-enforced.** Nothing in the
  verified kiro-cli schema lets a config hard-bind a tool call's arguments; the system prompt
  (§3.1) plus the absence of any discovery tool is the only lever found, and that's judged
  sufficient for a demo (the LLM has no way to *learn* a different thread id, only to
  hypothetically mistype the hardcoded one). If this is judged insufficient, the escape hatch is
  the proxy-server alternative rejected in §3.2 — a materially bigger build, flagged here rather
  than silently assumed away.
- **`allowedTools` auto-approval is a judgment call** (§3.1) — recommended for demo smoothness,
  trivially reversible by deleting two array entries if the presenter wants to show the
  confirmation prompt instead. Left as an explicit, easy override rather than baked in either way.
- **The interactive `@` file-reference completion behavior is unverified by this plan** (§2.3,
  §5) — only the non-interactive path was empirically checked (no client-side rejection). The
  true interactive TUI behavior when typing `@assistant` needs a live pass in coordination unit 5.
  Low risk (worst case: a visual dropdown the presenter ignores by continuing to type), but
  explicitly a gap, not silently assumed fine.
- **Hardcoded MCP URL (`http://localhost:8000/mcp`)** — matches `start_server.sh`'s own default
  banner exactly; if someone runs falkor-chat on a non-default port, the checked-in config won't
  follow without a manual one-line edit. Accepted trade-off: AC-3's "no manual wiring" is scoped
  to the default setup path documented by falkor-chat's own scripts, same assumption the
  requirements already make about falkor-chat being "set up per its own docs."
- **`kiro-cli` version drift.** Everything in §2.3 was verified against v2.14.1 specifically. If
  the schema changes in a future CLI update, the config in §3.1 may need adjustment — not
  something this plan can pre-empt, flagged so unit 3/5 know to re-check `kiro-cli --version` if
  something in this plan's grounding stops matching observed behavior.
- **BACKLOG.md-for-`kiro/` timing** (§3.4) is a judgment call, not a hard rule — surfaced
  explicitly in case the stakeholder wants it created now instead of deferred.
