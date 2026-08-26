# Claude agents — context for AI agents working here

This directory (`claude/`) holds custom Claude Code subagents. Each agent is a folder: `<name>/<name>.md` (Markdown + YAML frontmatter) plus `<name>/kaizen/{plan,history}.md` — there is **no `inbox.md`**, and `audit-team.sh` check 1 requires only that pair. Every agent's raw learnings capture (durable environment facts discovered during runs) writes directly into one **shared** working-memory FalkorDB graph, `kaizen_team`, as `:KaizenEntry` nodes via `mcp__cypher__query`, so one query reaches every agent's raw learnings. An entry is tied to its producer by a `(:Agent {agentId})-[:PRODUCED]->(:KaizenEntry)` edge, plus an optional `(:KaizenEntry)-[:MENTIONS]->(:Agent)` edge when `cobb` tags it as being about a different agent during distillation; older entries instead carry a plain `author` string property and no edges — read both shapes, write only the edge shape. `cobb` distills the shared graph periodically per the `agent-maintenance` skill §5 (verify → route to prompt/knowledge base/project docs → log in the entry's own agent's `history.md` → clear, a curator-scoped `DETACH DELETE` through `mcp__cypher__query` against `kaizen_team`). **Skills do not live here** — their home is the repo-root [`skills/`](../skills/) (see [`skills/README.md`](../skills/README.md)); cobb's `agent-maintenance` and `agent-standards` skills are there.

**The full agent catalog — what each does, when to use it, handoff contracts, hook enforcement — lives once, in [`README.md`](./README.md).** Each agent's frontmatter `description` is its routing contract and is auto-injected into sessions; each `<name>/<name>.md` is the source of truth for its behavior. This file keeps only the index plus directory-level conventions.

## Agents

Roster — behavior source is always `<name>/<name>.md`, kaizen at `<name>/kaizen/`; what each
does and when to use it lives in the injected descriptions and [`README.md`](./README.md):

`teco` (coordinator) · `tico` (product owner, project explainer, user-manual curator; **interactive**: `claude --agent tico`) ·
`architect` · `coder` · `tdd-engineer` · `frontend-engineer` ·
`qa-engineer` (carries an on-demand knowledge base: `qa-testing-techniques.md` — environment/
tooling techniques such as the WSL2 browser-automation fallback, driving an interactive TUI, and
CLI health-check gotchas) ·
`analyst` (carries an on-demand knowledge base: `review-techniques.md` — review-methodology
techniques consulted on demand) ·
`data-scientist` (carries an on-demand knowledge base: `lm-studio-model-notes.md` —
live-verified small-model realism notes for this lab's LM Studio stack) ·
`graph-dba` (carries two on-demand knowledge bases: `falkordb-quirks.md`,
live-verified and perishable — re-verify on upgrades — and `falkordb-reference.md`; also drives
Joern via the `joern-cpg` skill, on demand, to build a repo's CPG and export/load it into
FalkorDB as Cypher — a rare capability, not a proactive default) ·
`devops` (user-scoped — runs in every project; carries an on-demand knowledge base:
`ops-quirks.md` — live-verified Docker/BuildKit/Bash-scripting traps) ·
`security-expert` (on-demand deep security reviewer: code/app security, agent/prompt-safety,
secrets/infra-hardening, compliance checklists — advisory to `analyst`/`cobb`/`devops`; the only
agent on this team gated to attempt active exploitation, local/dev targets only, fresh explicit
approval every time) ·
`cobb` (team maintainer: `agent-maintenance`/`agent-standards` skills,
`scripts/audit-team.sh`, testing standards in `cobb/TESTING.md`).

## Hook machinery

**Three shared cores**, all under `scripts/`, all thin-wrapped per agent via frontmatter
`hooks:` → `$HOME/.claude/agents/<name>/hooks/<script>.sh` (resolves through the deployment
symlink):

- **`scripts/guard-doc-writes.sh`** — an ALLOW-LIST core for a doc-scoped (or topic-scoped)
  agent: escalate everything except a small set of paths that ARE the whole remit. Eight
  `Write|Edit` wrappers sit on this core today: the original five doc-scoped agents (`architect`,
  `analyst`, `data-scientist`, `teco`, `tico`), plus `security-expert`'s review guard, plus
  `cobb` (topic-bounded — any agent's own
  definition file, kaizen curation for the team, and a small explicitly-maintained list of
  MCP/agent-standards docs outside `claude/`/`skills/`, e.g. `cypher-mcp/README.md` — cuts across
  folders rather than living in one, so its allowlist is a wider glob union than the others') and
  `qa-engineer` (`docs/test-plans/*`, `docs/test-reports/*`). Each wrapper passes its allowed-path
  globs (every `claude/`/`skills/`-rooted glob doubled — a bare form plus a `*/`-prefixed sibling
  — because `tool_input.file_path` can arrive absolute, and a leading `*` is what lets the doubled
  form absorb an arbitrary absolute prefix ahead of the literal directory), an escalation message,
  and an optional third arg, `on_mismatch` (`ask`, the default — seven of the eight wrappers use
  this; or `pass`, used only by `qa-engineer`, whose remit is genuinely wider than its two doc
  kinds — it also authors source/test files as part of execution, and those must fall through to
  the ambient permission flow unmediated rather than newly escalate). On a match the core emits an
  explicit `permissionDecision: "allow"`; on a mismatch (`on_mismatch="ask"`) it emits
  `permissionDecision: "ask"`. `teco`'s wrapper is **not** purely thin: before deferring to the
  core it auto-allows one mechanically-verified edit shape — an `Edit` on a `docs/**.md` file
  whose old/new strings differ only in the canonical `Status:` field flipping to `archived` (the
  milestone-close archival flip) — checked in
  python3 by masking the Status field on both strings and requiring byte-equality of the rest.
- **`scripts/guard-broad-write.sh`** — the DENY-LIST inverse: for an
  implementer agent whose remit is genuinely "the whole codebase, this task" and has no single
  folder/kind to allowlist. Allow everything except a small set of paths KNOWN to belong to a
  *different* specialist's documented deliverable-path convention (every other agent's doc kind,
  `claude/`/`skills/`-rooted agent-standards paths, and `docs/BACKLOG.md` — the last one
  deliberately kept escalating rather than resolved either way). One wrapper today:
  `tdd-engineer/hooks/guard-tdd-broad-write.sh`.
- **`scripts/guard-destructive-ops.sh`** — thin-wrapped by the three destructive-ops guards
  (`devops`, `graph-dba`, `qa-engineer`; each passes its agent name; the core matches Bash command
  patterns — `GRAPH.DELETE`, `FLUSHALL`/`FLUSHDB`, volume wipes, `docker rm -f`, and
  `pipeline.sh ... --reset` — a wrapper invocation matched ad hoc because the
  script runs `GRAPH.DELETE` internally, where the literal string never reaches the guard — not
  write paths). Unlike the two cores above it is **`ask`-only, with no allow branch**.

Each agent named above carries one or two
`PreToolUse` hooks under a single frontmatter `hooks:` block — one hook per agent is the common
case, not a hard rule. `security-expert`'s `Write|Edit` guard
(`security-expert/hooks/guard-review-doc-writes.sh`) is a normal thin wrapper over
`guard-doc-writes.sh`, scoped to `docs/reviews/*` only; its `Bash` guard
(`security-expert/hooks/guard-exploitation-approval.sh`), enforcing FR-10's "active exploitation
needs a fresh, explicit approval every time, local/dev targets only"
(`claude/docs/requirements/security-expert.md`), is deliberately **not** layered on
`guard-destructive-ops.sh` — that core's catalog is shared-state-destruction literals
(`GRAPH.DELETE`, `FLUSHALL`, volume wipes), a different hazard class from the offensive-tool/
network-exploitation patterns this guard matches (named tools like `sqlmap`/`nmap`/`msfconsole`,
listener setups, or a network-reaching command with no visible local/dev marker). It's a
standalone, agent-owned script with the same mechanics/contract as the shared cores (fail-open,
`ask`-only, jq→python3 extraction) — extract it into a shared core only if a second
exploitation-shaped agent is ever added (`security-expert/kaizen/plan.md` K-003). `qa-engineer`
carries two: its `Bash` destructive-ops guard alongside the `Write|Edit` doc-write guard above.
`teco` also carries two: alongside its `Write|Edit` wrapper, an `Agent|Task` dispatch guard
(`teco/hooks/guard-agent-dispatch.sh`, standalone agent-owned script, same
fail-open/`ask`-only/jq→python3 contract as the shared cores) that escalates any `Agent` dispatch
missing `subagent_type` — an omitted field silently spawns a `general-purpose` delegate with none
of the named agent's prompt, tools, or hooks, and raises no error.

**Git-commit authority is prompt-level, not hook-enforced.** No `PreToolUse` hook matches `git
commit` (the destructive-ops guards match Bash command patterns like `GRAPH.DELETE`, not
versioning commands), so this is entirely self-discipline, backstopped only by
`scripts/audit-team.sh` check 8. The policy has two layers:

- **Standing broad grants — `tico` and `teco` only, unconditioned on invocation mode.** `teco` may
  commit any coordinated specialist's already-verified deliverable **by explicit path** (its
  integrator role, deliberately wider than its own Write/Edit guard). `tico` may commit three
  things: its own doc kinds (requirements, manuals); the returned artifact of a
  `qa-engineer`/`analyst` verification pass it itself offered under Mode 3 and the stakeholder
  accepted, once tico has confirmed the artifact fits; and **a file it itself wrote in the current
  session that its `Write`/`Edit` guard let through**, including a one-off the human approved at
  the guard's escalation prompt (e.g. a `docs/BACKLOG.md` entry). In that third case the
  write-approval is what confers committability — a human who has just approved the write has
  already supplied the review — so **that case alone** covers only files `tico` wrote itself this
  session; it does not stretch to an earlier session's writes, and it does not narrow the other
  two. Explicit-path staging applies to all three. Both agents' grants hold whether they are
  running interactively or as a delegated subagent — they're tied to the agent's role, not its
  invocation mode.
- **Universal interactive-mode grant — every agent.** Every agent in this directory may `git
  add`/`git commit` its **own verified work from the current session**, by explicit path only
  (never `git add -A`/`git add .`/`git commit -a`, never `git push`/`reset`/`rebase`, never amend
  history) — **but only when it is running interactively** (`claude --agent <name>`, a human
  conversing with it turn-by-turn). **The grant does not apply when the same agent is spawned as a
  delegated subagent** (via `Agent`/`Task`, isolated context, no live human turn) — there,
  committing stays the coordinating agent's (`teco`'s) integration step, after its own
  verification. The distinction exists because a human directly steering an interactive session
  *is* the review that routing commits through `teco` otherwise supplies; an isolated subagent
  chain has no such reviewer in the loop. Each agent's own Guardrails/Principles/Boundaries section
  states this grant in its own words; `scripts/audit-team.sh` check 8 verifies every agent's file
  both claims it and states the delegated-subagent carve-out.

This second layer **does not touch** the first: `tico`'s and `teco`'s broader, mode-unconditioned
grants stand exactly as documented above, and this still isn't a delegation of *coordination* —
an agent running interactively still only commits what it itself verified, never another agent's
in-flight work.

**`git add` then `git commit` is not atomic against a concurrent process sharing the same working
tree.** A staged file sits in the shared git index until the commit actually runs — a second
agent/process that stages and commits its own files in that window can commit *both* sets
together, bundling the first agent's file into a commit message it never approved (observed
2026-08-21: an `analyst` review doc staged cleanly, but a concurrent `qa-engineer` pass staged and
committed first, sweeping the review doc in under its own unrelated message). File-disjoint work
does not avoid this — the race is on the index, not on any one path. Immediately before running
`git commit`, re-check `git status`/`git diff --cached --name-only` to confirm only your own
explicit path(s) are staged; if anything else has appeared, stop and reconcile before committing.

## Maintenance rules

- **A stakeholder proposal for a new team member is a `tico` requirements interview, not a
  straight-to-`cobb` request.** It's an ordinary Mode 1 interview with `claude/` treated as the
  component — WHAT/WHY only (the new agent's remit, its boundaries with existing agents, any
  destructive-shaped capability), landing at `claude/docs/requirements/<slug>.md`. Only once that
  doc reaches **Ready for design** does `cobb` design the actual agent (name, prompt, tools,
  hooks) from it. Precedent: `claude/docs/requirements/security-expert.md`.
- Adding/editing/renaming/removing an agent → update the agent source, its `kaizen/{plan,history}.md`, the full catalog entry in [`README.md`](./README.md), and the name roster here, in the same change.
- Skills live in the repo-root [`skills/`](../skills/) home, not here. Their catalog is [`skills/README.md`](../skills/README.md); cobb's kaizen logs changes to `agent-maintenance`/`agent-standards`.
- Don't paste full system prompts or duplicate the README catalog here — point to the source.
