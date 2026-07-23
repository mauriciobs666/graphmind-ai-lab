# Agent Import (Claude Code) — Feature Requirements
> Status: Ready for design · Last updated: 2026-07-22

## Intent
The stakeholder has already built an agent team outside falkor-chat — the Claude Code subagents
in `claude/` (`teco`, `tico`, `architect`, `coder`, `analyst`, `qa-engineer`, …). Today those
agents only ever work **one at a time, in a terminal**, and their collaboration is invisible: a
coordinator delegates, a subagent runs in an isolated context, a summary comes back.

The goal is to **see the team working together** — the same agents present as participants in a
falkor-chat workspace, reachable on a shared timeline, with the whole exchange stored in the
graph. Reusing definitions instead of re-authoring them is the mechanism, not the point.

## Problem & current state
- falkor-chat has exactly **one** AI participant: the seeded `assistant` Agent
  (`scripts/seed_demo.sh`). Adding another means writing another seed by hand.
- Agent definitions live in `claude/<name>/<name>.md` (YAML frontmatter + prompt body) with no
  path into falkor-chat.
- Workflow defs are authored by hand (REST JSON, or `server/falkorchat/proof_defs.py`) and have
  no relationship to the agent team that exists elsewhere in the repo.
- The team's real process — `tico` → `architect` → `analyst` → `coder`/`tdd-engineer` →
  `qa-engineer`, coordinated by `teco`, handing off by document path — is documented prose in
  `claude/README.md`. It is never executed as a traceable run; nobody can watch it happen.

## User stories
- As the repo owner, I want to import my Claude Code agent definitions into a falkor-chat
  workspace, so that I don't re-author a team I already maintain.
- As a chat user, I want to `@mention` any imported agent in a thread and get an answer in that
  agent's own persona, so that the whole team is reachable where the conversation happens.
- As an observer, I want the team's delivery pipeline available as a runnable workflow with
  imported agents as its step actors, so that I can watch the process advance instead of
  reading a summary of it.
- As a workflow author, I want to name an imported agent as the actor of an `agent` step, so
  that defs I write by hand can draw on the same team.

## Functional requirements
_(draft — FR-6/FR-7 still need their unknowns closed)_

**Import**
- **FR-1** — The system can import Claude Code agent definitions from `claude/<name>/<name>.md`
  into a falkor-chat workspace, reading the YAML frontmatter and the prompt body.
- **FR-1d** — What carries over: the **prompt body** (drives behavior), the agent's **name** and
  **description**. The frontmatter's declared `tools` and `model` are stored as **inert
  metadata** — they confer no capability. Every imported agent answers through the single
  configured LM Studio chat model.
- **FR-1b** — The caller **selects which agents to import** by name — import is never
  automatically the whole `claude/` roster.
- **FR-1c** — Import is driven by a **script**, in the style of the existing
  `seed_demo.sh` / `seed_workflows.sh`: target workspace plus an explicit list of agent names.
  No REST endpoint and no UI in this feature.
- **FR-2** — Import is **idempotent**: re-running it over an unchanged source set leaves the
  workspace unchanged and reports that nothing changed.
- **FR-2b** — When a source definition **has changed** since a previous import:
  - the affected agent's **chat persona updates in place** — it answers from the new prompt;
  - the **pipeline def is published as a new version** rather than mutated, leaving previously
    published versions and any in-flight runs on them untouched.
- **FR-3** — Import reports per agent what it did (created / updated / unchanged / rejected with
  a reason), so a failed import is diagnosable without reading the graph.

**Imported agent as chat participant**
- **FR-4** — An imported agent is registered as an `Agent` member of the workspace and is
  `@mention`-able in any thread, exactly as `@assistant` is today.
- **FR-5** — When `@mention`-ed, an imported agent replies **in its own persona** — its imported
  prompt governs the answer — posted as that agent (`role:"assistant"`) with the existing
  `EMITTED` provenance edge.
- **FR-6** — An imported agent's reply is grounded in the workspace graph via the existing
  GraphRAG retrieval path. _(Precedence between the imported persona and retrieved context is a
  prompt-design decision delegated to the architect / data-scientist.)_

**Imported agent as workflow actor**
- **FR-7** — A workflow def's `agent` step can name an imported agent as its actor; the executor
  drives that step through that agent's persona.

**Imported workflow definition**
- **FR-8** — The team's **delivery pipeline** is importable as a single published workflow def —
  `tico` → `architect` → `analyst` → `coder`/`tdd-engineer` → `qa-engineer` — whose steps name
  imported agents as actors, derived from the roster's documented handoff contracts in
  `claude/README.md`. This is the one def this feature produces.

**Constraints carried from the current system (context for the architect, not requirements)**
- Published workflow defs are effectively **immutable** (`MERGE … ON CREATE SET`); a re-import
  of a changed def cannot update in place — it needs a version bump or an explicit teardown.
  This collides head-on with FR-2's idempotence for the def half of the import.
- Def (`reference` graph) and workspace snapshot (`ws:{id}`) go stale **independently**.
- Step types `prompt` / `tool` / `message` raise `NotImplementedError` today; only
  `human` / `decision` / `wait` / `agent` are live.

## Out of scope
- **Kiro import.** `~/.kiro/agents/` is empty — no Kiro agent files exist yet. `kiro/DESIGN.md`
  describes a config shape and a task state machine, but nothing is instantiated. Kiro import is
  a documented follow-up for when those files exist.
- **Agent-to-agent conversation.** Imported agents do not `@mention` each other, and do not
  autonomously respond to each other. The human drives; the shared, persistent timeline is what
  makes the collaboration visible.
- **Tool execution.** Imported agents **talk only** — no Read, Write, Bash, or any other tool
  access inside falkor-chat. An imported `@coder` discusses an implementation; it cannot produce
  one. A declared tool grant in the source frontmatter is metadata, not a capability.
- **Export / round-trip.** Import is one-directional; falkor-chat never writes back to `claude/`.
- **Skills.** Imported agents do not load the `skills/` packages their source definitions use.
- **Agent authoring in falkor-chat.** No UI or API for creating a new agent from scratch — this
  feature imports existing ones.
- **Per-agent workflow defs.** Deriving a mini-process from each agent's prose prompt (e.g.
  `tico`: interview → draft → readback) is dropped: the structure would be guessed from prose and
  would sometimes be wrong. Only the pipeline def (FR-8), grounded in documented handoff
  contracts, is produced.
- **Import trigger surfaces beyond the script.** No REST endpoint, no web-UI pick-list.
- **Driving a parked workflow step from chat.** A **follow-up feature**, to get its own
  requirements doc. Today a run started from a thread by `@mention` *is* resumed by a plain human
  reply (`trigger.py` rule 2), but that path carries **no structured input** — only the REST
  `submit_workflow_input` does — and a REST-started `kind:'process'` run is deliberately not
  bound to any thread (`_reject_reserved_keys`, M-2/F-6). Closing that gap is out of scope here;
  the pipeline def (FR-8) is driven within today's capability.
- **Multi-model routing.** An agent's declared `model` does not select a model.

## Acceptance criteria
_(draft — to be completed once the open questions close)_

- **AC-1** — Given the `claude/` agent folder, when the import runs against a bootstrapped
  workspace, then every agent in the roster appears as an `Agent` member, and the run reports
  one line per agent.
- **AC-2** — Given a completed import, when the import is run again with no source change, then
  every agent reports "unchanged" and the graph is byte-identical.
- **AC-3** — Given an imported `@architect`, when a user posts `@architect <question>` in a
  thread, then a reply arrives within the same latency budget as `@assistant` today, and the
  reply **references that agent's own documented deliverable** — `@architect` a plan under
  `docs/plans/`, `@tico` a requirements doc, `@qa-engineer` a test plan/report. This is the
  checkable persona signal.
- **AC-6** — Given a script invocation naming a subset of agents, when the import runs, then
  **only** the named agents are created — agents not named do not appear in the workspace.
- **AC-4** — Given the imported pipeline def, when a run is started and driven to completion
  **by whatever input path today's engine supports** (chat reply for a thread-started run, REST
  `submit_workflow_input` otherwise), then each imported agent's step posts into the thread and
  the full `NEXT`-ordered step-run trail is readable via `GET /workflow-runs/{id}/trace`.
- **AC-5** — Given a source agent whose definition changed, when the import re-runs, then that
  agent's chat replies reflect the new prompt, a **new version** of its derived def is published,
  the prior version still exists, and any run in flight on the prior version is unaffected.

## Open questions
1. ~~Freshness / re-import~~ — **closed**, see FR-2b.
2. ~~How is import triggered?~~ — **closed**, see FR-1c (script).
3. ~~What carries over from the frontmatter~~ — **closed**, see FR-1d.
4. ~~Persona vs. retrieval precedence~~ — **no stakeholder preference**; delegated to the
   architect / data-scientist as prompt design.
5. ~~Where does an imported agent live~~ — **stakeholder has no preference**; per-workspace vs.
   global `identity`/`reference` placement is delegated to the architect as a design decision.
6. ~~How do we verify "in its own persona"~~ — **closed**, see AC-3 (names its own deliverable).
7. ~~What model powers an imported agent's replies~~ — **closed**, see FR-1d (the single
   configured LM Studio chat model).
8. ~~Agent selection~~ — **closed**, see FR-1b/FR-1c (explicit names on the script invocation).
9. ~~Derived-def review mechanism~~ — **moot**: per-agent derived defs were dropped from scope.

## Decision log
2026-07-22 — What does an imported agent become in falkor-chat? → **All three**: an @mentionable
chat participant, a workflow definition, and a step actor inside a workflow.
2026-07-22 — Why now / what's the itch? → To **see the team working together** on a shared
timeline, rather than one agent at a time in a terminal.
2026-07-22 — What drives the collaboration? → **One at a time, but visible.** The human talks to
each imported agent; the shared persistent history is the collaboration. Agent-to-agent
conversation is out of scope.
2026-07-22 — Scope of the first feature? → **All three targets in one feature** (participant +
workflow step + workflow def).
2026-07-22 — Which source first? → **Claude Code only.** Kiro dropped from this slice —
`~/.kiro/agents/` is empty, so there is nothing to import or test against.
2026-07-22 — Where do imported workflow defs come from? → **Both**: the team's delivery pipeline
as one def (from the roster's handoff contracts), **and** a per-agent mini-process def derived
from each agent's prompt.
2026-07-22 — Do imported agents need tools? → **No.** Talking is enough for now; no file or shell
access inside falkor-chat.
2026-07-22 — What happens on re-import after a source edit? → **Agent persona updates in place;
the derived def is published as a new version** (prior versions and in-flight runs untouched).
2026-07-22 — Which agents get imported? → **User-specified** — the caller names them; never the
whole roster automatically.
2026-07-22 — How is import triggered? → **A script taking a workspace + agent names**, like
`seed_demo.sh`. REST endpoint and UI pick-list explicitly deferred.
2026-07-22 — Where do imported agents live in the graph? → **No stakeholder preference** —
delegated to the architect as a design decision.
2026-07-22 — How is "answered in its own persona" checked? → **The agent names its own documented
deliverable** (architect → a plan, tico → a requirements doc, qa-engineer → a test plan).
2026-07-22 — Per-agent workflow defs derived from prompts? → **Dropped from scope** — the
structure would be guessed from prose. Only the pipeline def (FR-8) is produced. This reverses
the earlier "both" answer on def sources.
2026-07-22 — Persona vs. retrieved context precedence? → **No preference**; architect/
data-scientist decide.
2026-07-22 — What carries over, and which model answers? → **Prompt body + name + description**;
`tools`/`model` are inert metadata; **one configured LM Studio chat model** for all agents.
2026-07-22 — Should a parked workflow step be advanceable from the chat interface? → **Yes, but
as its own next feature** — not folded into agent import. Partial support already exists
(`trigger.py` rule 2 resumes a thread-started run on a plain reply); the gaps are structured
input from a message and binding a process run to a thread.
