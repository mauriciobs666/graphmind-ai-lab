# Kaizen — Learnings Inbox: tico

> Append-only capture of durable, non-obvious environment facts the `tico` agent
> discovers during runs — raw observations, not conclusions. The maintainer (cobb)
> periodically distills this inbox (agent-maintenance skill §5): verifies each entry,
> routes it (prompt / knowledge base / project docs / discard), logs the promotion in
> `history.md`, and clears it. The agent only appends here; it never promotes.
>
> Entry format (append at the end):
>
> ```markdown
> ## YYYY-MM-DD — <the fact, one line>
> - **Evidence:** what was run/read/observed (command, file:line, output)
> - **Context:** the task where it surfaced, one line
> - **Suggested home:** prompt | knowledge base | project docs | unsure
> ```

## 2026-08-17 — `AskUserQuestion` hard-rejects any question with fewer than 2 options — there is no way to route a stakeholder's free-text "let me describe it" answer back through the tool as a single-choice follow-up
- **Evidence:** Calling `AskUserQuestion` with a question offering one option (`"Let me describe it in free text"`) raised `InputValidationError: array to have >=2 items` before the user ever saw it, with explicit guidance not to invent a filler second option and to fall back to plain conversation instead.
- **Context:** `generic-cypher-mcp` requirements interview — the stakeholder picked "Something else — I'll describe it" on a multiSelect `AskUserQuestion`, and the natural next move (a single-option follow-up question) was rejected by the tool; had to drop back to a plain conversational prompt to get the free-text answer.
- **Suggested home:** prompt (a one-line addition to Tico's own interview-craft guidance: when a stakeholder's answer needs open-ended free-text follow-up, ask in plain conversation, not via `AskUserQuestion`).

## 2026-08-17 — FalkorDB's bundled web console is already live on `:3000` wherever `falkordb-dev` runs — no new work needed for "can I just look at the graph"
- **Evidence:** `falkor-chat/scripts/start_falkordb.sh` publishes `-p "${FALKORDB_WEB_PORT}:3000"` (default `3000`) alongside the `:6379` Redis/Cypher port, and prints `Web console: http://localhost:${FALKORDB_WEB_PORT}` on every start.
- **Context:** `generic-cypher-mcp` requirements interview — stakeholder asked "can I use the web tool to monitor created nodes/relationships," which resolved a whole access-tier open question (deferring their own read/write access to a later phase) once confirmed this already exists today with zero new work.
- **Suggested home:** unsure (candidate: a line in `tico`'s own explanation reflexes, or `cypher-mcp/README.md`'s troubleshooting/observability section, since agents debugging graph state might not know this console exists either).

## 2026-08-17 — Introducing a new *agent* is a tico requirements interview too, filed under claude/docs/requirements/
- **Fact:** A stakeholder request to add a new team member (a new Claude Code subagent, e.g.
  "security expert") is not a special case that skips straight to `cobb` — it's a normal Mode 1
  requirements interview, with `claude/` treated as the component. The doc landed at
  `claude/docs/requirements/security-expert.md`. `claude/` had **no `docs/` directory at all**
  before this — agents there previously only used `<name>/kaizen/{plan,history,inbox}.md` (per
  `claude/AGENTS.md`), never the generic module-docs convention from root `AGENTS.md`
  (`docs/requirements/`, `docs/plans/`, etc.). This session established the precedent that the
  generic convention applies to `claude/` itself once the topic is the *team*, not a specific
  existing agent's kaizen plan.
- **Evidence:** Session where the stakeholder said "i want to introduce a new team member, the
  security expert." Root `AGENTS.md`'s module-documentation-convention section is generic across
  "modules"; `claude/` had never exercised the `docs/requirements/` branch of it before. WHAT/WHY
  (the new agent's remit, boundaries with `analyst`/`cobb`/`devops`, the one destructive-shaped
  capability) went through the normal interview; the actual agent design (name, prompt, tools,
  hook) is explicitly left to `cobb` as the next step, not designed here.
- **Suggested home:** `claude/AGENTS.md` or `claude/README.md` — worth a line noting that a
  *new*-agent proposal gets a `claude/docs/requirements/<slug>.md` via `tico` before `cobb`
  designs it, so this doesn't need re-deriving next time. Also worth `cobb` knowing this doc
  exists as the WHAT/WHY input the next time it's asked to design the security-expert agent.

## 2026-08-19 — Stakeholder flagged commit cadence as too fine-grained: one commit per single Edit/decision-log append reads as noisy, even though each was individually a "meaningful update"
- **Evidence:** one session covering two requirements docs (`generic-cypher-mcp2.md`,
  `cpg-mcp-rename.md`) produced **11 separate commits**, several of them a single decision-log
  entry or one FR tweak (e.g. `generic-cypher-mcp2 — scope (all 11 agents) and trigger settled`
  immediately followed by `— team-wide query surface, cobb cadence/self-migration settled`, two
  commits apart by one `AskUserQuestion` round each). The stakeholder's own words: "you seen to be
  committing too often." My prompt's guidance ("after a meaningful update lands... stage and
  commit exactly the file(s) you just wrote") was followed literally per-edit rather than batched
  per settled *thread* or natural pause point in the conversation.
- **Context:** live M6 (`generic-cypher-mcp2`) + follow-on (`cpg-mcp-rename`) requirements
  interviews, same session, back-to-back `AskUserQuestion` rounds each followed by its own commit.
- **Suggested home:** prompt — tighten `tico.md`'s "Commit as you go" guidance to commit at
  natural batch points (a full readback, a settled cluster of decisions, or before switching
  topics/modes) rather than after every individual `Edit` call, while still never batching two
  *different* documents' changes into one commit.
