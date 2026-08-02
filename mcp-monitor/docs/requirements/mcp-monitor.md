# MCP Monitor — Feature Requirements
> **Status:** Interviewing · **Owner:** `tico` · **Tracks:** — · **Last updated:** 2026-08-01

## Intent
A standalone, generic watcher: poll a configured MCP tool in real time, and when its result
matches a configurable regular expression, automatically launch a configured command line —
so nobody has to watch MCP tool output by hand and react manually, and every agent/consumer
that needs to react to some MCP activity doesn't have to build its own bespoke polling loop.

The concrete driver is `falkor-chat`: today, waking up a headless agent CLI when it's
`@mention`-ed in a thread requires a human to notice and act (e.g. the `kiro-demo-agent` demo
requires someone to type "read the latest messages"). `mcp-monitor` would let that wake-up
happen automatically — poll `falkor-chat`'s message-reading MCP tool, match an `@mention`
pattern, and launch a headless CLI instance to handle it. This is related to, but distinct
from, two already-tracked gaps: `kiro/docs/requirements/kiro-vision-followups.md` item 4 (no
agent is wired to auto-wake yet) and `falkor-chat/docs/BACKLOG.md` K-018 (server-side real-time
push, deferred) — `mcp-monitor` is a client-side polling watcher, not a push mechanism, and it
is being built as a generic tool rather than something falkor-chat-specific.

## Problem & current state
No such capability exists anywhere in the repo. The closest thing today is the `kiro-demo-agent`
(`kiro/docs/requirements/kiro-demo-agent.md`, archived/shipped), which requires a person to
manually type a "read messages" request in a live session — there is no automatic reaction to
MCP tool output today, for falkor-chat or anything else.

## User stories
- As an operator running headless agent CLIs, I want a standalone process to poll a configured
  MCP tool on an interval and detect when its result matches a pattern, so that I don't have to
  build a bespoke polling loop into every agent/consumer that needs to react to MCP activity.
- As an operator, I want a match to automatically launch a configured command line, so that a
  headless CLI instance is triggered without a human noticing and starting it.
- As a user configuring a watch, I want to define the regex, the target MCP tool, and the
  command to run, in a text config file (JSON/YAML/etc.), so that I can set up or change watches
  without touching code.
- As an operator, I want the launched command to receive the full raw tool result, the matched
  text, and which tool/server it came from, so the triggered process has enough context to act
  without needing to re-fetch everything itself.
- As someone who needs to trust this tool isn't secretly falkor-chat-specific, I want it
  exercised against at least a second, different MCP server/tool in v1, so genericity is
  demonstrated, not just claimed.

## Functional requirements
- **FR-1** — A text config file (format TBD — JSON/YAML/etc.) defines one or more independent
  "watches," each specifying at least: target MCP server, target MCP tool (+ any arguments it
  needs), poll interval, a regular expression, and a command line to execute on match.
- **FR-2** — mcp-monitor acts as an MCP client itself, calling each watch's configured tool on
  its configured interval — it does not require being placed in front of/behind an existing
  client-server session.
- **FR-3** — Each poll's result is checked against the watch's configured regular expression.
- **FR-4** — On a match, mcp-monitor executes the watch's configured command line as a new
  process, without waiting for a human to act.
- **FR-5** — The launched command receives, at minimum: the full raw tool result, the matched
  text, and an identifier of which tool/server/watch produced the match.
- **FR-6** — mcp-monitor supports multiple watches running concurrently (e.g. different
  tools/servers, or different patterns against the same tool).
- **FR-7** — v1 is demonstrated against at least two distinct MCP servers/tools (one being
  falkor-chat's message-reading tool), not just one, to prove the tool is genuinely generic.

_Still to be settled — see Open questions: duplicate/repeat matches, concurrent-trigger
behavior, failure handling, and observability._

## Out of scope
_To be drafted._

## Acceptance criteria
_To be drafted._

## Open questions
- **Repeat/duplicate matches.** If a poll re-fetches the same underlying item (e.g. the same
  chat message still present in the tool's result on the next poll), should it re-trigger the
  command again, or be treated as already-handled? This affects correctness materially and is
  the next thing to resolve.
- **Concurrent trigger behavior.** If a new match arrives while a previously-launched command is
  still running, should mcp-monitor launch another instance in parallel, queue it, or skip it?
- **Failure handling.** What should happen if a poll to the MCP tool fails (server down, error
  response) — retry, log and continue, alert, stop the watch?
- **Observability.** Does the operator need visibility (a log, a status view) into what
  mcp-monitor has polled/matched/triggered, to verify it's working and to debug it?
- **Second MCP server/tool for v1's genericity proof.** Which second server/tool should stand in
  alongside falkor-chat's message-reading tool?

## Decision log
- 2026-08-01 — Which component does this belong to? → New standalone component, `mcp-monitor`.
- 2026-08-01 — How should the regex/tool/command be configured? → A text config file
  (JSON, YAML, or similar — format not yet decided).
- 2026-08-01 — Real-time vs after-the-fact reaction? → Real time, as MCP calls happen.
- 2026-08-01 — Concrete driving scenario? → Trigger headless CLI agent instances when
  `@mention`-ed in a falkor-chat thread.
- 2026-08-01 — How does mcp-monitor obtain results? → It polls the configured tool itself, as
  its own MCP client (not a proxy/observer of someone else's session).
- 2026-08-01 — Does the triggered command need match details, or fire blind? → It needs match
  details.
- 2026-08-01 — Is falkor-chat the only v1 scenario, or must genericity be proven? → Genericity
  must be proven — v1 exercises at least a second MCP server/tool, not falkor-chat alone.
- 2026-08-01 — What must the triggered command receive? → The full raw tool result, the matched
  text, and which tool/server/watch it came from.

## Decision log
- 2026-08-01 — Which component does this belong to? → New standalone component, `mcp-monitor`.
- 2026-08-01 — How should the regex/tool/command be configured? → A text config file
  (JSON, YAML, or similar — format not yet decided).
