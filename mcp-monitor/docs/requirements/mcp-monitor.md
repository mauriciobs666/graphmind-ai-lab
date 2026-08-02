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
- **FR-8** — Whether a watch re-triggers on a match it has already fired on before (vs.
  suppressing it as already-handled) is itself a per-watch setting in the config file, not a
  fixed, one-size-fits-all behavior.
- **FR-9** — If a match fires while a command from an earlier match on the same (or another)
  watch is still running, mcp-monitor launches the new command in parallel rather than waiting
  or skipping it.
- **FR-10** — If a poll to a watch's MCP tool fails (server unreachable, error response),
  mcp-monitor logs the failure and continues — it does not stop the watch, and does not need to
  surface the failure any more visibly than the log.
- **FR-11** — mcp-monitor logs both failures (FR-10) and successful activity — matches found and
  commands triggered — so an operator can verify after the fact that a watch is working and
  debug it if not.
- **FR-12** — v1's second, genericity-proving MCP server/tool (alongside falkor-chat's, FR-7) is
  a minimal fake/test MCP server built for this purpose, not an existing repo component.

## Out of scope
- **What the triggered command/headless CLI does once launched.** mcp-monitor's job ends at
  launching the configured command with the match payload (FR-5) — the behavior of whatever it
  starts (e.g. an agent CLI's own logic) is not this feature's concern.
- **Turn-taking/backoff among multiple agents responding to the same trigger.** A known open
  question tracked separately in `kiro/docs/requirements/kiro-vision-followups.md` — not
  resolved here.
- **Server-side real-time push** (as opposed to client-side polling). Tracked separately at
  `falkor-chat/docs/BACKLOG.md` K-018 — mcp-monitor is a polling watcher by decision (see
  decision log), not a push mechanism.
- **Any change to falkor-chat's, cpg's, or any other existing MCP server.** mcp-monitor only
  consumes MCP tools as a client; it does not modify the servers it watches.
- **A UI or dashboard.** Observability is via logs (FR-11), not a visual interface.
- **Authentication/production hardening of mcp-monitor itself.** Not addressed in this pass.
- **Live config hot-reload.** Picking up a changed config file without restarting mcp-monitor is
  not required — restarting the process to apply config changes is acceptable.

## Acceptance criteria
- **AC-1** — Given a config file defining a watch (MCP server, tool, poll interval, regex,
  command line), when mcp-monitor is started pointed at that config, then it polls the tool at
  the configured interval without further manual input.
- **AC-2** — Given a poll result that matches the watch's regex, when the match is detected, then
  mcp-monitor launches the configured command, and that command receives the full raw tool
  result, the matched text, and an identifier of the watch/tool/server that produced it.
- **AC-3** — Given falkor-chat running with a watch configured against its message-reading tool
  and a regex for an `@mention` pattern, when a message mentioning the watched name is posted,
  then mcp-monitor detects it on its next poll and launches the configured command — demonstrated
  live, end to end.
- **AC-4** — Given a second, different MCP server (a minimal fake/test server built for this
  purpose) with its own watch configured, when its tool result matches that watch's regex, then
  mcp-monitor triggers that watch's command too — proving the same mcp-monitor instance handles
  more than one kind of MCP server/tool.
- **AC-5** — Given a watch configured with repeat-match re-triggering **on**, when the same
  underlying item matches again on a later poll, then the command is launched again; given the
  same watch configured with it **off**, when the same underlying item matches again, then the
  command is *not* launched a second time.
- **AC-6** — Given a match on one watch while an earlier-triggered command (from the same or a
  different watch) is still running, when the new match is detected, then the new command is
  launched in parallel — it is not blocked or skipped.
- **AC-7** — Given a watch whose MCP tool poll fails (server unreachable or errors), when the
  failure happens, then mcp-monitor logs it, does not crash or stop the watch, and tries again on
  the next scheduled poll.
- **AC-8** — Given mcp-monitor has been running for a while, when the operator inspects its logs,
  then both failures (AC-7) and successful matches/triggers (AC-2) are visible in them.

## Open questions
_(none — all resolved, see decision log)_

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
- 2026-08-01 — Should mcp-monitor guarantee "never trigger twice on the same match," or is that
  acceptable to leave dependent on the watched tool/query? → Neither, fixed — it's a **per-watch
  config setting**: each watch declares whether repeat matches re-trigger or are suppressed as
  already-handled (FR-8).
- 2026-08-01 — Concurrent triggers (a new match while an earlier command is still running)? →
  Parallel is fine (FR-9).
- 2026-08-01 — MCP poll failure handling? → Fail silently — log it, retry on the next poll,
  don't stop the watch (FR-10).
- 2026-08-01 — Log failures only, or matches/triggers too? → Both (FR-11).
- 2026-08-01 — Second MCP server/tool for v1's genericity proof? → A minimal fake/test MCP
  server built for this purpose, not an existing repo component (FR-12).
- 2026-08-01 — Config hot-reload while running? → Not required — restarting to apply config
  changes is acceptable.
- 2026-08-01 — Out-of-scope list reviewed with stakeholder → confirmed correct, nothing missing.
