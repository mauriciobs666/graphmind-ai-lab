# Minimal Kiro Demo Agent for falkor-chat — Feature Requirements
> **Status:** Ready for design · **Owner:** `tico` · **Tracks:** — (—) · **Last updated:** 2026-08-01

## Intent
Prove that a Kiro agent can reach `falkor-chat` as a real MCP client — not just described, but
demonstrable — as the first slice of the still-open "Kiro connected to falkor-chat" gap
(`kiro/docs/requirements/kiro-vision-followups.md`, item 4). This is for a demo: someone types a
message in a Kiro chat session and sees it land in falkor-chat, then reads a response back,
showing the cross-tool wiring is real.

## Problem & current state
- `falkor-chat`'s MCP server (Streamable-HTTP, mounted at `/mcp`) has 7 conversational tools
  (`send_message`, `read_messages`, `create_thread`, `create_channel`, `list_channels`,
  `list_threads`, `search_messages`) — see `falkor-chat/docs/DESIGN.md` §15.
- No Kiro agent config exists yet pointing at it. `~/.kiro/agents/` currently has nothing wired to
  falkor-chat (confirmed empty of falkor-chat-related config). Only Kiro's **CLI** tooling
  (`kiro-cli`) is present on this machine — no evidence of the IDE surface — noted here as context
  for whoever designs this, not as a stakeholder requirement.
- `falkor-chat/docs/requirements/agent-import.md` (a different, larger feature — importing
  Claude Code agent *definitions* into falkor-chat as chat participants) explicitly deferred Kiro
  as a future follow-up. This is not that feature — this is a Kiro agent reaching *out* to
  falkor-chat as an MCP client, the opposite direction.
- Assumed precondition, not part of this feature's scope: falkor-chat's server and the demo
  workspace (`scripts/start_server.sh`, `scripts/seed_demo.sh`) are already stood up and running,
  per falkor-chat's own docs, before the demo starts.

## User stories
- As the person running the demo, I want to type a message in a Kiro chat session and see a
  response read back from falkor-chat, so that I can show the audience the integration is real.
- As a teammate who pulls the repo, I want the demo agent's configuration already checked in, so
  that I can run the same demo without re-wiring anything myself.

## Functional requirements
- **FR-1** — A Kiro agent configuration, checked into this repository, connects to falkor-chat's
  MCP server as a client.
- **FR-2** — The demo agent's reachable tool set is limited to exactly `send_message` and
  `read_messages` — no other falkor-chat MCP tool is exposed through it.
- **FR-3** — The demo agent targets the existing seeded demo channel/thread
  (`scripts/seed_demo.sh`) as a fixed target — it does not discover or select a thread at runtime.
- **FR-4** — In a live Kiro chat session, a person can type a message that gets posted into the
  falkor-chat demo thread, `@mention`-ing `@assistant`.
- **FR-5** — In the same live session, the person can then have the Kiro agent read back
  `@assistant`'s reply from falkor-chat.
- **FR-6** — Messages the demo agent sends are attributed to falkor-chat's existing single MCP
  actor (§15.2) — the feature does not add or require a distinct Kiro persona/identity.

## Out of scope
- **Distinct Kiro identity/persona in falkor-chat.** Explicitly not needed now (decision log).
- **Discovery tools / dynamic thread selection** (`list_channels`/`list_threads`, or any other of
  the 7 MCP tools beyond the two named in FR-2). The hardcoded demo thread is sufficient.
- **A scripted/automated one-command demo run.** This is a live, manually-driven demo — typing in
  Kiro, watching falkor-chat.
- **Any change to falkor-chat's MCP server itself.** No new tools, no new capabilities — this
  feature only adds a Kiro-side client against what already exists.
- **Authentication / production hardening.** falkor-chat's MCP is already unauthenticated by
  design in M1; unchanged here.
- **Multi-agent coordination, turn-taking, artifact provenance.** Tracked separately in
  `kiro/docs/requirements/kiro-vision-followups.md`.
- **Importing Claude Code agent definitions into falkor-chat.** A different, separate feature
  (`falkor-chat/docs/requirements/agent-import.md`).
- **Standing up falkor-chat/the demo workspace/LM Studio.** Assumed already running per existing
  scripts and docs — not built or automated by this feature.

## Acceptance criteria
- **AC-1** — Given falkor-chat running locally with the demo workspace/channel/thread seeded, and
  the Kiro demo agent loaded, when a person types a message `@mention`-ing `@assistant` in a live
  Kiro chat session, then that message appears in the falkor-chat demo thread (visible e.g. in the
  web UI).
- **AC-2** — Given `@assistant` has replied in the demo thread, when the person then has the Kiro
  agent read messages back, then `@assistant`'s reply is returned and shown in the Kiro session.
- **AC-3** — Given a fresh clone of this repository with falkor-chat set up per its own docs, when
  someone loads the repo-committed Kiro agent config, then the demo agent is available with no
  manual MCP wiring beyond what's checked in.
- **AC-4** — Given the demo agent's configuration, when its tool set is inspected, then only
  `send_message` and `read_messages` are reachable — every other falkor-chat MCP tool is absent.

## Open questions
_(none — all resolved, see decision log)_

## Decision log
2026-08-01 — Config checked into the repo, or personal one-off? → **Checked into the repo** —
repeatable/shareable.
2026-08-01 — Dedicated demo space, or reuse existing? → **Reuse** the existing seeded demo
channel/thread (`scripts/seed_demo.sh`).
2026-08-01 — Minimal tool surface — hardcoded thread vs. discovery tools? → **Hardcoded thread is
fine** — `send_message`/`read_messages` only, no `list_channels`/`list_threads`.
2026-08-01 — How is the demo run/observed? → **Live action** — someone typing in Kiro, watching
falkor-chat, not a scripted one-command run.
2026-08-01 — What should the demo agent actually do? → Type a message in Kiro, see it land in
falkor-chat, read a response back.
2026-08-01 — Who/what supplies the response? → falkor-chat's existing `@assistant` (the seeded,
LLM-backed K-013 responder) — the Kiro agent `@mention`s it and reads its answer.
2026-08-01 — Does the demo need a distinct Kiro persona/identity in falkor-chat, given the
MCP actor-identity constraint (every caller is attributed to the same single hardcoded actor)? →
No — name doesn't matter for now. This is an explicit, revisitable-later assumption, not a
requirement to solve identity in this feature.
