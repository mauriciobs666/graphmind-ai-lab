# kiro

A checked-in Kiro CLI agent (`falkor-chat-demo`) that connects to `falkor-chat`'s MCP server as a
client, restricted to `send_message`/`read_messages`, hardcoded to the seeded demo thread — for a
live demo of Kiro-to-falkor-chat MCP connectivity: type a message in Kiro, watch it land in
falkor-chat, read a reply back. See `kiro/docs/requirements/kiro-demo-agent.md` and
`kiro/docs/plans/kiro-demo-agent.md` for the concrete, built slice. `kiro/DESIGN.md` is a broader,
still-Draft vision for a multi-agent Kiro ecosystem — not the spec for what's actually built here.

## Prerequisite

`falkor-chat` running locally, set up per its own docs — see `../falkor-chat/README.md` /
`../falkor-chat/AGENTS.md`.

## Run

```bash
# Terminal 1 — from repo root, start falkor-chat (if not already running):
cd falkor-chat && ./scripts/start_server.sh

# Terminal 2 — from repo root:
cd kiro
kiro-cli chat --agent falkor-chat-demo --require-mcp-startup
```

In the Kiro chat session:
- **Post a message** (AC-1): type something like "post 'hello from the kiro demo' and mention
  assistant" — it lands in falkor-chat's seeded demo thread, `@mention`-ing the `assistant`
  responder.
- **Read the reply back** (AC-2): once `assistant` has replied, type "read the latest messages" —
  the reply is shown back in the Kiro session.
