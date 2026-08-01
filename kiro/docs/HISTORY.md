# Change History — kiro

> Dated log of actual changes to the `kiro` component. Most recent first.

## 2026-08-01 — Minimal Kiro demo agent for falkor-chat, and kiro/ as a structured component

**What:** Implemented `kiro/docs/plans/kiro-demo-agent.md` (reviewed, approve with suggestions).
Shipped a repo-checked-in Kiro CLI agent config (`kiro/.kiro/agents/falkor-chat-demo.json`) that
connects to falkor-chat's existing MCP server (`http://localhost:8000/mcp`) as a client, restricted
to exactly `send_message`/`read_messages` (no other falkor-chat MCP tool, no built-in tool
reachable), hardcoded to the seeded demo thread (`demo-welcome` in channel `demo-general`,
workspace `ws:acme`) — for a live demo: someone types a message in Kiro, sees it land in
falkor-chat, reads a reply back. This is also the first real build-out of `kiro/` as a structured
component.

- **Agent config** — `kiro/.kiro/agents/falkor-chat-demo.json`, written verbatim per the plan's
  §3.1 (`tools`/`allowedTools` both listing exactly the two `@falkor-chat/...` tool references,
  `includeMcpJson: false`, `model: null`). Statically validated: `kiro-cli agent validate --path
  kiro/.kiro/agents/falkor-chat-demo.json` (exit 0, no output) and `kiro-cli agent list` from
  `kiro/` (lists `falkor-chat-demo`, scope `Workspace`), both against `kiro-cli` v2.14.1.
- **`kiro/README.md`** — quick-start doc: what this is, the falkor-chat prerequisite, and the
  two-terminal run block (`start_server.sh` then `kiro-cli chat --agent falkor-chat-demo
  --require-mcp-startup`), plus plain-English example phrasing for AC-1/AC-2 (deliberately not a
  literal `@assistant`-typed example, to avoid kiro-cli's own `@`-file-completion syntax colliding
  with the demo's `@mention` intent).
- **`kiro/docs/` scaffold** — this file (`HISTORY.md`, first entry) and `kiro/docs/requirements/`
  (new directory, populated by the relocation below). `kiro/docs/plans/` already existed
  (`kiro-demo-agent.md`, `kiro-demo-agent-coordination.md`); `reviews/` already held the plan
  review. `BACKLOG.md` deliberately not created yet (plan §3.4 — `kiro/`'s only forward work is
  this single feature, already fully captured elsewhere).
- **Requirements relocation** — `git mv`'d `docs/requirements/kiro-demo-agent.md` and
  `docs/requirements/kiro-vision-followups.md` to `kiro/docs/requirements/` (plan §3.5's
  recommendation), so both co-locate with the rest of the `kiro-demo-agent` document family
  (`kiro/docs/plans/kiro-demo-agent.md`, `.../kiro-demo-agent-coordination.md`,
  `kiro/docs/reviews/kiro-demo-agent.md`). Fixed the two internal self-references inside the moved
  requirements file and the five references in `kiro/docs/plans/kiro-demo-agent-coordination.md`
  pointing at the old root-`docs/` paths.
- **Root `AGENTS.md`** — added a `kiro/` bullet to the Structure list and a `kiro/` row to the
  Component-docs table.

**Verified:** `git status --porcelain -- falkor-chat/` empty — no file under `falkor-chat/` touched
by this change, per the plan's hard constraint. Live AC-1…AC-4 execution is a separate,
qa-engineer-owned coordination unit — not run as part of this implementation.

**Files touched:** `kiro/.kiro/agents/falkor-chat-demo.json` (new), `kiro/README.md` (new),
`kiro/docs/HISTORY.md` (new), `kiro/docs/requirements/kiro-demo-agent.md` (moved from
`docs/requirements/`, 2 references fixed), `kiro/docs/requirements/kiro-vision-followups.md`
(moved from `docs/requirements/`), `kiro/docs/plans/kiro-demo-agent-coordination.md` (5 references
fixed), root `AGENTS.md` (Structure + Component-docs rows).
