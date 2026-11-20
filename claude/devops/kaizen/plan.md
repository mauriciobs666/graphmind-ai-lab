# Kaizen — Improvement Plan: devops

> Forward-looking backlog for the `devops` agent.
> Status: 🔵 proposed · 🟡 in-progress · ✅ done (then moved to history.md) · ⚪ rejected/deferred
> Last reviewed: 2026-09-07

## Active


### K-002 — Infra-docs WebFetch allowlist

- **Added:** 2026-07-02
- **Status:** 🔵 proposed
- **Priority:** low
- **Rationale:** Like cobb verifies agent-tool docs live, devops will want to check Docker/Compose/
  GitHub Actions docs. Those domains aren't in the current `~/.claude/settings.json` allowlist, so
  fetches prompt.
- **Proposed change:** Add `WebFetch(domain:docs.docker.com)`, `docker.com`, `docs.github.com` to
  the user-scope allowlist if verification-heavy infra work becomes common. Personal/global file —
  document in `claude/README.md` alongside cobb's block.

> Summary: Add a WebFetch allowlist for infra doc domains (docker.com, docs.docker.com, docs.github.com)

### K-003 — Persona vs. role framing

- **Added:** 2026-07-02
- **Status:** 🔵 proposed
- **Priority:** low
- **Rationale:** User chose the plain role name `devops` over a persona. If the repo's persona
  culture (teco/saul/severino) proves to matter for how this agent is invoked/relatable, a persona
  alias could be layered on. Low stakes.

> Summary: Revisit persona vs. role framing once the agent has run a few real tasks

### K-004 — Guard hook: degraded extraction and fixed pattern list

- **Added:** 2026-07-02
- **Status:** 🔵 proposed
- **Priority:** low
- **Rationale:** the hook path is already machine-independent (`$HOME/.claude/agents/devops/hooks/…` via the deploy symlink). What remains: the script prefers `jq`→`python3`, and a box with neither falls back to grep on the raw payload; the blocked-pattern list is not configurable.
- **Proposed change:** only if the agent is deployed across machines — make the pattern list configurable.

> Summary: Guard hook script prefers `jq`→`python3` and degrades to grep extraction without either; make the blocked-pattern list tunable (only if deployed across machines)

## Parking lot / ideas

### GAL-I02-K0007 — Compose coverage for `salesperson`

- **Status:** 🔵 proposed — parking-lot origin; owner confirms status

(this repo). Partially delivered: `falkor-chat/compose.yaml`
already models the shared FalkorDB service (`v4.18.11`, 6379/3000, `external: true` on the
`falkordb-data` volume) plus the falkor-chat server, but it is component-scoped — `salesperson`
still boots via `start_falkordb.sh` + a manual Streamlit run, so the *whole stack* is not yet
`compose up`-able. Note the two are mutually exclusive on :6379 by design.

### GAL-I02-K0008 — First GitHub Actions CI workflow

- **Status:** 🔵 proposed — parking-lot origin; owner confirms status

(this repo): lint + `pytest` + `./scripts/test_queries.sh`
against a FalkorDB service container) — the obvious greenfield deliverable. Note this is a
*re*-introduction: `.github/workflows/falkor-chat.yml` existed and was removed in `5a97821`, so
start by reading that deleted file rather than from scratch.

### GAL-I02-K0009 — `salesperson` Streamlit app image

- **Status:** 🔵 proposed — parking-lot origin; owner confirms status

(this repo). `falkor-chat/Dockerfile` and
`cypher-mcp/Dockerfile` already exist; salesperson is the remaining gap.

### GAL-I02-K0010 — `salesperson/build.sh` node-branch `/mnt/` gap

- **Status:** 🔵 proposed — parking-lot origin; owner confirms status

(this repo). The third resolution branch (`elif command -v node`) accepts
whatever `node` is on `PATH` without applying the `/mnt/` rejection that its own `npm`-only branch
and `claude/devops/ops-quirks.md` both state. Harmless today (there is no Windows `node` on this
box's `PATH`, only `npm`), but it is the one gap between the script and the rule. Observed
2026-09-07 during the U3 distillation pass; `salesperson/` is not `cobb`'s to edit.

### GAL-I02-K0011 — A behavioral eval in the cobb/TESTING.md harness style

- **Status:** 🔵 proposed — parking-lot origin; owner confirms status

A behavioral eval in the cobb/TESTING.md harness style — e.g. assert the agent (a) reads the
project's README/docs before proposing infra changes, and (b) refuses to `docker volume rm` a
shared data volume without confirmation.
