# Kaizen — Improvement Plan: devops

> Forward-looking backlog for the `devops` agent.
> Status: 🔵 proposed · 🟡 in-progress · ✅ done (then moved to history.md) · ⚪ rejected/deferred
> Last reviewed: 2026-07-02

## Active

| ID | Added | Priority | Status | Summary |
|-------|------------|----------|--------|---------|
| K-002 | 2026-07-02 | low | 🔵 | Add a WebFetch allowlist for infra doc domains (docker.com, docs.docker.com, docs.github.com) |
| K-003 | 2026-07-02 | low | 🔵 | Revisit persona vs. role framing once the agent has run a few real tasks |
| K-004 | 2026-07-02 | low | 🔵 | Guard hook is single-machine (absolute path + machine deps) — make it relocatable / tunable |

> ✅ **K-001 — Enforce the destructive-op guardrail deterministically** — DONE 2026-07-02.
> Implemented as a subagent-scoped `PreToolUse` hook (`devops/hooks/guard-destructive-ops.sh`,
> wired in `devops.md` frontmatter), returning `permissionDecision: "ask"`. Kept it **in-repo** via
> the frontmatter `hooks:` field (not `settings.json`) — the original trade-off note is resolved:
> subagent frontmatter hooks are supported and scoped to this agent. See history 2026-07-02.

### K-004 — Make the guard hook relocatable
- **Status:** 🔵 proposed
- **Priority:** low
- **Rationale:** The frontmatter hook uses an **absolute path** to the script in this repo, and the
  script prefers `jq`→`python3`. On a new machine/clone-path both the deploy symlink *and* this path
  need re-pointing, and a box with neither jq nor python3 would fail-open (patterns still run on the
  raw payload via grep, so it mostly holds, but extraction is degraded).
- **Proposed change:** Consider a portable indirection (e.g. resolve the script relative to a stable
  env var, or a tiny installer that rewrites the path on deploy) and/or make the blocked-pattern list
  configurable. Only worth it if this agent gets deployed across machines.
- **Notes:** Same machine-specificity as the `~/.claude/agents/devops` symlink — document both in the
  "re-add on a new machine" checklist.

### K-002 — Infra-docs WebFetch allowlist
- **Status:** 🔵 proposed
- **Priority:** low
- **Rationale:** Like cobb verifies agent-tool docs live, devops will want to check Docker/Compose/
  GitHub Actions docs. Those domains aren't in the current `~/.claude/settings.json` allowlist, so
  fetches prompt.
- **Proposed change:** Add `WebFetch(domain:docs.docker.com)`, `docker.com`, `docs.github.com` to
  the user-scope allowlist if verification-heavy infra work becomes common. Personal/global file —
  document in `claude/README.md` alongside cobb's block.

### K-003 — Persona vs. role framing
- **Status:** 🔵 proposed
- **Priority:** low
- **Rationale:** User chose the plain role name `devops` over a persona. If the repo's persona
  culture (teco/saul/severino) proves to matter for how this agent is invoked/relatable, a persona
  alias could be layered on. Low stakes.

## Parking lot / ideas
- **(this repo)** Extend Compose coverage to `salesperson`. Partially delivered: `falkor-chat/compose.yaml`
  already models the shared FalkorDB service (`v4.18.11`, 6379/3000, `external: true` on the
  `falkordb-data` volume) plus the falkor-chat server, but it is component-scoped — `salesperson`
  still boots via `start_falkordb.sh` + a manual Streamlit run, so the *whole stack* is not yet
  `compose up`-able. Note the two are mutually exclusive on :6379 by design.
- **(this repo)** A first GitHub Actions CI workflow (lint + `pytest` + `./scripts/test_queries.sh`
  against a FalkorDB service container) — the obvious greenfield deliverable. Note this is a
  *re*-introduction: `.github/workflows/falkor-chat.yml` existed and was removed in `5a97821`, so
  start by reading that deleted file rather than from scratch.
- **(this repo)** A `salesperson` Streamlit app image. `falkor-chat/Dockerfile` and
  `cypher-mcp/Dockerfile` already exist; salesperson is the remaining gap.
- A behavioral eval in the cobb/TESTING.md harness style — e.g. assert the agent (a) reads the
  project's README/docs before proposing infra changes, and (b) refuses to `docker volume rm` a
  shared data volume without confirmation.
- ⚪ **Keep the graphmind-ai-lab example in the prompt lightweight** — **resolved by deletion,
  2026-08-24 (C4).** The item's own remedy (refresh it when the repo's infra changes materially) was
  applied twice, on 2026-07-09 and 2026-07-11, and the block rotted a third time anyway — in a
  *different* slot each time, so repairing the fact that broke last time never protected the ones
  that broke next. Deleted outright rather than refreshed a third time; this agent is user-scoped, so
  a memorized snapshot of one repo was a false anchor in every other project, on the one agent whose
  whole remit is "don't generalize from another repo." The item's own closing sentence had it right —
  the authoritative detail lives in the repo's `AGENTS.md`, which orientation reads anyway.
