# Claude Agents

Custom [Claude Code subagents](https://code.claude.com/docs) for this repo. Each lives in its own folder with the agent source (`<name>.md`, Markdown + YAML frontmatter) and a `kaizen/` folder holding its improvement plan and change history.

| Agent | What it does | When to use it | Model |
|-------|--------------|----------------|-------|
| [`cobb`](./cobb/cobb.md) | Senior practitioner of agentic development; deep, current knowledge of Claude Code, Kiro, and OpenCode agent formats and the cross-tool standards (`AGENTS.md`, Agent Skills). | Designing, authoring, reviewing, porting, or debugging agents, subagents, skills, steering docs, slash commands, hooks, or system prompts. | opus |
| [`dra-claudia`](./dra-claudia/dra-claudia.md) | Dra. Cláudia — médica homeopatia e medicina alternativa; mantém prontuário em markdown de cada paciente. | Perguntas sobre saúde, sintomas, tratamentos, remédios homeopáticos/fitoterápicos, abordagens integrativas, ou registrar/consultar histórico clínico. | opus |
| [`graph-dba`](./graph-dba/graph-dba.md) | Graph database administrator & data architect specialized in **FalkorDB** (Redis-module, GraphBLAS sparse-matrix engine; RedisGraph successor; built for GraphRAG). Covers its OpenCypher dialect, modeling, vector/full-text indexing, constraints, multi-graph tenancy, in-memory sizing, replication/clustering, and tuning via `GRAPH.PROFILE`. Fluent in the wider LPG world (Neo4j, openCypher, GQL) for porting. | Designing a graph data model, writing/optimizing FalkorDB Cypher, indexes/constraints, FalkorDB deployment (RAM sizing, persistence, replication, Redis Cluster), tuning slow traversals, bulk ingestion/migration, building a GraphRAG/knowledge-graph layer, or FalkorDB ops. | opus |
| [`tdd-engineer`](./tdd-engineer/tdd-engineer.md) | Senior engineer who implements features and fixes strictly via Test-Driven Development (red → green → refactor), keeping the suite green at every step. | Implementing a feature, fixing a bug, refactoring with a safety net, or adding/improving tests. | opus |

## Kaizen

Each agent carries a living improvement plan and change log:

- `cobb/kaizen/` — [plan](./cobb/kaizen/plan.md) · [history](./cobb/kaizen/history.md)
- `dra-claudia/kaizen/` — [plan](./dra-claudia/kaizen/plan.md) · [history](./dra-claudia/kaizen/history.md)
- `graph-dba/kaizen/` — [plan](./graph-dba/kaizen/plan.md) · [history](./graph-dba/kaizen/history.md)
- `tdd-engineer/kaizen/` — [plan](./tdd-engineer/kaizen/plan.md) · [history](./tdd-engineer/kaizen/history.md)

`cobb` additionally maintains [`cobb/TESTING.md`](./cobb/TESTING.md) (agent testing standards).

## Skills

Skills were unified into the repo-root [`skills/`](../skills/) home — see [`skills/README.md`](../skills/README.md) for the full catalog. `cobb` relies on two of them:

- [**`agent-maintenance`**](../skills/agent-maintenance/SKILL.md) — kaizen/documentation/file-location/drift-audit machinery `cobb` follows when it creates/edits/reviews an agent or skill.
- [**`agent-standards`**](../skills/agent-standards/SKILL.md) — perishable per-tool reference specifics (frontmatter fields, paths, inclusion modes) for Claude Code, Kiro, and OpenCode, each `Verified:`-stamped.

## Deployment

These agents live in this repo but run from Claude Code's config dir via symlinks:

- **Agents:** `~/.claude/agents/<name>` → `claude/<name>` (one symlink per agent folder).
- **Skills:** now in the repo-root [`skills/`](../skills/) home, deployed via `~/.claude/skills` →
  `skills/` (whole-dir symlink; all 7 skills visible to Claude Code). Also symlinked into OpenCode
  and Kiro — see [`skills/README.md`](../skills/README.md#deployment).

So edits here are picked up live, and the definitions stay version-controlled.

### WebFetch / WebSearch allowlist (for `cobb`)

`cobb`'s mandate is to **verify version-sensitive specifics against live official docs** (and the
[`agent-standards`](../skills/agent-standards/SKILL.md) skill re-checks its `Verified:` stamps the
same way). To let those fetches run without a confirmation prompt, the official doc domains are
allowlisted in **`~/.claude/settings.json`** (user scope) under `permissions.allow`:

```json
{
  "permissions": {
    "allow": [
      "WebFetch(domain:code.claude.com)",
      "WebFetch(domain:platform.claude.com)",
      "WebFetch(domain:docs.anthropic.com)",
      "WebFetch(domain:kiro.dev)",
      "WebFetch(domain:opencode.ai)",
      "WebSearch"
    ]
  }
}
```

Notes:
- **User scope, so it's session-wide** — every agent in the session (not just `cobb`) can fetch
  these five hosts and run `WebSearch` unprompted; any other URL still prompts. For a strictly
  cobb-only allowlist you'd use a subagent-scoped `PreToolUse` hook instead (more machinery; see
  cobb's kaizen history).
- This file is **personal/global and not in this repo** — re-add the block on a new machine.
- Takes effect on new sessions; open `/permissions` or restart if it doesn't load immediately.
- Match the *final* host: `WebFetch` returns cross-host redirects to be re-fetched, so the
  redirect destination must also be listed (e.g. `docs.anthropic.com`).

## Conventions

- **Folder per agent:** `<name>/<name>.md` is the source; `<name>/kaizen/{plan,history}.md` track improvements.
- **Frontmatter** drives routing: the `description` says *what the agent does and precisely when to use it* so Claude Code can auto-delegate.
- When you add, edit, rename, or remove an agent, keep this catalog and `CLAUDE.md` in sync, and update the agent's `kaizen/` files.
