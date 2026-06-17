# Claude agents — context for AI agents working here

This directory (`claude/`) holds custom Claude Code subagents. Each agent is a folder: `<name>/<name>.md` (Markdown + YAML frontmatter) plus `<name>/kaizen/{plan,history}.md`. **Skills no longer live here** — they were unified into the repo-root [`skills/`](../skills/) home (see [`skills/README.md`](../skills/README.md)); cobb's `agent-maintenance` and `agent-standards` skills are there. See [`README.md`](./README.md) for the human-facing agent catalog.

## Agents

- **cobb** — agentic-development expert (Claude Code, Kiro, OpenCode agent formats + cross-tool standards). Use for designing/authoring/reviewing/porting/debugging agents, skills, steering docs, hooks, system prompts. Source: `cobb/cobb.md`. Kaizen: `cobb/kaizen/`. Testing standards: `cobb/TESTING.md`. Relies on two skills in the repo-root [`skills/`](../skills/) home: **`agent-maintenance`** for its kaizen/documentation/drift-audit procedures, and **`agent-standards`** for the perishable per-tool reference specifics (frontmatter fields, paths, inclusion modes). The resident prompt keeps only the mandate + stable mental models and points at the skills.
- **dra-claudia** — médica de homeopatia/medicina alternativa; mantém prontuários em markdown. Use para perguntas de saúde/sintomas/tratamentos e registro/consulta de histórico clínico (não substitui consulta presencial). Source: `dra-claudia/dra-claudia.md`. Kaizen: `dra-claudia/kaizen/`.
- **tdd-engineer** — implements features/fixes strictly via TDD (red → green → refactor). Use for implementing a feature, fixing a bug, refactoring with a safety net, or adding/improving tests. Source: `tdd-engineer/tdd-engineer.md`. Kaizen: `tdd-engineer/kaizen/`.
- **graph-dba** — graph database administrator & data architect specialized in **FalkorDB** (Redis-module, GraphBLAS sparse-matrix engine; RedisGraph successor; GraphRAG-focused), fluent in the wider LPG world (Neo4j, openCypher, GQL) for porting. Use for graph data modeling, writing/optimizing FalkorDB Cypher, vector/full-text indexing & constraints, deployment (RAM sizing, persistence, replication, Redis Cluster), query tuning (`GRAPH.PROFILE`), bulk ingestion/migration, GraphRAG/knowledge-graph layers, and ops. Source: `graph-dba/graph-dba.md`. Kaizen: `graph-dba/kaizen/`.

All four use `model: opus`.

## Maintenance rules

- Adding/editing/renaming/removing an agent → update the agent source, its `kaizen/{plan,history}.md`, this file, and `README.md` in the same change.
- Skills live in the repo-root [`skills/`](../skills/) home, not here. Their catalog is [`skills/README.md`](../skills/README.md); cobb's kaizen logs changes to `agent-maintenance`/`agent-standards`.
- Don't paste full system prompts here — point to the source file.
