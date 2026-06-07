# graphmind-ai-lab

A monorepo of independent, self-contained components — there is **no root-level build/test
script**. Each component carries its own docs and run instructions. Two themes run through the
repo: **graph-backed AI apps** (FalkorDB + LLMs) and **agent/skill engineering** (Claude Code
and OpenCode artifacts).

## Structure

- `salesperson/` — Streamlit sales-assistant chatbot. FalkorDB knowledge graph + LangChain +
  LangGraph; optional local LLM via LM Studio. See `salesperson/AGENTS.md`.
- `falkor-chat/` — Hybrid chat system (humans + AI) where **FalkorDB is the single store for
  everything**: chat history, workspace/reference data, workflow definitions and execution
  traces. GraphRAG (in-graph vector + traversal) and graph-state-machine workflows. Design and
  query library are locked and live-verified; M0 complete. See `falkor-chat/README.md` and
  `falkor-chat/CLAUDE.md`.
- `opencode/` — Personal OpenCode configuration: custom agents and skills.
  - `agents/` — `rpg`, `coding-senior`, and `severino/` (a full LM-Studio-backed local agent project).
  - `skills/` — `python-coding`, `write-tutorial`, `comparison-driver`, `skill-builder`, `user-preferences`.
  - `local-llm.md` — notes on running OpenCode against a local LM Studio server.
- `claude/` — Custom Claude Code subagents, one folder per agent, each with a `kaizen/` plan +
  history. See `claude/README.md` (human catalog) and `claude/CLAUDE.md` (agent context).

## Component docs (read before working in a component)

| Component | Entry doc(s) |
|---|---|
| `salesperson/` | `salesperson/AGENTS.md` · `salesperson/README.md` |
| `falkor-chat/` | `falkor-chat/README.md` · `falkor-chat/CLAUDE.md` · `falkor-chat/docs/DESIGN.md` · `falkor-chat/docs/QUERIES.md` |
| `opencode/` | `opencode/skills/*/SKILL.md` · `opencode/agents/severino/README.md` · `opencode/local-llm.md` |
| `claude/` | `claude/README.md` · `claude/CLAUDE.md` |

## Claude Code subagents (`claude/`)

Folder-per-agent: `<name>/<name>.md` (Markdown + YAML frontmatter) plus `<name>/kaizen/{plan,history}.md`.
All currently use `model: opus`.

| Agent | What it does | When to use |
|---|---|---|
| `cobb` | Agentic-development expert across Claude Code, Kiro, OpenCode, and cross-tool standards (`AGENTS.md`, Agent Skills). | Designing/authoring/reviewing/porting/debugging agents, subagents, skills, steering docs, hooks, slash commands, system prompts. |
| `dra-claudia` | Médica de homeopatia/medicina alternativa; mantém prontuário markdown por paciente. | Perguntas de saúde/sintomas/tratamentos e registro/consulta de histórico clínico (PT-BR; não substitui consulta presencial). |
| `graph-dba` | Graph DBA & data architect specialized in **FalkorDB** (GraphBLAS engine, RedisGraph successor, GraphRAG-focused); fluent in Neo4j/openCypher/GQL for porting. | Graph data modeling, FalkorDB Cypher authoring/tuning, indexes/constraints, deployment & RAM sizing, `GRAPH.PROFILE` query tuning, bulk ingestion, GraphRAG layers, ops. |
| `tdd-engineer` | Implements features/fixes strictly via TDD (red → green → refactor), keeping the suite green. | Implementing a feature, fixing a bug, refactoring with a safety net, adding/improving tests. |

## OpenCode agents (`opencode/agents/`)

- `coding-senior` — subagent; senior architect that does impact analysis and plans before
  changing code (`edit`/`bash` set to `ask`).
- `rpg` — friendly conversational primary agent; uses the `user-preferences` skill to remember
  the user across sessions.
- `severino/` — a full local agent project (own `opencode.json`, `README.md`, tests). Read-only
  coding advisor backed by **LM Studio** (`lmstudio/<model-id>`, default Nemotron 3 Nano 4B).
  Gotchas: top-level key is `agent` (singular), no `name` field on the agent, LM Studio context
  ≥16K. See `opencode/agents/severino/CLAUDE.md`.

## User-preferences skill (shared memory pattern)

`opencode/skills/user-preferences/` gives conversational agents persistent memory:
- Storage: `storage/{work,hobbies,communication,general}.md`
- Protocol: read preference files at conversation start, grep to search, write new prefs to the
  right category file.
- Used by the `rpg` agent.

## Key commands

**salesperson/** (run from that directory):
```bash
./start_falkordb.sh            # start FalkorDB (required before the app)
python create_kg_pastel.py     # seed graph data (wipes kg_pastel)
streamlit run chatbot.py       # run app
python visualize_agent_graph.py
```

**falkor-chat/** (run from that directory):
```bash
./start_falkordb.sh                          # FalkorDB in Docker (foreground); web console :3000
./scripts/bootstrap_schema.sh <workspaceId>  # create indexes + constraints (idempotent)
./scripts/test_queries.sh                    # end-to-end query suite — baseline 64/64 passed
```

**severino** (run from `opencode/agents/severino/`):
```bash
opencode --agent severino      # requires LM Studio server running at :1234
```

## Working in this repo

- **Chatbot tasks** → `salesperson/`, follow `salesperson/AGENTS.md`. No pytest/lint scripts; manual checks.
- **FalkorDB chat platform** → `falkor-chat/`, follow `falkor-chat/CLAUDE.md`. This is FalkorDB
  OpenCypher (not Neo4j): no APOC/GDS, vector indexes via DDL, index-before-constraint. Keep the
  query suite green (`./scripts/test_queries.sh`, 64/64).
- **OpenCode skill/agent tasks** → `opencode/`, follow each `SKILL.md` / the severino docs.
- **Claude subagent tasks** → `claude/`, follow `claude/CLAUDE.md`. Adding/editing/renaming an
  agent means updating its source, its `kaizen/{plan,history}.md`, `claude/README.md`, and
  `claude/CLAUDE.md` in the same change.
- The root `CLAUDE.md` is intentionally empty; per-component context files carry the detail.
</content>
</invoke>
