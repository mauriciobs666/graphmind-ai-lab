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
  `falkor-chat/AGENTS.md`.
- `opencode/` — Personal OpenCode configuration: custom agents.
  - `agents/` — `rpg`, `coding-senior`, and `severino/` (a full LM-Studio-backed local agent project).
  - `local-llm.md` — notes on running OpenCode against a local LM Studio server.
- `claude/` — Custom Claude Code subagents (one folder per agent, each with a `kaizen/` plan +
  history). See `claude/README.md` (human catalog) and `claude/AGENTS.md` (agent context;
  `claude/CLAUDE.md` is a `@AGENTS.md` import stub).
- `skills/` — **Unified Agent Skills home** (`SKILL.md` packages, the open
  `agentskills.io` standard) shared across the repo's tools. `agent-maintenance` +
  `agent-standards` (cobb's machinery) and `python-coding`, `write-tutorial`,
  `comparison-driver`, `skill-builder`, `user-preferences` (OpenCode-authored). See
  `skills/README.md`. Format ports across Claude Code/OpenCode/Kiro; tool-gating &
  activation behavior do not — verify per tool.

## Component docs (read before working in a component)

| Component | Entry doc(s) |
|---|---|
| `salesperson/` | `salesperson/AGENTS.md` · `salesperson/README.md` |
| `falkor-chat/` | `falkor-chat/README.md` · `falkor-chat/AGENTS.md` · `falkor-chat/docs/DESIGN.md` · `falkor-chat/docs/QUERIES.md` |
| `opencode/` | `opencode/agents/severino/README.md` · `opencode/local-llm.md` |
| `claude/` | `claude/README.md` · `claude/AGENTS.md` (Claude Code reads it via the `claude/CLAUDE.md` import) |
| `skills/` | `skills/README.md` · `skills/*/SKILL.md` |

## Claude Code subagents (`claude/`)

Folder-per-agent: `<name>/<name>.md` (Markdown + YAML frontmatter) plus `<name>/kaizen/{plan,history}.md`.
All currently use `model: opus`. The skills cobb loads on demand (`agent-maintenance`,
`agent-standards`) now live in the repo-root `skills/` home (see the `skills/` section below),
not under `claude/`.

| Agent | What it does | When to use |
|---|---|---|
| `teco` | Technical coordinator; decomposes a goal into a sequenced work breakdown and delegates each piece to the right specialist (architect, coder, tdd-engineer, graph-dba, cobb), then integrates. Hybrid: drives execution but returns to the user at decision points. Coordinates — doesn't design/code. | A multi-step / multi-specialty task, orchestration, or end-to-end feature delivery. |
| `architect` | Software architect; investigates code, weighs trade-offs, produces a step-by-step implementation plan/spec. **Read-only on code** (Write/Edit = plan doc only). Planning half of an architect→coder handoff. | Wanting a design, approach, impact analysis, or plan before any code is written. |
| `coder` | Software engineer who implements an approved plan/spec end-to-end — clean, tested, repo-idiomatic code, suite kept green. Implementation half of the handoff. | Building from a ready plan/spec or clear task. (Strict test-first → `tdd-engineer`.) |
| `cobb` | Agentic-development expert across Claude Code, Kiro, OpenCode, and cross-tool standards (`AGENTS.md`, Agent Skills). | Designing/authoring/reviewing/porting/debugging agents, subagents, skills, steering docs, hooks, slash commands, system prompts. |
| `dra-claudia` | Médica de homeopatia/medicina alternativa; mantém prontuário markdown por paciente. | Perguntas de saúde/sintomas/tratamentos e registro/consulta de histórico clínico (PT-BR; não substitui consulta presencial). |
| `saul` | Assessor jurídico (foro de SP) de direito civil e penal, com foco aprofundado em **direito condominial** (condomínio edilício). Monta dossiês markdown por caso (`$AGENT_WORKDIR/saul/dossies/{cliente}/{caso}.md`), cita os dispositivos legais, busca a vigência atual na web e separa fato verificado de tese; redige minutas tarjadas como rascunho. Não substitui advogado(a) OAB (PT-BR). | Perguntas de direito civil/penal, condomínio, cobrança de cotas, assembleias/convenção, contratos, notificações, defesa/acusação, ou montar/consultar o dossiê de um caso. |
| `graph-dba` | Graph DBA & data architect specialized in **FalkorDB** (GraphBLAS engine, RedisGraph successor, GraphRAG-focused); fluent in Neo4j/openCypher/GQL for porting. | Graph data modeling, FalkorDB Cypher authoring/tuning, indexes/constraints, deployment & RAM sizing, `GRAPH.PROFILE` query tuning, bulk ingestion, GraphRAG layers, ops. |
| `tdd-engineer` | Implements features/fixes strictly via TDD (red → green → refactor), keeping the suite green. | Implementing a feature, fixing a bug, refactoring with a safety net, adding/improving tests. |
| `qa-engineer` | QA / functional-testing engineer at the behavior/acceptance altitude (black-box complement to `tdd-engineer`). Reasons a risk-based **strategy** → writes a versioned **test plan** (`docs/test-plans/<kebab>.md`) → **executes** it (authors automated functional/acceptance tests, runs existing suites, and drives the running app black-box) → delivers a **test report** (`docs/test-reports/<kebab>-report.md`) with results, reproducible defects, coverage/gaps, and feedback. Evidence-over-assertion; doesn't fix code under test unless asked. | Wanting a test strategy/plan, functional/acceptance/integration/e2e/exploratory testing, a QA pass on a feature or release, or a written report of what was tested and what broke. |

## OpenCode agents (`opencode/agents/`)

- `coding-senior` — subagent; senior architect that does impact analysis and plans before
  changing code (`edit`/`bash` set to `ask`).
- `rpg` — friendly conversational primary agent; uses the `user-preferences` skill to remember
  the user across sessions.
- `severino/` — a full local agent project (own `opencode.json`, `README.md`, tests). Read-only
  coding advisor backed by **LM Studio** (`lmstudio/<model-id>`, default Nemotron 3 Nano 4B).
  Gotchas: top-level key is `agent` (singular), no `name` field on the agent, LM Studio context
  ≥16K. See `opencode/agents/severino/AGENTS.md` (with a `CLAUDE.md` = `@AGENTS.md` stub).

## Skills (`skills/`)

Unified Agent Skills home — one `SKILL.md` package per folder, the open `agentskills.io`
standard. Shared across the repo's tools; the **format** ports (Claude Code, OpenCode, Kiro all
read `SKILL.md`), but **tool-gating and activation behavior do not** — verify per tool. See
`skills/README.md` for the catalog.

- `agent-maintenance`, `agent-standards` — cobb's machinery (kaizen/doc/drift procedures;
  perishable per-tool reference specifics). Loaded on demand so cobb's prompt stays lean.
- `python-coding`, `write-tutorial`, `comparison-driver`, `skill-builder`, `user-preferences` —
  OpenCode-authored skills.

> **Deployment:** all three harnesses point at this home via whole-dir symlinks (all 7 skills
> visible to each): `~/.claude/skills`, `~/.config/opencode/skills`, and `~/.kiro/skills` →
> `skills/`. See `skills/README.md` to recreate on a new machine.

## User-preferences skill (shared memory pattern)

`skills/user-preferences/` gives conversational agents persistent memory:
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
./scripts/start_falkordb.sh                  # FalkorDB in Docker (foreground; -d for headless); web console :3000
./scripts/bootstrap_schema.sh <workspaceId>  # create indexes + constraints (idempotent)
./scripts/test_queries.sh                    # end-to-end query suite — baseline 67/67 passed
```

**severino** (run from `opencode/agents/severino/`):
```bash
opencode --agent severino      # requires LM Studio server running at :1234
```

## Working in this repo

- **Chatbot tasks** → `salesperson/`, follow `salesperson/AGENTS.md`. No pytest/lint scripts; manual checks.
- **FalkorDB chat platform** → `falkor-chat/`, follow `falkor-chat/AGENTS.md` (`falkor-chat/CLAUDE.md`
  imports it). This is FalkorDB
  OpenCypher (not Neo4j): no APOC/GDS, vector indexes via DDL, index-before-constraint. Keep the
  query suite green (`./scripts/test_queries.sh`, 67/67).
- **OpenCode agent tasks** → `opencode/`, follow the severino docs / `opencode/local-llm.md`.
- **Skill tasks** (any tool) → `skills/`, follow each `skills/<name>/SKILL.md` and `skills/README.md`.
- **Claude subagent / skill tasks** → `claude/` (agents) and `skills/` (skills), follow
  `claude/AGENTS.md`. Adding/editing/renaming an agent or skill means updating its source, its
  `kaizen/{plan,history}.md`, the relevant catalog (`claude/README.md` for agents, `skills/README.md`
  for skills), and `claude/AGENTS.md` in the same change.
- The root `CLAUDE.md` contains only `@AGENTS.md` — this file is the single source of truth for
  root-level context; per-component context files carry the detail.
</content>
</invoke>
