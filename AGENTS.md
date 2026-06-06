# graphmind-ai-lab

Monorepo with two self-contained components — no root-level build/test scripts.

## Structure
- `salesperson/` — Streamlit chatbot (FalkorDB + LangChain + LangGraph)
- `opencode/` — Personal OpenCode skill configurations
  - `skills/` — OpenCode skills (python-coding, write-tutorial, comparison-driver, skill-builder, user-preferences)
  - `agents/` — Custom OpenCode agents (rpg, coding-senior; creation/edition per OpenCode standards)
- `claude/` — Custom Claude Code subagents (cobb, dra-claudia, tdd-engineer); folder-per-agent with `kaizen/` plan+history

## Component docs
- `salesperson/AGENTS.md` — Chatbot-specific guidance
- `opencode/skills/*/SKILL.md` — Skill instructions
- `claude/README.md` — Claude subagents catalog · `claude/CLAUDE.md` — context for agents working there

## User Preferences Skill

The `user-preferences` skill provides persistent memory for conversational agents:
- Storage: `opencode/skills/user-preferences/storage/` (work.md, hobbies.md, communication.md, general.md)
- Protocol: Read preference files at conversation start, search with grep, write new preferences to category files
- Used by: RPG agent

## Key commands (run from salesperson/)

```bash
# Start FalkorDB (required before app)
./start_falkordb.sh

# Seed graph data (wipes kg_pastel)
python create_kg_pastel.py

# Run app
streamlit run chatbot.py

# Visualize LangGraph workflow
python visualize_agent_graph.py
```

## Working in this repo
- Chatbot tasks: use `salesperson/` workdir, follow `salesperson/AGENTS.md`
- Skill tasks: use `opencode/` workdir
- Claude subagent tasks: use `claude/` workdir, follow `claude/CLAUDE.md`
- No pytest or lint scripts; manual code checks only