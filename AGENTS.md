# graphmind-ai-lab

Monorepo with two self-contained components — no root-level build/test scripts.

## Structure
- `salesperson/` — Streamlit chatbot (FalkorDB + LangChain + LangGraph)
- `opencode/` — Personal OpenCode skill configurations
  - `skills/` — OpenCode skills (python-coding, write-tutorial, comparison-driver, skill-creator)
  - `agents/` — Custom OpenCode agents (placeholder for future agents; creation/edition per OpenCode standards)

## Component docs
- `salesperson/AGENTS.md` — Chatbot-specific guidance
- `opencode/skills/*/SKILL.md` — Skill instructions

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
- No pytest or lint scripts; manual code checks only