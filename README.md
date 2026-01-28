# GraphMind AI Lab

A Streamlit chatbot for “Pastel do Mau” (Brazilian pastel shop) powered by FalkorDB + LLMs.

## Overview

- **create_kg_pastel.py** – seeds the `kg_pastel` graph with flavors, ingredients, and prices.
- **graph.py** – central FalkorDB connector plus schema helpers.
- **cypher.py** – LangChain tool that builds and runs Cypher queries.
- **agent.py / chatbot.py** – Streamlit UI driving a ReAct-style agent with tools.
- **prompts.py** – central prompt strings for the agent (English internals, PT-BR customer copy).
- **diagnostics.py** – lightweight session snapshot helper for debugging/support.
- **visualize_agent_graph.py** – optional helper to export the LangGraph workflow (Mermaid + PNG).

## Requirements

- Python 3.12+ (virtualenv recommended)
- FalkorDB access (local Docker or remote instance)
- OpenAI API key compatible with your selected model (gpt‑4, gpt‑4o, etc.)

## Run FalkorDB

Use the helper script (see `start_falkordb.sh` for port/image overrides):

```bash
./start_falkordb.sh
```

## App configuration

1. Copy `.streamlit/secrets.toml.example` → `.streamlit/secrets.toml`.
2. Fill in your credentials:

```toml
OPENAI_API_KEY = "sk-..."
OPENAI_MODEL = "gpt-4o"
FALKORDB_URL = "redis://localhost:6379"
FALKORDB_GRAPH = "kg_pastel"
LOG_LEVEL = "DEBUG"
```

Prefer host/port credentials? Remove `FALKORDB_URL` and set:

```toml
FALKORDB_HOST = "localhost"
FALKORDB_PORT = 6379
# FALKORDB_USERNAME = ""
# FALKORDB_PASSWORD = ""
LOG_LEVEL = "DEBUG"

Prefer environment variables instead of Streamlit secrets? Use `.env.example` as a reference.

### Configuration reference

| Key | Purpose | Default |
| --- | --- | --- |
| `OPENAI_API_KEY` | OpenAI API key | none |
| `OPENAI_MODEL` | OpenAI model name | none |
| `FALKORDB_URL` | FalkorDB URL | `redis://localhost:6379` |
| `FALKORDB_GRAPH` | Graph name | `kg_pastel` |
| `FALKORDB_HOST` | FalkorDB host (when not using URL) | `localhost` |
| `FALKORDB_PORT` | FalkorDB port (when not using URL) | `6379` |
| `FALKORDB_USERNAME` | FalkorDB username | empty |
| `FALKORDB_PASSWORD` | FalkorDB password | empty |
| `LOG_LEVEL` | Logging level | `DEBUG` |
| `LOG_DIR` | Log directory | `logs` |
| `LOG_SESSION_HANDLER_LIMIT` | Max active session log handlers | `100` |
| `LOG_EXCLUDE_PREFIXES` | Comma-separated logger prefixes to exclude from file logs | `watchdog,streamlit,httpcore` |
```

## Populate the graph

```bash
python create_kg_pastel.py
```

This wipes `kg_pastel`, recreates all `Pastel`/`Ingrediente` nodes, adds prices (R$19–50), and prints a summary.

## Launch the chatbot

```bash
streamlit run chatbot.py
```

Want to inspect the agent graph?

```bash
./.venv/bin/python visualize_agent_graph.py
```

This writes `agent_workflow.mmd` (Mermaid) and tries to save `agent_workflow.png`. If network rendering is blocked, open the `.mmd` file in a Mermaid viewer (e.g., mermaid.live) or render locally with `mmdc`.

Highlights:
- Per-session memory to keep the conversation coherent.
- The `menu` tool automatically generates Cypher queries for flavors/ingredients/prices.
- Integrated cart workflow: the assistant can add/view/clear the cart via LangGraph tools, and the Streamlit sidebar reflects the current items and total.
- Structured customer profile capture: the agent collects the customer's name upfront and, once there is an order, confirms the delivery address, showing everything in the sidebar with a quick reset button for demos.
- Verbose logging so you can inspect the generated Cypher and results (see logs below).

## Logging

- App-wide logs: `logs/app.log`.
- Per-session logs: `logs/session_<session_id>.log` (created once a session is established).
- File logs exclude `watchdog`, `streamlit`, and `httpcore` by default to focus on AI/graph activity.

## Troubleshooting

- **No data returned**: rerun `create_kg_pastel.py` and confirm the FalkorDB container is running.
- **Credential errors**: double-check `.streamlit/secrets.toml` and restart Streamlit.
- **Tool skipping the graph**: watch the `menu` logs in the terminal; they show the Cypher queries and results.
- **Missing file logs**: ensure a session was created and restart Streamlit after changing log-related env vars.

## Support snapshot

Need a quick health check while debugging? Use the session snapshot helper:

```bash
python3 -c "from diagnostics import get_session_snapshot; print(get_session_snapshot())"
```

Happy frying! :)
