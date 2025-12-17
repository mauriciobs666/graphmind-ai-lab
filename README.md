# GraphMind AI Lab

A Streamlit chatbot for “Pastel do Mau” (Brazilian pastel shop) powered by FalkorDB + LLMs.

## Overview

- **create_kg_pastel.py** – seeds the `kg_pastel` graph with flavors, ingredients, and prices.
- **graph.py** – central FalkorDB connector plus schema helpers.
- **cypher.py** – LangChain tool that builds and runs Cypher queries.
- **agent.py / chatbot.py** – Streamlit UI driving a ReAct-style agent with tools.

## Requirements

- Python 3.12+ (virtualenv recommended)
- FalkorDB access (local Docker or remote instance)
- OpenAI API key compatible with your selected model (gpt‑4, gpt‑4o, etc.)

## Run FalkorDB

```bash
docker run -p 6379:6379 -p 3000:3000 -it --rm falkordb/falkordb:edge
```
- `6379` exposes the Redis/FalkorDB endpoint.
- `3000` exposes the optional web console.

## App configuration

1. Copy `.streamlit/secrets.toml.example` → `.streamlit/secrets.toml`.
2. Fill in your credentials:

```toml
OPENAI_API_KEY = "sk-..."
OPENAI_MODEL = "gpt-4o"
FALKORDB_URL = "redis://localhost:6379"
FALKORDB_GRAPH = "kg_pastel"
```

Prefer host/port credentials? Remove `FALKORDB_URL` and set:

```toml
FALKORDB_HOST = "localhost"
FALKORDB_PORT = 6379
# FALKORDB_USERNAME = ""
# FALKORDB_PASSWORD = ""
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

Highlights:
- Per-session memory to keep the conversation coherent.
- The `cardapio` tool automatically generates Cypher queries for flavors/ingredients/prices.
- Integrated cart workflow: the assistant can adicionar/ver/limpar o carrinho via LangGraph tools, and the Streamlit sidebar reflects the current items and total.
- Structured customer profile capture: the agent now collects the customer's name upfront and, once there is an order, confirms delivery address and payment method, echoing everything in the sidebar with a quick reset button for demos.
- Verbose logging so you can inspect the generated Cypher and results.

## Troubleshooting

- **No data returned**: rerun `create_kg_pastel.py` and confirm the FalkorDB container is running.
- **Credential errors**: double-check `.streamlit/secrets.toml` and restart Streamlit.
- **Tool skipping the graph**: watch the `cardapio` logs in the terminal; they show the Cypher queries and results.

Happy frying! :)
