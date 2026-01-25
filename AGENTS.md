# AGENTS.md

## Project overview
- Streamlit salesperson chatbot.
- Backend: FalkorDB graph + LangChain + LangGraph tools.
- UI: `chatbot.py` sidebar shows cart + customer profile.

## Goals for agent changes
- Keep the customer-facing interface in Brazilian Portuguese.
- Prompts, tool descriptions, and internal LLM instructions should be in English.
- Avoid changing business logic unless explicitly requested.

## File map
- `chatbot.py`: Streamlit UI and sidebar labels.
- `agent.py`: tools and LangGraph workflow.
- `prompts.py`: system prompts, tool instructions, and customer-facing copy.
- `diagnostics.py`: session health snapshot for support/debugging.
- `cart.py`: cart operations, LLM extraction prompts.
- `cypher.py`: LLM prompts for Cypher and answers.
- `create_kg_pastel.py`: seed data + console output.
- `graph.py`: FalkorDB connection + schema snapshot.
- `customer_profile.py`, `session_manager.py`: session and profile state.

## Coding standards
- Prefer `rg` for searches.
- Keep edits minimal and localized.
- Use ASCII by default; preserve existing accents.
- Add comments only if logic is complex.

## Testing / verification
- If modifying prompts/tools: run a quick smoke check via Streamlit if possible.

## Safety
- Do not delete data or reset graphs unless asked.
- Never run destructive git commands without explicit request.
