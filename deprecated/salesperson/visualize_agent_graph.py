#!/usr/bin/env python3
"""Render the LangGraph workflow to a PNG for quick inspection."""
from pathlib import Path

from agent import agent_workflow
from langchain_core.runnables.graph_mermaid import MermaidDrawMethod


def main() -> None:
    graph = agent_workflow.get_graph()

    mermaid = graph.draw_mermaid()
    mermaid_path = Path("agent_workflow.mmd")
    mermaid_path.write_text(mermaid, encoding="utf-8")

    output = Path("agent_workflow.png")
    try:
        png_bytes = graph.draw_mermaid_png(
            draw_method=MermaidDrawMethod.API, max_retries=3, retry_delay=1.0
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Could not render PNG (wrote Mermaid instead): {exc}")
        print(f"Mermaid saved to {mermaid_path.resolve()}")
    else:
        output.write_bytes(png_bytes)
        print(f"Wrote {output.resolve()}")
        print(f"Mermaid saved to {mermaid_path.resolve()}")


if __name__ == "__main__":
    main()
