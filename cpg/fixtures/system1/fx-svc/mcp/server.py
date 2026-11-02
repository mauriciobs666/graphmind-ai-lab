"""MCP server exposing item and order tools."""
import argparse

from fastmcp import FastMCP

from fx_svc.mcp.names import LIST_ORDERS

mcp = FastMCP("fx-svc")


@mcp.tool()
def get_item(item_id: int) -> dict:
    return {"id": item_id}


@mcp.tool(name=LIST_ORDERS)
def list_orders() -> list:
    return []


@mcp.tool()
def reindex() -> str:
    return "done"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8200)
    args = parser.parse_args()
    mcp.run(transport="streamable-http", host="127.0.0.1", port=args.port, path="/mcp")


if __name__ == "__main__":
    main()
