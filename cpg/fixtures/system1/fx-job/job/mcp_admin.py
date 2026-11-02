"""Admin tool server for the job."""
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("fx-job-admin")


@mcp.tool()
def reindex() -> str:
    """Rebuild the job's local report index."""
    return "ok"
