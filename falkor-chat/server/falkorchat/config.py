"""Configuration and the single auth/tenancy seam (DESIGN §14.3).

The hardcoded M1 scope lives here and is injected at exactly one place
(`api.get_context`). When real auth lands (token -> user + workspace claim,
or the `identity` graph as source of truth) only `get_context` changes —
services and the repository already take `ws`/`actor` as parameters.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# ── M1 single hardcoded tenant (DESIGN §14.1) ──────────────────────────────────
WS_ID: str = os.environ.get("FALKORCHAT_WS_ID", "acme")
USER_ID: str = os.environ.get("FALKORCHAT_USER_ID", "u1")

# ── FalkorDB connection ────────────────────────────────────────────────────────
FALKORDB_HOST: str = os.environ.get("FALKORDB_HOST", "127.0.0.1")
FALKORDB_PORT: int = int(os.environ.get("FALKORDB_PORT", "6379"))


@dataclass(frozen=True)
class CallContext:
    """The resolved actor + workspace for one call.

    `ws` is the workspace id (graph key is ``ws:{ws}``); `actor` is the member
    id (a `userId` or an `agentId`) attributed as author / reader.
    """

    ws: str
    actor: str


def get_context() -> CallContext:
    """The single auth/tenancy seam (DESIGN §14.3).

    M1 resolves every call to one hardcoded tenant. Both front doors (REST and
    MCP) attribute calls through here, so when real auth lands (token -> user +
    workspace claim) only this function changes. MCP ignores any client-supplied
    `from` and attributes to this configured actor (plan Q#1).
    """
    return CallContext(ws=WS_ID, actor=USER_ID)
