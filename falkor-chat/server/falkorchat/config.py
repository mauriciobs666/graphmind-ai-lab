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
# Client socket timeouts in seconds (DEF-2). An unreachable instance must fail
# fast and loud — on WSL2 a dead port can blackhole (no RST) instead of
# refusing, so without a connect timeout startup hangs for minutes with zero
# output. `SOCKET_TIMEOUT` bounds each command round-trip client-side; long
# GraphRAG reads that need more pass a per-query `timeout=` override instead
# (DESIGN §10 posture).
FALKORDB_CONNECT_TIMEOUT: float = float(os.environ.get("FALKORDB_CONNECT_TIMEOUT", "5"))
FALKORDB_SOCKET_TIMEOUT: float = float(os.environ.get("FALKORDB_SOCKET_TIMEOUT", "10"))

# ── GraphRAG / embeddings (K-008, DESIGN §1.3) ─────────────────────────────────
# The embedding dimension is FIXED at vector-index creation and must match the
# workspace's `Message.embedding`/`Chunk.embedding` indexes. Default tracks the
# model-neutral bootstrap default (1536); real M2 GraphRAG workspaces are created
# at 1024 (Qwen3-Embedding-0.6B) — set FALKORCHAT_EMBEDDING_DIM to match, because
# a wrong-dim vecf32 write is silently accepted and then drops out of ANN.
EMBEDDING_DIM: int = int(os.environ.get("FALKORCHAT_EMBEDDING_DIM", "1536"))
# LM Studio, OpenAI-compatible /v1/embeddings (the embedding worker's default
# backend). Kept off the message write path — computed out-of-band (DESIGN §9).
EMBEDDING_BASE_URL: str = os.environ.get("FALKORCHAT_EMBEDDING_BASE_URL", "http://localhost:1234/v1")
EMBEDDING_MODEL: str = os.environ.get(
    "FALKORCHAT_EMBEDDING_MODEL", "text-embedding-qwen3-embedding-0.6b"
)
# LM Studio chat model (K-013 AI responder). OpenAI-compatible /v1/chat/completions,
# same backend as the embedder. Default is the non-thinking Qwen3-4B (NOT the
# `-thinking-` variant). Kept off the guarded write path — the responder calls the
# LLM before posting, so latency/failure never corrupts the thread (failure
# isolation by ordering).
LLM_BASE_URL: str = os.environ.get("FALKORCHAT_LLM_BASE_URL", "http://localhost:1234/v1")
LLM_MODEL: str = os.environ.get("FALKORCHAT_LLM_MODEL", "qwen/qwen3-4b-2507")

# ── AI agent participant (K-013/K-014, DESIGN §M2) ─────────────────────────────
# The workspace Agent the responder posts as. `AGENT_ID` must match the `agentId`
# registered in the workspace (see `scripts/seed_demo.sh`) and is wired into
# `AgentResponder(agent_id=…)`; `@mention`-ing it in a message triggers a reply.
AGENT_ID: str = os.environ.get("FALKORCHAT_AGENT_ID", "assistant")
AGENT_NAME: str = os.environ.get("FALKORCHAT_AGENT_NAME", "Assistant")


def _env_flag(name: str, default: bool = False) -> bool:
    """Parse a boolean env var. Truthy: 1/true/yes/on (case-insensitive)."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# Whether `falkorchat.app:app` wires the live LM-Studio-backed embedder + LLM +
# responder. **Off by default** so importing the module (and the pytest baseline)
# stays network-free — the served app turns it on via `FALKORCHAT_ENABLE_AGENT=1`
# (see `scripts/start_server.sh`). Constructing the clients is itself offline; the
# network is only touched when a posted message schedules the background tasks.
ENABLE_AGENT: bool = _env_flag("FALKORCHAT_ENABLE_AGENT", default=False)


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
