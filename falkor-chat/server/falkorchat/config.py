"""Configuration and the single auth/tenancy seam (DESIGN §14.3).

The hardcoded M1 scope lives here and is injected at exactly one place
(`api.get_context`). When real auth lands (token -> user + workspace claim,
or the `identity` graph as source of truth) only `get_context` changes —
services and the repository already take `ws`/`actor` as parameters.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

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
#
# NOT touched by K-042/FR-20: this is DDL-time/write-path input (§4.5), a different
# thing from a model's own declared output width (which the overlay's per-model
# `dim` now carries, authoritative when present — this stays the fallback).
EMBEDDING_DIM: int = int(os.environ.get("FALKORCHAT_EMBEDDING_DIM", "1536"))

# ── LLM/embedding provider & model config (K-042, FR-1/FR-2/FR-11/FR-20) ───────
# Model choice is no longer an env var (FR-20) — see `modelconfig.ModelGateway`.
# Two files: the pristine shared OpenCode file (no product default — a home-dir
# default is the "works on my box" failure mode; `scripts/start_server.sh` supplies
# the dev convenience default) and falkor-chat's own overlay (in-repo default).
OPENCODE_CONFIG_PATH: str | None = os.environ.get("FALKORCHAT_OPENCODE_CONFIG")
_DEFAULT_MODEL_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "models.json"
MODEL_CONFIG_PATH: str = os.environ.get(
    "FALKORCHAT_MODEL_CONFIG", str(_DEFAULT_MODEL_CONFIG_PATH)
)

# AC-13 tripwire: the four env vars K-042 replaced. Kept as a tuple (not inlined into
# the function below) so a test can assert against the exact list.
LEGACY_MODEL_ENV_VARS: tuple[str, ...] = (
    "FALKORCHAT_LLM_BASE_URL", "FALKORCHAT_LLM_MODEL",
    "FALKORCHAT_EMBEDDING_BASE_URL", "FALKORCHAT_EMBEDDING_MODEL",
)


def assert_no_legacy_model_env() -> None:
    """FR-20/AC-13: refuse to start if a legacy model-config env var is set.

    Called from `ModelGateway.from_env()` — i.e. only when an LLM consumer is
    actually being wired (§4.1's "required only when wired" rule), so a library
    import with no consumer enabled never trips this. Names every legacy var that
    is set (not just the first) and points at the two replacement files.
    """
    present = [name for name in LEGACY_MODEL_ENV_VARS if name in os.environ]
    if present:
        raise RuntimeError(
            "legacy model configuration env var(s) " + ", ".join(present) +
            " are set, but K-042 replaced them with two config files: "
            "FALKORCHAT_OPENCODE_CONFIG (providers, no product default) and "
            f"FALKORCHAT_MODEL_CONFIG (falkor-chat overlay, default "
            f"{MODEL_CONFIG_PATH}). Unset the legacy var(s) and configure models "
            "via those two files instead — see config/opencode.example.json and "
            "config/models.json."
        )

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

# ── M3 workflow engine (K-022/K-023) ───────────────────────────────────────────
# Whether `falkorchat.app:app` wires the LLM-native workflow executor + trigger. When
# on, an `@mention` of `AGENT_ID` starts (or a plain reply resumes) a run of the
# `TRIGGER_DEF_KEY`@`TRIGGER_DEF_VERSION` def; when off (the default) the app is exactly
# the M2 wiring and the import + pytest baseline stay network-free. The trigger holds the
# responder for its no-workflow fall-through, so exactly one handler fires per message.
WORKFLOW_ENABLED: bool = _env_flag("FALKORCHAT_WORKFLOW_ENABLED", default=False)
# The materialized def an `@mention` starts — must match the seeded `key`/`version`
# (see `scripts/seed_workflows.sh`) or the trigger's @mention-to-start step never fires.
TRIGGER_DEF_KEY: str = os.environ.get("FALKORCHAT_TRIGGER_DEF_KEY", "triage")
TRIGGER_DEF_VERSION: str = os.environ.get("FALKORCHAT_TRIGGER_DEF_VERSION", "v1")


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
