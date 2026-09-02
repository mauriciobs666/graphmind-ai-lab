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


def _env_csv(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    """Parse a comma-separated env var into an order-preserving tuple.

    An unset **or effectively empty** value (blank, `","`, all-whitespace) falls
    back to `default`: the one consumer is the storefront's locale enum, and an
    empty tuple there would reject every language a participant could pick,
    turning a typo in the operator's shell into a demo where nobody can join.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    parts = tuple(part.strip() for part in raw.split(",") if part.strip())
    return parts or default


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
# Whether `WorkflowTrigger` keeps the M2 responder for its step-4 fall-through
# (`trigger.py` rule 4). **On by default** — every existing deployment keeps the
# behaviour it has today. Off (`FALKORCHAT_TRIGGER_RESPONDER_FALLTHROUGH=0`) wires
# `WorkflowTrigger(responder=None)`, so a message that matches no workflow reaches
# nothing at all. The storefront demo turns it off (`salesperson-ui.md` §4.3 part 4):
# the responder's retrieval is workspace-WIDE (`services.hybrid_search` with
# `channel_id=None`), which would let one participant's question surface another
# participant's messages — so the fall-through is made structurally unreachable
# rather than merely unlikely.
TRIGGER_RESPONDER_FALLTHROUGH: bool = _env_flag(
    "FALKORCHAT_TRIGGER_RESPONDER_FALLTHROUGH", default=True
)
# K-028 — how often the in-process sweep tick runs (seconds), read only where
# `WORKFLOW_ENABLED` is already read (`app._build_default_app`). `POST
# /workflow-runs/due` (the manual/cron entry point) is unaffected by this value.
WORKFLOW_SWEEP_INTERVAL_S: float = float(
    os.environ.get("FALKORCHAT_WORKFLOW_SWEEP_INTERVAL_S", "30")
)

# ── Salesperson storefront deployment (salesperson-ui §4.9) ──────────────────
# Whether `falkorchat.app:app` is the *storefront* deployment rather than the dev
# one. **Off by default**, so `uvicorn falkorchat.app:app` and the whole existing
# test suite keep today's app shape. When on, `_build_default_app` derives BOTH
# `create_app(mount_mcp=...)` and `create_app(dev_surface=...)` from it as
# `not STOREFRONT_ENABLED`, so the unauthenticated surfaces — `api.build_router`,
# the `/` static mount and `/mcp` — are not registered at all.
#
# Note what is deliberately absent: there is **no env var for `dev_surface`**. It is
# a `create_app` parameter only (§4.9 move 1), so no operator setting can put the
# legacy surface back while storefront participants exist — the dangerous
# configuration is not expressible, rather than merely discouraged.
STOREFRONT_ENABLED: bool = _env_flag("FALKORCHAT_STOREFRONT_ENABLED", default=False)

# The **served** SPA build directory, mounted at `/shop` by `create_app` and the
# root of the product-image manifest (`<dir>/products/`, §4.7). `None` when unset
# — deliberately not `""`, which `Path("")` turns into the process working
# directory and would silently serve whatever the operator happened to `cd` into.
# The storefront deployment must set it (`scripts/start_demo.sh`, S11); it is the
# *built* output (`salesperson/dist/`), never the source tree, because the image
# manifest is built from what is actually served (§4.7).
STOREFRONT_DIR: str | None = os.environ.get("FALKORCHAT_STOREFRONT_DIR") or None

# The single operator secret for the presenter surface (§4.3, OQ-5) — typed once
# at `/shop/presenter` and exchanged for a presenter bearer token. It is demo-
# session scoping, not authentication: no accounts, no per-user credentials.
#
# **Empty means "no presenter surface", and must never authenticate.** Every
# check of it goes through `hmac.compare_digest`, and `compare_digest("", "")` is
# `True` — so a presenter login path must reject an unset key *before* comparing,
# or an unconfigured deployment hands the reset-everyone button to whoever sends
# an empty key first.
STOREFRONT_PRESENTER_KEY: str = os.environ.get("FALKORCHAT_STOREFRONT_PRESENTER_KEY", "")

# §4.4 measure 1: the size of the storefront's own bounded turn executor, sized to
# LM Studio's configured parallelism. Agent turns run there instead of on
# `BackgroundTasks`, so a deep turn queue never touches anyio's default thread
# limiter and poll reads stay instant.
STOREFRONT_TURN_WORKERS: int = int(os.environ.get("FALKORCHAT_STOREFRONT_TURN_WORKERS", "4"))

# §4.8/§7 of the graph note: how long either reset waits for in-flight turns to
# drain after intake stops, before giving up and changing nothing (`503`).
# Comfortably under the 180 s agent timeout, so a stuck turn cannot hold the
# reset past the point where the presenter gives up on it.
STOREFRONT_QUIESCE_S: float = float(os.environ.get("FALKORCHAT_STOREFRONT_QUIESCE_S", "30"))

# The languages a participant may join in (FR-3/AC-9) — the enum `POST
# /shop/api/session` validates against and the set the SPA ships bundles for
# (S12c). Comma-separated; order is the UI's offer order.
STOREFRONT_LOCALES: tuple[str, ...] = _env_csv(
    "FALKORCHAT_STOREFRONT_LOCALES", ("en", "pt-BR", "es")
)

# §4.4 measure 2: the anyio thread limiter the storefront raises **inside
# `_lifespan`, before `yield`** (`to_thread.current_default_thread_limiter()` is
# event-loop scoped and raises outside a running loop). Headroom for the poll
# path, explicitly **not** load-bearing — measure 1 is what keeps turns off this
# limiter in the first place.
THREAD_LIMIT: int = int(os.environ.get("FALKORCHAT_THREAD_LIMIT", "100"))


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
