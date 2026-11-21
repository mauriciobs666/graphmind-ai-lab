"""The one configuration module (plan section 3.1).

Every value is env-overridable so a test never touches shared state; `Config.from_env`
takes the environment mapping explicitly for the same reason.
"""
from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

# The per-section KB lease TTL (plan section 3.5). Every falkor-chat HTTP call made
# under a renewed lease must time out well inside it: asserted < TTL/3 at start-up.
LEASE_TTL_S = 60

# `COORDGRAPH_HOOK_DISABLED` disables the hook only for one of these (case-insensitive,
# surrounding whitespace ignored); `false`, `0`, `no`, `off`, empty and anything else leave it on.
HOOK_DISABLED_VALUES = frozenset({"1", "true", "yes", "on"})

DEFAULT_KB_URL = "http://localhost:8200"
DEFAULT_KB_GRAPH = "ws:agent-team"
# `start_agent_team.sh` supplies no embedder URL (the model gateway reads it from the
# OpenCode config), so the fallback is LM Studio's standard local endpoint.
DEFAULT_EMBED_URL = "http://localhost:1234/v1"


class ConfigError(ValueError):
    """A configuration value is unusable; raised at start-up, never mid-run."""


def package_repo_root() -> Path:
    """The repo containing this package: `<repo>/coordination-graph/coordgraph/config.py`."""
    return Path(__file__).resolve().parents[2]


def _float(env: Mapping[str, str], name: str, default: float) -> float:
    raw = env.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        raise ConfigError(f"{name}={raw!r} is not a number") from None


def _int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = env.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        raise ConfigError(f"{name}={raw!r} is not an integer") from None


def _text(env: Mapping[str, str], name: str, default: str) -> str:
    raw = env.get(name)
    return default if raw is None or raw.strip() == "" else raw.strip()


@dataclass(frozen=True)
class Config:
    repo_root: Path
    falkordb_host: str
    falkordb_port: int
    key: str
    state_dir: Path
    kb_url: str
    kb_graph: str
    embed_url: str
    kb_budget_s: float
    kb_http_timeout_s: float
    hook_disabled: bool
    kb_retry_max: int = 3  # COORDGRAPH_KB_RETRY_MAX: failed-document re-submissions per section
    kb_retry_backoff_s: float = 1800.0  # COORDGRAPH_KB_RETRY_BACKOFF_S: first wait; doubles per attempt
    kb_retry_run_cap: int = 4  # COORDGRAPH_KB_RETRY_RUN_CAP: sections one run may re-submit
    shared_confirm: str | None = None  # COORDGRAPH_SHARED_CONFIRM: the standing (env) shared-instance confirmation

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Config:
        env = os.environ if env is None else env
        root_raw = env.get("COORDGRAPH_REPO_ROOT") or env.get("CLAUDE_PROJECT_DIR")
        repo_root = Path(root_raw) if root_raw else package_repo_root()
        state_raw = env.get("COORDGRAPH_STATE_DIR")
        state_dir = Path(state_raw) if state_raw else repo_root / "coordination-graph" / ".state"
        embed = env.get("COORDGRAPH_EMBED_URL") or env.get("FALKORCHAT_EMBEDDING_BASE_URL")
        timeout = _float(env, "COORDGRAPH_KB_HTTP_TIMEOUT_S", 10.0)
        if not 0 < timeout < LEASE_TTL_S / 3:
            raise ConfigError(
                f"COORDGRAPH_KB_HTTP_TIMEOUT_S={timeout} must be > 0 and < lease TTL/3 "
                f"({LEASE_TTL_S / 3:g} s): a renewed lease must not expire inside a call"
            )
        disabled = env.get("COORDGRAPH_HOOK_DISABLED", "").strip().casefold() in HOOK_DISABLED_VALUES
        return cls(
            repo_root=repo_root,
            falkordb_host=_text(env, "COORDGRAPH_FALKORDB_HOST", "localhost"),
            falkordb_port=_int(env, "COORDGRAPH_FALKORDB_PORT", 6379),
            key=_text(env, "COORDGRAPH_KEY", "coordination"),
            state_dir=state_dir,
            kb_url=_text(env, "COORDGRAPH_KB_URL", DEFAULT_KB_URL),
            kb_graph=_text(env, "COORDGRAPH_KB_GRAPH", DEFAULT_KB_GRAPH),
            embed_url=(embed.strip() if embed and embed.strip() else DEFAULT_EMBED_URL),
            kb_budget_s=_float(env, "COORDGRAPH_KB_BUDGET_S", 10.0),
            kb_http_timeout_s=timeout,
            hook_disabled=disabled,
            kb_retry_max=_int(env, "COORDGRAPH_KB_RETRY_MAX", 3),
            kb_retry_backoff_s=_float(env, "COORDGRAPH_KB_RETRY_BACKOFF_S", 1800.0),
            kb_retry_run_cap=_int(env, "COORDGRAPH_KB_RETRY_RUN_CAP", 4),
            shared_confirm=(env.get("COORDGRAPH_SHARED_CONFIRM") or "").strip() or None,
        )
