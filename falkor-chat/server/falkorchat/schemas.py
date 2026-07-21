"""Request models for the REST surface (FastAPI validation boundary).

Responses are plain dicts straight from the service layer — one shape, both
front doors. Only request bodies need declared models.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field, field_validator

# Size bounds: RAM is the binding constraint (AGENTS.md rule 6) — message text
# lands in graph memory AND the full-text index, so the boundary caps it.
MAX_NAME_LEN = 200
MAX_TEXT_LEN = 8000
MAX_MENTIONS = 50
# Server-minted ids (runId, msgId, …) are hex uuids; bound path params defensively so a
# pathological id can't reach a query (rule 6 — the boundary caps every string input).
MAX_ID_LEN = 200


class CreateChannelIn(BaseModel):
    name: str = Field(min_length=1, max_length=MAX_NAME_LEN)


class CreateThreadIn(BaseModel):
    title: str = Field(min_length=1, max_length=MAX_NAME_LEN)


class PostMessageIn(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_TEXT_LEN)
    # REST mention parity with the MCP tool
    mentions: list[str] | None = Field(None, max_length=MAX_MENTIONS)


# ── §11 Workflow definition publish (M3 Slice 1) ────────────────────────────────
# Size bounds are the RAM guard (rule 6): a def is a handful of steps/transitions
# in the global `reference` graph; `config`/`guard` are opaque strings the caller
# pre-serializes (the service stores them verbatim, never queries inside).
MAX_KEY_LEN = 200
MAX_STEPS = 200
MAX_TRANSITIONS = 500
MAX_CONFIG_LEN = 8000


class WorkflowStepIn(BaseModel):
    key: str = Field(min_length=1, max_length=MAX_KEY_LEN)
    type: str = Field(min_length=1, max_length=50)
    config: str | None = Field(None, max_length=MAX_CONFIG_LEN)
    # Exactly one step marks itself the start; the service validates + derives it.
    start: bool = False


class WorkflowTransitionIn(BaseModel):
    # `from` is a Python keyword — accept it on the wire via alias.
    from_: str = Field(alias="from", min_length=1, max_length=MAX_KEY_LEN)
    to: str = Field(min_length=1, max_length=MAX_KEY_LEN)
    on: str = Field(min_length=1, max_length=MAX_KEY_LEN)
    guard: str | None = Field(None, max_length=MAX_CONFIG_LEN)
    order: int = 0


class PublishWorkflowDefIn(BaseModel):
    key: str = Field(min_length=1, max_length=MAX_KEY_LEN)
    version: str = Field(min_length=1, max_length=MAX_KEY_LEN)
    name: str = Field(min_length=1, max_length=MAX_NAME_LEN)
    kind: str = Field(min_length=1, max_length=50)
    steps: list[WorkflowStepIn] = Field(min_length=1, max_length=MAX_STEPS)
    transitions: list[WorkflowTransitionIn] = Field(
        default_factory=list, max_length=MAX_TRANSITIONS
    )


# ── §12 Workflow-run start + human/signal input (K-024 U3, D-B/D-F/D-H) ─────────
# **Which layer owns which bound (plan m-5).** Pydantic bounds only what it can
# see: the *submitted* dict (key count, key length, serialized size) — it never
# sees the stored run `ctx` this merges into, nor does MCP / a direct service
# caller ever reach it. So the **merged** ctx bound, the reserved-key rule (M-2)
# and the parked-step declaration check (D-H) all live in `services.py`; this
# model is the convenience bound at the HTTP door, not the contract.
MAX_RUN_CTX_KEYS = 32
# The run-level step budget a caller may declare (D-H part c: `access-request@v1`
# passes 24; omitting it falls back to the executor's global default of 12).
MAX_RUN_STEPS = 50


def _bounded_flat_dict(value: dict[str, Any] | None) -> dict[str, Any] | None:
    """Shared bound for a caller-supplied ctx/input dict (rule 6)."""
    if value is None:
        return None
    if len(value) > MAX_RUN_CTX_KEYS:
        raise ValueError(f"at most {MAX_RUN_CTX_KEYS} keys allowed")
    for key in value:
        if not key or len(key) > MAX_KEY_LEN:
            raise ValueError(f"key must be 1..{MAX_KEY_LEN} characters")
    if len(json.dumps(value, separators=(",", ":"), sort_keys=True)) > MAX_CONFIG_LEN:
        raise ValueError(f"serialized payload exceeds {MAX_CONFIG_LEN} characters")
    return value


class StartWorkflowRunIn(BaseModel):
    """`POST /workflow-runs` — start a run from a snapshot with no chat trigger."""

    defKey: str = Field(min_length=1, max_length=MAX_KEY_LEN)
    version: str = Field(min_length=1, max_length=MAX_KEY_LEN)
    # The run's initial state. Reserved keys (`threadId`, `error`) are rejected by
    # the SERVICE (M-2/F-6) — not here — because MCP and direct callers bypass this.
    ctx: dict[str, Any] | None = None
    trace: bool = False
    maxSteps: int | None = Field(None, ge=1, le=MAX_RUN_STEPS)

    @field_validator("ctx")
    @classmethod
    def _check_ctx(cls, v):
        return _bounded_flat_dict(v)


class SubmitWorkflowInputIn(BaseModel):
    """`POST /workflow-runs/{runId}/input` — human/signal input for a parked run."""

    input: dict[str, Any] = Field(default_factory=dict)

    @field_validator("input")
    @classmethod
    def _check_input(cls, v):
        return _bounded_flat_dict(v)
