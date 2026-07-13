"""Request models for the REST surface (FastAPI validation boundary).

Responses are plain dicts straight from the service layer — one shape, both
front doors. Only request bodies need declared models.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

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
