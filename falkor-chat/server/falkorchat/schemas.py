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


class CreateChannelIn(BaseModel):
    name: str = Field(min_length=1, max_length=MAX_NAME_LEN)


class CreateThreadIn(BaseModel):
    title: str = Field(min_length=1, max_length=MAX_NAME_LEN)


class PostMessageIn(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_TEXT_LEN)
    # REST mention parity with the MCP tool
    mentions: list[str] | None = Field(None, max_length=MAX_MENTIONS)
