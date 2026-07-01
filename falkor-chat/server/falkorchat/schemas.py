"""Request models for the REST surface (FastAPI validation boundary).

Responses are plain dicts straight from the service layer — one shape, both
front doors. Only request bodies need declared models.
"""

from __future__ import annotations

from pydantic import BaseModel


class CreateChannelIn(BaseModel):
    name: str


class CreateThreadIn(BaseModel):
    title: str


class PostMessageIn(BaseModel):
    text: str
    mentions: list[str] | None = None  # REST mention parity with the MCP tool
