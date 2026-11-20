"""Requirements documents: the header block only, no children (graph note section 2.1)."""
from __future__ import annotations

from .source import SourceDoc, base_fields


def parse_requirements(path: str, raw: bytes) -> SourceDoc:
    _, base = base_fields(path, "requirement", raw)
    return SourceDoc(**base)
