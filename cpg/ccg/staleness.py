"""Staleness: the closed ``Unit.failReason`` vocabulary (``-p1-graph.md`` §1 G4).

Created by P1-18 with the constant only; P1-20 adds ``unit_staleness(unit, head)`` to this file, the one staleness module.
A ``failReason`` is **value-free**: one code of :data:`FAIL_REASONS`, never exception text, source text or a configuration value.
A new code is an edit of this constant and of the G4 row.
"""
from __future__ import annotations

FAIL_REASONS: tuple[str, ...] = (
    "manifest:never-built",
    "build:shallow-clone",
    "extract:failed",
    "extract:timeout",
    "bundle:invalid",
    "bundle:schema-minor",
    "bundle:bind-key-check",
    "bundle:audit-missing",
)
