"""Backlog scope from `.work-items/project.json` `legacy_paths` (plan3 section 3.3).

`legacy_paths` maps a path pattern to a legacy home (`*` is one path segment, captured as
`$1`). The same semantics as `wilib.check._home_of`, re-implemented here so the sync path
stays free of `wilib`; a test pins the two together. Stdlib only.
"""
from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path


def load_legacy_paths(repo_root: str | Path) -> dict[str, str]:
    """The `legacy_paths` mapping of the repo's work-item project file; `{}` when there is none."""
    path = Path(repo_root) / ".work-items" / "project.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    mapping = data.get("legacy_paths") if isinstance(data, dict) else None
    return {str(k): str(v) for k, v in mapping.items()} if isinstance(mapping, dict) else {}


def scope_of(path: str, legacy_paths: Mapping[str, str]) -> str | None:
    """The scope (legacy home) a backlog path belongs to, or None when no pattern matches."""
    for pattern, home in legacy_paths.items():
        rx = "".join("([^/]+)" if c == "*" else re.escape(c) for c in pattern)
        m = re.fullmatch(rx, path)
        if m:
            return home.replace("$1", m.group(1)) if m.groups() else home
    return None
