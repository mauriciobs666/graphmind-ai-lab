"""The "is X mapped?" lookup and the File+line anchor read (S17 note sections 1.3, 1.4).

The store holds many repos and a path repeats across them, so every read is **repo-scoped**. The lookup is three
separate queries in this order (no ``OR``, no ``IS NULL OR`` filters; the index seeks on ``path``, ``repoId``
filters the rows). ``run(cypher, params) -> list[list]`` is any read-only executor over one graph key.
"""
from __future__ import annotations

from typing import Any, Callable

Run = Callable[[str, dict], list]

MAPPED_CYPHER = "MATCH (f:File {path: $p, repoId: $r}) RETURN f.unitId"
SKIPPED_CYPHER = "MATCH (s:SkippedFile {path: $p, repoId: $r}) RETURN s.unitId, s.reason"
#: section 1.4: every symbol of the file, because an owned region can lie outside the span
ANCHOR_CYPHER = ("MATCH (f:File {path: $p, repoId: $r})-[:DEFINES]->(s:Symbol) "
                 "RETURN s.symbolId, s.lineStart, s.lineEnd, s.ownedRegions, s.depth")


def lookup_path(run: Run, repo_id: str, path: str) -> dict[str, Any]:
    """``{"state": "mapped", "unitId"}`` | ``{"state": "not mapped", "unitId", "reason"}`` | ``{"state": "not found"}``.

    A declaration file, a vendored path or anything outside the units is in neither set: "not found"."""
    params = {"p": path, "r": repo_id}
    if rows := run(MAPPED_CYPHER, params):
        return {"state": "mapped", "unitId": rows[0][0]}
    if rows := run(SKIPPED_CYPHER, params):
        return {"state": "not mapped", "unitId": rows[0][0], "reason": rows[0][1]}
    return {"state": "not found"}


def file_symbols(run: Run, repo_id: str, path: str) -> list[list]:
    """Every symbol of ``path`` in ``repo_id``: ``[symbolId, lineStart, lineEnd, ownedRegions, depth]`` rows."""
    return run(ANCHOR_CYPHER, {"p": path, "r": repo_id})
