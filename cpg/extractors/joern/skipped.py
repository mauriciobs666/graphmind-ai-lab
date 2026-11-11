"""``header.skippedFiles`` for one unit (C-1001 P1-08).

Rule: ``docs/plans/connected-code-graph-bundle-spec4.md`` §1; plan: ``docs/plans/connected-code-graph-p1-extraction.md``
§ P1-08. A *source file* of the unit has no ``file`` record by design (``test``) or by fallback (``frontend-ignore``);
this module only decides which of the two, or neither. It never reads the file system and never writes a record: the
caller passes every candidate source path and whether the frontend produced a ``file`` for it, and drops the ``file``
record (and everything under it) of each path returned.

Why ``override`` exists although none is available: ``jssrc2cpg`` 4.0.579 has no switch that re-includes its default
ignores (``extractors/joern/README.md``, "jssrc2cpg ignore override"), so the runner always passes ``override=False``.
The argument states the writer's belief about the extraction it is describing; with ``True`` the frontend is asserted to
have parsed every non-test source, and one it did not is a contradiction, never a ``frontend-ignore`` entry.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Mapping

from ccg.bundle import JS_TEST_EXTENSIONS
from ccg.units import Unit
from extractors.joern.run import is_staged_source

#: the writer's default ``vendoredDirs`` (``normalize/assemble_bundle.py``; ``test_skipped_files`` keeps the two equal).
DEFAULT_VENDORED = ("_vendor", "node_modules", "third-party", "third_party", "vendor", "vendored")
#: spec3 §3.3 "Which files": the source extensions of the ``javascript`` and ``python`` tokens.
JS_EXTENSIONS = tuple("." + e for e in JS_TEST_EXTENSIONS)
PY_EXTENSIONS = (".py", ".pyi")
DECLARATION_SUFFIXES = (".d.ts", ".d.mts", ".d.cts", ".pyi")
TEST_DIRS = ("e2e", "__tests__")


def _below_root(path: str, root: str) -> str | None:
    """``path`` relative to ``root``, or ``None`` when it is not strictly below it (a string prefix is not enough)."""
    if root == ".":
        return path
    return path[len(root) + 1:] if path.startswith(root + "/") else None


def _is_source(base: str, language: str) -> bool:
    if base.endswith(DECLARATION_SUFFIXES):
        return False
    if language == "javascript":
        return base.endswith(JS_EXTENSIONS)
    if language == "python":
        return base.endswith(PY_EXTENSIONS)
    return False


def _is_test(dirs: list[str], base: str) -> bool:
    if any(d in TEST_DIRS for d in dirs):
        return True
    return any(base.endswith(f".{kind}.{ext}") for kind in ("test", "spec") for ext in JS_TEST_EXTENSIONS)


def skipped_files(unit: Unit, parsed: Mapping[str, bool], override: bool) -> list[dict]:
    """The ``header.skippedFiles`` entries of ``unit``: sorted by UTF-8 bytes, unique, each ``{path, reason}``.

    ``parsed`` maps every candidate path (repo-relative POSIX) to whether the frontend gave it a ``file``. A path not
    strictly below ``unit.rootPath``, under a vendored directory, a declaration file (``.d.ts``, ``.pyi``) or not a
    source file of the unit's language is in neither list. Only ``javascript`` units have entries in v1: a source file
    matching the closed ``test`` pattern is ``test`` whether or not it was parsed (precedence over ``frontend-ignore``);
    any other unparsed one is ``frontend-ignore``. A Python or other-language file is never listed.

    With ``override`` true an unparsed non-test source raises ``ValueError``: the override was supposed to bring it in.
    """
    if unit.language != "javascript":
        return []
    out: list[dict] = []
    contradicted: list[str] = []
    for path in sorted(parsed, key=lambda p: p.encode("utf-8")):
        rel = _below_root(path, unit.rootPath)
        if rel is None:
            continue
        *dirs, base = rel.split("/")
        if any(d in DEFAULT_VENDORED for d in dirs) or not _is_source(base, unit.language):
            continue
        if _is_test(dirs, base):
            out.append({"path": path, "reason": "test"})
        elif not parsed[path]:
            if override:
                contradicted.append(path)
            else:
                out.append({"path": path, "reason": "frontend-ignore"})
    if contradicted:
        raise ValueError("override is on but the frontend did not parse: " + ", ".join(contradicted))
    return out


def candidate_sources(stage: str | Path, unit_root: str) -> list[str]:
    """Repo-relative POSIX paths of the staged files that count as source (``run.is_staged_source``: regular and
    non-empty, the predicate of ``_count_sources``), sorted. Every extension is returned; ``skipped_files`` decides
    which are source files of the unit's language. ``stage`` is rooted at the unit (``run.stage_unit``)."""
    stage = Path(stage)
    out = []
    for dirpath, _dirs, names in os.walk(stage):
        for name in names:
            p = Path(dirpath, name)
            if is_staged_source(p):
                rel = p.relative_to(stage).as_posix()
                out.append(rel if unit_root == "." else f"{unit_root}/{rel}")
    return sorted(out)


def parsed_map(stage: str | Path, unit_root: str, parsed_paths: Iterable[str]) -> dict[str, bool]:
    """The ``parsed`` argument of ``skipped_files``: every ``candidate_sources`` path -> False, then each repo-relative
    path in ``parsed_paths`` (the files the frontend gave a ``file``) -> True. A parsed path is added even when the walk
    would not return it: ``jssrc2cpg`` emits a ``FILE`` for an empty ``.ts`` (README), and a parsed file the ``test``
    pattern matches must still be listed so its record is dropped."""
    out = dict.fromkeys(candidate_sources(stage, unit_root), False)
    for p in parsed_paths:
        out[p] = True
    return out
