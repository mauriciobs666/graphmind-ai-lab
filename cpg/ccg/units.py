"""Unit discovery over a system manifest's repositories (C-1001 P1-03).

Plan: ``docs/plans/connected-code-graph-p1-extraction.md`` § P1-03. ``Unit = (unitId, repoId, rootPath, language)``;
``unitId`` is ``ccg.ids.unit_id``, ``rootPath`` is repo-relative POSIX with no trailing ``/`` (the repo root is ``.``)
and ``language`` is a bundle spec3 §3.3 token.

Code units, in two kinds:

- **manifest units**: a directory holding ``pyproject.toml`` or ``requirements.txt`` (python) or ``package.json`` or
  ``tsconfig.json`` (javascript, TypeScript included). Nested manifests give nested units; a file belongs to its
  nearest enclosing unit of its language.
- **fallback units**: a ``.py``/``.pyi`` or JS/TS file under no manifest unit of its language is rooted at its
  highest ancestor directory that holds no manifest unit of that language below it (see ``_fallback_root``; ``.`` for
  a file at the repo root), one unit per (root, language).

Only python and javascript have code units in P1; other languages (Java, Go, ...) are P2's. With ``per_file`` (the
default) each shell, config and Markdown file is also a unit of its own, ``rootPath`` being the file, for the non-Joern
extractors.

Files come from ``git ls-files`` when the repository root is a non-empty git work tree (so untracked scratch files
never change the units and the golden is the committed state), else from a directory walk. Exclusions are the repo's
``exclusions.paths`` and ``exclusions.contentCheck.files``, relative to the repo root; vendored and cache directories
are skipped either way.
"""
from __future__ import annotations

import os
import posixpath
import re
import subprocess
from pathlib import Path
from typing import Iterable, NamedTuple

from ccg import ids
from ccg.manifest import Manifest, Repo

#: ``export_kit.py``'s SKIP_DIRS plus caches and virtualenvs; applied to every directory segment.
SKIP_DIRS = frozenset({
    "node_modules", "vendor", "vendored", "_vendor", "third_party", "third-party",
    ".git", "__pycache__", ".venv", "venv", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox",
})
MANIFEST_FILES = {
    "pyproject.toml": "python", "requirements.txt": "python", "package.json": "javascript",
    "tsconfig.json": "javascript",
}
_SOURCE_EXT = {".py": "python", ".pyi": "python"} | {
    e: "javascript" for e in (".ts", ".tsx", ".mts", ".cts", ".js", ".jsx", ".mjs", ".cjs")
}
_CONFIG_EXT = {".json", ".yaml", ".yml", ".toml"}
_SHELLS = frozenset({"sh", "bash", "zsh"})


class Unit(NamedTuple):
    unitId: str
    repoId: str
    rootPath: str
    language: str


# --- file listing and exclusion --------------------------------------------------------------------------------


_MCP_JSON = re.compile(r"\.mcp(?:\..+)?\.json")


def is_mcp_json(path: str) -> bool:
    """True iff ``path`` is an MCP server configuration file: a basename ``.mcp.json``, ``.mcp.<x>.json`` (``.mcp.example.json``,
    ``.mcp.local.json``), outside a vendored ``node_modules`` directory. The one predicate of the config extractor (its ``mcp-json``
    format) and of the linker's binder (a server's entry is read only from such a file); ``.mcp.json.example`` is not one."""
    *dirs, base = path.split("/")
    return _MCP_JSON.fullmatch(base) is not None and "node_modules" not in dirs


def _skipped_dir(name: str) -> bool:
    return name in SKIP_DIRS or name.endswith(".egg-info")


def _git_files(root: Path) -> list[str] | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z"], capture_output=True, check=True, timeout=60
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    files = [f for f in out.decode("utf-8", "surrogateescape").split("\0") if f]
    return files or None


def _walk_files(root: Path) -> list[str]:
    out = []
    for dirpath, dirs, names in os.walk(root):
        dirs[:] = sorted(d for d in dirs if not _skipped_dir(d))
        rel = os.path.relpath(dirpath, root)
        out.extend(posixpath.join("" if rel == "." else Path(rel).as_posix(), n) for n in sorted(names))
    return out


def _list_files(root: Path) -> list[str]:
    files = _git_files(root)
    if files is None:
        files = _walk_files(root)
    return sorted(f for f in files if not any(_skipped_dir(seg) for seg in f.split("/")[:-1]))


def _glob_re(pattern: str) -> re.Pattern[str]:
    """``**/`` is any run of directories (or none), ``**`` anything, ``*`` and ``?`` stay inside one segment."""
    out, i = [], 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif pattern[i] == "*":
            out.append("[^/]*")
            i += 1
        elif pattern[i] == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    return re.compile("".join(out))


def _excluder(repo: Repo):
    """Every entry of ``exclusions.paths`` and ``contentCheck.files`` excludes the path it names and everything below it.

    A plain entry (``cpg/``, ``dep``, ``docs/BACKLOG.md``) is a path prefix on whole segments; a glob entry is matched
    against the path and against every leading run of its segments, so ``docs/**/connected-code-graph*`` also excludes
    files below a matching directory. This is the census matcher's (``export_kit.path_excluded``) behaviour.
    """
    plain = {p.rstrip("/") for p in repo.exclusion_paths if not any(c in p for c in "*?")} | set(repo.content_check_files)
    globs = [_glob_re(p.rstrip("/")) for p in repo.exclusion_paths if any(c in p for c in "*?")]

    def excluded(rel: str) -> bool:
        segs = rel.split("/")
        prefixes = ["/".join(segs[:i]) for i in range(1, len(segs) + 1)]
        return any(pre in plain or any(g.fullmatch(pre) for g in globs) for pre in prefixes)

    return excluded


# --- classification ---------------------------------------------------------------------------------------------


def _file_language(root: Path, rel: str) -> str | None:
    """The per-file token (shell, config, markdown), or None."""
    name = posixpath.basename(rel)
    ext = posixpath.splitext(name)[1].lower()
    if ext in (".sh", ".bash"):
        return "shell"
    if ext == ".md":
        return "markdown"
    if ext in _CONFIG_EXT or name == ".env" or name.startswith((".env.", "Dockerfile")):
        return "config"
    if not ext:
        try:
            with open(root / rel, "rb") as fh:
                head = fh.readline(256)
        except OSError:
            return None
        if _shebang_is_shell(head):
            return "shell"
    return None


def _shebang_is_shell(first_line: bytes) -> bool:
    if not first_line.startswith(b"#!"):
        return False
    argv = first_line[2:].decode("utf-8", "replace").split()
    if argv and posixpath.basename(argv[0]) == "env":
        argv = [a for a in argv[1:] if not a.startswith("-")]
    return bool(argv) and posixpath.basename(argv[0]) in _SHELLS


def _under(rel_dir: str, path: str) -> bool:
    return rel_dir == "." or path.startswith(rel_dir + "/")


def _fallback_root(path: str, lang: str, manifest_roots: set[tuple[str, str]]) -> str:
    """The highest ancestor directory of an unowned file (below the repo root) with no manifest unit of ``lang`` under it.

    Fallback units therefore never enclose a manifest unit's files. The one exception is forced: a file directly in
    a directory that holds a nested manifest unit (a root-level file, say) can only root at its own directory, and the
    nearest-enclosing-unit rule gives the nested unit its files.
    """
    segs = path.split("/")[:-1]
    for i in range(1, len(segs) + 1):
        d = "/".join(segs[:i])
        if not any(l == lang and r != "." and r.startswith(d + "/") for r, l in manifest_roots):
            return d
    return "/".join(segs) or "."


def _repo_units(root: Path, repo: Repo, per_file: bool) -> list[Unit]:
    repo_root = root / repo.path
    excluded = _excluder(repo)
    files = [f for f in _list_files(repo_root) if not excluded(f)]

    manifest_roots: set[tuple[str, str]] = set()
    for f in files:
        lang = MANIFEST_FILES.get(posixpath.basename(f))
        if lang:
            manifest_roots.add((posixpath.dirname(f) or ".", lang))
    roots = set(manifest_roots)
    # Coverage is judged against manifest roots only, so the result never depends on file order.
    for f in files:
        lang = _SOURCE_EXT.get(posixpath.splitext(f)[1].lower())
        if lang and not any(l == lang and _under(d, f) for d, l in manifest_roots):
            roots.add((_fallback_root(f, lang, manifest_roots), lang))
    found = {(d, lang) for d, lang in roots}
    if per_file:
        for f in files:
            lang = _file_language(repo_root, f)
            if lang:
                found.add((f, lang))
    return [Unit(ids.unit_id(repo.repo_id, d, lang), repo.repo_id, d, lang) for d, lang in found]


def discover_units(repo_root: str | Path, manifest: Manifest, per_file: bool = True) -> list[Unit]:
    """Units of every repository of ``manifest``; each repo's ``path`` is resolved against ``repo_root``.

    Sorted by ``(repoId, rootPath, language)``, so two runs over one tree are identical.
    """
    root = Path(repo_root)
    units: list[Unit] = []
    for repo in manifest.repos:
        units.extend(_repo_units(root, repo, per_file))
    return sorted(units, key=lambda u: (u.repoId, u.rootPath, u.language))


def _golden_main(argv: Iterable[str]) -> int:  # pragma: no cover - the golden's regeneration command
    import json
    import sys

    from ccg.manifest import load_manifest

    cpg = Path(__file__).resolve().parent.parent
    manifest = load_manifest(cpg / "ccg" / "system.yaml")
    units = discover_units(cpg.parent, manifest, per_file=False)
    doc = {"system": manifest.system, "units": [u._asdict() for u in units]}
    target = cpg / "tests" / "golden" / "system-units.json"
    if "--golden" in argv:
        target.write_text(json.dumps(doc, indent=1) + "\n")
        print(f"wrote {target} ({len(units)} units)")
    else:
        print(json.dumps(doc, indent=1))
    return 0


if __name__ == "__main__":  # pragma: no cover
    import sys

    raise SystemExit(_golden_main(sys.argv[1:]))
