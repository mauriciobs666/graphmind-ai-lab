"""``:Snapshot`` identity and the ``:Pack`` set (``-p1-graph.md`` §3).

Pure functions over bytes and records: no clock, no database. The identity is
``manifestHash`` = SHA-256 of the canonical JSON of ``{graphModelVersion, packSetHash, codeHash, units}``; ``snapshotId`` is
``tuple_id(["snapshot", systemId, manifestHash])``. ``bundleHash`` is a ``manifestHash`` input only and is never stored (it covers
``bindParts``/``bindKeyCheck``, which never become graph properties). ``auditSetHash`` is provenance outside the identity.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from ccg import ids
from ccg.bundle import canonical_line

GRAPH_MODEL_VERSION = "1.1"
#: ``ids.tuple_id(["verdictLog"])``, the stamp of an empty verdict log (P1 stores offset 0 and this hash; answer spec §4.2).
EMPTY_VERDICT_LOG_HASH = ids.tuple_id(["verdictLog"])
VERDICT_LOG_OFFSET = 0

#: Header fields exempt from determinism (``-bundle-spec3.md`` §8.5): the only two that may differ between builds of one unit.
BUNDLE_HASH_EXEMPT = ("builtAt", "sandboxAuditIds")

#: The source of the modules that turn bundles and packs into rows (``-p1-graph.md`` §3), as globs relative to the ``cpg/`` root, plus
#: ``reasons.yaml``. ``ccg/engine``, ``ccg/front`` and the store are not in it; ``flow-availability.yaml`` is not either (its effect enters
#: through the hashed ``flowAvailable`` values). A test runs an assembly in a fresh ``-I`` interpreter and fails if any ``ccg.*`` module
#: then loaded is missing here, so the list cannot fall behind the import graph.
ASSEMBLY_CODE: tuple[str, ...] = (
    "ccg/__init__.py",
    "ccg/assemble/**/*.py",
    "ccg/linker/**/*.py",
    "ccg/packrt/**/*.py",
    "ccg/ids.py",
    "ccg/keynorm.py",
    "ccg/rulepack.py",
    "ccg/bundle.py",
    "ccg/valuehash.py",
    "ccg/sysname.py",
    "ccg/sandbox_audit.py",
    "ccg/staleness.py",
    "ccg/units.py",
    "ccg/gitsafe.py",
    "ccg/indexed.py",
    "ccg/store/__init__.py",
    "ccg/store/ddl_v1.cypher",
    "ccg/globs.py",
    "ccg/manifest.py",
    "ccg/atrest.py",
    "ccg/registry/reasons.yaml",
)

CPG_ROOT = Path(__file__).resolve().parents[2]


class AssembleError(ValueError):
    """A contract violation in the assembler's inputs. Messages hold ids and group names, never a record value."""


class SnapshotError(AssembleError):
    """An identity input that cannot be hashed (a missing listed file, a symlink in a pack directory)."""


def assembly_code_files(root: Path | None = None) -> list[str]:
    """The files :data:`ASSEMBLY_CODE` names under ``root`` (default: this checkout's ``cpg/``), relative POSIX paths, sorted.

    A pattern that matches nothing is an error: a code list that silently shrank would stop covering a module."""
    root = CPG_ROOT if root is None else Path(root)
    found: set[str] = set()
    for pattern in ASSEMBLY_CODE:
        hits = [p for p in root.glob(pattern) if p.is_file() and "__pycache__" not in p.parts]
        if not hits:
            raise SnapshotError(f"ASSEMBLY_CODE entry {pattern!r} matches no file under the code root")
        found |= {p.relative_to(root).as_posix() for p in hits}
    return sorted(found)


def _files_hash(base: Path, rels: Iterable[str]) -> str:
    h = hashlib.sha256()
    for rel in sorted(rels):
        h.update(rel.encode("utf-8") + b"\0" + (base / rel).read_bytes() + b"\0")
    return h.hexdigest()


def code_hash(root: Path | None = None) -> str:
    """SHA-256 hex over :func:`assembly_code_files`, each contributing ``path + "\\0" + bytes + "\\0"``."""
    root = CPG_ROOT if root is None else Path(root)
    return _files_hash(root, assembly_code_files(root))


def pack_content_hash(directory: Path) -> str:
    """SHA-256 hex over a pack directory's files by the :func:`code_hash` byte rule (relative POSIX path order); a symlink is refused."""
    directory = Path(directory)
    if not directory.is_dir():
        raise SnapshotError("a pack directory is not a directory")
    rels = []
    for p in directory.rglob("*"):
        if p.is_symlink():
            raise SnapshotError("a pack directory holds a symbolic link")
        if p.is_file():
            rels.append(p.relative_to(directory).as_posix())
    return _files_hash(directory, rels)


def pack_triples(packs: Sequence[Mapping], pack_dirs: Mapping[str, Path]) -> list[tuple[str, str, str]]:
    """``(packId, version, contentHash)`` of every pack, sorted by ``packId``."""
    out = []
    for p in packs:
        if p["id"] not in pack_dirs:
            raise SnapshotError("a pack has no directory to hash")
        out.append((p["id"], str(p["version"]), pack_content_hash(pack_dirs[p["id"]])))
    if len({t[0] for t in out}) != len(out):
        raise SnapshotError("a pack id is given twice")
    return sorted(out)


def pack_set_hash(triples: Sequence[tuple[str, str, str]]) -> str:
    """O-10: ``tuple_id(["packSet", id1, version1, contentHash1, id2, ...])``, packs sorted by ``packId``."""
    flat: list[str] = ["packSet"]
    for t in sorted(triples):
        flat += list(t)
    return ids.tuple_id(flat)


def bundle_hash(records: Sequence[Mapping]) -> str:
    """SHA-256 hex over the bundle's canonical lines, the header first with the §8.5 exempt fields removed."""
    if not records or records[0].get("type") != "header":
        raise SnapshotError("a bundle starts with its header")
    h = hashlib.sha256()
    for i, rec in enumerate(records):
        if i == 0:
            rec = {k: v for k, v in rec.items() if k not in BUNDLE_HASH_EXEMPT}
        h.update(canonical_line(dict(rec)))
    return h.hexdigest()


def audit_set_hash(audit_ids: Iterable[str]) -> str:
    """SHA-256 hex of the sorted, de-duplicated ids joined by ``\\n``."""
    return hashlib.sha256("\n".join(sorted(set(audit_ids))).encode("utf-8")).hexdigest()


def manifest_hash(pack_set: str, code: str, unit_rows: Iterable[Sequence]) -> str:
    """``unit_rows`` are ``[unitId, buildStatus, bundleHash|"", flowAvailable, failReason|""]``, one per ``Unit`` node."""
    body = {"graphModelVersion": GRAPH_MODEL_VERSION, "packSetHash": pack_set, "codeHash": code,
            "units": sorted([list(r) for r in unit_rows], key=lambda r: r[0])}
    return hashlib.sha256(json.dumps(body, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")).hexdigest()


def snapshot_id(system: str, manifest: str) -> str:
    return ids.tuple_id(["snapshot", system, manifest])
