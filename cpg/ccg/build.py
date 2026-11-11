"""Build dispatch by unit kind (C-1001 P1-44): one unit at one commit -> one bundle, or one value-free failure code.

Plan: ``docs/plans/connected-code-graph-p1-linking.md`` § P1-44; failure codes ``-p1-graph.md`` §1 G4 (``FAIL_REASONS``).

``build_unit`` picks the extractor by ``unit.language``: python and javascript -> the Joern run (P1-04) and normalizer
(P1-05; the P1-15 SCIP merge is not adopted, nothing here calls the indexer); shell -> P1-10; config -> P1-11;
markdown -> P1-12. Every bundle is written by ``ccg.atrest.write_bundle`` and every key is read by ``ccg.atrest.hmac_key``.

A unit that fails fails alone: ``build_unit`` returns a ``BundleResult`` carrying one code of ``FAIL_CODES`` and nothing
else about the failure (no exception text, no source bytes). An unknown kind is the exception: ``UnknownUnitKind`` is raised,
never a silent skip, since only a caller bug produces one. **P1-20 (the refresh loop) catches ``UnknownUnitKind`` per unit, records
``extract:failed`` for that unit with a separate ``unknownKind`` count in the refresh report, and never aborts the refresh**
(owner ruling; ``build_unit`` keeps raising it). A ``commit`` that is not a full 40/64-hex object id raises ``InvalidCommit``
(a caller bug as well: a ref could move between the git calls of one build). ``build_units`` builds a whole system in the order the markdown
vocabulary needs: code units first, then ``known_tools`` once (the ``fastmcp`` pack over the code bundles), then the rest.

Non-Joern units read the tree of the built commit (``git ls-tree`` / ``cat-file``, all through ``ccg.gitsafe``), never the working tree.
The repository is untrusted: every path a unit selects is validated (``_check_tree_path``) before any file is written, and the sizes
listed by ``ls-tree -l`` bound what is read (``MAX_*`` constants) before a blob is fetched. ``require_full_history`` also raises
``HistoryError`` for a git timeout, so a timed-out ``rev-parse`` is reported as ``build:shallow-clone`` (plan-conformant). Config sizes: a config blob over 512 KiB
(the extractor's parse cap) up to ``MAX_CONFIG_BLOB_BYTES`` (8 MiB) is read and gets its File record while the extractor skips parsing it;
a config blob over 8 MiB is skipped and counted in ``skippedFiles`` and gets **no** File record. A non-Joern unit is
one file (``rootPath`` names a blob) or a directory of files of its language; directories are walked with no special case for
dot directories (``.claude/agents/*.md`` are prompt files).

Header: a host bundle (shell, config, markdown) never runs in the sandbox, so its ``sandboxAuditIds`` is ``[]`` (bundle spec6 §2; a code
bundle's ids come from its sandbox runs and are never empty: ``attestation`` reports a host bundle ``host-unattested``). The non-Joern header is built from the unit's *merged* record set; ``writer_rules.bind_header_fields`` decides
``bundleSchemaVersion`` (``"1.1"`` iff some entry carries ``bindParts``) and ``bindKeyCheck``. A Joern bundle is written by
``assemble_bundle.build_bundle`` and carries no config entry, so it is ``"1.0"`` by construction.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Mapping, NamedTuple, Sequence

from ccg import atrest, bundle, ids, rulepack, units as units_mod
from ccg.packrt import run_packs
from ccg.units import Unit
from extractors.common import writer_rules as wr
from extractors.common.errors import SourceRefused
from extractors.config import formats as config_formats
from extractors.config.extract import ConfigResult, extract_config, file_role as config_file_role
from extractors.joern import run as joern_run
from extractors.joern.normalize import assemble_bundle, blame
from extractors.markdown import extract as markdown_mod
from extractors.markdown.extract import MarkdownError, extract_markdown
from extractors.markdown.globs import Globs, load_globs
from extractors.shell import extract as shell_mod
from extractors.shell import extract_shell

CPG = Path(__file__).resolve().parent.parent
MCP_PACK = CPG / "packs" / "fastmcp" / "pack.yaml"

#: The codes this module assigns; each is a member of ``FAIL_REASONS`` (``ccg/staleness.py``, P1-18).
SHALLOW_CLONE = "build:shallow-clone"
EXTRACT_FAILED = "extract:failed"
EXTRACT_TIMEOUT = "extract:timeout"
BUNDLE_INVALID = "bundle:invalid"
FAIL_CODES = frozenset({SHALLOW_CLONE, EXTRACT_FAILED, EXTRACT_TIMEOUT, BUNDLE_INVALID})

CODE_KINDS = ("python", "javascript")
HOST_KINDS = ("shell", "config", "markdown")
KINDS = CODE_KINDS + HOST_KINDS
_ORDER = {k: i for i, k in enumerate(KINDS)}                 # code units, then shell, config, markdown (known_tools first)
_REGULAR = ("100644", "100755")                             # a symlink (120000) and a gitlink (160000) are never read or followed

# Bounds on what is read from the commit's tree (the repository is untrusted; a multi-GB blob must not exhaust memory or the temp
# disk). A blob over its extractor's own cap is treated as the extractor treats it: shell refuses (the unit fails), markdown skips
# (``skippedDocs``), config skips (``skippedFiles``) once over ``MAX_CONFIG_BLOB_BYTES`` (the config extractor itself skips a file over
# its 512 KiB parse cap but still decodes it, so a larger blob is not even fetched). A unit over the total or the file count fails.
MAX_SHELL_BLOB_BYTES = shell_mod.MAX_FILE_BYTES
MAX_MARKDOWN_BLOB_BYTES = markdown_mod.MAX_FILE_BYTES
MAX_CONFIG_BLOB_BYTES = 8 * 1024 * 1024
MAX_UNIT_BYTES = 64 * 1024 * 1024
MAX_UNIT_FILES = 5000
MAX_PATH_BYTES = 4096
MAX_SEGMENT_BYTES = 255


class UnknownUnitKind(ValueError):
    """``unit.language`` has no extractor: a caller bug (discovery yields only ``KINDS``), never skipped."""


class InvalidCommit(ValueError):
    """``commit`` is not a full lowercase object id: a caller bug (a ref or a prefix could move between git calls)."""


@dataclass(frozen=True)
class BundleResult:
    """The outcome of one unit. ``fail_reason`` is one code of ``FAIL_CODES`` (``ok`` False) or None; ``report`` holds counts only."""
    unit: Unit
    commit: str
    ok: bool
    path: Path | None = None
    bundle_schema_version: str | None = None
    fail_reason: str | None = None
    report: Mapping[str, int] = field(default_factory=dict)


@dataclass
class BuildContext:
    """What one build shares between its units; every field has a default, so a single ``build_unit`` call needs none.

    ``units`` is every unit of the system (descendant roots for Joern; the config files of a repository for W2).
    ``known_tools`` is the markdown vocabulary."""
    units: Sequence[Unit] = ()
    root: str | Path | None = None
    keys_root: str | Path | None = None
    built_at: str | None = None
    known_tools: frozenset[str] = frozenset()
    globs: Globs | None = None
    manifest: str | Path = CPG / "ccg" / "system.yaml"          # read for the prompt-bearing globs when ``globs`` is None
    runner: str | Path = joern_run.RUNNER
    timeout: float = joern_run.STAGE_TIMEOUT_S
    kill_grace: float = joern_run.KILL_GRACE_S
    wall_cap: float = joern_run.INDEXER_WALL_CAP_S
    _trees: dict = field(default_factory=dict, repr=False)
    _declared: dict = field(default_factory=dict, repr=False)


@dataclass(frozen=True)
class BuildReport:
    results: tuple[BundleResult, ...]
    known_tools: frozenset[str]
    skipped_docs: int


# --- failure mapping -------------------------------------------------------------------------------------------


def fail_code(exc: BaseException) -> str:
    """The one code of ``FAIL_CODES`` for an exception raised after the history check (plan P1-44 (3))."""
    if isinstance(exc, joern_run.RunnerTimeout):
        return EXTRACT_TIMEOUT
    if isinstance(exc, bundle.BundleError):
        return BUNDLE_INVALID
    return EXTRACT_FAILED          # ExtractError, SourceRefused, RuntimeRefused, BuildError, HistoryError and anything unlisted


# --- the commit's tree ------------------------------------------------------------------------------------------


class _Entry(NamedTuple):
    mode: str
    oid: str
    size: int


def _check_tree_path(path: str) -> str:
    """Refuse a tree path that is not an ordinary repository-relative name; the rule of the shell extractor (``_check_rel``: no empty,
    ``.`` or ``..`` segment, not absolute, no NUL) plus a backslash, a control character, a ``.git`` segment (any case) and a
    name or path over the filesystem's limits. Raises ``SourceRefused`` (value-free): the unit fails, nothing is written."""
    shell_mod.check_rel(path)
    raw = path.encode("utf-8", "surrogateescape")
    segs = path.split("/")
    if ("\\" in path or any(ord(c) < 32 or ord(c) == 127 for c in path) or any(x.lower() == ".git" for x in segs)
            or len(raw) > MAX_PATH_BYTES or any(len(x.encode("utf-8", "surrogateescape")) > MAX_SEGMENT_BYTES for x in segs)):
        raise SourceRefused("a tree path is not an ordinary repository-relative name")
    return path


def _assert_inside(dest: Path, root: Path) -> None:
    """``dest`` resolves below ``root`` (checked before the write: a ``..`` or a symlink must not leave the stage)."""
    if not Path(dest).resolve().is_relative_to(Path(root).resolve()):
        raise SourceRefused("a stage path resolves outside the stage directory")


def _tree(ctx: BuildContext, repo: Path, sha: str) -> dict[str, _Entry]:
    """``{path: _Entry}`` of every blob of ``sha`` (paths relative to ``repo``), listed once per build. Names are NOT validated
    here: ``_unit_files`` validates the paths it selects, so an odd name outside a unit is never selected and never written."""
    key = (str(repo), sha)
    if key not in ctx._trees:
        out = blame.git(repo, "ls-tree", "-r", "-l", "-z", sha)
        tree = {}
        for rec in out.split(b"\0"):
            if not rec:
                continue
            meta, _, path = rec.partition(b"\t")
            mode, typ, oid, size = meta.decode().split()
            if typ == "blob":
                tree[path.decode("utf-8", "surrogateescape")] = _Entry(mode, oid, int(size))
        ctx._trees[key] = tree
    return ctx._trees[key]


def _blob(repo: Path, oid: str) -> bytes:
    return blame.git(repo, "cat-file", "blob", oid)


def _language_of(repo: Path, path: str, entry: _Entry) -> str | None:
    """The per-file token of a file of a directory unit: ``units.classify_file`` over the commit's bytes. An extensionless
    blob over the shell cap is never fetched to look for a shebang (a script that large would be refused anyway)."""
    def first_line() -> bytes:
        return _blob(repo, entry.oid).split(b"\n", 1)[0][:256] if entry.size <= MAX_SHELL_BLOB_BYTES else b""

    return units_mod.classify_file(path, first_line)


def _unit_files(repo: Path, tree: Mapping[str, _Entry], unit: Unit) -> list[str]:
    """The regular files of a non-Joern unit at the commit: its one file, or the files of its language below its directory.
    Only the selected paths (and the unit root) are validated: a hostile name inside the unit fails that unit,
    one outside it (another directory, another language) is not selected and costs nothing. A unit over ``MAX_UNIT_FILES`` files or ``MAX_UNIT_BYTES`` bytes fails (``SourceRefused``) before any blob is read."""
    root = unit.rootPath
    if root != ".":
        _check_tree_path(root)
    if root in tree:
        out = [root] if tree[root].mode in _REGULAR else []
    else:
        prefix = "" if root == "." else root + "/"
        out = sorted(_check_tree_path(path) for path, e in tree.items()          # a selected path is validated before any use
                     if path.startswith(prefix) and e.mode in _REGULAR
                     and not any(units_mod.is_skipped_dir(seg) for seg in path.split("/")[:-1])     # dot directories are NOT skipped
                     and _language_of(repo, path, e) == unit.language)
    if len(out) > MAX_UNIT_FILES or sum(tree[p].size for p in out) > MAX_UNIT_BYTES:
        raise SourceRefused("a unit has more files or bytes than a build reads")
    return out


# --- history -----------------------------------------------------------------------------------------------------


class _History:
    """``(lastChangedAt, lastChangedCommit)`` from the source repo at the built commit, blamed once per file."""

    def __init__(self, repo: Path, commit: str):
        self.repo, self.commit, self._blamed = repo, commit, {}

    def of(self, path: str, regions: list[tuple[int, int]]) -> tuple[str, str]:
        if path not in self._blamed:
            self._blamed[path] = blame.blame(self.repo, self.commit, path)
        per_line, times = self._blamed[path]
        try:
            return blame.last_changed(per_line, times, regions)
        except ValueError:                                  # no blamed line: an empty file
            return blame.path_last_commit(self.repo, self.commit, path)

    def whole_file(self, path: str, data: bytes) -> tuple[str, str]:
        lines = data.count(b"\n") + (0 if data.endswith(b"\n") or not data else 1)
        return self.of(path, [(1, max(lines, 1))])


# --- host extractors ---------------------------------------------------------------------------------------------


def _line_count(data: bytes) -> int:
    return data.count(b"\n") + (0 if data.endswith(b"\n") or not data else 1)


def _declared_services(ctx: BuildContext, repo: Path, sha: str, tree, unit: Unit, key: bytes) -> frozenset[str]:
    """W2's declared-service set of the unit's repository at ``sha``: ``extract_config`` once over the Compose files of its config
    units that decode as UTF-8 (cached per commit and per set of Compose paths, so a context reused across commits never goes stale).

    A Compose file that does not decode is left out, so it fails *its own* unit (that unit's own ``extract_config`` call refuses it)
    and no other. Services declared only in such a file are therefore unknown to W2, which stays conservative: a reference to
    one is not bound (``bindValue`` is None), never guessed."""
    config_units = [u for u in ctx.units if u.repoId == unit.repoId and u.language == "config"] or [unit]
    paths = sorted({p for u in config_units for p in _unit_files(repo, tree, u)
                    if config_formats.is_compose(p) and tree[p].size <= MAX_CONFIG_BLOB_BYTES})
    k = (str(repo), sha, hashlib.sha256("\0".join(paths).encode("utf-8", "surrogateescape")).hexdigest())
    if k not in ctx._declared:
        compose = {}
        for p in paths:
            data = _blob(repo, tree[p].oid)
            try:
                data.decode("utf-8-sig")
            except UnicodeDecodeError:
                continue
            compose[p] = data
        ctx._declared[k] = extract_config(unit.repoId, compose, key=key).declared_services if compose else frozenset()
    return ctx._declared[k]


def _file_record(unit: Unit, path: str, data: bytes, kind: str, role: str, lang: str) -> dict:
    return {"type": "file", "id": ids.file_id(unit.repoId, path), "repoId": unit.repoId, "unitId": unit.unitId, "path": path,
            "lang": lang, "kind": kind, "role": role, "contentHash": hashlib.sha256(data).hexdigest(),
            "lineCount": _line_count(data)}


def _shell_records(repo: Path, tree, unit: Unit, files: list[str], hist: _History, report: Counter) -> list[dict]:
    with tempfile.TemporaryDirectory(prefix="ccg-shell-") as td:        # mkdtemp is 0700; removed on every path
        for path in files:
            dest = Path(td, path)
            _assert_inside(dest, Path(td))                               # before mkdir and write; _tree already refused hostile names
            if tree[path].size > MAX_SHELL_BLOB_BYTES:
                raise SourceRefused("a shell file is over the size cap")  # the extractor's own rule: refuse the unit, nothing is skipped
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(_blob(repo, tree[path].oid))
        warnings: list = []
        records = extract_shell(td, files, repo_id=unit.repoId, unit_id=unit.unitId, history=hist.of, warnings=warnings)
    report["warnings"] += len(warnings)
    return records


def _config_records(ctx: BuildContext, repo: Path, sha: str, tree, unit: Unit, files: list[str], key: bytes,
                    report: Counter) -> list[dict]:
    declared = _declared_services(ctx, repo, sha, tree, unit, key)
    big = [p for p in files if tree[p].size > MAX_CONFIG_BLOB_BYTES]      # not fetched: skipped and counted, no File record
    report["skippedFiles"] += len(big)
    files = [p for p in files if p not in big]
    data = {p: _blob(repo, tree[p].oid) for p in files}
    result: ConfigResult = extract_config(unit.repoId, data, key=key, extra_declared_services=declared)
    records = list(result.records)
    for path in files:                                                   # a file with no records still gets its File
        role = result.roles.get(path) or config_file_role(path)
        records.append(_file_record(unit, path, data[path], "config", role, "config"))
    report["skippedFiles"] += result.report.get("skippedFiles", 0)
    return records


def _markdown_records(ctx: BuildContext, repo: Path, tree, unit: Unit, files: list[str], hist: _History,
                      report: Counter) -> list[dict]:
    globs = ctx.globs or load_globs(ctx.manifest)
    records: list[dict] = []
    for path in files:
        if tree[path].size > MAX_MARKDOWN_BLOB_BYTES:                    # skipped before the fetch and before any blame
            report["skippedDocs"] += 1
            continue
        data = _blob(repo, tree[path].oid)
        try:
            ex = extract_markdown(path, data, repo_id=unit.repoId, unit_id=unit.unitId, globs=globs, bundle_path=None,
                                  history=hist.whole_file(path, data), known_tools=ctx.known_tools)
        except MarkdownError:                                            # an oversize file: skipped, the unit builds
            report["skippedDocs"] += 1
            continue
        records += ex.records
    return records


_EXTRACTOR_MODULES = {                       # every module whose logic a host bundle's records depend on (cpg-relative paths)
    "shell": ("extractors/shell/extract.py", "extractors/common/writer_rules.py", "extractors/common/errors.py"),
    "config": ("extractors/config/extract.py", "extractors/config/formats.py", "extractors/config/launch.py",
               "extractors/common/writer_rules.py", "extractors/common/errors.py"),
    "markdown": ("extractors/markdown/extract.py", "extractors/markdown/globs.py", "ccg/globs.py",      # the matcher and classify live in ccg
                 "extractors/common/writer_rules.py"),
}
_FRONTEND = {"shell": "tree-sitter-bash", "config": "none", "markdown": "none"}
_NORMALIZER = {"shell": "ccg-shell-extract", "config": "ccg-config-extract", "markdown": "ccg-markdown-extract"}


def _extractor_hash(kind: str) -> str:
    """``extractScriptHash``: sha256 over the sorted (path, sha256 of bytes) of *every* module of the kind's extractor, so a change
    in any of them changes the header (a one-file hash would claim "same extractor" for different logic)."""
    h = hashlib.sha256()
    for rel in sorted(_EXTRACTOR_MODULES[kind]):
        h.update(rel.encode() + b"\0" + hashlib.sha256((CPG / rel).read_bytes()).digest())
    return h.hexdigest()


def _versions(kind: str) -> tuple[str, str]:
    """``(frontendVersion, normalizerVersion)``. The extractors declare no version of their own, so the normalizer's version is the
    first 12 hex digits of ``_extractor_hash`` (it changes exactly when its code does); the shell frontend is the installed
    ``tree-sitter-bash`` distribution's version, and the config and markdown extractors have no separate frontend (``"none"``)."""
    front = importlib.metadata.version("tree-sitter-bash") if kind == "shell" else "none"
    return front, _extractor_hash(kind)[:12]


def _header(ctx: BuildContext, unit: Unit, sha: str, tree_id: str, records: list[dict], key: bytes) -> dict:
    front_version, norm_version = _versions(unit.language)
    counts = Counter(r["type"] for r in records)
    # vendoredDirs is every directory name ``_unit_files`` skips (``units.SKIP_DIRS``), so each excluded file is explained by the
    # header. A ``*.egg-info`` directory is skipped as well but is a suffix rule a name list cannot state; a per-repository
    # ``vendoredDirs`` of the manifest is not honoured here (open question for the architect).
    header = {
        "type": "header", "bundleSchemaVersion": "1.0",
        "unit": {"repoId": unit.repoId, "rootPath": unit.rootPath, "lang": unit.language}, "unitId": unit.unitId,
        "builtCommit": sha, "builtTree": tree_id,
        "builtAt": ctx.built_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tools": {"joern": "none", "frontend": _FRONTEND[unit.language], "frontendVersion": front_version,
                  "normalizer": _NORMALIZER[unit.language], "normalizerVersion": norm_version, "extractScript": "extract.py",
                  "extractScriptHash": _extractor_hash(unit.language)},
        "sandboxAuditIds": [], "egressMode": "offline", "flowProjected": False,
        "fieldDepthCap": 3, "vendoredDirs": sorted(units_mod.SKIP_DIRS), "history": "available",
        "counts": {t: counts.get(t, 0) for t in bundle.RECORD_TYPES[1:]},
    }
    header.update(wr.bind_header_fields(key, records))                   # (8): the merged set decides the version
    return header


def _build_host(ctx: BuildContext, unit: Unit, repo: Path, sha: str, system: str, report: Counter) -> tuple[Path, str]:
    key = atrest.hmac_key(system, ctx.keys_root)
    tree = _tree(ctx, repo, sha)
    files = _unit_files(repo, tree, unit)
    hist = _History(repo, sha)
    if unit.language == "shell":
        records = _shell_records(repo, tree, unit, files, hist, report)
    elif unit.language == "config":
        records = _config_records(ctx, repo, sha, tree, unit, files, key, report)
    else:
        records = _markdown_records(ctx, repo, tree, unit, files, hist, report)
    rev = f"{sha}^{{tree}}" if unit.rootPath == "." else f"{sha}:{unit.rootPath}"
    tree_id = blame.git(repo, "rev-parse", "--verify", "--end-of-options", rev).decode().strip()
    merged = [_header(ctx, unit, sha, tree_id, records, key), *records]
    path = atrest.write_bundle(system, unit.unitId, sha, merged, root=ctx.root)
    return path, merged[0]["bundleSchemaVersion"]


def _build_code(ctx: BuildContext, unit: Unit, repo: Path, sha: str, system: str, report: dict) -> tuple[Path, str]:
    area = Path(ctx.root) if ctx.root is not None else atrest.WORK_ROOT
    units = list(ctx.units) or [unit]
    raw = joern_run.extract_unit(repo, unit, units, sha, area / "joern" / unit.unitId, area=area, runner=ctx.runner,
                                 timeout=ctx.timeout, kill_grace=ctx.kill_grace, wall_cap=ctx.wall_cap)
    path = assemble_bundle.build_bundle(unit, raw, repo, sha, system=system, root=ctx.root, built_at=ctx.built_at,
                                        report=report)
    return path, "1.0"


# --- the entry points ----------------------------------------------------------------------------------------------


def build_unit(unit: Unit, repo: str | Path, commit: str, system: str, ctx: BuildContext | None = None) -> BundleResult:
    """Build ``unit`` of the git work tree ``repo`` at ``commit`` into system ``system``'s bundle area.

    Returns a failed ``BundleResult`` (one code of ``FAIL_CODES``) for any extraction failure; raises ``UnknownUnitKind`` for a
    kind with no extractor, ``InvalidCommit`` for a commit that is not a full object id; a host unit needs no audit id."""
    ctx = ctx or BuildContext(units=(unit,))
    if unit.language not in KINDS:
        raise UnknownUnitKind(f"unit language {unit.language!r} has no extractor (known: {', '.join(KINDS)})")
    if not blame.is_commit_id(commit):
        raise InvalidCommit("commit must be a full lowercase object id")
    repo = Path(repo)
    report: Counter | dict = Counter() if unit.language in HOST_KINDS else {}
    try:
        try:
            blame.require_full_history(repo)                            # before extraction: a shallow clone fails here
        except blame.HistoryError:
            return BundleResult(unit, commit, False, fail_reason=SHALLOW_CLONE)
        if unit.language in HOST_KINDS:
            path, version = _build_host(ctx, unit, repo, commit, system, report)
        else:
            path, version = _build_code(ctx, unit, repo, commit, system, report)
    except Exception as exc:                                            # noqa: BLE001 - every failure fails this unit alone
        return BundleResult(unit, commit, False, fail_reason=fail_code(exc), report=_counts(report))
    return BundleResult(unit, commit, True, path=path, bundle_schema_version=version, report=_counts(report))


def _counts(report: Mapping) -> dict[str, int]:
    """Only the integer entries: a report never carries text from a source file."""
    return {k: v for k, v in report.items() if isinstance(v, int) and not isinstance(v, bool)}


def known_tools_of(bundles: Iterable[list[dict]]) -> frozenset[str]:
    """The system's MCP tool names: the ``fastmcp`` pack's ``PROVIDES mcp_tool`` keys over the code ``bundles`` (P1-16)."""
    facts = run_packs(list(bundles), [rulepack.load_pack(MCP_PACK)])
    return frozenset(f["key"] for f in facts.providers if f["kind"] == "mcp_tool" and f["key"])


def build_units(units: Sequence[Unit], repos: Mapping[str, str | Path], commits: Mapping[str, str], system: str,
                ctx: BuildContext | None = None, *, build: Callable[..., BundleResult] | None = None) -> BuildReport:
    """Build every unit: code units first, then ``known_tools`` once, then shell, config and markdown units.

    ``repos`` maps a ``repoId`` to its git work tree and ``commits`` to the commit built. Each unit fails alone."""
    build = build or build_unit
    ctx = ctx or BuildContext()
    ctx.units = tuple(units)
    ordered = sorted(units, key=lambda u: (_ORDER.get(u.language, len(KINDS)), u.repoId, u.rootPath))
    code = [u for u in ordered if u.language in CODE_KINDS]
    rest = [u for u in ordered if u.language not in CODE_KINDS]
    results: list[BundleResult] = []
    code_bundles: list[list[dict]] = []
    for u in code:
        r = build(u, repos[u.repoId], commits[u.repoId], system, ctx)
        results.append(r)
        if r.ok:
            code_bundles.append(bundle.loads(atrest.read_bundle(system, u.unitId, r.commit, ctx.root)))
    if any(u.language == "markdown" for u in rest):                    # once per build, after every code bundle exists
        ctx.known_tools = known_tools_of(code_bundles)
    for u in rest:
        results.append(build(u, repos[u.repoId], commits[u.repoId], system, ctx))
    skipped = sum(r.report.get("skippedDocs", 0) for r in results)
    return BuildReport(tuple(results), ctx.known_tools, skipped)
