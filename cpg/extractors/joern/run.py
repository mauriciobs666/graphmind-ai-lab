"""Joern raw-fact extraction for one unit (C-1001 P1-04; replaces the slice's ``run_extract.sh``).

Plan: ``docs/plans/connected-code-graph-p1-extraction.md`` § P1-04. Per unit, every container step goes through
``cpg/sandbox/offline-run.sh --backend container --mode offline`` (``docs/plans/connected-code-graph-egress-spec3.md``
§7.0, §7.4, §7.8); nothing here runs ``npm``, ``pip`` or ``uv``, and nothing but the runner starts a container.

1. **stage**   the tree of ``<commit>`` at ``<rootPath>`` (``git ls-tree`` + ``git cat-file``, never ``git archive``) on the
   host, rooted at the unit, minus every path under a
   descendant unit root of the *same language* (so no file is staged in two bundles);
2. **parse**   ``joern-parse`` (``pythonsrc`` / ``jssrc``);
3. **check**   ``cpg/sandbox/scripts/cpg-content-check.sc``: a parse is certified by content (D15, risk R11), because
   ``joern-parse`` can exit 0 with an empty CPG;
4. **astgen**  javascript only: the Babel ASTs the normalizer reads for imports and type end lines;
5. **extract** ``extract.sc``: the raw facts JSONL.

``extract_unit`` returns the raw-facts path and the ``sandboxAuditIds`` of the bundle's inputs, in invocation order
(parse, astgen, extract; the content check is a record of its own and not an input). It writes no bundle.
"""
from __future__ import annotations

import argparse
import io
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from ccg import atrest, gitsafe
from ccg.units import Unit

HERE = Path(__file__).resolve().parent
CPG = HERE.parent.parent
REPO = CPG.parent
RUNNER = CPG / "sandbox" / "offline-run.sh"
CONTENT_CHECK_SC = CPG / "sandbox" / "scripts" / "cpg-content-check.sc"
EXTRACT_SC = HERE / "extract.sc"
ASTGEN_SC = HERE / "astgen.sc"

#: unit language -> (``joern-parse --language`` value, source-file extensions that count toward "the unit has sources").
#: The extension sets are the slice's own (``run_extract.sh``): equivalence before improvement.
LANGUAGES = {
    "python": ("pythonsrc", frozenset({".py"})),
    "javascript": ("jssrc", frozenset({".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"})),
}
STAGE_TIMEOUT_S = 1800
INDEXER_WALL_CAP_S = 1800      # the runner's own watchdog, ``SBX_INDEXER_WALL_S`` (cpg/sandbox/README.md); its exit 124 at this age is a timeout
CAVEAT = ("Caveat: units are discovered from the git index, not from --commit (P1-20 owns discover_units(commit=...)); "
          "run with the index equal to the commit.")
KILL_GRACE_S = 60


class ExtractError(RuntimeError):
    """Any failure of the extraction of one unit; the subclasses say which."""


class StageError(ExtractError):
    """The unit could not be staged (bad commit, bad archive, no source file)."""


class EmptyCpgError(ExtractError):
    """The content check's own verdict: the parse produced a CPG with no non-external METHOD (R11, H12)."""


class RunnerError(ExtractError):
    """A runner invocation failed for any other reason (refusal, backend, tool or script error)."""


class RunnerTimeout(RunnerError):
    """A runner invocation ran past its wall-clock cap (``_run`` terminated it); a subclass, so ``except RunnerError`` still
    catches it, and P1-44 maps it to its own failure code (``extract:timeout``)."""


# --- staging ----------------------------------------------------------------------------------------------------


def _strictly_under(child: str, parent: str) -> bool:
    return child != parent and (child.startswith(parent + "/") if parent != "." else True)


def descendant_roots(unit: Unit, units: Iterable[Unit]) -> list[str]:
    """Root paths of the units of ``units`` that lie below ``unit`` in the same repository and language.

    These are the paths staging leaves out of ``unit``: each belongs to its own unit's bundle. A unit of another
    language (a shell or config file, a javascript unit under a python one) is not an exclusion: its files are not
    this unit's language and stay in the stage, as the slice's stages did.
    """
    return sorted({u.rootPath for u in units
                   if u.repoId == unit.repoId and u.language == unit.language and _strictly_under(u.rootPath, unit.rootPath)})


def _resolve_commit(repo: Path, commit: str) -> str:
    try:
        out = gitsafe.git(repo, "rev-parse", "--verify", "--end-of-options", f"{commit}^{{commit}}", timeout=60)
    except (OSError, subprocess.SubprocessError) as e:
        raise StageError(f"{commit!r} is not a commit of {repo}") from e
    return out.decode().strip()


def is_staged_source(p: Path) -> bool:
    """A staged file that counts as source: a regular (not symlink) non-empty file. The one predicate of
    ``_count_sources`` and of ``skipped.candidate_sources`` (P1-08), so the two cannot disagree."""
    return not p.is_symlink() and p.stat().st_size > 0


def _count_sources(stage: Path, extensions: frozenset[str]) -> int:
    n = 0
    for dirpath, _dirs, names in os.walk(stage):
        for name in names:
            p = Path(dirpath, name)
            if p.suffix in extensions and is_staged_source(p):
                n += 1
    return n


def _git(repo: Path, *args: str) -> list[str]:
    """``git`` argv of ``ccg.gitsafe`` (hardened configuration, literal pathspecs: a ``rootPath`` such as ``a*`` or ``a[1]`` names that
    directory, not a glob); run it with ``gitsafe.environment()``."""
    return gitsafe.checked_command(repo, *args)


def _tree_entries(repo: Path, sha: str, root: str) -> list[tuple[str, str, str]]:
    """``(mode, blob id, path)`` of every blob and symlink of ``sha`` at or below ``root``, as ``git ls-tree`` lists them.

    No ``--full-tree``: without it the pathspec and the listed paths are relative to ``repo``, as ``discover_units``'
    roots are, so a repository that is a subdirectory of its git work tree stages its own files (a bare ``ls-tree``
    inside it lists only that subtree, so its paths still start at ``root``'s own components)."""
    try:
        out = gitsafe.git(repo, "ls-tree", "-r", "-z", sha, paths=[] if root == "." else [root], timeout=300)
    except (OSError, subprocess.SubprocessError) as e:
        raise StageError(f"git ls-tree {root!r} at {sha[:12]} failed: {e}") from e
    entries = []
    for rec in out.split(b"\0"):
        if not rec:
            continue
        meta, _, path = rec.partition(b"\t")
        mode, typ, oid = meta.decode().split(" ")
        if typ == "blob":                                  # a submodule (`commit`) has no files of ours
            entries.append((mode, oid, path.decode("utf-8", "surrogateescape")))
    return entries


MAX_BLOB_BYTES = 64 * 1024 * 1024        # one staged blob; the repository under analysis is hostile (a multi-GB blob is a memory/disk DoS)
MAX_STAGE_BYTES = 1024 * 1024 * 1024     # all blobs of one unit
STAGE_TIMEOUT = 600                      # seconds for the whole ``cat-file --batch`` stream


def stage_unit(repo: str | Path, commit: str, unit: Unit, exclude: Sequence[str], dest: str | Path) -> int:
    """Stage ``unit.rootPath`` of ``commit`` into ``dest``, rooted at the unit, minus ``exclude``.

    The files are the commit's tree exactly: ``git ls-tree -r`` for the list and ``git cat-file --batch`` for the
    blobs, never ``git archive``, whose ``export-ignore`` and ``export-subst`` attributes (read from the commit's own
    ``.gitattributes``) would drop files and rewrite text, so the bundle would cover other bytes than ``builtCommit``
    names. Pathspecs are literal. ``exclude`` holds repo-relative root paths (``descendant_roots``); a path at or below
    one is dropped. The entries are written to a private temporary tar and extracted under tarfile's ``data`` filter,
    so a symlink (absolute or leaving ``dest``) or a path escaping ``dest`` fails the stage: loud on purpose, where the
    slice's ``tar -x`` kept a dangling link. Returns the number of non-empty source files of the unit's language; none
    is a ``StageError`` (an empty stage would parse as success). History is never staged (BF12).
    """
    repo, dest = Path(repo), Path(dest)
    lang = LANGUAGES.get(unit.language)
    if lang is None:
        raise StageError(f"language {unit.language!r} has no Joern extractor in P1")
    sha = _resolve_commit(repo, commit)
    root = unit.rootPath
    strip = 0 if root == "." else len(root.split("/"))
    rel_excl = []
    for e in exclude:
        if not _strictly_under(e, root):
            raise StageError(f"exclusion {e!r} is not below the unit root {root!r}")
        rel_excl.append(e if root == "." else e[len(root) + 1:])
    dest.mkdir(parents=True, exist_ok=True)
    wanted = []
    for mode, oid, path in _tree_entries(repo, sha, root):
        rel = "/".join(path.split("/")[strip:])
        if not rel or any(rel == e or rel.startswith(e + "/") for e in rel_excl):
            continue
        wanted.append((mode, oid, rel))
    try:
        argv = _git(repo, "cat-file", "--batch")                       # checked like every ``gitsafe.git`` call, not by the ones before it
    except gitsafe.GitError as e:
        raise StageError(f"staging {root!r} at {sha[:12]} refused: {e}") from e
    proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, env=gitsafe.environment())
    timed_out = threading.Event()

    def expire() -> None:                                                # a hung or endless git must not hold the build
        timed_out.set()
        proc.kill()
    deadline = threading.Timer(STAGE_TIMEOUT, expire)
    deadline.start()
    staged = 0
    try:
        with tempfile.TemporaryFile(dir=dest.parent) as spool:
            with tarfile.open(fileobj=spool, mode="w") as out:
                for mode, oid, rel in wanted:
                    proc.stdin.write(oid.encode() + b"\n")
                    proc.stdin.flush()
                    head = proc.stdout.readline().split()
                    if timed_out.is_set():
                        raise StageError(f"staging {root!r} at {sha[:12]} timed out after {STAGE_TIMEOUT} s")
                    if len(head) != 3 or head[1] != b"blob":
                        raise StageError(f"git cat-file: no blob {oid} for {rel!r}")
                    size = int(head[2])
                    staged += size
                    if size > MAX_BLOB_BYTES:
                        raise StageError(f"{rel!r} is {size} bytes: over the per-blob cap of {MAX_BLOB_BYTES}")
                    if staged > MAX_STAGE_BYTES:
                        raise StageError(f"staging {root!r} at {sha[:12]}: total over the cap of {MAX_STAGE_BYTES} bytes")
                    data = proc.stdout.read(size)
                    proc.stdout.read(1)                                # the newline after the object
                    ti = tarfile.TarInfo(rel)
                    if mode == "120000":
                        ti.type, ti.linkname, ti.mode = tarfile.SYMTYPE, data.decode("utf-8", "surrogateescape"), 0o777
                        out.addfile(ti)
                    else:
                        ti.size, ti.mode = len(data), 0o755 if mode == "100755" else 0o644
                        out.addfile(ti, io.BytesIO(data))
            if timed_out.is_set():                                       # the deadline fired inside the last read: its blob is truncated
                raise StageError(f"staging {root!r} at {sha[:12]} timed out after {STAGE_TIMEOUT} s")
            spool.seek(0)
            with tarfile.open(fileobj=spool, mode="r:") as tar:
                for m in tar:
                    tar.extract(m, dest, filter="data")
    except (tarfile.TarError, OSError, ValueError) as e:
        raise StageError(f"staging {root!r} at {sha[:12]} failed: {e}") from e
    finally:
        deadline.cancel()
        proc.stdin.close()
        proc.stdout.close()
        proc.kill()
        proc.wait()
    n = _count_sources(dest, lang[1])
    if n < 1:
        raise StageError(f"unit {unit.repoId}:{root} ({unit.language}): no non-empty source file staged "
                         f"(counted: {' '.join(sorted(lang[1]))}; a symlink or an empty file does not count)")
    return n


# --- the runner argv of each stage -------------------------------------------------------------------------------


def _runner_argv(runner: Path, frontend: str, source: Path, output: Path, repo_id: str, script: Path | None,
                 tool_argv: Sequence[str]) -> list[str]:
    argv = [str(runner), "--backend", "container", "--mode", "offline", "--frontend", frontend,
            "--source", str(source), "--output", str(output), "--codebase", repo_id]
    if script is not None:
        argv += ["--script-file", str(script)]
    return argv + ["--", *tool_argv]


def parse_argv(runner: Path, source: Path, output: Path, repo_id: str, language: str) -> list[str]:
    return _runner_argv(runner, "joern-parse", source, output, repo_id, None, ["/src", "--language", LANGUAGES[language][0]])


def content_check_argv(runner: Path, source: Path, output: Path, repo_id: str, cpg_name: str) -> list[str]:
    return _runner_argv(runner, "joern", source, output, repo_id, CONTENT_CHECK_SC,
                        ["--script", f"/scripts/{CONTENT_CHECK_SC.name}", "--param", f"cpgFiles=/out/{cpg_name}"])


def astgen_argv(runner: Path, source: Path, output: Path, repo_id: str) -> list[str]:
    return _runner_argv(runner, "joern", source, output, repo_id, ASTGEN_SC,
                        ["--script", f"/scripts/{ASTGEN_SC.name}", "--param", "cpgFile=unused",
                         "--param", "outDir=/out/@AUDIT_ID@.astgen"])


def extract_argv(runner: Path, source: Path, output: Path, repo_id: str, cpg_name: str, language: str) -> list[str]:
    return _runner_argv(runner, "joern", source, output, repo_id, EXTRACT_SC,
                        ["--script", f"/scripts/{EXTRACT_SC.name}", "--param", f"cpgFile=/out/{cpg_name}",
                         "--param", "out=/out/@AUDIT_ID@.facts.jsonl", "--param", f"lang={language}"])


# --- running ----------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Completed:
    rc: int
    stdout: bytes
    stderr: bytes
    elapsed: float = 0.0       # seconds of wall-clock the invocation ran


def _run(argv: Sequence[str], timeout: float, kill_grace: float = KILL_GRACE_S) -> Completed:
    """Run one runner invocation. On timeout the runner gets SIGTERM first: its trap removes its own container.

    If it is still alive ``kill_grace`` seconds later it is SIGKILLed, and that path cannot remove the container: a
    killed runner never reaches its cleanup, so its container runs to its end under ``--rm``. ``run.py`` cannot do it
    either (only the runner starts or removes containers), so the fallback only guarantees that no runner outlives the
    call. ``tests/test_joern_run.py`` pins the kill, not a container removal."""
    started = time.monotonic()
    try:
        p = subprocess.Popen(list(argv), stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL,
                             start_new_session=True)
    except OSError as e:
        raise RunnerError(f"runner {argv[0]} could not be started: {e}") from e
    try:
        out, err = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        p.send_signal(signal.SIGTERM)
        try:
            p.communicate(timeout=kill_grace)
        except subprocess.TimeoutExpired:
            p.kill()
            p.communicate()
        raise RunnerTimeout(f"runner invocation timed out after {timeout}s: {Path(argv[0]).name} {' '.join(argv[1:8])}") from None
    return Completed(p.returncode, out, err, time.monotonic() - started)


def _new_with_suffix(out_dir: Path, before: set[str], suffix: str) -> Path:
    new = sorted(n for n in set(os.listdir(out_dir)) - before if n.endswith(suffix))
    if len(new) != 1:
        raise RunnerError(f"expected exactly one new *{suffix} in {out_dir}, found {new}")
    return out_dir / new[0]


def _stem(path: Path, suffix: str) -> str:
    return path.name[: -len(suffix)]


def _mkdir_private(path: Path, area: Path) -> None:
    """Create ``path`` (and parents below ``area``) at 0700 and check the chain; raw facts carry source text."""
    atrest.ensure_private(path, area)
    area = Path(area)
    cur = Path(os.path.realpath(area))
    rel = Path(os.path.abspath(path)).relative_to(Path(os.path.abspath(area)))
    for part in rel.parts:
        cur = cur / part
        if not cur.exists():
            os.mkdir(cur, 0o700)
            os.chmod(cur, 0o700)
    atrest.ensure_private(path, area)


@dataclass(frozen=True)
class RawFacts:
    unit: Unit
    commit: str
    facts: Path
    sandbox_audit_ids: tuple[str, ...]
    stage: Path
    cpg_dir: Path
    astgen_dir: Path | None
    content_check: Path
    source_files: int


def extract_unit(repo: str | Path, unit: Unit, units: Iterable[Unit], commit: str, work: str | Path, *,
                 area: str | Path = atrest.WORK_ROOT, runner: str | Path = RUNNER,
                 timeout: float = STAGE_TIMEOUT_S, kill_grace: float = KILL_GRACE_S,
                 wall_cap: float = INDEXER_WALL_CAP_S) -> RawFacts:
    """Build the raw facts of ``unit`` at ``commit`` under ``work`` (``<work>/stage``, ``<work>/cpg``).

    ``repo`` is the git work tree of the unit's repository, ``units`` every unit of the system (the descendants of
    ``unit`` are read from it). ``work`` must lie below ``area`` (default the work area, ``.ccg-work``) and is
    recreated empty. One unit at a time: the 6 GB host runs one Joern JVM.
    """
    repo, work, area, runner = Path(repo), Path(work), Path(area), Path(runner)
    if unit.language not in LANGUAGES:
        raise StageError(f"language {unit.language!r} has no Joern extractor in P1")
    _mkdir_private(work, area)
    stage, cpg_dir = work / "stage", work / "cpg"
    for d in (stage, cpg_dir):
        if d.exists():
            shutil.rmtree(d)
    sha = _resolve_commit(repo, commit)        # once: a moving ref (HEAD, a branch) must not split the stage from the label
    stage_files = stage_unit(repo, sha, unit, descendant_roots(unit, units), stage)
    _mkdir_private(cpg_dir, area)
    lang = unit.language

    def go(stage_name: str, argv: list[str]) -> Completed:
        c = _run(argv, timeout, kill_grace)
        (work / f"{stage_name}.log").write_bytes(c.stdout + b"\n--stderr--\n" + c.stderr)
        return c

    def exit_error(c: Completed) -> type[RunnerError]:
        """Exit 124 is the runner's watchdog *or* a tool's own 124 (cpg/sandbox/README.md): it is a timeout only when the
        invocation ran for the wall-clock cap. A boundary race that says 124 just under the cap stays a plain failure (fail-closed)."""
        return RunnerTimeout if c.rc == 124 and c.elapsed >= wall_cap else RunnerError

    def need_ok(stage_name: str, c: Completed) -> None:
        if c.rc != 0:
            raise exit_error(c)(f"unit {unit.repoId}:{unit.rootPath}: {stage_name} exited {c.rc} (see {work / (stage_name + '.log')})")

    # 2. parse
    before = set(os.listdir(cpg_dir))
    need_ok("parse", go("parse", parse_argv(runner, stage, cpg_dir, unit.repoId, lang)))
    cpg_file = _new_with_suffix(cpg_dir, before, ".cpg.bin")
    parse_id = _stem(cpg_file, ".cpg.bin")

    # 3. content check (D15): the stage has >= 1 non-empty source file, so zero non-external METHODs is a failed parse.
    c = go("contentcheck", content_check_argv(runner, stage, cpg_dir, unit.repoId, cpg_file.name))
    check_txt = work / "contentcheck.txt"
    check_txt.write_bytes(c.stdout)
    out_text = c.stdout.decode("utf-8", "replace")
    if c.rc != 0:
        # Only the check's own verdict line may claim an empty CPG; any other non-zero is a runner failure.
        if f"CONTENT /out/{cpg_file.name}=EMPTY" in out_text.splitlines():
            raise EmptyCpgError(f"unit {unit.repoId}:{unit.rootPath}: the parse produced a CPG with no non-external METHOD "
                                f"although the unit has {stage_files} non-empty {lang} source files (R11, H12); see {check_txt}")
        raise exit_error(c)(f"unit {unit.repoId}:{unit.rootPath}: content-check exited {c.rc} without a CONTENT=EMPTY verdict "
                          f"(runner or script failure, not an empty CPG); see {check_txt}")
    if f"CONTENT /out/{cpg_file.name}=ok" not in out_text.splitlines():
        raise RunnerError(f"unit {unit.repoId}:{unit.rootPath}: content-check printed no CONTENT=ok line; see {check_txt}")

    # 4. astgen (javascript)
    ids = [parse_id]
    astgen_dir = None
    if lang == "javascript":
        before = set(os.listdir(cpg_dir))
        need_ok("astgen", go("astgen", astgen_argv(runner, stage, cpg_dir, unit.repoId)))
        astgen_dir = _new_with_suffix(cpg_dir, before, ".astgen")
        ids.append(_stem(astgen_dir, ".astgen"))

    # 5. extract
    before = set(os.listdir(cpg_dir))
    need_ok("extract", go("extract", extract_argv(runner, stage, cpg_dir, unit.repoId, cpg_file.name, lang)))
    facts = _new_with_suffix(cpg_dir, before, ".facts.jsonl")
    ids.append(_stem(facts, ".facts.jsonl"))
    return RawFacts(unit, sha, facts, tuple(ids), stage, cpg_dir, astgen_dir, check_txt, stage_files)


# --- command line -----------------------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m extractors.joern.run --root falkor-chat/server --language python --commit <sha> --work <dir>``.

    **Caveat: the units come from the git index, not from ``--commit``.** ``ccg.units.discover_units`` lists
    ``git ls-files``, while the stage is read from ``--commit``; a unit root added or removed between the two selects
    the wrong exclusion set, and a unit that exists only at ``--commit`` cannot be selected. Run it with the index
    equal to the commit. ``discover_units(commit=...)`` is P1-20's (``docs/plans/connected-code-graph-p1-linking.md``
    §8 (8)); until then ``extract_unit`` is exact because its caller supplies ``units``.
    """
    from ccg import manifest as mf, units as un

    ap = argparse.ArgumentParser(prog="extractors.joern.run", description=__doc__.split("\n")[0],
                                 epilog=CAVEAT)
    ap.add_argument("--repo-root", default=str(REPO), help="directory the manifest's repo paths resolve against")
    ap.add_argument("--manifest", default=str(CPG / "ccg" / "system.yaml"))
    ap.add_argument("--root", required=True, help="unit rootPath")
    ap.add_argument("--language", required=True, choices=sorted(LANGUAGES))
    ap.add_argument("--repo-id", default=None, help="needed only when several repos carry the same root")
    ap.add_argument("--commit", required=True)
    ap.add_argument("--work", required=True, help="a directory below the work area")
    ap.add_argument("--area", default=str(atrest.WORK_ROOT))
    a = ap.parse_args(argv)
    man = mf.load_manifest(a.manifest)
    units = un.discover_units(a.repo_root, man, per_file=False)
    hit = [u for u in units if u.rootPath == a.root and u.language == a.language and a.repo_id in (None, u.repoId)]
    if len(hit) != 1:
        print(f"extract: {len(hit)} units match root={a.root!r} language={a.language!r}", file=sys.stderr)
        return 2
    repo = next(r for r in man.repos if r.repo_id == hit[0].repoId)
    try:
        rf = extract_unit(Path(a.repo_root) / repo.path, hit[0], units, a.commit, a.work, area=a.area)
    except ExtractError as e:
        print(f"extract: FAIL: {e}", file=sys.stderr)
        return 1
    print(f"extract: ok facts={rf.facts} sandboxAuditIds={','.join(rf.sandbox_audit_ids)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
