"""Run one SCIP indexer over a unit, through the sandbox runner only (C-1001 P1-15).

Plan: ``docs/plans/connected-code-graph-p1-extraction.md`` § P1-15; contract: ``docs/plans/connected-code-graph-egress-spec3.md``
§7.4 (the closed argv), §7.8, R-E9, R-E10, and ``docs/plans/connected-code-graph-egress-spec4.md`` §1 (the runner's wall-clock
watchdog and this module's outer bound). Every container step is ``cpg/sandbox/offline-run.sh --backend container --mode
offline --frontend scip-typescript|scip-python``; **never** ``--frontend joern`` (script mode bounds a script's arguments,
not what the script executes), and nothing here runs ``npm``, ``pip`` or ``uv`` or starts a container itself.

- **One indexer at a time**, in a staged directory of its own (``<work>/indexer/stage``), never the one a Joern parse
  reads (``extractors.joern.run``'s ``<work>/stage``). The unit's tree is staged by the Joern stager; the
  **operator-provisioned** dependencies (``node_modules``; a venv and the ``env.json`` of ``scip-python
  --environment``) are copied in from ``Provisioned.root`` and arrive read-only through ``/src``.
- **The stager rejects** (``StageRejected``), anywhere in the staged directory, a symlink with an absolute target or one
  that resolves outside it, a hardlink (a regular file with ``st_nlink > 1``), and any device, FIFO or socket. A
  relative link that stays inside is allowed (``node_modules/.bin``).
- **Dependencies absent** (no ``Provisioned``, or the elements the tool needs are not there): nothing is staged or
  run, and the result is ``Degraded`` with the existing registry reason ``headers-unavailable``; the caller builds
  with Joern alone. No ``cpg/`` module provisions anything.
- **Outer bound.** The runner kills its own container after ``SBX_INDEXER_WALL_S`` (1800 s) and exits 124. This module's
  bound is longer by the margin spec4 §1 states, so the inner watchdog fires first: TERM to the runner (its trap
  removes the container), then, after ``kill_grace_s``, KILL to its whole process group (a runner wedged in its cleanup
  ends only by SIGKILL); the group is swept after every TERM, even when the runner left on it. The bound is a blocking
  wait that returns when the runner does, so there is no timer to cancel. An exception in this process
  (``KeyboardInterrupt`` included) runs the same TERM/KILL sequence before it propagates: the runner is never orphaned.
- **Output is capped.** The runner's stdout and stderr (the container's output reaches them) are drained by reader threads
  that keep the first and last ``OUTPUT_KEEP_BYTES`` of each and count the rest, so a hostile indexer cannot grow the
  parent's memory; ``indexer.log`` holds what was kept.
- **Operator provisioning** is found by ``load_provisioned(system, unit_id)`` (``<CCG_PROVISION_DIR>/<system>/<unitId>/``,
  default ``~/.local/share/ccg/provision``; nothing created, ``~/.config/ccg`` refused); the provisioned
  tree is bounded in entries and bytes before it is copied; ``env_json`` and ``lockfile`` are checked against the §7.4
  path grammar before anything is read; the ``.scip`` must be named by a runner audit id.
- **Exit 124 is a timeout failure but is judged by ``durationMs``** of the runner's own audit record: 124 is also what
  a tool that exits 124 itself yields. ``RunnerTimeout.watchdog`` is True iff the duration reached the cap, False if
  it did not, None if the record could not be found; the failure is the same and no artifact is read. Any non-zero
  exit means no artifact (spec3 §7.8): a partial ``.scip`` may remain in the output directory and is never parsed.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import signal
import stat
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from ccg import atrest
from ccg.units import Unit
from extractors.joern.run import RUNNER, RunnerError, StageError, _mkdir_private, descendant_roots, stage_unit, _resolve_commit

#: the runner's constant (``SBX_INDEXER_WALL_S``, a literal in ``offline-run.sh``; ``tests/test_indexers_run.py`` pins the copy)
WALL_S = 1800
MARGIN_S = 60                      # spec4 §1: the runner's cleanup (measured worst case about 40 s) plus headroom
OUTER_SLACK_S = 30                 # the outer bound strictly exceeds WALL_S + MARGIN_S
OUTER_S = WALL_S + MARGIN_S + OUTER_SLACK_S
KILL_GRACE_S = 60
OUTPUT_KEEP_BYTES = 256 * 1024         # per stream: the first and the last this many bytes of the runner's output are kept
MAX_PROVISIONED_ENTRIES = 500_000
MAX_PROVISIONED_BYTES = 8 * 1024 ** 3
AUDIT_ID_RE = re.compile(r"^[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}$")      # the runner's uuid
PROVISION_JSON = "provision.json"
SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
DEGRADED_REASON = "headers-unavailable"        # an existing registry code (``ccg/registry/reasons.yaml``); no new code
FRONTENDS = {"python": "scip-python", "javascript": "scip-typescript"}
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
COMPONENT_RE = re.compile(r"^[A-Za-z0-9._@+-]+$")
IMAGE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/@:-]{0,200}$")


class StageRejected(StageError):
    """The staged directory holds something the stager refuses (spec3 §7.4)."""


class RunnerTimeout(RunnerError):
    """The indexer did not finish: the runner's watchdog fired (exit 124) or the outer bound expired."""

    def __init__(self, message: str, *, watchdog: bool | None, duration_ms: int | None = None):
        super().__init__(message)
        self.watchdog = watchdog
        self.duration_ms = duration_ms


@dataclass(frozen=True)
class Timeouts:
    wall_s: float = WALL_S             # the runner's cap, as this module believes it (judges exit 124)
    outer_s: float = OUTER_S
    kill_grace_s: float = KILL_GRACE_S


@dataclass(frozen=True)
class Provisioned:
    """What the operator placed outside the system. ``root``'s entries are copied to the top of the staged directory
    (``node_modules``, ``venv``, ``env.json``); ``env_json`` names the ``--environment`` file for ``scip-python``;
    ``lockfile`` (relative to the staged directory) is only hashed, for the pre-registration record."""
    root: Path
    env_json: str | None = None
    lockfile: str | None = None


@dataclass(frozen=True)
class Degraded:
    reason: str
    detail: str


@dataclass(frozen=True)
class IndexerRun:
    frontend: str
    scip: Path
    audit_id: str
    stage: Path
    out_dir: Path
    duration_ms: int | None
    env_json_sha256: str | None
    lockfile_sha256: str | None


# --- the §7.4 argv -----------------------------------------------------------------------------------------------


def _path_ok(p: str) -> bool:
    if p == "/src":
        return True
    if not p.startswith("/src/"):
        return False
    rest = p[5:]
    return rest == "" or all(c not in ("", ".", "..") and COMPONENT_RE.fullmatch(c) for c in rest.split("/"))


def tool_argv(frontend: str, *, env_json: str | None = None, project_name: str | None = None,
              project_version: str | None = None) -> list[str]:
    """``toolArgv`` of spec3 §7.4 (+ spec4 §1): ``index --cwd /src`` for ``scip-typescript``; ``index --cwd /src
    --environment /src/<env.json> --project-name <n> --project-version <v>`` for ``scip-python``."""
    if frontend == "scip-typescript":
        if env_json or project_name or project_version:
            raise ValueError("scip-typescript takes only --cwd")
        return ["index", "--cwd", "/src"]
    if frontend == "scip-python":
        if not env_json or not project_name or not project_version:
            raise ValueError("scip-python needs an env.json, a project name and a project version")
        env_path = f"/src/{env_json}"
        if not _path_ok(env_path) or env_json.endswith("/"):
            raise ValueError(f"env.json path {env_json!r} is outside the §7.4 path grammar")
        for what, v in (("project name", project_name), ("project version", project_version)):
            if not NAME_RE.fullmatch(v):
                raise ValueError(f"{what} {v!r} does not match {NAME_RE.pattern}")
        return ["index", "--cwd", "/src", "--environment", env_path, "--project-name", project_name,
                "--project-version", project_version]
    raise ValueError(f"{frontend!r} is not an indexer frontend (only {sorted(FRONTENDS.values())}; never joern)")


def runner_argv(runner: Path, frontend: str, source: Path, output: Path, repo_id: str, argv: Sequence[str]) -> list[str]:
    if frontend not in FRONTENDS.values():
        raise ValueError(f"{frontend!r} is not an indexer frontend (only {sorted(FRONTENDS.values())}; never joern)")
    return [str(runner), "--backend", "container", "--mode", "offline", "--frontend", frontend,
            "--source", str(source), "--output", str(output), "--codebase", repo_id, "--", *argv]


def indexer_image_tag() -> str:
    """The ``indexer`` image's tag (``build.sh --indexer --print-tag``: prints the tag, builds nothing)."""
    out = subprocess.run([str(RUNNER.parent.parent / "infra" / "toolchain" / "build.sh"), "--indexer", "--print-tag"],
                         capture_output=True, check=True, timeout=120, text=True).stdout.strip()
    if not IMAGE_RE.fullmatch(out):
        raise RunnerError(f"build.sh --indexer --print-tag printed {out!r}")
    return out


# --- the stager's rejections -------------------------------------------------------------------------------------


def validate_tree(root: str | os.PathLike, *, max_entries: int | None = None, max_bytes: int | None = None) -> None:
    """Raise ``StageRejected`` for a symlink with an absolute target or one resolving outside ``root``, a hardlink, or a
    device/FIFO/socket anywhere below ``root``, and (when given) for more than ``max_entries`` entries or ``max_bytes``
    of regular-file content. Nothing is opened, so a FIFO cannot hang the walk."""
    root_s = os.path.abspath(root)
    if os.path.islink(root_s) or not os.path.isdir(root_s):
        raise StageRejected(f"{root_s!r} is not a plain directory")
    real_root = os.path.realpath(root_s)

    def inside(p: str, base: str) -> bool:
        return p == base or p.startswith(base + os.sep)

    entries = total = 0
    for dirpath, dirs, files in os.walk(root_s, followlinks=False):
        for name in dirs + files:
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, root_s)
            st = os.lstat(full)
            entries += 1
            if max_entries is not None and entries > max_entries:
                raise StageRejected(f"more than {max_entries} entries (the provisioned-tree cap)")
            if stat.S_ISLNK(st.st_mode):
                target = os.readlink(full)
                if os.path.isabs(target):
                    raise StageRejected(f"{rel!r}: symlink with an absolute target")
                lexical = os.path.normpath(os.path.join(dirpath, target))
                if not inside(lexical, root_s) or not inside(os.path.realpath(full), real_root):
                    raise StageRejected(f"{rel!r}: symlink leaving the directory")
            elif stat.S_ISREG(st.st_mode):
                if st.st_nlink > 1:
                    raise StageRejected(f"{rel!r}: hardlink (st_nlink {st.st_nlink})")
                total += st.st_size
                if max_bytes is not None and total > max_bytes:
                    raise StageRejected(f"more than {max_bytes} bytes (the provisioned-tree cap)")
            elif not stat.S_ISDIR(st.st_mode):
                raise StageRejected(f"{rel!r}: device, FIFO or socket file")


def _overlay(provisioned: Provisioned, stage: Path) -> None:
    root = provisioned.root
    validate_tree(root, max_entries=MAX_PROVISIONED_ENTRIES, max_bytes=MAX_PROVISIONED_BYTES)   # before the copy: a copy turns a hardlink into a plain file
    for entry in sorted(os.listdir(root)):
        if entry == PROVISION_JSON:                          # the operator's note to cpg/, not a dependency
            continue
        dst = stage / entry
        if os.path.lexists(dst):
            raise StageRejected(f"provisioned {entry!r} collides with a path of the unit's own tree")
        src = root / entry
        if src.is_symlink() or src.is_file():
            shutil.copy2(src, dst, follow_symlinks=False)
        else:
            shutil.copytree(src, dst, symlinks=True)


def check_provisioned_names(provisioned: Provisioned) -> None:
    """``env_json`` and ``lockfile`` are paths inside ``/src``: the §7.4 component grammar, checked before anything is
    hashed or read (``ValueError``)."""
    for what, v in (("env_json", provisioned.env_json), ("lockfile", provisioned.lockfile)):
        if v is not None and (v.endswith("/") or not _path_ok(f"/src/{v}") or v == ""):
            raise ValueError(f"{what} {v!r} is outside the §7.4 path grammar")


def provision_base() -> Path:
    """``$CCG_PROVISION_DIR`` (absolute), else ``~/.local/share/ccg/provision``."""
    env = os.environ.get("CCG_PROVISION_DIR")
    if env:
        if not os.path.isabs(env):
            raise ValueError("CCG_PROVISION_DIR must be an absolute path")
        return Path(env)
    return Path.home() / ".local" / "share" / "ccg" / "provision"


def load_provisioned(system: str, unit_id: str) -> Provisioned | None:
    """The operator's dependencies for one unit, or None ("not run: dependencies not provisioned", reported to ``teco``;
    ``cpg/`` never provisions). The directory is ``<provision_base()>/<system>/<unitId>/`` with an optional
    ``provision.json`` ``{"envJson": "env.json", "lockfile": "package-lock.json"}``. ``system`` and ``unit_id`` are single
    safe components, nothing is created or written, and a
    location under ``~/.config/ccg`` (HMAC keys, ACL credentials) is refused: a ``..`` in the base, then the normalised
    path, then the real path (``realpath`` only reads symlinks on the way; it never opens a file there)."""
    for what, v in (("system", system), ("unit id", unit_id)):
        if not SAFE_NAME_RE.fullmatch(v):
            raise ValueError(f"{what} {v!r} is not a single safe path component")
    base = provision_base()
    if ".." in base.parts:
        raise ValueError(f"the provision location {base} has a '..' component")
    root = base / system / unit_id
    keys = Path.home() / ".config" / "ccg"
    for form in (os.path.normpath, os.path.realpath):        # lexical first, then through every symlink
        r, k = Path(form(root)), Path(form(keys))
        if r == k or k in r.parents:
            raise ValueError(f"{root} is under {keys}, which holds keys and credentials and is never read here")
    try:
        if root.is_symlink() or not root.is_dir():
            return None
    except OSError:
        return None
    env_json, lockfile = "env.json", None
    cfg = root / PROVISION_JSON
    if cfg.is_symlink():
        raise ValueError("provision.json must not be a symlink")
    if cfg.is_file():
        if cfg.stat().st_size > 65536:
            raise ValueError("provision.json is over 64 KiB")
        try:
            doc = json.loads(cfg.read_text("utf-8"))
        except (ValueError, UnicodeDecodeError) as e:
            raise ValueError(f"provision.json is not valid JSON: {e}") from e
        if not isinstance(doc, dict) or set(doc) - {"envJson", "lockfile"} or not all(isinstance(v, str) for v in doc.values()):
            raise ValueError('provision.json must be an object of strings with the keys "envJson" and "lockfile" only')
        env_json, lockfile = doc.get("envJson", env_json), doc.get("lockfile")
    prov = Provisioned(root, env_json=env_json, lockfile=lockfile)
    check_provisioned_names(prov)
    return prov


def _missing(frontend: str, provisioned: Provisioned | None) -> str | None:
    if provisioned is None or not provisioned.root.is_dir():
        return "no operator-provisioned dependency directory"
    if frontend == "scip-typescript":
        return None if (provisioned.root / "node_modules").is_dir() else "no node_modules in the provisioned directory"
    env = provisioned.env_json
    if not env or not COMPONENT_RE.fullmatch(env.split("/")[-1]) or not (provisioned.root / env).is_file():
        return "no env.json (the scip-python --environment file) in the provisioned directory"
    return None


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --- running -----------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Completed:
    rc: int
    stdout: bytes
    stderr: bytes


class _Capped(threading.Thread):
    """Drain one pipe, keeping only its first and last ``keep`` bytes, so a chatty or hostile runner costs no memory."""

    def __init__(self, stream, keep: int):
        super().__init__(daemon=True)
        self.stream, self.keep = stream, keep
        self.head, self.tail, self.total = bytearray(), b"", 0

    def run(self) -> None:
        try:
            while True:
                chunk = self.stream.read1(65536)
                if not chunk:
                    return
                if len(self.head) < self.keep:
                    self.head += chunk[: self.keep - len(self.head)]
                self.tail = (self.tail + chunk)[-self.keep:]
                self.total += len(chunk)
        except (OSError, ValueError):
            return

    def result(self) -> bytes:
        if self.total <= self.keep:
            return bytes(self.head)
        if self.total <= 2 * self.keep:
            return bytes(self.head) + self.tail[-(self.total - self.keep):]
        return bytes(self.head) + f"\n...[{self.total - 2 * self.keep} bytes dropped]...\n".encode() + self.tail


def _end(p: subprocess.Popen, kill_grace_s: float) -> None:
    """TERM to the runner (its trap removes the container), KILL to its whole process group after the grace, and the
    group is swept even when the runner left on TERM: a member that holds no pipe (the watchdog timer, a Docker client)
    must not outlive the call."""
    if p.poll() is None:
        p.send_signal(signal.SIGTERM)
        try:
            p.wait(timeout=kill_grace_s)
        except subprocess.TimeoutExpired:
            pass
    _killpg(p.pid)
    p.wait()


def _run(argv: Sequence[str], env: dict[str, str], t: Timeouts, keep: int = OUTPUT_KEEP_BYTES) -> Completed:
    """Run the runner under the outer bound. On expiry TERM, then (``kill_grace_s`` later) KILL to its process group
    (``RunnerTimeout``, ``watchdog=None``); on any exception in this process (``KeyboardInterrupt`` included) the same
    sequence runs before the exception propagates, so the runner is never orphaned. Output is drained into capped buffers."""
    try:
        p = subprocess.Popen(list(argv), stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL,
                             env=env, start_new_session=True)
    except OSError as e:
        raise RunnerError(f"runner {argv[0]} could not be started: {e}") from e
    readers = [_Capped(p.stdout, keep), _Capped(p.stderr, keep)]
    for r in readers:
        r.start()
    try:
        try:
            p.wait(timeout=t.outer_s)
            _killpg(p.pid)                                   # a descendant that outlived the runner must not hold the pipes
        except subprocess.TimeoutExpired:
            _end(p, t.kill_grace_s)
            raise RunnerTimeout(f"runner did not finish within the outer bound of {t.outer_s}s (TERM, then KILL after "
                                f"{t.kill_grace_s}s)", watchdog=None) from None
    except BaseException:
        _end(p, t.kill_grace_s)
        raise
    finally:
        for r in readers:
            r.join(timeout=5)
        for r, pipe in zip(readers, (p.stdout, p.stderr)):
            if not r.is_alive():                             # closing a pipe a live reader is blocked on would hang
                pipe.close()
    return Completed(p.returncode, readers[0].result(), readers[1].result())


def _killpg(pgid: int) -> None:
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def _audit_dir() -> Path:
    return Path(os.environ.get("CCG_AUDIT_DIR") or RUNNER.parent.parent / ".audit")


def _records_since(path: Path, offset: int) -> list[dict]:
    try:
        with open(path, "rb") as f:
            f.seek(offset)
            data = f.read(1 << 22)
    except OSError:
        return []
    out = []
    for line in data.splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def _my_record(recs: Iterable[dict], frontend: str, stage: Path) -> dict | None:
    for r in reversed(list(recs)):
        if r.get("frontend") == frontend and any(m.get("target") == "/src" and m.get("source") == str(stage)
                                                for m in r.get("mounts", [])):
            return r
    return None


def _new_scip(out_dir: Path, before: set[str]) -> Path:
    new = sorted(n for n in set(os.listdir(out_dir)) - before if n.endswith(".scip"))
    if len(new) != 1:
        raise RunnerError(f"expected exactly one new *.scip in {out_dir}, found {new}")
    return out_dir / new[0]


def run_indexer(repo: str | Path, unit: Unit, units: Iterable[Unit], commit: str, work: str | Path,
                provisioned: Provisioned | None, *, image: str, project_name: str | None = None,
                area: str | Path = atrest.WORK_ROOT, runner: str | Path = RUNNER,
                timeouts: Timeouts = Timeouts()) -> IndexerRun | Degraded:
    """Stage ``unit`` at ``commit`` with ``provisioned``'s dependencies and run its language's indexer once.

    Returns ``Degraded`` (nothing staged, nothing run) when the dependencies are not there. ``image`` is the
    ``indexer`` image (``indexer_image_tag()``), passed to the runner as ``CCG_TOOLCHAIN_IMAGE``. Raises
    ``RunnerTimeout`` or ``RunnerError`` on any failure; a failed run leaves no result to use.
    """
    repo, work, area, runner = Path(repo), Path(work), Path(area), Path(runner)
    frontend = FRONTENDS.get(unit.language)
    if frontend is None:
        raise StageError(f"language {unit.language!r} has no SCIP indexer in P1")
    if not IMAGE_RE.fullmatch(image):
        raise ValueError(f"image {image!r} is not a plain image reference")
    if runner == RUNNER and (timeouts.wall_s != WALL_S or timeouts.outer_s <= WALL_S + MARGIN_S):
        raise ValueError(f"the outer bound must exceed SBX_INDEXER_WALL_S ({WALL_S}) + {MARGIN_S}s (spec4 §1)")
    if provisioned is not None:
        check_provisioned_names(provisioned)
    why = _missing(frontend, provisioned)
    if why is not None:
        return Degraded(DEGRADED_REASON, f"{unit.repoId}:{unit.rootPath} ({unit.language}): {why}; Joern runs alone")
    assert provisioned is not None
    sha = _resolve_commit(repo, commit)
    base = work / "indexer"
    _mkdir_private(base, area)
    stage, out_dir = base / "stage", base / "out"
    for d in (stage, out_dir):
        if d.exists():
            shutil.rmtree(d)
    stage_unit(repo, sha, unit, descendant_roots(unit, units), stage)
    _overlay(provisioned, stage)
    validate_tree(stage)
    _mkdir_private(out_dir, area)
    env_sha = _sha256(stage / provisioned.env_json) if provisioned.env_json and (stage / provisioned.env_json).is_file() else None
    lock = stage / provisioned.lockfile if provisioned.lockfile else None
    lock_sha = _sha256(lock) if lock is not None and lock.is_file() else None
    name = project_name if project_name is not None else re.sub(r"[^A-Za-z0-9._-]", "-", unit.repoId).lstrip("-._")[:64] or "unit"
    argv = tool_argv(frontend, env_json=provisioned.env_json if frontend == "scip-python" else None,
                     project_name=name if frontend == "scip-python" else None,
                     project_version=sha if frontend == "scip-python" else None)
    full = runner_argv(runner, frontend, stage, out_dir, unit.repoId, argv)
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "CCG_TOOLCHAIN_IMAGE": image}
    if os.environ.get("CCG_AUDIT_DIR"):
        env["CCG_AUDIT_DIR"] = os.environ["CCG_AUDIT_DIR"]
    sandbox_jsonl = _audit_dir() / "sandbox.jsonl"
    offset = sandbox_jsonl.stat().st_size if sandbox_jsonl.exists() else 0
    before = set(os.listdir(out_dir))
    c = _run(full, env, timeouts)
    (base / "indexer.log").write_bytes(c.stdout + b"\n--stderr--\n" + c.stderr)
    rec = _my_record(_records_since(sandbox_jsonl, offset), frontend, stage)
    duration = rec.get("durationMs") if rec else None
    if c.rc == 124:
        fired = None if duration is None else duration >= timeouts.wall_s * 1000
        how = {True: "the runner's watchdog fired", False: "the tool itself exited 124 (below the cap)",
               None: "the cause is unknown (no audit record found)"}[fired]
        raise RunnerTimeout(f"unit {unit.repoId}:{unit.rootPath}: {frontend} exited 124, durationMs={duration}: {how}; "
                            f"no artifact is used (see {base / 'indexer.log'})", watchdog=fired, duration_ms=duration)
    if c.rc != 0:
        raise RunnerError(f"unit {unit.repoId}:{unit.rootPath}: {frontend} exited {c.rc}; no artifact is used "
                          f"(see {base / 'indexer.log'})")
    scip = _new_scip(out_dir, before)
    audit_id = scip.name[: -len(".scip")]
    if not AUDIT_ID_RE.fullmatch(audit_id):
        raise RunnerError(f"the .scip in {out_dir} is not named by a runner audit id; no artifact is used")
    return IndexerRun(frontend, scip, audit_id, stage, out_dir, duration, env_sha, lock_sha)
