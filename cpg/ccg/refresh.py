"""The refresh (C-1001 P1-20): one entry point, ``refresh(system, scope=None, override=None) -> RefreshReport``.

Plan: ``docs/plans/connected-code-graph-p1-linking.md`` § P1-20; failure codes ``-p1-graph.md`` §1 G4 (``ccg.staleness.FAIL_REASONS``);
host bundles are ``host-unattested`` (``-bundle-spec6.md``); the broker run ``-egress-spec3.md`` §2.1, §2.5.

The stages, in this order and no other: **sweep** (``atrest.sweep_stale_tmp``) -> **discover** (units from each repository's **commit**, not
the index or the work tree) -> **build** (``build_units``, one unit at a time, code units first so ``known_tools`` exists) -> **resolve** (each
unit's bundle is read through ``atrest.read_bundle``/``bundle.loads`` and gated by ``classify_bundle``; nothing reads a record any other
way) -> **link** (packs, binder, linker) -> **assemble** (``build_rows``) -> **load** -> **swap**. A failure before ``swap`` leaves the live key
untouched: the loader owns build keys only and removes its own.

Per unit: a built and accepted bundle is ``fresh``; a unit whose build failed, or whose bundle the assembler would refuse, keeps its last good
bundle as ``stale-kept`` with ``failedAt`` and the ONE code ``FAIL_REASONS`` names (the build's: this module invents none); a unit with no good
bundle is ``not-mapped`` (its reason is that code too, or ``manifest:never-built`` when nothing was tried), and every link into it is
``possible``/``target-not-mapped``. ``UnknownUnitKind`` is caught per unit, recorded ``extract:failed`` and counted separately
(``RefreshReport.unknown_kind``); it never aborts the refresh. A unit whose bundle carries ``bindParts`` that the system key cannot verify (a
``bind-key-mismatch <unitId>`` line of the linker) becomes ``stale-kept`` with ``bundle:bind-key-check``, its bundle still loaded.

**The head is the local clone's** (``staleness.HEAD_NOTE``): nothing is fetched. A full rebuild is the default. ``scope`` (a set of ``repoId``)
selects which units are **rebuilt**, never which units the graph holds (plan (2a)): every manifest unit is discovered and loaded, an
out-of-scope unit from its last good bundle with its own ``builtCommit`` and its **previous status and failure marker carried** (the
staleness reader shows any commit drift; it cannot show that the last refresh of the unit failed) or ``not-mapped``/``manifest:never-built`` when
it has none, so a scoped refresh never swaps a partial graph over the live key. The marker is read from the work directory, not the live graph
(the build credential holds the live key write-only): after each refresh ``atrest.write_unit_state`` records the status of every unit that
refresh rebuilt (``fresh``, or ``stale-kept`` with ``failedAt`` and its code), and a carried unit reads only its own record. A unit that was
``stale-kept`` therefore stays ``stale-kept`` with the same ``failedAt`` and ``failReason`` across scoped refreshes until it is rebuilt; a rebuild
overwrites the record; a unit with no record is ``fresh``. A record that is unreadable, a link, malformed, names a code outside ``FAIL_REASONS``
or belongs to another build than the bundle carried **fails closed**: the unit is ``stale-kept`` with ``extract:failed`` and the refresh time.
The record is a bind-key-mismatch-blind snapshot of the rebuild only (a mismatch is recomputed on every refresh and never recorded).
Scoped or full, the build stage always sees the whole system: the carried fresh code bundles join ``known_tools`` and ``ctx.units`` is the full
unit list, so a Markdown unit rebuilt in a scoped refresh matches the bytes a full refresh builds. A carried code unit that is ``stale-kept``
adds no tools, as a failed code build adds none in a full refresh.
``override`` (``"local"``/``"hosted"``), only when given, opens a broker ``refresh`` run first and fails with the broker's own error if there
is no broker: there is no fallback, and the run's ``policy`` record id is in the report (the token never is).

Report fields are counts, codes and ids: no source text, no exception text, no key; the few strings a repository controls (a unit's path) are
escaped to printable ASCII and capped in :meth:`RefreshReport.lines`. Memory is **measured** per stage two ways: ``Stage.peak_rss_kb`` is this
Python process's own high-water mark (reset at the stage's start where ``/proc/self/clear_refs`` allows, else the lifetime maximum), and
``Stage.mem_available_min_kb``/``mem_available_drop_kb`` are the **host's** low-water mark of ``MemAvailable`` (``/proc/meminfo``, sampled at
least once a second by a thread) and its fall from the stage's start. The second is what R-4 needs: the Joern JVM runs under ``docker run``,
outside this process and ``RUSAGE_CHILDREN``, and shows up only as a fall of the host's free memory.

``target`` is required: the default connection (the ``build`` and ``verdict`` users from the credentials dir) is **P1-26's** to wire; this
module opens no connection of its own.
"""
from __future__ import annotations

from collections import Counter
import re
import resource
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Iterable, Sequence

from ccg import atrest, bundle, rulepack
from ccg.assemble import NotMappedUnit, build_rows, classify_bundle
from ccg.broker.client import BrokerClient
from ccg.build import CODE_KINDS, BuildContext, BundleResult, UnknownUnitKind, build_unit, build_units
from ccg.linker import Binder, link
from ccg.manifest import Manifest, load_manifest
from ccg.packrt import run_packs
from ccg.sandbox_audit import default_audit_dir
from ccg.staleness import FAIL_REASONS, HEAD_NOTE, Staleness, local_head, unit_staleness
from ccg.store.loader import LoadReport, load
from ccg.store.swap import SwapResult, swap
from ccg.sysname import check_system_name
from ccg.units import Unit, discover_units

CPG = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = CPG / "ccg" / "system.yaml"
#: Every pack ``run_packs`` runs (``paper-inhouse-bus`` has a ``query`` subject it refuses): the plan leaves the set to the caller.
DEFAULT_PACKS = ("fastapi", "fastmcp", "fetch", "infra-signatures", "kafka", "severity-rules", "shell-mcp-call")
OVERRIDES = ("local", "hosted")
STAGES = ("sweep", "discover", "build", "resolve", "link", "assemble", "load", "swap")

FRESH, STALE_KEPT, NOT_MAPPED = "fresh", "stale-kept", "not-mapped"
NEVER_BUILT = "manifest:never-built"
BIND_KEY_CHECK = "bundle:bind-key-check"
_MISMATCH_LINE = re.compile(r"bind-key-mismatch ([0-9a-f]{32})")
_BUILD_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")
MEMINFO = "/proc/meminfo"
SAMPLE_INTERVAL_S = 1.0
_MAX_FIELD = 120


class RefreshError(RuntimeError):
    """The refresh cannot produce a graph (a misconfiguration, or no unit left to load). Value-free. ``report`` is what was known."""

    def __init__(self, message: str, report: "RefreshReport | None" = None):
        super().__init__(message)
        self.report = report


@dataclass(frozen=True)
class Target:
    """The store the refresh loads and swaps into: ``store`` is a ``ccg.store.loader.Store``, the two connections are falkordb-py clients
    authenticated as the system's ``build`` and ``verdict`` users (``ccg.store.swap``). Injected, never opened here."""
    store: object
    build_conn: object
    verdict_conn: object


@dataclass(frozen=True)
class Stage:
    name: str
    seconds: float
    peak_rss_kb: int
    #: the host's lowest ``MemAvailable`` seen during the stage and its fall from the stage's start (None: ``/proc/meminfo`` unreadable)
    mem_available_min_kb: int | None = None
    mem_available_drop_kb: int | None = None


@dataclass
class UnitReport:
    unit_id: str
    repo_id: str
    root_path: str
    language: str
    status: str
    built_commit: str | None = None
    failed_at: str | None = None
    fail_reason: str | None = None
    staleness: Staleness | None = None


@dataclass
class RefreshReport:
    system: str
    head_note: str = HEAD_NOTE
    heads: dict[str, str] = field(default_factory=dict)
    #: the ``repoId`` set whose units were rebuilt (None: every repository); the graph holds every manifest unit either way
    scope: tuple[str, ...] | None = None
    swept: int = 0
    units: list[UnitReport] = field(default_factory=list)
    unknown_kind: int = 0
    skipped_docs: int = 0
    #: paths excluded from unit discovery because they are not valid Unicode (a count only)
    undecodable_paths: int = 0
    pruned: int = 0
    #: retention failures after the swap (a link or a foreign file in the bundle directory): the graph is live, the count is all that is kept
    prune_errors: int = 0
    #: unit-state records that could not be written or cleared after the swap (the graph is live; a lost record reads as ``fresh`` next time)
    state_errors: int = 0
    unattested_units: list[str] = field(default_factory=list)
    bind_key_mismatch: list[str] = field(default_factory=list)
    dropped: dict[str, int] = field(default_factory=dict)
    link_counts: dict = field(default_factory=dict)
    stages: list[Stage] = field(default_factory=list)
    load: LoadReport | None = None
    swap: SwapResult | None = None
    #: ``auditId`` of the broker run's ``policy`` record, when an override opened a run (None otherwise)
    policy_audit_id: str | None = None
    snapshot_id: str | None = None

    def of(self, status: str) -> list[UnitReport]:
        return [u for u in self.units if u.status == status]

    @property
    def fresh(self) -> list[UnitReport]:
        return self.of(FRESH)

    @property
    def stale_kept(self) -> list[UnitReport]:
        return self.of(STALE_KEPT)

    @property
    def not_mapped(self) -> list[UnitReport]:
        return self.of(NOT_MAPPED)

    def lines(self) -> list[str]:
        """The report as text: counts, codes and ids only; every string that comes from a repository is escaped (:func:`safe_text`)."""
        t = safe_text
        out = [f"refresh {t(self.system)}: {len(self.fresh)} fresh, {len(self.stale_kept)} stale-kept, {len(self.not_mapped)} not-mapped "
               f"({t(self.head_note)})",
               f"swept {self.swept} stale temp file(s); unknownKind {self.unknown_kind}; skippedDocs {self.skipped_docs}; undecodablePaths {self.undecodable_paths}; pruned {self.pruned}; "
               f"prune errors {self.prune_errors}; state errors {self.state_errors}"]
        out += [f"{t(u.status)} {t(u.unit_id)} {t(u.language)} {t(u.root_path)} {t(u.fail_reason or '')}".rstrip()
                for u in self.units if u.status != FRESH]
        out += [f"unattested {len(self.unattested_units)}: {', '.join(t(i) for i in self.unattested_units)}" if self.unattested_units
                else "unattested 0"]
        out += [t(line) for line in self.bind_key_mismatch]
        out += [f"dropped {t(k)} {v}" for k, v in sorted(self.dropped.items())]
        if self.policy_audit_id:
            out.append(f"broker policy record {t(self.policy_audit_id)}")
        for s in self.stages:
            mem = ("host MemAvailable n/a" if s.mem_available_min_kb is None
                   else f"host MemAvailable min {s.mem_available_min_kb} KB (drop {s.mem_available_drop_kb} KB)")
            out.append(f"stage {s.name} {s.seconds:.2f}s peak {s.peak_rss_kb} KB (this process); {mem}")
        return out


def safe_text(value: object, cap: int = _MAX_FIELD) -> str:
    """``value`` as one printable-ASCII token for a report line: every control character (newline, ESC, NUL), backslash and non-ASCII
    character is backslash-escaped, so no repository-controlled text can end a line, forge one or drive a terminal; longer text is cut."""
    text = str(value).encode("unicode_escape", "backslashreplace").decode("ascii")
    return text if len(text) <= cap else text[:cap - 3] + "..."


# --- measurement ------------------------------------------------------------------------------------------------------------


def _hwm_reset() -> bool:
    """Reset this process's peak-RSS counter (Linux ``clear_refs`` 5); False where the kernel or the sandbox refuses."""
    try:
        with open("/proc/self/clear_refs", "w") as fh:
            fh.write("5")
        return True
    except OSError:
        return False


def _peak_kb(reset_ok: bool) -> int:
    if reset_ok:
        try:
            with open("/proc/self/status") as fh:
                for line in fh:
                    if line.startswith("VmHWM:"):
                        return int(line.split()[1])
        except (OSError, ValueError, IndexError):
            pass
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _mem_available_kb() -> int | None:
    try:
        with open(MEMINFO) as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return None


class _MemSampler(threading.Thread):
    """Samples the host's ``MemAvailable`` (a sample at the start, one every ``SAMPLE_INTERVAL_S``, one at the end) and keeps the lowest."""

    def __init__(self, interval: float):
        super().__init__(daemon=True)
        self.interval, self.start_kb, self.min_kb = interval, None, None
        self._halt = threading.Event()
        self._sample()
        self.start_kb = self.min_kb

    def _sample(self) -> None:
        kb = _mem_available_kb()
        if kb is not None and (self.min_kb is None or kb < self.min_kb):
            self.min_kb = kb

    def run(self) -> None:
        while not self._halt.wait(self.interval):
            self._sample()

    def finish(self) -> None:
        self._halt.set()
        self.join()
        self._sample()


class _Timed:
    """``with _Timed(report, "build"): ...`` appends one :class:`Stage` (wall time, this process's peak, the host's ``MemAvailable`` low-water
    mark) even when the body raises."""

    def __init__(self, report: RefreshReport, name: str, clock: Callable[[], float]):
        self.report, self.name, self.clock = report, name, clock

    def __enter__(self):
        self._reset = _hwm_reset()
        self._mem = _MemSampler(min(SAMPLE_INTERVAL_S, 1.0))
        self._mem.start()
        self._t0 = self.clock()
        return self

    def __exit__(self, *exc):
        seconds = self.clock() - self._t0
        self._mem.finish()
        low, first = self._mem.min_kb, self._mem.start_kb
        self.report.stages.append(Stage(self.name, seconds, _peak_kb(self._reset), low, None if low is None else first - low))
        return False


# --- units and bundles --------------------------------------------------------------------------------------------------------


def _rfc3339(t: float | None) -> str:
    return datetime.fromtimestamp(time.time() if t is None else t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _gate(system: str, unit_id: str, build: str, root, audit_dir: Path) -> tuple[list[dict] | None, str | None]:
    """``(records, None)`` for a stored bundle the assembler accepts, else ``(None, code)``. Records pass ``atrest.read_bundle`` (the
    custody checks) and ``bundle.loads`` (the validator), then ``classify_bundle``; a bundle whose header names another unit or build is invalid."""
    try:
        records = bundle.loads(atrest.read_bundle(system, unit_id, build, root))
    except (atrest.AtRestError, bundle.BundleError, OSError):
        return None, "bundle:invalid"
    head = records[0]
    if head.get("unitId") != unit_id or head.get("builtCommit") != build:
        return None, "bundle:invalid"
    code = classify_bundle(records, audit_dir)
    return (records, None) if code is None else (None, code)


def _stored_builds(system: str, unit_id: str, root) -> list[str]:
    """The builds with a stored bundle of ``unit_id``, newest file first (a link or a non-regular file is not a bundle)."""
    d = atrest.bundle_dir(system, root)
    found = []
    pre, suf = f"{unit_id}.", atrest.BUNDLE_SUFFIX
    for p in d.iterdir():
        n = p.name
        if not (n.startswith(pre) and n.endswith(suf)) or not _BUILD_TOKEN.fullmatch(n[len(pre):-len(suf)]):
            continue
        st = p.lstat()
        if p.is_symlink() or not p.is_file():
            continue
        found.append((-st.st_mtime_ns, n[len(pre):-len(suf)]))
    return [b for _, b in sorted(found)]


def _code_of(reason: str | None) -> str:
    return reason if reason in FAIL_REASONS else "extract:failed"


def _resolve(system: str, unit: Unit, result: BundleResult | None, root, audit_dir: Path, now: str):
    """``(status, records, builtCommit, failedAt, reason)`` of one unit: its fresh bundle, else its last good one, else nothing."""
    reason = None
    if result is not None and result.ok:
        records, reason = _gate(system, unit.unitId, result.commit, root, audit_dir)
        if records is not None:
            return FRESH, records, result.commit, None, None
    else:
        reason = _code_of(result.fail_reason) if result is not None else None
    for build in _stored_builds(system, unit.unitId, root):          # a refused fresh bundle is gated again here and refused again
        records, _ = _gate(system, unit.unitId, build, root, audit_dir)
        if records is not None:
            return STALE_KEPT, records, build, now, reason or "extract:failed"
    return NOT_MAPPED, None, None, None, reason or NEVER_BUILT


def _carried_marker(system: str, unit_id: str, build: str, root, failed_at: str) -> tuple[str, str | None, str | None]:
    """``(status, failedAt, reason)`` of a carried unit from its own unit-state record: none is ``fresh``; a bad or foreign record fails closed."""
    closed = (STALE_KEPT, failed_at, "extract:failed")
    try:
        state = atrest.read_unit_state(system, unit_id, root)
    except atrest.AtRestError:
        return closed
    if state is None:
        return FRESH, None, None
    if state["builtCommit"] != build or state["failReason"] not in (None, *FAIL_REASONS):
        return closed
    return state["status"], state["failedAt"], state["failReason"]


def _carry(system: str, unit: Unit, root, audit_dir: Path, failed_at: str):
    """An out-of-scope unit (not rebuilt): its last good bundle with its own ``builtCommit`` and its carried status and failure marker
    (:func:`_carried_marker`), else ``not-mapped`` (``manifest:never-built`` when no bundle is stored, the gate's code for the newest
    one when every stored bundle is refused)."""
    refused = None
    for build in _stored_builds(system, unit.unitId, root):
        records, code = _gate(system, unit.unitId, build, root, audit_dir)
        if records is not None:
            status, when, reason = _carried_marker(system, unit.unitId, build, root, failed_at)
            return status, records, build, when, reason
        refused = refused or code
    return NOT_MAPPED, None, None, None, refused or NEVER_BUILT


# --- the entry point ------------------------------------------------------------------------------------------------------------


def _scope_ids(manifest: Manifest, scope: Iterable[str] | None) -> tuple[str, ...] | None:
    """The sorted ``repoId`` tuple to rebuild (None: all); a ``str`` or empty scope is a ``ValueError``, an undeclared id a ``RefreshError``."""
    if scope is None:
        return None
    if isinstance(scope, str):
        raise ValueError("scope is a collection of repoId, not a string")
    wanted = frozenset(scope)
    if not wanted:
        raise ValueError("an empty scope refreshes nothing")
    known = {r.repo_id for r in manifest.repos}
    if wanted - known:
        raise RefreshError(f"scope names {len(wanted - known)} repoId(s) the manifest does not declare")
    return tuple(sorted(wanted))


def refresh(system: str, scope: Iterable[str] | None = None, override: str | None = None, *, target: Target | None = None,
            manifest: str | Path = DEFAULT_MANIFEST, repo_root: str | Path | None = None, work_root: str | Path | None = None,
            keys_root: str | Path | None = None, audit_dir: str | Path | None = None, packs: Sequence[str] = DEFAULT_PACKS,
            broker: BrokerClient | None = None, build: Callable[..., BundleResult] | None = None, built_at: str | None = None,
            staleness_mode: str = "files", sweep_older_than_s: float = 3600, clock: Callable[[], float] = time.monotonic,
            now: float | None = None) -> RefreshReport:
    """Rebuild ``system`` from the local clones at their current ``HEAD``, assemble, load and swap; return the :class:`RefreshReport`.

    ``manifest`` is the system manifest's path (its ``system`` must equal ``system``); ``repo_root`` resolves each repository's ``path``
    (default: the monorepo root). ``target`` is the store (a refresh without one raises :class:`RefreshError`: the default connection
    from the credentials dir is P1-26's). ``build`` replaces ``build_unit`` (a test's stub); ``now`` fixes the time ``failedAt`` and
    the sweep's age cut read."""
    system = check_system_name(system)
    if override is not None and override not in OVERRIDES:
        raise ValueError(f"override is one of {OVERRIDES} or None")
    if target is None:
        raise RefreshError("no store given: refresh loads and swaps into a Target (the credentials-dir default is not wired yet)")
    manifest_path = Path(manifest)
    full = load_manifest(manifest_path)
    if full.system != system:
        raise RefreshError("the manifest's system is not the system being refreshed")
    scope_ids = _scope_ids(full, scope)
    mf = full
    report = RefreshReport(system, scope=scope_ids)
    root = Path(repo_root) if repo_root is not None else CPG.parent
    audit = Path(audit_dir) if audit_dir is not None else default_audit_dir()

    grant = None
    if override is not None:                                       # first: a refused or absent broker stops the refresh before any work
        client = broker or BrokerClient()
        grant = client.open_run("refresh", system, list(scope_ids) if scope_ids else "*", override)
        report.policy_audit_id = grant.audit_id
    try:
        return _run(report, mf, manifest_path, root, target, work_root, keys_root, audit, packs, build, built_at, staleness_mode,
                    sweep_older_than_s, clock, now)
    finally:
        if grant is not None:
            try:
                client.close_run(grant.token)
            except Exception:  # noqa: BLE001 - an unreachable broker must not replace the error (or the report) the run produced
                pass


def _run(report: RefreshReport, mf: Manifest, manifest_path: Path, root: Path, target: Target, work_root, keys_root, audit: Path,
         packs: Sequence[str], build, built_at, staleness_mode: str, sweep_older_than_s: float, clock, now) -> RefreshReport:
    system = report.system
    scope_ids = report.scope
    failed_at = _rfc3339(now)

    with _Timed(report, "sweep", clock):
        report.swept = len(atrest.sweep_stale_tmp(system, sweep_older_than_s, work_root, now))

    repos = {r.repo_id: root / r.path for r in mf.repos}
    with _Timed(report, "discover", clock):
        report.heads = {rid: local_head(path) for rid, path in repos.items()}
        found = Counter()
        units = discover_units(root, mf, commit=report.heads, report=found)
        report.undecodable_paths = found["undecodable_paths"]
    rebuilt = [u for u in units if scope_ids is None or u.repoId in scope_ids]
    in_scope = {u.unitId for u in rebuilt}
    if not units:
        raise RefreshError("the manifest's repositories yield no unit at their heads", report)

    unknown = [0]
    real = build or build_unit

    def one(unit, repo, commit, sys_name, ctx):
        try:
            return real(unit, repo, commit, sys_name, ctx)
        except UnknownUnitKind:                                    # a caller bug, recorded per unit (owner ruling): never aborts the refresh
            unknown[0] += 1
            return BundleResult(unit, commit, False, fail_reason="extract:failed")

    ctx = BuildContext(root=work_root, keys_root=keys_root, built_at=built_at, manifest=manifest_path)
    with _Timed(report, "build", clock):
        carried = {u.unitId: _carry(system, u, work_root, audit, failed_at) for u in units if u.unitId not in in_scope}
        carried_code = [c[1] for u in units if u.language in CODE_KINDS and (c := carried.get(u.unitId)) and c[0] == FRESH]
        built = build_units(rebuilt, repos, report.heads, system, ctx, build=one, carried_code_bundles=carried_code, all_units=units)
    report.unknown_kind, report.skipped_docs = unknown[0], built.skipped_docs
    results = {r.unit.unitId: r for r in built.results}

    kept: dict[str, str] = {}
    records_by_unit: dict[str, list[dict]] = {}
    unit_reports: dict[str, UnitReport] = {}
    not_mapped: list[NotMappedUnit] = []
    stale: dict[str, tuple[str, str]] = {}
    rebuilt_state: dict[str, tuple[str, str, str | None, str | None]] = {}      # what the next scoped refresh carries for a unit rebuilt now
    with _Timed(report, "resolve", clock):
        for u in units:
            status, records, commit, when, reason = (_resolve(system, u, results.get(u.unitId), work_root, audit, failed_at)
                                                     if u.unitId in in_scope else carried[u.unitId])
            ur = UnitReport(u.unitId, u.repoId, u.rootPath, u.language, status, commit, when, reason)
            unit_reports[u.unitId] = ur
            if records is None:
                not_mapped.append(NotMappedUnit(u.unitId, u.repoId, u.rootPath, u.language, reason))
                continue
            records_by_unit[u.unitId] = records
            if u.unitId in in_scope:
                kept[u.unitId] = commit
                rebuilt_state[u.unitId] = (status, commit, when, reason)
            if status == STALE_KEPT:
                stale[u.unitId] = (when, reason)
        report.units = [unit_reports[u.unitId] for u in units]
    if not records_by_unit:
        raise RefreshError("no unit built and none has a last good bundle: nothing to load", report)

    bundles = [records_by_unit[u.unitId] for u in units if u.unitId in records_by_unit]
    pack_dir = CPG / "packs"
    loaded = [rulepack.load_pack(pack_dir / name / "pack.yaml") for name in packs]
    with _Timed(report, "link", clock):
        facts = run_packs(bundles, loaded)
        binder = Binder(facts, bundles, key=atrest.hmac_key(system, keys_root), repos={r.repo_id: r.path for r in mf.repos},
                        packs=loaded, units=units)
        links = link(facts, bundles, packs=loaded, bindings=binder)
    report.link_counts = dict(links.counts)
    report.bind_key_mismatch = list(links.bind_key_mismatch)
    for line in report.bind_key_mismatch:                          # (0): the linker's value-free line becomes that unit's failReason
        m = _MISMATCH_LINE.fullmatch(line)
        ur = unit_reports.get(m[1]) if m else None
        if ur is not None and ur.status == FRESH:
            ur.status, ur.failed_at, ur.fail_reason = STALE_KEPT, failed_at, BIND_KEY_CHECK
            stale[ur.unit_id] = (failed_at, BIND_KEY_CHECK)

    with _Timed(report, "assemble", clock):
        rows = build_rows(bundles, links, facts, system=system, packs=loaded, audit_dir=audit, stale=stale, not_mapped=not_mapped)
    report.unattested_units, report.dropped = list(rows.unattested_units), dict(rows.dropped)
    report.snapshot_id = rows.snapshot["snapshotId"]

    for ur in report.units:
        if ur.built_commit is not None:
            ur.staleness = unit_staleness(SimpleNamespace(rootPath=ur.root_path, builtCommit=ur.built_commit), report.heads[ur.repo_id],
                                          repo=repos[ur.repo_id], mode=staleness_mode)

    del bundles, records_by_unit, facts, binder, links         # the rows are all the load needs (6 GB host)
    with _Timed(report, "load", clock):
        report.load = load(target.store, rows, system, drop_stale=True)
    with _Timed(report, "swap", clock):
        report.swap = swap(target.build_conn, target.verdict_conn, system, rows.snapshot["snapshotId"],
                           timeout=report.load.timeout_max or None)
    # retention (infra README): only after the graph is live, so a failure here is counted, never raised (the report is all the caller has)
    for unit_id, build_id in kept.items():
        try:
            report.pruned += len(atrest.prune_unit(system, unit_id, build_id, work_root))
        except atrest.AtRestError:
            report.prune_errors += 1
    for u in rebuilt:                                              # a rebuilt unit overwrites its own carried state (none: no bundle, no state)
        try:
            if u.unitId in rebuilt_state:
                status, commit, when, reason = rebuilt_state[u.unitId]
                atrest.write_unit_state(system, u.unitId, status, commit, when, reason, work_root)
            else:
                atrest.clear_unit_state(system, u.unitId, work_root)
        except atrest.AtRestError:
            report.state_errors += 1
    if scope_ids is None:                                          # a full refresh knows every live unit: the rest are orphans
        try:
            report.pruned += len(atrest.prune_orphans(system, {u.unitId for u in units}, work_root))
        except atrest.AtRestError:
            report.prune_errors += 1
    return report
