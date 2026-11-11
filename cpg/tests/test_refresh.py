"""P1-20 — ``ccg.refresh.refresh``, offline.

Real extractors run over small temporary git repositories of host files (shell, config, Markdown): no Joern, no Docker, no network, no database.
The store is ``loader_helpers.FakeStore`` plus the fake swap connections; the broker is a recording fake or an absent socket. Nothing reads or
writes ``~/.config/ccg`` (keys and the work area live under ``tmp_path``; ``tests/conftest.py`` makes a test that forgets that fail, never write) or touches ``falkordb-dev``/``ccg-falkordb``.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from loader_helpers import EMPTY_HASH, Boom, FakeStore, fail_at, fake_pair

from ccg import atrest, ids, sysname
from ccg import refresh as RF
from ccg.assemble import AssembleError
from ccg.broker.client import BrokerClient, BrokerError, BrokerUnavailable
from ccg.build import BundleResult
from ccg.staleness import FAIL_REASONS
from ccg.store import loader as L
from ccg.store import swap as SW
from ccg.units import Unit

SYSTEM = "rfs"
NOW = 1_800_000_000.0
NOW_Z = "2027-01-15T08:00:00Z"
ENV = {"GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull, "PATH": os.environ["PATH"], "GIT_AUTHOR_NAME": "t",
       "GIT_AUTHOR_EMAIL": "t@example.invalid", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid"}

MANIFEST = """schemaVersion: 1
system: {system}
egress: {{schemaVersion: 1, local: [], outbound: [], hosted: []}}
promptBearingMarkdown:
  include: ["**/AGENTS.md"]
repos:
{repos}
"""
FILES = {"bin/run.sh": "#!/bin/sh\necho hi\n", "conf/app.yaml": "name: app\nport: 8080\n", "docs/note.md": "# Note\n\nplain text\n"}


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True, env=ENV).stdout.strip()


def commit(repo: Path, files: dict[str, str | None], msg: str = "c") -> str:
    if not (repo / ".git").exists():
        repo.mkdir(parents=True, exist_ok=True)
        git(repo, "init", "-q", "-b", "main")
    for rel, text in files.items():
        p = repo / rel
        if text is None:
            p.unlink()
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "--allow-empty", "-m", msg)
    return git(repo, "rev-parse", "HEAD")


class World:
    def __init__(self, tmp: Path, repos=("r1",), files=None, system: str = SYSTEM):
        self.root, self.work, self.keys, self.audit = tmp / "root", tmp / ".ccg-work", tmp / "keys", tmp / "audit"
        self.work.mkdir(mode=0o700)
        self.audit.mkdir()
        self.repos = repos
        for r in repos:
            commit(self.root / r, dict(FILES if files is None else files))
        lines = "\n".join(f"  - {{repoId: {r}, path: {r}, policy: {{egress: local-only}}}}" for r in repos)
        self.manifest = tmp / "system.yaml"
        self.manifest.write_text(MANIFEST.format(system=system, repos=lines))

    def repo(self, name="r1") -> Path:
        return self.root / name

    def kw(self, target, **extra):
        return {**dict(target=target, manifest=self.manifest, repo_root=self.root, work_root=self.work, keys_root=self.keys,
                       audit_dir=self.audit, built_at="2027-01-01T00:00:00Z", now=NOW), **extra}


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


class Rig(FakeStore):
    """A ``FakeStore`` that also tells the fake swap connections about the build key once its ``:Snapshot`` is written."""

    def __init__(self, shared, stamps, **kw):
        super().__init__(**kw)
        self.shared, self.stamps = shared, stamps

    def query(self, key, cypher, params=None, timeout=None):
        res = super().query(key, cypher, params, timeout)
        if cypher == L.SNAPSHOT_STATEMENT:
            self.shared[key] = "new"
            self.stamps[key] = (0, EMPTY_HASH)
        return res


def make_target(**store_kw):
    bdb, vdb, log, shared, stamps = fake_pair(SW.RELEASE_LUA, keys={sysname.live_key(SYSTEM): "old"})
    stamps[sysname.live_key(SYSTEM)] = (0, EMPTY_HASH)
    store = Rig(shared, stamps, **store_kw)
    return RF.Target(store, bdb, vdb), SimpleNamespace(log=log, shared=shared, store=store, bdb=bdb, vdb=vdb)


def renames(fakes):
    return [e for e in fakes.log if e[1] == "rename"]


@pytest.fixture
def spies(monkeypatch):
    """Records ``sweep, build_units, build_rows, load, swap`` (in call order) and keeps the rows the loader was given."""
    calls, keep = [], SimpleNamespace(rows=None, units_seen=None)

    def wrap(mod, attr, label):
        real = getattr(mod, attr)

        def spy(*a, **k):
            calls.append(label)
            if label == "load":
                keep.rows = a[1]
            return real(*a, **k)

        monkeypatch.setattr(mod, attr, spy)

    wrap(RF.atrest, "sweep_stale_tmp", "sweep")
    for attr, label in (("build_units", "build_units"), ("build_rows", "build_rows"), ("load", "load"), ("swap", "swap")):
        wrap(RF, attr, label)
    keep.calls = calls
    return keep


def unit_rows(rows):
    return {u["unitId"]: u for u in rows.nodes["Unit"]}


def failing(*unit_ids_codes, real=None):
    """A ``build`` stub: the named units fail with the given code, every other unit is built by the real ``build_unit``."""
    from ccg import build as B
    plan = dict(unit_ids_codes)
    seen = []

    def build(unit, repo, commit, system, ctx):
        seen.append(unit.unitId)
        if unit.unitId in plan:
            return BundleResult(unit, commit, False, fail_reason=plan[unit.unitId])
        return (real or B.build_unit)(unit, repo, commit, system, ctx)

    build.seen = seen
    return build


def uid(root: str, lang: str, repo="r1") -> str:
    return ids.unit_id(repo, root, lang)


ALL_UNITS = [("bin/run.sh", "shell"), ("conf/app.yaml", "config"), ("docs/note.md", "markdown")]


# --- the happy path ----------------------------------------------------------------------------------------------------------


def test_a_full_refresh_builds_every_unit_fresh_loads_and_swaps(world):
    target, fakes = make_target()
    rep = RF.refresh(SYSTEM, **world.kw(target))
    assert sorted(u.unit_id for u in rep.fresh) == sorted(uid(*u) for u in ALL_UNITS)
    assert not rep.stale_kept and not rep.not_mapped and rep.unknown_kind == 0 and rep.swept == 0
    assert rep.heads == {"r1": git(world.repo(), "rev-parse", "HEAD")} and rep.head_note == "head is the local clone's"
    assert all(u.built_commit == rep.heads["r1"] and u.staleness.stale is False for u in rep.units)
    assert rep.swap.renamed and len(renames(fakes)) == 1 and fakes.shared[sysname.live_key(SYSTEM)] == "new"
    assert rep.load.build_key == sysname.build_key(SYSTEM, rep.snapshot_id) and rep.snapshot_id == rep.load.snapshot_id
    assert sysname.build_key(SYSTEM, rep.snapshot_id) not in fakes.shared                       # renamed onto live


def test_the_report_has_wall_time_a_peak_and_the_hosts_memory_low_water_mark_for_every_stage_in_order(world):
    rep = RF.refresh(SYSTEM, **world.kw(make_target()[0]))
    assert [s.name for s in rep.stages] == list(RF.STAGES)
    assert all(s.seconds >= 0 and s.peak_rss_kb > 0 and s.mem_available_min_kb > 0 and 0 <= s.mem_available_drop_kb for s in rep.stages)
    assert not hasattr(rep.stages[0], "children_peak_rss_kb")
    text = "\n".join(rep.lines())
    assert "head is the local clone's" in text and all(f"stage {n} " in text for n in RF.STAGES) and "host MemAvailable min" in text


def _meminfo(path, kb):
    path.write_text(f"MemTotal:  8000000 kB\nMemFree: 1 kB\nMemAvailable: {kb} kB\nBuffers: 5 kB\n")


def test_a_stage_reports_the_lowest_host_memory_available_seen_and_the_fall_from_its_start(tmp_path, monkeypatch):
    fake = tmp_path / "meminfo"
    _meminfo(fake, 5000)
    monkeypatch.setattr(RF, "MEMINFO", str(fake))
    monkeypatch.setattr(RF, "SAMPLE_INTERVAL_S", 0.005)
    report = RF.RefreshReport("s")
    with RF._Timed(report, "build", time.monotonic):
        _meminfo(fake, 3000)                                   # the JVM takes memory ...
        time.sleep(0.1)
        _meminfo(fake, 4200)                                   # ... and gives some back
        time.sleep(0.05)
    (st,) = report.stages
    assert (st.mem_available_min_kb, st.mem_available_drop_kb) == (3000, 2000)
    with RF._Timed(report, "link", time.monotonic):
        _meminfo(fake, 9000)                                   # memory only grew: the drop is 0, the minimum is the start
        time.sleep(0.05)
    assert (report.stages[1].mem_available_min_kb, report.stages[1].mem_available_drop_kb) == (4200, 0)
    line = [ln for ln in report.lines() if ln.startswith("stage build")][0]
    assert "host MemAvailable min 3000 KB (drop 2000 KB)" in line and "(this process)" in line


def test_an_unreadable_meminfo_is_reported_as_unavailable_not_as_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(RF, "MEMINFO", str(tmp_path / "absent"))
    report = RF.RefreshReport("s")
    with RF._Timed(report, "sweep", time.monotonic):
        pass
    st = report.stages[0]
    assert st.mem_available_min_kb is None and st.mem_available_drop_kb is None
    assert "host MemAvailable n/a" in "\n".join(report.lines())


def test_a_stage_peak_is_this_processs_own_and_follows_an_allocation_made_inside_it(tmp_path):
    if not RF._hwm_reset():
        pytest.skip("the kernel refuses /proc/self/clear_refs here")
    report = RF.RefreshReport("s")
    with RF._Timed(report, "big", time.monotonic):
        blob = bytearray(150 * 1024 * 1024)
        blob[::4096] = b"x" * len(blob[::4096])                 # touch every page
    del blob
    with RF._Timed(report, "small", time.monotonic):
        pass
    big, small = report.stages
    assert big.peak_rss_kb - small.peak_rss_kb > 100 * 1024


def test_stages_run_in_the_documented_order(world, spies):
    RF.refresh(SYSTEM, **world.kw(make_target()[0]))
    assert spies.calls == ["sweep", "build_units", "build_rows", "load", "swap"]


def test_the_assembly_reports_unattested_host_units_in_the_refresh_report(world):
    rep = RF.refresh(SYSTEM, **world.kw(make_target()[0]))
    assert rep.unattested_units == sorted(uid(*u) for u in ALL_UNITS)
    assert rep.load.unattested_units == rep.unattested_units and f"unattested 3: {', '.join(rep.unattested_units)}" in rep.lines()


def test_a_refresh_without_a_store_or_for_another_systems_manifest_is_refused(world):
    with pytest.raises(RF.RefreshError, match="no store"):
        RF.refresh(SYSTEM, **{k: v for k, v in world.kw(None).items() if k != "target"})
    with pytest.raises(RF.RefreshError, match="manifest's system"):
        RF.refresh("another", **world.kw(make_target()[0]))


# --- (8) units come from the commit ----------------------------------------------------------------------------------------


def test_units_are_discovered_from_the_commit_not_the_index_or_the_work_tree(world):
    repo = world.repo()
    (repo / "bin/run.sh").unlink()                                               # deleted in the work tree only
    (repo / "staged.sh").write_text("#!/bin/sh\n")
    git(repo, "add", "staged.sh")                                                # staged, never committed
    (repo / "loose.sh").write_text("#!/bin/sh\n")                                # untracked
    (repo / "conf/app.yaml").write_text("changed: true\n")                       # modified after the commit
    rep = RF.refresh(SYSTEM, **world.kw(make_target()[0]))
    assert sorted((u.root_path, u.language) for u in rep.units) == sorted(ALL_UNITS)
    assert len(rep.fresh) == 3
    from ccg import bundle
    cfg = bundle.loads(atrest.read_bundle(SYSTEM, uid("conf/app.yaml", "config"), rep.heads["r1"], world.work))
    files = [r for r in cfg if r["type"] == "file"]
    import hashlib
    assert files[0]["contentHash"] == hashlib.sha256(FILES["conf/app.yaml"].encode()).hexdigest()   # the committed bytes, not the edit


# --- (1) stale-kept, not-mapped -----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("code", ["build:shallow-clone", "extract:failed", "extract:timeout", "bundle:invalid"])
def test_a_failed_unit_keeps_its_last_good_bundle_marked_stale_kept_with_the_builds_code(world, spies, code):
    first = RF.refresh(SYSTEM, **world.kw(make_target()[0]))
    c1 = first.heads["r1"]
    c2 = commit(world.repo(), {"conf/app.yaml": "name: app\nport: 9090\n", "bin/run.sh": "#!/bin/sh\necho two\n"})
    victim = uid("conf/app.yaml", "config")
    target, fakes = make_target()
    rep = RF.refresh(SYSTEM, **world.kw(target, build=failing((victim, code))))
    by = {u.unit_id: u for u in rep.units}
    v = by[victim]
    assert (v.status, v.built_commit, v.failed_at, v.fail_reason) == ("stale-kept", c1, NOW_Z, code)
    assert v.staleness.stale is True and v.staleness.files_changed == 1 and v.staleness.head == c2
    assert by[uid("bin/run.sh", "shell")].status == "fresh" and by[uid("bin/run.sh", "shell")].built_commit == c2
    row = unit_rows(spies.rows)[victim]
    assert (row["buildStatus"], row["builtCommit"], row["failedAt"], row["failReason"]) == ("stale-kept", c1, NOW_Z, code)
    assert row["failReason"] in FAIL_REASONS and unit_rows(spies.rows)[uid("bin/run.sh", "shell")]["buildStatus"] == "fresh"
    d = atrest.bundle_dir(SYSTEM, world.work)
    assert (d / f"{victim}.{c1}.bundle.jsonl").is_file() and not (d / f"{victim}.{c2}.bundle.jsonl").exists()     # the last good one is retained
    other = uid("bin/run.sh", "shell")
    assert (d / f"{other}.{c2}.bundle.jsonl").is_file() and not (d / f"{other}.{c1}.bundle.jsonl").exists()       # a fresh unit keeps only its new one


def test_the_failure_code_is_the_builds_never_text_it_carries(world):
    RF.refresh(SYSTEM, **world.kw(make_target()[0]))
    commit(world.repo(), {"conf/app.yaml": "name: app\nport: 9090\n"})
    victim = uid("conf/app.yaml", "config")
    rep = RF.refresh(SYSTEM, **world.kw(make_target()[0], build=failing((victim, "boom at /home/x/secret.txt: token=abc"))))
    assert {u.unit_id: u.fail_reason for u in rep.stale_kept} == {victim: "extract:failed"}
    assert "secret" not in "\n".join(rep.lines()) and "token" not in "\n".join(rep.lines())


def test_a_unit_with_no_last_good_bundle_is_not_mapped_with_the_builds_code_and_the_others_still_load(world, spies):
    victim = uid("bin/run.sh", "shell")
    rep = RF.refresh(SYSTEM, **world.kw(make_target()[0], build=failing((victim, "extract:timeout"))))
    assert [(u.unit_id, u.status, u.fail_reason, u.built_commit) for u in rep.not_mapped] == [(victim, "not-mapped", "extract:timeout", None)]
    assert len(rep.fresh) == 2
    row = unit_rows(spies.rows)[victim]
    assert row["buildStatus"] == "not-mapped" and row["failReason"] == "extract:timeout" and "builtCommit" not in row


def test_a_real_unbuildable_unit_fails_alone(world):
    commit(world.repo(), {"bad\\name.sh": "#!/bin/sh\n"})                         # a backslash in a tree path: the build refuses it
    rep = RF.refresh(SYSTEM, **world.kw(make_target()[0]))
    assert [(u.root_path, u.status, u.fail_reason) for u in rep.not_mapped] == [("bad\\name.sh", "not-mapped", "extract:failed")]
    assert len(rep.fresh) == 3


def test_a_unit_the_build_never_reported_is_not_mapped_as_never_built(world, monkeypatch, spies):
    real = RF.build_units

    def drop_one(units, repos, commits, system, ctx=None, *, build=None):
        rep = real(units, repos, commits, system, ctx, build=build)
        return rep.__class__(tuple(r for r in rep.results if r.unit.unitId != uid("docs/note.md", "markdown")), rep.known_tools, rep.skipped_docs)

    monkeypatch.setattr(RF, "build_units", drop_one)
    rep = RF.refresh(SYSTEM, **world.kw(make_target()[0], build=failing((uid("docs/note.md", "markdown"), "extract:failed"))))
    assert [(u.unit_id, u.fail_reason) for u in rep.not_mapped] == [(uid("docs/note.md", "markdown"), "manifest:never-built")]


def test_a_never_built_unit_reaches_the_binder_so_a_link_into_it_degrades(world, monkeypatch):
    seen = {}
    real = RF.Binder

    def spy(facts, bundles, **kw):
        seen["units"] = list(kw["units"])
        seen["built"] = {b[0]["unitId"] for b in bundles}
        return real(facts, bundles, **kw)

    monkeypatch.setattr(RF, "Binder", spy)
    victim = uid("bin/run.sh", "shell")
    RF.refresh(SYSTEM, **world.kw(make_target()[0], build=failing((victim, "extract:failed"))))
    assert victim in {u.unitId for u in seen["units"]} and victim not in seen["built"] and len(seen["units"]) == 3


def test_every_unit_failing_with_nothing_kept_raises_with_the_report_and_loads_nothing(world, spies):
    codes = [(uid(*u), "extract:failed") for u in ALL_UNITS]
    target, fakes = make_target()
    with pytest.raises(RF.RefreshError, match="nothing to load") as ei:
        RF.refresh(SYSTEM, **world.kw(target, build=failing(*codes)))
    assert len(ei.value.report.not_mapped) == 3 and "load" not in spies.calls and "swap" not in spies.calls
    assert fakes.store.calls == [] and fakes.shared == {sysname.live_key(SYSTEM): "old"}


# --- (7) a shallow clone is a build error --------------------------------------------------------------------------------------


def _shallow(world, name="r1"):
    repo = world.repo(name)
    commit(repo, {"extra.txt": "x"}, "second")
    clone = world.root / f"{name}-shallow"
    subprocess.run(["git", "clone", "-q", "--depth", "1", f"file://{repo}", str(clone)], check=True, capture_output=True, env=ENV)
    shutil.rmtree(repo)
    clone.rename(repo)


def test_a_shallow_clone_is_a_build_error_for_every_unit(world):
    _shallow(world)
    with pytest.raises(RF.RefreshError) as ei:
        RF.refresh(SYSTEM, **world.kw(make_target()[0]))
    assert {(u.status, u.fail_reason) for u in ei.value.report.units} == {("not-mapped", "build:shallow-clone")}


def test_a_shallow_clone_keeps_the_last_good_bundles_as_stale_kept(world):
    first = RF.refresh(SYSTEM, **world.kw(make_target()[0]))
    _shallow(world)
    target, fakes = make_target()
    rep = RF.refresh(SYSTEM, **world.kw(target))
    assert len(rep.stale_kept) == 3 and {u.fail_reason for u in rep.stale_kept} == {"build:shallow-clone"}
    assert {u.built_commit for u in rep.stale_kept} == {first.heads["r1"]} and rep.swap.renamed


# --- UnknownUnitKind per unit ---------------------------------------------------------------------------------------------------


def test_unknown_unit_kinds_are_recorded_per_unit_counted_apart_and_never_abort(world, monkeypatch, spies):
    real = RF.discover_units

    def with_unknown(*a, **k):
        extra = [Unit(ids.unit_id("r1", f"x{i}.kt", "kotlin"), "r1", f"x{i}.kt", "kotlin") for i in range(2)]
        return sorted(real(*a, **k) + extra, key=lambda u: (u.repoId, u.rootPath, u.language))

    monkeypatch.setattr(RF, "discover_units", with_unknown)
    rep = RF.refresh(SYSTEM, **world.kw(make_target()[0]))
    assert rep.unknown_kind == 2 and len(rep.fresh) == 3
    assert sorted((u.root_path, u.status, u.fail_reason) for u in rep.not_mapped) == [("x0.kt", "not-mapped", "extract:failed"),
                                                                                       ("x1.kt", "not-mapped", "extract:failed")]
    assert spies.calls[-2:] == ["load", "swap"] and unit_rows(spies.rows)[uid("x0.kt", "kotlin")]["failReason"] == "extract:failed"


def test_another_exception_from_a_build_is_not_swallowed(world):
    def build(unit, repo, commit, system, ctx):
        raise RuntimeError("a caller bug")

    with pytest.raises(RuntimeError, match="caller bug"):
        RF.refresh(SYSTEM, **world.kw(make_target()[0], build=build))


# --- (0) bind-key-mismatch ------------------------------------------------------------------------------------------------------


def _with_mismatch(monkeypatch, *lines):
    real = RF.link
    monkeypatch.setattr(RF, "link", lambda *a, **k: real(*a, **k)._replace(bind_key_mismatch=list(lines)))


def test_a_bind_key_mismatch_line_becomes_that_units_fail_reason(world, monkeypatch, spies):
    victim = uid("conf/app.yaml", "config")
    _with_mismatch(monkeypatch, f"bind-key-mismatch {victim}")
    rep = RF.refresh(SYSTEM, **world.kw(make_target()[0]))
    by = {u.unit_id: u for u in rep.units}
    assert (by[victim].status, by[victim].fail_reason, by[victim].failed_at) == ("stale-kept", "bundle:bind-key-check", NOW_Z)
    assert by[victim].built_commit == rep.heads["r1"] and by[uid("bin/run.sh", "shell")].status == "fresh"
    assert rep.bind_key_mismatch == [f"bind-key-mismatch {victim}"] and f"bind-key-mismatch {victim}" in rep.lines()
    row = unit_rows(spies.rows)[victim]
    assert (row["buildStatus"], row["failReason"]) == ("stale-kept", "bundle:bind-key-check") and "bundle:bind-key-check" in FAIL_REASONS


def test_a_mismatch_line_does_not_overwrite_a_build_failure_code_nor_invent_a_unit(world, monkeypatch):
    first = RF.refresh(SYSTEM, **world.kw(make_target()[0]))
    commit(world.repo(), {"conf/app.yaml": "name: app\nport: 9090\n"})
    victim = uid("conf/app.yaml", "config")
    _with_mismatch(monkeypatch, f"bind-key-mismatch {victim}", f"bind-key-mismatch {'0' * 32}")
    rep = RF.refresh(SYSTEM, **world.kw(make_target()[0], build=failing((victim, "extract:timeout"))))
    assert {u.unit_id: u.fail_reason for u in rep.stale_kept} == {victim: "extract:timeout"}
    assert len(rep.bind_key_mismatch) == 2 and first.heads["r1"] != rep.heads["r1"]


# --- (4) staleness against the local head ----------------------------------------------------------------------------------------


def test_staleness_uses_the_local_head_and_a_unit_the_commit_did_not_touch_stays_fresh(world):
    RF.refresh(SYSTEM, **world.kw(make_target()[0]))
    commit(world.repo(), {"conf/app.yaml": "name: app\nport: 9090\n"})
    victim = uid("bin/run.sh", "shell")                                            # fails, but nothing under its root changed
    rep = RF.refresh(SYSTEM, **world.kw(make_target()[0], build=failing((victim, "extract:failed"))))
    u = next(u for u in rep.units if u.unit_id == victim)
    assert u.status == "stale-kept" and u.staleness.stale is False and u.staleness.files_changed == 0
    assert rep.heads["r1"] == git(world.repo(), "rev-parse", "HEAD")


def test_staleness_mode_is_passed_through(world):
    RF.refresh(SYSTEM, **world.kw(make_target()[0]))
    commit(world.repo(), {"conf/app.yaml": "name: app\nport: 9090\n"})
    victim = uid("bin/run.sh", "shell")
    rep = RF.refresh(SYSTEM, **world.kw(make_target()[0], build=failing((victim, "extract:failed")), staleness_mode="commits"))
    u = next(u for u in rep.units if u.unit_id == victim)
    assert u.staleness.basis == "commits" and u.staleness.stale is True and u.staleness.commits_behind == 1


# --- (sweep) --------------------------------------------------------------------------------------------------------------------


def _plant_tmp(world, name: str, age: float) -> Path:
    d = atrest.bundle_dir(SYSTEM, world.work)
    p = d / name
    p.write_bytes(b"partial")
    p.chmod(0o600)
    os.utime(p, (NOW - age, NOW - age))
    return p


def test_the_sweep_runs_first_counts_what_it_removed_and_leaves_young_files(world, spies):
    old = _plant_tmp(world, ".u1.b1.bundle.jsonl.0123456789abcdef.tmp", 7200)
    old2 = _plant_tmp(world, ".u2.b1.bundle.jsonl.01234567.tmp", 7200)
    young = _plant_tmp(world, ".u3.b1.bundle.jsonl.0123456789abcdef.tmp", 60)
    rep = RF.refresh(SYSTEM, **world.kw(make_target()[0]))
    assert rep.swept == 2 and not old.exists() and not old2.exists() and young.exists()
    assert spies.calls[0] == "sweep" and rep.stages[0].name == "sweep"
    assert "swept 2 stale temp file(s)" in "\n".join(rep.lines())


def test_the_sweep_age_is_the_callers(world):
    young = _plant_tmp(world, ".u3.b1.bundle.jsonl.0123456789abcdef.tmp", 60)
    rep = RF.refresh(SYSTEM, **world.kw(make_target()[0], sweep_older_than_s=30))
    assert rep.swept == 1 and not young.exists()


def test_a_sweep_that_cannot_run_stops_the_refresh_before_any_build(world, monkeypatch):
    monkeypatch.setattr(RF.atrest, "sweep_stale_tmp", lambda *a, **k: (_ for _ in ()).throw(atrest.AtRestError("planted")))
    build = failing()
    with pytest.raises(atrest.AtRestError):
        RF.refresh(SYSTEM, **world.kw(make_target()[0], build=build))
    assert build.seen == []


# --- scope ----------------------------------------------------------------------------------------------------------------------


SCOPE_FILES = {**FILES, "app/main.py": "x = 1\n"}                      # a code-language unit besides the three host ones
PER_REPO = [*ALL_UNITS, ("app", "python")]


def _every_unit(repos=("r1", "r2")):
    return {uid(root, lang, r) for r in repos for root, lang in PER_REPO}


def _build_python_fails(*repos):
    return failing(*[(uid("app", "python", r), "extract:failed") for r in repos])


@pytest.mark.parametrize("scope", [["r1"], ["r2"], ["r1", "r2"]])
@pytest.mark.parametrize("state", ["stored", "none", "refused", "newer-commit"])
def test_a_scoped_refresh_always_loads_every_manifest_unit_whatever_the_stored_state_of_the_rest(tmp_path, spies, monkeypatch, scope, state):
    """The coverage probe of the ruling (2a): scope x stored-bundle state x language; the live unit set is the full manifest's in every cell."""
    w = World(tmp_path, repos=("r1", "r2"), files=SCOPE_FILES)
    first = RF.refresh(SYSTEM, **w.kw(make_target()[0], build=_build_python_fails("r1", "r2"))) if state != "none" else None
    c_old = git(w.repo("r1"), "rev-parse", "HEAD")
    if state == "newer-commit":
        commit(w.repo("r1"), {"conf/app.yaml": "name: app\nport: 9090\n"})
    if state == "refused":
        real, outside = RF.classify_bundle, _every_unit(tuple({"r1", "r2"} - set(scope)))
        monkeypatch.setattr(RF, "classify_bundle", lambda rec, audit: "bundle:audit-missing" if rec[0]["unitId"] in outside else real(rec, audit))
    target, fakes = make_target()
    rep = RF.refresh(SYSTEM, scope, **w.kw(target, build=_build_python_fails("r1", "r2")))
    assert {u["unitId"] for u in spies.rows.nodes["Unit"]} == _every_unit()
    assert {u.unit_id for u in rep.units} == _every_unit(), sorted((u.root_path, u.language) for u in rep.units)
    assert rep.scope == tuple(sorted(scope)) and set(rep.heads) == {"r1", "r2"} and rep.swap.renamed
    for u in rep.units:
        row = unit_rows(spies.rows)[u.unit_id]
        assert row["buildStatus"] == u.status
        if u.repo_id in scope:                                                    # rebuilt: the host units are fresh at their head
            assert u.status == ("not-mapped" if u.language == "python" else "fresh")
            continue
        if state == "none" or u.language == "python":                             # no bundle was ever stored (the python build failed)
            assert (u.status, u.fail_reason, u.built_commit) == ("not-mapped", "manifest:never-built", None)
        elif state == "refused":
            assert (u.status, u.fail_reason, u.built_commit) == ("not-mapped", "bundle:audit-missing", None)
        else:                                                                     # carried over unchanged: fresh, own commit, nothing failed
            assert (u.status, u.built_commit, u.failed_at, u.fail_reason) == ("fresh", c_old, None, None) and row["builtCommit"] == c_old
            assert u.staleness.stale is (state == "newer-commit" and u.repo_id == "r1" and u.root_path == "conf/app.yaml")
    assert first is None or first.heads["r1"] == c_old


def test_a_scoped_refresh_keeps_the_links_that_cross_the_scope_boundary(tmp_path, spies):
    w = World(tmp_path, repos=("r1", "r2"))
    full = RF.refresh(SYSTEM, **w.kw(make_target()[0]))
    full_rows = {k: len(v) for k, v in spies.rows.nodes.items()}
    scoped = RF.refresh(SYSTEM, ["r2"], **w.kw(make_target()[0]))
    assert {k: len(v) for k, v in spies.rows.nodes.items()} == full_rows and len(scoped.units) == len(full.units) == 6
    assert scoped.snapshot_id == full.snapshot_id and scoped.scope == ("r2",) and full.scope is None


def test_a_scoped_refresh_rebuilds_only_the_scoped_units_and_prunes_only_those(tmp_path):
    w = World(tmp_path, repos=("r1", "r2"))
    RF.refresh(SYSTEM, **w.kw(make_target()[0]))
    c1_old = git(w.repo("r1"), "rev-parse", "HEAD")
    commit(w.repo("r1"), {"conf/app.yaml": "name: app\nport: 9090\n"})
    commit(w.repo("r2"), {"conf/app.yaml": "name: app\nport: 9090\n"})
    build = failing()
    rep = RF.refresh(SYSTEM, ["r2"], **w.kw(make_target()[0], build=build))
    assert build.seen and all(i in {uid(*u, repo="r2") for u in ALL_UNITS} for i in build.seen)
    assert {u.unit_id: u.built_commit for u in rep.units if u.repo_id == "r1"} == {uid(*u, repo="r1"): c1_old for u in ALL_UNITS}


def test_a_bad_scope_is_refused_before_any_work(tmp_path):
    w = World(tmp_path)
    build = failing()
    with pytest.raises(RF.RefreshError, match="does not declare"):
        RF.refresh(SYSTEM, ["nope"], **w.kw(make_target()[0], build=build))
    for bad in ([], "r1"):
        with pytest.raises(ValueError):
            RF.refresh(SYSTEM, bad, **w.kw(make_target()[0], build=build))
    assert build.seen == []


# --- the order and a failed load -----------------------------------------------------------------------------------------------


def test_a_failed_load_leaves_the_live_graph_and_the_stored_bundles_intact(world, spies):
    first = RF.refresh(SYSTEM, **world.kw(make_target()[0]))
    c2 = commit(world.repo(), {"conf/app.yaml": "name: app\nport: 9090\n"})
    target, fakes = make_target(fail_on=fail_at(3))
    spies.calls.clear()
    with pytest.raises(Boom):
        RF.refresh(SYSTEM, **world.kw(target))
    assert spies.calls == ["sweep", "build_units", "build_rows", "load"]              # swap never ran
    assert fakes.shared == {sysname.live_key(SYSTEM): "old"} and not renames(fakes) and not [e for e in fakes.log if e[1] == "set"]
    assert not fakes.store.keys                                                        # the loader removed its own build key
    d = atrest.bundle_dir(SYSTEM, world.work)
    victim = uid("conf/app.yaml", "config")
    assert (d / f"{victim}.{first.heads['r1']}.bundle.jsonl").is_file() and (d / f"{victim}.{c2}.bundle.jsonl").is_file()   # no retention yet


def test_a_failed_swap_does_not_prune_either(world, monkeypatch):
    first = RF.refresh(SYSTEM, **world.kw(make_target()[0]))
    commit(world.repo(), {"conf/app.yaml": "name: app\nport: 9090\n"})
    target, fakes = make_target()
    fakes.shared[sysname.lock_key(SYSTEM)] = "someone-else"                             # the verdict lock is held: the swap refuses
    real = RF.swap
    monkeypatch.setattr(RF, "swap", lambda *a, **k: real(*a, **{**k, "wait_s": 0}))
    with pytest.raises(SW.SwapError, match="lock"):
        RF.refresh(SYSTEM, **world.kw(target))
    assert not renames(fakes) and fakes.shared[sysname.live_key(SYSTEM)] == "old"
    victim = uid("conf/app.yaml", "config")
    assert (atrest.bundle_dir(SYSTEM, world.work) / f"{victim}.{first.heads['r1']}.bundle.jsonl").is_file()


def test_the_newest_stored_bundle_is_the_last_good_one(world, monkeypatch):
    RF.refresh(SYSTEM, **world.kw(make_target()[0]))
    c2 = commit(world.repo(), {"conf/app.yaml": "name: app\nport: 9090\n"})
    target, fakes = make_target()
    fakes.shared[sysname.lock_key(SYSTEM)] = "someone-else"
    real = RF.swap
    monkeypatch.setattr(RF, "swap", lambda *a, **k: real(*a, **{**k, "wait_s": 0}))
    with pytest.raises(SW.SwapError):                                                  # c1's and c2's bundles are now both on disk
        RF.refresh(SYSTEM, **world.kw(target))
    monkeypatch.undo()
    commit(world.repo(), {"conf/app.yaml": "name: app\nport: 7070\n"})
    victim = uid("conf/app.yaml", "config")
    rep = RF.refresh(SYSTEM, **world.kw(make_target()[0], build=failing((victim, "extract:failed"))))
    assert {u.unit_id: u.built_commit for u in rep.stale_kept} == {victim: c2}


# --- a fresh bundle the assembler would refuse ---------------------------------------------------------------------------------


def _refuse_fresh(monkeypatch, unit_id, commit_id, code="bundle:audit-missing"):
    """``classify_bundle`` refuses the bundle of ``unit_id`` built at ``commit_id`` (an older one of the unit still passes)."""
    real = RF.classify_bundle
    monkeypatch.setattr(RF, "classify_bundle", lambda records, audit: code if (records[0]["unitId"], records[0]["builtCommit"]) == (unit_id, commit_id)
                        else real(records, audit))


def test_a_fresh_bundle_the_gate_refuses_falls_back_to_the_last_good_one_with_the_gates_code(world, monkeypatch, spies):
    first = RF.refresh(SYSTEM, **world.kw(make_target()[0]))
    c2 = commit(world.repo(), {"conf/app.yaml": "name: app\nport: 9090\n"})
    victim = uid("conf/app.yaml", "config")
    _refuse_fresh(monkeypatch, victim, c2)
    rep = RF.refresh(SYSTEM, **world.kw(make_target()[0]))
    u = next(u for u in rep.units if u.unit_id == victim)
    assert (u.status, u.built_commit, u.fail_reason) == ("stale-kept", first.heads["r1"], "bundle:audit-missing")
    assert unit_rows(spies.rows)[victim]["failReason"] == "bundle:audit-missing" and len(rep.fresh) == 2


def test_a_fresh_bundle_the_gate_refuses_with_no_last_good_is_not_mapped(world, monkeypatch):
    c1 = git(world.repo(), "rev-parse", "HEAD")
    victim = uid("conf/app.yaml", "config")
    _refuse_fresh(monkeypatch, victim, c1, "bundle:schema-minor")
    rep = RF.refresh(SYSTEM, **world.kw(make_target()[0]))
    assert [(u.unit_id, u.status, u.fail_reason) for u in rep.not_mapped] == [(victim, "not-mapped", "bundle:schema-minor")]


def test_a_stored_bundle_whose_header_names_another_build_is_not_a_last_good_one(world):
    RF.refresh(SYSTEM, **world.kw(make_target()[0]))
    d = atrest.bundle_dir(SYSTEM, world.work)
    victim = uid("conf/app.yaml", "config")
    src = next(d.glob(f"{victim}.*.bundle.jsonl"))
    for p in d.iterdir():
        if not p.name.startswith(victim):
            p.unlink()
    src.rename(d / f"{victim}.{'a' * 40}.bundle.jsonl")                               # a bundle filed under a build its header does not name
    commit(world.repo(), {"conf/app.yaml": "name: app\nport: 9090\n"})
    rep = RF.refresh(SYSTEM, **world.kw(make_target()[0], build=failing((victim, "extract:failed"))))
    assert [(u.unit_id, u.status) for u in rep.not_mapped] == [(victim, "not-mapped")]


def test_a_leftover_build_key_of_the_same_snapshot_is_dropped_and_the_refresh_goes_on(world):
    first_target, first = make_target()
    rep = RF.refresh(SYSTEM, **world.kw(first_target))
    target, fakes = make_target(existing={rep.load.build_key})                         # an interrupted load of this very snapshot
    again = RF.refresh(SYSTEM, **world.kw(target))
    assert again.snapshot_id == rep.snapshot_id and ("delete", rep.load.build_key) in fakes.store.calls and again.swap.renamed


def test_the_swap_reads_the_stamps_under_the_servers_query_timeout(world):
    target, fakes = make_target()
    rep = RF.refresh(SYSTEM, **world.kw(target))
    assert rep.load.timeout_max == 10000 and fakes.bdb.timeouts and {t for _, _, t in fakes.bdb.timeouts + fakes.vdb.timeouts} == {10000}


# --- the broker ---------------------------------------------------------------------------------------------------------------------


class FakeBroker:
    def __init__(self, calls, fail=None):
        self.calls, self.fail = calls, fail
        self.token = "t" * 64

    def open_run(self, kind, system, codebases, override=None):
        self.calls.append(("open_run", kind, system, codebases, override))
        if self.fail:
            raise self.fail
        return SimpleNamespace(run_id="run-1", token=self.token, effective={"r1": "local-only"}, system_local_only=True,
                               audit_id="11111111-1111-4111-8111-111111111111")

    def close_run(self, token):
        self.calls.append(("close_run", token))


def test_no_override_opens_no_broker_run_and_never_builds_a_client(world, monkeypatch):
    monkeypatch.setattr(RF, "BrokerClient", lambda *a, **k: pytest.fail("a broker client was built without an override"))
    rep = RF.refresh(SYSTEM, **world.kw(make_target()[0]))
    assert rep.policy_audit_id is None and "broker" not in "\n".join(rep.lines())


def test_an_override_opens_one_refresh_run_first_names_its_policy_record_and_closes_it(world, spies):
    calls = spies.calls
    rep = RF.refresh(SYSTEM, None, "local", **world.kw(make_target()[0], broker=FakeBroker(calls)))
    assert calls[0] == ("open_run", "refresh", SYSTEM, "*", "local")
    assert calls[1] == "sweep" and calls[-1] == ("close_run", "t" * 64)
    assert rep.policy_audit_id == "11111111-1111-4111-8111-111111111111"
    text = "\n".join(rep.lines())
    assert "broker policy record 11111111-1111-4111-8111-111111111111" in text and "t" * 64 not in text


def test_the_run_carries_the_scope_as_its_codebases(tmp_path, spies):
    w = World(tmp_path, repos=("r1", "r2"))
    RF.refresh(SYSTEM, ["r2", "r1"], "hosted", **w.kw(make_target()[0], broker=FakeBroker(spies.calls)))
    assert spies.calls[0] == ("open_run", "refresh", SYSTEM, ["r1", "r2"], "hosted")


def test_the_run_is_closed_even_when_the_refresh_fails_later(world, spies):
    target, _ = make_target(fail_on=fail_at(2))
    with pytest.raises(Boom):
        RF.refresh(SYSTEM, None, "local", **world.kw(target, broker=FakeBroker(spies.calls)))
    assert spies.calls[-1] == ("close_run", "t" * 64)


def test_an_absent_broker_fails_the_overridden_refresh_with_no_fallback(world, tmp_path, spies):
    absent = BrokerClient(socket_path=str(tmp_path / "no-such.sock"), timeout=1)
    target, fakes = make_target()
    build = failing()
    with pytest.raises(BrokerUnavailable):
        RF.refresh(SYSTEM, None, "local", **world.kw(target, broker=absent, build=build))
    assert spies.calls == [] and build.seen == [] and fakes.store.calls == [] and fakes.log == []     # no sweep, no build, no store touched


def test_a_refused_run_fails_the_refresh_with_the_brokers_error(world, spies):
    target, fakes = make_target()
    with pytest.raises(BrokerError, match="audit-unwritable"):
        RF.refresh(SYSTEM, None, "local", **world.kw(target, broker=FakeBroker(spies.calls, fail=BrokerError("audit-unwritable"))))
    assert fakes.store.calls == [] and "sweep" not in spies.calls and ("close_run", "t" * 64) not in spies.calls


def test_an_unknown_override_is_refused_before_the_broker_is_asked(world, spies):
    with pytest.raises(ValueError):
        RF.refresh(SYSTEM, None, "mixed", **world.kw(make_target()[0], broker=FakeBroker(spies.calls)))
    assert spies.calls == []


# --- (6) retention and no fetch --------------------------------------------------------------------------------------------------------


def test_after_a_good_refresh_each_unit_keeps_exactly_one_bundle(world):
    RF.refresh(SYSTEM, **world.kw(make_target()[0]))
    commit(world.repo(), {"conf/app.yaml": "name: app\nport: 9090\n", "bin/run.sh": "#!/bin/sh\necho 2\n", "docs/note.md": "# N\n\nx\n"})
    rep = RF.refresh(SYSTEM, **world.kw(make_target()[0]))
    names = sorted(p.name for p in atrest.bundle_dir(SYSTEM, world.work).iterdir())
    assert names == sorted(f"{uid(*u)}.{rep.heads['r1']}.bundle.jsonl" for u in ALL_UNITS) and rep.pruned == 3


def test_refresh_never_asks_git_for_anything_but_local_reads(world, monkeypatch):
    from ccg import gitsafe
    seen = set()
    real = gitsafe.git

    def spy(repo, *a, **k):
        seen.add(a[0])
        return real(repo, *a, **k)

    monkeypatch.setattr(gitsafe, "git", spy)
    from extractors.joern.normalize import blame
    monkeypatch.setattr(blame.gitsafe, "git", spy, raising=False)
    RF.refresh(SYSTEM, **world.kw(make_target()[0]))
    assert seen and seen <= gitsafe.READ_ONLY and not {"fetch", "pull", "clone", "ls-remote", "remote"} & seen
    assert re.fullmatch(r"[0-9a-f]{40}", git(world.repo(), "rev-parse", "HEAD"))


def test_a_complete_assembly_failure_before_the_load_is_not_swallowed(world, monkeypatch, spies):
    monkeypatch.setattr(RF, "build_rows", lambda *a, **k: (_ for _ in ()).throw(AssembleError("planted")))
    target, fakes = make_target()
    with pytest.raises(AssembleError):
        RF.refresh(SYSTEM, **world.kw(target))
    assert fakes.store.calls == [] and not renames(fakes)


# --- the default pack set -----------------------------------------------------------------------------------------------------------


def test_the_default_pack_set_is_every_pack_directory_but_the_one_run_packs_refuses():
    on_disk = {d.name for d in (RF.CPG / "packs").iterdir() if d.is_dir()}
    assert set(RF.DEFAULT_PACKS) == on_disk - {"paper-inhouse-bus"} and len(RF.DEFAULT_PACKS) == len(set(RF.DEFAULT_PACKS))
    assert set(RF.DEFAULT_PACKS) == {"fastapi", "fastmcp", "fetch", "infra-signatures", "kafka", "severity-rules", "shell-mcp-call"}


def test_the_packs_reach_the_linker_and_the_assembler_as_loaded(world, monkeypatch):
    seen = {}
    real = RF.build_rows
    monkeypatch.setattr(RF, "build_rows", lambda *a, **k: (seen.setdefault("ids", sorted(p["id"] if isinstance(p, dict) else p.id for p in k["packs"])), real(*a, **k))[1])
    RF.refresh(SYSTEM, **world.kw(make_target()[0]))
    assert len(seen["ids"]) == len(RF.DEFAULT_PACKS)
    seen.clear()
    RF.refresh(SYSTEM, **world.kw(make_target()[0], packs=("fetch",)))
    assert len(seen["ids"]) == 1


# --- key wiring (the build's key and the binder's key are the system key from keys_root, and nowhere else) -----------------------------------


URL_FILES = {"conf/app.yaml": "name: app\nurl: http://api.example.com:8000/v1\n",
             "docker-compose.yml": "services:\n  web:\n    image: x\n    environment:\n      API_URL: http://api:8000/v1\n"}


def _header_of(world, rep, root):
    from ccg import bundle
    u = next(u for u in rep.units if u.root_path == root)
    return bundle.loads(atrest.read_bundle(SYSTEM, u.unit_id, u.built_commit, world.work))


def test_the_system_key_lives_only_under_keys_root_and_both_build_and_binder_use_it(tmp_path):
    w = World(tmp_path, files=URL_FILES)
    rep = RF.refresh(SYSTEM, **w.kw(make_target()[0]))
    assert sorted(p.name for p in w.keys.iterdir()) == [f"{SYSTEM}.hmac"]                    # created here, and by nobody else
    assert not Path(os.environ["CCG_CREDENTIALS_DIR"]).exists()                                # the env area (a stand-in for the default) is untouched
    key = atrest.hmac_key(SYSTEM, w.keys)
    for root in ("conf/app.yaml", "docker-compose.yml"):
        recs = _header_of(w, rep, root)
        assert recs[0]["bindKeyCheck"] == atrest.bind_key_check(key)                          # the build used keys_root's key
        assert any("bindParts" in r for r in recs)                                            # the fixture really produces bindParts
    assert rep.bind_key_mismatch == []  # the binder used the same key: no mismatch
    assert not [u for u in rep.units if u.fail_reason == "bundle:bind-key-check"]


def test_a_binder_with_another_key_makes_every_bindparts_unit_a_bind_key_mismatch(tmp_path, monkeypatch):
    w = World(tmp_path, files=URL_FILES)
    real = RF.Binder
    monkeypatch.setattr(RF, "Binder", lambda *a, **k: real(*a, **{**k, "key": b"x" * 32}))
    rep = RF.refresh(SYSTEM, **w.kw(make_target()[0]))
    wired = sorted(u.unit_id for u in rep.units if u.root_path in ("conf/app.yaml", "docker-compose.yml"))
    assert rep.bind_key_mismatch == [f"bind-key-mismatch {i}" for i in wired]
    assert {(u.status, u.fail_reason) for u in rep.units if u.unit_id in wired} == {("stale-kept", "bundle:bind-key-check")}


def test_a_refresh_that_forgets_keys_root_fails_in_test_mode_without_touching_the_real_area(world, monkeypatch):
    """What a mutant that drops ``keys_root`` meets: ``tests/conftest.py`` refuses the default area before a byte is written."""
    real = Path(os.path.expanduser("~")) / ".config" / "ccg" / "keys"
    before = sorted(p.name for p in real.iterdir()) if real.is_dir() else None
    with pytest.raises(RF.RefreshError, match="nothing to load"):                          # every unit's build is refused, none is written
        RF.refresh(SYSTEM, **{**world.kw(make_target()[0]), "keys_root": None})
    assert (sorted(p.name for p in real.iterdir()) if real.is_dir() else None) == before


# --- a failure after the swap ------------------------------------------------------------------------------------------------------------


def test_a_prune_failure_after_the_swap_is_counted_and_the_report_still_comes_back(world):
    first = RF.refresh(SYSTEM, **world.kw(make_target()[0]))
    commit(world.repo(), {"conf/app.yaml": "name: app\nport: 9090\n"})
    d = atrest.bundle_dir(SYSTEM, world.work)
    victim = uid("conf/app.yaml", "config")
    (d / f"{victim}.zz.bundle.jsonl").symlink_to(world.root)                                 # a planted link: retention refuses to touch it
    target, fakes = make_target()
    rep = RF.refresh(SYSTEM, **world.kw(target))
    assert rep.swap.renamed and fakes.shared[sysname.live_key(SYSTEM)] == "new"
    assert rep.prune_errors == 1 and "prune errors 1" in "\n".join(rep.lines())
    assert (d / f"{victim}.zz.bundle.jsonl").is_symlink() and world.root.is_dir() and first.heads["r1"] != rep.heads["r1"]
    assert rep.pruned >= 2                                                                    # the other units were pruned all the same


def test_a_clean_refresh_reports_no_prune_errors(world):
    rep = RF.refresh(SYSTEM, **world.kw(make_target()[0]))
    assert rep.prune_errors == 0 and "prune errors 0" in "\n".join(rep.lines())


# --- hostile repository-controlled text in the report -----------------------------------------------------------------------------------------


HOSTILE = "a\nrefresh rfs: 99 fresh\x1b[2J.sh"


def _printable_ascii(line: str) -> bool:
    return all(0x20 <= ord(c) <= 0x7e for c in line)


def test_a_hostile_file_name_cannot_forge_a_report_line_or_inject_an_escape(world):
    commit(world.repo(), {HOSTILE: "#!/bin/sh\n"})                                           # a committed name with a newline and an ESC
    rep = RF.refresh(SYSTEM, **world.kw(make_target()[0], build=failing()))
    lines = rep.lines()
    assert not [ln for ln in lines if ln.startswith("refresh rfs: 99")] and all(_printable_ascii(ln) for ln in lines)
    assert any("a\\nrefresh rfs: 99 fresh\\x1b[2J.sh" in ln for ln in lines)               # visible, escaped


@pytest.mark.parametrize("name", ["a\nb", "a\x1bb", "a\x00b", "a\rb", "a\u2028b", "a\u202eb", "é\udc80", "a\\nb", "x" * 4100, " "])
def test_every_repository_controlled_string_is_one_short_printable_ascii_token(name):
    rep = RF.RefreshReport("s")
    rep.units = [RF.UnitReport(name, name, name, name, "not-mapped", fail_reason=name)]
    rep.unattested_units = [name]
    rep.bind_key_mismatch = [name]
    rep.dropped = {name: 1}
    rep.policy_audit_id = name
    lines = rep.lines()
    assert all(_printable_ascii(ln) and len(ln) < 1000 for ln in lines) and len(lines) == 7


def test_safe_text_escapes_and_caps():
    assert RF.safe_text("a\nb\x1b") == "a\\nb\\x1b" and RF.safe_text("plain-1.sh") == "plain-1.sh"
    out = RF.safe_text("y" * 500)
    assert len(out) == 120 and out.endswith("...")


# --- retention of orphans ----------------------------------------------------------------------------------------------------------------


def test_a_full_refresh_removes_the_bundles_of_units_that_no_longer_exist(world):
    RF.refresh(SYSTEM, **world.kw(make_target()[0]))
    gone = uid("docs/note.md", "markdown")
    commit(world.repo(), {"docs/note.md": None})
    rep = RF.refresh(SYSTEM, **world.kw(make_target()[0]))
    names = [p.name for p in atrest.bundle_dir(SYSTEM, world.work).iterdir()]
    assert not [n for n in names if n.startswith(gone)] and len(names) == 2 and rep.pruned == 3     # 2 old builds of the live units + the orphan


def test_a_scoped_refresh_never_removes_orphans(tmp_path):
    w = World(tmp_path, repos=("r1", "r2"))
    RF.refresh(SYSTEM, **w.kw(make_target()[0]))
    commit(w.repo("r1"), {"docs/note.md": None})
    gone = uid("docs/note.md", "markdown", "r1")
    RF.refresh(SYSTEM, ["r2"], **w.kw(make_target()[0]))
    assert [p for p in atrest.bundle_dir(SYSTEM, w.work).iterdir() if p.name.startswith(gone)]


# --- survivors of the first gate --------------------------------------------------------------------------------------------------------


def test_a_mismatch_line_is_read_only_when_it_is_exactly_the_linkers_line(world, monkeypatch):
    victim = uid("conf/app.yaml", "config")
    _with_mismatch(monkeypatch, f"seen: bind-key-mismatch {victim}", f"bind-key-mismatch {victim[:8]}", f"bind-key-mismatch {victim}x")
    rep = RF.refresh(SYSTEM, **world.kw(make_target()[0]))
    assert not rep.stale_kept and len(rep.bind_key_mismatch) == 3


def test_an_oversize_markdown_is_counted_as_a_skipped_doc(world):
    big = "# Big\n\n" + "x" * (2_097_152 + 10)
    commit(world.repo(), {"docs/big.md": big})
    rep = RF.refresh(SYSTEM, **world.kw(make_target()[0]))
    assert rep.skipped_docs == 1 and "skippedDocs 1" in "\n".join(rep.lines())


def test_a_manifest_whose_repositories_have_no_unit_is_refused_with_the_report(tmp_path):
    w = World(tmp_path, files={"README.txt": "nothing a unit is made of\n"})
    build = failing()
    with pytest.raises(RF.RefreshError, match="no unit") as ei:
        RF.refresh(SYSTEM, **w.kw(make_target()[0], build=build))
    assert build.seen == [] and ei.value.report.heads


def test_a_code_unit_is_classified_with_the_audit_dir_the_caller_gave(world, monkeypatch):
    seen = set()
    real = RF.classify_bundle
    monkeypatch.setattr(RF, "classify_bundle", lambda rec, audit: (seen.add(audit), real(rec, audit))[1])
    RF.refresh(SYSTEM, **world.kw(make_target()[0]))
    assert seen == {world.audit}


def test_a_unit_the_build_never_reported_keeps_its_last_good_bundle_with_the_extract_failed_code(world, monkeypatch):
    RF.refresh(SYSTEM, **world.kw(make_target()[0]))
    commit(world.repo(), {"conf/app.yaml": "name: app\nport: 9090\n"})
    victim = uid("docs/note.md", "markdown")
    real = RF.build_units

    def drop_one(units, repos, commits, system, ctx=None, *, build=None):
        rep = real(units, repos, commits, system, ctx, build=build)
        return rep.__class__(tuple(r for r in rep.results if r.unit.unitId != victim), rep.known_tools, rep.skipped_docs)

    monkeypatch.setattr(RF, "build_units", drop_one)
    rep = RF.refresh(SYSTEM, **world.kw(make_target()[0]))
    u = next(u for u in rep.units if u.unit_id == victim)
    assert (u.status, u.built_commit, u.fail_reason) == ("stale-kept", rep.heads["r1"], "extract:failed")   # the build did store one


def test_an_override_builds_one_broker_client_and_reuses_it_to_close_the_run(world, monkeypatch):
    built, calls = [], []

    class OneClient(FakeBroker):
        def __init__(self, *a, **k):
            super().__init__(calls)
            built.append(self)

    monkeypatch.setattr(RF, "BrokerClient", OneClient)
    RF.refresh(SYSTEM, None, "local", **world.kw(make_target()[0]))
    assert len(built) == 1 and calls[0][0] == "open_run" and calls[-1] == ("close_run", "t" * 64)


# --- a path that is not valid Unicode --------------------------------------------------------------------------------------------------------


def _commit_raw_names(repo: Path, names: list[bytes]) -> None:
    for n in names:
        raw = os.path.join(os.fsencode(repo), n)
        os.makedirs(os.path.dirname(raw), exist_ok=True)
        with open(raw, "wb") as fh:
            fh.write(b"#!/bin/sh\necho x\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "raw names")


def test_a_non_utf8_git_path_is_excluded_and_counted_and_the_refresh_completes(world):
    _commit_raw_names(world.repo(), [b"bad\xff\xfe.sh", b"dir\xc3(/x.sh"])
    rep = RF.refresh(SYSTEM, **world.kw(make_target()[0]))
    assert rep.undecodable_paths == 2 and "undecodablePaths 2" in "\n".join(rep.lines())
    assert sorted((u.root_path, u.language) for u in rep.units) == sorted(ALL_UNITS) and len(rep.fresh) == 3
    assert rep.swap.renamed and "\\xff" not in "\n".join(rep.lines())


def test_a_clean_tree_has_zero_undecodable_paths(world):
    assert RF.refresh(SYSTEM, **world.kw(make_target()[0])).undecodable_paths == 0
