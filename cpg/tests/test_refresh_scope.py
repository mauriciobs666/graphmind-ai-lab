"""P1-20 follow-up — ``ccg.refresh`` scope: the system-wide inputs, the carried unit state, and the pinned retention/measurement behaviours.

Offline like ``test_refresh.py`` (its ``World``/fake store are reused): host units are built by the real extractors; the one code unit is a
synthetic, valid bundle written through ``atrest.write_bundle`` in place of the Joern step (``build._build_code``), so no Joern, Docker, network
or database runs. Everything lives under ``tmp_path``; ``tests/conftest.py`` keeps the real ``~/.config/ccg`` out of reach.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from loader_helpers import Boom, fail_at
from test_assemble_rows import sbx
from test_build_dispatch import AUDIT, synthetic_mcp_bundle
from test_refresh import NOW, NOW_Z, SYSTEM, World, commit, failing, git, make_target, uid, unit_rows

from ccg import atrest, bundle, ids
from ccg import build as B
from ccg import refresh as RF
from ccg.staleness import FAIL_REASONS

LATER_Z = "2027-01-15T08:08:20Z"                                   # NOW + 500 s


@pytest.fixture
def rows(monkeypatch):
    """The rows each refresh gave the loader (the last one)."""
    box = SimpleNamespace(rows=None)
    real = RF.load

    def spy(store, r, *a, **k):
        box.rows = r
        return real(store, r, *a, **k)

    monkeypatch.setattr(RF, "load", spy)
    return box


# --- N1: a scoped refresh builds with the whole system's tool vocabulary ------------------------------------------------------------

PROMPT = "# Support agent\n\nWhen a customer asks about an item, call the `get_item` tool and summarize the result.\n"


def tools_world(tmp_path, monkeypatch, tools=("get_item", "list_orders")):
    """Repository ``a``: one Python unit (stub-built, defines MCP tools); repository ``b``: a prompt-bearing Markdown unit naming one of them."""
    w = World(tmp_path, repos=("a", "b"), files={})
    commit(w.repo("a"), {"main.py": "x = 1\n"})
    commit(w.repo("b"), {"docs/AGENTS.md": PROMPT})
    (w.audit / "sandbox.jsonl").write_text(json.dumps(sbx(AUDIT)) + "\n")

    def fake(ctx, u, repo, sha, system, report):
        recs = synthetic_mcp_bundle(u.repoId, list(tools))
        recs[0].update(builtCommit=sha)
        return atrest.write_bundle(system, u.unitId, sha, recs, root=ctx.root), "1.0"

    monkeypatch.setattr(B, "_build_code", fake)
    return w


def markdown_bytes(w):
    u = uid("docs/AGENTS.md", "markdown", "b")
    (path,) = [p for p in atrest.bundle_dir(SYSTEM, w.work).iterdir() if p.name.startswith(u + ".") and p.name.endswith(atrest.BUNDLE_SUFFIX)]
    return path.read_bytes()


def mentions(w):
    recs = bundle.loads(markdown_bytes(w))
    return sorted(r["calleeName"] for r in recs if r["type"] == "callsite")


@pytest.mark.parametrize("scope", [["a"], ["b"], ["a", "b"]])
def test_a_scoped_refresh_builds_the_same_markdown_bytes_and_snapshot_as_a_full_refresh(tmp_path, monkeypatch, scope):
    w = tools_world(tmp_path, monkeypatch)
    full = RF.refresh(SYSTEM, **w.kw(make_target()[0]))
    assert mentions(w) == ["get_item"], "the probe is vacuous unless the full refresh sees the tool defined in the other repository"
    reference = markdown_bytes(w)
    scoped = RF.refresh(SYSTEM, scope, **w.kw(make_target()[0]))
    assert mentions(w) == ["get_item"] and markdown_bytes(w) == reference
    assert scoped.snapshot_id == full.snapshot_id and [u.status for u in scoped.units] == ["fresh"] * 2


def test_the_scoped_build_reads_the_full_unit_list_and_the_out_of_scope_code_bundles(tmp_path, monkeypatch):
    w = tools_world(tmp_path, monkeypatch)
    RF.refresh(SYSTEM, **w.kw(make_target()[0]))
    seen = []

    def stub(unit, repo, commit_id, system, ctx):
        seen.append((unit.repoId, len(ctx.units), ctx.known_tools))
        return B.build_unit(unit, repo, commit_id, system, ctx)

    RF.refresh(SYSTEM, ["b"], **w.kw(make_target()[0], build=stub))
    assert seen == [("b", 2, frozenset({"get_item", "list_orders"}))]


def test_a_carried_code_unit_that_is_stale_kept_adds_no_tools(tmp_path, monkeypatch):
    w = tools_world(tmp_path, monkeypatch)
    RF.refresh(SYSTEM, **w.kw(make_target()[0]))
    commit(w.repo("a"), {"main.py": "x = 2\n"})
    code = uid(".", "python", "a")
    RF.refresh(SYSTEM, **w.kw(make_target()[0], build=failing((code, "extract:timeout"))))
    rep = RF.refresh(SYSTEM, ["b"], **w.kw(make_target()[0]))
    assert {u.unit_id: u.fail_reason for u in rep.stale_kept} == {code: "extract:timeout"} and mentions(w) == []


# --- N2 (ruled): a carried unit keeps its failure marker until it is rebuilt -----------------------------------------------------------


VICTIM = uid("conf/app.yaml", "config", "r1")


def failed_world(tmp_path, rows):
    """Two repositories; a good full refresh, then a full refresh at ``NOW`` where r1's config build times out (the unit is stale-kept)."""
    w = World(tmp_path, repos=("r1", "r2"))
    RF.refresh(SYSTEM, **w.kw(make_target()[0]))
    good = rows.rows
    bad = RF.refresh(SYSTEM, **w.kw(make_target()[0], build=failing((VICTIM, "extract:timeout"))))
    return w, good, bad


def only(rep, unit_id):
    (u,) = [u for u in rep.units if u.unit_id == unit_id]
    return u


def test_a_unit_that_failed_stays_stale_kept_with_its_marker_across_a_scoped_refresh_of_another_repository(tmp_path, rows):
    w, _, bad = failed_world(tmp_path, rows)
    assert (only(bad, VICTIM).status, only(bad, VICTIM).failed_at) == ("stale-kept", NOW_Z)
    scoped = RF.refresh(SYSTEM, ["r2"], **w.kw(make_target()[0], now=NOW + 500))
    u = only(scoped, VICTIM)
    assert (u.status, u.fail_reason, u.failed_at, u.built_commit) == ("stale-kept", "extract:timeout", NOW_Z, only(bad, VICTIM).built_commit)
    row = unit_rows(rows.rows)[VICTIM]
    assert (row["buildStatus"], row["failReason"], row["failedAt"]) == ("stale-kept", "extract:timeout", NOW_Z)
    assert scoped.snapshot_id == bad.snapshot_id                       # the graph reads as the failed full refresh did, not as a fresh one


def test_the_carried_marker_changes_the_snapshot_a_fresh_carry_would_have_produced(tmp_path, rows):
    w, _, bad = failed_world(tmp_path, rows)
    RF.refresh(SYSTEM, ["r1"], **w.kw(make_target()[0]))              # a good rebuild of r1
    cleared = RF.refresh(SYSTEM, ["r2"], **w.kw(make_target()[0]))
    assert cleared.snapshot_id != bad.snapshot_id and only(cleared, VICTIM).status == "fresh"


def test_rebuilding_the_unit_clears_the_marker_and_a_later_scoped_refresh_keeps_it_clear(tmp_path, rows):
    w, _, _ = failed_world(tmp_path, rows)
    RF.refresh(SYSTEM, ["r1"], **w.kw(make_target()[0], now=NOW + 500))
    rep = RF.refresh(SYSTEM, ["r2"], **w.kw(make_target()[0], now=NOW + 900))
    u = only(rep, VICTIM)
    assert (u.status, u.failed_at, u.fail_reason) == ("fresh", None, None) and atrest.read_unit_state(SYSTEM, VICTIM, w.work)["status"] == "fresh"


def test_a_rebuild_that_fails_again_replaces_the_marker(tmp_path, rows):
    w, _, _ = failed_world(tmp_path, rows)
    RF.refresh(SYSTEM, ["r1"], **w.kw(make_target()[0], now=NOW + 500, build=failing((VICTIM, "extract:failed"))))
    u = only(RF.refresh(SYSTEM, ["r2"], **w.kw(make_target()[0], now=NOW + 900)), VICTIM)
    assert (u.status, u.failed_at, u.fail_reason) == ("stale-kept", LATER_Z, "extract:failed")


def test_a_unit_with_no_state_record_is_fresh_as_before(tmp_path, rows):
    w = World(tmp_path, repos=("r1", "r2"))
    RF.refresh(SYSTEM, **w.kw(make_target()[0]))
    for p in atrest.bundle_dir(SYSTEM, w.work).glob("*" + atrest.STATE_SUFFIX):
        p.unlink()
    rep = RF.refresh(SYSTEM, ["r2"], **w.kw(make_target()[0]))
    assert only(rep, VICTIM).status == "fresh" and not rep.stale_kept


def test_a_scoped_refresh_writes_no_state_for_a_carried_unit(tmp_path, rows):
    w, _, _ = failed_world(tmp_path, rows)
    before = atrest.unit_state_path(SYSTEM, VICTIM, w.work).read_bytes()
    RF.refresh(SYSTEM, ["r2"], **w.kw(make_target()[0], now=NOW + 500))
    assert atrest.unit_state_path(SYSTEM, VICTIM, w.work).read_bytes() == before


def test_the_state_is_written_only_after_the_swap(tmp_path, rows):
    w = World(tmp_path, repos=("r1",))
    first = RF.refresh(SYSTEM, **w.kw(make_target()[0]))
    commit(w.repo("r1"), {"conf/app.yaml": "name: app\nport: 9090\n"})
    with pytest.raises(Boom):                                          # a failing load: no swap, so no new state either
        RF.refresh(SYSTEM, **w.kw(make_target(fail_on=fail_at(3))[0]))
    assert atrest.read_unit_state(SYSTEM, VICTIM, w.work)["builtCommit"] == first.heads["r1"]


@pytest.mark.parametrize("damage", ["garbage", "extra-field", "unknown-code", "marker-without-status", "other-build", "link", "empty", "bad-stamp"])
def test_a_damaged_state_record_fails_closed_value_free_for_a_carried_unit(tmp_path, rows, damage):
    w = World(tmp_path, repos=("r1", "r2"))
    first = RF.refresh(SYSTEM, **w.kw(make_target()[0]))
    path = atrest.unit_state_path(SYSTEM, VICTIM, w.work)
    commit_id = first.heads["r1"]
    good = {"status": "stale-kept", "builtCommit": commit_id, "failedAt": NOW_Z, "failReason": "extract:timeout"}
    bad = {"garbage": None, "extra-field": {**good, "note": "SECRET-text"}, "unknown-code": {**good, "failReason": "boom:unlisted"},
           "marker-without-status": {**good, "status": "fresh"}, "other-build": {**good, "builtCommit": "0" * 40},
           "link": None, "empty": None, "bad-stamp": {**good, "failedAt": "yesterday"}}[damage]
    path.unlink()
    if damage == "garbage":
        path.write_bytes(b"\xff\x00 not json SECRET")
    elif damage == "empty":
        path.write_bytes(b"")
    elif damage == "link":
        path.symlink_to(w.root)
    else:
        path.write_text(json.dumps(bad))
    if not path.is_symlink():
        path.chmod(0o600)                                              # a bad mode alone would fail closed: isolate the damage under test
    rep = RF.refresh(SYSTEM, ["r2"], **w.kw(make_target()[0], now=NOW + 500))
    u = only(rep, VICTIM)
    assert (u.status, u.fail_reason, u.failed_at, u.built_commit) == ("stale-kept", "extract:failed", LATER_Z, commit_id)
    text = "\n".join(rep.lines())
    assert "SECRET" not in text and str(w.root) not in text
    assert u.fail_reason in FAIL_REASONS


def test_a_state_record_open_to_group_or_other_fails_closed(tmp_path, rows):
    w = World(tmp_path, repos=("r1", "r2"))
    RF.refresh(SYSTEM, **w.kw(make_target()[0]))
    atrest.unit_state_path(SYSTEM, VICTIM, w.work).chmod(0o644)
    u = only(RF.refresh(SYSTEM, ["r2"], **w.kw(make_target()[0], now=NOW + 500)), VICTIM)
    assert (u.status, u.fail_reason, u.failed_at) == ("stale-kept", "extract:failed", LATER_Z)


def test_a_damaged_record_of_a_rebuilt_unit_is_ignored_and_replaced(tmp_path, rows):
    w = World(tmp_path, repos=("r1", "r2"))
    RF.refresh(SYSTEM, **w.kw(make_target()[0]))
    path = atrest.unit_state_path(SYSTEM, VICTIM, w.work)
    path.write_bytes(b"garbage")
    path.chmod(0o600)
    rep = RF.refresh(SYSTEM, ["r1"], **w.kw(make_target()[0]))
    assert only(rep, VICTIM).status == "fresh" and rep.state_errors == 0
    assert atrest.read_unit_state(SYSTEM, VICTIM, w.work)["status"] == "fresh"


def test_a_state_record_that_cannot_be_written_is_counted_and_the_report_comes_back(tmp_path, rows):
    w = World(tmp_path, repos=("r1",))
    RF.refresh(SYSTEM, **w.kw(make_target()[0]))
    path = atrest.unit_state_path(SYSTEM, VICTIM, w.work)
    path.unlink()
    path.symlink_to(w.root)                                            # a link where the record goes: the writer refuses it
    target, fakes = make_target()
    rep = RF.refresh(SYSTEM, **w.kw(target))
    assert rep.swap.renamed and rep.state_errors == 1 and "state errors 1" in "\n".join(rep.lines()) and path.is_symlink()


def test_a_unit_with_no_bundle_after_the_refresh_has_no_state_record(tmp_path, rows):
    w = World(tmp_path, repos=("r1",))
    RF.refresh(SYSTEM, **w.kw(make_target()[0]))
    atrest.unit_state_path(SYSTEM, VICTIM, w.work).write_text("stale")
    for p in atrest.bundle_dir(SYSTEM, w.work).glob(VICTIM + ".*" + atrest.BUNDLE_SUFFIX):
        p.unlink()
    RF.refresh(SYSTEM, **w.kw(make_target()[0], build=failing((VICTIM, "extract:failed"))))
    assert not atrest.unit_state_path(SYSTEM, VICTIM, w.work).exists()


# --- the state record in atrest ----------------------------------------------------------------------------------------------------------


def test_the_state_record_round_trips_is_private_and_value_free(tmp_path):
    p = atrest.write_unit_state("s", "u1", "stale-kept", "c" * 40, NOW_Z, "extract:timeout", tmp_path)
    assert atrest.read_unit_state("s", "u1", tmp_path) == {"status": "stale-kept", "builtCommit": "c" * 40, "failedAt": NOW_Z,
                                                           "failReason": "extract:timeout"}
    assert oct(p.stat().st_mode & 0o777) == "0o600" and p.name == "u1" + atrest.STATE_SUFFIX
    assert atrest.read_unit_state("s", "nobody", tmp_path) is None
    atrest.write_unit_state("s", "u1", "fresh", "d" * 40, root=tmp_path)
    assert atrest.read_unit_state("s", "u1", tmp_path)["failReason"] is None
    assert atrest.clear_unit_state("s", "u1", tmp_path) and not atrest.clear_unit_state("s", "u1", tmp_path)


@pytest.mark.parametrize("args", [("stale", "c" * 40, None, None), ("stale-kept", "c" * 40, None, None), ("fresh", "c" * 40, NOW_Z, "extract:failed"),
                                  ("stale-kept", "c" * 40, NOW_Z, "SECRET text"), ("stale-kept", "../x", NOW_Z, "extract:failed"),
                                  ("stale-kept", "c" * 40, "yesterday", "extract:failed")])
def test_the_state_writer_refuses_anything_but_the_closed_value_free_shape(tmp_path, args):
    with pytest.raises(atrest.AtRestError):
        atrest.write_unit_state("s", "u1", *args, root=tmp_path)
    assert not list(atrest.bundle_dir("s", tmp_path).iterdir())


@pytest.mark.parametrize("name", ["../x", "a/b", "a.b", "", ".", "x" * 65, "a\nb"])
def test_a_hostile_unit_name_never_reaches_a_path(tmp_path, name):
    for call in (lambda: atrest.write_unit_state("s", name, "fresh", "c" * 40, root=tmp_path), lambda: atrest.read_unit_state("s", name, tmp_path),
                 lambda: atrest.clear_unit_state("s", name, tmp_path)):
        with pytest.raises(atrest.AtRestError):
            call()


def test_orphan_pruning_removes_the_state_of_a_dead_unit_only_and_pruning_a_unit_keeps_its_state(tmp_path):
    for u in ("live", "dead"):
        atrest.write_unit_state("s", u, "fresh", "c" * 40, root=tmp_path)
    d = atrest.bundle_dir("s", tmp_path)
    (d / "dead.state.json.extra").write_text("x")                      # not a state record's name: never touched
    (d / ".dead.state.json.0123456789abcdef.tmp").write_text("x")
    removed = atrest.prune_orphans("s", {"live"}, tmp_path)
    assert [p.name for p in removed] == ["dead.state.json"] and (d / "live.state.json").exists()
    assert (d / "dead.state.json.extra").exists()
    with pytest.raises(atrest.AtRestError):
        atrest.prune_unit("s", "live", "nobuild", tmp_path)            # nothing to keep: nothing pruned
    assert (d / "live.state.json").exists()


def test_a_dead_units_state_link_is_refused_not_followed(tmp_path):
    d = atrest.bundle_dir("s", tmp_path)
    (d / "dead.state.json").symlink_to(tmp_path)
    with pytest.raises(atrest.AtRestError):
        atrest.prune_orphans("s", set(), tmp_path)
    assert tmp_path.is_dir() and (d / "dead.state.json").is_symlink()


def test_the_sweep_removes_an_old_crashed_state_write_and_leaves_a_young_one(tmp_path):
    d = atrest.bundle_dir("s", tmp_path)
    old, young, other = d / ".u1.state.json.0123456789abcdef.tmp", d / ".u2.state.json.0123456789abcdef.tmp", d / ".u3.state.json.zz.tmp"
    for p in (old, young, other):
        p.write_text("x")
    os.utime(old, (1, 1))
    os.utime(other, (1, 1))
    assert atrest.sweep_stale_tmp("s", 3600, tmp_path, now=100000.0) == [old] and young.exists() and other.exists()


def test_a_full_refresh_prunes_the_state_of_a_unit_that_no_longer_exists(tmp_path, rows):
    w = World(tmp_path, repos=("r1",))
    RF.refresh(SYSTEM, **w.kw(make_target()[0]))
    gone = uid("docs/note.md", "markdown", "r1")
    commit(w.repo("r1"), {"docs/note.md": None})
    RF.refresh(SYSTEM, **w.kw(make_target()[0]))
    assert not atrest.unit_state_path(SYSTEM, gone, w.work).exists() and atrest.unit_state_path(SYSTEM, VICTIM, w.work).exists()


# --- N3: behaviours the first gate found unpinned ---------------------------------------------------------------------------------------


def _timed(report=None):
    return RF._Timed(report or RF.RefreshReport(SYSTEM), "x", lambda: 0.0)


def test_the_memory_sampler_thread_is_stopped_after_a_stage_and_after_an_exception():
    t = _timed()
    with t:
        assert t._mem.is_alive()
    assert not t._mem.is_alive()
    t = _timed()
    with pytest.raises(ZeroDivisionError):
        with t:
            assert t._mem.is_alive()
            1 / 0
    assert not t._mem.is_alive() and len(t.report.stages) == 1


def test_the_stage_waits_for_a_sampler_that_is_mid_sample(monkeypatch):
    """The sampler is inside a slow read when the stage ends: the stage must join it, not just ask it to stop."""
    real = RF._mem_available_kb

    def slow():
        if threading.current_thread() is not threading.main_thread():
            time.sleep(0.2)
        return real()

    monkeypatch.setattr(RF, "_mem_available_kb", slow)
    monkeypatch.setattr(RF, "SAMPLE_INTERVAL_S", 0.001)
    t = _timed()
    with t:
        time.sleep(0.05)                                               # the thread is now inside its first slow sample
    assert not t._mem.is_alive()


def test_a_refresh_leaves_no_sampler_thread_behind_ok_failed_load_or_raising_build(tmp_path):
    w = World(tmp_path, repos=("r1",))
    base = threading.active_count()
    RF.refresh(SYSTEM, **w.kw(make_target()[0]))
    assert threading.active_count() == base
    with pytest.raises(Boom):
        RF.refresh(SYSTEM, **w.kw(make_target(fail_on=fail_at(3))[0]))
    assert threading.active_count() == base

    def boom(*a):
        raise RuntimeError("build bug")

    with pytest.raises(RuntimeError):
        RF.refresh(SYSTEM, **w.kw(make_target()[0], build=boom))
    assert threading.active_count() == base


def test_a_planted_link_where_an_orphan_bundle_was_is_counted_as_a_prune_error_and_the_graph_is_live(tmp_path):
    w = World(tmp_path, repos=("r1",))
    RF.refresh(SYSTEM, **w.kw(make_target()[0]))
    d = atrest.bundle_dir(SYSTEM, w.work)
    (d / ("f" * 32 + ".zz.bundle.jsonl")).symlink_to(w.root)
    target, fakes = make_target()
    rep = RF.refresh(SYSTEM, **w.kw(target))
    assert rep.swap.renamed and rep.prune_errors == 1 and w.root.is_dir()


def test_a_live_unit_with_only_refused_bundles_keeps_them_through_a_full_refresh(tmp_path, monkeypatch):
    w = World(tmp_path, repos=("r1",))
    RF.refresh(SYSTEM, **w.kw(make_target()[0]))
    real = RF.classify_bundle
    monkeypatch.setattr(RF, "classify_bundle", lambda rec, audit: "bundle:unindexable" if rec[0]["unitId"] == VICTIM else real(rec, audit))
    rep = RF.refresh(SYSTEM, **w.kw(make_target()[0]))
    assert [(u.unit_id, u.status, u.fail_reason) for u in rep.not_mapped] == [(VICTIM, "not-mapped", "bundle:unindexable")]
    assert list(atrest.bundle_dir(SYSTEM, w.work).glob(VICTIM + ".*" + atrest.BUNDLE_SUFFIX)) and rep.pruned == 0


def test_a_carried_unit_whose_parts_the_key_cannot_verify_is_stale_kept_but_the_flip_is_not_recorded(tmp_path, monkeypatch, rows):
    w = World(tmp_path, repos=("r1", "r2"))
    RF.refresh(SYSTEM, **w.kw(make_target()[0]))
    real = RF.link
    monkeypatch.setattr(RF, "link", lambda *a, **k: real(*a, **k)._replace(bind_key_mismatch=[f"bind-key-mismatch {VICTIM}"]))
    rep = RF.refresh(SYSTEM, ["r2"], **w.kw(make_target()[0], now=NOW + 500))
    u = only(rep, VICTIM)
    assert (u.status, u.fail_reason, u.failed_at) == ("stale-kept", "bundle:bind-key-check", LATER_Z)
    assert unit_rows(rows.rows)[VICTIM]["failReason"] == "bundle:bind-key-check"
    monkeypatch.setattr(RF, "link", real)                              # the key is right again: nothing sticks
    assert only(RF.refresh(SYSTEM, ["r2"], **w.kw(make_target()[0])), VICTIM).status == "fresh"


def test_a_carried_unit_with_several_refused_bundles_reports_the_newest_ones_code(tmp_path, monkeypatch):
    w = World(tmp_path, repos=("r1", "r2"))
    first = RF.refresh(SYSTEM, **w.kw(make_target()[0]))
    monkeypatch.setattr(RF.atrest, "prune_unit", lambda *a, **k: [])    # keep both builds of the unit
    second_commit = commit(w.repo("r1"), {"conf/app.yaml": "name: app\nport: 9090\n"})
    RF.refresh(SYSTEM, ["r1"], **w.kw(make_target()[0]))
    d = atrest.bundle_dir(SYSTEM, w.work)
    old, new = (d / f"{VICTIM}.{first.heads['r1']}{atrest.BUNDLE_SUFFIX}"), (d / f"{VICTIM}.{second_commit}{atrest.BUNDLE_SUFFIX}")
    assert old.exists() and new.exists()
    os.utime(old, (1, 1))
    codes = {first.heads["r1"]: "bundle:audit-missing", second_commit: "bundle:unindexable"}
    real = RF.classify_bundle
    monkeypatch.setattr(RF, "classify_bundle", lambda rec, audit: codes[rec[0]["builtCommit"]] if rec[0]["unitId"] == VICTIM else real(rec, audit))
    rep = RF.refresh(SYSTEM, ["r2"], **w.kw(make_target()[0]))
    assert [(u.unit_id, u.status, u.fail_reason) for u in rep.not_mapped] == [(VICTIM, "not-mapped", "bundle:unindexable")]
