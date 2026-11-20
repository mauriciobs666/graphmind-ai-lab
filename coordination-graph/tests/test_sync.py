"""Offline tests of `sync`, `rebuild`'s parity and `drift` over an in-memory graph and tmp_path trees.

The Cypher is the live suite's (`test_sync_live.py`); this file pins the decisions around it: what is
re-written, what is pruned, what parity compares, what drift labels.
"""
import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from coordgraph import drift, store, sync
from coordgraph.parse.source import PARSER_VERSION
from tests.memgraph import MemGraph

LEDGER = """# L

> **Status:** active · **Owner:** `teco` · **Tracks:** —

| Unit | Owner | Status | Deliverable |
|---|---|---|---|
| U1 first | coder | **accepted** | x |
| U2 | coder | **queued** | y |
"""
BACKLOG = "# B\n\n### K-001 — One\n\n- **Status:** 🔵 proposed\n\n### K-002 — Two\n\n- **Status:** 🔵 proposed\n"
REVIEW = "# R\n\n> **Status:** active · **Owner:** `analyst` · **Tracks:** —\n\n## Pass 1\n\nVerdict: approve\n"
PROJECT = {"key": "GAL", "legacy_paths": {"*/docs/BACKLOG.md": "$1", "docs/BACKLOG.md": "cpg"}}
L = "a/docs/plans/x-coordination.md"
B = "a/docs/BACKLOG.md"
R = "a/docs/reviews/x.md"


def put(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


@pytest.fixture
def repo(tmp_path):
    put(tmp_path, ".work-items/project.json", json.dumps(PROJECT))
    put(tmp_path, L, LEDGER)
    put(tmp_path, B, BACKLOG)
    put(tmp_path, R, REVIEW)
    return tmp_path


def parser(repo) -> sync.Parser:
    return sync.Parser.for_root(repo)


def statuses(outcomes):
    return {o.path: o.status for o in outcomes}


# --- sync ---------------------------------------------------------------------------------------------

def test_sync_all_writes_every_source_then_only_what_changed(repo):
    g = MemGraph()
    assert statuses(sync.sync_all(g, parser(repo))) == {L: "synced", B: "synced", R: "synced"}
    assert statuses(sync.sync_all(g, parser(repo))) == {L: "unchanged", B: "unchanged", R: "unchanged"}
    g.writes.clear()
    put(repo, B, BACKLOG + "\n### K-003 — Three\n\n- **Status:** 🔵 proposed\n")
    assert statuses(sync.sync_changed(g, parser(repo))) == {L: "unchanged", B: "synced", R: "unchanged"}
    assert g.writes == [B]
    assert len(g.sources[B][2]) == 3 and g.sources[B][1]["rowCount"] == 3


def test_a_parser_version_bump_resyncs_an_unchanged_file(repo):
    g = MemGraph()
    sync.sync_all(g, parser(repo))
    g.sources[R][1]["parserVersion"] = PARSER_VERSION - 1
    assert statuses(sync.sync_paths(g, parser(repo), [R, B])) == {R: "synced", B: "unchanged"}


def test_sync_all_force_rewrites_everything(repo):
    g = MemGraph()
    sync.sync_all(g, parser(repo))
    g.writes.clear()
    sync.sync_all(g, parser(repo), force=True)
    assert sorted(g.writes) == sorted([L, B, R])


def test_a_named_path_replaces_only_that_source(repo):
    g = MemGraph()
    out = sync.sync_paths(g, parser(repo), [L])
    assert statuses(out) == {L: "synced"} and list(g.sources) == [L]
    assert [r["unitKey"] for r in g.sources[L][2]] == ["U1 first", "U2"]


def test_a_gone_file_is_deleted_and_an_uncovered_path_is_skipped(repo):
    g = MemGraph()
    sync.sync_all(g, parser(repo))
    (repo / R).unlink()
    put(repo, "a/notes.md", "x")
    out = sync.sync_paths(g, parser(repo), [R, "a/notes.md"])
    assert statuses(out) == {R: "deleted", "a/notes.md": "skipped"}
    assert R not in g.sources


def test_sync_all_prunes_sources_no_longer_on_disk_but_changed_does_not(repo):
    g = MemGraph()
    sync.sync_all(g, parser(repo))
    (repo / R).unlink()
    assert R not in statuses(sync.sync_changed(g, parser(repo))) and R in g.sources
    out = sync.sync_all(g, parser(repo))
    assert statuses(out)[R] == "deleted" and R not in g.sources


def test_one_failing_path_does_not_stop_the_rest_and_keeps_its_previous_state(repo):
    g = MemGraph()
    sync.sync_all(g, parser(repo))
    before = g.sources[L]
    put(repo, L, LEDGER + "| U3 | coder | **queued** | z |\n")
    put(repo, B, BACKLOG + "\n### K-001 — Dup\n\n- **Status:** 🔵 proposed\n")  # duplicate item: unparseable
    g.fail_paths = set()
    out = statuses(sync.sync_all(g, parser(repo)))
    assert out[B] == "failed" and out[L] == "synced"
    assert g.sources[B][2] and len(g.sources[L][2]) == 3 and before is not g.sources[L]


def test_a_store_error_on_one_path_is_reported_not_raised(repo):
    g = MemGraph()
    g.fail_paths = {R}
    out = statuses(sync.sync_all(g, parser(repo)))
    assert out[R] == "failed" and out[L] == "synced"


def test_a_file_that_keeps_changing_is_a_failure_not_a_silent_success(repo):
    g = MemGraph()
    p = parser(repo)
    orig = p.parse

    def churn(path):
        got = orig(path)
        put(repo, path, (repo / path).read_text() + "\n")  # the file moves under every parse
        return got

    p.parse = churn
    out = sync.sync_paths(g, p, [R])
    assert out[0].status == "failed" and "still changing" in out[0].detail


# --- rebuild and parity -----------------------------------------------------------------------------------

@pytest.fixture
def built(repo, monkeypatch):
    monkeypatch.setattr(store, "bootstrap", lambda *a, **k: None)
    g = MemGraph()
    res = sync.rebuild(None, g, "k", parser(repo), now_ms=lambda: 1234)
    return repo, g, res


def test_rebuild_is_an_unconditional_replace_with_a_passing_parity_and_a_stamp(built, monkeypatch):
    repo, g, res = built
    assert res.ok and res.parity.expected == res.parity.actual == 3, res.render()
    assert g.sync_state == {"now": 1234, "pv": PARSER_VERSION}
    g.writes.clear()
    res2 = sync.rebuild(None, g, "k", parser(repo))
    assert sorted(g.writes) == sorted([L, B, R]) and res2.ok


def test_rebuild_prunes_a_source_that_left_the_disk(built):
    repo, g, _ = built
    (repo / R).unlink()
    res = sync.rebuild(None, g, "k", parser(repo))
    assert R not in g.sources and res.ok and f"deleted\t{R}" in res.render()


def test_parity_passes_on_a_graph_whose_properties_came_back_in_another_order(built):
    repo, g, _ = built
    meta = g.sources[L][1]
    g.sources[L] = (g.sources[L][0], dict(reversed(list(meta.items()))), g.sources[L][2])
    assert sync.parity(g, parser(repo)).ok


def mutated(built, mutate):
    repo, g, _ = built
    g = copy.deepcopy(g)  # each case mutates its own copy
    mutate(g)
    return sync.parity(g, parser(repo))


def test_parity_names_a_property_changed_on_one_unit(built):
    p = mutated(built, lambda g: g.sources[L][2][1].__setitem__("statusToken", "paused"))
    assert p.diffs == [f"{L}: child '{L}#U2' differs"]


def test_parity_distinguishes_absent_from_empty_on_a_child(built):
    p = mutated(built, lambda g: g.sources[L][2][0].__setitem__("needs", []))
    assert p.diffs == [f"{L}: child '{L}#U1 first' differs"]


def test_parity_reports_a_dropped_child_with_the_row_count(built):
    p = mutated(built, lambda g: g.sources[L][2].pop())
    assert f"{L}: child '{L}#U2' missing in graph" in p.diffs


def test_parity_reports_an_altered_source_property_a_dropped_warnings_and_a_wrong_hash(built):
    p = mutated(built, lambda g: g.sources[R][1].__setitem__("owner", "`other`"))
    assert p.diffs == [f"{R}: source properties differ"]
    p = mutated(built, lambda g: g.sources[R][1].pop("warnings"))
    assert p.diffs == [f"{R}: source properties differ"]
    p = mutated(built, lambda g: g.sources[B][1].__setitem__("sha256", "0" * 64))
    assert p.diffs == [f"{B}: sha256 {'0' * 64!r}, expected {sync.sha_of(built[0] / B)!r}"]


def test_parity_reports_a_missing_an_extra_and_an_unparseable_source(built):
    repo, g, _ = built
    del g.sources[R]
    g.sources["z/docs/reviews/old.md"] = g.sources[L]
    put(repo, B, BACKLOG + "\n### K-001 — Dup\n\n- **Status:** 🔵 proposed\n")
    p = sync.parity(g, parser(repo))
    assert f"{R}: missing in graph" in p.diffs
    assert "z/docs/reviews/old.md: in graph, not on disk" in p.diffs
    assert any(d.startswith(f"{B}: does not parse") for d in p.diffs)
    assert not p.ok


def test_parity_compares_a_fresh_parse_not_the_stored_hashes(built):
    """A stale `rowHash` stored in the graph must not satisfy parity: children are re-hashed."""
    repo, g, _ = built
    g.sources[L][2][0]["rowHash"] = "stale"
    assert sync.parity(g, parser(repo)).diffs == [f"{L}: child '{L}#U1 first' differs"]


def test_a_nul_in_a_cell_is_scrubbed_in_the_graph_and_parity_still_passes(repo, monkeypatch):
    monkeypatch.setattr(store, "bootstrap", lambda *a, **k: None)
    put(repo, L, LEDGER.replace("| y |", "| y\x00z |"))
    g = MemGraph()
    res = sync.rebuild(None, g, "k", parser(repo))
    assert res.ok, res.render()
    assert "�" in g.sources[L][2][1]["deliverable"]
    assert any("NUL" in w for w in g.sources[L][1]["warnings"])


# --- drift --------------------------------------------------------------------------------------------------

def by_path(report):
    return {d.path: d for d in report.drifts}


def test_drift_is_none_right_after_a_sync(repo):
    g = MemGraph()
    sync.sync_all(g, parser(repo))
    rep = drift.check(g, parser(repo))
    assert rep.clean and rep.source_count == 3
    assert drift.render_header(rep, True).splitlines()[-1] == "Drift: none"


def test_drift_labels_changed_unsynced_and_gone_and_never_repairs(repo):
    g = MemGraph()
    sync.sync_all(g, parser(repo))
    put(repo, L, LEDGER.replace("**queued**", "**accepted**"))
    (repo / R).unlink()
    put(repo, "a/docs/requirements/new.md", "# N\n\n> **Status:** Interviewing · **Owner:** `tico` · **Tracks:** —\n")
    before = dict(g.sources)
    d = by_path(drift.check(g, parser(repo)))
    assert {p: x.state for p, x in d.items()} == {L: "changed", R: "gone", "a/docs/requirements/new.md": "unsynced"}
    assert g.sources == before and not g.writes[3:]


def test_drift_diffs_rows_by_key_changed_added_and_removed(repo):
    g = MemGraph()
    sync.sync_all(g, parser(repo))
    put(repo, L, LEDGER.replace("**queued**", "**accepted**").replace("| U1 first |", "| U1 renamed |")
        + "| U3 | coder | **queued** | z |\n")
    d = by_path(drift.check(g, parser(repo)))[L]
    assert sorted(d.rows) == sorted([(f"{L}#U1 renamed", "added in file"), (f"{L}#U1 first", "removed from file"),
                                     (f"{L}#U2", "changed"), (f"{L}#U3", "added in file")])


def test_drift_restricts_to_the_covered_paths(repo):
    g = MemGraph()
    sync.sync_all(g, parser(repo))
    put(repo, L, LEDGER + "\nmore\n")
    put(repo, B, BACKLOG + "\nmore\n")
    assert list(by_path(drift.check(g, parser(repo), [B]))) == [B]
    assert drift.check(g, parser(repo), [R]).clean


def test_an_unparseable_changed_file_is_flagged_with_the_reason(repo):
    g = MemGraph()
    sync.sync_all(g, parser(repo))
    put(repo, B, BACKLOG + "\n### K-001 — Dup\n\n- **Status:** 🔵 proposed\n")
    d = by_path(drift.check(g, parser(repo)))[B]
    assert d.state == "changed" and "duplicate item" in d.note


def test_header_shape_hook_state_registry_line_and_row_flags(repo):
    g = MemGraph()
    sync.sync_all(g, parser(repo))
    for m in g.sources.values():
        m[1]["syncedAt"] = 1_700_000_000_000
    g.sources[B][1]["syncedAt"] = 1_700_000_100_000
    put(repo, L, LEDGER.replace("**queued**", "**accepted**"))
    rep = drift.check(g, parser(repo))
    text = drift.render_header(rep, False, "Registry: 5 items")
    lines = text.splitlines()
    assert lines[0] == "Synced: 2023-11-14T22:13:20Z – 2023-11-14T22:15:00Z (3 sources) · Hook: disabled"
    assert lines[1] == "Registry: 5 items" and lines[2] == "Drift: 1 source(s)"
    assert lines[3] == f"  changed\t{L}\t→ file wins; run coordgraph sync"
    assert lines[4] == f"    ⚠ drift\t{L}#U2\tchanged"
    assert rep.flagged_rows() == {(L, f"{L}#U2")}
    assert drift.render_header(drift.check(MemGraph(), parser(repo), [R]), True).startswith(
        "Synced: never – never (0 sources) · Hook: active")


# --- the sync path stays offline (plan3 section 6 test 12) -------------------------------------------------

def run_py(code, env=None, cwd=None):
    base = {"PYTHONPATH": ".", "PATH": "", **(env or {})}
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=base,
                          cwd=cwd or str(Path(__file__).resolve().parents[1]))


def test_the_sync_modules_and_cli_import_neither_identity_nor_wilib():
    code = ("import sys\nimport coordgraph.sync, coordgraph.drift, coordgraph.commands_store, coordgraph.cli, "
            "coordgraph.store, coordgraph.dryrun\n"
            "bad = [m for m in ('coordgraph.identity', 'wilib', 'falkordb') if m in sys.modules]\n"
            "assert not bad, bad\n")
    out = run_py(code, {"COORDGRAPH_WI_ROOT": "/nonexistent"})
    assert out.returncode == 0, out.stderr


def test_a_sync_run_with_a_bogus_work_items_root_still_mirrors_and_loads_no_wilib(repo, tmp_path):
    """Red if the sync path imports wilib: the bogus root would break it (or `wilib` shows in sys.modules)."""
    code = ("import sys\nfrom coordgraph import sync, store\nfrom tests.memgraph import MemGraph\n"
            f"g = MemGraph()\nout = sync.sync_all(g, sync.Parser.for_root({str(repo)!r}))\n"
            "assert {o.status for o in out} == {'synced'}, out\n"
            "assert not [m for m in ('coordgraph.identity', 'wilib') if m in sys.modules]\n")
    out = run_py(code, {"COORDGRAPH_WI_ROOT": "/nonexistent"})
    assert out.returncode == 0, out.stderr
