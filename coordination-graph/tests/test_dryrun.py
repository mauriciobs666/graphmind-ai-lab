"""Offline tests of `rebuild --dry-run` (graph note v4 section 5.5 step 1; plan S3).

Two levels: tiny repos on disk through the real parsers (the end-to-end path), and hand-built
`Load`s for the inputs a parser would never produce (a crashing key, a cross-file duplicate).
"""
import copy
import hashlib
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from coordgraph import commands_store, dryrun
from coordgraph.dryrun import Load, SourceLoad, validate

REPO = Path(__file__).resolve().parents[2]

LEDGER = """# L

> **Status:** active · **Owner:** `teco` · **Tracks:** —

| Unit | Owner | Status | Deliverable |
|---|---|---|---|
| U1 first thing | coder | **accepted** | x |
| U2 | coder | **relayed** | "why?" — relayed 2026-11-01 |

| Unit | Owner |
|---|---|
| N1 | coder |
"""
ARCHIVED = LEDGER.replace("**Status:** active", "**Status:** archived")


def write(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def codes(report) -> list[str]:
    return [v.code for v in report.violations]


def clean_ledger(path="a/docs/plans/x-coordination.md") -> SourceLoad:
    return SourceLoad("ledger", path, {
        "path": path, "kind": "ledger", "component": "a", "slug": "x", "sha256": "0" * 64, "bytes": 10,
        "parserVersion": 1, "warnings": [], "rowCount": 1, "citedReviews": [], "resolvedReviews": [],
    }, [{
        "key": f"{path}#U1", "unitKey": "U1", "unitLabel": "U1", "statusToken": "accepted",
        "ledgerPath": path, "slug": "x", "component": "a", "tableIdx": 1, "rowIdx": 1,
        "unit": "U1", "cites": [], "rowHash": "h", "statusText": "accepted",
    }])


def clean_backlog(path="a/docs/BACKLOG.md", ref="GAL-F0042") -> SourceLoad:
    return SourceLoad("backlog", path, {
        "path": path, "kind": "backlog", "component": "a", "slug": "BACKLOG", "sha256": "1" * 64, "bytes": 10,
        "parserVersion": 1, "warnings": [], "rowCount": 1, "scope": "a",
    }, [{"ref": ref, "itemId": ref, "scope": "a", "backlogPath": path, "ord": 1, "itemType": "F",
         "title": "t", "open": True, "itemHash": "h"}])


def clean_review(path="a/docs/reviews/x.md") -> SourceLoad:
    return SourceLoad("review", path, {
        "path": path, "kind": "review", "component": "a", "slug": "x", "sha256": "2" * 64, "bytes": 3,
        "parserVersion": 1, "warnings": [], "rowCount": 0, "latestPass": 1, "verdictClass": "approve",
    })


def one(*sources: SourceLoad) -> Load:
    return Load(sources=list(sources))


# --- end to end through the real parsers -------------------------------------------------------

def test_tiny_repo_is_clean_and_reports_everything(tmp_path):
    write(tmp_path, "a/docs/plans/x-coordination.md", LEDGER)
    write(tmp_path, "a/docs/plans/old-coordination.md", ARCHIVED)
    report = dryrun.rebuild_dry_run(tmp_path)
    assert report.ok, report.render()
    assert report.kind_counts == {"ledger": 2}
    assert report.status_tally == {"accepted": 2, "paused": 2, "unknown": 2}
    raw = (tmp_path / "a/docs/plans/x-coordination.md").read_bytes()
    assert ("a/docs/plans/x-coordination.md", hashlib.sha256(raw).hexdigest()) in report.manifest
    # the one unknown raw value (no Status column -> statusText None) and the warning are printed
    assert {(p, k.split("#")[1]) for p, k, _ in report.unknown_values} >= {("a/docs/plans/x-coordination.md", "N1")}
    assert any("no Status column" in w for _, w in report.warnings)
    text = report.render()
    assert "statusToken tally" in text and "dry-run: ok" in text


def test_a_backlog_item_scheme_ref_headed_in_two_files_is_named_with_both_paths(tmp_path):
    body = "# B\n\n### GAL-F0042 — Thing\n\n- **Status:** 🔵 proposed\n"
    write(tmp_path, "a/docs/BACKLOG.md", body)
    write(tmp_path, "b/docs/BACKLOG.md", body)
    report = dryrun.rebuild_dry_run(tmp_path, {"a/docs/BACKLOG.md": "a", "b/docs/BACKLOG.md": "b"})
    dup = [v for v in report.violations if v.code == "duplicate-unique"]
    assert len(dup) == 1
    assert ("scheme ref GAL-F0042 headed in both a/docs/BACKLOG.md (ord 1) and b/docs/BACKLOG.md (ord 1)"
            in dup[0].message)
    assert "a wi move may be in progress" in dup[0].message


def test_a_duplicate_item_inside_one_file_is_a_parse_error_not_a_crash(tmp_path):
    body = ("# B\n\n### GAL-F0042 — Thing\n\n- **Status:** 🔵 proposed\n\n"
            "### GAL-F0042 — Again\n\n- **Status:** 🔵 proposed\n")
    write(tmp_path, "a/docs/BACKLOG.md", body)
    report = dryrun.rebuild_dry_run(tmp_path, {"a/docs/BACKLOG.md": "a"})
    assert [v.code for v in report.violations if v.code == "parse-error"] == ["parse-error"]


def test_the_dry_run_opens_no_socket_and_loads_no_database_client(tmp_path, monkeypatch):
    write(tmp_path, "a/docs/plans/x-coordination.md", LEDGER)

    def boom(*a, **k):
        raise AssertionError("network used")

    monkeypatch.setattr(socket.socket, "connect", boom)
    monkeypatch.setattr(socket, "create_connection", boom)
    assert dryrun.rebuild_dry_run(tmp_path).ok
    code = ("import sys, coordgraph.dryrun as d; d.rebuild_dry_run('.'); "
            "sys.exit('falkordb' in sys.modules)")
    assert subprocess.run([sys.executable, "-c", code], cwd=tmp_path, check=False,
                          env={"PYTHONPATH": str(REPO / "coordination-graph")}).returncode == 0


def test_command_exit_codes_and_manifest(tmp_path, capsys):
    write(tmp_path, "a/docs/plans/x-coordination.md", LEDGER)
    manifest = tmp_path / "manifest.txt"
    args = type("A", (), {"repo_root": tmp_path, "manifest": str(manifest)})()
    assert commands_store.rebuild_dry_run(args) == 0
    assert "a/docs/plans/x-coordination.md" in manifest.read_text()
    assert "dry-run: ok" in capsys.readouterr().out
    write(tmp_path, "a/docs/BACKLOG.md", "# B\n\n### GAL-F0042 — T\n\n- **Status:** 🔵 proposed\n")
    write(tmp_path, "b/docs/BACKLOG.md", "# B\n\n### GAL-F0042 — T\n\n- **Status:** 🔵 proposed\n")
    assert commands_store.rebuild_dry_run(args) == 1


# --- crashing keys (note 4.1): hand-built, a parser would not emit them --------------------------

CRASHING = ["set ", " a", "a\t", "a\n", "a\r", "a\v", "a\f"]


@pytest.mark.parametrize("bad", CRASHING + ["k" * 513])
def test_a_crashing_unit_key_is_reported_by_path_and_key(bad):
    s = clean_ledger()
    s.rows[0]["key"] = bad
    report = validate(one(s))
    hit = [v for v in report.violations if v.code == "unique-key-unsafe"]
    assert hit and "Unit.key" in hit[0].message and s.path in hit[0].where


@pytest.mark.parametrize("bad", CRASHING + ["k" * 513])
def test_a_crashing_source_path_and_item_ref_are_reported(bad):
    s = clean_backlog()
    s.rows[0]["ref"] = bad
    assert "unique-key-unsafe" in codes(validate(one(s)))
    r = clean_review()
    r.meta["path"] = bad
    r.path = bad
    assert "unique-key-unsafe" in codes(validate(one(r)))


def test_kb_keys_are_checked(tmp_path):
    (tmp_path / "coordination-graph").mkdir()
    (tmp_path / "coordination-graph" / "kb-files.txt").write_text("a/kb.md\n b.md \n", encoding="utf-8")
    write(tmp_path, "a/kb.md", "## One\n\ntext\n\n## One\n\nmore\n")
    report = dryrun.rebuild_dry_run(tmp_path)
    assert "kb-file-missing" in codes(report)  # `b.md` is not on disk
    assert report.ok is False and any(p == "a/kb.md" for p, _ in report.manifest)


def test_an_unsafe_kb_section_key_or_file_path_is_reported():
    load = one()
    load.kb = [dryrun.KbLoad("a.md ", "s", ["a.md#x "])]
    got = [v.message.split()[0] for v in validate(load).violations if v.code == "unique-key-unsafe"]
    assert sorted(got) == ["KbFile.path", "KbSection.key"]


def test_duplicate_kb_section_key_and_file_path():
    load = one()
    load.kb = [dryrun.KbLoad("a.md", "s", ["a.md#x", "a.md#x"]), dryrun.KbLoad("a.md", "s", ["a.md#y"])]
    found = codes(validate(load))
    assert found.count("duplicate-unique") == 2


# --- cross-load duplicates ---------------------------------------------------------------------

def test_cross_file_scheme_ref_duplicate_message():
    a, b = clean_backlog("a/docs/BACKLOG.md"), clean_backlog("b/docs/BACKLOG.md")
    v = [v for v in validate(one(a, b)).violations if v.code == "duplicate-unique"]
    assert len(v) == 1
    assert v[0].message.startswith(
        "scheme ref GAL-F0042 headed in both a/docs/BACKLOG.md (ord 1) and b/docs/BACKLOG.md (ord 1)")
    assert v[0].message.endswith("a wi move may be in progress")


def test_legacy_ref_duplicate_across_files_is_not_called_a_scheme_ref():
    a, b = clean_backlog("a/docs/BACKLOG.md", "a:K-001"), clean_backlog("b/docs/BACKLOG.md", "a:K-001")
    for s in (a, b):
        s.rows[0].pop("itemType")
        s.rows[0]["itemId"] = "K-001"
    v = [v for v in validate(one(a, b)).violations if v.code == "duplicate-unique"]
    assert len(v) == 1 and v[0].message.startswith("ref a:K-001 headed in both")


def test_same_ref_twice_in_one_file_and_duplicate_unit_key_and_path():
    s = clean_backlog()
    s.rows.append(dict(s.rows[0], ord=2))
    s.meta["rowCount"] = 2
    assert "duplicate-unique" in codes(validate(one(s)))
    a, b = clean_ledger("a/docs/plans/x-coordination.md"), clean_ledger("b/docs/plans/x-coordination.md")
    b.rows[0]["key"] = a.rows[0]["key"]
    assert "duplicate-unique" in codes(validate(one(a, b)))
    assert "duplicate-unique" in codes(validate(one(clean_review(), clean_review())))


# --- indexed values ----------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [" GAL-F1", "GAL-F1\n", "", "x" * 513, "two words"])
def test_bad_indexed_values_are_reported(bad):
    r = clean_review()
    r.meta["itemId"] = bad
    assert codes(validate(one(r))) and codes(validate(one(r)))[0] in ("indexed-value-bad", "text-on-indexed-property")
    s = clean_ledger()
    s.rows[0]["itemId"] = bad
    assert codes(validate(one(s)))


def test_status_token_outside_the_closed_set_is_text_on_an_index():
    s = clean_ledger()
    s.rows[0]["statusToken"] = "accepted — APPROVE WITH SUGGESTIONS"
    assert "text-on-indexed-property" in codes(validate(one(s)))


def test_backlog_scope_and_ledger_slug_are_checked():
    b = clean_backlog()
    b.meta["scope"] = " a"
    assert "indexed-value-bad" in codes(validate(one(b)))
    s = clean_ledger()
    s.meta["slug"] = "x "
    assert "indexed-value-bad" in codes(validate(one(s)))


# --- property-set invariants ---------------------------------------------------------------------

def test_clean_hand_built_load_is_ok():
    assert validate(one(clean_ledger(), clean_backlog(), clean_review())).ok


@pytest.mark.parametrize("kind_factory", [clean_ledger, clean_backlog, clean_review])
@pytest.mark.parametrize("name", ["warnings", "rowCount"])
def test_every_source_must_carry_warnings_and_rowcount(kind_factory, name):
    s = kind_factory()
    del s.meta[name]
    hit = [v for v in validate(one(s)).violations if v.code == "missing-property" and name in v.message]
    assert hit and s.path in hit[0].where


def test_warnings_and_rowcount_shapes_and_agreement():
    s = clean_ledger()
    s.meta["warnings"] = "none"
    s.meta["rowCount"] = 3
    got = codes(validate(one(s)))
    assert "bad-shape" in got and "rowcount-mismatch" in got


def test_unit_label_must_be_derived_and_nonempty_with_the_key():
    s = clean_ledger()
    s.rows[0].update(unitKey="U1 K-091 band write", unitLabel="U1 K-091 band write")
    assert "unit-label-wrong" in codes(validate(one(s)))
    s.rows[0]["unitLabel"] = "U1"
    assert validate(one(s)).ok
    for key, label in [("U3:", "U3"), ("U1(a)", "U1"), ("U1;", "U1"), ("(x) y", "(x) y"), ("U2.3 between", "U2.3")]:
        s.rows[0].update(unitKey=key, unitLabel=label)
        assert validate(one(s)).ok, key
    s.rows[0].update(unitKey="U3:", unitLabel="U3:")  # "equals unitKey when no space" is not the rule
    assert "unit-label-wrong" in codes(validate(one(s)))
    s.rows[0].update(unitKey="U1", unitLabel="")
    assert "unit-label-wrong" in codes(validate(one(s)))
    del s.rows[0]["unitLabel"]
    assert "missing-property" in codes(validate(one(s)))


def test_no_status_column_row_has_no_statustext_and_reads_unknown():
    s = clean_ledger()
    s.meta["warnings"] = ["table 1: no Status column"]
    s.rows[0].update(statusToken="unknown")
    assert "status-text-present" in codes(validate(one(s)))  # statusText 'accepted' present
    s.rows[0]["statusText"] = ""
    assert "status-text-present" in codes(validate(one(s)))
    del s.rows[0]["statusText"]
    assert validate(one(s)).ok
    s.rows[0]["statusToken"] = "accepted"
    assert "status-token-wrong" in codes(validate(one(s)))


def test_slug_is_derived_per_kind_a_requirement_keeps_its_graph_suffix():
    req = SourceLoad("requirement", "a/docs/requirements/coordination-graph.md", {
        "path": "a/docs/requirements/coordination-graph.md", "kind": "requirement", "component": "a",
        "slug": "coordination-graph", "sha256": "3" * 64, "bytes": 1, "parserVersion": 1,
        "warnings": [], "rowCount": 0}, [])
    assert validate(one(req)).ok
    req.meta["slug"] = "coordination"
    assert "slug-wrong" in codes(validate(one(req)))
    led = clean_ledger("a/docs/plans/coordination-graph-coordination.md")
    led.meta["slug"] = "coordination-graph"
    led.rows[0]["slug"] = "coordination-graph"
    assert validate(one(led)).ok
    led.meta["slug"] = "coordination-graph-coordination"
    assert "slug-wrong" in codes(validate(one(led)))
    rev = clean_review("a/docs/reviews/x-impl.md")
    rev.meta["slug"] = "x"
    assert validate(one(rev)).ok


def test_minimal_unit_of_an_archived_ledger_carries_no_cell_text():
    s = clean_ledger()
    s.meta["docStatus"] = "archived"
    assert {"minimal-unit-extra"} == {v.code for v in validate(one(s)).violations}  # unit/cites/rowHash/statusText
    for k in ("unit", "cites", "rowHash", "statusText"):
        s.rows[0].pop(k)
    assert validate(one(s)).ok
    s.rows[0]["needs"] = ["T1"]
    assert "minimal-unit-extra" in codes(validate(one(s)))


def test_list_and_enum_shapes():
    s = clean_ledger()
    s.rows[0].update(needs=["T1"], needsError="boom")
    assert "bad-shape" in codes(validate(one(s)))
    s.rows[0].pop("needsError")
    s.rows[0]["needs"] = "T1"
    assert "bad-shape" in codes(validate(one(s)))
    b = clean_backlog()
    b.rows[0]["itemType"] = "X"
    assert "bad-shape" in codes(validate(one(b)))
    b = clean_backlog()
    b.rows[0]["dependsOn"] = [1]
    assert "bad-shape" in codes(validate(one(b)))
    r = clean_review()
    r.meta["aliases"] = "x"
    assert "bad-shape" in codes(validate(one(r)))


def test_only_note_properties_are_allowed():
    b = clean_backlog()
    b.meta["itemCount"] = 1  # the pre-note name: no longer a property
    b.rows[0]["milestoneOrder"] = 2
    got = [v for v in validate(one(b)).violations if v.code == "unknown-property"]
    assert {"itemCount", "milestoneOrder"} <= {v.message.split()[0] for v in got}


def test_item_ref_form():
    b = clean_backlog(ref="a:GAL-F0042")  # scheme ID qualified: wrong
    b.rows[0]["itemId"] = "GAL-F0042"
    assert "ref-form" in codes(validate(one(b)))
    legacy = clean_backlog(ref="K-001")  # legacy ID unqualified: wrong
    legacy.rows[0]["itemId"] = "K-001"
    legacy.rows[0].pop("itemType")
    assert "ref-form" in codes(validate(one(legacy)))


def test_label_coverage_probe_fires_when_a_holder_is_not_returned():
    s = clean_ledger()
    second = dict(s.rows[0], key=s.rows[0]["key"] + "~2", unitKey="U1 gate", unitLabel="U9", rowIdx=2)
    s.rows.append(second)
    s.meta["rowCount"] = 2
    got = validate(one(s))
    assert "label-probe" in codes(got) and "unit-label-wrong" in codes(got)
    second["unitLabel"] = "U1"
    assert validate(one(s)).ok


def test_parameter_size_bound(monkeypatch):
    monkeypatch.setattr(dryrun, "PARAM_LIMIT", 50)
    got = validate(one(clean_ledger()))
    assert {"$meta", "$rows"} <= {v.message.split()[0] for v in got.violations if v.code == "parameter-too-large"}


def test_status_tally_and_unknown_values_are_listed():
    s = clean_ledger()
    s.rows.append(dict(copy.deepcopy(s.rows[0]), key=s.rows[0]["key"] + "~2", rowIdx=2, unitKey="U2",
                       unitLabel="U2", statusToken="unknown", statusText="blocked (x)"))
    s.meta["rowCount"] = 2
    report = validate(one(s))
    assert report.status_tally == {"accepted": 1, "unknown": 1}
    assert report.unknown_values == [(s.path, f"{s.path}#U1~2", "blocked (x)")]


# --- the real repo (read-only) ---------------------------------------------------------------

def test_real_repo_dry_run_is_clean():
    report = dryrun.rebuild_dry_run(REPO, _legacy_paths())
    assert report.ok, report.render()[-1500:]


def _legacy_paths():
    from coordgraph.scope import load_legacy_paths

    return load_legacy_paths(REPO)
