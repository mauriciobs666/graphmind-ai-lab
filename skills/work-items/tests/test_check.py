"""Step 7: ``wi check`` (plan 3.8; AC-7, AC-14 check half; plan test 17).  tmp_path repos only."""
import json
import subprocess
import sys

import pytest

from conftest import SCRIPTS
from wilib import alloc, check, headers, move, project
from wilib.store import Store

WI = SCRIPTS / "wi.py"
HEAD = "> **Status:** active · **Owner:** `tico` · **Tracks:** —"


def wi(cwd, *args):
    return subprocess.run([sys.executable, str(WI), *args], cwd=cwd, capture_output=True, text=True)


def doc(root, rel, header=HEAD, body="body\n"):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(f"# Title\n\n{header}\n\n{body}", encoding="utf-8")
    return p


def put(root, rel, text):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


@pytest.fixture
def proj(repo):
    project.init(repo, "GAL", ["a", "cpg"])
    Store(repo, "GAL").flush()
    return repo


def findings(root, *a):
    return [(f.kind, f.path) for f in check.check(root, *a) if not f.informational]


def kinds(root, **kw):
    return sorted(f.kind for f in check.check(root, **kw) if not f.informational)


def test_clean_repo_has_no_findings(proj):
    alloc.new(proj, "F", slug="feat")
    Store(proj, "GAL").flush()
    doc(proj, "docs/plans/feat.md", HEAD + " · **Item:** `GAL-F0001`")
    assert check.check(proj) == []
    r = wi(proj, "check")
    assert (r.returncode, r.stdout) == (0, "")


# ------------------------------------------------------------------ AC-7
def test_ac7_set_with_no_id_one_malformed_id_and_two_sharing_one_id_reports_all_three(proj):
    alloc.new(proj, "F")
    alloc.new(proj, "F")
    Store(proj, "GAL").flush()
    s = put(proj, "set.json", json.dumps([
        {"title": "no id here"},
        {"id": "GAL-F01"},
        {"id": "GAL-F0001"}, {"id": "GAL-F0001"}]))
    found = [(f.kind, f.path) for f in check.check(proj, set_file=s)]
    assert found == [("no-id", "set.json#0"), ("malformed", "set.json#1"),
                     ("duplicate", "set.json#2")]
    r = wi(proj, "check", "--set", "set.json")
    assert r.returncode == 1
    assert r.stdout.splitlines() == [
        "no-id: set.json#0: entry has no ID",
        "malformed: set.json#1: GAL-F01: non-canonical segment 'F01' in 'GAL-F01'",
        "duplicate: set.json#2: GAL-F0001 is shared by entries ['set.json#2', 'set.json#3']"]


def test_set_entry_with_a_dangling_id_is_reported(proj):
    s = put(proj, "set.json", '{"items": [{"id": "GAL-F0009"}]}')
    assert [(f.kind) for f in check.check(proj, set_file=s)] == ["dangling"]


def test_set_entry_with_a_legacy_id_is_informational_and_a_non_object_entry_is_no_id(proj):
    s = put(proj, "set.json", json.dumps([{"id": "K-030"}, "GAL-F0001", 7]))
    got = [(f.kind, f.informational) for f in check.check(proj, set_file=s)]
    assert got == [("legacy-id", True), ("no-id", False), ("no-id", False)]
    assert wi(proj, "check", "--set", "set.json").returncode == 1


def test_unreadable_set_file_is_a_clean_error_not_a_traceback(proj):
    put(proj, "bad.json", "{nope")
    r = wi(proj, "check", "--set", "bad.json")
    assert r.returncode == 1 and r.stderr.startswith("wi: cannot read --set file")


def test_two_new_events_with_one_id_are_a_duplicate(proj):
    alloc.new(proj, "F")
    store = Store(proj, "GAL")
    ev = {"evid": "zz00000000000001", "ts": 1, "ev": "new", "id": "GAL-F0001", "type": "F",
          "parent": None, "home": "repo"}
    with open(store.journal_path, "a") as fh:
        fh.write(json.dumps(ev) + "\n")
    got = check.check(proj)
    assert [(f.kind, f.detail) for f in got if f.kind == "duplicate"] == [
        ("duplicate", "GAL-F0001: 2 distinct new events")]


def test_one_item_shared_by_two_slugs_is_a_duplicate(proj):
    alloc.new(proj, "F", slug="one")
    alloc.new(proj, "F", slug="two")
    Store(proj, "GAL").flush()
    doc(proj, "docs/plans/one.md", HEAD + " · **Item:** `GAL-F0001`")
    doc(proj, "docs/plans/two.md", HEAD + " · **Item:** `GAL-F0001`")
    got = check.check(proj)
    assert ("duplicate", "", "Item GAL-F0001 is shared by slugs ['one', 'two']") in \
        [(f.kind, f.path, f.detail) for f in got]


def test_duplicate_registry_lines_are_informational_only(proj):
    alloc.new(proj, "F")
    store = Store(proj, "GAL")
    store.flush()
    lines = store.registry_path.read_text().splitlines()
    store.registry_path.write_text("\n".join(lines + [lines[-1]]) + "\n")
    got = check.check(proj)
    assert [(f.kind, f.informational) for f in got] == [("duplicate-lines", True)]
    assert wi(proj, "check").returncode == 0


# ------------------------------------------------------------------ AC-14 / family
def test_ac14_a_plan_created_after_new_without_item_is_reported_until_synced(proj):
    alloc.new(proj, "F", slug="feat")
    Store(proj, "GAL").flush()
    doc(proj, "docs/requirements/feat.md", HEAD + " · **Item:** `GAL-F0001`")
    doc(proj, "docs/plans/feat.md")
    doc(proj, "docs/reviews/feat.md")
    assert findings(proj) == [("family-no-item", "docs/plans/feat.md"),
                              ("family-no-item", "docs/reviews/feat.md")]
    headers.sync(proj)
    assert check.check(proj) == []


def test_family_mismatch_covers_every_kind_not_only_requirements(proj):
    alloc.new(proj, "I")
    alloc.new(proj, "F", slug="feat")
    move.move(proj, "GAL-F0001", "GAL-I03")
    Store(proj, "GAL").flush()
    doc(proj, "a/docs/test-plans/feat.md", HEAD + " · **Item:** `GAL-F0001`")   # stale: an alias
    got = check.check(proj)
    assert [(f.kind, f.path) for f in got] == [("family-mismatch", "a/docs/test-plans/feat.md")]
    assert "GAL-F0001 differs from registry ID GAL-I03-F0001" in got[0].detail


def test_item_naming_another_slugs_item_is_an_item_slug_mismatch(proj):
    alloc.new(proj, "F", slug="one")
    alloc.new(proj, "F", slug="two")
    Store(proj, "GAL").flush()
    doc(proj, "docs/plans/one.md", HEAD + " · **Item:** `GAL-F0002`")
    doc(proj, "docs/plans/two.md", HEAD + " · **Item:** `GAL-F0002`")
    assert ("item-slug", "docs/plans/one.md") in findings(proj)


def test_new_requirements_document_not_in_baseline_without_item_is_no_id(proj):
    doc(proj, "docs/requirements/old.md")
    (proj / ".work-items" / "baseline.txt").write_text("docs/requirements/old.md\n")
    doc(proj, "docs/requirements/fresh.md")
    assert findings(proj) == [("no-id", "docs/requirements/fresh.md")]


def test_only_a_family_of_a_registered_slug_needs_item(proj):
    doc(proj, "docs/plans/unregistered.md")
    assert check.check(proj) == []


# ------------------------------------------------------------------ malformed / dangling scope
def test_malformed_dangling_in_headers_and_every_header_field_is_scanned(proj):
    alloc.new(proj, "F")
    Store(proj, "GAL").flush()
    doc(proj, "docs/plans/p.md", HEAD.replace("—", "`GAL-F0001`, `GAL-F9`") +
        " · **Aliases:** `GAL-F0007`")
    got = [(f.kind, f.detail.split(":")[0]) for f in check.check(proj)]
    assert got == [("malformed", "Tracks"), ("dangling", "Aliases")]


def test_prose_placeholder_and_legacy_ids_never_fire(proj):
    doc(proj, "docs/plans/p.md", HEAD.replace("—", "C-1001, K-030, `K-055`"),
        body="Use GAL-F<4 digits> and GAL-F01 or GAL-ZZ-9 in prose; see K-030 and C-1001.\n")
    assert check.check(proj) == []


def test_forgotten_flush_shows_as_dangling_only_through_a_missing_event(proj):
    alloc.new(proj, "F", slug="feat")                       # journal only
    doc(proj, "docs/plans/feat.md", HEAD + " · **Item:** `GAL-F0001`")
    got = check.check(proj)
    assert [(f.kind, f.informational) for f in got] == [("unflushed", True)]
    # a branch that cites the ID but whose registry never received the event:
    Store(proj, "GAL").registry_path.write_text("")
    Store(proj, "GAL").journal_path.unlink()
    assert ("dangling", "docs/plans/feat.md") in findings(proj)


def test_fr_ac_citation_present_or_absent_in_the_requirements_document(proj):
    alloc.new(proj, "F", slug="feat")
    Store(proj, "GAL").flush()
    doc(proj, "docs/requirements/feat.md", HEAD + " · **Item:** `GAL-F0001`",
        body="- FR-9 (checkable): x\n- AC-7 (FR-9): y\n- FR-2a: z\n")
    doc(proj, "docs/plans/feat.md",
        HEAD.replace("—", "`GAL-F0001-FR9`, `GAL-F0001-AC7`, `GAL-F0001-FR2a`, `GAL-F0001-FR2`")
        + " · **Item:** `GAL-F0001`")
    got = check.check(proj)
    assert [(f.kind, f.detail.split(":")[1].strip()) for f in got] == [
        ("absent", "GAL-F0001-FR2")]


def test_a_citation_of_an_unknown_feature_requirement_is_dangling(proj):
    doc(proj, "docs/plans/p.md", HEAD.replace("—", "`GAL-F0004-FR1`"))
    assert [f.kind for f in check.check(proj)] == ["dangling"]


def test_registry_item_whose_parent_disagrees_with_its_path_is_reported(proj):
    alloc.new(proj, "I")
    alloc.new(proj, "F", parent="GAL-I03")
    store = Store(proj, "GAL")
    with store.transaction() as tx:                          # a corrupted registry: parent dropped
        tx.append({"ev": "move", "uid": "GAL-I03-F0001", "parent": None,
                   "moves": [{"uid": "GAL-I03-F0001", "old": "GAL-I03-F0001",
                              "new": "GAL-I03-F0001"}]})
    store.flush()
    got = check.check(proj)
    assert [(f.kind, f.detail) for f in got] == [
        ("parent", "GAL-I03-F0001: parent is None, its path says GAL-I03")]


def test_a_path_that_breaks_fr25_in_a_header_is_a_parent_finding(proj):
    doc(proj, "docs/plans/p.md", HEAD.replace("—", "`GAL-F0001-D0001`"))
    assert [f.kind for f in check.check(proj)] == ["parent"]


# ------------------------------------------------------------------ namespaces
@pytest.fixture
def ns(proj):
    cfg = proj / ".work-items" / "project.json"
    data = json.loads(cfg.read_text())
    data["legacy_namespaces"] = {"K-": ["architect"], "C-": ["cpg"]}
    data["legacy_paths"] = {"docs/BACKLOG.md": "cpg", "claude/*/kaizen/plan.md": "$1",
                            "*/docs/BACKLOG.md": "$1"}
    cfg.write_text(json.dumps(data))
    return proj


def test_plan_test17_citing_only_backlog_is_silent_and_a_root_definer_with_missing_home_fires(ns):
    put(ns, "a/docs/BACKLOG.md", "# Backlog\n\nSee K-004 and C-310 in prose.\n- mentions K-005 here\n")
    put(ns, "claude/architect/kaizen/plan.md", "## K-004 — title\n")
    put(ns, "docs/BACKLOG.md", "- **C-310 — thing**\n")
    assert check.check(ns) == []
    cfg = ns / ".work-items" / "project.json"
    data = json.loads(cfg.read_text())
    data["legacy_namespaces"]["C-"] = []
    cfg.write_text(json.dumps(data))
    got = check.check(ns)
    assert [(f.kind, f.path) for f in got] == [("namespace", "docs/BACKLOG.md")]
    assert "home 'cpg' is not in legacy_namespaces" in got[0].detail


@pytest.mark.parametrize("line", ["## K-004 — t", "#### K-004 t", "- **K-004** t", "| K-004 | t |"])
def test_each_definer_shape_fires(ns, line):
    put(ns, "a/docs/BACKLOG.md", line + "\n")
    assert [f.kind for f in check.check(ns)] == ["namespace"]       # home 'a' is not declared for K-


@pytest.mark.parametrize("line", ["##### K-004 t", "# K-004 t", "text K-004", "- K-004 plain bullet"])
def test_non_definer_shapes_stay_silent(ns, line):
    put(ns, "a/docs/BACKLOG.md", line + "\n")
    assert check.check(ns) == []


def test_agent_kaizen_plan_home_comes_from_the_path_and_unmapped_files_are_reported(ns):
    put(ns, "claude/tdd/kaizen/plan.md", "### K-001 x\n")
    put(ns, "claude/architect/kaizen/plan.md", "### K-001 x\n")
    got = check.check(ns)
    assert [(f.kind, f.path) for f in got] == [("namespace", "claude/tdd/kaizen/plan.md")]
    cfg = ns / ".work-items" / "project.json"
    data = json.loads(cfg.read_text())
    del data["legacy_paths"]["claude/*/kaizen/plan.md"]
    cfg.write_text(json.dumps(data))
    assert "no legacy_paths entry" in [f.detail for f in check.check(ns)][0]


def test_paths_argument_limits_the_document_scan(proj):
    alloc.new(proj, "F", slug="a1")
    alloc.new(proj, "F", slug="b1")
    Store(proj, "GAL").flush()
    doc(proj, "docs/plans/a1.md")
    doc(proj, "docs/plans/b1.md")
    assert findings(proj, None, ["docs/plans/a1.md"]) == [("family-no-item", "docs/plans/a1.md")]
    assert len(findings(proj)) == 2


def test_check_after_two_moves_is_clean_once_synced(proj):
    alloc.new(proj, "I")
    alloc.new(proj, "I")
    alloc.new(proj, "F", slug="feat")
    doc(proj, "docs/plans/feat.md")
    move.move(proj, "GAL-F0001", "GAL-I03")
    move.move(proj, "GAL-I03-F0001", "GAL-I04")
    Store(proj, "GAL").flush()
    headers.sync(proj)
    assert check.check(proj) == []


def _lost_then_rebuilt(proj):
    store = Store(proj, "GAL")
    store.journal_path.unlink()
    store.registry_path.unlink()
    r = wi(proj, "rebuild", "--write", "--accept-reissue")
    assert r.returncode == 0, r.stderr


def test_check_reports_rebuilt_informationally_until_the_marker_is_removed(proj):
    _lost_then_rebuilt(proj)
    first = wi(proj, "check")
    assert first.returncode == 0
    assert first.stdout.startswith("info rebuilt: ") and "project.json" in first.stdout
    assert wi(proj, "check").stdout == first.stdout              # every run, not once
    found = check.check(proj)
    assert [(f.kind, f.informational) for f in found] == [("rebuilt", True)]
    cfg = proj / ".work-items" / "project.json"
    data = json.loads(cfg.read_text())
    del data["rebuilt"]
    cfg.write_text(json.dumps(data))
    after = wi(proj, "check")
    assert (after.returncode, after.stdout) == (0, "")


def test_check_without_a_rebuild_has_no_rebuilt_finding(proj):
    assert check.check(proj) == []


# ---------------------------------------------------------------- N5: no private reaches
def test_check_references_no_private_name_of_another_module():
    import re
    src = (SCRIPTS / "wilib" / "check.py").read_text(encoding="utf-8")
    attr = re.findall(r"\b(?:store|headers|rebuild|alloc|events|project|rules)\._\w+", src)
    imported = re.findall(r"from \.\w+ import [^\n]*\b_\w+", src)
    assert attr == [] and imported == []


def test_store_read_union_is_the_reconciled_union_and_reseeds(repo):
    project.init(repo, "GAL", ["a"])
    alloc.new(repo, "F")
    store = Store(repo, "GAL")
    store.flush()
    store.journal_path.unlink()                       # journal lost; tracked copy remains
    union = store.read_union()
    assert [e["ev"] for e in union].count("new") == 3 and store.journal_path.exists()


def test_public_header_helpers_match_the_family_rules(tmp_path):
    assert headers.owner("feat-coordination.md", {"feat", "fe"}) == "feat"
    assert headers.header_index(["# T\n", "\n", "> **Status:** active\n"]) == 2
    root = tmp_path
    (root / "docs" / "plans").mkdir(parents=True)
    (root / "docs" / "plans" / "feat.md").write_text("# T\n")
    assert [p.name for p in headers.family_files(root, "feat", {"feat"})] == ["feat.md"]
