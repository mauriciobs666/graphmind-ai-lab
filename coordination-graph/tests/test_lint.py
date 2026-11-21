"""`lint`: malformed IDs through wilib.rules, unresolvable `Depends on`, milestone lints."""
import json
from pathlib import Path

import pytest

from coordgraph.lint import item_counts, lint_repo
from coordgraph.shapes import qualify

PROJECT = {"key": "GAL", "legacy_paths": {
    "claude/*/kaizen/plan.md": "$1", "*/docs/BACKLOG.md": "$1", "docs/BACKLOG.md": "cpg"}}


def put(root, rel, text):
    p = Path(root) / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


@pytest.fixture
def repo(tmp_path):
    put(tmp_path, ".work-items/project.json", json.dumps(PROJECT))
    return tmp_path


def item(ident, *fields, title="T"):
    status = "- **Status:** 🔵 proposed\n"
    return f"### {ident} — {title}\n\n{status}" + "".join(f"- **{k}:** {v}\n" for k, v in fields) + "\n"


def lint(repo, backlog="falkor-chat/docs/BACKLOG.md", **kw):
    return lint_repo(repo, [backlog], **kw)


def codes(findings, severity="error"):
    return sorted((f.code, f.message) for f in findings if f.severity == severity)


def test_a_clean_backlog_has_no_findings(repo):
    put(repo, "falkor-chat/docs/BACKLOG.md", "# B\n\n" + item("K-001") + item("GAL-I02-K0001"))
    assert lint(repo) == []


def test_a_shape_valid_but_non_canonical_id_is_reported_malformed_through_wilib_rules(repo):
    """plan3 test 1: GAL-F00042 passes the parser's shape and `lint` calls it malformed."""
    put(repo, "falkor-chat/docs/BACKLOG.md", "# B\n\n" + item("GAL-F00042") + item("GAL-F0042"))
    ((code, msg),) = codes(lint(repo))
    assert code == "malformed-id" and "GAL-F00042" in msg


def test_an_item_under_an_illegal_parent_is_reported(repo):
    put(repo, "falkor-chat/docs/BACKLOG.md", "# B\n\n" + item("GAL-I02-F0001"))
    ((code, msg),) = codes(lint(repo))
    assert code == "malformed-id" and "GAL-I02-F0001" in msg


@pytest.mark.parametrize("ident", ["GAL-I03-F0012-S01-T01", "GAL-I02-K0001", "GAL-I01-D0001", "GAL-F0042-S01"])
def test_the_canonical_scheme_ids_of_the_table_lint_clean(repo, ident):
    put(repo, "falkor-chat/docs/BACKLOG.md", "# B\n\n" + item(ident))
    assert lint(repo) == []


def test_a_scheme_ref_headed_in_two_backlogs_is_an_error_naming_both_paths(repo):
    put(repo, "falkor-chat/docs/BACKLOG.md", "# B\n\n" + item("GAL-F0042"))
    put(repo, "cpg-x/docs/BACKLOG.md", "# B\n\n" + item("GAL-F0042"))
    findings = lint_repo(repo)
    (msg,) = [f.message for f in findings if f.code == "duplicate-ref"]
    assert "falkor-chat/docs/BACKLOG.md" in msg and "cpg-x/docs/BACKLOG.md" in msg and "wi move" in msg


def test_a_duplicate_id_in_one_file_is_an_error(repo):
    put(repo, "falkor-chat/docs/BACKLOG.md", "# B\n\n" + item("K-001") + item("K-001"))
    ((code, _),) = codes(lint(repo))
    assert code == "duplicate-id"


def test_missing_status_and_delivered_marker_are_errors(repo):
    put(repo, "falkor-chat/docs/BACKLOG.md",
        "# B\n\n### K-001 — no status\n\nprose\n\n### K-002 — T\n\n- **Status:** ✅ delivered\n")
    assert [c for c, _ in codes(lint(repo))] == ["delivered-in-backlog", "status"]


# --- Depends on ----------------------------------------------------------------------------

def test_depends_on_resolves_same_backlog_qualified_and_cross_backlog_scheme(repo):
    """plan3 test 7: a bare scheme ID in a `cpg` backlog joins an item in `falkor-chat`, unqualified."""
    put(repo, "falkor-chat/docs/BACKLOG.md", "# B\n\n" + item("K-065") + item("GAL-I02-K0042"))
    put(repo, "docs/BACKLOG.md", "# B\n\n" + item("C-310", ("Depends on", "GAL-I02-K0042, falkor-chat:K-065")))
    assert lint_repo(repo) == []
    assert qualify("GAL-I02-K0042", "cpg") == "GAL-I02-K0042"


def test_a_bare_legacy_ref_means_the_referrers_own_backlog(repo):
    put(repo, "falkor-chat/docs/BACKLOG.md", "# B\n\n" + item("K-065"))
    put(repo, "docs/BACKLOG.md", "# B\n\n" + item("C-310", ("Depends on", "K-065")))
    ((code, msg),) = codes(lint_repo(repo))
    assert code == "unresolvable-dependency" and "K-065" in msg


def test_a_bare_legacy_ref_resolves_to_an_open_item_of_the_referrers_own_backlog(repo):
    """Base 3.6: bare = same backlog. Pins the scope the resolver is asked in (a wrong scope leaves the
    bare ref unqualified-for-this-backlog and the open item unfound)."""
    put(repo, "falkor-chat/docs/BACKLOG.md",
        "# B\n\n" + item("K-065") + item("K-066", ("Depends on", "K-065")))
    assert lint(repo) == []


def test_a_bare_ref_in_a_backlog_with_no_such_item_is_unresolved_even_if_another_backlog_has_it(repo):
    put(repo, "falkor-chat/docs/BACKLOG.md", "# B\n\n" + item("K-001", ("Depends on", "K-065")))
    put(repo, "opencode/docs/BACKLOG.md", "# B\n\n" + item("K-065"))
    ((code, msg),) = codes(lint_repo(repo, ["falkor-chat/docs/BACKLOG.md"]))
    assert code == "unresolvable-dependency" and "K-065" in msg


def test_the_scope_passed_to_the_resolver_is_the_referrers(repo):
    seen = []

    def candidates(token, scope):
        seen.append((token, scope))
        return {qualify(token, scope)}

    put(repo, "opencode/docs/BACKLOG.md",
        "# B\n\n" + item("K-065") + item("K-066", ("Depends on", "K-065")))
    assert lint(repo, "opencode/docs/BACKLOG.md", candidates=candidates) == []
    assert ("K-065", "opencode") in seen and not any(scope == "unscoped" for _, scope in seen)


def test_an_unavailable_wilib_is_an_error_where_it_is_expected_and_a_warning_where_it_is_not(
        repo, monkeypatch):
    """No canonical-ID verdict can be given: visible either way, exit-1 (error) when wilib should exist."""
    from coordgraph import lint as lint_mod
    put(repo, "falkor-chat/docs/BACKLOG.md", "# B\n\n" + item("K-001"))
    monkeypatch.setattr(lint_mod, "load_wilib_rules", lambda: None)
    monkeypatch.setenv("COORDGRAPH_WI_ROOT", str(repo / "nowhere"))
    (f,) = lint(repo)
    assert (f.severity, f.code) == ("error", "wilib-unavailable") and "NOT run" in f.message
    monkeypatch.delenv("COORDGRAPH_WI_ROOT")
    monkeypatch.setattr(lint_mod, "default_wi_root", lambda: repo / "no-such-scripts")
    (f,) = lint(repo)
    assert (f.severity, f.code) == ("warn", "wilib-unavailable")
    monkeypatch.setattr(lint_mod, "default_wi_root", lambda: repo)
    (f,) = lint(repo)
    assert f.severity == "error"


def test_a_dependency_open_nowhere_needs_a_delivery_shaped_history_heading(repo):
    put(repo, "falkor-chat/docs/BACKLOG.md",
        "# B\n\n" + item("K-001", ("Depends on", "K-066, K-067, K-068, none-such")))
    put(repo, "falkor-chat/docs/HISTORY.md",
        "# H\n\n## 2026-10-21 — K-066 closed: x\n\n"
        "## 2026-10-15 — K-067: chunk jobs retry (mechanism delivered; live proof outstanding)\n\n"
        "The fixes are filed as K-068.\n")
    bad = sorted(m for c, m in codes(lint(repo)) if c == "unresolvable-dependency")
    assert len(bad) == 3
    assert any("K-067" in m and "delivered" in m for m in bad)
    assert any("K-068" in m for m in bad) and any("none-such" in m and "unrecognised" in m for m in bad)
    assert not any("K-066" in m for m in bad)


def test_planted_non_canonical_and_free_text_dependencies_are_shown_unresolved(repo):
    """plan3 test 2: stored verbatim by the parser, reported by lint (never silently dropped)."""
    put(repo, "falkor-chat/docs/BACKLOG.md",
        "# B\n\n" + item("K-001", ("Depends on", "GAL-F42, the design doc")))
    msgs = [m for _, m in codes(lint(repo))]
    assert any("GAL-F42" in m for m in msgs) and any("the design doc" in m for m in msgs)


def test_an_injected_candidate_function_resolves_a_renamed_dependency(repo):
    """The registry-backed read side (S1d) plugs in here: an old name joins the new heading."""
    put(repo, "falkor-chat/docs/BACKLOG.md",
        "# B\n\n" + item("GAL-I02-K0042") + item("K-002", ("Depends on", "falkor-chat:K-065")))
    assert [c for c, _ in codes(lint(repo))] == ["unresolvable-dependency"]

    def candidates(token, scope):
        base = {token, qualify(token, scope)}
        return base | {"GAL-I02-K0042"} if "falkor-chat:K-065" in base else base

    assert lint(repo, candidates=candidates) == []


# --- milestones -----------------------------------------------------------------------------

def test_items_of_one_milestone_must_agree_on_started_and_a_wip_item_needs_it(repo):
    put(repo, "falkor-chat/docs/BACKLOG.md", "# B\n\n"
        + item("K-001", ("Milestone", "M7 — started 2026-11-01"))
        + item("K-002", ("Milestone", "M7"))
        + "### K-003 — wip\n\n- **Status:** 🟡 in-progress\n- **Milestone:** M8\n")
    assert [c for c, _ in codes(lint(repo))] == ["milestone-started", "milestone-started"]


MAP = ("<!-- coordgraph:milestone-map -->\n| Milestone | Items |\n|---|---|\n"
       "| **M2.5 — Hardening** | K-016 → K-017, K-099 |\n<!-- /coordgraph:milestone-map -->\n\n")


def test_milestone_map_rows_and_item_fields_must_agree_open_ids_only(repo):
    put(repo, "falkor-chat/docs/BACKLOG.md", "# B\n\n" + MAP
        + item("K-016", ("Milestone", "M2.5")) + item("K-017", ("Milestone", "M3"))
        + item("K-018", ("Milestone", "M2.5")))
    msgs = [m for c, m in codes(lint(repo)) if c == "milestone-map"]
    assert len(msgs) == 2  # K-017 named in M2.5's row but has M3; K-018 has M2.5 but is absent from the row
    assert any("K-017" in m for m in msgs) and any("K-018" in m for m in msgs)  # K-099 (no open item) ignored


def test_a_clean_map_and_a_file_without_markers_are_fine(repo):
    put(repo, "falkor-chat/docs/BACKLOG.md", "# B\n\n" + MAP.replace("K-099", "K-018")
        + item("K-016", ("Milestone", "M2.5")) + item("K-017", ("Milestone", "M2.5"))
        + item("K-018", ("Milestone", "M2.5")))
    assert lint(repo) == []


def test_item_counts_per_file(repo):
    put(repo, "falkor-chat/docs/BACKLOG.md", "# B\n\n" + item("K-001") + item("K-002"))
    put(repo, "docs/BACKLOG.md", "# B\n\n" + item("C-310"))
    put(repo, "mcp-monitor/docs/BACKLOG.md", "# B\n\n- **bullet.** no ids\n")
    assert item_counts(repo) == {"docs/BACKLOG.md": 1, "falkor-chat/docs/BACKLOG.md": 2,
                                 "mcp-monitor/docs/BACKLOG.md": 0}


# --- ledger `unknown` statusToken (plan S1: "every unknown statusToken in a non-archived ledger") ---

LEDGER = "falkor-chat/docs/plans/x-coordination.md"


def ledger(status_line, *cells):
    rows = "".join(f"| U{i} | {c} |\n" for i, c in enumerate(cells, 1))
    return f"# L\n\n{status_line}\n\n| Unit | Status |\n|---|---|\n{rows}"


ACTIVE = "> **Status:** active · **Owner:** `teco` · **Tracks:** —"
ARCHIVED = "> **Status:** archived · **Owner:** `teco` · **Tracks:** —"


def unit_errors(repo, **kw):
    return [f for f in lint_repo(repo, **kw) if f.code == "unknown-status"]


def test_an_unknown_status_in_an_active_ledger_is_an_error(repo):
    put(repo, LEDGER, ledger(ACTIVE, "delivered", "mystery"))
    (f,) = unit_errors(repo)
    assert (f.severity, f.path) == ("error", LEDGER)
    assert "U2" in f.message and "mystery" in f.message


def test_every_unknown_unit_is_reported_across_tables(repo):
    text = ledger(ACTIVE, "bogus") + "\n| Unit | Status |\n|---|---|\n| V1 | nonsense |\n"
    put(repo, LEDGER, text)
    assert len(unit_errors(repo)) == 2


def test_an_unknown_status_in_an_archived_ledger_is_silent(repo):
    put(repo, LEDGER, ledger(ARCHIVED, "mystery"))
    assert unit_errors(repo) == []


def test_known_statuses_in_an_active_ledger_are_silent(repo):
    put(repo, LEDGER, ledger(ACTIVE, "queued", "✅ delivered", "in-flight", "done"))
    assert unit_errors(repo) == []


def test_a_ledger_without_a_header_status_counts_as_live(repo):
    put(repo, LEDGER, ledger("", "mystery"))
    assert len(unit_errors(repo)) == 1


def test_an_odd_header_status_is_not_the_archived_exemption(repo):
    put(repo, LEDGER, ledger("> **Status:** Archived-ish · **Owner:** `teco`", "mystery"))
    assert len(unit_errors(repo)) == 1


def test_ledger_findings_respect_the_paths_filter(repo):
    put(repo, LEDGER, ledger(ACTIVE, "mystery"))
    assert unit_errors(repo, paths=["falkor-chat/docs/BACKLOG.md"]) == []
    assert len(unit_errors(repo, paths=[LEDGER])) == 1


def no_unit_findings(repo, **kw):
    return [f for f in lint_repo(repo, **kw) if f.code == "no-units"]


def test_a_ledger_whose_table_does_not_parse_is_a_lint_warning_not_an_error(repo):
    put(repo, LEDGER, f"# L\n\n{ACTIVE}\n\n| Unit | Status |\n| U1 | queued |\n")
    (f,) = no_unit_findings(repo)
    assert (f.severity, f.path) == ("warn", LEDGER)
    assert [x for x in lint_repo(repo) if x.severity == "error"] == []


def test_a_parsed_or_prose_only_ledger_has_no_no_units_warning(repo):
    put(repo, LEDGER, ledger(ACTIVE, "queued"))
    put(repo, "falkor-chat/docs/plans/y-coordination.md", f"# L\n\n{ACTIVE}\n\nprose only\n")
    assert no_unit_findings(repo) == []


def test_the_no_units_warning_also_applies_to_an_archived_ledger_and_respects_paths(repo):
    put(repo, LEDGER, f"# L\n\n{ARCHIVED}\n\n| Unit | Status |\n| U1 | queued |\n")
    assert len(no_unit_findings(repo)) == 1
    assert no_unit_findings(repo, paths=["falkor-chat/docs/BACKLOG.md"]) == []
