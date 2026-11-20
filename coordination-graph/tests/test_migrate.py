"""`migrate`: the one-time reformat mints scheme IDs only (plan3 section 3.5; section 6, test 10).

HARD FENCE: `--apply` allocates in a work-item registry. Every apply here runs in a scratch
root of its own (a copy of project.json and a freshly `wi init`-ed registry) in `tmp_path`; the
module-wide guard below fails the run if the repo's real `.work-items/` or its journal changed.
"""
import collections
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from coordgraph.discover import discover
from coordgraph.lint import item_counts, lint_repo
from coordgraph.migrate import (ApplyError, TitleReviewRequired, analyze, apply_file, dry_run_report,
                                placeholder_ids, title_issues, wi_flush, wi_minter)
from coordgraph.parse.backlog import parse_backlog
from coordgraph.parse.tables import find_tables
from coordgraph.scope import load_legacy_paths
from coordgraph.shapes import classify

REPO = Path(__file__).resolve().parents[2]
WI = REPO / "skills" / "work-items" / "scripts" / "wi.py"
LEGACY = {"claude/*/kaizen/plan.md": "$1", "*/docs/BACKLOG.md": "$1", "docs/BACKLOG.md": "cpg"}


def _real_registry_state() -> tuple:
    status = subprocess.run(["git", "status", "--short", ".work-items"], cwd=REPO,
                            capture_output=True, text=True, check=True).stdout
    common = subprocess.run(["git", "rev-parse", "--git-common-dir"], cwd=REPO,
                            capture_output=True, text=True, check=True).stdout.strip()
    journal = (REPO / common / "work-items" / "GAL" / "journal.jsonl")
    files = [REPO / ".work-items" / "registry.jsonl", REPO / ".work-items" / "project.json", journal]
    return status, [hashlib.sha256(f.read_bytes()).hexdigest() if f.exists() else None for f in files]


@pytest.fixture(scope="module", autouse=True)
def real_work_items_untouched():
    before = _real_registry_state()
    yield
    assert _real_registry_state() == before, "the repo's real .work-items registry/journal changed"


@pytest.fixture
def scratch(tmp_path):
    root = tmp_path / "scratch"
    (root / ".work-items").mkdir(parents=True)
    assert REPO not in root.resolve().parents
    (root / ".work-items" / "project.json").write_bytes((REPO / ".work-items" / "project.json").read_bytes())
    subprocess.run([sys.executable, str(WI), "init", "--key", "GAL"], cwd=root, check=True, capture_output=True)
    return root


def registry_lines(root):
    return (root / ".work-items" / "registry.jsonl").read_text().splitlines()


def put(root, rel, text):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


def plan(path, text):
    return analyze(path, text)


def apply(root, path):
    """`apply_file` against the scratch registry, titles acknowledged."""
    return apply_file(root, path, wi_minter(root, WI), flush=lambda: wi_flush(root, WI), titles_reviewed=True)


# --- analysis (pure, no allocation) ---------------------------------------------------------

PARKING = (
    "# Kaizen — Improvement Plan: x\n\n## Active\n\n### K-001 — Existing\n\n- **Status:** 🔵 proposed\n\n"
    "## Parking lot / ideas\n"
    "- **Growth-stops pointer: confirm on first live use (2026-10-29).** The U2 pointer is\n"
    "  judgment-enforced.\n"
    "  - nested detail kept\n\n"
    "- **A lead that wraps\n  onto a second line.** Body text.\n"
    "- A bullet with no bold lead at all that goes on and on and on and on and on and on and on forever and ever\n"
)


def test_parking_lot_bullets_become_items_with_ids_to_mint_and_no_text_dropped():
    rf = plan("claude/cobb/kaizen/plan.md", PARKING)
    assert [n.title for n in rf.new_items] == [
        "Growth-stops pointer: confirm on first live use (2026-10-29)",
        "A lead that wraps onto a second line",
        "A bullet with no bold lead at all that goes on and on and on and on and on and on and on",
    ]
    assert {(n.wi_type, n.home, n.origin) for n in rf.new_items} == {("K", "claude", "parking-lot")}
    out = rf.render(placeholder_ids(rf.new_items))
    assert "### <new K #1> — Growth-stops pointer: confirm on first live use (2026-10-29)\n\n" \
           "- **Status:** 🔵 proposed — parking-lot origin; owner confirms status\n\n" \
           "The U2 pointer is\njudgment-enforced.\n- nested detail kept\n" in out
    assert "Body text." in out and "forever and ever" in out  # the no-lead bullet keeps its whole text
    assert "<new K #3>" in out and len(rf.status_defaulted) == 3
    assert "- **Growth-stops" not in out and "Depends on" not in out and "Touches" not in out


def test_component_open_item_bullets_mint_features_homed_in_the_component():
    text = "# Backlog — opencode\n\n## Open\n\n- **Wire the scheduler.** Body.\n- **Second one.** More.\n"
    rf = plan("opencode/docs/BACKLOG.md", text)
    assert [(n.wi_type, n.home, n.origin) for n in rf.new_items] == [("F", "opencode", "open-items")] * 2
    assert rf.new_items[0].command == ["wi", "new", "F", "--home", "opencode", "--title", "Wire the scheduler"]


def test_cpg_root_backlog_homes_in_cpg_and_keeps_bullet_ids_taking_status_from_emoji_or_group():
    text = ("# B\n\n## Open\n\n### In progress 🟡\n\n- **C-1001 — Connected graph.** 🟡 The use cases.\n\n"
            "### Proposed 🔵\n\n- **C-310 — OpenCode wiring.** 🔵 `.mcp.json` only.\n"
            "  more text\n\n### Deferred ⚪\n\n- **C-323 — Bulk repath.**\n  ⚪ **Deliberately deferred.**\n")
    rf = plan("docs/BACKLOG.md", text)
    assert rf.new_items == []  # the IDs exist: kept, never reminted
    out = rf.render([])
    assert "### C-1001 — Connected graph\n\n- **Status:** 🟡 in-progress\n\nThe use cases." in out
    assert "### C-310 — OpenCode wiring\n\n- **Status:** 🔵 proposed\n\n`.mcp.json` only.\nmore text" in out
    assert "### C-323 — Bulk repath\n\n- **Status:** ⚪ deferred" in out
    assert "### Proposed 🔵" in out  # group headings stay
    d = parse_backlog("docs/BACKLOG.md", out.encode(), LEGACY)
    assert [(i.ref, i.status) for i in d.items] == [
        ("cpg:C-1001", "in-progress"), ("cpg:C-310", "proposed"), ("cpg:C-323", "deferred")]


def test_a_bullet_under_a_group_without_its_own_emoji_takes_the_groups():
    rf = plan("docs/BACKLOG.md", "# B\n\n## Open\n\n### Deferred ⚪\n\n- **C-409 — No marker.** body\n")
    assert "- **Status:** ⚪ deferred" in rf.render([]) and rf.status_defaulted == []


def test_id_less_headings_under_an_open_section_are_items_other_headings_are_not():
    text = ("# B\n\n## Open items\n\n### Hand-off prompt parity\n\nThe guard's prompt.\n\n- **Owner:** x\n\n"
            "## Notes\n\n### Not an item\n\nprose\n")
    rf = plan("decision-models/docs/BACKLOG.md", text)
    assert [(n.title, n.origin) for n in rf.new_items] == [("Hand-off prompt parity", "open-heading")]
    out = rf.render(["GAL-F0009"])
    assert "### GAL-F0009 — Hand-off prompt parity\n\n- **Status:** 🔵 proposed — defaulted; owner confirms status" \
           "\n\nThe guard's prompt.\n\n- **Owner:** x" in out
    assert "### Not an item" in out


def test_bullets_outside_an_open_or_parking_section_and_inside_fences_or_items_are_untouched():
    text = ("# B\n\n## Cross-references\n\n- **Tracked elsewhere.** x\n\n## Open items\n\n"
            "```markdown\n- **fenced.** not an item\n```\n\n### GAL-F0001 — Real\n\n- **Status:** 🔵 proposed\n\n"
            "- a body bullet\n")
    rf = plan("mcp-monitor/docs/BACKLOG.md", text)
    assert rf.new_items == [] and rf.render([]) == text


def test_heading_status_paren_moves_into_the_status_field_even_with_nested_parens():
    text = ("# B\n\n## Open follow-ups\n\n### K-079 — Fabricated prices (🟡 in-progress — F1 delivered (2026-10-22), "
            "F2 too)\n\n- **Owner:** `architect`\n\n### K-082 — Rerun (🔵 proposed — carried from K-066)\n\n"
            "> **Why it exists.** quote\n")
    out = plan("falkor-chat/docs/BACKLOG.md", text).render([])
    assert ("### K-079 — Fabricated prices\n\n- **Status:** 🟡 in-progress — F1 delivered (2026-10-22), F2 too\n"
            "- **Owner:** `architect`\n") in out
    assert "### K-082 — Rerun\n\n- **Status:** 🔵 proposed — carried from K-066\n\n> **Why it exists.** quote" in out
    d = parse_backlog("falkor-chat/docs/BACKLOG.md", out.encode(), LEGACY)
    assert [(i.ref, i.status_marker) for i in d.items] == [
        ("falkor-chat:K-079", "🟡 in-progress"), ("falkor-chat:K-082", "🔵 proposed")]


def test_an_existing_status_field_is_never_overwritten_and_an_item_with_none_defaults():
    text = ("# B\n\n### K-001 — Has (🔵 proposed — x)\n\n- **Status:** 🟡 in-progress\n\n"
            "### K-002 — Bare\n\nprose\n")
    rf = plan("falkor-chat/docs/BACKLOG.md", text)
    out = rf.render([])
    assert "### K-001 — Has (🔵 proposed — x)" in out and out.count("- **Status:** 🟡 in-progress") == 1
    assert "### K-002 — Bare\n\n- **Status:** 🔵 proposed — defaulted; owner confirms status\n\nprose" in out
    assert rf.status_defaulted == ["K-002"]


TABLE = (
    "# Kaizen — x\n\n## Active\n\n| ID | Added | Priority | Status | Summary |\n|---|---|---|---|---|\n"
    "| K-001 | 2026-07-09 | high | 🔵 | With a heading |\n"
    "| K-002 | 2026-07-10 | low | 🟡 | Table only |\n\n"
    "### K-001 — With a heading\n\n- **Rationale:** r\n"
)


def test_an_agent_plan_summary_table_is_folded_into_its_items_and_removed():
    rf = plan("claude/tico/kaizen/plan.md", TABLE)
    out = rf.render([])
    assert "| ID |" not in out and rf.new_items == []
    d = parse_backlog("claude/tico/kaizen/plan.md", out.encode(), LEGACY)
    by = {i.item_id: i for i in d.items}
    assert (by["K-001"].priority, by["K-001"].status) == ("high", "proposed")
    assert by["K-001"].fields["Added"] == "2026-07-09"
    assert (by["K-002"].priority, by["K-002"].status, by["K-002"].title) == ("low", "in-progress", "Table only")
    assert by["K-001"].fields["Rationale"] == "r"  # the existing block is kept


MILESTONES = ("# B\n\n## Milestones still open\n\n| Milestone | Reaches ✅ when | Items |\n|---|---|---|\n"
              "| **M2.5 — Hardening** ⚪ *(deferred)* | Real auth | K-016 → K-017, K-018 |\n\n"
              "### K-016 — Auth (🔵 proposed — M2.5, deferred)\n\n### K-017 — Agent path (🔵 proposed)\n\n"
              "- **Milestone:** M9\n\n### K-018 — Push (🔵 proposed)\n")


def test_an_open_milestone_table_puts_milestone_on_its_items_and_gets_the_markers():
    out = plan("falkor-chat/docs/BACKLOG.md", MILESTONES).render([])
    assert "<!-- coordgraph:milestone-map -->\n| Milestone" in out and out.count("<!-- /coordgraph:milestone-map -->") == 1
    d = parse_backlog("falkor-chat/docs/BACKLOG.md", out.encode(), LEGACY)
    assert [(i.item_id, i.milestone) for i in d.items] == [("K-016", "M2.5"), ("K-017", "M9"), ("K-018", "M2.5")]
    assert [i.milestone_order for i in d.items] == [None] * 3  # no order, no approved date: none is recorded


@pytest.mark.parametrize("path,text", [
    ("claude/cobb/kaizen/plan.md", PARKING), ("docs/BACKLOG.md", "# B\n\n## Open\n\n### Proposed 🔵\n\n- **C-310 — x.** y\n"),
    ("claude/tico/kaizen/plan.md", TABLE), ("falkor-chat/docs/BACKLOG.md", MILESTONES),
    ("opencode/docs/BACKLOG.md", "# B\n\n## Open\n\n- **One.** b\n"),
])
def test_analysing_the_output_again_finds_nothing_to_do(path, text):
    rf = plan(path, text)
    ids = [f"GAL-F{n + 1:04d}" for n in range(len(rf.new_items))]
    once = rf.render(ids)
    again = plan(path, once)
    assert again.new_items == [] and again.render([]) == once


def test_reformat_never_emits_a_legacy_id_for_a_minted_item():
    rf = plan("claude/cobb/kaizen/plan.md", PARKING + "\n")
    out = rf.render(placeholder_ids(rf.new_items))
    headings = re.findall(r"^### (\S+)", out, re.M)
    assert [h for h in headings if re.fullmatch(r"[A-Z]-\d+", h)] == ["K-001"]  # the one that existed


# --- dry-run: allocates nothing ---------------------------------------------------------------

def test_dry_run_reports_commands_and_counts_and_allocates_nothing(scratch, monkeypatch, capsys):
    from coordgraph.cli import main
    put(scratch, "opencode/docs/BACKLOG.md", "# B\n\n## Open\n\n- **One.** b\n- **Two.** c\n")
    before = registry_lines(scratch)
    journal = {p: p.read_bytes() for p in (scratch / ".work-items").rglob("*") if p.is_file()}
    monkeypatch.setenv("COORDGRAPH_REPO_ROOT", str(scratch))
    assert main(["migrate", "--dry-run", "opencode/docs/BACKLOG.md"]) == 0
    out = capsys.readouterr().out
    assert "items: 0 now, 2 after the reformat" in out and "new ids to mint: 2" in out
    assert "wi new F --home opencode --title One" in out and "+### <new F #1> — One" in out
    assert registry_lines(scratch) == before
    assert {p: p.read_bytes() for p in (scratch / ".work-items").rglob("*") if p.is_file()} == journal
    assert (scratch / "opencode/docs/BACKLOG.md").read_text().startswith("# B\n\n## Open\n\n- **One.**")


def test_dry_run_report_counts_match_what_apply_then_produces(scratch):
    path = "opencode/docs/BACKLOG.md"
    put(scratch, path, "# B\n\n## Open\n\n- **One.** b\n- **Two.** c\n")
    report = dry_run_report(analyze(path, (scratch / path).read_text()), LEGACY)
    after = int(re.search(r"now, (\d+) after", report).group(1))
    apply(scratch, path)
    assert item_counts(scratch)[path] == after == 2


# --- apply (scratch registry only) ------------------------------------------------------------

def new_ids(before, after):
    """Heading IDs the reformat introduced: not present anywhere in the original (a table row's ID is kept)."""
    return {i for i in re.findall(r"^### (\S+) — ", after, re.M) if not re.search(rf"\b{re.escape(i)}\b", before)}


def test_apply_mints_scheme_ids_only_flushes_and_is_idempotent(scratch):
    path = "claude/cobb/kaizen/plan.md"
    put(scratch, path, PARKING)
    before = (scratch / path).read_text()
    rf = apply(scratch, path)
    after = (scratch / path).read_text()
    minted = new_ids(before, after)
    assert len(rf.new_items) == len(minted) == 3
    assert all(classify(i) == "scheme" and i.startswith("GAL-I02-K") for i in minted)  # `wi new K` -> kaizen
    assert not any(re.fullmatch(r"K-\d+", i) for i in minted)
    reg = "\n".join(registry_lines(scratch))
    assert all(i in reg for i in minted)  # flushed into the tracked registry
    assert lint_repo(scratch, [path]) == []
    again = apply(scratch, path)
    assert again.new_items == [] and (scratch / path).read_text() == after
    assert len(registry_lines(scratch)) == len(reg.splitlines())


def test_apply_allocates_before_writing_so_a_failed_allocation_leaves_the_file_alone(scratch):
    path = "opencode/docs/BACKLOG.md"
    text = "# B\n\n## Open\n\n- **One.** b\n- **Two.** c\n"
    put(scratch, path, text)
    calls = []

    def flaky(item):
        calls.append(item.title)
        if len(calls) == 2:
            raise RuntimeError("wi failed")
        return "GAL-F0001"

    with pytest.raises(RuntimeError):
        apply_file(scratch, path, flaky, titles_reviewed=True)
    assert (scratch / path).read_text() == text and not list((scratch / "opencode/docs").glob(".migrate-*"))


def test_a_failing_wi_surfaces_as_an_error_not_a_blank_id(scratch):
    from coordgraph.migrate import WiError
    put(scratch, "opencode/docs/BACKLOG.md", "# B\n\n## Open\n\n- **One.** b\n")
    with pytest.raises(WiError, match="failed"):
        apply_file(scratch, "opencode/docs/BACKLOG.md", wi_minter(scratch, scratch / "no-such-wi.py"),
                   titles_reviewed=True)


REAL_BACKLOGS = sorted(f.path for f in discover(REPO) if f.kind == "backlog")


@pytest.mark.parametrize("path", REAL_BACKLOGS)
def test_apply_over_a_scratch_copy_of_each_real_backlog(scratch, path):
    """plan3 test 10: only scheme IDs, lint and `wi check` clean, idempotent, counts equal the dry-run's."""
    put(scratch, path, (REPO / path).read_text())
    original = (scratch / path).read_text()
    rf = analyze(path, original)
    dry = dry_run_report(rf, load_legacy_paths(scratch))
    dry_after = int(re.search(r"now, (\d+) after", dry).group(1))
    assert f"new ids to mint: {len(rf.new_items)}" in dry
    apply(scratch, path)
    after = (scratch / path).read_text()
    minted = new_ids(original, after)
    assert all(classify(i) == "scheme" for i in minted)
    assert len(minted) == len(rf.new_items)  # table-only rows add headings that keep their legacy IDs
    assert item_counts(scratch)[path] == dry_after
    assert [f for f in lint_repo(scratch, [path]) if f.severity == "error"] == []
    check = subprocess.run([sys.executable, str(WI), "check"], cwd=scratch, capture_output=True, text=True)
    assert check.returncode == 0, check.stdout + check.stderr
    again = apply(scratch, path)
    assert again.new_items == [] and (scratch / path).read_text() == after


def test_the_real_registry_is_not_where_scratch_runs_allocate(scratch):
    """Pinned separately from the module guard: the scratch registry, not the repo's, takes the events."""
    put(scratch, "opencode/docs/BACKLOG.md", "# B\n\n## Open\n\n- **One.** b\n")
    before = len(registry_lines(scratch))
    apply(scratch, "opencode/docs/BACKLOG.md")
    assert len(registry_lines(scratch)) == before + 1
    assert json.loads(registry_lines(scratch)[-1])["id"].startswith("GAL-F")


# --- Priority and summary text of the removed index table is never dropped (B1, M1) -------------

ROWS = (
    "# Kaizen — x\n\n## Active\n\n| ID | Added | Priority | Status | Summary |\n|---|---|---|---|---|\n"
    "| K-001 | 2026-08-21 | **high — filed upstream** | 🔵 | Systemic hook defect. A second sentence with detail "
    "that goes on and on and on and on well past the title limit. |\n"
    "| K-002 | 2026-09-13 | medium (raised 2026-09-16 from low — see below) | 🔵 | Another row-only item |\n"
    "| K-003 | 2026-09-14 | **blocking** | 🔵 | Odd priority |\n"
    "| K-004 | 2026-09-15 | med | 🔵 | Short |\n"
    "| K-005 | 2026-09-16 | low | 🔵 | Headed row whose summary carries zebra words the item lacks |\n"
    "| K-006 | 2026-09-17 | low | 🔵 | Headed row restated |\n\n"
    "### K-005 — Headed five\n\n- **Rationale:** body without the special words\n\n"
    "### K-006 — Headed six\n\n- **Rationale:** restated: Headed row restated\n"
)


def _out(path, text):
    return plan(path, text).render([])


def test_a_priority_with_a_note_keeps_the_word_and_the_note_in_the_field():
    out = _out("claude/tico/kaizen/plan.md", ROWS)
    assert "- **Priority:** high — filed upstream\n" in out
    assert "- **Priority:** medium — (raised 2026-09-16 from low — see below)\n" in out
    d = parse_backlog("claude/tico/kaizen/plan.md", out.encode(), LEGACY)
    assert {i.item_id: i.priority for i in d.items}["K-001"] == "high"
    assert {i.item_id: i.priority for i in d.items}["K-002"] == "medium"


def test_an_unrecognised_priority_cell_is_emitted_verbatim_and_med_is_medium():
    out = _out("claude/tico/kaizen/plan.md", ROWS)
    assert "- **Priority:** blocking\n" in out  # not skipped: lint warns it unrecognised
    assert "### K-004 — Short\n\n- **Status:** 🔵 proposed\n- **Priority:** medium\n" in out


def test_a_row_only_item_is_titled_by_its_first_sentence_and_keeps_the_whole_summary():
    out = _out("claude/tico/kaizen/plan.md", ROWS)
    assert "### K-001 — Systemic hook defect\n" in out and "…" not in out
    assert ("A second sentence with detail that goes on and on and on and on well past the title limit."
            in out.split("### K-002")[0])
    assert "### K-003 — Odd priority\n\n- **Status:** 🔵 proposed\n- **Priority:** blocking\n" in out
    assert out.count("Odd priority") == 1  # a summary that is its own title is not repeated as a body


def test_a_summary_the_item_does_not_carry_stays_as_a_quote_under_the_fields_one_it_carries_goes():
    out = _out("claude/tico/kaizen/plan.md", ROWS)
    assert ("### K-005 — Headed five\n\n- **Status:** 🔵 proposed\n- **Priority:** low\n- **Added:** 2026-09-16\n"
            "- **Rationale:** body without the special words\n\n"
            "> Summary: Headed row whose summary carries zebra words the item lacks\n") in out
    assert "> Summary: Headed row restated" not in out
    d = parse_backlog("claude/tico/kaizen/plan.md", out.encode(), LEGACY)
    assert [i.fields["Rationale"] for i in d.items if i.item_id == "K-005"] == ["body without the special words"]


def test_a_row_priority_that_differs_from_an_existing_priority_field_is_kept_too():
    text = ("# K\n\n## Active\n\n| ID | Added | Priority | Status | Summary |\n|---|---|---|---|---|\n"
            "| K-001 | 2026-08-21 | **blocking** | 🔵 | Same text as item |\n\n"
            "### K-001 — Same text as item\n\n- **Priority:** low\n")
    out = _out("claude/tico/kaizen/plan.md", text)
    assert "> Index row: Priority blocking" in out and "- **Priority:** low" in out


def test_a_priority_written_inline_on_the_status_line_is_not_added_a_second_time():
    text = ("# K\n\n## Active\n\n| ID | Added | Priority | Status | Summary |\n|---|---|---|---|---|\n"
            "| K-001 | 2026-08-21 | medium | 🔵 | Same text as item |\n\n"
            "### K-001 — Same text as item\n\n- **Status:** 🔵 proposed · **Priority:** medium\n")
    out = _out("claude/tico/kaizen/plan.md", text)
    assert out.count("**Priority:**") == 1 and "> Index row" not in out


_WORD = re.compile(r"\w+")


def _words(text):
    return collections.Counter(w.casefold() for w in _WORD.findall(text))


ALIAS = {"med": "medium"}


@pytest.mark.parametrize("path", REAL_BACKLOGS)
def test_no_word_of_a_real_backlog_is_lost_by_applying_the_reformat(scratch, path):
    """The one-time rewriter is checked by a word-multiset diff, old minus new, over every real input:
    outside the removed summary table nothing is lost (multiset); each removed table cell's words
    (`med` read as `medium`) are all in the item that replaced its row."""
    put(scratch, path, (REPO / path).read_text())
    original = (scratch / path).read_text()
    apply(scratch, path)
    new = (scratch / path).read_text()
    table_lines, rows = set(), []
    for t in find_tables(original):
        names = [h.strip().casefold() for h in t.header]
        if names[:1] == ["id"] and "summary" in names:
            table_lines.update(range(t.row_lines[0] - 3, t.row_lines[-1]))
            rows += [(c[0].strip().strip("`"), c, names) for c in t.rows]
    kept = "\n".join(ln for i, ln in enumerate(original.splitlines()) if i not in table_lines)
    assert dict(_words(kept) - _words(new)) == {}
    for ident, cells, names in rows:
        start = new.index(f"### {ident} —")
        nxt = re.search(r"^#{1,3} ", new[start + 5:], re.M)
        have = set(_words(new[start: start + 5 + nxt.start()] if nxt else new[start:]))
        for col in ("added", "priority", "summary"):
            cell = " ".join(ALIAS.get(w.casefold(), w) for w in _WORD.findall(cells[names.index(col)]))
            assert set(_words(cell)) <= have, (path, ident, col, set(_words(cell)) - have)


# --- titles are checked before an ID is minted (M2) ----------------------------------------------

@pytest.mark.parametrize("title,force_cut,expected", [
    ("Wire the scheduler into the nightly job", False, []),
    ("Something cut off…", False, ["contains an ellipsis"]),
    ("Something cut off...", False, ["contains an ellipsis"]),
    ("(this repo)", False, ["a stub or bare parenthetical, not a title"]),
    ("(optional) Wire the scheduler into the nightly job", False, []),
    ("Short", False, ["a stub or bare parenthetical, not a title"]),
    ("Judged and kept, do not re-litigate (2026-08-24, C6 lint)", False,
     ["reads as a record of past work, not an open item"]),
    ("No `analyst` backlog item opened by U37", False, ["reads as a record of past work, not an open item"]),
    ("Done 2026-09-16", False, ["reads as a record of past work, not an open item"]),
    ("Resolved 2026-09-02 and fixed by deletion", False, ["reads as a record of past work, not an open item"]),
    ("A title that was cut at a word boundary", True, ["cut mid-clause to fit the title length"]),
    ("x" * 130, False, ["over 120 characters"]),
    ("Done right: the guard now fails loudly on bad input", False,
     ["reads as a record of past work, not an open item"]),
    ("Fixture cleanup for the live suite", False, []),
])
def test_title_fitness(title, force_cut, expected):
    assert title_issues(title, force_cut) == expected


def test_the_dry_run_lists_flagged_titles_and_apply_refuses_before_minting_anything(scratch, monkeypatch, capsys):
    from coordgraph.cli import main
    path = "opencode/docs/BACKLOG.md"
    text = ("# B\n\n## Open\n\n- **Wire the scheduler into the nightly job.** Fine.\n"
            "- **(this repo).** a stub\n- **Done 2026-09-16.** a record\n")
    put(scratch, path, text)
    monkeypatch.setenv("COORDGRAPH_REPO_ROOT", str(scratch))
    assert main(["migrate", "--dry-run", path]) == 0
    out = capsys.readouterr().out
    assert "TITLE REVIEW: 2 title(s)" in out and "'(this repo)'" in out and "'Done 2026-09-16'" in out
    assert "'Wire the scheduler into the nightly job'" not in out.split("TITLE REVIEW")[1].split("\n   +")[0]
    before = registry_lines(scratch)
    with pytest.raises(TitleReviewRequired) as exc:
        apply_file(scratch, path, wi_minter(scratch, WI))
    assert [i.title for i in exc.value.flagged] == ["(this repo)", "Done 2026-09-16"]
    assert main(["migrate", "--apply", path]) == 2
    assert "refusing to mint" in capsys.readouterr().err
    assert registry_lines(scratch) == before and (scratch / path).read_text() == text
    assert main(["migrate", "--apply", "--titles-reviewed", path]) == 0
    assert len(registry_lines(scratch)) == len(before) + 3


def test_dry_run_of_a_missing_path_is_a_message_not_a_traceback(scratch, monkeypatch, capsys):
    from coordgraph.cli import main
    monkeypatch.setenv("COORDGRAPH_REPO_ROOT", str(scratch))
    assert main(["migrate", "--dry-run", "opencode/docs/BACKLOG.md"]) == 2
    assert "no such file" in capsys.readouterr().err


# --- apply does not leave a half-done state (m1) ------------------------------------------------

THREE = "# B\n\n## Open\n\n- **First wired item.** a\n- **Second wired item.** b\n- **Third wired item.** c\n"


def test_a_mint_failing_midway_writes_nothing_and_reports_the_ids_already_minted(scratch):
    path = "opencode/docs/BACKLOG.md"
    put(scratch, path, THREE)
    calls = []

    def flaky(item):
        calls.append(item.title)
        if len(calls) == 3:
            raise RuntimeError("wi failed")
        return f"GAL-F{len(calls):04d}"

    with pytest.raises(ApplyError, match="GAL-F0001, GAL-F0002") as exc:
        apply_file(scratch, path, flaky)
    assert exc.value.minted == ["GAL-F0001", "GAL-F0002"] and "wi failed" in str(exc.value)
    assert (scratch / path).read_text() == THREE and not list((scratch / "opencode/docs").glob(".migrate-*"))


def test_a_file_edited_since_analysis_is_not_overwritten(scratch):
    path = "opencode/docs/BACKLOG.md"
    put(scratch, path, THREE)
    edited = THREE + "\nan edit that landed while ids were being minted\n"

    def minting(item, _n=[0]):
        _n[0] += 1
        if _n[0] == 3:
            (scratch / path).write_text(edited)
        return f"GAL-F{_n[0]:04d}"

    with pytest.raises(ApplyError, match="changed since it was analysed") as exc:
        apply_file(scratch, path, minting)
    assert len(exc.value.minted) == 3
    assert (scratch / path).read_text() == edited and not list((scratch / "opencode/docs").glob(".migrate-*"))


@pytest.mark.parametrize("mode", [0o644, 0o664, 0o600])
def test_apply_keeps_the_files_mode(scratch, mode):
    path = "opencode/docs/BACKLOG.md"
    put(scratch, path, THREE)
    os.chmod(scratch / path, mode)
    apply(scratch, path)
    assert stat.S_IMODE(os.stat(scratch / path).st_mode) == mode
    assert "GAL-F" in (scratch / path).read_text()
