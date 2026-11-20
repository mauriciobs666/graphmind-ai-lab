"""Backlog parser: heading grammar (plan3 section 3.6), raw tokens, ref/scope policy, field block."""
import pytest

from coordgraph.parse.backlog import BacklogParseError, parse_backlog

LEGACY = {"claude/*/kaizen/plan.md": "$1", "*/docs/BACKLOG.md": "$1", "docs/BACKLOG.md": "cpg"}
FC = "falkor-chat/docs/BACKLOG.md"


def backlog(body, path=FC, legacy=LEGACY):
    return parse_backlog(path, ("# Backlog\n\n" + body).encode(), legacy)


def item(heading_id, status="- **Status:** 🔵 proposed\n"):
    return f"### {heading_id} — Title\n\n{status}\n"


# --- heading grammar: every row of the plan3 section 3.6 table ---------------------------

POSITIVE = [
    "K-004", "C-310", "K-0001", "GAL-F0042", "GAL-I03-F0012-S01-T01", "GAL-I02-K0001",
    "GAL-I01-D0001", "GAL-F0042-S01",
    "GAL-F00042",  # passes shape; `lint` reports it malformed (the documented split)
]
NEGATIVE = [
    "GAL-", "gal-F0042", "GAL-F4", "GAL-F42", "GAL-F042", "K-04", "GAL-X0001", "GAL-0042",
    "GAL-F0042-T3",
    "GAL-F0042-FR2", "GAL-F0042-FR2a",  # FR/AC are citations, not items
]


@pytest.mark.parametrize("ident", POSITIVE)
def test_heading_positive_is_an_item_with_its_id_as_written(ident):
    (i,) = backlog(f"### {ident} — A title\n").items  # no field block needed to be an item
    assert i.item_id == ident and i.title == "A title" and not i.id_less


@pytest.mark.parametrize("ident", NEGATIVE)
def test_heading_negative_is_not_an_item(ident):
    assert backlog(f"### {ident} — A title\n\nprose\n").items == []


@pytest.mark.parametrize("ident", NEGATIVE)
def test_a_negative_heading_with_a_status_block_is_an_id_less_item_never_an_id(ident):
    (i,) = backlog(item(ident)).items
    assert i.item_id is None and i.id_less
    assert i.ref == "falkor-chat:@1"


def test_only_level_three_headings_are_items_and_fenced_ones_are_not():
    d = backlog("## K-001 — level two\n\n#### K-002 — level four\n\n```markdown\n### K-003 — example\n```\n"
                + item("K-004"))
    assert [i.item_id for i in d.items] == ["K-004"]


def test_the_title_is_the_whole_rest_of_the_heading():
    (i,) = backlog("### K-082 — Optional rerun (🔵 proposed — carried from K-066)\n").items
    assert i.title == "Optional rerun (🔵 proposed — carried from K-066)"


# --- Item.ref policy, scope, itemType -----------------------------------------------------

def test_legacy_heading_ref_is_scope_qualified_scheme_ref_is_not():
    d = backlog(item("K-065") + item("GAL-I02-K0042"))
    assert [(i.ref, i.scope) for i in d.items] == [
        ("falkor-chat:K-065", "falkor-chat"), ("GAL-I02-K0042", "falkor-chat")]


def test_scope_comes_from_legacy_paths_not_from_the_path_shape():
    d = backlog(item("K-1001").replace("K-1001", "C-1001"), path="docs/BACKLOG.md")
    assert d.items[0].ref == "cpg:C-1001" and d.scope == "cpg"
    d = backlog(item("K-001"), path="claude/cobb/kaizen/plan.md")
    assert d.items[0].ref == "cobb:K-001"
    other = {"*/docs/BACKLOG.md": "lab-$1"}
    assert backlog(item("K-001"), legacy=other).items[0].ref == "lab-falkor-chat:K-001"


def test_a_backlog_no_pattern_covers_warns_and_still_parses():
    d = backlog(item("K-001"), legacy={})
    assert d.scope is None and d.items[0].ref == "unscoped:K-001"
    assert any("scope" in w for w in d.warnings)


@pytest.mark.parametrize("ident,typ", [
    ("GAL-F0042", "F"), ("GAL-I03-F0012-S01-T01", "T"), ("GAL-I02-K0001", "K"),
    ("GAL-I01-D0001", "D"), ("GAL-F0042-S01", "S"),
])
def test_item_type_is_the_last_segments_letter_parsed_offline(ident, typ):
    assert backlog(item(ident)).items[0].item_type == typ


def test_item_type_is_absent_for_legacy_ids():
    assert backlog(item("K-065")).items[0].item_type is None
    assert "itemType" not in backlog(item("K-065")).items[0].props()


def test_id_less_item_gets_a_positional_ref_and_a_warning_until_the_reformat():
    d = backlog(item("K-001") + "### Open question\n\n- **Status:** 🔵 proposed\n")
    first, second = d.items
    assert (second.item_id, second.ref, second.ord) == (None, "falkor-chat:@2", 2)
    assert any("work-item ID" in w for w in d.warnings)
    assert "itemId" not in second.props()


def test_duplicate_id_in_one_file_is_a_parse_error_naming_both_lines():
    with pytest.raises(BacklogParseError, match=r"K-001.*lines 3 and 7"):
        backlog(item("K-001") + item("K-001"))


# --- raw, unqualified tokens (plan3 section 3.6 opacity; test 2) --------------------------

def test_depends_on_tokens_are_stored_raw_unqualified_and_never_shape_validated():
    d = backlog("### K-001 — T\n\n- **Status:** 🔵 proposed\n"
                "- **Depends on:** K-070, falkor-chat:K-065, GAL-F0042, GAL-F42, see the design doc\n")
    assert d.items[0].depends_on == ["K-070", "falkor-chat:K-065", "GAL-F0042", "GAL-F42",
                                     "see the design doc"]


@pytest.mark.parametrize("value,expected", [("none", []), ("None", []), ("K-070", ["K-070"])])
def test_depends_on_none_is_empty_list(value, expected):
    d = backlog(f"### K-001 — T\n\n- **Status:** 🔵 proposed\n- **Depends on:** {value}\n")
    assert d.items[0].depends_on == expected


def test_absent_depends_on_is_not_recorded_distinct_from_none():
    absent = backlog(item("K-001")).items[0]
    none = backlog("### K-001 — T\n\n- **Status:** 🔵 proposed\n- **Depends on:** none\n").items[0]
    assert absent.depends_on is None and none.depends_on == []
    assert "dependsOn" not in absent.props() and none.props()["dependsOn"] == []


def test_an_empty_token_between_commas_is_dropped_with_a_warning():
    d = backlog("### K-001 — T\n\n- **Status:** 🔵 proposed\n- **Depends on:** K-070,, K-071\n")
    assert d.items[0].depends_on == ["K-070", "K-071"]
    assert any("empty token" in w for w in d.warnings)


# --- the field block ------------------------------------------------------------------------

def test_field_block_after_heading_one_blank_allowed_continuations_and_unknown_labels():
    d = backlog(
        "### K-001 — T\n\n- **Status:** 🟡 in-progress — started\n- **Priority:** high — urgent\n"
        "- **Touches:** file:a/b.py; graph:ws:test; fact:x/y.py#f (relies)\n"
        "- **Owner:** `architect` and\n  a continuation\n- **Milestone:** M7 — order 2 — approved 2026-10-21\n"
        "\nprose - **Depends on:** K-9\n")
    (i,) = d.items
    assert (i.status_marker, i.status, i.open) == ("🟡 in-progress", "in-progress", True)
    assert i.priority == "high"
    assert i.touches == ["file:a/b.py", "graph:ws:test", "fact:x/y.py#f (relies)"]
    assert i.fields["Owner"] == "`architect` and a continuation"
    assert (i.milestone, i.milestone_order, i.milestone_approved, i.milestone_started) == (
        "M7", 2, "2026-10-21", None)
    assert i.depends_on is None  # prose after the block is not a field


def test_a_field_two_blank_lines_below_the_heading_is_not_read():
    (i,) = backlog("### K-001 — T\n\n\n- **Status:** 🔵 proposed\n").items
    assert i.status is None and "Status" not in i.fields


def test_status_is_keyed_on_the_word_and_the_emoji_must_agree():
    d = backlog(item("K-001", "- **Status:** ✅ delivered\n") + item("K-002", "- **Status:** ⚪ rejected — record\n")
                + item("K-003", "- **Status:** 🔵 deferred\n") + item("K-004", "- **Status:** soon\n"))
    a, b, c, e = d.items
    assert (a.status, a.open) == ("delivered", False)
    assert (b.status, b.open, b.status_marker) == ("rejected", True, "⚪ rejected")
    assert c.status == "deferred" and e.status is None
    assert any("emoji" in w for w in d.warnings) and any("unrecognised Status" in w for w in d.warnings)


def test_absent_priority_touches_milestone_are_not_recorded_and_none_is_recorded_empty():
    i = backlog("### K-001 — T\n\n- **Status:** 🔵 proposed\n- **Touches:** none\n").items[0]
    assert i.touches == [] and i.priority is None and i.milestone is None
    props = i.props()
    assert props["touches"] == [] and "priority" not in props and "milestone" not in props


def test_milestone_with_started_and_a_bad_part_warns():
    d = backlog("### K-001 — T\n\n- **Status:** 🟡 in-progress\n- **Milestone:** M2.5 — started 2026-11-01 — x\n")
    assert d.items[0].milestone_started == "2026-11-01"
    assert any("Milestone part" in w for w in d.warnings)


def test_touches_entry_without_prefix_or_fact_mode_warns():
    d = backlog("### K-001 — T\n\n- **Status:** 🔵 proposed\n- **Touches:** a/b.py; fact:x#y\n")
    assert sum("Touches" in w for w in d.warnings) == 2


def test_item_props_order_and_hash_change_with_the_item_text_only():
    a = backlog(item("K-001") + item("K-002"))
    b = backlog(item("K-001") + "### K-002 — Title\n\n- **Status:** 🔵 proposed\n\nedited\n")
    assert a.items[0].item_hash == b.items[0].item_hash
    assert a.items[1].item_hash != b.items[1].item_hash
    assert [i.ord for i in a.items] == [1, 2]
    assert a.source_props()["itemCount"] == 2 and a.source_props()["scope"] == "falkor-chat"


def test_item_hash_ignores_trailing_whitespace_and_surrounding_blank_lines():
    a = backlog("### K-001 — T\n\n- **Status:** 🔵 proposed\n\nbody\n")
    b = backlog("### K-001 — T\n\n- **Status:** 🔵 proposed   \n\nbody  \n\n\n")
    assert a.items[0].item_hash == b.items[0].item_hash


def test_planted_ledger_needs_tokens_are_stored_verbatim_too():
    from coordgraph.parse.ledger import parse_needs
    assert parse_needs("GAL-F42, free text, `U3`") == (["GAL-F42", "free text", "U3"], None)
