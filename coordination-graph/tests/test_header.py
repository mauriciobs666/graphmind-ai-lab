import sys
from pathlib import Path

import pytest

from coordgraph.parse.header import header_fields, parse_header, tokens

REPO = Path(__file__).resolve().parents[2]
WI_SCRIPTS = REPO / "skills" / "work-items" / "scripts"

FULL = (
    "# Title\n\n"
    "> **Status:** active · **Owner:** `architect` · **Tracks:** K-022 (M3) · **Version:** 2 · "
    "**Extends:** `a.md` · **Reviews:** `r.md` · **Last updated:** 2026-11-20 · "
    "**Item:** `GAL-F0042` · **Aliases:** `GAL-F0001`, `falkor-chat:K-065`\n\nBody.\n"
)


def test_all_optional_fields_parse_with_item_and_aliases_set_and_nothing_lost():
    h = parse_header(FULL)
    assert h.item_id == "GAL-F0042"
    assert h.aliases == ["GAL-F0001", "falkor-chat:K-065"]
    assert (h.doc_status, h.owner, h.tracks) == ("active", "`architect`", "K-022 (M3)")
    assert set(h.fields) == {
        "Status", "Owner", "Tracks", "Version", "Extends", "Reviews", "Last updated",
        "Item", "Aliases",
    }
    assert h.fields["Extends"] == "`a.md`"


def test_blank_line_under_h1_is_allowed_and_nothing_else_may_precede_the_header():
    assert parse_header("# T\n> **Status:** active\n") is not None
    assert parse_header("# T\n\n> **Status:** active\n") is not None
    assert parse_header("# T\n\n\n> **Status:** active\n") is None
    assert parse_header("# T\nintro\n> **Status:** active\n") is None
    assert parse_header("> **Status:** active\n") is None  # no H1


def test_no_header_block_means_none_and_no_aliases_list():
    assert parse_header("# T\n\nno header\n") is None
    assert header_fields("") is None


def test_header_without_item_or_aliases():
    h = parse_header("# T\n\n> **Status:** archived · **Owner:** `teco` · **Tracks:** —\n")
    assert h.item_id is None and h.aliases == [] and h.item_tokens == []
    assert h.doc_status == "archived"


@pytest.mark.parametrize("value,token", [
    ("active", "active"),
    ("Active", "active"),
    ("`active`", "active"),
    ("**active**", "active"),
    ("Interviewing", "Interviewing"),
    ("Ready for design", "Ready for design"),
    ("Ready for design (v4)", "Ready for design"),
    ("active — amended in place", "active"),
    ("archived", "archived"),
    ("superseded", "superseded"),
    ("draft", "draft"),  # outside the closed set: the leading word, visible rather than hidden
    ("activeish", "activeish"),  # a prefix match is not a token match
])
def test_status_token_is_the_first_thing_after_the_label(value, token):
    assert parse_header(f"# T\n\n> **Status:** {value} · **Owner:** `x`\n").doc_status == token


def test_status_free_text_after_the_token_is_not_the_token():
    h = parse_header("# T\n\n> **Status:** active · **Owner:** `x`\n")
    assert h.doc_status == "active"


def test_item_token_hygiene_unusable_item_is_not_stored_and_warns():
    h = parse_header("# T\n\n> **Status:** active · **Item:** `   `\n")
    assert h.item_id is None and any("Item" in w for w in h.warnings)
    long = "G" * 600
    h = parse_header(f"# T\n\n> **Status:** active · **Item:** `{long}`\n")
    assert h.item_id is None and h.warnings
    # a planted malformed id is stored verbatim: the parser never shape-validates
    assert parse_header("# T\n\n> **Status:** active · **Item:** `GAL-F42`\n").item_id == "GAL-F42"


def test_tokens_backticked_else_whole_value_and_dash_means_none():
    assert tokens("`a`, `b`") == ["a", "b"]
    assert tokens("plain") == ["plain"]
    assert tokens("—") == [] and tokens("") == []


def test_unparseable_part_is_tolerated():
    h = parse_header("# T\n\n> **Status:** active · free text · **Item:** `GAL-F1`\n")
    assert h.item_id == "GAL-F1"


# --- parity with wi's own header reader ---------------------------------------------------

@pytest.fixture(scope="module")
def wilib():
    sys.path.insert(0, str(WI_SCRIPTS))
    try:
        import wilib.rebuild as rebuild
    finally:
        sys.path.remove(str(WI_SCRIPTS))
    return rebuild


def test_parity_with_wilib_over_every_family_document_in_the_repo(wilib):
    docs = wilib.iter_docs(REPO)
    assert len(docs) > 300, "the family walk found suspiciously few documents"
    with_header = 0
    for path in docs:
        text = path.read_bytes().decode("utf-8", "replace")
        theirs = wilib.header_fields(path)
        ours = header_fields(text)
        assert ours == theirs, path
        if theirs is None:
            assert parse_header(text) is None
            continue
        with_header += 1
        h = parse_header(text)
        assert h.item_tokens == wilib.tokens(theirs.get("Item", "")), path
        assert h.alias_tokens == wilib.tokens(theirs.get("Aliases", "")), path
    assert with_header > 300


def test_parity_with_wilib_on_a_header_carrying_every_optional_field(wilib, tmp_path):
    f = tmp_path / "x.md"
    f.write_text(FULL)
    assert header_fields(FULL) == wilib.header_fields(f)
    h = parse_header(FULL)
    assert h.item_tokens == wilib.tokens(wilib.header_fields(f)["Item"])
    assert h.alias_tokens == wilib.tokens(wilib.header_fields(f)["Aliases"])


def test_production_code_imports_no_wilib():
    import ast

    pkg = Path(__file__).resolve().parents[1] / "coordgraph"
    offenders = []
    for py in pkg.rglob("*.py"):
        for node in ast.walk(ast.parse(py.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            if any(n.split(".")[0] == "wilib" for n in names):
                offenders.append(str(py))
    assert offenders == []
