"""History matching: delivery-shaped headings, scheme IDs, siblings (plan section 3.7; plan3 3.3)."""
import pytest

from coordgraph.history import (
    HistoryIndex,
    backlog_path_for,
    delivery_headings,
    find_id_tokens,
    history_path_for,
    mentioned_tokens,
)
from coordgraph.shapes import classify, offline_candidates, qualify

LEGACY = {"claude/*/kaizen/plan.md": "$1", "*/docs/BACKLOG.md": "$1", "docs/BACKLOG.md": "cpg"}


def tokens(heading):
    return [t for h in delivery_headings(heading + "\n") for t in h.tokens]


@pytest.mark.parametrize("heading,ids", [
    ("## 2026-10-21 — K-065 closed: Mitigation D holds", ["K-065"]),
    ("## 2026-10-17 — K-069 delivered: a shipped log config", ["K-069"]),
    ("## 2026-10-17 — K-075 fixed: an underscore-only title", ["K-075"]),
    ("## 2026-09-18 — K-065/DEF-6 closed: Mitigation D", ["K-065"]),
    ("### K-070/K-071 delivered", ["K-070", "K-071"]),
    ("## 2026-10-26 — K-078 closed: carts fill; K-079 F1 and F3 delivered", ["K-078"]),
    ("## ✅ K-077 and the docs", ["K-077"]),
    ("## 2026-11-20 — GAL-F0042 delivered", ["GAL-F0042"]),
    ("## 2026-11-20 — GAL-I02-K0007/K-070/GAL-F0043 closed", ["GAL-I02-K0007", "K-070", "GAL-F0043"]),
    ("## 2026-11-20 — falkor-chat:K-065 shipped", ["falkor-chat:K-065"]),
    ("## 2026-11-20 — K-065 Closed: capitalised", ["K-065"]),
    ("## 2026-11-20 — K-065 DELIVERED", ["K-065"]),
    ("## 2026-11-20 — K-065 Done", ["K-065"]),
    ("## 2026-11-20 — K-065 fixed.", ["K-065"]),
])
def test_delivery_shaped_headings_yield_their_id_tokens(heading, ids):
    assert tokens(heading) == ids


@pytest.mark.parametrize("heading", [
    "## 2026-10-15 — K-067: chunk extract and embed jobs retry (mechanism delivered; live proof outstanding)",
    "## 2026-10-14 — K-054 revision: analyst Pass 1",
    "## 2026-10-15 — The fixes are filed as K-077",
    "## K-070, K-071 delivered",  # a comma list matches neither (P3-n2)
    "## Fixes shipped this week",  # a verb with no ID in front of it
    "K-065 closed: not a heading",
    "# K-065 closed: level one is not a delivery heading",
    "##### K-065 closed: nor level five",
    "## 2026-11-20 — K-065 fixed-up: a hyphenated word is not the verb",
    "## 2026-11-20 — K-065 done criteria",
    "## 2026-11-20 — K-065 Done Definition: a noun phrase, not a delivery",
    "## 2026-11-20 — K-065 closed-loop check",
])
def test_non_delivery_shapes_match_nothing(heading):
    assert tokens(heading) == []


def test_headings_inside_fenced_code_are_ignored_and_lines_are_numbered():
    text = "intro\n\n## K-001 closed: x\n\n```\n## K-002 closed: y\n```\n"
    assert [(h.line, h.tokens) for h in delivery_headings(text)] == [(3, ("K-001",))]


def test_id_tokens_are_found_by_shape_only():
    assert find_id_tokens("see K-065, falkor-chat:K-070. GAL-F0042-FR2 and DEF-6 and k-1 A-1b") == [
        "K-065", "falkor-chat:K-070", "GAL-F0042-FR2"]
    assert mentioned_tokens("K-1 K-1 GAL-F0042-FR2 GAL-F0042 K-2.") == ["K-1", "GAL-F0042", "K-2"]


@pytest.mark.parametrize("token,kind", [
    ("K-065", "legacy-bare"), ("falkor-chat:K-065", "legacy-qualified"), ("GAL-F0042", "scheme"),
    ("GAL-F0042-FR2a", "scheme"), ("GAL-F42", "scheme"), ("free text", None), ("k-1", None),
    ("bogus:KK-1", None),
])
def test_shape_routing(token, kind):
    assert classify(token) == kind


def test_a_scheme_reference_is_never_scope_qualified_a_bare_legacy_one_always_is():
    assert qualify("GAL-F0042", "cpg") == "GAL-F0042"
    assert qualify("K-065", "falkor-chat") == "falkor-chat:K-065"
    assert qualify("falkor-chat:K-065", "cpg") == "falkor-chat:K-065"
    assert not any(c.startswith("cpg:") for c in offline_candidates("GAL-I02-K0042", "cpg"))


def test_history_and_backlog_paths_pair_up():
    assert history_path_for("falkor-chat/docs/BACKLOG.md") == "falkor-chat/docs/HISTORY.md"
    assert history_path_for("docs/BACKLOG.md") == "docs/HISTORY.md"
    assert history_path_for("claude/cobb/kaizen/plan.md") == "claude/cobb/kaizen/history.md"
    assert history_path_for("docs/plans/x.md") is None
    assert backlog_path_for("claude/cobb/kaizen/history.md") == "claude/cobb/kaizen/plan.md"
    assert backlog_path_for("falkor-chat/docs/HISTORY.md") == "falkor-chat/docs/BACKLOG.md"


def write(root, rel, text):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def registry(*groups):
    """A fake registry-backed `Candidates`: every token of a group denotes the same identity."""
    def candidates(token, scope):
        q = qualify(token, scope)
        for g in groups:
            if q in g or token in g:
                return set(g) | {token, q}
        return {token, q}
    return candidates


@pytest.fixture
def index(tmp_path):
    write(tmp_path, "falkor-chat/docs/HISTORY.md",
          "# H\n\n## 2026-11-20 — GAL-F0042 delivered: the thing\n\n## 2026-10-21 — K-066 closed: x\n"
          "\n## 2026-11-21 — K-070/GAL-F0043 delivered\n")
    write(tmp_path, "docs/HISTORY.md", "# H\n\n## 2026-08-01 — C-310 closed\n")
    return HistoryIndex.from_repo(tmp_path, ["falkor-chat/docs/BACKLOG.md", "docs/BACKLOG.md"], LEGACY)


def test_a_legacy_ref_matches_only_its_owning_backlogs_history(index):
    assert index.delivered("K-066", "falkor-chat")[0] == "falkor-chat/docs/HISTORY.md"
    assert index.delivered("falkor-chat:K-066", "cpg") is not None
    assert index.delivered("K-066", "cpg") is None  # bare means the referrer's backlog
    assert index.delivered("C-310", "cpg")[0] == "docs/HISTORY.md"


def test_scheme_refs_search_every_history_file_and_siblings_count(index):
    assert index.delivered("GAL-F0042", "cpg")[0] == "falkor-chat/docs/HISTORY.md"
    assert index.delivered("GAL-F0043", "cpg") is not None  # a scheme ID among slash-joined siblings
    assert index.delivered("K-070", "falkor-chat") is not None
    assert index.delivered("GAL-F0099", "cpg") is None and index.delivered("free text", "cpg") is None


def test_a_dependency_written_as_the_old_name_is_delivered_by_a_scheme_heading_via_candidates(index):
    """plan3 test 8: `## ... GAL-F0042 delivered` satisfies `falkor-chat:K-065`."""
    reg = registry(("falkor-chat:K-065", "GAL-F0042"))
    assert index.delivered("falkor-chat:K-065", "cpg") is None  # raw-string equality cannot see it
    hit = index.delivered("falkor-chat:K-065", "cpg", reg)
    assert hit is not None and "GAL-F0042" in hit[1].tokens
    # and the reverse: a scheme dependency delivered under the old name
    write_old = HistoryIndex({"falkor-chat": [("h", delivery_headings("## K-065 closed\n"))]})
    assert write_old.delivered("GAL-F0042", "cpg") is None
    assert write_old.delivered("GAL-F0042", "cpg", reg) is not None
