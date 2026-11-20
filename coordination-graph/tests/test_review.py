import pytest

from coordgraph.parse.review import classify_verdict, parse_review

HDR = "# R\n\n> **Status:** active · **Owner:** `analyst` · **Tracks:** —\n\n"


def review(body, path="docs/reviews/x.md"):
    return parse_review(path, (HDR + body).encode())


@pytest.mark.parametrize("body,cls", [
    ("**Verdict: approve.** No blockers.\n", "approve"),
    ("**Verdict: approve with suggestions.** One minor.\n", "approve-with-suggestions"),
    ("**Verdict: needs changes.** 1 blocker.\n", "needs-changes"),
    ("**Verdict: needs changes** (one blocker).\n", "needs-changes"),
    ("Verdict (Pass 1): needs changes\n", "needs-changes"),
    ("**Verdict (pass 1): needs changes** (one blocker). Current verdict: `## Pass 2`.\n", "needs-changes"),
    ("## Verdict: **needs changes**\n", "needs-changes"),
    ("**Verdict: NEEDS CHANGES**\n", "needs-changes"),
    ("**Verdict: needs one change (N1)** before the record pins the tools.\n", "needs-changes"),
    ("**Verdict: reject.** Wrong approach.\n", "reject"),
    ("Scope narrow: foo. **Verdict: approve.**\n", "approve"),
    ("Re-gate of the fixes. **Verdict: approve** (no blocker).\n", "approve"),
    ("**Updated verdict: approve.** Pass 1's major is closed.\n", "approve"),
    ("**Verdict**: approve with suggestions\n", "approve-with-suggestions"),
])
def test_each_verdict_spelling(body, cls):
    assert review(body).verdict_class == cls


def test_verdict_heading_with_the_verdict_on_the_next_line():
    d = review("## Verdict\n\n**approve with suggestions** — no blocker.\n")
    assert d.verdict_class == "approve-with-suggestions"
    assert d.latest_verdict.startswith("approve with suggestions")


def test_last_pass_section_wins_over_earlier_verdicts():
    d = review(
        "## Pass 1\n\n**Verdict: needs changes.** 1 blocker.\n\n"
        "## Pass 2\n\n**Verdict: needs changes.** new major.\n\n"
        "## Pass 3 — 2026-11-01\n\n**Verdict: approve with suggestions.** All fixed.\n")
    assert (d.verdict_class, d.latest_pass) == ("approve-with-suggestions", 3)
    assert d.latest_verdict.startswith("approve with suggestions")


def test_earlier_pass_needs_changes_is_not_the_latest():
    d = review("## Pass 1\n\n**Verdict: needs changes.**\n\n## Pass 2\n\n**Verdict: approve.**\n")
    assert d.verdict_class == "approve"


def test_single_unnumbered_verdict_is_pass_one():
    d = review("## Scope & verdict\n\n**Verdict: approve.**\n")
    assert (d.verdict_class, d.latest_pass) == ("approve", 1)


def test_pass_marked_verdicts_without_pass_headings_take_the_highest_pass():
    d = review("Verdict (Pass 1): needs changes\n\nlater\n\nVerdict (Pass 2): approve\n")
    assert (d.verdict_class, d.latest_pass) == ("approve", 2)


def test_no_verdict_is_unknown_with_a_warning_and_never_invented():
    d = review("## Findings\n\nThe cause was a race.\n")
    assert d.verdict_class == "unknown" and d.latest_verdict is None
    assert "no verdict found" in d.warnings
    # a last pass that carries no verdict is unknown, not the previous pass's verdict
    d = review("## Pass 1\n\n**Verdict: needs changes.**\n\n## Pass 2\n\nDispositions only.\n")
    assert d.verdict_class == "unknown"


def test_prose_mentioning_a_verdict_without_a_phrase_is_not_a_verdict():
    assert review("My Pass 1 split verdict is discharged.\n").verdict_class == "unknown"


def test_classify_verdict_takes_the_earliest_phrase():
    assert classify_verdict("approve with suggestions (earlier: needs changes)") == "approve-with-suggestions"
    assert classify_verdict("needs changes, then approve") == "needs-changes"
    assert classify_verdict("") == "unknown"


def test_source_props_carry_status_and_verdict_and_header_fields():
    d = review("**Verdict: approve.**\n")
    p = d.source_props()
    assert p["kind"] == "review" and p["docStatus"] == "active" and p["owner"] == "`analyst`"
    assert p["verdictClass"] == "approve" and p["latestPass"] == 1
    assert p["latestVerdict"] == "approve." and p["component"] == "root"
    assert p["warnings"] == []
