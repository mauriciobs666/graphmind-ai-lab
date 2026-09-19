"""`scripts/consolidate_sweep_reports.py` — Unit C of the small-model catalog sweep's report
feature (`docs/plans/small-model-catalog-sweep.md` §3.5, §4/§5 Unit C).

Light, fixture-based tests matching `scripts/refresh_golden.py`'s own precedent (a modest
convenience-script test file, not the core library's mutation-testing bar) — with one exception:
`test_assembler_never_combines_two_packs_values` *is* mutation-tested by hand (see its own
docstring), because the no-cross-pack-arithmetic invariant is exactly the kind of thing this
component's honesty rules exist to protect, even in a "light touch" script.

Every fixture report file here is **hand-built**, matching the plan's §3.5 marker format exactly
— `rank`, the command that would produce a real one, does not exist yet (Unit B, not yet built).
Real end-to-end verification against actual `rank` output is a follow-up once Units A and B land.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import consolidate_sweep_reports as csr  # noqa: E402

# --------------------------------------------------------------------------------------------
# Fixture report bodies — one marker for four packs, TWO for guard-judge (plan §3.5's own
# explicitly-called-out shape: six markers across five files, not five).
# --------------------------------------------------------------------------------------------

_EMBEDDER_REPORT = """\
# embedder-graphrag-retrieval — ranked report

| rank | model | mean | 95% CI |
|---|---|---|---|
| 1 | text-embedding-qwen3-embedding-0.6b | 0.812 | [0.701, 0.905] |

<!-- rank-report: pack=embedder-graphrag-retrieval metric=mrr \
top=text-embedding-qwen3-embedding-0.6b value=0.812 ci=[0.701,0.905] -->
"""

_NLQ_REPORT = """\
# nlq-structured-query — ranked report

| rank | model | k/n | rate | 95% CI |
|---|---|---|---|---|
| 1 | qwen/qwen3-4b-2507 | 30/34 | 0.882 | [0.73, 0.95] |

<!-- rank-report: pack=nlq-structured-query metric=layer1ExactMatchRate top=qwen/qwen3-4b-2507 \
value=0.882 ci=[0.73,0.95] -->
"""

_TOOL_CALLER_REPORT = """\
# tool-caller-shop-assistant — ranked report

| rank | model | k/n | rate | 95% CI |
|---|---|---|---|---|
| 1 | mistralai/ministral-3-3b | 10/12 | 0.833 | [0.55, 0.97] |

<!-- rank-report: pack=tool-caller-shop-assistant metric=cleanThroughTurnH \
top=mistralai/ministral-3-3b value=0.833 ci=[0.55,0.97] -->
"""

_CHAT_RESPONDER_REPORT = """\
# chat-responder-grounded-answers — ranked report

| rank | model | k/n | rate | 95% CI |
|---|---|---|---|---|
| 1 | qwen/qwen3-4b-2507 | 40/40 | 1.0 | [0.91, 1.0] |

<!-- rank-report: pack=chat-responder-grounded-answers metric=groundingRate \
top=qwen/qwen3-4b-2507 value=1.0 ci=[0.91,1.0] -->
"""

# The one pack with no headline metric — two ranked tables, one per verdictMetrics member, each
# with its OWN marker line (plan §3.1/§3.5; review Pass-1 finding 2.2, fixed in v2).
_GUARD_JUDGE_REPORT = """\
# guard-judge-understanding — ranked report

## falseAdvanceRate

| rank | model | k/n | rate | 95% CI |
|---|---|---|---|---|
| 1 | google/gemma-3-4b | 2/40 | 0.05 | [0.01, 0.17] |

<!-- rank-report: pack=guard-judge-understanding metric=falseAdvanceRate \
top=google/gemma-3-4b value=0.05 ci=[0.01,0.17] -->

## falseSuspendRate

| rank | model | k/n | rate | 95% CI |
|---|---|---|---|---|
| 1 | qwen/qwen3-4b-2507 | 3/40 | 0.075 | [0.02, 0.20] |

<!-- rank-report: pack=guard-judge-understanding metric=falseSuspendRate \
top=qwen/qwen3-4b-2507 value=0.075 ci=[0.02,0.20] -->
"""


def _write(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


@pytest.fixture
def five_reports(tmp_path: Path) -> list[Path]:
    """The sweep's own five report files, in pack order — six markers total across them."""
    return [
        _write(tmp_path, "embedder-graphrag-retrieval-rank-2026-09-19-01.md", _EMBEDDER_REPORT),
        _write(
            tmp_path, "guard-judge-understanding-rank-2026-09-19-01.md", _GUARD_JUDGE_REPORT
        ),
        _write(tmp_path, "nlq-structured-query-rank-2026-09-19-01.md", _NLQ_REPORT),
        _write(
            tmp_path, "tool-caller-shop-assistant-rank-2026-09-19-01.md", _TOOL_CALLER_REPORT
        ),
        _write(
            tmp_path,
            "chat-responder-grounded-answers-rank-2026-09-19-01.md",
            _CHAT_RESPONDER_REPORT,
        ),
    ]


# --------------------------------------------------------------------------------------------
# parse_markers / load_markers
# --------------------------------------------------------------------------------------------


def test_parse_markers_extracts_a_single_marker_line(tmp_path: Path) -> None:
    path = _write(tmp_path, "embedder.md", _EMBEDDER_REPORT)
    markers = csr.parse_markers(_EMBEDDER_REPORT, source=path)
    assert len(markers) == 1
    marker = markers[0]
    assert marker.pack == "embedder-graphrag-retrieval"
    assert marker.metric == "mrr"
    assert marker.top == "text-embedding-qwen3-embedding-0.6b"
    assert marker.value == "0.812"
    assert marker.ci_low == "0.701"
    assert marker.ci_high == "0.905"
    assert marker.source == path


def test_parse_markers_extracts_two_markers_from_the_guard_judge_shaped_file(
    tmp_path: Path,
) -> None:
    """The exact shape both the plan (§3.5) and the review (finding 2.2) call out: one report
    file, two marker lines, keyed by different metrics."""
    path = _write(tmp_path, "guard-judge.md", _GUARD_JUDGE_REPORT)
    markers = csr.parse_markers(_GUARD_JUDGE_REPORT, source=path)
    assert len(markers) == 2
    assert [m.metric for m in markers] == ["falseAdvanceRate", "falseSuspendRate"]
    assert [m.top for m in markers] == ["google/gemma-3-4b", "qwen/qwen3-4b-2507"]
    assert all(m.pack == "guard-judge-understanding" for m in markers)


def test_parse_markers_returns_empty_list_when_no_marker_present(tmp_path: Path) -> None:
    path = _write(tmp_path, "no-marker.md", "# just a heading\n\nno marker here.\n")
    assert csr.parse_markers("# just a heading\n\nno marker here.\n", source=path) == []


def test_load_markers_concatenates_in_file_then_in_document_order(
    five_reports: list[Path],
) -> None:
    """Six markers total across five files — not five (plan §3.5, review finding 2.2) — in the
    exact order the files were given, guard-judge's own two markers kept adjacent and in their
    own in-file order."""
    markers = csr.load_markers(five_reports)
    assert len(markers) == 6
    assert [m.pack for m in markers] == [
        "embedder-graphrag-retrieval",
        "guard-judge-understanding",
        "guard-judge-understanding",
        "nlq-structured-query",
        "tool-caller-shop-assistant",
        "chat-responder-grounded-answers",
    ]
    assert [m.metric for m in markers][1:3] == ["falseAdvanceRate", "falseSuspendRate"]


# --------------------------------------------------------------------------------------------
# build_consolidated_document
# --------------------------------------------------------------------------------------------


def test_build_consolidated_document_index_has_one_row_per_marker(
    tmp_path: Path, five_reports: list[Path]
) -> None:
    markers = csr.load_markers(five_reports)
    out_path = tmp_path / "consolidated.md"
    document = csr.build_consolidated_document(markers, out_path=out_path)

    # Six data rows (plus the header + separator row) — six markers, not five.
    index_lines = [
        line
        for line in document.splitlines()
        if line.startswith("|") and "Pack" not in line and "---" not in line
    ]
    assert len(index_lines) == 6
    assert "`falseAdvanceRate`" in index_lines[1]
    assert "`falseSuspendRate`" in index_lines[2]


def test_build_consolidated_document_index_values_are_copied_verbatim(
    tmp_path: Path, five_reports: list[Path]
) -> None:
    """No rounding, no reformatting — a value/CI string in the index table is byte-identical to
    the one in the marker line it came from."""
    markers = csr.load_markers(five_reports)
    document = csr.build_consolidated_document(markers, out_path=tmp_path / "out.md")
    assert "0.812" in document
    assert "[0.701, 0.905]" in document
    assert "1.0" in document  # chat-responder's own value/high-CI, unrounded


def test_build_consolidated_document_links_are_relative_to_the_output_directory(
    tmp_path: Path,
) -> None:
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    report_path = _write(reports_dir, "embedder-rank.md", _EMBEDDER_REPORT)
    markers = csr.load_markers([report_path])

    # Same directory as the reports -> a bare filename link.
    same_dir_doc = csr.build_consolidated_document(markers, out_path=reports_dir / "out.md")
    assert "(embedder-rank.md)" in same_dir_doc

    # A different directory -> a relative `../` link, never an absolute path leaking the host.
    other_dir = tmp_path / "elsewhere"
    other_dir.mkdir()
    other_dir_doc = csr.build_consolidated_document(markers, out_path=other_dir / "out.md")
    expected_relative = os.path.relpath(report_path.resolve(), start=other_dir.resolve())
    assert f"({expected_relative})" in other_dir_doc
    assert str(report_path) not in other_dir_doc


def test_build_consolidated_document_has_exactly_one_preamble_todo(
    tmp_path: Path, five_reports: list[Path]
) -> None:
    markers = csr.load_markers(five_reports)
    document = csr.build_consolidated_document(markers, out_path=tmp_path / "out.md")
    assert document.count("<!-- TODO: state plainly") == 1


def test_build_consolidated_document_has_exactly_one_narrative_todo_per_distinct_role(
    tmp_path: Path, five_reports: list[Path]
) -> None:
    """Five roles represented (guard-judge's two markers share one role) -> exactly five
    narrative TODO placeholders, one per `### <role>` heading — never six (one per marker) and
    never four (a dropped role)."""
    markers = csr.load_markers(five_reports)
    document = csr.build_consolidated_document(markers, out_path=tmp_path / "out.md")
    assert document.count(csr._NARRATIVE_TODO) == 5
    for role in ("embedder", "guard-judge", "nlq-generator", "tool-caller", "chat-responder"):
        assert f"### {role}" in document
    # guard-judge's heading appears exactly once despite contributing two index rows.
    assert document.count("### guard-judge") == 1


def test_build_consolidated_document_only_headers_roles_actually_present(tmp_path: Path) -> None:
    """A `--reports` call given fewer than five files must not manufacture placeholder headings
    for a role nobody supplied a report for."""
    report_path = _write(tmp_path, "embedder.md", _EMBEDDER_REPORT)
    markers = csr.load_markers([report_path])
    document = csr.build_consolidated_document(markers, out_path=tmp_path / "out.md")
    assert document.count(csr._NARRATIVE_TODO) == 1
    assert "### embedder" in document
    assert "### guard-judge" not in document


def test_build_consolidated_document_raises_on_no_markers_at_all(tmp_path: Path) -> None:
    with pytest.raises(csr.ConsolidateSweepReportsError, match="no `<!-- rank-report"):
        csr.build_consolidated_document([], out_path=tmp_path / "out.md")


def test_build_consolidated_document_raises_on_unknown_pack(tmp_path: Path) -> None:
    marker = csr.RankMarker(
        pack="some-future-pack",
        metric="x",
        top="model-a",
        value="0.5",
        ci_low="0.4",
        ci_high="0.6",
        source=tmp_path / "future.md",
    )
    with pytest.raises(csr.ConsolidateSweepReportsError, match="some-future-pack"):
        csr.build_consolidated_document([marker], out_path=tmp_path / "out.md")


# --------------------------------------------------------------------------------------------
# The no-cross-pack-arithmetic invariant — this one IS worth a mutation test.
#
# Method: two markers from two DIFFERENT packs carry deliberately distinctive numeric values
# whose sum/difference/average are themselves distinctive decimal strings, unlikely to appear in
# the document by any legitimate, single-marker path. This test asserts none of those computed
# strings appear anywhere in the assembled document.
#
# This was verified as a real mutation test during development, not just asserted in the
# abstract: temporarily editing `_index_table` to append an extra column computed as
# `float(a.value) + float(b.value)` across two markers from different packs reddened this test
# immediately (the injected sum, "1.353", appeared in the document and the assertion below
# failed); reverting the mutation restored green. That edit was made and reverted only in this
# development session — the shipped script never parses a marker's `value`/`ci_*` fields to a
# number anywhere (see `_MARKER_RE`'s own docstring note), so there is no code path left that
# could reintroduce this silently.
# --------------------------------------------------------------------------------------------


def test_assembler_never_combines_two_packs_values(tmp_path: Path) -> None:
    marker_a = csr.RankMarker(
        pack="embedder-graphrag-retrieval",
        metric="mrr",
        top="model-a",
        value="0.137",
        ci_low="0.100",
        ci_high="0.200",
        source=tmp_path / "a.md",
    )
    marker_b = csr.RankMarker(
        pack="nlq-structured-query",
        metric="layer1ExactMatchRate",
        top="model-b",
        value="0.941",
        ci_low="0.800",
        ci_high="0.980",
        source=tmp_path / "b.md",
    )
    document = csr.build_consolidated_document([marker_a, marker_b], out_path=tmp_path / "out.md")

    forbidden_combinations = {
        "sum": str(float(marker_a.value) + float(marker_b.value)),  # "1.078"
        "difference": str(float(marker_b.value) - float(marker_a.value)),  # "0.804"
        "average": str((float(marker_a.value) + float(marker_b.value)) / 2),  # "0.539"
    }
    for label, computed in forbidden_combinations.items():
        assert computed not in document, (
            f"consolidated document contains a cross-pack {label} ({computed!r}) that no single "
            "marker's own value could have produced — the assembler combined two packs' numbers"
        )

    # Positive control: each pack's OWN value is present verbatim (never dropped).
    assert marker_a.value in document
    assert marker_b.value in document


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------


def test_main_writes_the_consolidated_document_and_prints_its_path(
    tmp_path: Path, five_reports: list[Path], capsys: pytest.CaptureFixture[str]
) -> None:
    out_path = tmp_path / "out" / "consolidated.md"
    exit_code = csr.main(
        ["--reports", *[str(p) for p in five_reports], "--out", str(out_path)]
    )
    assert exit_code == 0
    assert out_path.exists()
    written = out_path.read_text(encoding="utf-8")
    assert written.count("<!-- rank-report") == 0  # markers are the SOURCE report's, not copied
    assert "## Index" in written
    captured = capsys.readouterr()
    assert str(out_path) in captured.out


def test_main_exits_2_on_a_missing_report_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "does-not-exist.md"
    exit_code = csr.main(["--reports", str(missing), "--out", str(tmp_path / "out.md")])
    assert exit_code == 2
    captured = capsys.readouterr()
    assert "no such report file" in captured.err
    assert not (tmp_path / "out.md").exists()


def test_main_exits_1_when_no_reports_carry_any_marker(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    empty_report = _write(tmp_path, "empty.md", "# nothing to see here\n")
    exit_code = csr.main(
        ["--reports", str(empty_report), "--out", str(tmp_path / "out.md")]
    )
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "no `<!-- rank-report" in captured.err
    assert not (tmp_path / "out.md").exists()
