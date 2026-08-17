"""Unit tests for the eval report generator's rendering/branching logic (K-047,
`docs/BACKLOG.md` "K-047 — `server/tests/eval/generate_report.py` has zero
automated test coverage").

**Genuinely network/DB-free**: `generate_report.py` only reads local JSON/JSONL
fixture files and a filesystem path for its output — no FalkorDB, no live LLM.
This file never touches the real committed eval artifacts
(`retrieval_baseline.json`, `judge_calibration.json`, `corpus_provenance.json`,
`golden_retrieval.jsonl`); every test either monkeypatches the module's own path
constants to point at `tmp_path`-created fixtures, or calls the lower-level
`_render_*`/`_self_retrieval_guard_failures`/`_load_retrieval_baseline` functions
directly with hand-built dicts.

Four branches were flagged (analyst code gate at K-026 close, non-blocking
per decision D1; independently re-confirmed correct by `qa-engineer`'s
acceptance pass, `docs/test-reports/graphrag-eval-report.md` "Exploratory
findings (TP-011)") as manually/statically verified but automated-test-free:

  1. not-run marker when `judge_calibration.json` is absent;
  2. same-model vs. differs caveat selection, verbatim, adjacent to the numbers;
  3. self-retrieval-guard failure path (PASS/FAIL + offending row ids);
  4. missing-baseline `ReportError`, propagated uncaught by `build_report()`
     and caught by `main()` (stderr + non-zero exit, no report file written).
"""

from __future__ import annotations

import json

import generate_report as gr
import pytest

# ── fixtures / builders ────────────────────────────────────────────────────


def _judge_dict(*, same_model: bool) -> dict:
    return {
        "judgeModel": "qwen3-judge",
        "agentUnderTestModel": "qwen3-judge" if same_model else "qwen3-agent",
        "generatedAt": "2026-08-16T00:00:00+00:00",
        "sameModelAsAgentUnderTest": same_model,
        "calibration": {
            "sampleSize": 10,
            "faithfulnessAgreement": 0.9,
            "relevanceAgreement": 0.7,
            "parseFailures": 0,
        },
        "generation": {
            "sampleSize": 20,
            "faithfulTrue": 18,
            "faithfulFalse": 1,
            "faithfulAbstained": 1,
            "relevantTrue": 15,
            "relevantFalse": 5,
            "parseFailures": 0,
        },
    }


def _golden_row(row_id: str, query: str, target_text: str) -> dict:
    return {"id": row_id, "query": query, "target_text": target_text}


# ── branch 1: not-run marker ────────────────────────────────────────────────


def test_render_judge_section_not_run_marker_when_judge_is_none() -> None:
    section = gr._render_judge_section(None)
    assert "**Not run**" in section
    assert "judge_calibration.json` does not exist" in section


def test_render_judge_section_not_run_never_fabricates_numbers() -> None:
    """The not-run branch must never emit a percentage/number — those only ever
    come from an actual `judge_calibration.json`."""
    section = gr._render_judge_section(None)
    assert "%" not in section
    assert "sample size" not in section


# ── branch 2: same-model vs. differs caveat selection ───────────────────────


def test_render_judge_section_same_model_emits_caveat_verbatim_adjacent() -> None:
    judge = _judge_dict(same_model=True)
    section = gr._render_judge_section(judge)

    expected_caveat = gr._SAME_MODEL_CAVEAT_TEMPLATE.format(judge_model=judge["judgeModel"])
    assert expected_caveat in section, (
        "same-model caveat must be emitted verbatim, not paraphrased"
    )
    assert "differs from the agent-under-test" not in section

    # Adjacent, not a trailing footnote: the caveat must immediately follow the
    # generation sub-pass numbers, with nothing else rendered after it.
    parse_failures_idx = section.index("### Generation sub-pass")
    caveat_idx = section.index(expected_caveat)
    assert caveat_idx > parse_failures_idx
    assert section.rstrip().endswith(expected_caveat.rstrip())


def test_render_judge_section_differs_emits_plain_sentence_not_caveat() -> None:
    judge = _judge_dict(same_model=False)
    section = gr._render_judge_section(judge)

    assert "Same-model judge limitation" not in section
    assert (
        f"Judge model (`{judge['judgeModel']}`) differs from the agent-under-test"
        f" model (`{judge['agentUnderTestModel']}`)" in section
    )
    assert "self-preference-bias caveat does not apply" in section


# ── branch 3: self-retrieval-guard failure path ─────────────────────────────


def test_self_retrieval_guard_detects_query_substring_of_target() -> None:
    rows = [_golden_row("leak-1", "hello world", "say hello world now")]
    assert gr._self_retrieval_guard_failures(rows) == ["leak-1"]


def test_self_retrieval_guard_detects_target_substring_of_query() -> None:
    rows = [_golden_row("leak-2", "please say hello world now", "hello world")]
    assert gr._self_retrieval_guard_failures(rows) == ["leak-2"]


def test_self_retrieval_guard_clean_rows_report_no_failures() -> None:
    rows = [
        _golden_row("clean-1", "what is the deployment process", "we ship via CI"),
        _golden_row("clean-2", "who owns the retrieval baseline", "the DBA signs off"),
    ]
    assert gr._self_retrieval_guard_failures(rows) == []


@pytest.mark.parametrize("leak_position", ["first", "middle", "last"])
def test_self_retrieval_guard_scans_every_row_regardless_of_position(
    leak_position: str,
) -> None:
    """The guard loops over every row — a leak must be caught whether it's the
    first, middle, or last row scanned, not just row 0."""
    leaking = _golden_row("leak", "hello world", "say hello world now")
    clean_a = _golden_row("clean-a", "what is the deployment process", "we ship via CI")
    clean_b = _golden_row("clean-b", "who owns the retrieval baseline", "the DBA signs off")

    rows_by_position = {
        "first": [leaking, clean_a, clean_b],
        "middle": [clean_a, leaking, clean_b],
        "last": [clean_a, clean_b, leaking],
    }
    assert gr._self_retrieval_guard_failures(rows_by_position[leak_position]) == ["leak"]


def test_render_corpus_section_reports_fail_with_offending_id() -> None:
    rows = [_golden_row("leak-1", "hello world", "say hello world now")]
    section = gr._render_corpus_section(None, rows)
    assert "**FAIL**" in section
    assert "leak-1" in section


def test_render_corpus_section_reports_pass_when_clean() -> None:
    rows = [
        _golden_row("clean-1", "what is the deployment process", "we ship via CI"),
    ]
    section = gr._render_corpus_section(None, rows)
    assert "**PASS**" in section
    assert "**FAIL**" not in section


# ── branch 4: missing-baseline error ────────────────────────────────────────


def test_load_retrieval_baseline_raises_report_error_when_missing(tmp_path, monkeypatch) -> None:
    missing_path = tmp_path / "retrieval_baseline.json"
    assert not missing_path.exists()
    monkeypatch.setattr(gr, "_RETRIEVAL_BASELINE_PATH", missing_path)

    with pytest.raises(gr.ReportError, match="does not exist"):
        gr._load_retrieval_baseline()


def test_load_retrieval_baseline_returns_dict_when_present(tmp_path, monkeypatch) -> None:
    path = tmp_path / "retrieval_baseline.json"
    path.write_text(json.dumps({"recall_at_10": 0.9}), encoding="utf-8")
    monkeypatch.setattr(gr, "_RETRIEVAL_BASELINE_PATH", path)

    assert gr._load_retrieval_baseline() == {"recall_at_10": 0.9}


def test_build_report_propagates_report_error_uncaught_when_baseline_missing(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(gr, "_RETRIEVAL_BASELINE_PATH", tmp_path / "retrieval_baseline.json")

    with pytest.raises(gr.ReportError):
        gr.build_report()


def test_main_missing_baseline_prints_error_and_returns_nonzero_without_writing(
    tmp_path, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(gr, "_RETRIEVAL_BASELINE_PATH", tmp_path / "retrieval_baseline.json")
    reports_dir = tmp_path / "reports"
    monkeypatch.setattr(gr, "_REPORTS_DIR", reports_dir)

    exit_code = gr.main()

    assert exit_code == 1
    captured = capsys.readouterr()
    assert captured.out == "", "no success/wrote message on the error path"
    assert captured.err.startswith("error: ")
    assert "does not exist" in captured.err
    assert not reports_dir.exists(), (
        "main() must never create the reports dir or write a partial/garbage "
        "report file on the error path"
    )
