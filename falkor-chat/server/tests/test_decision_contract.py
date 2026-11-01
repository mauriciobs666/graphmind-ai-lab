"""The guard decision point's contract with model-bench (typed-decision-models U3b, plan
`docs/plans/typed-decision-models.md` §3.2, §3.2.2, §4 U3b, §5 U3b).

model-bench measures the guard by transcribing `app._render_judge_user` and the hand-off verdict
rule (`llm.extract_own_line_json_object` then `guards._coerce_verdict`) — never by importing them.
model-bench owns the canonical golden vectors (`model-bench/tests/fixtures/decision-contract/`);
this suite asserts falkor-chat's real functions against **byte-copies** of two of them under
`tests/data/decision-contract/`, so a change on either side reddens before anything is measured
against a stale copy:

* `render-guard.jsonl` — `_render_judge_user(condition, understanding, recent_turns)`;
* `handoff-parse-guard.jsonl` — a judge reply through the production judge
  (`_build_llm_judge` over a stub LLM) and `_coerce_verdict`: `advance` is the verdict's
  `decision`; `parseFailure` is whether the own-line parse found no verdict;
* `normalize-turns.jsonl` — `guards._recent_turns(thread)`, the turn normalization the guard
  render and the join-in render share (typed-decision-models impl review P2-F3).

All three are written by `scripts/generate_guard_decision_vectors.py` (human-invoked, this tree's
venv), which also iterates `guards`' cue and negator tables for its probe rows.

The guarded equality test asserts each copy byte-identical to model-bench's, and skips only when
`model-bench/` is absent from the checkout (§3.2.2's skip rule). A gate on this unit runs it
un-skipped (plan R-10). Offline: no FalkorDB, no LLM.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from falkorchat import guards, llm
from falkorchat.app import _build_llm_judge, _render_judge_user
from falkorchat.guards import _coerce_verdict

_COPIES = Path(__file__).resolve().parent / "data" / "decision-contract"
_MODEL_BENCH = Path(__file__).resolve().parents[3] / "model-bench"
_CANONICAL = _MODEL_BENCH / "tests" / "fixtures" / "decision-contract"
_VECTOR_FILES = ("render-guard.jsonl", "handoff-parse-guard.jsonl", "normalize-turns.jsonl")


def _rows(name: str) -> list[dict[str, Any]]:
    with (_COPIES / name).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


_RENDER_ROWS = _rows("render-guard.jsonl")
_PARSE_ROWS = _rows("handoff-parse-guard.jsonl")
_NORMALIZE_ROWS = _rows("normalize-turns.jsonl")


@pytest.mark.parametrize("name", _VECTOR_FILES)
def test_the_vector_copy_is_byte_identical_to_model_benchs(name: str) -> None:
    if not _MODEL_BENCH.is_dir():
        pytest.skip("model-bench/ is absent from this checkout (plan §3.2.2 skip rule)")
    assert (_COPIES / name).read_bytes() == (_CANONICAL / name).read_bytes()


@pytest.mark.parametrize(
    ("name", "rows"),
    [
        ("render-guard.jsonl", _RENDER_ROWS),
        ("handoff-parse-guard.jsonl", _PARSE_ROWS),
        ("normalize-turns.jsonl", _NORMALIZE_ROWS),
    ],
)
def test_every_vector_line_is_a_parametrized_row(name: str, rows: list[dict[str, Any]]) -> None:
    lines = [line for line in (_COPIES / name).read_bytes().splitlines() if line.strip()]
    assert len(rows) == len(lines)


def _true_verdict_rationale(reply: Any) -> str | None:
    """The lower-cased rationale of a whole-reply `{"decision": true, "rationale": <str>}`."""
    try:
        parsed = json.loads(reply)
    except (TypeError, ValueError):
        return None
    if not isinstance(parsed, dict) or parsed.get("decision") is not True:
        return None
    rationale = parsed.get("rationale")
    return rationale.lower() if isinstance(rationale, str) else None


def test_every_cue_and_negator_in_guards_has_a_row_of_its_polarity() -> None:
    """Every `guards._NEGATION_CUES` entry has a parsed `true` row where it stands alone,
    un-negated, and contradicts; every (`guards._NEGATORS`, cue) pair has one where it affirms.
    A cue or negator added here reddens until the vectors are regenerated
    (`scripts/generate_guard_decision_vectors.py`) and re-copied from model-bench."""
    cues, negators = guards._NEGATION_CUES, guards._NEGATORS
    rows = [(r["expected"], _true_verdict_rationale(r["reply"])) for r in _PARSE_ROWS]
    rows = [(e, r) for e, r in rows if r is not None and not e["parseFailure"]]
    for cue in cues:
        assert any(
            not expected["advance"]
            and cue in rationale
            and not any(neg + cue in rationale for neg in negators)
            and not any(other in rationale.replace(cue, "|") for other in cues if other != cue)
            for expected, rationale in rows
        ), f"no contradicting row for cue {cue!r}"
        for negator in negators:
            assert any(
                expected["advance"] and negator + cue in rationale for expected, rationale in rows
            ), f"no affirming row for {negator!r} + {cue!r}"


@pytest.mark.parametrize("row", _RENDER_ROWS, ids=[r["case"] for r in _RENDER_ROWS])
def test_render_judge_user_matches_render_guard_jsonl(row: dict[str, Any]) -> None:
    rendered = _render_judge_user(row["condition"], row["understanding"], row["recentTurns"])
    assert rendered == row["expected"]


class _ReplyLLM:
    """Stub LLM returning one canned judge reply."""

    def __init__(self, text: Any) -> None:
        self._text = text

    def complete(self, messages: Any) -> Any:
        return self._text


@pytest.mark.parametrize("row", _PARSE_ROWS, ids=[r["case"] for r in _PARSE_ROWS])
def test_the_guard_handoff_verdict_matches_handoff_parse_guard_jsonl(row: dict[str, Any]) -> None:
    raw = _build_llm_judge(_ReplyLLM(row["reply"]))(
        "enough info?", understanding={}, recent_turns=[], ctx={}, step_output=""
    )
    verdict = _coerce_verdict(raw)
    parsed = llm.extract_own_line_json_object(row["reply"], require_key="decision")
    assert verdict.decision is row["expected"]["advance"]
    assert (parsed is None) is row["expected"]["parseFailure"]


@pytest.mark.parametrize("row", _NORMALIZE_ROWS, ids=[r["case"] for r in _NORMALIZE_ROWS])
def test_recent_turns_matches_normalize_turns_jsonl(row: dict[str, Any]) -> None:
    assert guards._recent_turns(row["thread"]) == row["expected"]
