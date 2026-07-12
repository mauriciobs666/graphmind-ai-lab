"""Unit tests for fuzzy (LLM-judged) + deterministic transition-guard evaluation (U7).

`guards.evaluate_guard` dispatches on the serialized guard `kind`:
  * `""` (empty) → unconditional (fires whenever reached);
  * `{"kind":"llm", ...}` → the Q1 extract-then-judge path via the **injected** judge,
    with the ambiguity policy (malformed / contradictory judge output → bias-to-suspend);
  * any other kind (e.g. `expr`) → a pure `NotImplementedError` seam (M7).

The judge is an injected callable so the whole module is stub-testable offline — no LLM.
"""

from __future__ import annotations

import json

import pytest

from falkorchat.guards import GuardVerdict, evaluate_guard


class StubJudge:
    """A scripted LLM-guard judge returning a fixed `{decision, rationale}`-ish output.

    Records the extract-then-judge inputs it was handed so the tests can assert the
    compact `understanding` (not the raw transcript) is what reached the judge (Q1).
    """

    def __init__(self, output):
        self.output = output
        self.calls: list[dict] = []

    def __call__(self, condition, *, understanding, ctx, step_output):
        self.calls.append(
            {"condition": condition, "understanding": understanding,
             "ctx": ctx, "step_output": step_output}
        )
        return self.output


def _boom_judge(*args, **kwargs):  # a judge that must never be called
    raise AssertionError("judge must not be called for this guard")


# ── unconditional (empty guard) ──────────────────────────────────────────────

def test_empty_guard_is_unconditional_and_never_calls_the_judge():
    verdict = evaluate_guard(
        "", ctx={}, run={}, step_output="", thread=None, judge=_boom_judge
    )
    assert isinstance(verdict, GuardVerdict)
    assert verdict.decision is True


# ── llm guard — advance / suspend ────────────────────────────────────────────

def test_llm_guard_fires_when_judge_decides_true():
    judge = StubJudge({"decision": True, "rationale": "all fields present"})
    verdict = evaluate_guard(
        '{"kind":"llm","text":"enough info?"}',
        ctx={}, run={}, step_output="", thread=None, judge=judge,
    )
    assert verdict.decision is True
    assert verdict.rationale == "all fields present"
    assert judge.calls[0]["condition"] == "enough info?"


def test_llm_guard_suspends_when_judge_decides_false():
    judge = StubJudge({"decision": False, "rationale": "username still missing"})
    verdict = evaluate_guard(
        '{"kind":"llm","text":"enough info?"}',
        ctx={}, run={}, step_output="", thread=None, judge=judge,
    )
    assert verdict.decision is False
    assert "username" in verdict.rationale


# ── Q1 ambiguity policy — bias-to-suspend ────────────────────────────────────

def test_bias_to_suspend_on_non_mapping_judge_output():
    # parse failure / invalid output — the judge handed back a bare string
    verdict = evaluate_guard(
        '{"kind":"llm","text":"enough info?"}',
        ctx={}, run={}, step_output="", thread=None,
        judge=StubJudge("not a json object"),
    )
    assert verdict.decision is False


def test_bias_to_suspend_when_decision_missing_or_non_bool():
    for bad in ({"rationale": "no decision key"}, {"decision": "true"}, {"decision": 1}):
        verdict = evaluate_guard(
            '{"kind":"llm","text":"enough info?"}',
            ctx={}, run={}, step_output="", thread=None, judge=StubJudge(bad),
        )
        assert verdict.decision is False, bad


def test_bias_to_suspend_when_rationale_contradicts_a_true_decision():
    # decision says advance but the rationale explicitly says it is not enough —
    # an internally inconsistent verdict is treated as false (bias-to-suspend).
    judge = StubJudge({"decision": True, "rationale": "not enough information yet"})
    verdict = evaluate_guard(
        '{"kind":"llm","text":"enough info?"}',
        ctx={}, run={}, step_output="", thread=None, judge=judge,
    )
    assert verdict.decision is False


# ── extract-then-judge: the judge sees the compact understanding ─────────────

def test_understanding_is_extracted_from_step_output_and_handed_to_the_judge():
    understanding = {"request": "reset password", "known": ["email"],
                     "missing": ["username"]}
    judge = StubJudge({"decision": False, "rationale": "x"})

    evaluate_guard(
        '{"kind":"llm","text":"enough info?"}',
        ctx={}, run={}, step_output=json.dumps(understanding),
        thread=None, judge=judge,
    )

    assert judge.calls[0]["understanding"] == understanding


# ── expr / unknown kind — pure NotImplementedError seam (M7) ─────────────────

def test_expr_kind_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        evaluate_guard(
            '{"kind":"expr","expr":"x > 1"}',
            ctx={}, run={}, step_output="", thread=None, judge=_boom_judge,
        )


def test_unknown_non_empty_guard_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        evaluate_guard(
            "just some prose", ctx={}, run={}, step_output="",
            thread=None, judge=_boom_judge,
        )
