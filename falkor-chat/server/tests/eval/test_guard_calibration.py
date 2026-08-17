"""Offline, network-free unit tests for the guard-judge calibration harness
(K-027 item 3, `docs/plans/guard-judge-calibration-ml.md` §7).

Everything here drives `tests/eval/guard_calibration.py`'s deterministic plumbing
against **stub** judges — no live LM Studio, no network. The live calibration run
itself lives in `test_guard_calibration_live.py`, behind the `live` marker.

Covers:
  * fixture-row → `guards.evaluate_guard` call assembly (archived §5.1's table, with
    the ml-note §4 `run` addendum already applied by `guard_calibration.build_call`);
  * the path-was-taken assertion (archived §5.1: "not optional" — a `path: "turns"`
    case that silently ran the understanding branch is worthless data);
  * the G1 (per-call, not per-case-majority) / G2 (per-case-majority) counting rules
    (archived §4.1/§5.2 — these are the two rules most likely to get silently swapped,
    hence mutation-tested, see this module's kaizen inbox entry / the task report);
  * kappa with marginals + prevalence, `coercion_flip_rate`, `flip_rate`.
"""

from __future__ import annotations

import json

import pytest

from falkorchat.guards import evaluate_guard

from guard_calibration import (
    CaseResult,
    ReplicateResult,
    RecordingJudge,
    advance_recall,
    assert_path_matches,
    build_call,
    case_majority,
    coercion_flip_rate,
    cohens_kappa,
    confusion_matrix,
    false_advance_rate,
    false_advance_rate_all,
    flip_rate,
    materiality_probe_failed,
    path_taken,
    per_path_breakdown,
    run_case,
)


class StubJudge:
    """A scripted judge — mirrors `tests/test_guards.py`'s own `StubJudge`, kept
    local (rather than imported) since that one lacks `accepts_run`/`model` params
    and this harness must exercise the K-042 `accepts_run` capability flag too.
    """

    accepts_run = True

    def __init__(self, output):
        self.output = output
        self.calls: list[dict] = []

    def __call__(self, condition, *, understanding, recent_turns, ctx, step_output,
                 model=None, run=None):
        self.calls.append({
            "condition": condition, "understanding": understanding,
            "recent_turns": recent_turns, "ctx": ctx, "step_output": step_output,
            "model": model, "run": run,
        })
        return self.output


UNDERSTANDING_ROW = {
    "id": "ca-01", "tier": "clear_advance", "path": "understanding", "r1_probe": True,
    "condition": "the user has provided enough information to research their request",
    "understanding": {
        "request": "Find out why the checkout service started returning 502 errors",
        "known": ["service: checkout", "symptom: HTTP 502"], "missing": [],
    },
    "turns": [], "expected": True,
    "label_rationale": "irrelevant to the harness",
}

TURNS_ROW = {
    "id": "tn-04", "tier": "clear_suspend", "path": "turns", "r1_probe": False,
    "condition": "the user has provided enough information to research their request",
    "understanding": {},
    "turns": [{
        "msgId": "m-tn04-1", "text": "hey, got a problem", "role": "user",
        "createdAt": 1752570300000, "authorId": "u-dana", "displayName": "Dana",
        "authorType": ["User"],
    }],
    "expected": False,
    "label_rationale": "irrelevant to the harness",
}


# ── fixture → call assembly (archived §5.1 + ml-note §4 addendum) ────────────


def test_build_call_understanding_path_envelope_and_run_addendum():
    call = build_call(UNDERSTANDING_ROW)
    guard = json.loads(call["guard"])
    assert guard == {
        "kind": "llm",
        "text": "the user has provided enough information to research their request",
    }
    step_output = json.loads(call["step_output"])
    assert step_output == {"understanding": UNDERSTANDING_ROW["understanding"]}
    # ml-note §4: run = {"ws": "ws:golden-eval", "modelOverrides": {}} — NOT the
    # archived doc's literal run={}, which lands in a branch production never takes.
    assert call["run"] == {"ws": "ws:golden-eval", "modelOverrides": {}}
    assert call["thread"] == []


def test_build_call_turns_path_step_output_does_not_parse_and_ctx_has_no_understanding():
    call = build_call(TURNS_ROW)
    # F5: step_output must NOT parse to a non-empty dict, and ctx must carry no
    # `understanding` key — all three conditions, or the case silently tests the
    # wrong (primary) path.
    with pytest.raises((json.JSONDecodeError, ValueError)):
        parsed = json.loads(call["step_output"])
        if isinstance(parsed, dict) and parsed:
            raise ValueError("step_output must not parse to a non-empty dict")
    assert "understanding" not in call["ctx"]
    assert call["thread"] == TURNS_ROW["turns"]


def test_build_call_understanding_row_actually_drives_evaluate_guard_understanding_path():
    """Not just a shape check — asserts the *real* `evaluate_guard` derives the
    understanding evidence tier from this call, per archived F1's whole point."""
    judge = StubJudge({"decision": True, "rationale": "all present"})
    call = build_call(UNDERSTANDING_ROW)
    evaluate_guard(call["guard"], ctx=call["ctx"], run=call["run"],
                    step_output=call["step_output"], thread=call["thread"], judge=judge)
    assert judge.calls[0]["understanding"] == UNDERSTANDING_ROW["understanding"]
    assert judge.calls[0]["recent_turns"] == []


def test_build_call_turns_row_actually_drives_evaluate_guard_turns_fallback():
    judge = StubJudge({"decision": False, "rationale": "nothing to go on"})
    call = build_call(TURNS_ROW)
    evaluate_guard(call["guard"], ctx=call["ctx"], run=call["run"],
                    step_output=call["step_output"], thread=call["thread"], judge=judge)
    assert judge.calls[0]["understanding"] == {}
    assert judge.calls[0]["recent_turns"] != []
    assert judge.calls[0]["recent_turns"][0]["text"] == "hey, got a problem"


def test_run_addendum_forwards_run_only_when_judge_accepts_it():
    """A stub judge without `accepts_run` (the pre-K-042 shape, 24 existing stub
    judges in `test_guards.py`) must never receive `run=` — mirrors guards.py's own
    zero-churn contract."""
    class LegacyStubJudge:
        def __init__(self, output):
            self.output = output
            self.calls = []

        def __call__(self, condition, *, understanding, recent_turns, ctx, step_output):
            self.calls.append(True)
            return self.output

    judge = LegacyStubJudge({"decision": True, "rationale": "x"})
    call = build_call(UNDERSTANDING_ROW)
    # Must not raise TypeError even though run= is in the call dict.
    evaluate_guard(call["guard"], ctx=call["ctx"], run=call["run"],
                    step_output=call["step_output"], thread=call["thread"], judge=judge)
    assert judge.calls == [True]


# ── the path-was-taken assertion (archived §5.1: "not optional") ─────────────


def test_assert_path_matches_raises_when_turns_row_silently_used_understanding():
    call = {"understanding": {"request": "x"}, "recent_turns": []}
    with pytest.raises(AssertionError, match="turns"):
        assert_path_matches(TURNS_ROW, call)


def test_assert_path_matches_raises_when_understanding_row_silently_fell_back_to_turns():
    call = {"understanding": {}, "recent_turns": [{"speaker": "m", "role": "user", "text": "x"}]}
    with pytest.raises(AssertionError, match="understanding"):
        assert_path_matches(UNDERSTANDING_ROW, call)


def test_assert_path_matches_passes_on_a_correctly_routed_call():
    call = {"understanding": UNDERSTANDING_ROW["understanding"], "recent_turns": []}
    assert_path_matches(UNDERSTANDING_ROW, call)  # must not raise


def test_run_case_raises_when_a_live_judge_answer_silently_drove_the_wrong_path():
    """Guards against a harness regression where `build_call` stops synthesizing a
    non-parsing `step_output` for a turns case — `run_case` must fail loudly, not
    silently score a worthless data point."""
    class WrongPathRow(dict):
        pass

    bad_row = dict(TURNS_ROW)
    bad_row["understanding"] = {"request": "leaked in"}  # simulate F5 violation upstream

    def fake_build_call(row):
        # A deliberately broken assembly: step_output DOES parse and carries data,
        # so evaluate_guard takes the understanding path even though the fixture
        # declares path="turns".
        return {
            "guard": json.dumps({"kind": "llm", "text": row["condition"]}),
            "ctx": {}, "run": {"ws": "ws:golden-eval", "modelOverrides": {}},
            "step_output": json.dumps({"understanding": {"request": "leaked in"}}),
            "thread": row["turns"],
        }

    import guard_calibration as gc
    original = gc.build_call
    gc.build_call = fake_build_call
    try:
        with pytest.raises(AssertionError):
            run_case(bad_row, StubJudge({"decision": True, "rationale": "x"}), k=1)
    finally:
        gc.build_call = original


# ── run_case + metrics (archived §4, §5.2) ────────────────────────────────────


def _case(id_, tier, path, r1_probe, expected, final_decisions, raw_decisions=None):
    """Build a `CaseResult` directly (no live judge) for metric-function tests."""
    raw_decisions = raw_decisions or final_decisions
    replicates = [
        ReplicateResult(
            raw_decision=raw, raw_rationale="r", final_decision=final, final_rationale="r",
        )
        for raw, final in zip(raw_decisions, final_decisions)
    ]
    return CaseResult(id=id_, tier=tier, path=path, r1_probe=r1_probe, expected=expected,
                       replicates=replicates)


def test_run_case_executes_k_replicates_and_records_final_and_raw_decisions():
    judge = StubJudge({"decision": True, "rationale": "all present"})
    result = run_case(UNDERSTANDING_ROW, judge, k=3)
    assert result.id == "ca-01"
    assert len(result.replicates) == 3
    assert all(r.final_decision is True for r in result.replicates)
    assert all(r.raw_decision is True for r in result.replicates)
    assert len(judge.calls) == 3  # one real judge call per replicate


def test_run_case_coercion_flip_when_rationale_contradicts_decision():
    """`_coerce_verdict` bias-to-suspend: `decision:true` + a contradicting
    rationale coerces to `final_decision=False` while `raw_decision` stays True."""
    judge = StubJudge({"decision": True, "rationale": "not enough information yet"})
    result = run_case(UNDERSTANDING_ROW, judge, k=1)
    rep = result.replicates[0]
    assert rep.raw_decision is True
    assert rep.final_decision is False


def test_case_majority_two_of_three():
    reps_true = [ReplicateResult(True, "r", True, "r"), ReplicateResult(True, "r", True, "r"),
                 ReplicateResult(False, "r", False, "r")]
    assert case_majority(reps_true) is True
    reps_false = [ReplicateResult(False, "r", False, "r"), ReplicateResult(False, "r", False, "r"),
                  ReplicateResult(True, "r", True, "r")]
    assert case_majority(reps_false) is False


def test_g1_false_advance_rate_is_per_call_not_per_case_majority():
    """archived §4.1/§5.2: a `clear_suspend` case that advances 1-in-3 times must
    contribute 1 false advance to G1's numerator, NOT be washed out by majority
    voting to "suspend" (2 suspends beats 1 advance)."""
    cases = [
        # This case's majority is "suspend" (2 False, 1 True) — but the ONE
        # advancing call is still a real false-advance on the live (single-sample)
        # path, and G1 must count it.
        _case("cs-01", "clear_suspend", "understanding", False, False,
              [False, False, True]),
        _case("cs-02", "clear_suspend", "understanding", False, False,
              [False, False, False]),
    ]
    g1 = false_advance_rate(cases)
    assert g1["n_calls"] == 6
    assert g1["advances"] == 1  # the single True call, not 0 (which per-case-majority would give)
    assert g1["rate"] == pytest.approx(1 / 6)


def test_g1_excludes_boundary_cases_from_denominator():
    cases = [
        _case("cs-01", "clear_suspend", "understanding", False, False, [False, False, False]),
        _case("bd-01", "boundary", "understanding", False, False, [True, True, True]),
    ]
    g1 = false_advance_rate(cases)
    assert g1["n_calls"] == 3  # boundary excluded
    assert g1["advances"] == 0


def test_far_all_includes_boundary_cases():
    cases = [
        _case("cs-01", "clear_suspend", "understanding", False, False, [False, False, False]),
        _case("bd-01", "boundary", "understanding", False, False, [True, True, True]),
    ]
    far_all = false_advance_rate_all(cases)
    assert far_all["n_calls"] == 6
    assert far_all["advances"] == 3


def test_g2_advance_recall_is_per_case_majority():
    cases = [
        # Majority advance (2 True, 1 False) → counts as ONE advancing case for G2,
        # not 2/3 of a case.
        _case("ca-01", "clear_advance", "understanding", False, True, [True, True, False]),
        _case("ca-02", "clear_advance", "understanding", False, True, [False, False, True]),
    ]
    g2 = advance_recall(cases)
    assert g2["n_cases"] == 2
    assert g2["advances"] == 1
    assert g2["rate"] == pytest.approx(0.5)


def test_confusion_matrix_counts_tp_fp_tn_fn():
    cases = [
        _case("ca-01", "clear_advance", "understanding", False, True, [True, True, True]),
        _case("ca-02", "clear_advance", "understanding", False, True, [False, False, False]),
        _case("cs-01", "clear_suspend", "understanding", False, False, [False, False, False]),
        _case("cs-02", "clear_suspend", "understanding", False, False, [True, True, True]),
    ]
    cm = confusion_matrix(cases)
    assert cm == {"tp": 1, "fn": 1, "tn": 1, "fp": 1}


def test_cohens_kappa_perfect_agreement_is_one():
    cases = [
        _case("ca-01", "clear_advance", "understanding", False, True, [True, True, True]),
        _case("cs-01", "clear_suspend", "understanding", False, False, [False, False, False]),
    ]
    result = cohens_kappa(cases)
    assert result["kappa"] == pytest.approx(1.0)
    assert result["prevalence"] == pytest.approx(0.5)
    assert result["rater_a_positive_rate"] == pytest.approx(0.5)
    assert result["rater_b_positive_rate"] == pytest.approx(0.5)


def test_cohens_kappa_reports_marginals_even_with_no_agreement():
    cases = [
        _case("ca-01", "clear_advance", "understanding", False, True, [False, False, False]),
        _case("cs-01", "clear_suspend", "understanding", False, False, [True, True, True]),
    ]
    result = cohens_kappa(cases)
    assert result["kappa"] == pytest.approx(-1.0)


def test_coercion_flip_rate_counts_raw_true_final_false():
    cases = [
        _case("ca-01", "clear_advance", "understanding", True, True,
              final_decisions=[False], raw_decisions=[True]),  # coerced
        _case("ca-02", "clear_advance", "understanding", False, True,
              final_decisions=[True], raw_decisions=[True]),  # not coerced
    ]
    overall = coercion_flip_rate(cases)
    assert overall == {"flips": 1, "n_calls": 2, "rate": 0.5}
    r1_only = coercion_flip_rate(cases, r1_probe_only=True)
    assert r1_only == {"flips": 1, "n_calls": 1, "rate": 1.0}


def test_flip_rate_counts_cases_with_disagreeing_replicates():
    cases = [
        _case("ca-01", "clear_advance", "understanding", False, True, [True, True, True]),
        _case("ca-02", "clear_advance", "understanding", False, True, [True, False, True]),
    ]
    result = flip_rate(cases)
    assert result["unstable"] == 1
    assert result["unstable_ids"] == ["ca-02"]
    assert result["rate"] == pytest.approx(0.5)


def test_per_path_breakdown_separates_understanding_and_turns():
    cases = [
        _case("ca-01", "clear_advance", "understanding", False, True, [True, True, True]),
        _case("tn-01", "clear_advance", "turns", False, True, [False, False, False]),
    ]
    breakdown = per_path_breakdown(cases)
    assert breakdown["understanding"]["n_cases"] == 1
    assert breakdown["understanding"]["accuracy"] == pytest.approx(1.0)
    assert breakdown["turns"]["n_cases"] == 1
    assert breakdown["turns"]["accuracy"] == pytest.approx(0.0)


def test_materiality_probe_failed_true_only_when_all_probes_suspend_and_control_advances():
    cases_by_id = {
        "ca-04": _case("ca-04", "clear_advance", "understanding", False, True, [False, False, False]),
        "ca-05": _case("ca-05", "clear_advance", "understanding", False, True, [False, False, False]),
        "ca-08": _case("ca-08", "clear_advance", "understanding", False, True, [False, False, False]),
        "cs-04": _case("cs-04", "clear_suspend", "understanding", False, False, [True, True, True]),
    }
    assert materiality_probe_failed(cases_by_id) is True


def test_materiality_probe_not_failed_when_one_probe_advances():
    cases_by_id = {
        "ca-04": _case("ca-04", "clear_advance", "understanding", False, True, [True, True, True]),
        "ca-05": _case("ca-05", "clear_advance", "understanding", False, True, [False, False, False]),
        "ca-08": _case("ca-08", "clear_advance", "understanding", False, True, [False, False, False]),
        "cs-04": _case("cs-04", "clear_suspend", "understanding", False, False, [True, True, True]),
    }
    assert materiality_probe_failed(cases_by_id) is False
