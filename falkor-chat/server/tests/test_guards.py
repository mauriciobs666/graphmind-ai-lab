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

from falkorchat.guards import (
    TURN_TEXT_MAX,
    GuardVerdict,
    WorkflowConfigError,
    evaluate_guard,
)


class StubJudge:
    """A scripted LLM-guard judge returning a fixed `{decision, rationale}`-ish output.

    Records the extract-then-judge inputs it was handed so the tests can assert **both**
    tiers of the DS §Q1 evidence contract reached the judge: the compact `understanding`
    (primary) and the RECENT-TURNS fallback (`recent_turns`, only when no understanding
    was emitted).
    """

    def __init__(self, output):
        self.output = output
        self.calls: list[dict] = []

    def __call__(self, condition, *, understanding, recent_turns, ctx, step_output):
        self.calls.append(
            {"condition": condition, "understanding": understanding,
             "recent_turns": recent_turns, "ctx": ctx, "step_output": step_output}
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


# ── null guard — treated as unconditional (n-2 safety net) ───────────────────

def test_none_guard_is_unconditional_and_never_calls_the_judge():
    # A hand-crafted / pre-existing transition may carry a null guard. `None`
    # means "no condition", so it fires when reached (like ""), never raising
    # NotImplementedError (which would orphan a drive → M-1). n-2.
    verdict = evaluate_guard(
        None, ctx={}, run={}, step_output="", thread=None, judge=_boom_judge
    )
    assert verdict.decision is True


# ── llm guard with no judge wired — named config error (m-3) ──────────────────

def test_llm_guard_without_judge_raises_named_config_error():
    # An {kind:'llm'} guard driven by an executor built without a guard_judge must
    # raise a NAMED WorkflowConfigError (not a bare TypeError from calling None(...)),
    # so the M-1 fault net can fail the run with a clear diagnostic. m-3.
    with pytest.raises(WorkflowConfigError) as exc:
        evaluate_guard(
            '{"kind":"llm","text":"enough info?"}',
            ctx={}, run={}, step_output="", thread=None, judge=None,
        )
    assert "judge" in str(exc.value).lower()


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


# ── the RECENT-TURNS fallback + the DS precedence (Defect A) ─────────────────
#
# DS note §Q1: the judge is fed the compact `understanding` (primary) plus "RECENT TURNS
# (context only) … N = 6, newest last; omit if understanding is present". The `thread`
# parameter was a declared seam no callee honored — the judge was handed `{}` every turn
# and correctly biased to suspend forever (Defect A). These pin the restored contract.

LLM_GUARD = '{"kind":"llm","text":"enough info?"}'


def _rows(n, *, text=None):
    """`repository.read_thread`-shaped rows, chronological (oldest first)."""
    return [
        {"msgId": f"m{i}", "text": text or f"turn {i}", "role": "user",
         "createdAt": 1000 + i, "authorId": "u1", "displayName": "Alice",
         "authorType": "User"}
        for i in range(n)
    ]


def test_recent_turns_are_omitted_when_an_understanding_was_emitted():
    # T1 — the DS omit rule: the understanding is the primary evidence; the raw turns are
    # only the fallback, so feeding both would re-introduce the long-context dilution the
    # extract-then-judge method exists to avoid.
    judge = StubJudge({"decision": False, "rationale": "x"})
    understanding = {"request": "r", "known": ["k"], "missing": []}

    evaluate_guard(
        LLM_GUARD, ctx={}, run={},
        step_output=json.dumps({"understanding": understanding}),
        thread=_rows(10), judge=judge,
    )

    assert judge.calls[0]["understanding"] == understanding
    assert judge.calls[0]["recent_turns"] == []


def test_last_six_turns_are_handed_to_the_judge_when_no_understanding_was_emitted():
    # T2 — the fallback that Defect A removed: a prose intake output must NOT leave the
    # judge blind. Last N=6, newest last (read_thread is already chronological).
    judge = StubJudge({"decision": False, "rationale": "x"})

    evaluate_guard(
        LLM_GUARD, ctx={}, run={},
        step_output="Thank you for providing all the details, Alice.",
        thread=_rows(10), judge=judge,
    )

    turns = judge.calls[0]["recent_turns"]
    assert judge.calls[0]["understanding"] == {}
    assert len(turns) == 6
    assert turns[0]["text"] == "turn 4"   # newest-last window over rows 4..9
    assert turns[-1]["text"] == "turn 9"


def test_recent_turns_are_normalized_to_speaker_role_text():
    # T3 — a compact, prompt-ready shape; `displayName` → `authorId` → "member".
    judge = StubJudge({"decision": False, "rationale": "x"})
    thread = [
        {"text": "hi", "role": "user", "displayName": "Alice", "authorId": "u1"},
        {"text": "hello", "role": "assistant", "displayName": None, "authorId": "a1"},
        {"text": "yo", "role": None, "displayName": None, "authorId": None},
    ]

    evaluate_guard(
        LLM_GUARD, ctx={}, run={}, step_output="prose", thread=thread, judge=judge,
    )

    assert judge.calls[0]["recent_turns"] == [
        {"speaker": "Alice", "role": "user", "text": "hi"},
        {"speaker": "a1", "role": "assistant", "text": "hello"},
        {"speaker": "member", "role": "user", "text": "yo"},
    ]


def test_each_recent_turn_is_truncated_so_one_turn_cannot_dominate_the_prompt():
    # T4 — rule 6 / the DS ~1500-token judge cap: a single long turn must not crowd out
    # the rest of the window.
    judge = StubJudge({"decision": False, "rationale": "x"})

    evaluate_guard(
        LLM_GUARD, ctx={}, run={}, step_output="prose",
        thread=_rows(1, text="x" * 5000), judge=judge,
    )

    assert len(judge.calls[0]["recent_turns"][0]["text"]) == TURN_TEXT_MAX


@pytest.mark.parametrize(
    "thread", [None, [], [{}], "garbage", [None, 42], [{"text": ""}]]
)
def test_a_missing_or_malformed_thread_degrades_to_the_understanding_only_path(thread):
    # T5 — tolerance by design: the offline stub path and non-agent steps carry no thread
    # (`StepResult.thread == []`). The guard must degrade, never crash a drive.
    judge = StubJudge({"decision": True, "rationale": "all fields present"})

    verdict = evaluate_guard(
        LLM_GUARD, ctx={}, run={}, step_output="prose", thread=thread, judge=judge,
    )

    assert judge.calls[0]["recent_turns"] == []
    assert verdict.decision is True


def test_an_emitted_but_empty_understanding_still_falls_back_to_the_turns():
    # T6 — truthiness, not presence: a node that emits `{"understanding": {}}` knows
    # nothing, so the turns are still the only evidence available.
    judge = StubJudge({"decision": False, "rationale": "x"})

    evaluate_guard(
        LLM_GUARD, ctx={}, run={}, step_output='{"understanding":{}}',
        thread=_rows(3), judge=judge,
    )

    assert judge.calls[0]["understanding"] == {}
    assert len(judge.calls[0]["recent_turns"]) == 3


def test_an_empty_guard_never_calls_the_judge_even_with_a_live_thread():
    # T6b — the unconditional (D5) path short-circuits before any evidence is gathered.
    verdict = evaluate_guard(
        "", ctx={}, run={}, step_output="prose", thread=_rows(6), judge=_boom_judge
    )
    assert verdict.decision is True


# ── R-1: `_NEGATION_CUES` must not fire on an AFFIRMATIVE rationale ──────────
#
# The cue check is a backstop against an *internally inconsistent* verdict (decision:true
# + a rationale asserting a deficiency) — it is not a prose grep. Once the judge is fed a
# real transcript it writes wordier rationales, and a naive substring match turns a correct
# advance into a forced suspend — a failure INDISTINGUISHABLE from Defect A from the
# outside ("the guard never fires"). These pin the contract in both directions.

ADVANCING_RATIONALES = [
    # a negated cue AFFIRMS the advance — "no more info", "nothing is unclear"
    "The user provided the service, version and symptom; no more info is needed.",
    "Everything is known; nothing is unclear.",
    "The request is clear and no relevant details are missing.",
    "The user isn't missing anything; the request is fully specified.",
]

SUSPENDING_RATIONALES = [
    "Not enough information yet.",            # the real contradiction case
    "The user still needs to provide the version.",
    "The request is unclear.",
    "More info is needed about the deployment.",
    "The root cause is still missing.",
    "There is insufficient detail to research this.",
]


@pytest.mark.parametrize("rationale", ADVANCING_RATIONALES)
def test_an_affirmative_rationale_does_not_block_a_true_decision(rationale):
    verdict = evaluate_guard(
        LLM_GUARD, ctx={}, run={}, step_output="prose", thread=None,
        judge=StubJudge({"decision": True, "rationale": rationale}),
    )
    assert verdict.decision is True, f"false suspend on an advancing rationale: {rationale!r}"
    assert verdict.rationale == rationale


@pytest.mark.parametrize("rationale", SUSPENDING_RATIONALES)
def test_a_deficiency_asserting_rationale_still_contradicts_a_true_decision(rationale):
    verdict = evaluate_guard(
        LLM_GUARD, ctx={}, run={}, step_output="prose", thread=None,
        judge=StubJudge({"decision": True, "rationale": rationale}),
    )
    assert verdict.decision is False, f"missed contradiction: {rationale!r}"


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
