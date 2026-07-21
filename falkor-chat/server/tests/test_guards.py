"""Unit tests for fuzzy (LLM-judged) + deterministic transition-guard evaluation.

`guards.evaluate_guard` dispatches on the serialized guard `kind`:
  * `""` (empty) → unconditional (fires whenever reached);
  * `{"kind":"llm", ...}` → the Q1 extract-then-judge path via the **injected** judge,
    with the ambiguity policy (malformed / contradictory judge output → bias-to-suspend)
    — K-022 U7;
  * `{"kind":"cmp"|"all"|"any"|"not", ...}` → the deterministic structured comparator
    (K-024 U1): whitelisted ops, two whitelisted path roots, structural caps, and the
    publish-time `validate_cmp` twin. Total on data (missing → False), loud on structure;
  * any other kind (e.g. `expr`) → a pure `NotImplementedError` seam (M7).

The judge is an injected callable so the whole module is stub-testable offline — no LLM.
"""

from __future__ import annotations

import json

import pytest

from falkorchat.guards import (
    MAX_GUARD_DEPTH,
    MAX_GUARD_NODES,
    MAX_GUARD_WIDTH,
    TURN_TEXT_MAX,
    GuardVerdict,
    WorkflowConfigError,
    evaluate_guard,
    render_label,
    validate_cmp,
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


# ── the deterministic `cmp` comparator (K-024 U1, plan §3.2) ─────────────────
#
# A `cmp` guard is already-parsed JSON evaluated by whitelisted Python callables — no
# parser, no `eval`, no dependency (D-A option A1). The judge must never be consulted.


def _cmp(guard, *, ctx=None, step_output=""):
    """Evaluate a `cmp`-family guard offline, asserting the judge is never called."""
    return evaluate_guard(
        json.dumps(guard) if not isinstance(guard, str) else guard,
        ctx=ctx or {}, run={}, step_output=step_output,
        thread=None, judge=_boom_judge,
    )


def test_cmp_eq_fires_on_a_matching_ctx_path():
    guard = {"kind": "cmp", "path": "ctx.decision", "op": "eq", "value": "approve"}
    assert _cmp(guard, ctx={"decision": "approve"}).decision is True


def test_cmp_eq_does_not_fire_on_a_different_value():
    guard = {"kind": "cmp", "path": "ctx.decision", "op": "eq", "value": "approve"}
    assert _cmp(guard, ctx={"decision": "reject"}).decision is False


def test_cmp_traverses_a_dotted_path_into_a_nested_dict():
    # §4.2 transition #2 reads `ctx.request.role`. Traversal is dict-key lookup only.
    guard = {"kind": "cmp", "path": "ctx.request.role", "op": "eq", "value": "exec"}
    assert _cmp(guard, ctx={"request": {"role": "exec"}}).decision is True


@pytest.mark.parametrize(
    "value, expected", [(0, True), ("", True), (False, True), (None, True)]
)
def test_exists_is_true_for_a_present_but_falsy_value(value, expected):
    # §7 case 4 — `exists` asks "is the key there?", `truthy` asks "is it set?".
    # Conflating them would make `{"provisioned": false}` ("not yet") advance the run.
    guard = {"kind": "cmp", "path": "ctx.provisioned", "op": "exists"}
    assert _cmp(guard, ctx={"provisioned": value}).decision is expected


@pytest.mark.parametrize(
    "value, expected", [(True, True), (1, True), ("yes", True),
                        (0, False), ("", False), (False, False), (None, False)]
)
def test_truthy_follows_python_truthiness(value, expected):
    guard = {"kind": "cmp", "path": "ctx.provisioned", "op": "truthy"}
    assert _cmp(guard, ctx={"provisioned": value}).decision is expected


def test_in_tests_membership_of_the_path_value_in_a_list_literal():
    # §4.2 transition #2: `ctx.request.role` ∈ ["contractor","exec"].
    guard = {"kind": "cmp", "path": "ctx.request.role", "op": "in",
             "value": ["contractor", "exec"]}
    assert _cmp(guard, ctx={"request": {"role": "contractor"}}).decision is True
    assert _cmp(guard, ctx={"request": {"role": "engineer"}}).decision is False


def test_in_against_a_non_list_literal_is_false_not_a_raise():
    guard = {"kind": "cmp", "path": "ctx.role", "op": "in", "value": 7}
    assert _cmp(guard, ctx={"role": "exec"}).decision is False


def test_contains_works_on_a_list_and_on_a_string():
    on_list = {"kind": "cmp", "path": "ctx.tags", "op": "contains", "value": "urgent"}
    assert _cmp(on_list, ctx={"tags": ["urgent", "vpn"]}).decision is True
    assert _cmp(on_list, ctx={"tags": ["vpn"]}).decision is False

    on_str = {"kind": "cmp", "path": "ctx.note", "op": "contains", "value": "vpn"}
    assert _cmp(on_str, ctx={"note": "needs vpn access"}).decision is True
    assert _cmp(on_str, ctx={"note": "needs laptop"}).decision is False


def test_contains_on_a_non_container_is_false_not_a_raise():
    guard = {"kind": "cmp", "path": "ctx.tags", "op": "contains", "value": "urgent"}
    assert _cmp(guard, ctx={"tags": 7}).decision is False


def test_ne_fires_only_when_a_present_value_differs():
    guard = {"kind": "cmp", "path": "ctx.decision", "op": "ne", "value": "approve"}
    assert _cmp(guard, ctx={"decision": "reject"}).decision is True
    assert _cmp(guard, ctx={"decision": "approve"}).decision is False


@pytest.mark.parametrize("op, left, right, expected", [
    ("lt", 1, 2, True), ("lt", 2, 1, False), ("le", 2, 2, True), ("le", 3, 2, False),
    ("gt", 3, 2, True), ("gt", 2, 3, False), ("ge", 2, 2, True), ("ge", 1, 2, False),
    ("lt", "abc", "abd", True), ("gt", "b", "a", True), ("le", "a", "a", True),
])
def test_ordering_comparisons_on_ints_and_strings(op, left, right, expected):
    guard = {"kind": "cmp", "path": "ctx.n", "op": op, "value": right}
    assert _cmp(guard, ctx={"n": left}).decision is expected


@pytest.mark.parametrize("left, right", [("abc", 3), (3, "abc"), (None, 1), ([1], 1)])
def test_mismatched_types_on_an_ordering_op_are_false_not_a_type_error(left, right):
    # §7 case 8 — a TypeError here would escape into the drive and fail the run; a guard
    # that cannot decide must decline to fire, never crash.
    guard = {"kind": "cmp", "path": "ctx.n", "op": "lt", "value": right}
    assert _cmp(guard, ctx={"n": left}).decision is False


# ── totality: a missing path never fires (bias to not-fire) ──────────────────


@pytest.mark.parametrize("op, value", [
    ("eq", "approve"), ("ne", "approve"), ("lt", 3), ("le", 3), ("gt", 3), ("ge", 3),
    ("in", ["a"]), ("contains", "a"), ("truthy", None), ("exists", None),
])
def test_a_missing_path_is_false_for_every_op(op, value):
    # §7 case 3 — including `exists` and `ne`. A value that is not there can fire nothing:
    # the safe direction is to park (mirrors `_coerce_verdict`'s bias-to-suspend), because
    # a parked run is unblockable and a wrongly-advanced one is not.
    guard = {"kind": "cmp", "path": "ctx.nope", "op": op, "value": value}
    assert _cmp(guard, ctx={"decision": "approve"}).decision is False


def test_a_path_through_a_non_dict_is_missing_not_a_raise():
    guard = {"kind": "cmp", "path": "ctx.request.role", "op": "exists"}
    assert _cmp(guard, ctx={"request": "a bare string"}).decision is False


@pytest.mark.parametrize("path", ["foo.bar", "ctx", "", None, 7, "output.a.b"])
def test_an_unwhitelisted_or_unusable_path_resolves_to_missing(path):
    # §7 case 10 — exactly two roots are whitelisted (`ctx.` and `output.`, plus bare
    # `output`). Anything else — including a bare `ctx` and a non-string — is *missing*,
    # never an attribute lookup and never a raise.
    guard = {"kind": "cmp", "path": path, "op": "exists"}
    assert _cmp(guard, ctx={"decision": "approve"}).decision is False


def test_output_root_reads_the_current_steps_json_output():
    guard = {"kind": "cmp", "path": "output.awaiting.kind", "op": "eq", "value": "human"}
    out = json.dumps({"awaiting": {"kind": "human", "prompt": "Approve?"}})
    assert _cmp(guard, step_output=out).decision is True


def test_bare_output_reads_the_raw_step_output_string():
    guard = {"kind": "cmp", "path": "output", "op": "contains", "value": "approved"}
    assert _cmp(guard, step_output="the request was approved").decision is True
    assert _cmp(guard, step_output="the request was denied").decision is False


def test_a_dotted_output_path_against_non_json_output_is_missing():
    guard = {"kind": "cmp", "path": "output.awaiting", "op": "exists"}
    assert _cmp(guard, step_output="plain prose").decision is False


# ── combinators: all / any / not ─────────────────────────────────────────────

_T = {"op": "eq", "path": "ctx.decision", "value": "approve"}   # true  in CTX below
_F = {"op": "eq", "path": "ctx.decision", "value": "reject"}    # false in CTX below
CTX = {"decision": "approve"}


@pytest.mark.parametrize("of, expected", [
    ([_T, _T], True), ([_T, _F], False), ([_F, _F], False),
])
def test_all_is_conjunction(of, expected):
    assert _cmp({"kind": "all", "of": of}, ctx=CTX).decision is expected


@pytest.mark.parametrize("of, expected", [
    ([_T, _T], True), ([_T, _F], True), ([_F, _F], False),
])
def test_any_is_disjunction(of, expected):
    assert _cmp({"kind": "any", "of": of}, ctx=CTX).decision is expected


def test_not_negates_its_single_child():
    assert _cmp({"kind": "not", "of": [_F]}, ctx=CTX).decision is True
    assert _cmp({"kind": "not", "of": [_T]}, ctx=CTX).decision is False


def test_combinators_nest():
    spec = {"kind": "all", "of": [_T, {"kind": "any", "of": [_F, {"kind": "not", "of": [_F]}]}]}
    assert _cmp(spec, ctx=CTX).decision is True


def test_empty_of_follows_the_identity_of_the_operator():
    # §7 case 11 — defined, not accidental: `all` over nothing is vacuously true,
    # `any` over nothing has no witness and is false.
    assert _cmp({"kind": "all", "of": []}, ctx=CTX).decision is True
    assert _cmp({"kind": "any", "of": []}, ctx=CTX).decision is False


@pytest.mark.parametrize("of", [[], [_T, _F], [_T, _T, _F]])
def test_not_with_an_arity_other_than_one_is_a_named_config_error(of):
    with pytest.raises(WorkflowConfigError) as exc:
        _cmp({"kind": "not", "of": of}, ctx=CTX)
    assert "not" in str(exc.value).lower()


@pytest.mark.parametrize("kind", ["all", "any", "not"])
def test_a_combinator_without_a_list_of_is_a_named_config_error(kind):
    with pytest.raises(WorkflowConfigError):
        _cmp({"kind": kind, "of": "nope"}, ctx=CTX)


def test_ne_on_a_missing_path_and_not_eq_on_a_missing_path_disagree():
    # §7 case 3b / plan m-10 — **De Morgan does not hold here, deliberately.**
    # `ne` is a comparison against a value that is not there → False (bias to not-fire);
    # `not` negates a *verdict*, and the inner `eq` was False → True. Asserted side by
    # side so the asymmetry is pinned by the suite rather than discovered in production.
    ne_missing = {"kind": "cmp", "path": "ctx.nope", "op": "ne", "value": "approve"}
    not_eq_missing = {"kind": "not",
                      "of": [{"op": "eq", "path": "ctx.nope", "value": "approve"}]}

    assert _cmp(ne_missing, ctx=CTX).decision is False
    assert _cmp(not_eq_missing, ctx=CTX).decision is True


# ── loud failures: unknown op + the structural DoS caps ──────────────────────


def _nest(depth):
    """A `not`-chain of the given nesting depth around one leaf."""
    node = dict(_T)
    for _ in range(depth - 1):
        node = {"kind": "not", "of": [node]}
    return node


def test_an_unknown_op_is_a_named_config_error_not_a_silent_false():
    # §7 case 13 — a typo'd op must be loud. Silently returning False would park a live
    # run forever with nothing naming the cause.
    with pytest.raises(WorkflowConfigError) as exc:
        _cmp({"kind": "cmp", "path": "ctx.decision", "op": "equals", "value": "a"})
    assert "equals" in str(exc.value)


def test_an_unknown_op_nested_inside_a_combinator_is_also_loud():
    with pytest.raises(WorkflowConfigError):
        _cmp({"kind": "all", "of": [_T, {"op": "equals", "path": "ctx.d", "value": "a"}]})


def test_a_guard_at_the_depth_cap_still_evaluates():
    assert _cmp(_nest(MAX_GUARD_DEPTH), ctx=CTX).decision is not None


def test_exceeding_the_depth_cap_is_a_named_config_error():
    with pytest.raises(WorkflowConfigError) as exc:
        _cmp(_nest(MAX_GUARD_DEPTH + 1), ctx=CTX)
    assert "depth" in str(exc.value).lower()


def test_exceeding_the_node_cap_is_a_named_config_error():
    wide = {"kind": "all", "of": [{"kind": "any", "of": [dict(_T)] * MAX_GUARD_WIDTH}]
                                 * MAX_GUARD_WIDTH}
    with pytest.raises(WorkflowConfigError) as exc:
        _cmp(wide, ctx=CTX)
    assert "node" in str(exc.value).lower()


def test_exceeding_the_width_cap_is_a_named_config_error():
    with pytest.raises(WorkflowConfigError) as exc:
        _cmp({"kind": "any", "of": [dict(_T)] * (MAX_GUARD_WIDTH + 1)}, ctx=CTX)
    assert "width" in str(exc.value).lower()


# ── validate_cmp — the publish-time structural validator (M-4) ───────────────
#
# One implementation of the structural rules, two call sites: `evaluate_guard` (drive)
# and `services._validate_def_spec` (publish, U2). A typo'd op must be an authoring
# error caught at seed time, not a dead run discovered when a manager clicks approve.
# `validate_cmp` takes an ALREADY-PARSED dict — normalizing a string-shaped guard is the
# call site's job.

# Every guard shape the `access-request@v1` proof def uses (plan §4.2, transitions #1–#6).
ACCESS_REQUEST_GUARDS = [
    {"kind": "cmp", "path": "ctx.request.role", "op": "exists"},
    {"kind": "cmp", "path": "ctx.request.role", "op": "in",
     "value": ["contractor", "exec"]},
    {"kind": "cmp", "path": "ctx.decision", "op": "eq", "value": "approve"},
    {"kind": "cmp", "path": "ctx.decision", "op": "eq", "value": "reject"},
    {"kind": "cmp", "path": "ctx.provisioned", "op": "truthy"},
]


@pytest.mark.parametrize("guard", ACCESS_REQUEST_GUARDS)
def test_validate_cmp_accepts_every_guard_shape_the_proof_def_uses(guard):
    assert validate_cmp(guard) is None


def test_validate_cmp_accepts_the_combinators_and_a_kindless_child():
    validate_cmp({"kind": "all", "of": [
        {"op": "eq", "path": "ctx.a", "value": 1},
        {"kind": "not", "of": [{"op": "truthy", "path": "output.flag"}]},
        {"kind": "any", "of": []},
    ]})


@pytest.mark.parametrize("bad", [
    {"kind": "cmp", "path": "ctx.a", "op": "equals", "value": 1},        # unknown op
    {"kind": "cmp", "path": "ctx.a"},                                    # no op
    {"kind": "cmp", "path": "state.a", "op": "eq", "value": 1},          # bad root
    {"kind": "cmp", "path": "ctx", "op": "exists"},                      # bare ctx root
    {"kind": "cmp", "path": "ctx..a", "op": "exists"},                   # empty segment
    {"kind": "cmp", "op": "eq", "value": 1},                             # no path
    {"kind": "cmp", "path": 7, "op": "exists"},                          # non-string path
    {"kind": "cmp", "path": "ctx.a", "op": "eq"},                        # value op, no value
    {"kind": "not", "of": []},                                           # arity 0
    {"kind": "not", "of": [{"op": "truthy", "path": "ctx.a"},
                           {"op": "truthy", "path": "ctx.b"}]},          # arity 2
    {"kind": "all", "of": {"op": "truthy", "path": "ctx.a"}},            # `of` not a list
    {"kind": "llm", "text": "is it enough?"},                            # not cmp family
    "not even an object",
])
def test_validate_cmp_rejects_a_structurally_unsound_guard(bad):
    with pytest.raises(WorkflowConfigError):
        validate_cmp(bad)


@pytest.mark.parametrize("kind", ["expr", "llm", "sql"])
def test_validate_cmp_rejects_a_foreign_kind_even_when_it_carries_valid_cmp_fields(kind):
    # Without an explicit family check, `{"kind":"expr", …cmp fields…}` would validate as
    # a cmp leaf — quietly admitting the very kind that must stay a raising seam.
    with pytest.raises(WorkflowConfigError, match="cmp family"):
        validate_cmp({"kind": kind, "path": "ctx.a", "op": "eq", "value": 1})


def test_validate_cmp_enforces_the_same_caps_as_evaluation():
    with pytest.raises(WorkflowConfigError, match="depth"):
        validate_cmp(_nest(MAX_GUARD_DEPTH + 1))
    with pytest.raises(WorkflowConfigError, match="width"):
        validate_cmp({"kind": "any", "of": [dict(_T)] * (MAX_GUARD_WIDTH + 1)})
    with pytest.raises(WorkflowConfigError, match="node"):
        validate_cmp({"kind": "all",
                      "of": [{"kind": "any", "of": [dict(_T)] * MAX_GUARD_WIDTH}]
                            * MAX_GUARD_WIDTH})
    assert MAX_GUARD_NODES >= MAX_GUARD_WIDTH   # the caps must not contradict each other


def test_an_undeclared_value_is_rejected_at_publish_but_total_at_drive_time():
    # The one place validation is deliberately STRICTER than evaluation: an unwhitelisted
    # path root is an authoring defect worth failing the publish, yet at drive time it is
    # only "a value that is not there" → False. Pinning both halves keeps the asymmetry
    # a decision rather than a discrepancy.
    guard = {"kind": "cmp", "path": "state.decision", "op": "exists"}
    with pytest.raises(WorkflowConfigError):
        validate_cmp(guard)
    assert _cmp(guard, ctx={"decision": "approve"}).decision is False


# ── render_label + rationale — the trace inputs (M-6, FR-4) ──────────────────


def test_render_label_describes_a_cmp_guard_without_a_text_field():
    # §7 case 18 — a `cmp` guard has no `text`, and `_trace_step` formats `f"{text} -> …"`,
    # so an empty label would emit a trace line starting with a bare " -> ".
    label = render_label({"kind": "cmp", "path": "ctx.decision", "op": "eq",
                          "value": "approve"})
    assert label.strip()
    assert "ctx.decision" in label and "eq" in label and "approve" in label


def test_render_label_describes_a_combinator_including_its_children():
    label = render_label({"kind": "any", "of": [
        {"op": "truthy", "path": "ctx.provisioned"},
        {"op": "exists", "path": "ctx.request.role"},
    ]})
    assert label.startswith("any(")
    assert "ctx.provisioned" in label and "ctx.request.role" in label


@pytest.mark.parametrize("spec", [None, "prose", 7, {"kind": "all", "of": "nope"}, {}])
def test_render_label_never_raises_on_a_malformed_guard(spec):
    # A renderer must not be able to break a trace; raising loudly is `validate_cmp`'s job.
    assert isinstance(render_label(spec), str)


def test_a_cmp_verdict_carries_a_readable_rationale_in_both_directions():
    # §7 case 16 — U2 wires this into the debug trace; it must be populated and readable.
    guard = {"kind": "cmp", "path": "ctx.decision", "op": "eq", "value": "approve"}

    fired = _cmp(guard, ctx={"decision": "approve"})
    missed = _cmp(guard, ctx={"decision": "reject"})

    assert "ctx.decision" in fired.rationale and "true" in fired.rationale
    assert "ctx.decision" in missed.rationale and "false" in missed.rationale


# ── expr / unknown kind — pure NotImplementedError seam (M7) ─────────────────

def test_expr_kind_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        evaluate_guard(
            '{"kind":"expr","expr":"x > 1"}',
            ctx={}, run={}, step_output="", thread=None, judge=_boom_judge,
        )


def test_a_kindless_guard_object_is_not_silently_treated_as_a_cmp_leaf():
    # Kind inference exists for nested `of` children only. At the TOP level a guard with
    # no `kind` (the long-standing `{"expr":"x>0"}` publish fixture) must stay on the M7
    # seam — inferring `cmp` there would swallow it and change publish behaviour.
    with pytest.raises(NotImplementedError):
        evaluate_guard(
            '{"expr":"x>0"}', ctx={}, run={}, step_output="",
            thread=None, judge=_boom_judge,
        )


def test_cmp_is_named_cmp_so_the_expr_seam_stays_shut():
    # DESIGN §13 resolved "no expression library is built". `cmp` is deliberately NOT
    # `expr`, and this pins that the door stays visibly closed.
    from falkorchat.guards import CMP_KINDS
    assert "expr" not in CMP_KINDS


def test_unknown_non_empty_guard_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        evaluate_guard(
            "just some prose", ctx={}, run={}, step_output="",
            thread=None, judge=_boom_judge,
        )
