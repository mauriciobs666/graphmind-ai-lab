"""`modelbench.scoring.toolcalls` — S5 spec §5 Steps 3-5: the scorer's per-turn pure functions,
`clean_through_turn`/`hazard_points` (`-ml` §4.3 rule 5's three consumers), the determinism probe's
`outcome_vectors_differ` (plan item 12b), and `score_conversations` itself (Step 5's assembly,
§2.3/§2.6). Offline throughout: every fixture is a hand-built `TurnTrace`/`ConversationTrace`
(Steps 3-4) or `Conversation`/`Turn` script (Step 5) — no on-disk pack tree; Step 5's tests use a
small duck-typed `_FakePack` stand-in, matching `retrieval.py`'s/`test_runner.py`'s own offline-stub
convention.

Docstrings on the tests below cite the exact plan/`-ml` item they pin, per
`model-bench/AGENTS.md`'s guard-reach convention.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from modelbench.convo import Conversation, ConversationTrace, Turn, TurnTrace
from modelbench.scoring import toolcalls
from modelbench.tooling import DispatchRecord

# --------------------------------------------------------------------------------------------
# Shared fixtures — hand-built, offline (mirrors tests/test_convo.py's own helpers)
# --------------------------------------------------------------------------------------------


def make_turn(
    *,
    disposition: str = "replied",
    dispatches: tuple[DispatchRecord, ...] = (),
    iterations: int = 1,
    final_reply: str | None = "ok",
) -> TurnTrace:
    return TurnTrace(
        messagesSent=(),
        chatResults=(),
        dispatches=dispatches,
        envState={},
        iterations=iterations,
        turnDisposition=disposition,  # type: ignore[arg-type]
        finalReplyText=final_reply,
        wallClockMs=1.0,
    )


def make_dispatch(
    name: str, arguments: Mapping[str, Any], return_value: Any = None
) -> DispatchRecord:
    return DispatchRecord(
        name=name,
        rawArguments=arguments,
        parsedArguments=arguments,
        returnValue=return_value if return_value is not None else {"ok": True},
        timestamp="2026-09-13T00:00:00Z",
    )


def make_trace(
    script_id: str, turns: tuple[TurnTrace, ...], *, shape: str = "A", replicate: int = 1
) -> ConversationTrace:
    return ConversationTrace(scriptId=script_id, shape=shape, replicate=replicate, turns=turns)


_SCHEMAS_PATH = (
    Path(__file__).resolve().parent.parent
    / "packs"
    / "tool-caller-shop-assistant"
    / "tools"
    / "schemas.json"
)


def real_schemas() -> list[dict[str, Any]]:
    return json.loads(_SCHEMAS_PATH.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------------------------
# Step 5 — `score_conversations` fixtures: `Conversation`/`Turn` scripts + a duck-typed fake pack
# --------------------------------------------------------------------------------------------


def _restraint_turn(seq: int) -> Turn:
    """A turn with `R(t) = ∅` — `toolRequired` absent entirely, matching a script that never
    mentions it (§2.6's `_required_tool_names` reads `expect.get("toolRequired")`, `None`-safe)."""
    return Turn(seq=seq, user=f"turn {seq}", expect={"toolRequired": False})


def _required_turn(seq: int, tool: str = "lookup_product_fact") -> Turn:
    return Turn(seq=seq, user=f"turn {seq}", expect={"toolRequired": True, "tool": tool})


def make_script(
    script_id: str, n_turns: int, *, shape: str = "A", replicate: int = 1
) -> Conversation:
    """A script whose every turn is a restraint turn (`toolRequired: False`) — the simplest
    `Turn.expect` shape whose own cleanliness is exactly `restraint(dispatched_count)`, so a
    fixture built from `make_turn()` (no dispatches) alone is clean by construction and Step 5's
    tests can isolate the censoring/hazard/funnel mechanics from Steps 3-4's own FR-8 correctness
    (already tested there)."""
    return Conversation(
        scriptId=script_id, shape=shape, replicate=replicate,
        turns=tuple(_restraint_turn(i) for i in range(n_turns)),
    )


def _clean_nine_turn_script(script_id: str) -> Conversation:
    return make_script(script_id, 9)


def _clean_scored(script_id: str) -> tuple[Conversation, ConversationTrace, tuple[Any, ...]]:
    """A `scored` triple for a fully clean, fully driven 9-turn restraint script — the shared
    "nothing interesting happens" filler conversation Step 5's multi-conversation fixtures need
    beside the one conversation under test."""
    trace = make_trace(script_id, tuple(make_turn() for _ in range(9)))
    return (_clean_nine_turn_script(script_id), trace, ())


class _FakePack:
    """A minimal duck-typed stand-in for `packs.Pack` (mirrors `tests/test_runner.py`'s own
    `FakePack`) — `score_conversations` reads `pack.manifest` and `pack.prompt_config().toolSchemas`
    structurally, never `isinstance`-checked."""

    def __init__(
        self,
        *,
        h: int = 4,
        determinism_probe_scripts: tuple[str, ...] = (),
        tool_schemas: tuple[Mapping[str, Any], ...] = (),
    ) -> None:
        self.manifest = {
            "metrics": {"cleanThroughTurnH": {"H": h}},
            "sampling": {"determinismProbeScripts": list(determinism_probe_scripts)},
        }
        self._tool_schemas = tool_schemas

    def prompt_config(self) -> Any:
        from types import SimpleNamespace

        return SimpleNamespace(toolSchemas=self._tool_schemas)


def make_pack(
    *,
    h: int = 4,
    determinism_probe_scripts: tuple[str, ...] = (),
    tool_schemas: tuple[Mapping[str, Any], ...] = (),
) -> _FakePack:
    return _FakePack(
        h=h, determinism_probe_scripts=determinism_probe_scripts, tool_schemas=tool_schemas
    )


# ==================================================================================================
# Step 3 — per-turn pure functions
# ==================================================================================================

# --- turn_disposition_scores (`-ml` §4.3 rule 4) -------------------------------------------------


@pytest.mark.parametrize(
    ("disposition", "expected"),
    [
        ("replied", "scored"),
        ("cap-hit", "scored"),
        ("timed-out", "scored"),
        ("no-response", "unrunnable"),
        ("server-rejected", "unrunnable"),
    ],
)
def test_turn_disposition_scores_matches_rule_4s_table(disposition, expected) -> None:
    """`-ml` §4.3 rule 4: the five-row mechanism table, each row asserted."""
    assert toolcalls.turn_disposition_scores(disposition) == expected


# --- (a)+(b) emission_form (`-ml` §4.2(a)+(b)) ----------------------------------------------------


def test_emission_form_native_when_a_call_was_dispatched() -> None:
    """`-ml` §4.2(a)+(b): `|E(t)| >= 1` is `native`, regardless of `prose_detected`."""
    assert (
        toolcalls.emission_form(required=True, dispatched_count=1, prose_detected=True)
        == "native"
    )
    assert (
        toolcalls.emission_form(required=True, dispatched_count=2, prose_detected=False)
        == "native"
    )


def test_emission_form_prose_pseudo_call_when_nothing_dispatched_but_prose_detected() -> None:
    """`-ml` §4.2(a)+(b): `|E(t)| = 0` and `P(t)` fired."""
    assert (
        toolcalls.emission_form(required=True, dispatched_count=0, prose_detected=True)
        == "prose_pseudo_call"
    )


def test_emission_form_no_attempt_when_nothing_dispatched_and_no_prose() -> None:
    """`-ml` §4.2(a)+(b): `|E(t)| = 0` and not `P(t)` — the count that DOES NOT move relative to
    the prior two cases (same `dispatched_count=0`, differing only in `prose_detected`)."""
    assert (
        toolcalls.emission_form(required=True, dispatched_count=0, prose_detected=False)
        == "no_attempt"
    )


def test_emission_form_refuses_a_restraint_turn() -> None:
    """`required=False` is a caller bug (restraint's denominator, not (a)+(b)'s) — refused."""
    with pytest.raises(ValueError, match="restraint"):
        toolcalls.emission_form(required=False, dispatched_count=0, prose_detected=False)


# --- the prose-pseudo-call detector and its calibration (§2.7) ------------------------------------


def test_detect_prose_pseudo_call_matches_a_function_call_shaped_reply() -> None:
    assert toolcalls.detect_prose_pseudo_call('I will call add_to_cart("Pad", 1) now.') is True


def test_detect_prose_pseudo_call_does_not_match_ordinary_prose() -> None:
    """The count that does NOT move: a reply with no call-shaped text at all."""
    assert toolcalls.detect_prose_pseudo_call("The Pad costs $24.99.") is False


def test_detect_prose_pseudo_call_matches_first_person_calling_announcement() -> None:
    assert toolcalls.detect_prose_pseudo_call("I'm calling the cart tool for you.") is True


def test_detect_prose_pseudo_call_none_and_empty_never_match() -> None:
    assert toolcalls.detect_prose_pseudo_call(None) is False
    assert toolcalls.detect_prose_pseudo_call("") is False


def test_prose_detector_precision_recall_none_when_no_corpus() -> None:
    """§2.7: `None` when no labelled corpus is available at all."""
    assert toolcalls.prose_detector_precision_recall([]) is None


def test_prose_detector_precision_recall_perfect_detector() -> None:
    labelled = [
        ('add_to_cart("Pad", 1)', True),
        ("The Pad costs $24.99.", False),
    ]
    assert toolcalls.prose_detector_precision_recall(labelled) == (1.0, 1.0)


def test_prose_detector_precision_recall_a_false_positive_lowers_precision_not_recall() -> None:
    labelled = [
        ('add_to_cart("Pad", 1)', True),  # true positive
        ("I'm calling this a great deal.", False),  # false positive (matches "I'm calling")
    ]
    precision, recall = toolcalls.prose_detector_precision_recall(labelled)
    assert recall == 1.0
    assert precision == pytest.approx(0.5)


def test_prose_detector_precision_recall_zero_denominators_are_zero_not_a_crash() -> None:
    """No positive label and no positive prediction: both figures are `0.0`."""
    labelled = [("The Pad costs $24.99.", False)]
    assert toolcalls.prose_detector_precision_recall(labelled) == (0.0, 0.0)


# --- (c) right_tool_chosen (`-ml` §4.2(c)) --------------------------------------------------------


def test_right_tool_chosen_true_when_every_required_name_was_dispatched() -> None:
    assert toolcalls.right_tool_chosen({"lookup_product_fact"}, {"lookup_product_fact"}) is True


def test_right_tool_chosen_true_with_an_extra_call_alongside_the_required_one() -> None:
    """Extra calls are (e)'s business, never (c)'s — coverage alone is checked."""
    assert (
        toolcalls.right_tool_chosen({"lookup_product_fact"}, {"lookup_product_fact", "view_cart"})
        is True
    )


def test_right_tool_chosen_false_on_the_wrong_tool() -> None:
    """The count that moves: the required tool was never among the dispatched names."""
    assert toolcalls.right_tool_chosen({"lookup_product_fact"}, {"filter_products"}) is False


def test_right_tool_chosen_refuses_an_empty_required_or_dispatched_set() -> None:
    with pytest.raises(ValueError):
        toolcalls.right_tool_chosen(set(), {"lookup_product_fact"})
    with pytest.raises(ValueError):
        toolcalls.right_tool_chosen({"lookup_product_fact"}, set())


# --- (d) argument_correctness (`-ml` §4.2(d)) -----------------------------------------------------


def test_argument_correctness_all_correct() -> None:
    result = toolcalls.argument_correctness({"name": "Pad"}, {"name": "Pad"})
    assert result.allCorrect is True
    assert result.omittedRequired == ()
    assert result.wrongValue == ()
    assert result.boundaryUnit == ()


def test_argument_correctness_no_required_arguments_is_correct_by_construction() -> None:
    """A tool with no required arguments (e.g. `view_cart`) — `allCorrect=True`, nothing to
    check."""
    result = toolcalls.argument_correctness({}, {})
    assert result.allCorrect is True


def test_argument_correctness_omitted_required_argument() -> None:
    """The count that moves relative to 'all correct': the argument is simply absent."""
    result = toolcalls.argument_correctness({"name": "Pad"}, {})
    assert result.allCorrect is False
    assert result.omittedRequired == ("name",)
    assert result.wrongValue == ()
    assert result.boundaryUnit == ()


def test_argument_correctness_wrong_value_non_boundary() -> None:
    result = toolcalls.argument_correctness({"category": "Electronics"}, {"category": "Home"})
    assert result.allCorrect is False
    assert result.wrongValue == ("category",)
    assert result.boundaryUnit == ()  # a plain miss, not a declared boundary confusion


def test_scalar_equal_never_coerces_a_number_and_a_string_even_when_str_equal() -> None:
    """`-ml` §4.2(d): "never coerce across types" — a number and the string of that same number
    must NOT compare equal, however identical their `str()` forms are. One numeric/one non-numeric
    is refused regardless of which argument holds which type."""
    assert toolcalls._scalar_equal(49.99, "49.99") is False
    assert toolcalls._scalar_equal("49.99", 49.99) is False
    assert toolcalls._scalar_equal(10, "10") is False
    assert toolcalls._scalar_equal("10", 10) is False


def test_argument_correctness_flags_wrong_value_for_a_stringified_number() -> None:
    """The exact defect shape FR-8(d) exists to catch: a model returns a stringified number where
    a real number was expected. Must register as `wrong_value` — never silently score `allCorrect`
    via `str()`-coercion, and never spuriously classified `boundary_unit` (no `schema` given)."""
    result = toolcalls.argument_correctness({"maxPrice": 49.99}, {"maxPrice": "49.99"})
    assert result.allCorrect is False
    assert result.wrongValue == ("maxPrice",)
    assert result.boundaryUnit == ()


def test_argument_correctness_boundary_unit_is_a_named_subset_of_wrong_value_off_real_schema() -> (
    None
):
    """`-ml` §4.2(d): `boundaryRule` is read off the pack's own `tools/schemas.json` — never a
    regex — using the REAL `filter_products.maxPrice` entry, which declares
    `confusedWith: [4999, 5000]` (S5 spec §3.3)."""
    schemas = real_schemas()
    schema = toolcalls.properties_for_tool(schemas, "filter_products")
    assert "boundaryRule" in schema["maxPrice"]

    result = toolcalls.argument_correctness(
        {"maxPrice": 49.99}, {"maxPrice": 4999}, schema=schema
    )
    assert result.allCorrect is False
    assert result.wrongValue == ("maxPrice",)
    assert result.boundaryUnit == ("maxPrice",)  # named subset: also in wrongValue


def test_argument_correctness_wrong_value_not_matching_any_boundary_confusion_stays_plain() -> None:
    """The count that does NOT move into `boundaryUnit`: a wrong value the schema's own
    `confusedWith` list does not name."""
    schemas = real_schemas()
    schema = toolcalls.properties_for_tool(schemas, "filter_products")
    result = toolcalls.argument_correctness({"maxPrice": 49.99}, {"maxPrice": 1.0}, schema=schema)
    assert result.wrongValue == ("maxPrice",)
    assert result.boundaryUnit == ()


def test_properties_for_tool_unknown_tool_name_returns_empty_mapping() -> None:
    assert toolcalls.properties_for_tool(real_schemas(), "no_such_tool") == {}


# --- (e) spurious_and_duplicate (`-ml` §4.2(e)) ---------------------------------------------------


def test_spurious_and_duplicate_baseline_no_defect() -> None:
    result = toolcalls.spurious_and_duplicate(
        {"lookup_product_fact"}, [("lookup_product_fact", {"name": "Pad"})]
    )
    assert result.spurious is False
    assert result.duplicateWithinTurn is False
    assert result.duplicateCrossTurn is False


def test_spurious_and_duplicate_flags_a_call_to_a_tool_not_in_r_t() -> None:
    result = toolcalls.spurious_and_duplicate(
        {"lookup_product_fact"}, [("lookup_product_fact", {"name": "Pad"}), ("view_cart", {})]
    )
    assert result.spurious is True
    assert result.duplicateWithinTurn is False


def test_spurious_and_duplicate_flags_a_within_turn_repeat() -> None:
    """K-061's own defect: the same-turn duplicate `add_to_cart` call."""
    call = ("add_to_cart", {"productName": "Pad", "quantity": 1})
    result = toolcalls.spurious_and_duplicate({"add_to_cart"}, [call, call])
    assert result.duplicateWithinTurn is True
    assert result.duplicateCrossTurn is False
    assert result.spurious is False


def test_spurious_and_duplicate_flags_a_cross_turn_reissue() -> None:
    """The ministral defect: turn 2 re-issues turn 1's already-completed call."""
    call = ("add_to_cart", {"productName": "Pad", "quantity": 1})
    result = toolcalls.spurious_and_duplicate(
        {"add_to_cart"}, [call], prior_completed_calls=[call]
    )
    assert result.duplicateCrossTurn is True
    assert result.duplicateWithinTurn is False


# --- (f) stopping_when_done + iteration_summary (`-ml` §4.2(f)) -----------------------------------


def test_stopping_when_done_true_when_replied_and_nothing_after_satisfied() -> None:
    assert toolcalls.stopping_when_done("replied", 1, continued_after_satisfied=False) is True


def test_stopping_when_done_false_when_replied_but_continued_after_satisfied() -> None:
    """The count that moves relative to the prior case: same disposition/dispatch count, only
    `continued_after_satisfied` differs."""
    assert toolcalls.stopping_when_done("replied", 2, continued_after_satisfied=True) is False


def test_stopping_when_done_false_on_cap_hit_regardless_of_continued_flag() -> None:
    """`-ml` §4.3 rule 4: a cap-hit turn never stopped on its own — always fails, and the flag is
    irrelevant to that (passed `False` here and it still fails)."""
    assert toolcalls.stopping_when_done("cap-hit", 3, continued_after_satisfied=False) is False


def test_stopping_when_done_false_on_timed_out() -> None:
    """`-ml` §4.3 rule 4: a timeout scores `fail`, never `n_a`, wherever it lands."""
    assert toolcalls.stopping_when_done("timed-out", 1, continued_after_satisfied=False) is False


def test_stopping_when_done_none_when_no_call_was_dispatched() -> None:
    """`|E(t)| = 0` is outside the denominator entirely — including a cap-hit turn
    (item 4.3.1(4))."""
    assert toolcalls.stopping_when_done("cap-hit", 0, continued_after_satisfied=False) is None
    assert toolcalls.stopping_when_done("replied", 0, continued_after_satisfied=False) is None


def test_stopping_when_done_none_on_unrunnable_dispositions() -> None:
    assert toolcalls.stopping_when_done("no-response", 1, continued_after_satisfied=False) is None
    assert (
        toolcalls.stopping_when_done("server-rejected", 1, continued_after_satisfied=False) is None
    )


def test_iteration_summary_empty_when_no_included_observations() -> None:
    summary = toolcalls.iteration_summary([("no-response", 0), ("server-rejected", 0)])
    assert summary.n == 0
    assert summary.mean is None
    assert summary.p95 is None
    assert summary.meanCensored is False
    assert summary.p95Censored is False


def test_iteration_summary_mean_over_replied_turns_only() -> None:
    summary = toolcalls.iteration_summary([("replied", 2), ("replied", 4)])
    assert summary.n == 2
    assert summary.mean == pytest.approx(3.0)
    assert summary.capHitCount == 0
    assert summary.meanCensored is False


def test_iteration_summary_cap_hit_moves_the_mean_and_censors_it() -> None:
    """Appending a `cap-hit` turn: the mean moves AND `meanCensored` flips `True`."""
    baseline = toolcalls.iteration_summary([("replied", 2), ("replied", 4)])
    with_cap_hit = toolcalls.iteration_summary([("replied", 2), ("replied", 4), ("cap-hit", 8)])
    assert with_cap_hit.mean != baseline.mean
    assert with_cap_hit.capHitCount == 1
    assert with_cap_hit.meanCensored is True


def test_iteration_summary_timed_out_leaves_mean_and_p95_unchanged() -> None:
    """`-ml` §4.2(f): a `timed-out` turn is outside `{replied, cap-hit}` — the mean/p95/`n` do not
    move at all, even though it DOES count in `stopping_when_done`'s own denominator (a different
    function, a different denominator)."""
    baseline = toolcalls.iteration_summary([("replied", 2), ("replied", 4)])
    with_timed_out = toolcalls.iteration_summary(
        [("replied", 2), ("replied", 4), ("timed-out", 0)]
    )
    assert with_timed_out.n == baseline.n
    assert with_timed_out.mean == baseline.mean
    assert with_timed_out.p95 == baseline.p95


def test_iteration_summary_no_response_and_server_rejected_leave_it_unchanged_too() -> None:
    baseline = toolcalls.iteration_summary([("replied", 2), ("replied", 4)])
    with_no_response = toolcalls.iteration_summary(
        [("replied", 2), ("replied", 4), ("no-response", 0)]
    )
    with_server_rejected = toolcalls.iteration_summary(
        [("replied", 2), ("replied", 4), ("server-rejected", 0)]
    )
    assert with_no_response.n == baseline.n
    assert with_no_response.mean == baseline.mean
    assert with_server_rejected.n == baseline.n
    assert with_server_rejected.mean == baseline.mean


def test_iteration_summary_mean_is_distinct_from_the_unrestricted_calls_per_turn_average() -> None:
    """Item (7): `Y_calls / Y` (unrestricted, over every turn) and the `I(t)` mean (restricted to
    `{replied, cap-hit}`) are DIFFERENT quantities and must not be substituted for one another —
    demonstrated directly: a run with one non-`replied` turn makes them differ."""
    observations = [("replied", 5), ("replied", 5), ("no-response", 0)]
    restricted_mean = toolcalls.iteration_summary(observations).mean
    unrestricted_mean = sum(it for _, it in observations) / len(observations)
    assert restricted_mean == pytest.approx(5.0)
    assert unrestricted_mean == pytest.approx(10 / 3)
    assert restricted_mean != pytest.approx(unrestricted_mean)


def _independent_p95_rank(x: int) -> int:
    """`-ml` §11.2.1's rank, recomputed here with `math.ceil` over a plain `float` — deliberately
    NOT calling `toolcalls._p95_rank` (asserting that function against its own output would be the
    mirror trap: `iteration_summary` and this recomputation would then agree by construction on
    every mutant of the shared helper). `math.ceil(0.95 * x)` is a fine INDEPENDENT check at these
    sizes (`x <= 200`) even though `-ml` itself warns the float form drifts from the exact integer
    one at larger `x` — this test only needs a second, differently-derived opinion, not the
    package's own one-true-implementation."""
    import math

    return max(1, min(x, math.ceil(0.95 * x)))


def test_p95_exactness_matches_the_iff_rule_swept_both_directions() -> None:
    """Item (6): "the p95 is exact iff `r <= X - c`", swept over `X <= 200` and every `c <= X`, in
    BOTH directions of the iff (a test asserting only the forward half would bless a report that
    printed a bare p95 where a `>=` was owed). The expected rank is computed INDEPENDENTLY of
    `toolcalls._p95_rank` (see `_independent_p95_rank`), so a mutation to that helper cannot pass by
    agreeing with itself."""
    for x in range(1, 201):
        rank = _independent_p95_rank(x)
        for c in range(0, x + 1):
            observations = [("cap-hit", 1)] * c + [("replied", 1)] * (x - c)
            summary = toolcalls.iteration_summary(observations)
            expected_censored = rank > x - c
            assert summary.p95Censored == expected_censored, (x, c, rank)


# --- (g) reply_matches_tool (`-ml` §4.2(g)) -------------------------------------------------------


def test_reply_matches_tool_true_when_must_contain_present_and_must_not_contain_absent() -> None:
    assert (
        toolcalls.reply_matches_tool("Your cart now has 1 Pad.", ["1 Pad"], ["error"]) is True
    )


def test_reply_matches_tool_false_when_a_must_contain_value_is_missing() -> None:
    assert toolcalls.reply_matches_tool("Sure, done!", ["1 Pad"], []) is False


def test_reply_matches_tool_false_when_it_contradicts_the_tool_result() -> None:
    """The K-057-adjacent defect §8.3 documented: narrating success the negative list catches."""
    assert (
        toolcalls.reply_matches_tool(
            "Successfully removed the Pad from your cart!", [], ["still in your cart"]
        )
        is True
    )
    assert (
        toolcalls.reply_matches_tool(
            "Successfully removed the Pad from your cart, still in your cart otherwise.",
            [],
            ["still in your cart"],
        )
        is False
    )


def test_reply_matches_tool_false_when_reply_text_is_none() -> None:
    assert toolcalls.reply_matches_tool(None, [], []) is False


# --- restraint (`-ml` §4.2's own added count) -----------------------------------------------------


def test_restraint_true_when_nothing_dispatched() -> None:
    assert toolcalls.restraint(0) is True


def test_restraint_false_when_a_call_was_dispatched_on_a_restraint_turn() -> None:
    assert toolcalls.restraint(1) is False


# ==================================================================================================
# Step 3, item (9): the five-member disposition set, one distinct behavioural consequence each
# ==================================================================================================


def test_each_turn_disposition_has_a_distinct_scoring_consequence() -> None:
    """`-ml` §4.3.1 item 9 / §4.2(f)'s pin, its own second assertion: five members, five different
    observable consequences (not merely the union/disjointness pin already landed at Step 0)."""
    baseline = [("replied", 3), ("replied", 5)]
    baseline_summary = toolcalls.iteration_summary(baseline)

    # replied: moves the mean.
    with_replied = toolcalls.iteration_summary([*baseline, ("replied", 9)])
    assert with_replied.mean != baseline_summary.mean

    # cap-hit: moves the mean AND flips it to censored.
    with_cap_hit = toolcalls.iteration_summary([*baseline, ("cap-hit", 9)])
    assert with_cap_hit.mean != baseline_summary.mean
    assert with_cap_hit.meanCensored is True

    # timed-out: leaves mean/p95 unchanged, but DOES enter stopping_when_done's denominator and
    # fails it (a different function, a different, non-`I(t)` denominator).
    with_timed_out = toolcalls.iteration_summary([*baseline, ("timed-out", 0)])
    assert with_timed_out.mean == baseline_summary.mean
    assert with_timed_out.p95 == baseline_summary.p95
    assert toolcalls.stopping_when_done("timed-out", 1, continued_after_satisfied=False) is False

    # no-response: leaves mean/p95 unchanged, leaves stopping_when_done's denominator untouched
    # (None, not a scored fail), and is `unrunnable`.
    with_no_response = toolcalls.iteration_summary([*baseline, ("no-response", 0)])
    assert with_no_response.mean == baseline_summary.mean
    assert toolcalls.stopping_when_done("no-response", 1, continued_after_satisfied=False) is None
    assert toolcalls.turn_disposition_scores("no-response") == "unrunnable"

    # server-rejected: the same as no-response, by a different mechanism token.
    with_server_rejected = toolcalls.iteration_summary([*baseline, ("server-rejected", 0)])
    assert with_server_rejected.mean == baseline_summary.mean
    assert (
        toolcalls.stopping_when_done("server-rejected", 1, continued_after_satisfied=False) is None
    )
    assert toolcalls.turn_disposition_scores("server-rejected") == "unrunnable"


# ==================================================================================================
# Step 4 — the ten named adversarial traces (S5 spec §5 Step 4)
# ==================================================================================================


def test_adversarial_no_tool_call() -> None:
    """(1) no tool call: `R(t) >= 1`, nothing dispatched, no prose either."""
    turn = make_turn(dispatches=())
    assert (
        toolcalls.emission_form(
            required=True, dispatched_count=len(turn.dispatches), prose_detected=False
        )
        == "no_attempt"
    )


def test_adversarial_prose_pseudo_call() -> None:
    """(2) prose pseudo-call: the model narrates the call instead of dispatching it."""
    reply = 'I will call add_to_cart("Pad", 1) for you.'
    turn = make_turn(dispatches=(), final_reply=reply)
    prose_detected = toolcalls.detect_prose_pseudo_call(turn.finalReplyText)
    assert prose_detected is True
    assert (
        toolcalls.emission_form(
            required=True, dispatched_count=len(turn.dispatches), prose_detected=prose_detected
        )
        == "prose_pseudo_call"
    )


def test_adversarial_wrong_tool() -> None:
    """(3) wrong tool: the model called a real tool, just not the required one."""
    turn = make_turn(dispatches=(make_dispatch("filter_products", {}),))
    dispatched_names = {d.name for d in turn.dispatches}
    assert toolcalls.right_tool_chosen({"lookup_product_fact"}, dispatched_names) is False


def test_adversarial_omitted_required_argument() -> None:
    """(4) omitted required argument."""
    turn = make_turn(dispatches=(make_dispatch("lookup_product_fact", {}),))
    result = toolcalls.argument_correctness({"name": "Pad"}, turn.dispatches[0].parsedArguments)
    assert result.allCorrect is False
    assert result.omittedRequired == ("name",)


def test_adversarial_mis_translated_boundary() -> None:
    """(5) mis-translated boundary: the model used the boundary-confused cents/off-by-one value
    the pack's own `boundaryRule` names, off the REAL `tools/schemas.json`."""
    schema = toolcalls.properties_for_tool(real_schemas(), "filter_products")
    turn = make_turn(dispatches=(make_dispatch("filter_products", {"maxPrice": 5000}),))
    result = toolcalls.argument_correctness(
        {"maxPrice": 49.99}, turn.dispatches[0].parsedArguments, schema=schema
    )
    assert result.wrongValue == ("maxPrice",)
    assert result.boundaryUnit == ("maxPrice",)


def test_adversarial_duplicated_within_turn_call() -> None:
    """(6) duplicated within-turn call: K-061's own same-turn `add_to_cart` defect."""
    add = make_dispatch("add_to_cart", {"productName": "Pad", "quantity": 1})
    turn = make_turn(dispatches=(add, add))
    calls = [(d.name, d.parsedArguments) for d in turn.dispatches]
    result = toolcalls.spurious_and_duplicate({"add_to_cart"}, calls)
    assert result.duplicateWithinTurn is True


def test_adversarial_cross_turn_reissue() -> None:
    """(7) cross-turn re-issue: the ministral defect, turn 2 re-issuing turn 1's completed call."""
    add = make_dispatch("add_to_cart", {"productName": "Pad", "quantity": 1})
    turn1 = make_turn(dispatches=(add,))
    turn2 = make_turn(dispatches=(add,))
    prior_calls = [(d.name, d.parsedArguments) for d in turn1.dispatches]
    this_turn_calls = [(d.name, d.parsedArguments) for d in turn2.dispatches]
    result = toolcalls.spurious_and_duplicate(
        {"add_to_cart"}, this_turn_calls, prior_completed_calls=prior_calls
    )
    assert result.duplicateCrossTurn is True


def test_adversarial_kept_going_after_done() -> None:
    """(8) kept going after done: R(t) was already satisfied and the model called again anyway."""
    turn = make_turn(
        disposition="replied",
        dispatches=(make_dispatch("add_to_cart", {}), make_dispatch("view_cart", {})),
    )
    assert (
        toolcalls.stopping_when_done(
            turn.turnDisposition, len(turn.dispatches), continued_after_satisfied=True
        )
        is False
    )


def test_adversarial_contradicted_the_tool_result() -> None:
    """(9) contradicted the tool result: narrated a removal the cart state disagrees with —
    §8.3's price-shaped-token blind spot, caught by the negative list."""
    turn = make_turn(final_reply="Successfully removed the Pad from your cart!")
    assert (
        toolcalls.reply_matches_tool(turn.finalReplyText, [], ["still shows the Pad"]) is True
    )
    turn_contradicted = make_turn(
        final_reply="Successfully removed the Pad — it still shows the Pad in your cart though."
    )
    assert (
        toolcalls.reply_matches_tool(
            turn_contradicted.finalReplyText, [], ["still shows the Pad"]
        )
        is False
    )


def test_adversarial_called_a_tool_when_none_required() -> None:
    """(10) called a tool when none required: a restraint-turn violation."""
    turn = make_turn(dispatches=(make_dispatch("view_cart", {}),))
    assert toolcalls.restraint(len(turn.dispatches)) is False


# --- the two named discriminating pairs (S5 spec §5 Step 4 / `-ml` §4.3.1 item 4) -----------------


def test_discriminating_pair_cap_hit_with_empty_dispatch_trace_all_five_consequences() -> None:
    """A `cap-hit` turn with an EMPTY dispatch trace: all five consequences at once (item 4.3.1(4)).
    """
    turn = make_turn(disposition="cap-hit", dispatches=(), iterations=8, final_reply=None)

    # (1) no_attempt partition.
    assert (
        toolcalls.emission_form(
            required=True, dispatched_count=len(turn.dispatches), prose_detected=False
        )
        == "no_attempt"
    )
    # (2) DOES enter iteration_cap_hit_rate's numerator (kept on the scored side, disposition
    # alone — no `E(t)` condition on that count).
    assert toolcalls.turn_disposition_scores(turn.turnDisposition) == "scored"
    assert turn.turnDisposition == "cap-hit"
    # (3) DOES enter the I(t) summary.
    summary = toolcalls.iteration_summary([(turn.turnDisposition, turn.iterations)])
    assert summary.n == 1
    assert summary.capHitCount == 1
    # (4) ABSENT from stopping_when_done's denominator (|E(t)| = 0).
    assert (
        toolcalls.stopping_when_done(
            turn.turnDisposition, len(turn.dispatches), continued_after_satisfied=False
        )
        is None
    )
    # (5) ABSENT from (g)'s unscoreable bucket: (g)'s denominator/unscoreable split is itself
    # conditioned on `|E(t)| >= 1` (rule 4's table); this turn never reaches either side of it.
    assert len(turn.dispatches) == 0


def test_discriminating_pair_timed_out_scores_fail_no_response_scores_unrunnable() -> None:
    """A `timed-out` turn beside an otherwise-identical `no-response` turn: `fail` against
    `unrunnable` (item 4.3.1(4))."""
    dispatch = make_dispatch("lookup_product_fact", {"name": "Pad"})
    timed_out = make_turn(disposition="timed-out", dispatches=(dispatch,), final_reply=None)
    no_response = make_turn(disposition="no-response", dispatches=(dispatch,), final_reply=None)

    assert toolcalls.turn_disposition_scores(timed_out.turnDisposition) == "scored"
    assert toolcalls.turn_disposition_scores(no_response.turnDisposition) == "unrunnable"
    assert (
        toolcalls.stopping_when_done(
            timed_out.turnDisposition, len(timed_out.dispatches), continued_after_satisfied=False
        )
        is False
    )
    assert (
        toolcalls.stopping_when_done(
            no_response.turnDisposition,
            len(no_response.dispatches),
            continued_after_satisfied=False,
        )
        is None
    )


# --- clean_through_turn (`-ml` §4.3 rule 5 / plan §4.6) -----------------------------------------


def test_clean_through_turn_clean_when_every_considered_turn_is_clean() -> None:
    trace = make_trace("A-01", tuple(make_turn() for _ in range(4)))
    assert toolcalls.clean_through_turn(trace, [True, True, True, True], h=4) == "clean"


def test_clean_through_turn_failed_when_a_considered_turn_is_not_clean() -> None:
    trace = make_trace("A-01", tuple(make_turn() for _ in range(4)))
    assert toolcalls.clean_through_turn(trace, [True, True, False, True], h=4) == "failed"


def test_clean_through_turn_third_state_when_an_unrunnable_turn_is_at_or_before_h() -> None:
    """`-ml` §4.3 rule 5's third state: a conversation with an `unrunnable` turn at any `t <= H` is
    neither clean nor failed."""
    turns = (
        make_turn(),
        make_turn(),
        make_turn(disposition="no-response", final_reply=None),
        make_turn(),
    )
    trace = make_trace("A-01", turns)
    assert toolcalls.clean_through_turn(trace, [True, True, True, True], h=4) == "n_a"


def test_clean_through_turn_third_state_when_trace_is_shorter_than_h_tool_channel_censoring() -> (
    None
):
    """A tool-channel `ToolDispatchFailed` truncates the trace before `H` is ever reached."""
    trace = make_trace("A-01", (make_turn(), make_turn()))  # only 2 turns recorded, H=4
    assert toolcalls.clean_through_turn(trace, [True, True], h=4) == "n_a"


def test_clean_through_turn_discriminating_pair_h4_unrunnable_at_t5_vs_t3() -> None:
    """The `H = 4` discriminating pair: an `unrunnable` at `t = 5` (index 4) leaves the conversation
    IN `cleanThroughTurn4`'s denominator; one at `t = 3` (index 2) takes it OUT — same clean data
    otherwise."""
    turns_unrunnable_at_5 = (
        make_turn(),
        make_turn(),
        make_turn(),
        make_turn(),
        make_turn(disposition="server-rejected", final_reply=None),
    )
    trace_at_5 = make_trace("A-01", turns_unrunnable_at_5)
    assert toolcalls.clean_through_turn(trace_at_5, [True] * 5, h=4) == "clean"

    turns_unrunnable_at_3 = (
        make_turn(),
        make_turn(),
        make_turn(disposition="server-rejected", final_reply=None),
        make_turn(),
        make_turn(),
    )
    trace_at_3 = make_trace("A-02", turns_unrunnable_at_3)
    assert toolcalls.clean_through_turn(trace_at_3, [True] * 5, h=4) == "n_a"


def test_clean_through_turn_refuses_a_mismatched_turn_clean_length() -> None:
    trace = make_trace("A-01", (make_turn(), make_turn()))
    with pytest.raises(ValueError):
        toolcalls.clean_through_turn(trace, [True], h=2)


# --- hazard_points (`-ml` §4.3.1 item 11) -------------------------------------------------------


def _clean_nine_turn_trace(script_id: str) -> ConversationTrace:
    return make_trace(script_id, tuple(make_turn() for _ in range(9)))


def test_hazard_points_nine_turn_fixture_one_unrunnable_at_t2() -> None:
    """Item 11(a): a 9-turn conversation with one `unrunnable` turn at `t = 2` (index 1), three
    others clean throughout: in the risk set at `t = 1`, in NO risk set at `t >= 2`; `c_2 == 1`;
    the hazard at `t >= 3` is over 3 conversations, not 4."""
    censored_turns = tuple(
        make_turn() if i != 1 else make_turn(disposition="no-response", final_reply=None)
        for i in range(9)
    )
    traces = [
        make_trace("A-01", censored_turns),
        _clean_nine_turn_trace("A-02"),
        _clean_nine_turn_trace("A-03"),
        _clean_nine_turn_trace("A-04"),
    ]
    turn_clean = [[True] * 9 for _ in traces]

    points = toolcalls.hazard_points(traces, turn_clean)

    # position index 0 == "t=1": all four in the risk set, none censored yet (both mutation
    # directions' first half: turns 1..t-1 are NOT excluded).
    assert points[0].metric.n == 4
    assert points[0].censored == 0

    # position index 1 == "t=2": the unrunnable turn is newly censored HERE.
    assert points[1].censored == 1
    assert points[1].metric.n == 3  # excluded from its OWN position's risk set too (rule 4)

    # position index 2 == "t=3" onward: the hazard is over 3 conversations, not 4 (both mutation
    # directions' second half: it must NOT still be counted at t >= 3).
    for point in points[2:]:
        assert point.metric.n == 3
        assert point.censored == 0


def test_hazard_points_h4_discriminating_pair_t5_vs_t3() -> None:
    """Item 11(b): at `H = 4`, an `unrunnable` at `t = 5` (index 4) is OUT of the hazard from
    `t = 5` onward, distinct from `clean_through_turn`'s own treatment of the same conversation
    (which keeps it IN at `h=4`, tested separately above) — proving the two rules are genuinely
    different, not the same rule applied twice."""
    turns_unrunnable_at_5 = (
        make_turn(),
        make_turn(),
        make_turn(),
        make_turn(),
        make_turn(disposition="server-rejected", final_reply=None),
    )
    turns_unrunnable_at_3 = (
        make_turn(),
        make_turn(),
        make_turn(disposition="server-rejected", final_reply=None),
        make_turn(),
        make_turn(),
    )
    traces = [make_trace("A-01", turns_unrunnable_at_5), make_trace("A-02", turns_unrunnable_at_3)]
    turn_clean = [[True] * 5, [True] * 5]

    points = toolcalls.hazard_points(traces, turn_clean)

    # Both conversations present at index 0/1 ("t=1,2"): neither has been censored yet.
    assert points[0].metric.n == 2
    assert points[1].metric.n == 2

    # At index 2 ("t=3"): A-02 is unrunnable HERE (out of its own position's risk set, rule 4),
    # leaving only A-01.
    assert points[2].metric.n == 1
    assert points[2].censored == 1

    # At index 3 ("t=4"): A-02 already left at index 2 — only A-01 remains.
    assert points[3].metric.n == 1
    assert points[3].censored == 0

    # At index 4 ("t=5"): A-01 is unrunnable HERE too, leaving nobody.
    assert points[4].metric.n == 0
    assert points[4].censored == 1


def test_hazard_points_c_t_zero_when_nothing_is_newly_censored() -> None:
    """Item 11(d)'s data half: `c_t == 0` on an ordinary position with no censoring event."""
    traces = [_clean_nine_turn_trace("A-01"), _clean_nine_turn_trace("A-02")]
    turn_clean = [[True] * 9, [True] * 9]
    points = toolcalls.hazard_points(traces, turn_clean)
    assert all(point.censored == 0 for point in points)
    assert all(point.metric.n == 2 for point in points)


def test_hazard_points_a_failure_also_leaves_the_risk_set_for_later_positions() -> None:
    """A conversation that fails (not censored) at `t` is out of the risk set at every later
    position too — this is a time-to-FIRST-failure curve, not an independent per-position rate."""
    traces = [_clean_nine_turn_trace("A-01"), _clean_nine_turn_trace("A-02")]
    turn_clean = [[True, True, False, True, True, True, True, True, True], [True] * 9]
    points = toolcalls.hazard_points(traces, turn_clean)
    assert points[2].metric.n == 2  # both still at risk entering t=3 (index 2)
    assert points[2].metric.successes == 1  # A-01 fails here
    for point in points[3:]:
        assert point.metric.n == 1  # A-01 has left the risk set for good


def test_hazard_points_h_bounds_how_many_positions_are_computed() -> None:
    traces = [_clean_nine_turn_trace("A-01")]
    turn_clean = [[True] * 9]
    points = toolcalls.hazard_points(traces, turn_clean, h=4)
    assert len(points) == 4


def test_hazard_points_refuses_a_mismatched_turn_clean_shape() -> None:
    traces = [_clean_nine_turn_trace("A-01")]
    with pytest.raises(ValueError):
        toolcalls.hazard_points(traces, [[True, True]])
    with pytest.raises(ValueError):
        toolcalls.hazard_points(traces, [])


# --- outcome_vectors_differ (plan item 12b) -----------------------------------------------------


def test_outcome_vectors_differ_empty_on_an_identical_pair() -> None:
    dispatch = make_dispatch("lookup_product_fact", {"name": "Pad"})
    a = make_trace("A-01", (make_turn(dispatches=(dispatch,), final_reply="24.99"),))
    b = make_trace("A-01", (make_turn(dispatches=(dispatch,), final_reply="24.99"),))
    assert toolcalls.outcome_vectors_differ(a, b) == ()


def test_outcome_vectors_differ_reports_the_one_differing_turn_index() -> None:
    dispatch = make_dispatch("lookup_product_fact", {"name": "Pad"})
    a = make_trace(
        "A-01",
        (
            make_turn(dispatches=(dispatch,), final_reply="24.99"),
            make_turn(final_reply="Anything else?"),
        ),
    )
    b = make_trace(
        "A-01",
        (
            make_turn(dispatches=(dispatch,), final_reply="24.99"),
            make_turn(final_reply="Anything else I can help with?"),
        ),
    )
    assert toolcalls.outcome_vectors_differ(a, b) == (1,)


def test_outcome_vectors_differ_ignores_wall_clock_and_timestamp_only_differences() -> None:
    """Deliberately NOT part of the comparand: `wallClockMs` and `DispatchRecord.timestamp`."""
    a = make_trace(
        "A-01",
        (
            TurnTrace(
                messagesSent=(),
                chatResults=(),
                dispatches=(
                    DispatchRecord(
                        name="lookup_product_fact",
                        rawArguments={"name": "Pad"},
                        parsedArguments={"name": "Pad"},
                        returnValue={"price": 24.99},
                        timestamp="2026-09-13T00:00:00Z",
                    ),
                ),
                envState={},
                iterations=1,
                turnDisposition="replied",
                finalReplyText="24.99",
                wallClockMs=1.0,
            ),
        ),
    )
    b = make_trace(
        "A-01",
        (
            TurnTrace(
                messagesSent=(),
                chatResults=(),
                dispatches=(
                    DispatchRecord(
                        name="lookup_product_fact",
                        rawArguments={"name": "Pad"},
                        parsedArguments={"name": "Pad"},
                        returnValue={"price": 999.0},  # returnValue differs — not compared
                        timestamp="2026-09-13T00:05:00Z",  # timestamp differs — not compared
                    ),
                ),
                envState={},
                iterations=1,
                turnDisposition="replied",
                finalReplyText="24.99",
                wallClockMs=999.0,  # wallClockMs differs — not compared
            ),
        ),
    )
    assert toolcalls.outcome_vectors_differ(a, b) == ()


def test_outcome_vectors_differ_length_mismatch_counts_as_differing() -> None:
    dispatch = make_dispatch("lookup_product_fact", {"name": "Pad"})
    a = make_trace("A-01", (make_turn(dispatches=(dispatch,), final_reply="24.99"),))
    b = make_trace(
        "A-01",
        (
            make_turn(dispatches=(dispatch,), final_reply="24.99"),
            make_turn(final_reply="Anything else?"),
        ),
    )
    assert toolcalls.outcome_vectors_differ(a, b) == (1,)


# ==================================================================================================
# Step 5 — `score_conversations` (S5 spec §5 Step 5, §2.3/§2.6)
# ==================================================================================================


def test_score_conversations_e2_dispatch_censoring_funnel_and_hazard() -> None:
    """E2, re-tested against the REAL scorer (not just the runner-level truncation `test_runner.py`
    already covers, §2.1): a synthetic 9-turn/raise-at-`t=4`/`H=4` fixture. Funnel prints 1 under
    `unrunnableToolChannel`; the headline denominator is 3 with 1 in its n/a tally; the hazard risk
    set is 3 at `t >= 4` with `c_4 == 1`."""
    censored_trace = make_trace("A-01", tuple(make_turn() for _ in range(3)))  # raise at t=4
    scored = [
        (make_script("A-01", 9), censored_trace, ()),
        _clean_scored("A-02"),
        _clean_scored("A-03"),
        _clean_scored("A-04"),
    ]

    items, aggregates = toolcalls.score_conversations(scored, [], pack=make_pack(h=4))

    assert len(items) == 4
    assert aggregates.funnelCounts.unrunnableToolChannel == 1

    scoreable_count = sum(1 for it in items if it.scoreable.get("cleanThroughTurnH") is True)
    na_count = sum(1 for it in items if it.scoreable.get("cleanThroughTurnH") is False)
    assert scoreable_count == 3
    assert na_count == 1

    hazard = aggregates.hazard
    assert hazard[3].censored == 1  # "c_4 == 1": newly censored AT t=4 (index 3)
    for point in hazard[3:]:
        assert point.metric.n == 3  # the hazard risk set from t=4 onward is 3, not 4


def test_score_conversations_e3_censoring_after_h_stays_in_the_headline_denominator() -> None:
    """E3: the same fixture with the raise moved to `t = 5` — IN `cleanThroughTurn4`'s denominator
    (the censoring happens after H) and OUT of the hazard from `t = 5` onward."""
    censored_trace = make_trace("A-01", tuple(make_turn() for _ in range(4)))  # raise at t=5
    scored = [
        (make_script("A-01", 9), censored_trace, ()),
        _clean_scored("A-02"),
        _clean_scored("A-03"),
        _clean_scored("A-04"),
    ]

    items, aggregates = toolcalls.score_conversations(scored, [], pack=make_pack(h=4))

    censored_item = next(it for it in items if it.itemId == "A-01")
    assert censored_item.scoreable["cleanThroughTurnH"] is True  # IN the denominator
    assert censored_item.outcome == "pass"
    assert censored_item.counts["cleanThroughTurnH"] == 1

    # Still a tool-channel censoring event — FunnelCounts tracks it regardless of H.
    assert aggregates.funnelCounts.unrunnableToolChannel == 1

    hazard = aggregates.hazard
    assert hazard[3].metric.n == 4  # t=4 (index 3): not yet censored
    assert hazard[4].censored == 1  # t=5 (index 4): newly censored HERE
    for point in hazard[4:]:
        assert point.metric.n == 3  # OUT of the hazard from t=5 onward


def test_score_conversations_e4_censoring_differs_from_an_ordinary_scored_failure() -> None:
    """E4: the E2 fixture rendered beside one where turn 4 is an ordinary SCORED failure (a
    `replied` turn that fails an FR-8 check) rather than a dispatch-censoring raise — the
    headline/hazard figures must differ. If they did not, E2 would have passed on a tautology:
    censoring would look indistinguishable from a mundane scored miss."""
    # --- E2's own fixture ------------------------------------------------------------------------
    censored_trace = make_trace("A-01", tuple(make_turn() for _ in range(3)))
    scored_e2 = [
        (make_script("A-01", 9), censored_trace, ()),
        _clean_scored("A-02"),
        _clean_scored("A-03"),
        _clean_scored("A-04"),
    ]
    items_e2, aggregates_e2 = toolcalls.score_conversations(scored_e2, [], pack=make_pack(h=4))

    # --- the negative control: turn 4 (index 3) requires a tool and the model never calls it -----
    # (a genuine `replied`, FR-8-scored miss — never a dispatch raise, never truncated).
    failing_script_turns = [_restraint_turn(i) for i in range(9)]
    failing_script_turns[3] = _required_turn(3)
    failing_script = Conversation(
        scriptId="A-01", shape="A", replicate=1, turns=tuple(failing_script_turns)
    )
    failing_trace = make_trace("A-01", tuple(make_turn() for _ in range(9)))  # never dispatches
    scored_e4 = [
        (failing_script, failing_trace, ()),
        _clean_scored("A-02"),
        _clean_scored("A-03"),
        _clean_scored("A-04"),
    ]
    items_e4, aggregates_e4 = toolcalls.score_conversations(scored_e4, [], pack=make_pack(h=4))

    # The headline differs: E2 excludes A-01 (n/a); E4 scores it a genuine failure (still counted).
    scoreable_e2 = sum(1 for it in items_e2 if it.scoreable.get("cleanThroughTurnH") is True)
    scoreable_e4 = sum(1 for it in items_e4 if it.scoreable.get("cleanThroughTurnH") is True)
    assert scoreable_e2 != scoreable_e4
    assert aggregates_e2.funnelCounts.unrunnableToolChannel == 1
    assert aggregates_e4.funnelCounts.unrunnableToolChannel == 0

    # The hazard differs at t=4 (index 3): E2 excludes A-01 from the risk set THERE (censored);
    # E4 keeps it in the risk set (it is a real, observed failure, not a censoring event).
    assert aggregates_e2.hazard[3].metric.n == 3
    assert aggregates_e2.hazard[3].censored == 1
    assert aggregates_e4.hazard[3].metric.n == 4
    assert aggregates_e4.hazard[3].censored == 0
    assert aggregates_e4.hazard[3].metric.successes == 1


def test_laundering_a_trace_collapsed_at_turn_two_does_not_outscore_one_reaching_turn_eight() -> (
    None
):
    """Plan item 7's laundering guard: a conversation censored after ONE completed turn must never
    register as clean, and must not out-score — on `cleanThroughTurn`'s own numerator OR
    denominator — a conversation driven the full nine turns that only fails once, well after `H`."""
    collapsed_script = make_script("SHORT", 9)
    collapsed_trace = make_trace("SHORT", (make_turn(),))  # censored after turn 1 (raise at t=2)

    long_script_turns = [_restraint_turn(i) for i in range(9)]
    long_script_turns[7] = _required_turn(7)  # a late (post-H) failure — irrelevant to H=4
    long_script = Conversation(
        scriptId="LONG", shape="A", replicate=1, turns=tuple(long_script_turns)
    )
    long_trace = make_trace("LONG", tuple(make_turn() for _ in range(9)))  # no dispatch at t=8

    scored = [(collapsed_script, collapsed_trace, ()), (long_script, long_trace, ())]
    items, aggregates = toolcalls.score_conversations(scored, [], pack=make_pack(h=4))

    short_item = next(it for it in items if it.itemId == "SHORT")
    long_item = next(it for it in items if it.itemId == "LONG")

    assert short_item.outcome == "unrunnable"
    assert short_item.scoreable["cleanThroughTurnH"] is False

    assert long_item.outcome == "pass"  # the late, post-H miss does not touch H=4's own verdict
    assert long_item.scoreable["cleanThroughTurnH"] is True
    assert long_item.counts["cleanThroughTurnH"] == 1

    # The pooled headline excludes SHORT from both the numerator AND the denominator — it cannot
    # inflate a rate it never entered, and it cannot be read as "worse" either: it is simply absent.
    assert aggregates.cleanThroughTurn.successes == 1
    assert aggregates.cleanThroughTurn.n == 1


# --- S5 spec §4.4 item 4 / §5 Step 6: `iterationSummary` -----------------------------------------


def test_score_conversations_populates_iteration_summary_restricted_and_unrestricted() -> None:
    """`iterationSummary` carries `-ml` §4.2(f)'s restricted `I(t)` mean/p95 (over
    `replied`/`cap-hit` turns only, `iteration_summary`'s own filter) alongside the UNRESTRICTED
    `yCalls`/`y` pair `-ml` §11.4's `Y_calls / Y` reads off of — summed/counted over EVERY turn
    driven, including the `no-response` turn `iteration_summary` itself excludes. The two must
    differ whenever a non-`replied`/`cap-hit` turn was driven, which is the distinctness the
    report's own sentence asserts (§4.4 item 4)."""
    replied_script = Conversation(
        scriptId="A-01", shape="A", replicate=1, turns=(_restraint_turn(0), _restraint_turn(1))
    )
    replied_trace = make_trace(
        "A-01",
        (
            make_turn(disposition="replied", iterations=2),
            make_turn(disposition="replied", iterations=4),
        ),
    )
    cap_hit_script = make_script("A-02", 1)
    cap_hit_trace = make_trace(
        "A-02", (make_turn(disposition="cap-hit", iterations=8, final_reply=None),)
    )
    unrunnable_script = make_script("A-03", 1)
    unrunnable_trace = make_trace(
        "A-03", (make_turn(disposition="no-response", iterations=0, final_reply=None),)
    )
    scored = [
        (replied_script, replied_trace, ()),
        (cap_hit_script, cap_hit_trace, ()),
        (unrunnable_script, unrunnable_trace, ()),
    ]

    _, aggregates = toolcalls.score_conversations(scored, [], pack=make_pack(h=1))

    expected = toolcalls.iteration_summary(
        [("replied", 2), ("replied", 4), ("cap-hit", 8), ("no-response", 0)]
    )
    summary = aggregates.iterationSummary
    assert summary["n"] == expected.n == 3
    assert summary["capHitCount"] == expected.capHitCount == 1
    assert summary["mean"] == pytest.approx(expected.mean)
    assert summary["meanCensored"] is expected.meanCensored is True
    assert summary["p95"] == pytest.approx(expected.p95)
    assert summary["p95Censored"] == expected.p95Censored
    # unrestricted: every turn driven (4), including the `no-response` one `iteration_summary`
    # excludes from its own `n` — plus `-ml` §11.4's own `+1` for that turn's own non-returning
    # attempt (its disposition is in `ITERATION_SUMMARY_EXCLUDED`); `iterations=0` here happens to
    # make `+0` and `+1` indistinguishable in THIS fixture (review Finding 2's own gap — a fixture
    # with a non-zero `iterations` excluded-disposition turn discriminates the two, see
    # `test_iteration_summary_ycalls_counts_the_turns_own_nonreturning_attempt`).
    assert summary["yCalls"] == 2 + 4 + 8 + 0 + 1
    assert summary["y"] == 4
    assert summary["mean"] != summary["yCalls"] / summary["y"]  # the distinctness §4.4 item 4 names


@pytest.mark.parametrize("disposition", ["timed-out", "no-response", "server-rejected"])
def test_iteration_summary_ycalls_counts_the_turns_own_nonreturning_attempt(
    disposition: str,
) -> None:
    """`analyst` review `docs/reviews/small-model-benchmarking-s5.md` Finding 2 (major): `-ml`
    §11.4's own formula is `a_i = callCount_i + [D(t_i) in {timed-out, no-response,
    server-rejected}]` per item — this module's own `ITERATION_SUMMARY_EXCLUDED` names exactly
    that set. A turn that completed `iterations=2` calls before its own disposition landed in
    that set must contribute `2 + 1 = 3` to the unrestricted `yCalls`, not the bare `2` a plain
    `sum(iterations)` gives — the turn's own final, non-returning call attempt is real cost that
    was never recorded as a completed iteration. The existing `iterations=0` fixture in
    `test_score_conversations_populates_iteration_summary_restricted_and_unrestricted` cannot
    discriminate `+0` from `+1` since both give the same total (`0`); this fixture's `iterations=2`
    can."""
    script = make_script("A-01", 1)  # one restraint turn (toolRequired: False)
    turn = make_turn(disposition=disposition, iterations=2, final_reply=None)
    trace = make_trace("A-01", (turn,))

    _, aggregates = toolcalls.score_conversations([(script, trace, ())], [], pack=make_pack(h=1))

    assert aggregates.iterationSummary["yCalls"] == 3
    assert aggregates.iterationSummary["y"] == 1


def test_score_conversations_model_channel_unrunnable_is_not_tool_channel() -> None:
    """The sharp edge §2.6/§2.3 name explicitly: a MODEL-channel `unrunnable` turn
    (`no-response`/`server-rejected`) makes the item `scoreable=False` (n_a) exactly like
    tool-channel censoring does, but must NOT be counted in `FunnelCounts.unrunnableToolChannel` —
    that field is the `len(trace.turns) < len(script.turns)` mechanism alone (§2.3), never the
    model-channel one. A scorer that conflated the two, or double-counted, would pass every other
    test in this file and still misreport E5's own dispatch-failure disclosure line."""
    turns = (
        make_turn(),
        make_turn(),
        make_turn(),
        make_turn(disposition="no-response", final_reply=None),  # t=4 — model channel, NOT a raise
        make_turn(),
        make_turn(),
        make_turn(),
        make_turn(),
        make_turn(),
    )
    trace = make_trace("A-01", turns)  # full 9 turns recorded — never truncated
    scored = [(make_script("A-01", 9), trace, ())]

    items, aggregates = toolcalls.score_conversations(scored, [], pack=make_pack(h=4))

    assert items[0].scoreable["cleanThroughTurnH"] is False
    assert items[0].outcome == "unrunnable"
    assert aggregates.funnelCounts.unrunnableToolChannel == 0
    assert aggregates.funnelCounts.unrunnableModelChannel == 1
    assert aggregates.funnelCounts.turnsDriven == 9  # every scripted turn was still driven


def test_score_conversations_cap_hit_with_a_correct_dispatched_call_is_still_not_clean() -> None:
    """Another sharp edge: `-ml` §4.3 rule 4 — `cap-hit` "never stopped on its own — always
    fails" (`stopping_when_done`'s own rule, already pinned at Step 3). A turn that dispatches the
    right tool with fully correct arguments must STILL fail `cleanThroughTurnH` if its own
    disposition is `cap-hit` rather than `replied` — right-tool-and-right-args is not sufficient
    for "clean" on its own, and a scorer that dropped the disposition check here would pass every
    other test in this file while silently crediting a conversation that never actually finished
    its turn."""
    script = Conversation(
        scriptId="A-01", shape="A", replicate=1,
        turns=(Turn(seq=0, user="turn 0", expect={"toolRequired": True, "tool": "view_cart"}),),
    )
    call = make_dispatch("view_cart", {})
    cap_hit_turn = make_turn(disposition="cap-hit", dispatches=(call,), final_reply=None)
    trace = make_trace("A-01", (cap_hit_turn,))

    items, _ = toolcalls.score_conversations([(script, trace, ())], [], pack=make_pack(h=1))

    assert items[0].outcome == "fail"
    assert items[0].counts["cleanThroughTurnH"] == 0


@pytest.mark.parametrize("disposition", ["cap-hit", "timed-out"])
def test_score_conversations_restraint_turn_that_never_replied_is_still_not_clean(
    disposition: str,
) -> None:
    """The restraint-branch analogue of the `cap-hit`-with-a-correct-call test above: a restraint
    turn (`R(t) = ∅`) that dispatches NOTHING (`restraint(dispatched_count)` is `True`) but whose
    own disposition is `cap-hit` (rambled through every iteration without ever calling a tool or
    finishing) or `timed-out` (never came back at all) rather than `replied` must still fail
    `cleanThroughTurnH` at that turn — dispatching nothing is not sufficient for "clean" on a
    restraint turn that never actually replied. A scorer that scored `clean and dispatched_count
    == 0` alone, dropping the disposition check, would pass every other test in this file while
    silently crediting a conversation that never finished its restraint turn."""
    script = make_script("A-01", 1)  # one restraint turn (toolRequired: False)
    non_replied_turn = make_turn(disposition=disposition, dispatches=(), final_reply=None)
    trace = make_trace("A-01", (non_replied_turn,))

    items, _ = toolcalls.score_conversations([(script, trace, ())], [], pack=make_pack(h=1))

    assert items[0].outcome == "fail"
    assert items[0].counts["cleanThroughTurnH"] == 0


@pytest.mark.parametrize("disposition", ["cap-hit", "timed-out"])
def test_score_conversations_restraint_turn_that_never_replied_is_not_a_restraint_success(
    disposition: str,
) -> None:
    """`analyst` review `docs/reviews/small-model-benchmarking-s5.md` Finding 1 (blocker): U142
    gated `turn_clean` on `t_turn.turnDisposition == "replied"` for a restraint turn but left the
    sibling `tally.restraintSuccesses` tally, three lines above it in the same branch, ungated —
    it credits `restraint(dispatched_count)` alone. The standalone `aggregates.restraint`
    `BinaryMetric` (which reaches `report.py`'s "## Arms" table unchanged via `named_metrics()`)
    must not count a restraint turn that dispatched nothing but never actually replied
    (`cap-hit`/`timed-out`) as a restraint SUCCESS, even though the same turn's `items[0].outcome`
    is already `"fail"` (the test above). The existing parametrized headline test only asserts
    `items[0].outcome`/`.counts`, never `aggregates.restraint` — this is exactly that gap."""
    script = make_script("A-01", 1)  # one restraint turn (toolRequired: False)
    non_replied_turn = make_turn(disposition=disposition, dispatches=(), final_reply=None)
    trace = make_trace("A-01", (non_replied_turn,))

    _, aggregates = toolcalls.score_conversations([(script, trace, ())], [], pack=make_pack(h=1))

    assert aggregates.restraint.successes == 0
    assert aggregates.restraint.n == 1


# --- determinismProbe (plan `:2262-2264`) -----------------------------------------------------


def test_determinism_probe_identical_when_the_probe_matches_its_scored_counterpart() -> None:
    trace = make_trace("A-01", tuple(make_turn() for _ in range(9)))
    probe_trace = make_trace("A-01", tuple(make_turn() for _ in range(9)))
    scored = [(_clean_nine_turn_script("A-01"), trace, ())]
    probes = [(_clean_nine_turn_script("A-01"), probe_trace, ())]

    _, aggregates = toolcalls.score_conversations(
        scored, probes, pack=make_pack(h=4, determinism_probe_scripts=("A-01",))
    )
    probe = aggregates.determinismProbe
    assert probe["ran"] is True
    assert probe["identical"] is True
    assert probe["differingTurns"] == []


def test_determinism_probe_not_identical_when_a_probe_turn_differs() -> None:
    trace = make_trace("A-01", tuple(make_turn() for _ in range(9)))
    probe_turns = list(make_turn() for _ in range(9))
    probe_turns[2] = make_turn(final_reply="a completely different reply")
    probe_trace = make_trace("A-01", tuple(probe_turns))
    scored = [(_clean_nine_turn_script("A-01"), trace, ())]
    probes = [(_clean_nine_turn_script("A-01"), probe_trace, ())]

    _, aggregates = toolcalls.score_conversations(
        scored, probes, pack=make_pack(h=4, determinism_probe_scripts=("A-01",))
    )
    probe = aggregates.determinismProbe
    assert probe["ran"] is True
    assert probe["identical"] is False
    assert probe["differingTurns"] == [{"scriptId": "A-01", "turns": [2]}]


def test_determinism_probe_not_ran_when_fewer_probes_returned_than_declared() -> None:
    """The fail-safe default (plan `:2260-2261`): an unrun/incomplete probe never buys
    `identical: True` by omission."""
    trace = make_trace("A-01", tuple(make_turn() for _ in range(9)))
    scored = [(_clean_nine_turn_script("A-01"), trace, ())]

    _, aggregates = toolcalls.score_conversations(
        scored, [], pack=make_pack(h=4, determinism_probe_scripts=("A-01", "B-01"))
    )
    probe = aggregates.determinismProbe
    assert probe["ran"] is False
    assert probe["identical"] is False


# ==================================================================================================
# 2026-09-14 correction (review `small-model-benchmarking-s5.md` Finding 3, option (a)):
# `argument_correctness`'s per-argument failure decomposition wired into `_Tally`/`FunnelCounts`
# ==================================================================================================

_DECOMPOSITION_SCHEMAS: tuple[Mapping[str, Any], ...] = (
    {
        "type": "function",
        "function": {
            "name": "custom_tool",
            "parameters": {
                "type": "object",
                "properties": {
                    "productName": {"type": "string"},
                    "quantity": {"type": "integer"},
                    "maxPrice": {
                        "type": "number",
                        "boundaryRule": {"confusedWith": [5000], "inclusive": True},
                    },
                },
                "required": ["productName", "quantity", "maxPrice"],
            },
        },
    },
)


def _decomposition_script(script_id: str, *, args: Mapping[str, Any]) -> Conversation:
    """A 3-turn script whose middle turn requires `custom_tool`, called with `args` as the
    scoring oracle's expected arguments — flanked by two ordinary restraint turns so the fixture
    exercises `score_conversations`'s real assembly path, not just a single isolated turn."""
    turns = [
        _restraint_turn(0),
        Turn(
            seq=1,
            user="turn 1",
            expect={"toolRequired": True, "tool": "custom_tool", "args": dict(args)},
        ),
        _restraint_turn(2),
    ]
    return Conversation(scriptId=script_id, shape="A", replicate=1, turns=tuple(turns))


def test_score_conversations_wires_argument_correctness_decomposition_into_funnel_counts() -> None:
    """2026-09-14 correction (review Finding 3, option (a)): `argument_correctness`'s three
    per-argument failure tuples — previously computed and immediately discarded at
    `_score_one_conversation`'s `matching_calls` loop — now move `FunnelCounts.argsOmittedRequired`
    /`.argsWrongValue`/`.argsBoundaryUnit`. One dispatched call, three simultaneous argument
    defects: a missing required `productName` (omitted), a wrong, non-boundary `quantity`, and a
    boundary-confused `maxPrice` (the schema's own `confusedWith` value) — so `wrongValue` counts
    both `quantity` and `maxPrice` while `boundaryUnit` counts only `maxPrice`, exactly `-ml`
    §4.2(d)'s "`wrong_value: 12, of which boundary/unit: 7`" shape at a small scale."""
    expected_args = {"productName": "Pad", "quantity": 2, "maxPrice": 49.99}
    dispatched_args = {"quantity": 3, "maxPrice": 5000}  # productName omitted entirely
    script = _decomposition_script("A-01", args=expected_args)
    trace = make_trace(
        "A-01",
        (
            make_turn(),
            make_turn(dispatches=(make_dispatch("custom_tool", dispatched_args),)),
            make_turn(),
        ),
    )

    _, aggregates = toolcalls.score_conversations(
        [(script, trace, ())], [], pack=make_pack(h=1, tool_schemas=_DECOMPOSITION_SCHEMAS)
    )

    fc = aggregates.funnelCounts
    assert fc.argsOmittedRequired == 1
    assert fc.argsWrongValue == 2
    assert fc.argsBoundaryUnit == 1


def test_score_conversations_argument_decomposition_counters_stay_zero_when_all_correct() -> None:
    """The mirror case: a dispatched call whose every expected argument matches leaves all three
    new counters at their `FunnelCounts` default of 0 — the correction's own no-op path."""
    expected_args = {"productName": "Pad", "quantity": 2, "maxPrice": 49.99}
    script = _decomposition_script("A-01", args=expected_args)
    trace = make_trace(
        "A-01",
        (
            make_turn(),
            make_turn(dispatches=(make_dispatch("custom_tool", dict(expected_args)),)),
            make_turn(),
        ),
    )

    _, aggregates = toolcalls.score_conversations(
        [(script, trace, ())], [], pack=make_pack(h=1, tool_schemas=_DECOMPOSITION_SCHEMAS)
    )

    fc = aggregates.funnelCounts
    assert fc.argsOmittedRequired == 0
    assert fc.argsWrongValue == 0
    assert fc.argsBoundaryUnit == 0


_MULTI_CALL_SCHEMAS: tuple[Mapping[str, Any], ...] = (
    {
        "type": "function",
        "function": {
            "name": "custom_tool",
            "parameters": {
                "type": "object",
                "properties": {
                    "productName": {"type": "string"},
                    "quantity": {"type": "integer"},
                    "maxPrice": {
                        "type": "number",
                        "boundaryRule": {"confusedWith": [5000, 4999], "inclusive": True},
                    },
                },
                "required": ["productName", "quantity", "maxPrice"],
            },
        },
    },
)


def test_score_conversations_argument_decomposition_sums_across_every_matching_call() -> None:
    """`_Tally`'s three new accumulators must SUM every matching call's own contribution across
    the whole run (`+=`), never retain only the last one's (`=`) — a distinction the two tests
    above cannot make, since each drives exactly one qualifying call, where `+=`/`=` are
    indistinguishable.

    Four turns, four dispatched calls to the one required tool, each contributing a distinct,
    known defect count:
    - turn 0 omits `productName` only: `(omitted +1, wrong +0, boundary +0)`.
    - turn 1 has a wrong, non-boundary `quantity` only: `(+0, +1, +0)`.
    - turn 2's `maxPrice` hits the schema's FIRST `confusedWith` value: `(+0, +1, +1)`.
    - turn 3's `maxPrice` hits the schema's SECOND, DIFFERENT `confusedWith` value: `(+0, +1, +1)`.

    Summed: `argsOmittedRequired=1`, `argsWrongValue=3`, `argsBoundaryUnit=2`. Under a
    `tally.x = len(...)` overwrite instead of `+=`, the final `FunnelCounts` would read turn 3's
    own contribution alone (`0, 1, 1`) — every one of the three assertions below would then fail,
    not just one, since turns 0-2's contributions are each fully lost on every axis they set."""
    expected_args = {"productName": "Pad", "quantity": 2, "maxPrice": 49.99}
    turns = tuple(
        Turn(
            seq=i,
            user=f"turn {i}",
            expect={"toolRequired": True, "tool": "custom_tool", "args": dict(expected_args)},
        )
        for i in range(4)
    )
    script = Conversation(scriptId="A-01", shape="A", replicate=1, turns=turns)
    trace = make_trace(
        "A-01",
        (
            make_turn(
                dispatches=(make_dispatch("custom_tool", {"quantity": 2, "maxPrice": 49.99}),)
            ),
            make_turn(
                dispatches=(
                    make_dispatch(
                        "custom_tool", {"productName": "Pad", "quantity": 9, "maxPrice": 49.99}
                    ),
                )
            ),
            make_turn(
                dispatches=(
                    make_dispatch(
                        "custom_tool", {"productName": "Pad", "quantity": 2, "maxPrice": 5000}
                    ),
                )
            ),
            make_turn(
                dispatches=(
                    make_dispatch(
                        "custom_tool", {"productName": "Pad", "quantity": 2, "maxPrice": 4999}
                    ),
                )
            ),
        ),
    )

    _, aggregates = toolcalls.score_conversations(
        [(script, trace, ())], [], pack=make_pack(h=4, tool_schemas=_MULTI_CALL_SCHEMAS)
    )

    fc = aggregates.funnelCounts
    assert fc.argsOmittedRequired == 1
    assert fc.argsWrongValue == 3
    assert fc.argsBoundaryUnit == 2
