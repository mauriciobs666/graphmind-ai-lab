"""Unit tests for `nlq_scoring` (K-055 M6, unit U29c, `docs/plans/workflow-nl-
query-generation-ml.md` §3 — the Layer 1/Layer 2 comparison rules).

**Genuinely network/DB-free.** Every test feeds `score_pair`/`layer2_contains`/
`wilson_interval` hand-built golden rows and hand-built raw tool results (the
`{"items": [...]}` shape `QueryGraphDataTool.run()` returns, per
`server/tests/test_tools.py`'s own fixtures) — no FalkorDB, no LLM. Runs in the
default offline suite.
"""

from __future__ import annotations

import math

import pytest

from nlq_scoring import layer2_contains, score_pair, wilson_interval


def _row(*, shape: str, expected: dict, question: str = "q?") -> dict:
    return {
        "id": "nlq-test",
        "dataset": "catalog",
        "question": question,
        "shape": shape,
        "expected": expected,
        "rationale": "test fixture",
    }


# ── scalar ────────────────────────────────────────────────────────────────────


def test_scalar_string_exact_match_after_case_and_whitespace_folding():
    row = _row(shape="single-fact", expected={"type": "scalar", "value": "Storage"})
    result = {"items": [{"p.category": "  storage  "}]}
    outcome = score_pair(row, result)
    assert outcome.correct is True


def test_scalar_string_mismatch_is_incorrect():
    row = _row(shape="single-fact", expected={"type": "scalar", "value": "Storage"})
    result = {"items": [{"p.category": "Audio"}]}
    assert score_pair(row, result).correct is False


def test_scalar_number_within_epsilon_is_correct():
    row = _row(shape="single-fact", expected={"type": "scalar", "value": 24.99})
    result = {"items": [{"p.price": 24.995}]}  # 0.005 off — within the 0.01 epsilon
    assert score_pair(row, result).correct is True


def test_scalar_number_at_epsilon_boundary_is_correct():
    row = _row(shape="single-fact", expected={"type": "scalar", "value": 24.99})
    result = {"items": [{"p.price": 25.00}]}  # exactly 0.01 off — boundary, inclusive
    assert score_pair(row, result).correct is True


def test_scalar_number_just_outside_epsilon_is_incorrect():
    row = _row(shape="single-fact", expected={"type": "scalar", "value": 24.99})
    result = {"items": [{"p.price": 25.01}]}  # 0.02 off
    assert score_pair(row, result).correct is False


def test_scalar_never_string_equal_on_a_formatted_price():
    """The ml note's own explicit caution: a numeric expected value compared
    against a string-typed actual value must not pass via bare `==`/string
    comparison — it must be treated as a type mismatch (incorrect), not
    silently stringified and matched."""
    row = _row(shape="single-fact", expected={"type": "scalar", "value": 24.99})
    result = {"items": [{"p.price": "$24.99"}]}
    assert score_pair(row, result).correct is False


def test_scalar_extraction_fails_on_zero_rows():
    row = _row(shape="single-fact", expected={"type": "scalar", "value": "Storage"})
    result = {"items": []}
    assert score_pair(row, result).correct is False


def test_scalar_extraction_fails_on_multiple_rows():
    row = _row(shape="single-fact", expected={"type": "scalar", "value": "Storage"})
    result = {"items": [{"p.category": "Storage"}, {"p.category": "Storage"}]}
    assert score_pair(row, result).correct is False


def test_scalar_extraction_fails_on_multiple_columns():
    row = _row(shape="single-fact", expected={"type": "scalar", "value": "Storage"})
    result = {"items": [{"p.category": "Storage", "p.price": 1.0}]}
    assert score_pair(row, result).correct is False


# ── set (filter-list / aggregation-list) ─────────────────────────────────────


def test_set_unordered_match_regardless_of_result_order():
    row = _row(
        shape="filter-list",
        expected={"type": "set", "values": ["Alpha", "Beta", "Gamma"]},
    )
    result = {"items": [{"p.name": "Gamma"}, {"p.name": "Alpha"}, {"p.name": "Beta"}]}
    assert score_pair(row, result).correct is True


def test_set_match_is_case_and_whitespace_folded():
    row = _row(shape="filter-list", expected={"type": "set", "values": ["Alpha"]})
    result = {"items": [{"p.name": "  ALPHA "}]}
    assert score_pair(row, result).correct is True


def test_set_missing_a_member_is_incorrect():
    row = _row(
        shape="filter-list",
        expected={"type": "set", "values": ["Alpha", "Beta"]},
    )
    result = {"items": [{"p.name": "Alpha"}]}
    assert score_pair(row, result).correct is False


def test_set_with_an_extra_member_is_incorrect():
    """Layer 1 is exact-set-match, not containment — an over-broad result (e.g.
    the mechanism returned an extra unrequested column/value) must not score
    correct just because it happens to contain every expected member too."""
    row = _row(shape="filter-list", expected={"type": "set", "values": ["Alpha"]})
    result = {"items": [{"p.name": "Alpha"}, {"p.name": "Beta"}]}
    assert score_pair(row, result).correct is False


def test_set_singleton_is_still_a_set_comparison_not_a_scalar_one():
    row = _row(shape="filter-list", expected={"type": "set", "values": ["Alpha"]})
    result = {"items": [{"p.name": "Alpha"}]}
    assert score_pair(row, result).correct is True


# ── not_found ─────────────────────────────────────────────────────────────────


def test_not_found_correct_when_result_is_genuinely_empty():
    row = _row(shape="not-found", expected={"type": "not_found"})
    result = {"items": [], "finding": "no matching data found"}
    assert score_pair(row, result).correct is True


def test_not_found_incorrect_when_mechanism_fabricates_a_value():
    row = _row(shape="not-found", expected={"type": "not_found"})
    result = {"items": [{"p.category": "Storage"}]}
    assert score_pair(row, result).correct is False


# ── conflicting-facts ─────────────────────────────────────────────────────────


def test_conflicting_facts_correct_only_when_all_expected_values_present():
    row = _row(
        shape="conflicting-facts",
        expected={"type": "set", "values": ["62", "140 employees"]},
    )
    result = {"items": [{"o.value": "62"}, {"o.value": "140 employees"}]}
    assert score_pair(row, result).correct is True


def test_conflicting_facts_incorrect_when_only_one_value_present():
    """Silently picking one of two conflicting facts is a wrong answer for this
    category, never partial credit (ml note §3)."""
    row = _row(
        shape="conflicting-facts",
        expected={"type": "set", "values": ["62", "140 employees"]},
    )
    result = {"items": [{"o.value": "62"}]}
    assert score_pair(row, result).correct is False


def test_conflicting_facts_incorrect_when_result_is_empty():
    row = _row(
        shape="conflicting-facts",
        expected={"type": "set", "values": ["62", "140 employees"]},
    )
    result = {"items": []}
    assert score_pair(row, result).correct is False


def test_conflicting_facts_tolerates_extra_values_present_unlike_plain_set_shape():
    """Containment, not exact-match, for this one shape — per the ml note's
    'correct only if all values in the expected set are present,' which does not
    forbid the mechanism from also returning something extra."""
    row = _row(
        shape="conflicting-facts",
        expected={"type": "set", "values": ["62", "140 employees"]},
    )
    result = {"items": [{"o.value": "62"}, {"o.value": "140 employees"}, {"o.value": "99"}]}
    assert score_pair(row, result).correct is True


# ── unknown expected type ────────────────────────────────────────────────────


def test_unknown_expected_type_raises():
    row = _row(shape="single-fact", expected={"type": "bogus"})
    with pytest.raises(ValueError):
        score_pair(row, {"items": []})


# ── wilson_interval ───────────────────────────────────────────────────────────


def test_wilson_interval_rejects_zero_n():
    with pytest.raises(ValueError):
        wilson_interval(0, 0)


def test_wilson_interval_all_successes_upper_bound_below_one():
    lo, hi = wilson_interval(39, 39)
    assert hi <= 1.0
    assert 0.85 < lo < 1.0  # a real interval, not a degenerate [1,1]


def test_wilson_interval_all_failures_lower_bound_is_zero():
    lo, hi = wilson_interval(0, 39)
    assert lo == 0.0
    assert hi < 0.15


def test_wilson_interval_is_symmetric_around_half_at_p_half():
    lo, hi = wilson_interval(20, 40)
    center = (lo + hi) / 2
    assert math.isclose(center, 0.5, abs_tol=0.03)
    assert lo < 0.5 < hi


def test_wilson_interval_widens_at_smaller_n_for_the_same_proportion():
    lo_big, hi_big = wilson_interval(20, 40)
    lo_small, hi_small = wilson_interval(5, 10)
    assert (hi_small - lo_small) > (hi_big - lo_big)


# ── layer2_contains (rendered-answer containment check) ──────────────────────


def test_layer2_scalar_value_contained_in_rendered_sentence():
    row = _row(shape="single-fact", expected={"type": "scalar", "value": 24.99})
    assert layer2_contains(row, "The Wireless Charging Pad costs $24.99.") is True


def test_layer2_scalar_value_absent_from_rendered_sentence():
    row = _row(shape="single-fact", expected={"type": "scalar", "value": 24.99})
    assert layer2_contains(row, "I couldn't find that product.") is False


def test_layer2_set_requires_every_value_present():
    row = _row(
        shape="filter-list",
        expected={"type": "set", "values": ["Alpha", "Beta"]},
    )
    assert layer2_contains(row, "We carry Alpha and Beta in that category.") is True
    assert layer2_contains(row, "We carry Alpha in that category.") is False


def test_layer2_not_found_recognizes_an_abstention_phrase():
    row = _row(shape="not-found", expected={"type": "not_found"})
    assert layer2_contains(row, "Sorry, I couldn't find that product.") is True


def test_layer2_not_found_incorrect_when_answer_states_a_value():
    row = _row(shape="not-found", expected={"type": "not_found"})
    assert layer2_contains(row, "That product costs $19.99.") is False


def test_layer2_conflicting_facts_requires_all_values_present():
    row = _row(
        shape="conflicting-facts",
        expected={"type": "set", "values": ["62", "140 employees"]},
    )
    assert layer2_contains(
        row, "Sources conflict: one says 62 employees, another says 140 employees."
    ) is True
    assert layer2_contains(row, "Marlowe Robotics has 62 employees.") is False
