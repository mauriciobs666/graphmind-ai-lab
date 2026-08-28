"""Unit tests for `pricing.compute_line_total` — pure, no I/O, no fixtures needed.

Covers the plan §3.1/§5 (AC-1/AC-2/AC-4) coverage list: empty, one line, many
lines, a malformed line skipped not raised, identical-inputs-identical-output.
"""

from __future__ import annotations

from falkorchat.pricing import compute_line_total


def test_empty_list_totals_zero():
    assert compute_line_total([]) == 0.0


def test_one_line():
    assert compute_line_total([{"price": 2.5, "quantity": 3}]) == 7.5


def test_many_lines_sum_across_items():
    items = [
        {"price": 10.0, "quantity": 2},
        {"price": 5.0, "quantity": 3},
        {"price": 1.5, "quantity": 4},
    ]
    # 20 + 15 + 6 = 41
    assert compute_line_total(items) == 41.0


def test_malformed_line_missing_price_is_skipped_not_raised():
    items = [{"quantity": 2}, {"price": 10.0, "quantity": 2}]
    assert compute_line_total(items) == 20.0


def test_malformed_line_missing_quantity_is_skipped_not_raised():
    items = [{"price": 10.0}, {"price": 10.0, "quantity": 2}]
    assert compute_line_total(items) == 20.0


def test_malformed_line_missing_both_keys_is_skipped_not_raised():
    items = [{}, {"price": 10.0, "quantity": 2}]
    assert compute_line_total(items) == 20.0


def test_malformed_line_non_numeric_price_is_skipped_not_raised():
    items = [
        {"price": "ten", "quantity": 2},
        {"price": 10.0, "quantity": 2},
    ]
    assert compute_line_total(items) == 20.0


def test_malformed_line_non_numeric_quantity_is_skipped_not_raised():
    items = [
        {"price": 10.0, "quantity": "two"},
        {"price": 10.0, "quantity": 2},
    ]
    assert compute_line_total(items) == 20.0


def test_malformed_line_none_values_are_skipped_not_raised():
    items = [
        {"price": None, "quantity": 2},
        {"price": 10.0, "quantity": None},
        {"price": 10.0, "quantity": 2},
    ]
    assert compute_line_total(items) == 20.0


def test_bool_values_are_not_treated_as_numeric():
    # bool is an int subclass in Python — must not silently pass as a price/qty.
    items = [
        {"price": True, "quantity": 2},
        {"price": 10.0, "quantity": False},
        {"price": 10.0, "quantity": 2},
    ]
    assert compute_line_total(items) == 20.0


def test_all_lines_malformed_totals_zero_never_raises():
    items = [{"price": "x", "quantity": 1}, {"price": 1, "quantity": "y"}, {}]
    assert compute_line_total(items) == 0.0


def test_identical_inputs_produce_identical_output_ac4():
    items = [
        {"price": 3.33, "quantity": 3},
        {"price": 1.11, "quantity": 7},
    ]
    first = compute_line_total(items)
    second = compute_line_total(items)
    assert first == second


def test_int_price_and_quantity_are_accepted():
    assert compute_line_total([{"price": 4, "quantity": 5}]) == 20.0


def test_does_not_mutate_input_list():
    items = [{"price": 1.0, "quantity": 1}]
    snapshot = list(items)
    compute_line_total(items)
    assert items == snapshot
