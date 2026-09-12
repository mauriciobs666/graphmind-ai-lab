"""`packs/nlq-structured-query/tools/exec.py` — Layer A/B validation + in-process execution
against `tables.json` (S4 spec §5.2.2, §7 Step 3 Pass A).

Pure, no pack loader: this module is imported directly off its file path (mirroring
`Pack.load_tool_module`'s own `importlib.util.spec_from_file_location` mechanism, S4 spec §7 Step
3's own "Pass A ... pure, no pack loader" framing) so these tests do not need `pack.json` to exist
yet.

Order, red -> green, matches the spec's own step sequence (§7 Step 3 Pass A):

1. Layer A (`validate_structure`) — one case per rule named in §2.5, each raising
   `MalformedSpecError`; one full valid spec raising nothing.
2. Layer B (`validate_against_schema`) — an unregistered label/property, a duplicate `returns`
   entry, a numeric-typed property given a non-parsing string value, a numeric-typed property
   given a parsing string value (accepted and coerced), a `*Normalized` property given
   un-normalized text (matched against a pre-normalized stored row) — each `SchemaViolationError`
   except the accepted/matched cases.
3. `compile_and_execute` against a small synthetic 4-row `tables.json` fixture: a filter query, a
   `count`/`avg`/`min`/`max` aggregate each, an `order_by` + `limit` superlative query, a query
   whose `returns` includes a `*Normalized` property (never itself normalized on output, only on
   the filter side).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

_MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "packs" / "nlq-structured-query" / "tools" / "exec.py"
)


def _load_exec_module():
    spec = importlib.util.spec_from_file_location("nlq_tools_exec_under_test", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


exec_mod = _load_exec_module()

# --------------------------------------------------------------------------------------------
# A small, synthetic 4-row `tables.json`-shaped fixture and matching `schema.json` block
# --------------------------------------------------------------------------------------------

_SCHEMA = {
    "labels": {
        "Product": {
            "name": "str",
            "nameNormalized": "str",
            "category": "str",
            "price": "float",
        },
    }
}

_TABLES = {
    "Product": [
        {"name": "Widget A", "nameNormalized": "widget a", "category": "Tools", "price": 10.0},
        {"name": "Widget B", "nameNormalized": "widget b", "category": "Tools", "price": 20.0},
        {"name": "Gizmo C", "nameNormalized": "gizmo c", "category": "Gadgets", "price": 30.0},
        {"name": "Gizmo D", "nameNormalized": "gizmo d", "category": "Gadgets", "price": 5.0},
    ]
}


def _valid_spec(**overrides: Any) -> dict[str, Any]:
    spec = {
        "matches": [
            {
                "var": "p",
                "label": "Product",
                "filters": [{"property": "category", "op": "=", "value": "Tools"}],
            }
        ],
        "returns": ["p.name"],
    }
    spec.update(overrides)
    return spec


# ==================================================================================================
# 1. Layer A — `validate_structure`
# ==================================================================================================


class TestValidateStructureRejections:
    def test_valid_spec_raises_nothing(self):
        exec_mod.validate_structure(_valid_spec())  # must not raise

    def test_valid_spec_with_every_optional_field_raises_nothing(self):
        exec_mod.validate_structure(
            _valid_spec(order_by="p.price", order_dir="DESC", limit=5)
        )

    def test_top_level_unknown_key_is_rejected(self):
        spec = _valid_spec()
        spec["bogus"] = 1
        with pytest.raises(exec_mod.MalformedSpecError):
            exec_mod.validate_structure(spec)

    def test_match_level_unknown_key_is_rejected(self):
        spec = _valid_spec()
        spec["matches"][0]["bogus"] = 1
        with pytest.raises(exec_mod.MalformedSpecError):
            exec_mod.validate_structure(spec)

    def test_filter_level_unknown_key_is_rejected(self):
        spec = _valid_spec()
        spec["matches"][0]["filters"][0]["bogus"] = 1
        with pytest.raises(exec_mod.MalformedSpecError):
            exec_mod.validate_structure(spec)

    def test_matches_length_zero_is_rejected(self):
        spec = _valid_spec(matches=[])
        with pytest.raises(exec_mod.MalformedSpecError):
            exec_mod.validate_structure(spec)

    def test_matches_length_two_is_rejected(self):
        spec = _valid_spec()
        spec["matches"] = spec["matches"] * 2
        with pytest.raises(exec_mod.MalformedSpecError):
            exec_mod.validate_structure(spec)

    def test_bad_var_regex_is_rejected(self):
        spec = _valid_spec()
        spec["matches"][0]["var"] = "P"  # uppercase, not allowed
        with pytest.raises(exec_mod.MalformedSpecError):
            exec_mod.validate_structure(spec)

    def test_bad_property_regex_on_filter_is_rejected(self):
        spec = _valid_spec()
        spec["matches"][0]["filters"][0]["property"] = "9bad"
        with pytest.raises(exec_mod.MalformedSpecError):
            exec_mod.validate_structure(spec)

    def test_op_outside_the_six_member_whitelist_is_rejected(self):
        spec = _valid_spec()
        spec["matches"][0]["filters"][0]["op"] = "contains"
        with pytest.raises(exec_mod.MalformedSpecError):
            exec_mod.validate_structure(spec)

    def test_filters_at_length_five_is_rejected(self):
        spec = _valid_spec()
        spec["matches"][0]["filters"] = [
            {"property": "category", "op": "=", "value": "Tools"} for _ in range(5)
        ]
        with pytest.raises(exec_mod.MalformedSpecError):
            exec_mod.validate_structure(spec)

    def test_filters_at_length_four_is_accepted(self):
        spec = _valid_spec()
        spec["matches"][0]["filters"] = [
            {"property": "category", "op": "=", "value": "Tools"} for _ in range(4)
        ]
        exec_mod.validate_structure(spec)  # must not raise

    def test_returns_at_length_zero_is_rejected(self):
        spec = _valid_spec(returns=[])
        with pytest.raises(exec_mod.MalformedSpecError):
            exec_mod.validate_structure(spec)

    def test_returns_at_length_seven_is_rejected(self):
        spec = _valid_spec(returns=["p.name"] * 7)
        with pytest.raises(exec_mod.MalformedSpecError):
            exec_mod.validate_structure(spec)

    def test_returns_at_length_six_is_accepted(self):
        exec_mod.validate_structure(_valid_spec(returns=["p.name"] * 6))  # must not raise

    def test_returns_entry_matching_neither_projection_nor_aggregate_is_rejected(self):
        with pytest.raises(exec_mod.MalformedSpecError):
            exec_mod.validate_structure(_valid_spec(returns=["not a valid expr!"]))

    def test_returns_aggregate_shape_is_accepted_structurally(self):
        exec_mod.validate_structure(_valid_spec(returns=["count(p)"]))  # must not raise
        exec_mod.validate_structure(_valid_spec(returns=["avg(p.price)"]))  # must not raise

    def test_non_projection_order_by_is_rejected(self):
        with pytest.raises(exec_mod.MalformedSpecError):
            exec_mod.validate_structure(_valid_spec(order_by="count(p)"))

    def test_order_dir_outside_asc_desc_is_rejected(self):
        with pytest.raises(exec_mod.MalformedSpecError):
            exec_mod.validate_structure(_valid_spec(order_dir="UP"))

    def test_limit_at_zero_is_rejected(self):
        with pytest.raises(exec_mod.MalformedSpecError):
            exec_mod.validate_structure(_valid_spec(limit=0))

    def test_limit_at_fifty_one_is_rejected(self):
        with pytest.raises(exec_mod.MalformedSpecError):
            exec_mod.validate_structure(_valid_spec(limit=51))

    def test_limit_at_one_and_fifty_are_accepted(self):
        exec_mod.validate_structure(_valid_spec(limit=1))
        exec_mod.validate_structure(_valid_spec(limit=50))

    def test_missing_matches_is_rejected(self):
        spec = _valid_spec()
        del spec["matches"]
        with pytest.raises(exec_mod.MalformedSpecError):
            exec_mod.validate_structure(spec)

    def test_missing_returns_is_rejected(self):
        spec = _valid_spec()
        del spec["returns"]
        with pytest.raises(exec_mod.MalformedSpecError):
            exec_mod.validate_structure(spec)


# ==================================================================================================
# 2. Layer B — `validate_against_schema`
# ==================================================================================================


class TestValidateAgainstSchema:
    def test_valid_spec_raises_nothing(self):
        exec_mod.validate_against_schema(_valid_spec(), _SCHEMA)

    def test_unregistered_label_is_rejected(self):
        spec = _valid_spec()
        spec["matches"][0]["label"] = "Nonexistent"
        with pytest.raises(exec_mod.SchemaViolationError):
            exec_mod.validate_against_schema(spec, _SCHEMA)

    def test_unregistered_filter_property_is_rejected(self):
        spec = _valid_spec()
        spec["matches"][0]["filters"][0]["property"] = "nope"
        with pytest.raises(exec_mod.SchemaViolationError):
            exec_mod.validate_against_schema(spec, _SCHEMA)

    def test_unregistered_return_property_is_rejected(self):
        spec = _valid_spec(returns=["p.nope"])
        with pytest.raises(exec_mod.SchemaViolationError):
            exec_mod.validate_against_schema(spec, _SCHEMA)

    def test_unregistered_order_by_property_is_rejected(self):
        spec = _valid_spec(order_by="p.nope")
        with pytest.raises(exec_mod.SchemaViolationError):
            exec_mod.validate_against_schema(spec, _SCHEMA)

    def test_duplicate_returns_entry_is_rejected(self):
        spec = _valid_spec(returns=["p.name", "p.name"])
        with pytest.raises(exec_mod.SchemaViolationError):
            exec_mod.validate_against_schema(spec, _SCHEMA)

    def test_non_parsing_numeric_filter_value_is_rejected(self):
        spec = _valid_spec()
        spec["matches"][0]["filters"][0] = {"property": "price", "op": "<", "value": "fifty"}
        with pytest.raises(exec_mod.SchemaViolationError):
            exec_mod.validate_against_schema(spec, _SCHEMA)

    def test_parsing_numeric_filter_value_is_accepted(self):
        spec = _valid_spec()
        spec["matches"][0]["filters"][0] = {"property": "price", "op": "<", "value": "50"}
        exec_mod.validate_against_schema(spec, _SCHEMA)  # must not raise

    def test_normalized_property_given_unnormalized_text_is_accepted(self):
        """`*Normalized` filter values are case-folded/whitespace-collapsed before matching, so
        un-normalized question text is legal input at this layer — the mismatch (if any) is an
        execution-time question, not a schema violation."""
        spec = _valid_spec()
        spec["matches"][0]["filters"][0] = {
            "property": "nameNormalized", "op": "=", "value": "  Widget   A  ",
        }
        exec_mod.validate_against_schema(spec, _SCHEMA)  # must not raise


# ==================================================================================================
# 3. `compile_and_execute` against the synthetic 4-row fixture
# ==================================================================================================


class TestCompileAndExecute:
    def test_filter_query_returns_matching_rows_projected(self):
        spec = _valid_spec()  # category = Tools -> Widget A, Widget B
        result = exec_mod.compile_and_execute(spec, tables=_TABLES, schema=_SCHEMA)
        assert result == {"items": [{"p.name": "Widget A"}, {"p.name": "Widget B"}]}

    def test_count_aggregate(self):
        spec = _valid_spec(returns=["count(p)"])
        result = exec_mod.compile_and_execute(spec, tables=_TABLES, schema=_SCHEMA)
        assert result == {"items": [{"count(p)": 2}]}

    def test_avg_aggregate(self):
        spec = _valid_spec(returns=["avg(p.price)"])
        result = exec_mod.compile_and_execute(spec, tables=_TABLES, schema=_SCHEMA)
        assert result == {"items": [{"avg(p.price)": 15.0}]}  # (10 + 20) / 2

    def test_min_aggregate(self):
        spec = _valid_spec(returns=["min(p.price)"])
        result = exec_mod.compile_and_execute(spec, tables=_TABLES, schema=_SCHEMA)
        assert result == {"items": [{"min(p.price)": 10.0}]}

    def test_max_aggregate(self):
        spec = _valid_spec(returns=["max(p.price)"])
        result = exec_mod.compile_and_execute(spec, tables=_TABLES, schema=_SCHEMA)
        assert result == {"items": [{"max(p.price)": 20.0}]}

    def test_order_by_and_limit_superlative_query(self):
        """Which product is the cheapest? — no filter, order_by price ASC, limit 1."""
        spec = {
            "matches": [{"var": "p", "label": "Product", "filters": []}],
            "returns": ["p.name"],
            "order_by": "p.price",
            "order_dir": "ASC",
            "limit": 1,
        }
        result = exec_mod.compile_and_execute(spec, tables=_TABLES, schema=_SCHEMA)
        assert result == {"items": [{"p.name": "Gizmo D"}]}  # price 5.0, cheapest

    def test_order_by_desc_and_limit(self):
        spec = {
            "matches": [{"var": "p", "label": "Product", "filters": []}],
            "returns": ["p.name"],
            "order_by": "p.price",
            "order_dir": "DESC",
            "limit": 1,
        }
        result = exec_mod.compile_and_execute(spec, tables=_TABLES, schema=_SCHEMA)
        assert result == {"items": [{"p.name": "Gizmo C"}]}  # price 30.0, most expensive

    def test_returns_a_normalized_property_never_normalizes_the_output_value(self):
        """`*Normalized` properties are normalized on the FILTER side only (querygen.compile's own
        asymmetry, S4 spec §7 Step 3 Pass A item 3) — a `returns` entry projecting one prints the
        row's own STORED value verbatim, never re-normalized a second time (it already is)."""
        spec = {
            "matches": [
                {
                    "var": "p", "label": "Product",
                    "filters": [{"property": "name", "op": "=", "value": "Widget A"}],
                }
            ],
            "returns": ["p.nameNormalized"],
        }
        result = exec_mod.compile_and_execute(spec, tables=_TABLES, schema=_SCHEMA)
        assert result == {"items": [{"p.nameNormalized": "widget a"}]}

    def test_no_matching_rows_returns_empty_items(self):
        spec = _valid_spec()
        spec["matches"][0]["filters"][0]["value"] = "Nonexistent Category"
        result = exec_mod.compile_and_execute(spec, tables=_TABLES, schema=_SCHEMA)
        assert result == {"items": []}

    def test_a_layer_a_violation_raises_malformed_spec_error_and_never_executes(self):
        spec = _valid_spec()
        spec["matches"][0]["filters"][0]["op"] = "contains"
        with pytest.raises(exec_mod.MalformedSpecError):
            exec_mod.compile_and_execute(spec, tables=_TABLES, schema=_SCHEMA)

    def test_a_layer_b_violation_raises_schema_violation_error_and_never_executes(self):
        spec = _valid_spec()
        spec["matches"][0]["label"] = "Nonexistent"
        with pytest.raises(exec_mod.SchemaViolationError):
            exec_mod.compile_and_execute(spec, tables=_TABLES, schema=_SCHEMA)

    def test_an_unnormalized_filter_value_against_normalized_property_matches_the_stored_row(self):
        spec = {
            "matches": [
                {
                    "var": "p", "label": "Product",
                    "filters": [
                        {"property": "nameNormalized", "op": "=", "value": "  Widget   A  "}
                    ],
                }
            ],
            "returns": ["p.name"],
        }
        result = exec_mod.compile_and_execute(spec, tables=_TABLES, schema=_SCHEMA)
        assert result == {"items": [{"p.name": "Widget A"}]}
