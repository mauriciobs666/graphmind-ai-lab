"""Unit tests for `falkorchat.querygen` (K-055 M6, implementation cluster 1).

Entirely offline: pure module, no FalkorDB/LLM dependency. Covers the plan's
§5 AC-3/AC-3a test strategy row (a), (d), (e) — hand-built `QueryRequest`s,
the exact `compile()` output for a filter case and an aggregate case, and the
security reviewer's own escape-attempt fixtures from
`docs/reviews/workflow-nl-query-generation-security.md` (MAJOR 1/MAJOR 2),
reproduced verbatim per the coordination brief.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from falkorchat.querygen import (
    CATALOG_SCHEMA,
    KNOWLEDGE_BASE_SCHEMA,
    CompiledQuery,
    QueryFilter,
    QueryMatch,
    QueryRequest,
    compile as qg_compile,
)


# ── Exact-output cases (§5 test strategy row (a)) ────────────────────────────


def test_compile_filter_case_produces_exact_cypher_and_params():
    request = QueryRequest(
        dataset="catalog",
        matches=[
            QueryMatch(
                var="p",
                label="Product",
                filters=[QueryFilter(property="category", op="=", value="Audio")],
            )
        ],
        returns=["p.name", "p.price"],
    )
    compiled = qg_compile(request, CATALOG_SCHEMA)

    assert isinstance(compiled, CompiledQuery)
    assert compiled.cypher == (
        "MATCH (p:Product) WHERE p.category = $p0 RETURN p.name, p.price LIMIT $limit"
    )
    assert compiled.params == {"p0": "Audio", "limit": 20}


def test_compile_aggregate_case_produces_exact_cypher_and_params():
    request = QueryRequest(
        dataset="catalog",
        matches=[QueryMatch(var="p", label="Product", filters=[])],
        returns=["count(p)"],
    )
    compiled = qg_compile(request, CATALOG_SCHEMA)

    assert compiled.cypher == "MATCH (p:Product) RETURN count(p) LIMIT $limit"
    assert compiled.params == {"limit": 20}


def test_compile_with_order_by_and_custom_limit():
    request = QueryRequest(
        dataset="catalog",
        matches=[QueryMatch(var="p", label="Product", filters=[])],
        returns=["p.name", "p.price"],
        order_by="p.price",
        order_dir="DESC",
        limit=5,
    )
    compiled = qg_compile(request, CATALOG_SCHEMA)

    assert compiled.cypher == (
        "MATCH (p:Product) RETURN p.name, p.price ORDER BY p.price DESC LIMIT $limit"
    )
    assert compiled.params == {"limit": 5}


def test_compile_value_and_limit_always_bound_never_formatted_into_text():
    # A value containing characters that would be dangerous if ever
    # string-formatted must appear ONLY in params, never in the cypher text.
    request = QueryRequest(
        dataset="catalog",
        matches=[
            QueryMatch(
                var="p",
                label="Product",
                filters=[
                    QueryFilter(
                        property="name", op="=", value="a\") DETACH DELETE (p) //"
                    )
                ],
            )
        ],
        returns=["p.name"],
    )
    compiled = qg_compile(request, CATALOG_SCHEMA)
    assert "DETACH" not in compiled.cypher
    assert "DELETE" not in compiled.cypher
    assert compiled.params["p0"] == 'a") DETACH DELETE (p) //'


def test_compile_second_dataset_knowledge_base_matches_ac2_example():
    # AC-2's own worked example (§3.5): "what type of entity is X" compiles to
    # a single-label filter against the knowledge-base dataset.
    request = QueryRequest(
        dataset="knowledge_base",
        matches=[
            QueryMatch(
                var="e",
                label="Entity",
                filters=[QueryFilter(property="nameNormalized", op="=", value="acme corp")],
            )
        ],
        returns=["e.type"],
    )
    compiled = qg_compile(request, KNOWLEDGE_BASE_SCHEMA)
    assert compiled.cypher == (
        "MATCH (e:Entity) WHERE e.nameNormalized = $p0 RETURN e.type LIMIT $limit"
    )
    assert compiled.params == {"p0": "acme corp", "limit": 20}


# ── `extra="forbid"` (MINOR 2) ────────────────────────────────────────────────


def test_query_request_rejects_unexpected_field():
    with pytest.raises(ValidationError):
        QueryRequest(
            dataset="catalog",
            matches=[QueryMatch(var="p", label="Product", filters=[])],
            returns=["p.name"],
            raw_cypher="MATCH (n) DETACH DELETE n",
        )


def test_query_match_rejects_unexpected_field():
    with pytest.raises(ValidationError):
        QueryMatch(var="p", label="Product", filters=[], extra_field="x")


def test_query_filter_rejects_unexpected_field():
    with pytest.raises(ValidationError):
        QueryFilter(property="name", op="=", value="x", raw="DROP")


# ── MAJOR 1 regression: returns/order_by decomposition must be fully anchored ──


@pytest.mark.parametrize(
    "bad_expr",
    [
        "count(v)) DETACH DELETE (v) //",
        "v.name//anything",
    ],
)
def test_returns_rejects_prefix_match_escape_attempts(bad_expr):
    with pytest.raises(ValidationError):
        QueryRequest(
            dataset="catalog",
            matches=[QueryMatch(var="v", label="Product", filters=[])],
            returns=[bad_expr],
        )


@pytest.mark.parametrize(
    "bad_expr",
    [
        "count(v)) DETACH DELETE (v) //",
        "v.name//anything",
    ],
)
def test_order_by_rejects_prefix_match_escape_attempts(bad_expr):
    # order_by only accepts a bare projection shape; either escape attempt
    # must be rejected whole, never truncated to a matching prefix.
    with pytest.raises(ValidationError):
        QueryRequest(
            dataset="catalog",
            matches=[QueryMatch(var="v", label="Product", filters=[])],
            returns=["v.name"],
            order_by=bad_expr,
        )


# ── MAJOR 2 regression: QueryMatch.var escape attempts ───────────────────────


@pytest.mark.parametrize(
    "bad_var",
    [
        "x) DETACH DELETE (x",
        "x WITH 1 AS y MATCH (m) DETACH DELETE m",
        "x//y",
    ],
)
def test_query_match_var_rejects_escape_attempts_at_pydantic_layer(bad_var):
    with pytest.raises(ValidationError):
        QueryMatch(var=bad_var, label="Product", filters=[])


def test_query_match_var_field_constraint_is_enforced_not_just_documented():
    # A plain valid-looking var passes; this pins down that the constraint is
    # actually wired (not merely a comment) by checking a borderline-invalid
    # one (uppercase) is rejected too.
    with pytest.raises(ValidationError):
        QueryMatch(var="P", label="Product", filters=[])


# ── compile()'s own independent re-check (defense-in-depth, bypassing Pydantic) ──


def test_compile_rejects_var_mismatch_from_hand_constructed_request():
    # Bypass Pydantic's own constructors/validators entirely, mirroring a
    # future refactor that builds a QueryMatch/QueryRequest by some path other
    # than the validated public constructors. `compile()` must still catch a
    # `returns` entry whose var disagrees with the declared match variable.
    match = QueryMatch.model_construct(var="p", label="Product", filters=[])
    request = QueryRequest.model_construct(
        dataset="catalog",
        matches=[match],
        returns=["q.name"],  # "q" was never declared — only "p" was
        order_by=None,
        order_dir="ASC",
        limit=20,
    )
    with pytest.raises(ValueError):
        qg_compile(request, CATALOG_SCHEMA)


def test_compile_rejects_order_by_var_mismatch_from_hand_constructed_request():
    match = QueryMatch.model_construct(var="p", label="Product", filters=[])
    request = QueryRequest.model_construct(
        dataset="catalog",
        matches=[match],
        returns=["p.name"],
        order_by="q.price",  # "q" was never declared
        order_dir="ASC",
        limit=20,
    )
    with pytest.raises(ValueError):
        qg_compile(request, CATALOG_SCHEMA)


def test_compile_rejects_more_than_one_match_from_hand_constructed_request():
    match_a = QueryMatch.model_construct(var="p", label="Product", filters=[])
    match_b = QueryMatch.model_construct(var="q", label="Product", filters=[])
    request = QueryRequest.model_construct(
        dataset="catalog",
        matches=[match_a, match_b],
        returns=["p.name"],
        order_by=None,
        order_dir="ASC",
        limit=20,
    )
    with pytest.raises(ValueError):
        qg_compile(request, CATALOG_SCHEMA)


def test_compile_rejects_unregistered_label():
    match = QueryMatch.model_construct(var="p", label="NotARealLabel", filters=[])
    request = QueryRequest.model_construct(
        dataset="catalog",
        matches=[match],
        returns=["p.name"],
        order_by=None,
        order_dir="ASC",
        limit=20,
    )
    with pytest.raises(ValueError):
        qg_compile(request, CATALOG_SCHEMA)


def test_compile_rejects_unregistered_label_for_bare_aggregate_with_no_property():
    # `returns=["count(p)"]` has no `.property` to decompose, so the
    # property-allowlist check never fires for this shape — the
    # label-registration guard is the SOLE line of defense here. Pins that
    # guard specifically, since `test_compile_rejects_unregistered_label`
    # above (`returns=["p.name"]`) happens to also be caught by the
    # property-allowlist check and so doesn't prove the label check alone is
    # load-bearing.
    match = QueryMatch.model_construct(var="p", label="NotARealLabel", filters=[])
    request = QueryRequest.model_construct(
        dataset="catalog",
        matches=[match],
        returns=["count(p)"],
        order_by=None,
        order_dir="ASC",
        limit=20,
    )
    with pytest.raises(ValueError):
        qg_compile(request, CATALOG_SCHEMA)


def test_compile_rejects_unregistered_property_on_filter():
    match = QueryMatch.model_construct(
        var="p",
        label="Product",
        filters=[QueryFilter.model_construct(property="secretInternalField", op="=", value="x")],
    )
    request = QueryRequest.model_construct(
        dataset="catalog",
        matches=[match],
        returns=["p.name"],
        order_by=None,
        order_dir="ASC",
        limit=20,
    )
    with pytest.raises(ValueError):
        qg_compile(request, CATALOG_SCHEMA)


def test_compile_rejects_unregistered_property_in_returns():
    match = QueryMatch.model_construct(var="p", label="Product", filters=[])
    request = QueryRequest.model_construct(
        dataset="catalog",
        matches=[match],
        returns=["p.secretInternalField"],
        order_by=None,
        order_dir="ASC",
        limit=20,
    )
    with pytest.raises(ValueError):
        qg_compile(request, CATALOG_SCHEMA)


def test_compile_rejects_unregistered_aggregate_function():
    match = QueryMatch.model_construct(var="p", label="Product", filters=[])
    request = QueryRequest.model_construct(
        dataset="catalog",
        matches=[match],
        returns=["p.name"],
        order_by=None,
        order_dir="ASC",
        limit=20,
    )
    # Directly probe the aggregate-func allowlist by hand-crafting a schema
    # whose `aggregates` set does not include "count" — a `returns` entry
    # using it must still be rejected even though the regex shape is valid.
    from falkorchat.querygen import DatasetSchema

    narrow_schema = DatasetSchema(
        graph_key="reference",
        labels={"Product": frozenset({"name"})},
        aggregates=frozenset(),
    )
    request2 = QueryRequest.model_construct(
        dataset="catalog",
        matches=[match],
        returns=["count(p)"],
        order_by=None,
        order_dir="ASC",
        limit=20,
    )
    with pytest.raises(ValueError):
        qg_compile(request2, narrow_schema)


def test_compile_never_emits_forbidden_tokens_for_any_valid_request():
    forbidden = ("CREATE", "MERGE", "SET", "DELETE", "REMOVE", "DROP", "FOREACH", "CALL")
    request = QueryRequest(
        dataset="catalog",
        matches=[
            QueryMatch(
                var="p",
                label="Product",
                filters=[QueryFilter(property="price", op=">", value=10.0)],
            )
        ],
        returns=["p.name", "count(p)"],
        order_by="p.price",
    )
    compiled = qg_compile(request, CATALOG_SCHEMA)
    for token in forbidden:
        assert token not in compiled.cypher.upper()
