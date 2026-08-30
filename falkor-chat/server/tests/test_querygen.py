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
        "MATCH (p:Product) WHERE p.category = $p0 "
        "RETURN DISTINCT p.name, p.price LIMIT $limit"
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
        "MATCH (p:Product) RETURN DISTINCT p.name, p.price ORDER BY p.price DESC LIMIT $limit"
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
        "MATCH (e:Entity) WHERE e.nameNormalized = $p0 RETURN DISTINCT e.type LIMIT $limit"
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
        labels={"Product": {"name": str}},
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


# ── U29f fix A: numeric filter values coerced by declared property type ──────


def test_compile_coerces_numeric_string_filter_value_against_numeric_property():
    # RCA category A (nlq-08 shape): the model serialized a numeric filter
    # value as a JSON string instead of a bare number.
    request = QueryRequest(
        dataset="catalog",
        matches=[
            QueryMatch(
                var="p", label="Product",
                filters=[QueryFilter(property="price", op="<", value="50")],
            )
        ],
        returns=["p.name"],
    )
    compiled = qg_compile(request, CATALOG_SCHEMA)
    assert compiled.params["p0"] == 50.0
    assert isinstance(compiled.params["p0"], float)


def test_compile_leaves_already_numeric_filter_value_unchanged():
    request = QueryRequest(
        dataset="catalog",
        matches=[
            QueryMatch(
                var="p", label="Product",
                filters=[QueryFilter(property="price", op="<", value=50)],
            )
        ],
        returns=["p.name"],
    )
    compiled = qg_compile(request, CATALOG_SCHEMA)
    assert compiled.params["p0"] == 50


def test_compile_rejects_non_numeric_string_against_numeric_property():
    request = QueryRequest(
        dataset="catalog",
        matches=[
            QueryMatch(
                var="p", label="Product",
                filters=[QueryFilter(property="price", op="=", value="cheap")],
            )
        ],
        returns=["p.name"],
    )
    with pytest.raises(ValueError):
        qg_compile(request, CATALOG_SCHEMA)


def test_compile_does_not_coerce_string_value_against_string_property():
    # A numeric-looking string against a plain string property (e.g. a
    # product literally named "50") must NOT be coerced — coercion is scoped
    # strictly to properties whose declared type is numeric.
    request = QueryRequest(
        dataset="catalog",
        matches=[
            QueryMatch(
                var="p", label="Product",
                filters=[QueryFilter(property="name", op="=", value="50")],
            )
        ],
        returns=["p.price"],
    )
    compiled = qg_compile(request, CATALOG_SCHEMA)
    assert compiled.params["p0"] == "50"


# ── U29f fix B: normalize filter values against *Normalized properties ──────


def test_compile_normalizes_mixed_case_value_against_normalized_property():
    # RCA category B (nlq-02/04 shape): the model filtered on
    # `nameNormalized` with the verbatim (un-normalized) question text.
    request = QueryRequest(
        dataset="catalog",
        matches=[
            QueryMatch(
                var="p", label="Product",
                filters=[
                    QueryFilter(property="nameNormalized", op="=", value="Portable SSD 1TB")
                ],
            )
        ],
        returns=["p.category"],
    )
    compiled = qg_compile(request, CATALOG_SCHEMA)
    assert compiled.params["p0"] == "portable ssd 1tb"


def test_compile_leaves_already_normalized_value_unchanged():
    request = QueryRequest(
        dataset="catalog",
        matches=[
            QueryMatch(
                var="p", label="Product",
                filters=[
                    QueryFilter(property="nameNormalized", op="=", value="portable ssd 1tb")
                ],
            )
        ],
        returns=["p.category"],
    )
    compiled = qg_compile(request, CATALOG_SCHEMA)
    assert compiled.params["p0"] == "portable ssd 1tb"


def test_compile_does_not_normalize_for_ordering_operators():
    # The fix is scoped to `=`/`<>` only (DESIGN §5.1's convention has no
    # ordering semantics over a normalized value) — a `>` filter must pass
    # its value through unchanged.
    request = QueryRequest(
        dataset="catalog",
        matches=[
            QueryMatch(
                var="p", label="Product",
                filters=[QueryFilter(property="nameNormalized", op=">", value="Zeta")],
            )
        ],
        returns=["p.category"],
    )
    compiled = qg_compile(request, CATALOG_SCHEMA)
    assert compiled.params["p0"] == "Zeta"


# ── U29f fix C: scoped DISTINCT + order_by/returns validation guard ─────────


def test_compile_adds_distinct_for_non_aggregate_returns():
    request = QueryRequest(
        dataset="knowledge_base",
        matches=[
            QueryMatch(
                var="e", label="Entity",
                filters=[QueryFilter(property="name", op="=", value="Marlowe Robotics")],
            )
        ],
        returns=["e.type"],
    )
    compiled = qg_compile(request, KNOWLEDGE_BASE_SCHEMA)
    assert compiled.cypher == (
        "MATCH (e:Entity) WHERE e.name = $p0 RETURN DISTINCT e.type LIMIT $limit"
    )


def test_compile_does_not_add_distinct_for_aggregate_returns():
    request = QueryRequest(
        dataset="knowledge_base",
        matches=[
            QueryMatch(
                var="e", label="Entity",
                filters=[QueryFilter(property="type", op="=", value="Organization")],
            )
        ],
        returns=["count(e)"],
    )
    compiled = qg_compile(request, KNOWLEDGE_BASE_SCHEMA)
    assert compiled.cypher == "MATCH (e:Entity) WHERE e.type = $p0 RETURN count(e) LIMIT $limit"
    assert "DISTINCT" not in compiled.cypher


def test_compile_uses_tuple_distinct_when_order_by_not_in_returns():
    # graph-dba (`claude/graph-dba/falkordb-quirks.md`, "distinct
    # projection, ordered by a column NOT in the projection" entry): a plain
    # `RETURN DISTINCT <returns> ORDER BY <order_by not in returns>` can
    # return a flat-out WRONG answer (DISTINCT collapses to the returned
    # columns before ORDER BY runs, discarding the sort-key pairing). The
    # confirmed fix dedups on the full tuple in a WITH, then RETURNs only
    # the requested columns re-aliased back to their original expression
    # text (the exact nlq-25 probe shape: order_by names a column not in
    # returns).
    request = QueryRequest(
        dataset="knowledge_base",
        matches=[
            QueryMatch(
                var="e", label="Entity",
                filters=[QueryFilter(property="type", op="=", value="Location")],
            )
        ],
        returns=["e.entityId"],
        order_by="e.name",
    )
    compiled = qg_compile(request, KNOWLEDGE_BASE_SCHEMA)
    assert compiled.cypher == (
        "MATCH (e:Entity) WHERE e.type = $p0 "
        "WITH DISTINCT e.entityId AS c0, e.name AS c1 "
        "ORDER BY c1 ASC LIMIT $limit "
        "RETURN c0 AS `e.entityId`"
    )


def test_compile_uses_tuple_distinct_for_superlative_shape_with_no_filter():
    # The DSL's only way to express a superlative ("which product is the
    # cheapest?"): no aggregate return, order_by/limit only, order_by not
    # among returns. Must compile (never raise) via the tuple-DISTINCT form.
    request = QueryRequest(
        dataset="catalog",
        matches=[QueryMatch(var="p", label="Product", filters=[])],
        returns=["p.name"],
        order_by="p.price",
        order_dir="ASC",
        limit=1,
    )
    compiled = qg_compile(request, CATALOG_SCHEMA)
    assert compiled.cypher == (
        "MATCH (p:Product) "
        "WITH DISTINCT p.name AS c0, p.price AS c1 "
        "ORDER BY c1 ASC LIMIT $limit "
        "RETURN c0 AS `p.name`"
    )
    assert compiled.params == {"limit": 1}


def test_compile_tuple_distinct_pairs_multiple_columns_correctly():
    # analyst review MAJOR 1: every prior tuple-DISTINCT test used exactly
    # ONE `returns` entry, so a mutation reversing the alias `zip()` order
    # (silently swapping which value lands under which key) shipped past the
    # whole suite undetected. Two non-aggregate `returns` columns plus a
    # third `order_by` column not among them — assert the per-column `AS`
    # mapping explicitly (not just "contains WITH DISTINCT"), so a swapped
    # pairing fails this assertion even though "WITH DISTINCT" is still
    # present in the string.
    request = QueryRequest(
        dataset="catalog",
        matches=[QueryMatch(var="p", label="Product", filters=[])],
        returns=["p.name", "p.category"],
        order_by="p.price",
    )
    compiled = qg_compile(request, CATALOG_SCHEMA)
    assert compiled.cypher == (
        "MATCH (p:Product) "
        "WITH DISTINCT p.name AS c0, p.category AS c1, p.price AS c2 "
        "ORDER BY c2 ASC LIMIT $limit "
        "RETURN c0 AS `p.name`, c1 AS `p.category`"
    )


# ── analyst review MAJOR 2: a duplicate `returns` entry must be rejected ────


def test_compile_rejects_duplicate_projection_in_returns():
    # A repeated `returns` expression compiles today into a Cypher `RETURN`
    # with two identically-named columns, which FalkorDB itself rejects at
    # execution time ("Multiple result columns with the same name are not
    # supported") — a crash `QueryGraphDataTool` cannot recover from, since
    # only `QueryRequest.model_validate`/`compile()` are wrapped in its
    # try/except, not the query execution after it. Reject at compile time,
    # like every other DSL-legal-but-engine-rejected shape this module guards
    # against.
    request = QueryRequest(
        dataset="catalog",
        matches=[QueryMatch(var="p", label="Product", filters=[])],
        returns=["p.name", "p.name"],
    )
    with pytest.raises(ValueError):
        qg_compile(request, CATALOG_SCHEMA)


def test_compile_rejects_duplicate_aggregate_in_returns():
    request = QueryRequest(
        dataset="catalog",
        matches=[QueryMatch(var="p", label="Product", filters=[])],
        returns=["count(p)", "count(p)"],
    )
    with pytest.raises(ValueError):
        qg_compile(request, CATALOG_SCHEMA)


def test_compile_allows_order_by_in_returns_when_distinct_applies():
    request = QueryRequest(
        dataset="knowledge_base",
        matches=[
            QueryMatch(
                var="e", label="Entity",
                filters=[QueryFilter(property="type", op="=", value="Location")],
            )
        ],
        returns=["e.name"],
        order_by="e.name",
    )
    compiled = qg_compile(request, KNOWLEDGE_BASE_SCHEMA)
    assert compiled.cypher == (
        "MATCH (e:Entity) WHERE e.type = $p0 RETURN DISTINCT e.name "
        "ORDER BY e.name ASC LIMIT $limit"
    )


def test_compile_does_not_validate_order_by_against_returns_for_aggregate_query():
    # DISTINCT never applies to an aggregate-shaped return, so the new
    # order_by/returns validation guard must not fire even when order_by
    # names a column absent from returns.
    request = QueryRequest(
        dataset="catalog",
        matches=[QueryMatch(var="p", label="Product", filters=[])],
        returns=["count(p)"],
        order_by="p.price",
    )
    compiled = qg_compile(request, CATALOG_SCHEMA)
    assert "DISTINCT" not in compiled.cypher
    assert compiled.cypher.endswith("ORDER BY p.price ASC LIMIT $limit")


# ── RCA regression reproductions (docs/reviews/workflow-nl-query-generation-rca.md §3) ──


def test_regression_nlq08_price_filter_with_quoted_numeric_value_now_matches():
    # nlq-08: "Which products cost less than $50?" — the live probe recovered
    # filters:[{"property":"price","op":"<","value":"50"}] (a quoted string).
    request = QueryRequest(
        dataset="catalog",
        matches=[
            QueryMatch(var="p", label="Product",
                       filters=[QueryFilter(property="price", op="<", value="50")])
        ],
        returns=["p.name"],
    )
    compiled = qg_compile(request, CATALOG_SCHEMA)
    assert compiled.params["p0"] == 50.0


def test_regression_nlq02_name_normalized_mixed_case_value_now_matches():
    # nlq-02: "What category is the Portable SSD 1TB in?" — the live probe
    # recovered filters:[{"property":"nameNormalized","op":"=",
    # "value":"Portable SSD 1TB"}] (verbatim, not normalized).
    request = QueryRequest(
        dataset="catalog",
        matches=[
            QueryMatch(var="p", label="Product",
                       filters=[QueryFilter(property="nameNormalized", op="=",
                                             value="Portable SSD 1TB")])
        ],
        returns=["p.category"],
    )
    compiled = qg_compile(request, CATALOG_SCHEMA)
    assert compiled.params["p0"] == "portable ssd 1tb"


def test_regression_nlq21_duplicate_entity_projection_now_deduped():
    # nlq-21: "What type of entity is Marlowe Robotics?" — the RCA's key
    # finding: filter and projection are already correct, the 9 un-fused
    # duplicate nodes need DISTINCT, not a filter/projection change.
    request = QueryRequest(
        dataset="knowledge_base",
        matches=[
            QueryMatch(var="e", label="Entity",
                       filters=[QueryFilter(property="name", op="=", value="Marlowe Robotics")])
        ],
        returns=["e.type"],
    )
    compiled = qg_compile(request, KNOWLEDGE_BASE_SCHEMA)
    assert "DISTINCT" in compiled.cypher


def test_regression_nlq25_order_by_not_in_returns_now_deduped_not_nondeterministic():
    # nlq-25: "Which entities are of type Location?" — the probe recovered
    # returns:["e.entityId"] with order_by:"e.name" (not in returns). The
    # confirmed fix compiles this via the tuple-DISTINCT WITH form (never
    # raises), which dedups correctly instead of the plain-DISTINCT form's
    # silently non-deterministic (and, per graph-dba, potentially wrong) sort.
    request = QueryRequest(
        dataset="knowledge_base",
        matches=[
            QueryMatch(var="e", label="Entity",
                       filters=[QueryFilter(property="type", op="=", value="Location")])
        ],
        returns=["e.entityId"],
        order_by="e.name",
    )
    compiled = qg_compile(request, KNOWLEDGE_BASE_SCHEMA)
    assert "WITH DISTINCT" in compiled.cypher
    assert "RETURN c0 AS `e.entityId`" in compiled.cypher


def test_regression_nlq16_superlative_cheapest_product_now_compiles():
    # nlq-16: "Which product is the cheapest?" — filters:[], order_by:
    # "p.price", limit:1, returns:["p.name"]. This exact shape is worked
    # example #4 in `_QUERY_REQUEST_INSTRUCTIONS` and is in the golden set
    # (nlq-16/17/20) — an earlier version of fix C rejected it outright,
    # which was itself a regression; it must compile successfully.
    request = QueryRequest(
        dataset="catalog",
        matches=[QueryMatch(var="p", label="Product", filters=[])],
        returns=["p.name"],
        order_by="p.price",
        order_dir="ASC",
        limit=1,
    )
    compiled = qg_compile(request, CATALOG_SCHEMA)
    assert "WITH DISTINCT p.name AS c0, p.price AS c1" in compiled.cypher
    assert "RETURN c0 AS `p.name`" in compiled.cypher


def test_regression_nlq31_aggregate_count_unaffected_by_distinct_scoping():
    # nlq-31: "How many Organization-type entities are there?" — golden value
    # is the raw un-fused count (17); DISTINCT must never touch this shape.
    request = QueryRequest(
        dataset="knowledge_base",
        matches=[
            QueryMatch(var="e", label="Entity",
                       filters=[QueryFilter(property="type", op="=", value="Organization")])
        ],
        returns=["count(e)"],
    )
    compiled = qg_compile(request, KNOWLEDGE_BASE_SCHEMA)
    assert compiled.cypher == "MATCH (e:Entity) WHERE e.type = $p0 RETURN count(e) LIMIT $limit"


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
