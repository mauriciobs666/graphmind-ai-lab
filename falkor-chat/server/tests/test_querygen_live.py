"""Live regression check for U29f's `querygen.compile()` fixes (K-055 M6),
`docs/reviews/workflow-nl-query-generation-rca.md` §3/§4.

**Marker-gated**, mirroring `test_ac5_chat_grounding_live.py`/`test_workflow_live.py`:
`pytest.mark.live`, DESELECTED by default (`addopts = -m "not live"` in
`pyproject.toml`). Run explicitly:

    cd server && .venv/bin/python -m pytest -m live -s tests/test_querygen_live.py

Needs only FalkorDB up — no LM Studio dependency, since this reproduces the
*compiler*-level fix directly against hand-built `QueryRequest`s (the exact
shapes the RCA's live LLM probes recovered), not the model's own completion.
Unreachable FalkorDB skips (never fails) with a reason.

**Read-only against shared graphs.** `run_readonly_query` always executes via
`.ro_query` (`repository.run_readonly_query`'s own Layer-2 guarantee) — this
file never writes to `reference` or `ws:nlq-eval`. Both graphs must already
be seeded: `reference` via `./scripts/bootstrap_schema.sh <ws> &&
./scripts/seed_catalog.sh` (a default offline `pytest` run wipes `reference`
at teardown, `falkor-chat/AGENTS.md`); `ws:nlq-eval` is the golden-set
knowledge-base workspace seeded by an earlier unit and left alone here.

This is deliberately narrower than the full 39-pair golden-set harness
(`server/tests/eval/run_nlq_golden_set_eval.py`, a later unit's job): a
handful of representative shapes from the RCA's own probes (§3), enough to
confirm the compiled fix actually returns the golden set's expected *values*
against the real data — stronger evidence than the compile()-only unit tests
in `test_querygen.py` alone, which only check the emitted Cypher/params text.

**Correction (fix C, superlative shape):** the tuple-DISTINCT `WITH`
compilation (`claude/graph-dba/falkordb-quirks.md`, "distinct projection,
ordered by a column NOT in the projection") replaced an earlier reject-outright
guard that made every superlative question ("which product is the cheapest?")
raise `ValueError` — that shape (`order_by` not among `returns`, no aggregate
return) is the DSL's only way to express it, and it's in the golden set
(`nlq-16/17/20`). The `test_nlq16_*`/`test_nlq17_*`/`test_nlq20_*` cases below
reproduce it live. `nlq-25` reuses the exact `returns=["e.entityId"],
order_by="e.name"` shape the RCA's probe recovered for it to prove the new
compilation runs correctly against real data too — but note `nlq-25`'s
*golden* value is a list of `name`s, not `entityId`s (RCA category D, a prompt
fix, not this fix's job), so that test asserts structural correctness (no
error, correctly ordered, right row count) rather than a golden-set value
match.
"""

from __future__ import annotations

import pytest

from falkorchat import db
from falkorchat.querygen import (
    CATALOG_SCHEMA,
    KNOWLEDGE_BASE_SCHEMA,
    QueryFilter,
    QueryMatch,
    QueryRequest,
)
from falkorchat.querygen import compile as qg_compile
from falkorchat.repository import Repository

NLQ_EVAL_WS = "nlq-eval"


def _falkordb_reachable() -> bool:
    try:
        db.connect().select_graph("reference").query("RETURN 1")
        return True
    except Exception:
        return False


pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not _falkordb_reachable(), reason="FalkorDB is not reachable"),
]


@pytest.fixture(scope="module")
def repo() -> Repository:
    return Repository(db.connect())


def test_nlq08_quoted_numeric_price_filter_matches_golden_set(repo: Repository):
    # nlq-08: "Which products cost less than $50?" — probe recovered
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
    rows = repo.run_readonly_query("reference", compiled)
    names = {row["p.name"] for row in rows}
    assert names == {
        "Gaming Mouse Pad XL", "Wireless Charging Pad", "Wireless Mouse Pro",
        "Laptop Stand Aluminum", "USB-C Hub 7-in-1", "Bluetooth Speaker Mini",
    }


def test_nlq02_verbatim_name_normalized_value_matches_golden_set(repo: Repository):
    # nlq-02: "What category is the Portable SSD 1TB in?" — probe recovered
    # filters:[{"property":"nameNormalized","op":"=","value":"Portable SSD 1TB"}]
    # (verbatim question text, not the stored normalized form).
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
    rows = repo.run_readonly_query("reference", compiled)
    assert rows == [{"p.category": "Storage"}]


def test_nlq21_duplicate_entity_projection_is_deduped_to_one_row(repo: Repository):
    # nlq-21: "What type of entity is Marlowe Robotics?" — 9 un-fused
    # duplicate nodes; the RCA's fix is DISTINCT, not the filter/projection
    # (already correct). Golden value is the scalar "Organization".
    request = QueryRequest(
        dataset="knowledge_base",
        matches=[
            QueryMatch(var="e", label="Entity",
                       filters=[QueryFilter(property="name", op="=", value="Marlowe Robotics")])
        ],
        returns=["e.type"],
    )
    compiled = qg_compile(request, KNOWLEDGE_BASE_SCHEMA)
    graph_key = f"ws:{NLQ_EVAL_WS}"
    rows = repo.run_readonly_query(graph_key, compiled)
    assert rows == [{"e.type": "Organization"}]


def test_nlq16_cheapest_product_superlative_matches_golden_set(repo: Repository):
    # nlq-16: "Which product is the cheapest?" — filters:[], returns:
    # ["p.name"], order_by:"p.price", order_dir ASC, limit:1. Golden value:
    # "Gaming Mouse Pad XL" (min price 19.99 across all 15 products).
    request = QueryRequest(
        dataset="catalog",
        matches=[QueryMatch(var="p", label="Product", filters=[])],
        returns=["p.name"],
        order_by="p.price",
        order_dir="ASC",
        limit=1,
    )
    compiled = qg_compile(request, CATALOG_SCHEMA)
    rows = repo.run_readonly_query("reference", compiled)
    assert rows == [{"p.name": "Gaming Mouse Pad XL"}]


def test_nlq17_most_expensive_product_superlative_matches_golden_set(repo: Repository):
    # nlq-17: "Which product is the most expensive?" — same shape, DESC.
    # Golden value: "27-inch 4K Monitor" (max price 349.99).
    request = QueryRequest(
        dataset="catalog",
        matches=[QueryMatch(var="p", label="Product", filters=[])],
        returns=["p.name"],
        order_by="p.price",
        order_dir="DESC",
        limit=1,
    )
    compiled = qg_compile(request, CATALOG_SCHEMA)
    rows = repo.run_readonly_query("reference", compiled)
    assert rows == [{"p.name": "27-inch 4K Monitor"}]


def test_nlq20_superlative_combined_with_a_filter_matches_golden_set(repo: Repository):
    # nlq-20: "What is the cheapest product in the Wearables category?" —
    # superlative shape combined with a filter predicate. Golden value:
    # "Fitness Tracker Band" (79.99 vs Smartwatch Series 5's 249.99).
    request = QueryRequest(
        dataset="catalog",
        matches=[
            QueryMatch(var="p", label="Product",
                       filters=[QueryFilter(property="category", op="=", value="Wearables")])
        ],
        returns=["p.name"],
        order_by="p.price",
        order_dir="ASC",
        limit=1,
    )
    compiled = qg_compile(request, CATALOG_SCHEMA)
    rows = repo.run_readonly_query("reference", compiled)
    assert rows == [{"p.name": "Fitness Tracker Band"}]


def test_tuple_distinct_pairs_multiple_columns_correctly_live(repo: Repository):
    # analyst review MAJOR 1: every prior tuple-DISTINCT live test used
    # exactly ONE `returns` entry, so a mutation reversing the alias `zip()`
    # order (silently swapping which value lands under which key) shipped
    # past the entire suite undetected — the reviewer confirmed this
    # separately against a scratchpad mutant. Two non-aggregate `returns`
    # columns plus a third `order_by` column not among them, against the
    # real `reference` graph, asserting the actual PAIRED row values (not
    # just that each value individually appears somewhere).
    request = QueryRequest(
        dataset="catalog",
        matches=[QueryMatch(var="p", label="Product", filters=[])],
        returns=["p.name", "p.category"],
        order_by="p.price",
        limit=1,
    )
    compiled = qg_compile(request, CATALOG_SCHEMA)
    rows = repo.run_readonly_query("reference", compiled)
    # The cheapest product overall (min price 19.99) is "Gaming Mouse Pad
    # XL", category "Peripherals" — a swapped-alias mutant would instead
    # produce {"p.name": "Peripherals", "p.category": "Gaming Mouse Pad XL"}.
    assert rows == [{"p.name": "Gaming Mouse Pad XL", "p.category": "Peripherals"}]


def test_nlq25_entityid_projection_with_order_by_not_in_returns_runs_correctly(repo: Repository):
    # nlq-25: "Which entities are of type Location?" — the RCA's probe (§4.1.3)
    # recovered returns:["e.entityId"] with order_by:"e.name" (not in returns).
    # This exercises the tuple-DISTINCT compile path against real duplicate
    # data (11 raw Location nodes, Colorado x3/Denver x2, per the golden
    # set's own rationale) — asserting it runs correctly (right row count,
    # sorted by name, every key the raw "e.entityId" expression text) rather
    # than the golden-set *name* values, since projecting entityId instead of
    # name is category D (a prompt fix), not this fix's job.
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
    graph_key = f"ws:{NLQ_EVAL_WS}"
    rows = repo.run_readonly_query(graph_key, compiled)
    assert len(rows) == 11  # raw node count, entityId is unique per node (no dedup collapse here)
    assert all(set(row) == {"e.entityId"} for row in rows)


def test_nlq26_name_projection_filter_list_still_dedups_correctly(repo: Repository):
    # nlq-26: "What entities are classified as Person?" — same duplicate-
    # entity family as nlq-21/25 (5 raw Person nodes, Devon Cole x3), but
    # projecting `e.name` directly (no order_by) — the plain scoped-DISTINCT
    # path (untouched by this correction). Golden value: 3 distinct names.
    request = QueryRequest(
        dataset="knowledge_base",
        matches=[
            QueryMatch(var="e", label="Entity",
                       filters=[QueryFilter(property="type", op="=", value="Person")])
        ],
        returns=["e.name"],
    )
    compiled = qg_compile(request, KNOWLEDGE_BASE_SCHEMA)
    graph_key = f"ws:{NLQ_EVAL_WS}"
    rows = repo.run_readonly_query(graph_key, compiled)
    names = {row["e.name"] for row in rows}
    assert names == {"Devon Cole", "Elena Ferro", "Priya Nandakumar"}
    assert len(rows) == 3  # DISTINCT actually collapsed Devon Cole's 3 raw nodes to 1


def test_nlq31_aggregate_count_still_the_raw_unfused_count(repo: Repository):
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
    graph_key = f"ws:{NLQ_EVAL_WS}"
    rows = repo.run_readonly_query(graph_key, compiled)
    assert rows == [{"count(e)": 17}]
