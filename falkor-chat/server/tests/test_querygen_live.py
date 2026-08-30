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
(`server/tests/eval/run_nlq_golden_set_eval.py`, a later unit's job): four
representative shapes from the RCA's own probes (§3), enough to confirm the
compiled fix actually returns the golden set's expected *values* against the
real data — stronger evidence than the compile()-only unit tests in
`test_querygen.py` alone, which only check the emitted Cypher/params text.
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
