"""Regression guard for the vector-index churn hazard fixed via
`conftest.rebuild_vector_indexes` (`docs/reviews/document-ingestion2-rca.md`).

`conftest._schema` rebuilds `ws:test`'s vector indexes only **once per pytest
session**; every embedding-writing test after that piles create/delete churn
onto the SAME never-mid-session-rebuilt HNSW index, and this FalkorDB build's
ANN recall for small `k` degrades **monotonically** with that cumulative
churn (RCA §2/§3, Appendix B). This is not a hypothesis here either — it is
reproduced live below, on a disposable probe graph, adapted from the RCA's
Appendix B churn script.

Runs against its own throwaway `ws:{PROBE_WS}` graph, created and torn down
inside this module (never `ws:test`), so it never spends any of the shared
suite's churn budget and never interferes with any other test.

Two things pinned here:
  1. the phenomenon itself (`test_ann_recall_...`) — recall at a small `k` is
     unreliable after enough churn on a never-rebuilt index, while a much
     larger `k` stays reliable through the same churn;
  2. the actual fix (`test_rebuilding_...`) — `conftest.rebuild_vector_indexes`
     (the mechanism behind the `fresh_vector_index` fixture used throughout
     the rest of this suite) restores recall regardless of how much churn
     preceded it.

Test 2 is the actual guard the RCA says was missing: break
`rebuild_vector_indexes` (make it a no-op, or drop `fresh_vector_index` from
an ANN-sensitive test module) and this file — not just the tests that
happened to be failing when the RCA was written — starts failing.
"""

from __future__ import annotations

import random

import pytest
from conftest import TEST_EMBEDDING_DIM, _falkordb_reachable, rebuild_vector_indexes

from falkorchat import db

PROBE_WS = "vector_churn_guard"
_TARGET_VEC = [1.0] + [0.0] * (TEST_EMBEDDING_DIM - 1)

# Cumulative churn batches. Live-verified (repeatedly, against the pinned
# FalkorDB build) to cross the k=4 recall cliff by the final batch while
# leaving k=50 reliable throughout — mirrors the RCA's own Appendix B shape
# ("k=4 recall starts failing around 150-200 cumulative create/delete cycles
# ... k=50 stayed reliable throughout every run attempted"). The exact
# cumulative count this build's cliff sits at is itself build-specific and not
# the point being pinned — see the module docstring.
_CHURN_BATCHES = [10, 40, 50, 100, 100]  # cumulative: 10, 50, 100, 200, 300


@pytest.fixture()
def probe_conn():
    """A connection plus a throwaway `ws:{PROBE_WS}` graph with a fresh
    Message+Chunk vector index at `TEST_EMBEDDING_DIM` — deleted again at
    teardown regardless of test outcome, so a failure here never leaves the
    probe graph behind for another run to trip over."""
    if not _falkordb_reachable(PROBE_WS):
        pytest.skip("FalkorDB not reachable — start it with ./scripts/start_falkordb.sh")
    connection = db.connect()
    graph = db.workspace_graph(connection, PROBE_WS)
    try:
        graph.delete()
    except Exception:
        pass  # graph may not exist yet
    for label in ("Message", "Chunk"):
        graph.query(
            f"CREATE VECTOR INDEX FOR (n:{label}) ON (n.embedding) "
            f"OPTIONS {{dimension:{TEST_EMBEDDING_DIM}, similarityFunction:'cosine'}}"
        )
    try:
        yield connection
    finally:
        graph.delete()


def _rand_vec() -> list[float]:
    return [random.random() for _ in range(TEST_EMBEDDING_DIM)]


def _churn(graph, cycles: int) -> None:
    """`cycles` create+delete cycles of randomly-vectored `Chunk`s — the same
    create/embed/delete shape every real embedding test performs."""
    for _ in range(cycles):
        graph.query("CREATE (c:Chunk {embedding: vecf32($v)})", {"v": _rand_vec()})
    graph.query("MATCH (c:Chunk) WHERE c.chunkId IS NULL DETACH DELETE c")


def _recall_at(graph, k: int) -> bool:
    """True iff an exact-match `Chunk`, seeded fresh at `_TARGET_VEC`, is
    retrieved by a `k`-NN query for that same vector. Creates and removes the
    target chunk itself — live-verified as essential to reproducing the
    cliff deterministically: this repeated churn *at the query's own
    coordinates* (exactly what happens when several real tests reuse the same
    stub vector, e.g. `test_graphrag.py`'s `_pad([1.0])`) degrades recall for
    that neighborhood far faster and more reliably than generic random churn
    checked only once at the end.
    """
    graph.query(
        "CREATE (c:Chunk {chunkId:'target', embedding: vecf32($v)})",
        {"v": _TARGET_VEC},
    )
    rows = graph.ro_query(
        "CALL db.idx.vector.queryNodes('Chunk','embedding',$k, vecf32($v)) "
        "YIELD node RETURN node.chunkId",
        {"k": k, "v": _TARGET_VEC},
    ).result_set
    found = "target" in [row[0] for row in rows]
    graph.query("MATCH (c:Chunk {chunkId:'target'}) DETACH DELETE c")
    return found


def _run_churn_and_track_k4(graph) -> dict[int, bool]:
    """Run `_CHURN_BATCHES` cumulatively, recording k=4 recall after each
    batch (each check itself contributing churn at the target's own
    coordinates — see `_recall_at`)."""
    cumulative = 0
    results: dict[int, bool] = {}
    for batch in _CHURN_BATCHES:
        _churn(graph, batch)
        cumulative += batch
        results[cumulative] = _recall_at(graph, 4)
    return results


def test_ann_recall_at_small_k_degrades_with_cumulative_churn_on_a_never_rebuilt_index(
    probe_conn,
):
    """Monotonic-in-churn characterization, adapted from the RCA's Appendix B
    script: on a near-fresh index small-`k` recall is reliable; past enough
    cumulative create/delete churn on the SAME never-rebuilt index it is not
    — while a much larger `k` stays reliable through the identical churn.
    This is the live-engine behavior `rebuild_vector_indexes` exists to
    bound.
    """
    graph = db.workspace_graph(probe_conn, PROBE_WS)

    k4_by_cumulative_churn = _run_churn_and_track_k4(graph)

    assert k4_by_cumulative_churn[50] is True, (
        "k=4 should still be reliable on a near-fresh index"
    )
    assert k4_by_cumulative_churn[200] is True, (
        "k=4 should still be reliable before this build's documented cliff"
    )
    assert k4_by_cumulative_churn[300] is False, (
        "k=4 should have crossed the recall cliff by the final churn batch — "
        "if this now passes, this build's HNSW behavior (or these churn "
        "constants) changed and the rebuild-boundary design needs revisiting"
    )
    assert _recall_at(graph, 50) is True, (
        "k=50 should stay reliable through the same cumulative churn that "
        "already broke k=4"
    )


def test_rebuilding_the_vector_index_recovers_recall_after_heavy_churn(probe_conn):
    """The actual fix under guard: `rebuild_vector_indexes` restores k=4
    recall regardless of how much prior churn degraded it. Break the fix
    (make it a no-op, or stop calling it before an ANN-sensitive test) and
    this test fails — the defect-catching mechanism the RCA says was missing,
    as opposed to a fixed-k bump on whichever tests happened to be failing
    when the RCA was written.
    """
    graph = db.workspace_graph(probe_conn, PROBE_WS)
    k4_by_cumulative_churn = _run_churn_and_track_k4(graph)
    assert k4_by_cumulative_churn[300] is False, (
        "setup check: this much churn should already have broken k=4 recall "
        "— if it hasn't, the fix-recovers assertion below would be "
        "meaningless"
    )

    rebuild_vector_indexes(probe_conn, ws=PROBE_WS, dim=TEST_EMBEDDING_DIM)

    assert _recall_at(graph, 4) is True
