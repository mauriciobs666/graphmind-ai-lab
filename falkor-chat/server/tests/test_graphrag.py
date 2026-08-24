"""Integration tests for the K-008 GraphRAG retrieval core (repository layer).

`set_embedding` and `hybrid_search` map 1:1 to `docs/QUERIES.md` §6 (verified in
`docs/archive/plans/m2-graphrag.md`). Tests run against the live `ws:test` graph, whose
vector index is bootstrapped at `TEST_EMBEDDING_DIM` (4) — stub vectors match.

Ranking is by **cosine distance ASC** (0 = identical). ANN recall is approximate
on a nearly-empty index (kNN may return fewer than k) — assert *ordering* and
*membership*, never an exact neighbor count.
"""

from __future__ import annotations

import pytest
from conftest import TEST_EMBEDDING_DIM

from falkorchat.repository import EmbeddingDimensionError

WS = "test"


def _pad(head: list[float]) -> list[float]:
    """A TEST_EMBEDDING_DIM vector from a leading fragment (zero-padded)."""
    return (head + [0.0] * TEST_EMBEDDING_DIM)[:TEST_EMBEDDING_DIM]


def _seed_thread(repo, *, channel_id, thread_id, messages):
    """Create channel→thread, author `u1`, post `messages` [(msg_id, text, vec)]."""
    repo.ensure_user(WS, user_id="u1")
    repo.create_channel(WS, channel_id=channel_id, name=channel_id, created_at=1)
    repo.create_thread(
        WS, channel_id=channel_id, thread_id=thread_id, title="t", created_at=1
    )
    first = True
    ts = 10
    for msg_id, text, vec in messages:
        write = repo.post_first_message if first else repo.post_subsequent_message
        write(
            WS, thread_id=thread_id, msg_id=msg_id, author_id="u1",
            text=text, role="user", created_at=ts,
        )
        repo.set_embedding(WS, msg_id=msg_id, embedding=vec, expected_dim=TEST_EMBEDDING_DIM)
        first = False
        ts += 1


# ── set_embedding ────────────────────────────────────────────────────────────


def test_set_embedding_rejects_wrong_dimension_loudly(repo):
    # The critical quirk: a wrong-dim vecf32 SET is silently accepted by FalkorDB
    # and the node then vanishes from the ANN index. Validate client-side first.
    repo.ensure_user(WS, user_id="u1")
    repo.create_channel(WS, channel_id="c1", name="c1", created_at=1)
    repo.create_thread(WS, channel_id="c1", thread_id="t1", title="t", created_at=1)
    repo.post_first_message(
        WS, thread_id="t1", msg_id="m1", author_id="u1",
        text="hi", role="user", created_at=10,
    )
    with pytest.raises(EmbeddingDimensionError):
        repo.set_embedding(
            WS, msg_id="m1", embedding=[1.0] * (TEST_EMBEDDING_DIM + 1),
            expected_dim=TEST_EMBEDDING_DIM,
        )


def test_set_embedding_writes_and_message_is_ann_retrievable(repo):
    _seed_thread(
        repo, channel_id="c1", thread_id="t1",
        messages=[("m1", "about cats", _pad([1.0]))],
    )
    rows = repo.hybrid_search(WS, q_vec=_pad([1.0]), k=4, limit=5)
    assert "m1" in [r["msgId"] for r in rows]


# ── hybrid_search: ranking ────────────────────────────────────────────────────


def test_hybrid_search_ranks_by_cosine_distance_asc(repo):
    _seed_thread(
        repo, channel_id="c1", thread_id="t1",
        messages=[
            ("m1", "about cats", _pad([1.0, 0.0])),
            ("m2", "more on cats", _pad([0.9, 0.1])),
            ("m3", "about dogs", _pad([0.0, 0.0, 1.0])),
        ],
    )
    # query vector identical to m1 → m1 scores 0 (most similar) and ranks first
    rows = repo.hybrid_search(WS, q_vec=_pad([1.0, 0.0]), k=4, limit=5)
    ids = [r["msgId"] for r in rows]

    assert ids[0] == "m1"
    assert "m2" in ids
    assert ids.index("m1") < ids.index("m2")
    # scores are non-decreasing (ASC) and the identical vector scores 0
    scores = [r["score"] for r in rows]
    assert scores == sorted(scores)
    assert rows[0]["score"] == pytest.approx(0.0, abs=1e-6)


def test_hybrid_search_returns_seed_text_and_role(repo):
    _seed_thread(
        repo, channel_id="c1", thread_id="t1",
        messages=[("m1", "about cats", _pad([1.0]))],
    )
    rows = repo.hybrid_search(WS, q_vec=_pad([1.0]), k=4, limit=5)
    row = next(r for r in rows if r["msgId"] == "m1")
    assert row["text"] == "about cats"
    assert row["role"] == "user"


# ── hybrid_search: Entity layer dormant (M2) ──────────────────────────────────


def test_hybrid_search_related_context_is_empty_list(repo):
    # The Entity co-occurrence expansion is present in the query but dormant in
    # M2 (no extraction pipeline) — the OPTIONAL MATCH no-ops → relatedContext [].
    _seed_thread(
        repo, channel_id="c1", thread_id="t1",
        messages=[("m1", "about cats", _pad([1.0]))],
    )
    rows = repo.hybrid_search(WS, q_vec=_pad([1.0]), k=4, limit=5)
    assert rows
    for r in rows:
        assert r["relatedContext"] == []


# ── hybrid_search: channel-scoped vs workspace-wide ───────────────────────────


def test_hybrid_search_channel_scoped_excludes_other_channels(repo):
    _seed_thread(
        repo, channel_id="c1", thread_id="t1",
        messages=[("m1", "in c1", _pad([1.0]))],
    )
    _seed_thread(
        repo, channel_id="c2", thread_id="t2",
        messages=[("mA", "in c2", _pad([1.0]))],
    )
    scoped = repo.hybrid_search(WS, q_vec=_pad([1.0]), k=10, limit=10, channel_id="c1")
    ids = [r["msgId"] for r in scoped]
    assert "m1" in ids
    assert "mA" not in ids


def test_hybrid_search_workspace_wide_spans_channels(repo):
    _seed_thread(
        repo, channel_id="c1", thread_id="t1",
        messages=[("m1", "in c1", _pad([1.0]))],
    )
    _seed_thread(
        repo, channel_id="c2", thread_id="t2",
        messages=[("mA", "in c2", _pad([1.0]))],
    )
    wide = repo.hybrid_search(WS, q_vec=_pad([1.0]), k=10, limit=10)
    ids = [r["msgId"] for r in wide]
    assert {"m1", "mA"} <= set(ids)


# ── read_index_dimension (K-042 Landing 2, FR-19) ───────────────────────────────
#
# `-graph.md` §3.2's `CALL db.indexes()` introspection, live against `ws:test`
# (bootstrapped at TEST_EMBEDDING_DIM=4, `conftest.py::_schema`). The four edge
# behaviours the design tabulates collapse to two outcomes at this layer: a real
# int (vector index present — index metadata, not data, so this holds even with
# zero vectors written) or `None` (every "no vector index to compare against"
# case, including the graph-key-absent one below).


def test_read_index_dimension_returns_the_configured_dimension_for_message(repo):
    assert repo.read_index_dimension(WS, label="Message") == TEST_EMBEDDING_DIM


def test_read_index_dimension_none_for_a_range_only_label(repo):
    # `User` carries only a plain range index (`bootstrap_schema.sh`) — no vector
    # index at all for this label, so the read must be "no index", not an error
    # and not a made-up dimension.
    assert repo.read_index_dimension(WS, label="User") is None


def test_read_index_dimension_none_for_an_unknown_label(repo):
    assert repo.read_index_dimension(WS, label="NoSuchLabelAtAll") is None


def test_read_index_dimension_returns_the_configured_dimension_for_chunk(repo):
    assert repo.read_index_dimension(WS, label="Chunk") == TEST_EMBEDDING_DIM


def test_read_index_dimension_none_when_the_graph_key_does_not_exist(repo, conn):
    # -graph.md §3.2 edge case 4: a `ws:{id}` graph that was never bootstrapped
    # makes FalkorDB raise "ERR Invalid graph operation on empty key" rather than
    # returning zero rows. The read must fold this into the same "no index"
    # answer as the other two cases above, and — critically — must NOT create
    # the graph as a side effect of merely checking it (routed via `ro_query`,
    # never a write).
    ghost_ws = "u11-fr19-ghost-probe-does-not-exist"
    before = set(conn.list_graphs())
    assert f"ws:{ghost_ws}" not in before

    assert repo.read_index_dimension(ghost_ws, label="Message") is None

    after = set(conn.list_graphs())
    assert f"ws:{ghost_ws}" not in after  # confirmed: the read created nothing


# ── set_chunk_embedding / search_chunks (K-050 M5 Stage 2, FR-3) ────────────────
#
# Mirrors the `set_embedding`/`hybrid_search` section above exactly, `Chunk` in
# place of `Message` — same live `ws:test` vector index (bootstrapped at
# TEST_EMBEDDING_DIM=4, `document-ingestion-graph.md` §2.1 confirms no new DDL
# was needed for it). `search_chunks` has no scope traversal (a chunk has no
# Thread/Channel) and no Entity expansion (ABOUT is dormant until Stage 3).


def _seed_document(repo, *, document_id, chunks):
    """Create `document_id` with `chunks` = [(chunk_id, text, vec_or_None)],
    embedding each chunk whose `vec` is not None. Actor is a fixed `u1` user,
    ensured idempotently (mirrors `_seed_thread` above)."""
    repo.ensure_user(WS, user_id="u1")
    repo.create_document(
        WS, document_id=document_id, title="t",
        text="".join(text for _cid, text, _vec in chunks),
        source_format="text", ingested_by="u1", created_at=1,
        chunks=[
            {"chunkId": cid, "text": text, "seq": i}
            for i, (cid, text, _vec) in enumerate(chunks)
        ],
    )
    for cid, _text, vec in chunks:
        if vec is not None:
            repo.set_chunk_embedding(
                WS, chunk_id=cid, embedding=vec, expected_dim=TEST_EMBEDDING_DIM
            )


def test_set_chunk_embedding_rejects_wrong_dimension_loudly(repo):
    repo.ensure_user(WS, user_id="u1")
    repo.create_document(
        WS, document_id="d1", title="t", text="x", source_format="text",
        ingested_by="u1", created_at=1,
        chunks=[{"chunkId": "c1", "text": "x", "seq": 0}],
    )
    with pytest.raises(EmbeddingDimensionError):
        repo.set_chunk_embedding(
            WS, chunk_id="c1", embedding=[1.0] * (TEST_EMBEDDING_DIM + 1),
            expected_dim=TEST_EMBEDDING_DIM,
        )


def test_set_chunk_embedding_writes_and_chunk_is_ann_retrievable(repo):
    _seed_document(
        repo, document_id="d1", chunks=[("c1", "about cats", _pad([1.0]))],
    )
    rows = repo.search_chunks(WS, q_vec=_pad([1.0]), k=4, limit=5)
    assert "c1" in [r["chunkId"] for r in rows]


def test_search_chunks_ranks_by_cosine_distance_asc(repo):
    _seed_document(
        repo, document_id="d1",
        chunks=[
            ("c1", "about cats", _pad([1.0, 0.0])),
            ("c2", "more on cats", _pad([0.9, 0.1])),
            ("c3", "about dogs", _pad([0.0, 0.0, 1.0])),
        ],
    )
    rows = repo.search_chunks(WS, q_vec=_pad([1.0, 0.0]), k=4, limit=5)
    ids = [r["chunkId"] for r in rows]

    assert ids[0] == "c1"
    assert "c2" in ids
    assert ids.index("c1") < ids.index("c2")
    scores = [r["score"] for r in rows]
    assert scores == sorted(scores)
    assert rows[0]["score"] == pytest.approx(0.0, abs=1e-6)


def test_search_chunks_returns_denormalized_document_id_and_seq(repo):
    _seed_document(
        repo, document_id="d7",
        chunks=[("c1", "about cats", _pad([1.0]))],
    )
    rows = repo.search_chunks(WS, q_vec=_pad([1.0]), k=4, limit=5)
    row = next(r for r in rows if r["chunkId"] == "c1")
    assert row["text"] == "about cats"
    assert row["documentId"] == "d7"
    assert row["seq"] == 0


def test_search_chunks_never_returns_a_message_seed(repo):
    # `search_chunks` is Chunk-only (FR-3, a genuinely separate capability from
    # `hybrid_search`'s Message-only ANN, plan §3.7/FR-14) — seeding a Message
    # at the identical vector must not leak into a Chunk search.
    _seed_thread(
        repo, channel_id="c1", thread_id="t1",
        messages=[("m1", "about cats", _pad([1.0]))],
    )
    rows = repo.search_chunks(WS, q_vec=_pad([1.0]), k=4, limit=5)
    assert rows == []
