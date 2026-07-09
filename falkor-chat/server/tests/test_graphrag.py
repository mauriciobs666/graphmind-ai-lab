"""Integration tests for the K-008 GraphRAG retrieval core (repository layer).

`set_embedding` and `hybrid_search` map 1:1 to `docs/QUERIES.md` §6 (verified in
`docs/plans/m2-graphrag.md`). Tests run against the live `ws:test` graph, whose
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
