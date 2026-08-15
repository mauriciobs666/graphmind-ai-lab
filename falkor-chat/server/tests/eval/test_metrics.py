"""Unit tests for the pure, network-free retrieval metrics (K-026,
`docs/plans/graphrag-eval.md` §5 Unit 2b).

Genuinely network/DB-free: `recall_at_k`/`mrr` are plain functions over a list of
message ids and a set of relevant ids, so every case here is exercised in-process
with no FalkorDB/LLM dependency — this file runs unconditionally in the default
suite, unlike `test_retrieval_eval.py` (which needs a seeded `ws:eval`).

Edge cases covered explicitly (plan §6):
  - multi-relevant golden pairs (a `relevant` set with more than one id);
  - `hybrid_search` returning fewer than `k` rows (ANN's documented non-guarantee)
    — a short/empty `retrieved_msg_ids` list must not index-error or overcount;
  - the hit position varied across first/middle/last of the retrieved list, not
    just position 0 — both `recall_at_k`'s top-k slice and `mrr`'s first-hit scan
    are positional rules, so a corpus uniform on "hit is always first" would not
    actually prove the slicing/rank-counting logic works.
"""

from __future__ import annotations

import pytest

from metrics import mrr, recall_at_k

# ── recall_at_k ─────────────────────────────────────────────────────────────


def test_recall_at_k_full_hit_within_k_is_one() -> None:
    assert recall_at_k(["a", "b", "c"], {"a"}, k=10) == 1.0


def test_recall_at_k_no_hit_is_zero() -> None:
    assert recall_at_k(["x", "y", "z"], {"a"}, k=10) == 0.0


@pytest.mark.parametrize(
    "retrieved,position",
    [
        (["a", "x", "y", "z", "w"], "first"),
        (["x", "y", "a", "z", "w"], "middle"),
        (["x", "y", "z", "w", "a"], "last"),
    ],
)
def test_recall_at_k_hit_counted_regardless_of_position_within_k(
    retrieved: list[str], position: str
) -> None:
    assert recall_at_k(retrieved, {"a"}, k=10) == 1.0, (
        f"hit at {position} position of the top-k window should still count"
    )


def test_recall_at_k_hit_outside_top_k_window_does_not_count() -> None:
    """A relevant id present in the full list but past the k-th slot must not
    count — this is what distinguishes recall@5 from recall@10 over the same
    underlying ranked list."""
    retrieved = ["x", "y", "z", "w", "v", "a"]  # "a" is rank 6
    assert recall_at_k(retrieved, {"a"}, k=5) == 0.0
    assert recall_at_k(retrieved, {"a"}, k=10) == 1.0


def test_recall_at_k_multi_relevant_partial_hit() -> None:
    """A golden pair with more than one relevant id: only some are retrieved."""
    retrieved = ["a", "c", "d"]
    relevant = {"a", "b"}
    assert recall_at_k(retrieved, relevant, k=10) == 0.5


def test_recall_at_k_multi_relevant_full_hit() -> None:
    retrieved = ["a", "c", "b", "d"]
    relevant = {"a", "b"}
    assert recall_at_k(retrieved, relevant, k=10) == 1.0


def test_recall_at_k_handles_retrieved_shorter_than_k() -> None:
    """ANN's documented non-guarantee: `hybrid_search` may return fewer than `k`
    rows. Slicing a short list must not index-error or otherwise misbehave."""
    assert recall_at_k(["a"], {"a"}, k=10) == 1.0
    assert recall_at_k(["a"], {"a", "b"}, k=10) == 0.5


def test_recall_at_k_handles_empty_retrieved_list() -> None:
    assert recall_at_k([], {"a"}, k=10) == 0.0


def test_recall_at_k_raises_on_empty_relevant_set() -> None:
    with pytest.raises(ValueError):
        recall_at_k(["a", "b"], set(), k=10)


# ── mrr ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "retrieved,expected_rr",
    [
        (["a", "x", "y"], 1.0),       # hit at rank 1
        (["x", "a", "y"], 1 / 2),     # hit at rank 2 (middle)
        (["x", "y", "a"], 1 / 3),     # hit at rank 3 (last)
    ],
)
def test_mrr_reciprocal_rank_of_first_hit(
    retrieved: list[str], expected_rr: float
) -> None:
    assert mrr(retrieved, {"a"}) == pytest.approx(expected_rr)


def test_mrr_no_hit_is_zero() -> None:
    assert mrr(["x", "y", "z"], {"a"}) == 0.0


def test_mrr_handles_empty_retrieved_list() -> None:
    assert mrr([], {"a"}) == 0.0


def test_mrr_multi_relevant_uses_earliest_matching_rank() -> None:
    """With multiple relevant ids, MRR is driven by whichever one is found
    first in ranked order — not by a specific id."""
    retrieved = ["x", "b", "a", "y"]
    relevant = {"a", "b"}
    assert mrr(retrieved, relevant) == pytest.approx(1 / 2)  # "b" hits at rank 2


def test_mrr_raises_on_empty_relevant_set() -> None:
    with pytest.raises(ValueError):
        mrr(["a", "b"], set())
