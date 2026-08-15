"""Pure, network-free retrieval metrics (K-026, `docs/plans/graphrag-eval.md`
§5 Unit 2b).

Both functions operate on whatever ordered list of retrieved message ids they are
given — `Services.hybrid_search`'s own docstring is explicit that ANN recall is
approximate and may return fewer than `k` rows, so neither function assumes a
fixed-length input. A short (or empty) `retrieved_msg_ids` list is handled the same
way a full one is: slicing/iterating just stops at whatever length it actually has,
with no special-casing required.
"""

from __future__ import annotations


def recall_at_k(retrieved_msg_ids: list[str], relevant: set[str], k: int) -> float:
    """Fraction of `relevant` ids present in the first `k` of `retrieved_msg_ids`.

    Standard recall@k: ``|top-k ∩ relevant| / |relevant|``. `retrieved_msg_ids` may
    have fewer than `k` entries (ANN's documented non-guarantee, `repository.py`'s
    `hybrid_search` docstring) — the slice below simply stops at whatever length the
    list actually has, so a short result list is scored correctly with no extra
    branching.

    Raises `ValueError` if `relevant` is empty — a golden pair with no relevant ids
    is a fixture defect (every pair's `relevant_msgIds` must be non-empty, enforced
    by `test_golden_set_integrity.py`), not a legitimate "0 recall" retrieval
    outcome to silently score.
    """
    if not relevant:
        raise ValueError("relevant must be non-empty")
    top_k = retrieved_msg_ids[:k]
    hits = sum(1 for msg_id in top_k if msg_id in relevant)
    return hits / len(relevant)


def mrr(retrieved_msg_ids: list[str], relevant: set[str]) -> float:
    """Reciprocal rank (1-indexed) of the first id in `retrieved_msg_ids` that is a
    member of `relevant`.

    Returns ``0.0`` if none of `relevant` appears anywhere in `retrieved_msg_ids`
    (including the empty-list case — a real possibility given ANN's non-guarantee).
    With more than one relevant id (a multi-relevant golden pair), the score is
    driven by whichever relevant id is found earliest in ranked order.

    Raises `ValueError` if `relevant` is empty, for the same reason as
    `recall_at_k`.
    """
    if not relevant:
        raise ValueError("relevant must be non-empty")
    for rank, msg_id in enumerate(retrieved_msg_ids, start=1):
        if msg_id in relevant:
            return 1.0 / rank
    return 0.0
