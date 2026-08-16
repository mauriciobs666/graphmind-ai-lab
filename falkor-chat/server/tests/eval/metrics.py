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


def check_regression(
    current: dict[str, float],
    baseline: dict[str, float],
    *,
    mrr_tolerance: float,
) -> list[str]:
    """D6's regression-detection gate (`docs/plans/graphrag-eval.md` §3 D6):
    compare `current` retrieval metrics against a committed `baseline`.

    Two independent checks, both applied (never short-circuits on the first
    hit, so a run regressing on both axes reports both):

    - **recall@10 — zero tolerance.** `current["recall_at_10"]` must be
      `>= baseline["recall_at_10"]`; any drop is a regression.
    - **MRR — relative tolerance.** `current["mrr"]` must be
      `>= baseline["mrr"] * (1 - mrr_tolerance)`; a drop within that band is
      accepted as noise, not a regression.

    Returns a list of human-readable regression-reason strings — empty means
    "no regression, the gate passes." Pure and network-free: operates only on
    the two dicts given, no FalkorDB/corpus dependency, so it's fully
    unit-testable in isolation from the live `ws_eval` integration path that
    `test_retrieval_eval.py` drives this same logic through.
    """
    reasons: list[str] = []

    if current["recall_at_10"] < baseline["recall_at_10"]:
        reasons.append(
            f"recall@10 regressed: {current['recall_at_10']:.4f} < baseline "
            f"{baseline['recall_at_10']:.4f}"
        )

    mrr_floor = baseline["mrr"] * (1 - mrr_tolerance)
    if current["mrr"] < mrr_floor:
        reasons.append(
            f"MRR regressed more than {mrr_tolerance:.0%} relative: "
            f"{current['mrr']:.4f} < floor {mrr_floor:.4f} (baseline "
            f"{baseline['mrr']:.4f})"
        )

    return reasons
