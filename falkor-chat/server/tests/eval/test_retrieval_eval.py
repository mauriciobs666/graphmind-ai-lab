"""Retrieval-metrics eval suite (K-026, `docs/plans/graphrag-eval.md` §5 Unit 2b).

Both tests below need `ws:eval` seeded (the `ws_eval` fixture, `conftest.py`, D2)
and **skip cleanly (not fail)** whenever it isn't — see `conftest.py`'s docstring.
No LLM/network dependency: the only I/O is `Services.hybrid_search` against
FalkorDB, which the default offline suite already assumes reachable (AGENTS.md).
Golden-query vectors come from the committed, pre-cached
`golden_retrieval.embeddings.json` (D3) — never embedded live here.

Edge cases exercised by the real corpus/golden-set (plan §6):
  - multi-relevant golden pairs (e.g. `gr-34`, two `relevant_msgIds`) — `metrics.py`
    handles the set case; unit coverage for that logic itself lives in
    `test_metrics.py`, which is genuinely network-free and runs unconditionally.
  - `hybrid_search` returning fewer than `k` rows (ANN's documented non-guarantee,
    `repository.py`'s `hybrid_search` docstring) — `_aggregate_metrics` below never
    assumes a fixed-length result; `recall_at_k`/`mrr` handle a short list directly.

D6's compare-or-establish branching itself (the actual regression *gate*, not
the recall/MRR math) is factored into `metrics.check_regression` precisely so
it has its own network-free unit coverage in `test_metrics.py` — proving the
gate fails when it should, not just that it runs against a live corpus that
(in every real run here) compares against itself
(`docs/reviews/graphrag-eval.md` finding M-1).
`test_retrieval_metrics_meet_or_beat_baseline` below is a thin integration
wrapper: real `current`/`baseline` in, `check_regression` out.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from falkorchat import db
from falkorchat.config import CallContext
from falkorchat.repository import Repository
from falkorchat.services import Services

from metrics import check_regression, mrr, recall_at_k

_EVAL_DIR = Path(__file__).resolve().parent
_GOLDEN_PATH = _EVAL_DIR / "golden_retrieval.jsonl"
_EMBEDDINGS_PATH = _EVAL_DIR / "golden_retrieval.embeddings.json"
_BASELINE_PATH = _EVAL_DIR / "retrieval_baseline.json"

_K = 10
_K_SMALL = 5
_ACTOR_ID = "eval-harness"  # unused by hybrid_search's read-only Cypher; any id is fine

# D6: MRR is allowed to drop up to this much (relative) against the baseline before
# it's treated as a regression; recall@10 must never drop at all.
_MRR_REGRESSION_TOLERANCE = 0.05


def _load_golden_set() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with _GOLDEN_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_embedding_cache() -> dict[str, Any]:
    with _EMBEDDINGS_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def test_golden_msgids_exist_in_corpus(ws_eval: str) -> None:
    """Every `relevant_msgId` referenced by `golden_retrieval.jsonl` actually
    exists in `ws:eval` — catches a golden pair drifting out of sync with the
    corpus (e.g. after a corpus edit that renamed/removed a message)."""
    rows = _load_golden_set()
    all_ids = sorted({msg_id for row in rows for msg_id in row["relevant_msgIds"]})

    graph = db.workspace_graph(db.connect(), ws_eval)
    res = graph.ro_query(
        "MATCH (m:Message) WHERE m.msgId IN $ids RETURN m.msgId", {"ids": all_ids}
    )
    found = {row[0] for row in res.result_set}
    missing = sorted(set(all_ids) - found)
    assert not missing, (
        f"{len(missing)} golden relevant_msgId(s) not found in ws:{ws_eval}: "
        f"{missing} — the golden set has drifted out of sync with the corpus"
    )


def _aggregate_metrics(ws: str) -> dict[str, Any]:
    """One `hybrid_search` round-trip per golden query (k=10); recall@5 is sliced
    from the same ordered result, never a second call."""
    rows = _load_golden_set()
    cache = _load_embedding_cache()
    ctx = CallContext(ws=ws, actor=_ACTOR_ID)
    services = Services(Repository(db.connect()))

    recall_10_scores: list[float] = []
    recall_5_scores: list[float] = []
    mrr_scores: list[float] = []
    for row in rows:
        golden_id = row["id"]
        relevant = set(row["relevant_msgIds"])
        q_vec = cache[golden_id]["vector"]

        results = services.hybrid_search(ctx, q_vec=q_vec, k=_K, limit=_K)
        retrieved = [r["msgId"] for r in results]

        recall_10_scores.append(recall_at_k(retrieved, relevant, k=_K))
        recall_5_scores.append(recall_at_k(retrieved, relevant, k=_K_SMALL))
        mrr_scores.append(mrr(retrieved, relevant))

    n = len(rows)
    return {
        "recall_at_10": sum(recall_10_scores) / n,
        "recall_at_5": sum(recall_5_scores) / n,
        "mrr": sum(mrr_scores) / n,
        "n": n,
    }


def test_retrieval_metrics_meet_or_beat_baseline(ws_eval: str) -> None:
    """D6: recall@10 (primary)/recall@5/MRR over the golden set, aggregated by mean.

    First-run baseline-establish mode: if `retrieval_baseline.json` doesn't exist,
    compute it, write it, and pass — the method note's "first run establishes the
    numbers; it does not gate." On subsequent runs, compare against the committed
    baseline using the method note's acceptance rule (recall@10 >= baseline, MRR
    not down more than 5% relative) and fail loudly on a real regression.
    `UPDATE_EVAL_BASELINE=1` forces a deliberate re-baseline (mirrors `KEEP_WS`).

    NOTE: per D6, a freshly-written `retrieval_baseline.json` here is not yet
    load-bearing — it requires a separate `data-scientist` methodology sign-off
    before being treated as a trustworthy regression gate (see the plan / this
    unit's dispatch brief).
    """
    current = _aggregate_metrics(ws_eval)
    print(
        f"\nretrieval eval (n={current['n']}): "
        f"recall@10={current['recall_at_10']:.4f} "
        f"recall@5={current['recall_at_5']:.4f} "
        f"mrr={current['mrr']:.4f}"
    )

    if not _BASELINE_PATH.exists() or os.environ.get("UPDATE_EVAL_BASELINE") == "1":
        _BASELINE_PATH.write_text(
            json.dumps(current, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        return  # baseline-establish mode: unconditional pass, per D6

    with _BASELINE_PATH.open(encoding="utf-8") as f:
        baseline = json.load(f)

    reasons = check_regression(
        current, baseline, mrr_tolerance=_MRR_REGRESSION_TOLERANCE
    )
    assert not reasons, (
        "; ".join(reasons)
        + " — set UPDATE_EVAL_BASELINE=1 to accept a deliberate retrieval change"
    )
