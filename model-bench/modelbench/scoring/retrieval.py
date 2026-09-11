"""The embedder's `ItemScorer` — exact brute-force cosine retrieval plus a BM25 reference arm.

Design: `docs/plans/small-model-benchmarking-s3-spec.md` §6 (this module's full signature set),
§3.2 (the `prime`/`embed_text` extension), §3.3 (the `deterministic_arm` CLI hook), and
`docs/plans/small-model-benchmarking-ml.md` §5.1-§5.5 ("`-ml`", the ML-methodology note this
module's formulas are pinned against — read directly rather than re-derived).

A **module**, not a class: `score_item`/`aggregate`/`prime`/`embed_text`/`deterministic_arm` are
module-level functions that satisfy `runner.ItemScorer` structurally (spec §7.1's own docstring —
never `isinstance`-checked). `_state` is process-local, written once per run by `prime()` and read
by `score_item`/`aggregate`/`embed_text` — safe because one `run` process drives exactly one model
x one pack, never concurrently or re-entrantly (spec §3.2). A direct unit test resets it by calling
`prime()` again before each case.

**Resolution of the spec's own flagged ambiguity (S3 coordination, not re-litigated here):** the
corpus-side raw-norm diagnostic in `prime()` runs unconditionally, on a cache HIT as well as a
cache miss — it is a property of whichever vectors end up in memory, not of whether a network call
fired. It is stored in `_state["corpusNormDiagnostic"]` (never only logged) so a caller building a
self-check report (S3 Step 2) can read it back.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import re
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any

import modelbench
from modelbench.fingerprint import Fingerprint
from modelbench.lmstudio import EmbedResult, LMStudio, ModelInfo
from modelbench.packs import Pack
from modelbench.results import (
    BENCH_SCHEMA_VERSION,
    BinaryMetric,
    ContinuousMetric,
    DistributionSummary,
    ItemResult,
    ItemTiming,
    RetrievalAggregates,
    RunResult,
)
from modelbench.stats import LEVEL_P50, percentile

#: `-ml` §5.2's p10 figures — not one of `stats.LEVEL_*` (those are the paired-comparison note's
#: own level space, §11.2.2); this module's own rational level, over the same `percentile()`.
_LEVEL_P10: Fraction = Fraction(1, 10)

# ==================================================================================================
# §6.1 — pure arithmetic: recall@k / MRR / precision@k, L2 normalization + cosine, score separation
# ==================================================================================================


def recall_at_k(retrieved: Sequence[str], relevant: Sequence[str] | set[str], k: int) -> float:
    """`|top-k ∩ R| / |R|` (`-ml` §5.1, verbatim port of `metrics.recall_at_k`'s semantics).

    Raises `ValueError` if `relevant` is empty — a golden pair with no relevant ids is a fixture
    defect, never a legitimate zero."""
    relevant_set = set(relevant)
    if not relevant_set:
        raise ValueError("recall_at_k: `relevant` is empty (golden-set defect, -ml §5.1)")
    top_k = set(retrieved[:k])
    return len(top_k & relevant_set) / len(relevant_set)


def mrr(retrieved: Sequence[str], relevant: Sequence[str] | set[str]) -> float:
    """Reciprocal rank (1-indexed) of the first relevant id; `0.0` if none found (`-ml` §5.1,
    verbatim port of `metrics.mrr`'s semantics). Raises `ValueError` if `relevant` is empty."""
    relevant_set = set(relevant)
    if not relevant_set:
        raise ValueError("mrr: `relevant` is empty (golden-set defect, -ml §5.1)")
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank
    return 0.0


def precision_at_k(retrieved: Sequence[str], relevant: Sequence[str] | set[str], k: int) -> float:
    """`|top-k ∩ R| / k` (`-ml` §5.1's own definition, not in `metrics.py`). `k > 0` always (pack
    config, never 0); no `ValueError` branch of its own — `recall_at_k`'s already covers the
    shared `relevant`-empty precondition, and both are always called on the same item."""
    relevant_set = set(relevant)
    top_k = set(retrieved[:k])
    return len(top_k & relevant_set) / k


def l2_norm(vector: Sequence[float]) -> float:
    return math.sqrt(sum(x * x for x in vector))


def l2_normalize(vector: Sequence[float]) -> tuple[float, ...]:
    """Returns `vector` unchanged, never divides, iff its norm is `0.0` — a genuinely zero
    embedding is a harness defect to surface downstream, not a `ZeroDivisionError` here."""
    norm = l2_norm(vector)
    if norm == 0.0:
        return tuple(vector)
    return tuple(x / norm for x in vector)


@dataclass(frozen=True)
class RankedList:
    """Every corpus doc, sorted by score descending — no truncation before scoring (§3.8.1: exact
    search needs every score for `sep_raw`/`sep_z`, and brute force gives them for free). Ties
    break on `docId` ascending, for a deterministic order no downstream reader has to guess at."""

    docIds: tuple[str, ...]
    scores: dict[str, float]


def rank_by_cosine(
    query_vector: Sequence[float], corpus_vectors: Mapping[str, Sequence[float]]
) -> RankedList:
    """Brute-force exact cosine over the WHOLE corpus (§3.8.1: no ANN, no FalkorDB, no top-k
    truncation). L2-normalizes `query_vector` and every corpus vector internally (`-ml` §5.1: not
    optional — raw magnitude differences would corrupt score separation without touching
    ranking)."""
    normalized_query = l2_normalize(query_vector)
    scores = {
        doc_id: sum(q * d for q, d in zip(normalized_query, l2_normalize(vector), strict=True))
        for doc_id, vector in corpus_vectors.items()
    }
    doc_ids = tuple(sorted(scores, key=lambda d: (-scores[d], d)))
    return RankedList(docIds=doc_ids, scores=scores)


def score_separation_raw(scores: Mapping[str, float], relevant: Sequence[str] | set[str]) -> float:
    """`max_{d in R(q)} score(d) - max_{d not in R(q)} score(d)` (`-ml` §5.2). Raises `ValueError`
    if `relevant` is empty, if no relevant doc from `relevant` appears in `scores`, or if `scores`
    names no non-relevant doc — all golden-set/corpus defects, never a legitimate outcome (mirrors
    `recall_at_k`'s discipline)."""
    relevant_set = set(relevant)
    if not relevant_set:
        raise ValueError("score_separation_raw: `relevant` is empty (golden-set defect, -ml §5.2)")
    relevant_scores = [score for doc_id, score in scores.items() if doc_id in relevant_set]
    nonrelevant_scores = [score for doc_id, score in scores.items() if doc_id not in relevant_set]
    if not relevant_scores:
        raise ValueError(
            "score_separation_raw: no doc in `relevant` appears in `scores` (golden-set defect, "
            "-ml §5.2)"
        )
    if not nonrelevant_scores:
        raise ValueError(
            "score_separation_raw: `scores` names no non-relevant doc (corpus defect, -ml §5.2)"
        )
    return max(relevant_scores) - max(nonrelevant_scores)


def score_separation_z(sep_raw_value: float, scores: Mapping[str, float]) -> float:
    """`sep_raw_value / sd({cos(q, d) : d in corpus})` (`-ml` §5.2).

    `sd` is the **population** standard deviation (`statistics.pstdev`, `-ml` v1.24 ruling) — the
    121-doc corpus is enumerated in full and scored exactly for every query, not a sample drawn
    from a larger, unobserved population, which is the condition that calls for Bessel's
    correction (sample stdev, `ddof=1`)."""
    return sep_raw_value / statistics.pstdev(scores.values())


# ==================================================================================================
# §6.1 — BM25 (`-ml` §5.3, all parameters from `pack.json`'s `embedding.bm25` block)
# ==================================================================================================

_BM25_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def bm25_tokenize(text: str) -> list[str]:
    """Unicode-aware `re.findall(r"\\w+", text.casefold())` (`-ml` §5.3 verbatim). No stemming."""
    return _BM25_TOKEN_RE.findall(text.casefold())


@dataclass(frozen=True)
class Bm25Index:
    docIds: tuple[str, ...]
    docTermFreqs: tuple[Mapping[str, int], ...]  # per doc, post-stopword-removal
    docLengths: tuple[int, ...]
    avgDocLength: float
    documentFrequency: Mapping[str, int]  # term -> number of docs containing it
    n: int
    k1: float
    b: float


def build_bm25_index(
    corpus_rows: Sequence[Mapping[str, str]], *, k1: float, b: float, stopwords: frozenset[str]
) -> Bm25Index:
    doc_ids: list[str] = []
    doc_term_freqs: list[dict[str, int]] = []
    doc_lengths: list[int] = []
    document_frequency: dict[str, int] = {}

    for row in corpus_rows:
        tokens = [t for t in bm25_tokenize(row["text"]) if t not in stopwords]
        term_freq: dict[str, int] = {}
        for token in tokens:
            term_freq[token] = term_freq.get(token, 0) + 1
        doc_ids.append(row["docId"])
        doc_term_freqs.append(term_freq)
        doc_lengths.append(len(tokens))
        for term in term_freq:
            document_frequency[term] = document_frequency.get(term, 0) + 1

    n = len(corpus_rows)
    avg_doc_length = sum(doc_lengths) / n if n else 0.0
    return Bm25Index(
        docIds=tuple(doc_ids),
        docTermFreqs=tuple(doc_term_freqs),
        docLengths=tuple(doc_lengths),
        avgDocLength=avg_doc_length,
        documentFrequency=document_frequency,
        n=n,
        k1=k1,
        b=b,
    )


def bm25_score(index: Bm25Index, query: str) -> dict[str, float]:
    """Every doc's score for `query`. Always-positive IDF (`-ml` §5.3 verbatim):
    `idf(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))`."""
    scores: dict[str, float] = {doc_id: 0.0 for doc_id in index.docIds}
    query_terms = set(bm25_tokenize(query))
    for term in query_terms:
        df = index.documentFrequency.get(term, 0)
        idf = math.log(1.0 + (index.n - df + 0.5) / (df + 0.5))
        for doc_id, term_freq, doc_length in zip(
            index.docIds, index.docTermFreqs, index.docLengths, strict=True
        ):
            tf = term_freq.get(term, 0)
            if tf == 0:
                continue
            denom = tf + index.k1 * (1 - index.b + index.b * doc_length / index.avgDocLength)
            scores[doc_id] += idf * (tf * (index.k1 + 1)) / denom
    return scores


def load_stopwords(pack: Pack) -> frozenset[str]:
    """Reads `pack.manifest["embedding"]["bm25"]["stopwordsFile"]` via `pack.root / <name>` — a
    plain-text, one-word-per-line file, `#`-prefixed comment lines and blank lines skipped."""
    name = pack.manifest["embedding"]["bm25"]["stopwordsFile"]
    text = (pack.root / name).read_text(encoding="utf-8")
    words: set[str] = set()
    for line in text.splitlines():
        line = line.strip().casefold()
        if not line or line.startswith("#"):
            continue
        words.add(line)
    return frozenset(words)


# ==================================================================================================
# §6.2 — prime / embed_text / score_item / aggregate
# ==================================================================================================

# Process-local state, written once per run by prime(), read by score_item/aggregate/embed_text.
# Safe under one-run-one-process (module docstring); a direct-unit-test caller resets it by calling
# prime() again before each case.
_state: dict[str, Any] = {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _compute_cache_key(
    *, model: str, quantization: str, document_prefix: str, corpus_bytes: bytes
) -> dict[str, str]:
    """`-ml` §5.5's cache key, all four components: `(model id, quantization, docPrefix, corpus
    version)`, `corpus version` realized as a sha256 of the corpus data's own bytes. Computed
    independently of `scripts/refresh_golden.py`'s own `compute_cache_key` (same shape, deliberately
    not imported — this package does not depend on the top-level `scripts/` directory)."""
    return {
        "model": model,
        "quantization": quantization,
        "documentPrefix": document_prefix,
        "corpusSha256": hashlib.sha256(corpus_bytes).hexdigest(),
    }


def _norm_diagnostic(norms: Sequence[float]) -> dict[str, Any]:
    if not norms:
        return {"median": None, "p10": None, "n": 0}
    return {
        "median": percentile(norms, level=LEVEL_P50),
        "p10": percentile(norms, level=_LEVEL_P10),
        "n": len(norms),
    }


def prime(
    *,
    lmstudio: LMStudio,
    model_info: ModelInfo,
    call_surface: str,
    timeout_s: float,
    pack: Pack,
) -> None:
    """Embeds/caches the 121-document corpus once, via the SAME live model under test (§3.1).

    Computes this run's cache key and compares it against the committed `corpus.embeddings.json`
    header. On a match: reuses the committed RAW vectors, no live call for the corpus. On a
    mismatch or an absent file: embeds `corpus.jsonl` live (`documentPrefix`-applied, one batched
    `lmstudio.embed()` call) — the result is NOT persisted.

    Also builds the BM25 index (§6.1) and records the raw corpus-norm distribution — computed
    UNCONDITIONALLY, on a cache hit as well as a miss (module docstring's own resolution)."""
    corpus_path = pack.data_path("corpus")
    corpus_rows = _read_jsonl(corpus_path)
    corpus_bytes = corpus_path.read_bytes()
    document_prefix = pack.manifest["embedding"]["documentPrefix"]

    cache_key = _compute_cache_key(
        model=model_info.id,
        quantization=model_info.quantization,
        document_prefix=document_prefix,
        corpus_bytes=corpus_bytes,
    )

    committed: Mapping[str, Any] | None = None
    committed_path = pack.data_path("corpusEmbeddings")
    if committed_path.exists():
        committed = json.loads(committed_path.read_text(encoding="utf-8"))

    if committed is not None and committed.get("cacheKey") == cache_key:
        raw_vectors: dict[str, tuple[float, ...]] = {
            doc_id: tuple(vector) for doc_id, vector in committed["vectors"].items()
        }
    else:
        texts = [document_prefix + row["text"] for row in corpus_rows]
        embed_result: EmbedResult = lmstudio.embed(texts, model=model_info.id, timeout_s=timeout_s)
        raw_vectors = {
            row["docId"]: vector
            for row, vector in zip(corpus_rows, embed_result.vectors, strict=True)
        }

    bm25_conf = pack.manifest["embedding"]["bm25"]
    stopwords = load_stopwords(pack)
    bm25_index = build_bm25_index(
        corpus_rows, k1=bm25_conf["k1"], b=bm25_conf["b"], stopwords=stopwords
    )

    norm_diagnostic = _norm_diagnostic([l2_norm(v) for v in raw_vectors.values()])

    _state.clear()
    _state["cacheKey"] = cache_key
    _state["corpusVectors"] = raw_vectors
    _state["bm25Index"] = bm25_index
    _state["corpusNormDiagnostic"] = norm_diagnostic


def embed_text(item_input: Mapping[str, Any], *, pack: Pack) -> str:
    """`pack.manifest["embedding"]["queryPrefix"] + item_input["query"]` — the exact string the
    runner embeds for this item (spec §3.2)."""
    return pack.manifest["embedding"]["queryPrefix"] + item_input["query"]


def _score_ranking(
    item_id: str,
    ranked_ids: Sequence[str],
    scores: Mapping[str, float],
    relevant: set[str],
    *,
    pack: Pack,
    timing: ItemTiming | None,
    detail: Mapping[str, Any],
) -> ItemResult:
    """The one scoring path both arms use (§6.3: "one scoring path, two arms") — `score_item`
    reaches it after `rank_by_cosine`, `deterministic_arm` after `bm25_score`."""
    ks = pack.manifest["metrics"]["recallAtK"]["ks"]
    counts: dict[str, int] = {}
    scoreable: dict[str, bool] = {}
    for k in ks:
        hits = len(set(ranked_ids[:k]) & relevant)
        counts[f"recallAt{k}"] = hits
        scoreable[f"recallAt{k}"] = True
    counts["precisionAt1"] = len(set(ranked_ids[:1]) & relevant)
    scoreable["precisionAt1"] = True

    mrr_value = mrr(list(ranked_ids), relevant)
    sep_raw_value = score_separation_raw(scores, relevant)
    sep_z_value = score_separation_z(sep_raw_value, scores)
    scoreable["mrr"] = True
    scoreable["separationRaw"] = True
    scoreable["separationZ"] = True

    return ItemResult(
        itemId=item_id,
        pairingKey=(item_id,),
        outcome="pass",
        scoreable=scoreable,
        counts=counts,
        timing=timing,
        measures={"mrr": mrr_value, "separationRaw": sep_raw_value, "separationZ": sep_z_value},
        detail=detail,
    )


def score_item(
    item_input: Mapping[str, Any],
    result: EmbedResult | None,
    timing: ItemTiming,
    *,
    pack: Pack,
) -> ItemResult:
    """One query. `result is None` (timeout/no-response) scores `outcome="fail"`/`"unrunnable"`
    per `timing.withheldFor`, `scoreable={}` — no `measures`/`counts` at all (a query the model
    never answered contributes nothing to recall@k/MRR: there IS no ranking to score)."""
    item_id = item_input["itemId"]
    if result is None:
        outcome = "fail" if timing.withheldFor == "timeout" else "unrunnable"
        return ItemResult(
            itemId=item_id,
            pairingKey=(item_id,),
            outcome=outcome,
            scoreable={},
            counts={},
            timing=timing,
        )

    if "corpusVectors" not in _state:
        raise RuntimeError(
            "scoring.retrieval.score_item: prime() was never called — this is a programming-"
            "contract failure, not a pack-config one"
        )

    corpus_vectors = _state["corpusVectors"]
    query_vector = result.vectors[0]
    ranked = rank_by_cosine(query_vector, corpus_vectors)
    relevant = set(item_input["relevantDocIds"])

    return _score_ranking(
        item_id,
        ranked.docIds,
        ranked.scores,
        relevant,
        pack=pack,
        timing=timing,
        detail={"queryNorm": l2_norm(query_vector)},
    )


def aggregate(items: Sequence[ItemResult], *, pack: Pack) -> RetrievalAggregates:
    """One pass over `items` (mirrors `LatencyBlock`'s "computed from run.items in one pass"
    discipline). Every item lacking an `"mrr"` entry in `measures` (a timed-out/unrunnable query)
    is excluded from all five aggregates' inputs, never coerced to 0."""
    scored = [it for it in items if "mrr" in it.measures]
    ks = pack.manifest["metrics"]["recallAtK"]["ks"]

    recall_at_k_metrics = tuple(
        BinaryMetric(
            name=f"recallAt{k}",
            successes=sum(1 for it in scored if it.counts.get(f"recallAt{k}", 0) > 0),
            n=len(scored),
            unit="query",
        )
        for k in ks
    )
    precision_at_1 = BinaryMetric(
        name="precisionAt1",
        successes=sum(1 for it in scored if it.counts.get("precisionAt1", 0) > 0),
        n=len(scored),
        unit="query",
    )

    if not scored:
        return RetrievalAggregates(
            recallAtK=recall_at_k_metrics,
            mrr=None,
            precisionAt1=precision_at_1,
            separationRaw=None,
            separationZ=None,
        )

    mrr_values = [it.measures["mrr"] for it in scored]
    mrr_metric = ContinuousMetric(
        name="mrr", mean=statistics.fmean(mrr_values), n=len(scored), support=(0.0, 1.0)
    )

    sep_raw_values = [it.measures["separationRaw"] for it in scored]
    sep_z_values = [it.measures["separationZ"] for it in scored]
    separation_raw = DistributionSummary(
        name="separationRaw",
        median=percentile(sep_raw_values, level=LEVEL_P50),
        p10=percentile(sep_raw_values, level=_LEVEL_P10),
        n=len(sep_raw_values),
        unit="query",
        support=(-1.0, 1.0),
    )
    separation_z = DistributionSummary(
        name="separationZ",
        median=percentile(sep_z_values, level=LEVEL_P50),
        p10=percentile(sep_z_values, level=_LEVEL_P10),
        n=len(sep_z_values),
        unit="query",
        support=None,
    )

    return RetrievalAggregates(
        recallAtK=recall_at_k_metrics,
        mrr=mrr_metric,
        precisionAt1=precision_at_1,
        separationRaw=separation_raw,
        separationZ=separation_z,
    )


# ==================================================================================================
# §6.3 — the BM25 deterministic arm
# ==================================================================================================


def _utc_stamp(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _build_deterministic_fingerprint(
    *, pack: Pack, started_at: datetime, ended_at: datetime
) -> Fingerprint:
    """The eleven required `deterministic`-profile fields (`fingerprint.py`'s own
    `_DETERMINISTIC_SCHEMA_1`). `armParametersHash` covers the arm's OWN knobs (k1, b,
    tokenization, IDF variant, stopwords) — deliberately not `packContentHash` again, which is a
    separate required field already covering the whole pack."""
    bm25_conf = pack.manifest["embedding"]["bm25"]
    stopwords_sha256 = hashlib.sha256(
        (pack.root / bm25_conf["stopwordsFile"]).read_bytes()
    ).hexdigest()
    params = {
        "k1": bm25_conf["k1"],
        "b": bm25_conf["b"],
        "tokenization": "unicode-word-casefold",
        "idfVariant": "always-positive",
        "stopwordsSha256": stopwords_sha256,
    }
    arm_parameters_hash = hashlib.sha256(
        json.dumps(params, sort_keys=True).encode("utf-8")
    ).hexdigest()
    fields = {
        "armId": "bm25",
        "armParametersHash": arm_parameters_hash,
        "packId": pack.packId,
        "packVersion": pack.packVersion,
        "packContentHash": pack.contentHash,
        "benchVersion": modelbench.__version__,
        "benchSchemaVersion": BENCH_SCHEMA_VERSION,
        "pythonVersion": platform.python_version(),
        "hostOs": platform.platform(),
        "startedAt": _utc_stamp(started_at),
        "endedAt": _utc_stamp(ended_at),
    }
    return Fingerprint(armKind="deterministic", callSurface=None, fields=fields)


def deterministic_arm(*, pack: Pack, session_id: str | None) -> RunResult:
    """§3.3's resolution: builds the BM25 reference-arm `RunResult` with NO live LM Studio call.
    `ItemResult.timing` is `None` on every item (BM25 makes no model call, so there is nothing to
    time); `RunResult.latency` is therefore `None` too."""
    started_at = datetime.now(timezone.utc)

    corpus_rows = _read_jsonl(pack.data_path("corpus"))
    query_rows = list(pack.iter_items())

    bm25_conf = pack.manifest["embedding"]["bm25"]
    stopwords = load_stopwords(pack)
    index = build_bm25_index(
        corpus_rows, k1=bm25_conf["k1"], b=bm25_conf["b"], stopwords=stopwords
    )

    items: list[ItemResult] = []
    for row in query_rows:
        scores = bm25_score(index, row["query"])
        ranked_ids = tuple(sorted(scores, key=lambda d: (-scores[d], d)))
        relevant = set(row["relevantDocIds"])
        items.append(
            _score_ranking(
                row["itemId"], ranked_ids, scores, relevant, pack=pack, timing=None, detail={}
            )
        )

    ended_at = datetime.now(timezone.utc)
    aggregates = aggregate(tuple(items), pack=pack)
    fingerprint = _build_deterministic_fingerprint(
        pack=pack, started_at=started_at, ended_at=ended_at
    )
    run_id = f"{pack.packId}-bm25-{_utc_stamp(started_at)}"

    return RunResult(
        runId=run_id,
        sessionId=session_id,
        role=pack.role,
        armKind="deterministic",
        fingerprint=fingerprint,
        items=tuple(items),
        aggregates=aggregates,
        designEffect=1.0,
        basis="by-construction",
        attestationTripWire=None,
        latency=None,
    )
