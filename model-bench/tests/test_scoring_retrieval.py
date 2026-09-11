"""`modelbench.scoring.retrieval` — the embedder's `ItemScorer` (S3 spec §6, §8 Step 1).

Order, red -> green, matches the spec's own step sequence (§8 Step 1):

1. Test 8's hand-built-ranked-list cases (§6.1) — pure arithmetic, no pack/corpus/live call.
2. BM25 unit tests against a small synthetic corpus with a hand-computed expected score.
3. `prime()`/`embed_text()` unit tests — cache-key match/mismatch, `StubLMStudio`.
4. `score_item`/`aggregate` unit tests.
5. `deterministic_arm()` unit test.

`tests/test_metrics_agreement.py` (the 20-case transcribed agreement fixture) is a separate file,
per the spec's own file layout (§4).
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any

import pytest

from modelbench.lmstudio import EmbedResult, ModelInfo
from modelbench.packs import Pack, load_pack
from modelbench.results import ItemTiming
from modelbench.scoring import retrieval

# ==================================================================================================
# 1. Pure arithmetic (§6.1) — hand-built ranked lists, no pack/corpus/live call (test 8)
# ==================================================================================================


class TestRecallAtK:
    def test_single_relevant_hit_within_k(self):
        assert retrieval.recall_at_k(["a", "b", "c"], {"a"}, k=10) == 1.0

    def test_single_relevant_no_hit(self):
        assert retrieval.recall_at_k(["x", "y", "z"], {"a"}, k=10) == 0.0

    def test_multi_relevant_partial_hit(self):
        assert retrieval.recall_at_k(["a", "c", "d"], {"a", "b"}, k=10) == 0.5

    def test_multi_relevant_full_hit(self):
        assert retrieval.recall_at_k(["a", "c", "b", "d"], {"a", "b"}, k=10) == 1.0

    def test_fewer_than_k_results(self):
        """`retrieved` has fewer than `k` entries — recall is over whatever was retrieved."""
        assert retrieval.recall_at_k(["a"], {"a", "b"}, k=10) == 0.5

    def test_zero_relevant_found(self):
        assert retrieval.recall_at_k([], {"a"}, k=10) == 0.0

    def test_raises_on_empty_relevant_set(self):
        with pytest.raises(ValueError):
            retrieval.recall_at_k(["a", "b"], set(), k=10)

    def test_hit_outside_top_k_window_excluded(self):
        assert retrieval.recall_at_k(["x", "y", "z", "w", "v", "a"], {"a"}, k=5) == 0.0

    def test_hit_included_at_larger_k(self):
        assert retrieval.recall_at_k(["x", "y", "z", "w", "v", "a"], {"a"}, k=10) == 1.0


class TestMrr:
    def test_hit_at_rank_1(self):
        assert retrieval.mrr(["a", "x", "y"], {"a"}) == 1.0

    def test_hit_at_rank_2(self):
        assert retrieval.mrr(["x", "a", "y"], {"a"}) == 0.5

    def test_hit_at_rank_3(self):
        assert retrieval.mrr(["x", "y", "a"], {"a"}) == pytest.approx(1.0 / 3.0)

    def test_no_hit_is_zero(self):
        assert retrieval.mrr(["x", "y", "z"], {"a"}) == 0.0

    def test_zero_relevant_found_empty_retrieved(self):
        assert retrieval.mrr([], {"a"}) == 0.0

    def test_multi_relevant_uses_earliest_rank(self):
        assert retrieval.mrr(["x", "b", "a", "y"], {"a", "b"}) == 0.5

    def test_raises_on_empty_relevant_set(self):
        with pytest.raises(ValueError):
            retrieval.mrr(["a", "b"], set())


class TestPrecisionAtK:
    def test_single_relevant_hit(self):
        assert retrieval.precision_at_k(["a", "b", "c"], {"a"}, k=3) == pytest.approx(1.0 / 3.0)

    def test_multi_relevant_partial(self):
        result = retrieval.precision_at_k(["a", "c", "d"], {"a", "b"}, k=3)
        assert result == pytest.approx(1.0 / 3.0)

    def test_multi_relevant_full_hit_at_k1(self):
        assert retrieval.precision_at_k(["a", "c", "b", "d"], {"a", "b"}, k=1) == 1.0

    def test_fewer_than_k_results(self):
        """`retrieved` shorter than `k` — the denominator is still `k`, per `-ml` §5.1's
        `|top-k ∩ R| / k` definition (never rescaled to `len(retrieved)`)."""
        assert retrieval.precision_at_k(["a"], {"a"}, k=10) == pytest.approx(1.0 / 10.0)

    def test_zero_relevant_found(self):
        assert retrieval.precision_at_k(["x", "y"], {"a"}, k=2) == 0.0


class TestL2NormAndNormalize:
    def test_l2_norm(self):
        assert retrieval.l2_norm([3.0, 4.0]) == pytest.approx(5.0)

    def test_l2_normalize_unit_length(self):
        normalized = retrieval.l2_normalize([3.0, 4.0])
        assert retrieval.l2_norm(normalized) == pytest.approx(1.0)
        assert normalized == pytest.approx((0.6, 0.8))

    def test_l2_normalize_zero_vector_returned_unchanged(self):
        """A genuinely zero embedding is a harness defect to surface downstream (score would be
        garbage), never a `ZeroDivisionError` here (§6.1)."""
        assert retrieval.l2_normalize([0.0, 0.0]) == (0.0, 0.0)


class TestRankByCosine:
    def test_ranks_by_descending_cosine_similarity(self):
        query = [1.0, 0.0]
        corpus = {
            "close": [1.0, 0.0],
            "orthogonal": [0.0, 1.0],
            "opposite": [-1.0, 0.0],
        }
        ranked = retrieval.rank_by_cosine(query, corpus)
        assert ranked.docIds == ("close", "orthogonal", "opposite")
        assert ranked.scores["close"] == pytest.approx(1.0)
        assert ranked.scores["orthogonal"] == pytest.approx(0.0)
        assert ranked.scores["opposite"] == pytest.approx(-1.0)

    def test_normalizes_query_and_corpus_vectors_internally(self):
        """Unnormalized inputs (different magnitudes) still rank by cosine, not by raw dot
        product — a longer but less-aligned vector must not out-rank a shorter, better-aligned
        one."""
        query = [2.0, 0.0]
        corpus = {
            "aligned_short": [1.0, 0.0],
            "misaligned_long": [10.0, 9.0],
        }
        ranked = retrieval.rank_by_cosine(query, corpus)
        assert ranked.docIds[0] == "aligned_short"

    def test_no_truncation_every_corpus_doc_scored(self):
        query = [1.0, 0.0]
        corpus = {f"d{i}": [float(i), 1.0] for i in range(20)}
        ranked = retrieval.rank_by_cosine(query, corpus)
        assert len(ranked.docIds) == 20
        assert set(ranked.docIds) == set(corpus)
        assert set(ranked.scores) == set(corpus)


class TestScoreSeparationRaw:
    def test_positive_when_top_hit_is_relevant(self):
        scores = {"rel": 0.9, "irrel1": 0.5, "irrel2": 0.3}
        assert retrieval.score_separation_raw(scores, {"rel"}) == pytest.approx(0.4)

    def test_negative_when_irrelevant_doc_outranks_every_relevant_one(self):
        scores = {"rel": 0.3, "irrel1": 0.9, "irrel2": 0.5}
        assert retrieval.score_separation_raw(scores, {"rel"}) == pytest.approx(-0.6)

    def test_uses_max_over_multi_relevant_docs(self):
        scores = {"rel1": 0.2, "rel2": 0.8, "irrel": 0.5}
        assert retrieval.score_separation_raw(scores, {"rel1", "rel2"}) == pytest.approx(0.3)

    def test_raises_on_empty_relevant(self):
        with pytest.raises(ValueError):
            retrieval.score_separation_raw({"a": 0.5, "b": 0.4}, set())

    def test_raises_when_scores_names_no_non_relevant_doc(self):
        with pytest.raises(ValueError):
            retrieval.score_separation_raw({"a": 0.5, "b": 0.4}, {"a", "b"})


class TestScoreSeparationZ:
    def test_population_stdev_ruling(self):
        """`-ml` v1.24 ruling: population stdev (`statistics.pstdev`), never sample (`ddof=1`) —
        the 121-doc corpus is the full population each query is scored against."""
        scores = {"a": 0.9, "b": 0.5, "c": 0.3, "d": 0.1}
        sep_raw = retrieval.score_separation_raw(scores, {"a"})
        import statistics

        expected = sep_raw / statistics.pstdev(scores.values())
        assert retrieval.score_separation_z(sep_raw, scores) == pytest.approx(expected)
        # And NOT the sample stdev (ddof=1) — the mutant this pins.
        sample_based = sep_raw / statistics.stdev(scores.values())
        assert retrieval.score_separation_z(sep_raw, scores) != pytest.approx(sample_based)


# ==================================================================================================
# 2. BM25 (§6.1, `-ml` §5.3) — synthetic corpus, hand-computed expected score
# ==================================================================================================


_STOPWORDS = frozenset({"the", "a", "an", "of", "to"})

_BM25_CORPUS = [
    {"docId": "d1", "text": "the cat sat on the mat"},
    {"docId": "d2", "text": "the dog sat on a log"},
    {"docId": "d3", "text": "cats and dogs are great pets"},
]


class TestBm25Tokenize:
    def test_casefolds_and_splits_on_word_boundaries(self):
        assert retrieval.bm25_tokenize("Cats, Dogs & Mice!") == ["cats", "dogs", "mice"]

    def test_unicode_word_characters_are_kept(self):
        assert retrieval.bm25_tokenize("café résumé") == ["café", "résumé"]


class TestBuildBm25Index:
    def test_removes_stopwords_and_counts_term_frequencies(self):
        index = retrieval.build_bm25_index(_BM25_CORPUS, k1=1.2, b=0.75, stopwords=_STOPWORDS)
        assert index.docIds == ("d1", "d2", "d3")
        # "the" (2x in d1) and "on"/"a" are stopword-filtered per-doc as declared, except "on" is
        # not in the stopword list here, so it survives.
        assert index.docTermFreqs[0] == {"cat": 1, "sat": 1, "on": 1, "mat": 1}
        assert index.docLengths[0] == 4  # "the","the" removed; "cat sat on mat" remains
        assert index.n == 3
        assert index.documentFrequency["sat"] == 2  # d1, d2
        assert index.documentFrequency["cat"] == 1  # d1 only

    def test_avg_doc_length_is_mean_of_doc_lengths(self):
        index = retrieval.build_bm25_index(_BM25_CORPUS, k1=1.2, b=0.75, stopwords=_STOPWORDS)
        assert index.avgDocLength == pytest.approx(sum(index.docLengths) / 3)


class TestBm25Score:
    def test_hand_computed_single_term_query(self):
        """A single-document-frequency term ("mat", df=1, only in d1) against a 3-doc corpus,
        hand-computed against `-ml` §5.3's always-positive IDF: `idf = ln(1 + (N-df+0.5)/(df+0.5))`
        and Okapi's `tf*(k1+1) / (tf + k1*(1-b+b*|d|/avgdl))`."""
        index = retrieval.build_bm25_index(_BM25_CORPUS, k1=1.2, b=0.75, stopwords=_STOPWORDS)
        scores = retrieval.bm25_score(index, "mat")

        n, df = 3, 1
        idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
        tf = 1
        doc_length = index.docLengths[0]
        avg_len = index.avgDocLength
        k1, b = 1.2, 0.75
        denom = tf + k1 * (1 - b + b * doc_length / avg_len)
        expected_d1 = idf * (tf * (k1 + 1)) / denom

        assert scores["d1"] == pytest.approx(expected_d1)
        assert scores["d2"] == 0.0
        assert scores["d3"] == 0.0

    def test_term_absent_from_corpus_scores_every_doc_zero(self):
        index = retrieval.build_bm25_index(_BM25_CORPUS, k1=1.2, b=0.75, stopwords=_STOPWORDS)
        scores = retrieval.bm25_score(index, "xylophone")
        assert all(v == 0.0 for v in scores.values())

    def test_idf_is_always_positive_even_for_a_common_term(self):
        """`-ml` §5.3: the classic IDF goes negative when a term appears in >N/2 docs; the
        always-positive `+1` variant never does. "sat" appears in 2 of 3 docs here."""
        index = retrieval.build_bm25_index(_BM25_CORPUS, k1=1.2, b=0.75, stopwords=_STOPWORDS)
        scores = retrieval.bm25_score(index, "sat")
        assert scores["d1"] > 0.0
        assert scores["d2"] > 0.0
        assert scores["d3"] == 0.0

    def test_stopword_query_term_contributes_nothing(self):
        """A stopword never appears in any doc's `docTermFreqs` (filtered at index time), so its
        `tf` is 0 everywhere and it contributes 0 to every doc's score regardless of its
        (very high, df=0) idf — no explicit stopword filtering needed in `bm25_score` itself."""
        index = retrieval.build_bm25_index(_BM25_CORPUS, k1=1.2, b=0.75, stopwords=_STOPWORDS)
        scores_with_stopword = retrieval.bm25_score(index, "the mat")
        scores_without = retrieval.bm25_score(index, "mat")
        assert scores_with_stopword == pytest.approx(scores_without)


class TestLoadStopwords:
    def test_reads_one_word_per_line_skipping_comments_and_blanks(self, tmp_path):
        (tmp_path / "stop.txt").write_text("# comment\na\nthe\n\nAN\n")
        pack = _FakePackForStopwords(tmp_path)
        words = retrieval.load_stopwords(pack)
        assert words == frozenset({"a", "the", "an"})


class _FakePackForStopwords:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.manifest = {"embedding": {"bm25": {"stopwordsFile": "stop.txt"}}}


# ==================================================================================================
# 3. `prime()` / `embed_text()` (§6.2) — cache-key match/mismatch, `StubLMStudio`
# ==================================================================================================


def _write_corpus(tmp_path: Path, rows: list[dict[str, str]]) -> Path:
    path = tmp_path / "corpus.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def _write_stopwords(tmp_path: Path) -> None:
    (tmp_path / "bm25_stopwords.txt").write_text("the\na\n", encoding="utf-8")


def _embedder_pack(tmp_path: Path, *, document_prefix: str = "", query_prefix: str = "") -> Pack:
    manifest: dict[str, Any] = {
        "packId": "embedder-test-fixture",
        "packVersion": "1.0.0",
        "role": "embedder",
        "data": {
            "items": "queries.jsonl",
            "corpus": "corpus.jsonl",
            "corpusEmbeddings": "corpus.embeddings.json",
        },
        "embedding": {
            "queryPrefix": query_prefix,
            "documentPrefix": document_prefix,
            "bm25": {"k1": 1.2, "b": 0.75, "stopwordsFile": "bm25_stopwords.txt"},
        },
        "metrics": {"recallAtK": {"ks": [1, 2]}},
    }
    return Pack(
        packId=manifest["packId"],
        packVersion=manifest["packVersion"],
        role="embedder",
        contentHash="e" * 64,
        manifest=manifest,
        root=tmp_path,
    )


def _model_info(*, model_id: str = "text-embedding-test", quantization: str = "Q8_0") -> ModelInfo:
    return ModelInfo(
        id=model_id,
        object="model",
        type="embeddings",
        publisher="pub",
        arch="arch",
        compatibility_type="gguf",
        quantization=quantization,
        state="loaded",
        max_context_length=8192,
        capabilities=None,
        loaded_context_length=8192,
    )


class _StubEmbedLMStudio:
    def __init__(self, *, vectors: list[tuple[float, ...]]) -> None:
        self.embed_calls: list[dict[str, Any]] = []
        self._vectors = vectors

    def embed(self, texts, *, model, timeout_s):
        self.embed_calls.append({"texts": list(texts), "model": model, "timeout_s": timeout_s})
        return EmbedResult(
            vectors=tuple(self._vectors),
            dimension=len(self._vectors[0]) if self._vectors else None,
            model=model,
            usage=None,
            wallClockMs=5.0,
        )


class _ExplodingLMStudio:
    """Any call is a test failure — proves a cache HIT makes no live embed call at all."""

    def embed(self, *args, **kwargs):
        raise AssertionError("embed() must not be called on a corpus cache hit")


@pytest.fixture(autouse=True)
def _reset_retrieval_state():
    """`_state` is process-local (module docstring) — reset before and after every test in this
    file so no case leaks its corpus/BM25 state into the next."""
    retrieval._state.clear()
    yield
    retrieval._state.clear()


class TestPrime:
    def test_cache_hit_reuses_committed_vectors_no_live_embed_call(self, tmp_path):
        rows = [{"docId": "d1", "text": "alpha"}, {"docId": "d2", "text": "beta"}]
        corpus_path = _write_corpus(tmp_path, rows)
        _write_stopwords(tmp_path)
        pack = _embedder_pack(tmp_path)
        info = _model_info()
        cache_key = retrieval._compute_cache_key(
            model=info.id,
            quantization=info.quantization,
            document_prefix="",
            corpus_bytes=corpus_path.read_bytes(),
        )
        committed_vectors = {"d1": [3.0, 4.0], "d2": [1.0, 0.0]}
        (tmp_path / "corpus.embeddings.json").write_text(
            json.dumps(
                {
                    "cacheKey": cache_key,
                    "generatedAt": "2026-09-01T00:00:00Z",
                    "normalized": False,
                    "vectors": committed_vectors,
                }
            )
        )

        retrieval.prime(
            lmstudio=_ExplodingLMStudio(),
            model_info=info,
            call_surface="embeddings",
            timeout_s=30.0,
            pack=pack,
        )

        assert retrieval._state["corpusVectors"] == {"d1": (3.0, 4.0), "d2": (1.0, 0.0)}

    def test_cache_hit_still_computes_the_norm_diagnostic(self, tmp_path):
        """The spec's own resolution: the raw-norm diagnostic runs unconditionally, on a cache
        HIT as well as a cache miss — it is a property of whichever vectors get loaded, not of
        whether a network call fired."""
        rows = [{"docId": "d1", "text": "alpha"}, {"docId": "d2", "text": "beta"}]
        corpus_path = _write_corpus(tmp_path, rows)
        _write_stopwords(tmp_path)
        pack = _embedder_pack(tmp_path)
        info = _model_info()
        cache_key = retrieval._compute_cache_key(
            model=info.id,
            quantization=info.quantization,
            document_prefix="",
            corpus_bytes=corpus_path.read_bytes(),
        )
        (tmp_path / "corpus.embeddings.json").write_text(
            json.dumps(
                {
                    "cacheKey": cache_key,
                    "generatedAt": "2026-09-01T00:00:00Z",
                    "normalized": False,
                    "vectors": {"d1": [3.0, 4.0], "d2": [1.0, 0.0]},
                }
            )
        )

        retrieval.prime(
            lmstudio=_ExplodingLMStudio(),
            model_info=info,
            call_surface="embeddings",
            timeout_s=30.0,
            pack=pack,
        )

        from modelbench.stats import LEVEL_P50, percentile

        diagnostic = retrieval._state["corpusNormDiagnostic"]
        assert diagnostic["n"] == 2
        # norms are 5.0 (d1: [3,4]) and 1.0 (d2: [1,0]) — `stats.percentile`'s own estimator.
        assert diagnostic["median"] == pytest.approx(percentile([5.0, 1.0], level=LEVEL_P50))

    def test_cache_key_mismatch_embeds_live_with_document_prefix(self, tmp_path):
        """The mutation-test target: a mismatch must trigger a REAL re-embed, never silently
        reuse the stale committed vectors."""
        rows = [{"docId": "d1", "text": "alpha"}, {"docId": "d2", "text": "beta"}]
        _write_corpus(tmp_path, rows)
        _write_stopwords(tmp_path)
        pack = _embedder_pack(tmp_path, document_prefix="doc: ")
        info = _model_info()

        stale_key = retrieval._compute_cache_key(
            model=info.id,
            quantization=info.quantization,
            document_prefix="STALE PREFIX",
            corpus_bytes=b"stale corpus bytes",
        )
        (tmp_path / "corpus.embeddings.json").write_text(
            json.dumps(
                {
                    "cacheKey": stale_key,
                    "generatedAt": "2026-01-01T00:00:00Z",
                    "normalized": False,
                    "vectors": {"d1": [9.0, 9.0], "d2": [9.0, 9.0]},
                }
            )
        )
        stub = _StubEmbedLMStudio(vectors=[(1.0, 0.0), (0.0, 1.0)])

        retrieval.prime(
            lmstudio=stub, model_info=info, call_surface="embeddings", timeout_s=30.0, pack=pack
        )

        assert len(stub.embed_calls) == 1
        assert stub.embed_calls[0]["texts"] == ["doc: alpha", "doc: beta"]
        assert stub.embed_calls[0]["model"] == info.id
        # NOT the stale committed vectors:
        assert retrieval._state["corpusVectors"] == {"d1": (1.0, 0.0), "d2": (0.0, 1.0)}

    def test_absent_committed_file_embeds_live(self, tmp_path):
        rows = [{"docId": "d1", "text": "alpha"}]
        _write_corpus(tmp_path, rows)
        _write_stopwords(tmp_path)
        pack = _embedder_pack(tmp_path)
        info = _model_info()
        stub = _StubEmbedLMStudio(vectors=[(1.0, 0.0)])

        retrieval.prime(
            lmstudio=stub, model_info=info, call_surface="embeddings", timeout_s=30.0, pack=pack
        )

        assert len(stub.embed_calls) == 1
        assert retrieval._state["corpusVectors"] == {"d1": (1.0, 0.0)}

    def test_builds_bm25_index_into_state(self, tmp_path):
        rows = [{"docId": "d1", "text": "the cat sat"}, {"docId": "d2", "text": "the dog ran"}]
        _write_corpus(tmp_path, rows)
        _write_stopwords(tmp_path)
        pack = _embedder_pack(tmp_path)
        info = _model_info()
        stub = _StubEmbedLMStudio(vectors=[(1.0, 0.0), (0.0, 1.0)])

        retrieval.prime(
            lmstudio=stub, model_info=info, call_surface="embeddings", timeout_s=30.0, pack=pack
        )

        index = retrieval._state["bm25Index"]
        assert index.n == 2
        assert index.docIds == ("d1", "d2")


class TestEmbedText:
    def test_applies_query_prefix(self, tmp_path):
        pack = _embedder_pack(tmp_path, query_prefix="Q: ")
        assert retrieval.embed_text({"query": "hello"}, pack=pack) == "Q: hello"

    def test_empty_prefix_is_identity(self, tmp_path):
        pack = _embedder_pack(tmp_path, query_prefix="")
        assert retrieval.embed_text({"query": "hello"}, pack=pack) == "hello"


# ==================================================================================================
# 5. `score_item` / `aggregate` (§6.2)
# ==================================================================================================


def _timeout_timing() -> ItemTiming:
    return ItemTiming(wallClockMs=None, calls=(), withheldFor="timeout")


def _no_response_timing() -> ItemTiming:
    return ItemTiming(wallClockMs=None, calls=(), withheldFor="no_response")


def _primed_pack(tmp_path: Path, *, corpus_vectors: dict[str, tuple[float, ...]]) -> Pack:
    """A pack whose `_state` is primed directly with `corpus_vectors` (bypassing a live embed
    call) — `prime()` itself is `TestPrime`'s own scope; this only needs the resulting `_state`."""
    rows = [{"docId": doc_id, "text": doc_id} for doc_id in corpus_vectors]
    _write_corpus(tmp_path, rows)
    _write_stopwords(tmp_path)
    pack = _embedder_pack(tmp_path)
    info = _model_info()
    stub = _StubEmbedLMStudio(vectors=[corpus_vectors[doc_id] for doc_id in corpus_vectors])
    retrieval.prime(
        lmstudio=stub, model_info=info, call_surface="embeddings", timeout_s=30.0, pack=pack
    )
    return pack


class TestScoreItem:
    def test_timeout_scores_fail_with_no_measures_or_counts(self, tmp_path):
        pack = _embedder_pack(tmp_path)
        item_input = {"itemId": "q1", "query": "x", "relevantDocIds": ["d1"]}
        result = retrieval.score_item(item_input, None, _timeout_timing(), pack=pack)
        assert result.outcome == "fail"
        assert result.scoreable == {}
        assert result.counts == {}
        assert result.measures == {}

    def test_no_response_scores_unrunnable_with_no_measures_or_counts(self, tmp_path):
        pack = _embedder_pack(tmp_path)
        item_input = {"itemId": "q1", "query": "x", "relevantDocIds": ["d1"]}
        result = retrieval.score_item(item_input, None, _no_response_timing(), pack=pack)
        assert result.outcome == "unrunnable"
        assert result.scoreable == {}
        assert result.counts == {}
        assert result.measures == {}

    def test_raises_if_prime_was_never_called(self, tmp_path):
        pack = _embedder_pack(tmp_path)
        item_input = {"itemId": "q1", "query": "x", "relevantDocIds": ["d1"]}
        embed_result = EmbedResult(
            vectors=((1.0, 0.0),), dimension=2, model="m", usage=None, wallClockMs=1.0
        )
        timing = ItemTiming(wallClockMs=5.0, calls=(), withheldFor=None)
        with pytest.raises(RuntimeError):
            retrieval.score_item(item_input, embed_result, timing, pack=pack)

    def test_clean_item_produces_measures_and_counts(self, tmp_path):
        corpus_vectors = {"d1": (1.0, 0.0), "d2": (0.0, 1.0), "d3": (-1.0, 0.0)}
        pack = _primed_pack(tmp_path, corpus_vectors=corpus_vectors)
        item_input = {"itemId": "q1", "query": "x", "relevantDocIds": ["d1"]}
        embed_result = EmbedResult(
            vectors=((1.0, 0.0),), dimension=2, model="m", usage=None, wallClockMs=1.0
        )
        timing = ItemTiming(wallClockMs=5.0, calls=(), withheldFor=None)

        result = retrieval.score_item(item_input, embed_result, timing, pack=pack)

        assert result.outcome == "pass"
        assert result.counts["recallAt1"] == 1  # d1 is the top hit
        assert result.counts["recallAt2"] == 1
        assert result.counts["precisionAt1"] == 1
        assert result.scoreable == {
            "recallAt1": True,
            "recallAt2": True,
            "precisionAt1": True,
            "mrr": True,
            "separationRaw": True,
            "separationZ": True,
        }
        assert result.measures["mrr"] == pytest.approx(1.0)  # d1 ranks first
        assert result.measures["separationRaw"] == pytest.approx(1.0)  # score 1.0 - 0.0
        expected_z = 1.0 / statistics.pstdev([1.0, 0.0, -1.0])
        assert result.measures["separationZ"] == pytest.approx(expected_z)
        assert result.detail["queryNorm"] == pytest.approx(1.0)

    def test_multi_relevant_item_recall_is_a_raw_hit_count_not_binarized(self, tmp_path):
        """`-ml` v1.24 ruling: `counts[metric]` stores the raw hit count (an int in `[0, |R|]`),
        never a pre-binarized flag — the mutation-test target: a mutant that stores
        `int(recall_at_k(...) == 1.0)` instead of the raw count must redden this."""
        corpus_vectors = {"d1": (1.0, 0.0), "d2": (0.0, 1.0), "d3": (-1.0, 0.0)}
        pack = _primed_pack(tmp_path, corpus_vectors=corpus_vectors)
        # relevant = {d1, d2}; top-2 by cosine to [1,0] is (d1, d2) — both relevant, full hit.
        item_input = {"itemId": "q1", "query": "x", "relevantDocIds": ["d1", "d2"]}
        embed_result = EmbedResult(
            vectors=((1.0, 0.0),), dimension=2, model="m", usage=None, wallClockMs=1.0
        )
        timing = ItemTiming(wallClockMs=5.0, calls=(), withheldFor=None)

        result = retrieval.score_item(item_input, embed_result, timing, pack=pack)

        # recallAt1: top-1 is d1 only -> 1 of 2 relevant found (a raw count of 1, not 0 or 1
        # binarized from `== 1.0`, which a `recall_at_k(...) > 0` reading and a mutant reading
        # `== 1.0` would disagree on for a k where the count is a genuine partial hit).
        assert result.counts["recallAt1"] == 1
        # recallAt2: top-2 is {d1, d2} -> both relevant found -> raw count 2.
        assert result.counts["recallAt2"] == 2


class TestAggregate:
    def test_excludes_items_with_no_mrr_measure(self, tmp_path):
        corpus_vectors = {"d1": (1.0, 0.0), "d2": (0.0, 1.0)}
        pack = _primed_pack(tmp_path, corpus_vectors=corpus_vectors)
        embed_result = EmbedResult(
            vectors=((1.0, 0.0),), dimension=2, model="m", usage=None, wallClockMs=1.0
        )
        timing = ItemTiming(wallClockMs=5.0, calls=(), withheldFor=None)

        scored = retrieval.score_item(
            {"itemId": "q1", "query": "x", "relevantDocIds": ["d1"]},
            embed_result,
            timing,
            pack=pack,
        )
        timed_out = retrieval.score_item(
            {"itemId": "q2", "query": "y", "relevantDocIds": ["d2"]}, None, _timeout_timing(),
            pack=pack,
        )

        aggregates = retrieval.aggregate((scored, timed_out), pack=pack)

        assert aggregates.mrr.n == 1  # only the scored item, never coerced from the timeout
        assert aggregates.mrr.mean == pytest.approx(1.0)
        for recall_metric in aggregates.recallAtK:
            assert recall_metric.n == 1
        assert aggregates.precisionAt1.n == 1
        assert aggregates.separationRaw.n == 1
        assert aggregates.separationZ.n == 1

    def test_binarizes_a_partial_multi_relevant_hit_as_a_success(self, tmp_path):
        """`-ml` v1.24 ruling, and the mutation-test target the spec calls out by name: a
        per-item success is "at least one relevant doc retrieved in the top k"
        (`recall_at_k(...) > 0`), never "every relevant doc retrieved" (`== 1.0`). This item is
        a genuine partial hit — 1 of 2 relevant docs in the top 2 — so the two readings
        disagree: `> 0` counts it a success, `== 1.0` (equivalently, a raw-count threshold of
        `>= len(relevant)`) would not."""
        corpus_vectors = {"d1": (1.0, 0.0), "d2": (0.0, 1.0), "d3": (-1.0, 0.0)}
        pack = _primed_pack(tmp_path, corpus_vectors=corpus_vectors)
        # ranked order for query [1,0] is (d1, d2, d3); relevant={d1,d3} -> top-2 hits only d1:
        # a partial hit (1 of 2 relevant), never a full one.
        item_input = {"itemId": "q1", "query": "x", "relevantDocIds": ["d1", "d3"]}
        embed_result = EmbedResult(
            vectors=((1.0, 0.0),), dimension=2, model="m", usage=None, wallClockMs=1.0
        )
        timing = ItemTiming(wallClockMs=5.0, calls=(), withheldFor=None)
        scored = retrieval.score_item(item_input, embed_result, timing, pack=pack)
        assert scored.counts["recallAt2"] == 1  # raw hit count, partial (not 0, not 2)

        aggregates = retrieval.aggregate((scored,), pack=pack)

        recall_at_2 = next(m for m in aggregates.recallAtK if m.name == "recallAt2")
        assert recall_at_2.successes == 1
        assert recall_at_2.n == 1

    def test_empty_scored_set_yields_none_continuous_aggregates(self, tmp_path):
        pack = _embedder_pack(tmp_path)
        timed_out = retrieval.score_item(
            {"itemId": "q1", "query": "x", "relevantDocIds": ["d1"]}, None, _timeout_timing(),
            pack=pack,
        )

        aggregates = retrieval.aggregate((timed_out,), pack=pack)

        assert aggregates.mrr is None
        assert aggregates.separationRaw is None
        assert aggregates.separationZ is None
        for recall_metric in aggregates.recallAtK:
            assert recall_metric.n == 0
        assert aggregates.precisionAt1.n == 0


# ==================================================================================================
# 6. `deterministic_arm()` (§6.3) — a fixed synthetic pack, no live LM Studio call
# ==================================================================================================

_DETERMINISTIC_CORPUS = [
    {"docId": "d1", "text": "apple banana cherry"},
    {"docId": "d2", "text": "dog cat mouse"},
    {"docId": "d3", "text": "car train plane"},
    {"docId": "d4", "text": "apple dog car"},
    {"docId": "d5", "text": "banana cat train"},
]

_DETERMINISTIC_QUERIES = [
    {"itemId": "q1", "query": "apple banana cherry", "relevantDocIds": ["d1"]},
    {"itemId": "q2", "query": "dog cat mouse", "relevantDocIds": ["d2"]},
    {"itemId": "q3", "query": "car train plane", "relevantDocIds": ["d3"]},
]


def _deterministic_arm_pack(tmp_path: Path) -> Pack:
    (tmp_path / "corpus.jsonl").write_text(
        "\n".join(json.dumps(r) for r in _DETERMINISTIC_CORPUS) + "\n", encoding="utf-8"
    )
    (tmp_path / "queries.jsonl").write_text(
        "\n".join(json.dumps(r) for r in _DETERMINISTIC_QUERIES) + "\n", encoding="utf-8"
    )
    _write_stopwords(tmp_path)
    return _embedder_pack(tmp_path)


class TestDeterministicArm:
    def test_produces_a_deterministic_run_result_with_no_timings(self, tmp_path):
        pack = _deterministic_arm_pack(tmp_path)

        run = retrieval.deterministic_arm(pack=pack, session_id="sess-1")

        assert run.armKind == "deterministic"
        assert run.sessionId == "sess-1"
        assert run.attestationTripWire is None
        assert run.latency is None
        assert len(run.items) == 3
        for item in run.items:
            assert item.timing is None
        assert run.fingerprint.armKind == "deterministic"
        assert run.fingerprint.get("armId") == "bm25"
        # A complete, well-formed deterministic fingerprint — all 11 required fields present.
        assert run.fingerprint.validate() == []

    def test_arm_parameters_hash_is_stable_across_runs(self, tmp_path):
        """Deterministic given (corpus version, query set, parameters) — `armParametersHash`
        must not depend on wall-clock timing."""
        pack = _deterministic_arm_pack(tmp_path)
        run_a = retrieval.deterministic_arm(pack=pack, session_id=None)
        run_b = retrieval.deterministic_arm(pack=pack, session_id=None)
        assert run_a.fingerprint.get("armParametersHash") == run_b.fingerprint.get(
            "armParametersHash"
        )

    def test_reuses_the_same_scoring_path_as_score_item(self, tmp_path):
        """§6.3: "one scoring path, two arms" — every item carries the same measure/count names
        `score_item` produces."""
        pack = _deterministic_arm_pack(tmp_path)
        run = retrieval.deterministic_arm(pack=pack, session_id=None)
        for item in run.items:
            assert set(item.measures) == {"mrr", "separationRaw", "separationZ"}
            assert "precisionAt1" in item.counts
            assert "recallAt1" in item.counts and "recallAt2" in item.counts


# ==================================================================================================
# 6. Done-condition 4 (S3 spec §8 Step 2 item 2) — the offline ranking self-test against the REAL,
# committed pack data: `corpus.embeddings.json` (written live by `refresh_golden.py --embed-corpus`,
# once, against `text-embedding-qwen3-embedding-0.6b`) and `golden_retrieval.embeddings.json`
# (copied verbatim at Step 0, itself embedded by the same underlying model via falkor-chat's own
# pipeline). NO live embedding call anywhere in this class — `prime()` is driven with an
# `_ExplodingLMStudio` that fails the test the instant `.embed()` is called, and every query vector
# comes from the committed fixture, never from a fresh call. This isolates "is the ranking code
# right" from "is the live embedding call right" (`-ml` §5.4's own stated purpose for shipping
# `golden_retrieval.embeddings.json` at all) — the live self-check (done-condition 5, S3 Step 2 item
# 3) is the counterpart that DOES make live calls, and is deliberately not duplicated here.
# ==================================================================================================

_REAL_PACK_ROOT = Path(__file__).resolve().parents[1] / "packs" / "embedder-graphrag-retrieval"

#: The exact model identity `corpus.embeddings.json` was generated against (its own `cacheKey`
#: header, read back and asserted below) — reproduced here rather than read from the file, so a
#: cache-key regression in either `embed_corpus` or `prime()`'s own `_compute_cache_key` fails
#: this test loudly (a HIT silently degrading to a live call would instead be caught by
#: `_ExplodingLMStudio` raising).
_REAL_MODEL_INFO = ModelInfo(
    id="text-embedding-qwen3-embedding-0.6b",
    object="model",
    type="embeddings",
    publisher="lmstudio",
    arch="qwen3",
    compatibility_type="gguf",
    quantization="Q8_0",
    state="loaded",
    max_context_length=8192,
    capabilities=None,
    loaded_context_length=8192,
)


class _ExplodingLMStudioNoCorpusCall:
    """Any call is a test failure — proves the offline self-test makes no live embedding call at
    all (spec §8 Step 2 item 2: "with no live embedding call")."""

    def embed(self, *args, **kwargs):
        raise AssertionError("embed() must not be called — done-condition 4 is offline-only")


def _score_all_golden_queries(pack: Pack, golden_vectors: dict[str, Any]):
    """One pass over every `queries.jsonl` row, scoring against its own fixed vector from
    `golden_retrieval.embeddings.json` — never a live call."""
    items = []
    for row in pack.iter_items():
        vector = golden_vectors[row["itemId"]]["vector"]
        embed_result = EmbedResult(
            vectors=(tuple(vector),),
            dimension=len(vector),
            model=_REAL_MODEL_INFO.id,
            usage=None,
            wallClockMs=1.0,
        )
        timing = ItemTiming(wallClockMs=5.0, calls=(), withheldFor=None)
        items.append(retrieval.score_item(row, embed_result, timing, pack=pack))
    return retrieval.aggregate(tuple(items), pack=pack)


@pytest.fixture
def _real_pack() -> Pack:
    if not (_REAL_PACK_ROOT / "corpus.embeddings.json").exists():
        pytest.fail(
            f"{_REAL_PACK_ROOT / 'corpus.embeddings.json'} does not exist — run "
            "`refresh_golden.py --embed-corpus` (S3 spec §8 Step 2 item 1) before this test can "
            "run; done-condition 4 needs the real artifact, not a synthetic one."
        )
    return load_pack(_REAL_PACK_ROOT)


class TestOfflineRankingSelfTestAgainstTheRealPackData:
    def test_prime_cache_hits_against_the_committed_corpus_embeddings_with_no_live_call(
        self, _real_pack: Pack
    ) -> None:
        retrieval._state.clear()
        retrieval.prime(
            lmstudio=_ExplodingLMStudioNoCorpusCall(),
            model_info=_REAL_MODEL_INFO,
            call_surface="embeddings",
            timeout_s=30.0,
            pack=_real_pack,
        )
        assert len(retrieval._state["corpusVectors"]) == 121

    def test_ranking_path_reproduces_identical_rankings_from_the_two_fixed_vector_files(
        self, _real_pack: Pack
    ) -> None:
        """"the ranking path is shown to reproduce identical rankings from the two fixed vector
        files with no live embedding call at all" (plan, cited S3 spec §8 Step 2 item 2) — proven
        here as determinism: the same two fixed files, scored twice, must produce byte-identical
        `RetrievalAggregates` and per-item rankings, since nothing in the path (cosine, sorting,
        aggregation) has any source of non-determinism to reproduce identically from otherwise."""
        golden_vectors = json.loads(
            (_REAL_PACK_ROOT / "golden_retrieval.embeddings.json").read_text(encoding="utf-8")
        )

        retrieval._state.clear()
        retrieval.prime(
            lmstudio=_ExplodingLMStudioNoCorpusCall(),
            model_info=_REAL_MODEL_INFO,
            call_surface="embeddings",
            timeout_s=30.0,
            pack=_real_pack,
        )
        aggregates_a = _score_all_golden_queries(_real_pack, golden_vectors)
        aggregates_b = _score_all_golden_queries(_real_pack, golden_vectors)

        assert aggregates_a == aggregates_b

    def test_aggregate_metrics_are_well_formed_over_all_38_golden_queries(
        self, _real_pack: Pack
    ) -> None:
        """Sanity bounds only — NOT an assertion against `retrieval_baseline.json`'s pinned
        figures. `-ml` §5.4 is explicit that exact-vs-ANN and vector-only-vs-hybrid push in
        opposite directions, so a match OR a mismatch against that baseline is uninterpretable as
        a correctness signal here; that comparison is the live self-check's own job (S3 Step 2
        item 3, `docs/test-reports/embedder-self-check-report.md`), never this offline test's."""
        golden_vectors = json.loads(
            (_REAL_PACK_ROOT / "golden_retrieval.embeddings.json").read_text(encoding="utf-8")
        )

        retrieval._state.clear()
        retrieval.prime(
            lmstudio=_ExplodingLMStudioNoCorpusCall(),
            model_info=_REAL_MODEL_INFO,
            call_surface="embeddings",
            timeout_s=30.0,
            pack=_real_pack,
        )
        aggregates = _score_all_golden_queries(_real_pack, golden_vectors)

        assert aggregates.mrr is not None
        assert aggregates.mrr.n == 38
        assert 0.0 <= aggregates.mrr.mean <= 1.0
        for recall_metric in aggregates.recallAtK:
            assert recall_metric.n == 38
            assert 0 <= recall_metric.successes <= recall_metric.n
        assert aggregates.precisionAt1 is not None
        assert aggregates.precisionAt1.n == 38
