"""`scripts/refresh_golden.py` — the embedder pack's one-way importer (this stage's own spec,
`docs/plans/small-model-benchmarking-s3-spec.md` §5, §8 Step 0: "static pack data, no live call").

Every test here is either **pure** (a synthetic in-memory/tmp_path fixture, never a real
`falkor-chat` path — `-ml` §5.5's cache key is asserted against a synthetic corpus/model-info
fixture, not the real corpus, which is Step 2's) or reads only this pack's own **already-copied**
data under `model-bench/packs/embedder-graphrag-retrieval/` (inside `model-bench`, so FR-23's
"standalone" rule is untouched — that data was itself produced by this script, once, against the
real `falkor-chat` tree, confirmed this session by running `--check-origins` directly; see this
unit's own report). Nothing here depends on `falkor-chat/` being present on disk, matching the
same discipline plan §3.1 point 2(b) states for `test_metrics_agreement.py` (§5 test 20: "the
default suite still passes with `falkor-chat/` renamed away").
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import refresh_golden  # noqa: E402

_PACK_ROOT = Path(__file__).resolve().parents[1] / "packs" / "embedder-graphrag-retrieval"

# A tiny stand-in for `seed_eval_corpus.py`'s real shape (two topics, a couple of messages each) —
# annotated exactly like the real file (`_CORPUS: list[dict[str, object]] = [...]`), which is an
# `ast.AnnAssign` node, not a plain `ast.Assign` (confirmed against the real file this session).
_SYNTHETIC_SEED_SCRIPT = '''
_CORPUS: list[dict[str, object]] = [
    {
        "slug": "topic-one",
        "messages": [
            ("user", "first message"),
            ("assistant", "second message"),
        ],
    },
    {
        "slug": "topic-two",
        "messages": [
            ("user", "third message"),
        ],
    },
]
'''

_SYNTHETIC_SEED_SCRIPT_PLAIN_ASSIGN = '''
_CORPUS = [
    {"slug": "topic-one", "messages": [("user", "only message")]},
]
'''

_SYNTHETIC_SEED_SCRIPT_NO_CORPUS = '''
_OTHER = [1, 2, 3]
'''


# --------------------------------------------------------------------------------------------
# `compute_cache_key` — `-ml` §5.5's four-component cache key, against a synthetic fixture
# --------------------------------------------------------------------------------------------


def test_compute_cache_key_has_exactly_the_four_ml_5_5_components() -> None:
    """`-ml` §5.5: "the cache key must include (model id, quantization, docPrefix, corpus
    version) — all four". This is what that document's own line means by "asserted in a unit
    test, not just implemented" — a synthetic corpus/model-info fixture, never the real 121-doc
    corpus (that fixture is Step 2's)."""
    key = refresh_golden.compute_cache_key(
        model="text-embedding-qwen3-embedding-0.6b",
        quantization="Q8_0",
        document_prefix="passage: ",
        corpus_bytes=b'{"docId": "d-001", "text": "hello", "topic": "t"}\n',
    )
    assert set(key.keys()) == {"model", "quantization", "documentPrefix", "corpusSha256"}
    assert key["model"] == "text-embedding-qwen3-embedding-0.6b"
    assert key["quantization"] == "Q8_0"
    assert key["documentPrefix"] == "passage: "
    assert key["corpusSha256"] == refresh_golden._sha256_bytes(
        b'{"docId": "d-001", "text": "hello", "topic": "t"}\n'
    )


def test_compute_cache_key_changes_when_any_one_component_changes() -> None:
    """The cache key's whole job is to invalidate on ANY of the four changes (`-ml` §5.5's own
    trap: "a cached one shared across prefix settings of the same model is equally wrong") — a
    key built from otherwise-identical inputs but one changed component must differ."""
    base = dict(
        model="m", quantization="q", document_prefix="p", corpus_bytes=b"corpus-bytes-one"
    )
    baseline = refresh_golden.compute_cache_key(**base)

    assert refresh_golden.compute_cache_key(**{**base, "model": "m2"}) != baseline
    assert refresh_golden.compute_cache_key(**{**base, "quantization": "q2"}) != baseline
    assert refresh_golden.compute_cache_key(**{**base, "document_prefix": "p2"}) != baseline
    assert (
        refresh_golden.compute_cache_key(**{**base, "corpus_bytes": b"corpus-bytes-two"})
        != baseline
    )


def test_compute_cache_key_is_stable_for_identical_inputs() -> None:
    key_a = refresh_golden.compute_cache_key(
        model="m", quantization="q", document_prefix="p", corpus_bytes=b"same-bytes"
    )
    key_b = refresh_golden.compute_cache_key(
        model="m", quantization="q", document_prefix="p", corpus_bytes=b"same-bytes"
    )
    assert key_a == key_b


# --------------------------------------------------------------------------------------------
# `CorpusEmbeddingsFile` — round trip, and the "normalized": false header key (spec §9's risk)
# --------------------------------------------------------------------------------------------


def test_corpus_embeddings_file_round_trips_and_flags_unnormalized() -> None:
    cache_key = refresh_golden.compute_cache_key(
        model="m", quantization="q", document_prefix="", corpus_bytes=b"x"
    )
    original = refresh_golden.CorpusEmbeddingsFile(
        cacheKey=cache_key,
        generatedAt="2026-09-10T00:00:00Z",
        vectors={"doc-1": [1.0, 2.0, 3.0], "doc-2": [4.0, 5.0, 6.0]},
    )

    as_dict = original.to_dict()
    assert as_dict["normalized"] is False  # spec §9: must not be assumable by omission
    assert as_dict["cacheKey"] == cache_key
    assert as_dict["vectors"] == {"doc-1": [1.0, 2.0, 3.0], "doc-2": [4.0, 5.0, 6.0]}

    round_tripped = refresh_golden.CorpusEmbeddingsFile.from_dict(as_dict)
    assert round_tripped.cacheKey == original.cacheKey
    assert round_tripped.generatedAt == original.generatedAt
    assert round_tripped.vectors == original.vectors


# --------------------------------------------------------------------------------------------
# `embed_corpus` — Step 2's live mode, deliberately NOT built in Step 0
# --------------------------------------------------------------------------------------------


def test_embed_corpus_is_not_implemented_in_this_stage() -> None:
    with pytest.raises(NotImplementedError):
        refresh_golden.embed_corpus(
            [], lmstudio=None, model_key="m", document_prefix="", timeout_s=1.0
        )


# --------------------------------------------------------------------------------------------
# `_read_corpus_literal` / `_corpus_rows_from_literal` — the AST-literal path, synthetic input
# --------------------------------------------------------------------------------------------


def test_read_corpus_literal_handles_the_real_files_annotated_assignment_form() -> None:
    """`_CORPUS`'s real declaration is `_CORPUS: list[dict[str, object]] = [...]` — an
    `ast.AnnAssign` node. An extractor that only matches plain `ast.Assign` silently finds
    nothing on the real file; this pins the annotated form specifically."""
    corpus = refresh_golden._read_corpus_literal(_SYNTHETIC_SEED_SCRIPT)
    assert len(corpus) == 2
    assert corpus[0]["slug"] == "topic-one"
    assert len(corpus[0]["messages"]) == 2


def test_read_corpus_literal_also_handles_a_plain_assign() -> None:
    corpus = refresh_golden._read_corpus_literal(_SYNTHETIC_SEED_SCRIPT_PLAIN_ASSIGN)
    assert corpus == [{"slug": "topic-one", "messages": [("user", "only message")]}]


def test_read_corpus_literal_raises_when_no_corpus_assignment_exists() -> None:
    """The origin's own shape changing must be loud, never a silent `[]` (spec §5's own
    docstring)."""
    with pytest.raises(refresh_golden.RefreshGoldenError, match="_CORPUS"):
        refresh_golden._read_corpus_literal(_SYNTHETIC_SEED_SCRIPT_NO_CORPUS)


def test_corpus_rows_from_literal_reproduces_the_eval_msg_id_convention() -> None:
    """`msg_id = f"eval-{slug}-{n:03d}"`, `n` 1-indexed per topic in message order (confirmed
    against `seed_eval_corpus.py:629` this session) — reproduced here so `docId` matches
    `golden_retrieval.jsonl`'s `relevant_msgIds` verbatim."""
    corpus = refresh_golden._read_corpus_literal(_SYNTHETIC_SEED_SCRIPT)
    rows = refresh_golden._corpus_rows_from_literal(corpus)
    assert rows == [
        {"docId": "eval-topic-one-001", "text": "first message", "topic": "topic-one"},
        {"docId": "eval-topic-one-002", "text": "second message", "topic": "topic-one"},
        {"docId": "eval-topic-two-001", "text": "third message", "topic": "topic-two"},
    ]


# --------------------------------------------------------------------------------------------
# `_queries_rows_from_golden_retrieval` — the jsonl-transform path, synthetic input
# --------------------------------------------------------------------------------------------


def test_queries_rows_from_golden_retrieval_renames_fields_per_pack_convention() -> None:
    """`id` -> `itemId`, `relevant_msgIds` -> `relevantDocIds`, `target_text` -> `targetText`;
    `query`/`topic`/`rationale` carried through unchanged (spec §4.1)."""
    line = json.dumps(
        {
            "id": "gr-01",
            "query": "a query",
            "relevant_msgIds": ["eval-topic-one-001"],
            "topic": "topic-one",
            "target_text": "the target",
            "rationale": "because",
        }
    )
    rows = refresh_golden._queries_rows_from_golden_retrieval([line, "", "   "])
    assert rows == [
        {
            "itemId": "gr-01",
            "query": "a query",
            "relevantDocIds": ["eval-topic-one-001"],
            "topic": "topic-one",
            "targetText": "the target",
            "rationale": "because",
        }
    ]


# --------------------------------------------------------------------------------------------
# `_pack_version_gate` — AC-3's "bump packVersion first" refusal, synthetic tmp_path pack
# --------------------------------------------------------------------------------------------


def test_pack_version_gate_allows_a_first_run_with_no_provenance_yet(tmp_path: Path) -> None:
    pack_root = tmp_path / "pack"
    pack_root.mkdir()
    (pack_root / "pack.json").write_text(json.dumps({"packVersion": "1.0.0"}))
    refresh_golden._pack_version_gate(pack_root, pack_root / "PROVENANCE.md")  # must not raise


def test_pack_version_gate_refuses_a_rerun_with_an_unchanged_pack_version(tmp_path: Path) -> None:
    pack_root = tmp_path / "pack"
    pack_root.mkdir()
    (pack_root / "pack.json").write_text(json.dumps({"packVersion": "1.0.0"}))
    refresh_golden._write_provenance(pack_root, "1.0.0", records=())

    with pytest.raises(refresh_golden.RefreshGoldenError, match="packVersion"):
        refresh_golden._pack_version_gate(pack_root, pack_root / "PROVENANCE.md")


def test_pack_version_gate_allows_a_rerun_once_pack_version_is_bumped(tmp_path: Path) -> None:
    pack_root = tmp_path / "pack"
    pack_root.mkdir()
    (pack_root / "pack.json").write_text(json.dumps({"packVersion": "1.0.0"}))
    refresh_golden._write_provenance(pack_root, "1.0.0", records=())

    (pack_root / "pack.json").write_text(json.dumps({"packVersion": "1.1.0"}))
    refresh_golden._pack_version_gate(pack_root, pack_root / "PROVENANCE.md")  # must not raise


# --------------------------------------------------------------------------------------------
# `_check_origins` — read-only drift detection, synthetic origins (monkeypatched, no real repo)
# --------------------------------------------------------------------------------------------


def _write_synthetic_provenance(
    pack_root: Path, *, origin_path: str, dest_path: str, source_bytes: bytes
) -> None:
    """Builds a `PROVENANCE.md` by hand, the way `_write_provenance` would, without going through
    `_run_import` — which would need `repo_root` to be a real git checkout (`_git_file_sha` shells
    out to `git log`). `_check_origins` only ever reads the SHA-256 column back out, so a
    fabricated `sourceGitSha` is harmless here."""
    record = refresh_golden.ProvenanceRecord(
        originPath=origin_path,
        destPath=dest_path,
        sourceGitSha="0" * 40,
        sourceSha256=refresh_golden._sha256_bytes(source_bytes),
        copiedAt="2026-09-10T00:00:00Z",
    )
    refresh_golden._write_provenance(pack_root, "1.0.0", (record,))


def test_check_origins_reports_unchanged_when_hashes_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = tmp_path / "repo"
    pack_root = tmp_path / "pack"
    (repo_root / "origin").mkdir(parents=True)
    pack_root.mkdir()
    origin_file = repo_root / "origin" / "a.txt"
    origin_file.write_bytes(b"stable content")

    monkeypatch.setattr(
        refresh_golden,
        "_TRACKED_ORIGINS",
        (refresh_golden.OriginSpec("origin/a.txt", "a.copy", "copy"),),
    )
    _write_synthetic_provenance(
        pack_root, origin_path="origin/a.txt", dest_path="a.copy", source_bytes=b"stable content"
    )

    results = refresh_golden._check_origins(repo_root, pack_root)
    assert len(results) == 1
    assert results[0].status == "unchanged"


def test_check_origins_reports_drifted_when_the_origin_changed_since_the_last_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = tmp_path / "repo"
    pack_root = tmp_path / "pack"
    (repo_root / "origin").mkdir(parents=True)
    pack_root.mkdir()
    origin_file = repo_root / "origin" / "a.txt"
    origin_file.write_bytes(b"original content")

    monkeypatch.setattr(
        refresh_golden,
        "_TRACKED_ORIGINS",
        (refresh_golden.OriginSpec("origin/a.txt", "a.copy", "copy"),),
    )
    _write_synthetic_provenance(
        pack_root,
        origin_path="origin/a.txt",
        dest_path="a.copy",
        source_bytes=b"original content",
    )

    origin_file.write_bytes(b"drifted content")
    results = refresh_golden._check_origins(repo_root, pack_root)
    assert results[0].status == "DRIFTED"


def test_check_origins_reads_the_check_only_originss_hash_from_the_fixtures_own_header(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The `check-only` kind (`test_metrics.py`) never gets a `PROVENANCE.md` row — its recorded
    hash lives on the transcribed fixture's own `sourceSha256` header (spec §3.1 point 2(c))."""
    repo_root = tmp_path / "repo"
    pack_root = tmp_path / "pack"
    (repo_root / "origin").mkdir(parents=True)
    (pack_root / "fixtures").mkdir(parents=True)
    origin_file = repo_root / "origin" / "check_me.py"
    origin_file.write_bytes(b"def f(): pass\n")

    fixture_path = pack_root / "fixtures" / "agreement.json"
    fixture_path.write_text(
        json.dumps({"sourceSha256": refresh_golden._sha256_bytes(b"def f(): pass\n")})
    )

    monkeypatch.setattr(
        refresh_golden,
        "_TRACKED_ORIGINS",
        (
            refresh_golden.OriginSpec(
                "origin/check_me.py", "fixtures/agreement.json", "check-only"
            ),
        ),
    )
    results = refresh_golden._check_origins(repo_root, pack_root)
    assert results[0].status == "unchanged"

    fixture_path.write_text(json.dumps({"sourceSha256": "0" * 64}))
    results = refresh_golden._check_origins(repo_root, pack_root)
    assert results[0].status == "DRIFTED"


# --------------------------------------------------------------------------------------------
# `main()` — mode exclusivity and the `--embed-corpus` loud-failure surface
# --------------------------------------------------------------------------------------------


def test_main_refuses_check_origins_and_embed_corpus_together(tmp_path: Path) -> None:
    exit_code = refresh_golden.main(
        ["--pack", str(tmp_path), "--check-origins", "--embed-corpus"]
    )
    assert exit_code == 2


def test_main_embed_corpus_raises_rather_than_silently_no_opping(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError):
        refresh_golden.main(["--pack", str(tmp_path), "--embed-corpus"])


# --------------------------------------------------------------------------------------------
# The shipped pack's own data (Step 0's "Done when", spec §8): docId/relevantDocIds set-equality
# --------------------------------------------------------------------------------------------


def test_shipped_pack_every_relevant_doc_id_resolves_inside_the_shipped_corpus() -> None:
    """`docId` values in `corpus.jsonl` must cover every `relevantDocIds` entry in
    `queries.jsonl` exactly (a direct set-equality/subset check, not eyeballed) — otherwise a
    golden query points at a document the corpus does not have, and every metric on that item is
    silently meaningless. Reads only this pack's own already-copied data under
    `model-bench/packs/`, not `falkor-chat/`."""
    corpus_doc_ids = {
        json.loads(line)["docId"]
        for line in (_PACK_ROOT / "corpus.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    assert len(corpus_doc_ids) == 121

    relevant_doc_ids: set[str] = set()
    query_rows = [
        json.loads(line)
        for line in (_PACK_ROOT / "queries.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(query_rows) == 38
    for row in query_rows:
        relevant_doc_ids.update(row["relevantDocIds"])

    assert relevant_doc_ids <= corpus_doc_ids
    assert relevant_doc_ids - corpus_doc_ids == set()
