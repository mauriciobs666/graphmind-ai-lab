"""typed-decision-models U2.4 — the `prime` guard (plan5 §3.1).

The embedder's `prime` corpus embed runs once, before any item. On a corpus cache miss its
`lmstudio.embed` can raise `LMStudioCallTimeout`/`LMStudioCallFailed`; the runner maps both to
`RunRefused(exitCode=3)` (the warm-up rule: nothing was scored). Every other exception passes
through, and a cache hit makes no corpus call at all, so it is unaffected.

The refusal is driven through `cli.main` with the real `retrieval` scorer over an on-disk fixture
pack (only `cli` stores, impl review P9-n1, so "stores nothing" is asserted there), against a
stubbed `LMStudio`. Offline throughout.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest
import test_cli as cli_tests
import test_runner as chat

from modelbench import cli
from modelbench.lmstudio import EmbedResult, LMStudioCallFailed, LMStudioCallTimeout
from modelbench.runner import RunRefused, _drive_single_call_items
from modelbench.scoring import retrieval

_MODEL_ID = "text-embedding-fixture"


@pytest.fixture(autouse=True)
def _reset_retrieval_state():
    """`retrieval._state` is process-global — cleared on both sides so no corpus state leaks."""
    retrieval._state.clear()
    yield
    retrieval._state.clear()


def _embedding_model_info():
    return dataclasses.replace(chat.model_info(model_id=_MODEL_ID), type="embeddings")


def _workspace(tmp_path: Path, *, cache_hit: bool) -> Path:
    cli_tests._write_embedder_pack_fixture(tmp_path)
    cli_tests._write_host_json(tmp_path)
    if cache_hit:
        pack_dir = tmp_path / "packs" / cli_tests.EMBEDDER_PACK
        info = _embedding_model_info()
        cache_key = retrieval._compute_cache_key(
            model=info.id,
            quantization=info.quantization,
            document_prefix="",
            corpus_bytes=(pack_dir / "corpus.jsonl").read_bytes(),
        )
        (pack_dir / "corpus.embeddings.json").write_text(
            json.dumps(
                {
                    "cacheKey": cache_key,
                    "generatedAt": "2026-09-01T00:00:00Z",
                    "normalized": False,
                    "vectors": {"d1": [1.0, 0.0], "d2": [0.0, 1.0]},
                }
            )
        )
    return tmp_path


def _embed_result() -> EmbedResult:
    return EmbedResult(
        vectors=((1.0, 0.0),), dimension=2, model=_MODEL_ID, usage=None, wallClockMs=5.0
    )


def _stub(embed_responses: list[Any]) -> chat.StubLMStudio:
    return chat.StubLMStudio(
        catalog_result=[_embedding_model_info()],
        embed_responses=embed_responses,
    )


def _cli_run(root: Path, monkeypatch, lms: chat.StubLMStudio) -> int:
    monkeypatch.setattr(cli, "LMStudio", lambda base_url: lms)
    return cli.main(
        ["run", "--root", str(root), "--pack", cli_tests.EMBEDDER_PACK, "--model", _MODEL_ID]
    )


_PRIME_FAILURES = pytest.mark.parametrize(
    "failure",
    [
        LMStudioCallTimeout("timed out after 120s"),
        LMStudioCallFailed("connection refused", status=None, body=None),
    ],
    ids=["timeout", "failed"],
)


@_PRIME_FAILURES
def test_a_failed_corpus_embed_on_a_cache_miss_exits_3_storing_nothing(
    tmp_path, monkeypatch, capsys, failure
):
    root = _workspace(tmp_path, cache_hit=False)
    lms = _stub([failure])

    code = _cli_run(root, monkeypatch, lms)

    err = capsys.readouterr().err
    assert code == 3, err
    assert "corpus embed" in err
    assert err.rstrip().endswith("nothing was scored")
    assert len(lms.embed_calls) == 1  # the corpus embed; no item was ever driven
    assert not (root / "results").exists()


@_PRIME_FAILURES
def test_the_refusal_chains_the_lmstudio_exception(tmp_path, monkeypatch, failure):
    """`from exc`: the cause is the very exception `embed` raised."""
    root = _workspace(tmp_path, cache_hit=False)
    pack = cli.load_pack(root / "packs" / cli_tests.EMBEDDER_PACK)
    with pytest.raises(RunRefused) as caught:
        _drive_single_call_items(
            pack,
            chat._cfg(modelKey=_MODEL_ID),
            lmstudio=_stub([failure]),
            model_info=_embedding_model_info(),
            call_surface="embeddings",
            baseline_residency=chat.resident(),
        )
    assert caught.value.exitCode == 3
    assert caught.value.__cause__ is failure


def test_a_cache_hit_makes_no_corpus_call_and_the_run_is_stored(tmp_path, monkeypatch, capsys):
    """The stub's one embed response is the item's; a corpus call would consume it first."""
    root = _workspace(tmp_path, cache_hit=True)
    lms = _stub([_embed_result()])

    code = _cli_run(root, monkeypatch, lms)

    err = capsys.readouterr().err
    assert code == 0, err
    assert [c["texts"] for c in lms.embed_calls] == [["apple banana"]]
    assert len(list((root / "results" / "runs").glob("*.json"))) == 2  # model + deterministic


class _RaisingPrimeScorer(chat.FakeItemScorer):
    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    def prime(self, *, lmstudio, model_info, call_surface, timeout_s, pack):
        raise self._exc


def test_a_non_lmstudio_error_from_prime_propagates_unchanged(monkeypatch):
    """`RunRefused` subclasses `RuntimeError`, so the type is checked exactly: an `except
    Exception` guard would turn this into a refusal."""
    boom = RuntimeError("scorer bug")
    scorer = _RaisingPrimeScorer(boom)
    monkeypatch.setattr("modelbench.runner._load_item_scorer", lambda pack: scorer)
    pack = chat.FakePack(role="embedder", items=[{"id": "a"}], prompt_cfg=chat.make_prompt_cfg())
    with pytest.raises(RuntimeError) as caught:
        _drive_single_call_items(
            pack,
            chat._cfg(),
            lmstudio=chat.StubLMStudio(),
            model_info=_embedding_model_info(),
            call_surface="embeddings",
            baseline_residency=chat.resident(),
        )
    assert caught.value is boom
