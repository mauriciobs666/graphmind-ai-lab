"""Structural integrity checks for the golden retrieval set (K-026,
`docs/plans/graphrag-eval.md` §5 Unit 2a).

**Genuinely network/DB-free for both checks below** — this is the whole point of the
v3 `target_text` field (plan §5 Unit 2a "Confirmed (v3)"): neither check reads
`ws:eval`, and this file needs neither FalkorDB nor a live LLM/embedder to pass.

1. Self-retrieval-inflation guard (method note finding 5/risk 2): no golden `query`
   is a case-insensitive substring of, or a superstring containing, its own pair's
   `target_text` — a pure string comparison against the fixture's own `target_text`
   field, never a live `Message.text` read.
2. Embedding-cache-currency guard (plan §3 D3): every golden query id has a cache
   entry in `golden_retrieval.embeddings.json` whose `model` matches
   `config/models.json`'s current `defaults.embedding`.

**Both checks read real config/fixture *files* directly, never `ModelGateway.from_env()`**
(plan §3 D7 mechanism 1: `modelconfig.Overlay.load(str(modelconfig.DEFAULT_MODEL_CONFIG_PATH))`).
Inside a pytest context, `ModelGateway.from_env()` is silently redirected by
`server/tests/conftest.py`'s autouse `_model_config_env` fixture to the dim-4 test
fixture config — using it here would validate the embedding cache against the wrong
source of truth.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from falkorchat import modelconfig

_EVAL_DIR = Path(__file__).resolve().parent
_GOLDEN_PATH = _EVAL_DIR / "golden_retrieval.jsonl"
_EMBEDDINGS_PATH = _EVAL_DIR / "golden_retrieval.embeddings.json"


def _load_golden_set() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with _GOLDEN_PATH.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise AssertionError(
                    f"{_GOLDEN_PATH.name}:{line_no}: invalid JSON — {exc}"
                ) from exc
    return rows


_GOLDEN_ROWS = _load_golden_set()
_GOLDEN_IDS = [row["id"] for row in _GOLDEN_ROWS]


def test_golden_set_file_exists_and_nonempty() -> None:
    assert _GOLDEN_PATH.exists(), f"missing fixture: {_GOLDEN_PATH}"
    assert _GOLDEN_ROWS, f"{_GOLDEN_PATH.name} has no rows"


def test_golden_set_size_in_expected_range() -> None:
    """Plan §5 Unit 2a: 30-50 golden pairs."""
    assert 30 <= len(_GOLDEN_ROWS) <= 50, (
        f"expected 30-50 golden pairs, found {len(_GOLDEN_ROWS)}"
    )


def test_golden_ids_are_unique() -> None:
    assert len(_GOLDEN_IDS) == len(set(_GOLDEN_IDS)), "duplicate golden ids found"


@pytest.mark.parametrize("row", _GOLDEN_ROWS, ids=_GOLDEN_IDS)
def test_golden_row_has_required_fields(row: dict[str, Any]) -> None:
    for field in ("id", "query", "relevant_msgIds", "topic", "target_text", "rationale"):
        assert field in row, f"{row.get('id', '?')}: missing field {field!r}"
    assert row["relevant_msgIds"], f"{row['id']}: relevant_msgIds must be non-empty"


@pytest.mark.parametrize("row", _GOLDEN_ROWS, ids=_GOLDEN_IDS)
def test_query_is_not_verbatim_self_retrieval(row: dict[str, Any]) -> None:
    """Self-retrieval-inflation guard (method note finding 5/risk 2).

    A paraphrased query must never verbatim-match its own target message's text —
    neither as a substring of the target, nor (degenerately) the target being a
    substring of the query. Pure string comparison against the fixture's own
    `target_text` field; no graph read.
    """
    query = row["query"].strip().lower()
    target_text = row["target_text"].strip().lower()
    assert query not in target_text, (
        f"{row['id']}: query is a verbatim substring of its own target_text — "
        f"paraphrase it instead of quoting the message"
    )
    assert target_text not in query, (
        f"{row['id']}: target_text is a substring of the query — "
        f"paraphrase the query instead of quoting the message"
    )


def _expected_embedding_model_ref() -> str:
    """D7 mechanism 1: read the real `config/models.json` directly, bypassing
    `ModelGateway.from_env()` (which is redirected to the test-fixture config for
    every pytest test by `server/tests/conftest.py`'s autouse fixture)."""
    overlay = modelconfig.Overlay.load(str(modelconfig.DEFAULT_MODEL_CONFIG_PATH))
    ref = overlay.default_for("embedding")
    assert ref, (
        f"{modelconfig.DEFAULT_MODEL_CONFIG_PATH} has no defaults.embedding — "
        f"cannot validate the golden-query embedding cache"
    )
    return ref


def _load_embedding_cache() -> dict[str, Any]:
    if not _EMBEDDINGS_PATH.exists():
        pytest.fail(
            f"{_EMBEDDINGS_PATH.name} does not exist — run "
            f"embed_golden_queries.py to generate it"
        )
    with _EMBEDDINGS_PATH.open(encoding="utf-8") as f:
        return json.load(f)


@pytest.mark.parametrize("row", _GOLDEN_ROWS, ids=_GOLDEN_IDS)
def test_embedding_cache_is_current_for_configured_model(row: dict[str, Any]) -> None:
    """Plan §3 D3: the committed embedding cache must be keyed to the *current*
    configured embedding model — a mismatch means the cache is stale and must be
    refreshed, never silently scored against an outdated vector."""
    expected_ref = _expected_embedding_model_ref()
    cache = _load_embedding_cache()
    golden_id = row["id"]
    assert golden_id in cache, (
        f"{golden_id}: no cache entry in {_EMBEDDINGS_PATH.name} — "
        f"run embed_golden_queries.py to refresh the cache for {expected_ref!r}"
    )
    entry = cache[golden_id]
    cached_model = entry.get("model")
    assert cached_model == expected_ref, (
        f"{golden_id}: cached embedding model {cached_model!r} does not match "
        f"config/models.json's current defaults.embedding {expected_ref!r} — "
        f"run embed_golden_queries.py to refresh the cache for {expected_ref!r}"
    )
    assert isinstance(entry.get("vector"), list) and entry["vector"], (
        f"{golden_id}: cache entry has no non-empty 'vector'"
    )
