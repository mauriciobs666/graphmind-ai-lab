"""Unit tests for the async embedding worker + LM Studio embedder client (K-008).

The worker is **decoupled from the message write path** (DESIGN §9): a message is
readable before its embedding lands. It is exercised here with an injected stub
embedder — unit tests never touch the network. The real `LMStudioEmbedder` is
tested with an injected transport, so its response parsing is pinned without a
live LM Studio server.
"""

from __future__ import annotations

import pytest

from falkorchat.embedding import EmbeddingWorker, LMStudioEmbedder
from falkorchat.repository import EmbeddingDimensionError


class SpyRepo:
    """Records set_embedding calls; mimics its length validation."""

    def __init__(self, expected_dim: int):
        self._dim = expected_dim
        self.calls: list[tuple] = []

    def set_embedding(self, ws, *, msg_id, embedding, expected_dim=None):
        dim = self._dim if expected_dim is None else expected_dim
        if len(embedding) != dim:
            raise EmbeddingDimensionError(f"len {len(embedding)} != {dim}")
        self.calls.append((ws, msg_id, tuple(embedding), expected_dim))
        return True


class StubEmbedder:
    """Deterministic embedder — returns a fixed vector per text, no network."""

    def __init__(self, vector: list[float]):
        self._vector = vector
        self.seen: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.seen.append(text)
        return list(self._vector)


# ── worker: happy path ────────────────────────────────────────────────────────


def test_worker_embeds_then_writes_via_repository():
    repo = SpyRepo(expected_dim=4)
    embedder = StubEmbedder([1.0, 0.0, 0.0, 0.0])
    worker = EmbeddingWorker(repo, embedder, expected_dim=4)

    worker.embed_message("test", msg_id="m1", text="about cats")

    assert embedder.seen == ["about cats"]
    assert len(repo.calls) == 1
    ws, msg_id, vec, expected_dim = repo.calls[0]
    assert (ws, msg_id) == ("test", "m1")
    assert vec == (1.0, 0.0, 0.0, 0.0)
    assert expected_dim == 4


# ── worker: length validation is loud ─────────────────────────────────────────


def test_worker_rejects_wrong_dimension_from_embedder_loudly():
    repo = SpyRepo(expected_dim=4)
    embedder = StubEmbedder([1.0, 0.0, 0.0])  # 3 dims, expected 4
    worker = EmbeddingWorker(repo, embedder, expected_dim=4)

    with pytest.raises(EmbeddingDimensionError):
        worker.embed_message("test", msg_id="m1", text="oops")

    assert repo.calls == []  # nothing written on a bad vector


# ── LMStudioEmbedder: parses the OpenAI-compatible response ────────────────────


def test_lmstudio_embedder_parses_embedding_and_posts_expected_payload():
    captured: dict = {}

    def fake_transport(url: str, payload: dict) -> dict:
        captured["url"] = url
        captured["payload"] = payload
        return {"data": [{"embedding": [0.1, 0.2, 0.3]}]}

    embedder = LMStudioEmbedder(
        base_url="http://localhost:1234/v1", model="qwen3", transport=fake_transport
    )

    vec = embedder.embed("hello")

    assert vec == [0.1, 0.2, 0.3]
    assert captured["url"] == "http://localhost:1234/v1/embeddings"
    assert captured["payload"] == {"model": "qwen3", "input": "hello"}


def test_lmstudio_embedder_strips_trailing_slash_on_base_url():
    def fake_transport(url: str, payload: dict) -> dict:
        assert url == "http://localhost:1234/v1/embeddings"
        return {"data": [{"embedding": [0.0]}]}

    embedder = LMStudioEmbedder(
        base_url="http://localhost:1234/v1/", model="m", transport=fake_transport
    )
    embedder.embed("x")
