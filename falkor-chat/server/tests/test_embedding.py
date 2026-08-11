"""Unit tests for the async embedding worker + OpenAI-compatible embedder client
(K-008; K-042 Landing 1 renamed `LMStudioEmbedder` to `OpenAICompatibleEmbedder`).

The worker is **decoupled from the message write path** (DESIGN §9): a message is
readable before its embedding lands. It is exercised here with an injected stub
embedder — unit tests never touch the network. The real `OpenAICompatibleEmbedder` is
tested with an injected transport, so its response parsing is pinned without a
live LM Studio server.
"""

from __future__ import annotations

import pytest

from falkorchat.embedding import EmbeddingWorker, OpenAICompatibleEmbedder
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


# ── worker: resolves through a ModelGateway (K-042 FR-4) ───────────────────────


class _StubGateway:
    """A minimal `ModelGateway`-shaped stub: resolve() reports a declared `dim`,
    embedder() returns a fixed stub client — exercises the worker's `models=` path
    without any real config file or network."""

    def __init__(self, embedder, *, dim):
        self._embedder = embedder
        self._resolution = _StubResolution(dim)

    def resolve(self, kind, *, requested=None, ws=None, overrides=None):
        return self._resolution

    def embedder(self, kind, *, requested=None, ws=None, overrides=None):
        return self._embedder


class _StubResolution:
    def __init__(self, dim):
        self.primary = _StubResolvedModel(dim)


class _StubResolvedModel:
    def __init__(self, dim):
        self.dim = dim


def test_worker_resolves_dim_from_the_gateway_when_no_expected_dim_given():
    embedder = StubEmbedder([1.0, 0.0, 0.0, 0.0])
    gateway = _StubGateway(embedder, dim=4)
    repo = SpyRepo(expected_dim=4)
    worker = EmbeddingWorker(repo, models=gateway)

    worker.embed_message("test", msg_id="m1", text="about cats")

    assert embedder.seen == ["about cats"]
    assert len(repo.calls) == 1
    assert repo.calls[0][3] == 4  # dim came from the gateway's resolved model, not config


def test_worker_falls_back_to_config_embedding_dim_when_gateway_declares_none():
    from falkorchat import config

    embedder = StubEmbedder([0.0] * config.EMBEDDING_DIM)
    gateway = _StubGateway(embedder, dim=None)
    repo = SpyRepo(expected_dim=config.EMBEDDING_DIM)
    worker = EmbeddingWorker(repo, models=gateway)

    worker.embed_message("test", msg_id="m1", text="x")

    assert repo.calls[0][3] == config.EMBEDDING_DIM


def test_worker_wraps_a_directly_injected_embedder_via_static_gateway_sugar():
    # FR-4 sugar: EmbeddingWorker(repo, embedder) still works unmodified (§3).
    repo = SpyRepo(expected_dim=4)
    embedder = StubEmbedder([1.0, 0.0, 0.0, 0.0])
    worker = EmbeddingWorker(repo, embedder, expected_dim=4)

    worker.embed_message("test", msg_id="m2", text="dogs too")

    assert embedder.seen == ["dogs too"]
    assert len(repo.calls) == 1


# ── OpenAICompatibleEmbedder: parses the OpenAI-compatible response ────────────


def test_lmstudio_embedder_parses_embedding_and_posts_expected_payload():
    captured: dict = {}

    def fake_transport(url: str, payload: dict) -> dict:
        captured["url"] = url
        captured["payload"] = payload
        return {"data": [{"embedding": [0.1, 0.2, 0.3]}]}

    embedder = OpenAICompatibleEmbedder(
        "http://localhost:1234/v1", "qwen3", transport=fake_transport
    )

    vec = embedder.embed("hello")

    assert vec == [0.1, 0.2, 0.3]
    assert captured["url"] == "http://localhost:1234/v1/embeddings"
    assert captured["payload"] == {"model": "qwen3", "input": "hello"}


def test_lmstudio_embedder_strips_trailing_slash_on_base_url():
    def fake_transport(url: str, payload: dict) -> dict:
        assert url == "http://localhost:1234/v1/embeddings"
        return {"data": [{"embedding": [0.0]}]}

    embedder = OpenAICompatibleEmbedder(
        "http://localhost:1234/v1/", "m", transport=fake_transport
    )
    embedder.embed("x")


def test_embedder_merges_params_into_the_payload():
    captured: dict = {}

    def fake_transport(url: str, payload: dict) -> dict:
        captured["payload"] = payload
        return {"data": [{"embedding": [0.0]}]}

    embedder = OpenAICompatibleEmbedder(
        "http://x/v1", "m", transport=fake_transport, params={"dimensions": 512}
    )
    embedder.embed("x")

    assert captured["payload"]["dimensions"] == 512


def test_embedder_omits_params_key_when_none_given():
    captured: dict = {}

    def fake_transport(url: str, payload: dict) -> dict:
        captured["payload"] = payload
        return {"data": [{"embedding": [0.0]}]}

    embedder = OpenAICompatibleEmbedder("http://x/v1", "m", transport=fake_transport)
    embedder.embed("x")

    assert "dimensions" not in captured["payload"]


def test_embedder_raises_provider_call_error_on_missing_data():
    from falkorchat.transport import ProviderCallError

    def fake_transport(url: str, payload: dict) -> dict:
        return {"unexpected": "shape"}

    embedder = OpenAICompatibleEmbedder("http://x/v1", "m", transport=fake_transport)
    with pytest.raises(ProviderCallError) as excinfo:
        embedder.embed("x")
    assert "m" in str(excinfo.value)
