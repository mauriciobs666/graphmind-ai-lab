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


_UNSET = object()


class SpyRepo:
    """Records set_embedding calls; mimics its length validation.

    `index_dim` (K-042 FR-19) is what `read_index_dimension` reports back for the
    `Message` label — defaults to `expected_dim` so every pre-existing test (which
    never passes it) keeps exercising the "the index matches the declared dim"
    path unchanged, exactly like `ws:test`'s real dim-4 fixture. Pass an explicit
    `index_dim` (including `None`, for "no vector index") to simulate the FR-19
    guard's other branches.
    """

    def __init__(self, expected_dim: int, *, index_dim=_UNSET):
        self._dim = expected_dim
        self._index_dim = expected_dim if index_dim is _UNSET else index_dim
        self.calls: list[tuple] = []
        self.index_dim_calls: list[tuple] = []

    def read_index_dimension(self, ws, *, label, prop="embedding"):
        self.index_dim_calls.append((ws, label, prop))
        return self._index_dim

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

    def __init__(self, embedder, *, dim, ref="stub/embed-model"):
        self._embedder = embedder
        self._resolution = _StubResolution(dim, ref)

    def resolve(self, kind, *, requested=None, ws=None, overrides=None):
        return self._resolution

    def embedder(self, kind, *, requested=None, ws=None, overrides=None):
        return self._embedder


class _StubResolution:
    def __init__(self, dim, ref="stub/embed-model"):
        self.primary = _StubResolvedModel(dim, ref)


class _StubResolvedModel:
    def __init__(self, dim, ref="stub/embed-model"):
        self.dim = dim
        self.ref = ref


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


# ── worker: the FR-19 embedding-dimension guard (K-042 Landing 2) ──────────────
#
# `embed_message` must compare the resolved model's *declared* dim against the
# workspace's *introspected* vector-index dim for `Message` — the label it
# writes — BEFORE calling the embedder at all. `ws:test`'s real dim-4 fixture is
# the "declared == index" happy path already exercised by every test above
# (`SpyRepo`'s `index_dim` defaults to `expected_dim`); these pin the mismatch
# branch, the "no index at all" branches, the cache, and the `Chunk`-is-never-
# consulted guarantee.


def test_worker_raises_before_calling_embedder_on_index_dimension_mismatch():
    embedder = StubEmbedder([1.0, 0.0, 0.0, 0.0])
    gateway = _StubGateway(embedder, dim=4, ref="lmstudio/embed-4")
    # workspace's Message index is dimension 8 — the model declares 4
    repo = SpyRepo(expected_dim=4, index_dim=8)
    worker = EmbeddingWorker(repo, models=gateway)

    with pytest.raises(EmbeddingDimensionError) as excinfo:
        worker.embed_message("acme", msg_id="m1", text="about cats")

    # no HTTP call, no vector computed, nothing written
    assert embedder.seen == []
    assert repo.calls == []
    # all five required facts (-graph.md §3.4): workspace, label, index dim,
    # model ref, model's declared dim — plus the remedy text.
    msg = str(excinfo.value)
    assert "acme" in msg
    assert "Message" in msg
    assert "8" in msg
    assert "lmstudio/embed-4" in msg
    assert "4" in msg
    assert "cannot be changed in place" in msg
    assert "re-bootstrapping" in msg.lower()


def test_worker_raises_when_no_vector_index_exists_for_message_at_all():
    # -graph.md §3.2's three "no index" edge cases (RANGE-only label, unknown
    # label, nonexistent graph key) all collapse to `read_index_dimension`
    # returning None at the repository layer (pinned directly in
    # test_repository.py); here the worker's own response to `None` is pinned —
    # it must refuse exactly like a real numeric mismatch, never proceed.
    embedder = StubEmbedder([1.0, 0.0, 0.0, 0.0])
    gateway = _StubGateway(embedder, dim=4)
    repo = SpyRepo(expected_dim=4, index_dim=None)
    worker = EmbeddingWorker(repo, models=gateway)

    with pytest.raises(EmbeddingDimensionError):
        worker.embed_message("fresh-ws", msg_id="m1", text="hello")

    assert embedder.seen == []
    assert repo.calls == []


def test_worker_ws_test_dim4_fixture_keeps_working_via_declared_dim_path():
    # AC-11's explicit regression guard: the repo's real ws:test fixture (dim 4,
    # DESIGN §14.7) must keep embedding successfully once declared dim and index
    # dim agree — no behaviour change on the matching path.
    embedder = StubEmbedder([1.0, 0.0, 0.0, 0.0])
    gateway = _StubGateway(embedder, dim=4, ref="lmstudio/embed-4")
    repo = SpyRepo(expected_dim=4, index_dim=4)
    worker = EmbeddingWorker(repo, models=gateway)

    worker.embed_message("test", msg_id="m1", text="about cats")

    assert embedder.seen == ["about cats"]
    assert len(repo.calls) == 1


def test_worker_caches_the_index_dimension_per_ws_label_for_process_lifetime():
    embedder = StubEmbedder([1.0, 0.0, 0.0, 0.0])
    gateway = _StubGateway(embedder, dim=4)
    repo = SpyRepo(expected_dim=4, index_dim=4)
    worker = EmbeddingWorker(repo, models=gateway)

    worker.embed_message("acme", msg_id="m1", text="one")
    worker.embed_message("acme", msg_id="m2", text="two")
    worker.embed_message("acme", msg_id="m3", text="three")

    # three embeds, but the introspection query only ran once — cached after
    # the first successful read (-graph.md §3.3 layer 2).
    assert repo.index_dim_calls == [("acme", "Message", "embedding")]
    assert len(repo.calls) == 3


def test_worker_never_caches_a_failed_index_lookup_reprobes_every_call():
    # An un-bootstrapped workspace can become bootstrapped without a process
    # restart — a cached `None` would permanently and wrongly refuse it, so a
    # miss must never be cached (-graph.md §3.3 layer 2, explicit).
    embedder = StubEmbedder([1.0, 0.0, 0.0, 0.0])
    gateway = _StubGateway(embedder, dim=4)
    repo = SpyRepo(expected_dim=4, index_dim=None)
    worker = EmbeddingWorker(repo, models=gateway)

    with pytest.raises(EmbeddingDimensionError):
        worker.embed_message("fresh-ws", msg_id="m1", text="one")
    with pytest.raises(EmbeddingDimensionError):
        worker.embed_message("fresh-ws", msg_id="m2", text="two")

    # re-probed both times — never cached the None
    assert repo.index_dim_calls == [
        ("fresh-ws", "Message", "embedding"),
        ("fresh-ws", "Message", "embedding"),
    ]

    # workspace becomes bootstrapped between calls (no restart) — the very next
    # call must succeed via a fresh read, not stay stuck on a stale refusal.
    repo._index_dim = 4
    worker.embed_message("fresh-ws", msg_id="m3", text="three")

    assert repo.index_dim_calls == [
        ("fresh-ws", "Message", "embedding"),
        ("fresh-ws", "Message", "embedding"),
        ("fresh-ws", "Message", "embedding"),
    ]
    assert len(repo.calls) == 1


def test_worker_never_queries_or_considers_chunk_when_writing_a_message():
    # FR-19's write-gate scope (m-4, -graph.md §3.3/§3.4, plan §4.5): a `Message`
    # write is gated ONLY on the `Message` index. A divergent (or absent) `Chunk`
    # index must never be consulted, let alone block the embed.
    embedder = StubEmbedder([1.0, 0.0, 0.0, 0.0])
    gateway = _StubGateway(embedder, dim=4)
    repo = SpyRepo(expected_dim=4, index_dim=4)
    worker = EmbeddingWorker(repo, models=gateway)

    worker.embed_message("acme", msg_id="m1", text="hello")

    assert repo.index_dim_calls == [("acme", "Message", "embedding")]
    assert all(label == "Message" for (_ws, label, _prop) in repo.index_dim_calls)


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
