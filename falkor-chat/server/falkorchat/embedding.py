"""Async embedding worker + OpenAI-compatible embedder client (K-008, DESIGN §9;
K-042 Landing 1: `LMStudioEmbedder` renamed/generalized to `OpenAICompatibleEmbedder`
— LM Studio is one OpenAI-compatible backend among several, not the only one).

Embeddings are computed **out-of-band, decoupled from the message write path** —
a message must be readable before its embedding lands, and provider latency must
stay off the guarded write. The worker takes a posted `(msgId, text)`, asks an
injectable `Embedder` for the vector, validates its length, then calls
`repository.set_embedding` (which validates again as the last line of defense
against the silent wrong-dim ANN-drop quirk — `docs/archive/plans/m2-graphrag.md` item 2).

The `Embedder` seam is what makes this testable: unit tests inject a deterministic
stub (fixed vectors, no network); production resolves a client through `ModelGateway`
(FR-4 — `modelconfig.py` is the only module besides `tests/` allowed to construct
`OpenAICompatibleEmbedder` directly). The worker is a plain callable — it is **not**
wired into the post path here; a caller (e.g. FastAPI `BackgroundTasks` or a queue
consumer) schedules `embed_message` after `services.post_message` returns.
"""

from __future__ import annotations

from typing import Any, Protocol

from . import config
from .modelconfig import StaticModelGateway
from .repository import EmbeddingDimensionError
from .transport import ProviderCallError, Transport, make_http_transport

# Used only when a caller constructs the client with no injected transport (direct
# construction is a tests-only / dev affordance — production always builds the
# transport via `modelconfig.py`, which knows the resolved model's real timeout).
_DIRECT_CONSTRUCTION_TIMEOUT = 180.0


class Embedder(Protocol):
    """Anything that turns text into an embedding vector."""

    def embed(self, text: str) -> list[float]: ...


def _default_transport() -> Transport:
    """The fallback transport for a directly-constructed client (tests/dev only —
    production always builds the transport in `modelconfig.py`)."""
    return make_http_transport(timeout=_DIRECT_CONSTRUCTION_TIMEOUT)


class OpenAICompatibleEmbedder:
    """OpenAI-compatible `/v1/embeddings` client (K-042: generalized from the
    LM-Studio-only `LMStudioEmbedder`; any `protocol="openai"` provider works the
    same way).

    `base_url`/`model` are **required** — no config-derived defaults (FR-20). `transport`
    is injectable for tests; it defaults to a bare urllib POST via a generic timeout
    fallback (real production timeouts come from `modelconfig.py`). `params` are
    per-model settings passed through into the request payload (L1-2 passthrough rule).

    **FR-4:** only `modelconfig.py` and `tests/` may construct this client directly.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        transport: Transport | None = None,
        params: dict[str, Any] | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._transport = transport or _default_transport()
        self._params = dict(params) if params else {}

    def embed(self, text: str) -> list[float]:
        resp = self._transport(
            f"{self._base_url}/embeddings",
            {"model": self._model, "input": text, **self._params},
        )
        try:
            return resp["data"][0]["embedding"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderCallError(
                f"{self._model} @ {self._base_url}: response missing data: {exc}"
            ) from exc


class EmbeddingWorker:
    """Compute + persist a message's embedding, out-of-band from the write path.

    `embedder` is the pre-K-042 direct-injection seam, kept as sugar (FR-4, §3):
    `__init__` wraps it into a `StaticModelGateway` when `models` is not given, so
    the many existing `EmbeddingWorker(repo, stub_embedder)` test constructions keep
    working unmodified. `models` is the real `ModelGateway` production wiring passes.
    """

    def __init__(
        self,
        repo: Any,
        embedder: Embedder | None = None,
        *,
        models: Any = None,
        expected_dim: int | None = None,
    ) -> None:
        self._repo = repo
        self._models = models or StaticModelGateway(embedder=embedder)
        self._expected_dim = expected_dim

    def embed_message(self, ws: str, *, msg_id: str, text: str) -> list[float]:
        """Embed `text` and write it onto message `msg_id` in workspace `ws`.

        Validates the embedder's output length before writing — a wrong-length
        vector (a buggy or misconfigured model) is rejected loudly rather than
        silently corrupting the message's ANN membership. Returns the vector.

        Dimension precedence (§4.5): an explicit `expected_dim` constructor override
        wins outright; otherwise the resolved model's declared `dim` (overlay
        `models."<ref>".dim`, authoritative when present) is used; otherwise
        `config.EMBEDDING_DIM` is the final fallback.
        """
        embedder = self._models.embedder("embedding", ws=ws)
        if self._expected_dim is not None:
            dim = self._expected_dim
        else:
            resolution = self._models.resolve("embedding", ws=ws)
            dim = resolution.primary.dim
            if dim is None:
                dim = config.EMBEDDING_DIM

        vector = embedder.embed(text)
        if len(vector) != dim:
            raise EmbeddingDimensionError(
                f"embedder returned a {len(vector)}-dim vector, expected {dim} "
                f"(msgId={msg_id!r}) — refusing to write a wrong-dimension embedding "
                f"(it would silently drop the message out of the ANN index)"
            )
        self._repo.set_embedding(
            ws, msg_id=msg_id, embedding=vector, expected_dim=dim
        )
        return vector
