"""Async embedding worker + LM Studio embedder client (K-008, DESIGN §9).

Embeddings are computed **out-of-band, decoupled from the message write path** —
a message must be readable before its embedding lands, and LM Studio latency must
stay off the guarded write. The worker takes a posted `(msgId, text)`, asks an
injectable `Embedder` for the vector, validates its length, then calls
`repository.set_embedding` (which validates again as the last line of defense
against the silent wrong-dim ANN-drop quirk — `docs/archive/plans/m2-graphrag.md` item 2).

The `Embedder` seam is what makes this testable: unit tests inject a deterministic
stub (fixed vectors, no network); production injects `LMStudioEmbedder`. The
worker is a plain callable — it is **not** wired into the post path here; a caller
(e.g. FastAPI `BackgroundTasks` or a queue consumer) schedules `embed_message`
after `services.post_message` returns.
"""

from __future__ import annotations

import json
import urllib.request
from typing import Any, Callable, Protocol

from . import config
from .repository import EmbeddingDimensionError

# A transport is `(url, json_payload) -> parsed_json_dict`. Injected so the
# embedder's response parsing is unit-testable without a live LM Studio server.
Transport = Callable[[str, dict[str, Any]], dict[str, Any]]


class Embedder(Protocol):
    """Anything that turns text into an embedding vector."""

    def embed(self, text: str) -> list[float]: ...


def _urllib_transport(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Default HTTP transport (stdlib only — no runtime HTTP dependency).

    POSTs `payload` as JSON and returns the decoded JSON response. Network/HTTP
    errors propagate to the worker, which runs out-of-band from the write path.
    """
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req) as resp:  # noqa: S310 (fixed, config-controlled URL)
        return json.loads(resp.read().decode("utf-8"))


class LMStudioEmbedder:
    """OpenAI-compatible `/v1/embeddings` client (LM Studio backend).

    `transport` is injectable for tests; it defaults to a stdlib urllib POST.
    """

    def __init__(
        self,
        base_url: str = config.EMBEDDING_BASE_URL,
        model: str = config.EMBEDDING_MODEL,
        *,
        transport: Transport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._transport = transport or _urllib_transport

    def embed(self, text: str) -> list[float]:
        resp = self._transport(
            f"{self._base_url}/embeddings",
            {"model": self._model, "input": text},
        )
        return resp["data"][0]["embedding"]


class EmbeddingWorker:
    """Compute + persist a message's embedding, out-of-band from the write path."""

    def __init__(
        self, repo: Any, embedder: Embedder, *, expected_dim: int | None = None
    ) -> None:
        self._repo = repo
        self._embedder = embedder
        self._dim = config.EMBEDDING_DIM if expected_dim is None else expected_dim

    def embed_message(self, ws: str, *, msg_id: str, text: str) -> list[float]:
        """Embed `text` and write it onto message `msg_id` in workspace `ws`.

        Validates the embedder's output length before writing — a wrong-length
        vector (a buggy or misconfigured model) is rejected loudly rather than
        silently corrupting the message's ANN membership. Returns the vector.
        """
        vector = self._embedder.embed(text)
        if len(vector) != self._dim:
            raise EmbeddingDimensionError(
                f"embedder returned a {len(vector)}-dim vector, expected {self._dim} "
                f"(msgId={msg_id!r}) — refusing to write a wrong-dimension embedding "
                f"(it would silently drop the message out of the ANN index)"
            )
        self._repo.set_embedding(
            ws, msg_id=msg_id, embedding=vector, expected_dim=self._dim
        )
        return vector
