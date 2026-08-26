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
    """Compute + persist a message's or chunk's embedding, out-of-band from the
    write path.

    `embedder` is the pre-K-042 direct-injection seam, kept as sugar (FR-4, §3):
    `__init__` wraps it into a `StaticModelGateway` when `models` is not given, so
    the many existing `EmbeddingWorker(repo, stub_embedder)` test constructions keep
    working unmodified. `models` is the real `ModelGateway` production wiring passes.

    K-050 M5 Stage 2 adds `embed_chunk`, a sibling of `embed_message` — same
    worker, same `ModelGateway`/dimension-guard machinery, writing `Chunk`
    instead of `Message`. The two write paths are gated independently (each
    consults only its own label's vector index, §3.3/§3.4 of
    `document-ingestion-graph.md` — `embed_message` still never consults
    `Chunk`, pinned by `test_worker_never_queries_or_considers_chunk_when_writing_a_message`,
    and `embed_chunk` never consults `Message`).
    """

    # `EmbeddingWorker` is a process-lifetime singleton (`app.py::_build_default_app`
    # constructs exactly one) — the two labels it writes today (§3.3 layer 2).
    _WRITE_LABEL = "Message"
    _CHUNK_WRITE_LABEL = "Chunk"

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
        # K-042 Landing 2 (FR-19, `-graph.md` §3.3 layer 2 — "the correctness
        # boundary"): the introspected index dimension, cached per `(ws, label)` for
        # the process lifetime — the dimension provably cannot change in place
        # (§3.1: only drop+recreate, an out-of-band admin action, changes it). A
        # `None` (no vector index yet — e.g. an un-bootstrapped workspace) is
        # **never** cached: a workspace can become bootstrapped without a process
        # restart, so a cached miss would permanently and wrongly refuse it.
        self._index_dim_cache: dict[tuple[str, str], int] = {}

    @property
    def repo(self) -> Any:
        """The injected repository (K-051) — lets `background._safe_embed_chunk`
        report a chunk-embed job's completion back onto its owning `Document`
        (`repository.report_document_job_done`) without this worker having to
        know anything about document-level status itself."""
        return self._repo

    def _index_dimension(self, ws: str, label: str) -> int | None:
        key = (ws, label)
        cached = self._index_dim_cache.get(key)
        if cached is not None:
            return cached
        dim = self._repo.read_index_dimension(ws, label=label)
        if dim is not None:
            self._index_dim_cache[key] = dim
        return dim

    def _resolve_and_embed(
        self, ws: str, *, write_label: str, text: str, id_desc: str,
    ) -> tuple[list[float], int]:
        """Shared pre-flight + embed for `embed_message`/`embed_chunk` (K-050
        Stage 2 factors this out of what was originally `embed_message` alone —
        the FR-19 guard and dimension-resolution logic are identical for both
        labels, only *which* label's index is consulted differs).

        **Pre-flight (FR-19):** before calling the embedder at all, the resolved
        model's *declared* dimension is compared against `write_label`'s
        *introspected* vector-index dimension — the label the caller is about to
        write. A caller writing `Message` never consults `Chunk`'s index (and
        vice versa) — each label's write is gated only on its own index
        (`-graph.md` §3.3/§3.4, plan §4.5). On mismatch (including "no vector
        index exists for this label in this workspace at all" — an
        un-bootstrapped workspace, a `RANGE`-only label, or an unknown label all
        read back as `None` and are treated as a mismatch, since none of them is
        a dimension the model's output could ever satisfy) this raises
        `EmbeddingDimensionError` **before any HTTP call**: no wasted inference,
        no vector computed.

        Validates the embedder's output length before returning — a wrong-length
        vector (a buggy or misconfigured model) is rejected loudly rather than
        silently corrupting the node's ANN membership.

        Dimension precedence (§4.5): an explicit `expected_dim` constructor
        override wins outright; otherwise the resolved model's declared `dim`
        (overlay `models."<ref>".dim`, authoritative when present) is used;
        otherwise `config.EMBEDDING_DIM` is the final fallback.

        Returns `(vector, dim)` — the caller (`embed_message`/`embed_chunk`)
        still owns the actual `repo.set_*` write, since each label's write
        method differs.
        """
        embedder = self._models.embedder("embedding", ws=ws)
        if self._expected_dim is not None:
            dim = self._expected_dim
            model_ref = "(explicit expected_dim override)"
        else:
            resolution = self._models.resolve("embedding", ws=ws)
            model_ref = resolution.primary.ref
            dim = resolution.primary.dim
            if dim is None:
                dim = config.EMBEDDING_DIM

        index_dim = self._index_dimension(ws, write_label)
        if index_dim != dim:
            raise EmbeddingDimensionError(
                f"embedding dimension mismatch for workspace {ws!r}, label "
                f"{write_label!r}: the workspace's vector index is "
                f"dimension {index_dim!r}, but the configured embedding model "
                f"{model_ref!r} declares dimension {dim!r} ({id_desc}) — "
                f"refusing to embed before calling the model (no HTTP call made, "
                f"no vector written). The index dimension cannot be changed in "
                f"place, and re-bootstrapping does NOT change it (dropping and "
                f"recreating the index is the only way) — either configure a "
                f"model whose declared dimension matches the index, or create a "
                f"new workspace at the desired dimension."
            )

        vector = embedder.embed(text)
        if len(vector) != dim:
            raise EmbeddingDimensionError(
                f"embedder returned a {len(vector)}-dim vector, expected {dim} "
                f"({id_desc}) — refusing to write a wrong-dimension embedding "
                f"(it would silently drop the node out of the ANN index)"
            )
        return vector, dim

    def embed_message(self, ws: str, *, msg_id: str, text: str) -> list[float]:
        """Embed `text` and write it onto message `msg_id` in workspace `ws`.

        See `_resolve_and_embed` for the FR-19 pre-flight guard and dimension
        precedence — unchanged from before K-050 Stage 2 factored it out.
        """
        vector, dim = self._resolve_and_embed(
            ws, write_label=self._WRITE_LABEL, text=text, id_desc=f"msgId={msg_id!r}",
        )
        self._repo.set_embedding(
            ws, msg_id=msg_id, embedding=vector, expected_dim=dim
        )
        return vector

    def embed_chunk(self, ws: str, *, chunk_id: str, text: str) -> list[float]:
        """Embed `text` and write it onto chunk `chunk_id` in workspace `ws`.

        K-050 M5 Stage 2 (FR-3): mirrors `embed_message` exactly, writing
        `Chunk` instead of `Message` — same out-of-band, decoupled-from-the-
        write-path posture (a chunk is readable before its embedding lands,
        plan §4 Stage 2's "done" condition) and the same FR-19 pre-flight
        dimension guard, gated on `Chunk`'s own vector index only (never
        `Message`'s). See `_resolve_and_embed`.
        """
        vector, dim = self._resolve_and_embed(
            ws, write_label=self._CHUNK_WRITE_LABEL, text=text,
            id_desc=f"chunkId={chunk_id!r}",
        )
        self._repo.set_chunk_embedding(
            ws, chunk_id=chunk_id, embedding=vector, expected_dim=dim
        )
        return vector
