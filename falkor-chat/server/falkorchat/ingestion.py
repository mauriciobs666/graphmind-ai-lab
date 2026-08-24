"""Entity/relationship extraction write path (K-050 M5 Stage 3, FR-7a/AC-10).

`IngestionPipeline` is a peer of `AgentResponder`/`EmbeddingWorker` (plan §2.2):
a component `background._safe_extract` calls, out-of-band, per chunk — right
after a document is written (Stage 1) and independent of that chunk's own
background embed (Stage 2's `_safe_embed_chunk`; the two are scheduled side by
side from `api.py`/`mcp.py`, never chained). It calls `extraction.extract` for
the LLM-bearing part, then writes whatever came back via
`repository.create_entity`/`link_chunk_about_entity`/
`create_entity_relationship`.

**Stage 3 posture — no fusion yet (plan §3.1/§3.3).** Every extracted entity
mention becomes a *fresh* `Entity` node, even when an identically-named-and-
typed entity already exists from an earlier chunk or document — extraction
never looks up or reuses an existing entity. This is a deliberate, testable
degenerate case ("fusion permanently at always-create-new"), not a bug: Stage 4
is what adds identity fusion. Do not add any matching/dedup logic here.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from typing import Any

from . import extraction
from .modelconfig import StaticModelGateway


def _default_id() -> str:
    return uuid.uuid4().hex


def _default_clock() -> int:
    """Server clock in milliseconds since the epoch."""
    return int(time.time() * 1000)


class IngestionPipeline:
    """Extract entities/relationships from one chunk and write them.

    `llm`/`models` mirror `EmbeddingWorker`'s FR-4 sugar: a bare injected
    `llm=` client wraps into a `StaticModelGateway`; `models=` is the real
    production `ModelGateway`, resolved through its `extraction` kind
    (`config/models.json`).

    `id_gen`/`clock` mirror `Services`'s own injection seam
    (`services._default_id`/`_default_clock`, same default shapes) — this
    component mints its own fresh entity ids and timestamps because, unlike
    `EmbeddingWorker` (which only ever writes onto a chunk/message id that
    already exists), it creates brand-new `Entity` nodes. Routing that back
    through `Services` would mean a background thread making a synchronous
    call into the request-serving layer — an inversion this codebase's
    layering doesn't have anywhere else; every other background component
    (`EmbeddingWorker`) writes via `repo` directly, and this one does too.
    """

    def __init__(
        self,
        repo: Any,
        llm: Any = None,
        *,
        models: Any = None,
        id_gen: Callable[[], str] = _default_id,
        clock: Callable[[], int] = _default_clock,
    ) -> None:
        self._repo = repo
        self._models = models or StaticModelGateway(llm=llm)
        self._id = id_gen
        self._clock = clock

    def extract_chunk(
        self, ws: str, *, chunk_id: str, document_id: str, text: str
    ) -> None:
        """Extract entities/relationships from `text` and write them, all
        traceable back to `chunk_id`/`document_id` (AC-10).

        Per extracted entity mention (including stub-repaired ones,
        `extraction.py`): a fresh `Entity` + an `ABOUT` edge from `chunk_id`.
        Per extracted relationship: a `RELATES_TO` fact edge between the two
        entities resolved from THIS call's own extraction (matched by
        `extraction.normalize_name`, the same shared normalization helper
        `create_entity`'s `nameNormalized` is written with) — never an entity
        from another chunk/document. A relationship whose subject/object
        can't be resolved to an entity created in this same call is skipped
        defensively (should not happen given `extraction.extract`'s own
        stub-repair guarantee, but one odd relationship must never cost the
        rest of the chunk's extraction, same failure-isolation spirit as
        `background._safe_extract` one layer up).
        """
        llm = self._models.llm("extraction", ws=ws)
        result = extraction.extract(text, llm)

        entity_ids_by_name: dict[str, str] = {}
        for entity in result.entities:
            entity_id = self._id()
            self._repo.create_entity(
                ws, entity_id=entity_id, name=entity["name"],
                name_normalized=extraction.normalize_name(entity["name"]),
                type=entity["type"], created_at=self._clock(),
            )
            self._repo.link_chunk_about_entity(
                ws, chunk_id=chunk_id, entity_id=entity_id,
            )
            # Last-write-wins on a within-chunk normalized-name collision (two
            # mentions of "the same" name in one chunk) — a routine, low-stakes
            # tie-break; every mention still gets its own Entity node + ABOUT
            # edge above, only relationship-endpoint resolution below picks one.
            entity_ids_by_name[extraction.normalize_name(entity["name"])] = entity_id

        for rel in result.relationships:
            subject_id = entity_ids_by_name.get(
                extraction.normalize_name(rel["subject"])
            )
            object_id = entity_ids_by_name.get(
                extraction.normalize_name(rel["object"])
            )
            if subject_id is None or object_id is None:
                continue
            self._repo.create_entity_relationship(
                ws, subject_id=subject_id, object_id=object_id,
                label=rel["predicate"], source_chunk_id=chunk_id,
                source_document_id=document_id, created_at=self._clock(),
            )
