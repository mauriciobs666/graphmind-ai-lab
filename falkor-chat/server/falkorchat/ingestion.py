"""Entity/relationship extraction + fusion write path (K-050 M5 Stages 3-4,
FR-7a/AC-10, FR-6/7/8/9/10).

`IngestionPipeline` is a peer of `AgentResponder`/`EmbeddingWorker` (plan §2.2):
a component `background._safe_extract` calls, out-of-band, per chunk — right
after a document is written (Stage 1) and independent of that chunk's own
background embed (Stage 2's `_safe_embed_chunk`; the two are scheduled side by
side from `api.py`/`mcp.py`, never chained). It calls `extraction.extract` for
the LLM-bearing part, then writes whatever came back via
`repository.create_entity_with_auto_match`/`link_chunk_about_entity`/
`create_entity_relationship`.

**Stage 4 posture — fusion, not "always create new" (plan §3.4).** Every
extracted entity mention still becomes a *fresh* `Entity` node (fusion never
blocks creation, only decides *linking* after the fact), but now each one is
created via `repository.create_entity_with_auto_match` — the atomic call that
also resolves the FR-8 exact tier (identical normalized name + type) and
auto-links with `status='confirmed'` when a candidate exists, in the same
round trip. When it reports `exactMatched=False`, `fuse_entity` (below) runs
the FR-9 suggested tier: a fuzzy full-text lookup (`fusion.find_fuzzy_
candidates`), and if any candidate comes back, `repository.create_or_reopen_
match` with `status='pending'` for the top-ranked one.

**Per-entity failure isolation for the fusion step (plan §4 Stage 4).** The
fuzzy-tier lookup/write for one entity runs through `background._safe_fuse`
(try/except-log-never-raise), called from *inside* `extract_chunk`'s per-
entity loop — not just at the chunk level `background._safe_extract` already
covers one layer up. This is what keeps a fusion failure for one entity from
aborting the rest of the chunk's entities/relationships (the exact-tier
`create_entity_with_auto_match` call itself is NOT wrapped this way: an entity
that fails to even get created is a more fundamental failure than "found no/
failed on a fuzzy suggestion for an entity that exists," and stays subject to
`_safe_extract`'s existing chunk-level isolation only, same posture as Stage
3's `create_entity` failures).
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from typing import Any

from . import extraction, fusion
from .background import _safe_fuse
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

    @property
    def repo(self) -> Any:
        """The injected repository (K-051) — lets `background._safe_extract`
        report a chunk-extract job's completion back onto its owning
        `Document` (`repository.report_document_job_done`) without this
        pipeline having to know anything about document-level status
        itself."""
        return self._repo

    def extract_chunk(
        self, ws: str, *, chunk_id: str, document_id: str, text: str
    ) -> None:
        """Extract entities/relationships from `text` and write them, all
        traceable back to `chunk_id`/`document_id` (AC-10) — then fuse each
        entity (Stage 4, FR-6/7/8/9/10).

        Per extracted entity mention (including stub-repaired ones,
        `extraction.py`): a fresh `Entity` via `create_entity_with_auto_match`
        (which also resolves/auto-links the FR-8 exact tier atomically) + an
        `ABOUT` edge from `chunk_id`, then `fuse_entity` for the FR-9
        suggested tier when no exact match was found. Per extracted
        relationship: a `RELATES_TO` fact edge between the two entities
        resolved from THIS call's own extraction (matched by
        `extraction.normalize_name`, the same shared normalization helper
        `nameNormalized` is written with) — never an entity from another
        chunk/document. A relationship whose subject/object can't be resolved
        to an entity created in this same call is skipped defensively (should
        not happen given `extraction.extract`'s own stub-repair guarantee,
        but one odd relationship must never cost the rest of the chunk's
        extraction, same failure-isolation spirit as `background._safe_extract`
        one layer up).
        """
        llm = self._models.llm("extraction", ws=ws)
        result = extraction.extract(text, llm)

        entity_ids_by_name: dict[str, str] = {}
        for entity in result.entities:
            entity_id = self._id()
            match = self._repo.create_entity_with_auto_match(
                ws, entity_id=entity_id, name=entity["name"],
                name_normalized=extraction.normalize_name(entity["name"]),
                type=entity["type"], created_at=self._clock(),
                match_id=self._id(),
            )
            self._repo.link_chunk_about_entity(
                ws, chunk_id=chunk_id, entity_id=entity_id,
            )
            # Last-write-wins on a within-chunk normalized-name collision (two
            # mentions of "the same" name in one chunk) — a routine, low-stakes
            # tie-break; every mention still gets its own Entity node + ABOUT
            # edge above, only relationship-endpoint resolution below picks one.
            entity_ids_by_name[extraction.normalize_name(entity["name"])] = entity_id

            # FR-9 suggested tier — only when the atomic call found no exact
            # candidate. Per-entity failure isolation (`_safe_fuse`, see this
            # module's docstring): a fuzzy lookup/write failure for THIS
            # entity must not abort the rest of the chunk's entities/
            # relationships.
            if not match["exactMatched"]:
                _safe_fuse(self, ws, entity_id, entity["name"], entity["type"])

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

    def fuse_entity(self, ws: str, entity_id: str, name: str, type: str) -> None:
        """FR-9 suggested-tier fusion for one entity that did not exact-match.

        `background._safe_fuse` calls this wrapped in try/except-log-never-
        raise, one entity at a time from `extract_chunk`'s own loop, so this
        method itself is free to let a genuine failure (a RediSearch syntax
        error, a transient connection hiccup) propagate — isolation is the
        caller's job, not this method's.

        Looks up fuzzy candidates (`fusion.find_fuzzy_candidates`), **excluding
        `entity_id` itself** — by the time this runs the new entity already
        exists (`create_entity_with_auto_match` created it before reporting
        `exactMatched=False`), so a RediSearch fuzzy query against its own
        name can legitimately return the entity as its own top hit; without
        this filter that would write a spurious `(e)-[:SAME_AS]->(e)`
        self-loop, which no part of this schema's fusion model (`document-
        ingestion-graph.md` §1.5) anticipates or handles. A `'none'`
        classification (no candidates left after that filter) is a silent
        no-op — nothing is written. A `'suggested'` classification takes the
        single top-ranked remaining candidate (already score-sorted by the
        repository) and records it via `create_or_reopen_match` with
        `status='pending'` — never `'confirmed'`, since nothing on this tier
        is ever auto-linked without a human/agent decision (plan §3.4).
        """
        candidates = [
            c for c in fusion.find_fuzzy_candidates(self._repo, ws, name, type)
            if c["entityId"] != entity_id
        ]
        if fusion.classify_fuzzy(candidates) == "none":
            return
        top = candidates[0]
        self._repo.create_or_reopen_match(
            ws, new_entity_id=entity_id, candidate_entity_id=top["entityId"],
            match_id=self._id(), status="pending", confidence=top["score"],
            technique="fuzzy_fulltext", created_at=self._clock(),
        )
