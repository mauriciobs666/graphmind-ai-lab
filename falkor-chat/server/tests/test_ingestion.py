"""Unit tests for `IngestionPipeline` (K-050 M5 Stage 3, FR-7a/AC-10).

Exercises the write-side orchestration with a fake repository and a stub LLM —
no live FalkorDB needed (mirrors `test_embedding.py`'s `SpyRepo` pattern). The
extraction step itself (`extraction.extract`) is unit-tested independently in
`test_extraction.py`; here the focus is what `IngestionPipeline` DOES with an
`ExtractionResult`: which repository calls it makes, in what shape, and how it
resolves relationship endpoints from the SAME call's own extraction.
"""

from __future__ import annotations

import json

from falkorchat import ingestion as ingestion_mod
from falkorchat.extraction import ExtractionResult, normalize_name
from falkorchat.ingestion import IngestionPipeline


class _ReplyLLM:
    def __init__(self, text: str) -> None:
        self._text = text

    def complete(self, messages):
        return self._text


class SpyRepo:
    """Records create_entity/link_chunk_about_entity/create_entity_relationship
    calls; mimics `create_entity`'s `{entityId}` return shape."""

    def __init__(self):
        self.entities: list[dict] = []
        self.about_edges: list[tuple] = []
        self.relationships: list[dict] = []

    def create_entity(self, ws, *, entity_id, name, name_normalized, type, created_at):
        self.entities.append({
            "ws": ws, "entityId": entity_id, "name": name,
            "nameNormalized": name_normalized, "type": type, "createdAt": created_at,
        })
        return {"entityId": entity_id}

    def link_chunk_about_entity(self, ws, *, chunk_id, entity_id):
        self.about_edges.append((ws, chunk_id, entity_id))

    def create_entity_relationship(
        self, ws, *, subject_id, object_id, label, source_chunk_id,
        source_document_id, created_at,
    ):
        self.relationships.append({
            "ws": ws, "subjectId": subject_id, "objectId": object_id, "label": label,
            "sourceChunkId": source_chunk_id, "sourceDocumentId": source_document_id,
            "createdAt": created_at,
        })


def _reply(entities, relationships):
    return json.dumps({"entities": entities, "relationships": relationships})


def _make(reply_text, *, ids=None, clock=None):
    repo = SpyRepo()
    ids = ids or iter(f"e{i}" for i in range(1000))
    clock = clock or (lambda: 500)
    pipeline = IngestionPipeline(
        repo, _ReplyLLM(reply_text), id_gen=lambda: next(ids), clock=clock,
    )
    return pipeline, repo


# ── happy path: entities + relationships written, traceable to chunk/doc ────


def test_extract_chunk_creates_an_entity_per_mention_with_about_edge():
    reply = _reply([{"name": "Alice", "type": "Person"}], [])
    pipeline, repo = _make(reply)

    pipeline.extract_chunk("acme", chunk_id="c1", document_id="d1", text="Alice.")

    assert len(repo.entities) == 1
    entity = repo.entities[0]
    assert entity["ws"] == "acme"
    assert entity["name"] == "Alice"
    assert entity["type"] == "Person"
    assert entity["nameNormalized"] == "alice"
    assert repo.about_edges == [("acme", "c1", entity["entityId"])]


def test_extract_chunk_creates_relates_to_between_entities_from_the_same_extraction():
    reply = _reply(
        [{"name": "Alice", "type": "Person"}, {"name": "Acme", "type": "Organization"}],
        [{"subject": "Alice", "predicate": "works at", "object": "Acme"}],
    )
    pipeline, repo = _make(reply)

    pipeline.extract_chunk("acme", chunk_id="c1", document_id="d1", text="x")

    alice_id = next(e["entityId"] for e in repo.entities if e["name"] == "Alice")
    acme_id = next(e["entityId"] for e in repo.entities if e["name"] == "Acme")
    assert len(repo.relationships) == 1
    rel = repo.relationships[0]
    assert rel["subjectId"] == alice_id
    assert rel["objectId"] == acme_id
    assert rel["label"] == "works at"
    assert rel["sourceChunkId"] == "c1"
    assert rel["sourceDocumentId"] == "d1"


def test_extract_chunk_writes_relationships_for_stub_repaired_entities_too():
    # extraction.extract's own stub-repair adds "Acme" as a stub — the
    # pipeline must resolve the relationship against it just like a real one.
    reply = _reply(
        [{"name": "Alice", "type": "Person"}],
        [{"subject": "Alice", "predicate": "works at", "object": "Acme"}],
    )
    pipeline, repo = _make(reply)

    pipeline.extract_chunk("acme", chunk_id="c1", document_id="d1", text="x")

    names = {e["name"] for e in repo.entities}
    assert names == {"Alice", "Acme"}
    stub = next(e for e in repo.entities if e["name"] == "Acme")
    assert stub["type"] == "Other"
    assert len(repo.relationships) == 1


def test_extract_chunk_mints_a_fresh_id_and_timestamp_per_entity():
    reply = _reply(
        [{"name": "Alice", "type": "Person"}, {"name": "Bob", "type": "Person"}], [],
    )
    ids = iter(["id-a", "id-b"])
    clock_calls = iter([111, 222, 333, 444])
    pipeline, repo = _make(reply, ids=ids, clock=lambda: next(clock_calls))

    pipeline.extract_chunk("acme", chunk_id="c1", document_id="d1", text="x")

    assert [e["entityId"] for e in repo.entities] == ["id-a", "id-b"]
    assert repo.entities[0]["createdAt"] == 111
    assert repo.entities[1]["createdAt"] == 222


# ── one shared normalizer, not two independently-written ones ───────────────


def test_extract_chunk_writes_name_normalized_via_the_shared_helper():
    # Pins that create_entity's nameNormalized argument is computed by
    # extraction.normalize_name specifically, not some other/inline logic —
    # a name whose normalized form is non-trivial (mixed case AND internal
    # whitespace) makes a naive `.lower()`-only normalizer diverge visibly.
    reply = _reply([{"name": "  Acme   CORP  ", "type": "Organization"}], [])
    pipeline, repo = _make(reply)

    pipeline.extract_chunk("acme", chunk_id="c1", document_id="d1", text="x")

    assert repo.entities[0]["nameNormalized"] == normalize_name("  Acme   CORP  ")
    assert repo.entities[0]["nameNormalized"] == "acme corp"


def test_extract_chunk_resolves_a_relationship_across_internal_whitespace_variance():
    # The entity is mentioned with double internal whitespace; the
    # relationship's subject references the same name with single spacing.
    # Only a whitespace-COLLAPSING normalizer (extraction.normalize_name)
    # matches them to the same entity — a normalizer that merely lower()s
    # (no collapse) would treat them as different names and silently drop
    # the relationship, exactly the drift this module's docstring warns
    # against (one shared helper, not two that can diverge).
    reply = _reply(
        [{"name": "Acme  Corp", "type": "Organization"}],
        [{"subject": "Alice", "predicate": "works at", "object": "Acme Corp"}],
    )
    pipeline, repo = _make(reply)

    pipeline.extract_chunk("acme", chunk_id="c1", document_id="d1", text="x")

    assert len(repo.relationships) == 1
    acme_id = next(e["entityId"] for e in repo.entities if e["name"] == "Acme  Corp")
    assert repo.relationships[0]["objectId"] == acme_id


# ── no fusion (stage 3 posture): repeated mentions always create fresh nodes ─


def test_extract_chunk_never_looks_up_or_reuses_an_existing_entity():
    # SpyRepo has no lookup method at all — if IngestionPipeline ever called
    # one, this test would fail with an AttributeError, proving no fusion
    # lookup is attempted at this stage.
    reply = _reply([{"name": "Acme", "type": "Organization"}], [])
    pipeline, repo = _make(reply)

    pipeline.extract_chunk("acme", chunk_id="c1", document_id="d1", text="x")
    pipeline.extract_chunk("acme", chunk_id="c2", document_id="d1", text="x")

    assert len(repo.entities) == 2  # two identical mentions, two fresh nodes
    assert repo.entities[0]["entityId"] != repo.entities[1]["entityId"]


# ── defensive skip: an unresolvable relationship endpoint is dropped, not fatal ─


def test_extract_chunk_skips_a_relationship_whose_endpoint_never_resolves(monkeypatch):
    # extraction.extract's own stub-repair guarantees this can't happen via a
    # real LLM reply — pinned directly against IngestionPipeline's defensive
    # posture by monkeypatching extraction.extract to hand back a dangling
    # relationship (a subject with no corresponding entity), confirming the
    # pipeline skips it rather than raising or writing a broken edge.
    def _fake_extract(chunk_text, llm):
        return ExtractionResult(
            entities=[{"name": "Alice", "type": "Person"}],
            relationships=[
                {"subject": "Alice", "predicate": "knows", "object": "Ghost"},
            ],
        )

    monkeypatch.setattr(ingestion_mod.extraction, "extract", _fake_extract)
    pipeline, repo = _make("irrelevant — extraction.extract is monkeypatched")

    pipeline.extract_chunk("acme", chunk_id="c1", document_id="d1", text="x")

    assert len(repo.entities) == 1  # Alice was still created + linked
    assert repo.relationships == []  # the dangling relationship was skipped


# ── empty extraction result: no writes at all, no crash ─────────────────────


def test_extract_chunk_writes_nothing_on_an_empty_extraction_result():
    pipeline, repo = _make('{"entities": [], "relationships": []}')

    pipeline.extract_chunk("acme", chunk_id="c1", document_id="d1", text="x")

    assert repo.entities == []
    assert repo.about_edges == []
    assert repo.relationships == []


def test_extract_chunk_writes_nothing_on_an_unparseable_reply():
    pipeline, repo = _make("no json here at all")

    pipeline.extract_chunk("acme", chunk_id="c1", document_id="d1", text="x")

    assert repo.entities == []
    assert repo.relationships == []


# ── FR-4 sugar: bare llm= wraps into a StaticModelGateway ────────────────────


def test_pipeline_wraps_a_directly_injected_llm_via_static_gateway_sugar():
    reply = _reply([{"name": "Alice", "type": "Person"}], [])
    repo = SpyRepo()
    pipeline = IngestionPipeline(repo, _ReplyLLM(reply))

    pipeline.extract_chunk("acme", chunk_id="c1", document_id="d1", text="x")

    assert len(repo.entities) == 1
