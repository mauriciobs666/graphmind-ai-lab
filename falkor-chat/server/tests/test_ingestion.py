"""Unit tests for `IngestionPipeline` (K-050 M5 Stages 3-4, FR-7a/AC-10,
FR-6/7/8/9/10).

Exercises the write-side orchestration with a fake repository and a stub LLM —
no live FalkorDB needed (mirrors `test_embedding.py`'s `SpyRepo` pattern). The
extraction step itself (`extraction.extract`) is unit-tested independently in
`test_extraction.py`; here the focus is what `IngestionPipeline` DOES with an
`ExtractionResult`: which repository calls it makes, in what shape, how it
resolves relationship endpoints from the SAME call's own extraction, and (new
this stage) how it drives fusion off `create_entity_with_auto_match`'s
`exactMatched` flag. The atomic query's own concurrency guarantee and the
fuzzy-tier Cypher are proven live against real FalkorDB in
`test_repository.py`; this file only pins what Python calls get made.

**One deliberate exception** (bottom of the file, "AC-8 cross-document
fusion"): a single live-FalkorDB-backed test using the real `repo`/`conn`
fixtures instead of `SpyRepo` — plan §5's AC-8 test-strategy row names the
*pipeline* producing a cross-document `SAME_AS` link, not just the repository
primitive in isolation (already proven by `test_repository.py`'s
`test_create_entity_with_auto_match_links_an_existing_candidate` and its
concurrent variant); a `SpyRepo`-scripted version of that scenario would only
re-assert what those primitive-level tests already cover, since `SpyRepo`'s
`create_entity_with_auto_match` is scripted, not a real matcher.
"""

from __future__ import annotations

import json

from falkorchat import db, ingestion as ingestion_mod
from falkorchat.extraction import ExtractionResult, normalize_name
from falkorchat.ingestion import IngestionPipeline


class _ReplyLLM:
    def __init__(self, text: str) -> None:
        self._text = text

    def complete(self, messages):
        return self._text


class SpyRepo:
    """Records create_entity_with_auto_match/link_chunk_about_entity/
    create_entity_relationship/find_fuzzy_candidates/create_or_reopen_match
    calls; mimics each real method's return shape.

    Defaults to "no exact match, no fuzzy candidates" for every entity —
    the same always-fresh-node behavior Stage 3's tests already pin, now
    expressed as fusion finding nothing rather than fusion not existing.
    Script an exact match via `exact_matches[(nameNormalized, type)] =
    {"entityId": ...}`; script fuzzy candidates via `fuzzy_results` (a FIFO
    queue, one entry per `find_fuzzy_candidates` call, mirroring this
    codebase's existing `first_status`-style scripting idiom, `test_services.
    py`'s `FakeRepo`).
    """

    def __init__(self):
        self.entities: list[dict] = []
        self.about_edges: list[tuple] = []
        self.relationships: list[dict] = []
        self.fuzzy_lookups: list[tuple] = []
        self.reopen_calls: list[dict] = []
        self.exact_matches: dict[tuple[str, str], dict] = {}
        self.fuzzy_results: list[list[dict]] = []

    def create_entity_with_auto_match(
        self, ws, *, entity_id, name, name_normalized, type, created_at, match_id,
    ):
        self.entities.append({
            "ws": ws, "entityId": entity_id, "name": name,
            "nameNormalized": name_normalized, "type": type, "createdAt": created_at,
        })
        candidate = self.exact_matches.get((name_normalized, type))
        if candidate is None:
            return {
                "entityId": entity_id, "exactMatched": False,
                "candidateEntityId": None, "matchId": None,
            }
        return {
            "entityId": entity_id, "exactMatched": True,
            "candidateEntityId": candidate["entityId"], "matchId": match_id,
        }

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

    def find_fuzzy_candidates(self, ws, *, fuzzy_query, type, limit=5):
        self.fuzzy_lookups.append((ws, fuzzy_query, type, limit))
        if self.fuzzy_results:
            return self.fuzzy_results.pop(0)
        return []

    def create_or_reopen_match(
        self, ws, *, new_entity_id, candidate_entity_id, match_id, status,
        confidence, technique, created_at,
    ):
        self.reopen_calls.append({
            "ws": ws, "newEntityId": new_entity_id,
            "candidateEntityId": candidate_entity_id, "matchId": match_id,
            "status": status, "confidence": confidence, "technique": technique,
            "createdAt": created_at,
        })
        return {"created": True, "reopened": False, "matchId": match_id, "status": status}


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
    # Each entity now draws TWO ids (its own entityId, plus the matchId
    # `create_entity_with_auto_match` always consumes as a call argument,
    # even when unused because exactMatched comes back False) — the entity
    # ids of interest are draws 1 and 3, not 1 and 2.
    reply = _reply(
        [{"name": "Alice", "type": "Person"}, {"name": "Bob", "type": "Person"}], [],
    )
    ids = iter(["id-a", "match-a", "id-b", "match-b"])
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


# ── fusion never blocks creation: every mention still gets a fresh node ─────


def test_extract_chunk_always_creates_a_fresh_node_even_across_mentions():
    # SpyRepo's default (no scripted exact/fuzzy match) means fusion finds
    # nothing for either mention — still, each call creates its own fresh
    # Entity node (fusion only ever decides LINKING after the fact, never
    # blocks creation, plan §3.4).
    reply = _reply([{"name": "Acme", "type": "Organization"}], [])
    pipeline, repo = _make(reply)

    pipeline.extract_chunk("acme", chunk_id="c1", document_id="d1", text="x")
    pipeline.extract_chunk("acme", chunk_id="c2", document_id="d1", text="x")

    assert len(repo.entities) == 2  # two identical mentions, two fresh nodes
    assert repo.entities[0]["entityId"] != repo.entities[1]["entityId"]
    # No exact/fuzzy match scripted -> no SAME_AS edge attempted for either.
    assert repo.reopen_calls == []


# ── Stage 4 fusion wiring (FR-6/7/8/9/10) ────────────────────────────────────


def test_extract_chunk_exact_match_skips_the_fuzzy_lookup_entirely():
    # create_entity_with_auto_match reporting exactMatched=True means the
    # atomic call already auto-linked (status='confirmed') — the pipeline
    # must not also run a fuzzy lookup/create_or_reopen_match for this entity.
    reply = _reply([{"name": "Acme", "type": "Organization"}], [])
    repo = SpyRepo()
    repo.exact_matches[("acme", "Organization")] = {"entityId": "existing-1"}
    ids = iter(["e1", "m1"])
    pipeline = IngestionPipeline(repo, _ReplyLLM(reply), id_gen=lambda: next(ids))

    pipeline.extract_chunk("acme", chunk_id="c1", document_id="d1", text="x")

    assert len(repo.entities) == 1
    assert repo.fuzzy_lookups == []
    assert repo.reopen_calls == []


def test_extract_chunk_no_exact_match_runs_fuzzy_lookup():
    # exactMatched=False -> the pipeline runs the FR-9 fuzzy tier for this
    # entity. No candidates scripted -> classify_fuzzy is 'none' -> no
    # create_or_reopen_match call, but the lookup itself must still happen.
    reply = _reply([{"name": "Acme", "type": "Organization"}], [])
    pipeline, repo = _make(reply)

    pipeline.extract_chunk("acme", chunk_id="c1", document_id="d1", text="x")

    assert len(repo.fuzzy_lookups) == 1
    ws, fuzzy_query, type_, limit = repo.fuzzy_lookups[0]
    assert ws == "acme"
    assert fuzzy_query == "%Acme%"
    assert type_ == "Organization"
    assert repo.reopen_calls == []


def test_extract_chunk_fuzzy_candidate_writes_a_pending_match_for_the_top_hit():
    reply = _reply([{"name": "Acme", "type": "Organization"}], [])
    repo = SpyRepo()
    repo.fuzzy_results.append([
        {"entityId": "cand-2", "name": "Acme Co", "type": "Organization", "score": 1.5},
        {"entityId": "cand-1", "name": "Acme Corp", "type": "Organization", "score": 3.0},
    ])
    ids = iter(["e1", "m1", "m2"])
    pipeline = IngestionPipeline(repo, _ReplyLLM(reply), id_gen=lambda: next(ids))

    pipeline.extract_chunk("acme", chunk_id="c1", document_id="d1", text="x")

    assert len(repo.reopen_calls) == 1
    call = repo.reopen_calls[0]
    assert call["newEntityId"] == "e1"
    # top-ranked candidate = repository's own order (index 0), not re-sorted
    # by score client-side — pins that IngestionPipeline trusts the
    # repository's ORDER BY score DESC rather than re-deriving it.
    assert call["candidateEntityId"] == "cand-2"
    assert call["status"] == "pending"
    assert call["confidence"] == 1.5
    assert call["technique"] == "fuzzy_fulltext"


def test_extract_chunk_excludes_the_just_created_entity_from_its_own_fuzzy_candidates():
    # By the time fuse_entity runs, the new entity already exists (created by
    # create_entity_with_auto_match) — a fuzzy full-text query against its
    # own name could legitimately return itself as the top hit. Without the
    # self-exclusion filter this would write a spurious self-loop.
    reply = _reply([{"name": "Acme", "type": "Organization"}], [])
    repo = SpyRepo()
    ids = iter(["e1", "m1"])
    pipeline = IngestionPipeline(repo, _ReplyLLM(reply), id_gen=lambda: next(ids))
    repo.fuzzy_results.append([
        {"entityId": "e1", "name": "Acme", "type": "Organization", "score": 5.0},
    ])

    pipeline.extract_chunk("acme", chunk_id="c1", document_id="d1", text="x")

    assert repo.reopen_calls == []  # the only candidate was self -> no-op


def test_extract_chunk_multi_word_name_builds_one_fuzzy_term_per_token():
    reply = _reply([{"name": "Acme Corp", "type": "Organization"}], [])
    pipeline, repo = _make(reply)

    pipeline.extract_chunk("acme", chunk_id="c1", document_id="d1", text="x")

    assert repo.fuzzy_lookups[0][1] == "%Acme% %Corp%"


# ── entity-level fusion failure isolation ────────────────────────────────────


class _FlakyFuzzyRepo(SpyRepo):
    """Raises from `find_fuzzy_candidates` for one specific entity name only —
    proves a fusion failure for ONE entity does not abort the rest of the
    chunk's entities/relationships (unlike `_safe_extract`'s chunk-level
    isolation one layer up, this is per-entity, inside `extract_chunk`
    itself, via `_safe_fuse`)."""

    def __init__(self, *, fails_for: str):
        super().__init__()
        self._fails_for = fails_for

    def find_fuzzy_candidates(self, ws, *, fuzzy_query, type, limit=5):
        if self._fails_for in fuzzy_query:
            raise RuntimeError(f"boom fuzzy lookup for {fuzzy_query}")
        return super().find_fuzzy_candidates(
            ws, fuzzy_query=fuzzy_query, type=type, limit=limit
        )


def test_extract_chunk_one_entitys_fusion_failure_does_not_block_siblings():
    reply = _reply(
        [
            {"name": "Alice", "type": "Person"},
            {"name": "Acme", "type": "Organization"},
        ],
        [{"subject": "Alice", "predicate": "works at", "object": "Acme"}],
    )
    repo = _FlakyFuzzyRepo(fails_for="Alice")
    pipeline = IngestionPipeline(repo, _ReplyLLM(reply))

    # Must not raise — _safe_fuse swallows the per-entity fusion failure.
    pipeline.extract_chunk("acme", chunk_id="c1", document_id="d1", text="x")

    # Both entities were still created, both linked to the chunk, and the
    # relationship between them still got written despite Alice's own
    # fuzzy-lookup blowing up.
    assert {e["name"] for e in repo.entities} == {"Alice", "Acme"}
    assert len(repo.about_edges) == 2
    assert len(repo.relationships) == 1
    # Acme's own (non-failing) fuzzy lookup still ran normally.
    assert any("Acme" in q for _, q, _, _ in repo.fuzzy_lookups)


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


# ── AC-8 cross-document fusion (plan §5 test-strategy row, live-backed) ─────
#
# The one test in this file that uses the REAL `repo`/`conn` fixtures instead
# of `SpyRepo` — see the module docstring for why. Proves IngestionPipeline's
# wiring, not just the repository primitive: two "documents" (two
# `extract_chunk` calls, different `document_id`/`chunk_id`) that each mention
# an entity of the same normalized name + type must end up fused via a
# confirmed `SAME_AS` edge, not just each independently exact-tier-checked in
# isolation.


def test_ac8_two_documents_mentioning_the_same_entity_fuse_via_extract_chunk(repo, conn):
    reply = _reply([{"name": "Acme", "type": "Organization"}], [])
    pipeline = IngestionPipeline(repo, _ReplyLLM(reply))

    pipeline.extract_chunk(
        "test", chunk_id="c-doc-a", document_id="doc-a", text="Acme is a company."
    )
    pipeline.extract_chunk(
        "test", chunk_id="c-doc-b", document_id="doc-b", text="Acme again, elsewhere."
    )

    g = db.workspace_graph(conn, "test")
    # Directed, unlabeled, no undirected+relationship-property-filter combo
    # (the live-verified FalkorDB quirk documented in the kaizen graph —
    # that combination can silently miss a real edge depending on which
    # bound node is declared first in the pattern text). This graph only
    # ever holds these two entities in this test, so a global confirmed-edge
    # count together with the entity count is unambiguous: SAME_AS only ever
    # connects two Entity nodes by construction, so exactly one confirmed
    # edge over exactly two entities can only be the doc-a/doc-b pair.
    [[entity_count]] = g.ro_query("MATCH (n:Entity) RETURN count(n)").result_set
    [[edge_count]] = g.ro_query(
        "MATCH ()-[r:SAME_AS {status:'confirmed'}]->() RETURN count(r)"
    ).result_set

    assert entity_count == 2  # one Entity per "document"
    assert edge_count == 1    # doc-a's entity and doc-b's entity are fused, not two islands
