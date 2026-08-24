"""Unit tests for LLM-based entity/relationship extraction (K-050 M5 Stage 3, FR-7a).

Mirrors K-027's parser-robustness fixture shapes (`test_app.py`'s guard-judge
tests) for the "does the tolerant parse survive a fenced/prose-wrapped reply"
angle, plus this module's own app-side schema validation, enum coercion,
stub-repair, and per-chunk caps — none of which `extract_own_line_json_object`
itself provides (ML note F1).
"""

from __future__ import annotations

import json

from falkorchat.extraction import (
    ENTITY_TYPES,
    MAX_ENTITIES_PER_CHUNK,
    MAX_RELATIONSHIPS_PER_CHUNK,
    ExtractionResult,
    extract,
    normalize_name,
)


class _ReplyLLM:
    """Stub LLM returning one canned reply, recording the messages it was sent."""

    def __init__(self, text: str) -> None:
        self._text = text
        self.calls: list[list[dict]] = []

    def complete(self, messages):
        self.calls.append(messages)
        return self._text


def _extract(reply: str) -> ExtractionResult:
    return extract("some chunk text", _ReplyLLM(reply))


# ── normalize_name ───────────────────────────────────────────────────────────


def test_normalize_name_case_folds_and_collapses_whitespace():
    assert normalize_name("  Acme   Corp  ") == "acme corp"
    assert normalize_name("ACME CORP") == "acme corp"
    assert normalize_name("Acme\nCorp") == "acme corp"


def test_normalize_name_is_the_matching_key_used_by_repair():
    assert normalize_name("IBM") == normalize_name("  ibm ")


# ── happy path: bare JSON, fenced JSON, prose-wrapped ────────────────────────


def test_extract_parses_bare_json_entities_and_relationships():
    reply = json.dumps({
        "entities": [
            {"name": "Alice", "type": "Person"},
            {"name": "Acme", "type": "Organization"},
        ],
        "relationships": [
            {"subject": "Alice", "predicate": "works at", "object": "Acme"},
        ],
    })

    result = _extract(reply)

    assert result.entities == [
        {"name": "Alice", "type": "Person"},
        {"name": "Acme", "type": "Organization"},
    ]
    assert result.relationships == [
        {"subject": "Alice", "predicate": "works at", "object": "Acme"},
    ]


def test_extract_sends_chunk_text_as_user_message_and_a_system_prompt():
    llm = _ReplyLLM('{"entities": [], "relationships": []}')
    extract("Alice works at Acme.", llm)

    assert len(llm.calls) == 1
    messages = llm.calls[0]
    assert messages[0]["role"] == "system"
    assert messages[1] == {"role": "user", "content": "Alice works at Acme."}
    # the closed taxonomy is given to the model explicitly, one-line definitions
    for etype in ENTITY_TYPES:
        assert etype in messages[0]["content"]


def test_extract_parses_a_fenced_json_reply():
    # K-027 item 1: a correct reply wrapped in a ```json fence must not break.
    payload = {
        "entities": [{"name": "Bob", "type": "Person"}], "relationships": [],
    }
    reply = "```json\n" + json.dumps(payload) + "\n```"

    result = _extract(reply)

    assert result.entities == [{"name": "Bob", "type": "Person"}]


def test_extract_parses_an_own_line_json_reply_amid_prose():
    payload = {"entities": [{"name": "Bob", "type": "Person"}], "relationships": []}
    reply = "Here is my extraction:\n" + json.dumps(payload) + "\nDone."

    result = _extract(reply)

    assert result.entities == [{"name": "Bob", "type": "Person"}]


def test_extract_returns_empty_result_on_unparseable_prose():
    result = _extract("I could not find anything meaningful in this passage.")

    assert result == ExtractionResult()


def test_extract_returns_empty_result_on_a_quoted_verdict_not_asserted():
    # gate B-1 style: a JSON object merely quoted mid-sentence must not be
    # lifted out as if asserted (extract_own_line_json_object's own discipline).
    reply = (
        'If I had to guess I would say {"entities": [{"name": "X", "type": '
        '"Other"}], "relationships": []} but nothing is clearly named here.'
    )

    result = _extract(reply)

    assert result == ExtractionResult()


def test_extract_handles_the_explicit_empty_result_shape():
    result = _extract('{"entities": [], "relationships": []}')

    assert result == ExtractionResult(entities=[], relationships=[])


# ── mandatory app-side schema validation (ML note F1) ────────────────────────


def test_extract_rejects_payload_missing_entities_key():
    result = _extract(json.dumps({"relationships": []}))

    assert result == ExtractionResult()


def test_extract_rejects_payload_missing_relationships_key():
    result = _extract(json.dumps({"entities": []}))

    assert result == ExtractionResult()


def test_extract_rejects_when_entities_is_not_a_list():
    result = _extract(json.dumps({"entities": "not a list", "relationships": []}))

    assert result == ExtractionResult()


def test_extract_rejects_when_relationships_is_not_a_list():
    result = _extract(json.dumps({"entities": [], "relationships": "nope"}))

    assert result == ExtractionResult()


def test_extract_skips_an_entity_item_missing_a_name_but_keeps_the_rest():
    reply = json.dumps({
        "entities": [
            {"type": "Person"},  # no name — dropped, not fatal
            {"name": "Bob", "type": "Person"},
        ],
        "relationships": [],
    })

    result = _extract(reply)

    assert result.entities == [{"name": "Bob", "type": "Person"}]


def test_extract_skips_a_relationship_item_missing_a_field_but_keeps_the_rest():
    reply = json.dumps({
        "entities": [
            {"name": "Alice", "type": "Person"}, {"name": "Acme", "type": "Organization"},
        ],
        "relationships": [
            {"subject": "Alice", "predicate": "works at"},  # no object — dropped
            {"subject": "Alice", "predicate": "works at", "object": "Acme"},
        ],
    })

    result = _extract(reply)

    assert result.relationships == [
        {"subject": "Alice", "predicate": "works at", "object": "Acme"},
    ]


# ── closed entity-type enum: out-of-enum / missing coerces to Other ─────────


def test_extract_coerces_an_out_of_enum_type_to_other():
    reply = json.dumps({
        "entities": [{"name": "Acme", "type": "Company"}], "relationships": [],
    })

    result = _extract(reply)

    assert result.entities == [{"name": "Acme", "type": "Other"}]


def test_extract_coerces_a_missing_type_to_other():
    reply = json.dumps({"entities": [{"name": "Acme"}], "relationships": []})

    result = _extract(reply)

    assert result.entities == [{"name": "Acme", "type": "Other"}]


def test_extract_keeps_every_valid_enum_type_unchanged():
    reply = json.dumps({
        "entities": [{"name": n, "type": t} for n, t in zip(
            ["a", "b", "c", "d", "e", "f", "g"], ENTITY_TYPES
        )],
        "relationships": [],
    })

    result = _extract(reply)

    assert [e["type"] for e in result.entities] == list(ENTITY_TYPES)


# ── stub-repair: a dangling subject/object gets a synthesized stub entity ───


def test_extract_repairs_a_dangling_relationship_object_with_a_stub_entity():
    reply = json.dumps({
        "entities": [{"name": "Alice", "type": "Person"}],
        "relationships": [
            {"subject": "Alice", "predicate": "works at", "object": "Acme Corp"},
        ],
    })

    result = _extract(reply)

    names = {e["name"] for e in result.entities}
    assert names == {"Alice", "Acme Corp"}
    stub = next(e for e in result.entities if e["name"] == "Acme Corp")
    assert stub["type"] == "Other"


def test_extract_stub_repair_matches_case_and_whitespace_insensitively():
    # "Acme" (already an entity, differently cased/spaced) must NOT get a
    # duplicate stub — normalize_name is the shared matching key.
    reply = json.dumps({
        "entities": [{"name": "  ACME  ", "type": "Organization"}],
        "relationships": [
            {"subject": "Alice", "predicate": "works at", "object": "acme"},
        ],
    })

    result = _extract(reply)

    org_entities = [e for e in result.entities if e["name"] == "  ACME  "]
    assert len(org_entities) == 1  # not duplicated
    # Alice (the subject) still gets its own stub, since it wasn't declared
    assert any(e["name"] == "Alice" and e["type"] == "Other" for e in result.entities)


def test_extract_never_silently_drops_a_relationship_fact_for_a_missing_entity():
    # ML note §3.2: losing the FACT is the costlier failure than a rough stub.
    reply = json.dumps({
        "entities": [],
        "relationships": [
            {"subject": "Alice", "predicate": "works at", "object": "Acme"},
        ],
    })

    result = _extract(reply)

    assert result.relationships == [
        {"subject": "Alice", "predicate": "works at", "object": "Acme"},
    ]
    assert {e["name"] for e in result.entities} == {"Alice", "Acme"}


# ── caps: truncate, never error ──────────────────────────────────────────────


def test_extract_truncates_relationships_at_the_cap():
    entities = [{"name": f"e{i}", "type": "Other"} for i in range(2)]
    relationships = [
        {"subject": "e0", "predicate": f"rel{i}", "object": "e1"}
        for i in range(MAX_RELATIONSHIPS_PER_CHUNK + 5)
    ]
    reply = json.dumps({"entities": entities, "relationships": relationships})

    result = _extract(reply)

    assert len(result.relationships) == MAX_RELATIONSHIPS_PER_CHUNK


def test_extract_truncates_raw_entities_at_the_cap_when_no_repair_is_needed():
    entities = [
        {"name": f"e{i}", "type": "Other"} for i in range(MAX_ENTITIES_PER_CHUNK)
    ]
    reply = json.dumps({"entities": entities, "relationships": []})

    result = _extract(reply)

    assert len(result.entities) == MAX_ENTITIES_PER_CHUNK


def test_extract_stub_repair_is_not_truncated_away_by_the_entity_cap():
    # Regression for Pass 3 MAJOR 1: entities already AT the cap, and a
    # relationship references a name not among them. The raw entities must be
    # capped BEFORE repair runs, so the stub repair adds ON TOP (entity count
    # may exceed MAX_ENTITIES_PER_CHUNK afterward) rather than being sliced
    # back off by a truncation that runs after repair — which would silently
    # drop the relationship fact, exactly what stub-repair exists to prevent.
    entities = [
        {"name": f"e{i}", "type": "Other"} for i in range(MAX_ENTITIES_PER_CHUNK)
    ]
    relationships = [
        {"subject": "e0", "predicate": "relatesTo", "object": "StubOnly"},
    ]
    reply = json.dumps({"entities": entities, "relationships": relationships})

    result = _extract(reply)

    # the fact survives — this is the whole point of the fix
    assert result.relationships == [
        {"subject": "e0", "predicate": "relatesTo", "object": "StubOnly"},
    ]
    # its endpoint is resolvable: the stub was NOT truncated away
    assert any(e["name"] == "StubOnly" for e in result.entities)
    assert len(result.entities) == MAX_ENTITIES_PER_CHUNK + 1


def test_extract_a_relationship_dropped_by_the_cap_gets_no_stub_entity():
    # Relationships are truncated BEFORE stub-repair runs, so a relationship
    # cut by the cap never spawns a stub for a name found nowhere else.
    entities = [{"name": "e0", "type": "Other"}]
    kept = [
        {"subject": "e0", "predicate": f"rel{i}", "object": "e0"}
        for i in range(MAX_RELATIONSHIPS_PER_CHUNK)
    ]
    dropped = [
        {"subject": "e0", "predicate": "only-in-dropped-rel", "object": "ghost-entity"}
    ]
    reply = json.dumps({"entities": entities, "relationships": kept + dropped})

    result = _extract(reply)

    assert len(result.relationships) == MAX_RELATIONSHIPS_PER_CHUNK
    assert all(e["name"] != "ghost-entity" for e in result.entities)
