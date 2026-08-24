"""LLM-based entity/relationship extraction (K-050 M5 Stage 3, FR-7a).

Turns one chunk's text into a structured `ExtractionResult` — the entities and
relationships stage 3's `IngestionPipeline` (`ingestion.py`) then writes as fresh
`Entity` nodes + `ABOUT`/`RELATES_TO` edges (no fusion yet, plan §3.1/§3.3).

Design is `data-scientist`'s recommendation, adopted as-is
(`docs/plans/document-ingestion-ml.md` §3): one LLM call per chunk, entities and
relationships combined (not two calls), parsed via the same fence-tolerant,
conservative-by-construction helper the K-027 guard judge uses
(`llm.extract_own_line_json_object`) rather than a bare `json.loads` — this
codebase runs against small local models, and "the model wrapped its JSON reply
in a ```json fence" is a proven, recurring failure mode (K-027 item 1), not a
hypothetical one.

**`require_key` does not fully validate the shape (ML note F1).**
`extract_own_line_json_object`'s "the whole reply is one JSON object" acceptance
path does not check `require_key` at all (only its second, disambiguation path
does) — since the extraction prompt asks for exactly one top-level JSON object,
almost every real reply lands on that first path. So this module does its own,
independent, mandatory schema validation below (`_coerce_entities`/
`_coerce_relationships`) — `require_key="entities"` is still passed (it still
helps disambiguate the second path), but nothing here relies on it to reject a
malformed top-level shape.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .llm import extract_own_line_json_object

# The closed 7-value entity-type taxonomy (ML note §3.1). Closed, not
# open-vocabulary, because FR-8's exact-tier auto-merge (stage 4) is gated on
# `Entity.type` equality — an open type field would let the same real-world
# entity land under two different labels purely from LLM word-choice variance,
# silently defeating that tier's recall for reasons that have nothing to do
# with entity identity. `Other` is the mandatory catch-all so the model always
# has a safe fallback instead of refusing to extract or inventing a new label.
ENTITY_TYPES: tuple[str, ...] = (
    "Person", "Organization", "Location", "Product", "Event", "Concept", "Other",
)
_ENTITY_TYPE_SET = frozenset(ENTITY_TYPES)
_FALLBACK_TYPE = "Other"

# Bounding extraction per chunk (RAM/supernode risk, plan §3.3/§6): truncate
# rather than error if the model returns more than this. Both caps apply to
# the RAW, pre-repair lists, in this order: relationships truncated first (a
# dropped relationship needs no stub-repair support), then the raw entities
# list, THEN stub-repair runs on top, uncapped (post-review fix, Pass 3 MAJOR
# 1 — analyst live-verified that capping entities AFTER repair could slice off
# a stub the just-truncated relationships still reference, silently dropping
# the very fact stub-repair exists to preserve — document-ingestion-ml.md
# §3.2's "avoids silently dropping a relationship fact" guarantee). Stub-repair
# adds at most 2 entities per surviving relationship, so the true worst case
# is `MAX_ENTITIES_PER_CHUNK + 2 * MAX_RELATIONSHIPS_PER_CHUNK` — still a
# small, RAM-safe bound, not the unbounded raw model output the cap exists to
# guard against.
MAX_ENTITIES_PER_CHUNK = 20
MAX_RELATIONSHIPS_PER_CHUNK = 20

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_name(name: str) -> str:
    """Case-fold + whitespace-collapse — the ONE shared normalization helper.

    Both this module's subject/object stub-repair (`_repair_stub_entities`,
    below) and `repository.create_entity`'s `nameNormalized` computation MUST
    use this exact function, not two independently-written normalizers that
    can silently drift apart (`document-ingestion-ml.md` §3.2,
    `document-ingestion-graph.md` §2.2/§7). `IngestionPipeline` is what calls
    this on the repository-write side (`ingestion.py`) — this module only
    calls it for its own in-memory matching.
    """
    return _WHITESPACE_RE.sub(" ", name.strip()).casefold()


_SYSTEM_PROMPT = (
    "You extract entities and relationships from a passage of text for a knowledge "
    "graph. Classify every entity into EXACTLY one of these seven types:\n"
    "- Person: a named individual human being\n"
    "- Organization: a company, institution, government body, or other formal group\n"
    "- Location: a place — a city, country, region, address, or landmark\n"
    "- Product: a named product, service, or technology\n"
    "- Event: a named occurrence tied to a time/place — a meeting, launch, incident\n"
    "- Concept: an abstract idea, topic, or category with no physical form\n"
    "- Other: anything that does not clearly fit one of the above — always use "
    "this instead of inventing a new type\n\n"
    "Reply with a single JSON object and nothing else, in exactly this shape:\n"
    '{"entities": [{"name": "<string>", "type": "<one of the seven types above>"}], '
    '"relationships": [{"subject": "<entity name>", "predicate": "<short string>", '
    '"object": "<entity name>"}]}\n\n'
    "If the passage mentions nothing extractable, reply with exactly "
    '{"entities": [], "relationships": []} — never omit the JSON object, and '
    "never reply with prose."
)


@dataclass(frozen=True)
class ExtractionResult:
    """One chunk's extraction: `{name, type}` entities, `{subject, predicate,
    object}` relationships. Both default to empty — the degenerate "nothing
    extractable, or the reply was unparseable/invalid" result is a *valid*
    `ExtractionResult`, not an exception (mirrors `_safe_embed`'s "never raise"
    posture one layer up, in `background._safe_extract`)."""

    entities: list[dict[str, str]] = field(default_factory=list)
    relationships: list[dict[str, str]] = field(default_factory=list)


_EMPTY = ExtractionResult()


def _coerce_entities(raw: Any) -> list[dict[str, str]] | None:
    """`None` ⇒ reject the whole payload (not a list at all — the one
    top-level shape failure this function can produce). A malformed
    **individual** item (no string `name`) is dropped, not fatal to the rest —
    losing one bad item is cheaper than discarding an otherwise-good chunk's
    extraction. An out-of-enum (or missing/non-string) `type` is coerced to
    `Other` rather than rejected (ML note §3.2: "a wrong type is recoverable
    via fusion review; a missing key is not something to guess at" — that
    "missing key" is about the top-level `entities`/`relationships` keys, not
    a per-item `type`)."""
    if not isinstance(raw, list):
        return None
    out: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        etype = item.get("type")
        if not isinstance(etype, str) or etype not in _ENTITY_TYPE_SET:
            etype = _FALLBACK_TYPE
        out.append({"name": name, "type": etype})
    return out


def _coerce_relationships(raw: Any) -> list[dict[str, str]] | None:
    """Mirrors `_coerce_entities`: `None` ⇒ reject (not a list); a malformed
    individual item (missing/blank `subject`/`predicate`/`object`) is dropped,
    not fatal."""
    if not isinstance(raw, list):
        return None
    out: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        subject, predicate, obj = (
            item.get("subject"), item.get("predicate"), item.get("object"),
        )
        if not all(
            isinstance(v, str) and v.strip() for v in (subject, predicate, obj)
        ):
            continue
        out.append({"subject": subject, "predicate": predicate, "object": obj})
    return out


def _repair_stub_entities(
    entities: list[dict[str, str]], relationships: list[dict[str, str]]
) -> list[dict[str, str]]:
    """Deterministic app-side repair for a `subject`/`object` name the model
    forgot to also list in `entities` — a common LLM extraction bookkeeping
    lapse, not exceptional (ML note §3.2). For any relationship endpoint name
    not found in `entities` (case-fold + whitespace-collapse compare via
    `normalize_name`), synthesize a stub `{name, type: "Other"}` and append it.
    Matching is exact-normalized only, never fuzzy — an under-specified
    reference (e.g. a pronoun) simply won't resolve, an accepted v1 miss."""
    known = {normalize_name(e["name"]) for e in entities}
    repaired = list(entities)
    for rel in relationships:
        for name in (rel["subject"], rel["object"]):
            key = normalize_name(name)
            if key not in known:
                known.add(key)
                repaired.append({"name": name, "type": _FALLBACK_TYPE})
    return repaired


def extract(chunk_text: str, llm: Any) -> ExtractionResult:
    """One LLM call, entities + relationships combined (ML note §3.3 — the v1
    default, not a two-stage entities-then-relationships call).

    `llm` is anything `.complete(messages) -> str`-shaped (the `llm.LLM`
    protocol) — `IngestionPipeline` resolves it through `ModelGateway`'s
    `extraction` kind before calling this.

    Returns `ExtractionResult(entities=[], relationships=[])` — never raises —
    for every "no usable result" case: an unparseable reply, a reply missing
    either top-level key, or a top-level `entities`/`relationships` that isn't
    a list. This mirrors the guard judge's own bias-to-suspend posture
    (`extract_own_line_json_object` returning `None`) one layer up: a
    generation failure here costs one chunk's worth of extraction, never a
    raised exception into the background-scheduling caller (which
    `background._safe_extract` treats as belt-and-suspenders, not the only
    line of defense).
    """
    reply = llm.complete([
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": chunk_text},
    ])
    parsed = extract_own_line_json_object(reply, require_key="entities")
    if parsed is None:
        return _EMPTY
    if "entities" not in parsed or "relationships" not in parsed:
        return _EMPTY

    entities = _coerce_entities(parsed["entities"])
    relationships = _coerce_relationships(parsed["relationships"])
    if entities is None or relationships is None:
        return _EMPTY

    relationships = relationships[:MAX_RELATIONSHIPS_PER_CHUNK]
    # Cap the RAW entities BEFORE repair (Pass 3 MAJOR 1 fix): repair must
    # never have its own just-added stubs sliced off by a truncation that
    # runs after it, or a relationship survives the cap while its endpoint
    # doesn't — the exact silent-fact-loss bug stub-repair exists to prevent.
    entities = entities[:MAX_ENTITIES_PER_CHUNK]
    entities = _repair_stub_entities(entities, relationships)

    return ExtractionResult(entities=entities, relationships=relationships)
