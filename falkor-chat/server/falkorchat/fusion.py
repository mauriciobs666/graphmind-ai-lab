"""Suggested-tier entity-fusion candidate generation (K-050 M5 Stage 4, FR-9).

The FR-8 exact tier is **not** here — as of the concurrency fix
(`docs/plans/document-ingestion.md` §3.4's "Concurrency note",
`docs/plans/document-ingestion-graph.md` §1.8), the exact-tier candidate
lookup, the new entity's creation, and its conditional auto-link are folded
into one atomic repository call, `Repository.create_entity_with_auto_match`.
This module holds only the FR-9 suggested tier, unaffected by that fix (a
missed/duplicated fuzzy suggestion under concurrent timing still lands in the
reviewed `pending` queue either way — nothing here is ever auto-confirmed
without a human/agent decision, plan §3.4):

* `find_fuzzy_candidates` — RediSearch fuzzy full-text lookup against
  `Entity.name` (`document-ingestion-graph.md` §2.3), type-filtered.
* `classify_fuzzy` — the OQ-1 default's fuzzy branch: any fuzzy hit at all is
  `'suggested'`, since v1 has no calibrated numeric threshold layered on top
  of RediSearch's raw relevance score (`document-ingestion-ml.md` §4.3 — the
  score is stored for audit/UI purposes, not used to gate a second tier here).

`IngestionPipeline` (`ingestion.py`) calls both, only when
`create_entity_with_auto_match` reports `exactMatched=False` for a given
entity.
"""

from __future__ import annotations

from typing import Any, Literal

# {entityId, name, type, score} — score is RediSearch's raw relevance score,
# not a calibrated probability (document-ingestion-ml.md §4.3).
MatchCandidate = dict[str, Any]


def _fuzzy_query(name: str) -> str:
    """One RediSearch 1-edit fuzzy term per name token (`%token%` syntax,
    `document-ingestion-graph.md` §2.3), space-joined. Empty/whitespace-only
    input (should not reach here — extraction never hands back a blank name,
    `extraction._coerce_entities`) yields an empty string; the caller treats
    that as "no candidates" rather than issuing a degenerate query.
    """
    tokens = name.split()
    return " ".join(f"%{tok}%" for tok in tokens)


def find_fuzzy_candidates(
    repo: Any, ws: str, name: str, type: str, limit: int = 5
) -> list[MatchCandidate]:
    """FR-9 suggested-tier candidate generation for one newly-created entity
    that did NOT exact-match (`create_entity_with_auto_match` reported
    `exactMatched=False`).

    Builds the app-side fuzzy query string from `name` (`document-ingestion-
    graph.md` §2.3) and delegates the actual RediSearch call to
    `repo.find_fuzzy_candidates` — this module holds no Cypher itself
    (`repository.py` is "the only place Cypher lives", its own docstring).
    Results come back ranked by score, highest first, `type`-filtered
    (a routine judgment call, not plan-mandated — avoids a fuzzy name hit
    surfacing a nonsensical cross-type suggestion, graph note §2.3).
    """
    fuzzy_query = _fuzzy_query(name)
    if not fuzzy_query:
        return []
    return repo.find_fuzzy_candidates(
        ws, fuzzy_query=fuzzy_query, type=type, limit=limit
    )


def classify_fuzzy(fuzzy: list[MatchCandidate]) -> Literal["suggested", "none"]:
    """The OQ-1 default's fuzzy branch: any candidate at all is `'suggested'`.

    No calibrated numeric threshold in v1 (`document-ingestion-ml.md` §4.3) —
    classification is really just "did `find_fuzzy_candidates` return
    anything." The caller (`IngestionPipeline`) only acts on `'suggested'`
    (taking the top-ranked candidate); `'none'` means no `SAME_AS` edge is
    written for this entity at all.
    """
    return "suggested" if fuzzy else "none"
