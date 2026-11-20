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

import re
import string
from typing import Any, Literal

# {entityId, name, type, score} — score is RediSearch's raw relevance score,
# not a calibrated probability (document-ingestion-ml.md §4.3).
MatchCandidate = dict[str, Any]


# The `Entity.name` index's own separators, measured live (K-068): ASCII space,
# tab, and ASCII punctuation except `_` and `\\`. Everything else — letters,
# `_`, non-ASCII punctuation and symbols — stays inside an indexed term.
_INDEX_SEPARATORS = re.compile(
    "[" + re.escape(" \t" + "".join(c for c in string.punctuation if c not in "_\\")) + "]+"
)
# Characters the query parser rejects inside a term but the index does not
# separate on: `\\` (an escape to the tokenizer) and the ASCII control
# characters other than tab. Deleted before splitting.
_DELETED = re.compile(r"[\x00-\x08\x0a-\x1f\x7f\\]")


def _fuzzy_query(name: str) -> str:
    """One RediSearch 1-edit fuzzy term per name token (`%token%` syntax,
    `document-ingestion-graph.md` §2.3), space-joined (implicit AND — K-064
    owns that decision).

    Tokens follow the `Entity.name` index's own tokenizer (K-068), measured
    live per character on a throwaway graph: the name is split on ASCII
    space, tab and ASCII punctuation other than `_` and `\\` — `model-bench/
    tests.py` -> `%model% %bench% %tests% %py%`; `\\` and the other ASCII
    control characters are deleted; everything else stays inside a term, as
    it does in the index — `A1–A15` -> `%A1–A15%`. Why:

    * Between word characters, RediSearch's query parser rejects every ASCII
      punctuation and control character except `_` (`RediSearch: Syntax
      error`), so a raw extracted name made the lookup crash and `_safe_fuse`
      drop that entity's suggested tier.
    * The index separates on exactly the characters split on here, so each
      term is one the index can hold for the same name. Stripping punctuation
      inside a token instead (`repository._escape_fuzzy_token`, the title
      path) glues `model-bench` into `%modelbench%`, a term the index never
      holds; splitting on *every* non-word character (`[^\\w]+`) breaks a
      name like `A1–A15` into terms the index never holds either.

    A token made only of underscores is dropped: it carries no word content,
    and a lone `%_%` is itself a syntax error. A name left with no token
    (`--`, `...`, `_`, `\\`) yields an empty string, as does empty/whitespace-
    only input (should not reach here — extraction never hands back a blank
    name, `extraction._coerce_entities`); the caller treats that as "no
    candidates" rather than issuing a degenerate query.

    Known limitation: a name with an emoji inside a word (`aa😀bb`) is not
    found even when stored verbatim — RediSearch misses it under every
    query shape tried.
    """
    return " ".join(
        f"%{tok}%"
        for tok in _INDEX_SEPARATORS.split(_DELETED.sub("", name))
        if tok.strip("_")
    )


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
