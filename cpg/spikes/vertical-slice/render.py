"""Render-time helpers over the stored graph (S17 note section 1.2, D6).

``Link.candidateCount`` is gone: a Link is one provider, so "may call any of N" is a property of a
``VIA`` edge. For the edge being rendered, N is the number of Links, the rendered one included, that have
a ``VIA`` edge from a ``CallSite`` of the same consumer symbol (reached through ``HAS_CALL <-``) with an
equal ``VIA.consumerKey``. One home: :func:`candidate_count` (rows) and :data:`CANDIDATE_COUNT_CYPHER` (store).
"""
from __future__ import annotations

from typing import Iterable

#: one parameterized read; ``$site``, ``$link`` and ``$key`` name the VIA edge being rendered (a site can reach
#: one Link under two consumer keys: two edges, two N)
CANDIDATE_COUNT_CYPHER = (
    "MATCH (s:CallSite {callSiteId: $site})-[v:VIA]->(:Link {linkId: $link}) WHERE v.consumerKey = $key "
    "MATCH (c:Symbol)-[:HAS_CALL]->(s) "
    "MATCH (c)-[:HAS_CALL]->(:CallSite)-[w:VIA]->(l:Link) WHERE w.consumerKey = v.consumerKey "
    "RETURN count(DISTINCT l)"
)


def candidate_count(via_rows: Iterable[tuple[str, str, str, str]], site_id: str, link_id: str, consumer_key: str) -> int:
    """N for the ``VIA`` edge ``site_id -> link_id`` carrying ``consumer_key``, over
    ``(consumer symbol, site, link, consumerKey)`` rows.

    ``KeyError`` when that edge is not among the rows (an edge that is not there has no N)."""
    rows = list(via_rows)
    owners = {sym for sym, s, l, k in rows if (s, l, k) == (site_id, link_id, consumer_key)}
    if not owners:
        raise KeyError(f"no VIA edge {site_id} -> {link_id} with consumerKey {consumer_key!r}")
    (sym,) = owners
    return len({l for sy, _, l, k in rows if sy == sym and k == consumer_key})
