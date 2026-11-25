"""Engine constants (``docs/plans/connected-code-graph-p1-answers.md`` P1-21 Files; ``-answer-spec2.md`` §1.2, §4.1)."""
from __future__ import annotations

#: the default depth bound of every resolved answer (``-answer-spec2.md`` §4.1); read by ``ccg.engine.bfs.walk`` and part of its definition's version
DEFAULT_DEPTH_BOUND = 2
#: the mention bound: a hop whose reason is in ``MENTION_REASONS`` is never followed to an arrival past this distance (``min(request bound, MENTION_BOUND)``), by the walk's
#: ``unexplored`` cut; read by ``ccg.engine.bfs.walk`` for the shipped table and part of its definition's version (``docs/plans/connected-code-graph-p1-noise-fix.md`` §4)
MENTION_BOUND = 1
#: the mention class: the edge reasons of a name that occurs in prose (it asserts no call, data or control dependence); read by ``ccg.engine.bfs.walk``. Not part of the definition's
#: fingerprint (the reason set is code, like the cost order: a change here is a version bump by review, not by the fingerprint)
MENTION_REASONS: frozenset[str] = frozenset({"prompt-mention"})
#: the level-synchronous walk's guard (a cycle that escapes the visited set raises, never loops); read by ``ccg.engine.bfs.walk``, not part of its algorithm version
LEVEL_GUARD = 64
#: ids per query; a reply that fills ``RESULTSET_SIZE`` halves it (``-p1-graph.md`` §5)
CHUNK = 1000
#: the image default of ``GRAPH.CONFIG RESULTSET_SIZE`` (10000, v4.18.11), not set by ``up``: a reply of this many rows may have been cut short by the server, so it
#: raises (or halves the chunk). An instance configured lower would cut replies this guard takes as whole; ``cpg/infra/ccg-falkordb.sh status`` reports it and flags a difference.
RESULTSET_SIZE = 10000

#: per-query timeout in milliseconds, sent on every ``GRAPH.RO_QUERY``. The server refuses a value above its ``TIMEOUT_MAX`` (10,000 on the pinned
#: instance); the engine role cannot read ``TIMEOUT_MAX`` (``-p1-graph.md`` §7), so an instance configured lower needs this lowered with it.
QUERY_TIMEOUT_MS = 10_000
#: options an ambiguous anchor lists at most (the reply says when it cut the list)
OPTIONS_MAX = 20
