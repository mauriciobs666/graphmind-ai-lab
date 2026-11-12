"""The answer engine (C-1001 P1-21 onward): the reader layer and the anchors.

Plan: ``docs/plans/connected-code-graph-p1-answers.md`` P1-21. The reader (``ccg.engine.reader``) is the ONLY code in the engine that talks to the store: read-only
(``GRAPH.RO_QUERY`` as ``ccg_engine_<s>``), parameters only, a timeout on every call, bounded replies. Everything it returns is untrusted text from a repo.

Seams left for later steps (each is named where it lives, with a ``SEAM(P1-xx)`` comment):
  * P1-22 (BFS rows, guards): the walk calls ``Reader.read(<expansion name>, ...)`` and sorts nothing itself; the label of a ``name_in``/``name_out`` candidate
    (``possible``/``name-match`` versus ``resolution-unverified``) is derived from its ``count(x)`` column there, not here.
  * P1-23 (flow fallback): nothing is read here; ``units`` returns ``flowAvailable`` raw.
  * P1-39 (staleness, stamp, builder): the ``snapshot`` and ``units`` reads are registered raw; the stamp and the staleness derivation are not built here.
"""
