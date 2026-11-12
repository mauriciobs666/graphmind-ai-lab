"""The answer engine (C-1001 P1-21 onward): the reader layer and the anchors.

Plan: ``docs/plans/connected-code-graph-p1-answers.md`` P1-21. The reader (``ccg.engine.reader``) is the ONLY code in the engine that talks to the store: read-only
(``GRAPH.RO_QUERY`` as ``ccg_engine_<s>``), parameters only, a timeout on every call, bounded replies. Everything it returns is untrusted text from a repo.

Built by P1-22 (``ccg.engine.bfs``, ``ccg.engine.state``; the flow fallback P1-23, ``ccg.engine.flow``): the walk, which calls ``Reader.read(<expansion name>, ...)`` and sorts nothing itself; the label of a
``name_in``/``name_out`` candidate (``possible``/``name-match`` versus ``resolution-unverified``) is derived from its ``count(x)`` column there, not in the reader.

Seams left for later steps (each is named where it lives, with a ``SEAM(P1-xx)`` comment):
  * P1-23 (grouping, distance, the reverse question, the ``site`` hop schema): not built. Built by P1-23 behaviour (1): ``ccg.engine.flow`` and the row gates and labelled
    rows of ``bfs.ROWS`` (the flow fallback, the candidate row tables).
  * P1-39 (staleness, stamp, builder): the ``snapshot`` and ``units`` reads are registered raw; the stamp and the staleness derivation are not built here.
"""
