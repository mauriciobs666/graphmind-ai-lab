"""The answer engine (C-1001 P1-21 onward): the reader layer and the anchors.

Plan: ``docs/plans/connected-code-graph-p1-answers.md`` P1-21. The reader (``ccg.engine.reader``) is the ONLY code in the engine that talks to the store: read-only
(``GRAPH.RO_QUERY`` as ``ccg_engine_<s>``), parameters only, a timeout on every call, bounded replies. Everything it returns is untrusted text from a repo.

Built by P1-22 (``ccg.engine.bfs``, ``ccg.engine.state``; the flow fallback P1-23, ``ccg.engine.flow``; item grouping, the fold and the reverse question P1-23c, ``ccg.engine.group``): the walk, which calls ``Reader.read(<expansion name>, ...)`` and sorts nothing itself; the label of a
``name_in``/``name_out`` candidate (``possible``/``name-match`` versus ``resolution-unverified``) is derived from its ``count(x)`` column there, not in the reader.

Built by P1-39 (``ccg.engine.staleness``, the staleness block and the stamp, P1-39a; ``ccg.engine.answer_build``, the answer builder that assembles the canonical document and validates it with ``ccg.answer``, P1-39b).

Seam left for a later step: P1-39c, the flow-fallback measurement over the four candidate walks (``ccg.engine.flow.CANDIDATES``); it is not built, and reads ``Walk.items()`` rather than a document (V18 refuses a candidate table's added hops).
"""
