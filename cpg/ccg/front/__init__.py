"""The plain-language front (C-1001 P1-26): from a question to a validated canonical answer (``docs/plans/connected-code-graph-p1-answers.md`` P1-26).

    front = Front(repos=("graphmind-ai-lab",), build=lambda anchor, direction: build_answer(fresh_reader(), anchor, question=direction))
    doc = front("impact falkorchat/api.py:603", reader=reader)

A :class:`Front` is the ``front`` seam of ``clients.common.Client`` (``front(question, reader=, cache_bypass=) -> answer``). It normalizes the question
(``ccg.answer.normalize_question``), runs the deterministic parsers (:mod:`ccg.front.parsers`; a question they read fully makes no model call), and hands the draft to
the resolver (:mod:`ccg.front.resolve`). Then:

* ``resolved``: ``build(anchor, direction)`` builds the answer (the client passes a builder over a **fresh** reader, so ``recipe.queries`` lists the walk's reads only,
  as for a structured request: AC-16), and a named file that failed its lookup is added as ``interpretation.lookups``;
* ``ambiguous`` / ``unresolved``: :func:`asking_document`, which ran no query (``-answer-spec.md`` §3);
* not built (a diff, a stack trace, an unbuilt use case, a scope, two anchors, or free prose with no interpreter): :class:`NotBuiltAsk`, which the clients report as
  their typed ``not-built`` error.

The interpreter, its cache and the summary are dispatch 2; until then free prose is not built. A question the parsers resolve carries no
``provenance`` (no model read it, so there is no fingerprint to record). Nothing here imports the broker, a network module or a store driver.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from ccg import answer
from ccg.engine import bfs, config
from ccg.engine import staleness as ST
from ccg.front.parsers import Surface, parse
from ccg.front.draft import Draft
from ccg.front.resolve import RESOLVER_VERSION, NotBuiltAsk, Outcome, refuse_unbuilt, resolve_draft

__all__ = ["Front", "NotBuiltAsk", "RESOLVER_VERSION", "asking_document"]

ANSWER_SCHEMA_VERSION = "1.2"


def asking_document(reader, outcome: Outcome) -> dict[str, Any]:
    """The canonical ``ambiguous`` or ``unresolved`` answer: no query ran, so ``items``, ``groups``, ``staleness.units`` and ``recipe.queries`` are empty and
    ``depthBound`` is ``{"unexplored": 0}``; ``stamp`` is the snapshot's; ``recipe.algorithm`` names the walk that would have run."""
    if outcome.status == "ambiguous":
        interp: dict[str, Any] = {"status": "ambiguous", "options": [dict(o) for o in outcome.options], "ambiguityKind": outcome.ambiguity_kind}
    elif outcome.status == "unresolved":
        interp = {"status": "unresolved", "lookups": [dict(x) for x in outcome.lookups]}
    else:
        raise ValueError("a resolved outcome is built by the engine, not here")
    doc = {
        "answerSchemaVersion": ANSWER_SCHEMA_VERSION,
        "interpretation": interp,
        "recipe": {"algorithm": {"id": "bfs-v1", "version": bfs.algorithm_version()}, "queries": [],
                   "guards": {"visitedSeedsAnchor": True, "levelGuard": config.LEVEL_GUARD}},
        "stamp": ST.stamp(reader),
        "staleness": {"units": []},
        "items": [],
        "groups": [],
        "depthBound": {"unexplored": 0},
    }
    answer.validate(doc)
    return answer.canonicalize(doc)


class Front:
    """The plain-language front of one system. ``repos``: the system's repoIds (a named file is looked up in each); ``build(anchor, direction)``: the engine's builder
    over a fresh reader."""

    def __init__(self, repos: Sequence[str], build: Callable[[Any, str], dict]):
        self.repos = tuple(repos)
        self._build = build

    def precheck(self, question: str) -> None:
        """Raise :class:`NotBuiltAsk` for a question that cannot be answered in this build, before any read (the client calls it before opening a reader)."""
        normalized = answer.normalize_question(question)
        refuse_unbuilt(self._draft(normalized), normalized)

    def _draft(self, normalized: str) -> Draft:
        parsed = parse(normalized)
        if isinstance(parsed, Surface):
            raise NotBuiltAsk(f"a pasted {parsed.kind} is the {parsed.use_case} use case, not built in this phase",
                              detail={"request": {"useCase": parsed.use_case}, "surface": parsed.kind})
        if parsed is None:
            raise NotBuiltAsk("this question needs the interpreter, which is not built yet; name the anchor (path:line, path::symbol, a route or A.b), "
                              "optionally with 'impact' or 'depends-on', or send a structured request", detail={"surface": "plain-language-question"})
        return parsed

    def __call__(self, question: str, *, reader, cache_bypass: bool = False) -> dict[str, Any]:
        normalized = answer.normalize_question(question)
        draft = self._draft(normalized)
        outcome = resolve_draft(reader, draft, normalized, self.repos)
        if outcome.status != "resolved":
            return asking_document(reader, outcome)
        doc = self._build(outcome.anchor, outcome.direction)
        if outcome.lookups:
            doc = {**doc, "interpretation": {**doc["interpretation"], "lookups": [dict(x) for x in outcome.lookups]}}
        answer.validate(doc)
        return answer.canonicalize(doc)
