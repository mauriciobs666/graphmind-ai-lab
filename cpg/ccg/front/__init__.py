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

Free prose goes to the interpreter (:mod:`ccg.front.interpret`: one grammar-constrained model call through the broker, its cache in :mod:`ccg.front.cache`), whose draft the
same resolver reads; an interpreted answer carries ``interpretation.provenance`` (``cacheHit``, and ``cacheBypass: true`` when ``cache_bypass`` skipped the cache), a question the
parsers resolve carries none (no model read it, so there is no fingerprint to record) and opens no broker run. A reply that does not validate against the grammar is never
repaired by a second call: it is an ``ambiguous`` answer of kind ``no-anchor`` with no options. :meth:`Front.answer` adds the optional aggregate-only summary
(:mod:`ccg.front.summary`) on the same question run. This module imports neither the broker, a network module nor a store driver; the interpreter is injected.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
import threading
from dataclasses import dataclass
from typing import Any

from ccg import answer
from ccg.engine import bfs, config
from ccg.engine import staleness as ST
from ccg.front.parsers import Surface, parse
from ccg.front.draft import Draft
from ccg.front.errors import ModelError
from ccg.front.resolve import RESOLVER_VERSION, NotBuiltAsk, Outcome, refuse_unbuilt, resolve_draft

__all__ = ["Answered", "Front", "ModelError", "NotBuiltAsk", "RESOLVER_VERSION", "asking_document"]

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


@dataclass(frozen=True)
class Answered:
    """``doc``: the canonical answer; ``summary``: ``ccg.front.summary.Summary`` when one was asked for."""
    doc: dict[str, Any]
    summary: Any = None


class Front:
    """The plain-language front of one system. ``repos``: the system's repoIds (a named file is looked up in each); ``build(anchor, direction)``: the engine's builder
    over a fresh reader. ``interpreter`` (``ccg.front.interpret.Interpreter``) and ``run_factory`` (``() -> ModelRun``, one fresh run per question) enable free prose, or
    ``model_loader() -> (interpreter, run_factory)`` does, called once at the first free-prose question (so a client that asks only parseable questions never loads the broker
    client); without any of them free prose is not built."""

    def __init__(self, repos: Sequence[str], build: Callable[[Any, str], dict], *, interpreter: Any = None, run_factory: Callable[[], Any] | None = None,
                 model_loader: Callable[[], tuple[Any, Callable[[], Any]]] | None = None):
        if (interpreter is None) != (run_factory is None):
            raise ValueError("an interpreter needs a run_factory, and a run_factory an interpreter")
        self.repos = tuple(repos)
        self._build = build
        self._interpreter = interpreter
        self._run_factory = run_factory
        self._model_loader = model_loader
        self._load_lock = threading.Lock()

    def _wired(self) -> bool:
        with self._load_lock:                                   # one load, whatever the number of concurrent first questions
            if self._interpreter is None and self._model_loader is not None:
                self._interpreter, self._run_factory = self._model_loader()
        return self._interpreter is not None

    def precheck(self, question: str) -> None:
        """Raise :class:`NotBuiltAsk` for a question that cannot be answered in this build, before any read (the client calls it before opening a reader); free prose
        passes when an interpreter is wired (the interpreter's draft is checked after it runs)."""
        normalized = answer.normalize_question(question)
        parsed = self._parsed(normalized)
        if parsed is not None:
            refuse_unbuilt(parsed, normalized)

    def _parsed(self, normalized: str) -> Draft | None:
        parsed = parse(normalized)
        if isinstance(parsed, Surface):
            raise NotBuiltAsk(f"a pasted {parsed.kind} is the {parsed.use_case} use case, not built in this phase",
                              detail={"request": {"useCase": parsed.use_case}, "surface": parsed.kind})
        if parsed is None and self._interpreter is None and self._model_loader is None:
            raise NotBuiltAsk("this question needs the interpreter, which is not wired here; name the anchor (path:line, path::symbol, a route or A.b), "
                              "optionally with 'impact' or 'depends-on', or send a structured request", detail={"surface": "plain-language-question"})
        return parsed

    def __call__(self, question: str, *, reader, cache_bypass: bool = False) -> dict[str, Any]:
        return self.answer(question, reader=reader, cache_bypass=cache_bypass).doc

    def answer(self, question: str, *, reader, cache_bypass: bool = False, summary: bool = False) -> Answered:
        normalized = answer.normalize_question(question)
        parsed = self._parsed(normalized)
        run = self._run_factory() if (parsed is None or summary) and self._wired() else None
        if summary and run is None:
            raise ModelError("a summary needs a model run, and none is wired here")
        try:
            provenance = None
            if parsed is None:
                result = self._interpreter.interpret(normalized, ST.stamp(reader)["snapshotId"], run, bypass=cache_bypass)
                provenance = result.provenance
                if result.draft is None:                            # an unusable reply is never repaired: the question names nothing
                    doc = self._finish(asking_document(reader, Outcome("ambiguous", options=(), ambiguity_kind="no-anchor")), provenance)
                    return Answered(doc, self._summarize(doc, run) if summary else None)
            draft = parsed if parsed is not None else result.draft
            outcome = resolve_draft(reader, draft, normalized, self.repos)
            if outcome.status != "resolved":
                doc = asking_document(reader, outcome)
            else:
                doc = self._build(outcome.anchor, outcome.direction)
                if outcome.lookups:
                    doc = {**doc, "interpretation": {**doc["interpretation"], "lookups": [dict(x) for x in outcome.lookups]}}
            doc = self._finish(doc, provenance)
            return Answered(doc, self._summarize(doc, run) if summary else None)
        finally:
            if run is not None:
                run.close()

    @staticmethod
    def _finish(doc: dict[str, Any], provenance: dict[str, Any] | None) -> dict[str, Any]:
        if provenance is not None:
            doc = {**doc, "interpretation": {**doc["interpretation"], "provenance": dict(provenance)}}
        answer.validate(doc)
        return answer.canonicalize(doc)

    @staticmethod
    def _summarize(doc: dict[str, Any], run) -> Any:
        from ccg.front.summary import summarize            # lazy: the module is the summary's, loaded only when one is asked for
        return summarize(doc, run)
