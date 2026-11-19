"""The deterministic resolver of the plain-language front (C-1001 P1-26 behaviours (3), (3b); ``docs/plans/connected-code-graph-p1-ml.md`` §4.2).

**The draft** (:class:`ccg.front.draft.Draft`, ``-p1-ml.md`` §4.2's ``InterpretDraft``) is what a question names before any id is known; this module alone turns it into ids.

The resolver turns a draft into an :class:`Outcome` over ``ccg.engine.anchors`` and the anchor reads of ``ccg.engine.reader`` (plus ``provides_in`` for a tool's server;
no other read, no model): ``resolved`` (one anchor and one direction), ``ambiguous`` (``options``, each a canonical ``Request`` with ``rank``) or ``unresolved``
(``lookups``: every file the question named failed the path lookup). It never picks between readings.

* **Fabrication guard.** A span whose text is not a verbatim substring of the question is dropped; a group left empty names no anchor.
* **Per-group join rule.** ``symbol`` + ``file``: the symbol's qualified name matches and the file matches its path as a **path suffix** (``decisionmodels/runner.py``
  matches ``decision-models/decisionmodels/runner.py``); ``file`` + ``line`` (or ``path:line`` in one span): the element enclosing the line (``resolve_location``);
  ``symbol``/``tool`` + ``server``: candidates restricted to those whose path names the server; ``route``: the ``http`` contract point by ``keyNorm``; ``tool``: the
  ``mcp_tool`` contract point; several candidates left: ``options``.
* **Directions.** ``direction: null`` (and a ``null`` reading) expands into **both** directions as options; ``readings`` add theirs.
* **Options.** Complete readings (several candidates, or a null direction) are **every** reading, ranked candidate-major (the candidate's rank key, then ``impact`` before
  ``depends-on``): never cut mid-candidate, bounded only by ``config.OPTIONS_MAX`` (whole candidates dropped past it; ``-answer-spec.md`` §3 "every reading"). Only a
  **did-you-mean** list is capped, at :data:`CAP` candidates (``-p1-ml.md`` §4.3), ordered by edit distance to the asked name's last dotted segment, then qname.
* **Partial paths.** A named path is read exactly in every repo first (an exact repo-relative path wins, a skipped exact hit is "not mapped"); only when no repo has it at all
  are the two suffix reads (``file_by_suffix``, ``skipped_by_suffix``) run. Several matching files (or repos) are never picked between: the answer is ``ambiguous`` over the
  anchors they yield (``-p1-graph.md`` §5); skipped-only matches are "not mapped" lookups; a skipped match beside one file match rides as a lookup.
* **A group that found nothing** (an unknown symbol, route or tool, a line in a mapped file with no element; not a group of only ``other``/``server`` spans or a plain
  file, which name nothing to find) is not dropped when another group resolves: the answer is ``ambiguous``
  (``unknown-name`` / ``no-anchor``) over the resolved readings plus the did-you-mean list, because the answer schema has no field for an unfound name (``lookups`` are
  paths, ``notices`` are the flow notice; an ``ambiguous`` answer carries no ``lookups``). A group that failed only its path lookup still rides as ``lookups`` on the resolved answer.
* **Not built** (:class:`NotBuiltAsk`): a scope, an unbuilt use case as the draft's own or in its ``readings`` (the request echoed), or two groups that each resolve (one anchor per ask in P1).

The answer's ``unresolved`` status needs a failed path (``lookups`` is required, ``-answer-spec.md`` §3); a group that names no usable file and no candidate is therefore
``ambiguous`` with ``no-anchor`` (or ``unknown-name`` for a symbol, contract or tool not found), with ``options`` ``[]`` or a did-you-mean list.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ccg.engine import anchors as A
from ccg.engine import config
from ccg.engine.reader import GraphDataError
from ccg.front.draft import (BUILT_USE_CASE, DIRECTIONS, KIND_HINTS, MAX_GROUPS, MAX_SPANS, UNBUILT_USE_CASES, Draft, Group, Reading, Span)  # noqa: F401  (re-exported)

class NotBuiltAsk(Exception):
    """The question is understood but asks for something P1 does not build (an unbuilt use case, a scope, a diff, a stack trace, several anchors), or needs the
    interpreter, which this build lacks. ``detail`` is JSON-able and echoes what was understood; clients report it as their typed ``not-built`` error."""

    def __init__(self, message: str, *, detail: Mapping[str, Any] | None = None):
        super().__init__(message)
        self.detail = dict(detail or {})


RESOLVER_VERSION = "1.0"
#: the pre-registered cap on a did-you-mean list (``-p1-ml.md`` §4.3, EP §10.4 ``capFloor``); complete readings are not capped by it
CAP = 5
#: the contract kind a span's ``kindHint`` names
CONTRACT_KINDS = {"route": "http", "tool": "mcp_tool"}
#: words of a server span that name no server ("the falkor-chat MCP server")
SERVER_NOISE = frozenset({"mcp", "server", "servidor", "the", "o", "do", "da", "de"})


@dataclass(frozen=True)
class Outcome:
    status: str                                  # "resolved" | "ambiguous" | "unresolved"
    anchor: Any = None                           # the engine anchor (``ccg.engine.anchors``) iff resolved
    direction: str | None = None                 # iff resolved
    options: tuple[dict, ...] = ()               # canonical ``Request`` objects with ``rank`` iff ambiguous
    ambiguity_kind: str | None = None            # iff ambiguous
    lookups: tuple[dict, ...] = ()               # failed path lookups (unresolved; or resolved with a failed named path)


@dataclass(frozen=True)
class _Found:
    """One group's result: ``candidates`` (ranked engine anchors), whether they are only a did-you-mean list, and the failed path lookups."""
    candidates: tuple = ()
    suggestion: bool = False
    lookups: tuple[dict, ...] = ()
    several_files: bool = False                  # a partial path matched several files: never a pick, whatever the candidates


# ---------------------------------------------------------------------------------------------
# Small helpers.
# ---------------------------------------------------------------------------------------------

def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance (insert, delete, substitute; each 1)."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _symbol_part(qname: str) -> str:
    return qname.rpartition("::")[2] if "::" in qname else qname


def _names_symbol(qname: str, asked: str) -> bool:
    """Whether ``qname`` names the asked symbol: its symbol part is ``asked`` or ends with ``.asked`` (``ServerConnection.call_tool`` names ``call_tool``)."""
    part = _symbol_part(qname)
    return part == asked or part.endswith("." + asked)


def path_suffix_match(path: str | None, asked: str) -> bool:
    """``asked`` names ``path``: equal, or a suffix starting at a ``/`` boundary (``a/b.py`` names ``x/a/b.py``, never ``xa/b.py``)."""
    asked = asked.removeprefix("./")
    return path is not None and bool(asked) and (path == asked or path.endswith("/" + asked))


def _server_token(text: str) -> str:
    words = [w for w in text.casefold().replace("_", "-").split() if w not in SERVER_NOISE]
    return " ".join(words)


def server_matches(path: str | None, server: str) -> bool:
    """A server span names a path when its words (``mcp``/``server`` dropped) are one of the path's segments, case-folded."""
    token = _server_token(server)
    if not token or path is None:
        return True                                  # "MCP" alone names no server: no restriction
    return token in path.casefold().split("/")


def _rank_key(anchor: Any) -> tuple:
    if isinstance(anchor, A.SymbolAnchor):
        return (anchor.qname, anchor.repo_id, anchor.symbol_id)
    if isinstance(anchor, A.ContractAnchor):
        return (anchor.kind, anchor.key_norm, anchor.cp_id)
    return (anchor.path, anchor.key_path, anchor.entry_id)


def request_anchor(anchor: Any) -> dict[str, Any]:
    """The answer's ``Anchor`` object of an engine anchor (``-answer-spec.md`` §3), as ``ccg.engine.answer_build`` writes it."""
    if isinstance(anchor, A.SymbolAnchor):
        return {"type": "symbol", "symbolId": anchor.symbol_id, "qname": anchor.qname, "repoId": anchor.repo_id, "path": anchor.path}
    if isinstance(anchor, A.ContractAnchor):
        return {"type": "contract", "cpId": anchor.cp_id, "kind": anchor.kind, "keyNorm": anchor.key_norm}
    return {"type": "config", "entryId": anchor.entry_id, "repoId": anchor.repo_id, "path": anchor.path, "keyPath": anchor.key_path}


def _lookup(path: str, repo: str, res: A.Resolution) -> dict[str, Any]:
    if res.state == "not mapped":
        return {"path": path, "repoId": repo, "state": "skipped-file", "reason": res.reason, "unitId": res.unit_id}
    return {"path": path, "repoId": repo, "state": "not-found"}


# ---------------------------------------------------------------------------------------------
# Reads.
# ---------------------------------------------------------------------------------------------

def _by_name(reader, name: str) -> list[A.SymbolAnchor]:
    """Every stored symbol whose ``nameNorm`` is ``name``'s, uncapped (a path or server filter must see them all). Symbols stored without a path cannot be a
    request anchor and are left out."""
    norm = A.name_norm(name)
    if not norm:
        return []
    out = []
    for row in reader.read("symbol_by_name", {"name": norm}):
        if len(row) != 6 or not all(isinstance(v, str) for v in (row[0], row[1], row[3])) or not (row[2] is None or isinstance(row[2], str)) \
                or not (row[4] is None or isinstance(row[4], str)) or isinstance(row[5], bool) or not isinstance(row[5], int):
            raise GraphDataError("query 'symbol_by_name' returned a malformed row")
        sid, repo, unit, qname, path, start = row
        if path is not None:
            out.append(A.SymbolAnchor(sid, qname, repo, path, unit, start, None))
    return out


@dataclass(frozen=True)
class _Located:
    """A named path found: the ``(repo, stored path, resolution)`` of each mapped file, the lookups to report, and whether a partial path matched several files."""
    mapped: tuple = ()
    lookups: tuple[dict, ...] = ()
    several: bool = False


def _str_row(name: str, row: tuple, n: int) -> tuple:
    if len(row) != n or not all(isinstance(v, str) for v in row):
        raise GraphDataError(f"query {name!r} returned a malformed row")
    return row


def _by_suffix(reader, repos: Sequence[str], path: str, line: int | None) -> _Located:
    """The fallback for a partial path: ``file_by_suffix`` and ``skipped_by_suffix`` over the repos (``-p1-graph.md`` §5), at a ``/`` boundary. A file match is resolved
    by its stored path; skipped matches are "not mapped" lookups (the whole answer when no file matched). Over :data:`config.OPTIONS_MAX` file matches the list is truncated."""
    suffix = "/" + path
    if not repos or len(suffix.encode("utf-8", "replace")) > A.MAX_TEXT_BYTES:
        return _Located()
    params = {"repos": list(repos), "sep": "/", "suffix": suffix}
    files = [_str_row("file_by_suffix", r, 4) for r in reader.read("file_by_suffix", params)]
    skipped = [_str_row("skipped_by_suffix", r, 4) for r in reader.read("skipped_by_suffix", params)]
    lookups = tuple({"path": p, "repoId": repo, "state": "skipped-file", "reason": reason, "unitId": unit} for repo, p, unit, reason in skipped[:config.OPTIONS_MAX])
    mapped = []
    for repo, hit, _file_id, _unit in files[:config.OPTIONS_MAX]:
        res = A.resolve_location(reader, repo, hit, line)
        if res.state == "mapped":
            mapped.append((repo, hit, res))
    if not mapped:
        return _Located(lookups=lookups)
    return _Located(tuple(mapped), lookups, several=len(files) > 1)


def _locate(reader, repos: Sequence[str], path: str, line: int | None) -> _Located:
    """``resolve_location`` of ``path`` in every known repo: where it is mapped, else one failed lookup per repo. An exact repo-relative path wins; only when no repo holds
    the path at all (neither as a file nor as a skipped file) is it read as a partial path (:func:`_by_suffix`)."""
    if not repos:
        raise NotBuiltAsk("a question naming a file needs the system's repositories, which this client does not know", detail={"surface": "file-without-repos"})
    mapped, failed, known = [], [], []
    for repo in repos:
        try:
            res = A.resolve_location(reader, repo, path, line)
        except A.UnknownRepo:
            continue
        except A.AnchorError:                            # a path no store can hold (too long, NUL): nothing to look up
            return _Located()
        known.append(repo)
        if res.state == "mapped":
            mapped.append((repo, path, res))
        else:
            failed.append((repo, res))
    if mapped:
        return _Located(tuple(mapped))
    if all(res.state == "not found" for _, res in failed):
        found = _by_suffix(reader, known, path, line)
        if found.mapped or found.lookups:
            return found
    return _Located(lookups=tuple(_lookup(path, repo, res) for repo, res in failed))


def _symbols_of_file(reader, repo: str, path: str) -> list[A.SymbolAnchor]:
    out = []
    for row in reader.read("symbols_of_file", {"repo": repo, "path": path}):
        if len(row) != 6 or not isinstance(row[0], str) or not isinstance(row[1], str) or isinstance(row[2], bool) or not isinstance(row[2], int):
            raise GraphDataError("query 'symbols_of_file' returned a malformed row")
        out.append(A.SymbolAnchor(row[0], row[1], repo, path, None, row[2], row[3] if isinstance(row[3], int) else None))
    return out


# ---------------------------------------------------------------------------------------------
# One group.
# ---------------------------------------------------------------------------------------------

def _first(group: Sequence[Span], kind: str) -> str | None:
    return next((s.text for s in group if s.kind_hint == kind), None)


def _line_number(text: str) -> int | None:
    digits = text.rpartition(":")[2] if ":" in text else "".join(c for c in text if c.isdigit())
    return int(digits) if digits.isdigit() and 0 < len(digits) <= 9 and int(digits) > 0 else None


def _suggest(cands: Iterable[Any], asked: str) -> _Found:
    """A did-you-mean list, capped at :data:`CAP`: by edit distance, then rank key. A symbol is compared by its last dotted segment (``Outer.helper`` is as near to
    ``helper`` as ``helper`` is); ``asked`` is the asked name's last segment for a symbol, the whole key for a contract."""
    def distance(a: Any) -> int:
        qname = getattr(a, "qname", None)
        return edit_distance(asked, _symbol_part(qname).rpartition(".")[2] if qname else a.key_norm)
    return _Found(tuple(sorted(cands, key=lambda a: (distance(a), _rank_key(a)))[:CAP]), suggestion=True)


def _contract(reader, key: str, kind: str, server: str | None) -> _Found:
    try:
        res = A.resolve_contract(reader, key, kind)
    except A.AnchorError:
        return _Found(suggestion=True)
    cands = list(res.anchors or res.options)
    if server and _server_token(server) and cands:      # restricted to the contract points a symbol under the named server provides
        named = set()
        for row in reader.read("provides_in", {"ids": sorted(a.cp_id for a in cands)}):
            if len(row) != 9 or not isinstance(row[0], str) or not (row[5] is None or isinstance(row[5], str)):
                raise GraphDataError("query 'provides_in' returned a malformed row")
            if row[5] is not None and server_matches(row[5], server):
                named.add(row[0])
        kept = [a for a in cands if a.cp_id in named]
        if not kept:
            return _suggest(cands, key)
        cands = kept
    if not cands:
        return _Found(suggestion=True)
    return _Found(tuple(sorted(cands, key=_rank_key)))


def _symbol(reader, repos: Sequence[str], asked: str, file: str | None, server: str | None) -> _Found:
    asked = asked.removesuffix("()")
    name = asked.rpartition(".")[2]
    try:
        hits = _by_name(reader, name)
    except A.AnchorError:
        hits = []
    if file:
        hits = [a for a in hits if path_suffix_match(a.path, file)]
    if server:
        hits = [a for a in hits if server_matches(a.path, server)]
    exact = [a for a in hits if _names_symbol(a.qname, asked)]
    if exact:
        return _Found(tuple(sorted(exact, key=_rank_key)))
    if file:                                             # the file's own symbols are the did-you-mean list, when the file is found as named
        path = file.removeprefix("./")
        loc = _locate(reader, repos, path, None)
        if not loc.mapped:
            return _Found(suggestion=True, lookups=loc.lookups)
        near = [sym for repo, hit, _ in loc.mapped for sym in _symbols_of_file(reader, repo, hit)]
        return _suggest(near or hits, name)
    return _suggest(hits, name)


def _location(reader, repos: Sequence[str], path: str, line: int | None) -> _Found:
    """A file and a line: the enclosing element. A file alone: a config file's entries are its candidates; any other file names no anchor."""
    path = path.removeprefix("./")
    loc = _locate(reader, repos, path, line)
    if not loc.mapped:
        return _Found(lookups=loc.lookups)
    cands: list = []
    for repo, hit, res in loc.mapped:
        if line is not None:
            cands += list(res.anchors or res.options)
            continue
        try:
            conf = A.resolve_config(reader, repo, hit)
        except A.AnchorError:
            continue
        cands += list(conf.anchors or conf.options)
    return _Found(tuple(sorted(set(cands), key=_rank_key)), lookups=loc.lookups, several_files=loc.several)


def _group(reader, repos: Sequence[str], spans: Sequence[Span]) -> _Found:
    server = _first(spans, "server")
    for hint, kind in CONTRACT_KINDS.items():
        if (key := _first(spans, hint)) is not None:
            return _contract(reader, key, kind, server)
    symbol, file, line = _first(spans, "symbol"), _first(spans, "file"), _first(spans, "line")
    if symbol is not None:
        return _symbol(reader, repos, symbol, file, server)
    if line is not None and file is None and ":" in line:
        file, line = line.rpartition(":")[0], line
    if file is not None:
        n = _line_number(line) if line is not None else None
        if line is not None and n is None:
            return _Found()
        return _location(reader, repos, file, n)
    return _Found()


# ---------------------------------------------------------------------------------------------
# The draft.
# ---------------------------------------------------------------------------------------------

def echo(draft: Draft) -> dict[str, Any]:
    """What was understood, as a not-built error carries it (behaviour (7): the request echoed): use case, direction, the span groups, the scope span."""
    out: dict[str, Any] = {"useCase": draft.use_case, "direction": draft.direction,
                           "anchors": [[{"text": s.text, "kindHint": s.kind_hint} for s in g.spans] for g in draft.anchors]}
    if draft.scope is not None:
        out["scope"] = draft.scope
    if draft.readings:
        out["readings"] = [{"useCase": r.use_case, "direction": r.direction} for r in draft.readings]
    return out


def directions(draft: Draft) -> tuple[str, ...]:
    """The built directions the draft asks for: its own and its readings', a ``None`` direction standing for both; in :data:`DIRECTIONS` order."""
    wanted: set[str] = set()
    for use_case, direction in [(draft.use_case, draft.direction), *((r.use_case, r.direction) for r in draft.readings)]:
        if use_case == BUILT_USE_CASE:
            wanted.update(DIRECTIONS if direction is None else (direction,))
    return tuple(d for d in DIRECTIONS if d in wanted)


def guarded(question: str, group: Group) -> tuple[Span, ...]:
    """The fabrication guard: the spans of ``group`` that are verbatim substrings of ``question``."""
    return tuple(s for s in group.spans if s.text in question)


def options(candidates: Sequence[Any], dirs: Sequence[str]) -> tuple[dict, ...]:
    """Every (candidate, direction) reading as a ranked ``Request``, candidate-major. Never cut mid-candidate: past ``config.OPTIONS_MAX`` readings whole candidates are
    dropped (the engine's own bound on an option list)."""
    keep = max(1, config.OPTIONS_MAX // max(1, len(dirs)))
    out = [{"useCase": BUILT_USE_CASE, "direction": d, "anchors": [request_anchor(c)]} for c in candidates[:keep] for d in dirs]
    return tuple({**o, "rank": i} for i, o in enumerate(out, 1))


def refuse_unbuilt(draft: Draft, question: str) -> tuple[str, ...]:
    """Raise :class:`NotBuiltAsk` for a scope or an unbuilt use case, as the draft's own or as one of its ``readings`` (no read needed to know; a question that also
    names an unbuilt use case is refused whole, never answered for its built half alone); else the directions asked for."""
    if draft.scope is not None and draft.scope in question:
        raise NotBuiltAsk("a repository scope is not built in this phase; ask about the whole system", detail={"request": echo(draft)})
    unbuilt = [u for u in dict.fromkeys([draft.use_case, *(r.use_case for r in draft.readings)]) if u != BUILT_USE_CASE]
    if unbuilt:                                         # never answer the built half and drop the rest: the answer schema has no place to say what was left out
        raise NotBuiltAsk(f"the use case {', '.join(repr(u) for u in unbuilt)} is not built in this phase", detail={"request": echo(draft), "built": [BUILT_USE_CASE]})
    return directions(draft)


def _names_something(spans: Sequence[Span]) -> bool:
    """Whether a group carries a span the asker could have meant as a thing to find (a symbol, route, tool, or a line): only such a group that finds nothing is an unfound
    name. A group of ``other``/``server`` spans, or a plain file (mapped, with no config entries), names nothing to find and is ignored beside a resolving group."""
    return any(s.kind_hint in ("symbol", "route", "tool", "line") for s in spans)


def resolve_draft(reader, draft: Draft, question: str, repos: Sequence[str]) -> Outcome:
    """The :class:`Outcome` of ``draft`` (read from ``question``, the normalized question its spans must be substrings of) over ``reader``; ``repos`` are the
    system's repoIds (a named file is looked up in each). Raises :class:`NotBuiltAsk` for a scope, an unbuilt use case or two anchors.

    ``lookups`` ride a resolved or an unresolved outcome only (the answer schema); an ambiguous outcome carries none."""
    dirs = refuse_unbuilt(draft, question)
    groups = [spans for spans in (guarded(question, g) for g in draft.anchors) if spans]
    found = [_group(reader, repos, spans) for spans in groups]
    lookups = tuple(sorted({(x["repoId"], x["path"]): x for f in found for x in f.lookups}.values(), key=lambda x: (x["repoId"], x["path"])))
    real = [f for f in found if f.candidates and not f.suggestion]
    if len({f.candidates for f in real}) > 1:
        raise NotBuiltAsk("an ask names one anchor in this phase; ask about each separately", detail={"request": echo(draft)})
    suggestions = [c for f in found if f.suggestion for c in f.candidates]
    unfound = [f for spans, f in zip(groups, found) if _names_something(spans) and not f.lookups and (f.suggestion or not f.candidates)]
    if real:
        cands = real[0].candidates
        several = any(f.several_files for f in real)
        if len(cands) == 1 and len(dirs) == 1 and not several and not unfound:
            return Outcome("resolved", anchor=cands[0], direction=dirs[0], lookups=lookups)
        kind = "no-anchor" if several and len(cands) <= 1 else "several-symbols" if len(cands) > 1 else "two-use-cases"
        if unfound:                                     # a name the asker gave found nothing: not silently dropped (see the module docstring)
            kind = "unknown-name" if any(f.suggestion for f in unfound) else "no-anchor"
            cands = tuple(dict.fromkeys([*cands, *suggestions]))
        return Outcome("ambiguous", options=options(cands, dirs), ambiguity_kind=kind)
    if lookups and not suggestions:
        return Outcome("unresolved", lookups=lookups)
    kind = "unknown-name" if any(f.suggestion for f in found) else "no-anchor"
    return Outcome("ambiguous", options=options(list(dict.fromkeys(suggestions))[:CAP], dirs), ambiguity_kind=kind)
