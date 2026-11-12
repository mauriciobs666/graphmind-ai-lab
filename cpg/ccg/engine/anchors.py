"""Anchor resolution (C-1001 P1-21): from what an asker named to the ids the walk starts at.

Contract: ``docs/plans/connected-code-graph-p1-answers.md`` P1-21 behaviours (2)-(6); ``docs/plans/connected-code-graph-p0-graph.md`` §1.3 (the three
repo-scoped location reads, in order) and §1.4 (File+line: the symbol with the narrowest containing interval, span or owned region; at equal width a span
match beats an owned-region match, then the larger ``depth``); ``-p1-graph.md`` §5 (the texts, in ``ccg.engine.reader``).

Every resolver returns a :class:`Resolution`: ``resolved`` (exactly one anchor), ``ambiguous`` (``options``, never a pick: a tie the rule cannot break, several
qnames, several contract points) or ``unresolved`` (with why: ``state``/``reason`` for a path, ``detail`` otherwise). Bad input and an unknown repo are errors, raised
before any read that could mislead (:class:`AnchorError`, :class:`UnknownRepo`); a row of the wrong shape is :class:`ccg.engine.reader.GraphDataError`.
Everything read (qnames, paths, key paths) is untrusted text: it is carried into the anchors verbatim, as data, and is never interpreted or placed in a text or key.

Contract points match client-side through ``ccg.linker.matchers`` (the one implementation the linker uses): a stored key is the *provider* and the asked key the
*consumer*, exactly as the linker relates them, so an anchor is "the contract point this key links to".
SEAM(P1-26): the plain-language front maps these results onto ``interpretation`` (``status``, ``options``, ``lookups``); nothing here builds an answer.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from ccg.engine import config
from ccg.engine.reader import GraphDataError, Reader
from ccg.linker.matchers import http_key, match_keys

_REPO_RE = re.compile(r"\A[a-z0-9._-]+\Z")                 # the schema's repoId pattern
_KIND_RE = re.compile(r"\A[a-z][a-z0-9_]*\Z")
MAX_TEXT_BYTES = 4096                                      # the indexed-value ceiling (``ccg.indexed``): nothing longer can be stored, so nothing longer is asked
#: the link matcher per kind, as the linker picks it (``ccg.linker.link``: ``http`` is ``path-template``, every other kind the pack's ``link`` row, default ``exact``).
#: The engine does not read packs; ``tests/test_engine_anchors.py`` compares this table with every shipped pack so the two cannot drift.
_KIND_MATCHER = {"http": "path-template"}


def contract_matcher(kind: str) -> str:
    """The matcher the linker relates a consumer's key to a provider's with, for ``kind``."""
    return _KIND_MATCHER.get(kind, "exact")


class AnchorError(ValueError):
    """The asker's input is unusable (a clean error, raised before any read)."""


class UnknownRepo(AnchorError):
    """The repo is not in the store."""


@dataclass(frozen=True)
class SymbolAnchor:
    symbol_id: str
    qname: str
    repo_id: str
    path: str | None                                       # None when the stored symbol carries no path (read as stored, never invented)
    unit_id: str | None
    line_start: int
    line_end: int | None


@dataclass(frozen=True)
class ContractAnchor:
    cp_id: str
    kind: str
    key_norm: str


@dataclass(frozen=True)
class ConfigAnchor:
    entry_id: str
    repo_id: str
    path: str
    key_path: str
    unit_id: str | None
    line: int


@dataclass(frozen=True)
class Resolution:
    status: str                                            # "resolved" | "ambiguous" | "unresolved"
    anchors: tuple = ()                                    # exactly one iff resolved
    options: tuple = ()                                    # the candidates iff ambiguous, at most config.OPTIONS_MAX
    options_total: int = 0                                 # candidates before the cap
    state: str | None = None                               # path lookups: "mapped" | "not mapped" | "not found"
    unit_id: str | None = None
    reason: str | None = None                              # the SkippedFile reason, when state is "not mapped"
    via: str | None = None                                 # how it resolved: span | owned-region | qname | name | exact | path-template | keyPath
    detail: str | None = None                              # why it did not resolve or is ambiguous


# ---------------------------------------------------------------------------------------------
# Input and row checks.
# ---------------------------------------------------------------------------------------------

def _text(value: Any, what: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise AnchorError(f"{what} is a non-empty string without NUL")
    try:
        size = len(value.encode("utf-8"))
    except UnicodeEncodeError:
        raise AnchorError(f"{what} is not valid Unicode") from None
    if size > MAX_TEXT_BYTES:
        raise AnchorError(f"{what} is at most {MAX_TEXT_BYTES} bytes")
    return value


def _repo(repo: Any) -> str:
    if not isinstance(repo, str) or not _REPO_RE.match(repo) or len(repo) > 128:
        raise AnchorError("repo is a repoId: lowercase letters, digits, '.', '_' and '-'")
    return repo


def _kind(kind: Any) -> str | None:
    if kind is None:
        return None
    if not isinstance(kind, str) or not _KIND_RE.match(kind) or len(kind) > 64:
        raise AnchorError("kind is a contract kind: lowercase letters, digits and '_'")
    return kind


def _line(line: Any) -> int:
    if isinstance(line, bool) or not isinstance(line, int) or line < 1:
        raise AnchorError("line is an integer of at least 1")
    return line


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _row(name: str, row: tuple, shape: str) -> tuple:
    """Check a row against ``shape``, one letter per column: ``s`` str, ``S`` str or None, ``i`` int, ``I`` int or None, ``?`` anything; raises :class:`GraphDataError`."""
    ok = len(row) == len(shape)
    if ok:
        for v, code in zip(row, shape, strict=True):
            ok = ok and {"s": isinstance(v, str), "S": v is None or isinstance(v, str), "i": _is_int(v), "I": v is None or _is_int(v), "?": True}[code]
    if not ok:
        raise GraphDataError(f"query {name!r} returned a malformed row")
    return row


def _ensure_repo(reader: Reader, repo: str) -> None:
    if not reader.read("repo_exists", {"repo": repo}):
        raise UnknownRepo(f"unknown repo {repo!r}")


def _capped(items: list) -> tuple[tuple, int]:
    return tuple(items[:config.OPTIONS_MAX]), len(items)


# ---------------------------------------------------------------------------------------------
# Code location: file, file + line, file + symbol.
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class _Sym:
    symbol_id: str
    qname: str
    start: int
    end: int
    regions: tuple[tuple[int, int], ...]
    depth: int


def _parse_symbol(row: tuple) -> _Sym:
    symbol_id, qname, start, end, regions, depth = _row("symbols_of_file", row, "ssii?I")
    if start > end:
        raise GraphDataError("query 'symbols_of_file' returned a symbol whose span ends before it starts")
    pairs: list[tuple[int, int]] = []
    if regions is not None:
        if not (isinstance(regions, tuple) and len(regions) % 2 == 0 and all(_is_int(v) for v in regions)):
            raise GraphDataError("query 'symbols_of_file' returned malformed ownedRegions")
        pairs = [(regions[i], regions[i + 1]) for i in range(0, len(regions), 2)]
        if any(a > b for a, b in pairs):
            raise GraphDataError("query 'symbols_of_file' returned an owned region that ends before it starts")
    return _Sym(symbol_id, qname, start, end, tuple(pairs), 0 if depth is None else depth)


def _lookup(reader: Reader, repo: str, path: str) -> tuple[str, str | None, str | None]:
    """The three repo-scoped reads, in order, stopping at the first that answers: ``(state, unitId, reason)``."""
    files = reader.read("file_by_path", {"repo": repo, "path": path})
    if files:
        if len(files) > 1:
            raise GraphDataError("query 'file_by_path' returned more than one file for one path")
        _, unit = _row("file_by_path", files[0], "sS")
        return "mapped", unit, None
    skipped = reader.read("skipped_by_path", {"repo": repo, "path": path})
    if skipped:
        unit, reason = _row("skipped_by_path", skipped[0], "SS")
        return "not mapped", unit, reason
    return "not found", None, None


def _anchor_of(sym: _Sym, repo: str, path: str, unit: str | None) -> SymbolAnchor:
    return SymbolAnchor(sym.symbol_id, sym.qname, repo, path, unit, sym.start, sym.end)


def _pick_by_line(syms: list[_Sym], line: int) -> list[tuple[_Sym, int]]:
    """The symbols tied at the best ``(width, kind, -depth)`` containing ``line`` (kind 0 a span, 1 an owned region), each with its kind."""
    scored: list[tuple[tuple[int, int, int], _Sym, int]] = []
    for sym in syms:
        spans = [(sym.end - sym.start, 0)] if sym.start <= line <= sym.end else []
        spans += [(b - a, 1) for a, b in sym.regions if a <= line <= b]
        if spans:
            width, kind = min(spans)
            scored.append(((width, kind, -sym.depth), sym, kind))
    if not scored:
        return []
    best = min(key for key, _, _ in scored)
    return sorted(((s, k) for key, s, k in scored if key == best), key=lambda t: t[0].symbol_id)


def resolve_location(reader: Reader, repo: str, path: str, line: int | None = None, symbol: str | None = None) -> Resolution:
    """Resolve a code location named by ``repo`` and ``path``, optionally with a ``line`` or a ``symbol`` qname (not both).

    The lookup is the three repo-scoped reads in order (``File`` -> mapped; else ``SkippedFile`` -> "not mapped" with its reason; else "not found": a ``.d.ts``,
    a ``.pyi`` or a vendored path is in neither set). Only a mapped file with a ``line`` or ``symbol`` reads its symbols (every symbol of the file: an owned region can lie
    outside the span).
    """
    repo, path = _repo(repo), _text(path, "path")
    if line is not None:
        line = _line(line)
    if symbol is not None:
        symbol = _text(symbol, "symbol")
    if line is not None and symbol is not None:
        raise AnchorError("name a line or a symbol, not both")
    _ensure_repo(reader, repo)
    state, unit, reason = _lookup(reader, repo, path)
    if state != "mapped":
        return Resolution("unresolved", state=state, unit_id=unit, reason=reason)
    if line is None and symbol is None:
        return Resolution("unresolved", state=state, unit_id=unit, detail="no-line-or-symbol")
    syms = [_parse_symbol(r) for r in reader.read("symbols_of_file", {"repo": repo, "path": path})]
    if symbol is not None:
        hits = sorted((s for s in syms if s.qname == symbol), key=lambda s: s.symbol_id)
        if not hits:
            return Resolution("unresolved", state=state, unit_id=unit, detail="no-such-symbol")
        anchors = [_anchor_of(s, repo, path, unit) for s in hits]
        return _decide(anchors, state, unit, via="qname")
    tied = _pick_by_line(syms, line)
    if not tied:
        return Resolution("unresolved", state=state, unit_id=unit, detail="no-symbol-at-line")
    via = "span" if tied[0][1] == 0 else "owned-region"
    return _decide([_anchor_of(s, repo, path, unit) for s, _ in tied], state, unit, via=via)


def _decide(anchors: list, state: str | None = None, unit: str | None = None, *, via: str | None = None, detail: str | None = None) -> Resolution:
    """One anchor resolves; several are options (capped), never a pick."""
    if len(anchors) == 1:
        return Resolution("resolved", anchors=(anchors[0],), state=state, unit_id=unit, via=via)
    options, total = _capped(anchors)
    return Resolution("ambiguous", options=options, options_total=total, state=state, unit_id=unit, detail=detail)


# ---------------------------------------------------------------------------------------------
# Symbol anchors by qname or by name.
# ---------------------------------------------------------------------------------------------

def name_norm(name: str) -> str:
    """``nameNorm`` as the bundle stores it: ``NFC(casefold(NFC(name))).strip()`` (``ccg.bundle``)."""
    return unicodedata.normalize("NFC", unicodedata.normalize("NFC", name).casefold()).strip()


def resolve_symbol(reader: Reader, repo: str, qname: str) -> Resolution:
    """A symbol of ``repo`` by its exact ``qname`` (the ``Symbol.qname`` index)."""
    repo, qname = _repo(repo), _text(qname, "qname")
    _ensure_repo(reader, repo)
    rows = [_row("symbol_by_qname", r, "sSsii?")[:5] for r in reader.read("symbol_by_qname", {"repo": repo, "qname": qname})]
    anchors = sorted((SymbolAnchor(sid, qname, repo, path, unit, start, end) for sid, unit, path, start, end in rows), key=lambda a: a.symbol_id)
    if not anchors:
        return Resolution("unresolved", detail="no-such-symbol")
    return _decide(anchors, via="qname")


def resolve_symbol_by_name(reader: Reader, name: str, repo: str | None = None) -> Resolution:
    """Symbols whose normalized name is ``name`` (system-wide, or in ``repo``): several hits are options."""
    name = _text(name, "name")
    if repo is not None:
        repo = _repo(repo)
    norm = name_norm(name)
    if not norm:
        raise AnchorError("name is empty after normalization")
    if repo is not None:
        _ensure_repo(reader, repo)
    rows = [_row("symbol_by_name", r, "ssSsSi") for r in reader.read("symbol_by_name", {"name": norm})]
    anchors = sorted((SymbolAnchor(sid, qname, repo_id, path, unit, start, None) for sid, repo_id, unit, qname, path, start in rows
                      if repo is None or repo_id == repo), key=lambda a: (a.symbol_id, a.repo_id))
    if not anchors:
        return Resolution("unresolved", detail="unknown-name")
    return _decide(anchors, via="name")


# ---------------------------------------------------------------------------------------------
# Contract-point anchors.
# ---------------------------------------------------------------------------------------------

def resolve_contract(reader: Reader, key: str, kind: str | None = None) -> Resolution:
    """A contract point by ``keyNorm``; with a ``kind``, a key that is not stored also matches client-side by the kind's linker matcher (``contract_matcher``: ``http`` templates, others exact).

    An ``http`` key is normalized first (``ccg.linker.matchers.http_key``, the consumer form), as the linker normalizes a consumer's. Without a ``kind`` an unstored key
    is unresolved: the key scan reads one kind.
    """
    key, kind = _text(key, "key"), _kind(kind)
    if kind == "http":
        try:
            key = http_key(key, "consumer")
        except ValueError:
            raise AnchorError("key is not a usable http key") from None
    exact = [ContractAnchor(*_row("cp_by_key", r, "sss")) for r in reader.read("cp_by_key", {"key": key})]
    exact = sorted((a for a in exact if kind is None or a.kind == kind), key=lambda a: a.cp_id)
    if exact:
        return _decide(exact, via="exact")
    if kind is None:
        return Resolution("unresolved", detail="no-contract-point")
    matcher = contract_matcher(kind)
    if matcher == "exact":                                  # an unstored key cannot link: the stored-key read above was the whole match
        return Resolution("unresolved", detail="no-contract-point")
    hits: list[tuple[ContractAnchor, str]] = []
    for cp_id, stored in (_row("cp_keys", r, "ss") for r in reader.read("cp_keys", {"kind": kind})):
        got = match_keys(kind, matcher, stored, key)
        if got is not None:
            hits.append((ContractAnchor(cp_id, kind, stored), got.matcher))
    hits.sort(key=lambda t: t[0].cp_id)
    if not hits:
        return Resolution("unresolved", detail="no-contract-point")
    return _decide([a for a, _ in hits], via=hits[0][1] if len(hits) == 1 else None)


# ---------------------------------------------------------------------------------------------
# Config anchors.
# ---------------------------------------------------------------------------------------------

def resolve_config(reader: Reader, repo: str, path: str, key_path: str | None = None) -> Resolution:
    """A ``configEntry``-origin entry of a config file by ``keyPath``. A launch entry (``launchKind``) is never an anchor; without a ``key_path`` every entry is an option.

    ``key_path`` is matched exactly against the stored (masked) key path: a segment the build masked cannot be asked for.
    """
    repo, path = _repo(repo), _text(path, "path")
    if key_path is not None:
        key_path = _text(key_path, "key_path")
    _ensure_repo(reader, repo)
    rows = [_row("config_by_path", r, "sSiSS") for r in reader.read("config_by_path", {"repo": repo, "path": path})]
    entries = sorted((ConfigAnchor(eid, repo, path, kp, unit, line) for eid, unit, line, kp, launch in rows if launch is None and kp is not None),
                     key=lambda a: (a.key_path, a.entry_id))
    if not entries:
        state, unit, reason = _lookup(reader, repo, path)
        return Resolution("unresolved", state=state, unit_id=unit, reason=reason, detail="no-entries" if state == "mapped" else None)
    if key_path is None:
        options, total = _capped(entries)
        return Resolution("ambiguous", options=options, options_total=total, detail="key-path-not-given")
    hits = [e for e in entries if e.key_path == key_path]
    if not hits:
        return Resolution("unresolved", detail="no-such-key")
    return _decide(hits, via="keyPath")
