"""Parse a SCIP index to ``(file, line, col) -> targets`` (C-1001 P1-15).

Plan: ``docs/plans/connected-code-graph-p1-extraction.md`` § P1-15; method note ``docs/plans/connected-code-graph-p1-ml.md``
§2.2. The ``.scip`` file is **hostile input**: the process that wrote it ran inside the indexer container, which may
have executed staged code. So this module

- refuses a file over ``max_bytes`` *before* reading it, allocates one buffer of ``st_size + 1`` bytes and fills it with
  ``readv`` (a file that grows after the ``fstat`` is caught by the extra byte), and refuses anything that is not a
  regular file (a FIFO would hang a reader; a symlink is not followed);
- decodes the protobuf wire format itself (stdlib only; no generated code, no third-party parser) with every
  length checked against the bytes actually present, so a field that *claims* gigabytes costs nothing. Documents,
  occurrences and symbol informations are capped and **counted while scanning, before anything is stored** (a document
  is scanned twice: the first pass only counts), so a file of empty sub-messages raises ``ScipError`` at the cap and
  never ``MemoryError``; an over-long symbol, a bad range or a position's 65th distinct target is skipped and counted;
  de-duplication is on interned symbol ids (linear);
- reads each document's ``relative_path`` only as a **dictionary key**. A path that is absolute, has a ``..`` or
  ``.`` component, an empty component, a control character, a backslash or a colon, or is over-long, drops the whole document
  (``rejected_documents``). Nothing here opens, creates or names a file from the index's content: the module has
  no write-capable call at all (``tests/test_scip_parse.py`` checks that structurally).

``parse_scip`` returns every occurrence that is neither a definition nor an import keyed by ``(relative file, 1-based line, 0-based column)``,
the convention of the call-site positions of ``assemble_bundle.locate_callees``. Columns are SCIP's, taken as
given: on BMP text both real tools (scip-typescript 0.6.0, scip-python 0.6.8) count code points and declare no
``position_encoding`` (``ScipIndex.position_encodings`` is empty); with a character above U+FFFF earlier on the line
they are unverified, so the merge leaves such a site alone (``merge_scip``'s ``astral_skipped``).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

DEFAULT_MAX_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_DOCUMENTS = 100_000
DEFAULT_MAX_OCCURRENCES = 1_000_000
DEFAULT_MAX_SYMBOL_INFOS = 1_000_000
MAX_TARGETS_PER_POSITION = 64
MAX_SYMBOL_BYTES = 2048
MAX_PATH_BYTES = 1024
MAX_STRING_BYTES = 4096                       # tool name/version
_INT32 = 2 ** 31

ROLE_DEFINITION = 0x1
ROLE_IMPORT = 0x2


class ScipError(ValueError):
    """The index is unusable: over a cap, not a regular file, or not well-formed SCIP/protobuf."""


@dataclass(frozen=True)
class ParsedSymbol:
    chain: tuple[str, ...] | None        # type/term/method names, namespaces dropped; None: no usable chain
    callable: bool                       # the last descriptor is a method
    is_type: bool                        # the last descriptor is a type (a class: instantiated, not called)
    local: bool


@dataclass(frozen=True)
class ScipTarget:
    symbol: str
    file: str | None                     # the document that defines it, when this index does; None: external or local
    chain: tuple[str, ...] | None
    callable: bool
    is_type: bool
    local: bool


@dataclass
class ScipIndex:
    tool: tuple[str, str] = ("", "")
    refs: dict[tuple[str, int, int], tuple[ScipTarget, ...]] = field(default_factory=dict)
    documents: int = 0
    rejected_documents: int = 0
    rejected_occurrences: int = 0
    position_encodings: frozenset[int] = frozenset()


# --- symbol strings ----------------------------------------------------------------------------------------------

_SIMPLE = re.compile(r"[A-Za-z0-9_+\-$]+")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_NO_CHAIN = ParsedSymbol(None, False, False, False)


def _header_token(s: str, i: int) -> tuple[str, int] | None:
    out = []
    n = len(s)
    while i < n:
        c = s[i]
        if c == " ":
            if i + 1 < n and s[i + 1] == " ":          # an escaped space
                out.append(" ")
                i += 2
                continue
            return "".join(out), i + 1
        out.append(c)
        i += 1
    return None


def parse_symbol(s: str) -> ParsedSymbol:
    """The descriptor chain of a SCIP symbol string (scip.proto, "Symbol"). Anything unparsable has no chain."""
    if s.startswith("local "):
        return ParsedSymbol(None, False, False, True)
    i = 0
    for _ in range(4):                                   # scheme, manager, package name, version
        tok = _header_token(s, i)
        if tok is None:
            return _NO_CHAIN
        i = tok[1]
    names: list[str] = []
    last = ""
    n = len(s)
    while i < n:
        c = s[i]
        if c == "`":
            j, buf = i + 1, []
            while True:
                if j >= n:
                    return _NO_CHAIN
                if s[j] == "`":
                    if j + 1 < n and s[j + 1] == "`":
                        buf.append("`")
                        j += 2
                        continue
                    break
                buf.append(s[j])
                j += 1
            name, i = "".join(buf), j + 1
        else:
            m = _SIMPLE.match(s, i)
            if m is None:                                # `(param)`, `[T]` or junk: no chain
                return _NO_CHAIN
            name, i = m.group(), m.end()
        if i >= n:
            return _NO_CHAIN
        suf = s[i]
        i += 1
        if suf == "/":
            last = "ns"
        elif suf == "#":
            names.append(name)
            last = "type"
        elif suf == ".":
            names.append(name)
            last = "term"
        elif suf == ":":
            last = "meta"
        elif suf == "!":
            last = "macro"
        elif suf == "(":
            j = s.find(")", i)
            if j < 0 or j + 1 >= n or s[j + 1] != ".":
                return _NO_CHAIN
            names.append(name)
            last = "method"
            i = j + 2
        else:
            return _NO_CHAIN
    if not names or last in ("ns", "meta", "macro"):
        return _NO_CHAIN
    return ParsedSymbol(tuple(names), last == "method", last == "type", False)


# --- the wire format ---------------------------------------------------------------------------------------------


def _varint(buf: memoryview, pos: int, end: int) -> tuple[int, int]:
    val = shift = 0
    for k in range(10):
        if pos >= end:
            raise ScipError("truncated varint")
        b = buf[pos]
        pos += 1
        val |= (b & 0x7F) << shift
        if not b & 0x80:
            return val, pos
        shift += 7
    raise ScipError("varint longer than 10 bytes")


def _fields(buf: memoryview, pos: int, end: int):
    """Yield ``(field number, wire type, value)``: an int for a varint, ``(start, end)`` for a length-delimited field."""
    while pos < end:
        tag, pos = _varint(buf, pos, end)
        num, wt = tag >> 3, tag & 7
        if num == 0:
            raise ScipError("field number 0")
        if wt == 0:
            v, pos = _varint(buf, pos, end)
            yield num, wt, v
        elif wt == 2:
            ln, pos = _varint(buf, pos, end)
            if ln > end - pos:
                raise ScipError(f"field length {ln} runs past the end of its message")
            yield num, wt, (pos, pos + ln)
            pos += ln
        elif wt == 1 or wt == 5:
            w = 8 if wt == 1 else 4
            if end - pos < w:
                raise ScipError("truncated fixed-width field")
            pos += w
            yield num, wt, None
        else:
            raise ScipError(f"unsupported wire type {wt}")


def _text(buf: memoryview, span: tuple[int, int], limit: int) -> str | None:
    if span[1] - span[0] > limit:
        return None
    try:
        return bytes(buf[span[0]:span[1]]).decode("utf-8")
    except UnicodeDecodeError:
        return None


def _path_ok(p: str | None) -> bool:
    if not p or len(p.encode("utf-8", "replace")) > MAX_PATH_BYTES or p.startswith("/"):
        return False
    if _CONTROL.search(p) or "\\" in p or ":" in p:
        return False
    return all(part not in ("", ".", "..") for part in p.split("/"))


def _range(buf: memoryview, spans: list, singles: list[int]) -> list[int]:
    vals = list(singles)
    for s, e in spans:
        pos = s
        while pos < e:
            v, pos = _varint(buf, pos, e)
            vals.append(v)
            if len(vals) > 4:
                break
    return vals


def parse_bytes(raw: bytes | bytearray | memoryview, *, max_documents: int = DEFAULT_MAX_DOCUMENTS,
                max_occurrences: int = DEFAULT_MAX_OCCURRENCES,
                max_symbol_infos: int = DEFAULT_MAX_SYMBOL_INFOS) -> ScipIndex:
    """Parse one index. Nothing is materialised per sub-message before its cap is checked: a document is scanned twice
    (the first pass only counts, so a file of empty occurrences raises ``ScipError`` at the cap, never ``MemoryError``)."""
    buf = memoryview(raw)
    idx = ScipIndex()
    syms: dict[str, int] = {}
    sym_list: list[str] = []
    defs: dict[int, str] = {}
    refs: dict[tuple[str, int, int], tuple] = {}              # position -> interned symbol ids, at most MAX_TARGETS_PER_POSITION
    occ_count = sym_count = 0
    encodings: set[int] = set()

    def sid(sym: str) -> int:
        k = syms.get(sym)
        if k is None:
            k = syms[sym] = len(sym_list)
            sym_list.append(sym)
        return k

    for num, wt, v in _fields(buf, 0, len(buf)):
        if num == 1 and wt == 2:                          # metadata
            for n2, w2, v2 in _fields(buf, v[0], v[1]):
                if n2 == 2 and w2 == 2:                   # tool_info
                    name = ver = ""
                    for n3, w3, v3 in _fields(buf, v2[0], v2[1]):
                        if w3 == 2 and n3 in (1, 2):
                            t = _text(buf, v3, MAX_STRING_BYTES) or ""
                            name, ver = (t, ver) if n3 == 1 else (name, t)
                    idx.tool = (name, ver)
        elif num == 2 and wt == 2:                        # a document
            idx.documents += 1
            if idx.documents > max_documents:
                raise ScipError(f"more than {max_documents} documents (the document cap)")
            path: str | None = None
            for n2, w2, v2 in _fields(buf, v[0], v[1]):   # pass 1: the path, and the counts against the caps
                if n2 == 1 and w2 == 2:
                    path = _text(buf, v2, MAX_PATH_BYTES + 1)
                elif n2 == 2 and w2 == 2:
                    occ_count += 1
                    if occ_count > max_occurrences:
                        raise ScipError(f"more than {max_occurrences} occurrences (the occurrence cap)")
                elif n2 == 3 and w2 == 2:
                    sym_count += 1
                    if sym_count > max_symbol_infos:
                        raise ScipError(f"more than {max_symbol_infos} symbol informations (the symbol cap)")
                elif n2 == 6 and w2 == 0:
                    encodings.add(v2)
            if not _path_ok(path):
                idx.rejected_documents += 1
                continue
            for n2, w2, v2 in _fields(buf, v[0], v[1]):   # pass 2: the content
                if n2 == 3 and w2 == 2:                   # SymbolInformation.symbol (1): defined in this document
                    for n3, w3, v3 in _fields(buf, v2[0], v2[1]):
                        if n3 == 1 and w3 == 2:
                            t = _text(buf, v3, MAX_SYMBOL_BYTES)
                            if t:
                                defs.setdefault(sid(t), path)
                elif n2 == 2 and w2 == 2:
                    spans: list[tuple[int, int]] = []
                    singles: list[int] = []
                    nrange = 0
                    sym_span = None
                    roles = 0
                    for n3, w3, v3 in _fields(buf, v2[0], v2[1]):
                        if n3 == 1:
                            nrange += 1
                            if nrange <= 4:               # a range has at most 4 elements: more is rejected below
                                if w3 == 2:
                                    spans.append(v3)
                                else:
                                    singles.append(v3)
                        elif n3 == 2 and w3 == 2:
                            sym_span = v3
                        elif n3 == 3 and w3 == 0:
                            roles = v3
                    rng = _range(buf, spans, singles) if nrange <= 4 else []
                    sym = _text(buf, sym_span, MAX_SYMBOL_BYTES) if sym_span else None
                    if len(rng) not in (3, 4) or any(x >= _INT32 for x in rng) or not sym:
                        idx.rejected_occurrences += 1
                        continue
                    k = sid(sym)
                    if roles & ROLE_DEFINITION:
                        defs.setdefault(k, path)
                    elif not roles & ROLE_IMPORT:
                        key = (path, rng[0] + 1, rng[1])
                        cur = refs.get(key)
                        if cur is None:
                            refs[key] = (k,)
                        elif k in cur:
                            pass
                        elif len(cur) >= MAX_TARGETS_PER_POSITION:
                            idx.rejected_occurrences += 1
                        else:
                            refs[key] = cur + (k,)

    targets: dict[int, ScipTarget] = {}
    for key, ks in refs.items():                          # replace the values in place: no second dict of keys
        out = []
        for k in ks:
            t = targets.get(k)
            if t is None:
                sym = sym_list[k]
                p = parse_symbol(sym)
                t = targets[k] = ScipTarget(sym, defs.get(k) if not p.local else None, p.chain, p.callable,
                                            p.is_type, p.local)
            out.append(t)
        refs[key] = tuple(out)
    idx.refs = refs
    idx.position_encodings = frozenset(encodings)
    return idx


def parse_scip(path: str | os.PathLike, *, max_bytes: int = DEFAULT_MAX_BYTES, max_documents: int = DEFAULT_MAX_DOCUMENTS,
               max_occurrences: int = DEFAULT_MAX_OCCURRENCES, max_symbol_infos: int = DEFAULT_MAX_SYMBOL_INFOS) -> ScipIndex:
    """Read and parse the ``.scip`` at ``path`` (a regular file of at most ``max_bytes``; a symlink is refused).

    The buffer is allocated once at ``st_size + 1`` and filled by ``readv``: no chunk list, no join, and a file that
    grows after the ``fstat`` is caught by the extra byte."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    except OSError as e:
        raise ScipError(f"cannot open {os.fspath(path)!r}: {e.strerror}") from e
    try:
        st = os.fstat(fd)
        if (st.st_mode & 0o170000) != 0o100000:
            raise ScipError(f"{os.fspath(path)!r} is not a regular file")
        if st.st_size > max_bytes:
            raise ScipError(f"{os.fspath(path)!r} is {st.st_size} bytes, which exceeds the size cap {max_bytes}")
        buf = bytearray(st.st_size + 1)
        view = memoryview(buf)
        got = 0
        while got < len(buf):
            n = os.readv(fd, [view[got:]])
            if n == 0:
                break
            got += n
    finally:
        os.close(fd)
    if got > st.st_size:
        raise ScipError(f"{os.fspath(path)!r} changed while it was read (it was {st.st_size} bytes)")
    return parse_bytes(view[:got], max_documents=max_documents, max_occurrences=max_occurrences,
                       max_symbol_infos=max_symbol_infos)
