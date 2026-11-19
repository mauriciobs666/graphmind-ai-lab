"""The deterministic parsers of the plain-language front (C-1001 P1-26 behaviour (1); ``docs/plans/connected-code-graph-p1-answers.md`` P1-26 Files).

They run first, on ``ccg.answer.normalize_question``'s output, and make no model call. :func:`parse` returns one of:

* :class:`Surface`: a pasted **diff** (code review) or **stack trace** (root cause): understood, and not built in P1 (P5);
* a complete :class:`ccg.front.draft.Draft`: every word of the question is an anchor, a scope, or one of the closed use-case words (:data:`DIRECTION_WORDS`, one literal
  token each, :data:`UNBUILT_WORDS`); with no such word the direction is ``None`` and the resolver offers both;
* ``None``: free prose, which only the interpreter can read.

Anchors recognized, each one span group whose spans are verbatim substrings of the question:

* a route, ``VERB /path`` with an upper-case HTTP verb (``kindHint`` ``route``);
* ``path::symbol`` (a ``file`` and a ``symbol`` span);
* ``path:line`` (one ``line`` span; ``:col`` after it is ignored) and ``path line N`` / ``path linha N`` (a ``file`` and a ``line`` span);
* a qualified name ``A.b`` (dotted identifiers, the last segment not a file extension) or a call form ``name()`` (``symbol``);
* a file path alone (``file``).

The parser is token-based (whitespace tokens, surrounding punctuation stripped), so no pattern backtracks over the whole question.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from ccg.front.draft import BUILT_USE_CASE, Draft, Group, Span

HTTP_VERBS = ("GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS")
#: the extensions that make a dotted token a file rather than a qualified name
FILE_EXTENSIONS = frozenset({"py", "pyi", "ts", "tsx", "js", "jsx", "mjs", "cjs", "sh", "bash", "json", "yaml", "yml", "toml", "ini", "cfg", "go", "rs",
                             "java", "kt", "rb", "php", "c", "h", "cc", "cpp", "hpp", "cs", "swift", "sql", "html", "css", "scss", "vue", "svelte", "txt",
                             "env", "conf", "xml", "lock", "md", "dockerfile"})
#: the closed set of whole-remainder tokens that state a direction: the use-case names the structured request takes, as one literal token each (case-insensitive). The spaced
#: phrase ``depends on`` states none: ``depends on X`` asks what depends on X (impact) and ``X depends on`` the reverse, so the phrase is the interpreter's.
DIRECTION_WORDS = {"impact": "impact", "depends-on": "depends-on"}
#: whole tokens that look like a file or a qualified name and are neither (product names, abbreviations); never an anchor
NOT_ANCHORS = frozenset({"node.js", "vue.js", "next.js", "nuxt.js", "react.js", "express.js", "d3.js", "three.js", "e.g", "i.e", "a.k.a", "vs", "etc"})
#: the closed set of whole-remainder words naming an unbuilt use case
UNBUILT_WORDS = {"root cause": "root-cause", "rca": "root-cause", "code review": "code-review", "review": "code-review",
                 "test coverage": "test-coverage", "coverage": "test-coverage", "security": "security"}
LINE_WORDS = ("line", "linha", "l")
SCOPE_KEYS = ("scope", "repo", "repoid", "repository", "repositório")
MAX_TOKEN = 512                     # a longer token is never an anchor

_LEAD = "([{\"'`<«“‘"
_TRAIL = ")]}\"'`>»”’.,;:?!"
_OPENER = {")": "(", "]": "[", "}": "{", ">": "<", "»": "«", "”": "“", "’": "‘"}
_IDENT = r"[A-Za-z_$][\w$]*"
_QNAME_RE = re.compile(rf"{_IDENT}(?:\.{_IDENT})+(?:\(\))?")
_CALL_RE = re.compile(rf"{_IDENT}\(\)")
_SYMBOL_RE = re.compile(rf"{_IDENT}(?:\.{_IDENT})*")
_PATH_RE = re.compile(r"[\w.\-/@+]+")
_FILE_LINE_RE = re.compile(r"([^:\s]+):(\d{1,9})(?::\d{1,9})?")
_SCOPE_RE = re.compile(r"([^:=\s]+)[:=](\S+)")
_HUNK_RE = re.compile(r"@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@")
_PY_FRAME_RE = re.compile(r"File \"[^\"]+\", line \d+")
_AT_FRAME_RE = re.compile(r"at (?:\S+ ?\([^()\s]+:\d+(?::\d+)?\)|[^()\s]+:\d+:\d+)")      # "at f (src/a.js:10:5)", "at a.B.c(B.java:12)", "at src/a.js:10:5"
_SEPARATORS = re.compile(r"[\s—–\-:;,.?!()\[\]{}\"'`]+")


@dataclass(frozen=True)
class Surface:
    """A question whose shape is a use case P1 does not build: ``kind`` ``diff`` (code review) or ``stack-trace`` (root cause)."""
    kind: str
    use_case: str


@dataclass(frozen=True)
class _Tok:
    start: int
    end: int
    text: str                       # surrounding punctuation stripped


def _surface(question: str) -> Surface | None:
    lines = [ln.strip() for ln in question.split("\n")]
    if any(ln.startswith("diff --git ") or _HUNK_RE.match(ln) for ln in lines) or \
            (any(ln.startswith("--- ") for ln in lines) and any(ln.startswith("+++ ") for ln in lines)):
        return Surface("diff", "code-review")
    if "Traceback (most recent call last)" in question or any(_PY_FRAME_RE.match(ln) or _AT_FRAME_RE.fullmatch(ln) for ln in lines):
        return Surface("stack-trace", "root-cause")
    return None


def _tokens(question: str) -> list[_Tok]:
    out = []
    for m in re.finditer(r"\S+", question):
        start, end = m.start(), m.end()
        while start < end and question[start] in _LEAD:
            start += 1
        while end > start and question[end - 1] in _TRAIL and question[end - 2:end] != "()":     # a call form keeps its parentheses
            if (opener := _OPENER.get(question[end - 1])) and opener in question[start:end - 1]:      # a closer with its opener in the token ({id}) is part of it
                break
            end -= 1
        out.append(_Tok(start, end, question[start:end]))
    return out


def is_path(text: str) -> bool:
    """A file path: path characters only, and a last segment with a known file extension (``a/b.py``, ``image-tag.sh``)."""
    if not text or len(text) > MAX_TOKEN or not _PATH_RE.fullmatch(text):
        return False
    stem, dot, ext = text.rpartition("/")[2].rpartition(".")
    return bool(dot and stem and ext.lower() in FILE_EXTENSIONS)


def _is_qname(text: str) -> bool:
    return len(text) <= MAX_TOKEN and bool(_QNAME_RE.fullmatch(text)) and not is_path(text.removesuffix("()"))


class _Parse:
    def __init__(self, question: str):
        self.q = question
        self.toks = _tokens(question)
        self.used: set[int] = set()
        self.groups: list[Group] = []
        self.scope: str | None = None

    def span(self, a: int, b: int, kind: str) -> Span:
        return Span(self.q[a:b], kind)

    def take(self, *idx: int) -> None:
        self.used.update(idx)

    def run(self) -> None:
        toks = self.toks
        for i, t in enumerate(toks):
            if i in self.used or not t.text or len(t.text) > MAX_TOKEN or t.text.lower() in NOT_ANCHORS:
                continue
            nxt = toks[i + 1] if i + 1 < len(toks) else None
            if t.text in HTTP_VERBS and nxt is not None and nxt.text.startswith("/") and len(nxt.text) <= MAX_TOKEN:
                self.groups.append(Group((self.span(t.start, nxt.end, "route"),)))
                self.take(i, i + 1)
            elif self._scope(i, t, nxt):
                pass
            elif "::" in t.text:
                path, _, sym = t.text.partition("::")
                sym = sym.removesuffix("()")
                if is_path(path) and _SYMBOL_RE.fullmatch(sym):
                    a = t.start + len(path) + 2
                    self.groups.append(Group((self.span(t.start, t.start + len(path), "file"), self.span(a, a + len(sym), "symbol"))))
                    self.take(i)
            elif (m := _FILE_LINE_RE.fullmatch(t.text)) and is_path(m.group(1)):
                self.groups.append(Group((self.span(t.start, t.start + m.end(2), "line"),)))
                self.take(i)
            elif is_path(t.text):
                after = toks[i + 2] if i + 2 < len(toks) else None
                if nxt is not None and after is not None and nxt.text.lower() in LINE_WORDS and after.text.isdigit() and len(after.text) <= 9:
                    self.groups.append(Group((self.span(t.start, t.end, "file"), self.span(nxt.start, after.end, "line"))))
                    self.take(i, i + 1, i + 2)
                else:
                    self.groups.append(Group((self.span(t.start, t.end, "file"),)))
                    self.take(i)
            elif _is_qname(t.text) or _CALL_RE.fullmatch(t.text):
                self.groups.append(Group((self.span(t.start, t.start + len(t.text.removesuffix("()")), "symbol"),)))
                self.take(i)

    def _scope(self, i: int, t: _Tok, nxt: _Tok | None) -> bool:
        m = _SCOPE_RE.fullmatch(t.text)
        if m and m.group(1).lower() in SCOPE_KEYS:
            self.scope = self.q[t.start + len(m.group(1)) + 1:t.end]
            self.take(i)
            return True
        raw = self.q[t.start:t.end + 1]
        if raw.endswith(":") and t.text.lower() in SCOPE_KEYS and nxt is not None:
            self.scope = nxt.text
            self.take(i, i + 1)
            return True
        return False

    def remainder(self) -> str:
        words = [t.text.lower() if t.text.lower() in DIRECTION_WORDS else _SEPARATORS.sub(" ", t.text).lower() for i, t in enumerate(self.toks) if i not in self.used]
        return " ".join(" ".join(words).split())


def parse(question: str) -> Surface | Draft | None:
    """The deterministic reading of ``question`` (already ``normalize_question``-ed): a not-built :class:`Surface`, a complete draft, or ``None`` (free prose)."""
    surface = _surface(question)
    if surface is not None:
        return surface
    p = _Parse(question)
    p.run()
    rest = p.remainder()
    if rest in UNBUILT_WORDS:
        return Draft(UNBUILT_WORDS[rest], None, tuple(p.groups[:3]), scope=p.scope) if len(p.groups) <= 3 else None
    if not p.groups or len(p.groups) > 3:
        return None
    if rest and rest not in DIRECTION_WORDS:
        return None
    return Draft(BUILT_USE_CASE, DIRECTION_WORDS.get(rest), tuple(p.groups), scope=p.scope)
