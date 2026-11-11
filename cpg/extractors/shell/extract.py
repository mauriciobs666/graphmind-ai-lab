"""Shell extractor (C-1001 P1-10): tree-sitter-bash -> bundle records for shell scripts.

Plan: ``docs/plans/connected-code-graph-p1-extraction.md`` P1-10; shapes: ``docs/plans/connected-code-graph-bundle-spec3.md``
§4.1 rule 4 and "Shell" (one ``<script>`` file symbol, every function top level), §5.4 ``param``, §5.6 ``callsite``,
§5.8 ``valueSet``, §5.9 ``constant``, §5.10 ``data``, §7.

``extract_shell(unit_root, files, ...)`` returns the records of the given files: ``file``, ``symbol``, ``param``,
``callsite``, ``valueSet``, ``constant``, ``data``. No header (the caller's assembly owns it, the counts and the
``lastChanged*`` history). It never writes ``configEntry``/``launchFact``: a ``${var:-default}`` is a value-set
alternative, never a ``bindValue`` (W3), and launch facts are P1-11's.

**Untrusted input.** A shell file is repository content: it is parsed, never run, never ``source``d or ``eval``-ed. Every
walk is an explicit-stack loop, nesting deeper than ``MAX_DEPTH`` and a file over ``MAX_FILE_BYTES`` are refused,
per-command arguments, the expansion-resolution depth and the records per file are capped. A name that leaves
``unit_root`` (``..``, an absolute name, a symlink out), a file that is not regular and bytes that are not UTF-8
(index D12) raise ``SourceRefused``: the unit fails, nothing is skipped silently. A file tree-sitter cannot parse
cleanly is a warning (``warnings``) and keeps its ``<script>`` symbol with no calls.

**What is modelled** (everything else is a recorded limit, README):

- *Scopes.* The script and each function are separate symbols; a command belongs to the innermost one. Positional
  parameters are per scope: ``$1``..``$N`` are ``param`` indexes ``0..N-1`` (named ``$1``..), ``$@``/``$*`` one more
  param named ``$@`` at index N (N = the highest numbered one the scope mentions); ``shift`` is not modelled.
- *Call sites.* Every ``command`` node is a ``callsite`` of kind ``command``; ``calleeName`` is the literal command word as
  written, ``<dynamic>`` when it is computed. A name that is a function of the same file resolves to it
  (``confirmed``/``resolved-call``); any other is ``possible``/``name-match``; ``eval``, a computed command name, a computed
  ``source``/``.`` path and an argument with indirect expansion (``${!n}``) are ``possible``/``shell-dynamic``.
  An ``eval`` with exactly one argument keeps its call site and *adds* one inner ``possible``/``shell-dynamic`` call site
  per complete simple command found by parsing the argument's leading literal as shell (``_eval_sites``; P1-10 10b);
  an inner command naming a function of the file resolves to it like any other command (the bundle schema ties ``resolved`` to ``confirmed``).
- *Values.* Each argument (``a0``..) has a ``valueSet``: a literal, an interpolated string (``template``), the reaching
  assignment(s) of a variable in the same scope, a ``ref`` to a once-assigned script-level variable (``constant``), or
  an ``unknown``. A literal that fails the token screen (``bundle.screen_fails``) and the value assigned to a
  variable whose name fails the name rule, or that follows a flag whose name does, is withheld as ``unknown``.
- *Flow.* ``data`` records (use = the call site, ``a<i>``; def = a ``param`` ``p<k>`` or an inner call site ``r``) for an
  argument whose value comes from a positional parameter or a command substitution, through assignments.
"""
from __future__ import annotations

import bisect
import hashlib
import json
import os
import re
import time
import unicodedata
from collections.abc import Callable, Iterable
from functools import cache
from pathlib import Path
from typing import NamedTuple

import tree_sitter_bash
from tree_sitter import Language, Node, Parser

from ccg import bundle, ids
from extractors.common.errors import SourceRefused     # the one class P1-44 maps to `extract:failed`

LANG = "shell"
MAX_FILE_BYTES = 2 * 1024 * 1024    # the largest real script is 171 kB; over the cap the unit fails (SourceRefused)
MAX_DEPTH = 120                     # tree nesting; deeper is a warning and a call-less <script>
MAX_ARGS = 32                       # arguments modelled per command
MAX_POSITIONAL = 64                 # $N with N above this is a runtime-built value, not a param
MAX_RECORDS = 100_000               # records per file; past it the remaining commands are dropped with a warning
MAX_CONST = 8192                    # a literal longer than this is withheld
MAX_ALTERNATIVES = 16               # spec3 §5.8
MAX_RESOLVE_DEPTH = 8               # variable -> variable chains
MAX_TEMPLATE_PARTS = 64             # parts of one interpolated string; more is a value-set-cap
MAX_EVAL_CHARS = 4096               # the leading literal of an `eval` argument that is parsed as shell; the rest is dropped
MAX_EVAL_COMMANDS = 8               # inner call sites per `eval` string
PARSE_TIMEOUT_SECONDS = 20.0

_UNKNOWN = {"unknown": "runtime-built"}
#: the statement nodes an `eval` string's commands are looked for under (a compound command's body is not entered)
_EVAL_WALK = frozenset({"program", "list", "pipeline", "redirected_statement", "negated_command", "subshell",
                        "compound_statement"})
_REDIRECTS = frozenset({"file_redirect", "heredoc_redirect", "herestring_redirect", "comment"})
#: constructs whose assignments may not run, or do not leak out: an assignment inside is added, never a kill.
_CONDITIONAL = frozenset({
    "if_statement", "elif_clause", "else_clause", "case_statement", "case_item", "while_statement", "for_statement",
    "c_style_for_statement", "list", "pipeline", "subshell", "command_substitution", "process_substitution"})
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_POSITIONAL = re.compile(r"[1-9][0-9]*")
_DEFAULT_OPS = frozenset({":-", "-", ":=", "="})
_READ_NAMES = frozenset({"read", "mapfile", "readarray"})
_READ_FLAGS_WITH_VALUE = frozenset({"-p", "-d", "-n", "-N", "-t", "-u", "-i", "-a", "-C", "-c", "-s", "-O"})


@cache
def _parser() -> Parser:
    return Parser(Language(tree_sitter_bash.language()))


def _parse(data: bytes):
    """The syntax tree, or ``None`` when the parser gave up after ``PARSE_TIMEOUT_SECONDS``."""
    deadline, expired = time.monotonic() + PARSE_TIMEOUT_SECONDS, []

    def read(offset: int, _point) -> bytes:
        if time.monotonic() > deadline:
            expired.append(offset)
            return b""                                    # end the input: the tree is discarded below
        return data[offset:offset + 65536]

    tree = _parser().parse(read)
    return None if expired else tree


# --- small helpers --------------------------------------------------------------------------------------

def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def _name_norm(name: str) -> str:
    return unicodedata.normalize("NFC", _nfc(name).casefold()).strip()


def _canon(el: object) -> bytes:
    return json.dumps(el, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _file_role(rel: str) -> str:
    base = rel.rsplit("/", 1)[-1].lower()
    parts = rel.lower().split("/")
    if (".example" in base or ".sample" in base or base.startswith(("example", "sample"))
            or "examples" in parts[:-1] or "samples" in parts[:-1]):
        return "example"
    return "primary"


def _check_rel(rel: object) -> str:
    if not isinstance(rel, str) or not rel or "\x00" in rel or rel.startswith("/") \
            or any(p in ("", ".", "..") for p in rel.split("/")):
        raise SourceRefused("a file name is empty, absolute, not normalized or leaves the unit root")
    return rel


check_rel = _check_rel          # public name for callers outside this module (``ccg.build`` validates tree paths with it)


def _read(root: Path, rel: str) -> bytes:
    try:
        target = (root / rel).resolve()
    except (OSError, ValueError, RuntimeError):
        raise SourceRefused(f"{rel!r} cannot be resolved") from None
    if not target.is_relative_to(root):
        raise SourceRefused(f"{rel!r} resolves outside the unit root")
    try:
        if not target.is_file():
            raise SourceRefused(f"{rel!r} is not a regular file")
        with open(target, "rb") as f:
            data = f.read(MAX_FILE_BYTES + 1)
    except OSError as e:
        raise SourceRefused(f"{rel!r} cannot be read ({type(e).__name__})") from None
    if len(data) > MAX_FILE_BYTES:
        raise SourceRefused(f"{rel!r} is over {MAX_FILE_BYTES} bytes")
    return data


def _text(data: bytes, node: Node) -> str:
    return data[node.start_byte:node.end_byte].decode("utf-8")


def _scan(root: Node) -> tuple[int, Node | None]:
    """(max depth, the first error/missing node in source order) of a tree, by an explicit stack."""
    depth, first = 0, None
    stack = [(root, 1)]
    while stack:
        n, d = stack.pop()
        depth = max(depth, d)
        if (n.is_error or n.is_missing) and (first is None or n.start_byte < first.start_byte):
            first = n
        stack.extend((c, d + 1) for c in n.children)
    return depth, first


def _unescape(s: str, dq: bool) -> str:
    """Backslash removal: in a double-quoted string only before ``\\ " $ ```, anywhere else before any character."""
    if "\\" not in s:
        return s
    if dq:
        return re.sub(r'\\\n|\\([\\"$`])', lambda m: m.group(1) or "", s)
    return re.sub(r"\\\n|\\(.)", lambda m: m.group(1) or "", s, flags=re.S)


# --- value model ----------------------------------------------------------------------------------------

class _Part(NamedTuple):
    alts: tuple        # alternatives (elements) this part may be
    flows: tuple       # ((kind, ref, dynamic), ...): kind "p" (param index) or "r" (start byte of the inner command)
    dyn: bool          # indirect expansion somewhere in it
    text: str | None   # the literal text when the whole part is literal
    spread: bool = False


def _lit(text: str) -> _Part:
    return _Part(({"const": _nfc(text)},), (), False, text)


def _unk(reason: str, flows: tuple = (), dyn: bool = False) -> _Part:
    return _Part(({"unknown": reason},), flows, dyn, None)


def _merge_flows(*groups) -> tuple:
    seen: dict[tuple, bool] = {}
    for g in groups:
        for kind, ref, dyn in g:
            seen[(kind, ref)] = seen.get((kind, ref), False) or dyn
    return tuple((k, r, d) for (k, r), d in seen.items())


def _collapse(alts: tuple) -> dict:
    """One element standing for several alternatives, where only one fits (inside a template)."""
    uniq = {_canon(a): a for a in alts}
    if len(uniq) == 1:
        return next(iter(uniq.values()))
    derived = any(a.get("unknown") == "point-derived" for a in alts)
    return {"unknown": "point-derived" if derived else "runtime-built"}


def _assemble(parts: list[_Part]) -> _Part:
    flows = _merge_flows(*(p.flows for p in parts))
    dyn = any(p.dyn for p in parts)
    if not parts:
        return _Part(({"const": ""},), (), False, "")
    if len(parts) == 1:
        return parts[0]
    pieces: list[dict] = []
    for p in parts:
        el = _collapse(p.alts)
        for piece in el["template"] if "template" in el else [el]:
            if pieces and "const" in piece and "const" in pieces[-1] and isinstance(piece["const"], str) \
                    and isinstance(pieces[-1]["const"], str):
                pieces[-1] = {"const": pieces[-1]["const"] + piece["const"]}
            else:
                pieces.append(piece)
    text = "".join(p.text for p in parts) if all(p.text is not None for p in parts) else None
    if len(pieces) > MAX_TEMPLATE_PARTS or sum(len(x["const"]) for x in pieces if isinstance(x.get("const"), str)) > MAX_CONST:
        return _Part(({"unknown": "value-set-cap"},), flows, dyn, None)
    el = pieces[0] if len(pieces) == 1 else {"template": pieces}
    return _Part((el,), flows, dyn, text)


class _Def(NamedTuple):
    name: str
    start: int
    end: int
    kind: str           # "assign" | "append" | "bare" | "read" | "for"
    value: Node | None
    conditional: bool
    line: int


class _Scope:
    def __init__(self, sym_id: str, qname: str, is_script: bool, body: Node, line: int):
        self.sym_id, self.qname, self.is_script, self.body, self.line = sym_id, qname, is_script, body, line
        self.commands: list[Node] = []
        self.defs: dict[str, list[_Def]] = {}
        self.starts: dict[str, list[int]] = {}
        self.pmax = 0               # highest numbered positional the scope mentions
        self.varargs = False
        self.cs_by_start: dict[int, str] = {}      # command node start byte -> callsite id
        self.cache: dict[tuple[int, int], tuple] = {}   # (def start byte, chain depth) -> (alts, flows, dyn)

    def add_def(self, d: _Def) -> None:
        self.defs.setdefault(d.name, []).append(d)


class _Func(NamedTuple):
    node: Node
    name: str
    k: int              # 1 for the first definition of the name, 2.. for later ones (#N)


# --- the extractor of one file --------------------------------------------------------------------------

class _FileExtractor:
    def __init__(self, path: str, data: bytes, text: str, *, repo_id: str, unit_id: str, history, warnings: list):
        self.path, self.data, self.text = path, data, text
        self.repo, self.unit_id, self.history, self.warnings = repo_id, unit_id, history, warnings
        self.lines = text.split("\n")
        self.line_count = text.count("\n") + (0 if text.endswith("\n") or text == "" else 1)
        self.file_id = ids.file_id(repo_id, path)
        self.records: list[dict] = []
        self.constants: dict[str, tuple[Node, int]] = {}     # name -> (value node, line); script-level, assigned once
        self.func_names: set[str] = set()
        self.by_name: dict[str, list[tuple[str, str]]] = {}  # function name -> [(symbolId, qname)]
        self.targets: dict[str, list[dict]] = {}
        self.cap_warned = False

    # -- warnings and records
    def warn(self, code: str, line: int | None, detail: str) -> None:
        self.warnings.append({"path": self.path, "code": code, "line": line, "detail": detail})

    def _span_text(self, ls: int, le: int) -> str:
        wanted = (self.lines[n - 1].strip() for n in range(ls, min(le, len(self.lines)) + 1))
        return "\n".join(x for x in wanted if x)

    def symbol(self, name: str, qname: str, kind: str, ls: int, le: int, parent: str | None) -> dict:
        rec = {"type": "symbol", "id": ids.symbol_id(LANG, self.repo, qname), "lang": LANG, "repoId": self.repo,
               "qname": qname, "fileId": self.file_id, "path": self.path, "kind": kind, "name": name,
               "nameNorm": _name_norm(name), "lineStart": ls, "lineEnd": le,
               "fingerprint": _sha256(self._span_text(ls, le).encode("utf-8")), "complexity": 1}
        if parent is not None:
            rec["parentId"] = parent
        if self.history is not None:
            rec["lastChangedAt"], rec["lastChangedCommit"] = self.history(self.path, [(ls, le)])
        return rec

    def file_record(self) -> dict:
        return {"type": "file", "id": self.file_id, "repoId": self.repo, "unitId": self.unit_id, "path": self.path,
                "lang": LANG, "kind": "script", "role": _file_role(self.path), "contentHash": _sha256(self.data),
                "lineCount": self.line_count}

    def script_symbol(self) -> dict:
        return self.symbol("<script>", f"{self.path}::<script>", "script", 1, max(1, self.line_count), None)

    # -- driver
    def run(self, root: Node) -> list[dict]:
        script = self.script_symbol()
        self.records = [self.file_record(), script]
        funcs = self._functions(root)
        scopes = [_Scope(script["id"], script["qname"], True, root, 1)]
        for f in funcs:
            q = f"{self.path}::{f.name}" + (f"#{f.k}" if f.k > 1 else "")
            ls, le = self._lines(f.node)
            sym = self.symbol(f.name, q, "function", ls, le, script["id"])
            self.records.append(sym)
            self.by_name.setdefault(f.name, []).append((sym["id"], q))
            name_node = next(c for c in f.node.children if c.type == "word")
            body = next((c for c in f.node.named_children if c.start_byte >= name_node.end_byte), f.node)
            scopes.append(_Scope(sym["id"], q, False, body, ls))
        self.func_names = set(self.by_name)
        for s in scopes:
            self._collect(s)
        self._find_constants(scopes)
        for s in scopes:
            self._process(s)
        return self.records

    def _lines(self, node: Node) -> tuple[int, int]:
        ls, le = node.start_point.row + 1, node.end_point.row + 1
        if node.end_point.column == 0 and le > ls:
            le -= 1
        return ls, max(ls, le)

    def _functions(self, root: Node) -> list[_Func]:
        found: list[tuple[int, Node, str]] = []
        stack = [root]
        while stack:
            n = stack.pop()
            if n.type == "function_definition":
                nm = next((c for c in n.children if c.type == "word"), None)
                name = _nfc(_text(self.data, nm)).strip() if nm is not None else ""
                if name:
                    found.append((n.start_byte, n, name))
            stack.extend(n.children)
        found.sort(key=lambda t: t[0])
        seen: dict[str, int] = {}
        out = []
        for _, n, name in found:
            seen[name] = seen.get(name, 0) + 1
            out.append(_Func(n, name, seen[name]))
        return out

    # -- one scope's nodes: commands, definitions, positional references
    def _collect(self, sc: _Scope) -> None:
        stack: list[tuple[Node, bool, str]] = [(sc.body, False, "")]
        while stack:
            n, cond, parent = stack.pop()
            t = n.type
            if t == "function_definition":
                continue
            if t == "command":
                sc.commands.append(n)
                self._read_defs(sc, n, cond)
            elif t == "variable_assignment" and parent != "command":
                self._assign_def(sc, n, cond)
            elif t == "declaration_command":
                for c in n.children:
                    if c.type == "variable_name":
                        sc.add_def(_Def(_text(self.data, c), c.start_byte, c.end_byte, "bare", None, cond, c.start_point.row + 1))
            elif t == "for_statement":
                v = next((c for c in n.children if c.type == "variable_name"), None)
                if v is not None:
                    sc.add_def(_Def(_text(self.data, v), n.start_byte, v.end_byte, "for", None, True, n.start_point.row + 1))
            elif t in ("simple_expansion", "expansion"):
                self._note_positional(sc, n)
            child_cond = cond or t in _CONDITIONAL
            stack.extend((c, child_cond, t) for c in reversed(n.children))
        sc.commands.sort(key=lambda n: n.start_byte)
        for name, defs in sc.defs.items():
            defs.sort(key=lambda d: d.start)
            sc.starts[name] = [d.start for d in defs]

    def _assign_def(self, sc: _Scope, n: Node, cond: bool) -> None:
        var = n.children[0] if n.children else None
        if var is None or var.type != "variable_name":
            return                                     # a[i]=v: an element, not the variable
        op = next((c.type for c in n.children[1:] if c.type in ("=", "+=")), "=")
        value = next((c for c in n.children[1:] if c.is_named), None)
        sc.add_def(_Def(_text(self.data, var), n.start_byte, n.end_byte, "assign" if op == "=" else "append",
                        value, cond or op == "+=", n.start_point.row + 1))       # an append keeps the earlier value

    def _read_defs(self, sc: _Scope, cmd: Node, cond: bool) -> None:
        name = cmd.child_by_field_name("name")
        if name is None or _text(self.data, name) not in _READ_NAMES:
            return
        skip = False
        for c in (c for c in cmd.named_children if c.start_byte >= name.end_byte):
            txt = _text(self.data, c)
            if skip:
                skip = False
            elif c.type == "word" and txt in _READ_FLAGS_WITH_VALUE:
                skip = True
            elif c.type == "word" and _IDENT.fullmatch(txt):
                sc.add_def(_Def(txt, cmd.start_byte, cmd.end_byte, "read", None, cond, cmd.start_point.row + 1))

    def _note_positional(self, sc: _Scope, n: Node) -> None:
        kind, num = self._positional(n)
        if kind == "n" and num <= MAX_POSITIONAL:
            sc.pmax = max(sc.pmax, num)
        elif kind == "all":
            sc.varargs = True

    def _var_node(self, n: Node) -> Node | None:
        return next((c for c in n.children if c.type in ("variable_name", "special_variable_name", "subscript")), None)

    def _positional(self, n: Node) -> tuple[str, int]:
        v = self._var_node(n)
        if v is None or v.type == "subscript":
            return "", 0
        txt = _text(self.data, v)
        if v.type == "special_variable_name":
            return ("all", 0) if txt in ("@", "*") else ("", 0)
        if n.type == "simple_expansion" and txt[0] in "123456789":      # the grammar reads `$1y` as one name; bash reads `${1}y`
            return "n", int(txt[0])
        return ("n", int(txt)) if _POSITIONAL.fullmatch(txt) else ("", 0)

    # -- constants: a script-level variable assigned exactly once, unconditionally, to a value set (spec3 §5.9)
    def _find_constants(self, scopes: list[_Scope]) -> None:
        count: dict[str, int] = {}
        for s in scopes:
            for name, defs in s.defs.items():
                count[name] = count.get(name, 0) + len(defs)
        for name, defs in scopes[0].defs.items():
            d = defs[0]
            if count[name] == 1 and d.kind == "assign" and not d.conditional and d.value is not None \
                    and name not in self.func_names and not bundle.name_fails(name):
                self.constants[name] = (d.value, d.line)

    # -- evaluation of words
    def _ev(self, node: Node, sc: _Scope, at: int, depth: int, dq: bool = False) -> _Part:
        return _assemble(self._parts(node, sc, at, depth, dq))

    def _parts(self, n: Node, sc: _Scope, at: int, depth: int, dq: bool) -> list[_Part]:
        t = n.type
        if t in ("word", "number", "regex", "extglob_pattern"):
            return [_lit(_unescape(_text(self.data, n), False))]
        if t == "string_content":
            return [_lit(_unescape(_text(self.data, n), True))]
        if t == "raw_string":
            return [_lit(_text(self.data, n)[1:-1])]
        if t == "string":
            out: list[_Part] = []
            for c in n.children:
                if c.type == '"':
                    continue
                out.extend(self._parts(c, sc, at, depth, True) if c.is_named else [_lit(_text(self.data, c))])
            return out
        if t in ("concatenation", "command_name"):
            return [p for c in n.children for p in self._parts(c, sc, at, depth, dq)]
        if t in ("simple_expansion", "expansion"):
            part = self._expansion(n, sc, at, depth)
            v = self._var_node(n)
            if t == "simple_expansion" and v is not None and v.type == "variable_name":
                txt = _text(self.data, v)
                if len(txt) > 1 and txt[0] in "0123456789":              # `$1y`: the positional, then the literal `y`
                    return [part, _lit(txt[1:])]
            return [part]
        if t == "command_substitution":
            inner = self._substituted_command(n)
            if inner is None:
                return [_unk("runtime-built")]
            return [_unk("point-derived", (("r", inner.start_byte, False),))]
        return [_unk("runtime-built")]

    def _substituted_command(self, n: Node) -> Node | None:
        stmts = [c for c in n.named_children if c.type != "comment"]
        cur = stmts[-1] if stmts else None
        while cur is not None and cur.type in ("pipeline", "redirected_statement", "negated_command"):
            named = [c for c in cur.named_children if c.type not in _REDIRECTS]
            cur = named[-1] if named else None
        return cur if cur is not None and cur.type == "command" else None

    def _expansion(self, n: Node, sc: _Scope, at: int, depth: int) -> _Part:
        v = self._var_node(n)
        if v is None or v.type == "subscript":
            return _unk("runtime-built")
        kind, num = self._positional(n)
        if kind == "all":
            base = _Part(({"unknown": "spread-argument"},), (("p", sc.pmax, False),), False, None, spread=True)
        elif kind == "n" and num <= MAX_POSITIONAL:
            base = _unk("point-derived", (("p", num - 1, False),))
        elif kind == "n" or v.type == "special_variable_name" or _text(self.data, v)[0] in "0123456789":
            base = _unk("runtime-built")
        else:
            base = self._resolve(_text(self.data, v), sc, at, depth)
        if n.type == "simple_expansion":
            return base
        kids = n.children
        if len(kids) > 1 and kids[1].type == "!":                          # ${!name}: the value of the variable it names
            return _unk("runtime-built", tuple((k, r, True) for k, r, _ in base.flows), True)
        if len(kids) > 1 and kids[1].type == "#":                          # ${#name}: a length
            return _unk("runtime-built", base.flows)
        op = next((c for c in kids if c.type in _DEFAULT_OPS), None)
        if op is not None:                                                  # ${v:-d}: v or d (never a bindValue, W3)
            nxt = op.next_named_sibling
            dflt = self._ev(nxt, sc, at, depth + 1) if nxt is not None else _lit("")
            return _Part(base.alts + dflt.alts, _merge_flows(base.flows, dflt.flows), base.dyn or dflt.dyn, None)
        if any(c.type not in ("${", "}", "variable_name", "special_variable_name") for c in kids):
            return _unk("runtime-built", base.flows)                        # ${v%x}, ${v/a/b}, ${v:+x}, ${v:1}: derived
        return base

    def _resolve(self, name: str, sc: _Scope, at: int, depth: int) -> _Part:
        cands = self._reaching(sc, name, at)
        if cands is None:
            return _unk("value-set-cap")
        if not cands:
            if not sc.is_script and name in self.constants:
                return _Part(({"ref": f"{self.path}::{name}"},), (), False, None)
            return _unk("unresolved-ref")
        parts = [self._def_value(d, sc, depth) for d in cands]
        alts = tuple({_canon(a): a for p in parts for a in p.alts}.values())
        return _Part(alts, _merge_flows(*(p.flows for p in parts)), any(p.dyn for p in parts), None)

    def _reaching(self, sc: _Scope, name: str, at: int) -> list[_Def] | None:
        """The definitions of ``name`` that reach ``at``: back from the last one before it, up to and including the first
        unconditional one (it kills what precedes). ``None`` when more than ``MAX_ALTERNATIVES`` reach."""
        defs = sc.defs.get(name)
        if not defs:
            return []
        out: list[_Def] = []
        for i in range(bisect.bisect_left(sc.starts[name], at) - 1, -1, -1):
            d = defs[i]
            if d.end > at:
                continue                                            # the assignment this very use is inside
            out.append(d)
            if not d.conditional:
                break
            if len(out) > MAX_ALTERNATIVES:
                return None
        out.reverse()
        return out

    def _def_value(self, d: _Def, sc: _Scope, depth: int) -> _Part:
        """The value of definition ``d`` reached at chain depth ``depth``. Memoized on (definition, depth), so a use's value
        never depends on which uses were evaluated before it, and the chain limit holds on every path."""
        if d.kind != "assign" or d.value is None or depth >= MAX_RESOLVE_DEPTH:
            return _unk("runtime-built")
        key = (d.start, depth)
        if key not in sc.cache:
            sc.cache[key] = ((_UNKNOWN,), (), False)         # a cycle resolves to unknown, never recurses
            p = self._ev(d.value, sc, d.start, depth + 1)
            sc.cache[key] = (self._mask(p.alts) if bundle.name_fails(d.name) else p.alts, p.flows, p.dyn)
        alts, flows, dyn = sc.cache[key]
        return _Part(alts, flows, dyn, None)

    def _mask(self, alts):
        def one(el):
            if "const" in el:
                return _UNKNOWN
            if "template" in el:
                return {"template": [one(x) for x in el["template"]]}
            return el
        return tuple(one(a) for a in alts)

    def _screen(self, el: dict) -> dict:
        if "const" in el:
            c = el["const"]
            return _UNKNOWN if not isinstance(c, str) or len(c) > MAX_CONST or bundle.screen_fails(c) else el
        if "template" in el:
            return {"template": [self._screen(x) for x in el["template"]]}
        return el

    def _values(self, alts: tuple, mask: bool) -> list[dict]:
        els = list(self._mask(alts)) if mask else list(alts)
        uniq = {_canon(e): e for e in map(self._screen, els)}
        ordered = [uniq[k] for k in sorted(uniq)]
        if len(ordered) > MAX_ALTERNATIVES:
            capped = {"unknown": "value-set-cap"}
            uniq = {_canon(e): e for e in ordered[:MAX_ALTERNATIVES - 1] + [capped]}
            ordered = [uniq[k] for k in sorted(uniq)]
        return ordered

    # -- one scope: params, call sites, value sets, data
    def _process(self, sc: _Scope) -> None:
        sym = sc.sym_id
        for name, (vnode, line) in (self.constants.items() if sc.is_script else ()):
            self._emit_constant(sc, name, vnode, line)
        for i in range(sc.pmax + (1 if sc.varargs else 0)):
            self.records.append({"type": "param", "id": ids.point_id(sym, "param", i), "symbolId": sym, "index": i,
                                 "name": f"${i + 1}" if i < sc.pmax else "$@", "line": sc.line})
        sites = []            # (command node, callsite record)
        counts: dict[str, int] = {}
        pending = 0           # records the accepted call sites will still write
        for cmd in sc.commands:
            cost = 1 + 2 * min(cmd.named_child_count, MAX_ARGS)
            if len(self.records) + pending + cost > MAX_RECORDS:
                if not self.cap_warned:
                    self.warn("record-cap", cmd.start_point.row + 1, f"more than {MAX_RECORDS} records; the rest dropped")
                    self.cap_warned = True
                break
            name_node = cmd.child_by_field_name("name")
            if name_node is None:
                continue
            nm = self._callee(name_node, sc)
            k = counts.get(nm, 0)
            counts[nm] = k + 1
            targets = self._targets(nm)
            rec = {"type": "callsite", "id": ids.callsite_id(sym, nm, k), "symbolId": sym, "calleeName": nm,
                   "ordinal": k, "callKind": "command", "line": name_node.start_point.row + 1,
                   "resolution": "resolved" if targets else "unresolved", "targets": targets,
                   "status": "confirmed" if targets else "possible",
                   "reason": "resolved-call" if targets else "name-match"}
            sc.cs_by_start[cmd.start_byte] = rec["id"]
            sites.append((cmd, rec))
            pending += cost
        for cmd, rec in sites:
            self._arguments(sc, cmd, rec)
            self.records.append(rec)
        # the inner sites of an `eval` string come after every written command, so no written site is renumbered; each
        # takes the next ordinal of its callee in source order of the `eval`s
        for cmd, rec in sites:
            if rec["calleeName"] == "eval" and not self._eval_sites(sc, cmd, counts):
                break

    def _lead(self, n: Node, data: bytes) -> tuple[str, bool]:
        """(the literal text a word starts with, whether the whole word is that literal): the text up to the first
        expansion or substitution, read from the syntax only (a variable's value is never substituted)."""
        t = n.type
        if t in ("word", "number"):
            return _unescape(data[n.start_byte:n.end_byte].decode("utf-8"), False), True
        if t == "string_content":
            return _unescape(data[n.start_byte:n.end_byte].decode("utf-8"), True), True
        if t == "raw_string":
            return data[n.start_byte + 1:n.end_byte - 1].decode("utf-8"), True
        if t in ("concatenation", "command_name", "string"):
            out: list[str] = []
            for c in n.children:
                if c.type == '"':
                    continue
                text, whole = self._lead(c, data)
                out.append(text)
                if not whole or sum(map(len, out)) > MAX_EVAL_CHARS:
                    return "".join(out), False
            return "".join(out), True
        return "", False

    def _eval_sites(self, sc: _Scope, cmd: Node, counts: dict[str, int]) -> bool:
        """Add the inner call sites of the ``eval`` command ``cmd`` (P1-10 10b). ``eval`` joins its arguments and runs the
        result as shell, so the leading literal of its one argument is parsed as shell, never split on spaces, and every
        complete simple command in it is a call site (``possible``/``shell-dynamic``: it runs under ``eval``). A word is
        complete only when a delimiter follows it inside the literal, or the argument ends there; an open word is dropped
        and, when it was an argument, stands as one ``unknown``. Nothing is added for an argument that does not start with
        a literal, a literal that does not parse cleanly, a command name that is not a plain literal, or an ``eval``.
        Returns False when the record cap stopped the file (the caller stops)."""
        name_end = cmd.child_by_field_name("name").end_byte
        argv = [c for c in cmd.named_children if c.start_byte >= name_end and c.type not in _REDIRECTS]
        if len(argv) != 1:
            return True
        text, whole = self._lead(argv[0], self.data)
        text, whole = (text[:MAX_EVAL_CHARS], False) if len(text) > MAX_EVAL_CHARS else (text, whole)
        if not text.strip():
            return True
        src = text.encode("utf-8")
        tree = _parse(src)
        if tree is None:
            return True
        depth, bad = _scan(tree.root_node)
        if bad is not None or depth > MAX_DEPTH:
            return True
        found: list[Node] = []
        stack = [tree.root_node]
        while stack:
            n = stack.pop()
            if n.type == "command":
                found.append(n)
            elif n.type in _EVAL_WALK:
                stack.extend(n.named_children)
        found.sort(key=lambda n: n.start_byte)
        done = 0
        for inner in found:
            iname = inner.child_by_field_name("name")
            if iname is None:
                continue
            nm, nm_whole = self._lead(iname, src)
            nm = _nfc(nm).strip()
            if not nm_whole or not nm or nm == "eval" or re.search(r"[\s\x00-\x1f\x7f]", nm):
                continue
            iargv = [c for c in inner.named_children if c.start_byte >= iname.end_byte and c.type not in _REDIRECTS]
            trailing = False
            if not whole:
                last = inner.children[-1]
                if last.end_byte == len(src):                       # the literal ends inside this word: it is open
                    if not iargv or last.id != iargv[-1].id:
                        continue                                    # an open command name, or a redirect
                    iargv, trailing = iargv[:-1], True
                elif not src[inner.end_byte:].strip(b" \t"):        # a space follows: the next word starts there
                    trailing = True
            if done == MAX_EVAL_COMMANDS:
                break
            cost = 1 + 2 * (min(len(iargv), MAX_ARGS) + 1)
            if len(self.records) + cost > MAX_RECORDS:
                if not self.cap_warned:
                    self.warn("record-cap", cmd.start_point.row + 1, f"more than {MAX_RECORDS} records; the rest dropped")
                    self.cap_warned = True
                return False
            done += 1
            k = counts.get(nm, 0)
            counts[nm] = k + 1
            targets = self._targets(nm)
            cs = ids.callsite_id(sc.sym_id, nm, k)
            line = argv[0].start_point.row + 1 + src[:inner.start_byte].count(b"\n")
            self.records.append({"type": "callsite", "id": cs, "symbolId": sc.sym_id, "calleeName": nm, "ordinal": k,
                                 "callKind": "command", "line": line, "resolution": "resolved" if targets else "unresolved",
                                 "targets": targets, "status": "confirmed" if targets else "possible",
                                 "reason": "resolved-call" if targets else "shell-dynamic"})   # the schema: resolved => confirmed
            prev_flag = None
            parts = []
            for a in iargv[:MAX_ARGS]:
                text_a, whole_a = self._lead(a, src)
                parts.append(_lit(text_a) if whole_a else _unk("runtime-built"))
            if trailing and len(parts) < MAX_ARGS:
                parts.append(_unk("runtime-built"))
            for i, p in enumerate(parts):
                mask, prev_flag = self._flag_mask(p, prev_flag)
                arg = f"a{i}"
                self.records.append({"type": "valueSet", "id": ids.value_set_id(cs, arg), "callSiteId": cs, "arg": arg,
                                     "values": self._values(p.alts, mask)})
        return True

    def _targets(self, name: str) -> list[dict]:
        """The in-file definitions of function ``name`` (the first ``MAX_ALTERNATIVES`` of a redefined one), by symbolId."""
        if name not in self.targets:
            self.targets[name] = sorted(({"symbolId": i, "qname": q} for i, q in self.by_name.get(name, ())[:MAX_ALTERNATIVES]),
                                        key=lambda t: t["symbolId"])
        return [dict(t) for t in self.targets[name]]

    def _callee(self, name_node: Node, sc: _Scope) -> str:
        p = self._ev(name_node, sc, name_node.start_byte, 0)
        t = _nfc(p.text).strip() if p.text is not None else ""
        return t if t and not re.search(r"[\s\x00-\x1f\x7f]", t) else "<dynamic>"

    def _emit_constant(self, sc: _Scope, name: str, vnode: Node, line: int) -> None:
        p = self._ev(vnode, sc, vnode.start_byte, 1)
        vals = self._values(p.alts, False)
        if all("unknown" in v for v in vals):
            return
        q = f"{self.path}::{name}"
        self.records.append({"type": "constant", "id": ids.constant_id(LANG, self.repo, q), "qname": q,
                             "fileId": self.file_id, "line": line, "values": vals})

    @staticmethod
    def _flag_mask(p: _Part, prev_flag: str | None) -> tuple[bool, str | None]:
        """(is this argument's value withheld, the flag name the next argument follows) for an argument ``p``."""
        mask = prev_flag is not None and bundle.name_fails(prev_flag)
        if p.text is not None and re.match(r"-{1,2}[^=\s]+=", p.text):
            mask = mask or bundle.name_fails(p.text.lstrip("-").split("=", 1)[0])
        return mask, (p.text.lstrip("-") if p.text is not None and p.text.startswith("-") and "=" not in p.text else None)

    def _arguments(self, sc: _Scope, cmd: Node, rec: dict) -> None:
        sym, cs_id = sc.sym_id, rec["id"]
        is_eval = rec["calleeName"] == "eval"
        name_end = cmd.child_by_field_name("name").end_byte
        argv = [c for c in cmd.named_children if c.start_byte >= name_end and c.type not in _REDIRECTS]
        dynamic, prev_flag, first_text = rec["calleeName"] == "<dynamic>" or is_eval, None, None
        for i, a in enumerate(argv[:MAX_ARGS]):
            p = self._ev(a, sc, a.start_byte, 0)
            if i == 0:
                first_text = p.text
            mask, prev_flag = self._flag_mask(p, prev_flag)
            arg = f"a{i}"
            self.records.append({"type": "valueSet", "id": ids.value_set_id(cs_id, arg), "callSiteId": cs_id,
                                 "arg": arg, "values": self._values(p.alts, mask)})
            for kind, ref, dyn in p.flows:
                if kind == "p":
                    def_id, from_path = ids.point_id(sym, "param", ref), f"p{ref}"
                else:
                    def_id, from_path = sc.cs_by_start.get(ref), "r"
                    if def_id is None:
                        continue
                dyn = dyn or is_eval
                self.records.append({
                    "type": "data", "id": ids.data_id(cs_id, def_id, arg, from_path), "use": cs_id, "def": def_id,
                    "toPath": arg, "fromPath": from_path,
                    "status": "possible" if dyn else "confirmed", "reason": "shell-dynamic" if dyn else "intra-flow"})
            dynamic = dynamic or p.dyn
        if rec["calleeName"] in ("source", ".") and (argv and first_text is None):
            dynamic = True
        if dynamic and not rec["targets"]:
            rec["reason"] = "shell-dynamic"


# --- the entry point ------------------------------------------------------------------------------------

History = Callable[[str, list[tuple[int, int]]], tuple[str, str]]


def extract_shell(unit_root: str | os.PathLike, files: Iterable[str], *, repo_id: str, unit_id: str,
                  history: History | None = None, warnings: list | None = None) -> list[dict]:
    """The records of the shell ``files`` (repo-relative POSIX paths) under ``unit_root``, the directory of the checkout
    or stage they resolve against; ``repo_id``/``unit_id`` are the unit's. ``history(path, [(lineStart, lineEnd)])`` returns
    ``(lastChangedAt, lastChangedCommit)`` for a symbol (the caller's blame, spec3 §5.3); without it the symbols carry no
    ``lastChanged*``. ``warnings``, when given, receives one dict per warning (``path``, ``code``, ``line``, ``detail``;
    codes ``parse-error``, ``parse-timeout``, ``too-deep``, ``record-cap``). Records come file by file, in no §8.3 order
    (``ccg.bundle.dumps`` sorts). Raises ``SourceRefused`` (see the module doc) and ``ValueError`` for a repeated file."""
    sink = warnings if warnings is not None else []
    rels = [_check_rel(f) for f in files]
    if len(set(rels)) != len(rels):
        raise ValueError("a file is listed twice")
    try:
        root = Path(unit_root).resolve()
    except (OSError, RuntimeError):
        raise SourceRefused("the unit root cannot be resolved") from None
    out: list[dict] = []
    for rel in sorted(rels):
        data = _read(root, rel)
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as e:
            raise SourceRefused(f"{rel!r} is not UTF-8 ({e.reason} at byte {e.start}); the unit cannot be extracted") from None
        fx = _FileExtractor(rel, data, text, repo_id=repo_id, unit_id=unit_id, history=history, warnings=sink)
        tree = _parse(data)
        if tree is None:
            fx.warn("parse-timeout", None, f"tree-sitter gave up after {PARSE_TIMEOUT_SECONDS} s")
            out += [fx.file_record(), fx.script_symbol()]
            continue
        depth, bad = _scan(tree.root_node)
        if depth > MAX_DEPTH:
            fx.warn("too-deep", None, f"syntax tree nesting {depth} is over {MAX_DEPTH}")
            out += [fx.file_record(), fx.script_symbol()]
        elif bad is not None:
            fx.warn("parse-error", bad.start_point.row + 1, "the file does not parse cleanly; no calls extracted")
            out += [fx.file_record(), fx.script_symbol()]
        else:
            out += fx.run(tree.root_node)
    return out
