"""Per-format ``keyPath`` derivation (F-6) and the bounded parsers behind it (P1-11).

Spec: `docs/plans/connected-code-graph-bundle-spec3.md` §5.17 (``configEntry``: ``format``, ``keyPath``), §1.7.4 W3;
`docs/plans/connected-code-graph-p1-extraction.md` § P1-11. Every config file is untrusted repo content: each parser is
bounded (bytes, nesting depth, node count) and builds a ``Node`` tree from the text; nothing is constructed, executed
or interpolated (YAML is *composed*, never constructed, so no tag can name a Python object). A parser raises
``FormatError`` carrying a reason code and never the text.

``FORMATS`` has one row per ``format`` value of §5.17 (``derivation`` is the row's keyPath rule, one sentence, which a test
pins). ``binding_role`` is the E3 input of `-bundle-spec5.md` §6.1 and of ``writer_rules.bind_value``/``launch_default``:
whether an entry is in a binding role (SA-3, allow-by-role).
"""
from __future__ import annotations

import json
import re
import shlex
import tomllib
import unicodedata
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from itertools import islice
from json.decoder import JSONDecodeError, scanstring

import yaml

from ccg.units import is_mcp_json

MAX_BYTES = 512 * 1024          # a larger file is skipped (counted), not parsed
MAX_DEPTH = 32                  # nesting depth of a keyPath, and of a JSON document
MAX_NODES = 50_000              # nodes visited by one walk, aliases expanded: a YAML alias bomb stops here
MAX_ENTRIES = 10_000            # configEntry records per file
MAX_KEY_CHARS = 512             # a longer keyPath segment, or a keyPath over 1024, drops its entry
MAX_DOCS = 1                    # YAML: the first document only


class FormatError(ValueError):
    """A config file that cannot be read as its format. The message is a reason code, never file text."""


# --- the tree --------------------------------------------------------------------------------------------------

class Node:
    """``kind`` ``s`` (scalar: ``text`` as the format's unescaping yields it), ``m`` (``items``: ``(key, Node, merged_from)``
    in source order) or ``l`` (``items``: Nodes). ``alias_of`` marks a YAML alias occurrence (W3). ``path`` is set when the
    walker visits the node itself, so a later alias can name its anchor's keyPath."""
    __slots__ = ("kind", "text", "items", "line", "alias_of", "path", "merges")

    def __init__(self, kind: str, line: int, text: str = "", items: list | None = None, alias_of: "Node | None" = None,
                 merges: list | None = None):
        self.kind, self.line, self.text, self.alias_of, self.path = kind, line, text, alias_of, None
        self.items = [] if items is None else items
        self.merges = [] if merges is None else merges          # YAML ``<<``: the merged maps, expanded lazily by ``map_entries``


def scalar(text: str, line: int) -> Node:
    return Node("s", line, text=text)


def mapping(line: int, items: list | None = None) -> Node:
    return Node("m", line, items=items)


def sequence(line: int, items: list | None = None) -> Node:
    return Node("l", line, items=items)


@dataclass(frozen=True)
class Leaf:
    segs: tuple                    # str (a named segment) and int (an index)
    node: Node                     # the scalar (an alias leaf: the occurrence; ``text`` is the target's)
    text: str
    line: int
    via_alias: bool
    anchor_segs: tuple | None      # the anchor's segments for a via_alias leaf, ``None`` when not known (W3 then withholds)


def map_entries(node: Node) -> Iterator[tuple]:
    """The ``(key, child, merged_from)`` entries of a map in order: each merged map's entries first (recursively, lazily: a merge
    chain is never copied), then the map's own. ``merged_from`` is the anchor map an entry came from (W3's second end)."""
    t = node.alias_of or node
    for m in t.merges:
        mt = m.alias_of or m
        for key, child, inner in map_entries(mt):
            yield key, child, inner or mt
    for key, child, _ in t.items:
        yield key, child, None


class Walk:
    """Doc-order iteration over the scalar leaves of a tree, bounded in depth and visited nodes (``visited`` counts them).
    ``truncated`` is true when a bound stopped it. It is iterative and lazy (a stack of child iterators, so a merge or alias
    expansion is never materialised): ``MAX_NODES`` bounds an alias or merge bomb."""

    def __init__(self, root: Node, max_depth: int = MAX_DEPTH, max_nodes: int = MAX_NODES):
        self._root, self._depth, self._max = root, max_depth, max_nodes
        self.truncated = False
        self.visited = 0

    def _children(self, target: Node, segs: tuple, via: bool, anchor):
        if target.kind == "m":
            for key, child, merged in map_entries(target):
                c_via, c_anchor = via, (None if anchor is None else anchor + (key,))
                if merged is not None:
                    c_via, c_anchor = True, (None if merged.path is None else merged.path + (key,))
                yield child, segs + (key,), c_via, c_anchor
        else:
            for i, child in enumerate(target.items):
                yield child, segs + (i,), via, None if anchor is None else anchor + (i,)

    def __iter__(self) -> Iterator[Leaf]:
        stack = [iter([(self._root, (), False, None)])]
        while stack:
            nxt = next(stack[-1], None)
            if nxt is None:
                stack.pop()
                continue
            node, segs, via, anchor = nxt
            self.visited += 1
            if self.visited > self._max:
                self.truncated = True
                return
            target = node
            if node.alias_of is not None:
                target, via, anchor = node.alias_of, True, node.alias_of.path
            elif node.path is None and not via:
                node.path = segs
            if target.kind == "s":
                if segs:
                    yield Leaf(segs, node, target.text, node.line, via, anchor)
                continue
            if len(segs) >= self._depth:
                self.truncated = True
                continue
            stack.append(self._children(target, segs, via, anchor))


# --- keyPath rendering and the role rule -----------------------------------------------------------------------

def norm_name(text: str) -> str:
    """NFC then strip: the form every identity string is written in (§8.1); also the form W2's declared set is built from."""
    return unicodedata.normalize("NFC", text).strip()


def _norm_segment(seg: str) -> str:
    """§8.1: ``keyPath`` is an identity field, NFC and stripped; a segment is normalised the same way before it is written."""
    return norm_name(seg)


def _plain_segment(seg: str) -> bool:
    return bool(seg) and "." not in seg and "[" not in seg and all(ord(c) >= 0x20 and ord(c) != 0x7F for c in seg)


def render_key_path(segs: tuple) -> str:
    """§5.17: segments as written, joined by ``.``; an index is ``[i]``; a segment containing ``.`` or ``[`` (or empty, or a
    control character, which a bare segment cannot carry through the schema's pattern) is quoted ``["…"]`` as JSON writes it."""
    out = []
    for s in segs:
        if isinstance(s, str):
            s = _norm_segment(s)
        if isinstance(s, int):
            out.append(f"[{s}]")
        elif _plain_segment(s):
            out.append(s if not out else "." + s)
        else:
            out.append("[" + json.dumps(s, ensure_ascii=False) + "]")
    return "".join(out)


ROLE_TOKENS = frozenset({"port", "host", "hostname", "addr", "address", "url", "uri", "endpoint", "base", "server",
                         "listen", "bind"})
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def role_tokens(segment: str) -> list[str]:
    """The segment split on ``_`` ``-`` ``.`` and at camelCase boundaries, lower-cased."""
    parts = []
    for p in re.split(r"[_.\-]", segment):
        parts += _CAMEL.sub(" ", p).split()
    return [p.lower() for p in parts]


def has_role_token(segment: str) -> bool:
    return any(t in ROLE_TOKENS for t in role_tokens(segment))


def binding_role(fmt: str, segs: tuple) -> bool:
    """E3 (SA-3, allow-by-role): the entry is in a binding role iff it is Compose ``services.*.ports[i]``/``expose[i]``,
    ``mcpServers.*.url`` of a ``.mcp.json`` or Kiro agent, or its **last** segment (a named one: an index is not a
    name) holds a role token. This only *admits* an entry to ``writer_rules``; W1-W4 and the grammar still decide."""
    if fmt == "compose" and len(segs) == 4 and segs[0] == "services" and segs[2] in ("ports", "expose") \
            and isinstance(segs[3], int):
        return True
    if fmt in ("mcp-json", "kiro-agent") and len(segs) == 3 and segs[0] == "mcpServers" and segs[2] == "url":
        return True
    return bool(segs) and isinstance(segs[-1], str) and has_role_token(segs[-1])


# --- JSON ------------------------------------------------------------------------------------------------------

_NUMBER = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?")
_WS = " \t\r\n"


def parse_json(text: str) -> Node:
    """Strict JSON with line numbers; a number is kept as written. Depth is bounded; duplicate keys stay (the last wins at
    the keyPath level); no ``NaN``/``Infinity``, comments or trailing commas."""
    s, n = text, len(text)
    pos = 0
    line = 1

    def ws() -> None:
        nonlocal pos, line
        while pos < n and s[pos] in _WS:
            if s[pos] == "\n":
                line += 1
            pos += 1

    def string() -> str:
        nonlocal pos
        try:
            val, pos = scanstring(s, pos + 1, True)
        except (JSONDecodeError, IndexError):
            raise FormatError("json-string") from None
        return val

    def value(depth: int) -> Node:
        nonlocal pos
        if depth > MAX_DEPTH:
            raise FormatError("depth")
        ws()
        if pos >= n:
            raise FormatError("json")
        c, ln = s[pos], line
        if c == "{":
            pos += 1
            items: list = []
            ws()
            if pos < n and s[pos] == "}":
                pos += 1
                return mapping(ln, items)
            while True:
                ws()
                if pos >= n or s[pos] != '"':
                    raise FormatError("json")
                key = string()
                ws()
                if pos >= n or s[pos] != ":":
                    raise FormatError("json")
                pos += 1
                items.append((key, value(depth + 1), None))
                ws()
                if pos < n and s[pos] == ",":
                    pos += 1
                    continue
                if pos < n and s[pos] == "}":
                    pos += 1
                    return mapping(ln, items)
                raise FormatError("json")
        if c == "[":
            pos += 1
            elems: list = []
            ws()
            if pos < n and s[pos] == "]":
                pos += 1
                return sequence(ln, elems)
            while True:
                elems.append(value(depth + 1))
                ws()
                if pos < n and s[pos] == ",":
                    pos += 1
                    continue
                if pos < n and s[pos] == "]":
                    pos += 1
                    return sequence(ln, elems)
                raise FormatError("json")
        if c == '"':
            return scalar(string(), ln)
        for lit in ("true", "false", "null"):
            if s.startswith(lit, pos):
                pos += len(lit)
                return scalar(lit, ln)
        m = _NUMBER.match(s, pos)
        if not m:
            raise FormatError("json")
        pos = m.end()
        return scalar(m.group(0), ln)

    root = value(0)
    ws()
    if pos != n:
        raise FormatError("json")
    return root


# --- YAML (composed, never constructed) ------------------------------------------------------------------------

class _AliasMark:
    __slots__ = ("target", "line")

    def __init__(self, target, line: int):
        self.target, self.line = target, line


class _ComposeLoader(yaml.SafeLoader):
    """Only ``get_single_node``-style composition is used. An alias occurrence is wrapped so the converter can tell it from
    the anchor's own node (W3 judges both ends)."""

    def compose_node(self, parent, index):
        if self.check_event(yaml.AliasEvent):
            line = self.peek_event().start_mark.line + 1
            return _AliasMark(super().compose_node(parent, index), line)
        return super().compose_node(parent, index)


def parse_yaml(text: str) -> Node:
    """The first document of ``text`` as a ``Node`` tree. Merge keys (``<<``) are flattened in front of the map's own keys, each
    merged entry remembering the anchor it came from; an alias occurrence is a ``Node`` whose ``alias_of`` is its anchor's."""
    loader = _ComposeLoader(text)
    try:
        try:
            loader.get_event()                                  # STREAM-START; later documents are never composed
            ynode = loader.get_node()
        finally:
            loader.dispose()
    except (yaml.YAMLError, RecursionError):
        raise FormatError("yaml") from None
    if ynode is None:
        return mapping(1)
    memo: dict[int, Node] = {}
    expanded: dict[int, int] = {}       # a map's entry count with its merges expanded (saturating checks below)
    count = 0                           # nodes built so far: bounded by MAX_NODES like the walk

    def made(node: Node) -> Node:
        nonlocal count
        count += 1
        if count > MAX_NODES:
            raise FormatError("nodes")
        return node

    def conv(y, depth: int) -> Node:
        if depth > MAX_DEPTH:
            raise FormatError("depth")
        if isinstance(y, _AliasMark):
            target = conv(y.target, depth)
            return made(Node(target.kind, y.line, text=target.text, items=target.items, alias_of=target))
        if id(y) in memo:
            return memo[id(y)]
        line = y.start_mark.line + 1
        if isinstance(y, yaml.ScalarNode):
            node = made(scalar(y.value, line))
        elif isinstance(y, yaml.SequenceNode):
            node = made(sequence(line, [conv(c, depth + 1) for c in y.value]))
        else:
            merges: list = []
            own: list = []
            for k, v in y.value:
                if isinstance(k, _AliasMark) or not isinstance(k, yaml.ScalarNode):
                    continue                                    # a complex or aliased key has no keyPath segment
                if k.tag == "tag:yaml.org,2002:merge":
                    child = conv(v, depth + 1)
                    sources = child.items if child.kind == "l" else [child]
                    merges += [src.alias_of or src for src in sources if (src.alias_of or src).kind == "m"]
                    continue
                own.append((k.value, conv(v, depth + 1), None))
            node = made(mapping(line, own))
            node.merges = merges
            size = len(own) + sum(expanded[id(m)] for m in merges)
            if size > MAX_NODES:
                raise FormatError("nodes")                      # a merge chain that would expand past the bound
            expanded[id(node)] = size
        memo[id(y)] = node
        return node

    try:
        return conv(ynode, 0)
    except RecursionError:
        raise FormatError("depth") from None


# --- TOML ------------------------------------------------------------------------------------------------------

def parse_toml(text: str) -> Node:
    """``tomllib`` for the structure (an int or datetime is written in its canonical form, a float as written), plus a lexical
    scan for the line of each key (best effort: a key it cannot place takes its table's, then 1)."""
    try:
        data = tomllib.loads(text, parse_float=str)
    except (tomllib.TOMLDecodeError, RecursionError):
        raise FormatError("toml") from None
    lines = _toml_lines(text)

    def at(segs: tuple) -> int:
        for k in range(len(segs), 0, -1):
            if segs[:k] in lines:
                return lines[segs[:k]]
        return 1

    def conv(v, segs: tuple, depth: int) -> Node:
        if depth > MAX_DEPTH:
            raise FormatError("depth")
        if isinstance(v, dict):
            return mapping(at(segs), [(k, conv(x, segs + (k,), depth + 1), None) for k, x in v.items()])
        if isinstance(v, list):
            return sequence(at(segs), [conv(x, segs + (i,), depth + 1) for i, x in enumerate(v)])
        if isinstance(v, bool):
            return scalar("true" if v else "false", at(segs))
        return scalar(str(v) if not hasattr(v, "isoformat") else v.isoformat(), at(segs))

    try:
        return conv(data, (), 0)
    except RecursionError:
        raise FormatError("depth") from None


def _toml_key(s: str, pos: int) -> tuple[list[str], int]:
    """A dotted TOML key starting at ``pos``: (segments, position after it)."""
    out: list[str] = []
    n = len(s)
    while True:
        while pos < n and s[pos] in " \t":
            pos += 1
        if pos < n and s[pos] in "\"'":
            q = s[pos]
            end = s.find(q, pos + 1)
            while q == '"' and end > 0 and s[end - 1] == "\\":
                end = s.find(q, end + 1)
            if end < 0:
                raise FormatError("toml")
            raw = s[pos + 1:end]
            out.append(raw if q == "'" else raw.encode().decode("unicode_escape", "replace") if "\\" in raw else raw)
            pos = end + 1
        else:
            m = re.compile(r"[A-Za-z0-9_-]+").match(s, pos)
            if not m:
                raise FormatError("toml")
            out.append(m.group(0))
            pos = m.end()
        while pos < n and s[pos] in " \t":
            pos += 1
        if pos < n and s[pos] == ".":
            pos += 1
            continue
        return out, pos


def _toml_lines(text: str) -> dict[tuple, int]:
    """Key path (names, with an int for each array-of-tables element) to the line it is written on."""
    lines: dict[tuple, int] = {}
    counters: dict[tuple, int] = {}
    last_idx: dict[tuple, int] = {}
    table: tuple = ()
    ml: str | None = None           # an open multi-line string's delimiter
    depth = 0                       # open ``[``/``{`` across lines

    def resolve(names: list[str], array: bool) -> tuple:
        segs: tuple = ()
        for k, name in enumerate(names):
            segs += (name,)
            raw = tuple(names[:k + 1])
            final = k == len(names) - 1
            if array and final:
                idx = counters.get(segs, -1) + 1
                counters[segs] = idx
                last_idx[raw] = idx
                segs += (idx,)
            elif raw in last_idx:
                segs += (last_idx[raw],)
        return segs

    def scan(rest: str) -> None:
        nonlocal ml, depth
        i, n = 0, len(rest)
        while i < n:
            if ml:
                j = rest.find(ml, i)
                if j < 0:
                    return
                i, ml = j + 3, None
                continue
            c = rest[i]
            if rest.startswith('"""', i) or rest.startswith("'''", i):
                ml, i = rest[i:i + 3], i + 3
            elif c in "\"'":
                j = i + 1
                while j < n and rest[j] != c:
                    j += 2 if (c == '"' and rest[j] == "\\") else 1
                i = j + 1
            elif c == "#":
                return
            else:
                depth += (c in "[{") - (c in "]}")
                i += 1

    for no, raw in enumerate(text.split("\n"), start=1):
        if ml or depth > 0:
            scan(raw)
            continue
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        try:
            if s.startswith("[["):
                names, end = _toml_key(s, 2)
                table = resolve(names, True)
                lines.setdefault(table, no)
            elif s.startswith("["):
                names, end = _toml_key(s, 1)
                table = resolve(names, False)
                lines.setdefault(table, no)
            else:
                names, end = _toml_key(s, 0)
                if end < len(s) and s[end] == "=":
                    segs = table
                    for k, name in enumerate(names):
                        segs += (name,)
                        lines.setdefault(segs, no)
                    scan(s[end + 1:])
        except FormatError:
            continue
    return lines


# --- INI -------------------------------------------------------------------------------------------------------

def parse_ini(text: str) -> Node:
    """``[section]`` and ``key = value`` / ``key: value``; ``;`` and ``#`` start a comment line; the value is the rest of the
    line, trimmed, as written (no interpolation, no inline comments). Keys before a section sit at the root."""
    root = mapping(1)
    sections: dict[str, Node] = {}
    cur = root
    for no, raw in enumerate(text.split("\n"), start=1):
        s = raw.strip()
        if not s or s[0] in ";#":
            continue
        if raw[0] in " \t" and cur is not root and cur.items:
            continue                                            # a continuation line: not a key
        if s.startswith("[") and s.endswith("]"):
            name = s[1:-1].strip()
            if name not in sections:
                sections[name] = mapping(no)
                root.items.append((name, sections[name], None))
            cur = sections[name]
            continue
        m = re.match(r"([^=:]+?)\s*[=:]\s*(.*)$", s)
        if m:
            cur.items.append((m.group(1), scalar(m.group(2).strip(), no), None))
    return root


# --- .env ------------------------------------------------------------------------------------------------------

_ENV_LINE = re.compile(r"(?:export\s+)?([A-Za-z_][A-Za-z0-9_.-]*)\s*=\s*(.*)$")


def _env_value(rest: str) -> str:
    if rest.startswith('"'):
        out, i = [], 1
        while i < len(rest) and rest[i] != '"':
            if rest[i] == "\\" and i + 1 < len(rest) and rest[i + 1] in '"\\n':
                out.append({"n": "\n"}.get(rest[i + 1], rest[i + 1]))
                i += 2
            else:
                out.append(rest[i])
                i += 1
        return "".join(out)
    if rest.startswith("'"):
        end = rest.find("'", 1)
        return rest[1:] if end < 0 else rest[1:end]
    m = re.search(r"\s#", rest)
    return (rest[:m.start()] if m else rest).strip()


def parse_env(text: str) -> Node:
    """``KEY=value`` lines (``export`` allowed); quotes are removed, an unquoted value ends at `` #``. One flat map."""
    root = mapping(1)
    for no, raw in enumerate(text.split("\n"), start=1):
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        m = _ENV_LINE.match(s)
        if m:
            root.items.append((m.group(1), scalar(_env_value(m.group(2)), no), None))
    return root


# --- Dockerfile ------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Instruction:
    name: str                      # upper-cased
    rest: str
    line: int                      # the instruction's first physical line


def dockerfile_instructions(text: str) -> list[Instruction]:
    """Instructions with ``\\`` continuations joined and comment lines dropped. Heredoc bodies are not interpreted."""
    out: list[Instruction] = []
    buf: list[str] = []
    start = 0
    for no, raw in enumerate(text.split("\n"), start=1):
        s = raw.strip()
        if buf and s.startswith("#"):
            continue
        if not buf:
            if not s or s.startswith("#"):
                continue
            start = no
        cont = s.endswith("\\")
        buf.append(s[:-1].rstrip() if cont else s)
        if not cont:
            joined = " ".join(buf)
            buf = []
            head, _, rest = joined.partition(" ")
            out.append(Instruction(head.upper(), rest.strip(), start))
    if buf:
        head, _, rest = " ".join(buf).partition(" ")
        out.append(Instruction(head.upper(), rest.strip(), start))
    return out


@dataclass
class DockerStage:
    """What the image runs: the final stage's ``ENV``, ``EXPOSE``, ``CMD`` and ``ENTRYPOINT`` (last of each), and the ``ARG``s
    with a default declared before the first ``FROM`` or in the final stage."""
    env: list[tuple[str, str, int]]
    arg: list[tuple[str, str, int]]
    expose: list[tuple[str, int]]
    cmd: Instruction | None
    entrypoint: Instruction | None


def _default_tokens(rest: str) -> list[tuple[str, str]]:
    """``ARG A=1 B C=3`` gives A and C: only a name with a default is an entry."""
    try:
        return [tuple(t.split("=", 1)) for t in shlex.split(rest, posix=True) if "=" in t]
    except ValueError:
        return []


def _assignments(rest: str) -> list[tuple[str, str]]:
    """``K=V K2="v 2"`` or the legacy ``K V``; a malformed line yields nothing."""
    try:
        toks = shlex.split(rest, posix=True)
    except ValueError:
        return []
    if toks and "=" not in toks[0]:
        key, _, value = rest.partition(" ")
        return [(key, value.strip())] if value.strip() else []
    return [tuple(t.split("=", 1)) for t in toks if "=" in t]


def dockerfile_stage(instrs: list[Instruction]) -> DockerStage:
    froms = [i for i, ins in enumerate(instrs) if ins.name == "FROM"]
    first = froms[0] if froms else len(instrs)
    final = froms[-1] if froms else -1
    st = DockerStage([], [], [], None, None)
    for idx, ins in enumerate(instrs):
        in_final = idx > final
        if ins.name == "ARG" and (idx < first or in_final):
            st.arg += [(k, v, ins.line) for k, v in _default_tokens(ins.rest)
                       if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", k)]
        elif not in_final:
            continue
        elif ins.name == "ENV":
            st.env += [(k, v, ins.line) for k, v in _assignments(ins.rest) if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", k)]
        elif ins.name == "EXPOSE":
            st.expose += [(t, ins.line) for t in ins.rest.split()]
        elif ins.name == "CMD":
            st.cmd = ins
        elif ins.name == "ENTRYPOINT":
            st.entrypoint = ins
    return st


def dockerfile_argv(ins: Instruction | None) -> list[str] | None:
    """The argv an ``ENTRYPOINT``/``CMD`` names: a JSON array of strings (exec form), else the shell form split like a POSIX
    shell would split it (a split, nothing more: no expansion). ``None`` when it cannot be split."""
    if ins is None:
        return None
    rest = ins.rest
    if rest.startswith("["):
        try:
            arr = json.loads(rest)
        except (ValueError, RecursionError):
            arr = None
        if isinstance(arr, list) and all(isinstance(a, str) for a in arr):
            return arr
    try:
        return shlex.split(rest, posix=True)
    except ValueError:
        return None


def parse_dockerfile(text: str) -> Node:
    """The derivation tree: ``ENV.<NAME>``, ``ARG.<NAME>`` (with a default), ``EXPOSE[i]`` (file order), ``entrypoint[i]`` and
    ``command[i]`` (the argv, so W1 reads the lower-case names it names)."""
    st = dockerfile_stage(dockerfile_instructions(text))
    root = mapping(1)
    if st.env:
        root.items.append(("ENV", mapping(st.env[0][2], [(k, scalar(v, ln), None) for k, v, ln in st.env]), None))
    if st.arg:
        root.items.append(("ARG", mapping(st.arg[0][2], [(k, scalar(v, ln), None) for k, v, ln in st.arg]), None))
    if st.expose:
        root.items.append(("EXPOSE", sequence(st.expose[0][1], [scalar(t, ln) for t, ln in st.expose]), None))
    for name, ins in (("entrypoint", st.entrypoint), ("command", st.cmd)):
        argv = dockerfile_argv(ins)
        if ins is not None and argv:
            root.items.append((name, sequence(ins.line, [scalar(a, ins.line) for a in argv]), None))
    return root


# --- Compose ---------------------------------------------------------------------------------------------------

def parse_compose(text: str) -> Node:
    """YAML, then list-form ``environment`` (``- K=V``) becomes a map keyed by ``K`` (the name is read from inside the value; the
    value is what follows the first ``=``). An entry with no ``=`` names a host variable and carries no value: dropped."""
    root = parse_yaml(text)
    services = child_node(root, "services")
    if services is None or services.kind != "m" or services.alias_of is not None:
        return root
    for _, svc, _ in bounded_entries(services):
        if svc.kind != "m" or svc.alias_of is not None:
            continue
        for i, (k, env, merged) in enumerate(svc.items):
            if k == "environment" and env.kind == "l" and env.alias_of is None and merged is None:
                items = []
                for el in env.items:
                    if el.kind == "s" and el.alias_of is None and "=" in el.text:
                        name, _, value = el.text.partition("=")
                        items.append((name, scalar(value, el.line), None))
                svc.items[i] = (k, mapping(env.line, items), None)
    return root


def bounded_entries(node: Node) -> list[tuple]:
    """``map_entries`` capped at ``MAX_NODES`` (a list, for callers that look keys up)."""
    return list(islice(map_entries(node), MAX_NODES))


def child_node(node: Node, key: str) -> Node | None:
    """The value of ``key`` in a map (the last occurrence, merged keys included), else ``None``."""
    if node.kind != "m":
        return None
    return next((v for k, v, _ in reversed(bounded_entries(node)) if k == key), None)


def compose_declared_services(root: Node) -> set[str]:
    """Service names and ``container_name``s declared in a Compose tree (W2's declared set)."""
    out: set[str] = set()
    services = child_node(root, "services")
    if services is None or services.kind != "m":
        return out
    for name, svc, _ in bounded_entries(services):
        out.add(norm_name(name).lower())
        if svc.kind == "m":
            out |= {norm_name(v.text).lower() for kk, v, _ in bounded_entries(svc) if kk == "container_name" and v.kind == "s"}
    return out


# --- classification and the derivation table -------------------------------------------------------------------

def _base(path: str) -> str:
    return path.rsplit("/", 1)[-1]


def is_kiro_agent(path: str) -> bool:
    return re.search(r"(?:^|/)\.kiro/agents/[^/]+\.json$", path) is not None


def is_compose(path: str) -> bool:
    return re.fullmatch(r"(?:docker-)?compose(?:\.[^/]+)?\.ya?ml", _base(path)) is not None


def is_dockerfile(path: str) -> bool:
    b = _base(path)
    return b == "Dockerfile" or b.startswith("Dockerfile.") or b.endswith(".Dockerfile")


def is_env(path: str) -> bool:
    b = _base(path)
    return b == ".env" or b.startswith(".env.") or b.endswith(".env")


@dataclass(frozen=True)
class FormatRow:
    name: str                                  # the §5.17 ``format`` value
    matches: Callable[[str], bool]
    parse: Callable[[str], Node]
    derivation: str                            # the keyPath rule of this format, one sentence


FORMATS: tuple[FormatRow, ...] = (
    FormatRow("mcp-json", is_mcp_json, parse_json,
              "JSON path from the root: `mcpServers.<server>.command`, `.args[i]`, `.env.<VAR>`, `.url`; the server's launch chain "
              "is a `command` launchFact."),
    FormatRow("kiro-agent", is_kiro_agent, parse_json,
              "JSON path from the root, as `mcp-json`: `mcpServers.<server>.url`, `tools[i]`, `prompt`."),
    FormatRow("compose", is_compose, parse_compose,
              "YAML path from the root: `services.<svc>.ports[i]`, `.expose[i]`, `.environment.<VAR>` (a list-form entry `K=V` is "
              "keyed by `K`), `.build.context` (or `.build` in the short form), `.command[i]`."),
    FormatRow("dockerfile", is_dockerfile, parse_dockerfile,
              "`ENV.<NAME>`, `ARG.<NAME>` (with a default), `EXPOSE[i]` (file order), `entrypoint[i]` and `command[i]` (the argv of "
              "the last ENTRYPOINT/CMD of the final stage; shell form is split like a POSIX shell)."),
    FormatRow("env", is_env, parse_env, "`<KEY>`: one flat map; the last duplicate wins."),
    FormatRow("json", lambda p: p.lower().endswith(".json"), parse_json,
              "JSON path from the root; array elements are `[i]`."),
    FormatRow("yaml", lambda p: p.lower().endswith((".yaml", ".yml")), parse_yaml,
              "YAML path from the root; a merge key's entries sit under the merging map; array elements are `[i]`."),
    FormatRow("toml", lambda p: p.lower().endswith(".toml"), parse_toml,
              "TOML path from the root: `[a.b]` gives `a.b.<key>`, `[[a]]` gives `a[i].<key>`."),
    FormatRow("ini", lambda p: p.lower().endswith((".ini", ".cfg")), parse_ini,
              "`<section>.<key>`; a key before any section is `<key>`."),
)


def classify(path: str) -> FormatRow | None:
    """The first row of ``FORMATS`` whose ``matches`` accepts ``path`` (specific names precede the extension rows)."""
    return next((row for row in FORMATS if row.matches(path)), None)
