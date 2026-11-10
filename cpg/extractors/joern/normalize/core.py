"""Language-neutral normalizer core: staged-source access, the scope tree, ids' inputs, name and path rules, value sets.

Ported from ``cpg/spikes/vertical-slice/normalize.py`` (equivalence before improvement: `docs/plans/connected-code-graph-p1.md` §8 and D7): the bodies are the
slice's, the argparse namespace the slice's ``Builder`` took is replaced by explicit arguments.
Rules: ``docs/plans/connected-code-graph-bundle-spec3.md`` §4-§5 read with ``-spec4.md`` and ``-spec2.md`` §1.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

from ccg import ids

IDENT = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*\Z")
FIELD_CAP = 3


class RuntimeRefused(RuntimeError):
    """The normalizer runs only under CPython 3.12 (spec4 §3, host-Python note)."""


def require_runtime(version_info=None, implementation: str | None = None) -> None:
    """Raise ``RuntimeRefused`` unless the interpreter is CPython 3.12 (arguments default to the running one)."""
    vi = tuple(sys.version_info[:2]) if version_info is None else tuple(version_info[:2])
    impl = sys.implementation.name if implementation is None else implementation
    if impl != "cpython" or vi != (3, 12):
        raise RuntimeRefused(f"the normalizer needs CPython 3.12, found {impl} {vi[0]}.{vi[1]}")


# -- small helpers ---------------------------------------------------------------------------
def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def collapse(s: str, limit: int = 200) -> str:
    return re.sub(r"\s+", " ", s).strip()[:limit].rstrip()


def name_norm(name: str) -> str:
    """-spec2 §1.2: NFC(casefold(NFC(name))).strip()."""
    n = unicodedata.normalize("NFC", name)
    return unicodedata.normalize("NFC", n.casefold()).strip()


def quote_field(key: str) -> str:
    """-spec2 §1.1: an ASCII IDENT is written ``.key``; every other key is ``["..."]``."""
    if IDENT.match(key):
        return "." + key
    return "[" + json.dumps(key, ensure_ascii=False) + "]"


def path_from(head: str, fields: list[str]) -> tuple[str, bool]:
    """Head + fields -> (path string, truncated).  Depth cap counts fields after the head
    (-spec2 §1.1); a ``kw.<name>`` head's own field is not counted."""
    fields = list(fields)
    trunc = len(fields) > FIELD_CAP
    if trunc:
        fields = fields[:FIELD_CAP]
    return head + "".join(quote_field(f) for f in fields), trunc


def kw_head(name: str) -> str:
    return "kw" + quote_field(name)


class Source:
    """The staged unit's files: bytes, text, lines (split on \\n only)."""

    def __init__(self, stage: Path):
        self.stage = Path(stage)
        self._t: dict[str, str] = {}

    def text(self, rel: str) -> str:
        if rel not in self._t:
            self._t[rel] = (self.stage / rel).read_bytes().decode("utf-8")
        return self._t[rel]

    def lines(self, rel: str) -> list[str]:
        return self.text(rel).split("\n")

    def exists(self, rel: str) -> bool:
        return (self.stage / rel).is_file()


# -- scope tree ------------------------------------------------------------------------------
class Scope:
    __slots__ = ("key", "kind", "name", "line", "col", "lineEnd", "parent", "file", "cpg_ids",
                 "base", "dup", "qname", "sid", "accessor", "jname", "is_type", "decos",
                 "synthetic_alias", "node")

    def __init__(self, **kw):
        self.accessor = None
        self.dup = 1
        self.decos = []
        self.cpg_ids = []   # CPG METHOD ids whose own code belongs to this symbol
        self.qname = None
        self.sid = None
        self.is_type = False
        for k, v in kw.items():
            setattr(self, k, v)


def seg(scope: Scope) -> str:
    """One qname segment (spec §4.1): base [. accessor] [#N]; an <anon> always carries its #N."""
    base = scope.base + (("." + scope.accessor) if scope.accessor else "")
    if scope.base == "<anon>":
        return f"<anon>#{scope.dup}"
    return base + (f"#{scope.dup}" if scope.dup > 1 else "")


class Builder:
    """Per-unit state: the raw facts by kind, the staged source, the unit's identity."""

    def __init__(self, lang: str, unit_root: str, repo_id: str, facts, src: Source):
        require_runtime()    # for every language, deliberately: fail-closed, and the project pins its interpreter
        self.facts = facts
        self.src = src
        self.lang = lang
        self.root = unit_root
        self.repo = repo_id
        self.unit_id = ids.unit_id(repo_id, unit_root, lang)
        self.col_base = 1 if lang == "python" else 0
        self.records: list[dict] = []
        self.report: dict = {"notes": []}
        self.span_cache: dict[str, dict] = {}

    def P(self, rel: str) -> str:
        """Joern parse-root-relative -> repo-relative POSIX path (spec §8.2)."""
        return rel if self.root == "." else f"{self.root}/{rel}"


# -- reading facts ---------------------------------------------------------------------------
def load_facts(path: Path):
    by = defaultdict(list)
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            by[o["k"]].append(o)
    return by


def lang_files(facts, lang):
    exts = (".py",) if lang == "python" else (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".mts", ".cts")
    return sorted({f["name"] for f in facts["f"] if f["name"].endswith(exts)})


def number_and_qualify(B: Builder, scopes: dict) -> None:
    """Duplicate numbering (same parent chain, base and accessor: by position, then CPG id) and qnames, in place.

    The ``sort`` is what makes the numbering independent of the order Joern emitted the facts in."""
    groups = defaultdict(list)
    for s in scopes.values():
        if s.kind == "module":
            continue
        groups[(id(s.parent), s.base, s.accessor)].append(s)
    for (_, base, acc), grp in groups.items():
        grp.sort(key=lambda x: (x.line, x.col, x.node["id"] if "id" in x.node else ""))
        for i, s in enumerate(grp, start=1):
            s.dup = i

    def chain(s: Scope) -> str:
        parts = []
        cur = s
        while cur is not None and cur.kind != "module":
            parts.append(seg(cur))
            cur = cur.parent
        return ".".join(reversed(parts))

    for s in scopes.values():
        p = B.P(s.file)
        s.qname = f"{p}::{s.base}" if s.kind == "module" else f"{p}::{chain(s)}"


def sym_name(s: Scope) -> str:
    """Spec §4.1 rule 10: the last segment's base, without #N or an accessor suffix; for ``<init>`` the
    type's name."""
    if s.base == "<init>" and s.parent is not None:
        return s.parent.base
    return s.base


def span_start(B: Builder, s: Scope) -> int:
    start = s.line
    for _, d in s.decos:
        start = min(start, d["l"])
    return start


def fingerprint(B: Builder, s: Scope) -> str:
    """sha256 of the symbol's non-blank lines, stripped, joined by \\n (the span is decorator-inclusive)."""
    ls = span_start(B, s)
    le = max(s.lineEnd, ls)
    lines = B.src.lines(s.file)
    text = "\n".join(x for x in (l.strip() for l in lines[ls - 1:le]) if x)
    return sha256(text.encode("utf-8"))


def import_symbol_id(B: Builder, scopes: dict, file: str, line: int):
    """The ``symbolId`` of an import record: the symbol of ``file`` (never a module or a type) whose span holds ``line``;
    among several, the slice's rule, the widest span (``span_start - lineEnd`` ascending). ``None`` when there is none."""
    inner = [s for s in scopes.values() if s.file == file and s.kind not in ("module", "type")
             and span_start(B, s) <= line <= s.lineEnd]
    if not inner:
        return None
    inner.sort(key=lambda s: (span_start(B, s) - s.lineEnd))
    return inner[0].sid


# -- value sets ------------------------------------------------------------------------------
class Resolver:
    """Imports and constants -> {ref}/{importRef}; element trees -> bundle elements."""

    def __init__(self, B: Builder, imports_by_file: dict[str, list[dict]], constants: dict[str, set[str]]):
        self.B = B
        self.imports = imports_by_file
        self.constants = constants   # path -> {qname}
        self.alias: dict[str, dict[str, tuple]] = defaultdict(dict)  # file -> name -> (module, name, resolved)
        for f, recs in imports_by_file.items():
            for r in recs:
                for n in r["names"]:
                    if n["name"] == "*":
                        continue
                    local = n.get("alias", n["name"])
                    self.alias[f][local] = (r["module"], n["name"], r.get("_sub"), r.get("resolvedPath"))

    def lit(self, code: str):
        try:
            v = ast.literal_eval(code)
        except Exception:
            m = re.fullmatch(r"`([^`$\\]*)`", code)
            if m:
                return {"const": m.group(1)}
            return None
        if isinstance(v, bool):
            return {"const": v}
        if isinstance(v, int):
            return {"const": v}
        if isinstance(v, str):
            return {"const": unicodedata.normalize("NFC", v)}
        return None

    def element(self, e: dict, file: str, depth: int = 1):
        o = e["o"]
        if depth > 4:
            return {"unknown": "value-set-cap"}
        if o == "lit":
            r = self.lit(e["c"])
            return r if r is not None else {"unknown": "runtime-built"}
        if o == "name":
            n = e["n"]
            path = self.B.P(file)
            if f"{path}::{n}" in self.constants.get(path, set()):
                return {"ref": f"{path}::{n}"}
            al = self.alias.get(file, {}).get(n)
            if al:
                module, orig, sub, resolved = al
                if sub:   # `from pkg import submodule` -> the module file; a bare name is the module
                    return {"importRef": {"module": module, "name": orig}}
                if resolved:
                    return {"ref": f"{resolved}::{orig}"}
                return {"importRef": {"module": module, "name": orig}}
            if n in ("undefined", "None", "null", "NaN"):
                return {"unknown": "runtime-built"}
            return {"unknown": "point-derived"}
        if o == "attr":
            b = e["b"]
            if b["o"] == "name":
                al = self.alias.get(file, {}).get(b["n"])
                if al:
                    module, orig, sub, resolved = al
                    if sub:
                        return {"ref": f"{sub}::{e['f']}"}
                    if resolved and not sub:
                        return {"ref": f"{resolved}::{orig}.{e['f']}"}
                    return {"importRef": {"module": module, "name": f"{orig}.{e['f']}"}}
            return {"unknown": "point-derived"}
        if o in ("fmt", "add"):
            parts = [self.element(p, file, depth + 1) for p in e["p"]]
            parts = [p for p in parts if p != {"const": ""}]
            if not parts:
                return {"const": ""}
            if depth >= 3 and any("template" in p or "concat" in p for p in parts):
                return {"unknown": "value-set-cap"}
            return {"template" if o == "fmt" else "concat": parts}
        if o == "call":
            return {"unknown": "point-derived"}
        return {"unknown": "runtime-built"}

    def values(self, e: dict, file: str):
        return [self.element(e, file)]


def only_unknown(el) -> bool:
    return "unknown" in el
