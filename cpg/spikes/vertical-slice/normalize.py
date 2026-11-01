#!/usr/bin/env python3
"""normalize.py -- S11 (P0-10): extract.sc facts -> a bundle-v1 JSONL file.

Contract: docs/plans/connected-code-graph-p0-2.md S11 row and §4.4; bundle spec v1.3 plus
-spec2 v1.3 (``docs/plans/connected-code-graph-bundle-spec.md``, ``...-bundle-spec2.md``).
Runs on the HOST, outside the sandbox: it reads the facts file extract.sc wrote, the staged
source (for call-site positions, fingerprints, imports) and the source repository's history
(``git blame``, bundle spec BF12). It writes through ``ccg.bundle.write_bundle`` so the file is
sorted, canonical and validated on the way out.

What it owns (extract.sc emits CPG facts only):
  * path rewriting (Joern's parse-root-relative FILENAME + the unit's rootPath, spec §8.2);
  * the qname grammar (spec §4.1/§4.2) -- computed from the CPG scope tree, NEVER from
    Joern's FULL_NAME by string surgery (spec §4.0); ``verify_qnames.py`` re-derives every qname
    from the source alone and compares;
  * ids (ccg.ids), ordinals, nameNorm (-spec2 §1.2), ASCII-IDENT quoting (-spec2 §1.1),
    import.ordinal with repeated names de-duplicated (-spec2 §1.5);
  * cond->cond controlledBy from the CDG and G-4: self-edges dropped and (point, cond) pairs
    de-duplicated BEFORE writing (-spec2 §1.4), with the raw/dropped counts reported;
  * lastChanged* from ``git blame --porcelain <builtCommit>`` (outside the sandbox, BF12).
"""
from __future__ import annotations

import argparse
import ast
import bisect
import hashlib
import json
import re
import subprocess
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # cpg/
from ccg import bundle, ids  # noqa: E402

NORMALIZER = "ccg-slice-normalize"
NORMALIZER_VERSION = "0.1"
DEFAULT_VENDORED = ["_vendor", "node_modules", "third-party", "third_party", "vendor", "vendored"]
IDENT = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*\Z")
FIELD_CAP = 3
PY_SYNTHETIC = ("<body>", "<metaClassCallHandler>", "<fakeNew>")


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


def int_(v) -> int:
    return int(v)


class Source:
    """The staged unit's files: bytes, text, lines (split on \\n only)."""

    def __init__(self, stage: Path):
        self.stage = stage
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
    def __init__(self, a, facts, src: Source):
        self.a = a
        self.facts = facts
        self.src = src
        self.lang = a.lang
        self.root = a.unit_root
        self.repo = a.repo_id
        self.unit_id = ids.unit_id(self.repo, self.root, self.lang)
        self.col_base = 1 if self.lang == "python" else 0
        self.records: list[dict] = []
        self.report: dict = {"notes": []}

    def P(self, rel: str) -> str:
        """Joern parse-root-relative -> repo-relative POSIX path (spec §8.2)."""
        return rel if self.root == "." else f"{self.root}/{rel}"


# -- reading facts ---------------------------------------------------------------------------
def load_facts(path: Path):
    by = defaultdict(list)
    with path.open(encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            by[o["k"]].append(o)
    return by


def lang_files(facts, lang):
    exts = (".py",) if lang == "python" else (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".mts", ".cts")
    return sorted({f["name"] for f in facts["f"] if f["name"].endswith(exts)})


# -- Python/JS symbol construction -----------------------------------------------------------
def build_scopes(B: Builder):
    """Return (scopes, method_to_scope).  One Scope per symbol.  Parent links follow the CPG's
    AST parent chain; the qname chain is built from names, not from FULL_NAME."""
    facts, lang = B.facts, B.lang
    files = set(lang_files(facts, lang))
    methods = {m["id"]: m for m in facts["m"] if m["file"] in files}
    types = {t["id"]: t for t in facts["t"] if t["file"] in files}
    deco_by_target: dict[str, list[dict]] = defaultdict(list)
    for d in facts["deco"]:
        deco_by_target[d["target"]].append(d)
    scopes: dict[tuple, Scope] = {}
    m2s: dict[str, Scope] = {}

    def lines(file):
        return B.src.lines(file)

    # file-level scopes
    for mid, m in methods.items():
        if (lang == "python" and m["name"] == "<module>") or (lang == "javascript" and m["name"] == ":program"):
            kind = "module"
            s = Scope(key=("m", mid), kind=kind, name="<module>", base="<module>", line=1, col=1,
                      lineEnd=len(B.src.text(m["file"]).splitlines()) or 1, parent=None, file=m["file"],
                      jname="<module>", node=m)
            s.cpg_ids.append(mid)
            scopes[s.key] = s
            m2s[mid] = s

    def is_class_py(t):
        return (t["full"] + ".<body>") in body_fulls

    body_fulls = {}
    if lang == "python":
        for mid, m in methods.items():
            if m["name"] == "<body>":
                body_fulls[m["full"]] = mid

    # type scopes
    if lang == "python":
        for tid, t in types.items():
            bid = body_fulls.get(t["full"] + ".<body>")
            if t["name"].endswith("<meta>") or bid is None or t["name"] in PY_SYNTHETIC:
                continue
            bm = methods[bid]
            s = Scope(key=("t", tid), kind="type", name=t["name"], base=t["name"], line=bm["l"], col=bm["c"],
                      lineEnd=bm["le"], parent=None, file=t["file"], jname=t["name"], is_type=True, node=t)
            s.cpg_ids.append(bid)
            scopes[s.key] = s
            m2s[bid] = s
    else:
        # JS/TS: a TYPE_DECL is a type iff no METHOD shares its fullName and it is not an <init>.
        method_fulls = {m["full"] for m in methods.values()}
        for tid, t in types.items():
            if t["full"] in method_fulls or t["name"] == "<init>" or t["name"].startswith("<lambda>"):
                continue
            if t["name"] == ":program":
                continue
            if t["name"].startswith("<anon-class>"):
                continue   # Joern's object-type-literal artifacts; a source anonymous class is a ClassExpression
            anon = False
            base = t["name"]
            s = Scope(key=("t", tid), kind="type", name=t["name"], base=base, line=t["l"], col=t["c"],
                      lineEnd=t["l"], parent=None, file=t["file"], jname=t["name"], is_type=True, node=t)
            scopes[s.key] = s

    # method scopes
    for mid, m in methods.items():
        if mid in m2s:
            continue
        name = m["name"]
        if lang == "python":
            if name in PY_SYNTHETIC or name.endswith("<metaClassAdapter>"):
                continue
        else:
            if name == "<init>":
                # an explicit constructor is a symbol; the synthetic one's code belongs to the type
                ln = lines(m["file"])[m["l"] - 1] if 1 <= m["l"] <= len(lines(m["file"])) else ""
                if "constructor" not in ln:
                    continue
        anon = name.startswith("<lambda>")
        base = "<anon>" if anon else ("<init>" if name == "<init>" else name)
        s = Scope(key=("m", mid), kind="lambda" if anon else "function", name=name, base=base,
                  line=m["l"], col=m["c"], lineEnd=m["le"], parent=None, file=m["file"], jname=name, node=m)
        s.cpg_ids.append(mid)
        scopes[s.key] = s
        m2s[mid] = s

    # parents
    for s in scopes.values():
        if s.kind == "module":
            continue
        n = s.node
        pk, pid = n["pk"], n["pid"]
        parent = None
        if pk == "METHOD":
            parent = m2s.get(pid)
            if parent is None and lang == "python" and methods.get(pid, {}).get("name") in PY_SYNTHETIC:
                parent = None
        elif pk == "TYPE_DECL":
            parent = scopes.get(("t", pid))
            if parent is None:
                # JS: the TYPE_DECL parent may be a function's or <init>'s TYPE_DECL; climb to the
                # method with the same fullName, else the type's parent chain.
                tpar = types.get(pid)
                if tpar is not None:
                    for m in methods.values():
                        if m["full"] == tpar["full"] and m["id"] in m2s:
                            parent = m2s[m["id"]]
                            break
        if parent is None and pk == "METHOD":
            # a method nested in a synthetic method (Python class body)
            pm = methods.get(pid)
            if pm is not None:
                par2 = None
                if pm["pk"] == "TYPE_DECL":
                    par2 = scopes.get(("t", pm["pid"]))
                parent = par2
        if parent is None:
            parent = next((x for x in scopes.values() if x.kind == "module" and x.file == s.file), None)
        s.parent = parent

    # JS/TS: attribute the synthetic <init> methods' code to the type
    if lang == "javascript":
        for mid, m in methods.items():
            if m["name"] == "<init>" and mid not in m2s:
                tdecl = None
                if m["pk"] == "TYPE_DECL":
                    tdecl = scopes.get(("t", m["pid"]))
                if tdecl is not None:
                    tdecl.cpg_ids.append(mid)
                    m2s[mid] = tdecl

    # python property accessors, overload stubs
    if lang == "python":
        for s in list(scopes.values()):
            if s.kind != "function":
                continue
            ds = deco_by_target.get(s.node["full"], [])
            names = []
            for d in ds:
                nm = decorator_name(B, s.file, d)
                names.append((nm, d))
            s.decos = names
            for nm, d in names:
                if nm in ("property",):
                    s.accessor = "<get>"
                elif nm == f"{s.base}.setter":
                    s.accessor = "<set>"
                elif nm == f"{s.base}.deleter":
                    s.accessor = "<del>"
            if any(nm in ("overload", "typing.overload") for nm, _ in names):
                s.kind = "OVERLOAD"
        for k in [k for k, v in scopes.items() if v.kind == "OVERLOAD"]:
            s = scopes.pop(k)
            for mid in s.cpg_ids:
                m2s.pop(mid, None)
        for s in scopes.values():
            if s.accessor:
                s.kind = "accessor"
            elif s.kind == "function" and s.parent is not None and s.parent.is_type:
                s.kind = "method"
    else:
        for s in scopes.values():
            if s.kind == "function" and s.parent is not None and s.parent.is_type:
                s.kind = "method"
            if s.base == "<init>":
                s.kind = "constructor"

    # duplicates and anonymous numbering: same parent chain
    groups = defaultdict(list)
    for s in scopes.values():
        if s.kind == "module":
            continue
        groups[(id(s.parent), s.base, s.accessor)].append(s)
    for (_, base, acc), grp in groups.items():
        grp.sort(key=lambda x: (x.line, x.col, x.node["id"] if "id" in x.node else ""))
        for i, s in enumerate(grp, start=1):
            s.dup = i
    # qnames
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
    return scopes, m2s, deco_by_target


def decorator_name(B: Builder, file: str, d: dict) -> str:
    """The decorator expression without its argument list, as written, whitespace removed."""
    lines = B.src.lines(file)
    ln, c0 = d["l"], d["c"] - B.col_base
    text = lines[ln - 1][c0:] if 1 <= ln <= len(lines) else ""
    m = re.match(r"[A-Za-z_][A-Za-z0-9_.]*", text)
    if not m:
        return d["name"]
    return m.group(0)


def py_spans(B: Builder, file: str):
    """(name, line) -> (lineStart incl. decorators, lineEnd) from the stdlib ``ast``.  pysrc2cpg's
    LINE_NUMBER_END overshoots (often by 2 lines: trailing blank/comment lines), so a Python symbol's end
    line is taken from the source parse; names, nesting and qnames still come from the CPG."""
    cache = B.__dict__.setdefault("_pyspan", {})
    if file not in cache:
        out = {}
        tree = ast.parse(B.src.text(file))
        for n in ast.walk(tree):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                st = min([d.lineno for d in n.decorator_list] + [n.lineno])
                out[(n.name, n.lineno)] = (st, n.end_lineno)
            elif isinstance(n, ast.Lambda):
                out[("<lambda>", n.lineno, n.col_offset + 1)] = (n.lineno, n.end_lineno)
        cache[file] = out
    return cache[file]


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
                    target = sub or resolved
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


# -- imports (source-derived; Joern emits duplicated IMPORT nodes) ---------------------------
def py_resolve(B: Builder, importer: str, module: str, names: list[str]):
    """Resolve a Python import specifier to a file in the staged unit.  Returns
    (resolvedPath or None, {name: submodule file})."""
    src = B.src
    level = len(module) - len(module.lstrip("."))
    mod = module.lstrip(".")
    base = Path(importer).parent
    if level:
        for _ in range(level - 1):
            base = base.parent
    else:
        base = Path(".")
    parts = [p for p in mod.split(".") if p]
    d = base.joinpath(*parts) if parts else base
    cands = [f"{d}.py", f"{d}/__init__.py"] if parts else [f"{d}/__init__.py"]
    res = next((c.lstrip("./") if c.startswith("./") else c for c in cands if src.exists(c)), None)
    subs = {}
    for n in names:
        for c in (f"{d}/{n}.py", f"{d}/{n}/__init__.py"):
            if src.exists(c):
                subs[n] = B.P(c)
                break
    return (B.P(res) if res else None), subs


def py_imports(B: Builder, file: str, scopes_in_file: list[Scope]):
    text = B.src.text(file)
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for al in node.names:
                out.append(dict(line=node.lineno, col=node.col_offset, module=al.name, names=[], dynamic=False,
                                resolvedPath=None, _sub=None))
        elif isinstance(node, ast.ImportFrom):
            module = "." * node.level + (node.module or "")
            nm = [(a.name, a.asname) for a in node.names]
            resolved, subs = py_resolve(B, file, module, [n for n, _ in nm if n != "*"])
            names = []
            seen = set()
            for n, asn in nm:
                key = (n, asn or "")
                if key in seen:
                    continue
                seen.add(key)
                d = {"name": n}
                if asn:
                    d["alias"] = asn
                names.append(d)
            names.sort(key=lambda x: (x["name"], 0 if "alias" not in x else 1, x.get("alias", "")))
            rec = dict(line=node.lineno, col=node.col_offset, module=module, names=names, dynamic=False,
                       resolvedPath=resolved, _sub=None)
            rec["_subs"] = subs
            out.append(rec)
    out.sort(key=lambda r: (r["line"], r["col"]))
    return out


# -- JS/TS imports and type spans from astgen's Babel AST ------------------------------------
JS_EXTS = (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".mts", ".cts")


def js_resolve(B: Builder, importer: str, spec: str):
    if not spec.startswith("."):
        return None
    base = Path(importer).parent
    p = (base / spec)
    norm = Path(*[x for x in p.parts if x != "."])  # may contain ".."
    parts = []
    for x in norm.parts:
        if x == "..":
            if not parts:
                return None
            parts.pop()
        else:
            parts.append(x)
    d = "/".join(parts)
    for cand in [d] + [d + e for e in JS_EXTS] + [d + "/index" + e for e in JS_EXTS]:
        if B.src.exists(cand) and cand.endswith(JS_EXTS):
            return B.P(cand)
    return None


def babel(astgen_dir: Path, file: str):
    p = astgen_dir / (file + ".json")
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))["ast"]


def walk_babel(node):
    stack = [node]
    while stack:
        n = stack.pop()
        if isinstance(n, dict):
            if "type" in n:
                yield n
            for v in n.values():
                if isinstance(v, (dict, list)):
                    stack.append(v)
        elif isinstance(n, list):
            stack.extend(n)


def js_imports(B: Builder, file: str, ast_):
    out = []
    for n in walk_babel(ast_):
        t = n.get("type")
        if t == "ImportDeclaration":
            spec = n["source"]["value"]
            names, seen = [], set()
            for sp in n.get("specifiers", []):
                st = sp["type"]
                if st == "ImportSpecifier":
                    nm = sp["imported"].get("name") or sp["imported"].get("value")
                    loc = sp["local"]["name"]
                    d = {"name": nm}
                    if loc != nm:
                        d["alias"] = loc
                elif st == "ImportDefaultSpecifier":
                    d = {"name": "default", "alias": sp["local"]["name"]}
                else:
                    d = {"name": "*", "alias": sp["local"]["name"]}
                key = (d["name"], d.get("alias", ""))
                if key not in seen:
                    seen.add(key)
                    names.append(d)
            names.sort(key=lambda x: (x["name"], 0 if "alias" not in x else 1, x.get("alias", "")))
            out.append(dict(line=n["loc"]["start"]["line"], col=n["loc"]["start"]["column"], module=spec, names=names,
                            dynamic=False, resolvedPath=js_resolve(B, file, spec), _sub=None, _subs={}))
        elif t == "CallExpression" and n.get("callee", {}).get("type") == "Import":
            a0 = (n.get("arguments") or [{}])[0]
            lit = a0.get("type") in ("StringLiteral",)
            spec = a0.get("value") if lit else "<dynamic>"
            out.append(dict(line=n["loc"]["start"]["line"], col=n["loc"]["start"]["column"], module=spec, names=[],
                            dynamic=True, resolvedPath=js_resolve(B, file, spec) if lit else None, _sub=None, _subs={}))
    out.sort(key=lambda r: (r["line"], r["col"]))
    return out


TYPE_NODES = {"TSInterfaceDeclaration", "TSTypeAliasDeclaration", "TSEnumDeclaration", "ClassDeclaration",
              "ClassExpression", "TSModuleDeclaration"}


def utf16_line_index(text: str):
    """Sorted UTF-16 offsets of every ``\n`` (Babel's ``start``/``end`` are UTF-16 indexes)."""
    out, off = [], 0
    for ch in text:
        if ch == "\n":
            out.append(off)
        off += 2 if ord(ch) > 0xFFFF else 1
    return out


def end_line(nl: list[int], end_index: int) -> int:
    """1-based line holding the last character of a node whose exclusive end index is given."""
    return bisect.bisect_left(nl, end_index - 1) + 1


def js_type_spans(ast_, text: str):
    """name -> list of (startLine, endLine, nodeType) for named type declarations.  astgen's
    nested nodes carry loc.start only; the end comes from the node's ``end`` index."""
    nl = utf16_line_index(text)
    out = defaultdict(list)
    for n in walk_babel(ast_):
        if n.get("type") in TYPE_NODES and n.get("id") and n["id"].get("name"):
            out[n["id"]["name"]].append((n["loc"]["start"]["line"], end_line(nl, n["end"]), n["type"]))
    return out


# -- git blame (outside the sandbox, BF12) ----------------------------------------------------
def blame(repo: Path, commit: str, repo_path: str):
    """Per-line [(committer epoch, commit)] for a file at ``commit``."""
    cp = subprocess.run(["git", "-C", str(repo), "blame", "--porcelain", commit, "--", repo_path],
                        capture_output=True, check=True)
    lines = cp.stdout.decode("utf-8", "surrogateescape").split("\n")
    times: dict[str, int] = {}
    per_line: dict[int, str] = {}
    i, cur = 0, None
    hdr = re.compile(r"^([0-9a-f]{40}) \d+ (\d+)(?: \d+)?$")
    while i < len(lines):
        m = hdr.match(lines[i])
        if m:
            cur = m.group(1)
            per_line[int(m.group(2))] = cur
            i += 1
            while i < len(lines) and not lines[i].startswith("\t"):
                if lines[i].startswith("committer-time "):
                    times[cur] = int(lines[i].split()[1])
                i += 1
        i += 1
    return per_line, times


def last_changed(per_line, times, regions):
    best = None
    for a, b in regions:
        for ln in range(a, b + 1):
            c = per_line.get(ln)
            if c is None:
                continue
            key = (-times[c], c)   # latest time; ties -> the lower commit id
            if best is None or key < best:
                best = key
    if best is None:
        raise ValueError("no blame for span")
    t, c = -best[0], best[1]
    return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), c


# -- call-site positions ---------------------------------------------------------------------
def callee_name(c: dict) -> str:
    if c["name"] == "<operator>.new":
        m = re.match(r"\s*new\s+([A-Za-z_$][\w$.]*)", c["code"])
        return m.group(1).split(".")[-1] if m else "<dynamic>"
    return c["name"] or "<dynamic>"


def locate_callees(B: Builder, file: str, calls: list[dict]):
    """(line, col) of each call's callee last identifier (spec §5.7), derived from source.  A call whose
    callee name cannot be found in the source is absent from the result (synthetic: see below)."""
    lines = B.src.lines(file)
    starts: dict[tuple, list[dict]] = defaultdict(list)
    for c in calls:
        starts[(c["l"], c["c"], callee_name(c))].append(c)
    pos = {}
    for (ln, cl, name), grp in starts.items():
        grp.sort(key=lambda c: (len(c["code"]), c["id"]))
        off = cl - B.col_base
        # the text from the call start to the end of the file, restricted to a window
        text = "\n".join(lines[ln - 1: ln + 400])
        if off < 0 or off > len(lines[ln - 1]):
            off = 0
        text = text[off:]
        pat = re.compile(r"(?<![\w$])" + re.escape(name) + r"(?![\w$])" +
                         ("" if grp[0]["name"] == "<operator>.new" else r"\s*(?:\?\.)?\s*(?:<[^()]*?>)?\s*\("))
        hits = [m for m in pat.finditer(text)]
        for k, c in enumerate(grp):
            if k < len(hits):
                before = text[:hits[k].start()]
                dl = before.count("\n")
                col = (len(before) - (before.rfind("\n") + 1)) + (off if dl == 0 else 0)
                pos[c["id"]] = (ln + dl, col)
            # else: the callee name never occurs in the source at/after the call's start -- a call the
            # frontend synthesized (Python `__iter__`/`__next__`/comprehension `append`, JS `require`
            # for imports, `__ecma.Array.factory`); not a source-level call, so not a call site (§5.6)
    return pos


# -- assembly ---------------------------------------------------------------------------------
def g4_filter(edges, mutant=None):
    """-spec2 §1.4 (G-4): drop CDG self-edges (``point == cond``) and de-duplicate (point, cond) pairs, in
    first-seen order.  Returns (kept, self_dropped, duplicates_dropped).  ``mutant`` = "self" or "dedup"
    deliberately KEEPS the named kind (verification only: the validator must then reject the bundle)."""
    kept, seen, n_self, n_dup = [], set(), 0, 0
    for point, cond in edges:
        if point == cond and mutant != "self":
            n_self += 1
            continue
        if (point, cond) in seen and mutant != "dedup":
            n_dup += 1
            continue
        seen.add((point, cond))
        kept.append((point, cond))
    return kept, n_self, n_dup


def file_role(rel: str) -> str:
    base = rel.rsplit("/", 1)[-1].lower()
    parts = rel.lower().split("/")
    if (re.search(r"\.example", base) or re.search(r"\.sample", base) or base.startswith("example")
            or base.startswith("sample") or "examples" in parts[:-1] or "samples" in parts[:-1]):
        return "example"
    return "primary"


def is_vendored(B: Builder, rel: str) -> bool:
    return any(s in DEFAULT_VENDORED for s in rel.split("/")[:-1])


def assemble(a, facts) -> tuple[list[dict], dict]:
    B = Builder(a, facts, Source(Path(a.stage)))
    lang = a.lang
    files = [f for f in lang_files(facts, lang) if B.src.exists(f) and not is_vendored(B, f)]
    fileset = set(files)
    scopes, m2s, deco_by_target = build_scopes(B)
    scopes = {k: s for k, s in scopes.items() if s.file in fileset}
    m2s = {k: s for k, s in m2s.items() if s.file in fileset}
    astgen_dir = Path(a.astgen) if a.astgen else None
    babel_cache = {f: babel(astgen_dir, f) for f in files} if (lang == "javascript" and astgen_dir) else {}

    # Python spans from the source parse (pysrc2cpg's end lines overshoot)
    if lang == "python":
        miss = 0
        for s in scopes.values():
            if s.kind == "module":
                continue
            sp = py_spans(B, s.file)
            key = ("<lambda>", s.line, s.col) if s.kind == "lambda" else (s.node["name"], s.line)
            hit = sp.get(key)
            if hit is None:
                miss += 1
                continue
            s.line, s.lineEnd = hit[0], hit[1]
        report_span_miss = miss
        report_spans = {"pythonSpanLookupMisses": miss}
        B.report.update(report_spans)

    # type spans for JS/TS from Babel
    if lang == "javascript":
        for s in scopes.values():
            if s.is_type:
                ast_ = babel_cache.get(s.file)
                spans = js_type_spans(ast_, B.src.text(s.file)).get(s.node["name"]) if ast_ else None
                if spans:
                    best = min(spans, key=lambda x: abs(x[0] - s.line))
                    s.line, s.lineEnd = best[0], best[1]
                s_ = s

    # symbol ids (qnames are final once duplicate numbering is done)
    for s in scopes.values():
        s.sid = ids.symbol_id(lang, B.repo, s.qname)
    sid_by_cpg = {mid: s.sid for mid, s in m2s.items()}
    scope_by_sid = {s.sid: s for s in scopes.values()}
    qn_collisions = Counter(s.sid for s in scopes.values())
    dup = [k for k, v in qn_collisions.items() if v > 1]
    if dup:
        raise SystemExit(f"qname collision on {len(dup)} symbols, e.g. {scope_by_sid[dup[0]].qname}")

    records: list[dict] = []
    report = B.report

    # files
    file_ids = {}
    for f in files:
        path = B.P(f)
        data = (Path(a.stage) / f).read_bytes()
        txt = data.decode("utf-8")
        lc = txt.count("\n") + (0 if (txt.endswith("\n") or txt == "") else 1)
        fid = ids.file_id(B.repo, path)
        file_ids[f] = fid
        records.append({"type": "file", "id": fid, "repoId": B.repo, "unitId": B.unit_id, "path": path, "lang": lang,
                        "kind": "code", "role": file_role(path), "contentHash": sha256(data), "lineCount": lc})

    # blame
    blames = {}
    for f in files:
        blames[f] = blame(Path(a.repo), a.commit, B.P(f))

    # symbols
    cond_by_scope = defaultdict(list)
    for c in facts["c"]:
        s = m2s.get(c["m"])
        if s:
            cond_by_scope[s.sid].append(c)
    for s in scopes.values():
        ls = span_start(B, s)
        le = max(s.lineEnd, ls)
        lines = B.src.lines(s.file)
        fp_text = "\n".join(x for x in (l.strip() for l in lines[ls - 1:le]) if x)
        per, times = blames[s.file]
        try:
            at, commit = last_changed(per, times, [(ls, le)])
        except ValueError:
            # a span with no blamed line (an empty file's <module>): the file's last-touching commit
            out = subprocess.run(["git", "-C", a.repo, "log", "-1", "--format=%ct %H", a.commit, "--", B.P(s.file)],
                                 capture_output=True, text=True, check=True).stdout.split()
            at = datetime.fromtimestamp(int(out[0]), timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            commit = out[1]
            report.setdefault("blameFallbackFiles", []).append(B.P(s.file))
        rec = {"type": "symbol", "id": s.sid, "lang": lang, "repoId": B.repo, "qname": s.qname,
               "fileId": file_ids[s.file], "path": B.P(s.file), "kind": s.kind,
               "name": sym_name(s), "nameNorm": name_norm(sym_name(s)),
               "lineStart": ls, "lineEnd": le, "fingerprint": sha256(fp_text.encode("utf-8")),
               "complexity": 1 + len(cond_by_scope[s.sid]), "lastChangedAt": at, "lastChangedCommit": commit}
        if s.parent is not None and s.kind != "module":
            rec["parentId"] = s.parent.sid
        records.append(rec)
    S = {s.sid: s for s in scopes.values()}

    # params
    params_by = {}
    pfacts = defaultdict(list)
    for p in facts["p"]:
        pfacts[p["m"]].append(p)
    for s in scopes.values():
        if s.kind in ("module", "type"):
            continue
        mid = s.cpg_ids[0]
        ps = sorted(pfacts.get(mid, []), key=lambda p: int(p["i"]))
        static = any(nm in ("staticmethod",) for nm, _ in s.decos)
        out = []
        if lang == "python":
            recv_first = s.parent is not None and s.parent.is_type and not static and s.kind in ("method", "accessor")
            idx = 0
            for k, p in enumerate(ps):
                name = p["name"]
                # Joern drops the sigil (`*args` -> code `args`); spec §5.4 keeps it: read it from the source
                srcl = B.src.lines(s.file)[p["l"] - 1] if 1 <= p["l"] <= len(B.src.lines(s.file)) else ""
                m = re.search(r"(\*{1,2})\s*$", srcl[: max(p["c"] - 1, 0)])
                if m and srcl[p["c"] - 1: p["c"] - 1 + len(name)] == name:
                    name = m.group(1) + name
                if k == 0 and recv_first:
                    out.append((-1, name, p))
                else:
                    out.append((idx, name, p))
                    idx += 1
        else:
            for p in ps:
                if int(p["i"]) == 0:
                    continue
                name = p["name"]
                if name.startswith("_tmp_") or not name:
                    name = collapse(p.get("code", "") or "param", 60)
                out.append((int(p["i"]) - 1, name, p))
        params_by[s.sid] = {}
        for index, name, p in out:
            pid = ids.point_id(s.sid, "param", index)
            params_by[s.sid][(int(p["i"]))] = (index, pid)
            records.append({"type": "param", "id": pid, "symbolId": s.sid, "index": index, "name": name,
                            "line": max(p["l"], 1)})

    # rets
    rets_by_scope = defaultdict(list)
    for r in facts["r"]:
        s = m2s.get(r["m"])
        if s:
            rets_by_scope[s.sid].append(r)
    ret_pid = {}
    for sid, rs in rets_by_scope.items():
        rs.sort(key=lambda r: (r["l"], r["c"], int(r["id"])))
        for k, r in enumerate(rs):
            pid = ids.point_id(sid, "ret", k)
            ret_pid[r["id"]] = pid
            records.append({"type": "ret", "id": pid, "symbolId": sid, "ordinal": k, "line": max(r["l"], 1)})
    synth_rets = 0
    for s in scopes.values():
        if s.kind == "lambda" and s.sid not in rets_by_scope:
            pid = ids.point_id(s.sid, "ret", 0)
            records.append({"type": "ret", "id": pid, "symbolId": s.sid, "ordinal": 0, "line": max(s.line, 1)})
            synth_rets += 1
    report["syntheticRetsForExpressionLambdas"] = synth_rets

    # conds
    cond_pid = {}
    for sid, cs in cond_by_scope.items():
        cs.sort(key=lambda c: (c["l"], c["c"], len(c["code"]), c["code"], int(c["id"])))
        for k, c in enumerate(cs):
            pid = ids.point_id(sid, "cond", k)
            cond_pid[c["id"]] = pid
            records.append({"type": "cond", "id": pid, "symbolId": sid, "ordinal": k, "line": max(c["l"], 1),
                            "code": collapse(c["code"], 200)})

    # call sites
    calls_by_scope = defaultdict(list)
    for c in facts["cs"]:
        s = m2s.get(c["m"])
        if s and (c["name"] or c["name"] == "<operator>.new"):
            calls_by_scope[s.sid].append(c)
    cs_pid, cs_info = {}, {}
    synthetic = Counter()
    cpg_full2scope = {m["id"]: m2s.get(m["id"]) for m in facts["m"]}
    # class constructor target for Python class instantiation (`Cls(...)` resolves to <fakeNew>)
    init_of_class = {}
    for s in scopes.values():
        if s.base == "__init__" and s.parent is not None and s.parent.is_type:
            init_of_class[s.parent.sid] = s.sid
    for sid, cs in calls_by_scope.items():
        s = S[sid]
        pos = locate_callees(B, s.file, cs)
        keyed = defaultdict(list)
        for c in cs:
            if c["id"] not in pos:
                synthetic[callee_name(c)] += 1
                continue
            keyed[callee_name(c)].append(c)
        for name, grp in keyed.items():
            grp.sort(key=lambda c: (pos[c["id"]], len(c["code"]), int(c["id"])))
            for k, c in enumerate(grp):
                tgt = {}
                for t in c["in"]:
                    ts = m2s.get(t["id"])
                    if ts is None:
                        continue
                    tsid = ts.sid
                    if ts.is_type:
                        tsid = init_of_class.get(ts.sid)
                        if tsid is None:
                            continue
                    tgt[tsid] = S[tsid].qname
                targets = [{"symbolId": k2, "qname": v} for k2, v in sorted(tgt.items())]
                cid = ids.callsite_id(sid, name, k)
                cs_pid[c["id"]] = cid
                cs_info[c["id"]] = (cid, sid, pos[c["id"]])
                kind = "new" if c["name"] == "<operator>.new" else ("super" if name == "super" else "call")
                rec = {"type": "callsite", "id": cid, "symbolId": sid, "calleeName": name, "ordinal": k,
                       "callKind": kind, "line": max(pos[c["id"]][0], 1)}
                if targets:
                    rec.update(resolution="resolved", targets=targets, status="confirmed", reason="resolved-call")
                else:
                    ext = [o for o in c["out"] if o and not o.startswith("<unknownFullName>")]
                    ext = [o for o in ext if not o.startswith("__builtin")] or ext
                    if ext and c["mfn"] and c["mfn"] != "<unknownFullName>" and not c["in"]:
                        rec.update(resolution="external", externalTarget=c["mfn"], targets=[], status="possible",
                                   reason="name-match")
                    else:
                        rec.update(resolution="unresolved", targets=[], status="possible", reason="name-match")
                records.append(rec)

    report["syntheticCallsDropped"] = {"total": sum(synthetic.values()), "byName": dict(synthetic.most_common(12))}

    # imports (source-derived) and constants
    imports_by_file = {}
    for f in files:
        if lang == "python":
            imps = py_imports(B, f, [])
        else:
            ast_ = babel_cache.get(f)
            imps = js_imports(B, f, ast_) if ast_ else []
        imports_by_file[f] = imps
    consts_names = defaultdict(set)
    const_facts = defaultdict(lambda: defaultdict(list))
    for c in facts["const"]:
        s = m2s.get(c["m"])
        if s and s.kind in ("module", "type"):
            const_facts[s.sid][c["name"]].append(c)
    const_recs = []
    for sid, d in const_facts.items():
        s = S[sid]
        for name, lst in d.items():
            if len(lst) != 1:
                continue
            q = f"{B.P(s.file)}::{name}" if s.kind == "module" else f"{s.qname}.{name}"
            consts_names[B.P(s.file)].add(q)
            const_recs.append((s, name, q, lst[0]))
    # resolver tables: name -> (module, name, sub, resolved)
    alias_imps = {}
    for f, imps in imports_by_file.items():
        alias_imps[f] = imps
    R = Resolver(B, {}, consts_names)
    for f, imps in imports_by_file.items():
        for r in imps:
            for n in r["names"]:
                if n["name"] == "*":
                    continue
                local = n.get("alias", n["name"])
                R.alias[f][local] = (r["module"], n["name"], (r.get("_subs") or {}).get(n["name"]), r.get("resolvedPath"))

    for s, name, q, c in const_recs:
        el = R.element(c["e"], s.file)
        if "unknown" in el:
            continue
        records.append({"type": "constant", "id": ids.constant_id(lang, B.repo, q), "qname": q,
                        "fileId": file_ids[s.file], "line": max(c["l"], 1), "values": [el]})

    # import records
    spans_by_file = defaultdict(list)
    for s in scopes.values():
        if s.kind not in ("module", "type"):
            spans_by_file[s.file].append(s)
    n_imports = 0
    for f, imps in imports_by_file.items():
        for k, r in enumerate(imps):
            rec = {"type": "import", "id": ids.import_id(file_ids[f], k), "fileId": file_ids[f], "line": r["line"],
                   "ordinal": k, "module": r["module"], "names": r["names"], "dynamic": r["dynamic"]}
            if r.get("resolvedPath"):
                rec["resolvedPath"] = r["resolvedPath"]
            inner = [s for s in spans_by_file[f] if span_start(B, s) <= r["line"] <= s.lineEnd]
            if inner:
                inner.sort(key=lambda s: (span_start(B, s) - s.lineEnd))
                rec["symbolId"] = inner[0].sid
            records.append(rec)
            n_imports += 1

    # value sets (call-site arguments)
    vs_seen = set()
    n_vs = 0
    for c in facts["cs"]:
        if c["id"] not in cs_info:
            continue
        cid, sid, _ = cs_info[c["id"]]
        s = S[sid]
        for ar in c["args"]:
            i = int(ar["i"])
            if i == 0 and not ar["an"]:
                continue   # the receiver: values of `recv` are not value sets worth a record
            head = kw_head(ar["an"]) if ar["an"] else f"a{i - 1}"
            path, trunc = path_from(head, ar["f"])
            if trunc:
                continue
            el = R.element(ar["e"], s.file)
            if "unknown" in el and el["unknown"] != "point-derived":
                continue
            vid = ids.value_set_id(cid, path)
            if vid in vs_seen:
                continue
            vs_seen.add(vid)
            records.append({"type": "valueSet", "id": vid, "callSiteId": cid, "arg": path, "values": [el]})
            n_vs += 1

    # decorators
    full2mid = {m["full"]: m["id"] for m in facts["m"]}
    for d in facts["deco"]:
        mid = full2mid.get(d["target"])
        s = m2s.get(mid) if mid else None
        if s is None or s.sid not in S:
            continue
        name = re.sub(r"\s+", "", decorator_name(B, s.file, d))
        args = {}
        for ar in d["args"]:
            i = int(ar["i"])
            head = kw_head(ar["an"]) if ar["an"] else f"a{i - 1}"
            path, trunc = path_from(head, ar["f"])
            if trunc:
                continue
            el = R.element(ar["e"], s.file)
            if "unknown" in el and el["unknown"] != "point-derived":
                continue
            args[path] = [el]
        records.append({"type": "decorator", "id": ids.decorator_id(s.sid, int(d["o"])), "targetId": s.sid,
                        "ordinal": int(d["o"]), "form": "decorator", "name": name, "line": max(d["l"], 1),
                        "args": [{"arg": k, "values": v} for k, v in sorted(args.items())]})

    # data
    point_of_use = {}
    for cid_cpg, (cid, sid, _) in cs_info.items():
        point_of_use[("cs", cid_cpg)] = cid
    for rid, pid in ret_pid.items():
        point_of_use[("r", rid)] = pid
    for k, pid in cond_pid.items():
        point_of_use[("c", k)] = pid
    data_seen, n_data, dropped_data = set(), 0, Counter()
    for d in facts["d"]:
        s = m2s.get(d["m"])
        if s is None:
            dropped_data["no-symbol"] += 1
            continue
        use = point_of_use.get((d["uk"], d["uid"]))
        if use is None:
            dropped_data["no-use-point"] += 1
            continue
        if d["dk"] == "cs":
            defp = point_of_use.get(("cs", d["did"]))
        elif d["dk"] == "p":
            pm = params_by.get(s.sid, {}).get(int(d["di"]))
            defp = pm[1] if pm else None
        else:
            defp = None
        if defp is None:
            dropped_data["no-def-point"] += 1
            continue
        th = d["th"]
        if th == "":
            to_path, t1 = "", False
        elif th == "ret":
            to_path, t1 = path_from("ret", d["tf"])
        elif th.startswith("kw."):
            to_path, t1 = path_from(kw_head(th[3:]), d["tf"])
        else:
            to_path, t1 = path_from(th, d["tf"])
        h = d["h"]
        if h == "r":
            from_path, t2 = path_from("r", d["ff"])
        else:
            pm = params_by.get(s.sid, {}).get(int(d["di"]))
            idx = pm[0]
            from_path, t2 = path_from("recv" if idx == -1 else f"p{idx}", d["ff"])
        did = ids.data_id(use, defp, to_path, from_path)
        if did in data_seen:
            dropped_data["duplicate"] += 1
            continue
        data_seen.add(did)
        trunc = t1 or t2
        records.append({"type": "data", "id": did, "use": use, "def": defp, "toPath": to_path, "fromPath": from_path,
                        "status": "possible" if trunc else "confirmed",
                        "reason": "field-truncated" if trunc else "intra-flow"})
        n_data += 1
    report["dataDropped"] = dict(dropped_data)

    # controlledBy: G-4 -- drop self-edges, de-duplicate (point, cond) pairs, before writing
    edges, cb_unmapped = [], 0
    for e in facts["cb"]:
        if m2s.get(e["m"]) is None:
            continue
        point = point_of_use.get((e["pk"], e["pid"]))
        cond = cond_pid.get(e["cond"])
        if point is None or cond is None:
            cb_unmapped += 1
            continue
        edges.append((point, cond))
    kept, cb_self, cb_dup = g4_filter(edges, getattr(a, "g4_mutant", None))
    for point, cond in kept:
        records.append({"type": "controlledBy", "id": ids.controlled_by_id(point, cond), "point": point, "cond": cond})
    report["controlledBy"] = {"rawEdges": len(edges) + cb_unmapped, "selfEdgesDropped": cb_self,
                              "duplicatePairsDropped": cb_dup, "unmapped": cb_unmapped, "written": len(kept)}
    condset = set(cond_pid.values())
    report["controlledBy"]["condToCond"] = sum(1 for (p, c) in kept if p in condset)

    # header
    counts = Counter(r["type"] for r in records)
    tree = subprocess.run(["git", "-C", a.repo, "rev-parse", f"{a.commit}:{B.root}"], capture_output=True, text=True,
                          check=True).stdout.strip()
    es = Path(a.extract_script)
    header = {
        "type": "header", "bundleSchemaVersion": "1.0",
        "unit": {"repoId": B.repo, "rootPath": B.root, "lang": lang}, "unitId": B.unit_id,
        "builtCommit": a.commit, "builtTree": tree,
        "builtAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tools": {"joern": "v4.0.579", "frontend": "pysrc2cpg" if lang == "python" else "jssrc2cpg",
                  "frontendVersion": "v4.0.579", "normalizer": NORMALIZER, "normalizerVersion": NORMALIZER_VERSION,
                  "extractScript": "extract.sc", "extractScriptHash": sha256(es.read_bytes())},
        "sandboxAuditIds": a.audit_ids.split(","), "egressMode": "offline", "flowProjected": True,
        "fieldDepthCap": FIELD_CAP, "vendoredDirs": sorted(DEFAULT_VENDORED), "history": "available",
        "counts": {t: counts.get(t, 0) for t in bundle.RECORD_TYPES[1:]},
    }
    report["counts"] = header["counts"]
    report["filesParsed"] = len(files)
    report["symbols"] = len(scopes)
    return [header] + records, report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--facts", required=True)
    ap.add_argument("--stage", required=True, help="staged unit dir (Joern's parse root)")
    ap.add_argument("--unit-root", required=True, help="rootPath of the unit in the repo ('.' for the repo root)")
    ap.add_argument("--lang", required=True, choices=["python", "javascript"])
    ap.add_argument("--repo-id", default="graphmind-ai-lab")
    ap.add_argument("--repo", required=True, help="source repository (for git blame / rev-parse)")
    ap.add_argument("--commit", required=True)
    ap.add_argument("--audit-ids", required=True, help="comma-separated sandbox auditIds, invocation order")
    ap.add_argument("--extract-script", required=True)
    ap.add_argument("--astgen", help="astgen output dir (JS/TS: Babel AST for imports and type spans)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", help="write the measurement report (JSON) here")
    ap.add_argument("--g4-mutant", choices=["self", "dedup"], help="verification only: KEEP the self-edges "
                    "(self) or the repeated (point, cond) pairs (dedup); the validator must then reject the bundle")
    a = ap.parse_args(argv)
    facts = load_facts(Path(a.facts))
    records, report = assemble(a, facts)
    bundle.write_bundle(a.out, records)   # sorts, validates, canonical bytes
    if a.report:
        Path(a.report).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
