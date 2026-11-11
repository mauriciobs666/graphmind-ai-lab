"""TS/JS frontend of the normalizer: scope tree, imports and type spans from astgen's Babel ASTs, parameters.

Ported from ``cpg/spikes/vertical-slice/normalize.py`` (the ``lang == "javascript"`` branches of ``build_scopes`` and
``assemble``, ``js_resolve``, ``babel``/``walk_babel``, ``js_imports``, ``js_type_spans``): equivalence before
improvement (`docs/plans/connected-code-graph-p1.md` §8, D7). Names, nesting and qnames come from the CPG's scope tree
(spec3 §4.0); imports and the end lines of named types come from the Babel AST, because ``jssrc2cpg`` gives a type
declaration no end line. Babel's ``start``/``end`` are UTF-16 indexes.
"""
from __future__ import annotations

import bisect
import json
from collections import defaultdict
from pathlib import Path

from .core import Builder, Scope, collapse, number_and_qualify

JS_EXTS = (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".mts", ".cts")
TYPE_NODES = {"TSInterfaceDeclaration", "TSTypeAliasDeclaration", "TSEnumDeclaration", "ClassDeclaration",
              "ClassExpression", "TSModuleDeclaration"}


def function_of_type_decl(tdecl: dict, methods: dict, m2s: dict):
    """The scope of the function a TYPE_DECL belongs to. Joern gives a function and its TYPE_DECL one fullName, which two
    declarations of the same function share, so the owner is the candidate that starts last at or before the TYPE_DECL's own
    position (the first one when none does), by the total key ``(line, column, id)``, never by fact order."""
    cands = sorted((m for m in methods.values() if m["full"] == tdecl["full"] and m["id"] in m2s),
                   key=lambda m: (m["l"], m["c"], m["id"]))
    if not cands:
        return None
    at = (tdecl["l"], tdecl["c"])
    before = [m for m in cands if (m["l"], m["c"]) <= at]
    return m2s[(before[-1] if before else cands[0])["id"]]


def build_scopes(B: Builder):
    """Return ``(scopes, method_to_scope, {})``: one ``Scope`` per JS/TS symbol, qnames final.

    A TYPE_DECL is a type iff no METHOD shares its fullName and it is not an ``<init>``, a lambda, the ``:program``
    declaration or one of Joern's ``<anon-class>`` object-type-literal artifacts. An ``<init>`` method is a symbol (a
    ``constructor``) only when its source line says ``constructor``; the synthetic one's code belongs to its type."""
    facts = B.facts
    files = {f["name"] for f in facts["f"] if f["name"].endswith(JS_EXTS)}
    methods = {m["id"]: m for m in facts["m"] if m["file"] in files}
    types = {t["id"]: t for t in facts["t"] if t["file"] in files}
    scopes: dict[tuple, Scope] = {}
    m2s: dict[str, Scope] = {}

    for mid, m in methods.items():
        if m["name"] == ":program":
            s = Scope(key=("m", mid), kind="module", name="<module>", base="<module>", line=1, col=1,
                      lineEnd=len(B.src.text(m["file"]).splitlines()) or 1, parent=None, file=m["file"],
                      jname="<module>", node=m)
            s.cpg_ids.append(mid)
            scopes[s.key] = s
            m2s[mid] = s

    method_fulls = {m["full"] for m in methods.values()}
    for tid, t in types.items():
        if t["full"] in method_fulls or t["name"] == "<init>" or t["name"].startswith("<lambda>"):
            continue
        if t["name"] == ":program" or t["name"].startswith("<anon-class>"):
            continue
        s = Scope(key=("t", tid), kind="type", name=t["name"], base=t["name"], line=t["l"], col=t["c"],
                  lineEnd=t["l"], parent=None, file=t["file"], jname=t["name"], is_type=True, node=t)
        scopes[s.key] = s

    for mid, m in methods.items():
        if mid in m2s:
            continue
        name = m["name"]
        if name == "<init>":
            lines = B.src.lines(m["file"])
            ln = lines[m["l"] - 1] if 1 <= m["l"] <= len(lines) else ""
            if "constructor" not in ln:
                continue
        anon = name.startswith("<lambda>")
        base = "<anon>" if anon else name
        s = Scope(key=("m", mid), kind="lambda" if anon else "function", name=name, base=base,
                  line=m["l"], col=m["c"], lineEnd=m["le"], parent=None, file=m["file"], jname=name, node=m)
        s.cpg_ids.append(mid)
        scopes[s.key] = s
        m2s[mid] = s

    # A type's CPG parent is a BLOCK (pk neither METHOD nor TYPE_DECL), so a type hangs under its file's module by the
    # fallback below: the slice's rule, and no count of it is kept because it is the common case.
    for s in scopes.values():
        if s.kind == "module":
            continue
        n = s.node
        pk, pid = n["pk"], n["pid"]
        parent = None
        if pk == "METHOD":
            parent = m2s.get(pid)
        elif pk == "TYPE_DECL":
            parent = scopes.get(("t", pid))
            if parent is None:
                # the TYPE_DECL parent may be a function's or <init>'s TYPE_DECL: climb to the method with its fullName
                tpar = types.get(pid)
                if tpar is not None:
                    parent = function_of_type_decl(tpar, methods, m2s)
        if parent is None and pk == "METHOD":
            pm = methods.get(pid)
            if pm is not None and pm["pk"] == "TYPE_DECL":
                parent = scopes.get(("t", pm["pid"]))
        if parent is None:
            parent = next((x for x in scopes.values() if x.kind == "module" and x.file == s.file), None)
        s.parent = parent

    # the synthetic <init> methods' code belongs to the type
    for mid, m in methods.items():
        if m["name"] == "<init>" and mid not in m2s and m["pk"] == "TYPE_DECL":
            tdecl = scopes.get(("t", m["pid"]))
            if tdecl is not None:
                tdecl.cpg_ids.append(mid)
                m2s[mid] = tdecl

    for s in scopes.values():
        if s.kind == "function" and s.parent is not None and s.parent.is_type:
            s.kind = "method"
        if s.base == "<init>":
            s.kind = "constructor"

    number_and_qualify(B, scopes)
    return scopes, m2s, {}


def js_resolve(B: Builder, importer: str, spec: str):
    """Resolve a relative specifier to a file of the staged unit (extension, then ``/index`` + extension); a bare
    specifier, or a ``..`` that climbs past the unit root, resolves to nothing."""
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
    """The Babel AST astgen wrote for ``file`` (``<astgen_dir>/<file>.json``), or ``None`` when it wrote none."""
    p = Path(astgen_dir) / (file + ".json")
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
            names.sort(key=lambda x: (x["name"], x.get("alias", "")))
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


def utf16_line_index(text: str):
    """Sorted UTF-16 offsets of every ``\\n`` (Babel's ``start``/``end`` are UTF-16 indexes)."""
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


def apply_js_type_spans(B: Builder, scopes: dict, babel_cache: dict) -> None:
    """Overwrite each type scope's span from the Babel AST: of the same-named declarations, the one starting nearest
    the CPG's line. A type with no AST (or no named declaration) keeps the CPG's one-line span."""
    for s in scopes.values():
        if not s.is_type:
            continue
        ast_ = babel_cache.get(s.file)
        spans = js_type_spans(ast_, B.src.text(s.file)).get(s.node["name"]) if ast_ else None
        if spans:
            best = min(spans, key=lambda x: abs(x[0] - s.line))
            s.line, s.lineEnd = best[0], best[1]


def js_params(ps: list[dict]):
    """``(index, name, fact)`` per parameter of a JS/TS symbol: the ``this`` parameter (index 0) is dropped, a Joern
    ``_tmp_N`` or empty name is replaced by the parameter's source code (collapsed, 60 characters)."""
    out = []
    for p in ps:
        if int(p["i"]) == 0:
            continue
        name = p["name"]
        if name.startswith("_tmp_") or not name:
            name = collapse(p.get("code", "") or "param", 60)
        out.append((int(p["i"]) - 1, name, p))
    return out
