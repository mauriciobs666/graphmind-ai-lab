"""Python frontend of the normalizer: scope tree, spans from the source parse, imports.

Ported from ``cpg/spikes/vertical-slice/normalize.py`` (``build_scopes`` with its JavaScript branches left to the
TS/JS frontend, ``decorator_name``, ``py_spans``, ``py_resolve``, ``py_imports``). Names, nesting and qnames come from
the CPG's scope tree, never from ``FULL_NAME`` surgery (spec3 §4.0); end lines and imports come from the stdlib ``ast``.
"""
from __future__ import annotations

import ast
import re
from collections import defaultdict
from pathlib import Path, PurePosixPath

from .core import Builder, Scope, number_and_qualify

PY_SYNTHETIC = ("<body>", "<metaClassCallHandler>", "<fakeNew>")


def build_scopes(B: Builder):
    """Return ``(scopes, method_to_scope, decorators_by_target)``: one ``Scope`` per Python symbol, qnames final."""
    facts = B.facts
    files = {f["name"] for f in facts["f"] if f["name"].endswith(".py")}
    methods = {m["id"]: m for m in facts["m"] if m["file"] in files}
    types = {t["id"]: t for t in facts["t"] if t["file"] in files}
    deco_by_target: dict[str, list[dict]] = defaultdict(list)
    for d in facts["deco"]:
        deco_by_target[d["target"]].append(d)
    scopes: dict[tuple, Scope] = {}
    m2s: dict[str, Scope] = {}

    # file-level scopes
    for mid, m in methods.items():
        if m["name"] == "<module>":
            s = Scope(key=("m", mid), kind="module", name="<module>", base="<module>", line=1, col=1,
                      lineEnd=len(B.src.text(m["file"]).splitlines()) or 1, parent=None, file=m["file"],
                      jname="<module>", node=m)
            s.cpg_ids.append(mid)
            scopes[s.key] = s
            m2s[mid] = s

    # A class's ``<body>`` method names its TYPE_DECL by id. Not by fullName: a class defined twice in one scope keeps
    # the same fullName (live pysrc2cpg), so a fullName lookup is decided by whichever record came last.
    body_of = {m["pid"]: mid for mid, m in methods.items() if m["name"] == "<body>" and m["pk"] == "TYPE_DECL"}

    # type scopes: a class is a TYPE_DECL with a ``<body>`` method
    for tid, t in types.items():
        bid = body_of.get(tid)
        if t["name"].endswith("<meta>") or bid is None or t["name"] in PY_SYNTHETIC:
            continue
        bm = methods[bid]
        s = Scope(key=("t", tid), kind="type", name=t["name"], base=t["name"], line=bm["l"], col=bm["c"],
                  lineEnd=bm["le"], parent=None, file=t["file"], jname=t["name"], is_type=True, node=t)
        s.cpg_ids.append(bid)
        scopes[s.key] = s
        m2s[bid] = s

    # method scopes
    for mid, m in methods.items():
        if mid in m2s:
            continue
        name = m["name"]
        if name in PY_SYNTHETIC or name.endswith("<metaClassAdapter>"):
            continue
        anon = name.startswith("<lambda>")
        base = "<anon>" if anon else name
        s = Scope(key=("m", mid), kind="lambda" if anon else "function", name=name, base=base,
                  line=m["l"], col=m["c"], lineEnd=m["le"], parent=None, file=m["file"], jname=name, node=m)
        s.cpg_ids.append(mid)
        scopes[s.key] = s
        m2s[mid] = s

    # parents
    fallbacks = 0
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
                # the TYPE_DECL parent may be a function's TYPE_DECL; climb to the method with the same fullName
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
            fallbacks += 1
            parent = next((x for x in scopes.values() if x.kind == "module" and x.file == s.file), None)
        s.parent = parent

    B.report["pythonParentFallbacks"] = fallbacks    # symbols hung under their file's module because no parent resolved
    owned = decorators_by_owner(scopes, deco_by_target)

    # property accessors, overload stubs
    for s in list(scopes.values()):
        if s.kind == "lambda":
            # extract.sc reports a call that takes a lambda (`f(lambda: ...)`) as a decorator application of the lambda; the
            # slice emits it as a `decorator` record (3 of the stored 213) without letting it widen the lambda's span
            s.lambda_decos = [(decorator_name(B, s.file, d), d) for d in owned.get(s.key, [])]
            continue
        if s.kind != "function":
            continue
        names = [(decorator_name(B, s.file, d), d) for d in owned.get(s.key, [])]
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

    number_and_qualify(B, scopes)
    return scopes, m2s, deco_by_target


def decorators_by_owner(scopes: dict, deco_by_target: dict) -> dict:
    """Scope key -> its decorator facts, in written order. A decorator fact names its function by fullName, which two
    functions can share (a redefined class's members), so among the candidates it belongs to the first one at or
    after its line (a decorator precedes its ``def``)."""
    by_full: dict[str, list[Scope]] = defaultdict(list)
    for s in scopes.values():
        if s.kind in ("function", "lambda"):
            by_full[s.node["full"]].append(s)
    owned: dict[tuple, list[dict]] = defaultdict(list)
    for target, ds in deco_by_target.items():
        cands = by_full.get(target, [])
        for d in ds:
            if len(cands) > 1:
                after = [c for c in cands if c.node["l"] >= d["l"]]
                owner = min(after, key=lambda c: (c.node["l"], c.node["c"], c.node["id"])) if after else \
                    max(cands, key=lambda c: (c.node["l"], c.node["c"], c.node["id"]))
            elif cands:
                owner = cands[0]
            else:
                continue
            owned[owner.key].append(d)
    for ds in owned.values():
        ds.sort(key=lambda d: (d["l"], d["c"], d["o"]))
    return owned


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
    cache = B.span_cache
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


def apply_py_spans(B: Builder, scopes: dict) -> int:
    """Overwrite each non-module scope's line span from the source parse; return the lookup misses (spans kept)."""
    miss = 0
    for s in scopes.values():
        if s.kind == "module":
            continue
        key = ("<lambda>", s.line, s.col) if s.kind == "lambda" else (s.node["name"], s.line)
        hit = py_spans(B, s.file).get(key)
        if hit is None:
            miss += 1
            continue
        s.line, s.lineEnd = hit[0], hit[1]
    B.report["pythonSpanLookupMisses"] = miss
    return miss


# -- imports (source-derived; Joern emits duplicated IMPORT nodes) ---------------------------
def _norm(path: str) -> str:
    """A candidate path without ``.`` segments (``./a.py`` -> ``a.py``): the module path and every submodule path alike."""
    return str(PurePosixPath(path))


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
    res = next((c for c in map(_norm, cands) if src.exists(c)), None)
    subs = {}
    for n in names:
        for c in map(_norm, (f"{d}/{n}.py", f"{d}/{n}/__init__.py")):
            if src.exists(c):
                subs[n] = B.P(c)
                break
    return (B.P(res) if res else None), subs


def py_imports(B: Builder, file: str, scopes_in_file: list[Scope] | None = None):
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
            names.sort(key=lambda x: (x["name"], x.get("alias", "")))
            rec = dict(line=node.lineno, col=node.col_offset, module=module, names=names, dynamic=False,
                       resolvedPath=resolved, _sub=None)
            rec["_subs"] = subs
            out.append(rec)
    out.sort(key=lambda r: (r["line"], r["col"]))
    return out


def py_params(B: Builder, s: Scope, ps: list[dict]):
    """``(index, name, fact)`` per parameter of a Python symbol. A method's first parameter (the receiver) gets index
    ``-1`` unless the function is a ``staticmethod``; the others count from 0. Joern drops the sigil (``*args`` is
    code ``args``) where spec3 §5.4 keeps it, so the name is read back from the source line."""
    static = any(nm in ("staticmethod",) for nm, _ in s.decos)
    recv_first = s.parent is not None and s.parent.is_type and not static and s.kind in ("method", "accessor")
    out = []
    idx = 0
    lines = B.src.lines(s.file)
    for k, p in enumerate(ps):
        name = p["name"]
        srcl = lines[p["l"] - 1] if 1 <= p["l"] <= len(lines) else ""
        m = re.search(r"(\*{1,2})\s*$", srcl[: max(p["c"] - 1, 0)])
        if m and srcl[p["c"] - 1: p["c"] - 1 + len(name)] == name:
            name = m.group(1) + name
        if k == 0 and recv_first:
            out.append((-1, name, p))
        else:
            out.append((idx, name, p))
            idx += 1
    return out
