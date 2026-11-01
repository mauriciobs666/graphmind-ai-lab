#!/usr/bin/env python3
"""verify_qnames.py -- S11 (P0-10): re-derive every slice symbol's qname from the SOURCE alone.

Bundle spec §4.0: "S11 checks that on the slice's units: for every symbol record, it re-derives the
qname from source by this section and compares.  A mismatch is an SL-7 delta."

This file shares no code with normalize.py.  Python is derived with the stdlib ``ast``; TS/JS with the
Babel AST that astgen (the parser jssrc2cpg itself runs) writes, walked by §4.2's rules.  Neither
derivation reads the CPG, Joern's FULL_NAME, or the bundle.  The bundle is read only to compare.

  verify_qnames.py --lang python|javascript --stage <staged unit> --root <rootPath> --bundle <file>
                   [--astgen <dir>] [--json <report>]

Compared per qname: presence in both, and (lineStart, lineEnd) of the symbol.  Exit 1 on any difference.
"""
from __future__ import annotations

import argparse
import ast
import bisect
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

IDENT = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*\Z")


class Node:
    def __init__(self, base, parent, line, col, start, end, accessor=None):
        self.base, self.parent, self.line, self.col = base, parent, line, col
        self.start, self.end, self.accessor, self.dup = start, end, accessor, 1


def finish(path, nodes):
    """Apply rule 7 (#N duplicates; <anon> is always #N) and build the qnames."""
    groups = defaultdict(list)
    for n in nodes:
        groups[(id(n.parent), n.base, n.accessor)].append(n)
    for (_, base, acc), grp in groups.items():
        grp.sort(key=lambda n: (n.line, n.col))
        for i, n in enumerate(grp, start=1):
            n.dup = i

    def seg(n):
        b = n.base + (("." + n.accessor) if n.accessor else "")
        if n.base == "<anon>":
            return f"<anon>#{n.dup}"
        return b + (f"#{n.dup}" if n.dup > 1 else "")

    def chain(n):
        parts = []
        while n is not None:
            parts.append(seg(n))
            n = n.parent
        return ".".join(reversed(parts))

    out = {f"{path}::<module>": None}
    res = {}
    for n in nodes:
        res[f"{path}::{chain(n)}"] = (n.start, n.end)
    return res


# -- Python -------------------------------------------------------------------------------------
def py_file(path, text):
    tree = ast.parse(text)
    nodes = []

    def deco_start(d, n):
        return min([x.lineno for x in getattr(n, "decorator_list", [])] + [n.lineno])

    def visit(n, scope):
        """scope = the Node of the nearest enclosing symbol (evaluation scope), or None for module."""
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # decorators, defaults and annotations are evaluated in the ENCLOSING scope (§4.1 rule 2)
            for d in n.decorator_list:
                visit(d, scope)
            for d in list(n.args.defaults) + [k for k in n.args.kw_defaults if k is not None]:
                visit(d, scope)
            for a in list(n.args.args) + list(n.args.posonlyargs) + list(n.args.kwonlyargs) + [n.args.vararg, n.args.kwarg]:
                if a is not None and a.annotation is not None:
                    visit(a.annotation, scope)
            if n.returns is not None:
                visit(n.returns, scope)
            names = []
            for d in n.decorator_list:
                if isinstance(d, ast.Name):
                    names.append(d.id)
                elif isinstance(d, ast.Attribute):
                    names.append(ast.unparse(d))
            if any(x in ("overload", "typing.overload") for x in names):
                return                                             # not a symbol; its body is a stub
            acc = None
            for d in n.decorator_list:
                if isinstance(d, ast.Name) and d.id == "property":
                    acc = "<get>"
                elif isinstance(d, ast.Attribute) and isinstance(d.value, ast.Name) and d.value.id == n.name:
                    acc = {"setter": "<set>", "deleter": "<del>"}.get(d.attr, acc)
            me = Node(n.name, scope, n.lineno, n.col_offset, deco_start(None, n), n.end_lineno, acc)
            nodes.append(me)
            for s in n.body:
                visit(s, me)
            return
        if isinstance(n, ast.ClassDef):
            for d in n.decorator_list:
                visit(d, scope)
            for b in n.bases:
                visit(b, scope)
            for k in n.keywords:
                visit(k, scope)
            me = Node(n.name, scope, n.lineno, n.col_offset, deco_start(None, n), n.end_lineno)
            nodes.append(me)
            for s in n.body:
                visit(s, me)
            return
        if isinstance(n, ast.Lambda):
            for d in list(n.args.defaults) + [k for k in n.args.kw_defaults if k is not None]:
                visit(d, scope)
            me = Node("<anon>", scope, n.lineno, n.col_offset, n.lineno, n.end_lineno)
            nodes.append(me)
            visit(n.body, me)
            return
        for c in ast.iter_child_nodes(n):
            visit(c, scope)

    visit(tree, None)
    res = finish(path, nodes)
    nlines = len(text.splitlines()) or 1
    res[f"{path}::<module>"] = (1, nlines)
    return res


# -- TS/JS (Babel AST from astgen) --------------------------------------------------------------
def utf16_nl(text):
    out, off = [], 0
    for ch in text:
        if ch == "\n":
            out.append(off)
        off += 2 if ord(ch) > 0xFFFF else 1
    return out


def js_file(path, text, ast_):
    nl = utf16_nl(text)

    def end_line(n):
        return bisect.bisect_left(nl, n["end"] - 1) + 1

    def start_line(n):
        return n["loc"]["start"]["line"]

    def start_col(n):
        return n["loc"]["start"]["column"]

    nodes = []
    hint = {}   # id(function node) -> binding name (ES2015 inference restricted to §4.2's forms)
    FUNCS = ("FunctionExpression", "ArrowFunctionExpression")

    def unwrap(x):
        while isinstance(x, dict) and x.get("type") in ("TSAsExpression", "TSSatisfiesExpression", "TSNonNullExpression",
                                                          "ParenthesizedExpression", "TSTypeAssertion"):
            x = x["expression"]
        return x

    def keyname(k, computed):
        if computed:
            return None
        if k["type"] == "Identifier":
            return k["name"]
        if k["type"] == "StringLiteral" and IDENT.match(k["value"]):
            return k["value"]
        if k["type"] == "PrivateName":
            return "#" + k["id"]["name"]
        return None

    def visit(n, scope, parent_ctx=None):
        if isinstance(n, list):
            for x in n:
                visit(x, scope)
            return
        if not isinstance(n, dict) or "type" not in n:
            return
        t = n["type"]
        if t == "VariableDeclarator" and n.get("id", {}).get("type") == "Identifier":
            v = unwrap(n.get("init"))
            if isinstance(v, dict) and v.get("type") in FUNCS and not (v.get("id") and v["id"].get("name")):
                hint[id(v)] = n["id"]["name"]
        if t in ("ClassProperty", "ClassPrivateProperty", "PropertyDefinition"):
            v = unwrap(n.get("value"))
            nm = keyname(n["key"], n.get("computed")) if n.get("key") else None
            if isinstance(v, dict) and v.get("type") in FUNCS and nm and not (v.get("id") and v["id"].get("name")):
                hint[id(v)] = nm
        if t == "ObjectProperty":
            v = unwrap(n.get("value"))
            nm = keyname(n["key"], n.get("computed"))
            if isinstance(v, dict) and v.get("type") in FUNCS and nm and not (v.get("id") and v["id"].get("name")):
                hint[id(v)] = nm
        me = None
        if t == "FunctionDeclaration":
            nm = (n.get("id") or {}).get("name")
            if nm is None and parent_ctx == "default":
                nm = "default"
            me = Node(nm or "<anon>", scope, start_line(n), start_col(n), start_line(n), end_line(n))
        elif t in FUNCS:
            own = (n.get("id") or {}).get("name")
            nm = own or hint.get(id(n))
            if nm is None and parent_ctx == "default":
                nm = "default"
            me = Node(nm or "<anon>", scope, start_line(n), start_col(n), start_line(n), end_line(n))
        elif t in ("ClassMethod", "ClassPrivateMethod", "ObjectMethod"):
            nm = keyname(n["key"], n.get("computed"))
            acc = None
            if t != "ObjectMethod" and n.get("kind") == "constructor":
                nm = "<init>"
            elif n.get("kind") == "get":
                acc = "<get>"
            elif n.get("kind") == "set":
                acc = "<set>"
            me = Node(nm or "<anon>", scope, start_line(n), start_col(n), start_line(n), end_line(n), acc)
        elif t == "TSDeclareMethod" and n.get("abstract"):
            nm = keyname(n["key"], n.get("computed"))
            me = Node(nm or "<anon>", scope, start_line(n), start_col(n), start_line(n), end_line(n))
        elif t in ("ClassDeclaration", "ClassExpression"):
            nm = (n.get("id") or {}).get("name")
            if nm is None and parent_ctx == "default":
                nm = "default"
            me = Node(nm or "<anon>", scope, start_line(n), start_col(n), start_line(n), end_line(n))
        elif t in ("TSInterfaceDeclaration", "TSTypeAliasDeclaration", "TSEnumDeclaration"):
            me = Node(n["id"]["name"], scope, start_line(n), start_col(n), start_line(n), end_line(n))
        elif t == "TSModuleDeclaration" and n.get("id", {}).get("type") == "Identifier" and not n.get("declare"):
            me = Node(n["id"]["name"], scope, start_line(n), start_col(n), start_line(n), end_line(n))
        elif t == "TSMethodSignature" and scope is not None and scope_is_type(scope):
            nm = keyname(n["key"], n.get("computed"))
            if nm:
                me = Node(nm, scope, start_line(n), start_col(n), start_line(n), end_line(n))
        if me is not None:
            nodes.append(me)
            me_t = t
            type_nodes.add(id(me)) if t in ("TSInterfaceDeclaration", "TSTypeAliasDeclaration", "ClassDeclaration", "ClassExpression", "TSEnumDeclaration") else None
            scope = me
        for k, v in n.items():
            if k in ("loc", "extra", "leadingComments", "trailingComments", "innerComments", "typeAnnotation") and t not in ("TSTypeAliasDeclaration",):
                if k != "typeAnnotation":
                    continue
            if isinstance(v, (dict, list)):
                ctx = "default" if t == "ExportDefaultDeclaration" else None
                visit(v, scope, ctx)

    type_nodes = set()

    def scope_is_type(s):
        return id(s) in type_nodes

    visit(ast_["program"], None)
    res = finish(path, nodes)
    res[f"{path}::<module>"] = (1, len(text.splitlines()) or 1)
    return res


def py_param_check(stage, root, recs):
    """Python param records vs the ``ast`` signature: names with sigils, indexes, the receiver (-1: the first
    parameter of a function directly in a class body, unless @staticmethod; spec §5.4)."""
    syms = {r["id"]: r for r in recs if r["type"] == "symbol"}
    params = defaultdict(list)
    for r in recs:
        if r["type"] == "param":
            params[r["symbolId"]].append((r["index"], r["name"]))
    trees, bad = {}, []
    pre = len(root) + 1 if root != "." else 0
    for s in syms.values():
        if s["kind"] not in ("function", "method", "accessor", "lambda"):
            continue
        rel = s["path"][pre:]
        if rel not in trees:
            trees[rel] = ast.parse((stage / rel).read_text(encoding="utf-8"))
        node = None
        for n in ast.walk(trees[rel]):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and s["kind"] != "lambda" and n.name == s["name"] \
                    and min([d.lineno for d in n.decorator_list] + [n.lineno]) == s["lineStart"]:
                node = n
                break
            if isinstance(n, ast.Lambda) and s["kind"] == "lambda" and n.lineno == s["lineStart"]:
                node = n
                break
        if node is None:
            continue
        a = node.args
        names = [x.arg for x in a.posonlyargs + a.args]
        if a.vararg:
            names.append("*" + a.vararg.arg)
        names += [x.arg for x in a.kwonlyargs]
        if a.kwarg:
            names.append("**" + a.kwarg.arg)
        static = any(isinstance(d, ast.Name) and d.id == "staticmethod" for d in getattr(node, "decorator_list", []))
        in_class = s.get("parentId") and syms[s["parentId"]]["kind"] == "type"
        exp, idx = [], 0
        for k, nm in enumerate(names):
            if k == 0 and in_class and not static:
                exp.append((-1, nm))
            else:
                exp.append((idx, nm))
                idx += 1
        if sorted(params[s["id"]]) != exp:
            bad.append({"qname": s["qname"], "expected": exp, "got": sorted(params[s["id"]])})
    return bad


def callsite_line_check(stage, root, recs):
    """Plausibility: the line a callsite record names holds its calleeName as written (skips <dynamic>)."""
    syms = {r["id"]: r for r in recs if r["type"] == "symbol"}
    pre = len(root) + 1 if root != "." else 0
    texts, bad = {}, []
    for r in recs:
        if r["type"] != "callsite" or r["calleeName"] == "<dynamic>":
            continue
        path = syms[r["symbolId"]]["path"][pre:]
        if path not in texts:
            texts[path] = (stage / path).read_text(encoding="utf-8").split("\n")
        lines = texts[path]
        if not (1 <= r["line"] <= len(lines)) or r["calleeName"] not in lines[r["line"] - 1]:
            bad.append({"id": r["id"], "callee": r["calleeName"], "line": r["line"], "path": syms[r["symbolId"]]["path"]})
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lang", required=True, choices=["python", "javascript"])
    ap.add_argument("--stage", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--astgen")
    ap.add_argument("--json")
    a = ap.parse_args()
    stage = Path(a.stage)
    P = (lambda rel: rel) if a.root == "." else (lambda rel: f"{a.root}/{rel}")

    recs = [json.loads(l) for l in open(a.bundle, encoding="utf-8")]
    files = {r["path"] for r in recs if r["type"] == "file"}
    bsyms = {r["qname"]: (r["lineStart"], r["lineEnd"]) for r in recs if r["type"] == "symbol"}

    derived = {}
    for rel in sorted(f[len(a.root) + 1:] if a.root != "." else f for f in files):
        text = (stage / rel).read_text(encoding="utf-8")
        if a.lang == "python":
            d = py_file(P(rel), text)
        else:
            j = json.loads((Path(a.astgen) / (rel + ".json")).read_text(encoding="utf-8"))
            d = js_file(P(rel), text, j["ast"])
        derived.update(d)

    extra = {}
    if a.lang == "python":
        extra["paramMismatch"] = py_param_check(stage, a.root, recs)
    extra["callsiteLineWithoutCalleeName"] = callsite_line_check(stage, a.root, recs)
    only_src = sorted(set(derived) - set(bsyms))
    only_bundle = sorted(set(bsyms) - set(derived))
    span = sorted(q for q in set(derived) & set(bsyms) if derived[q] != bsyms[q])
    rep = {"lang": a.lang, "sourceSymbols": len(derived), "bundleSymbols": len(bsyms),
           "matching": len(set(derived) & set(bsyms)) - len(span),
           "onlyInSource": only_src, "onlyInBundle": only_bundle,
           "spanMismatch": [{"qname": q, "source": derived[q], "bundle": bsyms[q]} for q in span], **extra}
    if a.json:
        Path(a.json).write_text(json.dumps(rep, indent=1, sort_keys=True) + "\n")
    print(f"verify_qnames[{a.lang}]: source={len(derived)} bundle={len(bsyms)} qnameMatch={len(set(derived) & set(bsyms))} "
          f"spanMismatch={len(span)} onlyInSource={len(only_src)} onlyInBundle={len(only_bundle)} "
          f"paramMismatch={len(extra.get('paramMismatch', []))} "
          f"callsiteLineWithoutCalleeName={len(extra['callsiteLineWithoutCalleeName'])}")
    return 1 if (only_src or only_bundle or span or extra.get("paramMismatch")) else 0


if __name__ == "__main__":
    sys.exit(main())
