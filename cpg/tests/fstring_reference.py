"""The independent reference for the constant text of Python f-strings (P1-50 Part B): the stdlib `ast` over source files, never the normalizer.

For a `JoinedStr`: each run of constant text is one `{"const": <raw source segment>}` (`ast.get_source_segment` on the `Constant` node; on
Python 3.12 `ast` merges adjacent text, where Joern splits it only at a replacement field) and each `FormattedValue` is one
`{"unknown": "runtime-built"}`. The text stays source text: escapes (`\\n`), `{{` and `}}` are written as in the file.
"""
from __future__ import annotations

import ast
import collections
import io
import json
import re
import sys
import tokenize
from pathlib import Path

assert sys.version_info[:2] == (3, 12), "the reference relies on 3.12's `ast` (PEP 701 positions, adjacent text merged)"
UNKNOWN = {"unknown": "runtime-built"}
IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*$")
FIELD_CAP = 3                           # spec2 §1.1 depth cap on a value set's field path (the stored `arg`)


def pieces(src: str, node: ast.JoinedStr) -> list[dict]:
    out = []
    for v in node.values:
        if isinstance(v, ast.Constant):
            seg = ast.get_source_segment(src, v)
            assert seg is not None
            if out and "const" in out[-1]:            # not produced by 3.12's `ast`; kept so a future parser still merges the run
                out[-1] = {"const": out[-1]["const"] + seg}
            else:
                out.append({"const": seg})
        else:
            out.append(dict(UNKNOWN))
    return out


def is_implicit_concatenation(src: str, node: ast.JoinedStr) -> bool:
    """More than one string literal side by side (`f"a{x}" "b"`, a multi-line message): Joern writes one `stringExpressionList`, not a `formatString`."""
    depth = starts = 0
    for t in tokenize.generate_tokens(io.StringIO(ast.get_source_segment(src, node)).readline):
        if t.type == tokenize.FSTRING_START:
            starts += depth == 0
            depth += 1
        elif t.type == tokenize.FSTRING_END:
            depth -= 1
        elif t.type == tokenize.STRING and depth == 0:
            starts += 1
    return starts > 1


def _quote(key: str) -> str:
    return "." + key if IDENT.match(key) else "[" + json.dumps(key, ensure_ascii=False) + "]"


def _arg_joined(e: ast.expr, path: str, fields: int, out: list) -> None:
    """The JoinedStr values an argument expression holds at the value-set paths the normalizer writes: the argument itself, or the
    value of a dict literal's string key (at most `FIELD_CAP` keys deep)."""
    if isinstance(e, ast.JoinedStr):
        out.append((path, e))
    elif isinstance(e, ast.Dict) and fields < FIELD_CAP:
        for k, v in zip(e.keys, e.values):
            if isinstance(k, ast.Constant) and isinstance(k.value, str):
                _arg_joined(v, path + _quote(k.value), fields + 1, out)


class Found:
    """What the reference finds in one unit's sources. `calls`: (repo path, line of the callee, argument path) -> the list of
    (pieces, implicit?) of every f-string argument there; `constants`: qname -> pieces of a module-level `NAME = f"..."`;
    `decorators`: ((repo path, line, argument path), pieces) of f-strings that are arguments of a decorator call; `everywhere`: every f-string's pieces."""

    def __init__(self):
        self.calls: dict[tuple, list] = collections.defaultdict(list)
        self.constants: dict[str, list[dict]] = {}
        self.decorators: list[tuple] = []      # ((repo path, line, argument path), pieces)
        self.everywhere: list[list[dict]] = []     # the pieces of every f-string of the unit, wherever it sits (not one nested in another)


def scan(stage: Path, root: str) -> Found:
    stage = Path(stage)
    found = Found()
    for p in sorted(stage.rglob("*.py")):
        rel = p.relative_to(stage).as_posix()
        repo_path = f"{root}/{rel}" if root != "." else rel
        src = p.read_text(encoding="utf-8")
        tree = ast.parse(src)
        in_decorator = {id(n) for d in ast.walk(tree) if isinstance(d, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                        for dec in d.decorator_list for n in ast.walk(dec)}
        nested = {id(n) for j in ast.walk(tree) if isinstance(j, ast.JoinedStr) for v in j.values if isinstance(v, ast.FormattedValue)
                  for n in ast.walk(v)}
        found.everywhere += [pieces(src, n) for n in ast.walk(tree) if isinstance(n, ast.JoinedStr) and id(n) not in nested]
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                out: list = []
                for k, a in enumerate(node.args):
                    _arg_joined(a, f"a{k}", 0, out)
                for kw in node.keywords:
                    if kw.arg:
                        _arg_joined(kw.value, "kw" + _quote(kw.arg), 0, out)
                for path, j in out:
                    key = (repo_path, node.func.end_lineno, path)
                    if id(node) in in_decorator:
                        found.decorators.append((key, pieces(src, j)))
                    else:
                        found.calls[key].append((pieces(src, j), is_implicit_concatenation(src, j)))
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) \
                    and isinstance(node.value, ast.JoinedStr):
                found.constants[f"{repo_path}::{node.targets[0].id}"] = pieces(src, node.value)
    return found
