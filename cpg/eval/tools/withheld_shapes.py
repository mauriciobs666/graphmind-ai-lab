#!/usr/bin/env python3
"""withheld_shapes.py - the Q8 withheld-shape predicate for held-out scripts items (``docs/reviews/connected-code-graph-p1-closeout-ml.md``
``## Q8 ruling``; plan ``docs/plans/connected-code-graph-p1-closeout.md``).

A PURE FUNCTION of three inputs: the gold locator (``{"path": ..., "lines": [...]}`` of the impacted element), the snapshot text of that file, and
a language tag (``markdown`` / ``shell`` / ``python``, see :func:`language_tag`). It returns the withheld shape letter or ``None``. It reads no
engine output, no tier, no segment count, no uniqueness and no hit status, and imports nothing of ``ccg`` (stdlib only): the same answer under
any engine commit. The custodian runs it once on the held-out list and reports the withheld counts beside the hit counts.

Shapes (the ruling's family letters), each judged on EVERY gold line of the item (an item with one plain code line is not withheld):

* ``C``  markdown: the gold line is inside the YAML front matter (a hook ``command:`` key).
* ``B``  markdown: the gold line is inside a fenced block (the shipped contract excludes fences).
* ``J``  shell: the line launches a python package with ``-m`` (``exec "$PY" -m pkg``).
* ``M``  shell: the line is inside embedded Python (a heredoc fed to a python interpreter, or ``python -c '...'``).
* ``I``  shell: the line is inside a ``bash -c '...'`` string (the ruling's I-``bash-c``).
* ``K``  python: the line is inside a ``subprocess`` call whose command word or first argument is an indirect path (``str(NAME)``, a name or
  an attribute) instead of a literal.

Not withheld by design: G and H one-segment tails (built through the own-directory tier), A, D, E, F, L, and the I ``awk`` read.
"""
from __future__ import annotations

import ast
import re
from typing import Mapping

SHAPES = ("B", "C", "I", "J", "K", "M")

_EXT = {".md": "markdown", ".markdown": "markdown", ".sh": "shell", ".bash": "shell", ".py": "python"}
_FENCE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")
_HEREDOC = re.compile(r"<<-?\s*(['\"]?)(\w+)\1")
_PYTHON = re.compile(r"\bpython[\d.]*\b|\"?\$\{?(?:PY|PYTHON)\w*\}?\"?\s+-\s")
_DASH_C = re.compile(r"\b(?P<sh>(?:ba|z|da)?sh|python[\d.]*)\s+-c\s+(?P<q>['\"])")
_DASH_M = re.compile(r"""(?:\bpython[\d.]*|"\$\{?\w+\}?"|\$\{?\w+\}?)\s+-m\s+[A-Za-z_][\w.]*""")
_SUBPROCESS = {"run", "Popen", "call", "check_call", "check_output"}


def language_tag(path: str) -> str | None:
    """The language tag of a path by extension (``markdown`` / ``shell`` / ``python``); ``None`` for anything else."""
    for ext, tag in _EXT.items():
        if path.lower().endswith(ext):
            return tag
    return None


def _front_matter_end(lines: list[str]) -> int:
    """Index (0-based) of the closing ``---`` of a leading YAML front matter, else -1."""
    if not lines or lines[0].strip() != "---":
        return -1
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            return i
    return -1


def _fenced(lines: list[str]) -> set[int]:
    """1-based line numbers inside a fenced block (fence lines included)."""
    out: set[int] = set()
    mark = ""
    for i, ln in enumerate(lines, 1):
        m = _FENCE.match(ln)
        if mark:
            out.add(i)
            if m and m.group(1)[0] == mark[0] and len(m.group(1)) >= len(mark) and not ln.strip()[len(m.group(1)):].strip():
                mark = ""
        elif m:
            mark = m.group(1)
            out.add(i)
    return out


def _shell_embedded(lines: list[str]) -> tuple[set[int], set[int]]:
    """1-based line numbers of embedded Python (heredoc to a python interpreter, ``python -c '..'``) and of ``bash -c '..'`` strings."""
    py: set[int] = set()
    sh: set[int] = set()
    i = 0
    n = len(lines)
    while i < n:
        ln = lines[i]
        h = _HEREDOC.search(ln)
        if h and _PYTHON.search(ln[: h.start()]):
            tag = h.group(2)
            j = i + 1
            while j < n and lines[j].strip() != tag:
                py.add(j + 1)
                j += 1
            i = j + 1
            continue
        c = _DASH_C.search(ln)
        if c:
            target = py if c.group("sh").startswith("python") else sh
            target.add(i + 1)
            q = c.group("q")
            inside = ln[c.end():].count(q) % 2 == 0   # even: the string is still open after the opener's own quote
            j = i
            while inside and j + 1 < n:
                j += 1
                target.add(j + 1)
                if lines[j].count(q) % 2 == 1:
                    inside = False
            i = j + 1
            continue
        i += 1
    return py, sh


def _indirect(node: ast.expr) -> bool:
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "str" and len(node.args) == 1:
        return _indirect(node.args[0])
    return isinstance(node, (ast.Name, ast.Attribute))


def _subprocess_lines(text: str) -> set[int]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return set()
    out: set[int] = set()
    for n in ast.walk(tree):
        if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in _SUBPROCESS
                and isinstance(n.func.value, ast.Name) and n.func.value.id == "subprocess" and n.args):
            continue
        argv = n.args[0]
        if isinstance(argv, (ast.List, ast.Tuple)) and any(_indirect(e) for e in argv.elts[:2]):
            out.update(range(n.lineno, (n.end_lineno or n.lineno) + 1))
    return out


def withheld_shape(locator: Mapping, text: str, language: str) -> str | None:
    """The withheld shape letter (``B C I J K M``) of an item, or ``None`` (built, or not a withheld shape)."""
    gold = [int(x) for x in (locator.get("lines") or [])]
    lines = text.split("\n")
    if not gold or any(g < 1 or g > len(lines) for g in gold):
        return None
    if language == "markdown":
        end = _front_matter_end(lines)
        if end > 0 and all(1 < g <= end for g in gold):
            return "C"
        fenced = _fenced(lines)
        if all(g in fenced for g in gold):
            return "B"
    elif language == "shell":
        py, sh = _shell_embedded(lines)
        if all(g in py for g in gold):
            return "M"
        if all(g in sh for g in gold):
            return "I"
        if all(_DASH_M.search(lines[g - 1]) for g in gold):
            return "J"
    elif language == "python":
        sub = _subprocess_lines(text)
        if all(g in sub for g in gold):
            return "K"
    return None
