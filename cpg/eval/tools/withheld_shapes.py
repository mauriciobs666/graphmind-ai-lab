#!/usr/bin/env python3
"""withheld_shapes.py - the Q8 withheld-shape predicate for held-out scripts items (``docs/reviews/connected-code-graph-p1-closeout-ml.md``
``## Q8 ruling``; plan ``docs/plans/connected-code-graph-p1-closeout.md``).

A PURE FUNCTION of three inputs: the gold locator (``{"path": ..., "lines": [...]}`` of the impacted element), the snapshot text of that file, and
a language tag (``markdown`` / ``shell`` / ``python``, see :func:`language_tag`). It returns the withheld shape letter or ``None``. It reads no
engine output, no tier, no segment count, no uniqueness and no hit status, and imports nothing of ``ccg`` (stdlib only): the same answer under
any engine commit. The custodian runs it once on the held-out list and reports the withheld counts beside the hit counts.

Shapes (the ruling's family letters), each judged on EVERY gold line of the item (an item with one plain code line is not withheld):

* ``C``  markdown: the gold line is inside the YAML front matter (Markdown grammar as the shipped contract: container-nested fences, a backtick info string is no fence, a 200-line front matter cap; a hook ``command:`` key).
* ``B``  markdown: the gold line is inside a fenced block (the shipped contract excludes fences).
* ``J``  shell: the line launches a python package with ``-m`` (``exec "$PY" -m pkg``; any python launcher word, flags between it and ``-m``).
* ``M``  shell: the line is inside embedded Python (a heredoc fed to a python interpreter, or ``python -c '...'``).
* ``I``  shell: the line is inside a ``bash -c '...'`` string (the ruling's I-``bash-c``).
* ``K``  python: the line is inside a ``subprocess`` call whose command word or first argument is an indirect path (``str(NAME)``, a name or
  an attribute) instead of a literal. A literal-argv launch (``subprocess.run(["bash", "scripts/x.sh"])``) is not withheld by this predicate.

Not withheld by design: G and H one-segment tails (built through the own-directory tier), A, D, E, F, L, and the I ``awk`` read.
"""
from __future__ import annotations

import ast
import re
from typing import Mapping

_EXT = {".md": "markdown", ".markdown": "markdown", ".sh": "shell", ".bash": "shell", ".py": "python"}
_FRONT_MATTER_MAX_LINES = 200
_FM_KEY = re.compile(r"[A-Za-z_][\w-]{0,63}:")

# Launcher words. A python word: a literal or versioned interpreter, bare or behind a path, quoted or not (``python3``, ``"$VENV/bin/python"``);
# a python variable: any shell variable whose name contains ``py`` (``$PY``, ``"${PY[@]}"``, ``$VENV_PY``); a shell word: ``sh``/``bash``/``zsh``/
# ``dash``/``ksh`` behind an optional path, or ``$SHELL``/``$BASH``.
_PYWORD = r"""["']?(?:[^\s"']*/)?python[\d.]*["']?"""
_PYVAR = r"""["']?\$\{?\w*[Pp][Yy]\w*(?:\[[@*]\])?\}?["']?"""
_PY = rf"(?:{_PYWORD}|{_PYVAR})"
_SHWORD = r"""(?:["']?(?:[^\s"']*/)?(?:ba|z|da|k)?sh["']?|["']?\$\{?\w*(?:SHELL|BASH)\w*\}?["']?)"""
_START = r"(?:^|(?<=[\s;&|(`]))"
_FLAGS = r"(?:\s+-[A-Za-z]+(?:\s+(?!-)[^\s'\"<]+)?)*"            # interpreter flags between the launcher and -m / -c (-u, -I, -X dev)
_FENCE_LIKE = re.compile(r"^(`{3,}|~{3,})")
_HEREDOC = re.compile(r"(?<!<)<<(?!<)(-?)[ \t]*(['\"]?)(\w+)\2")
_PY_HEREDOC_HEAD = re.compile(rf"{_START}{_PY}{_FLAGS}(?:\s+-(?:\s.*)?)?\s*$")     # the command text before ``<<`` ends in a python launch
_DASH_C = re.compile(rf"{_START}(?P<sh>{_SHWORD}|{_PY}){_FLAGS}\s+-[A-Za-z]*c[ \t]+(?P<q>['\"])")
_DASH_M = re.compile(rf"{_START}{_PY}{_FLAGS}\s+-m\s+[A-Za-z_][\w.]*")
_SUBPROCESS = {"run", "Popen", "call", "check_call", "check_output"}


def language_tag(path: str) -> str | None:
    """The language tag of a path by extension (``markdown`` / ``shell`` / ``python``, case-insensitive); ``None`` for anything else."""
    low = path.lower()
    for ext, tag in _EXT.items():
        if low.endswith(ext):
            return tag
    return None


def _strip_container(line: str) -> str:
    """The text of a line after the block-container prefixes a fence may sit behind: indentation, ``>`` quotes and list markers (bullet or ordered)."""
    s = line
    for _ in range(8):
        t = s.lstrip(" \t")
        if t.startswith(">"):
            s = t[1:]
        elif t[:2] in ("- ", "* ", "+ "):
            s = t[2:]
        else:
            word = t.split(" ", 1)[0]
            if 1 < len(word) <= 10 and word[:-1].isdigit() and word[-1] in ".)" and len(t) > len(word):
                s = t[len(word) + 1:]
            else:
                return t
    return s.lstrip(" \t")


def _fence_run(line: str) -> tuple[str, int, str] | None:
    """``(char, length, rest)`` when the line is a fence run (three or more backticks or tildes after container prefixes), else ``None``."""
    t = _strip_container(line.rstrip("\r"))
    m = _FENCE_LIKE.match(t)
    if not m:
        return None
    n = len(t) - len(t.lstrip(t[0]))
    return t[0], n, t[n:]


def _front_matter_end(lines: list[str]) -> int:
    """The 1-based line of the closing ``---`` of a leading YAML front matter, else 0. Opens on line 1 with ``---``, closes within 200 lines, and
    the first non-blank line between is YAML-looking (``key:``, ``#``, ``- ``); otherwise the ``---`` is a horizontal rule."""
    if not lines or lines[0].rstrip() != "---":
        return 0
    for n in range(2, min(len(lines), _FRONT_MATTER_MAX_LINES) + 1):
        if lines[n - 1].rstrip() == "---":
            for body in lines[1:n - 1]:
                t = body.strip()
                if t:
                    return n if (_FM_KEY.match(t) or t.startswith("#") or t == "-" or t.startswith("- ")) else 0
            return 0
    return 0


def _fenced(lines: list[str]) -> set[int]:
    """1-based line numbers inside a fenced block (fence lines included), outside the front matter. A backtick fence whose info string holds a
    backtick is not a fence (CommonMark); a closer uses the opener's character, at least its length, and carries no text."""
    fm = _front_matter_end(lines)
    out: set[int] = set()
    fence: tuple[str, int] | None = None
    for i, ln in enumerate(lines, 1):
        if i <= fm:
            continue
        run = _fence_run(ln)
        if fence is not None:
            out.add(i)
            if run and run[0] == fence[0] and run[1] >= fence[1] and not run[2].strip():
                fence = None
        elif run and not (run[0] == "`" and "`" in run[2]):
            fence = (run[0], run[1])
            out.add(i)
    return out


def _in_comment(prefix: str) -> bool:
    """True when the text before a match sits in a ``#`` comment (a ``#`` at a word start)."""
    return bool(re.search(r"(?:^|\s)#", prefix))


def _shell_embedded(lines: list[str]) -> tuple[set[int], set[int]]:
    """1-based line numbers of embedded Python (heredoc to a python launcher, ``python -c '..'``) and of ``bash -c '..'`` strings."""
    py: set[int] = set()
    sh: set[int] = set()
    i = 0
    n = len(lines)
    while i < n:
        ln = lines[i].rstrip("\r")
        h = _HEREDOC.search(ln)
        if h and not _in_comment(ln[: h.start()]) and _PY_HEREDOC_HEAD.search(ln[: h.start()]):
            dash, tag = h.group(1), h.group(3)
            j = i + 1
            while j < n:
                body = lines[j].rstrip("\r")
                if (body.lstrip("\t") if dash else body) == tag:
                    break
                py.add(j + 1)
                j += 1
            i = j + 1
            continue
        c = _DASH_C.search(ln)
        if c and not _in_comment(ln[: c.start()]):
            is_py = bool(re.fullmatch(_PY, c.group("sh")))
            target = py if is_py else sh
            q = c.group("q")
            rest = ln[c.end():]
            inside = rest.count(q) % 2 == 0   # even: the string is still open after the opener's own quote
            if not inside and re.search(r"[;&|]\s*\S", rest[rest.find(q) + 1:] if q in rest else ""):
                i += 1                         # a one-line string followed by another command: the line is not wholly the string
                continue
            target.add(i + 1)
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
        if end and all(1 < g < end for g in gold):
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
