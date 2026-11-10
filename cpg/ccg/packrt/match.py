"""Rule ``match``: matchers (spec §1.3), the file ``scope`` gate and the decorator / callsite clauses (spec §1.2).

Matchers are the format's six: ``exact`` (a bare scalar), ``template``, ``glob``, ``basename``, ``prefix``, ``regex``. A path
operand is normalized once on both sides (steps 4-6) before ``exact``/``prefix``/``template`` (:func:`path_hit`).
"""
from __future__ import annotations

import re
from typing import Any

from ccg.packrt.extract import Corpus
from ccg.packrt.httpkey import normalize_path_rfc

__all__ = ["Hit", "UnsupportedSubject", "clause_hit", "path_hit", "scope_ok", "text_hit"]


class UnsupportedSubject(NotImplementedError):
    """A rule names a subject this runtime does not evaluate (``config``, ``launch``, ``symbol``, ``marker``, ``query``...)."""


class Hit:
    """A successful match; ``group(n)`` is a regex group of the matcher (a non-regex matcher has only group 0)."""

    def __init__(self, text: str, match: re.Match | None = None):
        self._text, self._m = text, match

    def group(self, n: int = 0) -> str | None:
        if self._m is not None:
            return self._m.group(n)
        if n == 0:
            return self._text
        raise IndexError(n)


def _glob_re(g: str) -> re.Pattern:
    out, i = "", 0
    while i < len(g):
        if g.startswith("**", i):
            out += ".*"
            i += 2
        elif g[i] == "*":
            out += "[^/]*"
            i += 1
        elif g[i] == "?":
            out += "."
            i += 1
        else:
            out += re.escape(g[i])
            i += 1
    return re.compile(out, re.S)


def _template_re(t: str) -> re.Pattern:
    out, i = "", 0
    while i < len(t):
        if t.startswith("{**}", i):
            out += ".+"
            i += 4
        elif t.startswith("{}", i):
            out += "[^/]+"
            i += 2
        else:
            out += re.escape(t[i])
            i += 1
    return re.compile(out, re.S)


def text_hit(m: Any, s: str, *, path: bool = False) -> Hit | None:
    """Apply one matcher (a bare scalar is ``exact``; a list is any-of) to ``s``; ``path`` makes ``prefix`` stop at ``/``."""
    if isinstance(m, list):
        for x in m:
            hit = text_hit(x, s, path=path)
            if hit:
                return hit
        return None
    if not isinstance(m, dict):
        return Hit(s) if s == m else None
    ((kind, v),) = [(k, v) for k, v in m.items() if k != "ignoreCase"]
    if kind == "exact":
        return Hit(s) if s == v else None
    if kind == "basename":
        return Hit(s) if s.rsplit("/", 1)[-1] == v else None
    if kind == "prefix":
        return Hit(s) if s == v or s.startswith(v + ("/" if path or v.startswith("/") else ".")) else None
    if kind == "glob":
        return Hit(s) if _glob_re(v).fullmatch(s) else None
    if kind == "template":
        return Hit(s) if _template_re(v).fullmatch(s) else None
    if kind == "regex":
        got = re.fullmatch(v, s, re.I if m.get("ignoreCase") else 0)
        return Hit(s, got) if got else None
    raise ValueError(f"unknown matcher {kind!r}")


def path_hit(m: dict, s: str) -> bool:
    """A path-like operand: operand and subject are each normalized once, from their original text."""
    ((kind, v),) = m.items()
    return bool(text_hit({kind: normalize_path_rfc(v)}, normalize_path_rfc(s)))


# ---------------------------------------------------------------------------------------------
# scope
# ---------------------------------------------------------------------------------------------


def _imports_ok(gates: list[dict], file_id: str, corpus: Corpus) -> bool:
    imports = corpus.imports_of_file.get(file_id, ())
    for gate in gates:
        for imp in imports:
            if not text_hit(gate["module"], imp["module"]):
                continue
            wanted = gate.get("names")
            if wanted is None or any(n["name"] in wanted for n in imp["names"]):
                return True
    return False


def scope_ok(scope: dict | None, file_id: str | None, corpus: Corpus) -> bool:
    """The file gate: ``imports`` (any entry), ``path``/``notPath`` (matchers over the file path), ``role`` (any of)."""
    if not scope:
        return True
    f = corpus.files.get(file_id, {})
    if "imports" in scope and not _imports_ok(scope["imports"], file_id, corpus):
        return False
    if "path" in scope and not text_hit(scope["path"], f.get("path", ""), path=True):
        return False
    if "notPath" in scope and text_hit(scope["notPath"], f.get("path", ""), path=True):
        return False
    if "role" in scope and f.get("role") not in scope["role"]:
        return False
    return True


# ---------------------------------------------------------------------------------------------
# clauses
# ---------------------------------------------------------------------------------------------


def _args_ok(entries: list[dict], record: dict, corpus: Corpus) -> bool:
    for e in entries:
        values = corpus.arg_values(record, e["arg"])
        if values is None:
            return False
        if "value" in e and not any("const" in v and text_hit(e["value"], str(v["const"])) for v in values):
            return False
    return True


def _target_ok(spec: dict, record: dict, corpus: Corpus) -> bool:
    sym = corpus.symbols.get(record["targetId"])
    if sym is None:
        return False
    if "kind" in spec and sym["kind"] not in _listed(spec["kind"]):
        return False
    if "name" in spec and not text_hit(spec["name"], sym["name"]):
        return False
    if "parentKind" in spec:
        parent = corpus.symbols.get(sym.get("parentId"))
        if parent is None or parent["kind"] not in _listed(spec["parentKind"]):
            return False
    return True


def _listed(v) -> list:
    return v if isinstance(v, list) else [v]


def clause_hit(subject: str, clause: dict, record: dict, corpus: Corpus) -> Hit | None:
    """Does the ``decorator`` or ``callsite`` ``clause`` hold on ``record``? The returned :class:`Hit` carries the regex
    groups of the clause's ``name`` (decorator) or ``callee`` (callsite) matcher."""
    if subject == "decorator":
        if "form" in clause and record["form"] not in _listed(clause["form"]):
            return None
        hit = text_hit(clause["name"], record["name"]) if "name" in clause else Hit(record["name"])
        if not hit:
            return None
        if "target" in clause and not _target_ok(clause["target"], record, corpus):
            return None
    elif subject == "callsite":
        hit = text_hit(clause["callee"], record["calleeName"]) if "callee" in clause else Hit(record["calleeName"])
        if not hit:
            return None
        if "callKind" in clause and record["callKind"] not in _listed(clause["callKind"]):
            return None
        if "resolution" in clause and record["resolution"] not in _listed(clause["resolution"]):
            return None
        if "externalTarget" in clause and not text_hit(clause["externalTarget"], record.get("externalTarget", "")):
            return None
        if "target" in clause and not any(text_hit(clause["target"], t["qname"]) for t in record["targets"]):
            return None
    else:
        raise UnsupportedSubject(f"match subject {subject!r} is not evaluated by the P1-16 runtime")
    if "args" in clause and not _args_ok(clause["args"], record, corpus):
        return None
    return hit
