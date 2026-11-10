"""Indexes over the unit bundles and the evaluation of ``extract`` expressions (rule-pack spec §1.2, §1.4).

:class:`Corpus` is the S12 recognizer's ``Corpus`` (``cpg/spikes/vertical-slice/assemble.py``, the conformance oracle) widened
with the file, import and launch/config records the declarations need. Values are *alternatives*: strings with
:data:`~ccg.packrt.httpkey.UNKNOWN` standing for an unresolved piece.
"""
from __future__ import annotations

import itertools
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from ccg.packrt.httpkey import UNKNOWN

MAX_ALTERNATIVES = 64
MAX_REF_DEPTH = 8


class UnsupportedExpression(NotImplementedError):
    """A pack names an expression form this runtime does not evaluate (refused, never read as empty)."""


class Corpus:
    def __init__(self, bundles: list[list[dict]]):
        self.files: dict[str, dict] = {}
        self.symbols: dict[str, dict] = {}
        self.callsites: dict[str, dict] = {}
        self.params: dict[str, dict] = {}
        self.rets: dict[str, dict] = {}
        self.conds: dict[str, dict] = {}
        self.constants: dict[str, dict] = {}
        self.value_sets: dict[tuple[str, str], list] = {}
        self.data_by_use: dict[str, list[dict]] = defaultdict(list)
        self.decorators: list[dict] = []
        self.launch_facts: list[dict] = []
        self.config_entries: list[dict] = []
        self.imports_of_file: dict[str, list[dict]] = defaultdict(list)
        self.controlled_by: dict[str, list[str]] = defaultdict(list)
        self.rets_of: dict[str, list[str]] = defaultdict(list)
        self.callsites_of: dict[str, list[dict]] = defaultdict(list)
        self.callers_of: dict[str, list[dict]] = defaultdict(list)
        for records in bundles:
            for r in records:
                t = r["type"]
                if t == "file":
                    self.files[r["id"]] = r
                elif t == "symbol":
                    self.symbols[r["id"]] = r
                elif t == "callsite":
                    self.callsites[r["id"]] = r
                elif t == "param":
                    self.params[r["id"]] = r
                elif t == "ret":
                    self.rets[r["id"]] = r
                    self.rets_of[r["symbolId"]].append(r["id"])
                elif t == "cond":
                    self.conds[r["id"]] = r
                elif t == "constant":
                    self.constants[r["qname"]] = r
                elif t == "valueSet":
                    self.value_sets[(r["callSiteId"], r["arg"])] = r["values"]
                elif t == "data":
                    self.data_by_use[r["use"]].append(r)
                elif t == "decorator":
                    self.decorators.append(r)
                elif t == "controlledBy":
                    self.controlled_by[r["point"]].append(r["cond"])
                elif t == "import":
                    self.imports_of_file[r["fileId"]].append(r)
                elif t == "launchFact":
                    self.launch_facts.append(r)
                elif t == "configEntry":
                    self.config_entries.append(r)
        for cid in sorted(self.callsites):
            c = self.callsites[cid]
            self.callsites_of[c["symbolId"]].append(c)
            for t in c["targets"]:
                self.callers_of[t["symbolId"]].append(c)
        for lst in self.rets_of.values():
            lst.sort()

    # -- where a record lives ------------------------------------------------------------------
    def file_id_of(self, record: dict) -> str | None:
        """The file of a decorator (its target symbol's) or a callsite (its enclosing symbol's)."""
        owner = record.get("targetId") or record.get("symbolId")
        sym = self.symbols.get(owner)
        return sym["fileId"] if sym else None

    def unit_of_file(self, file_id: str | None) -> str | None:
        f = self.files.get(file_id)
        return f["unitId"] if f else None

    # -- values --------------------------------------------------------------------------------
    def alternatives(self, element: dict, _seen: tuple = (), _depth: int = 0) -> list[str]:
        """Strings an ``element`` can take; ``UNKNOWN`` marks an unresolved piece."""
        if "const" in element:
            return [str(element["const"])]
        if "ref" in element:
            qn = element["ref"]
            const = self.constants.get(qn)
            if const is None or qn in _seen or _depth >= MAX_REF_DEPTH:
                return [UNKNOWN]
            return self.value_alternatives(const["values"], _seen + (qn,), _depth + 1)
        if "concat" in element or "template" in element:
            parts = [self.alternatives(p, _seen, _depth) for p in element.get("concat") or element["template"]]
            out: list[str] = []
            for combo in itertools.product(*parts):
                out.append("".join(combo))
                if len(out) >= MAX_ALTERNATIVES:
                    return sorted(set(out)) + [UNKNOWN]
            return sorted(set(out))
        return [UNKNOWN]  # importRef, unknown

    def value_alternatives(self, values: list[dict], _seen: tuple = (), _depth: int = 0) -> list[str]:
        out: set[str] = set()
        for el in values:
            out.update(self.alternatives(el, _seen, _depth))
        return sorted(out)

    def arg_values(self, record: dict, path: str) -> list[dict] | None:
        """The value elements of argument ``path`` of a decorator or callsite record; ``None`` when absent."""
        if record["type"] == "decorator":
            for a in record["args"]:
                if a["arg"] == path:
                    return a["values"]
            return None
        return self.value_sets.get((record["id"], path))

    def arg_alternatives(self, record: dict, path: str) -> list[str] | None:
        values = self.arg_values(record, path)
        return None if values is None else self.value_alternatives(values)

    def slot_alternatives(self, callsite_id: str, path: str) -> list[str] | None:
        values = self.value_sets.get((callsite_id, path))
        return None if values is None else self.value_alternatives(values)

    # -- flow ----------------------------------------------------------------------------------
    def flows_into(self, use_id: str, slot_path: str):
        """Yield ``(data, def_path)`` for every ``data`` record that feeds ``use_id`` at ``slot_path``.

        A flow into a prefix of the slot (``a1`` for ``a1.method``) carries the remainder over.
        """
        for d in self.data_by_use.get(use_id, ()):
            if _is_path_prefix(d["toPath"], slot_path):
                yield d, d["fromPath"] + slot_path[len(d["toPath"]):]


def _is_path_prefix(shorter: str, longer: str) -> bool:
    return longer == shorter or (longer.startswith(shorter) and longer[len(shorter)] in ".[")


# ---------------------------------------------------------------------------------------------
# Expressions (§1.2).
# ---------------------------------------------------------------------------------------------


@dataclass
class Ctx:
    """What an expression reads: the matched record, the clause's regex hit and, for a lifted node, its resolved roles."""
    corpus: Corpus
    record: dict
    hit: Any = None
    target: dict | None = None
    lifted: dict[str, list[str]] | None = None


def slot_mode(expr: dict, default: str) -> str:
    """The slot's one ``onUnknown`` (mixed nested modes are a pack error, N-29), else ``default``."""
    found = _modes(expr)
    return next(iter(found)) if found else default


def _modes(expr: dict) -> set[str]:
    out = {expr["onUnknown"]} if "onUnknown" in expr else set()
    for e in expr.get("firstOf", ()):
        out |= _modes(e)
    return out


def eval_expr(expr: dict, ctx: Ctx) -> list[str]:
    """The alternatives of one ``extract`` expression, ``[]`` when it is empty (``firstOf`` falls through on ``[]``)."""
    if "firstOf" in expr:
        for e in expr["firstOf"]:
            got = eval_expr(e, ctx)
            if got:
                return got
        return _default(expr)
    if "arg" in expr:
        got = ctx.corpus.arg_alternatives(ctx.record, expr["arg"]) or []
    elif "literal" in expr:
        got = [expr["literal"]]
    elif "group" in expr:
        g = ctx.hit.group(expr["group"]) if ctx.hit is not None else None
        got = [] if g is None else [g.upper() if expr.get("upper") else g]
    elif "target" in expr:
        got = [ctx.target[expr["target"]]] if ctx.target else []
    elif "lifted" in expr:
        got = list((ctx.lifted or {}).get(expr["lifted"], []))
    else:
        raise UnsupportedExpression(f"expression {sorted(expr)} is not evaluated by the P1-16 runtime")
    return got or _default(expr)


def _default(expr: dict) -> list[str]:
    return [expr["default"]] if "default" in expr else []
