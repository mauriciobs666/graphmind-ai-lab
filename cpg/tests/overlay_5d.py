"""The 5d overlay of the slice's oracle (`docs/plans/connected-code-graph-p1-extraction.md` § 5d, "Oracle and stored bundles").

The S14a oracle (raw facts and stored bundles) is not regenerated (D9). Step 5d changes two things in `extract.sc`: each `cs` fact
gains `rt`, and an export-block constant (`export const X = v`, `jssrc2cpg`) becomes a `const` fact. This module holds what both slice
tests need to absorb that without touching the oracle:

* ``tests/data/p1_05_5d_extras.json``, hand-listed per unit and derived once from a live Joern run (the stored CPGs re-queried with
  the new script, Joern ``v4.0.579``): ``sites`` (the call sites whose Joern targets are not all type-backed: Joern node id of the
  ``cs`` fact, its ``rt``, the bundle call-site id, and the type-backed target symbol ids), ``constFacts`` (the new ``const``
  facts) and ``constRecords`` (the bundle ``constant`` records they produce);
* ``overlaid_facts``: the stored facts plus exactly those ``rt`` values and ``const`` facts, fed to the real ``assemble``;
* ``strip``: the raw facts of a live run with the overlay removed, for comparison with the oracle;
* ``ast_self_sites``: the independent reference for the ``self`` rule, a plain ``ast`` pass over the staged sources that never
  imports ``assemble``.
"""
from __future__ import annotations

import ast
import collections
import json
from functools import lru_cache
from pathlib import Path

CPG = Path(__file__).resolve().parent.parent
EXTRAS_FILE = CPG / "tests" / "data" / "p1_05_5d_extras.json"


@lru_cache(maxsize=1)
def extras() -> dict:
    return json.loads(EXTRAS_FILE.read_text(encoding="utf-8"))


def sites(tag: str) -> list[dict]:
    """The listed call sites of unit ``tag`` as dicts: ``cpg`` (the fact id), ``site`` (call-site id), ``rt``, ``kept`` (symbol ids)."""
    u = extras()["units"][tag]
    return [{"cpg": c, "site": s, "rt": u["rt"][i], "kept": list(k)} for c, s, i, k in u["sites"]]


def rt_table(tag: str) -> dict[str, str]:
    return {s["cpg"]: s["rt"] for s in sites(tag)}


def _canon(o: dict) -> str:
    return json.dumps(o, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def overlaid_facts(src: Path, tag: str, dst: Path) -> Path:
    """``src`` (the oracle's facts file) copied to ``dst`` with ``rt`` set on the listed ``cs`` facts and the listed ``const`` facts
    appended. Every listed fact id must exist in ``src`` (a stale table is an error, not a silent no-op)."""
    table = rt_table(tag)
    seen = set()
    out = []
    for line in Path(src).read_text(encoding="utf-8").splitlines():
        o = json.loads(line)
        if o["k"] == "cs" and o["id"] in table:
            assert "rt" not in o
            o["rt"] = table[o["id"]]
            seen.add(o["id"])
        out.append(json.dumps(o))
    assert seen == set(table), f"{len(set(table) - seen)} listed cs facts are not in {src}"
    out += [json.dumps(f) for f in extras()["units"][tag]["constFacts"]]
    Path(dst).write_text("\n".join(out) + "\n", encoding="utf-8")
    return Path(dst)


def strip(records: list[dict], tag: str) -> list[dict]:
    """The raw facts of a live run less the 5d overlay: ``rt`` removed from every ``cs`` fact, the listed ``const`` facts removed
    (each exactly once). Whatever else differs from the oracle stays a difference."""
    listed = collections.Counter(_canon(f) for f in extras()["units"][tag]["constFacts"])
    out = []
    for o in records:
        if o["k"] == "cs":
            o = {k: v for k, v in o.items() if k != "rt"}
        elif o["k"] == "const" and listed[_canon(o)] > 0:
            listed[_canon(o)] -= 1
            continue
        out.append(o)
    left = +listed
    assert not left, f"listed const facts missing from the live facts: {sorted(left)}"
    return out


def rt_of(records: list[dict]) -> dict[str, str]:
    return {o["id"]: o["rt"] for o in records if o["k"] == "cs" and "rt" in o}


# -- the independent reference for the `self` rule ---------------------------------------------------------------------------
def _decorators(fn) -> set[str]:
    names = set()
    for d in fn.decorator_list:
        if isinstance(d, ast.Name):
            names.add(d.id)
        elif isinstance(d, ast.Attribute):
            names.add(d.attr)
    return names


def _methods(cls: ast.ClassDef):
    return [m for m in cls.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))]


def ast_self_sites(stage: Path, root: str, *, strict: bool = True) -> collections.Counter:
    """``Counter[(repo path, line, name)]`` of the ``self.m(...)`` calls the ``self`` rule confirms, read from the sources under
    ``stage`` by ``ast`` alone: a call ``self.m()`` lexically in a method (not a staticmethod; its first parameter is ``self``; a
    nested def, lambda or class inside it is out) of class ``C``, where ``C`` defines ``m`` exactly once as a plain method (no
    property/setter decorator) and, when ``strict``, no other class anywhere in the unit defines a method named ``m``. With
    ``strict=False`` the last condition is dropped: that is the superset Joern's own type recovery can add (when it had already
    narrowed the site to ``C.m`` the in-set holds one class method although another class names ``m``)."""
    stage = Path(stage)
    defs: dict[str, int] = collections.Counter()
    classes: list[tuple[str, ast.ClassDef]] = []
    for p in sorted(stage.rglob("*.py")):
        rel = p.relative_to(stage).as_posix()
        tree = ast.parse(p.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.append((rel, node))
                for m in _methods(node):
                    defs[m.name] += 1
    out: collections.Counter = collections.Counter()
    for rel, cls in classes:
        own = collections.Counter(m.name for m in _methods(cls))
        plain = {m.name for m in _methods(cls) if not (_decorators(m) & {"property", "setter", "getter", "deleter"})}
        for f in _methods(cls):
            if "staticmethod" in _decorators(f) or not f.args.args or f.args.args[0].arg != "self":
                continue
            stack = list(f.body)
            while stack:
                n = stack.pop()
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
                    continue
                if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name)
                        and n.func.value.id == "self"):
                    m = n.func.attr
                    if own[m] == 1 and m in plain and (not strict or defs[m] == 1):
                        out[(f"{root}/{rel}" if root != "." else rel, n.func.end_lineno, m)] += 1
                stack.extend(ast.iter_child_nodes(n))
    return out
