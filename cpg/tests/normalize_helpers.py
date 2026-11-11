"""Fixture helpers for the normalizer tests: raw facts of a small Python file, in the shape ``extract.sc`` writes.

``py_facts`` walks the file with the stdlib ``ast`` and emits the ``f``/``m``/``t``/``deco`` records the way the
real P1-04 output has them (checked against the stored slice facts: a class is a ``t`` of pk METHOD plus a ``<body>``
method of pk TYPE_DECL, a function's parent is the enclosing method, a decorator's ``c`` is its expression's 1-based
column, ``le`` overshoots the true end line by 2 as pysrc2cpg's does). Expected values in the tests are written by hand;
nothing here computes a qname.
"""
from __future__ import annotations

import ast
import random
from pathlib import Path

from extractors.joern.normalize import core

REPO = "r"


def py_facts(files: dict[str, str], *, shuffle: int | None = None) -> dict[str, list[dict]]:
    pool = list(range(1000, 3000))
    random.Random(7).shuffle(pool)       # CPG ids are not in source order: a sort by position must not lean on them
    ids_ = iter(pool)
    facts: dict[str, list[dict]] = {k: [] for k in ("f", "m", "t", "deco", "p", "r", "c", "cs", "const", "d", "cb")}
    for file, text in files.items():
        facts["f"].append({"k": "f", "name": file})
        mod = {"k": "m", "id": str(next(ids_)), "name": "<module>", "full": f"{file}:<module>", "file": file, "l": 1,
               "le": len(text.splitlines()) or 1, "c": 1, "pk": "NAMESPACE_BLOCK", "pid": "ns", "pfull": "", "code": ""}
        facts["m"].append(mod)
        facts["t"].append({"k": "t", "id": str(next(ids_)), "name": "<module>", "full": f"{file}:<module>", "file": file,
                           "l": 1, "c": 1, "pk": "NAMESPACE_BLOCK", "pid": "ns", "pfull": ""})
        counts: dict[tuple, int] = {}

        def visit(node, parent_m, parent_full):
            for ch in ast.iter_child_nodes(node):
                if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                    if isinstance(ch, ast.Lambda):
                        key = (parent_full, "<lambda>")
                        counts[key] = counts.get(key, -1) + 1
                        name = f"<lambda>{counts[key]}"
                    else:
                        name = ch.name
                    full = f"{parent_full}.{name}"
                    # pysrc2cpg (live probe, P1-05 gate): a redefined FUNCTION in one scope instance gets
                    # `$redefinitionN`; a redefined CLASS keeps the same fullName, and so do its members.
                    nth = counts[(parent_m["id"], name)] = counts.get((parent_m["id"], name), -1) + 1
                    if nth and not isinstance(ch, ast.ClassDef):
                        full += f"$redefinition{nth}"
                    m = {"k": "m", "id": str(next(ids_)), "name": name, "full": full, "file": file, "l": ch.lineno,
                         "le": ch.end_lineno + 2, "c": ch.col_offset + 1, "pk": parent_m["pk2"], "pid": parent_m["id"],
                         "pfull": parent_full, "code": ""}
                    if isinstance(ch, ast.ClassDef):
                        t = {"k": "t", "id": str(next(ids_)), "name": name, "full": full, "file": file, "l": ch.lineno,
                             "c": ch.col_offset + 1, "pk": "METHOD", "pid": parent_m["id"], "pfull": parent_full}
                        facts["t"].append(t)
                        meta = {"k": "t", "id": str(next(ids_)), "name": name + "<meta>", "full": full + "<meta>", "file": file,
                                "l": ch.lineno, "c": ch.col_offset + 1, "pk": "METHOD", "pid": parent_m["id"], "pfull": parent_full}
                        facts["t"].append(meta)
                        for synth in ("<metaClassCallHandler>", "<fakeNew>"):     # pysrc2cpg's class machinery, as in the slice
                            facts["m"].append({"k": "m", "id": str(next(ids_)), "name": synth, "full": f"{full}.{synth}",
                                               "file": file, "l": ch.lineno, "le": ch.lineno, "c": ch.col_offset + 1,
                                               "pk": "TYPE_DECL", "pid": meta["id"], "pfull": full + "<meta>", "code": ""})
                        for member in ch.body:
                            if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                facts["t"].append({"k": "t", "id": str(next(ids_)), "name": member.name + "<metaClassAdapter>",
                                                   "full": f"{full}.{member.name}<metaClassAdapter>", "file": file,
                                                   "l": member.lineno, "c": member.col_offset + 1, "pk": "TYPE_DECL",
                                                   "pid": meta["id"], "pfull": full + "<meta>"})
                        body = {"k": "m", "id": str(next(ids_)), "name": "<body>", "full": full + ".<body>", "file": file,
                                "l": ch.lineno, "le": ch.end_lineno + 2, "c": ch.col_offset + 1, "pk": "TYPE_DECL",
                                "pid": t["id"], "pfull": full, "code": ""}
                        facts["m"].append(body)
                        # a class member's parent is the class's <body> method (pk METHOD)
                        inner = dict(body, pk2="METHOD")
                        visit(ch, inner, full + ".<body>")
                        # the next sibling lines of the member's own parent are unaffected
                        continue
                    facts["m"].append({kk: vv for kk, vv in m.items()})
                    for o, d in enumerate(getattr(ch, "decorator_list", [])):     # `o`: 0-based, in source order on the target
                        facts["deco"].append({"k": "deco", "m": m["id"], "target": full, "o": o, "name": "cpg-name",
                                              "code": ast.unparse(d), "l": d.lineno, "c": d.col_offset + 1, "args": []})
                    visit(ch, dict(m, pk2="METHOD"), full)
                else:
                    visit(ch, parent_m, parent_full)

        visit(ast.parse(text), dict(mod, pk2="METHOD"), f"{file}:<module>")
    if shuffle is not None:
        rng = random.Random(shuffle)
        for v in facts.values():
            rng.shuffle(v)
    return facts


def builder(tmp_path: Path, files: dict[str, str], *, root: str = ".", shuffle: int | None = None) -> core.Builder:
    for rel, text in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return core.Builder("python", root, REPO, py_facts(files, shuffle=shuffle), core.Source(tmp_path))


def run_bounded(code: str, *, mem: int = 600_000_000, seconds: int = 20):
    """Run ``code`` in a fresh interpreter (cwd = cpg/) under an address-space cap and a wall-clock limit, so that an unbounded
    loop fails the calling test by an assertion on the result instead of exhausting the host. Returns the CompletedProcess."""
    import resource
    import subprocess
    import sys
    from pathlib import Path
    cpg = Path(__file__).resolve().parent.parent
    return subprocess.run([sys.executable, "-B", "-c", code], cwd=cpg, capture_output=True, text=True, timeout=seconds,
                          preexec_fn=lambda: resource.setrlimit(resource.RLIMIT_AS, (mem, mem)))
