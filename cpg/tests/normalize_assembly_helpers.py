"""A whole unit's raw facts, hand-built, plus a real git repository: the fixture of the bundle-assembly tests (P1-05, 5b).

``py_facts`` supplies ``f``/``m``/``t``/``deco``; this module adds the kinds ``assemble`` also reads (``p``, ``r``, ``c``,
``cs``, ``const``, ``d``, ``cb``) for ONE source, ``SRC``, written so that it holds the three shapes the determinism
claim needs: a class defined twice whose members share a ``fullName``, repeated ``(point, cond)`` pairs, and a ``data``
edge whose use is its own definition (``use == def``). Nothing here computes an expected value.
"""
from __future__ import annotations

import json
import os
import random
import subprocess
import textwrap
from pathlib import Path

from normalize_helpers import py_facts

SRC = textwrap.dedent('''\
    PREFIX = "/api"


    class B:
        @property
        def x(self):
            return 1


    class B:
        @property
        def x(self):
            if self.a:
                if self.b:
                    return g(self.a)
            return 2

        @x.setter
        def x(self, v):
            pass


    def g(n):
        return g(n)
''')
KINDS = ("f", "m", "t", "p", "r", "c", "cs", "const", "d", "cb", "deco")
COMMIT_DATE = "2026-01-02T03:04:05+00:00"


def line_of(text: str, needle: str, nth: int = 1) -> int:
    hits = [i for i, ln in enumerate(text.splitlines(), start=1) if needle in ln]
    return hits[nth - 1]


def full_facts(text: str = SRC, file: str = "r.py") -> dict[str, list[dict]]:
    facts = py_facts({file: text})
    ms = facts["m"]

    def method(name: str, line: int) -> dict:
        return next(m for m in ms if m["name"] == name and m["l"] == line)

    mod = next(m for m in ms if m["name"] == "<module>")
    getter2 = method("x", line_of(text, "def x(self):", 2))
    setter2 = method("x", line_of(text, "def x(self, v):"))
    getter1 = method("x", line_of(text, "def x(self):", 1))
    g = method("g", line_of(text, "def g(n):"))
    L = lambda needle, nth=1: line_of(text, needle, nth)
    facts["p"] += [
        {"k": "p", "m": getter1["id"], "i": 0, "name": "self", "l": getter1["l"], "c": 11, "code": "self"},
        {"k": "p", "m": getter2["id"], "i": 0, "name": "self", "l": getter2["l"], "c": 11, "code": "self"},
        {"k": "p", "m": setter2["id"], "i": 0, "name": "self", "l": setter2["l"], "c": 11, "code": "self"},
        {"k": "p", "m": setter2["id"], "i": 1, "name": "v", "l": setter2["l"], "c": 17, "code": "v"},
        {"k": "p", "m": g["id"], "i": 1, "name": "n", "l": g["l"], "c": 7, "code": "n"},
    ]
    r_getter1 = {"k": "r", "m": getter1["id"], "id": "5001", "l": L("return 1"), "c": 9}
    r_call = {"k": "r", "m": getter2["id"], "id": "5002", "l": L("return g(self.a)"), "c": 13}
    r_two = {"k": "r", "m": getter2["id"], "id": "5003", "l": L("return 2"), "c": 9}
    r_g = {"k": "r", "m": g["id"], "id": "5004", "l": L("return g(n)"), "c": 5}
    facts["r"] += [r_getter1, r_call, r_two, r_g]
    c1 = {"k": "c", "m": getter2["id"], "id": "6001", "l": L("if self.a:"), "c": 12, "code": "self.a", "lab": "CALL"}
    c2 = {"k": "c", "m": getter2["id"], "id": "6002", "l": L("if self.b:"), "c": 16, "code": "self.b", "lab": "CALL"}
    facts["c"] += [c1, c2]
    arg_self_a = {"i": 1, "an": "", "f": [], "e": {"o": "attr", "b": {"o": "name", "n": "self"}, "f": "a"}}
    call = {"k": "cs", "m": getter2["id"], "id": "7001", "name": "g", "mfn": g["full"], "l": L("return g(self.a)"), "c": 20,
            "code": "g(self.a)", "dt": "STATIC_DISPATCH", "in": [{"id": g["id"], "full": g["full"]}], "out": [],
            "args": [arg_self_a]}
    rec = {"k": "cs", "m": g["id"], "id": "7002", "name": "g", "mfn": g["full"], "l": L("return g(n)"), "c": 12,
           "code": "g(n)", "dt": "STATIC_DISPATCH", "in": [{"id": g["id"], "full": g["full"]}], "out": [],
           "args": [{"i": 1, "an": "", "f": [], "e": {"o": "name", "n": "n"}}]}
    facts["cs"] += [call, rec]
    facts["const"].append({"k": "const", "m": mod["id"], "name": "PREFIX", "l": 1, "e": {"o": "lit", "c": "'/api'"}})
    dat = lambda **kw: {"k": "d", "m": getter2["id"], "tf": [], "ff": [], **kw}
    facts["d"] += [
        dat(uk="cs", uid="7001", th="a0", h="p0", dk="p", did=getter2["id"], di=0),
        dat(uk="r", uid="5002", th="ret", h="r", dk="cs", did="7001"),
        # use == def: the call's result feeds its own receiver (spec4 §2, §5.10): kept
        {"k": "d", "m": g["id"], "uk": "cs", "uid": "7002", "th": "recv", "tf": [], "h": "r", "ff": [], "dk": "cs", "did": "7002"},
        # one record id, two raw edges: three fields and four fields (the fourth is cut, so the status differs)
        dat(uk="cs", uid="7001", th="a0", tf=["a", "b", "c"], h="p0", dk="p", did=getter2["id"], di=0, ff=["q"]),
        dat(uk="cs", uid="7001", th="a0", tf=["a", "b", "c", "d"], h="p0", dk="p", did=getter2["id"], di=0, ff=["q"]),
    ]
    cb = lambda pk, pid, cond: {"k": "cb", "m": getter2["id"], "pk": pk, "pid": pid, "cond": cond}
    facts["cb"] += [cb("cs", "7001", "6002")] * 3 + [cb("cs", "7001", "6001"), cb("c", "6002", "6001"),
                                                      cb("c", "6001", "6001")]     # 3 equal pairs, a cond->cond, a self-edge
    return facts


def shuffled(facts: dict[str, list[dict]], seed: int, kinds=KINDS) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    out = {k: list(v) for k, v in facts.items()}
    for k in kinds:
        rng.shuffle(out[k])
    return out


def git_repo(path: Path, files: dict[str, str], *, message: str = "c1") -> str:
    """A git repository at ``path`` holding ``files`` in one commit (fixed dates); returns the commit id."""
    path.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "GIT_AUTHOR_DATE": COMMIT_DATE, "GIT_COMMITTER_DATE": COMMIT_DATE, "GIT_CONFIG_GLOBAL": os.devnull,
           "GIT_CONFIG_SYSTEM": os.devnull, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid"}
    for rel, text in files.items():
        (path / rel).parent.mkdir(parents=True, exist_ok=True)
        (path / rel).write_text(text, encoding="utf-8")
    run = lambda *a: subprocess.run(["git", "-C", str(path), *a], check=True, capture_output=True, text=True, env=env)
    run("init", "-q", "-b", "main")
    run("add", "-A")
    run("commit", "-q", "-m", message)
    return run("rev-parse", "HEAD").stdout.strip()


def git_history(path: Path, steps: list[tuple[str, dict[str, str]]]) -> list[str]:
    """A repository at ``path`` with one commit per ``(committer date, {file: text})`` step (each step rewrites its files);
    returns the commit ids, oldest first."""
    path.mkdir(parents=True, exist_ok=True)
    base = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull, "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@example.invalid", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid"}
    subprocess.run(["git", "-C", str(path), "init", "-q", "-b", "main"], check=True, capture_output=True, env=base)
    commits = []
    for n, (date, files) in enumerate(steps, start=1):
        env = {**base, "GIT_AUTHOR_DATE": date, "GIT_COMMITTER_DATE": date}
        for rel, text in files.items():
            (path / rel).parent.mkdir(parents=True, exist_ok=True)
            (path / rel).write_text(text, encoding="utf-8")
        for cmd in (["add", "-A"], ["commit", "-q", "-m", f"c{n}", "--allow-empty"]):
            subprocess.run(["git", "-C", str(path), *cmd], check=True, capture_output=True, env=env)
        commits.append(subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], check=True, capture_output=True, text=True,
                                      env=base).stdout.strip())
    return commits
