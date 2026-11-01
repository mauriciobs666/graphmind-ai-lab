"""FR-8 by construction: `decisionmodels/` reaches nothing but the standard library and itself.

`docs/requirements/typed-decision-models.md` FR-8 (synthetic inputs only; no real falkor-chat
thread or trace) is made a test here, not a promise (`docs/plans/typed-decision-models.md` §3.4):

- **An AST import allowlist** — stdlib (`sys.stdlib_module_names`) plus `decisionmodels` and
  relative imports. So `redis`, `falkordb`, falkor-chat's `falkorchat` and model-bench's
  `modelbench` (plan3 §2: decision-models cannot import model-bench) are all refused, and so is
  any third-party package. A call to `__import__`/`import_module` — by bare name, by attribute,
  or under an alias bound by `from importlib|builtins import … as …` — counts as an import; one
  whose module name is not a literal is refused, since the allowlist cannot see it.
- **A source scan of str and bytes literals** (docstrings exempt — they describe, they do not
  reach) for the real-data reaches: a path or name containing `falkor-chat`/`falkorchat`/
  `falkor_chat`, `falkordb` or the word `redis` (letter-bounded, so `redis://`, `redis-cli` and
  `REDIS_HOST` match while "predisposition" does not), the side-check pack
  `guard-judge-understanding` (plan2 §3.5: `decisionmodels/` never reads the 85 human-curated
  items; only `scripts/import_side_check_shingles.py` does, outside this scan), and the integer
  6379, the FalkorDB/Redis port a stdlib `socket` would need.

The scan covers `decisionmodels/` only (plan2 §3.5). It is a static tripwire over source text,
not a proof. It does **not** see: an import function reached through `getattr`, `vars()` or any
other indirection; `exec`/`eval`, `runpy` or `importlib.util` loading by path; a forbidden name or
port assembled at run time (`"red" + "is"`, `6000 + 379`, a port given as a string, another
port); a glob that only matches a forbidden path; a docstring read back as data; and `redis`
glued to letters (`myredis`).
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest

PACKAGE_DIR = Path(__file__).resolve().parents[1] / "decisionmodels"

ALLOWED_TOP_LEVEL: frozenset[str] = frozenset(sys.stdlib_module_names) | {"decisionmodels"}

#: Reach name -> pattern over a lower-cased, non-docstring str/bytes literal. `redis` is
#: letter-bounded so ordinary words containing it ("predisposition", "redistribuir") pass.
FORBIDDEN_FRAGMENTS: dict[str, re.Pattern[str]] = {
    "falkor-chat": re.compile(re.escape("falkor-chat")),
    "falkorchat": re.compile(re.escape("falkorchat")),
    "falkor_chat": re.compile(re.escape("falkor_chat")),
    "falkordb": re.compile(re.escape("falkordb")),
    "redis": re.compile(r"(?<![a-z])redis(?![a-z])"),
    "guard-judge-understanding": re.compile(re.escape("guard-judge-understanding")),
}

#: The FalkorDB/Redis port.
FORBIDDEN_INTS: frozenset[int] = frozenset({6379})

_IMPORT_CALL_NAMES = frozenset({"__import__", "import_module"})
_IMPORT_FUNCTION_HOMES = frozenset({"importlib", "builtins"})


def _docstring_nodes(tree: ast.Module) -> set[int]:
    ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                ids.add(id(body[0].value))
    return ids


def _check_module_name(name: str, where: str) -> list[str]:
    top = name.split(".", 1)[0]
    if top in ALLOWED_TOP_LEVEL:
        return []
    return [f"{where}: import of {name!r} is outside the allowlist (stdlib + decisionmodels)"]


def _import_call_names(tree: ast.Module) -> frozenset[str]:
    """The import functions' own names plus any alias `from importlib|builtins import` binds."""
    names = set(_IMPORT_CALL_NAMES)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in _IMPORT_FUNCTION_HOMES:
            for alias in node.names:
                if alias.name in _IMPORT_CALL_NAMES and alias.asname:
                    names.add(alias.asname)
    return frozenset(names)


def _literal_reaches(text: str, kind: str, where: str) -> list[str]:
    lowered = text.lower()
    return [
        f"{where}: {kind} literal reaches {name!r}"
        for name, pattern in FORBIDDEN_FRAGMENTS.items()
        if pattern.search(lowered)
    ]


def _call_name(func: ast.expr) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def violations_in_source(source: str, filename: str) -> list[str]:
    """Every FR-8 violation in one module's source, each as a `file:line: reason` string."""
    tree = ast.parse(source, filename=filename)
    docstrings = _docstring_nodes(tree)
    import_calls = _import_call_names(tree)
    found: list[str] = []
    for node in ast.walk(tree):
        where = f"{filename}:{getattr(node, 'lineno', '?')}"
        if isinstance(node, ast.Import):
            for alias in node.names:
                found += _check_module_name(alias.name, where)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                found += _check_module_name(node.module or "", where)
        elif isinstance(node, ast.Call) and _call_name(node.func) in import_calls:
            first = node.args[0] if node.args else None
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                found += _check_module_name(first.value, where)
            else:
                found.append(f"{where}: dynamic import with a non-literal module name")
        elif isinstance(node, ast.Constant) and id(node) not in docstrings:
            value = node.value
            if isinstance(value, str):
                found += _literal_reaches(value, "string", where)
            elif isinstance(value, bytes):
                found += _literal_reaches(value.decode("latin-1"), "bytes", where)
            elif isinstance(value, int) and not isinstance(value, bool):
                if value in FORBIDDEN_INTS:
                    found.append(f"{where}: integer literal {value} (FalkorDB/Redis port)")
    return found


def violations_in_tree(root: Path) -> list[str]:
    found: list[str] = []
    for path in sorted(root.rglob("*.py")):
        found += violations_in_source(path.read_text(encoding="utf-8"), str(path))
    return found


# --- the real package ----------------------------------------------------------------------------


def test_the_scan_sees_the_real_package() -> None:
    """Guards against a vacuous green: a moved or renamed package would scan zero files."""
    scanned = sorted(p.relative_to(PACKAGE_DIR).as_posix() for p in PACKAGE_DIR.rglob("*.py"))
    assert "__init__.py" in scanned


def test_decisionmodels_reaches_only_stdlib_and_itself() -> None:
    assert violations_in_tree(PACKAGE_DIR) == []


def test_the_allowlist_is_the_runtime_stdlib_registry_plus_the_package() -> None:
    assert ALLOWED_TOP_LEVEL == frozenset(sys.stdlib_module_names) | {"decisionmodels"}


# --- planted-red cases: each forbidden reach, in a scratch package -------------------------------

PLANTED_RED = {
    "import redis": ("import redis\n", "'redis'"),
    "import falkordb": ("import falkordb\n", "'falkordb'"),
    "from falkordb import": ("from falkordb import FalkorDB\n", "'falkordb'"),
    "falkor-chat server package": ("import falkorchat.decisions\n", "'falkorchat.decisions'"),
    "model-bench package": ("from modelbench import packs\n", "'modelbench'"),
    "third-party package": ("import requests\n", "'requests'"),
    "import_module literal": (
        "import importlib\nimportlib.import_module('redis')\n",
        "import of 'redis'",
    ),
    "__import__ literal": ("__import__('falkordb')\n", "import of 'falkordb'"),
    "dynamic import": (
        "import importlib\nname = 'x'\nimportlib.import_module(name)\n",
        "non-literal",
    ),
    "falkor-chat path": (
        "open('../falkor-chat/server/config/models.json')\n",
        "string literal reaches 'falkor-chat'",
    ),
    "side-check items path": (
        "from pathlib import Path\n"
        "Path('model-bench/packs/guard-judge-understanding/items.jsonl')\n",
        "string literal reaches 'guard-judge-understanding'",
    ),
    "redis url": ("URL = 'redis://localhost:6379'\n", "string literal reaches 'redis'"),
    "redis env name": ("import os\nos.environ['REDIS_HOST']\n", "string literal reaches 'redis'"),
    "falkordb container name": (
        "import subprocess\nsubprocess.run(['docker', 'exec', 'falkordb-dev', 'true'])\n",
        "string literal reaches 'falkordb'",
    ),
    "falkor_chat path": (
        "from pathlib import Path\nPath('falkor_chat')\n",
        "string literal reaches 'falkor_chat'",
    ),
    "falkorchat module path as string": (
        "TARGET = 'falkorchat.app'\n",
        "string literal reaches 'falkorchat'",
    ),
    "bytes literal": ("PAYLOAD = b'GRAPH.QUERY via redis'\n", "bytes literal reaches 'redis'"),
    "aliased import_module": (
        "from importlib import import_module as im\nim('modelbench')\n",
        "'modelbench'",
    ),
    "redis-cli subprocess": (
        "import subprocess\nsubprocess.run(['redis-cli', 'GRAPH.QUERY'])\n",
        "string literal reaches 'redis'",
    ),
    "falkordb port": ("import socket\nsocket.create_connection(('localhost', 6379))\n", "6379"),
}


@pytest.mark.parametrize("case", sorted(PLANTED_RED))
def test_each_forbidden_reach_is_refused(case: str, tmp_path: Path) -> None:
    source, expected = PLANTED_RED[case]
    package = tmp_path / "decisionmodels"
    package.mkdir()
    (package / "__init__.py").write_text('"""Clean."""\n', encoding="utf-8")
    (package / "planted.py").write_text(source, encoding="utf-8")
    found = violations_in_tree(package)
    assert found, f"{case}: the planted reach was not refused"
    assert all("planted.py" in v for v in found)
    assert any(expected in v for v in found), found


def test_what_fr8_allows_is_not_refused(tmp_path: Path) -> None:
    """The negative control: stdlib, the package itself, relative imports, and docstrings that
    *describe* falkor-chat or redis (they reach nothing), and ordinary words that merely contain
    `redis` all pass."""
    source = (
        '"""Mirrors falkor-chat\'s guard render; never touches redis or FalkorDB."""\n'
        "from __future__ import annotations\n"
        "import importlib\n"
        "import json\n"
        "import urllib.request\n"
        "from collections import Counter\n"
        "from decisionmodels import seeds\n"
        "from . import render\n"
        "from .points import guard_judge\n"
        "importlib.import_module('decisionmodels.render')\n"
        "PORT = 1234\n"
        "WORDS = ('predisposition', 'redistribuir', b'predisposition')\n"
        "\n"
        "class Labels:\n"
        '    """A falkor-chat-shaped verdict, built from synthetic text only."""\n'
        "\n"
        "    def vote(self) -> str:\n"
        '        """Not a redis call."""\n'
        "        return 'yes'\n"
    )
    package = tmp_path / "decisionmodels"
    package.mkdir()
    (package / "clean.py").write_text(source, encoding="utf-8")
    assert violations_in_tree(package) == []
