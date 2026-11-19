"""P1-25 (7): no client holds a network credential or opens a socket except to FalkorDB, as the ONE import-boundary test (``-egress-spec3.md`` §10.1, I2).

Scope: every module under ``cpg/ccg/`` except ``cpg/ccg/broker/``, plus ``cpg/extractors/`` and ``cpg/clients/``, found by walking the trees (a module added later, ``cpg/ccg/front/``
included, is scanned without editing this file). Each is parsed with ``ast`` and every node is inspected (an import inside a function counts).

**Import names.** ``import a.b`` is judged as ``a.b``; ``from a import b`` (absolute or relative, ``from . import b`` resolved against the module's package) as ``a.b``
for every imported name (``a`` for ``*``), so ``from ccg import broker`` is ``ccg.broker``. A dotted attribute chain naming ``ccg.broker`` (``ccg.broker.x``) is judged too.

* **Network modules** (:data:`NETWORK`: ``socket``/``_socket``, ``ssl``, ``selectors``, ``http``, ``urllib`` but ``urllib.parse``, ``wsgiref``, ``asyncio``, ``anyio``,
  ``ctypes``, web frameworks and clients ...) are imported nowhere in scope.
* **``falkordb``/``redis``** only in :data:`STORE_CLIENTS`: the modules of ``ccg/store/``, ``ccg/engine/reader.py`` and ``clients/common.py`` (``raw_query``'s ``ccg_raw_<s>`` connection).
* **``ccg.broker``** only as ``ccg.broker.client`` and only in :data:`BROKER_CLIENTS`: ``ccg/refresh.py``, ``ccg/front/interpret.py``, ``ccg/front/summary.py``.
* **The ``mcp`` SDK** only in ``clients/mcp/server.py`` (stdio; an HTTP transport would listen on a socket).
* **No dynamic import or evaluation** (it hides every rule above): no import of :data:`DYNAMIC_MODULES` (``importlib`` but ``importlib.metadata``/``resources``,
  ``runpy``, ``pkgutil``, ``imp``, ``zipimport``, ``builtins``, ``code``); no name or attribute ``__import__``/``import_module``/``__builtins__``; no string equal to
  ``__import__``/``import_module`` (``getattr(x, "__import__")``); no call of ``eval``, ``exec`` or ``compile`` by name.
* **No network tool as a command** (:data:`NETWORK_TOOLS` ``curl``, ``wget``, ``nc``, ``ssh``, ``rsync`` ...; plus the pairs ``git clone|fetch|push|pull|ls-remote``,
  ``pip|uv|npm|... install|download|add``, and an interpreter with ``-c``):
  - every word of a multi-word string (a shell line: ``"FOO=1 curl x"``, ``"true; wget x"``), docstrings excepted;
  - every element of an argv-shaped list or tuple, one whose first element is a tool, a launcher wrapper (:data:`WRAPPERS`: ``sh``, ``env``, ``timeout``, ``git``,
    ``pip``, ``python`` ...) or ``sys.executable``;
  - every string, one word or many, inside a process-launching call (:data:`LAUNCHERS`: ``subprocess.*``, ``os.system``/``popen``/``exec*``/``spawn*``);
  - a bare tool name assigned to a variable (``TOOL = "curl"``), and a tool name concatenated from string literals (``"cu" + "rl"``).
  - a launcher bound to another name (``run = os.system``, a default ``go=subprocess.run``, ``from subprocess import run as go``): rule ``launcher-alias``. The reviewed-prose
    exemption below holds only while ``ManifestError`` is one class definition in ``ccg/manifest.py`` and bound nowhere else.
  **Not flagged (known limits, the static limit again):** a launcher passed as an argument (``map(os.system, cmds)``), held in a dict or list that is not an assignment
  value of the checked shape (``{"go": os.system}`` inside a call), wrapped by ``functools.partial``, fetched by ``getattr(os, "system")``, or reached through a module
  alias (``m = os; m.system(c)``, ``import os as m``): a call by the alias is judged by its call name only.

  **Not caught (the static limit):** a name assembled at run time from non-literal pieces (``"".join(parts)``, a value read from a file or the environment), and a tool
  word kept as data in a list that is not argv-shaped and never reaches a launcher call in the same expression. A tool name as data (a word list, a set of names) is
  not a command.

Every rule has named mutants, one per spelling, planted in a copy of the source by :func:`test_each_named_mutant_is_killed`, which runs the same scanner over it.
"""
from __future__ import annotations

import ast
import functools
import re
import shutil
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import pytest

CPG = Path(__file__).resolve().parent.parent
SCOPE = ("ccg", "extractors", "clients")
EXCLUDED = ("ccg/broker",)

#: modules that open or speak over a network socket, or reach native code that can (a dotted prefix matches its submodules); ``ALLOWED_SUBMODULES`` carves out the offline ones
NETWORK = ("socket", "_socket", "ssl", "_ssl", "selectors", "http", "urllib", "urllib3", "requests", "httpx", "aiohttp", "asyncio", "anyio", "trio", "socketserver",
           "wsgiref", "ftplib", "smtplib", "poplib", "imaplib", "nntplib", "telnetlib", "xmlrpc", "websocket", "websockets", "paramiko", "grpc", "uvicorn",
           "starlette", "tornado", "flask", "fastapi", "django", "gevent", "eventlet", "twisted", "boto3", "botocore", "pexpect", "webbrowser", "ctypes", "_ctypes",
           "cffi", "multiprocessing.connection", "multiprocessing.managers", "pycurl", "dns")
ALLOWED_SUBMODULES = ("urllib.parse",)
STORE_DRIVERS = ("falkordb", "redis")
STORE_CLIENTS = ("ccg/store/", "ccg/engine/reader.py", "clients/common.py")
BROKER = "ccg.broker"
BROKER_CLIENT = "ccg.broker.client"
BROKER_CLIENTS = ("ccg/refresh.py", "ccg/front/interpret.py", "ccg/front/summary.py")
MCP_SDK = "mcp"
MCP_SERVERS = ("clients/mcp/server.py",)
DYNAMIC_MODULES = ("importlib", "runpy", "pkgutil", "imp", "zipimport", "builtins", "code", "codeop")
IMPORTLIB_OK = ("importlib.metadata", "importlib.resources")
DYNAMIC_IMPORT_NAMES = ("__import__", "import_module", "__builtins__")
DYNAMIC_IMPORT_STRINGS = ("__import__", "import_module")
EVAL_CALLS = ("eval", "exec", "compile")
NETWORK_TOOLS = ("curl", "wget", "nc", "ncat", "netcat", "socat", "telnet", "ftp", "sftp", "scp", "ssh", "rsync", "aria2c", "lftp", "httpie")
#: (program, subcommands that reach the network)
NETWORK_SUBCOMMANDS = {"git": ("clone", "fetch", "push", "pull", "ls-remote"),
                       **{p: ("install", "download", "add", "get") for p in ("pip", "pip3", "uv", "npm", "npx", "pnpm", "yarn", "gem", "cargo", "go", "apt", "apt-get")}}
INTERPRETERS = ("python", "python3", "sys.executable", "node", "perl", "ruby", "bash", "sh")
#: first elements that make a list an argv: the program runs what follows
WRAPPERS = ("sh", "bash", "dash", "zsh", "env", "timeout", "nohup", "sudo", "xargs", "nice", "stdbuf", "setsid", "git", "pip", "pip3", "uv", "npm", "npx", "pnpm",
            "yarn", "python", "python3", "docker", "podman", "node")
#: (module, exception class, exact string) reviewed as prose, not commands: exempt only as the first argument of ``raise <class>(...)`` in that module; the same words
#: assigned to a variable, in any other string, or in a launch or argv, still fail
COMMAND_PROSE = {("ccg/manifest.py", "ManifestError", "git fetch is not enabled in P1: a repository must not name a `remote`")}
#: modules whose launchers must be called by their own name, never bound to another (``launcher-alias``)
LAUNCHER_MODULES = ("os", "subprocess", "pty", "asyncio")
#: call names that start a process (``subprocess``, ``os``, ``pty``, ``asyncio``)
LAUNCHERS = ("run", "Popen", "call", "check_call", "check_output", "getoutput", "getstatusoutput", "system", "popen", "execv", "execve", "execvp", "execvpe",
             "execl", "execle", "execlp", "execlpe", "spawnv", "spawnve", "spawnvp", "spawnvpe", "spawnl", "spawnle", "spawnlp", "spawnlpe", "posix_spawn",
             "posix_spawnp", "spawn", "create_subprocess_exec", "create_subprocess_shell")


@dataclass(frozen=True)
class Violation:
    path: str
    line: int
    rule: str
    what: str


def _under(name: str, prefix: str) -> bool:
    return name == prefix or name.startswith(prefix + ".")


def _allowed(rel: str, places: tuple[str, ...]) -> bool:
    return any(rel == p or (p.endswith("/") and rel.startswith(p)) for p in places)


def _package(rel: str) -> list[str]:
    return list(PurePosixPath(rel).with_suffix("").parts)[:-1]


def _absolute(rel: str, node: ast.ImportFrom) -> str:
    if not node.level:
        return node.module or ""
    pkg = _package(rel)
    base = pkg[:len(pkg) - (node.level - 1)] if node.level > 1 else pkg
    return ".".join(base + ([node.module] if node.module else []))


def _imported_names(rel: str, node: ast.Import | ast.ImportFrom) -> list[str]:
    """The dotted names an import statement is judged as: ``import a.b`` -> ``a.b``; ``from a import b, c`` -> ``a.b``, ``a.c``; ``from a import *`` -> ``a``."""
    if isinstance(node, ast.Import):
        return [a.name for a in node.names]
    mod = _absolute(rel, node)
    names = [f"{mod}.{a.name}" if mod else a.name for a in node.names if a.name != "*"]
    return names or [mod]


def _check_name(rel: str, name: str, line: int, out: list[Violation]) -> None:
    if any(_under(name, a) for a in ALLOWED_SUBMODULES):
        return
    if any(_under(name, n) for n in NETWORK):
        out.append(Violation(rel, line, "network-module", name))
    if any(_under(name, d) for d in STORE_DRIVERS) and not _allowed(rel, STORE_CLIENTS):
        out.append(Violation(rel, line, "store-driver-outside-allow-list", name))
    if _under(name, BROKER) and not (_under(name, BROKER_CLIENT) and _allowed(rel, BROKER_CLIENTS)):
        out.append(Violation(rel, line, "broker-import", name))
    if _under(name, MCP_SDK) and not _allowed(rel, MCP_SERVERS):
        out.append(Violation(rel, line, "mcp-sdk-outside-server", name))
    if any(_under(name, d) for d in DYNAMIC_MODULES) and not any(_under(name, ok) for ok in IMPORTLIB_OK):
        out.append(Violation(rel, line, "dynamic-import", name))


# ------------------------------------------------------------------------------------------------------------------ commands

def _str(node: ast.AST) -> str | None:
    """A string literal, or a ``+`` concatenation of string literals folded to one (``"cu" + "rl"``)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _str(node.left), _str(node.right)
        if left is not None and right is not None:
            return left + right
    return None


def _is_sys_executable(node: ast.AST) -> bool:
    return isinstance(node, ast.Attribute) and node.attr == "executable" and isinstance(node.value, ast.Name) and node.value.id == "sys"


def _words(text: str) -> list[str]:
    """Shell words of ``text``, each reduced to its basename (``/usr/bin/curl`` -> ``curl``); ``FOO=1``, quotes and ``; | & ( )`` separate words."""
    return [PurePosixPath(w).name or w for w in re.split(r"[\s;|&()<>`'\"]+", text) if w and "=" not in w]


def _hits(words: list[str]) -> list[str]:
    """The network-reaching commands in a word sequence: a tool anywhere, a (program, network subcommand) pair, or an interpreter given ``-c``."""
    out = [w for w in words if w in NETWORK_TOOLS]
    for i, w in enumerate(words):
        if w in NETWORK_SUBCOMMANDS and any(x in NETWORK_SUBCOMMANDS[w] for x in words[i + 1:]):
            out.append(f"{w} {next(x for x in words[i + 1:] if x in NETWORK_SUBCOMMANDS[w])}")
        if w in INTERPRETERS and "-c" in words[i + 1:]:
            out.append(f"{w} -c")
    return out


def _element_words(node: ast.AST) -> list[str]:
    if _is_sys_executable(node):
        return ["sys.executable"]
    text = _str(node)
    return _words(text) if text is not None else []


def _command_violations(rel: str, node: ast.AST, exempt: set[int]) -> list[Violation]:
    found: list[tuple[int, str]] = []
    text = _str(node)
    if text is not None and id(node) not in exempt and not isinstance(node, ast.BinOp) and len(text.split()) > 1:
        found += [(node.lineno, h) for h in _hits(_words(text))]                           # a shell line: every word
    elif isinstance(node, ast.BinOp) and text is not None:
        found += [(node.lineno, h) for h in _hits(_words(text))]                           # a concatenated literal: "cu" + "rl"
    if isinstance(node, (ast.List, ast.Tuple)) and node.elts:
        head = _element_words(node.elts[0])
        if head and (head[0] in NETWORK_TOOLS or head[0] in WRAPPERS or head[0] == "sys.executable"):     # argv-shaped: every element
            words = [w for e in node.elts for w in _element_words(e)]
            found += [(node.lineno, h) for h in _hits(words)]
    if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None and _str(node.value) is not None \
            and _str(node.value).strip() in NETWORK_TOOLS:                                  # a bare tool name kept for later: TOOL = "curl"
        found.append((node.lineno, _str(node.value).strip()))
    if isinstance(node, ast.Call):
        name = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else ""
        if name in LAUNCHERS:                                                               # inside a launch: every string, one word or many
            words = [w for sub in ast.walk(node) for w in _element_words(sub)]
            found += [(node.lineno, h) for h in _hits(words)]
    return [Violation(rel, line, "network-tool-command", what) for line, what in found]


def _docstrings(tree: ast.AST) -> set[int]:
    return {id(n.value) for n in ast.walk(tree) if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str)}


def _binds(tree: ast.AST, name: str) -> tuple[int, int]:
    """``(class definitions, every other binding)`` of ``name`` in the module: an assignment or other store, an import (``as`` included), a function definition."""
    classes = others = 0
    for n in ast.walk(tree):
        if isinstance(n, ast.ClassDef) and n.name == name:
            classes += 1
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name:
            others += 1
        elif isinstance(n, ast.Name) and n.id == name and isinstance(n.ctx, (ast.Store, ast.Del)):
            others += 1
        elif isinstance(n, ast.alias) and (n.asname or n.name.partition(".")[0]) == name:
            others += 1
        elif isinstance(n, ast.arg) and n.arg == name:
            others += 1
    return classes, others


def _reviewed_prose(rel: str, tree: ast.AST) -> set[int]:
    """The string nodes :data:`COMMAND_PROSE` exempts: the first argument of ``raise <class>("<text>")`` in the named module, with that exact text, where ``<class>`` is a
    class defined once in that module and bound nowhere else (an alias such as ``ManifestError = os.system`` would run the text)."""
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Raise) and isinstance(n.exc, ast.Call) and isinstance(n.exc.func, ast.Name) and n.exc.args:
            arg = n.exc.args[0]
            if (rel, n.exc.func.id, _str(arg)) in COMMAND_PROSE and _binds(tree, n.exc.func.id) == (1, 0):
                out.add(id(arg))
    return out


def _launcher_aliases(rel: str, tree: ast.AST) -> list[Violation]:
    """A launcher bound to another name instead of called (``run = os.system``, a default ``run=subprocess.run``, ``from subprocess import run as go``): the call by
    the new name is judged by its name, so the alias would hide it. A bare reference in an annotation or an ``except`` clause is not a binding and is not flagged."""
    out = []
    for n in ast.walk(tree):
        values: list[ast.AST] = []
        if isinstance(n, (ast.Assign, ast.AnnAssign, ast.NamedExpr)) and n.value is not None:
            values = [n.value]
        elif isinstance(n, ast.arguments):
            values = [d for d in (*n.defaults, *n.kw_defaults) if d is not None]
        elif isinstance(n, ast.ImportFrom) and n.module in LAUNCHER_MODULES:
            out += [Violation(rel, n.lineno, "launcher-alias", f"{n.module}.{a.name} as {a.asname}") for a in n.names if a.asname and a.name in LAUNCHERS]
        for v in values:
            for e in (v.elts if isinstance(v, (ast.Tuple, ast.List)) else [v]):
                dotted = _dotted(e) if isinstance(e, ast.Attribute) else None
                if dotted and dotted.partition(".")[0] in LAUNCHER_MODULES and dotted.rpartition(".")[2] in LAUNCHERS:
                    out.append(Violation(rel, e.lineno, "launcher-alias", dotted))
    return out


def _dotted(node: ast.AST) -> str | None:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        return ".".join([node.id, *reversed(parts)])
    return None


def scan_source(rel: str, source: str) -> list[Violation]:
    """The violations of one module, ``rel`` its path relative to ``cpg/`` (POSIX)."""
    tree = ast.parse(source, filename=rel)
    exempt = _docstrings(tree) | _reviewed_prose(rel, tree)
    inner = {id(n.value) for n in ast.walk(tree) if isinstance(n, ast.Attribute)}           # only the outermost attribute chain is judged as a dotted name
    out: list[Violation] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for name in _imported_names(rel, node):
                _check_name(rel, name, node.lineno, out)
        elif isinstance(node, ast.Name) and node.id in DYNAMIC_IMPORT_NAMES:
            out.append(Violation(rel, node.lineno, "dynamic-import", node.id))
        elif isinstance(node, ast.Attribute):
            if node.attr in DYNAMIC_IMPORT_NAMES:
                out.append(Violation(rel, node.lineno, "dynamic-import", node.attr))
            dotted = _dotted(node)
            if dotted and _under(dotted, BROKER) and id(node) not in inner:
                _check_name(rel, dotted, node.lineno, out)
        elif isinstance(node, (ast.Constant, ast.BinOp)) and _str(node) in DYNAMIC_IMPORT_STRINGS:
            out.append(Violation(rel, node.lineno, "dynamic-import", _str(node)))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in EVAL_CALLS:
            out.append(Violation(rel, node.lineno, "dynamic-eval", node.func.id))
        out.extend(_command_violations(rel, node, exempt))
    out.extend(_launcher_aliases(rel, tree))
    return out


def modules(root: Path = CPG) -> list[str]:
    """Every ``.py`` file of the scope under ``root``, as POSIX paths relative to it (``__pycache__`` and the broker excluded)."""
    found = []
    for top in SCOPE:
        for path in sorted((root / top).rglob("*.py")):
            rel = path.relative_to(root).as_posix()
            if "__pycache__" in path.parts or any(rel == e or rel.startswith(e + "/") for e in EXCLUDED):
                continue
            found.append(rel)
    return found


def scan_tree(root: Path = CPG) -> list[Violation]:
    out: list[Violation] = []
    for rel in modules(root):
        out.extend(scan_source(rel, (root / rel).read_text(encoding="utf-8")))
    return out


# ------------------------------------------------------------------------------------------------------------------ the boundary holds

def test_the_scope_is_walked_not_listed_and_is_not_vacuous():
    found = modules()
    for must in ("ccg/refresh.py", "ccg/engine/reader.py", "ccg/store/loader.py", "ccg/sandbox_audit.py", "clients/common.py", "clients/cli/main.py",
                 "clients/mcp/server.py", "extractors/indexers/run.py"):
        assert must in found, must
    assert not [m for m in found if m.startswith("ccg/broker/")]
    assert len(found) > 50 and all(m.split("/")[0] in SCOPE for m in found)


def test_no_module_in_scope_can_reach_the_network_outside_its_allow_list():
    assert scan_tree() == []


def test_the_allow_list_is_used_where_it_is_granted():
    """A rule that never fires on real code could be a scanner that sees nothing: the granted modules really import what they are granted."""
    granted = {rel: [v for v in scan_source("ccg/__x__.py", (CPG / rel).read_text(encoding="utf-8"))] for rel in
               ("ccg/refresh.py", "ccg/store/loader.py", "clients/common.py", "clients/mcp/server.py")}
    rules = {rel: {v.rule for v in vs} for rel, vs in granted.items()}
    assert rules == {"ccg/refresh.py": {"broker-import"}, "ccg/store/loader.py": {"store-driver-outside-allow-list"},
                     "clients/common.py": {"store-driver-outside-allow-list"}, "clients/mcp/server.py": {"mcp-sdk-outside-server"}}


def test_sandbox_audit_imports_nothing_from_the_broker():
    text = (CPG / "ccg/sandbox_audit.py").read_text(encoding="utf-8")
    assert not [v for v in scan_source("ccg/refresh.py", text) if v.rule == "broker-import"]      # even where a broker client import would be allowed
    assert "ccg.broker" not in text and "from .broker" not in text and "from . import broker" not in text


# ------------------------------------------------------------------------------------------------------------------ the named mutants

#: (name, module it is planted in, source appended to the module). Each must make the scan of the copy fail with the named rule.
MUTANTS = [
    ("import urllib.request as u inside a function", "clients/cli/main.py", "\ndef _m():\n    import urllib.request as u\n    return u\n", "network-module"),
    ("from urllib import request", "ccg/answer.py", "\nfrom urllib import request\n", "network-module"),
    ("import socket", "ccg/engine/bfs.py", "\nimport socket\n", "network-module"),
    ("from socket import create_connection", "extractors/indexers/run.py", "\nfrom socket import create_connection\n", "network-module"),
    ("import http.client", "ccg/manifest.py", "\nimport http.client\n", "network-module"),
    ("asyncio.open_connection", "clients/mcp/server.py", "\nimport asyncio\n", "network-module"),
    ("from importlib import import_module", "ccg/gitsafe.py", "\nfrom importlib import import_module\n", "dynamic-import"),
    ("importlib.util", "ccg/gitsafe.py", "\nimport importlib.util\n", "dynamic-import"),
    ("subprocess of curl by absolute path, via a variable", "extractors/joern/run.py", "\nimport subprocess\nCMD = ['/usr/bin/curl', 'x']\nsubprocess.run(CMD)\n",
     "network-tool-command"),
    ("subprocess of curl named by a variable", "clients/common.py", "\nimport subprocess\nTOOL = 'curl'\nsubprocess.run([TOOL, 'x'])\n", "network-tool-command"),
    ("subprocess of curl as a tuple argv", "extractors/indexers/run.py", "\nimport subprocess\nsubprocess.check_output(('wget', 'x'))\n", "network-tool-command"),
    ("ccg.broker outside the allow-list", "ccg/sandbox_audit.py", "\nfrom ccg.broker.client import BrokerClient\n", "broker-import"),
    ("ccg.broker in a client", "clients/common.py", "\nfrom ccg.broker import client\n", "broker-import"),
    ("ccg.broker in the MCP server, inside a function", "clients/mcp/server.py", "\ndef _m():\n    import ccg.broker.client\n", "broker-import"),
    ("a relative broker import", "ccg/answer.py", "\nfrom .broker import client\n", "broker-import"),
    ("a broker module other than the client, from refresh", "ccg/refresh.py", "\nfrom ccg.broker import server\n", "broker-import"),
    ("the broker package itself, from refresh", "ccg/refresh.py", "\nimport ccg.broker\n", "broker-import"),
    ("the broker policy, from the front's interpreter", "ccg/front/interpret.py", "\nfrom ccg.broker.policy import evaluate\n", "broker-import"),
    ("falkordb outside the allow-list", "ccg/engine/bfs.py", "\nfrom falkordb import FalkorDB\n", "store-driver-outside-allow-list"),
    ("redis in the CLI, inside a function", "clients/cli/main.py", "\ndef _m():\n    import redis\n", "store-driver-outside-allow-list"),
    ("falkordb in the MCP server", "clients/mcp/server.py", "\nimport falkordb\n", "store-driver-outside-allow-list"),
    ("the mcp SDK outside the server", "clients/cli/main.py", "\nfrom mcp.client.sse import sse_client\n", "mcp-sdk-outside-server"),
    ("an HTTP server framework in the MCP server", "clients/mcp/server.py", "\nimport uvicorn\n", "network-module"),
]


def _module_spellings(name: str, target: str, rule: str) -> list[tuple[str, str, str, str]]:
    """One planted source per import spelling of ``name``: ``import``, ``import as`` inside a function, ``from ... import``, ``from ... import as``, the dotted-parent form."""
    head, _, tail = name.rpartition(".")
    rows = [(f"import {name}", f"\nimport {name}\n"),
            (f"import {name} as z, in a function", f"\ndef _m():\n    import {name} as z\n    return z\n"),
            (f"from {name} import x", f"\nfrom {name} import x\n"),
            (f"from {name} import x as y, in a function", f"\ndef _m():\n    from {name} import x as y\n    return y\n")]
    if head:
        rows.append((f"from {head} import {tail}", f"\nfrom {head} import {tail}\n"))
    return [(f"{label} [{target}]", target, plant, rule) for label, plant in rows]


#: every banned module family x every import spelling (a new family is one entry here)
FAMILY_MUTANTS = [m for n in NETWORK for m in _module_spellings(n, "ccg/answer.py", "network-module")] + \
    [m for n in STORE_DRIVERS for t in ("ccg/engine/bfs.py", "clients/cli/main.py", "clients/mcp/server.py") for m in _module_spellings(n, t, "store-driver-outside-allow-list")] + \
    [m for t in ("clients/cli/main.py", "ccg/answer.py") for m in _module_spellings("mcp.client.sse", t, "mcp-sdk-outside-server")] + \
    [m for n in DYNAMIC_MODULES if n != "importlib" for m in _module_spellings(n, "ccg/units.py", "dynamic-import")] + \
    [m for n in ("importlib", "importlib.util", "importlib.machinery") for m in _module_spellings(n, "ccg/units.py", "dynamic-import")]

#: every spelling of a broker import: outside the grant, and inside ``ccg/refresh.py``'s grant for anything but ``ccg.broker.client``
BROKER_MUTANTS = [(f"{label} [{target}]", target, plant, "broker-import") for target, label, plant in [
    ("ccg/answer.py", "from ccg import broker", "\nfrom ccg import broker\n"),
    ("ccg/answer.py", "from . import broker", "\nfrom . import broker\n"),
    ("ccg/engine/bfs.py", "from .. import broker", "\nfrom .. import broker\n"),
    ("ccg/answer.py", "from .broker import client", "\nfrom .broker import client\n"),
    ("ccg/engine/bfs.py", "from ..broker.client import BrokerClient", "\nfrom ..broker.client import BrokerClient\n"),
    ("ccg/answer.py", "import ccg.broker.client", "\nimport ccg.broker.client\n"),
    ("ccg/answer.py", "import ccg.broker.client as bc, in a function", "\ndef _m():\n    import ccg.broker.client as bc\n    return bc\n"),
    ("ccg/answer.py", "from ccg.broker import *", "\nfrom ccg.broker import *\n"),
    ("ccg/answer.py", "attribute ccg.broker.client after import ccg", "\nimport ccg\n_c = ccg.broker.client\n"),
    ("clients/common.py", "from ccg import broker", "\nfrom ccg import broker\n"),
    ("ccg/refresh.py", "from ccg import broker (only .client is granted)", "\nfrom ccg import broker\n"),
    ("ccg/refresh.py", "from . import broker (only .client is granted)", "\nfrom . import broker\n"),
    ("ccg/refresh.py", "import ccg.broker (only .client is granted)", "\nimport ccg.broker\n"),
    ("ccg/refresh.py", "from ccg.broker import server", "\nfrom ccg.broker import server\n"),
    ("ccg/refresh.py", "from .broker import policy", "\nfrom .broker import policy\n"),
    ("ccg/refresh.py", "attribute ccg.broker.server", "\nimport ccg.broker.client\n_s = ccg.broker.server\n"),
    ("ccg/front/interpret.py", "from ccg.broker import policy", "\nfrom ccg.broker import policy\n"),
    ("ccg/front/interpret.py", "from .. import broker", "\nfrom .. import broker\n"),
]]

#: in-process dynamic import and evaluation, one spelling each
DYNAMIC_MUTANTS = [(label, "ccg/answer.py", plant, rule) for label, plant, rule in [
    ("__import__('socket')", "\n_s = __import__('socket')\n", "dynamic-import"),
    ("builtins.__import__", "\nimport builtins\n_s = builtins.__import__('socket')\n", "dynamic-import"),
    ("getattr(__builtins__, '__import__')", "\n_s = getattr(__builtins__, '__import__')('socket')\n", "dynamic-import"),
    ("getattr(x, '__imp' + 'ort__')", "\nimport os\n_s = getattr(os, '__imp' + 'ort__')\n", "dynamic-import"),
    ("vars(builtins)['__import__']", "\nimport builtins\n_s = vars(builtins)['__import__']\n", "dynamic-import"),
    ("importlib.import_module", "\nimport importlib\n_s = importlib.import_module('socket')\n", "dynamic-import"),
    ("import_module via importlib.metadata's binding", "\nimport importlib.metadata\n_s = importlib.import_module('socket')\n", "dynamic-import"),
    ("from importlib import import_module as f", "\nfrom importlib import import_module as f\n", "dynamic-import"),
    ("exec('import socket')", "\nexec('import socket')\n", "dynamic-eval"),
    ("eval(...)", "\n_s = eval('1')\n", "dynamic-eval"),
    ("compile(...)", "\n_c = compile('import socket', 'x', 'exec')\n", "dynamic-eval"),
    ("runpy.run_module", "\nimport runpy\nrunpy.run_module('socket')\n", "dynamic-import"),
    ("pkgutil.resolve_name", "\nimport pkgutil\n_s = pkgutil.resolve_name('socket.socket')\n", "dynamic-import"),
    ("__builtins__ held in a variable", "\n_b = __builtins__\n", "dynamic-import"),
    ("vars(__builtins__)[name]", "\nNAME = 'x'\n_f = vars(__builtins__)[NAME]\n", "dynamic-import"),
]]

#: process launches, one spelling each
COMMAND_MUTANTS = [(label, target, plant, "network-tool-command") for label, target, plant in [
    ("subprocess of curl (argv)", "ccg/gitsafe.py", "\nimport subprocess\nsubprocess.run(['curl', '-s', 'http://x'])\n"),
    ("subprocess of wget (shell text)", "ccg/staleness.py", "\nimport subprocess\nsubprocess.run('wget -q http://x', shell=True)\n"),
    ("curl by absolute path, via a variable", "extractors/joern/run.py", "\nimport subprocess\nCMD = ['/usr/bin/curl', 'x']\nsubprocess.run(CMD)\n"),
    ("os.system of wget", "ccg/units.py", "\nimport os\nos.system('wget http://x')\n"),
    ("os.execvp of curl, one word", "ccg/units.py", "\nimport os\nos.execvp('curl', ARGS)\n"),
    ("curl named by a variable", "clients/common.py", "\nimport subprocess\nTOOL = 'curl'\nsubprocess.run([TOOL, 'x'])\n"),
    ("tuple argv to check_output", "extractors/indexers/run.py", "\nimport subprocess\nsubprocess.check_output(('wget', 'x'))\n"),
    ("sh -c with curl after another command", "ccg/units.py", "\nimport subprocess\nsubprocess.run(['sh', '-c', 'true; curl x'])\n"),
    ("env-assignment prefix", "ccg/units.py", "\nimport subprocess\nsubprocess.run('FOO=1 curl x', shell=True)\n"),
    ("timeout wrapper", "ccg/units.py", "\nimport subprocess\nsubprocess.run('timeout 5 curl x', shell=True)\n"),
    ("env wrapper argv", "ccg/units.py", "\nCMD = ['env', 'X=1', 'wget', 'x']\n"),
    ("bash -c curl", "ccg/units.py", "\nimport subprocess\nsubprocess.run(['bash', '-c', 'curl x'])\n"),
    ("sys.executable -c", "ccg/units.py", "\nimport subprocess, sys\nsubprocess.run([sys.executable, '-c', 'print(1)'])\n"),
    ("python3 -c argv", "ccg/units.py", "\nCMD = ['python3', '-c', 'print(1)']\n"),
    ("git clone", "ccg/gitsafe.py", "\nimport subprocess\nsubprocess.run(['git', 'clone', 'https://x/y'])\n"),
    ("git fetch argv kept for later", "ccg/gitsafe.py", "\nCMD = ['git', '-C', 'r', 'fetch', 'origin']\n"),
    ("git push shell line", "ccg/gitsafe.py", "\n_C = 'git push origin main'\n"),
    ("pip install", "ccg/units.py", "\nimport subprocess\nsubprocess.run(['pip', 'install', 'x'])\n"),
    ("npm install shell line", "ccg/units.py", "\n_C = 'npm install left-pad'\n"),
    ("ssh as an argv element", "ccg/units.py", "\nimport subprocess\nsubprocess.run(['nice', 'ssh', 'host'])\n"),
    ("tool name concatenated from literals", "ccg/units.py", "\nimport subprocess\nsubprocess.run(['cu' + 'rl', 'x'])\n"),
    ("tool name concatenated, kept in a variable", "ccg/units.py", "\nTOOL = 'wg' + 'et'\n"),
    ("a launch whose argv starts with a variable, tool later", "ccg/units.py", "\nimport subprocess\nsubprocess.run([PROG, '-v', 'wget'])\n"),
    ("a launch keyword argument naming the tool", "ccg/units.py", "\nimport subprocess\nsubprocess.run(ARGS, cwd='/tmp', executable='/usr/bin/curl')\n"),
    ("git fetch in manifest.py, not the reviewed message", "ccg/manifest.py", "\nimport subprocess\nsubprocess.run(['git', 'fetch'])\n"),
    ("the reviewed message's words as a shell line elsewhere", "ccg/answer.py", "\n_C = 'git fetch is not enabled in P1: a repository must not name a `remote`'\n"),
    ("another git fetch line in manifest.py", "ccg/manifest.py", "\n_C = 'git fetch origin'\n"),
    # P1-25b review F7: the two scanner arms no other row pins, and the exemption held to its raise node
    ("a shell line concatenated from literals, kept in a variable", "ccg/units.py", "\n_M = 'cu' + 'rl -s http://x'\n"),
    ("a wrapper-led argv kept in a variable, launched later", "ccg/units.py", "\nimport subprocess\nCMD = ['git', 'clone']\nsubprocess.run(CMD)\n"),
    ("the reviewed message assigned in manifest.py and passed to os.system", "ccg/manifest.py",
     "\nimport os\nCMD = 'git fetch is not enabled in P1: a repository must not name a `remote`'\nos.system(CMD)\n"),
    ("the reviewed message raised as another exception in manifest.py", "ccg/manifest.py",
     "\ndef _m():\n    raise ValueError('git fetch is not enabled in P1: a repository must not name a `remote`')\n"),
    ("the reviewed raise in another module", "ccg/answer.py",
     "\nclass ManifestError(Exception):\n    pass\n\n\ndef _m():\n    raise ManifestError('git fetch is not enabled in P1: a repository must not name a `remote`')\n"),
    # P1-26 review m3: the exemption trusts the class name, so a rebinding of that name voids it
    ("ManifestError rebound to os.system beside the reviewed raise", "ccg/manifest.py",
     "\nimport os\nManifestError = os.system\n\n\ndef _m():\n    raise ManifestError('git fetch is not enabled in P1: a repository must not name a `remote`')\n"),
    ("ManifestError imported as os.system beside the reviewed raise", "ccg/manifest.py",
     "\nfrom os import system as ManifestError\n\n\ndef _m():\n    raise ManifestError('git fetch is not enabled in P1: a repository must not name a `remote`')\n"),
    ("ManifestError redefined as a function beside the reviewed raise", "ccg/manifest.py",
     "\ndef ManifestError(c):\n    return c\n\n\ndef _m():\n    raise ManifestError('git fetch is not enabled in P1: a repository must not name a `remote`')\n"),
]]

#: a launcher bound to another name is judged by the new name, so the binding itself is the violation
ALIAS_MUTANTS = [(label, target, plant, "launcher-alias") for label, target, plant in [
    ("a launcher assigned to a name", "ccg/units.py", "\nimport os\nrun_it = os.system\n"),
    ("a subprocess launcher assigned to a name", "ccg/units.py", "\nimport subprocess\nGO = subprocess.run\n"),
    ("a launcher assigned in a tuple", "ccg/units.py", "\nimport os\nA, B = os.system, os.popen\n"),
    ("a launcher as an annotated assignment", "ccg/units.py", "\nimport subprocess\nGO: object = subprocess.Popen\n"),
    ("a launcher as a default argument", "ccg/units.py", "\nimport subprocess\n\n\ndef _f(go=subprocess.run):\n    return go\n"),
    ("a launcher imported under another name", "ccg/units.py", "\nfrom subprocess import run as go\n"),
    ("a launcher bound by a walrus", "ccg/units.py", "\nimport os\nif (go := os.system):\n    pass\n"),
]]

MUTANTS = MUTANTS + FAMILY_MUTANTS + BROKER_MUTANTS + DYNAMIC_MUTANTS + COMMAND_MUTANTS + ALIAS_MUTANTS


def _copy_scope(dst: Path) -> None:
    for top in SCOPE:
        shutil.copytree(CPG / top, dst / top, ignore=shutil.ignore_patterns("__pycache__"))


@functools.lru_cache(maxsize=4)
def _walked(root: str) -> frozenset[str]:
    return frozenset(modules(Path(root)))


@functools.lru_cache(maxsize=64)
def _clean(rel: str, source: str) -> bool:
    return scan_source(rel, source) == []


@pytest.fixture(scope="module")
def copy_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("no-network-copy")
    _copy_scope(root)
    front = root / "ccg" / "front"
    front.mkdir(exist_ok=True)
    (front / "interpret.py").write_text("from ccg.broker.client import BrokerClient\n", encoding="utf-8")
    return root


def test_mutant_names_are_unique():
    names = [m[0] for m in MUTANTS]
    assert len(names) == len(set(names))


@pytest.mark.parametrize("name,rel,plant,rule", MUTANTS, ids=[m[0] for m in MUTANTS])
def test_each_named_mutant_is_killed(copy_root, name, rel, plant, rule):
    path = copy_root / rel
    original = path.read_text(encoding="utf-8")
    assert rel in _walked(str(copy_root)) and _clean(rel, original)                     # the walk reaches the module, and it is clean before the plant
    try:
        path.write_text(original + plant, encoding="utf-8")
        found = scan_source(rel, path.read_text(encoding="utf-8"))
        assert any(v.rule == rule for v in found), (name, found)
    finally:
        path.write_text(original, encoding="utf-8")


def test_the_future_front_passes_unchanged_with_its_granted_broker_imports(tmp_path):
    """``ccg/front`` (P1-26) is scanned by the same walk; its two model-calling modules may import ``ccg.broker.client`` in any of its spellings, the rest may not."""
    _copy_scope(tmp_path)
    front = tmp_path / "ccg" / "front"
    front.mkdir(exist_ok=True)
    files = {"__init__.py": "", "draft.py": "from dataclasses import dataclass\n", "interpret.py": "from ccg.broker.client import BrokerClient\n", "summary.py": "from ccg.broker import client\nimport ccg.broker.client\n",
             "parsers.py": "import re\nfrom urllib.parse import urlsplit\n", "resolve.py": "from ccg.engine import anchors\n",
             "cache.py": "from ccg.answer import cache_key\n"}
    for fname, text in files.items():
        (front / fname).write_text(text, encoding="utf-8")
    assert {m for m in modules(tmp_path) if m.startswith("ccg/front/")} == {f"ccg/front/{f}" for f in files}
    assert scan_tree(tmp_path) == []
    (front / "cache.py").write_text("from ccg.broker.client import BrokerClient\n", encoding="utf-8")
    assert {(v.path, v.rule) for v in scan_tree(tmp_path)} == {("ccg/front/cache.py", "broker-import")}


#: the families the P1-25b review named (F2), pinned as literals: the generated mutant table follows NETWORK/DYNAMIC_MODULES, so a family dropped from those
#: lists would drop its own mutants too; this list does not move with them
REVIEWED_BANS = ("socket", "_socket", "ssl", "selectors", "http", "urllib", "asyncio", "anyio", "ctypes", "wsgiref", "tornado", "flask", "fastapi", "webbrowser",
                 "runpy", "pkgutil", "imp", "zipimport", "importlib", "builtins")


@pytest.mark.parametrize("family", REVIEWED_BANS)
def test_each_reviewed_family_is_banned_in_every_spelling(family):
    for label, plant in [("import", f"\nimport {family}\n"), ("from", f"\nfrom {family} import x\n"),
                         ("in a function", f"\ndef _m():\n    import {family} as z\n    return z\n")]:
        assert scan_source("ccg/answer.py", plant), (family, label)


def test_the_allow_lists_are_exactly_the_specs():
    """``-egress-spec3.md`` §10.1: the store drivers for ``ccg/store/``, ``ccg/engine/reader.py`` and ``clients/common.py``; ``ccg.broker.client`` for three modules."""
    assert STORE_CLIENTS == ("ccg/store/", "ccg/engine/reader.py", "clients/common.py")
    assert BROKER_CLIENTS == ("ccg/refresh.py", "ccg/front/interpret.py", "ccg/front/summary.py")
    assert EXCLUDED == ("ccg/broker",) and SCOPE == ("ccg", "extractors", "clients")


def test_the_reviewed_prose_exemption_needs_manifest_error_to_be_a_class_defined_once_and_bound_nowhere_else():
    raise_ = f"\n\ndef _m():\n    raise ManifestError({next(iter(COMMAND_PROSE))[2]!r})\n"
    assert scan_source("ccg/manifest.py", "class ManifestError(ValueError):\n    pass\n" + raise_) == []
    for label, head in [("imported, not defined", "from ccg.other import ManifestError\n"), ("not defined at all", ""), ("defined twice", "class ManifestError(ValueError):\n    pass\n"
                                                                                                                      "class ManifestError(ValueError):\n    pass\n"),
                        ("rebound by assignment", "class ManifestError(ValueError):\n    pass\nManifestError = print\n")]:
        assert [v.rule for v in scan_source("ccg/manifest.py", head + raise_)] == ["network-tool-command"], label


def test_a_launcher_named_in_an_annotation_or_an_except_clause_or_called_by_its_own_name_is_not_an_alias():
    src = ("import subprocess\nimport os\n\n\ndef _f(p: subprocess.Popen) -> None:\n    try:\n        os.system(_c())\n    except subprocess.TimeoutExpired:\n        pass\n"
           "    p2: subprocess.Popen | None = None\n    from subprocess import run\n    run(['ls'])\n")
    assert [v for v in scan_source("ccg/units.py", src) if v.rule == "launcher-alias"] == []
