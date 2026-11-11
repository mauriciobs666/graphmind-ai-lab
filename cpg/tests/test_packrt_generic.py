"""P1-16 guard: the runtime names no pack or framework (`ccg/packrt/*.py` reads `pack.yaml`, it has no `fastapi` branch).

A guard over code *text*, so it is built the way `claude/tdd-engineer/guard-testing-techniques.md` asks. The forbidden words are
every string leaf of the shipped packs the runtime serves (`fastapi`, `fetch`, `fastmcp`), tokenized, minus the rule-pack
schema's own vocabulary (property names, enum and const values); a regex leaf is stripped of its character classes and escapes
first, and a whole leaf (a flag such as `--port`) is also forbidden as a literal. The reader looks at every place a word can hide
(identifiers, attributes, keywords, string, bytes and f-string pieces, dict keys, defaults, comparisons, imports, `except ... as`,
`global`, match patterns), docstrings and comments excepted. The few tokens the runtime must spell are justified one by one in
ALLOWED, by file; the positive controls plant one leaf per leaf kind of the packs in a copy of `run.py`.
"""
import ast
import functools
import json
import re

import pytest

from packrt_helpers import CPG, PACKS
from ccg.rulepack import load_pack

SRC = sorted((CPG / "ccg" / "packrt").glob("*.py"))
SERVED = ("fastapi", "fetch", "fastmcp", "shell-mcp-call")
SCHEMA = CPG / "ccg" / "schemas" / "rule-pack-v1.schema.json"
TOKEN = re.compile(r"[A-Za-z_]\w*")
# Tokens the runtime has to spell although a served pack also spells them: (file, kind, token); "*" is any file.
ALLOWED = {
    ("*", "attr", "get"),                 # `dict.get`, the method; the HTTP verb as a string would be a `str` hit
    ("run.py", "str", "mount"),           # the `{mount}` slot name of the key template (rule-pack spec 1.5): read in run.py only
    ("run.py", "str", "http"),            # the core contract kind `http` (the key normalizer applies only to it); not a pack's word
    ("run.py", "str", "unresolved"),      # the registry resolution value `unresolved`; a core vocabulary word
    ("run.py", "name", "unresolved"),     # a local variable holding the key slots that carry the UNKNOWN marker
    ("run.py", "name", "mount"),          # the local that names the `{mount}` slot's value (same slot, same file)
    ("run.py", "str", "partial-key"),     # a registry reason code (`ccg.rulepack.reason_row`), not a pack's word; the pack names it
    ("lift.py", "str", "partial-key"),    # as above: the lifted hop's status reason
    ("mount.py", "str", "partial-key"),   # as above: the `unmounted` default's reason
    ("mount.py", "str", "router"),        # the `unmounted-router` evidence token the spec's mount evidence names (`how`)
}


def _walk_leaves(x, kind=None):
    """``(kind, leaf)`` for every string leaf of a pack, ``kind`` being the nearest dict key above it."""
    if isinstance(x, dict):
        for k, v in x.items():
            yield from _walk_leaves(v, k)
    elif isinstance(x, list):
        for i in x:
            yield from _walk_leaves(i, kind)
    elif isinstance(x, str):
        yield kind, x


def _tokens(leaf: str, kind: str | None) -> set[str]:
    if kind == "regex":
        leaf = re.sub(r"\\[A-Za-z]|\[[^\]]*\]", "", leaf)
    return {t for t in TOKEN.findall(leaf) if len(t) > 1}


def schema_vocabulary() -> set[str]:
    """Property names, enum and const values of the rule-pack schema, as whole strings and as tokens."""
    out: set[str] = set()

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k == "properties" and isinstance(v, dict):
                    out.update(v)
                if k == "enum":
                    out.update(i for i in v if isinstance(i, str))
                if k == "const" and isinstance(v, str):
                    out.add(v)
                walk(v)
        elif isinstance(x, list):
            for i in x:
                walk(i)

    walk(json.loads(SCHEMA.read_text()))
    return out | {t for w in out for t in TOKEN.findall(w)}


@functools.lru_cache(maxsize=None)
def leaves() -> tuple[tuple[str | None, str], ...]:
    return tuple(kl for n in SERVED for kl in _walk_leaves(load_pack(PACKS / n / "pack.yaml")))


@functools.lru_cache(maxsize=None)
def forbidden() -> tuple[set[str], set[str]]:
    """``(tokens, whole literals)`` a runtime source may not spell, apart from ALLOWED."""
    vocab = schema_vocabulary()
    tokens = {t for kind, leaf in leaves() for t in _tokens(leaf, kind)} - vocab
    literals = {leaf for _, leaf in leaves() if len(leaf) > 1 and leaf not in vocab}
    return tokens, literals


def code_words(source: str) -> set[tuple[str, str]]:
    """``(kind, word)`` for every name and string a program spells, docstrings excepted (comments are not in the AST).
    ``kind`` is ``attr`` for an attribute name, ``str`` for a string/bytes piece (each token and the whole piece), else ``name``."""
    tree = ast.parse(source)
    docs = {id(n.body[0].value) for n in ast.walk(tree)
            if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and n.body and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)}
    out: set[tuple[str, str]] = set()

    def names(kind, *ws):
        out.update((kind, w) for w in ws if w)

    for n in ast.walk(tree):
        if isinstance(n, ast.Constant) and isinstance(n.value, (str, bytes)) and id(n) not in docs:
            text = n.value if isinstance(n.value, str) else n.value.decode("latin-1")
            names("str", text, *TOKEN.findall(text))
        elif isinstance(n, ast.Name):
            names("name", n.id)
        elif isinstance(n, ast.Attribute):
            names("attr", n.attr)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names("name", n.name)
        elif isinstance(n, ast.arg):
            names("name", n.arg)
        elif isinstance(n, ast.keyword):
            names("name", n.arg)
        elif isinstance(n, ast.alias):
            names("name", *n.name.split("."), n.asname)
        elif isinstance(n, ast.ImportFrom):
            names("name", *(n.module or "").split("."))
        elif isinstance(n, ast.ExceptHandler):
            names("name", n.name)
        elif isinstance(n, (ast.Global, ast.Nonlocal)):
            names("name", *n.names)
        elif isinstance(n, (ast.MatchAs, ast.MatchStar)):
            names("name", n.name)
        elif isinstance(n, ast.MatchMapping):
            names("name", n.rest)
    return out


def hits(source: str, filename: str = "") -> set[str]:
    tokens, literals = forbidden()
    found = set()
    for kind, w in code_words(source):
        if (w in tokens or w in literals) and not (
                ("*", kind, w) in ALLOWED or (filename, kind, w) in ALLOWED):
            found.add(w)
    return found


def test_the_forbidden_words_are_derived_from_the_shipped_packs_and_leave_out_the_schemas_own_vocabulary():
    tokens, literals = forbidden()
    assert {"fastapi", "fetch", "fastmcp", "include_router", "APIRouter", "FastAPI", "call_tool", "uvicorn",
            "mcpServers", "streamable_http_app", "streamablehttp_client", "mount", "tool"} <= tokens
    assert {"--port", "kw.prefix", "mcp-call"} <= literals
    assert "eval" in tokens                                             # P1-46: `shell-mcp-call` is a served pack too
    assert not ({"callee", "regex", "provides", "kind", "decorator", "firstOf", "arg"} & tokens)   # the format's words


def test_the_runtime_sources_spell_no_pack_or_framework_word():
    assert SRC, "no packrt sources found"
    assert {p.name: sorted(hits(p.read_text(), p.name)) for p in SRC if hits(p.read_text(), p.name)} == {}


def test_every_allowed_token_is_really_spelled_by_the_file_that_is_allowed_it():
    """An ALLOWED row that nothing needs is a standing hole; each must be a word the named file does spell."""
    for fname, kind, word in ALLOWED:
        files = SRC if fname == "*" else [p for p in SRC if p.name == fname]
        assert files and any((kind, word) in code_words(p.read_text()) for p in files), (fname, kind, word)


@pytest.mark.parametrize("snippet", [
    'if pack["id"] == "fastapi": pass',                                  # a comparison against a string
    'x = {"callee": "include_router"}',                                 # a dict value
    'x = {"fetch": 1}',                                                 # a dict key
    "def f(callee='call_tool'): pass",                                  # a default
    "fetch = 1",                                                        # an identifier
    "x.APIRouter()",                                                    # an attribute
    "f(uvicorn=1)",                                                     # a keyword
    "x = f'{a}/fastmcp/{b}'",                                           # a piece of an f-string
    "import fastapi",                                                   # an import
    "from fastapi.routing import APIRouter",                            # an import from
    "import a as fastapi",                                              # an import alias
    "class FastAPI: pass",                                              # a class name
    "x = ('a', 'b', 'FastAPI')",                                        # a tuple member
    "x = b'fastapi'",                                                   # a bytes constant
    "try:\n    pass\nexcept E as fastapi:\n    pass",                  # an except-as name
    "def f():\n    global fastapi",                                     # a global name
    "match x:\n    case {'a': 1, **fastapi}: pass",                    # a match-mapping rest
    "match x:\n    case [1, *fastapi]: pass",                          # a match star
    "match x:\n    case fastapi: pass",                                # a match capture
    "x.endswith('.tool')",                                              # a glob/regex piece
    "x = '--port'",                                                     # a flag, whole
])
def test_the_reader_sees_every_form_a_word_can_hide_in(snippet):
    assert hits(snippet), snippet


def test_a_docstring_and_a_comment_may_name_a_framework():
    assert hits('"""Spec: fastapi."""\n\ndef f():\n    """include_router"""\n    # fetch\n    return 1\n') == set()


def test_mount_is_allowed_in_run_py_only_not_in_any_other_runtime_file():
    assert hits('x = "mount"', "run.py") == set()
    assert hits('x = "mount"', "launch.py") == {"mount"}
    assert hits('if site["calleeName"] == "mount": pass', "launch.py") == {"mount"}
    assert hits("x.get('a')", "launch.py") == set() and hits("x = 'get'", "launch.py") == {"get"}   # the method, not the verb


def _leaf_kinds() -> dict[str, str]:
    """One planted candidate per leaf kind (the key a leaf sits under) that run.py is not allowed to spell, from the packs."""
    out: dict[str, str] = {}
    for kind, leaf in sorted(leaves(), key=lambda kl: (str(kl[0]), kl[1])):
        if kind not in out and hits(f"x = {leaf!r}", "run.py"):
            out[kind] = leaf
    return out


def test_the_planted_leaf_kinds_cover_a_regex_a_glob_a_flag_an_argument_a_callee_and_a_command():
    assert {"regex", "glob", "flags", "arg", "callee", "basename", "prefix"} <= set(_leaf_kinds())


@pytest.mark.parametrize("kind", ["regex", "glob", "flags", "arg", "callee", "basename", "prefix"])
def test_a_token_of_every_leaf_kind_planted_in_a_copy_of_the_real_runtime_is_caught(kind):
    src = (CPG / "ccg" / "packrt" / "run.py").read_text()
    assert not hits(src, "run.py")                                      # the unplanted copy is clean
    leaf = _leaf_kinds()[kind]
    planted = src.replace("def run_packs(", f"def _x(site):\n    return site == {leaf!r}\n\n\ndef run_packs(", 1)
    assert planted != src and hits(planted, "run.py"), (kind, leaf)


def test_a_branch_planted_in_a_copy_of_the_real_runtime_is_caught():
    src = (CPG / "ccg" / "packrt" / "run.py").read_text()
    planted = src.replace("def run_packs(", 'def _x(pack):\n    if pack["id"] == "fastapi":\n        return 1\n\n\ndef run_packs(', 1)
    assert hits(planted, "run.py") == {"fastapi"}
