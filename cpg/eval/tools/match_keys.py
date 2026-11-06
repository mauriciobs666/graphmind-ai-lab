#!/usr/bin/env python3
"""match_keys.py — the key normalizer, validator and leak-check library (S49 first half, P0-42a).

Spec: eval-protocol ``docs/plans/connected-code-graph-eval-protocol-ml.md`` §13.4 (EP) and
``docs/plans/connected-code-graph-eval-protocol2-ml.md`` §1-§2 (EP2, "step 8a"; K1-K48).

Library (pure, plain dicts; P1's gate harness imports these and never reimplements them):
  ``normalize_contract(loc, form="provider")`` EP2 §1.2   ``E(qname)`` EP §2.3
  ``h1_symbol_credit(x, e, lines)``  EP §5.2 H1           ``source_check_needed(e, lines, symbols)`` EP §5.4
  ``normalize_http_key(raw, form, mask=True)`` (``mask=False`` is the "pre-8a" form of EP2 §1.4)

Verbs (every one prints counts, ids and paths only; never item content):
  normalize          --writer KA|KB|KA2 --in F --out F [--log F] [--kit-src D --context C]   EP §13.4
  validate           --in F... --kit-src D --out-dir work [--decisions F]                     EP §7.1 V1-V3
  match              --in F... --out work/match.json [--ka2-strata S...]                      EP §6.2
  nearmiss           --match F --out work/nearmiss.jsonl                                      EP §6.3
  adjudicate         --match F --nearmiss F --verdicts F --out F --log work/adjudication.jsonl  EP §6.3, §13.4
  scope              --match F --context C --out F --out-of-scope F                           EP §7.1 V4, §3.7
  sample             --match F --qw F --eval-commit SHA --out-dir work                        EP §7.3
  renormalize-check  --in F                                                                   EP2 §1.5
  check-report       --report F --heldout F... --dev F... [--contact F]                       EP2 §1.4, EP §9.4

``match.json`` contract (what ``capture.py`` reads; keys are only ever added to it):
  one JSON object ``{schema: "ccg-match-v1", records, unclassified, stratumDisagreements, t}``.
  ``records`` is sorted by ``itemId``; each is ``{itemId, anchorId, impactedId, derivedStratum, members,
  sessions, strategies, history}``, plus ``inScope``/``counting`` once ``scope`` ran and ``absorbed`` (item ids)
  once ``adjudicate`` merged a ``same`` pair into it. ``sessions`` lists the writers that hold the item, in
  KA, KA2, KB order; ``strategies`` is sorted; ``history`` is a 3-character string, one character per session
  in (KA, KA2, KB) order, ``1`` where that session holds the item (Appendix A's tuples). ``t`` maps every
  derived stratum to the session count of EP §6.4: 3 where KA2 ran there (some KA2 record has that
  ``derivedStratum``, or ``--ka2-strata`` named it), else 2. Where ``t`` is 2 the KA2 column of ``history`` is
  not data: it is ``0`` because KA2 did not run, whereas where ``t`` is 3 a ``0`` means KA2 ran and missed it.
  An out-of-scope flow item stays in the file ``match`` and ``adjudicate`` write: ``scope`` rewrites it to its
  own ``--out`` with ``inScope`` set on every record, and ``capture.py`` reads THAT scoped file and keeps only
  records with ``inScope`` true (a record without the key is not scoped yet: refuse). Order of verbs:
  match, nearmiss, adjudicate, scope. ``spot.jsonl`` ends with ``{"record": "shortfall", ...}`` when short.

``--context`` is a JSON file: {"units": [{"rootPath","lang","ports"}], "cypherMcpTools": [...],
"flowAnchors": [qname...]} (what ``export_kit.py`` and S40 produce). Exit: 0 ok, 1 findings
(invalid items, drift, a leak), 2 usage, 4 missing input.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import itertools
import json
import os
import posixpath
import re
import sys
import tomllib
import unicodedata
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ccg import ids as _ids                      # noqa: E402  (the one hash implementation)
from ccg import keynorm as _keynorm              # noqa: E402  (the one 8a implementation, EP2 §1.1)
import export_kit as _kit                        # noqa: E402  (census-scope globs, shared with S48)

REPO_ID = "graphmind-ai-lab"
VERBS = ("GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "TRACE", "CONNECT")
STRATA = ("http", "mcp", "scripts", "wiring", "flow")
FLOW_HOPS = ("return-use", "callee-arg", "control", "shared-state")


def canon(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def nfc_strip(v):
    """NFC then strip every string, recursively (bundle spec §3.1)."""
    if isinstance(v, str):
        return unicodedata.normalize("NFC", v).strip()
    if isinstance(v, list):
        return [nfc_strip(x) for x in v]
    if isinstance(v, dict):
        return {nfc_strip(k): nfc_strip(x) for k, x in v.items()}
    return v


# =============================================================================================
# http keyNorm (EP2 §1.1): spec3 §6 steps 1, 4, 5, 6, 7, then 8a (the shared mask function)
# =============================================================================================

_UNRESERVED = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~")
_ESCAPE = re.compile(r"%([0-9A-Fa-f]{2})")
_PLACEHOLDERS = frozenset({"{}", "{**}", "{?}", "{#}"})
_WHOLE_PARAM = (                                    # step 7: only a segment that is wholly a parameter
    (re.compile(r"^\{[^{}/]*:path\}$"), "{**}"),
    (re.compile(r"^\{\*[^{}/]*\}$"), "{**}"),
    (re.compile(r"^\[\.\.\.[^\[\]/]*\]$"), "{**}"),
    (re.compile(r"^<path:[^<>/]*>$"), "{**}"),
    (re.compile(r"^\*$"), "{**}"),
    (re.compile(r"^\{[^{}/]*\}$"), "{}"),
    (re.compile(r"^:[A-Za-z_][A-Za-z0-9_]*$"), "{}"),
    (re.compile(r"^<[^<>/]*>$"), "{}"),
    (re.compile(r"^\[[^\[\]/]+\]$"), "{}"),
)


def _decode_unreserved(path: str) -> str:
    """Step 5: ONE pass. Escapes of unreserved characters are decoded, the hex of every other
    escape is uppercased. A pass can form a new escape from a stray ``%`` (K48): not decoded again."""
    def one(m):
        ch = chr(int(m.group(1), 16))
        return ch if ch in _UNRESERVED else "%" + m.group(1).upper()
    return _ESCAPE.sub(one, path)


def _collapse_and_resolve(path: str) -> str:
    """Step 6: collapse runs of ``/``; RFC 3986 §5.2.4 dot segments (a trailing ``.``/``..`` leaves
    the directory form; ``..`` at the root stays at the root)."""
    parts = re.sub(r"/{2,}", "/", path).split("/")[1:]
    out: list[str] = []
    last = len(parts) - 1
    for i, seg in enumerate(parts):
        if seg == ".":
            if i == last:
                out.append("")
        elif seg == "..":
            if out:
                out.pop()
            if i == last:
                out.append("")
        else:
            out.append(seg)
    return "/" + "/".join(out)


def _param_segment(seg: str) -> str:
    if seg in _PLACEHOLDERS:
        return seg
    for rx, repl in _WHOLE_PARAM:
        if rx.match(seg):
            return repl
    return seg


def normalize_http_key(raw: str, form: str = "provider", mask: bool = True) -> str:
    """``keyNorm`` of a writer's ``"<verb> <path>"``. ``mask=False`` stops after step 7 (the
    "pre-8a" form used by the leak check, EP2 §1.4). A string with no leading verb word is a bare
    path under the any-method verb ``*``. Raises ValueError for an unknown verb word."""
    if form not in ("provider", "consumer"):
        raise ValueError(f"form must be 'provider' or 'consumer', got {form!r}")
    s = nfc_strip(raw)
    if s.startswith("/") or " " not in s:
        verb, path = "*", s
    else:
        v, _, rest = s.partition(" ")
        if v.upper() in VERBS or v == "*":
            verb, path = v.upper(), rest.strip()
        else:
            raise ValueError(f"unknown verb word {v!r}")
    if not path.startswith("/"):                                   # step 4
        path = "/" + path
    path = _collapse_and_resolve(_decode_unreserved(path))         # steps 5, 6
    path = "/".join(_param_segment(x) for x in path.split("/"))    # step 7
    if mask:                                                       # step 8a
        path = _keynorm.mask_http_segments(path, "{#}" if form == "provider" else "{?}")
    return f"{verb} {path}"


_MCP_NS = re.compile(r"^mcp__(.+?)__(.+)$")


def normalize_contract(loc: dict, form: str = "provider") -> dict:
    """EP2 §1.2: a pure function over a plain dict; returns the normalized locator.
    ``http`` by §1.1 with 8a in the given ``form``; ``mcp_tool`` by EP §13.4 (``form`` has no effect)."""
    out = nfc_strip(dict(loc))
    kind, key = out.get("kind"), out.get("keyNorm")
    if kind == "http":
        out["keyNorm"] = normalize_http_key(key, form)
    elif kind == "mcp_tool":
        m = _MCP_NS.match(key or "")
        out["keyNorm"] = m.group(2) if m else key
    return out


def contract_id(loc: dict, form: str = "provider") -> str:
    n = normalize_contract(loc, form)
    return _ids.contract_point_id(n["kind"], n["keyNorm"])


def unserved_gold_candidate(loc: dict, served_ids) -> bool:
    """EP2 §1.2 (K39): True when the method owner must check a gold ``http`` anchor against source:
    its provider-form id is no key anchor / ``serves[]`` / listed-unit point (``served_ids``) AND it has a
    masked segment. Then, if the route is unserved, the owner sets the consumer form (``form='consumer'``)."""
    if loc.get("kind") != "http":
        return False
    n = normalize_contract(loc)["keyNorm"]
    return "{#}" in n and contract_id(loc) not in set(served_ids)


# =============================================================================================
# E(qname) and the hit-rule helpers (EP §2.3, §5.2 H1, §5.4)
# =============================================================================================

_FILE_SYMBOL = {".sh": "<script>", ".bash": "<script>", ".md": "<document>"}
_ANON_SEG = re.compile(r"^<anon>#\d+$")


def E(qname: str) -> str:
    """The eligible ancestor of a qname: every ``#k`` duplicate suffix removed, truncated before the
    first ``<anon>#N`` segment; if nothing is left, the file symbol (``<module>``, ``<script>``,
    ``<document>`` by the file's extension)."""
    path, sep, chain = qname.partition("::")
    if not sep:
        return qname
    segs: list[str] = []
    for seg in chain.split("."):
        if _ANON_SEG.match(seg):
            break
        segs.append(re.sub(r"#\d+$", "", seg))
    if not segs:
        segs = [_FILE_SYMBOL.get(os.path.splitext(path)[1].lower(), "<module>")]
    return f"{path}::{'.'.join(segs)}"


def _regions(x: dict) -> list[tuple[int, int]]:
    spans = [(x["lineStart"], x["lineEnd"])] if "lineStart" in x else []
    for r in x.get("ownedRegions") or []:
        spans.append((r[0], r[1]) if isinstance(r, (list, tuple)) else (r["lineStart"], r["lineEnd"]))
    return spans


def _contains(x: dict, lines) -> bool:
    return any(a <= l <= b for (a, b) in _regions(x) for l in lines)


def h1_symbol_credit(x: dict, e: dict, lines) -> bool:
    """EP §5.2 H1 for a symbol: ``E(qname(x)) = qname(e)`` AND x's span, or an owned region, holds
    some line of ``lines``. Both are required: there is no id-only branch (P2-2)."""
    return E(x["qname"]) == e["qname"] and _contains(x, lines)


def source_check_needed(e: dict, lines, symbols) -> bool:
    """EP §5.4's trigger: no snapshot symbol s with ``E(qname(s)) = qname(e)`` and a span or owned
    region holding some line of ``lines``."""
    return not any(h1_symbol_credit(s, e, lines) for s in symbols)


# =============================================================================================
# language, census scope, config grain (EP §2.2-§2.4, §3.3)
# =============================================================================================

_LANG_EXT = {".py": "python", ".sh": "shell", ".bash": "shell", ".md": "markdown",
             ".java": "java", ".kt": "kotlin", ".kts": "kotlin", ".cs": "csharp", ".go": "go",
             ".c": "c", ".h": "c", ".cc": "cpp", ".cpp": "cpp", ".cxx": "cpp", ".hpp": "cpp"}
_LANG_EXT.update({e: "javascript" for e in (".ts", ".tsx", ".mts", ".cts", ".js", ".jsx", ".mjs", ".cjs")})


def lang_of(path: str) -> str:
    return _LANG_EXT.get(os.path.splitext(path)[1].lower(), "unknown")


def in_census_scope(path: str) -> bool:
    if _kit.path_excluded(path) or _kit.is_test_path(path):
        return False
    if path.endswith(".md"):
        return _kit.is_prompt_md(path)
    return True


def keypath_segments(key_path: str) -> list[str]:
    """``a.b[0].c`` -> ['a','b','[0]','c']; a quoted ``["x.y"]`` is one segment (spec3 §5.17)."""
    segs, i, n = [], 0, len(key_path)
    while i < n:
        c = key_path[i]
        if c == ".":
            i += 1
        elif c == "[":
            j = key_path.find("]", i + 1)
            if key_path[i + 1:i + 2] == '"':
                j = key_path.find('"]', i + 2)
                j = j + 1 if j != -1 else -1
            j = n - 1 if j == -1 else j
            segs.append(key_path[i:j + 1])
            i = j + 1
        else:
            j = i
            while j < n and key_path[j] not in ".[":
                j += 1
            segs.append(key_path[i:j])
            i = j
    return segs


def join_keypath(segs: list[str]) -> str:
    out = ""
    for s in segs:
        out += s if s.startswith("[") else (("." if out else "") + s)
    return out


def config_format(path: str) -> str:
    base = path.rsplit("/", 1)[-1]
    if base == ".mcp.json":
        return "mcp-json"
    if _kit.glob_match("**/.kiro/agents/*.json", path):
        return "kiro-agent"
    if _kit.glob_match("**/.claude/settings*.json", path):
        return "claude-settings"
    if _kit.glob_match("mcp-monitor/*.toml", path):
        return "mcp-monitor"
    if _kit.any_glob(("compose*.y*ml", "docker-compose*.y*ml", "**/compose*.y*ml", "**/docker-compose*.y*ml"), path):
        return "compose"
    if base == ".env" or base.startswith(".env."):
        return "env"
    return "other"


_NOVALUE = object()


def config_grain(path: str, key_path: str, env_value: str | None = None, units=(), value=_NOVALUE) -> tuple[str, str]:
    """EP §2.4: ``(truncated keyPath, class)`` where class is ``server`` or ``contract``. The first
    pattern that is a segment-prefix of the keyPath wins and truncates it. For ``.env`` the class
    needs the variable's value (``env_value``) and the listed units' ports. Kiro ``includeMcpJson`` is a
    server entry only when its parsed ``value`` is ``True`` (when given): ``false`` falls through to the
    "any other key" row."""
    fmt = config_format(path)
    segs = keypath_segments(key_path)

    def idx(i):
        return i < len(segs) and segs[i].startswith("[")

    def take(n, cls):
        return join_keypath(segs[:n]), cls

    head = segs[0] if segs else ""
    if fmt in ("mcp-json", "kiro-agent") and head == "mcpServers" and len(segs) >= 2:
        return take(2, "server")
    if fmt == "kiro-agent":
        if head in ("tools", "allowedTools") and idx(1):
            return take(2, "contract")
        if head in ("toolAliases", "toolsSettings") and len(segs) >= 2:
            return take(2, "contract")
        if head == "includeMcpJson" and (value is _NOVALUE or value is True):
            return take(1, "server")
    if fmt == "claude-settings":
        if head == "enabledMcpjsonServers" and idx(1):
            return take(2, "server")
        if head == "permissions" and len(segs) >= 3 and segs[1] in ("allow", "ask", "deny") and idx(2):
            return take(3, "contract")
        if head == "hooks" and len(segs) >= 3 and idx(2):
            return take(3, "contract")
    if fmt == "mcp-monitor":
        if head == "server" and len(segs) >= 2:
            return take(2, "server")
        if head == "watch" and idx(1):
            return take(2, "contract")
    if fmt == "compose" and head == "services" and len(segs) >= 2:
        return take(2, "server")
    if fmt == "env":
        ports = {int(p) for u in units for p in (u.get("ports") or [])}
        cls = "contract"
        if env_value is not None:
            v = env_value.strip()
            m = re.match(r"^[A-Za-z][A-Za-z0-9+.\-]*://[^/:?#]*(?::(\d+))?", v)
            if m:
                port = int(m.group(1)) if m.group(1) else {"http": 80, "https": 443}.get(v.split(":", 1)[0].lower())
                cls = "server" if port in ports else "contract"
            elif v.isdigit() and int(v) in ports:
                cls = "server"
        return take(1, cls)
    return take(min(2, len(segs)), "contract")


def unit_of(path: str, units) -> dict | None:
    """The listed unit whose ``rootPath`` is the longest prefix of ``path`` (EP §3.7)."""
    hits = [u for u in units
            if u["rootPath"] == "." or path == u["rootPath"] or path.startswith(u["rootPath"] + "/")]
    return max(hits, key=lambda u: 0 if u["rootPath"] == "." else len(u["rootPath"]), default=None)


# =============================================================================================
# Python symbols (V2): bundle spec3 §4.1-§4.2, enough for Python
# =============================================================================================

def py_symbols(src: str) -> list[dict]:
    """Every symbol of a Python file as ``{qname_chain, lineStart, lineEnd}``: ``<module>``, classes,
    functions (decorators in the span), lambdas as ``<anon>#N``, ``#k`` duplicates, ``<get>``/
    ``<set>``/``<del>`` accessors. ``@overload`` stubs are not symbols. Decorators, defaults and
    annotations belong to the ENCLOSING scope (spec3 §4.1 rule 2)."""
    tree = ast.parse(src)
    nlines = max(1, len(src.splitlines()))
    out = [{"chain": "<module>", "lineStart": 1, "lineEnd": nlines}]
    dup: dict = {}
    anon: dict = {}

    def seg(chain, name, acc=()):
        key = (chain, name, acc)
        k = dup[key] = dup.get(key, 0) + 1
        return (name + (f"#{k}" if k > 1 else ""),) + acc

    def visit_list(nodes, chain):
        for n in nodes:
            visit(n, chain)

    def visit(node, chain):
        if isinstance(node, ast.ClassDef):
            visit_list(node.decorator_list, chain)
            visit_list(node.bases + [k.value for k in node.keywords], chain)
            new = chain + seg(chain, node.name)
            out.append({"chain": ".".join(new), "lineStart": min([d.lineno for d in node.decorator_list] + [node.lineno]),
                        "lineEnd": node.end_lineno})
            visit_list(node.body, new)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            decos = [ast.unparse(d) for d in node.decorator_list]
            visit_list(node.decorator_list, chain)
            a = node.args
            visit_list([d for d in a.defaults + a.kw_defaults if d is not None], chain)
            visit_list([x.annotation for x in a.posonlyargs + a.args + a.kwonlyargs if x.annotation] +
                       ([node.returns] if node.returns else []), chain)
            acc: tuple = ()
            for t in decos:
                if t == "property":
                    acc = ("<get>",)
                elif re.fullmatch(r"\w+\.(setter|deleter)", t):
                    acc = ("<set>",) if t.endswith("setter") else ("<del>",)
            if any(t.split(".")[-1] == "overload" for t in decos):
                new = chain + (node.name,)
            else:
                new = chain + seg(chain, node.name, acc)
                out.append({"chain": ".".join(new), "lineStart": min([d.lineno for d in node.decorator_list] + [node.lineno]),
                            "lineEnd": node.end_lineno})
            visit_list(node.body, new)
        elif isinstance(node, ast.Lambda):
            a = node.args
            visit_list([d for d in a.defaults + a.kw_defaults if d is not None], chain)
            n = anon[chain] = anon.get(chain, 0) + 1
            new = chain + (f"<anon>#{n}",)
            out.append({"chain": ".".join(new), "lineStart": node.lineno, "lineEnd": node.end_lineno})
            visit(node.body, new)
        else:
            for child in ast.iter_child_nodes(node):
                visit(child, chain)

    visit(tree, ())
    return out


# =============================================================================================
# V1 / V2 / V3 (EP §7.1)
# =============================================================================================

_NAME_SEG = re.compile(r"^(?:[^\s.#<>]+(?:#\d+)?|<anon>#\d+|<init>|<module>|<script>|<document>|<get>|<set>|<del>)$")


def _line_count(text: str) -> int:
    return len(text.splitlines())


def parse_config(path: str, text: str):
    """The parsed document of a config file by format; ``.env`` -> {VAR: value}."""
    fmt = config_format(path)
    base = path.rsplit("/", 1)[-1]
    if fmt == "env":
        env = {}
        for line in text.splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                env[k.strip()] = v.strip().strip("\"'")
        return env
    if base.endswith(".toml"):
        return tomllib.loads(text)
    if base.endswith((".yaml", ".yml")):
        import yaml                                  # pyyaml, from the `dev` extra
        return yaml.safe_load(text)
    return json.loads(text)


def resolve_keypath(doc, key_path: str) -> bool:
    cur = doc
    for s in keypath_segments(key_path):
        if s.startswith("["):
            inner = s[1:-1]
            if inner.startswith('"') and inner.endswith('"'):
                key = inner[1:-1]
                if not isinstance(cur, dict) or key not in cur:
                    return False
                cur = cur[key]
            else:
                if not inner.isdigit() or not isinstance(cur, list) or int(inner) >= len(cur):
                    return False
                cur = cur[int(inner)]
        else:
            if not isinstance(cur, dict) or s not in cur:
                return False
            cur = cur[s]
    return True


def check_locator(loc: dict, kit: Path | None) -> list[str]:
    """V1 (lexical) and V2 (parsed) for one NORMALIZED locator: a list of failure names."""
    bad: list[str] = []
    t = loc.get("type")
    if t == "contract":
        if loc.get("kind") not in ("http", "mcp_tool"):
            bad.append("V3:kind")
        return bad
    if loc.get("repoId") != REPO_ID:
        bad.append("V1:repoId")
    path = loc.get("path") if t != "unit" else None
    if t == "unit":
        return bad
    if not isinstance(path, str) or not in_census_scope(path):
        return bad + ["V1:path-scope"]
    text = None
    if kit is not None:
        f = kit / path
        if not f.is_file():
            return bad + ["V1:path-not-in-kit"]
        text = f.read_text(encoding="utf-8", errors="replace")
    if t == "symbol":
        q = loc.get("qname", "")
        if not q.startswith(path + "::"):
            return bad + ["V1:qname-prefix"]
        chain = q[len(path) + 2:]
        segs = chain.split(".")
        if not chain or not all(_NAME_SEG.match(s) for s in segs):
            return bad + ["V1:qname-grammar"]
        if text is not None:
            for s in E(q)[len(path) + 2:].split("."):
                if not s.startswith("<") and not re.search(r"(?<![A-Za-z0-9_$])" + re.escape(s) + r"(?![A-Za-z0-9_$])", text):
                    bad.append("V1:name-not-in-file")
                    break
            if path.endswith(".py") and not bad:
                try:
                    syms = py_symbols(text)
                except SyntaxError:
                    return bad + ["V2:unparseable"]
                eq = E(q)
                if eq not in {E(f"{path}::{s['chain']}") for s in syms}:
                    bad.append("V2:qname-not-derived")
    elif t == "config":
        if text is not None:
            try:
                doc = parse_config(path, text)
            except Exception:
                return bad + ["V2:config-unparseable"]
            if not resolve_keypath(doc, loc.get("keyPath", "")):
                bad.append("V2:keypath-unresolved")
            elif config_format(path) == "kiro-agent" and loc.get("keyPath") == "includeMcpJson" \
                    and doc.get("includeMcpJson") is not True:           # EP §7.1 V2 / §2.4: only when true
                bad.append("V2:includeMcpJson-not-true")
    else:
        bad.append("V1:locator-type")
    return bad


def check_lines(loc: dict, lines, kit: Path | None) -> list[str]:
    """V1: every ``lines`` element lies in 1..lineCount; V2 (Python): in the span of some symbol whose E is the qname."""
    bad: list[str] = []
    if not isinstance(lines, list) or not lines or not all(isinstance(l, int) and not isinstance(l, bool) for l in lines):
        return ["V1:lines-shape"]
    if kit is None or loc.get("type") not in ("symbol", "config"):
        return bad
    f = kit / loc["path"]
    if not f.is_file():
        return bad
    text = f.read_text(encoding="utf-8", errors="replace")
    if any(l < 1 or l > _line_count(text) for l in lines):
        return ["V1:line-out-of-range"]
    if loc["type"] == "symbol" and loc["path"].endswith(".py"):
        try:
            syms = py_symbols(text)
        except SyntaxError:
            return ["V2:unparseable"]
        mine = [(s["lineStart"], s["lineEnd"]) for s in syms
                if E(f"{loc['path']}::{s['chain']}") == E(loc["qname"])]
        if not all(any(a <= l <= b for a, b in mine) for l in lines):
            bad.append("V2:line-outside-span")
    return bad


# =============================================================================================
# normalize (EP §13.4)
# =============================================================================================

STRATEGY = {"KA": "provider-first", "KA2": "provider-first", "KB": "consumer-first"}


class Invalid(Exception):
    pass


class NeedsKit(Exception):
    """A ``.env`` locator was normalized without ``--kit-src``: its class reads the value from the kit
    (EP §2.4), so there is no quiet default. The CLI exits 2; this is not an invalid ITEM."""


def _norm_path(p: str) -> str:
    p = p.replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    return posixpath.normpath(p) if p else p


def _norm_locator(loc: dict, seq, log: list, ctx_units, kit: Path | None) -> dict:
    """NFC/strip done by the caller. Path, repoId, E, keyNorm, grain. Appends rewrite-log entries."""
    loc = dict(loc)
    t = loc.get("type")
    if t in ("symbol", "config", "unit"):
        if "repoId" not in loc:
            loc["repoId"] = REPO_ID
        elif loc["repoId"] != REPO_ID:
            raise Invalid("V1:repoId")
    if "path" in loc and isinstance(loc["path"], str):
        loc["path"] = _norm_path(loc["path"])
    if t == "symbol":
        if "within" in loc or "nth" in loc:
            raise Invalid("V1:ancestor-plus-line-form")        # E1: writers are never asked for it
        q = loc.get("qname")
        if not isinstance(q, str):
            raise Invalid("V1:qname")
        if "::" in q and not q.startswith(loc.get("path", "") + "::"):
            head = _norm_path(q.split("::", 1)[0])
            q = head + "::" + q.split("::", 1)[1]
        e = E(q)
        if e != q:
            log.append({"seq": seq, "rule": "E"})
        loc["qname"] = e
    elif t == "contract":
        try:
            pre = normalize_http_key(loc["keyNorm"], "provider", mask=False) if loc.get("kind") == "http" else None
            n = normalize_contract(loc)
        except (ValueError, KeyError, TypeError):
            raise Invalid("V3:keyNorm")
        if n["keyNorm"] != loc.get("keyNorm"):
            masked = (n["keyNorm"].count("{#}") - pre.count("{#}")) if pre is not None else 0
            log.append({"seq": seq, "rule": "8a", "masked": masked} if masked else {"seq": seq, "rule": "V3"})
        loc = n
    elif t == "config":
        value, parsed = None, _NOVALUE
        fmt = config_format(loc["path"])
        if fmt == "env" and kit is None:
            raise NeedsKit("a .env locator needs --kit-src (its class reads the value from the kit)")
        if fmt in ("env", "kiro-agent") and kit is not None and (kit / loc["path"]).is_file():
            try:
                doc = parse_config(loc["path"], (kit / loc["path"]).read_text(encoding="utf-8"))
                if fmt == "env":
                    value = doc.get(loc["keyPath"])
                elif isinstance(doc, dict) and "includeMcpJson" in doc:
                    parsed = doc["includeMcpJson"]
            except Exception:
                value = None
        kp, cls = config_grain(loc["path"], loc["keyPath"], value, ctx_units, parsed)
        if kp != loc["keyPath"]:
            log.append({"seq": seq, "rule": "config-grain"})
        loc["keyPath"] = kp
        loc["_class"] = cls
    return loc


def locator_id(loc: dict) -> str:
    t = loc["type"]
    if t == "contract":
        return _ids.contract_point_id(loc["kind"], loc["keyNorm"])
    if t == "symbol":
        return _ids.symbol_id(lang_of(loc["path"]), loc["repoId"], loc["qname"])
    if t == "config":
        return _ids.config_entry_id(loc["repoId"], loc["path"], loc["keyPath"])
    if t == "unit":
        return _ids.unit_id(loc["repoId"], loc["rootPath"], loc["lang"])
    raise Invalid("V1:locator-type")


def normalize_item(obj: dict, writer: str, units=(), kit: Path | None = None):
    """EP §13.4's per-item normalization -> ``(record, log)``. Raises ``Invalid(reason)``."""
    obj = nfc_strip(obj)
    log: list = []
    seq = obj.get("seq")
    if obj.get("schema") != "ccg-key-v1" or not isinstance(obj.get("anchor"), dict) or not isinstance(obj.get("impacted"), dict):
        raise Invalid("V1:schema")
    anchor = _norm_locator(obj["anchor"], seq, log, units, kit)
    impacted = _norm_locator(obj["impacted"], seq, log, units, kit)
    provider = _norm_locator(obj["provider"], seq, log, units, kit) if isinstance(obj.get("provider"), dict) else None
    serves = [_norm_locator(s, seq, log, units, kit) for s in obj.get("serves") or []]
    if impacted["type"] not in ("symbol", "config"):
        raise Invalid("V1:impacted-type")
    if anchor["type"] == "unit":
        if impacted["type"] != "config":
            raise Invalid("V1:unit-anchor-needs-config")
        if not any(u["rootPath"] == anchor["rootPath"] and u["lang"] == anchor["lang"] for u in units):
            raise Invalid("V1:unit-not-listed")
        if impacted.get("_class") == "contract":
            raise Invalid("V1:class-mismatch")
    if anchor["type"] == "contract" and impacted["type"] == "config" and impacted.get("_class") == "server":
        u = unit_of(provider["path"], units) if provider and provider.get("path") else None
        if u is None:
            raise Invalid("V1:reanchor-no-unit")
        serves = serves + [anchor]
        anchor = {"type": "unit", "repoId": REPO_ID, "rootPath": u["rootPath"], "lang": u["lang"]}
        log.append({"seq": seq, "rule": "reanchor"})
    impacted.pop("_class", None)
    if anchor["type"] in ("symbol", "contract") and impacted["type"] == "config" and anchor["type"] == "symbol":
        pass
    seen, uniq = set(), []
    for s in serves:
        k = canon(s)
        if k not in seen:
            seen.add(k)
            uniq.append(s)
    anchor_id, impacted_id = locator_id(anchor), locator_id(impacted)
    rec = {
        "schema": "ccg-norm-v1", "writer": writer, "strategy": STRATEGY.get(writer, "provider-first"),
        "seq": seq, "stratum": obj.get("stratum"),
        "anchor": anchor, "impacted": impacted, "provider": provider, "serves": uniq,
        "anchorId": anchor_id, "impactedId": impacted_id,
        "itemId": _ids.tuple_id(["evalItem", anchor_id, impacted_id]),
        "lines": sorted(set(obj.get("lines") or [])), "chain": obj.get("chain") or [], "hops": obj.get("hops"),
        "why": obj.get("why"), "deployment": obj.get("deployment"), "original": obj,
    }
    return rec, log


def merge_within_writer(recs: list[dict]) -> list[dict]:
    """Two records with one ``itemId`` merge: union of ``lines``, the first ``chain`` by ``seq``."""
    by: dict = {}
    for r in sorted(recs, key=lambda r: (r["seq"] if isinstance(r["seq"], int) else 0)):
        m = by.get(r["itemId"])
        if m is None:
            by[r["itemId"]] = dict(r, originals=[r["original"]])
            continue
        m["lines"] = sorted(set(m["lines"]) | set(r["lines"]))
        have = {canon(s) for s in m["serves"]}
        m["serves"] = m["serves"] + [s for s in r["serves"] if canon(s) not in have]
        m["provider"] = m["provider"] or r["provider"]
        m["originals"].append(r["original"])
    return sorted(by.values(), key=lambda r: r["itemId"])


class InputError(ValueError):
    """An input-file problem whose message names a path and a line number, never content."""


def read_jsonl(path: Path):
    """(records, dropped_final_line): an unparseable FINAL line is dropped (plan §6.1)."""
    lines = [l for l in Path(path).read_text(encoding="utf-8", errors="replace").split("\n") if l.strip()]
    recs, dropped = [], 0
    for i, l in enumerate(lines):
        try:
            recs.append(json.loads(l))
        except ValueError:
            if i == len(lines) - 1:
                dropped = 1
            else:
                raise InputError(f"{path}: line {i + 1} is not JSON")
    return recs, dropped


def derive_stratum(rec: dict, ctx: dict, kit: Path | None = None) -> str | None:
    """EP §3.1: the first matching rule, from the item's own fields."""
    a, imp = rec["anchor"], rec["impacted"]
    flow = set(ctx.get("flowAnchors") or [])
    if a["type"] == "symbol" and a["qname"] in flow:
        return "flow"
    if imp["type"] == "config":
        return "wiring"
    script = False
    if a["type"] == "symbol":
        path, chain = a["qname"].split("::", 1)
        if chain == "<script>":
            script = True
        elif chain == "<module>" and path.endswith(".py") and kit is not None and (kit / path).is_file():
            txt = (kit / path).read_text(encoding="utf-8", errors="replace")
            script = txt.startswith("#!") or bool(re.search(r"^if\s+__name__\s*==\s*['\"]__main__['\"]\s*:", txt, re.M))
    if (a["type"] == "contract" and a["kind"] == "mcp_tool" and a["keyNorm"] in set(ctx.get("cypherMcpTools") or [])) \
            or script or lang_of(imp["path"]) == "shell":
        return "scripts"
    if a["type"] == "contract" and a["kind"] == "http":
        return "http"
    if a["type"] == "contract" and a["kind"] == "mcp_tool":
        return "mcp"
    return None


def run_normalize(writer: str, in_path: Path, ctx: dict, kit: Path | None):
    objs, dropped = read_jsonl(in_path)
    units = ctx.get("units") or []
    recs, invalid, log = [], [], []
    for o in objs:
        try:
            r, lg = normalize_item(o, writer, units, kit)
        except Invalid as e:
            invalid.append({"writer": writer, "seq": o.get("seq") if isinstance(o, dict) else None, "check": str(e)})
            continue
        r["derivedStratum"] = derive_stratum(r, ctx, kit)
        recs.append(r)
        log.extend(lg)
    return merge_within_writer(recs), invalid, log, dropped


# =============================================================================================
# validate (V1-V3), match, nearmiss, scope (V4 + counting), sample
# =============================================================================================

def validate_records(recs: list[dict], kit: Path | None, decisions=()):
    """V1-V3 on normalized records -> ``(valid, invalid, dropped_counts)``. ``decisions`` are the method
    owner's, by item id: {"itemId","action":"drop"} or {"itemId","action":"fix","record":{...}}."""
    dec = {d["itemId"]: d for d in decisions}
    valid, invalid, dropped = [], [], {}
    for r in recs:
        checks = []
        for loc in [r["anchor"], r["impacted"]] + ([r["provider"]] if r.get("provider") else []) + r.get("serves", []):
            checks += check_locator(loc, kit)
        checks += check_lines(r["impacted"], r["lines"], kit)
        if not checks:
            valid.append(r)
            continue
        d = dec.get(r["itemId"])
        if d and d["action"] == "drop":
            for c in sorted(set(checks)):
                key = f"{r['writer']}|{c}"
                dropped[key] = dropped.get(key, 0) + 1
        elif d and d["action"] == "fix":
            valid.append(d["record"])
        else:
            invalid.append({"itemId": r["itemId"], "writer": r["writer"], "seq": r["seq"], "checks": sorted(set(checks))})
    return valid, invalid, dropped


SESSION_ORDER = ("KA", "KA2", "KB")


def _finish_group(g: dict) -> None:
    """(Re)derive a match group's session fields from its ``members`` (shared by ``match`` and ``adjudicate``)."""
    g["members"].sort(key=lambda m: SESSION_ORDER.index(m["writer"]) if m["writer"] in SESSION_ORDER else 9)
    g["sessions"] = [m["writer"] for m in g["members"]]
    g["strategies"] = sorted({m["strategy"] for m in g["members"]})
    g["history"] = "".join("1" if s in g["sessions"] else "0" for s in SESSION_ORDER)


def _disagreements(groups) -> int:
    return sum(1 for g in groups for m in g["members"] if m.get("stratum") not in (None, g["derivedStratum"]))


def match_records(recs: list[dict], ka2_strata=()) -> dict:
    """EP §6.2: records of different writers match when their ``itemId``s are equal. The result carries
    ``t`` per stratum (EP §6.4's session count: 3 where KA2 ran, else 2). KA2 ran in a stratum when
    some KA2 record has it as ``derivedStratum`` or the caller names it in ``ka2_strata`` (a KA2 pass
    that found nothing there leaves no record to infer it from)."""
    by: dict = {}
    unclassified = 0
    for r in recs:
        if r.get("derivedStratum") is None:
            unclassified += 1
            continue
        g = by.setdefault(r["itemId"], {"itemId": r["itemId"], "anchorId": r["anchorId"], "impactedId": r["impactedId"],
                                        "derivedStratum": r["derivedStratum"], "members": []})
        g["members"].append(r)
    out = list(by.values())
    for g in out:
        _finish_group(g)
    out.sort(key=lambda g: g["itemId"])
    ka2 = set(ka2_strata) | {r["derivedStratum"] for r in recs if r.get("writer") == "KA2" and r.get("derivedStratum")}
    return {"schema": "ccg-match-v1", "records": out, "unclassified": unclassified,
            "stratumDisagreements": _disagreements(out), "t": {s: 3 if s in ka2 else 2 for s in STRATA}}


class Inconsistent(Exception):
    """A verdict list that EP §6.3 sends back. ``args[0]`` is a list of ``(code, id)``: pair or item ids only."""


def apply_verdicts(match: dict, pairs: list[dict], verdicts: list[dict]):
    """EP §6.3 / §13.4 ``adjudicate``. ``pairs`` are the ``nearmiss`` lines, ``verdicts`` the method owner's
    ``{pairId, verdict: "same"|"different", keep: "x"|"y"}``. A verdict list is REJECTED (``Inconsistent``,
    nothing applied) when a pairId is unknown or repeated, a verdict or ``keep`` is malformed, a pair names an
    item that is not in ``match``, or one record is ``same`` in two pairs. A ``same`` pair becomes ONE matched
    record: the ``keep`` side's group survives (identity, locators) and takes the other's members, with
    sessions, strategies and history re-derived. A ``different`` or unadjudicated pair changes nothing (counted
    ``different``). -> ``(match, log, summary)``; ``log`` is ``work/adjudication.jsonl``'s lines."""
    match = copy.deepcopy(match)
    by_pair = {p["pairId"]: p for p in pairs}
    groups = {g["itemId"]: g for g in match["records"]}
    bad: list = []
    seen: set = set()
    same_use: dict = {}
    for v in verdicts:
        pid = v.get("pairId") if isinstance(v, dict) else None
        if pid not in by_pair:
            bad.append(("unknown-pair", pid if isinstance(pid, str) else "?"))
            continue
        if pid in seen:
            bad.append(("repeated-pair", pid))
        seen.add(pid)
        if v.get("verdict") not in ("same", "different"):
            bad.append(("bad-verdict", pid))
            continue
        p = by_pair[pid]
        ix, iy = p["x"]["itemId"], p["y"]["itemId"]
        if v["verdict"] == "same":
            if v.get("keep") not in ("x", "y"):
                bad.append(("bad-keep", pid))
            if ix not in groups or iy not in groups:
                bad.append(("item-not-in-match", pid))
            for i in (ix, iy):
                same_use.setdefault(i, []).append(pid)
    for i, pids in sorted(same_use.items()):
        if len(pids) > 1:
            bad += [("record-same-twice", pid) for pid in sorted(pids)]
    if bad:
        raise Inconsistent(sorted(set(bad)))
    log, merged = [], 0
    for v in sorted(verdicts, key=lambda v: v["pairId"]):
        p = by_pair[v["pairId"]]
        entry = {"pairId": v["pairId"], "verdict": v["verdict"], "keep": v.get("keep")}
        log.append(entry)
        if v["verdict"] != "same":
            continue
        ix, iy = p["x"]["itemId"], p["y"]["itemId"]
        keep, drop = (ix, iy) if v["keep"] == "x" else (iy, ix)
        k, d = groups[keep], groups.pop(drop)
        k["members"] = k["members"] + d["members"]
        k["absorbed"] = sorted(set(k.get("absorbed", [])) | {drop})
        _finish_group(k)
        merged += 1
    match = dict(match, records=sorted(groups.values(), key=lambda g: g["itemId"]))
    match["stratumDisagreements"] = _disagreements(match["records"])
    decided = {v["pairId"] for v in verdicts}
    summary = {"pairs": len(pairs), "verdicts": len(verdicts), "same": merged,
               "different": sum(1 for v in verdicts if v["verdict"] == "different"),
               "unadjudicated": len(pairs) - len(decided)}
    return match, log, summary


def _fold_key(k: str) -> str:
    k = k.casefold().rstrip("/") if k.endswith("/") and len(k) > 1 else k.casefold()
    return re.sub(r"\{[^{}/]*\}", "{}", k)


def _provider_id(m: dict):
    p = m.get("provider")
    try:
        return locator_id(p) if p and p.get("type") == "symbol" else None
    except Invalid:
        return None


def _kp_prefix(a: str, b: str) -> bool:
    sa, sb = keypath_segments(a), keypath_segments(b)
    n = min(len(sa), len(sb))
    return sa[:n] == sb[:n]


def nearmiss_pairs(match: dict) -> list[dict]:
    """EP §6.3: N1/N2/N3 candidate pairs, within each derived stratum, from each provider-first record
    and each consumer-first record that are both still unmatched (found by one strategy only)."""
    pairs = {}
    for stratum in STRATA:
        recs = [g for g in match["records"] if g["derivedStratum"] == stratum]
        xs = [g for g in recs if g["strategies"] == ["provider-first"]]        # found by ONE strategy only
        ys = [g for g in recs if g["strategies"] == ["consumer-first"]]
        for gx, gy in itertools.product(xs, ys):
            x, y = gx["members"][0], gy["members"][0]
            ax, ay, ix, iy = x["anchor"], y["anchor"], x["impacted"], y["impacted"]
            rule = None
            if x["anchorId"] == y["anchorId"] and x["impactedId"] != y["impactedId"] and ix.get("path") == iy.get("path") \
                    and (ix["type"] != "config" or iy["type"] != "config" or _kp_prefix(ix["keyPath"], iy["keyPath"])):
                rule = "N1"
            elif x["impactedId"] == y["impactedId"] and x["anchorId"] != y["anchorId"]:
                if ax["type"] == ay["type"] == "contract" and ax["kind"] == ay["kind"] and (
                        (_provider_id(x) is not None and _provider_id(x) == _provider_id(y))
                        or _fold_key(ax["keyNorm"]) == _fold_key(ay["keyNorm"])):
                    rule = "N2"
                elif ax["type"] == ay["type"] == "unit":
                    rule = "N3"
            if rule:
                pid = sha(f"ccg-p0f-pair-v1|{gx['itemId']}|{gy['itemId']}")[:12]
                pairs[pid] = {"pairId": pid, "rule": rule, "stratum": stratum, "x": x, "y": y}
    return [pairs[k] for k in sorted(pairs)]


def scope_records(match: dict, units: list[dict]):
    """V4 per ITEM after matching (m1) and the ``counting`` flag (EP §3.7).
    -> ``(match with inScope/counting, out_of_scope item ids, summary)``"""
    oos = []
    for g in match["records"]:
        g["inScope"] = True
        if g["derivedStratum"] == "flow":
            ok = any(len(m["chain"]) <= 2 and any(h.get("kind") in FLOW_HOPS for h in m["chain"]) for m in g["members"])
            g["inScope"] = ok
            if not ok:
                oos.append(g["itemId"])
        m0 = g["members"][0]
        if g["derivedStratum"] == "flow":
            ua = unit_of(m0["anchor"].get("path", ""), units) if m0["anchor"]["type"] != "contract" else None
            ui = unit_of(m0["impacted"].get("path", ""), units)
            g["counting"] = (ua is None) != (ui is None) or (ua is not None and ua["rootPath"] != ui["rootPath"]) \
                or m0["anchor"].get("repoId", REPO_ID) != m0["impacted"].get("repoId", REPO_ID)
        else:
            g["counting"] = True
    kept = [g for g in match["records"] if g["inScope"]]
    summary = {"H_all": len(kept), "H": sum(1 for g in kept if g["counting"]), "outOfScope": len(oos),
               "perStratum": {s: sum(1 for g in kept if g["derivedStratum"] == s) for s in STRATA}}
    return match, sorted(oos), summary


def hash_rank(seed: str, keys) -> list:
    """Ascending ``SHA-256(seed + "|" + key)`` (S05's convention)."""
    return sorted(keys, key=lambda k: sha(f"{seed}|{k}"))


def spot_sample(match: dict, eval_commit: str, per_cell: int = 4):
    """EP §7.3's union sample: ``per_cell`` singletons and doubletons per derived stratum, hash sorted
    by ``seed = "ccg-p0f-spot-v1|" + EVAL_COMMIT`` over ``itemId``. A cell short of ``per_cell`` takes all it has,
    and its deficit goes, in stratum order wrapping from ``flow`` to ``http``, to the SAME class of the
    next stratum with a surplus, then to the other class (from the same stratum on)."""
    seed = "ccg-p0f-spot-v1|" + eval_commit
    cls_of = lambda g: "singleton" if len(g["strategies"]) == 1 else "doubleton"
    ranked: dict = {}
    for s in STRATA:
        for c in ("singleton", "doubleton"):
            ids = [g["itemId"] for g in match["records"] if g["derivedStratum"] == s and cls_of(g) == c]
            ranked[(s, c)] = hash_rank(seed, ids)
    taken = {k: list(v[:per_cell]) for k, v in ranked.items()}
    spare = {k: list(v[per_cell:]) for k, v in ranked.items()}
    for s in STRATA:
        for c in ("singleton", "doubleton"):
            need = per_cell - len(taken[(s, c)])
            if need <= 0:
                continue
            i = STRATA.index(s)
            order_same = [STRATA[(i + d) % 5] for d in range(1, 5)]
            other = "doubleton" if c == "singleton" else "singleton"
            order_other = [STRATA[(i + d) % 5] for d in range(0, 5)]
            for (cls, strata) in ((c, order_same), (other, order_other)):
                for s2 in strata:
                    while need > 0 and spare[(s2, cls)]:
                        taken[(s, c)].append(spare[(s2, cls)].pop(0))
                        need -= 1
    by_id = {g["itemId"]: g for g in match["records"]}
    out = []
    for (s, c), ids in taken.items():
        for iid in ids:
            out.append({"cell": f"{s}/{c}", "class": cls_of(by_id[iid]), "stratum": by_id[iid]["derivedStratum"],
                        "itemId": iid, "record": by_id[iid]})
    return sorted(out, key=lambda r: r["itemId"])


def spot_expected(per_cell: int = 4) -> int:
    return per_cell * len(STRATA) * 2


def spot_shortfall_record(spot: list[dict], per_cell: int = 4) -> dict | None:
    """EP §7.3: with no surplus anywhere the sample stays short and ``work/spot.jsonl`` says so. -> the
    marker record (the file's last line; it has no ``itemId``) or ``None`` when the sample is full."""
    want = spot_expected(per_cell)
    if len(spot) >= want:
        return None
    return {"record": "shortfall", "expected": want, "got": len(spot), "short": want - len(spot)}


def lang_gold_sample(qw: list[dict], lang: str, seed_tag: str, eval_commit: str, n: int = 8, n_amb: int = 2):
    """EP §7.3: 8 items of ``lang`` drawn by hash sort over ``qid`` (6 unambiguous + 2 ambiguous if available;
    a short class is filled from the other)."""
    seed = f"{seed_tag}|{eval_commit}"
    pool = [q for q in qw if q.get("lang") == lang]
    amb = hash_rank(seed, [q["qid"] for q in pool if q.get("intent") == "ambiguous"])
    una = hash_rank(seed, [q["qid"] for q in pool if q.get("intent") != "ambiguous"])
    take_amb = min(n_amb, len(amb))
    take_una = min(n - n_amb, len(una))
    chosen = una[:take_una] + amb[:take_amb]
    rest = una[take_una:] + amb[take_amb:]
    chosen += rest[:max(0, n - len(chosen))]
    by = {q["qid"]: q for q in pool}
    return [by[i] for i in hash_rank(seed, chosen)]


# =============================================================================================
# renormalize-check (EP2 §1.5): the drift guard
# =============================================================================================

def renormalize_check(entries: list[dict]) -> dict:
    """Each entry: {"id": stored id, "locator": the writer-form contract locator, "form": "provider"|"consumer"}
    (key-item anchor/serves, AC-24 gold with its STORED form, and C: provider). Re-runs ``normalize_contract``
    with the screen as imported now and compares ids. -> counts and the differing stored ids only."""
    diffs = []
    for e in entries:
        stored = e.get("id") if isinstance(e, dict) else None
        try:
            now = contract_id(e["locator"], e.get("form", "provider"))
        except (ValueError, KeyError, TypeError, AttributeError):
            now = None                      # no longer normalizes (or malformed): drift. Never echo the exception text.
        if now != stored:
            diffs.append({"stored": stored, "now": now})
    return {"checked": len(entries), "differences": len(diffs), "diffs": diffs}


# =============================================================================================
# check-report (EP §9.4 as clarified by EP2 §1.4)
# =============================================================================================

_T = r"A-Za-z0-9_.:/#<>-"


def _tok_re(probe: str) -> re.Pattern:
    """EP §9.4 (v1.2, P2-3): ``(?:^|(?<=[^T]))(?:\\./)?P(?=[.:]*(?:$|[^T]))``."""
    return re.compile(rf"(?:^|(?<=[^{_T}]))(?:\./)?{re.escape(probe)}(?=[.:]*(?:$|[^{_T}]))")


def collapse(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def _locator_probes(loc: dict) -> list[tuple[str, str]]:
    """(kind, value) for a WRITER-ORIGINAL locator. keyNorm values are the normalized form and the
    pre-8a form (EP2 §1.4); the writer's raw spelling is not a probe."""
    out: list[tuple[str, str]] = []
    t = loc.get("type")
    try:
        if t == "contract":
            if loc.get("kind") == "http":
                out.append(("keyNorm", normalize_http_key(loc["keyNorm"], "provider", mask=True)))
                out.append(("keyNorm-pre8a", normalize_http_key(loc["keyNorm"], "provider", mask=False)))
            else:
                out.append(("keyNorm", normalize_contract(loc)["keyNorm"]))
                out.append(("keyNorm-pre8a", normalize_contract(loc)["keyNorm"]))
            out.append(("id", contract_id(loc)))
        elif t == "symbol":
            p = _norm_path(loc["path"])
            q = E(loc["qname"])
            out.append(("qname", q))
            out.append(("id", _ids.symbol_id(lang_of(p), REPO_ID, q)))
            if isinstance(loc.get("line"), int):
                out.append(("pathline", f"{p}:{loc['line']}"))
        elif t == "config":
            p = _norm_path(loc["path"])
            out.append(("pathkp", f"{p}#{loc['keyPath']}"))
    except (ValueError, KeyError, TypeError):
        pass
    return out


def _gold_locators(gold) -> list[dict]:
    locs = []
    if isinstance(gold, dict):
        locs += gold.get("anchors") or []
    return locs


def _id_variants(i: str) -> list[str]:
    return [i, i[:12]] if isinstance(i, str) and len(i) >= 12 else []


def record_probes(rec: dict) -> list[tuple[str, str]]:
    """Every probe of one held-out (or dev) record, computed from its WRITER ORIGINALS (EP2 §1.4):
    an AC-5 record {itemId, anchorId, impactedId, original(s)} or an AC-24 record {original{text, gold,
    goldOptions}} / {text, gold, goldOptions}."""
    out: list[tuple[str, str]] = []
    for k in ("itemId", "anchorId", "impactedId"):
        out += [("id", v) for v in _id_variants(rec.get(k))]
    originals = rec.get("originals") or ([rec["original"]] if isinstance(rec.get("original"), dict) else [])
    for o in originals:
        o = nfc_strip(o)
        locs = [o.get("anchor"), o.get("provider"), o.get("impacted")] + list(o.get("serves") or [])
        for opt in [o.get("gold")] + list(o.get("goldOptions") or []):
            locs += _gold_locators(opt)
        for loc in locs:
            if isinstance(loc, dict):
                out += _locator_probes(loc)
        imp = o.get("impacted") or {}
        if isinstance(imp.get("path"), str):
            for l in o.get("lines") or []:
                out.append(("pathline", f"{_norm_path(imp['path'])}:{l}"))
        for hop in o.get("chain") or []:
            for k in ("from", "to"):
                if isinstance(hop.get(k), str):
                    out.append(("pathline", hop[k]))
        text = o.get("text")
        if isinstance(text, str):
            out += [("shingle", s) for s in _shingles(text)]
    text = rec.get("text")
    if isinstance(text, str) and not originals:
        out += [("shingle", s) for s in _shingles(text)]
    for opt in [rec.get("gold")] + list(rec.get("goldOptions") or []) if not originals else []:
        for loc in _gold_locators(opt):
            out += _locator_probes(loc)
    return out


def _shingles(text: str, n: int = 6) -> list[str]:
    w = collapse(text).split(" ")
    return [" ".join(w[i:i + n]) for i in range(len(w) - n + 1)] if len(w) >= n else []


def contact_probes(loc: dict) -> list[tuple[str, str]]:
    return _locator_probes(loc)


def _bare_short(v: str) -> bool:
    """EP §9.4: the floor drops only a BARE probe under 8 characters (none of ``::``, ``/``, ``#`` or a
    space), such as a tool name like ``query``. A 7-character qname ``a.py::f`` stays a probe."""
    return len(v) < 8 and not any(c in v for c in ("::", "/", "#", " "))


def leak_check(report: str, heldout: list[dict], dev: list[dict], contact: list[dict]):
    """-> ``(hits, n_probes)`` where hits are probe hashes only. A held-out probe is removed when it is
    EXACTLY equal to the same kind of value in some dev item or in C (computed from writer originals)."""
    public: dict[str, set] = {}
    for rec in dev:
        for k, v in record_probes(rec):
            public.setdefault(k, set()).add(v)
    for loc in contact:
        for k, v in contact_probes(loc):
            public.setdefault(k, set()).add(v)
    probes: dict[str, str] = {}
    for rec in heldout:
        for k, v in record_probes(rec):
            if v in public.get(k, ()) or _bare_short(v):
                continue
            probes[collapse(v)] = k
    text = collapse(report)
    hits = []
    for p, k in sorted(probes.items()):
        if k in ("id", "shingle"):
            found = p in text
        else:
            found = bool(_tok_re(p).search(text))
        if found:
            hits.append(sha(p)[:12])
    return hits, len(probes)


# =============================================================================================
# CLI
# =============================================================================================

def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(canon(r) + "\n" for r in rows), encoding="utf-8")


def _ctx(p) -> dict:
    return json.loads(Path(p).read_text(encoding="utf-8")) if p else {}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="verb", required=True)

    def add(name, *flags):
        sp = sub.add_parser(name)
        for f, kw in flags:
            sp.add_argument(f, **kw)
        return sp

    P = lambda **kw: dict(type=Path, **kw)
    add("normalize", ("--writer", dict(required=True, choices=["KA", "KB", "KA2"])), ("--in", dict(dest="inp", required=True, type=Path)),
        ("--out", P(required=True)), ("--log", P()), ("--invalid", P()), ("--kit-src", P()), ("--context", P()))
    add("validate", ("--in", dict(dest="inp", required=True, type=Path, nargs="+")), ("--kit-src", P(required=True)),
        ("--out-dir", P(required=True)), ("--decisions", P()))
    add("match", ("--in", dict(dest="inp", required=True, type=Path, nargs="+")), ("--out", P(required=True)),
        ("--ka2-strata", dict(nargs="*", default=[], choices=list(STRATA))))
    add("adjudicate", ("--match", P(required=True)), ("--nearmiss", P(required=True)), ("--verdicts", P(required=True)),
        ("--out", P(required=True)), ("--log", P(required=True)))
    add("nearmiss", ("--match", P(required=True)), ("--out", P(required=True)))
    add("scope", ("--match", P(required=True)), ("--context", P(required=True)), ("--out", P(required=True)),
        ("--out-of-scope", P(required=True)))
    add("sample", ("--match", P(required=True)), ("--qw", P(required=True)), ("--eval-commit", dict(required=True)),
        ("--out-dir", P(required=True)))
    add("renormalize-check", ("--in", dict(dest="inp", required=True, type=Path)))
    add("check-report", ("--report", P(required=True)), ("--heldout", P(required=True, nargs="+")),
        ("--dev", P(nargs="*", default=[])), ("--contact", P()))
    a = ap.parse_args(argv)

    try:
        if a.verb == "normalize":
            ctx = _ctx(a.context)
            recs, invalid, log, dropped = run_normalize(a.writer, a.inp, ctx, a.kit_src)
            _write_jsonl(a.out, recs)
            if a.log:
                _write_jsonl(a.log, log)
            if a.invalid:
                _write_jsonl(a.invalid, invalid)
            by_rule: dict = {}
            for l in log:
                by_rule[l["rule"]] = by_rule.get(l["rule"], 0) + 1
            print(f"records: {len(recs)}\ninvalid: {len(invalid)}\ndropped-final-line: {dropped}\nrewrites: {canon(by_rule)}\nout: {a.out}")
            return 1 if invalid else 0
        if a.verb == "validate":
            recs = [r for p in a.inp for r in read_jsonl(p)[0]]
            dec = read_jsonl(a.decisions)[0] if a.decisions else []
            valid, invalid, dropped = validate_records(recs, a.kit_src, dec)
            _write_jsonl(a.out_dir / "invalid.jsonl", invalid)
            _write_jsonl(a.out_dir / "valid.jsonl", valid)
            print(f"valid: {len(valid)}\ninvalid: {len(invalid)}\ndropped: {canon(dropped)}")
            return 1 if invalid else 0
        if a.verb == "match":
            recs = [r for p in a.inp for r in read_jsonl(p)[0]]
            m = match_records(recs, a.ka2_strata)
            a.out.parent.mkdir(parents=True, exist_ok=True)
            a.out.write_text(canon(m) + "\n", encoding="utf-8")
            print(f"records: {len(m['records'])}\nunclassified: {m['unclassified']}\nstratum-disagreements: {m['stratumDisagreements']}")
            return 0
        if a.verb == "adjudicate":
            m = json.loads(a.match.read_text(encoding="utf-8"))
            pairs = read_jsonl(a.nearmiss)[0]
            verdicts = read_jsonl(a.verdicts)[0]
            try:
                m, log, summary = apply_verdicts(m, pairs, verdicts)
            except Inconsistent as e:
                print(f"adjudicate: verdict list rejected, nothing written: {len(e.args[0])} problem(s)", file=sys.stderr)
                for code, ident in e.args[0]:
                    print(f"  {code}: {ident}", file=sys.stderr)
                return 1
            a.out.parent.mkdir(parents=True, exist_ok=True)
            a.out.write_text(canon(m) + "\n", encoding="utf-8")
            _write_jsonl(a.log, log)
            print(f"pairs: {summary['pairs']}\nverdicts: {summary['verdicts']}\nsame: {summary['same']}\n"
                  f"different: {summary['different']}\nunadjudicated: {summary['unadjudicated']}\n"
                  f"records: {len(m['records'])}")
            return 0
        if a.verb == "nearmiss":
            pairs = nearmiss_pairs(json.loads(a.match.read_text(encoding="utf-8")))
            _write_jsonl(a.out, pairs)
            print(f"pairs: {len(pairs)}\n" + "".join(f"{r}: {sum(1 for p in pairs if p['rule'] == r)}\n" for r in ("N1", "N2", "N3")).rstrip())
            return 0
        if a.verb == "scope":
            m = json.loads(a.match.read_text(encoding="utf-8"))
            m, oos, summary = scope_records(m, _ctx(a.context).get("units") or [])
            a.out.write_text(canon(m) + "\n", encoding="utf-8")
            _write_jsonl(a.out_of_scope, [{"itemId": i, "reason": "V4"} for i in oos])
            print(canon(summary))
            return 0
        if a.verb == "sample":
            m = json.loads(a.match.read_text(encoding="utf-8"))
            qw = read_jsonl(a.qw)[0]
            spot = spot_sample(m, a.eval_commit)
            short = spot_shortfall_record(spot)
            _write_jsonl(a.out_dir / "spot.jsonl", spot + ([short] if short else []))
            pt = lang_gold_sample(qw, "pt", "ccg-p0f-ptgold-v1", a.eval_commit)
            en = lang_gold_sample(qw, "en", "ccg-p0f-engold-v1", a.eval_commit)
            _write_jsonl(a.out_dir / "ptgold.jsonl", pt)
            _write_jsonl(a.out_dir / "engold.jsonl", en)
            print(f"spot: {len(spot)}\nspot-shortfall: {short['short'] if short else 0}\nptgold: {len(pt)}\nengold: {len(en)}")
            return 0
        if a.verb == "renormalize-check":
            res = renormalize_check(read_jsonl(a.inp)[0])
            print(f"checked: {res['checked']}\ndifferences: {res['differences']}")
            for d in res["diffs"]:
                print(f"changed: {d['stored']} -> {d['now'] or '(does not normalize)'}")
            return 1 if res["differences"] else 0
        if a.verb == "check-report":
            ho = [r for p in a.heldout for r in read_jsonl(p)[0]]
            dev = [r for p in a.dev for r in read_jsonl(p)[0]]
            contact = read_jsonl(a.contact)[0] if a.contact else []
            hits, n = leak_check(a.report.read_text(encoding="utf-8"), ho, dev, contact)
            print(f"probes: {n}\nhits: {len(hits)}")
            for h in hits:
                print(f"hit: {h}")
            return 1 if hits else 0
    except NeedsKit as e:
        print(f"match_keys: {e}", file=sys.stderr)
        return 2
    except InputError as e:
        print(f"match_keys: {e}", file=sys.stderr)                  # a path and a line number only
        return 4
    except OSError as e:
        print(f"match_keys: {e}", file=sys.stderr)                  # a path
        return 4
    except (ValueError, KeyError, TypeError) as e:
        print(f"match_keys: unreadable input ({type(e).__name__})", file=sys.stderr)   # never a stored value (m-3)
        return 4
    return 2


if __name__ == "__main__":
    sys.exit(main())
