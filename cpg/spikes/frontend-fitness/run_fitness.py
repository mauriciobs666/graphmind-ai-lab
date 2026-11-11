#!/usr/bin/env python3
"""run_fitness.py -- S25 (P0-18a/b) harness for the seven-language Joern frontend-fitness measurement.

Spec: docs/plans/connected-code-graph-p0-ml.md (the ML note; "ML") and
docs/plans/connected-code-graph-p0-2.md §4.6 S25. Everything here is Joern-independent except the
`joern` stage (P0-18b), which only *builds the command lines* for cpg/sandbox/offline-run.sh.

Stages (each is idempotent, writes under cpg/spikes/frontend-fitness/ or cpg/.ccg-work/):

  fetch    <lang>   materialise the pinned corpus (git archive / shallow fetch of the pinned sha) into
                    cpg/.ccg-work/corpora/<lang>/src      (no .git; the labeller's and Joern-off's tree)
  census   <lang>   C1-C8 evidence + C6 idiom counts + exclusion greps -> stdout JSON
  frame    <lang>   tree-sitter census frame (ML §4.1)          -> frame/<lang>.jsonl
  draw     <lang>   hash-sort draw (ML §4.4)                    -> sample/<lang>.jsonl, .private.jsonl
  flowdraw <lang>   30 golden-flow methods (ML §6.4)            -> sample/<lang>-flow.jsonl
  flowdraw-p1 python --exclude F --counts C --tier2-out T   P1 draw (prereg §3; `--set p1` on flowbrief/flowcheck) -> sample/python-flow-p1.jsonl, T, C
  c5pilot  <lang>   first 50 of the c5 hash sort of the frame   -> truth/<lang>.c5-pilot.jsonl
  brief    <lang>   fill ML §6.5's template                     -> sample/<lang>.brief.md
  flowbrief <lang>  fill flow-brief.template.md (S56)           -> sample/<lang>-flow.brief.md
  flowcheck <lang>  schema-check labels/<lang>-flow.jsonl (S56)  -> stdout JSON; exit 1 on any error
  refmap   <lang>   map a SCIP JSON / callgraph output to sites -> truth/<lang>.ref.jsonl | c5 json
  genaudit csharp   the hash-sorted generic/nameof sites of the sample (ML §3.2, ccg-p0a-v1-gen|) -> truth/csharp.gen-audit.jsonl
  verify            re-derive every frame/sample/flow file from corpora.yaml alone; exit 1 on drift
  joern    <lang>   print (do not run) the runner command lines of the P0-18b Joern runs

Only the python stdlib + pyyaml + py-tree-sitter (cpg `spike` extra) are used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Optional

import yaml

HERE = Path(__file__).resolve().parent
CPG = HERE.parent.parent
REPO = CPG.parent
WORK = CPG / ".ccg-work"
CORPORA = WORK / "corpora"

LANGS = ["python", "javascript", "java", "kotlin", "csharp", "go", "cpp"]
SEED_PREFIX = {"sample": "ccg-p0a-v1|", "sym": "ccg-p0a-v1-sym|", "flow": "ccg-p0a-v1-flow|",
               "c5": "ccg-p0a-v1-c5|", "spur": "ccg-p0a-v1-spur|", "gen": "ccg-p0a-v1-gen|"}
# Flow sets (P1-34a). `p0` is the committed P0 draw and keeps every historical path; `p1` routes every flow path to a `-p1` name so a
# p1 run can never read-or-overwrite a P0 file. FLOW_SET is the process-wide selection (`--set`); path helpers also take it explicitly.
FLOW_SETS = {"p0": "", "p1": "-p1"}
FLOW_SET = "p0"
SEED_PREFIX_P1 = {"flow": "ccg-p1a-v1-flow|", "spur": "ccg-p1a-v1-spur|"}
FLOW_P1_TIER1, FLOW_P1_TIER2 = 30, 15
# The registered P1 corpus: the real `ccg-p1a-v1-flow` seed is only ever drawn on a tree that hashes to this digest (prereg §0.1).
P1_REAL_COMMIT = "0caf85e3ccb757801d15d6babe55a63477a486dd"
P1_REAL_SRC_DIGEST = "365761fbdd26604c538f12beceb05811a8d77a32c8dd4ad01abc624a484800d6"
P1_FRAME_COUNTS = (368, 338, 163)   # tier-1 population, after the P0 subtraction, tier-2 population (prereg §0): the frame depends on the grammar versions
CAP = 500
# ML §3.1 (v1.6): TS/JS is a CENSUS of the prod code: every frame site is labelled, in hash order, no cap, no stop rule.
CENSUS_LANGS = {"javascript"}
# Bundle spec4 §1 `test` pattern (closed; JS/TS only): `*.test.<ext>`, `*.spec.<ext>`, a directory segment `e2e` or `__tests__`.
# The frame and the flow population leave those files out (the bundle has no symbol for them by design).
TEST_DIR_SEGMENTS = {"e2e", "__tests__"}

# Bundle spec §5.0 default; a corpus may replace it in corpora.yaml (`vendoredDirs`).
DEFAULT_VENDORED = ["vendor", "vendored", "_vendor", "third_party", "third-party", "node_modules"]
# Directories that are never source in a *clean checkout's* frame either (tool caches); build output
# does not exist in the clean tree, this is belt-and-braces and is recorded in corpora.yaml too.
ALWAYS_SKIP_DIRS = {".git", "__pycache__", ".mypy_cache", ".pytest_cache", ".tox", ".gradle", ".idea", ".vs"}

# File sets: bundle spec3 §3.3 "Which files", minus declaration files; Kotlin `.kts` build scripts are
# excluded (kotlin2cpg takes .kt; Gradle scripts are DSL calls into the Gradle API, all external).
EXTS = {
    "python": {".py"},
    "javascript": {".ts", ".tsx", ".mts", ".cts", ".js", ".jsx", ".mjs", ".cjs"},
    "java": {".java"},
    "kotlin": {".kt"},
    "csharp": {".cs"},
    "go": {".go"},
    "cpp": {".c", ".cc", ".cpp", ".cxx", ".c++", ".h", ".hh", ".hpp", ".hxx", ".ino"},
}
# Other-language extensions, only for the C2 (>= 80 % single-language) share.
OTHER_EXTS = {
    "python": {".py"}, "javascript": {".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".mts", ".cts"},
    "java": {".java"}, "kotlin": {".kt", ".kts"}, "csharp": {".cs"}, "go": {".go"},
    "cpp": {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx"},
    "shell": {".sh", ".bash"}, "ruby": {".rb"}, "rust": {".rs"}, "php": {".php"}, "swift": {".swift"},
    "scala": {".scala"}, "groovy": {".groovy", ".gradle"}, "perl": {".pl", ".pm"}, "lua": {".lua"},
    "objc": {".m", ".mm"}, "dart": {".dart"}, "vb": {".vb"}, "fsharp": {".fs"},
}
GEN_MARK = re.compile(r"generated|do not edit", re.I)


# ------------------------------------------------------------------ small helpers

def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def ws(s: str) -> str:
    """Whitespace-collapsed (ML §2 callText)."""
    return " ".join(s.split())


def sha256hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def load_corpora() -> dict:
    with open(HERE / "corpora.yaml") as f:
        return yaml.safe_load(f)


def corpus_cfg(lang: str) -> dict:
    return load_corpora()["corpora"][lang]


def src_dir(lang: str) -> Path:
    return CORPORA / lang / "src"


def vendored(cfg: dict) -> list[str]:
    return cfg.get("vendoredDirs") or DEFAULT_VENDORED


def jdump(o) -> str:
    return json.dumps(o, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(jdump(r) + "\n")
            n += 1
    os.replace(tmp, path)
    return n


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


# ------------------------------------------------------------------ file walk

def is_test_file(lang: str, rel: str) -> bool:
    """Spec4 §1 `test` pattern; applies to JS/TS only (`javascript`), every other language has none."""
    if lang != "javascript":
        return False
    parts = rel.split("/")
    base = parts[-1]
    if any(seg in TEST_DIR_SEGMENTS for seg in parts[:-1]):
        return True
    m = re.fullmatch(r"(.+)\.(test|spec)(\.[^.]+)", base)
    return bool(m) and m.group(3).lower() in EXTS["javascript"]


def seg_excluded(rel_parts: tuple[str, ...], cfg: dict) -> bool:
    skip = set(vendored(cfg)) | ALWAYS_SKIP_DIRS | set(cfg.get("frameExcludeDirs") or [])
    return any(p in skip for p in rel_parts[:-1])


def is_generated(rel: str, head: str, cfg: dict) -> bool:
    for g in cfg.get("generatedPaths") or []:
        if rel == g or rel.startswith(g.rstrip("/") + "/"):
            return True
    lines = head.split("\n")[:5]
    return any(GEN_MARK.search(l) for l in lines)


def iter_source_files(lang: str, root: Path, cfg: dict, include_generated: bool = False) -> Iterator[tuple[str, Path]]:
    """Frame files: language extension, outside excluded/vendored dirs, not generated. Sorted by path."""
    exts = EXTS[lang]
    out = []
    for dp, dns, fns in os.walk(root):
        dns[:] = sorted(d for d in dns)
        for fn in fns:
            p = Path(dp) / fn
            rel = nfc(p.relative_to(root).as_posix())
            ext = p.suffix.lower()
            if ext not in exts:
                continue
            if lang == "javascript" and rel.endswith(".d.ts"):
                continue
            parts = tuple(rel.split("/"))
            if seg_excluded(parts, cfg):
                continue
            if not include_generated:
                try:
                    with open(p, "rb") as f:
                        head = f.read(2048).decode("utf-8", "replace")
                except OSError:
                    continue
                if is_generated(rel, head, cfg):
                    continue
            out.append((rel, p))
    out.sort(key=lambda t: t[0])
    yield from out


# ------------------------------------------------------------------ tree-sitter

_PARSERS: dict = {}


def _ts():
    import tree_sitter  # noqa: F401
    return tree_sitter


def grammar_for(lang: str, path: str):
    """(grammar-key, Language) for a file."""
    from tree_sitter import Language
    ext = os.path.splitext(path)[1].lower()
    if lang == "python":
        import tree_sitter_python as m; key, lg = "python", m.language()
    elif lang == "javascript":
        import tree_sitter_typescript as ts, tree_sitter_javascript as js
        if ext in (".tsx",):
            key, lg = "tsx", ts.language_tsx()
        elif ext in (".ts", ".mts", ".cts"):
            key, lg = "typescript", ts.language_typescript()
        else:
            key, lg = "javascript", js.language()
    elif lang == "java":
        import tree_sitter_java as m; key, lg = "java", m.language()
    elif lang == "kotlin":
        import tree_sitter_kotlin as m; key, lg = "kotlin", m.language()
    elif lang == "csharp":
        import tree_sitter_c_sharp as m; key, lg = "csharp", m.language()
    elif lang == "go":
        import tree_sitter_go as m; key, lg = "go", m.language()
    elif lang == "cpp":
        if ext == ".c":
            import tree_sitter_c as m; key, lg = "c", m.language()
        else:
            import tree_sitter_cpp as m; key, lg = "cpp", m.language()
    else:
        raise ValueError(lang)
    return key, Language(lg)


def parse_file(lang: str, rel: str, data: bytes):
    from tree_sitter import Parser
    key, lg = grammar_for(lang, rel)
    p = _PARSERS.get(key)
    if p is None:
        p = _PARSERS[key] = Parser(lg)
    return key, p.parse(data)


def count_errors(root) -> int:
    n = 0
    stack = [root]
    while stack:
        x = stack.pop()
        if x.type == "ERROR" or x.is_missing:
            n += 1
        if x.has_error:
            stack.extend(x.children)
    return n


class Lines:
    """Byte offset -> (1-based line, 1-based code-point column)."""

    def __init__(self, data: bytes):
        self.data = data
        self.starts = [0]
        for i, b in enumerate(data):
            if b == 0x0A:
                self.starts.append(i + 1)

    def lc(self, off: int) -> tuple[int, int]:
        import bisect
        i = bisect.bisect_right(self.starts, off) - 1
        s = self.starts[i]
        col = len(self.data[s:off].decode("utf-8", "replace")) + 1
        return i + 1, col

    def text_lines(self) -> list[str]:
        return self.data.decode("utf-8", "replace").split("\n")


# ---- identifier extraction (callee's last identifier; ML §2 `calleeName`/`nameLine`)

IDENT_TYPES = {"identifier", "property_identifier", "field_identifier", "type_identifier", "simple_identifier",
               "private_property_identifier", "namespace_identifier", "shorthand_property_identifier",
               "this", "super", "statement_identifier"}
# type -> field holding the member name (last identifier lives there)
MEMBER_FIELD = {"attribute": "attribute", "member_expression": "property", "field_access": "field",
                "member_access_expression": "name", "selector_expression": "field", "field_expression": "field",
                "qualified_identifier": "name", "template_function": "name", "qualified_type": "name", "index_expression": "operand",
                "subscript_expression": None, }


def last_ident(n):
    """The last identifier of an expression/type node, or None."""
    if n is None:
        return None
    t = n.type
    if t in IDENT_TYPES:
        return n
    f = MEMBER_FIELD.get(t)
    if f:
        return last_ident(n.child_by_field_name(f))
    if t == "navigation_expression":                       # kotlin: ... . identifier
        for c in reversed(n.children):
            if c.type == "navigation_suffix":
                for d in reversed(c.children):
                    r = last_ident(d)
                    if r is not None:
                        return r
            r = last_ident(c) if c.type in IDENT_TYPES else None
            if r is not None:
                return r
        return None
    if t in ("parenthesized_expression", "non_null_expression", "as_expression", "satisfies_expression",
             "type_assertion", "await_expression", "unary_expression"):
        return None                                           # computed callee: <dynamic>
    if t in ("string", "string_literal", "subscript", "call", "call_expression", "lambda", "lambda_expression",
             "function_expression", "arrow_function", "invocation_expression", "element_access_expression",
             "subscript_expression", "new_expression"):
        return None
    # generic fallback: rightmost identifier child, never inside type arguments / parameters
    for c in reversed(n.named_children):
        if "argument" in c.type or "parameters" in c.type:
            continue
        r = last_ident(c)
        if r is not None:
            return r
    return None


@dataclass
class Site:
    start: int              # byte offset of callee expression
    end: int                # byte offset end of callee expression
    node_type: str
    kind: str
    name_node: Optional[object]
    callee_node: object = None


def _in_ancestor(n, types: set[str]) -> bool:
    p = n.parent
    while p is not None:
        if p.type in types:
            return True
        p = p.parent
    return False


def _kind_from_callee(lang: str, callee) -> str:
    if callee is None:
        return "other"
    t = callee.type
    if t in ("identifier", "simple_identifier"):
        if lang == "cpp" and re.fullmatch(r"[A-Z][A-Z0-9_]+", callee.text.decode("utf-8", "replace")):
            return "macro-shaped"
        return "plain"
    if t in ("qualified_identifier", "scoped_identifier", "template_function"):
        return "qualified"
    if t in ("attribute", "member_expression", "field_access", "member_access_expression", "selector_expression",
             "field_expression", "navigation_expression"):
        # a receiver that is a bare Capitalised identifier reads as a static/qualified call
        recv = None
        for c in callee.children:
            if c.type in ("identifier", "simple_identifier", "type_identifier"):
                recv = c
            break
        if recv is not None and recv.text[:1].isupper() and lang in ("java", "csharp", "kotlin", "javascript"):
            return "qualified"
        return "method"
    return "other"


def frame_sites(lang: str, key: str, root, data: bytes) -> list[Site]:
    """Frame sites of one parsed file (ML §4.1 node-type table)."""
    out: list[Site] = []

    def add(start_node, end_node, node, kind, name_node, callee=None):
        out.append(Site(start_node.start_byte, end_node.end_byte, node.type, kind, name_node, callee or start_node))

    stack = [root]
    while stack:
        n = stack.pop()
        t = n.type
        if lang == "python":
            if t == "call" and not _in_ancestor(n, {"decorator"}):
                f = n.child_by_field_name("function")
                if f is not None:
                    add(f, f, n, _kind_from_callee(lang, f), last_ident(f), f)
        elif lang == "javascript":
            if t == "call_expression" and not _in_ancestor(n, {"decorator"}):
                f = n.child_by_field_name("function")
                a = n.child_by_field_name("arguments")
                if f is not None:
                    k = "tagged-template" if (a is not None and a.type == "template_string") else _kind_from_callee(lang, f)
                    if f.type == "import":
                        k = "other"
                    add(f, f, n, k, last_ident(f), f)
            elif t == "new_expression" and not _in_ancestor(n, {"decorator"}):
                c = n.child_by_field_name("constructor")
                a = n.child_by_field_name("arguments")
                if c is not None:
                    end = a.prev_sibling if a is not None and a.prev_sibling is not None else c
                    add(n, c, n, "ctor", last_ident(c), n)
            elif t in ("jsx_opening_element", "jsx_self_closing_element"):
                nm = n.child_by_field_name("name")
                if nm is not None:
                    first = nm
                    while first.type == "member_expression":
                        first = first.child_by_field_name("object") or first.children[0]
                    if first.text[:1].isupper():
                        add(nm, nm, n, "jsx", last_ident(nm), nm)
        elif lang == "java":
            if t == "method_invocation":
                a = n.child_by_field_name("arguments")
                nm = n.child_by_field_name("name")
                o = n.child_by_field_name("object")
                k = "plain" if o is None else ("qualified" if (o.type in ("identifier", "type_identifier") and o.text[:1].isupper()) else "method")
                if o is not None and o.type == "this":
                    k = "method"
                out.append(Site(n.start_byte, nm.end_byte, t, k, nm, n))
            elif t == "object_creation_expression":
                ty = n.child_by_field_name("type")
                if ty is not None:
                    out.append(Site(n.start_byte, ty.end_byte, t, "ctor", last_ident(ty), n))
            elif t == "explicit_constructor_invocation":
                a = n.child_by_field_name("arguments")
                kw = None
                for c in n.children:
                    if c.type in ("this", "super"):
                        kw = c
                if kw is not None:
                    out.append(Site(n.start_byte, kw.end_byte, t, "ctor", kw, n))
        elif lang == "kotlin":
            if t == "call_expression":
                kids = [c for c in n.children]
                callee = kids[0] if kids else None
                if callee is not None:
                    # `f(x) { ... }` parses as call_expression(call_expression(f, args), annotated_lambda):
                    # the outer node is the same call, so it is not a second site.
                    outer_dup = callee.type == "call_expression" and len(kids) > 1 and kids[1].type == "annotated_lambda"
                    if not outer_dup:
                        add(callee, callee, n, _kind_from_callee(lang, callee), last_ident(callee), callee)
            elif t == "constructor_invocation" and n.parent is not None and n.parent.type == "delegation_specifier":
                ty = n.children[0] if n.children else None
                if ty is not None:
                    add(ty, ty, n, "ctor", last_ident(ty), ty)
        elif lang == "csharp":
            if t == "invocation_expression":
                f = n.child_by_field_name("function")
                if f is not None:
                    add(f, f, n, _kind_from_callee(lang, f), last_ident(f), f)
            elif t == "object_creation_expression":
                ty = n.child_by_field_name("type")
                if ty is not None:
                    out.append(Site(n.start_byte, ty.end_byte, t, "ctor", last_ident(ty), n))
        elif lang == "go":
            if t == "call_expression":
                f = n.child_by_field_name("function")
                if f is not None:
                    add(f, f, n, _kind_from_callee(lang, f), last_ident(f), f)
            elif t == "type_conversion_expression" and n.named_children:
                # tree-sitter-go parses `F[T](x)` (a generic call) and `T(x)` conversions of generic-looking
                # types this way; the labeller marks real conversions notACall (ML §4.1).
                f = n.named_children[0]
                add(f, f, n, "other", last_ident(f), f)
        elif lang == "cpp":
            if t == "call_expression":
                f = n.child_by_field_name("function")
                if f is not None:
                    add(f, f, n, _kind_from_callee(lang, f), last_ident(f), f)
            elif t == "new_expression":
                ty = n.child_by_field_name("type")
                if ty is not None:
                    out.append(Site(n.start_byte, ty.end_byte, t, "ctor", last_ident(ty), n))
        stack.extend(reversed(n.children))
    return out


def iter_frame(lang: str, root: Path, cfg: dict) -> Iterator[dict]:
    """Yield frame records in (path, start byte) order. Pure function of the tree + cfg."""
    for rel, p in iter_source_files(lang, root, cfg):
        if is_test_file(lang, rel):
            continue
        data = p.read_bytes()
        key, tree = parse_file(lang, rel, data)
        L = Lines(data)
        sites = frame_sites(lang, key, tree.root_node, data)
        sites.sort(key=lambda s: (s.start, s.end))
        seen: dict[str, int] = {}
        for s in sites:
            line, col = L.lc(s.start)
            text = data[s.start:s.end].decode("utf-8", "replace")
            ct = ws(text)
            nn = s.name_node
            if nn is not None:
                nl, nc = L.lc(nn.start_byte)
                cname = nn.text.decode("utf-8", "replace")
            else:
                nl, nc, cname = line, col, "<dynamic>"
            if lang == "javascript" and s.kind == "ctor":
                pass
            siteKey = f"{rel}:{line}:{col}:{ct}"
            c = seen.get(siteKey, 0) + 1
            seen[siteKey] = c
            if c > 1:
                siteKey = f"{siteKey}#{c}"
            yield {"path": rel, "line": line, "column": col, "callText": ct, "calleeName": cname, "nameLine": nl, "nameColumn": nc,
                   "kind": s.kind, "nodeType": s.node_type, "grammar": key, "siteKey": siteKey}


# ------------------------------------------------------------------ the hash-sort draw (ML §4.4)

def seed_for(kind: str, lang: str, sha: str, flow_set: str = "p0") -> str:
    if flow_set == "p1":
        return f"{SEED_PREFIX_P1[kind]}{lang}|{sha}"
    if kind == "sample":
        return f"ccg-p0a-v1|{lang}|{sha}"
    return f"{SEED_PREFIX[kind]}{lang}|{sha}"


def hash_sort(items: list[dict], seed: str, keyfield: str = "siteKey") -> list[tuple[str, dict]]:
    scored = [(sha256hex(seed + "|" + it[keyfield]), it) for it in items]
    scored.sort(key=lambda t: (t[0], t[1][keyfield]))
    return scored


def excerpt(lines: list[str], line: int, before: int = 3, after: int = 3) -> str:
    lo = max(1, line - before)
    hi = min(len(lines), line + after)
    w = len(str(hi))
    return "\n".join(f"{i:>{w}}| {lines[i - 1]}" for i in range(lo, hi + 1))


# ------------------------------------------------------------------ census (C1-C8 evidence, C6 idioms, exclusions)

COMMENT_TYPES = {"comment", "line_comment", "block_comment", "multiline_comment", "documentation_comment"}


def code_lines(lang: str, rel: str, data: bytes) -> int:
    """cloc-style: lines holding at least one non-blank character outside comment nodes."""
    key, tree = parse_file(lang, rel, data)
    L = Lines(data)
    n_lines = len(L.starts)
    covered = [False] * (n_lines + 2)
    # mark comment spans char-by-char only for lines that are entirely comment
    lines = L.text_lines()
    comment_ranges: list[tuple[int, int]] = []
    stack = [tree.root_node]
    while stack:
        x = stack.pop()
        if x.type in COMMENT_TYPES or x.type.endswith("comment"):
            comment_ranges.append((x.start_byte, x.end_byte))
            continue
        stack.extend(x.children)
    comment_ranges.sort()
    cnt = 0
    byte_pos = 0
    ri = 0
    for i, ln in enumerate(data.split(b"\n")):
        start, end = byte_pos, byte_pos + len(ln)
        byte_pos = end + 1
        if not ln.strip():
            continue
        # bytes of the line not inside a comment
        cur = start
        has_code = False
        for (cs, ce) in comment_ranges:
            if ce <= start or cs >= end:
                continue
            if cs > cur and data[cur:min(cs, end)].strip():
                has_code = True
                break
            cur = max(cur, min(ce, end))
        else:
            if cur < end and data[cur:end].strip():
                has_code = True
        if has_code:
            cnt += 1
    return cnt


def lang_lines(lang: str, root: Path, cfg: dict) -> dict:
    """Non-blank line counts per language over the whole tree minus vendored/skip dirs (C2) and the
    target language's cloc-style code lines over frame files (C3)."""
    by_lang: dict[str, int] = {}
    skip = set(vendored(cfg)) | ALWAYS_SKIP_DIRS | set(cfg.get("frameExcludeDirs") or [])
    ext2lang = {}
    for lg, es in OTHER_EXTS.items():
        for e in es:
            ext2lang.setdefault(e, lg)
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if d not in skip]
        for fn in fns:
            e = os.path.splitext(fn)[1].lower()
            lg = ext2lang.get(e)
            if not lg:
                continue
            try:
                n = sum(1 for l in open(Path(dp) / fn, "rb") if l.strip())
            except OSError:
                continue
            by_lang[lg] = by_lang.get(lg, 0) + n
    target_key = {"javascript": "javascript"}.get(lang, lang)
    tgt = by_lang.get(target_key, 0)
    total = sum(by_lang.values())
    code = 0
    files = 0
    err_files = 0
    for rel, p in iter_source_files(lang, root, cfg):
        data = p.read_bytes()
        if not data.strip():
            continue
        files += 1
        code += code_lines(lang, rel, data)
        _, tree = parse_file(lang, rel, data)
        if tree.root_node.has_error:
            err_files += 1
    return {"nonBlankLinesByLanguage": dict(sorted(by_lang.items(), key=lambda kv: -kv[1])),
            "targetNonBlankLines": tgt, "allSourceNonBlankLines": total,
            "targetShare": round(tgt / total, 4) if total else None,
            "frameFiles": files, "frameFileCodeLines": code, "framefilesWithParseErrors": err_files}


BROKER_TERMS = ["kafka", "rabbitmq", "amqp", "nats", "pulsar", "aws-sdk-sqs", "sqs", "sns", "pubsub", "pub/sub",
                "servicebus", "azure.messaging", "activemq", "jms", "mqtt", "zeromq", "nsq", "rocketmq"]
EXCL_TERMS = ["opentelemetry", "microservices-demo"]
TEXT_EXTS = {".gradle", ".kts", ".xml", ".toml", ".mod", ".sum", ".csproj", ".props", ".targets", ".json",
             ".yml", ".yaml", ".txt", ".cmake", ".properties", ".lock", ".sln", ".cfg", ".md"}
BUILD_NAMES = {"pom.xml", "build.gradle", "build.gradle.kts", "settings.gradle", "settings.gradle.kts", "go.mod",
               "go.sum", "CMakeLists.txt", "package.json", "pyproject.toml", "requirements.txt", "Directory.Packages.props",
               "Directory.Build.props", "conanfile.txt", "vcpkg.json", "libs.versions.toml"}


def grep_terms(lang: str, root: Path, cfg: dict, terms: list[str]) -> dict:
    """Whole-word-ish, case-insensitive grep for terms in build manifests (dependency view) and in the
    language's source (import view). Returns {term: {"manifest": [files], "source": [files]}} (capped)."""
    res = {t: {"manifest": [], "source": []} for t in terms}
    pats = {t: re.compile(r"(?<![a-z0-9])" + re.escape(t) + r"(?![a-z0-9])", re.I) for t in terms}
    skip = set(vendored(cfg)) | ALWAYS_SKIP_DIRS
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if d not in skip]
        for fn in fns:
            p = Path(dp) / fn
            ext = p.suffix.lower()
            is_manifest = fn in BUILD_NAMES or ext in {".csproj", ".props", ".targets", ".gradle", ".kts", ".sln", ".mod", ".cmake"}
            is_src = ext in EXTS[lang] or ext in OTHER_EXTS.get(lang, set())
            if not (is_manifest or is_src):
                continue
            try:
                txt = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            rel = p.relative_to(root).as_posix()
            for t, pat in pats.items():
                if pat.search(txt):
                    bucket = "manifest" if is_manifest else "source"
                    if len(res[t][bucket]) < 8:
                        res[t][bucket].append(rel)
    return {t: v for t, v in res.items() if v["manifest"] or v["source"]}


def _txt(n) -> str:
    return n.text.decode("utf-8", "replace")


def walk(root):
    stack = [root]
    while stack:
        n = stack.pop()
        yield n
        stack.extend(reversed(n.children))


JAVA_DI = {"Component", "Service", "Repository", "Controller", "RestController", "Inject", "Autowired", "Bean",
           "Configuration", "Named", "Singleton", "Path", "Provider", "ApplicationScoped", "RequestScoped",
           "SessionScoped", "Dependent", "Resource", "EJB", "Stateless", "Stateful", "ConfigurationProperties",
           "SpringBootApplication", "ControllerAdvice", "RestControllerAdvice", "Produces", "Provides"}
CS_ROUTE = {"HttpGet", "HttpPost", "HttpPut", "HttpDelete", "HttpPatch", "Route", "HttpHead", "HttpOptions"}


def idiom_census(lang: str, root: Path, cfg: dict) -> dict:
    out: dict = {}
    files = list(iter_source_files(lang, root, cfg))
    if lang == "java":
        di_type = di_field = 0
        ifaces: dict[str, int] = {}
        impls: dict[str, set] = {}
        for rel, p in files:
            data = p.read_bytes(); _, tree = parse_file(lang, rel, data)
            for n in walk(tree.root_node):
                if n.type in ("marker_annotation", "annotation"):
                    nm = n.child_by_field_name("name")
                    base = _txt(nm).split(".")[-1] if nm is not None else ""
                    if base in JAVA_DI:
                        par = n.parent.parent if n.parent is not None else None  # modifiers -> declaration
                        if par is not None and par.type in ("class_declaration", "interface_declaration", "constructor_declaration",
                                                            "method_declaration", "record_declaration", "enum_declaration"):
                            if par.type == "method_declaration" and base != "Bean" and base != "Provides":
                                continue
                            di_type += 1
                        elif par is not None and par.type == "field_declaration":
                            di_field += 1
                if n.type == "interface_declaration":
                    nm = n.child_by_field_name("name")
                    if nm is not None:
                        ifaces[_txt(nm)] = ifaces.get(_txt(nm), 0) + 1
                if n.type in ("class_declaration", "record_declaration", "enum_declaration", "interface_declaration"):
                    nm = n.child_by_field_name("name")
                    for c in n.children:
                        if c.type in ("super_interfaces", "extends_interfaces"):
                            for t in walk(c):
                                if t.type == "type_identifier":
                                    impls.setdefault(_txt(t), set()).add(_txt(nm) if nm is not None else "?")
        with_impl = sorted(i for i in ifaces if impls.get(i))
        out = {"diAnnotationsOnTypesCtorsOrBeanMethods": di_type, "diAnnotationsOnFields": di_field,
               "interfaces": len(ifaces), "interfacesWithInCorpusImplementation": len(with_impl)}
    elif lang == "kotlin":
        suspend = 0; recv_lambda_defs = 0; builder_uses = 0; ktor_dsl = 0
        builder_names: set[str] = set()
        trees = []
        for rel, p in files:
            data = p.read_bytes(); _, tree = parse_file(lang, rel, data)
            trees.append(tree)
            for n in walk(tree.root_node):
                if n.type == "function_declaration":
                    mods = [c for c in n.children if c.type == "modifiers"]
                    if mods and re.search(r"\bsuspend\b", _txt(mods[0])):
                        suspend += 1
                    # builder: a parameter whose type is a function type with receiver `T.() -> R`
                    for c in walk(n):
                        if c.type == "function_type" and c.children and c.children[0].type == "receiver_type":
                            nm = n.child_by_field_name("name") or next((x for x in n.children if x.type == "identifier"), None)
                            if nm is not None:
                                builder_names.add(_txt(nm)); recv_lambda_defs += 1
                            break
                        if c.type == "function_type" and re.match(r"^\s*[\w<>?,.\s]+\.\(", _txt(c)):
                            nm = next((x for x in n.children if x.type == "identifier"), None)
                            if nm is not None:
                                builder_names.add(_txt(nm)); recv_lambda_defs += 1
                            break
        STD = {"apply", "with", "run", "also", "let", "routing", "route", "install", "embeddedServer", "buildString",
               "buildList", "buildMap", "buildSet", "install"}
        KTOR = {"routing", "route", "get", "post", "put", "delete", "patch", "install", "authenticate", "embeddedServer",
                "webSocket", "intercept"}
        for tree in trees:
            for n in walk(tree.root_node):
                if n.type == "call_expression":
                    kids = n.children
                    has_lambda = any(c.type in ("annotated_lambda", "lambda_literal") for c in kids) or \
                        any(c.type == "value_arguments" and "{" in _txt(c) for c in kids[1:2])
                    callee = last_ident(kids[0]) if kids else None
                    cn = _txt(callee) if callee is not None else ""
                    if has_lambda and cn in builder_names:
                        builder_uses += 1
                    elif has_lambda and cn in ("apply", "with"):
                        builder_uses += 1
                    elif has_lambda and cn in KTOR:
                        ktor_dsl += 1
        out = {"suspendFun": suspend, "receiverLambdaFunctionsDefined": recv_lambda_defs,
               "builderDslUsesInCorpusBuilders+apply/with": builder_uses, "ktorRoutingDslLikeCalls": ktor_dsl}
    elif lang == "csharp":
        actions = 0
        for rel, p in files:
            data = p.read_bytes(); _, tree = parse_file(lang, rel, data)
            for n in walk(tree.root_node):
                if n.type == "method_declaration":
                    for al in (c for c in n.children if c.type == "attribute_list"):
                        for a in (c for c in al.children if c.type == "attribute"):
                            nm = a.child_by_field_name("name")
                            if nm is not None and _txt(nm).split(".")[-1] in CS_ROUTE:
                                actions += 1
                                break
        projs = sorted(str(p.relative_to(root)) for p in root.rglob("*.csproj"))
        slns = sorted(str(p.relative_to(root)) for p in list(root.rglob("*.sln")) + list(root.rglob("*.slnx")))
        out = {"attributeRoutedActions": actions, "csprojFiles": len(projs), "solutions": slns}
    elif lang == "go":
        ifaces: dict[str, set] = {}
        methods: dict[str, set] = {}
        for rel, p in files:
            data = p.read_bytes(); _, tree = parse_file(lang, rel, data)
            for n in walk(tree.root_node):
                if n.type == "type_spec":
                    nm = n.child_by_field_name("name"); ty = n.child_by_field_name("type")
                    if ty is not None and ty.type == "interface_type" and nm is not None:
                        ms = {_txt(c.child_by_field_name("name")) for c in walk(ty) if c.type in ("method_elem", "method_spec") and c.child_by_field_name("name") is not None}
                        ifaces[_txt(nm)] = ms
                elif n.type == "method_declaration":
                    r = n.child_by_field_name("receiver"); nm = n.child_by_field_name("name")
                    if r is not None and nm is not None:
                        rt = [x for x in walk(r) if x.type == "type_identifier"]
                        if rt:
                            methods.setdefault(_txt(rt[0]), set()).add(_txt(nm))
        def impl_count(iface_methods: set) -> int:
            return sum(1 for t, ms in methods.items() if iface_methods and iface_methods <= ms)
        multi = {k: v for k, v in ifaces.items() if len(v) >= 1}
        ge2 = {k: impl_count(v) for k, v in multi.items()}
        out = {"namedInterfaces": len(ifaces), "interfacesWith>=1Method": len(multi),
               "interfacesWith>=2InCorpusImplementers(name-based method-set check)": sum(1 for c in ge2.values() if c >= 2),
               "interfacesWith>=2Methods&>=2Implementers": sum(1 for k, c in ge2.items() if c >= 2 and len(multi[k]) >= 2)}
        gen = sorted({r for r, _ in iter_source_files(lang, root, cfg, include_generated=True)} - {r for r, _ in files})
        out["generatedFilesExcludedFromFrame"] = len(gen)
    elif lang == "cpp":
        defs: dict[str, int] = {}
        macro_use_names: set[str] = set()
        virt = ovr = 0
        texts = []
        for rel, p in files:
            data = p.read_bytes(); _, tree = parse_file(lang, rel, data)
            for n in walk(tree.root_node):
                if n.type == "preproc_function_def":
                    nm = n.child_by_field_name("name")
                    if nm is not None:
                        defs[_txt(nm)] = defs.get(_txt(nm), 0) + 1
                elif n.type == "virtual_specifier" and _txt(n) == "override":
                    ovr += 1
                elif n.type == "virtual":
                    virt += 1
            texts.append(data.decode("utf-8", "replace"))
        blob = "\n".join(texts)
        used = 0
        for name in defs:
            # used = invoked (NAME followed by `(`) somewhere other than its own #define line
            n_inv = len(re.findall(r"(?<![A-Za-z0-9_])" + re.escape(name) + r"\s*\(", blob))
            if n_inv > defs[name]:
                used += 1
        out = {"functionLikeMacroDefinitions": len(defs), "functionLikeMacrosInvokedBeyondTheirDefinition": used,
               "virtualKeywordOccurrences": virt, "overrideSpecifiers": ovr}
    elif lang == "python":
        out = {}
    return out


def has_tests(root: Path, cfg: dict) -> list[str]:
    hits = []
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if d not in ALWAYS_SKIP_DIRS]
        for d in dns:
            if d.lower() in ("test", "tests", "__tests__", "spec", "specs", "unittests", "unit_tests") or d.lower().endswith(".tests") or d.lower().endswith("tests"):
                hits.append(str((Path(dp) / d).relative_to(root).as_posix()))
        if len(hits) > 12:
            break
    return hits[:12]


def licence_head(root: Path) -> dict:
    for n in ("LICENSE", "LICENSE.md", "LICENSE.txt", "LICENCE", "COPYING", "LICENSE.TXT", "license"):
        p = root / n
        if p.is_file():
            t = p.read_text(errors="replace")
            return {"path": n, "head": ws(t[:240]), "sha256": hashlib.sha256(t.encode()).hexdigest()[:16]}
    return {"path": None}


def census(lang: str, root: Path | None = None, cfg: dict | None = None) -> dict:
    cfg = cfg if cfg is not None else corpus_cfg(lang)
    root = root or src_dir(lang)
    r = {"lang": lang, "root": str(root)}
    r.update(lang_lines(lang, root, cfg))
    r["licenceFile"] = licence_head(root)
    r["testDirs"] = has_tests(root, cfg)
    r["idioms"] = idiom_census(lang, root, cfg)
    r["brokerGrep"] = grep_terms(lang, root, cfg, BROKER_TERMS)
    r["exclusionGrep"] = grep_terms(lang, root, cfg, EXCL_TERMS)
    return r


# ------------------------------------------------------------------ definitions and qnames (flow population, ML §6.4)
# A deliberately small re-implementation of the bundle spec §4.1/§4.2 scope chain, enough to name the
# *named, bodied, function-like* definitions. Anonymous scopes are counted (they appear in a chain) but are
# never candidates. Overload signatures are used only to decide the `#N` duplicate suffix.

FN_TYPES = {
    "python": {"function_definition"},
    "javascript": {"function_declaration", "generator_function_declaration", "method_definition",
                   "function_expression", "arrow_function", "function"},
    "java": {"method_declaration", "constructor_declaration", "compact_constructor_declaration"},
    "kotlin": {"function_declaration", "secondary_constructor"},
    "csharp": {"method_declaration", "constructor_declaration", "local_function_statement", "operator_declaration",
               "destructor_declaration", "conversion_operator_declaration"},
    "go": {"function_declaration", "method_declaration"},
    "cpp": {"function_definition"},
}
ANON_TYPES = {
    "python": {"lambda"}, "javascript": set(), "java": {"lambda_expression"}, "kotlin": {"lambda_literal", "anonymous_function"},
    "csharp": {"lambda_expression", "anonymous_method_expression"}, "go": {"func_literal"}, "cpp": {"lambda_expression"},
}
TYPE_SCOPES = {
    "python": {"class_definition"},
    "javascript": {"class_declaration", "abstract_class_declaration", "interface_declaration", "enum_declaration", "class"},
    "java": {"class_declaration", "interface_declaration", "enum_declaration", "record_declaration", "annotation_type_declaration"},
    "kotlin": {"class_declaration", "object_declaration"},
    "csharp": {"class_declaration", "struct_declaration", "interface_declaration", "record_declaration",
               "record_struct_declaration", "enum_declaration"},
    "go": set(),
    "cpp": {"class_specifier", "struct_specifier", "union_specifier"},
}
BRANCH_TYPES = {
    "python": {"if_statement", "conditional_expression", "match_statement", "while_statement", "for_statement"},
    "javascript": {"if_statement", "ternary_expression", "switch_statement", "while_statement", "for_statement",
                   "for_in_statement", "do_statement"},
    "java": {"if_statement", "ternary_expression", "switch_expression", "switch_statement", "while_statement",
             "for_statement", "enhanced_for_statement", "do_statement"},
    "kotlin": {"if_expression", "when_expression", "while_statement", "for_statement", "do_while_statement"},
    "csharp": {"if_statement", "conditional_expression", "switch_statement", "switch_expression", "while_statement",
               "for_statement", "foreach_statement", "do_statement"},
    "go": {"if_statement", "expression_switch_statement", "type_switch_statement", "for_statement", "select_statement"},
    "cpp": {"if_statement", "conditional_expression", "switch_statement", "while_statement", "for_statement",
            "for_range_loop", "do_statement"},
}


def _field_text(n, field):
    c = n.child_by_field_name(field)
    return _txt(c) if c is not None else None


def _py_deco_suffix(n) -> list[str]:
    par = n.parent
    if par is not None and par.type == "decorated_definition":
        for d in par.children:
            if d.type == "decorator":
                t = ws(_txt(d))
                if t == "@property":
                    return ["<get>"]
                m = re.fullmatch(r"@\w+\.(setter|deleter)", t)
                if m:
                    return ["<set>" if m.group(1) == "setter" else "<del>"]
    return []


def _js_binding(n):
    """ES2015 inferred name for an anonymous function value (spec §4.2), or None."""
    par = n.parent
    while par is not None and par.type in ("parenthesized_expression", "as_expression", "satisfies_expression", "non_null_expression"):
        n, par = par, par.parent
    if par is None:
        return None
    if par.type == "variable_declarator" and par.child_by_field_name("value") is n:
        nm = par.child_by_field_name("name")
        return _txt(nm) if nm is not None and nm.type == "identifier" else None
    if par.type in ("public_field_definition", "field_definition") and par.child_by_field_name("value") is n:
        nm = par.child_by_field_name("name") or par.child_by_field_name("property")
        return _txt(nm) if nm is not None else None
    if par.type == "pair" and par.child_by_field_name("value") is n:
        k = par.child_by_field_name("key")
        if k is not None:
            kt = _txt(k)
            if k.type == "string":
                kt = kt.strip("\"'`")
            if re.fullmatch(r"[A-Za-z_$][\w$]*", kt):
                return kt
    return None


def _params_of(lang: str, n):
    """Parameter nodes of a function-like node, excluding the implicit receiver."""
    if lang == "python":
        p = n.child_by_field_name("parameters")
        ps = [c for c in (p.named_children if p is not None else []) if c.type != "comment"]
        out = []
        for i, c in enumerate(ps):
            t = _txt(c)
            if i == 0 and t in ("self", "cls"):
                continue
            out.append(c)
        return out
    if lang == "go":
        p = n.child_by_field_name("parameters")
        out = []
        for d in (p.named_children if p is not None else []):
            if d.type in ("parameter_declaration", "variadic_parameter_declaration"):
                names = [c for c in d.children if c.type == "identifier"]
                out.extend(names if names else [d])
        return out
    if lang == "cpp":
        d = n.child_by_field_name("declarator")
        while d is not None and d.type not in ("function_declarator",):
            d = d.child_by_field_name("declarator")
        p = d.child_by_field_name("parameters") if d is not None else None
        ps = [c for c in (p.named_children if p is not None else []) if c.type in ("parameter_declaration", "optional_parameter_declaration", "variadic_parameter_declaration")]
        return [c for c in ps if _txt(c).strip() != "void"]
    p = n.child_by_field_name("parameters")
    if p is None:
        for c in n.children:
            if c.type in ("formal_parameters", "function_value_parameters", "parameter_list"):
                p = c
                break
    if p is None and n.type == "arrow_function":
        pr = n.child_by_field_name("parameter")
        return [pr] if pr is not None else []
    return [c for c in (p.named_children if p is not None else []) if "parameter" in c.type or c.type in ("identifier", "required_parameter", "optional_parameter")]


def _sig_text(lang: str, n) -> str:
    if lang not in ("java", "kotlin", "csharp", "cpp"):
        return ""
    ps = _params_of(lang, n)
    parts = []
    for p in ps:
        t = p.child_by_field_name("type")
        parts.append(re.sub(r"\s+", "", _txt(t) if t is not None else _txt(p)))
    return "(" + ",".join(parts) + ")"


def _cpp_decl_name(n):
    """(qualifier segments, base) from a function_definition declarator."""
    d = n.child_by_field_name("declarator")
    while d is not None and d.type in ("pointer_declarator", "reference_declarator", "parenthesized_declarator", "attributed_declarator"):
        d = d.child_by_field_name("declarator") or (d.named_children[-1] if d.named_children else None)
    if d is None:
        return [], None
    if d.type == "function_declarator":
        d = d.child_by_field_name("declarator")
    if d is None:
        return [], None
    txt = re.sub(r"<[^<>]*(<[^<>]*>[^<>]*)*>", "", _txt(d))     # drop template args
    txt = re.sub(r"\s+", "", txt)
    parts = txt.split("::")
    parts = [p for p in parts if p != ""]
    if not parts:
        return [], None
    base = parts[-1]
    if base.startswith("operator"):
        base = base
    return parts[:-1], base


def _scope_info(lang: str, n, parent_segments_text: list[str]):
    """(kind, added_segments, sig) if node n opens a named scope/definition; else None.
    kind in {"fn", "type", "ns", "anon"}."""
    t = n.type
    if t in ANON_TYPES[lang]:
        return ("anon", None, "")
    if lang == "python":
        if t == "class_definition":
            return ("type", [_field_text(n, "name")], "")
        if t == "function_definition":
            return ("fn", [_field_text(n, "name")] + _py_deco_suffix(n), "")
    elif lang == "javascript":
        if t in ("class_declaration", "abstract_class_declaration", "interface_declaration", "enum_declaration"):
            return ("type", [_field_text(n, "name")], "")
        if t == "class":
            nm = _field_text(n, "name")
            return ("type", [nm], "") if nm else ("anon", None, "")
        if t in ("internal_module", "module"):
            return ("ns", [_field_text(n, "name") or "?"], "")
        if t in ("function_declaration", "generator_function_declaration"):
            return ("fn", [_field_text(n, "name")], "")
        if t == "method_definition":
            nm = _field_text(n, "name")
            if nm is None or n.child_by_field_name("name").type == "computed_property_name":
                return ("anon", None, "")
            if nm == "constructor":
                nm = "<init>"
            acc = [c.type for c in n.children if c.type in ("get", "set")]
            seg = [nm.strip("\"'")] + (["<get>"] if "get" in acc else ["<set>"] if "set" in acc else [])
            return ("fn", seg, "")
        if t in ("method_signature", "abstract_method_signature"):
            nm = _field_text(n, "name")
            return ("fn", [nm], "") if nm else None
        if t in ("function_expression", "function", "arrow_function"):
            nm = _field_text(n, "name")
            if nm:
                return ("fn", [nm], "")
            b = _js_binding(n)
            return ("fn", [b], "") if b else ("anon", None, "")
    elif lang == "java":
        if t in TYPE_SCOPES["java"]:
            return ("type", [_field_text(n, "name")], "")
        if t == "method_declaration":
            return ("fn", [_field_text(n, "name")], _sig_text(lang, n))
        if t in ("constructor_declaration", "compact_constructor_declaration"):
            return ("fn", ["<init>"], _sig_text(lang, n))
        if t == "object_creation_expression" and any(c.type == "class_body" for c in n.children):
            return ("anon", None, "")
    elif lang == "kotlin":
        if t == "class_declaration":
            nm = next((c for c in n.children if c.type == "identifier"), None)
            return ("type", [_txt(nm) if nm is not None else "?"], "")
        if t == "object_declaration":
            nm = next((c for c in n.children if c.type == "identifier"), None)
            return ("type", [_txt(nm) if nm is not None else "?"], "")
        if t == "companion_object":
            nm = next((c for c in n.children if c.type == "identifier"), None)
            return ("type", [_txt(nm) if nm is not None else "Companion"], "")
        if t == "function_declaration":
            nm = next((c for c in n.children if c.type == "identifier"), None)
            return ("fn", [_txt(nm).strip("`") if nm is not None else "?"], _sig_text(lang, n))
        if t == "secondary_constructor":
            return ("fn", ["<init>"], _sig_text(lang, n))
        if t == "object_literal":
            return ("anon", None, "")
    elif lang == "csharp":
        if t in ("namespace_declaration", "file_scoped_namespace_declaration"):
            nm = _field_text(n, "name") or "?"
            return ("ns", nm.split("."), "")
        if t in TYPE_SCOPES["csharp"]:
            nm = _field_text(n, "name") or "?"
            tp = n.child_by_field_name("type_parameters")
            if tp is not None:
                nm += "`" + str(len([c for c in tp.named_children if c.type == "type_parameter"]))
            return ("type", [nm], "")
        if t in ("method_declaration", "local_function_statement"):
            return ("fn", [_field_text(n, "name")], _sig_text(lang, n))
        if t == "constructor_declaration":
            return ("fn", ["<init>"], _sig_text(lang, n))
        if t == "destructor_declaration":
            return ("fn", ["~" + (_field_text(n, "name") or "?")], "")
        if t in ("operator_declaration",):
            return ("fn", ["operator" + (_field_text(n, "operator") or "?")], _sig_text(lang, n))
    elif lang == "go":
        if t == "type_spec":
            return ("type", [_field_text(n, "name")], "")
        if t == "method_elem" and n.child_by_field_name("name") is not None:
            return ("fn", [_field_text(n, "name")], "")
        if t == "function_declaration":
            return ("fn", [_field_text(n, "name")], "")
        if t == "method_declaration":
            r = n.child_by_field_name("receiver")
            rt = [x for x in walk(r) if x.type == "type_identifier"] if r is not None else []
            return ("fn", [(_txt(rt[0]) if rt else "?"), _field_text(n, "name")], "")
    elif lang == "cpp":
        if t == "namespace_definition":
            nm = _field_text(n, "name")
            return ("ns", (nm.split("::") if nm else ["<anonns>"]), "")
        if t in ("class_specifier", "struct_specifier", "union_specifier"):
            nm = n.child_by_field_name("name")
            return ("type", [re.sub(r"<.*>", "", _txt(nm)) if nm is not None else "<anon>"], "") if nm is not None else None
        if t in ("function_definition", "field_declaration", "declaration") and (
                t == "function_definition" or any(c.type in ("pure_virtual_clause", "default_method_clause") for c in n.children)
                or "= 0" in _txt(n)[-12:] or "= default" in _txt(n)[-16:]):
            q, base = _cpp_decl_name(n)
            if base is None:
                return None
            encl = parent_segments_text[-1] if parent_segments_text else None
            if base == (q[-1] if q else encl):
                base = "<init>"
            return ("fn", q + [base], _sig_text(lang, n))
    return None


@dataclass
class Def:
    path: str
    qname: str
    node: object
    lineStart: int
    lineEnd: int
    nparams: int
    named_fn: bool


def _name_node(lang: str, n):
    nm = n.child_by_field_name("name")
    if nm is not None:
        return nm
    if lang == "kotlin":
        return next((c for c in n.children if c.type == "identifier"), None)
    if lang == "cpp":
        d = n.child_by_field_name("declarator")
        while d is not None and d.type != "function_declarator":
            d = d.child_by_field_name("declarator") or (d.named_children[0] if d.named_children else None)
        d = d.child_by_field_name("declarator") if d is not None else None
        return d
    return None


def enumerate_defs(lang: str, rel: str, data: bytes, key: str, root, scopes: Optional[list] = None) -> list[Def]:
    """Bodied function-like definitions (the flow population). If `scopes` is a list, every named
    type/function scope (bodiless declarations included) is appended as a dict for reference mapping."""
    out: list[Def] = []
    L = Lines(data)
    anon_ctr: dict[tuple, int] = {}
    dup_ctr: dict[tuple, int] = {}
    fn_types = FN_TYPES[lang]

    def rec(n, chain: tuple, tchain: list[str]):
        info = _scope_info(lang, n, tchain)
        new_chain, new_t = chain, tchain
        if info is not None:
            kind, segs, sig = info
            if kind == "anon":
                k = anon_ctr.get(chain, 0) + 1
                anon_ctr[chain] = k
                new_chain = chain + (f"<anon>#{k}",)
                new_t = tchain + [f"<anon>#{k}"]
            else:
                segs = [s for s in segs if s]
                if kind == "fn":
                    k = (chain, tuple(segs), sig)
                    c = dup_ctr.get(k, 0) + 1
                    dup_ctr[k] = c
                    segs = segs[:-1] + [segs[-1] + (f"#{c}" if c > 1 else "")] if segs else segs
                new_chain = chain + tuple(segs)
                new_t = tchain + [segs[-1] if segs else "?"] if kind != "ns" else tchain + ([segs[-1]] if segs else [])
                if scopes is not None and kind in ("fn", "type") and segs:
                    nmn = _name_node(lang, n)
                    nl = L.lc((nmn or n).start_byte)[0]
                    body = n.child_by_field_name("body")
                    if lang == "kotlin":
                        body = next((c for c in n.children if c.type in ("function_body", "class_body")), None)
                    scopes.append({"qname": f"{rel}::{'.'.join(new_chain)}", "nameLine": nl, "base": segs[-1].split("#")[0],
                                   "kind": kind, "bodied": body is not None, "node": n.type})
                if kind == "fn" and n.type in fn_types and (n.child_by_field_name("body") is not None or lang in ("kotlin",)):
                    body = n.child_by_field_name("body")
                    if lang == "kotlin":
                        body = next((c for c in n.children if c.type in ("function_body",)), None)
                    if body is not None:
                        ls, _ = L.lc(n.start_byte)
                        le, _ = L.lc(max(n.end_byte - 1, n.start_byte))
                        out.append(Def(rel, f"{rel}::{'.'.join(new_chain)}", n, ls, le, len(_params_of(lang, n)), True))
        for c in n.children:
            rec(c, new_chain, new_t)

    # C# file-scoped namespace applies to the rest of the file: treat as a wrapper over following siblings
    rec(root, (), [])
    return out


def method_has_branch_and_call(lang: str, key: str, d: Def, data: bytes) -> tuple[bool, bool]:
    node = d.node
    br = False
    for x in walk(node):
        if x.type in BRANCH_TYPES[lang]:
            br = True
            break
    calls = [s for s in frame_sites(lang, key, node, data)]
    return br, bool(calls)


def flow_population(lang: str, root: Path, cfg: dict, lo: int = 5, hi: int = 25) -> list[dict]:
    pop = []
    for rel, p in iter_source_files(lang, root, cfg):
        if is_test_file(lang, rel):
            continue
        data = p.read_bytes()
        key, tree = parse_file(lang, rel, data)
        for d in enumerate_defs(lang, rel, data, key, tree.root_node):
            n_lines = d.lineEnd - d.lineStart + 1
            if not (lo <= n_lines <= hi) or d.nparams < 1:
                continue
            if "<anon>" in d.qname.split("::", 1)[1]:
                continue
            br, call = method_has_branch_and_call(lang, key, d, data)
            if br and call:
                text = data[d.node.start_byte:d.node.end_byte].decode("utf-8", "replace")
                pop.append({"path": rel, "qname": d.qname, "lineStart": d.lineStart, "lineEnd": d.lineEnd,
                            "source": text, "siteKey": f"{rel}:{d.lineStart}:{d.qname}"})
    return pop


# ------------------------------------------------------------------ fetch / digest

def tree_digest(root: Path) -> str:
    """sha256 over sorted (relative path, sha256(content)) of every file: pins the corpus bytes."""
    h = hashlib.sha256()
    for dp, dns, fns in os.walk(root):
        dns.sort()
        for fn in sorted(fns):
            p = Path(dp) / fn
            rel = p.relative_to(root).as_posix()
            h.update(rel.encode("utf-8") + b"\0" + hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw)


def fetch(lang: str, force: bool = False) -> dict:
    cfg = corpus_cfg(lang)
    dst = src_dir(lang)
    sha = cfg["commit"]
    if dst.exists():
        if not force:
            return {"lang": lang, "src": str(dst), "digest": tree_digest(dst), "reused": True}
        import shutil
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    if cfg.get("repo") == "proving":
        root = cfg["rootPath"].strip("/")
        strip = len(root.split("/"))
        ar = subprocess.run(["git", "-C", str(REPO), "archive", sha, root], check=True, capture_output=True)
        subprocess.run(["tar", "-x", "-C", str(dst), f"--strip-components={strip}"], input=ar.stdout, check=True)
    else:
        tmp = WORK / "corpora" / lang / ".fetch"
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        run(["git", "init", "-q", str(tmp)])
        run(["git", "-C", str(tmp), "fetch", "-q", "--depth", "1", cfg["url"], sha])
        got = run(["git", "-C", str(tmp), "rev-parse", "FETCH_HEAD"]).stdout.strip()
        if got != sha:
            raise SystemExit(f"fetched {got}, pinned {sha}")
        ar = subprocess.run(["git", "-C", str(tmp), "archive", "FETCH_HEAD"], check=True, capture_output=True)
        subprocess.run(["tar", "-x", "-C", str(dst)], input=ar.stdout, check=True)
        shutil.rmtree(tmp, ignore_errors=True)
    return {"lang": lang, "src": str(dst), "digest": tree_digest(dst), "reused": False}


# ------------------------------------------------------------------ frame / draw

def frame_path(lang): return HERE / "frame" / f"{lang}.jsonl"
def sample_path(lang): return HERE / "sample" / f"{lang}.jsonl"
def private_path(lang): return HERE / "sample" / f"{lang}.private.jsonl"
def flow_path(lang, fs=None): return HERE / "sample" / f"{lang}-flow{FLOW_SETS[fs or FLOW_SET]}.jsonl"
def brief_path(lang): return HERE / "sample" / f"{lang}.brief.md"


def build_frame(lang: str) -> list[dict]:
    cfg = corpus_cfg(lang)
    root = src_dir(lang)
    rows = []
    for r in iter_frame(lang, root, cfg):
        rows.append(r)
    return rows


def cmd_frame(lang: str) -> dict:
    rows = build_frame(lang)
    n = write_jsonl(frame_path(lang), rows)
    kinds: dict[str, int] = {}
    for r in rows:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    files = len({r["path"] for r in rows})
    return {"lang": lang, "frameSites": n, "files": files, "kinds": kinds}


def build_sample(lang: str, rows: list[dict]) -> tuple[list[dict], list[dict]]:
    cfg = corpus_cfg(lang)
    sha = cfg["commit"]
    order = hash_sort(rows, seed_for("sample", lang, sha))
    root = src_dir(lang)
    cache: dict[str, list[str]] = {}
    pub, priv = [], []
    for i, (h, r) in enumerate(order if lang in CENSUS_LANGS else order[:CAP], start=1):
        lines = cache.get(r["path"])
        if lines is None:
            lines = cache[r["path"]] = Lines((root / r["path"]).read_bytes()).text_lines()
        pub.append({"ordinal": i, "siteId": h[:16], "path": r["path"], "line": r["line"], "column": r["column"],
                    "callText": r["callText"], "excerpt": excerpt(lines, r["line"])})
        priv.append({"ordinal": i, "siteId": h[:16], "kind": r["kind"], "nodeType": r["nodeType"],
                     "calleeName": r["calleeName"], "nameLine": r["nameLine"], "siteKey": r["siteKey"],
                     "frameSize": len(rows)})
    return pub, priv


def cmd_draw(lang: str) -> dict:
    rows = read_jsonl(frame_path(lang))
    pub, priv = build_sample(lang, rows)
    write_jsonl(sample_path(lang), pub)
    write_jsonl(private_path(lang), priv)
    return {"lang": lang, "frame": len(rows), "sample": len(pub)}


def cmd_c5pilot(lang: str) -> dict:
    rows = read_jsonl(frame_path(lang))
    sha = corpus_cfg(lang)["commit"]
    order = hash_sort(rows, seed_for("c5", lang, sha))[:50]
    out = [{"pilotOrdinal": i, "siteId": h[:16], **{k: r[k] for k in ("path", "line", "column", "callText", "calleeName", "nameLine", "kind", "siteKey")}}
           for i, (h, r) in enumerate(order, start=1)]
    n = write_jsonl(HERE / "truth" / f"{lang}.c5-pilot.jsonl", out)
    return {"lang": lang, "pilot": n}


def build_flow(lang: str) -> list[dict]:
    cfg = corpus_cfg(lang)
    pop = flow_population(lang, src_dir(lang), cfg)
    order = hash_sort(pop, seed_for("flow", lang, cfg["commit"]))[:30]
    out = []
    for i, (h, m) in enumerate(order, start=1):
        out.append({"ordinal": i, "path": m["path"], "qname": m["qname"], "lineStart": m["lineStart"],
                    "lineEnd": m["lineEnd"], "source": m["source"], "_population": len(pop)})
    for o in out:
        o.pop("_population")
    return out


def flow_tier2_path(lang, fs=None): return HERE / "sample" / f"{lang}-flow{FLOW_SETS[fs or FLOW_SET]}-tier2.jsonl"


def build_flow_tier2(lang: str) -> list[dict]:
    """ML §8.4 two-stage rule (a population drawn whole): the methods of 26-60 lines under the same conditions, same seed,
    at most 3, generated now so that it exists before any label; the labeller is asked for it only if the tier-1 pairs
    are below 120."""
    cfg = corpus_cfg(lang)
    pop = flow_population(lang, src_dir(lang), cfg, 26, 60)
    order = hash_sort(pop, seed_for("flow", lang, cfg["commit"]))[:3]
    return [{"ordinal": i, "path": m["path"], "qname": m["qname"], "lineStart": m["lineStart"], "lineEnd": m["lineEnd"],
             "source": m["source"]} for i, (h, m) in enumerate(order, start=1)]


def cmd_flowdraw(lang: str) -> dict:
    if FLOW_SET != "p0":
        raise SystemExit("flowdraw is the P0 draw; the p1 draw is `flowdraw-p1` (it applies the custodian exclusion)")
    cfg = corpus_cfg(lang)
    pop = flow_population(lang, src_dir(lang), cfg)
    order = hash_sort(pop, seed_for("flow", lang, cfg["commit"]))[:30]
    out = [{"ordinal": i, "path": m["path"], "qname": m["qname"], "lineStart": m["lineStart"], "lineEnd": m["lineEnd"],
            "source": m["source"]} for i, (h, m) in enumerate(order, start=1)]
    n = write_jsonl(flow_path(lang), out)
    res = {"lang": lang, "population": len(pop), "flowMethods": n}
    if lang in CENSUS_LANGS:
        res["tier2"] = write_jsonl(flow_tier2_path(lang), build_flow_tier2(lang))
    return res


# ------------------------------------------------------------------ P1 flow draw (P1-34a; prereg §1.1, §3)
# Frame = the P0-ML §6.4 population minus the 30 P0 methods minus every method whose span contains the definition line of a custodian-held
# anchor (file and span, so an enclosing method of a nested anchor goes too). LEAK CONTROL: the exclusion file is read into memory only; the
# unfiltered order is NEVER computed (no prefix count, no skipped position); only the two frame-wide exclusion counts leave.

def _strict_rel(p: str) -> bool:
    """The one accepted path form: relative to the corpus root, `/`-separated, no empty/`.`/`..` segment, no backslash, no leading slash."""
    return bool(p) and "\\" not in p and all(seg not in ("", ".", "..") for seg in p.split("/"))


def _line_count(f: Path) -> int:
    data = f.read_bytes()
    return data.count(b"\n") + (1 if data and not data.endswith(b"\n") else 0)


def _entry_in_corpus(src: Path, path: str, line: int, counts: dict) -> bool:
    """The entry names a file of the corpus (exact strict path, resolved inside it) and 1 <= defLine <= the file's line count."""
    if not _strict_rel(path):
        return False
    f = src / path
    try:
        if not f.is_file() or src.resolve() not in f.resolve().parents:
            return False
        n = counts.setdefault(path, _line_count(f))
    except OSError:
        return False
    return 1 <= line <= n


def load_exclusions(path: Path, src: Optional[Path] = None) -> list[tuple[str, int]]:
    """JSON lines {"path", "defLine"} -> [(path, defLine)]. Memory only; an error names a line number, never its content.
    With `src` (the corpus root) EVERY entry must also be a file of the corpus with 1 <= defLine <= its line count (fail closed: an entry in
    the wrong path form would match nothing and the draw would silently be the unfiltered one)."""
    out: list[tuple[str, int]] = []
    counts: dict = {}
    try:
        f = open(path, encoding="utf-8")
    except OSError:
        raise SystemExit("exclusion file: cannot be read") from None      # no path, no strerror: neither is echoed
    try:
        with f:
            for no, ln in enumerate(f, 1):
                if not ln.strip():
                    continue
                try:
                    d = json.loads(ln)
                    ok = isinstance(d, dict) and isinstance(d["path"], str) and type(d["defLine"]) is int
                except (ValueError, KeyError, RecursionError):
                    ok = False
                if not ok:
                    raise SystemExit(f"exclusion file: line {no} is not {{\"path\": str, \"defLine\": int}}") from None
                if src is not None and not _entry_in_corpus(src, d["path"], d["defLine"], counts):
                    raise SystemExit(f"exclusion file: line {no} is not a file of the corpus with 1 <= defLine <= its line count") from None
                out.append((d["path"], d["defLine"]))
    except UnicodeDecodeError:
        raise SystemExit("exclusion file: not valid UTF-8") from None     # the codec message carries a byte value and an offset: not echoed
    return out


def _is_excluded(m: dict, excl: list[tuple[str, int]]) -> bool:
    return any(m["path"] == p and m["lineStart"] <= line <= m["lineEnd"] for p, line in excl)


def draw_flow_p1(pop: list[dict], seed: str, n: int, drop_keys: set[str], excl: list[tuple[str, int]]) -> tuple[list[dict], int]:
    """-> (first n rows of the FILTERED hash order, excluded in the frame). The unfiltered order is never sorted: the filter runs first."""
    base = [m for m in pop if m["siteKey"] not in drop_keys]
    k_frame = sum(1 for m in base if _is_excluded(m, excl))
    kept = [m for m in base if not _is_excluded(m, excl)]
    rows = [{"ordinal": i, "path": m["path"], "qname": m["qname"], "lineStart": m["lineStart"], "lineEnd": m["lineEnd"],
             "source": m["source"]} for i, (_, m) in enumerate(hash_sort(kept, seed)[:n], start=1)]
    return rows, k_frame


def _p1_target(path: Path) -> Path:
    if "-p1" not in path.name:
        raise SystemExit(f"flowdraw-p1: refusing to write a non-p1 path: {path.name}")
    return path


def _versions() -> dict:
    from importlib import metadata
    return {"treeSitter": metadata.version("tree-sitter"), "grammarPython": metadata.version("tree-sitter-python")}


def _write_all(items: list[tuple[Path, str]]) -> None:
    """Atomic-or-nothing: stage EVERY output as a temp file first, rename only when all are staged. A failure while staging removes the temp
    files and leaves no output (and echoes no path)."""
    staged: list[tuple[Path, Path]] = []
    try:
        for path, text in items:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            staged.append((tmp, path))
            with open(tmp, "w", encoding="utf-8", newline="\n") as f:
                f.write(text)
    except OSError:
        for tmp, _ in staged:
            tmp.unlink(missing_ok=True)
        raise SystemExit("flowdraw-p1: cannot write the outputs; nothing written") from None
    for tmp, path in staged:
        os.replace(tmp, path)


def cmd_flowdraw_p1(lang: str, exclude: Path, counts_out: Path, tier2_out: Path, expect: Optional[tuple] = None) -> dict:
    if REPO in (tier2_out.resolve(), *tier2_out.resolve().parents):
        raise SystemExit("flowdraw-p1: --tier2-out must be outside the repository")
    cfg = corpus_cfg(lang)
    p0 = read_jsonl(flow_path(lang, "p0"))                       # read only: the 30 P0 methods (public)
    drop = {f"{r['path']}:{r['lineStart']}:{r['qname']}" for r in p0}
    seed = seed_for("flow", lang, cfg["commit"], "p1")
    registered = tree_digest(src_dir(lang)) == P1_REAL_SRC_DIGEST
    if seed == seed_for("flow", lang, P1_REAL_COMMIT, "p1") and not registered:
        raise SystemExit("flowdraw-p1: the registered seed is drawn only on the registered corpus (srcDigest mismatch); use another commit string for a test")
    if expect is None:
        if not registered:
            raise SystemExit("flowdraw-p1: the registered frame counts apply to the registered corpus only; pass --expect-counts for another one")
        expect = P1_FRAME_COUNTS
    excl = load_exclusions(exclude, src_dir(lang))
    if not excl:
        raise SystemExit("exclusion file: no entries (an empty exclusion would draw the unfiltered order)")
    pop1 = flow_population(lang, src_dir(lang), cfg)
    pop2 = flow_population(lang, src_dir(lang), cfg, 26, 60)
    if len(drop & {m["siteKey"] for m in pop1}) != len(drop):
        raise SystemExit("flowdraw-p1: a P0 sample method is not in the P0-ML 6.4 population (corpus drift?)")
    got = (len(pop1), len(pop1) - len(drop), len(pop2))          # asserted BEFORE the exclusion is applied
    if got != tuple(expect):
        raise SystemExit(f"flowdraw-p1: frame counts {got} differ from the expected {tuple(expect)} (population, minus P0, tier-2); nothing written")
    rows1, k1f = draw_flow_p1(pop1, seed, FLOW_P1_TIER1, drop, excl)
    rows2, k2f = draw_flow_p1(pop2, seed, FLOW_P1_TIER2, drop, excl)
    res = {"excludedTier1Frame": k1f, "excludedTier2Frame": k2f}      # exactly these two integers leave (the digests are of public files: `sha256sum`)
    _write_all([(tier2_out, "".join(jdump(r) + "\n" for r in rows2)),     # custodian-held: never inside the repo
                (_p1_target(flow_path(lang, "p1")), "".join(jdump(r) + "\n" for r in rows1)),
                (counts_out, json.dumps(res, indent=1, sort_keys=True) + "\n")])
    return {**res, **_versions()}


# ------------------------------------------------------------------ reference mapping (ML §5.2 "Mapping a reference answer to a site", §11 N1)
# ref.jsonl line: {ordinal, siteId, static, impls, tool, toolVersion}
#   static = {"kind": "internal"|"declared", "path", "qname", "defLine"}   in-corpus target (declared = bodiless)
#          | {"kind": "external", "package"}   | {"kind": "local"}   | {"kind": "dispatch"} (Go: several edges)
#          | {"kind": "silent"}                 no reference occurrence / edge at the site
#   impls  = [{"path","qname","defLine"}]       in-corpus implementations (SCIP is_implementation closure; Go: all edges)

SCIP_HASH = re.compile(r"^[0-9a-f]{16}$")
GO_BUILTINS = {"append", "cap", "clear", "close", "complex", "copy", "delete", "imag", "len", "make", "max", "min",
               "new", "panic", "print", "println", "real", "recover"}
# ML §3.2 C5: sites that are reference-silent by construction (closed list; tagged from the source alone, only on a
# site still silent after §5.2's mapping). A class is added only by a revision of the ML note.
CLOSED_CLASSES = {"go": ("builtin",), "csharp": ("generic-callee", "nameof"), "cpp": ("preproc-branch", "macro-callee")}
EXCLUSION_CAP = 10          # ML §3.2: the excluded sites are <= 10 of the 50



def scip_name(symbol: str) -> str:
    parts = symbol.split(" ", 4)
    desc = parts[4] if len(parts) >= 5 else symbol
    toks = re.findall(r"`[^`]*`|[^/#.\(\)\[\]:!\s]+", desc)
    toks = [t.strip("`") for t in toks if not SCIP_HASH.match(t.strip("`"))]
    return toks[-1] if toks else ""


def scip_package(symbol: str) -> str:
    parts = symbol.split(" ", 4)
    if len(parts) >= 3:
        return parts[2]
    return "?"


def _vendored_path(path: str, cfg: dict) -> bool:
    return any(seg in set(vendored(cfg)) for seg in path.split("/")[:-1])


class ScopeIndex:
    """(path, nameLine) -> scope records, built lazily from the corpus with tree-sitter."""

    def __init__(self, lang: str, root: Path, cfg: dict):
        self.lang, self.root, self.cfg = lang, root, cfg
        self._by_path: dict[str, dict[int, list[dict]]] = {}

    def at(self, path: str) -> dict[int, list[dict]]:
        m = self._by_path.get(path)
        if m is None:
            m = {}
            p = self.root / path
            if p.is_file() and p.suffix.lower() in EXTS[self.lang]:
                data = p.read_bytes()
                key, tree = parse_file(self.lang, path, data)
                scopes: list[dict] = []
                enumerate_defs(self.lang, path, data, key, tree.root_node, scopes)
                for sc in scopes:
                    m.setdefault(sc["nameLine"], []).append(sc)
            self._by_path[path] = m
        return m

    def lookup(self, path: str, line: int, name: str) -> Optional[dict]:
        recs = self.at(path).get(line, [])
        for r in recs:
            if r["base"] == name or (name == "<init>" and r["base"] == "<init>"):
                return r
        if name in ("<init>", "", "`.ctor`", ".ctor"):
            # an implicit/default constructor is the class itself (ML §6.5: "none explicit: the class")
            for r in recs:
                if r["kind"] == "type":
                    return r
        return recs[0] if len(recs) == 1 and not name else None


def _occ_range(o: dict) -> tuple[int, int, int, int]:
    """(line0, col0, line1, col1), 0-based. `scip print --json` writes either `range` (python/ts indexers) or a
    TypedRange {SingleLineRange | MultiLineRange} (java/dotnet/clang), depending on how the indexer built it."""
    r = o.get("range") or []
    if len(r) == 3:
        return r[0], r[1], r[0], r[2]
    if len(r) >= 4:
        return r[0], r[1], r[2], r[3]
    tr = o.get("TypedRange") or {}
    sl = tr.get("SingleLineRange")
    if sl is not None:
        return sl.get("line", 0), sl.get("start_character", 0), sl.get("line", 0), sl.get("end_character", 0)
    ml = tr.get("MultiLineRange")
    if ml is not None:
        return (ml.get("start_line", 0), ml.get("start_character", 0), ml.get("end_line", 0), ml.get("end_character", 0))
    return 0, 0, 0, 0


def _site_of(pub: dict, fr: dict) -> dict:
    """The public coordinates of a site (the same four fields sample/<lang>.jsonl shows the labeller), so that
    make_adjudication_input.py can work from labels/ and truth/ alone."""
    return {k: fr[k] for k in ("path", "line", "column", "callText")}


def refmap_scip(lang: str, scip_json: dict, sample: list[dict], priv: list[dict], tool_version: str,
                frame_by_key: dict) -> list[dict]:
    cfg = corpus_cfg(lang)
    root = src_dir(lang)
    sidx = ScopeIndex(lang, root, cfg)
    docs = scip_json.get("documents", [])
    # definitions: symbol -> (path, line1)
    defs: dict[str, tuple[str, int]] = {}
    local_defs: set[tuple[str, str]] = set()      # (path, local symbol) with a definition occurrence in that document
    occ_by_path_line: dict[tuple[str, int], list[dict]] = {}
    impl_edges: dict[str, set[str]] = {}
    for d in docs:
        path = d.get("relative_path", "")
        for o in d.get("occurrences", []):
            l0, c0, l1, c1 = _occ_range(o)
            roles = o.get("symbol_roles", 0)
            if roles & 1:
                defs.setdefault(o["symbol"], (path, l0 + 1))
                if o["symbol"].startswith("local "):
                    local_defs.add((path, o["symbol"]))
            else:
                occ_by_path_line.setdefault((path, l0 + 1), []).append({"sym": o["symbol"], "c0": c0, "c1": c1, "roles": roles})
        for s in d.get("symbols", []):
            for r in s.get("relationships", []):
                if r.get("is_implementation"):
                    impl_edges.setdefault(r["symbol"], set()).add(s["symbol"])
    src_lines: dict[str, list[str]] = {}

    def line_text(path: str, line: int) -> str:
        ls = src_lines.get(path)
        if ls is None:
            ls = src_lines[path] = (root / path).read_text(encoding="utf-8", errors="replace").split("\n")
        return ls[line - 1] if 0 < line <= len(ls) else ""

    def def_record(sym: str) -> Optional[dict]:
        pl = defs.get(sym)
        if pl is None:
            return None
        path, line = pl
        if _vendored_path(path, cfg) or not (root / path).is_file():
            return None          # vendored, or a document the tool generated that is not in the corpus => external
        sc = sidx.lookup(path, line, scip_name(sym))
        return {"path": path, "qname": sc["qname"] if sc else None, "defLine": line,
                "declared": bool(sc and not sc["bodied"])}

    def closure(sym: str) -> list[str]:
        seen, stack = set(), [sym]
        while stack:
            x = stack.pop()
            for y in impl_edges.get(x, ()):
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        return sorted(seen)

    def static_of(sym: str, site_path: str) -> tuple[dict, list]:
        if sym.startswith("local "):
            # a local with no definition in its document is how scip-python reports a builtin function
            # (len, isinstance, ...); a local that is defined there is a variable holding a callable.
            st = {"kind": "local"} if (site_path, sym) in local_defs else {"kind": "external", "package": "unresolved-local-no-definition"}
            return st, []
        rec = def_record(sym)
        if rec is None:
            return {"kind": "external", "package": scip_package(sym)}, []
        st = {"kind": "declared" if rec["declared"] else "internal", "path": rec["path"], "qname": rec["qname"], "defLine": rec["defLine"]}
        impls = []
        for y in closure(sym):
            yr = def_record(y)
            if yr is not None:
                impls.append({"path": yr["path"], "qname": yr["qname"], "defLine": yr["defLine"]})
        impls.sort(key=lambda r: (r["path"], r["defLine"]))
        return st, impls

    out = []
    ts_cache: dict = {}
    for pub, pr in zip(sample, priv):
        fr = frame_by_key[pr["siteKey"]]
        nl, nc, name = fr["nameLine"], fr["nameColumn"], fr["calleeName"]
        cands = occ_by_path_line.get((fr["path"], nl), [])
        text = line_text(fr["path"], nl)
        exact = [c for c in cands if c["c0"] == nc - 1]
        if not exact:
            exact = [c for c in cands if text[c["c0"]:c["c1"]] == name]
        # prefer a method-like symbol (ends with ").") over a type reference at the same token
        exact.sort(key=lambda c: (0 if c["sym"].endswith(").") else 1, c["sym"]))
        flags: dict = {}
        if exact:
            st, impls = static_of(exact[0]["sym"], fr["path"])
            tags: list[str] = []
        else:
            st, impls = {"kind": "silent"}, []
            tags = site_tags(lang, fr, ts_cache)
            if lang == "cpp":
                # ML §5.2 "C/C++ macro-argument sites": window from the outermost enclosing macro invocation to the site.
                # Applied to every still-silent site; the by-construction classes (§3.2) are tagged afterwards, on the
                # sites that stay silent ("still reference-silent after §5.2's mapping").
                start = macro_invocation_start(lang, fr, ts_cache)
                if start is not None:
                    syms = macro_window_symbols(occ_by_path_line, fr["path"], start, nl, name)
                    if len(syms) == 1:
                        st, impls = static_of(syms[0], fr["path"])
                        flags["relaxed"] = True
                        if len(macro_window_symbols(occ_by_path_line, fr["path"], start, nl, name, merge_type_ctor=False)) > 1:
                            flags["typeCtorMerged"] = True
                    elif len(syms) > 1:
                        flags["ambiguous"] = True
        row = {"ordinal": pub["ordinal"], "siteId": pub["siteId"], "site": _site_of(pub, fr), "static": st, "impls": impls,
               "tool": scip_json.get("metadata", {}).get("tool_info", {}).get("name", "scip"), "toolVersion": tool_version}
        row.update(flags)
        if st["kind"] == "silent":
            cl = [t for t in tags if t in CLOSED_CLASSES.get(lang, ())]
            if cl:
                row["byConstruction"] = cl
        out.append(row)
    return out


GO_EDGE = re.compile(r"^(?P<caller>.+?) -> (?P<callee>.+?) @ (?P<file>.+?):(?P<line>\d+):(?P<col>\d+)$")


def go_callee_parts(callee: str) -> tuple[str, Optional[str], str]:
    """'(*pkg/path.Type).Method' | 'pkg/path.Func' -> (pkg path, receiver type or None, name)."""
    m = re.fullmatch(r"\(\*?([^()]+)\.([A-Za-z_]\w*)\)\.([A-Za-z_]\w*)", callee)
    if m:
        return m.group(1), m.group(2), m.group(3)
    m = re.fullmatch(r"(.+)\.([A-Za-z_]\w*)", callee)
    if m:
        return m.group(1), None, m.group(2)
    return "", None, callee


def refmap_go(callgraph_text: str, module_path: str, sample: list[dict], priv: list[dict], tool_version: str,
              frame_by_key: dict, repo_prefix: str) -> list[dict]:
    """callgraph -algo=vta lines `caller -> callee @ file:line:col`. `repo_prefix` is the absolute dir the
    tool ran in (stripped from file names)."""
    cfg = corpus_cfg("go")
    root = src_dir("go")
    sidx = ScopeIndex("go", root, cfg)
    edges: dict[tuple[str, int], list[tuple[str, str]]] = {}
    for ln in callgraph_text.splitlines():
        m = GO_EDGE.match(ln.strip())
        if not m:
            continue
        f = m.group("file")
        if f.startswith(repo_prefix):
            f = f[len(repo_prefix):].lstrip("/")
        edges.setdefault((f, int(m.group("line"))), []).append((m.group("callee"), m.group("col")))
    pkg_dirs: dict[str, str] = {}

    def target(callee: str):
        pkg, recv, name = go_callee_parts(callee)
        if not pkg.startswith(module_path):
            return {"kind": "external", "package": ("stdlib:" + pkg) if "." not in pkg.split("/")[0] else pkg}
        d = pkg[len(module_path):].lstrip("/")
        want = f"{recv}.{name}" if recv else name
        dirpath = root / d if d else root
        for p in sorted(dirpath.glob("*.go")) if dirpath.is_dir() else []:
            rel = p.relative_to(root).as_posix()
            for line_recs in sidx.at(rel).values():
                for sc in line_recs:
                    if sc["qname"].split("::", 1)[1] == want:
                        return {"kind": "internal" if sc["bodied"] else "declared", "path": rel, "qname": sc["qname"], "defLine": sc["nameLine"]}
        return {"kind": "external", "package": pkg + " (unresolved in corpus)"}

    out = []
    for pub, pr in zip(sample, priv):
        fr = frame_by_key[pr["siteKey"]]
        es = [(c, col) for (c, col) in edges.get((fr["path"], fr["line"]), [])
              if go_callee_parts(c)[2] == fr["calleeName"]]
        if not es:
            es = [(c, col) for (c, col) in edges.get((fr["path"], fr["nameLine"]), []) if go_callee_parts(c)[2] == fr["calleeName"]]
        tg = []
        for c, _ in es:
            t = target(c)
            if t not in tg:
                tg.append(t)
        if not tg:
            st, impls = {"kind": "silent"}, []
        elif len(tg) == 1:
            st, impls = tg[0], []
        else:
            st = {"kind": "dispatch"}
            impls = [{"path": t["path"], "qname": t["qname"], "defLine": t["defLine"]} for t in tg if t["kind"] in ("internal", "declared")]
        row = {"ordinal": pub["ordinal"], "siteId": pub["siteId"], "site": _site_of(pub, fr), "static": st, "impls": impls,
               "tool": "callgraph-vta", "toolVersion": tool_version}
        if st["kind"] == "silent":
            cl = [t for t in site_tags("go", fr, {}) if t in CLOSED_CLASSES["go"]]
            if cl:
                row["byConstruction"] = cl
        out.append(row)
    return out


PP_COND = {"preproc_if", "preproc_ifdef", "preproc_elif", "preproc_else", "preproc_elifdef"}


def csharp_generic_classes(row: dict) -> list[str]:
    """C# by-construction classes read from the callee text alone: `scip-dotnet` emits no occurrence for a callee with a
    type-argument list (`F<T>(...)`, `new C<T>(...)`) or for `nameof` (ML §3.2)."""
    tags = []
    if row.get("calleeName") == "nameof":
        tags.append("nameof")
    if _ends_in_type_arguments(row["callText"]):
        tags.append("generic-callee")
    return tags


def _ends_in_type_arguments(call_text: str) -> bool:
    """True iff the callee text ends in a balanced `<...>` list that follows an identifier (`a.B<int>`, `new C<T>`,
    `F<Func<int, int>>`). A `<` elsewhere (a comparison in an argument, a generic argument inside a receiver call) does not
    make the callee generic."""
    t = call_text.rstrip()
    if not t.endswith(">"):
        return False
    depth = 0
    for i in range(len(t) - 1, -1, -1):
        c = t[i]
        if c == ">":
            depth += 1
        elif c == "<":
            depth -= 1
            if depth == 0:
                before = t[:i].rstrip()
                return bool(before) and (before[-1].isalnum() or before[-1] == "_")
    return False


def _site_node(lang: str, row: dict, cache: dict):
    """(node at the callee's first character, its byte offset, parse tree root) for a frame/pilot row."""
    path = row["path"]
    ent = cache.get(path)
    if ent is None:
        data = (src_dir(lang) / path).read_bytes()
        ent = cache[path] = (data, parse_file(lang, path, data)[1], Lines(data))
    data, t, L = ent
    off = L.starts[row["line"] - 1] + len("".join(L.text_lines()[row["line"] - 1][: row["column"] - 1]).encode("utf-8"))
    return t.root_node.descendant_for_byte_range(off, off + 1), off, t.root_node, data


def macro_invocation_start(lang: str, row: dict, cache: dict) -> Optional[int]:
    """ML §5.2: first line (1-based) of the OUTERMOST macro-shaped call whose argument list contains the site, or None."""
    if lang != "cpp":
        return None
    node, off, _, _ = _site_node(lang, row, cache)
    outer, x = None, node
    while x is not None and x.parent is not None:
        p = x.parent
        if p.type == "call_expression":
            fn, args = p.child_by_field_name("function"), p.child_by_field_name("arguments")
            if fn is not None and args is not None and _kind_from_callee("cpp", fn) == "macro-shaped" \
                    and args.start_byte <= off < args.end_byte:
                outer = p
        x = p
    return None if outer is None else outer.start_point[0] + 1


def macro_window_symbols(occ_by_path_line: dict, path: str, start_line: int, name_line: int, name: str,
                         merge_type_ctor: bool = True) -> list[str]:
    """ML §5.2: distinct symbols of the non-local, non-definition occurrences in lines start_line..name_line (both inclusive)
    whose last descriptor name equals the site's callee name. (`occ_by_path_line` holds reference occurrences only.)
    A constructor call has two occurrences at one token, the type `P/X#` and the constructor `P/X#X(<hash>).`; they are one
    reference, so the type symbol is dropped when its constructor is in the window (as the strict mapping prefers the
    method-like symbol at a token)."""
    syms = set()
    for ln in range(start_line, name_line + 1):
        for c in occ_by_path_line.get((path, ln), ()):
            if not c["sym"].startswith("local ") and scip_name(c["sym"]) == name:
                syms.add(c["sym"])
    if merge_type_ctor:
        for t in [x for x in syms if x.endswith("#")]:
            if any(u != t and u.startswith(t) and u.endswith(").") and "/" not in u[len(t):] and "#" not in u[len(t):] for u in syms):
                syms.discard(t)
    return sorted(syms)


def site_tags(lang: str, row: dict, cache: dict) -> list[str]:
    """Why a reference may legitimately be silent at a frame site (structure, not tool quality). The closed list of ML §3.2
       (CLOSED_CLASSES): go 'builtin'; csharp 'generic-callee', 'nameof'; cpp 'preproc-branch' (under a non-guard
       #if/#ifdef/#else group: the compilation database builds ONE configuration, the tree-sitter frame sees every branch),
       'macro-callee' (the callee is itself a macro). Informational, not a class: cpp 'error-region' (tree-sitter ERROR node)."""
    tags = []
    if lang == "csharp":
        tags += csharp_generic_classes(row)
    if lang == "go" and row["calleeName"] in GO_BUILTINS:
        tags.append("builtin")
    if lang == "cpp":
        n, _, root, data = _site_node(lang, row, cache)
        guard = None
        for c in root.children:
            if c.type in ("preproc_ifdef", "preproc_if") and (c.end_byte - c.start_byte) > 0.8 * len(data):
                guard = c
        x, in_pp, in_err = n, False, False
        while x is not None:
            if x.type in PP_COND and (guard is None or x != guard):
                in_pp = True
            if x.type == "ERROR":
                in_err = True
            x = x.parent
        if in_pp:
            tags.append("preproc-branch")
        if row.get("kind") == "macro-shaped":
            tags.append("macro-callee")
        if in_err:
            tags.append("error-region")
    return tags


def c5_rows(lang: str, pilot: list[dict], rows_by_site: dict[str, dict]) -> list[dict]:
    """The per-site C5 pilot answers (truth/<lang>.c5-rows.jsonl): everything `pilot_coverage` reads, so the figures in
    truth/<lang>.c5.json re-derive from committed files alone (this file + truth/<lang>.c5-pilot.jsonl). `tags` are the
    site tags of a silent site outside the closed classes (the informational counts); they need the corpus to compute."""
    cache: dict = {}
    out = []
    for p in pilot:
        r = rows_by_site[p["siteId"]]
        silent = r["static"]["kind"] == "silent"
        out.append({"pilotOrdinal": p["pilotOrdinal"], "siteId": p["siteId"], "kind": r["static"]["kind"],
                    "relaxed": bool(r.get("relaxed")), "ambiguous": bool(r.get("ambiguous")),
                    "typeCtorMerged": bool(r.get("typeCtorMerged")), "byConstruction": list(r.get("byConstruction") or []),
                    "tags": [t for t in site_tags(lang, p, cache) if t not in CLOSED_CLASSES.get(lang, ())] if silent else []})
    return out


def c5_rows_by_site(rows: list[dict]) -> dict[str, dict]:
    """Rows as `pilot_coverage` reads them, rebuilt from truth/<lang>.c5-rows.jsonl."""
    return {r["siteId"]: {"static": {"kind": r["kind"]}, "relaxed": r["relaxed"], "ambiguous": r["ambiguous"],
                          "byConstruction": r["byConstruction"], "tags": r["tags"]} for r in rows}


def pilot_coverage(lang: str, pilot: list[dict], rows_by_site: dict[str, dict]) -> dict:
    """C5 (ML §3.2): share of the 50-site pilot with a reference occurrence under §5.2's mapping (`coverage`, raw; a
    `relaxed` macro-argument match counts), and the adjusted figure that leaves the sites tagged into a closed
    by-construction class out of the denominator (`byConstruction` on a silent row). Raw stays next to adjusted."""
    n = len(pilot)
    rows = [rows_by_site[p["siteId"]] for p in pilot]
    silent = [r for r in rows if r["static"]["kind"] == "silent"]
    covered = n - len(silent)
    relaxed = sum(1 for r in rows if r.get("relaxed"))
    by_class: dict[str, int] = {}
    excluded = 0
    for r in silent:
        cl = r.get("byConstruction") or []
        excluded += 1 if cl else 0
        for c in cl:
            by_class[c] = by_class.get(c, 0) + 1
    den = n - excluded
    res = {"pilotSites": n, "withReference": covered, "silent": len(silent),
           "coverage": round(covered / n, 4) if n else None,
           "relaxed": relaxed, "coverageStrict": round((covered - relaxed) / n, 4) if n else None,
           "ambiguous": sum(1 for r in rows if r.get("ambiguous")),
           "excluded": excluded, "excludedByClass": dict(sorted(by_class.items())),
           "exclusionCap": EXCLUSION_CAP, "exclusionCapExceeded": excluded > EXCLUSION_CAP,
           "adjustedDenominator": den, "adjustedCoverage": round(covered / den, 4) if den else None}
    cache: dict = {}
    info: dict[str, int] = {}
    for p, r in zip(pilot, rows):
        if r["static"]["kind"] == "silent":
            for t in (r["tags"] if "tags" in r else [t for t in site_tags(lang, p, cache) if t not in CLOSED_CLASSES.get(lang, ())]):
                info[t] = info.get(t, 0) + 1
    res["silentInformationalTags"] = dict(sorted(info.items()))
    res["note"] = ("ML §3.2 C5 tests `adjustedCoverage` >= 0.90 when `exclusionCapExceeded` is false; more than "
                   f"{EXCLUSION_CAP} excluded sites means the reference is blind to too much of the frame (a C5 failure). "
                   "`coverage` is the raw figure, always reported beside it")
    return res


# ------------------------------------------------------------------ C# generic-site audit draw (ML §3.2, prefix ccg-p0a-v1-gen|)

GEN_AUDIT_K = 40


def c5_rows_path(lang): return HERE / "truth" / f"{lang}.c5-rows.jsonl"


def gen_audit_path(lang): return HERE / "truth" / f"{lang}.gen-audit.jsonl"


def build_gen_audit(lang: str, sample: list[dict], priv: list[dict], ref: list[dict]) -> list[dict]:
    """The hash-sorted order (ML §4.4 rule, seed `ccg-p0a-v1-gen|<lang>|<sha>`) of the sampled sites in the blind C# stratum:
    reference-silent sites in a by-construction class (a type-argument callee, `nameof`; `byConstruction` on the ref row).
    It reads the reference but no label and no Joern output. S57 takes, in this order, every site the label calls internal
    plus the first GEN_AUDIT_K the label calls external (gen_audit_select)."""
    if lang != "csharp":
        raise SystemExit("the generic-site audit draw is defined for C# only (ML §3.2)")
    blind = {r["ordinal"] for r in ref if r["static"]["kind"] == "silent" and r.get("byConstruction")}
    items = []
    for pub, pr in zip(sample, priv):
        cl = csharp_generic_classes({"callText": pub["callText"], "calleeName": pr["calleeName"]})
        if cl and pub["ordinal"] in blind:
            items.append({"ordinal": pub["ordinal"], "siteId": pub["siteId"], "siteKey": pr["siteKey"], "classes": cl})
    order = hash_sort(items, seed_for("gen", lang, corpus_cfg(lang)["commit"]))
    return [{"rank": i, "ordinal": it["ordinal"], "siteId": it["siteId"], "hash": h, "classes": it["classes"]}
            for i, (h, it) in enumerate(order, 1)]


def gen_audit_select(order: list[dict], labels: list[dict], k: int = GEN_AUDIT_K) -> list[dict]:
    """The sites S57 checks from source: every class site labelled internal (`internal`/`declared`), and the first `k`, in
    hash order, labelled external (all callees external). `notACall` and empty-callee labels are in neither stratum."""
    by_ord = {l["ordinal"]: l for l in labels}
    out, n_ext = [], 0
    for o in order:
        l = by_ord.get(o["ordinal"])
        if l is None or l.get("notACall") or not l.get("callees"):
            continue
        kinds = {c["kind"] for c in l["callees"]}
        if kinds & {"internal", "declared"}:
            out.append({**o, "stratum": "internal"})
        elif n_ext < k:
            n_ext += 1
            out.append({**o, "stratum": "external"})
    return out


def cmd_genaudit(lang: str, select: bool) -> dict:
    sample, priv = read_jsonl(sample_path(lang)), read_jsonl(private_path(lang))
    order = build_gen_audit(lang, sample, priv, read_jsonl(HERE / "truth" / f"{lang}.ref.jsonl"))
    if not select:
        n = write_jsonl(gen_audit_path(lang), order)
        return {"lang": lang, "classSites": n, "path": str(gen_audit_path(lang).relative_to(HERE))}
    # S57 (run by data-scientist, with labels in hand): write adjudication/<lang>.gen-audit.jsonl
    labels = read_jsonl(HERE / "labels" / f"{lang}.jsonl")
    chosen = gen_audit_select(order, labels)
    n = write_jsonl(HERE / "adjudication" / f"{lang}.gen-audit.jsonl", chosen)
    return {"lang": lang, "selected": n, "internal": sum(1 for c in chosen if c["stratum"] == "internal"),
            "external": sum(1 for c in chosen if c["stratum"] == "external")}


# ------------------------------------------------------------------ brief (ML §6.5)

ML_NOTE = REPO / "docs" / "plans" / "connected-code-graph-p0-ml.md"
SPEC3 = REPO / "docs" / "plans" / "connected-code-graph-bundle-spec3.md"
SPEC_LANG_HEADING = {
    "python": "#### Python", "javascript": "#### TypeScript / JavaScript", "java": "#### Java", "kotlin": "#### Kotlin",
    "csharp": "#### C#/.NET", "go": "#### Go", "cpp": "#### C and C++",
}
LANG_LABEL = {"python": "Python", "javascript": "TypeScript/JavaScript", "java": "Java", "kotlin": "Kotlin",
              "csharp": "C#", "go": "Go", "cpp": "C/C++"}


def brief_template() -> str:
    """The text between the two rules of ML §6.5, verbatim."""
    t = ML_NOTE.read_text(encoding="utf-8")
    a = t.index("**You are labelling call sites in a {LANG} codebase, blind.**")
    b = t.index("**Appendix: `qname` grammar**", a)
    end = t.index("\n---\n", b)
    return t[a:end].rstrip() + "\n"


def spec_excerpt(lang: str) -> tuple[str, str]:
    """(version, text): bundle spec3 §4.1 and the language's §4.2 subsection, verbatim."""
    t = SPEC3.read_text(encoding="utf-8")
    m = re.search(r"\*\*Version:\*\* ([0-9.]+)", t.split("\n", 4)[2])
    ver = m.group(1) if m else "?"
    a = t.index("### 4.1 Common grammar")
    b = t.index("### 4.2 Per language", a)
    sec41 = t[a:b].rstrip()
    h = SPEC_LANG_HEADING[lang]
    c = t.index(h, b)
    nxt = [t.find(x, c + 5) for x in ("\n#### ", "\n### ")]
    d = min(x for x in nxt if x != -1)
    return ver, sec41 + "\n\n" + t[c:d].rstrip()


def fill_brief(lang: str) -> str:
    cfg = corpus_cfg(lang)
    root = src_dir(lang)
    ver, excerpt_txt = spec_excerpt(lang)
    gen = sorted({r for r, _ in iter_source_files(lang, root, cfg, include_generated=True)} - {r for r, _ in iter_source_files(lang, root, cfg)})
    gen_paths = list(cfg.get("generatedPaths") or [])
    gen_txt = ", ".join(f"`{g}`" for g in gen_paths + [x for x in gen if not any(x.startswith(g.rstrip('/') + '/') for g in gen_paths)][:12]) or "none in this corpus"
    if len(gen) > 12:
        gen_txt += f", and {len(gen) - 12} more files whose first five lines say 'generated' or 'DO NOT EDIT'"
    slots = {
        "{LANG}": LANG_LABEL[lang],
        "{SAMPLE_FILE}": str(sample_path(lang)),
        "{CORPUS_ROOT}": str(root),
        "{LABEL_FILE}": str(HERE / "labels" / f"{lang}.jsonl"),
        "{GENERATED_DIRS}": gen_txt,
        "{VENDORED_DIRS}": ", ".join(f"`{d}`" for d in vendored(cfg)),
        "{CAP}": str(CAP),
        "{SPEC_VERSION}": f"{ver} (`docs/plans/connected-code-graph-bundle-spec3.md`; `-spec4` amends nothing in §4)",
    }
    body = brief_template()
    if lang in CENSUS_LANGS:
        # ML §3.1 (v1.6): a census labels every site; the template's 150-internal / {CAP} stop rule does not apply
        stop = ("- **Stop** after the line that brings the number of lines with at least one `internal` or\n"
                "  `declared` callee to\n  **150**, or after ordinal **{CAP}**, whichever comes first.")
        if stop not in body:
            raise SystemExit("brief: the template's stop rule no longer matches; the census variant needs a revision")
        body = body.replace(stop, "- **Stop** only after the line for ordinal **{CAP}**, the last site of the sample file: every site is\n"
                                  "  labelled (a census of {CAP} sites). There is no earlier stop and no count of internal sites to reach.")
        slots["{CAP}"] = str(len(read_jsonl(sample_path(lang))))
    for k, v in slots.items():
        body = body.replace(k, v)
    body = body.rstrip() + "\n\n" + excerpt_txt + "\n"
    return body


def validate_brief(text: str, lang: str = "") -> list[str]:
    errs = []
    left = re.findall(r"\{[A-Z_]{2,}\}", text)
    if left:
        errs.append(f"unfilled slots: {sorted(set(left))}")
    for must in ("**Output.**", "**Writing a target.**", "**Appendix: `qname` grammar**", "### 4.1 Common grammar"):
        if must not in text:
            errs.append(f"missing {must}")
    if lang in CENSUS_LANGS and ("**150**" in text or "whichever comes first" in text):
        errs.append("census brief carries the stop rule")
    return errs


def cmd_brief(lang: str) -> dict:
    txt = fill_brief(lang)
    errs = validate_brief(txt, lang)
    if errs:
        raise SystemExit(f"brief {lang}: {errs}")
    brief_path(lang).write_text(txt, encoding="utf-8")
    return {"lang": lang, "bytes": len(txt.encode())}


# ------------------------------------------------------------------ flow brief + label check (S56; ML §6.4, §11 N3)
# The template is flow-brief.template.md (between its BRIEF markers). Same shape as `brief`: deterministic,
# no network, no Joern, reads only the template, the bundle spec, corpora.yaml-derived paths and
# sample/<lang>-flow.jsonl. `flowcheck` opens only sample/<lang>-flow.jsonl and labels/<lang>-flow.jsonl.

FLOW_TEMPLATE = HERE / "flow-brief.template.md"
FLOW_BEGIN, FLOW_END = "<!-- BRIEF-BEGIN -->", "<!-- BRIEF-END -->"


def flow_brief_path(lang, fs=None): return HERE / "sample" / f"{lang}-flow{FLOW_SETS[fs or FLOW_SET]}.brief.md"
def flow_label_path(lang, fs=None): return HERE / "labels" / f"{lang}-flow{FLOW_SETS[fs or FLOW_SET]}.jsonl"


# Short-circuit operators per language: bundle spec §5.5 "short-circuit operands that guard a point".
# Kotlin's `?:` is the null-coalescing operator (JS/C# `??`); Kotlin's infix `and`/`or` do not short-circuit.
FLOW_OPS = {
    "python": "`and`, `or`",
    "javascript": "`&&`, `||` and `??`",
    "java": "`&&`, `||`",
    "kotlin": "`&&`, `||` and the elvis operator `?:` (Kotlin's null-coalescing operator; the infix functions `and`/`or` do not short-circuit and do not count)",
    "csharp": "`&&`, `||` and `??`",
    "go": "`&&`, `||`",
    "cpp": "`&&`, `||`",
}
# Invented snippets, not from any corpus (the labeller must not meet its own methods in the brief).
FLOW_OP_EXAMPLES = {
    "python": "`if token and verify(token): …` has the condition `token` guarding the call `verify`; `v = cache or load(key)` has the condition `cache` guarding `load`.",
    "javascript": "`props.onClick && props.onClick(e)` has the condition `props.onClick` guarding the call `onClick`; `const v = cached ?? load(key)` has the condition `cached` guarding `load`; `ready || init()` has the condition `ready` guarding `init`.",
    "java": "`if (user != null && audit.log(user)) { … }` has the condition `user != null` guarding the call `log`; `ok = hit || fetch(id)` has the condition `hit` guarding `fetch`.",
    "kotlin": "`val v = cache ?: load(key)` has the condition `cache` guarding `load`; `if (a != null && check(a)) { … }` has the condition `a != null` guarding `check`.",
    "csharp": "`var v = cache ?? Load(key);` has the condition `cache` guarding `Load`; `ok = user != null && Audit(user);` has the condition `user != null` guarding `Audit`.",
    "go": "`if ok && handle(x) { … }` has the condition `ok` guarding the call `handle`; `done = seen || mark(k)` has the condition `seen` guarding `mark`.",
    "cpp": "`if (p && p->run(x)) { … }` has the condition `p` guarding the call `run`; `ok = hit || fetch(id);` has the condition `hit` guarding `fetch`.",
}
FLOW_COND_FORMS = {
    "python": "`if`/`elif` conditions, `while` and `for` headers, conditional expressions (`x if c else y`), and `match` `case` labels,",
    "javascript": "`if` conditions, `while`, `do … while` and `for`/`for…of`/`for…in` headers, ternaries (`c ? a : b`), and `switch` `case` labels,",
    "java": "`if` conditions, `while`, `do … while` and `for`/enhanced-`for` headers, ternaries, and `switch` `case` labels (with their `when` guard, if any),",
    "kotlin": "`if` conditions (statement or expression), `while`, `do … while` and `for` headers, and `when` branch conditions,",
    "csharp": "`if` conditions, `while`, `do … while`, `for` and `foreach` headers, ternaries, `switch` `case` labels (with their `when` guard) and `catch … when (…)` filters,",
    "go": "`if` conditions, `for` conditions and range clauses, and `switch` `case` labels,",
    "cpp": "`if` conditions, `while`, `do … while` and `for`/range-`for` headers, ternaries, and `switch` `case` labels,",
}
FLOW_SCOPE_NOTES = {
    "python": "Nested `def`, `class` and `lambda` bodies are separate symbols (bundle spec §4.1 rule 5): their calls, returns and conditions are not this method's, and values captured by them are not followed. The call that creates or receives the nested function is this method's.",
    "javascript": "Nested functions, arrow functions, callbacks and classes are separate symbols (bundle spec §4.1 rule 5): their calls, returns and conditions are not this method's, and values captured by them are not followed. The call that receives the callback (`xs.map(x => …)`) is this method's.",
    "java": "Lambdas, local and anonymous classes are separate symbols (bundle spec §4.1 rule 5): their calls, returns and conditions are not this method's, and values captured by them are not followed. The call that receives the lambda is this method's.",
    "kotlin": "Lambdas (trailing lambdas included), local functions and `object :` expressions are separate symbols (bundle spec §4.1 rule 5): their calls, returns and conditions are not this method's, and values captured by them are not followed. The call that receives the lambda is this method's.",
    "csharp": "Lambdas, anonymous methods and local functions are separate symbols (bundle spec §4.1 rule 5): their calls, returns and conditions are not this method's, and values captured by them are not followed. The call that receives the lambda is this method's.",
    "go": "Function literals are separate symbols (bundle spec §4.1 rule 5): their calls, returns and conditions are not this method's, and values captured by them are not followed. The call that receives the function literal, or `go`/`defer` of it, is this method's.",
    "cpp": "Lambdas and local classes are separate symbols (bundle spec §4.1 rule 5): their calls, returns and conditions are not this method's, and values captured by them are not followed. The call that receives the lambda is this method's.",
}
FLOW_RECEIVER_NOTE = {
    "python": "The first parameter of a function that sits directly in a class body (`self`, `cls`), unless it is decorated `@staticmethod`, is the receiver: `param` index -1, and the next parameter is index 0. A read of `self.x` is a read of `param` -1.",
    "javascript": "`this` is implicit and is not a parameter (bundle spec §5.4): a value read from `this` or `this.x` has no source point, so it makes no pair.",
    "java": "`this` is implicit and is not a parameter (bundle spec §5.4): a value read from `this` or `this.x` has no source point, so it makes no pair.",
    "kotlin": "`this` is implicit and is not a parameter (bundle spec §5.4), except the receiver of an *extension function*, which is `param` index -1 (named `this`); the first ordinary parameter is then index 0. A value read from the implicit `this` has no source point.",
    "csharp": "`this` is implicit and is not a parameter (bundle spec §5.4), except the `this` parameter of an *extension method*, which is `param` index -1; the next parameter is index 0. A value read from the implicit `this` has no source point.",
    "go": "A method's receiver (`b` in `func (b *T) …`) is `param` index -1, and the first ordinary parameter is index 0.",
    "cpp": "Bundle spec §5.4 names no receiver parameter for C/C++. Treat the implicit `this` as the JVM's: it is not a parameter, and a value read from it has no source point, so it makes no pair.",
}


def flow_brief_template() -> str:
    t = FLOW_TEMPLATE.read_text(encoding="utf-8")
    a = t.index(FLOW_BEGIN) + len(FLOW_BEGIN)
    b = t.index(FLOW_END, a)
    return t[a:b].strip("\n") + "\n"


def _spec_between(t: str, start: str, end: str) -> str:
    a = t.index(start)
    return t[a:t.index(end, a)].rstrip()


def flow_spec_excerpt() -> str:
    """Bundle spec3 §5.4, §5.5, §5.6 (the call-site naming rows and the 'which calls' bullets) and §5.7,
    verbatim except two redactions: the sentences that name the comparison tool. A redaction that no longer
    matches the spec text fails loudly, so spec drift cannot slip a tool name into the brief."""
    t = SPEC3.read_text(encoding="utf-8")
    s54 = _spec_between(t, "### 5.4 `param`", "### 5.5 ")
    s55 = _spec_between(t, "### 5.5 `ret` and `cond`", "### 5.6 ")
    s55n = re.sub(r"\s*\(Joern's CDG sources decide what counts\..*?misses\.\)", "", s55, flags=re.S)
    if s55n == s55:
        raise SystemExit("flow brief: spec §5.5 redaction no longer matches the spec text")
    s56 = _spec_between(t, "### 5.6 `callsite`", "### 5.7 ")
    head, rows, bullets, cur = [], [], [], None
    for ln in s56.split("\n"):
        if ln.startswith("- "):
            cur = [ln]
            bullets.append(cur)
        elif cur is not None and ln.startswith("  "):
            cur.append(ln)
        elif ln.startswith("|"):
            cur = None
            if ln.startswith(("| Field", "|---")) or any(ln.startswith(f"| `{k}`") for k in ("calleeName", "ordinal", "callKind", "line")):
                rows.append(ln)
        elif ln.strip() and not rows:
            head.append(ln)
    keep = ["\n".join(b) for b in bullets if b[0].startswith(("- **Which calls are call sites:**", "- **Decorator and annotation"))]
    if len(keep) != 2:
        raise SystemExit("flow brief: spec §5.6 bullets no longer match")
    keep = [k.replace("(Joern's `<operator>.*` calls)", "(language operators)") for k in keep]
    if any("Joern" in k for k in keep):
        raise SystemExit("flow brief: spec §5.6 redaction no longer matches the spec text")
    s56n = "\n\n".join(head) + "\n\n" + "\n".join(rows) + "\n\n" + "\n".join(keep)
    s57 = _spec_between(t, "### 5.7 The call-site ordinal", "### 5.8 ")
    return "\n\n".join([s54, s55n, s56n, s57]) + "\n"


def spec_version() -> str:
    t = SPEC3.read_text(encoding="utf-8")
    m = re.search(r"\*\*Version:\*\* ([0-9.]+)", t.split("\n", 4)[2])
    return m.group(1) if m else "?"


def fill_flow_brief(lang: str) -> str:
    if not flow_path(lang).exists():
        raise SystemExit(f"flow brief {lang}: {flow_path(lang)} missing (run flowdraw first)")
    n = len(read_jsonl(flow_path(lang)))
    slots = {
        "{LANG}": LANG_LABEL[lang],
        "{N_METHODS}": str(n),
        "{SAMPLE_FILE}": str(flow_path(lang)),
        "{CORPUS_ROOT}": str(src_dir(lang)),
        "{LABEL_FILE}": str(flow_label_path(lang)),
        "{SHORT_CIRCUIT_OPS}": FLOW_OPS[lang],
        "{SHORT_CIRCUIT_EXAMPLES}": FLOW_OP_EXAMPLES[lang],
        "{COND_FORMS}": FLOW_COND_FORMS[lang],
        "{SCOPE_NOTES}": FLOW_SCOPE_NOTES[lang],
        "{RECEIVER_NOTE}": FLOW_RECEIVER_NOTE[lang],
        "{SPEC_VERSION}": f"{spec_version()} (`docs/plans/connected-code-graph-bundle-spec3.md`; `-spec4` §2 changes nothing for a named method)",
    }
    body = flow_brief_template()
    for k, v in slots.items():
        body = body.replace(k, v)
    return body.rstrip() + "\n\n" + flow_spec_excerpt()


FLOW_LEAKS = [r"results/", r"truth/", r"ref\.jsonl", r"adjudication", r"extract\.sc", r"\bprojection\b", r"CONTROLLED_BY",
              r"\bCDG\b", r"\bDATA\b", r"\bCALLS_NAME\b", r"\bFULL_NAME\b", r"cpg\.bin", r"\b0\.95\b", r"\brecall\b",
              r"\bthreshold\b", r"flowAvailable", r"labels/[a-z]+\.jsonl", r"\.brief\.md", r"S26|S28|S56|S57"]


def validate_flow_brief(lang: str, text: str) -> list[str]:
    errs = []
    left = re.findall(r"\{[A-Z_]{2,}\}", text)
    if left:
        errs.append(f"unfilled slots: {sorted(set(left))}")
    for must in ("**Output.**", "**What counts as a condition (`cond`).**", "short-circuit operands that guard a point",
                 "**What to label: pairs.**", "**Appendix: bundle spec points**", "### 5.4 `param`", "### 5.5 `ret` and `cond`",
                 "### 5.7 The call-site ordinal", FLOW_OPS[lang], FLOW_OP_EXAMPLES[lang]):
        if must not in text:
            errs.append(f"missing {must[:60]}")
    if lang == "javascript":  # the TS/JS instruction is mandatory and explicit
        for must in ("`&&`, `||` and `??`", "Every operand of such an operator", "`props.onClick && props.onClick(e)`"):
            if must not in text:
                errs.append(f"javascript: missing {must}")
    for pat in FLOW_LEAKS:
        if re.search(pat, text):
            errs.append(f"leak pattern {pat!r}")
    # `joern` is allowed once, in the prohibition sentence; nothing else about tools of the comparison
    if len(re.findall(r"joern", text, flags=re.I)) != 1:
        errs.append("expected exactly one mention of Joern (the prohibition)")
    return errs


def cmd_flowbrief(lang: str) -> dict:
    txt = fill_flow_brief(lang)
    errs = validate_flow_brief(lang, txt)
    if errs:
        raise SystemExit(f"flowbrief {lang}: {errs}")
    flow_brief_path(lang).write_text(txt, encoding="utf-8")
    return {"lang": lang, "bytes": len(txt.encode())}


FLOW_ENDPOINTS = {"param": {"index", "name"}, "call": {"calleeName", "ordinal", "line"}, "ret": {"ordinal", "line"},
                  "cond": {"line", "code"}}
FLOW_SOURCES = {"data": {"param", "call"}, "control": {"cond"}}
FLOW_SINKS = {"data": {"call", "ret", "cond"}, "control": {"call", "ret", "cond"}}


RETURN_TYPES = {"return_statement", "return_expression"}
_FACT_CACHE: dict = {}


def flow_src_dir(lang: str) -> Path:
    return src_dir(lang)


def method_facts(lang: str, m: dict) -> Optional[dict]:
    """Structure of a flow-sample method from the corpus tree-sitter parse (harness side, not the labeller's):
    `params` = the text of each declared parameter (receiver excluded; a C# `this` parameter is shifted out),
    `reliable` = False when the method's parse holds an error node (a macro in the signature, or a macro-named "function"): then the index/ordinal
    checks are skipped. `returns` = the method's own `return`s (nested functions and lambdas excluded). None when the corpus file or
    the method cannot be found."""
    f = flow_src_dir(lang) / m["path"]
    key = (str(f), m["qname"], m["lineStart"])
    if key in _FACT_CACHE:
        return _FACT_CACHE[key]
    facts = None
    if f.is_file():
        data = f.read_bytes()
        k, tree = parse_file(lang, m["path"], data)
        for d in enumerate_defs(lang, m["path"], data, k, tree.root_node):
            if d.qname == m["qname"] and d.lineStart == m["lineStart"]:
                params = [t for t in (_txt(c) for c in _params_of(lang, d.node)) if t not in ("*", "/")]     # python separators are not parameters
                if lang == "csharp" and params and re.match(r"this\s", params[0]):
                    params = params[1:]
                nested = FN_TYPES[lang] | ANON_TYPES[lang]
                rets = 0
                stack = list(d.node.children)
                while stack:
                    x = stack.pop()
                    if x.type in nested:
                        continue
                    rets += x.type in RETURN_TYPES
                    stack.extend(x.children)
                nn = _name_node(lang, d.node)
                macro_named = lang == "cpp" and nn is not None and re.fullmatch(r"[A-Z][A-Z0-9_]{2,}", _txt(nn)) is not None     # `MACRO(args) {` parsed as a function
                facts = {"params": params, "returns": rets, "reliable": not d.node.has_error and not macro_named}
                break
    _FACT_CACHE[key] = facts
    return facts


def _flow_endpoint_errs(e, role: str, kind: str, m: dict, text: str, ctext: str, facts: Optional[dict] = None) -> list[str]:
    if not isinstance(e, dict) or e.get("point") not in FLOW_ENDPOINTS:
        return [f"{role}: bad point"]
    p = e["point"]
    if set(e) != {"point"} | FLOW_ENDPOINTS[p]:
        return [f"{role}: keys {sorted(e)} != {sorted({'point'} | FLOW_ENDPOINTS[p])}"]
    errs = []
    allowed = (FLOW_SOURCES if role == "source" else FLOW_SINKS)[kind]
    if p not in allowed:
        errs.append(f"{role}: a {p} cannot be a {kind} {role}")
    ints = [k for k in ("index", "ordinal", "line") if k in e]
    if any(type(e[k]) is not int for k in ints):
        return errs + [f"{role}: non-integer field"]
    if "line" in e and not (m["lineStart"] <= e["line"] <= m["lineEnd"]):
        errs.append(f"{role}: line {e['line']} outside {m['lineStart']}-{m['lineEnd']}")
    if p == "param":
        nm = e["name"]
        if e["index"] < -1 or not isinstance(nm, str) or not nm:
            errs.append(f"{role}: bad param index/name")
        else:
            bare = nm.lstrip("*&.")
            word = r"(?<![\w$])" + re.escape(bare) + r"(?![\w$])"
            if not bare or not (re.search(word, text) or (e["index"] == -1 and bare == "this")):
                errs.append(f"{role}: param name {nm!r} is not a word of the method")
            elif facts is not None and facts["reliable"] and e["index"] >= 0:
                if e["index"] >= len(facts["params"]):
                    errs.append(f"{role}: param index {e['index']} beyond the {len(facts['params'])} declared parameter(s)")
                elif not re.search(word, facts["params"][e["index"]]):
                    errs.append(f"{role}: param {e['index']} is not named {nm!r}")
    elif p == "call":
        nm = e["calleeName"]
        if not isinstance(nm, str) or not nm or e["ordinal"] < 0:
            errs.append(f"{role}: bad call")
        elif nm != "<dynamic>" and e["ordinal"] >= len(re.findall(r"(?<![\w$])" + re.escape(nm) + r"(?![\w$])", text)):
            errs.append(f"{role}: call {nm}#{e['ordinal']} has fewer occurrences in the method")
    elif p == "ret":
        if e["ordinal"] < 0:
            errs.append(f"{role}: bad ret ordinal")
        elif facts is not None and facts["reliable"] and e["ordinal"] >= max(facts["returns"], 1):
            errs.append(f"{role}: ret ordinal {e['ordinal']} beyond the method's {facts['returns']} return(s)")
    else:
        c = e["code"]
        if not isinstance(c, str) or not c.strip() or len(c) > 200 or ws(c) not in ctext:
            errs.append(f"{role}: cond code is not (<= 200 chars of) the method's text")
    return errs


def cmd_flowcheck(lang: str) -> dict:
    """Schema-validate labels/<lang>-flow.jsonl against sample/<lang>-flow.jsonl (opens only those two files).
    Checks ordinal order and resumability, endpoint shapes, kind/endpoint compatibility, lines in span, evidence
    length (a citation has a digit or >= 3 word characters; one with no digit is only counted, `evidenceNoLine`), source != sink, duplicate pairs; counts pairs. Advisory counts (ML §8.4's 120-pair floor) are not errors.
    Param names/indices and `ret` ordinals are checked against the corpus parse (`method_facts`, the fetched tree);
    without it those checks are skipped and `structural` < `labelled`, which keeps `complete` false. `complete` also
    needs every method to carry a pair (S56's done-condition: >= 1 pair per method); `withoutPair` stays visible."""
    sample = read_jsonl(flow_path(lang))
    rows = []
    errs: list[str] = []
    if flow_label_path(lang).exists():
        for i, ln in enumerate(open(flow_label_path(lang), encoding="utf-8"), 1):
            if not ln.strip():
                continue
            try:
                rows.append((i, json.loads(ln)))
            except json.JSONDecodeError as ex:
                errs.append(f"line {i}: not JSON ({ex.msg})")
                break
    stats = {"data": 0, "control": 0, "unsure": 0, "withPair": 0, "withoutPair": 0, "evidenceNoLine": 0}
    structural = 0
    for k, (i, o) in enumerate(rows):
        pre = f"line {i}"
        if k >= len(sample):
            errs.append(f"{pre}: more lines than methods ({len(sample)})")
            break
        m = sample[k]
        if not isinstance(o, dict) or not {"ordinal", "path", "qname", "pairs"} <= set(o) or set(o) - {"ordinal", "path", "qname", "pairs", "unlisted", "noPairReason"}:
            errs.append(f"{pre}: keys {sorted(o) if isinstance(o, dict) else type(o).__name__}")
            continue
        if (o["ordinal"], o["path"], o["qname"]) != (m["ordinal"], m["path"], m["qname"]):
            errs.append(f"{pre}: expected ordinal {m['ordinal']} {m['qname']}")
            continue
        text, ctext = m["source"], ws(m["source"])
        facts = method_facts(lang, m)
        structural += facts is not None
        seen = set()
        if not isinstance(o["pairs"], list):
            errs.append(f"{pre}: pairs not a list")
            continue
        for j, p in enumerate(o["pairs"]):
            pp = f"{pre} pair {j}"
            if not isinstance(p, dict) or set(p) != {"kind", "source", "sink", "evidence", "confidence"}:
                errs.append(f"{pp}: keys")
                continue
            if p["kind"] not in FLOW_SOURCES or p["confidence"] not in ("sure", "unsure"):
                errs.append(f"{pp}: kind/confidence")
                continue
            if not isinstance(p["evidence"], str) or not p["evidence"].strip() or len(p["evidence"]) > 300:
                errs.append(f"{pp}: evidence missing or > 300 chars")
            elif not re.search(r"\d", p["evidence"]):
                if len(re.findall(r"\w", p["evidence"])) < 3:
                    errs.append(f"{pp}: evidence is not a citation (no digit and fewer than 3 word characters)")
                else:
                    stats["evidenceNoLine"] += 1
            ee = (_flow_endpoint_errs(p["source"], "source", p["kind"], m, text, ctext, facts)
                  + _flow_endpoint_errs(p["sink"], "sink", p["kind"], m, text, ctext, facts))
            errs += [f"{pp}: {e}" for e in ee]
            key = (p["kind"], jdump(p["source"]), jdump(p["sink"]))
            if p["source"] == p["sink"]:
                errs.append(f"{pp}: source equals sink")
            if key in seen:
                errs.append(f"{pp}: duplicate pair")
            seen.add(key)
            stats[p["kind"]] += 1
            stats["unsure"] += p["confidence"] == "unsure"
        if o["pairs"]:
            stats["withPair"] += 1
        else:
            stats["withoutPair"] += 1
            if not (isinstance(o.get("noPairReason"), str) and o["noPairReason"].strip() and len(o["noPairReason"]) <= 200):
                errs.append(f"{pre}: no pairs and no noPairReason")
        for u in o.get("unlisted", []):
            if not (isinstance(u, dict) and set(u) == {"line", "construct"} and type(u["line"]) is int and isinstance(u["construct"], str) and 0 < len(u["construct"]) <= 120):
                errs.append(f"{pre}: bad unlisted entry")
    pairs = stats["data"] + stats["control"]
    return {"lang": lang, "methods": len(sample), "labelled": len(rows), "structural": structural,
            "complete": len(rows) == len(sample) and not errs and stats["withoutPair"] == 0 and structural == len(rows),
            "pairs": pairs, **stats, "belowFloor120": pairs < 120, "errors": errs[:60], "nErrors": len(errs)}


# ------------------------------------------------------------------ firewall + verify

FIREWALL_PATTERNS = [r"joern", r"cpg\.bin", r"\.cpg\b", r"\bFULL_NAME\b", r"results/", r"truth/", r"ref\.jsonl",
                     r"adjudication", r"scip", r"callgraph"]
FIREWALL_ALLOW_IN_BRIEF = [r"joern"]  # the brief and the spec excerpt tell the labeller NOT to use these tools / name them


def firewall_scan() -> list[str]:
    """Nothing in sample/ (except the brief's own prohibition text) or frame/ may carry Joern output or
    reference truth. Mechanical part: forbidden keys, and fixed strings in the data files."""
    bad = []
    for d in ("frame", "sample"):
        for p in sorted((HERE / d).glob("*.jsonl")):
            for i, ln in enumerate(open(p, encoding="utf-8"), 1):
                o = json.loads(ln)
                ks = set(o)
                if ks & {"static", "impls", "tool", "toolVersion", "targets", "resolved", "res", "nm", "joern", "truth"}:
                    bad.append(f"{p.name}:{i}: forbidden key {sorted(ks & {'static', 'impls', 'tool', 'toolVersion', 'targets', 'resolved', 'res', 'nm', 'joern', 'truth'})}")
                    break
    return bad


_ROOT_PREFIX = re.compile(r"/[^\s`'\"]*?(?:/cpg)?(?=/(?:spikes/frontend-fitness|\.ccg-work)/)")


def same_modulo_root(committed: str, generated: str) -> bool:
    """A brief embeds absolute paths (sample file, corpus root, label file). Equal when the texts are equal after the checkout
    root in front of `[/cpg]/spikes/frontend-fitness/` and `[/cpg]/.ccg-work/` is replaced by a placeholder on both sides, so a
    relocated checkout (or a copy of the spike directory) verifies; only the root moves, any other difference in a path or in the text still differs."""
    return _ROOT_PREFIX.sub("<ROOT>", committed) == _ROOT_PREFIX.sub("<ROOT>", generated)


def cmd_verify(langs: list[str]) -> dict:
    """Re-derive frame, sample, private, flow, pilot and brief from corpora.yaml + the fetched tree and compare."""
    problems = []
    cfgs = load_corpora()
    for lang in langs:
        cfg = cfgs["corpora"][lang]
        dg = tree_digest(src_dir(lang))
        if cfg.get("srcDigest") and cfg["srcDigest"] != dg:
            problems.append(f"{lang}: srcDigest {dg} != corpora.yaml {cfg['srcDigest']}")
            continue
        rows = build_frame(lang)
        if rows != read_jsonl(frame_path(lang)):
            problems.append(f"{lang}: frame differs")
        pub, priv = build_sample(lang, rows)
        if pub != read_jsonl(sample_path(lang)):
            problems.append(f"{lang}: sample differs")
        if priv != read_jsonl(private_path(lang)):
            problems.append(f"{lang}: private differs")
        if flow_path(lang).exists():
            f2 = []
            pop = flow_population(lang, src_dir(lang), cfg)
            for i, (h, m) in enumerate(hash_sort(pop, seed_for("flow", lang, cfg["commit"]))[:30], 1):
                f2.append({"ordinal": i, "path": m["path"], "qname": m["qname"], "lineStart": m["lineStart"], "lineEnd": m["lineEnd"], "source": m["source"]})
            if f2 != read_jsonl(flow_path(lang)):
                problems.append(f"{lang}: flow sample differs")
            if lang in CENSUS_LANGS and (not flow_tier2_path(lang).exists() or build_flow_tier2(lang) != read_jsonl(flow_tier2_path(lang))):
                problems.append(f"{lang}: flow tier-2 sample differs")
        if brief_path(lang).exists() and not same_modulo_root(brief_path(lang).read_text(encoding="utf-8"), fill_brief(lang)):
            problems.append(f"{lang}: brief differs")
        if validate_brief(brief_path(lang).read_text(encoding="utf-8") if brief_path(lang).exists() else "", lang):
            problems.append(f"{lang}: brief invalid")
        if flow_brief_path(lang).exists():
            ftxt = flow_brief_path(lang).read_text(encoding="utf-8")
            if not same_modulo_root(ftxt, fill_flow_brief(lang)):
                problems.append(f"{lang}: flow brief differs")
            if validate_flow_brief(lang, ftxt):
                problems.append(f"{lang}: flow brief invalid")
    problems += firewall_scan()
    return {"checked": langs, "problems": problems}


# ------------------------------------------------------------------ refmap command (C5 and ref.jsonl)

def cmd_refmap(lang: str, tool_out: Path, tool_version: str, mode: str, module_path: str = "", repo_prefix: str = "",
               exit_status: int = 0, docs_indexed: Optional[int] = None, rows_only: bool = False) -> dict:
    """mode 'c5' maps the 50-site pilot and writes truth/<lang>.c5.json and truth/<lang>.c5-rows.jsonl (the per-site
    answers); with `rows_only` it writes only the rows file, keeps the c5.json (and its recordedAt, which gates the
    Joern run) and reports whether the recomputed `pilot` block equals the committed one. Mode 'ref' maps every
    sampled ordinal and writes truth/<lang>.ref.jsonl."""
    frame = read_jsonl(frame_path(lang))
    by_key = {r["siteKey"]: r for r in frame}
    if mode == "c5":
        pil = read_jsonl(HERE / "truth" / f"{lang}.c5-pilot.jsonl")
        sample, priv = [{"ordinal": p["pilotOrdinal"], "siteId": p["siteId"]} for p in pil], [{"siteKey": p["siteKey"]} for p in pil]
    else:
        sample, priv = read_jsonl(sample_path(lang)), read_jsonl(private_path(lang))
    if lang == "go":
        rows = refmap_go(tool_out.read_text(encoding="utf-8", errors="replace"), module_path, sample, priv, tool_version, by_key, repo_prefix)
        ndocs = docs_indexed
    else:
        j = json.loads(tool_out.read_text(encoding="utf-8"))
        rows = refmap_scip(lang, j, sample, priv, tool_version, by_key)
        ndocs = len(j.get("documents", []))
        tool_version = tool_version or j.get("metadata", {}).get("tool_info", {}).get("version", "")
    if mode == "c5":
        by_site = {r["siteId"]: r for r in rows}
        crow = c5_rows(lang, pil, by_site)
        cov = pilot_coverage(lang, pil, c5_rows_by_site(crow))
        write_jsonl(c5_rows_path(lang), crow)
        if rows_only:
            old = json.loads((HERE / "truth" / f"{lang}.c5.json").read_text())["pilot"]
            return {"lang": lang, "rows": len(crow), "pilotEqualsCommitted": cov == old,
                    "differs": sorted(k for k in set(cov) | set(old) if cov.get(k) != old.get(k))}
        rec = {"lang": lang, "tool": rows[0]["tool"] if rows else None, "toolVersion": tool_version, "exitStatus": exit_status,
               "documentsIndexed": ndocs, "seedPrefix": "ccg-p0a-v1-c5|", "pilot": cov,
               "pilotKinds": {k: sum(1 for r in rows if r["static"]["kind"] == k) for k in sorted({r["static"]["kind"] for r in rows})},
               "recordedAtUtc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               "recordedAtEpoch": int(__import__("time").time())}
        (HERE / "truth").mkdir(exist_ok=True)
        (HERE / "truth" / f"{lang}.c5.json").write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
        return rec
    n = write_jsonl(HERE / "truth" / f"{lang}.ref.jsonl", rows)
    kinds = {k: sum(1 for r in rows if r["static"]["kind"] == k) for k in sorted({r["static"]["kind"] for r in rows})}
    return {"lang": lang, "refLines": n, "kinds": kinds, "withImpls": sum(1 for r in rows if r["impls"]), "documents": ndocs}


# ------------------------------------------------------------------ joern (P0-18b): stage inputs and PRINT the commands

JOERN_LANG = {"python": "pythonsrc", "javascript": "jssrc", "java": "javasrc", "kotlin": "kotlin", "csharp": "csharpsrc",
              "go": "golang", "cpp": "c"}
# fetch-on flags (p0-2 §4.6 S25 item 6); `--jar4import-url` is never passed.
FETCH_ON_FLAG = {"java": "--fetch-dependencies", "go": "--fetch-dependencies", "kotlin": "--download-dependencies",
                 "csharp": "--download-dependencies"}


def cmd_joern(lang: str) -> dict:
    """Stage frame/<lang>.jsonl and sample/<lang>.private.jsonl (no label, no reference truth) for measure.sc and
    print the P0-18b command lines. Runs nothing. A Joern run is valid only if its start time is later than
    truth/<lang>.c5.json's recordedAtEpoch (checked here: the stage refuses otherwise only when the file is absent)."""
    c5 = HERE / "truth" / f"{lang}.c5.json"
    if not c5.exists():
        raise SystemExit(f"{lang}: no truth/{lang}.c5.json -- C5 must be recorded before any Joern run on this corpus")
    cfg = corpus_cfg(lang)
    stage = WORK / "frontend-fitness" / "joern" / lang / "stage"
    stage.mkdir(parents=True, exist_ok=True)
    write_jsonl(stage / "frame.jsonl", read_jsonl(frame_path(lang)))
    write_jsonl(stage / "sample.private.jsonl", read_jsonl(private_path(lang)))
    base = WORK / "frontend-fitness" / "joern" / lang
    src = src_dir(lang)
    fe = JOERN_LANG[lang]
    runner = "cpg/sandbox/offline-run.sh --backend container --mode offline"
    vend = ",".join(vendored(cfg))
    cmds = {
        "off.parse": f"{runner} --frontend joern-parse --source {src} --output {base}/off --codebase {lang} -- /src --language {fe}",
        "off.content-check": f"{runner} --frontend joern --script-file cpg/sandbox/scripts/cpg-content-check.sc --source {src} --output {base}/off -- --script /scripts/cpg-content-check.sc --param cpgFiles=/out/<parse auditId>.cpg.bin",
        "off.measure": f"{runner} --frontend joern --script-file cpg/spikes/frontend-fitness/measure.sc --source {stage} --output {base}/off -- --script /scripts/measure.sc --param cpgFile=/out/<parse auditId>.cpg.bin --param frame=/src/frame.jsonl --param sample=/src/sample.private.jsonl --param lang={lang} --param vendored={vend} --param out=/out/@AUDIT_ID@.measure.jsonl",
    }
    flag = FETCH_ON_FLAG.get(lang)
    tag = "$(cpg/infra/toolchain/build.sh --print-tag)"
    cmds["on.parse"] = (f"docker run --rm --hostname localhost --memory 4g --user 1000:1000 -v {src}:/src:ro -v {base}/on:/out {tag} "
                        f"joern-parse /src --language {fe} -o /out/on.cpg.bin" + (f" --frontend-args {flag}" if flag else "")
                        + "   # network ON in the toolchain image; never --jar4import-url")
    # the CPG file is local, so measuring the fetch-on CPG needs no network: the offline runner takes it from <base>/on
    cmds["on.measure"] = cmds["off.measure"].replace(f"{base}/off", f"{base}/on").replace("<parse auditId>.cpg.bin", "on.cpg.bin")
    return {"lang": lang, "stage": str(stage), "c5RecordedAtUtc": json.loads(c5.read_text())["recordedAtUtc"], "commands": cmds}


# ------------------------------------------------------------------ CLI

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("genaudit")
    s.add_argument("lang", choices=["csharp"])
    s.add_argument("--select", action="store_true", help="S57 only (needs labels/): write the sites to check from source")
    s = sub.add_parser("flowdraw-p1", help="P1 flow draw (prereg §3): applies the custodian exclusion, prints two counts only")
    s.add_argument("lang", choices=["python"])
    s.add_argument("--exclude", required=True, type=Path, help="custodian-private JSON lines {path, defLine}; read into memory only")
    s.add_argument("--counts", required=True, type=Path, help="output: exactly excludedTier1Frame and excludedTier2Frame")
    s.add_argument("--expect-counts", type=lambda t: tuple(int(x) for x in t.split(",")), help="population,minus-P0,tier-2 counts of a NON-registered corpus (tests); default 368,338,163 on the registered corpus only")
    s.add_argument("--tier2-out", required=True, type=Path, help="where the 15-method tier-2 file goes; must be outside the repository (custodian-held)")
    for c in ("fetch", "census", "frame", "draw", "flowdraw", "c5pilot", "brief", "flowbrief", "flowcheck", "joern"):
        s = sub.add_parser(c)
        s.add_argument("lang", choices=LANGS)
        if c == "fetch":
            s.add_argument("--force", action="store_true")
        if c in ("flowbrief", "flowcheck"):
            s.add_argument("--set", dest="flow_set", choices=sorted(FLOW_SETS), default="p0", help="flow set: p0 (default) or p1 (-p1 paths)")
    s = sub.add_parser("refmap")
    s.add_argument("lang", choices=LANGS)
    s.add_argument("--mode", choices=["c5", "ref"], required=True)
    s.add_argument("--tool-output", required=True, type=Path, help="`scip print --json` output, or callgraph text (go)")
    s.add_argument("--tool-version", default="")
    s.add_argument("--exit-status", type=int, default=0)
    s.add_argument("--module-path", default="")
    s.add_argument("--repo-prefix", default="")
    s.add_argument("--documents-indexed", type=int)
    s.add_argument("--rows-only", action="store_true", help="c5 mode: write only truth/<lang>.c5-rows.jsonl and compare with the committed c5.json")
    s = sub.add_parser("verify")
    s.add_argument("langs", nargs="*", default=LANGS)
    a = ap.parse_args(argv)
    global FLOW_SET
    FLOW_SET = getattr(a, "flow_set", "p0")
    if a.cmd == "fetch":
        r = fetch(a.lang, a.force)
    elif a.cmd == "census":
        r = census(a.lang)
    elif a.cmd == "frame":
        r = cmd_frame(a.lang)
    elif a.cmd == "draw":
        r = cmd_draw(a.lang)
    elif a.cmd == "flowdraw":
        r = cmd_flowdraw(a.lang)
    elif a.cmd == "flowdraw-p1":
        r = cmd_flowdraw_p1(a.lang, a.exclude, a.counts, a.tier2_out, a.expect_counts)
        print(json.dumps(r, indent=1, sort_keys=True))
        return 0
    elif a.cmd == "c5pilot":
        r = cmd_c5pilot(a.lang)
    elif a.cmd == "brief":
        r = cmd_brief(a.lang)
    elif a.cmd == "flowbrief":
        r = cmd_flowbrief(a.lang)
    elif a.cmd == "flowcheck":
        r = cmd_flowcheck(a.lang)
        print(json.dumps(r, indent=1))
        return 1 if r["nErrors"] else 0
    elif a.cmd == "joern":
        r = cmd_joern(a.lang)
    elif a.cmd == "genaudit":
        r = cmd_genaudit(a.lang, a.select)
    elif a.cmd == "refmap":
        r = cmd_refmap(a.lang, a.tool_output, a.tool_version, a.mode, a.module_path, a.repo_prefix, a.exit_status, a.documents_indexed, a.rows_only)
    elif a.cmd == "verify":
        r = cmd_verify([l for l in a.langs if (frame_path(l)).exists()])
        print(json.dumps(r, indent=1))
        return 1 if r["problems"] else 0
    print(json.dumps(r, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
