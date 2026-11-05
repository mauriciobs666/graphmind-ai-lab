#!/usr/bin/env python3
"""export_kit.py — the writers' kit exporter (S48, P0-41; eval-protocol §13.1, plan §6.2).

Modes (exactly one; no mode flag = export):

  export_kit.py --repo R --eval-commit SHA [--out-dir D] [--manifest P]
        §13.1 steps 1-4: ``git archive`` into ``<out-dir>/kit-src/``, drop plan §6.2's excluded
        paths, content-check (delete + record every hit, re-grep to zero), write
        ``kit-manifest.json``, ``chmod -R a-w``. Refuses an existing ``kit-src``.
  export_kit.py --update-briefs [--out-dir D] [--manifest P]
        re-hash ``<out-dir>/roles/*/BRIEF.md`` into an existing manifest's ``briefs``.
  export_kit.py --draw-flow-anchors --kit-src D --eval-commit SHA [--out F]
        step 5: §3.6's frame from the kit (Python ``ast``; py-tree-sitter for TS/JS, from
        ``cpg/pyproject.toml``'s ``spike`` extra), 8 per language, ``work/flow-anchors.jsonl``.
  export_kit.py --list-units --kit-src D [--out F]          step 6 (``ports`` emitted empty)
  export_kit.py --list-cypher-tools --kit-src D [--out F]   step 7 (reads ``cypher-mcp/server.py``)

Deterministic on its inputs; canonical JSON. Prints counts, ids and paths only, never item content
(§13: the tool lists are manifest data, not ground truth, and ARE printed by the two ``--list-*``
modes). Exit: 0 ok · 1 check failed · 2 usage · 3 refused (existing state / symlink) · 4 missing
dependency or input.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tarfile
from pathlib import Path

# ----------------------------------------------------------------------------- shared constants

EXCLUDED_PREFIXES = ("cpg/", "deprecated/")          # plan §6.2
EXCLUDED_FILES = ("docs/BACKLOG.md",)
EXCLUDED_DOC_GLOB = "docs/**/connected-code-graph*"
CONTENT_PATTERN = r"connected-code-graph|connected code graph|C-1001|\bccg[_ -]"
CONTENT_RE = re.compile(CONTENT_PATTERN.encode(), re.IGNORECASE)

TEST_GLOBS = ("**/tests/**", "**/test_*.py", "**/*_test.py", "**/conftest.py",
              "**/*.test.{ts,tsx,js,jsx}", "**/*.spec.{ts,tsx,js,jsx}")            # §3.3
PROMPT_MD_GLOBS = ("**/AGENTS.md", "**/CLAUDE.md", "claude/**/*.md", "skills/**/*.md",
                   "opencode/agents/**/*.md", "opencode/skills/**/*.md", ".kiro/steering/*.md",
                   ".claude/commands/**/*.md", ".claude/agents/**/*.md")
PROMPT_MD_EXCEPT = ("claude/**/kaizen/**", "claude/README.md", "skills/README.md")
CODE_EXT = {".py", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".mts", ".cts", ".sh", ".bash",
            ".java", ".kt", ".kts", ".cs", ".go", ".c", ".h", ".cc", ".cpp", ".cxx", ".hpp"}
CONFIG_EXT = {".json", ".yaml", ".yml", ".toml"}
SKIP_DIRS = {"node_modules", "vendor", "third_party", ".venv", "site-packages", "__pycache__"}
JS_EXT = {".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".mts", ".cts"}
FLOW_PER_LANG = 8
FLOW_LANGS = ("python", "javascript")


def canon(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------------- glob matching

def _expand_braces(pat: str) -> list[str]:
    m = re.search(r"\{([^{}]*)\}", pat)
    if not m:
        return [pat]
    out = []
    for alt in m.group(1).split(","):
        out += _expand_braces(pat[:m.start()] + alt + pat[m.end():])
    return out


def _glob_re(pat: str) -> re.Pattern:
    i, out = 0, ""
    while i < len(pat):
        if pat.startswith("**/", i):
            out += "(?:.*/)?"; i += 3
        elif pat.startswith("/**", i) and i + 3 == len(pat):
            out += "/.*"; i += 3
        elif pat.startswith("**", i):
            out += ".*"; i += 2
        elif pat[i] == "*":
            out += "[^/]*"; i += 1
        elif pat[i] == "?":
            out += "[^/]"; i += 1
        else:
            out += re.escape(pat[i]); i += 1
    return re.compile(out + r"\Z")


_GLOB_CACHE: dict[str, list[re.Pattern]] = {}


def glob_match(pattern: str, path: str) -> bool:
    res = _GLOB_CACHE.get(pattern)
    if res is None:
        res = _GLOB_CACHE[pattern] = [_glob_re(p) for p in _expand_braces(pattern)]
    return any(r.match(path) for r in res)


def any_glob(patterns, path: str) -> bool:
    return any(glob_match(p, path) for p in patterns)


def is_test_path(rel: str) -> bool:
    return any_glob(TEST_GLOBS, rel)


def is_prompt_md(rel: str) -> bool:
    return any_glob(PROMPT_MD_GLOBS, rel) and not any_glob(PROMPT_MD_EXCEPT, rel)


def path_excluded(rel: str) -> bool:
    """Plan §6.2's excluded paths (not the content check)."""
    if rel.startswith(EXCLUDED_PREFIXES) or rel in EXCLUDED_FILES:
        return True
    parts = rel.split("/")
    # docs/**/connected-code-graph*: any path segment under docs/ starting with the token
    return parts[0] == "docs" and any(p.startswith("connected-code-graph") for p in parts[1:])


def path_class(rel: str) -> str:
    """Path class of an excluded file (§13.1 step 3); never reads content."""
    if is_test_path(rel):
        return "test"
    base = rel.rsplit("/", 1)[-1]
    ext = os.path.splitext(base)[1].lower()
    if rel.endswith(".md"):
        return "prompt-md" if is_prompt_md(rel) else "doc"
    if ext in CODE_EXT:
        return "code"
    if ext in CONFIG_EXT or base == ".env" or base.startswith(".env."):
        return "config"
    return "other"


# ----------------------------------------------------------------------------- export (steps 1-4)

def _git(repo: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", repo, *args], capture_output=True)


def _walk_files(root: Path):
    for dp, dns, fns in os.walk(root):
        dns.sort()
        for fn in sorted(fns):
            p = Path(dp) / fn
            yield p, p.relative_to(root).as_posix()


def _default_out_dir(commit: str) -> Path:
    home = os.environ.get("CCG_EVAL_HOME") or str(Path.home() / ".local/share/ccg-eval")
    return Path(home) / commit[:12]


def _brief_hashes(out_dir: Path) -> dict:
    briefs = {}
    roles = out_dir / "roles"
    if roles.is_dir():
        for d in sorted(roles.iterdir()):
            b = d / "BRIEF.md"
            if b.is_file():
                briefs[f"roles/{d.name}/BRIEF.md"] = sha256_file(b)
    return briefs


def export(repo: str, eval_commit: str, out_dir: Path | None, manifest_path: Path | None) -> int:
    r = _git(repo, "rev-parse", "--verify", f"{eval_commit}^{{commit}}")
    if r.returncode != 0:
        print(f"export_kit: cannot resolve commit {eval_commit!r} in {repo}", file=sys.stderr)
        return 4
    full = r.stdout.decode().strip()
    out_dir = out_dir or _default_out_dir(eval_commit)
    kit = out_dir / "kit-src"
    if kit.exists():
        print(f"export_kit: {kit} exists; refusing to overwrite", file=sys.stderr)
        return 3
    manifest_path = manifest_path or out_dir / "kit-manifest.json"
    kit.mkdir(parents=True)
    path_excluded_n = 0
    with subprocess.Popen(["git", "-C", repo, "archive", "--format=tar", full],
                          stdout=subprocess.PIPE) as proc:
        with tarfile.open(fileobj=proc.stdout, mode="r|") as tf:
            for m in tf:
                rel = m.name.rstrip("/")
                if rel in ("", ".") or rel.startswith(".git/") or rel == ".git":
                    continue
                if m.issym() or m.islnk():
                    print(f"export_kit: archive holds a link ({rel}); flatten it first", file=sys.stderr)
                    proc.kill()
                    return 3
                if m.isdir():
                    continue
                if path_excluded(rel):
                    path_excluded_n += 1
                    continue
                tf.extract(m, kit, filter="data")
        if proc.wait() != 0:
            print("export_kit: git archive failed", file=sys.stderr)
            return 4

    # step 3: content check (case-insensitive), delete + record, then re-grep to zero
    hits = []
    for p, rel in _walk_files(kit):
        if CONTENT_RE.search(p.read_bytes()):
            hits.append(rel)
    exclusions = []
    by_class: dict[str, int] = {}
    for rel in hits:
        cls = path_class(rel)
        exclusions.append({"path": rel, "reason": "content-check", "class": cls})
        by_class[cls] = by_class.get(cls, 0) + 1
        (kit / rel).unlink()
    for dp, dns, fns in os.walk(kit, topdown=False):      # drop directories the deletions emptied
        if Path(dp) != kit and not os.listdir(dp):
            os.rmdir(dp)
    left = [rel for p, rel in _walk_files(kit) if CONTENT_RE.search(p.read_bytes())]
    if left:
        print(f"export_kit: re-grep still hits {len(left)} file(s)", file=sys.stderr)
        return 1

    # step 4: manifest, then chmod -R a-w
    files = sorted(({"path": rel, "sha256": sha256_file(p)} for p, rel in _walk_files(kit)),
                   key=lambda f: f["path"])
    manifest = {
        "evalCommit": full,
        "files": files,
        "briefs": _brief_hashes(out_dir),
        "exclusions": {
            "paths": sorted(list(EXCLUDED_PREFIXES) + list(EXCLUDED_FILES) + [EXCLUDED_DOC_GLOB]),
            "contentCheck": {"pattern": CONTENT_PATTERN, "files": exclusions},
        },
        "excludedScope": {
            "byClass": dict(sorted(by_class.items())),
            "count": sum(by_class.get(c, 0) for c in ("code", "config", "prompt-md")),
        },
        "counts": {"files": len(files), "pathExcluded": path_excluded_n, "contentExcluded": len(hits)},
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(canon(manifest) + "\n", encoding="utf-8")
    for dp, dns, fns in os.walk(kit):
        for n in dns + fns:
            q = os.path.join(dp, n)
            os.chmod(q, os.stat(q).st_mode & ~0o222)
    os.chmod(kit, os.stat(kit).st_mode & ~0o222)
    print(f"kit-src: {kit}\nfiles: {len(files)}\npath-excluded: {path_excluded_n}\n"
          f"content-check-excluded: {len(hits)}\nexcluded-scope: {manifest['excludedScope']['count']}\n"
          f"manifest: {manifest_path}")
    return 0


def update_briefs(out_dir: Path, manifest_path: Path | None) -> int:
    manifest_path = manifest_path or out_dir / "kit-manifest.json"
    if not manifest_path.is_file():
        print(f"export_kit: {manifest_path} not found", file=sys.stderr)
        return 4
    m = json.loads(manifest_path.read_text(encoding="utf-8"))
    m["briefs"] = _brief_hashes(out_dir)
    manifest_path.write_text(canon(m) + "\n", encoding="utf-8")
    print(f"briefs: {len(m['briefs'])}\nmanifest: {manifest_path}")
    return 0


# ----------------------------------------------------------------------------- --list-cypher-tools

def list_cypher_tools(kit_src: Path) -> list[str]:
    """Tool names registered by ``@<x>.tool`` decorators in ``cypher-mcp/server.py`` (§13.1 step 7)."""
    server = kit_src / "cypher-mcp" / "server.py"
    if not server.is_file():
        raise FileNotFoundError(f"{server} not found")
    tree = ast.parse(server.read_text(encoding="utf-8"))
    names: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for dec in node.decorator_list:
            call = dec if isinstance(dec, ast.Call) else None
            target = call.func if call else dec
            if not (isinstance(target, ast.Attribute) and target.attr == "tool"):
                continue
            name = node.name
            if call:
                if call.args and isinstance(call.args[0], ast.Constant) and isinstance(call.args[0].value, str):
                    name = call.args[0].value
                for kw in call.keywords:
                    if kw.arg == "name" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                        name = kw.value.value
            names.append(name)
    if not names:
        raise LookupError("no @<x>.tool registration found in cypher-mcp/server.py")
    return sorted(set(names))


# ----------------------------------------------------------------------------- --list-units

_MANIFESTS = {"pyproject.toml": "python", "requirements.txt": "python", "package.json": "javascript"}


def list_units(kit_src: Path) -> list[dict]:
    """One unit per (directory in census scope, ecosystem) holding a manifest file (§13.1 step 6)."""
    units = set()
    for p, rel in _walk_files(kit_src):
        base = rel.rsplit("/", 1)[-1]
        if base not in _MANIFESTS:
            continue
        if any(seg in SKIP_DIRS for seg in rel.split("/")[:-1]) or is_test_path(rel) or path_excluded(rel):
            continue
        d = rel.rsplit("/", 1)[0] if "/" in rel else "."
        units.add((d, _MANIFESTS[base]))
    return [{"rootPath": d, "lang": lang, "ports": []} for d, lang in sorted(units)]


# ----------------------------------------------------------------------------- --draw-flow-anchors

class Cand:
    __slots__ = ("path", "qname", "lineStart", "lineEnd")

    def __init__(self, path, qname, ls, le):
        self.path, self.qname, self.lineStart, self.lineEnd = path, qname, ls, le


def _py_frame(rel: str, src: str) -> list[Cand]:
    """Named, bodied Python functions and methods: a simplified bundle-spec §4.1 chain (classes and
    functions add segments; ``@property``/``.setter``/``.deleter`` add ``<get>``/``<set>``/``<del>``;
    the k-th same-base sibling gets ``#k``); ``#k`` duplicates are counted but never candidates (§2.3)."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    out: list[Cand] = []
    dup: dict[tuple, int] = {}

    def deco_names(fn):
        names = []
        for d in fn.decorator_list:
            t = ast.unparse(d)
            names.append(t)
        return names

    def stub(fn) -> bool:
        body = [b for b in fn.body if not (isinstance(b, ast.Expr) and isinstance(b.value, ast.Constant)
                                           and isinstance(b.value.value, str))]
        return all((isinstance(b, ast.Expr) and isinstance(b.value, ast.Constant) and b.value.value is Ellipsis)
                   or isinstance(b, ast.Pass) for b in body)

    def visit(node, chain: tuple):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, ast.ClassDef):
                segs, _k = _seg(chain, ch.name, ())
                visit(ch, chain + segs)
            elif isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef)):
                decos = deco_names(ch)
                acc: tuple = ()
                for t in decos:
                    if t == "property":
                        acc = ("<get>",)
                    elif re.fullmatch(r"\w+\.(setter|deleter)", t):
                        acc = ("<set>",) if t.endswith("setter") else ("<del>",)
                if any(t.split(".")[-1] == "overload" for t in decos):
                    visit(ch, chain + (ch.name,))              # @overload stubs are not symbols (rule 8)
                    continue
                segs, k = _seg(chain, ch.name, acc)
                new = chain + segs
                ls = min([d.lineno for d in ch.decorator_list] + [ch.lineno])
                bodyless = stub(ch) or any(t.split(".")[-1] in ("overload", "abstractmethod") for t in decos)
                if k == 1 and not bodyless:
                    out.append(Cand(rel, f"{rel}::{'.'.join(new)}", ls, ch.end_lineno))
                visit(ch, new)
            else:
                visit(ch, chain)

    def _seg(chain, name, acc):
        key = (chain, name, acc)
        k = dup.get(key, 0) + 1
        dup[key] = k
        return (name + (f"#{k}" if k > 1 else ""),) + acc, k

    visit(tree, ())
    return out


def _ts_parsers():
    try:
        import tree_sitter as ts
        import tree_sitter_javascript as tjs
        import tree_sitter_typescript as tts
    except ImportError as e:                                           # pragma: no cover
        raise RuntimeError(f"py-tree-sitter (cpg/pyproject.toml `spike` extra) is required: {e}")
    langs = {
        "js": ts.Language(tjs.language()),
        "ts": ts.Language(tts.language_typescript()),
        "tsx": ts.Language(tts.language_tsx()),
    }
    return {k: ts.Parser(v) for k, v in langs.items()}


_JS_CLASS = {"class_declaration", "abstract_class_declaration", "class"}
_JS_FN_DECL = {"function_declaration", "generator_function_declaration"}
_JS_FN_EXPR = {"function_expression", "function", "arrow_function", "generator_function"}


def _txt(n) -> str:
    return n.text.decode("utf-8", "replace")


def _js_binding(n):
    par = n.parent
    while par is not None and par.type in ("parenthesized_expression", "as_expression", "satisfies_expression",
                                           "non_null_expression"):
        n, par = par, par.parent
    if par is None:
        return None
    if par.type == "variable_declarator" and par.child_by_field_name("value") == n:
        nm = par.child_by_field_name("name")
        return _txt(nm) if nm is not None and nm.type == "identifier" else None
    if par.type in ("public_field_definition", "field_definition") and par.child_by_field_name("value") == n:
        nm = par.child_by_field_name("name") or par.child_by_field_name("property")
        return _txt(nm) if nm is not None else None
    if par.type == "pair" and par.child_by_field_name("value") == n:
        k = par.child_by_field_name("key")
        if k is not None:
            kt = _txt(k).strip("\"'`")
            if re.fullmatch(r"[A-Za-z_$][\w$]*", kt):
                return kt
    return None


def _js_frame(parsers, rel: str, data: bytes) -> list[Cand]:
    ext = os.path.splitext(rel)[1].lower()
    parser = parsers["tsx" if ext == ".tsx" else "ts" if ext in (".ts", ".mts", ".cts") else "js"]
    root = parser.parse(data).root_node
    out: list[Cand] = []
    dup: dict[tuple, int] = {}

    def seg(chain, segs):
        key = (chain, tuple(segs))
        k = dup.get(key, 0) + 1
        dup[key] = k
        return (list(segs[:-1]) + [segs[-1] + (f"#{k}" if k > 1 else "")]) if segs else [], k

    def rec(n, chain: tuple, anon: bool):
        t = n.type
        new_chain, new_anon = chain, anon
        added = None
        is_fn = False
        if t in _JS_CLASS:
            nm = n.child_by_field_name("name")
            if nm is None:
                new_anon = True
            else:
                s, _ = seg(chain, [_txt(nm)])
                new_chain = chain + tuple(s)
        elif t in ("internal_module", "module"):
            nm = n.child_by_field_name("name")
            s, _ = seg(chain, [_txt(nm) if nm is not None else "?"])
            new_chain = chain + tuple(s)
        elif t in _JS_FN_DECL:
            nm = n.child_by_field_name("name")
            if nm is None:
                new_anon = True
            else:
                added, is_fn = [_txt(nm)], True
        elif t == "method_definition":
            nm = n.child_by_field_name("name")
            if nm is None or nm.type == "computed_property_name":
                new_anon = True
            else:
                name = _txt(nm).strip("\"'")
                name = "<init>" if name == "constructor" else name
                acc = [c.type for c in n.children if c.type in ("get", "set")]
                added = [name] + (["<get>"] if "get" in acc else ["<set>"] if "set" in acc else [])
                is_fn = True
        elif t in _JS_FN_EXPR:
            nm = n.child_by_field_name("name")
            b = _txt(nm) if nm is not None else _js_binding(n)
            if b:
                added, is_fn = [b], True
            else:
                new_anon = True
        if added is not None:
            s, k = seg(chain, added)
            new_chain = chain + tuple(s)
            body = n.child_by_field_name("body")
            if is_fn and not anon and k == 1 and body is not None:
                out.append(Cand(rel, f"{rel}::{'.'.join(new_chain)}",
                                n.start_point[0] + 1, n.end_point[0] + 1))
        for c in n.children:
            rec(c, new_chain, new_anon)

    rec(root, (), False)
    return out


def flow_frame(kit_src: Path) -> dict[str, list[Cand]]:
    """§3.6's frame: eligible named bodied functions, ``lineEnd − lineStart ≥ 2``, outside test code."""
    frame = {"python": [], "javascript": []}
    parsers = None
    for p, rel in _walk_files(kit_src):
        ext = os.path.splitext(rel)[1].lower()
        if ext not in JS_EXT | {".py"}:
            continue
        if rel.endswith((".d.ts", ".pyi")) or is_test_path(rel) or path_excluded(rel):
            continue
        if any(seg in SKIP_DIRS for seg in rel.split("/")[:-1]):
            continue
        if ext == ".py":
            cands = _py_frame(rel, p.read_text(encoding="utf-8", errors="replace"))
            lang = "python"
        else:
            parsers = parsers or _ts_parsers()
            cands = _js_frame(parsers, rel, p.read_bytes())
            lang = "javascript"
        frame[lang] += [c for c in cands if c.lineEnd - c.lineStart >= 2 and "<anon>" not in c.qname]
    return frame


def draw_flow_anchors(kit_src: Path, eval_commit: str) -> list[dict]:
    """8 per language by S05's convention: ascending ``SHA-256(seed + "|" + qname)`` where
    ``seed = "ccg-p0f-flow-v1|" + lang + "|" + EVAL_COMMIT``."""
    frame = flow_frame(kit_src)
    rows = []
    for lang in FLOW_LANGS:
        if len(frame[lang]) < FLOW_PER_LANG:
            raise ValueError(f"the {lang} frame has {len(frame[lang])} anchors, fewer than {FLOW_PER_LANG}")
        seed = f"ccg-p0f-flow-v1|{lang}|{eval_commit}"
        ranked = sorted(frame[lang], key=lambda c: sha256_bytes((seed + "|" + c.qname).encode("utf-8")))
        for i, c in enumerate(ranked[:FLOW_PER_LANG], 1):
            rows.append({"lang": lang, "rank": i, "qname": c.qname, "path": c.path,
                         "lineStart": c.lineStart, "lineEnd": c.lineEnd})
    return rows


# ----------------------------------------------------------------------------- CLI

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--draw-flow-anchors", action="store_true")
    mode.add_argument("--list-units", action="store_true")
    mode.add_argument("--list-cypher-tools", action="store_true")
    mode.add_argument("--update-briefs", action="store_true")
    ap.add_argument("--repo", help="git repository to export (export mode)")
    ap.add_argument("--eval-commit", help="EVAL_COMMIT (seed input / commit to archive)")
    ap.add_argument("--out-dir", type=Path, help="default ~/.local/share/ccg-eval/<first 12 of EVAL_COMMIT>")
    ap.add_argument("--manifest", type=Path, help="kit-manifest.json path (default <out-dir>/kit-manifest.json)")
    ap.add_argument("--kit-src", type=Path, help="exported kit (list/draw modes)")
    ap.add_argument("--out", type=Path, help="output file (draw: default <kit-src>/../work/flow-anchors.jsonl)")
    a = ap.parse_args(argv)

    try:
        if a.draw_flow_anchors or a.list_units or a.list_cypher_tools:
            if not a.kit_src or not a.kit_src.is_dir():
                ap.error("--kit-src DIR is required")
            if a.list_cypher_tools:
                rows = [{"name": n} for n in list_cypher_tools(a.kit_src)]
            elif a.list_units:
                rows = list_units(a.kit_src)
            else:
                if not a.eval_commit:
                    ap.error("--eval-commit is required to draw")
                rows = draw_flow_anchors(a.kit_src, a.eval_commit)
            text = "".join(canon(r) + "\n" for r in rows)
            if a.draw_flow_anchors:
                out = a.out or a.kit_src.parent / "work" / "flow-anchors.jsonl"
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(text, encoding="utf-8")
                by = {l: sum(1 for r in rows if r["lang"] == l) for l in FLOW_LANGS}
                print(f"anchors: {len(rows)} {by}\nsha256: {sha256_bytes(text.encode())}\nfile: {out}")
            elif a.out:
                a.out.parent.mkdir(parents=True, exist_ok=True)
                a.out.write_text(text, encoding="utf-8")
                print(f"rows: {len(rows)}\nfile: {a.out}")
            else:
                sys.stdout.write(text)
            return 0
        if a.update_briefs:
            if not (a.out_dir or a.eval_commit):
                ap.error("--out-dir (or --eval-commit) is required")
            return update_briefs(a.out_dir or _default_out_dir(a.eval_commit), a.manifest)
        if not a.repo or not a.eval_commit:
            ap.error("export needs --repo and --eval-commit")
        return export(a.repo, a.eval_commit, a.out_dir, a.manifest)
    except ValueError as e:
        print(f"export_kit: {e}", file=sys.stderr)
        return 1
    except (FileNotFoundError, LookupError, RuntimeError) as e:
        print(f"export_kit: {e}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    sys.exit(main())
