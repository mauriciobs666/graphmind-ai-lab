#!/usr/bin/env python3
"""Development-contact list for the two proving-repo corpora (ML note v1.7 s3.1 circularity stop; R6).

Output: contact.json (beside this file). Deterministic: reads only blobs of HEAD (git ls-tree / git cat-file).
Never reads labels/, truth/, sample/, results/ or the (gitignored) frame/.  Corpus universe: tracked `.py` under
falkor-chat/server (107) and tracked salesperson TS/JS files outside bundle spec4 s1's `test` pattern
(`*.test.<ext>`, `*.spec.<ext>`, `e2e/`, `__tests__/`) -- the census frame's 40 files are a subset (the other 6 have no call site).

A corpus file is *contact* when any committed development artifact OUTSIDE cpg/spikes/frontend-fitness/
names it, by path (repo-rooted `falkor-chat/server/...` / `salesperson/...`, optionally behind a directory
prefix such as an absolute path; or corpus-relative `falkorchat/api.py`, `src/api/client.ts`) or by a defined
symbol's qualified name (`<path>::name` is a path match; `api.py::build_router.health` / `app.py:485` name a file
by its corpus-unique basename; the Python dotted form `falkorchat.storefront_api.f` or `falkorchat.app:app` is
resolved to the module file by longest-prefix).

`derivedFrom` is the commit SHA whose tree was read (HEAD at generation); --check re-derives from exactly that tree, so a later
commit to any file cannot move the list, and fails if that commit is unreachable (not an ancestor of HEAD) or the derivation differs.
Generating in the commit that adds contact.json is therefore safe: it records the parent (HEAD before that commit).
`flowAnchors` reads the committed `sample/<lang>-flow.jsonl` -- the one file of sample/ that is read -- and is NOT part of `files`.

Usage: make_contact.py [--check] [--summary] [--out PATH] [--tree-root DIR] [--from SHA]  (--summary: counts against the local frame/, not committed)   (--tree-root: read files from a directory copy
of the artifact tree instead of HEAD; used by the mutation tests).
"""
import json, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("CONTACT_REPO") or subprocess.run(["git", "-C", HERE, "rev-parse", "--show-toplevel"], capture_output=True,
                                                         text=True, check=True).stdout.strip()
OUT = os.path.join(HERE, "contact.json")
VERSION = 1
SELF_DIR = "cpg/spikes/frontend-fitness/"
# Not development sources (ML v1.8 R6): the measurement, synthetic snippets, this check's own test (mutation inputs)
SELF_EXCLUDE = {"cpg/tests/test_contact.py", "cpg/tests/test_frontend_fitness.py"}
LOG_NAMES = {"HISTORY.md", "BACKLOG.md"}              # logs, derivative of the sources
MAX_BYTES = 2_000_000
SKIP_EXT = (".jpg", ".png", ".svg", ".ico", ".lock", ".gz", ".zip", ".pyc", ".woff", ".woff2")

CORPORA = {
    "python": {"root": "falkor-chat/server", "exts": (".py",), "test_re": None},
    "javascript": {"root": "salesperson", "exts": (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"),
                   "test_re": re.compile(r"\.(test|spec)\.[A-Za-z]+$|/e2e/|/__tests__/")},
}
PY_PKG = "falkorchat"

# search sources, in evidence-priority order; first matching rule wins for an artifact
def source_of(p):
    base = p.rsplit("/", 1)[-1]
    if base in LOG_NAMES or base.endswith("-ml.md") or base.endswith("-coordination.md"):
        return None
    if p.startswith("cpg/fixtures/"):
        return "fixtures"
    if p.startswith("cpg/packs/"):
        return "packs"
    if p.startswith("cpg/tests/golden/") or p.startswith("cpg/tests/data/"):
        return "golden"
    if p.startswith("cpg/spikes/vertical-slice/") and "/tests/" not in p:
        return "slice"
    if re.match(r"cpg/tests/[^/]+$", p) or re.match(r"cpg/.*/tests/", p):
        return "tests"
    if re.match(r"docs/(plans|requirements|test-plans)/connected-code-graph[^/]*\.md$", p):
        return "docs"
    if p.startswith("cpg/"):
        return "cpg-other"
    return None
SOURCES = ["fixtures", "golden", "slice", "tests", "packs", "docs", "cpg-other"]

TOKEN = re.compile(r"[A-Za-z0-9_@+./-]+")
BASEQ = re.compile(r"(?<![\w./-])([A-Za-z0-9_-]+\.(?:py|tsx|ts|jsx|js|mjs|cjs))(?=::|:\d)")
DOTTED = re.compile(r"(?<![\w.])" + PY_PKG + r"((?:\.[A-Za-z_]\w*)+)")


def norm_token(t):
    """Path normalisation: strip sentence punctuation and a leading ./"""
    t = t.rstrip(".,-/")
    while t.startswith("./"):
        t = t[2:]
    return t


def git_blobs(treeish="HEAD", repo=None):
    out = subprocess.run(["git", "-C", repo or REPO, "ls-tree", "-r", "-z", treeish], capture_output=True, text=True, check=True).stdout
    ents = []
    for e in out.split("\0"):
        if not e:
            continue
        meta, path = e.split("\t", 1)
        ents.append((path, meta.split()[2]))
    return ents


def read_blobs(ents, want, repo=None):
    proc = subprocess.Popen(["git", "-C", repo or REPO, "cat-file", "--batch"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    res = {}
    for path, sha in ents:
        if path not in want:
            continue
        proc.stdin.write((sha + "\n").encode()); proc.stdin.flush()
        hdr = proc.stdout.readline().split()
        size = int(hdr[2])
        data = proc.stdout.read(size); proc.stdout.read(1)
        res[path] = data
    proc.stdin.close(); proc.wait()
    return res


def load_tree(tree_root=None, treeish="HEAD", repo=None):
    """-> {path: bytes} for every candidate file; path list of the whole tree."""
    if tree_root:
        allp = []
        for dp, dn, fn in os.walk(tree_root):
            dn.sort()
            for f in sorted(fn):
                allp.append(os.path.relpath(os.path.join(dp, f), tree_root).replace(os.sep, "/"))
        def rd(ps):
            return {p: open(os.path.join(tree_root, p), "rb").read() for p in ps if os.path.exists(os.path.join(tree_root, p))}
        return sorted(allp), rd
    ents = git_blobs(treeish, repo)
    allp = sorted(p for p, _ in ents)
    def rd(ps):
        return read_blobs(ents, set(ps), repo)
    return allp, rd


def corpus_universe(allp, rd):
    uni = {}
    for name, c in CORPORA.items():
        root = c["root"] + "/"
        files = [p for p in allp if p.startswith(root) and p.endswith(c["exts"]) and "/node_modules/" not in p
                 and "/.venv/" not in p and "/dist/" not in p and not (c["test_re"] and c["test_re"].search(p))]
        uni[name] = {"files": files}
    return uni


def module_files(py_rel):
    """module dotted name (after package) -> corpus-relative file"""
    m = {}
    for rel in py_rel:
        if not rel.startswith(PY_PKG + "/"):
            continue
        mod = rel[len(PY_PKG) + 1:-3].replace("/", ".")
        if mod.endswith(".__init__"):
            mod = mod[: -len(".__init__")]
        m[mod] = rel
    return m


def resolve_dotted(tail, mods):
    """`.a.b.c` -> file of the longest module prefix (a.b.c, a.b, a); None if none is a module."""
    parts = tail.lstrip(".").split(".")
    for n in range(len(parts), 0, -1):
        mod = ".".join(parts[:n])
        if mod in mods:
            return mods[mod]
    return None


def resolve_sha(treeish="HEAD", repo=None):
    r = subprocess.run(["git", "-C", repo or REPO, "rev-parse", "--verify", "--quiet", treeish + "^{commit}"], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def norm_flow_path(p, root):
    """sample/<lang>-flow.jsonl `path` -> corpus-relative, forward slashes, no ./ or repo-root prefix"""
    p = p.replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    if p.startswith(root + "/"):
        p = p[len(root) + 1:]
    return p


def flow_anchors(rd, universe_rel):
    out = {}
    for name, c in CORPORA.items():
        f = "cpg/spikes/frontend-fitness/sample/%s-flow.jsonl" % name
        txt = rd([f]).get(f, b"").decode("utf-8", "ignore")
        paths = set()
        for l in txt.splitlines():
            if l.strip():
                paths.add(norm_flow_path(json.loads(l)["path"], c["root"]))
        out[name] = sorted(paths)
    return out


def derive(tree_root=None, treeish="HEAD", repo=None):
    allp, rd = load_tree(tree_root, treeish, repo)
    uni = corpus_universe(allp, rd)
    # full-path and relative-path lookup tables
    full, rel = {}, {}
    for name, c in CORPORA.items():
        root = c["root"] + "/"
        for p in uni[name]["files"]:
            full[p] = (name, p)
            r = p[len(root):]
            if "/" in r:                      # root-level files (pyproject, vite.config) need the repo-rooted form
                rel.setdefault(r, []).append((name, p))
    by_base = {}
    for p in full:
        by_base.setdefault(p.rsplit("/", 1)[-1], []).append(p)
    base_unique = {b: ps[0] for b, ps in by_base.items() if len(ps) == 1}   # basename names one corpus file
    py_rel = [p[len(CORPORA["python"]["root"]) + 1:] for p in uni["python"]["files"]]
    mods = module_files(py_rel)
    markers = sorted({c["root"] + "/" for c in CORPORA.values()}, key=len, reverse=True)

    searched = [p for p in allp if source_of(p) and not p.startswith(SELF_DIR) and p not in SELF_EXCLUDE
                and not p.endswith(SKIP_EXT)]
    blobs = rd(searched)
    hits = {}       # corpus path -> {artifact: {(match, form)}}
    nsearched = {s: 0 for s in SOURCES}
    for art in searched:
        data = blobs.get(art)
        if data is None or len(data) > MAX_BYTES:
            continue
        src = source_of(art)
        nsearched[src] += 1
        text = data.decode("utf-8", "ignore")
        rel_ok = src != "fixtures" and "fixtures" not in os.path.basename(art)
        for tm in TOKEN.finditer(text):
            tok = tm.group(0)
            ntok = norm_token(tok)
            for mk in markers:
                start = 0
                while True:
                    i = ntok.find(mk, start)
                    if i < 0:
                        break
                    if i == 0 or ntok[i - 1] == "/":
                        cand = ntok[i:]
                        if cand in full:
                            hits.setdefault(cand, {}).setdefault(art, set()).add((cand, "path"))
                    start = i + 1
            if rel_ok and ntok in rel:
                for (_n, p) in rel[ntok]:
                    hits.setdefault(p, {}).setdefault(art, set()).add((ntok, "rel"))
        for bm in BASEQ.finditer(text):                  # `api.py::build_router.health`, `app.py:485`
            p = base_unique.get(bm.group(1))
            if p:
                hits.setdefault(p, {}).setdefault(art, set()).add((bm.group(1), "base"))
        for dm in DOTTED.finditer(text):
            f = resolve_dotted(dm.group(1), mods)
            if f:
                p = CORPORA["python"]["root"] + "/" + f
                hits.setdefault(p, {}).setdefault(art, set()).add((PY_PKG + dm.group(1), "qname"))

    order = {s: i for i, s in enumerate(SOURCES)}
    out = {"version": VERSION, "derivedFrom": None if tree_root else resolve_sha(treeish, repo),
           "flowAnchors": flow_anchors(rd, None),
           "rule": "file listed iff a committed artifact outside cpg/spikes/frontend-fitness/ names it by path or by a defined symbol's qname",
           "searchSources": {s: nsearched[s] for s in SOURCES},
           "s06": {"ownListOnDisk": False,
                   "note": "S06 (eval-protocol v1.2 s8.2) records a contact RULE and delegates the lists to S14 (docs/test-reports/connected-code-graph-p0-slice-report.md), "
                           "S30 (docs/plans/connected-code-graph-rule-pack-spec.md) and S56's flow methods (labels/<lang>-flow.jsonl, inside frontend-fitness: not read here); "
                           "the first two are covered by the 'docs' source below, S56 is not (S47 adds it)."},
           "corpora": {}}
    for name, c in CORPORA.items():
        files = {}
        for p in uni[name]["files"]:
            if p not in hits:
                continue
            arts = hits[p]
            ev_all = []
            for a in sorted(arts, key=lambda a: (order[source_of(a)], a)):
                for m, form in sorted(arts[a]):
                    ev_all.append({"artifact": a, "match": m, "form": form, "source": source_of(a)})
            seen, ev = set(), []
            for e in sorted(ev_all, key=lambda e: ({"path": 0, "qname": 1, "base": 2, "rel": 3}[e["form"]], order[e["source"]], e["artifact"], e["match"])):
                if e["artifact"] in seen:
                    continue
                seen.add(e["artifact"]); ev.append(e)
                if len(ev) == 3:
                    break
            rp = p[len(c["root"]) + 1:]
            files[rp] = {"artifacts": len(arts),
                         "forms": sorted({f for a in arts.values() for _, f in a}),
                         "sources": sorted({source_of(a) for a in arts}, key=order.get), "evidence": ev}
        out["corpora"][name] = {"root": c["root"], "corpusFiles": len(uni[name]["files"]), "contactFiles": len(files),
                                "files": dict(sorted(files.items()))}
    return out


def render(obj):
    return json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=False) + "\n"


def summary(obj):
    """Counts against the local (gitignored) frame/<lang>.jsonl and truth/<lang>.ref.jsonl -- reporting only."""
    for name in CORPORA:
        c = obj["corpora"][name]
        lang = name
        fr = os.path.join(HERE, "frame", lang + ".jsonl")
        ref = os.path.join(HERE, "truth", lang + ".ref.jsonl")
        frame = {json.loads(l)["path"] for l in open(fr)} if os.path.exists(fr) else set()
        cf = set(c["files"])
        row = {"corpusFiles": c["corpusFiles"], "contactFiles": c["contactFiles"], "frameFiles": len(frame),
               "contactInFrame": len(cf & frame)}
        smp = os.path.join(HERE, "sample", lang + ".jsonl")
        if os.path.exists(ref):
            n_in = n_con = 0
            site_path = {}
            if os.path.exists(smp):                     # python: ref rows carry no site path; join on siteId
                for l in open(smp):
                    r = json.loads(l); site_path[r["siteId"]] = r["path"]
            for l in open(ref):
                r = json.loads(l)
                if r.get("static", {}).get("kind") != "internal":
                    continue
                p = (r.get("site") or {}).get("path") or site_path.get(r["siteId"])
                if p is None:
                    continue
                n_in += 1; n_con += p in cf
            row.update(internalSitesWithPath=n_in, internalInContact=n_con, internalNonContact=n_in - n_con)
        print(name, json.dumps(row))


def check(path, repo=None):
    """-> (ok, message). Re-derives from the tree named by the file's own `derivedFrom`."""
    if not os.path.exists(path):
        return False, "contact.json missing"
    text = open(path).read()
    sha = json.loads(text).get("derivedFrom")
    if not sha or not re.fullmatch(r"[0-9a-f]{40}", sha):
        return False, "derivedFrom missing or not a full sha"
    if resolve_sha(sha, repo) != sha:
        return False, "derivedFrom %s is unreachable (not a commit here)" % sha
    if subprocess.run(["git", "-C", repo or REPO, "merge-base", "--is-ancestor", sha, "HEAD"]).returncode != 0:
        return False, "derivedFrom %s is not an ancestor of HEAD" % sha
    if render(derive(treeish=sha, repo=repo)) != text:
        return False, "contact.json differs from its derivation at %s" % sha
    return True, "ok"


def main(argv):
    tree_root = None
    out = OUT
    if "--tree-root" in argv:
        tree_root = argv[argv.index("--tree-root") + 1]
    if "--out" in argv:
        out = argv[argv.index("--out") + 1]
    if "--check" in argv:
        ok, msg = check(out)
        sys.stderr.write(msg + "\n")
        return 0 if ok else 1
    treeish = argv[argv.index("--from") + 1] if "--from" in argv else "HEAD"
    obj = derive(tree_root, treeish)
    text = render(obj)
    if "--summary" in argv:
        summary(obj); return 0
    open(out, "w").write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
