"""Raw facts -> one bundle (C-1001 P1-05, second dispatch): the production port of the slice's ``assemble()``.

``assemble`` turns the facts of one unit into the bundle's records (spec3 §5), ``build_bundle`` writes them with
``ccg.atrest.write_bundle`` (the only writer: ``0600`` file, ``0700`` directory, HMAC custody stays with ``atrest``).
Ported from ``cpg/spikes/vertical-slice/normalize.py`` under the slice profile: equivalence before improvement
(`docs/plans/connected-code-graph-p1.md` §8, D7), so the header still names the slice's normalizer and version.

**Determinism.** The raw facts are not byte-stable: Joern returns the ``d`` and ``cb`` edges in an order that varies from
run to run (the two runs agree as multisets). Bundle bytes therefore never depend on fact order here: ``d`` and ``cb``
are sorted by their whole canonical form before the first-wins de-duplication of ``data`` records (two raw edges
can share an id and differ in status, truncation) and the G-4 filter; every other kind is keyed by position and CPG id,
or ends in the record sort of ``ccg.bundle.dumps``.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from ccg import atrest, bundle, ids
from ccg.units import Unit

from . import blame as blame_mod
from . import javascript, python
from .core import (FIELD_CAP, Builder, Resolver, Source, collapse, fingerprint, import_symbol_id, kw_head, lang_files,
                   load_facts, name_norm, owned_regions, path_from, repo_relative, sha256, span_start, sym_name)
from .mask import mask_cond_code

#: The slice profile's header identity. A later step that changes output renames or re-versions it (the stamp reads it).
NORMALIZER = "ccg-slice-normalize"
NORMALIZER_VERSION = "0.4"     # bumped with each registered delta of tests/slice/test_normalize_equivalence.py (the last delta pins the same literal)
JOERN_VERSION = "v4.0.579"
DEFAULT_VENDORED = ["_vendor", "node_modules", "third-party", "third_party", "vendor", "vendored"]


class BuildError(RuntimeError):
    """The bundle cannot be built (a qname collision, a mismatching commit)."""


# -- call-site positions ---------------------------------------------------------------------
def callee_name(c: dict) -> str:
    if c["name"] == "<operator>.new":
        m = re.match(r"\s*new\s+([A-Za-z_$][\w$.]*)", c["code"])
        return m.group(1).split(".")[-1] if m else "<dynamic>"
    return c["name"] or "<dynamic>"


def locate_callees(B: Builder, file: str, calls: list[dict]):
    """(line, col) of each call's callee last identifier (spec §5.7), derived from source.  A call whose
    callee name cannot be found in the source is absent from the result (synthetic: see the caller)."""
    lines = B.src.lines(file)
    starts: dict[tuple, list[dict]] = defaultdict(list)
    for c in calls:
        starts[(c["l"], c["c"], callee_name(c))].append(c)
    pos = {}
    for (ln, cl, name), grp in starts.items():
        grp.sort(key=lambda c: (len(c["code"]), c["id"]))
        off = cl - B.col_base
        text = "\n".join(lines[ln - 1: ln + 400])      # from the call start, restricted to a window
        if off < 0 or off > len(lines[ln - 1]):
            off = 0
        text = text[off:]
        pat = re.compile(r"(?<![\w$])" + re.escape(name) + r"(?![\w$])" +
                         ("" if grp[0]["name"] == "<operator>.new" else r"\s*(?:\?\.)?\s*(?:<[^()]*?>)?\s*\("))
        hits = [m for m in pat.finditer(text)]
        for k, c in enumerate(grp):
            if k < len(hits):
                before = text[:hits[k].start()]
                dl = before.count("\n")
                col = (len(before) - (before.rfind("\n") + 1)) + (off if dl == 0 else 0)
                pos[c["id"]] = (ln + dl, col)
            # else: the callee name never occurs in the source at/after the call's start: a call the frontend
            # synthesized (Python `__iter__`/`__next__`/comprehension `append`, JS `require` for imports,
            # `__ecma.Array.factory`); not a source-level call, so not a call site (§5.6)
    return pos


# -- G-4, ordering ---------------------------------------------------------------------------
def g4_filter(edges):
    """-spec2 §1.4 (G-4): drop CDG self-edges (``point == cond``) and de-duplicate (point, cond) pairs, in
    first-seen order.  Returns (kept, self_dropped, duplicates_dropped).  A ``data`` self-edge (``use == def``) is a
    different record and is kept elsewhere (spec4 §2, §5.10)."""
    kept, seen, n_self, n_dup = [], set(), 0, 0
    for point, cond in edges:
        if point == cond:
            n_self += 1
            continue
        if (point, cond) in seen:
            n_dup += 1
            continue
        seen.add((point, cond))
        kept.append((point, cond))
    return kept, n_self, n_dup


def canonical_order(facts: list[dict]) -> list[dict]:
    """``d``/``cb`` facts sorted by their whole canonical JSON: a total order, independent of the order Joern returned."""
    return sorted(facts, key=lambda f: json.dumps(f, sort_keys=True, ensure_ascii=False))


def canonical_args(args: list[dict]) -> list[dict]:
    """A call's or decorator's arguments in a total order (position, then whole canonical JSON), so that "the first
    argument that names a path wins" does not depend on the order Joern listed them in. The spec is silent on a repeated
    path; both value sets and decorator arguments take the first."""
    return sorted(args, key=lambda a: (int(a["i"]), json.dumps(a, sort_keys=True, ensure_ascii=False)))


def file_role(rel: str) -> str:
    base = rel.rsplit("/", 1)[-1].lower()
    parts = rel.lower().split("/")
    if (re.search(r"\.example", base) or re.search(r"\.sample", base) or base.startswith("example")
            or base.startswith("sample") or "examples" in parts[:-1] or "samples" in parts[:-1]):
        return "example"
    return "primary"


def is_vendored(rel: str) -> bool:
    return any(s in DEFAULT_VENDORED for s in rel.split("/")[:-1])


def _blame_regions(s, ls: int, le: int) -> list[tuple[int, int]]:
    """A symbol's span plus its ``ownedRegions`` (spec3 §5.3): the lines ``lastChanged*`` is read from."""
    return [(ls, le)] + owned_regions(s)


# -- assembly --------------------------------------------------------------------------------
def assemble(*, lang: str, unit_root: str, repo_id: str, facts, stage, astgen_dir, repo, commit: str,
             audit_ids, extract_script, built_at: str | None = None, skipped=None) -> tuple[list[dict], dict]:
    """The header and the records of one unit's bundle, plus a measurement report.

    ``facts`` is ``core.load_facts``' result, ``stage`` the staged unit, ``astgen_dir`` the Babel ASTs (JS/TS; ``None``
    for Python), ``repo`` the source repository whose history ``lastChanged*`` reads, ``commit`` the built commit.
    ``builtAt`` (exempt from determinism, spec3 §8.5) defaults to now. ``skipped`` is ``header.skippedFiles`` (P1-08, repo-relative
    paths, computed by ``build_bundle``): written exactly when non-empty, and each listed path is dropped from the files."""
    if not blame_mod.is_commit_id(commit):
        raise BuildError(f"the built commit {commit!r} is not a commit id (40 or 64 lowercase hex digits)")
    B = Builder(lang, unit_root, repo_id, facts, Source(Path(stage)))       # asserts CPython 3.12 first
    blame_mod.require_full_history(repo)
    frontend = python if lang == "python" else javascript
    skipped = list(skipped or [])
    skip_paths = {e["path"] for e in skipped}                # repo-relative; ``files`` is stage-relative, so compare ``B.P(f)``
    files = [f for f in lang_files(facts, lang) if B.src.exists(f) and not is_vendored(f) and B.P(f) not in skip_paths]
    fileset = set(files)
    scopes, m2s, _ = frontend.build_scopes(B)
    scopes = {k: s for k, s in scopes.items() if s.file in fileset}
    m2s = {k: s for k, s in m2s.items() if s.file in fileset}
    babel_cache = {f: javascript.babel(Path(astgen_dir), f) for f in files} if (lang == "javascript" and astgen_dir) else {}

    if lang == "python":
        python.apply_py_spans(B, scopes)
    else:
        javascript.apply_js_type_spans(B, scopes, babel_cache)

    # symbol ids (qnames are final once duplicate numbering is done)
    for s in scopes.values():
        s.sid = ids.symbol_id(lang, B.repo, s.qname)
    scope_by_sid = {s.sid: s for s in scopes.values()}
    collisions = Counter(s.sid for s in scopes.values())
    dup = sorted(k for k, v in collisions.items() if v > 1)
    if dup:
        raise BuildError(f"qname collision on {len(dup)} symbols, e.g. {scope_by_sid[dup[0]].qname}")

    records: list[dict] = []
    report = B.report

    # files
    file_ids = {}
    for f in files:
        path = B.P(f)
        data = B.src.raw(f)
        txt = B.src.text(f)
        lc = txt.count("\n") + (0 if (txt.endswith("\n") or txt == "") else 1)
        fid = ids.file_id(B.repo, path)
        file_ids[f] = fid
        records.append({"type": "file", "id": fid, "repoId": B.repo, "unitId": B.unit_id, "path": path, "lang": lang,
                        "kind": "code", "role": file_role(path), "contentHash": sha256(data), "lineCount": lc})

    # blame: from the source repository, outside the sandbox (BF12)
    blames = {f: blame_mod.blame(Path(repo), commit, B.P(f)) for f in files}

    # symbols
    cond_by_scope = defaultdict(list)
    for c in facts["c"]:
        s = m2s.get(c["m"])
        if s:
            cond_by_scope[s.sid].append(c)
    for s in scopes.values():
        ls = span_start(B, s)
        le = max(s.lineEnd, ls)
        per, times = blames[s.file]
        try:
            at, commit_id = blame_mod.last_changed(per, times, _blame_regions(s, ls, le))
        except ValueError:
            # no blamed line in the span plus owned regions (an empty file's <module>): the path's last commit (spec4 §2)
            at, commit_id = blame_mod.path_last_commit(Path(repo), commit, B.P(s.file))
            report.setdefault("blameFallbackFiles", []).append(B.P(s.file))
        rec = {"type": "symbol", "id": s.sid, "lang": lang, "repoId": B.repo, "qname": s.qname,
               "fileId": file_ids[s.file], "path": B.P(s.file), "kind": s.kind,
               "name": sym_name(s), "nameNorm": name_norm(sym_name(s)),
               "lineStart": ls, "lineEnd": le, "fingerprint": fingerprint(B, s),
               "complexity": 1 + len(cond_by_scope[s.sid]), "lastChangedAt": at, "lastChangedCommit": commit_id}
        if s.parent is not None and s.kind != "module":
            rec["parentId"] = s.parent.sid
        if s.owned:
            rec["ownedRegions"] = [list(r) for r in owned_regions(s)]
        records.append(rec)
    S = scope_by_sid

    # params
    params_by = {}
    pfacts = defaultdict(list)
    for p in facts["p"]:
        pfacts[p["m"]].append(p)
    for s in scopes.values():
        if s.kind in ("module", "type"):
            continue
        ps = sorted(pfacts.get(s.cpg_ids[0], []), key=lambda p: int(p["i"]))
        out = python.py_params(B, s, ps) if lang == "python" else javascript.js_params(ps)
        params_by[s.sid] = {}
        for index, name, p in out:
            pid = ids.point_id(s.sid, "param", index)
            params_by[s.sid][int(p["i"])] = (index, pid)
            records.append({"type": "param", "id": pid, "symbolId": s.sid, "index": index, "name": name,
                            "line": max(p["l"], 1)})

    # rets
    rets_by_scope = defaultdict(list)
    for r in facts["r"]:
        s = m2s.get(r["m"])
        if s:
            rets_by_scope[s.sid].append(r)
    ret_pid = {}
    for sid, rs in rets_by_scope.items():
        rs.sort(key=lambda r: (r["l"], r["c"], int(r["id"])))
        for k, r in enumerate(rs):
            pid = ids.point_id(sid, "ret", k)
            ret_pid[r["id"]] = pid
            records.append({"type": "ret", "id": pid, "symbolId": sid, "ordinal": k, "line": max(r["l"], 1)})
    synth_rets = 0
    for s in scopes.values():
        if s.kind == "lambda" and s.sid not in rets_by_scope:       # spec4 §2 §5.5: an expression-bodied lambda
            pid = ids.point_id(s.sid, "ret", 0)
            records.append({"type": "ret", "id": pid, "symbolId": s.sid, "ordinal": 0, "line": max(s.line, 1)})
            synth_rets += 1
    report["syntheticRetsForExpressionLambdas"] = synth_rets

    # conds
    cond_pid = {}
    for sid, cs in cond_by_scope.items():
        cs.sort(key=lambda c: (c["l"], c["c"], len(c["code"]), c["code"], int(c["id"])))
        for k, c in enumerate(cs):
            pid = ids.point_id(sid, "cond", k)
            cond_pid[c["id"]] = pid
            records.append({"type": "cond", "id": pid, "symbolId": sid, "ordinal": k, "line": max(c["l"], 1),
                            "code": mask_cond_code(collapse(mask_cond_code(c["code"]), 200))})  # mask, cut, mask again

    # call sites
    calls_by_scope = defaultdict(list)
    for c in facts["cs"]:
        s = m2s.get(c["m"])
        if s and (c["name"] or c["name"] == "<operator>.new"):
            calls_by_scope[s.sid].append(c)
    cs_info = {}
    synthetic = Counter()
    # a class instantiation (`Cls(...)`) resolves to the class's constructor; a redefined ``__init__`` is the last one
    init_of_class: dict[str, tuple[int, str]] = {}
    for s in scopes.values():
        if s.base == "__init__" and s.parent is not None and s.parent.is_type:
            if s.parent.sid not in init_of_class or (s.dup, s.sid) > init_of_class[s.parent.sid]:
                init_of_class[s.parent.sid] = (s.dup, s.sid)
    for sid, cs in calls_by_scope.items():
        s = S[sid]
        pos = locate_callees(B, s.file, cs)
        keyed = defaultdict(list)
        for c in cs:
            if c["id"] not in pos:
                synthetic[callee_name(c)] += 1
                continue
            keyed[callee_name(c)].append(c)
        for name, grp in keyed.items():
            grp.sort(key=lambda c: (pos[c["id"]], len(c["code"]), int(c["id"])))
            for k, c in enumerate(grp):
                tgt = {}
                for t in c["in"]:
                    ts = m2s.get(t["id"])
                    if ts is None:
                        continue
                    tsid = ts.sid
                    if ts.is_type:
                        if ts.sid not in init_of_class:
                            continue
                        tsid = init_of_class[ts.sid][1]
                    tgt[tsid] = S[tsid].qname
                targets = [{"symbolId": k2, "qname": v} for k2, v in sorted(tgt.items())]
                cid = ids.callsite_id(sid, name, k)
                cs_info[c["id"]] = (cid, sid, pos[c["id"]])
                kind = "new" if c["name"] == "<operator>.new" else ("super" if name == "super" else "call")
                rec = {"type": "callsite", "id": cid, "symbolId": sid, "calleeName": name, "ordinal": k,
                       "callKind": kind, "line": max(pos[c["id"]][0], 1)}
                if targets:
                    rec.update(resolution="resolved", targets=targets, status="confirmed", reason="resolved-call")
                else:
                    ext = [o for o in c["out"] if o and not o.startswith("<unknownFullName>")]
                    if ext and c["mfn"] and c["mfn"] != "<unknownFullName>" and not c["in"]:
                        rec.update(resolution="external", externalTarget=c["mfn"], targets=[], status="possible",
                                   reason="name-match")
                    else:
                        rec.update(resolution="unresolved", targets=[], status="possible", reason="name-match")
                records.append(rec)
    report["syntheticCallsDropped"] = {"total": sum(synthetic.values()), "byName": dict(synthetic.most_common(12))}

    # imports (source-derived) and constants
    imports_by_file = {}
    for f in files:
        if lang == "python":
            imports_by_file[f] = python.py_imports(B, f, [])
        else:
            ast_ = babel_cache.get(f)
            imports_by_file[f] = javascript.js_imports(B, f, ast_) if ast_ else []
    consts_names = defaultdict(set)
    const_facts = defaultdict(lambda: defaultdict(list))
    for c in facts["const"]:
        s = m2s.get(c["m"])
        if s and s.kind in ("module", "type"):
            const_facts[s.sid][c["name"]].append(c)
    const_recs = []
    for sid, d in const_facts.items():
        s = S[sid]
        for name, lst in d.items():
            if len(lst) != 1:
                continue
            q = f"{B.P(s.file)}::{name}" if s.kind == "module" else f"{s.qname}.{name}"
            consts_names[B.P(s.file)].add(q)
            const_recs.append((s, name, q, lst[0]))
    R = Resolver(B, {}, consts_names)
    for f, imps in imports_by_file.items():
        for r in imps:
            for n in r["names"]:
                if n["name"] == "*":
                    continue
                local = n.get("alias", n["name"])
                R.alias[f][local] = (r["module"], n["name"], (r.get("_subs") or {}).get(n["name"]), r.get("resolvedPath"))
    for s, name, q, c in const_recs:
        el = R.element(c["e"], s.file)
        if "unknown" in el:
            continue
        records.append({"type": "constant", "id": ids.constant_id(lang, B.repo, q), "qname": q,
                        "fileId": file_ids[s.file], "line": max(c["l"], 1), "values": [el]})

    # import records
    for f, imps in imports_by_file.items():
        for k, r in enumerate(imps):
            rec = {"type": "import", "id": ids.import_id(file_ids[f], k), "fileId": file_ids[f], "line": r["line"],
                   "ordinal": k, "module": r["module"], "names": r["names"], "dynamic": r["dynamic"]}
            if r.get("resolvedPath"):
                rec["resolvedPath"] = r["resolvedPath"]
            sid = import_symbol_id(B, scopes, f, r["line"])
            if sid:
                rec["symbolId"] = sid
            records.append(rec)

    # value sets (call-site arguments)
    vs_seen = set()
    for c in facts["cs"]:
        if c["id"] not in cs_info:
            continue
        cid, sid, _ = cs_info[c["id"]]
        s = S[sid]
        for ar in canonical_args(c["args"]):
            i = int(ar["i"])
            if i == 0 and not ar["an"]:
                continue   # the receiver: values of `recv` are not value sets worth a record
            head = kw_head(ar["an"]) if ar["an"] else f"a{i - 1}"
            path, trunc = path_from(head, ar["f"])
            if trunc:
                continue
            el = R.element(ar["e"], s.file)
            if "unknown" in el and el["unknown"] != "point-derived":
                continue
            vid = ids.value_set_id(cid, path)
            if vid in vs_seen:
                continue
            vs_seen.add(vid)
            records.append({"type": "valueSet", "id": vid, "callSiteId": cid, "arg": path, "values": [el]})

    # decorators: from each scope's own, owner-resolved ``decos`` (never by fullName, which two owners can share)
    for s in scopes.values():
        for name, d in s.decos:
            name = re.sub(r"\s+", "", name)
            args = {}
            for ar in canonical_args(d["args"]):
                i = int(ar["i"])
                head = kw_head(ar["an"]) if ar["an"] else f"a{i - 1}"
                path, trunc = path_from(head, ar["f"])
                if trunc:
                    continue
                el = R.element(ar["e"], s.file)
                if "unknown" in el and el["unknown"] != "point-derived":
                    continue
                args.setdefault(path, [el])
            records.append({"type": "decorator", "id": ids.decorator_id(s.sid, int(d["o"])), "targetId": s.sid,
                            "ordinal": int(d["o"]), "form": "decorator", "name": name, "line": max(d["l"], 1),
                            "args": [{"arg": k, "values": v} for k, v in sorted(args.items())]})

    # data
    point_of_use = {}
    for cid_cpg, (cid, sid, _) in cs_info.items():
        point_of_use[("cs", cid_cpg)] = cid
    for rid, pid in ret_pid.items():
        point_of_use[("r", rid)] = pid
    for k, pid in cond_pid.items():
        point_of_use[("c", k)] = pid
    data_seen, dropped_data = set(), Counter()
    for d in canonical_order(facts["d"]):
        s = m2s.get(d["m"])
        if s is None:
            dropped_data["no-symbol"] += 1
            continue
        use = point_of_use.get((d["uk"], d["uid"]))
        if use is None:
            dropped_data["no-use-point"] += 1
            continue
        if d["dk"] == "cs":
            defp = point_of_use.get(("cs", d["did"]))
        elif d["dk"] == "p":
            pm = params_by.get(s.sid, {}).get(int(d["di"]))
            defp = pm[1] if pm else None
        else:
            defp = None
        if defp is None:
            dropped_data["no-def-point"] += 1
            continue
        th = d["th"]
        if th == "":
            to_path, t1 = "", False
        elif th == "ret":
            to_path, t1 = path_from("ret", d["tf"])
        elif th.startswith("kw."):
            to_path, t1 = path_from(kw_head(th[3:]), d["tf"])
        else:
            to_path, t1 = path_from(th, d["tf"])
        h = d["h"]
        if h == "r":
            from_path, t2 = path_from("r", d["ff"])
        else:
            idx = params_by.get(s.sid, {}).get(int(d["di"]))[0]
            from_path, t2 = path_from("recv" if idx == -1 else f"p{idx}", d["ff"])
        did = ids.data_id(use, defp, to_path, from_path)
        if did in data_seen:                     # first wins, in canonical order
            dropped_data["duplicate"] += 1
            continue
        data_seen.add(did)
        trunc = t1 or t2
        records.append({"type": "data", "id": did, "use": use, "def": defp, "toPath": to_path, "fromPath": from_path,
                        "status": "possible" if trunc else "confirmed",
                        "reason": "field-truncated" if trunc else "intra-flow"})
    report["dataDropped"] = dict(dropped_data)

    # controlledBy: G-4 -- drop self-edges, de-duplicate (point, cond) pairs, before writing
    edges, cb_unmapped = [], 0
    for e in canonical_order(facts["cb"]):
        if m2s.get(e["m"]) is None:
            continue
        point = point_of_use.get((e["pk"], e["pid"]))
        cond = cond_pid.get(e["cond"])
        if point is None or cond is None:
            cb_unmapped += 1
            continue
        edges.append((point, cond))
    kept, cb_self, cb_dup = g4_filter(edges)
    for point, cond in kept:
        records.append({"type": "controlledBy", "id": ids.controlled_by_id(point, cond), "point": point, "cond": cond})
    report["controlledBy"] = {"rawEdges": len(edges) + cb_unmapped, "selfEdgesDropped": cb_self,
                              "duplicatePairsDropped": cb_dup, "unmapped": cb_unmapped, "written": len(kept)}
    condset = set(cond_pid.values())
    report["controlledBy"]["condToCond"] = sum(1 for (p, c) in kept if p in condset)

    # header
    counts = Counter(r["type"] for r in records)
    rev = f"{commit}^{{tree}}" if B.root == "." else f"{commit}:{B.root}"
    tree = blame_mod.git(repo, "rev-parse", rev).decode().strip()
    header = {
        "type": "header", "bundleSchemaVersion": "1.0",
        "unit": {"repoId": B.repo, "rootPath": B.root, "lang": lang}, "unitId": B.unit_id,
        "builtCommit": commit, "builtTree": tree,
        "builtAt": built_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tools": {"joern": JOERN_VERSION, "frontend": "pysrc2cpg" if lang == "python" else "jssrc2cpg",
                  "frontendVersion": JOERN_VERSION, "normalizer": NORMALIZER, "normalizerVersion": NORMALIZER_VERSION,
                  "extractScript": "extract.sc", "extractScriptHash": sha256(Path(extract_script).read_bytes())},
        "sandboxAuditIds": list(audit_ids), "egressMode": "offline", "flowProjected": True,
        "fieldDepthCap": FIELD_CAP, "vendoredDirs": sorted(DEFAULT_VENDORED), "history": "available",
        "counts": {t: counts.get(t, 0) for t in bundle.RECORD_TYPES[1:]},
    }
    if skipped:
        header["skippedFiles"] = skipped
    report["skippedFiles"] = len(skipped)
    report["counts"] = header["counts"]
    report["filesParsed"] = len(files)
    report["symbols"] = len(scopes)
    return [header] + records, report


def build_bundle(unit: Unit, raw, repo, commit: str, *, system: str, root=None, build: str | None = None,
                 built_at: str | None = None, report: dict | None = None, extract_script=None) -> Path:
    """Build and write the bundle of ``unit`` from ``raw`` (an ``extractors.joern.run.RawFacts``) at ``commit``.

    ``repo`` is the source repository (history for ``lastChanged*``). The bytes go through ``ccg.atrest.write_bundle``
    under ``bundles/<system>/`` of the work area (``root``, default ``atrest.WORK_ROOT``), as ``<unitId>.<build>``;
    ``build`` defaults to ``commit``. ``report``, when given, is filled with the measurement report; ``extract_script`` (default the shipped
    ``extract.sc``) is the file whose sha256 the header records. Returns the path."""
    from extractors.joern.run import EXTRACT_SC                    # deferred: the runner is not needed to normalize
    from extractors.joern import skipped as skipped_mod            # deferred for the same reason (it imports the runner)
    if raw.commit != commit:
        raise BuildError(f"the raw facts were extracted at {raw.commit}, not {commit}")
    facts = load_facts(raw.facts)
    skipped = []
    if unit.language == "javascript":                              # `skipped_files` is [] for any other language
        parsed = skipped_mod.parsed_map(raw.stage, unit.rootPath,
                                        [repo_relative(unit.rootPath, f) for f in lang_files(facts, "javascript")])
        skipped = skipped_mod.skipped_files(unit, parsed, override=False)
    records, rep = assemble(lang=unit.language, unit_root=unit.rootPath, repo_id=unit.repoId, facts=facts, skipped=skipped,
                            stage=raw.stage, astgen_dir=raw.astgen_dir, repo=repo, commit=commit,
                            audit_ids=raw.sandbox_audit_ids, extract_script=extract_script or EXTRACT_SC, built_at=built_at)
    if report is not None:
        report.update(rep)
    return atrest.write_bundle(system, unit.unitId, build or commit, records, root=root)
