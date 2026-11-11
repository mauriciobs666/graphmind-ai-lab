"""P1-18 assembler: bundles + links -> graph rows (no database), ``ccg.sandbox_audit``, ``ccg.staleness.FAIL_REASONS``.

Contract: ``docs/plans/connected-code-graph-p1-linking.md`` P1-18 (Behaviours (1)-(11)), ``-p1-graph.md`` §1 (G1-G13), §2 (O-9,
``BOUND_BY``), §3 (identity), §8 P1-18 (a)-(i), ``-p0-graph.md`` D5/D6/D8, ``-egress-spec3.md`` §5.3-§5.4.

Every expected value below is written out here, never read from the constant or function under test. Rows are built by hand-made
``Link``/``PackFacts`` over small bundles; the slice equivalence is ``tests/slice/test_assemble_equivalence.py``.
"""
import ast
import copy
import hashlib
import json
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

CPG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from linker_helpers import cons, fact, prov  # noqa: E402

from ccg import bundle, ids  # noqa: E402
from ccg.assemble import rows as R  # noqa: E402
from ccg.assemble import snapshot as SN  # noqa: E402
from ccg.linker import Link, Links, Via  # noqa: E402
from ccg.packrt import PackFacts  # noqa: E402
from ccg.sandbox_audit import check_sandbox_ids  # noqa: E402
from ccg.staleness import FAIL_REASONS  # noqa: E402

REPO = "repo1"
SYSTEM = "shop"


def H(*p) -> str:
    return ids.tuple_id(["t18", *p])


UUID = lambda n: f"{n:08x}-0000-4000-8000-{n:012x}"          # noqa: E731  lowercase UUIDv4
EXEMPT_ID = {"U1": [UUID(1)], "U2": [UUID(2), UUID(3)], "U3": [UUID(4)]}

# ---------------------------------------------------------------------------------------------------------------------
# the sandbox audit file
# ---------------------------------------------------------------------------------------------------------------------
FRONTENDS = ("pysrc2cpg", "jssrc2cpg", "javasrc2cpg", "jimple2cpg", "kotlin2cpg", "csharpsrc2cpg", "gosrc2cpg", "c2cpg",
             "joern-parse", "joern", "scip-typescript", "scip-python")


def sbx(audit_id: str, **over) -> dict:
    r = {"schemaVersion": 1, "recordType": "sandbox", "auditId": audit_id, "ts": "2026-11-01T17:48:12.120Z",
         "backend": "container", "toolchain": {"image": "ccg-toolchain:d3184effb566", "imageId": "sha256:" + "0" * 64},
         "job": "parse", "egressMode": "offline", "network": "none", "networkName": None,
         "launcherArgv": ["docker", "run", "--rm"], "toolArgv": ["--script", "/scripts/extract.sc"], "frontend": "joern",
         "script": {"path": "/scripts/extract.sc", "sha256": "1" * 64},
         "flagCheck": {"listDigest": "2" * 64, "forbiddenHits": [], "allowedHits": [], "liftedHits": []},
         "envKeys": ["PATH"], "mounts": [{"target": "/src", "source": "/x", "mode": "ro,noexec,nosuid,nodev"}],
         "preconditions": {"interopPE": "enabled", "binfmtPinned": False, "seccomp": "builtin", "vsock": "blocked",
                           "netnsDistinct": True},
         "refused": False, "refusalReason": None, "exitCode": 0, "codebases": [], "runId": None, "durationMs": 11202}
    r.update(over)
    return r


def audit_dir(tmp_path: Path, records=(), raw: str | None = None, name="audit") -> Path:
    d = tmp_path / name
    d.mkdir(exist_ok=True)
    text = raw if raw is not None else "".join(json.dumps(r) + "\n" for r in records)
    (d / "sandbox.jsonl").write_text(text)
    return d


def all_audit(tmp_path: Path, extra=()) -> Path:
    recs = [sbx(i) for ids_ in EXEMPT_ID.values() for i in ids_]
    return audit_dir(tmp_path, recs + list(extra))


HDR = {"sandboxAuditIds": [UUID(1)], "egressMode": "offline"}


def test_a_valid_record_passes_and_each_condition_is_its_own_error(tmp_path):
    assert check_sandbox_ids(HDR, audit_dir(tmp_path, [sbx(UUID(1))])) == []
    cases = {
        "no record": [],
        "two records": [sbx(UUID(1)), sbx(UUID(1))],
        "refused": [sbx(UUID(1), refused=True, refusalReason="usage", exitCode=64)],
        "exit 1": [sbx(UUID(1), exitCode=1)],
        "exit 64 unrefused": [sbx(UUID(1), exitCode=64)],
        "refused with exit 0": [sbx(UUID(1), refused=True, refusalReason="usage", exitCode=0)],
        "record mode controlled": [sbx(UUID(1), egressMode="controlled")],
        "extra key": [{**sbx(UUID(1)), "extra": 1}],
        "missing key": [{k: v for k, v in sbx(UUID(1)).items() if k != "durationMs"}],
        "lib-launch frontend": [sbx(UUID(1), frontend="lib-launch")],
        "other id only": [sbx(UUID(2))],
    }
    for name, recs in cases.items():
        errs = check_sandbox_ids(HDR, audit_dir(tmp_path, recs, name=name.replace(" ", "_")))
        assert len(errs) == 1 and UUID(1) in errs[0], (name, errs)


def _lang_header(lang, ids_):
    return {"sandboxAuditIds": ids_, "egressMode": "offline", "unit": {"repoId": REPO, "rootPath": ".", "lang": lang}}


def test_empty_audit_ids_pass_exactly_for_host_languages(tmp_path):
    d = audit_dir(tmp_path, [])
    for lang in ("shell", "config", "markdown"):
        assert check_sandbox_ids(_lang_header(lang, []), d) == [], lang
    for lang in ("python", "javascript", "java", "kotlin", "csharp", "go", "c", "cpp"):
        assert check_sandbox_ids(_lang_header(lang, []), d) != [], lang
    assert check_sandbox_ids({"egressMode": "offline", "unit": {"lang": "config"}}, d) != []        # absent is not empty
    assert check_sandbox_ids({**_lang_header("config", []), "egressMode": "controlled"}, d) != []
    assert check_sandbox_ids({"sandboxAuditIds": [], "egressMode": "offline"}, d) != []             # no lang: refused
    # a host bundle that does name ids is checked like any other
    assert check_sandbox_ids(_lang_header("config", [UUID(1)]), d) != []


def test_host_langs_is_the_three_host_extractor_languages():
    from ccg.sandbox_audit import HOST_LANGS
    assert sorted(HOST_LANGS) == ["config", "markdown", "shell"]


def test_a_code_bundle_with_no_audit_ids_is_refused_and_a_host_bundle_with_none_is_assembled(tmp_path):
    b = world()
    b[0][0]["sandboxAuditIds"] = []
    assert R.classify_bundle(b[0], all_audit(tmp_path)) == "bundle:audit-missing"
    with pytest.raises(R.BundleRefused) as e:
        build(tmp_path, bundles=b)
    assert e.value.code == "bundle:audit-missing"
    h = world()
    h[1][0]["sandboxAuditIds"] = []
    assert R.classify_bundle(h[1], all_audit(tmp_path)) is None
    rows = build(tmp_path, bundles=h)
    assert rows.snapshot["auditSetHash"] == hashlib.sha256("\n".join([UUID(1)]).encode()).hexdigest()


def test_header_mode_other_than_offline_is_refused_even_when_the_record_agrees(tmp_path):
    d = audit_dir(tmp_path, [sbx(UUID(1), egressMode="controlled")])
    assert check_sandbox_ids({**HDR, "egressMode": "controlled"}, d) != []
    assert check_sandbox_ids({"sandboxAuditIds": [UUID(1)]}, audit_dir(tmp_path, [sbx(UUID(1))], name="nomode")) != []


def test_only_the_failing_ids_are_named_and_no_ids_is_refused(tmp_path):
    d = audit_dir(tmp_path, [sbx(UUID(1)), sbx(UUID(2), exitCode=1)])
    errs = check_sandbox_ids({"sandboxAuditIds": [UUID(1), UUID(2), UUID(3)], "egressMode": "offline"}, d)
    assert sorted(e for e in errs if UUID(2) in e) and len(errs) == 2
    assert not any(UUID(1) in e for e in errs)
    assert check_sandbox_ids({"sandboxAuditIds": [], "egressMode": "offline"}, d) != []
    assert check_sandbox_ids({"egressMode": "offline"}, d) != []


def test_a_missing_file_a_partial_last_line_and_garbage_are_handled(tmp_path):
    assert check_sandbox_ids(HDR, tmp_path / "nowhere") != []
    good = json.dumps(sbx(UUID(1)))
    assert check_sandbox_ids(HDR, audit_dir(tmp_path, raw="not json\n" + good + "\n" + good[:40], name="p1")) == []
    assert check_sandbox_ids(HDR, audit_dir(tmp_path, raw=good[:40], name="p2")) != []
    # a record merely MENTIONING the id in an argv is not a record with that auditId
    other = sbx(UUID(2), toolArgv=["--output", f"/out/{UUID(1)}.cpg.bin"])
    assert check_sandbox_ids(HDR, audit_dir(tmp_path, [other], name="p3")) != []


def test_exit_124_is_refused_whether_or_not_the_cap_fired_and_the_message_compares_duration_with_the_cap(tmp_path):
    cap_ms = 1800 * 1000                                    # SBX_INDEXER_WALL_S, cpg/sandbox/README.md
    fired = check_sandbox_ids(HDR, audit_dir(tmp_path, [sbx(UUID(1), exitCode=124, durationMs=cap_ms + 5)], name="a"))
    tool = check_sandbox_ids(HDR, audit_dir(tmp_path, [sbx(UUID(1), exitCode=124, durationMs=1000)], name="b"))
    assert len(fired) == len(tool) == 1
    assert "124" in fired[0] and "124" in tool[0] and "durationMs" in fired[0] and "durationMs" in tool[0]
    assert "at or above" in fired[0] and "below" in tool[0] and "below" not in fired[0] and "at or above" not in tool[0]
    edge = check_sandbox_ids(HDR, audit_dir(tmp_path, [sbx(UUID(1), exitCode=124, durationMs=cap_ms - 1)], name="c"))
    exact = check_sandbox_ids(HDR, audit_dir(tmp_path, [sbx(UUID(1), exitCode=124, durationMs=cap_ms)], name="d"))
    assert "at or above" in exact[0]                                # exactly the cap counts as fired
    assert "below" in edge[0]                                       # the cap itself is the boundary: one ms short is not "fired"


def test_every_closed_frontend_validates_and_an_unknown_one_does_not(tmp_path):
    for f in FRONTENDS:
        assert check_sandbox_ids(HDR, audit_dir(tmp_path, [sbx(UUID(1), frontend=f)], name=f)) == [], f
    assert check_sandbox_ids(HDR, audit_dir(tmp_path, [sbx(UUID(1), frontend="bash")], name="bash")) != []


def test_a_namespace_toolchain_validates_and_a_mixed_shape_does_not(tmp_path):
    ns = {"joernVersion": "4.0.0", "joernPath": "/opt/joern", "javaVersion": "openjdk 21"}
    assert check_sandbox_ids(HDR, audit_dir(tmp_path, [sbx(UUID(1), backend="namespace", toolchain=ns)], name="ns")) == []
    mixed = {"image": "x", "imageId": None, "joernVersion": "4"}
    assert check_sandbox_ids(HDR, audit_dir(tmp_path, [sbx(UUID(1), toolchain=mixed)], name="mix")) != []


REAL_AUDIT = CPG / ".audit" / "sandbox.jsonl"


@pytest.mark.skipif(not REAL_AUDIT.is_file(), reason="the real sandbox.jsonl is gitignored (cpg/.audit): UNRUN, not green")
def test_the_schema_admits_every_real_record_of_the_runner():
    """Positive control: a schema that rejects what the runner writes is useless; the real file (591 records at writing)."""
    from ccg.sandbox_audit import record_errors
    bad = []
    for n, line in enumerate(REAL_AUDIT.read_text().splitlines(), 1):
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if record_errors(rec):
            bad.append(n)
    assert bad == []


# ---------------------------------------------------------------------------------------------------------------------
# FAIL_REASONS, the flow registry, step 8a
# ---------------------------------------------------------------------------------------------------------------------
def test_fail_reasons_is_the_closed_set_of_p1_graph_g4():
    assert tuple(sorted(FAIL_REASONS)) == tuple(sorted((
        "manifest:never-built", "build:shallow-clone", "extract:failed", "extract:timeout", "bundle:invalid",
        "bundle:schema-minor", "bundle:bind-key-check", "bundle:audit-missing")))
    assert len(FAIL_REASONS) == 8 == len(set(FAIL_REASONS))


def test_the_flow_registry_has_the_six_measured_languages_each_false_and_citing_the_report():
    assert R.load_flow_availability() == {"python": False, "javascript": False, "go": False, "java": False, "csharp": False,
                                          "cpp": False}                         # kotlin: unmeasured, no row (report §a2.2.6)
    rows = R.load_flow_rows()
    assert sorted(r["language"] for r in rows) == ["cpp", "csharp", "go", "java", "javascript", "python"]
    assert all("docs/test-reports/connected-code-graph-p0-frontend-fitness-report.md" in r["measurement"] and "§a2" in r["measurement"]
               for r in rows)


def test_step_8a_is_imported_from_keynorm_and_defined_nowhere_else():
    from ccg import keynorm
    assert R.mask_http_segments is keynorm.mask_http_segments
    tree = ast.parse((CPG / "ccg" / "assemble" / "rows.py").read_text())
    imports = [n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module == "ccg.keynorm"]
    assert imports and any(a.name == "mask_http_segments" for n in imports for a in n.names)
    assert R.find_step_8a_definitions(CPG) == [Path("ccg/keynorm.py")]


def test_a_planted_second_definition_of_step_8a_is_found(tmp_path):
    """Control: the sweep is not vacuous. A re-implementation under another name (screen + the placeholder) and a copy by name."""
    (tmp_path / "ccg").mkdir()
    (tmp_path / "ccg" / "keynorm.py").write_text(
        "from ccg.bundle import screen_fails\ndef mask_http_segments(p, ph):\n    return [s for s in p if screen_fails(s)] and '{#}'\n")
    (tmp_path / "ccg" / "x.py").write_text(
        "from ccg.bundle import screen_fails\ndef mine(p):\n    return ['{?}' if screen_fails(s) else s for s in p.split('/')]\n")
    (tmp_path / "ccg" / "y.py").write_text("def mask_http_segments(p, ph):\n    return p\n")
    (tmp_path / "ccg" / "ok.py").write_text("from ccg.keynorm import mask_http_segments\nx = mask_http_segments('/a', '{#}')\n")
    assert R.find_step_8a_definitions(tmp_path) == [Path("ccg/keynorm.py"), Path("ccg/x.py"), Path("ccg/y.py")]


def _non_doc_strings(tree):
    docs = {id(n.body[0].value) for n in ast.walk(tree) if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef))
            and n.body and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)}
    return [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docs]


def test_flow_available_is_read_only_from_the_registry_loader():
    """No other module names the registry file or a ``flowAvailable`` key: only the assembler writes the flag."""
    for p in sorted((CPG / "ccg").rglob("*.py")) + sorted((CPG / "clients").rglob("*.py")):
        rel = p.relative_to(CPG).as_posix()
        strings = _non_doc_strings(ast.parse(p.read_text()))
        if rel not in ("ccg/assemble/rows.py", "ccg/assemble/snapshot.py"):
            assert "flowAvailable" not in strings and not any("flow-availability" in s for s in strings), rel
    rows_strings = _non_doc_strings(ast.parse((CPG / "ccg" / "assemble" / "rows.py").read_text()))
    assert sum("flow-availability" in s for s in rows_strings) == 1            # the loader's one path


# ---------------------------------------------------------------------------------------------------------------------
# a small world
# ---------------------------------------------------------------------------------------------------------------------
def counts(**kw):
    c = {k: 0 for k in ("file", "symbol", "param", "ret", "cond", "callsite", "valueSet", "constant", "data", "controlledBy",
                        "decorator", "fieldAccess", "import", "dependency", "testMarker", "configEntry", "launchFact",
                        "schemaEntry", "secretSighting")}
    c.update(kw)
    return c


def header(unit, root, lang, *, version="1.1", skipped=(), normalizer="1", built_at="2026-11-01T10:00:00Z",
           audit=None, commit="a" * 40, tree="b" * 40, extra=None):
    h = {"type": "header", "bundleSchemaVersion": version, "unit": {"repoId": REPO, "rootPath": root, "lang": lang},
         "unitId": H("unit", unit), "builtCommit": commit, "builtTree": tree, "builtAt": built_at,
         "tools": {"joern": "4", "frontend": "pysrc2cpg", "frontendVersion": "4", "normalizer": "n", "normalizerVersion": normalizer,
                   "extractScript": "extract.sc", "extractScriptHash": "c" * 64},
         "sandboxAuditIds": list(audit or EXEMPT_ID[unit]), "egressMode": "offline", "flowProjected": True, "fieldDepthCap": 3,
         "vendoredDirs": [], "history": "available", "counts": counts()}
    if skipped:
        h["skippedFiles"] = [{"path": p, "reason": r} for p, r in skipped]
    if version == "1.1":
        h["bindKeyCheck"] = "d" * 64
    h.update(extra or {})
    return h


def file_rec(fid, path, unit="U1", lang="python", kind="code", role="primary"):
    return {"type": "file", "id": H("file", fid), "repoId": REPO, "unitId": H("unit", unit), "path": path, "lang": lang, "kind": kind,
            "role": role, "contentHash": "e" * 64, "lineCount": 100}


def sym_rec(name, fid, path, *, lang="python", parent=None, kind="function", start=1, end=9, owned=None, lang_sig=None):
    r = {"type": "symbol", "id": H("sym", name), "lang": lang, "repoId": REPO, "qname": f"{path}::{name}", "fileId": H("file", fid),
         "path": path, "kind": kind, "name": name, "nameNorm": name.lower(), "lineStart": start, "lineEnd": end,
         "fingerprint": "f" * 64, "complexity": 2}
    if parent:
        r["parentId"] = H("sym", parent)
    if owned:
        r["ownedRegions"] = owned
    return r


def call_rec(cid, owner, callee, line, targets=(), status=None, reason=None, kind="call"):
    t = [{"symbolId": H("sym", x), "qname": f"q::{x}"} for x in targets]
    return {"type": "callsite", "id": H("cs", cid), "symbolId": H("sym", owner), "calleeName": callee, "ordinal": 0, "callKind": kind,
            "line": line, "resolution": "resolved" if t else "unresolved", "targets": t,
            "status": status or ("confirmed" if t else "possible"), "reason": reason or ("resolved-call" if t else "name-match")}


def data_rec(use, dfn, status="confirmed", reason="intra-flow", to="a0", frm="r"):
    return {"type": "data", "id": H("data", use, dfn, to, frm), "use": use, "def": dfn, "toPath": to, "fromPath": frm,
            "status": status, "reason": reason}


def world(*, owned=None, skipped=(), version="1.1", normalizer="1"):
    f1, f2 = "a.py", "b.py"
    u1 = [header("U1", "svc", "python", skipped=skipped, version=version, normalizer=normalizer),
          file_rec("a", "svc/a.py"), file_rec("b", "svc/b.py"),
          sym_rec("handler", "a", "svc/a.py", start=10, end=20, owned=owned),
          sym_rec("caller", "a", "svc/a.py", start=30, end=70),
          sym_rec("inner", "a", "svc/a.py", parent="caller", start=40, end=45),
          sym_rec("work", "b", "svc/b.py", start=1, end=5),
          sym_rec("work", "a", "svc/a.py", parent="caller", start=50, end=60) | {"id": H("sym", "work2")},
          {"type": "param", "id": H("p", "h0"), "symbolId": H("sym", "handler"), "index": 0, "name": "x", "line": 10},
          {"type": "ret", "id": H("r", "c"), "symbolId": H("sym", "caller"), "ordinal": 0, "line": 69},
          {"type": "cond", "id": H("k", "c"), "symbolId": H("sym", "caller"), "ordinal": 0, "line": 33, "code": "x"},
          call_rec("res", "caller", "handler", 31, targets=["handler"]),
          call_rec("name", "caller", "work", 32),
          call_rec("dyn", "caller", "<dynamic>", 34),
          call_rec("http", "caller", "fetch", 35),
          data_rec(H("r", "c"), H("cs", "res")),
          data_rec(H("cs", "name"), H("cs", "res"), status="possible", reason="field-truncated"),
          data_rec(H("cs", "http"), H("cs", "res"), status="confirmed", reason="field-truncated", to="a1"),
          data_rec(H("cs", "dyn"), H("cs", "res"), status="possible", reason="intra-flow", to="a2"),
          {"type": "controlledBy", "id": H("cb", 1), "point": H("cs", "res"), "cond": H("k", "c")},
          {"type": "controlledBy", "id": H("cb", 2), "point": H("k", "c"), "cond": H("k", "c")},
          {"type": "dependency", "id": H("dep", 1), "depId": H("depid", 1), "ecosystem": "PyPI", "package": "requests",
           "version": "1", "scope": "runtime", "manifestPath": "svc/requirements.txt", "line": 3}]
    mounts = [sym_rec("app", "a", "svc/a.py", start=80, end=90),
              call_rec("mnt_b", "app", "include_router", 85), call_rec("mnt_a", "app", "include_router", 82),
              call_rec("mcp_mnt", "app", "mount", 88)]
    u1 += mounts
    fq = "src/c.json"
    u2 = [header("U2", ".", "config"), file_rec("cfg", "compose.yaml", unit="U2", lang="config", kind="config"),
          {"type": "configEntry", "id": H("ce", 1), "repoId": REPO, "path": "compose.yaml", "line": 3, "format": "compose",
           "keyPath": "services.web.environment.API_BASE", "valueHash": "9" * 64, "example": False,
           "bindValue": "http://web:8000", "bindParts": {"scheme": "http", "hostKind": "service", "host": "web",
                                                       "hostHmac": "8" * 64, "pathHmacs": [], "port": 8000}},
          {"type": "configEntry", "id": H("ce", 2), "repoId": REPO, "path": "compose.yaml", "line": 9,
           "format": "compose", "keyPath": "services.web.ports[0]", "valueHash": "7" * 64, "example": False},
          {"type": "launchFact", "id": H("lf", 1), "path": "compose.yaml", "line": 4, "ordinal": 0, "kind": "command",
           "command": "uvicorn", "args": ["svc.a:app"], "origin": "explicit"},
          {"type": "launchFact", "id": H("lf", 2), "path": "compose.yaml", "line": 5, "ordinal": 1, "kind": "port", "port": 8000,
           "protocol": "http", "origin": "compose"},
          {"type": "launchFact", "id": H("lf", 3), "path": "compose.yaml", "line": 6, "ordinal": 2, "kind": "env-default",
           "envVar": "PORT", "defaultHash": "6" * 64, "default": "8000", "origin": "env-default"},
          {"type": "launchFact", "id": H("lf", 4), "path": "compose.yaml", "line": 7, "ordinal": 3, "kind": "mount",
           "mountPath": "/data", "service": "web", "origin": "compose"}]
    return [u1, u2]


def pack_dir(tmp_path: Path, pid="fastapi", body="packFormat: 1\n") -> Path:
    d = tmp_path / "packs" / pid
    d.mkdir(parents=True, exist_ok=True)
    (d / "pack.yaml").write_text(body)
    return d


def pack_def(pid="fastapi", version="0.1.0", kinds=("mcp_tool",), vocab="1.0"):
    return {"packFormat": 1, "id": pid, "version": version, "kind": "framework", "vocabVersion": vocab,
            "urlBindings": [{"id": "b", "kinds": list(kinds), "urlSources": []}]}


def links_of(*ls, mismatches=()):
    return Links(list(ls), list(mismatches), {})


def build(tmp_path, bundles=None, links=None, facts=None, **kw):
    bundles = world() if bundles is None else bundles
    if "audit_dir" not in kw:
        kw["audit_dir"] = all_audit(tmp_path)
    if "packs" not in kw:
        kw["packs"] = [pack_def()]
    if "pack_dirs" not in kw:
        kw["pack_dirs"] = {"fastapi": pack_dir(tmp_path)}
    kw.setdefault("system", SYSTEM)
    return R.build_rows(bundles, links if links is not None else Links([], [], {}),
                        facts if facts is not None else PackFacts([], [], [], [], [], [], []), **kw)


def by(rows, label, key):
    return {r[key]: r for r in rows.nodes.get(label, [])}


def edges(rows, src, typ, dst):
    return rows.edges.get((src, typ, dst), [])


def test_the_world_counts_every_node_and_edge_group(tmp_path):
    rows = build(tmp_path)
    assert {k: len(v) for k, v in rows.nodes.items()} == {
        "Repo": 1, "Unit": 2, "File": 3, "Symbol": 6, "Param": 1, "Ret": 1, "Cond": 1, "CallSite": 7, "Name": 5, "Dependency": 1,
        "ConfigEntry": 6, "Pack": 1}
    e = {k: len(v) for k, v in rows.edges.items()}
    assert e[("CallSite", "CALLS", "Symbol")] == 1
    assert e[("CallSite", "CALLS_NAME", "Name")] == 6               # every call site but the <dynamic> one
    assert e[("Name", "RESOLVES_TO", "Symbol")] == 3                # handler -> 1, work -> 2 (both symbols named work)
    assert e[("Symbol", "HAS_CALL", "CallSite")] == 7 and e[("Unit", "DEPENDS_ON", "Dependency")] == 1
    assert ("Cond", "CONTROLLED_BY", "Cond") not in e                # the cond -> cond self-edge is dropped (G-4)
    assert rows.dropped == {"controlled-by-self-edge": 1}
    assert rows.node_count() == 35                  # the sum of the table above; the Snapshot node is separate


def test_calls_name_pin_a_site_without_targets_has_calls_name_and_resolves_to_every_symbol_of_that_name(tmp_path):
    """P1-15 pin: no targets and calleeName X -> CALLS_NAME to Name(X) and RESOLVES_TO each symbol named X; <dynamic> gets none."""
    rows = build(tmp_path)
    work = ids.name_key("python", "work")
    cs_name = H("cs", "name")
    assert {"s": cs_name, "t": work} in edges(rows, "CallSite", "CALLS_NAME", "Name")
    assert sorted(r["t"] for r in edges(rows, "Name", "RESOLVES_TO", "Symbol") if r["s"] == work) == sorted([H("sym", "work"), H("sym", "work2")])
    assert not [r for r in edges(rows, "CallSite", "CALLS_NAME", "Name") if r["s"] == H("cs", "dyn")]
    assert ids.name_key("python", "<dynamic>") not in by(rows, "Name", "nameKey")
    assert [r for r in edges(rows, "CallSite", "CALLS", "Symbol") if r["s"] == cs_name] == []
    assert b"resolution-unverified" not in R.canonical_rows(rows)       # a read-time label (`name_in`'s count(x) > 0), never stored


def test_d8_calls_and_data_store_status_and_reason_only_when_not_the_default(tmp_path):
    b = world()
    b[0].append(call_rec("odd", "caller", "handler", 36, targets=["handler"], status="possible", reason="name-match"))
    b[0].append(call_rec("odd2", "caller", "handler", 37, targets=["handler"], status="confirmed", reason="verdict-confirmed"))
    rows = build(tmp_path, bundles=b)
    calls = {r["s"]: r.get("p") for r in edges(rows, "CallSite", "CALLS", "Symbol")}
    assert calls[H("cs", "res")] is None                                   # default: no props at all
    assert calls[H("cs", "odd")] == {"status": "possible", "reason": "name-match"}
    assert calls[H("cs", "odd2")] == {"reason": "verdict-confirmed"}      # only the field that differs
    data = {}
    for g in (("Ret", "DATA", "CallSite"), ("CallSite", "DATA", "CallSite")):
        data |= {(r["s"], r["t"]): r["p"] for r in edges(rows, *g)}
    assert data[(H("r", "c"), H("cs", "res"))] == {"fromPath": "r", "toPath": "a0"}
    assert data[(H("cs", "name"), H("cs", "res"))] == {"fromPath": "r", "toPath": "a0", "status": "possible", "reason": "field-truncated"}
    assert data[(H("cs", "http"), H("cs", "res"))] == {"fromPath": "r", "toPath": "a1", "reason": "field-truncated"}
    assert data[(H("cs", "dyn"), H("cs", "res"))] == {"fromPath": "r", "toPath": "a2", "status": "possible"}


def test_symbol_depth_owned_regions_and_the_stripped_props(tmp_path):
    rows = build(tmp_path, bundles=world(owned=[[12, 14], [16, 18]]))
    s = by(rows, "Symbol", "symbolId")
    assert s[H("sym", "caller")]["depth"] == 0 and s[H("sym", "inner")]["depth"] == 1
    assert s[H("sym", "handler")]["ownedRegions"] == [12, 14, 16, 18]
    assert "ownedRegions" not in s[H("sym", "caller")] and "parentId" not in s[H("sym", "inner")]


def test_skipped_files_are_nodes_under_skips_with_no_file_edges(tmp_path):
    rows = build(tmp_path, bundles=world(skipped=[("svc/t_a.py", "test"), ("svc/x.ts", "frontend-ignore")]))
    sk = rows.nodes["SkippedFile"]
    assert sk == [{"repoId": REPO, "unitId": H("unit", "U1"), "path": "svc/t_a.py", "reason": "test"},
                  {"repoId": REPO, "unitId": H("unit", "U1"), "path": "svc/x.ts", "reason": "frontend-ignore"}]
    assert edges(rows, "Unit", "SKIPS", "SkippedFile") == [{"s": H("unit", "U1"), "t": "svc/t_a.py"}, {"s": H("unit", "U1"), "t": "svc/x.ts"}]
    assert not [g for g in rows.edges if "SkippedFile" in g and g[1] != "SKIPS"]
    assert "SkippedFile" not in build(tmp_path).nodes


# ---------------------------------------------------------------------------------------------------------------------
# ConfigEntry (G1, G2)
# ---------------------------------------------------------------------------------------------------------------------
FORBIDDEN_KEYS = {"args", "default", "defaultHash", "bindValue", "bindParts", "format", "example", "bindKeyCheck", "origin", "ordinal",
                  "candidateCount", "paramMap", "hostHmac", "pathHmacs"}


def all_keys(rows) -> set:
    ks = set()
    for v in rows.nodes.values():
        for r in v:
            ks |= set(r)
    for v in rows.edges.values():
        for r in v:
            ks |= set(r.get("p", {}))
    ks |= set(rows.snapshot)
    return ks


def test_config_entry_rows_carry_exactly_one_of_keypath_and_launchkind_and_no_forbidden_key(tmp_path):
    rows = build(tmp_path)
    ce = by(rows, "ConfigEntry", "entryId")
    for r in ce.values():
        assert ("keyPath" in r) != ("launchKind" in r), r
        assert r["unitId"] == H("unit", "U2") and r["repoId"] == REPO and r["path"] == "compose.yaml"
    assert ce[H("ce", 1)] == {"entryId": H("ce", 1), "repoId": REPO, "unitId": H("unit", "U2"), "path": "compose.yaml", "line": 3,
                              "keyPath": "services.web.environment.API_BASE", "valueHash": "9" * 64}
    assert ce[H("lf", 1)]["argv0"] == "uvicorn" and ce[H("lf", 1)]["launchKind"] == "command" and "args" not in ce[H("lf", 1)]
    assert (ce[H("lf", 2)]["port"], ce[H("lf", 2)]["protocol"]) == (8000, "http")
    assert ce[H("lf", 3)]["envVar"] == "PORT" and "default" not in ce[H("lf", 3)] and "defaultHash" not in ce[H("lf", 3)]
    assert (ce[H("lf", 4)]["mountPath"], ce[H("lf", 4)]["service"]) == ("/data", "web")
    assert all_keys(rows).isdisjoint(FORBIDDEN_KEYS)


def test_keypath_segments_failing_the_screen_are_stored_masked_and_the_entry_id_is_the_records(tmp_path):
    secret = "AKIAIOSFODNN7EXAMPLE"
    assert bundle.screen_fails(secret)                                           # the premise
    b = world()
    b[1].append({"type": "configEntry", "id": H("ce", 3), "repoId": REPO, "path": "compose.yaml", "line": 11, "format": "compose",
                 "keyPath": f"services.web.environment.{secret}", "valueHash": "5" * 64, "example": False})
    rows = build(tmp_path, bundles=b)
    r = by(rows, "ConfigEntry", "entryId")[H("ce", 3)]
    assert r["keyPath"] == "services.web.environment.[#…]" and secret.encode() not in R.canonical_rows(rows)


# ---------------------------------------------------------------------------------------------------------------------
# contract points, links, evidence (G6-G9, G12, O-9)
# ---------------------------------------------------------------------------------------------------------------------
HANDLER, CALLER = H("sym", "handler"), H("sym", "caller")
SITE_A, SITE_B = H("cs", "res"), H("cs", "http")


def via(site, status="confirmed", reason="key-match", key="GET /items/{}", line=31):
    return Via(site, line, status, reason, key)


def lifted(site, ordn, kind="callsite", ident=None, line=31, **kw):
    r = {"edge": "LIFTED_VIA", "siteId": site, "kind": kind, "id": ident or site, "path": "svc/a.py", "line": line,
         "urlPath": "a0", "verbPath": None, "ord": ordn, "callee": "ignored", "qname": "ignored"}
    r.update(kw)
    return r


def bound(site, ordn, entry, **kw):
    return {"edge": "BOUND_BY", "siteId": site, "ord": ordn, "entryId": entry} | kw


def mk_link(lid="L1", status="confirmed", reason="key-match", sites=None, evidence=(), provider=HANDLER, key="GET /items/{}",
            to_unit=None, kind="http", param_map=None, matcher="exact"):
    return Link(H("link", lid), kind, key, provider, CALLER, status, reason, matcher, param_map or {}, to_unit,
                tuple(sites or [via(SITE_A)]), tuple(evidence))


def http_facts():
    return PackFacts([prov(HANDLER, "http", "GET /items/{}", site="decl", line=10)],
                     [cons(CALLER, "http", "GET /items/{}", SITE_A, status="confirmed", reason="lifted-role")], [], [], [], [], [])


def test_contract_points_links_and_their_edges(tmp_path):
    ev = (lifted(SITE_A, 0), lifted(SITE_A, 1, kind="symbol", ident=CALLER, line=30, viaSiteId=SITE_A, viaLine=31, verbPath="p0.m"))
    rows = build(tmp_path, links=links_of(mk_link(evidence=ev)), facts=http_facts())
    cp = ids.contract_point_id("http", "GET /items/{}")
    assert rows.nodes["ContractPoint"] == [{"cpId": cp, "kind": "http", "key": "GET /items/{}", "keyNorm": "GET /items/{}"}]
    assert rows.nodes["Link"] == [{"linkId": H("link", "L1"), "kind": "http", "status": "confirmed", "reason": "key-match", "matcher": "exact"}]
    assert edges(rows, "Symbol", "PROVIDES", "ContractPoint") == [{"s": HANDLER, "t": cp, "p": {"status": "confirmed", "reason": "key-match"}}]
    assert edges(rows, "CallSite", "CONSUMES", "ContractPoint") == [{"s": SITE_A, "t": cp, "p": {"status": "confirmed", "reason": "lifted-role"}}]
    assert edges(rows, "Link", "TO", "Symbol") == [{"s": H("link", "L1"), "t": HANDLER}]
    assert edges(rows, "Link", "ON", "ContractPoint") == [{"s": H("link", "L1"), "t": cp}]
    assert edges(rows, "CallSite", "VIA", "Link") == [{"s": SITE_A, "t": H("link", "L1"),
                                                       "p": {"status": "confirmed", "reason": "key-match", "consumerKey": "GET /items/{}"}}]
    assert edges(rows, "Link", "LIFTED_VIA", "CallSite") == [{"s": H("link", "L1"), "t": SITE_A,
                                                              "p": {"ord": 0, "siteId": SITE_A, "line": 31, "urlPath": "a0"}}]
    assert edges(rows, "Link", "LIFTED_VIA", "Symbol") == [{"s": H("link", "L1"), "t": CALLER, "p": {
        "ord": 1, "siteId": SITE_A, "line": 30, "urlPath": "a0", "verbPath": "p0.m", "viaSiteId": SITE_A, "viaLine": 31}}]
    assert "candidateCount" not in all_keys(rows) and "paramMap" not in all_keys(rows)


def test_candidate_count_is_not_stored_and_a_param_map_is_counted_as_dropped(tmp_path):
    rows = build(tmp_path, links=links_of(mk_link(param_map={"a": "b"}), mk_link("L2", sites=[via(SITE_A, key=None)])), facts=http_facts())
    assert rows.dropped["link-param-map"] == 1
    assert set(rows.nodes["Link"][0]) == {"linkId", "kind", "status", "reason", "matcher"}
    assert "candidateCount" not in all_keys(rows)
    l2 = [r for r in edges(rows, "CallSite", "VIA", "Link") if r["t"] == H("link", "L2")]
    assert [r["p"] for r in l2] == [{"status": "confirmed", "reason": "key-match"}]                 # an absent consumer key is absent


def test_two_facts_for_one_provider_and_key_write_one_edge_with_the_strongest_status(tmp_path):
    k = "GET /items/{}"
    ps = [prov(HANDLER, "http", k, site="s2", status="possible", reason="partial-key"), prov(HANDLER, "http", k, site="s1")]
    cs = [cons(CALLER, "http", k, SITE_A, status="possible", reason="partial-key"),
          cons(CALLER, "http", k, SITE_A, status="confirmed", reason="key-match")]
    for order in (ps, ps[::-1]):
        rows = build(tmp_path, facts=PackFacts(order, cs, [], [], [], [], []))
        assert [r["p"] for r in edges(rows, "Symbol", "PROVIDES", "ContractPoint")] == [{"status": "confirmed", "reason": "key-match"}]
        assert [r["p"] for r in edges(rows, "CallSite", "CONSUMES", "ContractPoint")] == [{"status": "confirmed", "reason": "key-match"}]


def test_one_site_reaching_a_link_under_two_consumer_keys_has_two_parallel_via_edges(tmp_path):
    lk = mk_link(sites=[via(SITE_A, key="GET /items/{}"), via(SITE_A, "possible", "partial-key", key="GET /items/{?}")])
    lk = lk._replace(status="confirmed", reason="key-match")
    rows = build(tmp_path, links=links_of(lk), facts=http_facts())
    assert sorted(r["p"]["consumerKey"] for r in edges(rows, "CallSite", "VIA", "Link")) == ["GET /items/{?}", "GET /items/{}"]


def test_to_unit_only_when_the_link_names_a_never_built_unit(tmp_path):
    nm = R.NotMappedUnit(H("unit", "NB"), REPO, "nb", "python")
    into = mk_link("N1", status="possible", reason="target-not-mapped", provider=None, to_unit=H("unit", "NB"),
                   sites=[via(SITE_A, "possible", "target-not-mapped")])
    outside = mk_link("N2", status="possible", reason="target-not-mapped", provider=None, to_unit=None,
                      sites=[via(SITE_A, "possible", "target-not-mapped")])
    rows = build(tmp_path, links=links_of(into, outside), facts=http_facts(), not_mapped=[nm])
    assert edges(rows, "Link", "TO_UNIT", "Unit") == [{"s": H("link", "N1"), "t": H("unit", "NB")}]
    assert edges(rows, "Link", "TO", "Symbol") == []
    assert {r["linkId"] for r in rows.nodes["Link"]} == {H("link", "N1"), H("link", "N2")}
    u = by(rows, "Unit", "unitId")[H("unit", "NB")]
    assert u == {"unitId": H("unit", "NB"), "repoId": REPO, "rootPath": "nb", "language": "python", "buildStatus": "not-mapped",
                 "flowAvailable": False, "failReason": "manifest:never-built"}
    with pytest.raises(R.AssembleError):                           # a TO_UNIT whose Unit row does not exist cannot load
        build(tmp_path, links=links_of(into), facts=http_facts())


def test_bound_by_one_edge_per_link_site_entry_with_port_evidence_and_a_loaded_pack(tmp_path):
    ev = (lifted(SITE_A, 0), lifted(SITE_B, 0, ident=SITE_B, line=35),
          bound(SITE_B, 0, H("ce", 1)), bound(SITE_B, 1, H("lf", 1), port=8000, portOrigin="pack-default", packId="fastapi"),
          bound(SITE_B, 2, H("lf", 2), port=8000, portOrigin="compose"))
    lk = mk_link(status="confirmed", reason="key-match", sites=[via(SITE_A), via(SITE_B, "possible", "port-shared", line=35)],
                 evidence=ev)
    rows = build(tmp_path, links=links_of(lk), facts=http_facts())
    b = edges(rows, "Link", "BOUND_BY", "ConfigEntry")
    assert [(r["t"], r["p"]) for r in b] == [
        (H("ce", 1), {"siteId": SITE_B, "ord": 0}),
        (H("lf", 1), {"siteId": SITE_B, "ord": 1, "port": 8000, "portOrigin": "pack-default", "packId": "fastapi"}),
        (H("lf", 2), {"siteId": SITE_B, "ord": 2, "port": 8000, "portOrigin": "compose"})]
    assert {p["packId"] for p in rows.nodes["Pack"]} >= {"fastapi"}


@pytest.mark.parametrize("bad", [
    {"packId": "ghost", "port": 1, "portOrigin": "pack-default"},         # not a loaded Pack
    {"packId": "fastapi", "port": 1, "portOrigin": "compose"},            # packId iff pack-default
    {"port": 1, "portOrigin": "pack-default"},                            # pack-default needs packId
    {"port": 70000, "portOrigin": "explicit"}, {"port": 0, "portOrigin": "explicit"},
    {"port": 1, "portOrigin": "guessed"}, {"portOrigin": "explicit"}, {"port": 1}])
def test_bound_by_port_evidence_is_validated(tmp_path, bad):
    ev = (lifted(SITE_A, 0), bound(SITE_A, 0, H("ce", 1), **bad))
    with pytest.raises(R.AssembleError):
        build(tmp_path, links=links_of(mk_link(sites=[via(SITE_A, reason="binding-confirmed")], reason="binding-confirmed", evidence=ev)),
              facts=http_facts())


def test_bound_by_siteid_must_be_a_via_site_of_its_link_and_one_edge_per_link_site_entry(tmp_path):
    ghost = "f" * 32
    for ev in ((bound(ghost, 0, H("ce", 1)),),                                           # BOUND_BY at a site that is not a VIA site
               (lifted(ghost, 0),)):                                                      # LIFTED_VIA likewise
        with pytest.raises(R.AssembleError, match="not a VIA site"):                      # the unbound key-match link is otherwise valid
            build(tmp_path, links=links_of(mk_link(evidence=ev)), facts=http_facts())
    dup = (bound(SITE_A, 0, H("ce", 1)), bound(SITE_A, 1, H("ce", 1)))                   # a duplicate (link, site, entry)
    with pytest.raises(R.AssembleError, match="two BOUND_BY"):
        build(tmp_path, links=links_of(mk_link(sites=[via(SITE_A, reason="binding-confirmed")], reason="binding-confirmed",
                                               evidence=dup)), facts=http_facts())


def o9_rows(rows):
    """Independent restatement of ``-p1-graph.md`` §2 over stored rows (the invariant, not the assembler's code)."""
    bound_sites = {}
    for r in edges(rows, "Link", "BOUND_BY", "ConfigEntry"):
        bound_sites.setdefault(r["s"], set()).add(r["p"]["siteId"])
    best = {}
    for r in edges(rows, "CallSite", "VIA", "Link"):
        p = r["p"]
        is_bound = p["status"] == "confirmed" and r["s"] in bound_sites.get(r["t"], set())
        assert (p["reason"] == "binding-confirmed") == is_bound, r
        best.setdefault(r["t"], []).append((p["status"], p["reason"]))
    for lk in rows.nodes.get("Link", []):
        vias = best[lk["linkId"]]
        assert (lk["reason"] == "binding-confirmed") == any(s == "confirmed" and why == "binding-confirmed" for s, why in vias)
        assert lk["status"] == ("confirmed" if any(s == "confirmed" for s, _ in vias) else "possible")


def test_o9_the_planted_case_site_a_unbound_confirmed_site_b_possible_with_a_chain(tmp_path):
    ev = (bound(SITE_B, 0, H("ce", 1)),)
    lk = mk_link(status="confirmed", reason="key-match", sites=[via(SITE_A), via(SITE_B, "possible", "port-shared", line=35)], evidence=ev)
    rows = build(tmp_path, links=links_of(lk), facts=http_facts())
    o9_rows(rows)
    assert rows.nodes["Link"][0]["reason"] == "key-match"
    ok = mk_link("L2", status="confirmed", reason="binding-confirmed", sites=[via(SITE_A, reason="binding-confirmed")],
                 evidence=(bound(SITE_A, 0, H("ce", 1)),))
    o9_rows(build(tmp_path, links=links_of(ok), facts=http_facts()))


@pytest.mark.parametrize("name,lk", [
    ("via binding-confirmed on a site whose chain binds the other site",
     mk_link(sites=[via(SITE_A, reason="binding-confirmed"), via(SITE_B, "possible", "port-shared", line=35)],
             reason="binding-confirmed", evidence=(bound(SITE_B, 0, H("ce", 1)),))),
    ("link reason derived from any BOUND_BY",
     mk_link(reason="binding-confirmed", sites=[via(SITE_A), via(SITE_B, "possible", "port-shared", line=35)],
             evidence=(bound(SITE_B, 0, H("ce", 1)),))),
    ("confirmed bound site that keeps key-match",
     mk_link(sites=[via(SITE_A)], evidence=(bound(SITE_A, 0, H("ce", 1)),))),
    ("possible site claiming binding-confirmed",
     mk_link(status="possible", reason="binding-confirmed", sites=[via(SITE_A, "possible", "binding-confirmed")],
             evidence=(bound(SITE_A, 0, H("ce", 1)),))),
    ("link confirmed with only possible sites",
     mk_link(status="confirmed", reason="port-shared", sites=[via(SITE_A, "possible", "port-shared")])),
    ("link possible with a confirmed site",
     mk_link(status="possible", reason="key-match", sites=[via(SITE_A)])),
    ("link reason is not the best site's reason",
     mk_link(status="possible", reason="partial-key", sites=[via(SITE_A, "possible", "port-shared")]))])
def test_o9_violations_are_refused(tmp_path, name, lk):
    with pytest.raises(R.AssembleError):
        build(tmp_path, links=links_of(lk), facts=http_facts())


def test_bind_parts_binding_and_clear_binding_give_identical_rows(tmp_path):
    """-p1-graph.md §8 (h): the entry's ``bindParts`` (or its clear ``bindValue``) never reaches a row."""
    ev = (lifted(SITE_A, 0), bound(SITE_A, 0, H("ce", 1), port=8000, portOrigin="explicit"))
    lk = mk_link(reason="binding-confirmed", sites=[via(SITE_A, reason="binding-confirmed")], evidence=ev)
    a = build(tmp_path, links=links_of(lk), facts=http_facts())
    b = world()
    ce = next(r for r in b[1] if r["type"] == "configEntry" and r["id"] == H("ce", 1))
    ce.pop("bindParts")
    c = build(tmp_path, bundles=b, links=links_of(lk), facts=http_facts())
    rows_of = lambda r: json.dumps([r.nodes, {"|".join(k): v for k, v in r.edges.items()}, r.dropped], sort_keys=True)   # noqa: E731
    assert rows_of(a) == rows_of(c) and "bindParts" not in rows_of(a) and "hostHmac" not in rows_of(a)
    assert a.snapshot["manifestHash"] != c.snapshot["manifestHash"]          # the bundle hash does cover bindParts (§3), the rows do not


# ---------------------------------------------------------------------------------------------------------------------
# MOUNTED_BY (G6)
# ---------------------------------------------------------------------------------------------------------------------
def mount_fact(site, prefixes, line, pack="fastapi"):
    return {"packId": pack, "mountId": "include-router", "siteId": H("cs", site), "symbolId": H("sym", "app"), "path": "svc/a.py",
            "line": line, "factoryIds": [], "prefixes": prefixes, "conditions": [], "status": "confirmed", "reason": "key-match"}


def mounted_provider(key, *sites):
    ms = [{"how": "mounted", "sites": [{"siteId": H("cs", s), "path": "svc/a.py", "line": 1}], "conditions": []} for s in sites]
    return prov(HANDLER, "http", key, site="decl", evidence={"mounts": ms}) | {"packId": "fastapi"}


def test_mounted_by_ord_is_by_mount_path_then_line_and_ids_are_symbol_and_call_site(tmp_path):
    # the path order and the line order disagree: /alpha is mounted at the later line
    facts = PackFacts([mounted_provider("GET /items/{}", "mnt_b", "mnt_a")], [],
                      [mount_fact("mnt_b", ["/zeta"], 82), mount_fact("mnt_a", ["/alpha"], 90)], [], [], [], [])
    rows = build(tmp_path, facts=facts)
    got = [(r["s"], r["t"], r["p"]) for r in edges(rows, "Symbol", "MOUNTED_BY", "CallSite")]
    assert sorted(got, key=lambda g: g[2]["ord"]) == [(HANDLER, H("cs", "mnt_a"), {"ord": 0, "mountPath": "/alpha"}),
                                                       (HANDLER, H("cs", "mnt_b"), {"ord": 1, "mountPath": "/zeta"})]


def test_a_handler_mounted_through_a_two_router_chain_gets_an_edge_to_each_mount_call_site(tmp_path):
    chain = {"how": "mounted", "conditions": [], "sites": [{"siteId": H("cs", "mnt_a"), "path": "svc/a.py", "line": 82},
                                                           {"siteId": H("cs", "mnt_b"), "path": "svc/a.py", "line": 85}]}
    p = prov(HANDLER, "http", "GET /items/{}", site="decl", evidence={"mounts": [chain]}) | {"packId": "fastapi"}
    facts = PackFacts([p], [], [mount_fact("mnt_a", ["/api"], 82), mount_fact("mnt_b", ["/v1"], 85)], [], [], [], [])
    rows = build(tmp_path, facts=facts)
    assert sorted((r["t"], r["p"]["mountPath"]) for r in edges(rows, "Symbol", "MOUNTED_BY", "CallSite")) == sorted(
        [(H("cs", "mnt_a"), "/api"), (H("cs", "mnt_b"), "/v1")])


def test_mounted_by_line_breaks_a_tie_on_mount_path_and_an_empty_prefix_is_the_root(tmp_path):
    facts = PackFacts([mounted_provider("GET /items/{}", "mnt_b", "mnt_a")], [],
                      [mount_fact("mnt_b", [""], 85), mount_fact("mnt_a", [""], 82)], [], [], [], [])
    rows = build(tmp_path, facts=facts)
    got = {r["t"]: r["p"] for r in edges(rows, "Symbol", "MOUNTED_BY", "CallSite")}
    assert got == {H("cs", "mnt_a"): {"ord": 0, "mountPath": "/"}, H("cs", "mnt_b"): {"ord": 1, "mountPath": "/"}}


def test_mount_path_goes_through_the_keynorm_screen(tmp_path):
    secret = "AKIAIOSFODNN7EXAMPLE"
    assert bundle.screen_fails(secret)
    facts = PackFacts([mounted_provider("GET /items/{}", "mnt_a")], [], [mount_fact("mnt_a", [f"/shop/{secret}"], 82)], [], [], [], [])
    rows = build(tmp_path, facts=facts)
    assert [r["p"]["mountPath"] for r in edges(rows, "Symbol", "MOUNTED_BY", "CallSite")] == ["/shop/{#}"]


def test_unit_level_mount_serves_the_providers_of_the_units_whose_kinds_the_pack_binds(tmp_path):
    mcp = prov(H("sym", "work"), "mcp_tool", "work", site="decl2") | {"packId": "fastapi"}
    other_unit = prov(H("sym", "work"), "other_kind", "k", site="decl3") | {"packId": "fastapi"}
    mp = {"fact": "mountPath", "packId": "fastapi", "status": "confirmed", "reason": None, "unitId": H("unit", "U1"), "path": "/mcp",
          "pathMatch": "prefix", "siteId": H("cs", "mcp_mnt"), "line": 88, "conditions": []}
    facts = PackFacts([mcp, other_unit], [], [], [], [mp], [], [])
    rows = build(tmp_path, facts=facts)
    assert edges(rows, "Symbol", "MOUNTED_BY", "CallSite") == [
        {"s": H("sym", "work"), "t": H("cs", "mcp_mnt"), "p": {"ord": 0, "mountPath": "/mcp"}}]


def test_a_mount_whose_site_is_missing_is_counted_not_written(tmp_path):
    facts = PackFacts([mounted_provider("GET /items/{}", "ghost")], [], [mount_fact("ghost", ["/x"], 1)], [], [], [], [])
    rows = build(tmp_path, facts=facts)
    assert edges(rows, "Symbol", "MOUNTED_BY", "CallSite") == [] and rows.dropped.get("mount-site-missing") == 1


# ---------------------------------------------------------------------------------------------------------------------
# Unit rows, flowAvailable, failReason
# ---------------------------------------------------------------------------------------------------------------------
def test_unit_rows_carry_the_header_fields_and_flow_available_from_the_registry_only(tmp_path):
    reg = tmp_path / "flow.yaml"
    reg.write_text("- {language: python, flowAvailable: true, measurement: 'docs/x.md §1'}\n")
    rows = build(tmp_path, flow_registry=reg)
    u = by(rows, "Unit", "unitId")
    assert u[H("unit", "U1")] == {"unitId": H("unit", "U1"), "repoId": REPO, "rootPath": "svc", "language": "python",
                                  "buildStatus": "fresh", "flowAvailable": True, "builtCommit": "a" * 40, "builtTree": "b" * 40,
                                  "builtAt": "2026-11-01T10:00:00Z", "bundleSchemaVersion": "1.1"}
    assert u[H("unit", "U2")]["flowAvailable"] is False            # `config` has no row
    assert all(isinstance(x["flowAvailable"], bool) for x in u.values())
    b = world()
    b[0][0]["flowProjected"] = False                                # the bundle's own flag decides nothing
    again = build(tmp_path, bundles=b, flow_registry=reg)
    assert by(again, "Unit", "unitId")[H("unit", "U1")]["flowAvailable"] is True
    assert by(build(tmp_path), "Unit", "unitId")[H("unit", "U1")]["flowAvailable"] is False      # the shipped registry: python false


def test_a_bad_flow_registry_is_refused(tmp_path):
    for body in ("- {language: python, flowAvailable: 'yes', measurement: m}\n",
                 "- {language: python, flowAvailable: true}\n",
                 "- {language: python, flowAvailable: true, measurement: m}\n- {language: python, flowAvailable: false, measurement: m}\n",
                 "- {language: cobol, flowAvailable: true, measurement: m}\n"):
        p = tmp_path / "bad.yaml"
        p.write_text(body)
        with pytest.raises(R.AssembleError):
            R.load_flow_availability(p)


def test_stale_kept_unit_carries_the_last_good_bundles_fields_plus_failed_at_and_reason(tmp_path):
    rows = build(tmp_path, stale={H("unit", "U1"): ("2026-11-02T01:02:03Z", "extract:timeout")})
    u = by(rows, "Unit", "unitId")[H("unit", "U1")]
    assert u["buildStatus"] == "stale-kept" and u["failedAt"] == "2026-11-02T01:02:03Z" and u["failReason"] == "extract:timeout"
    assert u["builtCommit"] == "a" * 40 and u["bundleSchemaVersion"] == "1.1"
    assert "failedAt" not in by(rows, "Unit", "unitId")[H("unit", "U2")]


@pytest.mark.parametrize("reason", ["extract:failed: boom", "SECRET-VALUE", "", "extract", None, "bundle:invalid ", "extract:timeout\n"])
def test_a_fail_reason_outside_the_closed_set_is_refused(tmp_path, reason):
    with pytest.raises(R.AssembleError) as e:
        build(tmp_path, stale={H("unit", "U1"): ("2026-11-02T01:02:03Z", reason)})
    assert "SECRET-VALUE" not in str(e.value)
    with pytest.raises(R.AssembleError):
        build(tmp_path, not_mapped=[R.NotMappedUnit(H("unit", "NB"), REPO, "nb", "python", fail_reason=reason)])


def test_failed_at_must_be_rfc3339_utc(tmp_path):
    for bad in ("2026-11-02", "2026-11-02T01:02:03+00:00", "yesterday", None):
        with pytest.raises(R.AssembleError):
            build(tmp_path, stale={H("unit", "U1"): (bad, "extract:failed")})


def test_a_failing_bundle_with_a_planted_bind_value_and_exception_yields_a_value_free_code(tmp_path):
    """-p1-graph.md §8 P1-18 (i): the code is a member of FAIL_REASONS and holds neither the bindValue nor the exception text."""
    planted = "PLANTED-BIND-VALUE-http://hunter2@evil"
    b = world()
    b[1][0]["unit"]["lang"] = planted                                   # an invalid header: schema error text would echo the value
    b[1].append({"type": "configEntry", "id": H("ce", 9), "bindValue": planted})
    code = R.classify_bundle(b[1], all_audit(tmp_path))
    assert code in FAIL_REASONS and planted not in code and "hunter2" not in code

    class Evil:                                                         # an exception whose message carries the value
        def __fspath__(self):
            raise RuntimeError(planted)
    thrown = R.classify_bundle(world()[0], Evil())
    assert thrown in FAIL_REASONS and planted not in thrown
    with pytest.raises(R.BundleRefused) as e:
        build(tmp_path, bundles=b)
    assert e.value.code == code and planted not in str(e.value) and "hunter2" not in str(e.value) and H("unit", "U2") in str(e.value)


def test_classify_bundle_maps_each_refusal_to_its_code(tmp_path):
    ad = all_audit(tmp_path)
    assert R.classify_bundle(world()[0], ad) is None
    assert R.classify_bundle(world(version="1.0")[0], ad) is None
    major = world()[0]
    major[0]["bundleSchemaVersion"] = "2.0"
    assert R.classify_bundle(major, ad) == "bundle:invalid"
    minor = world()[0]
    minor[0]["bundleSchemaVersion"] = "1.2"
    assert R.classify_bundle(minor, ad) == "bundle:schema-minor"
    assert R.classify_bundle(world()[0], audit_dir(tmp_path, [], name="empty")) == "bundle:audit-missing"
    assert R.classify_bundle([], ad) == "bundle:invalid" and R.classify_bundle([{"type": "file"}], ad) == "bundle:invalid"
    bad = world()[0]
    bad[0]["builtCommit"] = "xyz"
    assert R.classify_bundle(bad, ad) == "bundle:invalid"


def test_build_refuses_a_bundle_whose_audit_ids_do_not_all_check(tmp_path):
    ad = audit_dir(tmp_path, [sbx(UUID(1)), sbx(UUID(2)), sbx(UUID(3), exitCode=1)], name="one_bad")
    with pytest.raises(R.BundleRefused) as e:
        build(tmp_path, audit_dir=ad)
    assert e.value.code == "bundle:audit-missing" and e.value.unit_id == H("unit", "U2")
    with pytest.raises(R.BundleRefused):
        build(tmp_path, audit_dir=audit_dir(tmp_path, [sbx(UUID(1)), sbx(UUID(2))], name="miss_u3"))


def test_the_system_name_is_checked_again_here(tmp_path):
    from ccg.sysname import SystemNameError
    for bad in ("Shop", "shop_", "a__b", "", "-x", "x" * 49, "shop/x", None):
        with pytest.raises(SystemNameError):
            build(tmp_path, system=bad)


def test_a_system_with_no_fresh_or_stale_kept_unit_is_refused(tmp_path):
    nm = R.NotMappedUnit(H("unit", "NB"), REPO, "nb", "python")
    with pytest.raises(R.AssembleError):
        build(tmp_path, bundles=[], not_mapped=[nm])
    with pytest.raises(R.AssembleError):
        build(tmp_path, bundles=[])


def test_a_duplicate_unit_is_refused_and_an_unknown_stale_unit_too(tmp_path):
    b = world()
    with pytest.raises(R.AssembleError):
        build(tmp_path, bundles=[b[0], b[0]])
    with pytest.raises(R.AssembleError):
        build(tmp_path, stale={H("unit", "ghost"): ("2026-11-02T01:02:03Z", "extract:failed")})


# ---------------------------------------------------------------------------------------------------------------------
# LAUNCHES (the part the linker's API can feed) and the closed edge list
# ---------------------------------------------------------------------------------------------------------------------
def test_launch_unit_facts_give_launches_edges_to_the_unit(tmp_path):
    lu = {"fact": "launchUnit", "packId": "fastapi", "status": "confirmed", "reason": None, "launchId": H("lf", 1),
          "unitId": H("unit", "U1"), "target": "svc.a:app", "path": "compose.yaml"}
    ghost = lu | {"launchId": H("lf", 99)}
    rows = build(tmp_path, facts=PackFacts([], [], [], [], [lu, ghost], [], []))
    assert edges(rows, "ConfigEntry", "LAUNCHES", "Unit") == [{"s": H("lf", 1), "t": H("unit", "U1")}]
    assert rows.dropped == {"controlled-by-self-edge": 1, "launch-entry-missing": 1}


def test_every_edge_group_is_in_the_closed_list(tmp_path):
    ev = (lifted(SITE_A, 0), lifted(SITE_B, 0, ident=SITE_B, line=35), bound(SITE_B, 0, H("ce", 1)))
    facts = PackFacts(http_facts().providers, http_facts().consumers, [mount_fact("mnt_a", ["/a"], 82)], [], [], [], [])
    rows = build(tmp_path, links=links_of(mk_link(sites=[via(SITE_A), via(SITE_B, "possible", "port-shared", line=35)], evidence=ev)),
                 facts=facts)
    allowed = {("Unit", "IN_REPO", "Repo"), ("Unit", "HAS_FILE", "File"), ("File", "DEFINES", "Symbol"), ("Symbol", "HAS_CALL", "CallSite"),
               ("Symbol", "HAS_PARAM", "Param"), ("Symbol", "HAS_RETURN", "Ret"), ("Symbol", "HAS_COND", "Cond"),
               ("CallSite", "CALLS", "Symbol"), ("CallSite", "CALLS_NAME", "Name"), ("Name", "RESOLVES_TO", "Symbol"),
               ("CallSite", "DATA", "Param"), ("CallSite", "DATA", "CallSite"), ("Ret", "DATA", "Param"), ("Ret", "DATA", "CallSite"),
               ("Cond", "DATA", "Param"), ("Cond", "DATA", "CallSite"), ("CallSite", "CONTROLLED_BY", "Cond"),
               ("Ret", "CONTROLLED_BY", "Cond"), ("Cond", "CONTROLLED_BY", "Cond"), ("Param", "CONTROLLED_BY", "Cond"),
               ("Symbol", "PROVIDES", "ContractPoint"), ("CallSite", "CONSUMES", "ContractPoint"), ("CallSite", "VIA", "Link"),
               ("Link", "TO", "Symbol"), ("Link", "ON", "ContractPoint"), ("Link", "LIFTED_VIA", "CallSite"),
               ("Link", "LIFTED_VIA", "Symbol"), ("Unit", "DEPENDS_ON", "Dependency"), ("Unit", "SKIPS", "SkippedFile"),
               ("Link", "BOUND_BY", "ConfigEntry"), ("ConfigEntry", "LAUNCHES", "File"), ("ConfigEntry", "LAUNCHES", "Unit"),
               ("Symbol", "MOUNTED_BY", "CallSite"), ("Link", "TO_UNIT", "Unit")}
    assert set(rows.edges) <= allowed and len(allowed) == 34
    assert ("Link", "BOUND_BY", "ConfigEntry") in rows.edges


def test_a_config_entry_provider_is_refused_with_its_group_named(tmp_path):
    """-p1-graph.md §4: `(ConfigEntry)-[:PROVIDES]` and a Link `TO` a ConfigEntry are known refusals in P1."""
    f = PackFacts([prov(H("ce", 1), "config_key", "K")], [], [], [], [], [], [])
    with pytest.raises(R.AssembleError, match="PROVIDES"):
        build(tmp_path, facts=f)
    with pytest.raises(R.AssembleError, match="TO"):
        build(tmp_path, links=links_of(mk_link(provider=H("ce", 1), kind="config_key", key="K")), facts=PackFacts([], [], [], [], [], [], []))


def test_keyless_provider_and_consumer_have_no_contract_point_and_are_counted(tmp_path):
    f = PackFacts([prov(HANDLER, "http", None)], [cons(CALLER, "http", None, SITE_A)], [], [], [], [], [])
    rows = build(tmp_path, facts=f)
    assert "ContractPoint" not in rows.nodes
    assert rows.dropped["provider-key-none"] == 1 and rows.dropped["consumer-key-none"] == 1


# ---------------------------------------------------------------------------------------------------------------------
# Snapshot identity (§3)
# ---------------------------------------------------------------------------------------------------------------------
def snap(tmp_path, **kw):
    return build(tmp_path, **kw).snapshot


def test_snapshot_row_shape_and_constants(tmp_path):
    s = snap(tmp_path)
    assert set(s) == {"snapshotId", "systemId", "graphModelVersion", "bundleSchemaVersion", "vocabVersion", "manifestHash", "packSetHash",
                      "codeHash", "auditSetHash", "builtAt", "verdictLogOffset", "verdictLogHash"}
    assert s["graphModelVersion"] == "1.1" and s["verdictLogOffset"] == 0 and s["systemId"] == SYSTEM
    assert s["verdictLogHash"] == "c482b1cf54631b37eb4572dc86a68fb1"
    assert s["snapshotId"] == ids.tuple_id(["snapshot", SYSTEM, s["manifestHash"]])
    assert s["vocabVersion"] == "1.0" and s["builtAt"] == "2026-11-01T10:00:00Z"
    assert s["auditSetHash"] == hashlib.sha256("\n".join(sorted(UUID(i) for i in (1, 2, 3))).encode()).hexdigest()
    from ccg import answer
    assert answer.EMPTY_VERDICT_LOG_HASH == s["verdictLogHash"]


def test_manifest_hash_is_the_documented_canonical_json_of_the_documented_inputs(tmp_path):
    b = world()
    s = snap(tmp_path, bundles=b)
    units = sorted([[H("unit", u), "fresh", SN.bundle_hash(bb), False, ""] for u, bb in (("U1", b[0]), ("U2", b[1]))])
    body = {"graphModelVersion": "1.1", "packSetHash": s["packSetHash"], "codeHash": s["codeHash"], "units": units}
    assert s["manifestHash"] == hashlib.sha256(json.dumps(body, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()).hexdigest()


def test_bundle_hash_ignores_only_built_at_and_audit_ids(tmp_path):
    """-p1-graph.md §3: the exempt fields of spec3 §8.5; every other change (even a tool version) changes the hash."""
    base = world()[0]
    h0 = SN.bundle_hash(base)
    b = copy.deepcopy(base)
    b[0]["builtAt"] = "2030-01-01T00:00:00Z"
    b[0]["sandboxAuditIds"] = [UUID(99)]
    assert SN.bundle_hash(b) == h0
    for mut in (lambda r: r[0]["tools"].__setitem__("normalizerVersion", "2"), lambda r: r[0].__setitem__("builtCommit", "c" * 40),
                lambda r: r[3].__setitem__("lineStart", 2), lambda r: r.append({"type": "x"}), lambda r: r[0].__setitem__("flowProjected", False)):
        c = copy.deepcopy(base)
        mut(c)
        assert SN.bundle_hash(c) != h0
    ref = hashlib.sha256()
    for i, rec in enumerate(base):
        rec = {k: v for k, v in rec.items() if not (i == 0 and k in ("builtAt", "sandboxAuditIds"))}
        ref.update(json.dumps(rec, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode() + b"\n")
    assert h0 == ref.hexdigest()


def test_manifest_hash_changes_with_the_tool_version_a_flow_flip_a_pack_byte_and_a_code_byte(tmp_path):
    base = snap(tmp_path)["manifestHash"]
    assert snap(tmp_path, bundles=world(normalizer="2"))["manifestHash"] != base                           # mutant: drop bundleHash
    flip = tmp_path / "flip.yaml"
    flip.write_text("- {language: python, flowAvailable: true, measurement: m}\n")
    assert snap(tmp_path, flow_registry=flip)["manifestHash"] != base
    pd = pack_dir(tmp_path, "fastapi", "packFormat: 1\n# changed\n")
    assert snap(tmp_path, pack_dirs={"fastapi": pd})["manifestHash"] != base
    root = tmp_path / "code"
    for rel in SN.assembly_code_files(CPG):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes((CPG / rel).read_bytes())
    same = snap(tmp_path, code_root=root)["manifestHash"]
    assert same == base
    (root / "ccg" / "assemble" / "rows.py").write_bytes((root / "ccg" / "assemble" / "rows.py").read_bytes() + b"\n#x")
    assert snap(tmp_path, code_root=root)["manifestHash"] != base


def test_manifest_hash_ignores_a_flow_registry_edit_that_flips_no_loaded_unit_and_audit_ids_and_built_at(tmp_path):
    base = snap(tmp_path)
    other = tmp_path / "other.yaml"
    other.write_text("- {language: kotlin, flowAvailable: true, measurement: m}\n")
    assert snap(tmp_path, flow_registry=other)["manifestHash"] == base["manifestHash"]
    b = world()
    b[0][0]["builtAt"] = "2031-01-01T00:00:00Z"
    b[0][0]["sandboxAuditIds"] = [UUID(4)]
    again = snap(tmp_path, bundles=b)
    assert again["manifestHash"] == base["manifestHash"] and again["snapshotId"] == base["snapshotId"]
    assert again["auditSetHash"] != base["auditSetHash"] and again["builtAt"] != base["builtAt"]


def test_pack_set_hash_is_by_pack_id_order_and_content_not_by_list_order(tmp_path):
    d1, d2 = pack_dir(tmp_path, "a", "x: 1\n"), pack_dir(tmp_path, "b", "y: 2\n")
    p = [pack_def("b", "2"), pack_def("a", "1")]
    s1 = snap(tmp_path, packs=p, pack_dirs={"a": d1, "b": d2})
    s2 = snap(tmp_path, packs=list(reversed(p)), pack_dirs={"a": d1, "b": d2})
    assert s1["packSetHash"] == s2["packSetHash"]
    ch = lambda d: hashlib.sha256(b"".join(f.relative_to(d).as_posix().encode() + b"\0" + f.read_bytes() + b"\0"  # noqa: E731
                                           for f in sorted(d.rglob("*") , key=lambda f: f.relative_to(d).as_posix()) if f.is_file())).hexdigest()
    assert s1["packSetHash"] == ids.tuple_id(["packSet", "a", "1", ch(d1), "b", "2", ch(d2)])
    rows = build(tmp_path, packs=p, pack_dirs={"a": d1, "b": d2})
    assert rows.nodes["Pack"] == [{"packId": "a", "version": "1", "contentHash": ch(d1)}, {"packId": "b", "version": "2", "contentHash": ch(d2)}]


def test_pack_set_hash_sorts_by_pack_id_itself_and_a_nested_pack_file_is_covered(tmp_path):
    assert SN.pack_set_hash([("b", "2", "y"), ("a", "1", "x")]) == ids.tuple_id(["packSet", "a", "1", "x", "b", "2", "y"])
    d = pack_dir(tmp_path, "n", "x: 1\n")
    before = SN.pack_content_hash(d)
    (d / "sub").mkdir()
    (d / "sub" / "negatives.yaml").write_text("y: 2\n")
    assert SN.pack_content_hash(d) != before
    (d / "link").symlink_to(d / "pack.yaml")
    with pytest.raises(R.AssembleError):
        SN.pack_content_hash(d)


def test_an_edge_row_in_a_group_outside_the_closed_list_is_refused():
    rows = R.GraphRows()
    R.check_edge_groups(rows)
    rows.edges[("Link", "BOUND_BY", "ConfigEntry")] = [{"s": "a", "t": "b"}]
    R.check_edge_groups(rows)
    for g in (("ConfigEntry", "PROVIDES", "ContractPoint"), ("ConfigEntry", "CONSUMES", "ContractPoint"),
              ("ConfigEntry", "WRITES", "ContractPoint"), ("CallSite", "READS", "ContractPoint")):
        bad = R.GraphRows(edges={g: [{"s": "a", "t": "b"}]})
        with pytest.raises(R.AssembleError, match=g[1]):
            R.check_edge_groups(bad)


def test_a_mixed_vocab_version_is_refused(tmp_path):
    with pytest.raises(R.AssembleError):
        snap(tmp_path, packs=[pack_def("a", vocab="1.0"), pack_def("b", vocab="1.1")], pack_dirs={"a": pack_dir(tmp_path, "a"),
                                                                                                    "b": pack_dir(tmp_path, "b")})


def test_bundle_schema_version_is_the_highest_minor_loaded_and_per_unit_and_a_new_major_is_refused(tmp_path):
    b = world()
    b[1][0]["bundleSchemaVersion"] = "1.0"
    b[1][0].pop("bindKeyCheck")
    rows = build(tmp_path, bundles=b)
    assert rows.snapshot["bundleSchemaVersion"] == "1.1"
    u = by(rows, "Unit", "unitId")
    assert (u[H("unit", "U1")]["bundleSchemaVersion"], u[H("unit", "U2")]["bundleSchemaVersion"]) == ("1.1", "1.0")
    old = [world(version="1.0")[0]]
    assert build(tmp_path, bundles=old).snapshot["bundleSchemaVersion"] == "1.0"
    b[1][0]["bundleSchemaVersion"] = "2.0"
    with pytest.raises(R.BundleRefused) as e:
        build(tmp_path, bundles=b)
    assert e.value.code == "bundle:invalid"


def test_the_assembly_code_list_covers_every_ccg_module_the_assembler_imports(tmp_path):
    """-p1-graph.md §3 / §8 (f): a fresh ``-I`` interpreter, every ``ccg.*`` module in ``sys.modules`` after an assembly is listed."""
    import pickle
    job = tmp_path / "job.pkl"
    pickle.dump({"bundles": world(), "audit": str(all_audit(tmp_path)), "pack_dir": str(pack_dir(tmp_path))}, job.open("wb"))
    script = tmp_path / "run.py"
    script.write_text(textwrap.dedent(f"""
        import sys, pickle
        sys.path.insert(0, {str(CPG)!r})
        from pathlib import Path
        from ccg.assemble import rows as R, snapshot as SN
        from ccg.linker import Links
        from ccg.packrt import PackFacts
        j = pickle.load(open({str(job)!r}, "rb"))
        R.build_rows(j["bundles"], Links([], [], {{}}), PackFacts([], [], [], [], [], [], []), system="shop",
                     packs=[{{"packFormat": 1, "id": "fastapi", "version": "0.1.0", "kind": "framework", "vocabVersion": "1.0"}}],
                     pack_dirs={{"fastapi": Path(j["pack_dir"])}}, audit_dir=Path(j["audit"]))
        import ccg.sandbox_audit, ccg.staleness   # lazy imports are part of the set
        mods = sorted(m.__file__ for n, m in sys.modules.items() if (n == "ccg" or n.startswith("ccg.")) and getattr(m, "__file__", None))
        print("\\n".join(mods))
    """))
    out = subprocess.run([sys.executable, "-I", str(script)], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    listed = set(SN.assembly_code_files(CPG))
    mods = {Path(m).resolve().relative_to(CPG).as_posix() for m in out.stdout.split()}
    assert mods and mods <= listed, sorted(mods - listed)
    assert "ccg/assemble/rows.py" in mods and "ccg/linker/link.py" in mods


def test_assembly_code_files_are_the_declared_globs_plus_reasons_and_nothing_else():
    files = SN.assembly_code_files(CPG)
    assert "ccg/registry/reasons.yaml" in files and "ccg/registry/flow-availability.yaml" not in files
    assert not any(f.startswith(("ccg/engine", "ccg/front", "ccg/store")) for f in files)
    assert files == sorted(files)
    with pytest.raises(R.AssembleError):
        SN.assembly_code_files(CPG / "ccg")                                    # a root without the listed files is refused


# ---------------------------------------------------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------------------------------------------------
def test_two_builds_give_identical_rows_whatever_the_input_order(tmp_path):
    ev = (lifted(SITE_A, 0), lifted(SITE_B, 0, ident=SITE_B, line=35), bound(SITE_B, 0, H("ce", 1)))
    lks = [mk_link("L1", sites=[via(SITE_A), via(SITE_B, "possible", "port-shared", line=35)], evidence=ev), mk_link("L2", sites=[via(SITE_A)])]
    other = prov(H("sym", "work"), "http", "GET /other", site="decl9")
    facts = PackFacts([other] + http_facts().providers + [mounted_provider("GET /items/{}", "mnt_a", "mnt_b")], http_facts().consumers,
                      [mount_fact("mnt_b", ["/zeta"], 85), mount_fact("mnt_a", ["/alpha"], 82)], [], [], [], [])
    a = build(tmp_path, links=links_of(*lks), facts=facts)
    swapped = PackFacts(list(reversed(facts.providers)), facts.consumers, list(reversed(facts.mounts)), [], [], [], [])
    b = build(tmp_path, bundles=list(reversed(world())), links=links_of(*reversed(lks)), facts=swapped)
    assert R.canonical_rows(a) == R.canonical_rows(b) and a.snapshot == b.snapshot
    assert a.nodes and R.canonical_rows(a) == R.canonical_rows(build(tmp_path, links=links_of(*lks), facts=facts))


# ---------------------------------------------------------------------------------------------------------------------
# F3: every drop reason and every refusal in rows.py is reached by a planted input (a sweep of the source, not a case list)
# ---------------------------------------------------------------------------------------------------------------------
ROWS_SRC = CPG / "ccg" / "assemble" / "rows.py"


def _drop_literals() -> list[str]:
    """Every ``<x>.drop("reason")`` call of rows.py, one entry per call site (an f-string ``duplicate-{t}`` is the prefix)."""
    out = []
    for n in ast.walk(ast.parse(ROWS_SRC.read_text())):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "drop" and n.args:
            a = n.args[0]
            if isinstance(a, ast.Constant):
                out.append(a.value)
            elif isinstance(a, ast.JoinedStr):
                out.append("".join(v.value for v in a.values if isinstance(v, ast.Constant)))
    return sorted(out)


def _raise_literals() -> list[str]:
    """The longest constant text of every ``raise AssembleError(<message>)`` of rows.py, one entry per raise."""
    out = []
    for n in ast.walk(ast.parse(ROWS_SRC.read_text())):
        if isinstance(n, ast.Raise) and isinstance(n.exc, ast.Call) and getattr(n.exc.func, "id", "") == "AssembleError":
            a = n.exc.args[0]
            parts = [a.value] if isinstance(a, ast.Constant) else [v.value for v in a.values if isinstance(v, ast.Constant)]
            out.append(max(parts, key=len))
    return sorted(out)


GHOST_SYM, GHOST_CS, GHOST_CE = H("sym", "ghost"), H("cs", "ghost"), H("ce", "ghost")


def _with(unit_records, which=0):
    w = world()
    w[which] += list(unit_records)
    return w


def _planted_drops(tmp_path):
    """reason -> one planter per ``drop(...)`` call site of that reason; each returns ``build`` rows with exactly the planted input."""
    def facts_(prov_=(), cons_=(), mounts=(), launch=()):
        return PackFacts(list(prov_), list(cons_), list(mounts), [], list(launch), [], [])
    lu = {"fact": "launchUnit", "packId": "fastapi", "status": "confirmed", "reason": None, "launchId": H("lf", 1),
          "unitId": H("unit", "U1"), "target": "t", "path": "compose.yaml"}
    mp = {"fact": "mountPath", "packId": "fastapi", "status": "confirmed", "reason": None, "unitId": H("unit", "U1"), "path": "/mcp",
          "pathMatch": "prefix", "siteId": H("cs", "mcp_mnt"), "line": 88, "conditions": []}
    bound_lk = mk_link(reason="binding-confirmed", sites=[via(SITE_A, reason="binding-confirmed")],
                       evidence=(bound(SITE_A, 0, GHOST_CE),))
    return {
        "calls-target-not-in-snapshot": [lambda: build(tmp_path, bundles=_with([call_rec("g", "caller", "fetch", 36, targets=["ghostsym"])]))],
        "data-endpoint-missing": [lambda: build(tmp_path, bundles=_with([data_rec(H("nowhere"), H("cs", "res"))]))],
        "controlled-by-self-edge": [lambda: build(tmp_path, bundles=_with(
            [{"type": "controlledBy", "id": H("cb", 3), "point": H("k", "c"), "cond": H("k", "c")}]))],
        "controlled-by-endpoint-missing": [lambda: build(tmp_path, bundles=_with(
            [{"type": "controlledBy", "id": H("cb", 4), "point": H("nowhere"), "cond": H("k", "c")}]))],
        "provider-handler-missing": [lambda: build(tmp_path, facts=facts_([prov(GHOST_SYM, "http", "K")]))],
        "provider-key-none": [lambda: build(tmp_path, facts=facts_([prov(HANDLER, "http", None)]))],
        "consumer-site-missing": [lambda: build(tmp_path, facts=facts_(cons_=[cons(CALLER, "http", "K", GHOST_CS)]))],
        "consumer-key-none": [lambda: build(tmp_path, facts=facts_(cons_=[cons(CALLER, "http", None, SITE_A)]))],
        "link-param-map": [lambda: build(tmp_path, links=links_of(mk_link(param_map={"a": "b"})), facts=http_facts())],
        "via-site-missing": [lambda: build(tmp_path, links=links_of(mk_link(sites=[via(SITE_A), via(GHOST_CS)])), facts=http_facts())],
        "lifted-hop-missing": [
            lambda: build(tmp_path, links=links_of(mk_link(evidence=(lifted(SITE_A, 0, ident=GHOST_CS),))), facts=http_facts()),
            lambda: build(tmp_path, links=links_of(mk_link(evidence=(lifted(SITE_A, 0, kind="symbol", ident=GHOST_SYM),))),
                          facts=http_facts())],
        "bound-entry-missing": [lambda: build(tmp_path, links=links_of(bound_lk), facts=http_facts())],
        "mount-site-missing": [
            lambda: build(tmp_path, facts=facts_([mounted_provider("GET /items/{}", "ghost")], mounts=[mount_fact("ghost", ["/x"], 1)])),
            lambda: build(tmp_path, facts=facts_(launch=[mp | {"siteId": GHOST_CS}]))],
        "mount-fact-missing": [lambda: build(tmp_path, facts=facts_([mounted_provider("GET /items/{}", "mnt_a")]))],
        "mount-path-no-kinds": [lambda: build(tmp_path, facts=facts_(launch=[mp]), packs=[pack_def(kinds=())])],
        "launch-entry-missing": [lambda: build(tmp_path, facts=facts_(launch=[lu | {"launchId": GHOST_CE}]))],
        "launch-unit-missing": [lambda: build(tmp_path, facts=facts_(launch=[lu | {"unitId": H("unit", "ghost")}]))],
        "duplicate-": [lambda: build(tmp_path, bundles=_with([file_rec("a", "svc/a.py")]))],
    }


def test_every_drop_reason_in_rows_py_is_reached_by_a_planted_input_exactly_once(tmp_path):
    planted = _planted_drops(tmp_path)
    literals = _drop_literals()
    assert sorted(planted) == sorted(set(literals)), "a drop reason was added or removed without a planted input"
    for reason in planted:                                         # one planter per call site of the reason
        assert len(planted[reason]) >= literals.count(reason), reason
    baseline = build(tmp_path).dropped
    for reason, planters in planted.items():
        for k, plant in enumerate(planters):
            got = plant().dropped
            delta = {r: got.get(r, 0) - baseline.get(r, 0) for r in set(got) | set(baseline)}
            delta = {r: n for r, n in delta.items() if n}
            expect = {"duplicate-HAS_FILE": 1} if reason == "duplicate-" else {reason: 1}
            assert delta == expect, (reason, k, delta)


def _bad_registry(tmp_path, body):
    p = tmp_path / f"reg{abs(hash(body))}.yaml"
    p.write_text(body)
    return lambda: R.load_flow_rows(p)


def _stale(tmp_path, **kw):
    return lambda: build(tmp_path, **kw)


def _lk_raise(tmp_path, lk, **kw):
    return lambda: build(tmp_path, links=links_of(lk), facts=http_facts(), **kw)


def _planted_raises(tmp_path):
    """longest constant text of each ``raise AssembleError`` message -> a planter that must raise it."""
    ok_stale = ("2026-11-02T01:02:03Z", "extract:failed")
    pb = world()
    ev_bound = lambda **kw: (lifted(SITE_A, 0), bound(SITE_A, 0, H("ce", 1), **kw))      # noqa: E731
    sites_bound = [via(SITE_A, reason="binding-confirmed")]
    bound_lk = lambda **kw: mk_link(reason="binding-confirmed", sites=sites_bound, evidence=ev_bound(**kw))   # noqa: E731
    lo, hi = sorted([SITE_A, SITE_B])
    two_packs = [pack_def(), pack_def("other", vocab="2.0")]
    return {
        "flow registry unreadable (": lambda: R.load_flow_rows(tmp_path / "no-such.yaml"),
        "flow registry is a list of rows": _bad_registry(tmp_path, "42\n"),
        "flow registry row: keys are exactly language, flowAvailable, measurement": _bad_registry(tmp_path, "- {language: python}\n"),
        "flow registry row: an unknown or duplicate language": _bad_registry(
            tmp_path, "- {language: cobol, flowAvailable: true, measurement: m}\n"),
        "flow registry row: flowAvailable is a boolean and measurement cites the report": _bad_registry(
            tmp_path, "- {language: python, flowAvailable: 'x', measurement: m}\n"),
        ": failReason is not a member of FAIL_REASONS": _stale(tmp_path, stale={H("unit", "U1"): ("2026-11-02T01:02:03Z", "nope")}),
        ": failedAt is not an RFC 3339 UTC timestamp ending Z": _stale(tmp_path, stale={H("unit", "U1"): ("yesterday", "extract:failed")}),
        "two bundles carry one unitId": _stale(tmp_path, bundles=[pb[0], pb[0]]),
        "no fresh or stale-kept unit: there is nothing to answer from": _stale(tmp_path, bundles=[]),
        ": marked stale-kept but no bundle was given": _stale(tmp_path, stale={H("unit", "ghost"): ok_stale}),
        ": a stale-kept unit carries failedAt": _stale(tmp_path, stale={H("unit", "U1"): (None, "extract:failed")}),
        ": both built and not-mapped": _stale(tmp_path, not_mapped=[R.NotMappedUnit(H("unit", "U1"), REPO, "x", "python")]),
        "two not-mapped units carry one unitId": _stale(
            tmp_path, not_mapped=[R.NotMappedUnit(H("unit", "NB"), REPO, "x", "python")] * 2),
        "the packs declare more than one vocabVersion": _stale(
            tmp_path, packs=two_packs, pack_dirs={"fastapi": pack_dir(tmp_path), "other": pack_dir(tmp_path, "other")}),
        " is not in the P1 edge list (-p1-graph.md §4); no P1 pack emits it": _stale(
            tmp_path, facts=PackFacts([prov(SITE_A, "http", "K")], [], [], [], [], [], [])),
        ": given twice": lambda: build(tmp_path, links=links_of(mk_link(), mk_link()), facts=http_facts()),
        ": (Link)-[:TO]->(ConfigEntry) is not in the P1 edge list (-p1-graph.md §4)": _lk_raise(
            tmp_path, mk_link(provider=H("ce", 1), kind="config_key", key="K")),
        ": a link whose provider is outside the snapshot is a corrupt input": _lk_raise(tmp_path, mk_link(provider=GHOST_SYM)),
        ": TO_UNIT names a unit with no Unit row": _lk_raise(
            tmp_path, mk_link(provider=None, to_unit=H("unit", "NB"), status="possible", reason="target-not-mapped",
                              sites=[via(SITE_A, "possible", "target-not-mapped")])),
        ": no consumer site": _lk_raise(tmp_path, mk_link()._replace(sites=())),
        ": evidence names a site that is not a VIA site of the link": _lk_raise(tmp_path, mk_link(evidence=(lifted(SITE_B, 0, ident=SITE_B),))),
        ": two BOUND_BY edges for one (site, entry)": _lk_raise(
            tmp_path, mk_link(reason="binding-confirmed", sites=sites_bound,
                              evidence=(bound(SITE_A, 0, H("ce", 1)), bound(SITE_A, 1, H("ce", 1))))),
        ": BOUND_BY port and portOrigin come together": _lk_raise(tmp_path, bound_lk(port=80)),
        ": BOUND_BY port outside 1-65535": _lk_raise(tmp_path, bound_lk(port=0, portOrigin="explicit")),
        ": BOUND_BY portOrigin outside the closed set": _lk_raise(tmp_path, bound_lk(port=80, portOrigin="guess")),
        ": BOUND_BY packId is present iff portOrigin is pack-default": _lk_raise(tmp_path, bound_lk(port=80, portOrigin="explicit", packId="fastapi")),
        ": BOUND_BY packId names no loaded Pack": _lk_raise(tmp_path, bound_lk(port=80, portOrigin="pack-default", packId="nopack")),
        ": a VIA status outside the closed set": _lk_raise(tmp_path, mk_link(sites=[via(SITE_A, "maybe", "key-match")])),
        ": VIA reason violates O-9 (binding-confirmed iff confirmed and bound at that site)": _lk_raise(
            tmp_path, mk_link(sites=[via(SITE_A, reason="binding-confirmed")])),
        ": Link.status is not the best over its sites": _lk_raise(tmp_path, mk_link(status="possible")),
        ": Link.reason violates O-9 (binding-confirmed iff some confirmed site is)": _lk_raise(
            tmp_path, mk_link(reason="binding-confirmed")),
        ": Link.reason is not the reason of the site that sets the best status": _lk_raise(tmp_path, mk_link(reason="partial-key")),
        ") is not in the P1 list": lambda: R.check_edge_groups(R.GraphRows(edges={("A", "B", "C"): []})),
    }


def test_every_assemble_error_in_rows_py_is_reached_by_a_planted_input(tmp_path):
    planted = _planted_raises(tmp_path)
    literals = _raise_literals()
    assert sorted(planted) == sorted(set(literals)), (sorted(set(literals) - set(planted)), sorted(set(planted) - set(literals)))
    for text, plant in planted.items():
        with pytest.raises(R.AssembleError) as e:
            plant()
        assert text in str(e.value), (text, str(e.value))
    # the PROVIDES refusal has two anchors (a ConfigEntry, a CallSite): both reach it
    with pytest.raises(R.AssembleError, match="PROVIDES from a ConfigEntry"):
        build(tmp_path, facts=PackFacts([prov(H("ce", 1), "config_key", "K")], [], [], [], [], [], []))


def test_a_link_whose_provider_is_outside_the_snapshot_is_refused_with_ids_only(tmp_path):
    with pytest.raises(R.AssembleError) as e:
        build(tmp_path, links=links_of(mk_link(provider=GHOST_SYM)), facts=http_facts())
    assert H("link", "L1") in str(e.value) and GHOST_SYM not in str(e.value)


# ---------------------------------------------------------------------------------------------------------------------
# F3 survivors: stated behaviours that had no test
# ---------------------------------------------------------------------------------------------------------------------
def test_snapshot_built_at_is_the_maximum_over_fresh_units_only(tmp_path):
    b = world()
    b[0][0]["builtAt"] = "2026-11-03T10:00:00Z"                    # the stale-kept unit is the latest built
    b[1][0]["builtAt"] = "2026-11-01T09:00:00Z"
    s = build(tmp_path, bundles=b, stale={H("unit", "U1"): ("2026-11-04T00:00:00Z", "extract:failed")}).snapshot
    assert s["builtAt"] == "2026-11-01T09:00:00Z"
    both = build(tmp_path, bundles=b, stale={H("unit", "U1"): ("2026-11-04T00:00:00Z", "extract:failed"),
                                             H("unit", "U2"): ("2026-11-04T00:00:00Z", "extract:failed")}).snapshot
    assert "builtAt" not in both                                    # no fresh unit: absent


def test_the_link_reason_is_the_reason_of_the_lowest_site_id_among_equally_strong_sites(tmp_path):
    lo, hi = sorted([SITE_A, SITE_B])
    sites = [via(hi, "possible", "port-shared"), via(lo, "possible", "partial-key")]       # the higher id is listed first
    ok = mk_link(status="possible", reason="partial-key", sites=sites)
    build(tmp_path, links=links_of(ok), facts=http_facts())
    for wrong in ("port-shared",):
        with pytest.raises(R.AssembleError):
            build(tmp_path, links=links_of(mk_link(status="possible", reason=wrong, sites=sites)), facts=http_facts())
    with pytest.raises(R.AssembleError):                           # and in the other listing order
        build(tmp_path, links=links_of(mk_link(status="possible", reason="port-shared", sites=sites[::-1])), facts=http_facts())


def test_flow_available_of_a_not_mapped_unit_is_the_registry_value_or_false_and_is_a_manifest_input(tmp_path):
    reg = tmp_path / "flow.yaml"
    reg.write_text("- {language: python, flowAvailable: true, measurement: 'docs/x.md §1'}\n")
    nms = [R.NotMappedUnit(H("unit", "NB1"), REPO, "nb1", "python"), R.NotMappedUnit(H("unit", "NB2"), REPO, "nb2", "go")]
    rows = build(tmp_path, flow_registry=reg, not_mapped=nms)
    u = by(rows, "Unit", "unitId")
    assert u[H("unit", "NB1")]["flowAvailable"] is True and u[H("unit", "NB2")]["flowAvailable"] is False
    units = []
    for r in world():
        uid = r[0]["unitId"]
        units.append([uid, "fresh", SN.bundle_hash(r), u[uid]["flowAvailable"], ""])
    units += [[H("unit", "NB1"), "not-mapped", "", True, "manifest:never-built"],
              [H("unit", "NB2"), "not-mapped", "", False, "manifest:never-built"]]
    assert rows.snapshot["manifestHash"] == SN.manifest_hash(rows.snapshot["packSetHash"], rows.snapshot["codeHash"], units)


def test_repo_head_commit_is_the_first_fresh_units_commit_in_unit_id_order_else_the_first_stale_kept(tmp_path):
    lo, hi = sorted([H("unit", "U1"), H("unit", "U2")])
    commit = {H("unit", "U1"): "1" * 40, H("unit", "U2"): "2" * 40}
    b = world()
    for recs in b:
        recs[0]["builtCommit"] = commit[recs[0]["unitId"]]
    both = build(tmp_path, bundles=b)
    assert by(both, "Repo", "repoId")[REPO]["headCommit"] == commit[lo]                      # first wins
    stale_lo = build(tmp_path, bundles=b, stale={lo: ("2026-11-02T01:02:03Z", "extract:failed")})
    assert by(stale_lo, "Repo", "repoId")[REPO]["headCommit"] == commit[hi]                  # a fresh unit beats a stale-kept one
    stale_both = build(tmp_path, bundles=b, stale={lo: ("2026-11-02T01:02:03Z", "extract:failed"),
                                                   hi: ("2026-11-02T01:02:03Z", "extract:failed")})
    assert by(stale_both, "Repo", "repoId")[REPO]["headCommit"] == commit[lo]
    assert len(rows_of(both, "Repo")) == 1


def rows_of(rows, label):
    return rows.nodes.get(label, [])


# ---------------------------------------------------------------------------------------------------------------------
# F5, F6, F8
# ---------------------------------------------------------------------------------------------------------------------
def test_check_sandbox_ids_messages_never_echo_a_header_value(tmp_path):
    planted = "SECRET-ish value http://u:p@h"
    d = audit_dir(tmp_path, [sbx(UUID(1))])
    for bad in (planted, None, 5, planted.upper(), {"k": planted}):
        errs = check_sandbox_ids({"sandboxAuditIds": [UUID(1), bad], "egressMode": "offline"}, d)
        assert len(errs) == 1 and "sandboxAuditIds[1]" in errs[0], errs
        assert "SECRET" not in errs[0].upper() and "http" not in errs[0] and "u:p" not in errs[0]
    errs = check_sandbox_ids({"sandboxAuditIds": [planted, UUID(1)], "egressMode": planted}, d)
    assert errs and all("SECRET" not in e.upper() and "u:p" not in e for e in errs)
    assert any("sandboxAuditIds[0]" in e for e in errs)


def test_classify_bundle_reads_the_header_only_and_a_malformed_record_body_is_a_raw_error_naming_no_value(tmp_path):
    planted = "PLANTED-LANG-VALUE"
    b = world()
    b[0].append({"type": "symbol", "lang": planted})                # a symbol record without `id`: read_bundle would refuse it
    ad = all_audit(tmp_path)
    assert R.classify_bundle(b[0], ad) is None                      # the header is all that is classified
    with pytest.raises(KeyError) as e:
        build(tmp_path, bundles=b)
    assert planted not in str(e.value) and "id" in str(e.value)
    assert "read_bundle" in (R.classify_bundle.__doc__ or "") and "read_bundle" in (R.build_rows.__doc__ or "")


def test_via_lifted_via_and_bound_by_rows_do_not_depend_on_the_links_site_or_evidence_order(tmp_path):
    import random
    ev = (lifted(SITE_A, 0), lifted(SITE_A, 1, kind="symbol", ident=CALLER, line=30, viaSiteId=SITE_A, viaLine=31, verbPath="p0.m"),
          lifted(SITE_B, 0, ident=SITE_B, line=35), lifted(SITE_B, 1, kind="symbol", ident=HANDLER, line=11),
          bound(SITE_B, 0, H("ce", 1)), bound(SITE_B, 1, H("lf", 1), port=8000, portOrigin="pack-default", packId="fastapi"))
    sites = [via(SITE_A), via(SITE_A, "possible", "partial-key", key="GET /items/{?}"), via(SITE_B, "possible", "port-shared", line=35)]
    mk = lambda s, e: build(tmp_path, links=links_of(mk_link(sites=s, evidence=e)), facts=http_facts())    # noqa: E731
    want = R.canonical_rows(mk(sites, ev))
    rng = random.Random(7)
    for k in range(12):
        s, e = list(sites), list(ev)
        if k == 0:
            s, e = s[::-1], e[::-1]
        else:
            rng.shuffle(s), rng.shuffle(e)
        assert R.canonical_rows(mk(s, e)) == want, k
