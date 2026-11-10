"""Record builders for the pack-runtime tests (P1-16): readable ids, only the fields the runtime reads.

Records are not schema-valid on purpose (ids are words, not hex32): the runtime reads fields, it never validates, and a
planted fixture is easier to audit when ``h`` names the handler. The real-shape cases run over the stored slice bundles.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path

from ccg.rulepack import load_pack

CPG = Path(__file__).resolve().parents[1]
S14A = CPG / ".ccg-work" / "vertical-slice" / "s14a"
PACKS = CPG / "packs"


def file_(fid, path, unit="U", role="primary", lang="python"):
    return {"type": "file", "id": fid, "path": path, "unitId": unit, "role": role, "lang": lang}


def imp(fid, module, *names):
    return {"type": "import", "id": f"imp-{fid}-{module}", "fileId": fid, "module": module,
            "names": [{"name": n} for n in names], "line": 1, "ordinal": 0, "dynamic": False}


def sym(sid, name, fid="f1", path="a.py", parent=None, kind="function", start=1, qname=None):
    r = {"type": "symbol", "id": sid, "name": name, "fileId": fid, "path": path, "kind": kind,
         "qname": qname or f"{path}::{name}", "lineStart": start, "lineEnd": start + 5}
    if parent:
        r["parentId"] = parent
    return r


def call(cid, owner, callee, line=1, resolution="unresolved", targets=(), external=None, kind="call"):
    r = {"type": "callsite", "id": cid, "symbolId": owner, "calleeName": callee, "line": line, "callKind": kind,
         "resolution": resolution, "targets": [{"symbolId": t, "qname": f"q::{t}"} for t in targets], "ordinal": 0}
    if external:
        r["externalTarget"] = external
    return r


def vs(cid, arg, *values):
    """A valueSet; each value is a dict element or a bare string (a const)."""
    return {"type": "valueSet", "id": f"vs-{cid}-{arg}", "callSiteId": cid, "arg": arg,
            "values": [v if isinstance(v, dict) else {"const": v} for v in values]}


def dec(did, target, name, *args, line=1, form="decorator"):
    """``args`` are ``(path, value, ...)`` tuples; a value is a dict element or a const."""
    return {"type": "decorator", "id": did, "targetId": target, "name": name, "line": line, "form": form, "ordinal": 0,
            "args": [{"arg": a[0], "values": [v if isinstance(v, dict) else {"const": v} for v in a[1:]]} for a in args]}


def data(use, def_, to, frm, status="confirmed", reason="intra-flow"):
    return {"type": "data", "id": f"d-{use}-{to}-{def_}-{frm}", "use": use, "def": def_, "toPath": to,
            "fromPath": frm, "status": status, "reason": reason}


def param(pid, owner, index, name="x", line=1):
    return {"type": "param", "id": pid, "symbolId": owner, "index": index, "name": name, "line": line}


def ret(rid, owner):
    return {"type": "ret", "id": rid, "symbolId": owner, "line": 1, "ordinal": 0}


def cond(cid, owner, code, line=1):
    return {"type": "cond", "id": cid, "symbolId": owner, "code": code, "line": line, "ordinal": 0}


def ctrl(point, cid):
    return {"type": "controlledBy", "id": f"cb-{point}-{cid}", "point": point, "cond": cid}


def constant(qname, *values):
    return {"type": "constant", "id": f"c-{qname}", "qname": qname, "fileId": "f1", "line": 1,
            "values": [v if isinstance(v, dict) else {"const": v} for v in values]}


def launch(lid, path, kind, **kw):
    return {"type": "launchFact", "id": lid, "path": path, "line": 1, "ordinal": 0, "kind": kind,
            "origin": kw.pop("origin", "explicit"), **kw}


def config(cid, path, key_path, bind=None, fmt="mcp-json", example=False):
    r = {"type": "configEntry", "id": cid, "path": path, "line": 1, "format": fmt, "keyPath": key_path, "example": example}
    if bind is not None:
        r["bindValue"] = bind
    return r


def pack(name):
    """The shipped pack ``cpg/packs/<name>/pack.yaml`` through ``load_pack``.

    With ``CCG_PACKRT_PACKS_FROM_SPEC`` set, the pack is the YAML block of the rule-pack spec instead: the leaf-mutation
    harness (`test_rule_pack_leaf_mutation.py`) edits one value of that block in a scratch copy and runs the runtime tests,
    which is how a value only a runtime reads (a callee name, an import gate, a port) is shown to flip a result.
    """
    if os.environ.get("CCG_PACKRT_PACKS_FROM_SPEC"):
        import test_rule_pack_spec as spec
        return copy.deepcopy(spec.PACKS[name])
    return load_pack(PACKS / name / "pack.yaml")
