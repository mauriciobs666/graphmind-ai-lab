"""Launch facts a pack declares (rule-pack spec §1.6 ``launch``; walkthrough §4.4).

The bundle carries launch and config records but not what a framework does by default; a pack supplies that. Over the
``launchFact`` records and the call sites of the bundles this module derives, per unit: which unit a command launches
(``targetArg``), its ports (explicit ``portFlags``, ``envPorts`` heuristics, ``frameworkDefaults``) and its mount paths
(``mountPaths``). The URL-binding step (``urlBindings``: config and call-site URL sources, host/port/path matching) is the
linker's (P1-17).

Every fact is ``{"fact": ..., "packId": ..., "status": "confirmed", "reason": None}`` plus its own fields: none is an edge, so
no registry code applies, and ``origin`` (``explicit``, ``env-default``, ``pack-default``) says how a port was derived.
"""
from __future__ import annotations

import re
from collections import defaultdict

from ccg.bundle import REDACTED
from ccg.packrt.extract import Corpus
from ccg.packrt.httpkey import UNKNOWN
from ccg.packrt.match import clause_hit, text_hit

_MODULE_ATTR = re.compile(r"^([A-Za-z_][\w.]*):[A-Za-z_]\w*$")


def _base(pack_id: str, fact: str) -> dict:
    return {"fact": fact, "packId": pack_id, "status": "confirmed", "reason": None}


def _port(value) -> int | None:
    if isinstance(value, str) and value.isdigit() and 1 <= int(value) <= 65535:
        return int(value)
    return None


def _flag_value(args: list[str], flag: str) -> tuple[bool, int | None]:
    """``(flag present, its port)`` for ``--port 8000`` and ``--port=8000``."""
    for i, a in enumerate(args):
        if a == flag:
            return True, _port(args[i + 1]) if i + 1 < len(args) else None
        if a.startswith(flag + "="):
            return True, _port(a[len(flag) + 1:])
    return False, None


def _launched_unit(corpus: Corpus, launch: dict, entries: list[dict]) -> tuple[str, str] | None:
    """``(unitId, target)``: the in-repo file ``module/with/slashes.py`` that a ``module:attr`` positional argument names."""
    for e in entries:
        if "command" in e and not text_hit(e["command"], launch["command"]):
            continue
        args = launch["args"]
        if e["index"] >= len(args) or not (m := _MODULE_ATTR.match(args[e["index"]])):
            continue
        suffix = m.group(1).replace(".", "/") + ".py"
        hits = [f for f in corpus.files.values() if f["path"] == suffix or f["path"].endswith("/" + suffix)]
        if len(hits) == 1:
            return hits[0]["unitId"], args[e["index"]]
    return None


def launch_facts(corpus: Corpus, pack: dict) -> list[dict]:
    spec, pid = pack["launch"], pack["id"]
    out: list[dict] = []
    commands = [f for f in sorted(corpus.launch_facts, key=lambda f: f["id"]) if f["kind"] == "command"]
    units: dict[str, tuple[str, str]] = {}
    for lf in commands:
        got = _launched_unit(corpus, lf, spec.get("targetArg", ()))
        if got:
            units[lf["id"]] = got
            out.append(_base(pid, "launchUnit") | {"launchId": lf["id"], "unitId": got[0], "target": got[1], "path": lf["path"]})

    def protocol(lf: dict) -> str | None:
        return next((d.get("protocol") for d in spec.get("frameworkDefaults", ()) if text_hit(d["command"], lf["command"])), None)

    ports: dict[tuple, set] = defaultdict(set)         # (unit, port, origin, protocol) -> launch ids
    for lf in commands:
        if lf["id"] not in units:
            continue
        unit = units[lf["id"]][0]
        flags = [e for e in spec.get("portFlags", ()) if text_hit(e["command"], lf["command"])]
        present = False
        for e in flags:
            for flag in e["flags"]:
                seen, value = _flag_value(lf["args"], flag)
                present |= seen
                if value is not None:
                    ports[(unit, value, "explicit", protocol(lf))].add(lf["id"])
        if not present and not any(REDACTED in a for a in lf["args"]):   # a hidden argument may be the port flag (spec 1.6)
            for d in spec.get("frameworkDefaults", ()):
                if text_hit(d["command"], lf["command"]):
                    ports[(unit, d["port"], "pack-default", d.get("protocol"))].add(lf["id"])
    for ep in spec.get("envPorts", ()):
        for env in sorted((f for f in corpus.launch_facts if f["kind"] == "env-default"), key=lambda f: f["id"]):
            value = _port(env.get("default"))
            if value is None or not text_hit(ep["envVar"], env["envVar"]):
                continue
            if any(text_hit(x, env["envVar"]) for x in ep.get("excludeVars", ())):
                continue
            for lf in commands:                          # adoptedBy: same-file-launch
                if lf["path"] == env["path"] and lf["id"] in units:
                    ports[(units[lf["id"]][0], value, "env-default", protocol(lf))] |= {env["id"], lf["id"]}
    for (unit, port, origin, proto), ids in sorted(ports.items(), key=lambda kv: (kv[0][:3], kv[0][3] or "")):
        out.append(_base(pid, "port") | {"unitId": unit, "port": port, "origin": origin, "protocol": proto,
                                         "launchIds": sorted(ids)})

    for mp in spec.get("mountPaths", ()):
        for cid in sorted(corpus.callsites):
            site = corpus.callsites[cid]
            if not clause_hit("callsite", mp["call"], site, corpus):
                continue
            fed = any(d["toPath"] == mp["app"]["arg"] and d["fromPath"] == "r" and d["def"] in corpus.callsites
                      and clause_hit("callsite", mp["app"]["fromCall"], corpus.callsites[d["def"]], corpus)
                      for d in corpus.data_by_use.get(cid, ()))
            unit = corpus.unit_of_file(corpus.file_id_of(site))
            if not fed or unit is None:
                continue
            for path in corpus.arg_alternatives(site, mp["path"]["arg"]) or ():
                if UNKNOWN in path:
                    continue
                conds = [{"siteId": cid, "line": corpus.conds[k]["line"], "code": corpus.conds[k]["code"]}
                         for k in sorted(corpus.controlled_by.get(cid, ()))]
                out.append(_base(pid, "mountPath") | {"unitId": unit, "path": path, "pathMatch": mp["pathMatch"],
                                                      "siteId": cid, "line": site["line"], "conditions": conds})
    return out
