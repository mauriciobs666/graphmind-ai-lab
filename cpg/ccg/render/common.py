"""What the two renderers share: the fenced/masked data helpers, the grouping, and every sentence a fact becomes.

Spec: ``docs/plans/connected-code-graph-answer-spec.md`` §7.1-§7.3 (G-1, F-3, F-4) and ``-answer-spec2.md`` §3.2-§3.5, §4.2;
plan ``docs/plans/connected-code-graph-p1-answers.md`` P1-24. Deterministic templates only: every sentence is a fixed
template filled from the document, and no fact is model-written.

Two rules hold for every string this module emits:

* a **data field** (``answer.DATA_FIELDS``: anything an attacker can write into source) enters the text only through
  :func:`data`, which is ``answer.fence_data`` (F-3), after ``answer.mask_identifier`` for the F-4 field kinds;
* a value the engine mints (an id, enum, registry code, integer, timestamp) is printed as it is.

The renderers fence every ``DATA_FIELDS`` value and mask the F-4 field kinds, for whatever document they are given.
Fields the engine mints (reason codes, ``endpoint``, ``lang``, ``repoId``, ports, enums) are printed as they are, so
they rely on a document that has passed ``answer.validate``: **callers must run ``validate`` first** (P1-25, P1-26).
The one exception is ``ccg.render.summary_input``, the model-bound output, which validates its own input.
"""
from __future__ import annotations

from collections.abc import Iterator

from ccg.answer import fence_data, mask_identifier

#: ``ContractPoint.contractKind`` values whose ``keyNorm`` is an F-4 field, and the ``field_kind`` they use (§7.3).
_CONTRACT_F4 = frozenset({"config_key", "env_var", "http"})
_STATUS_RANK = {"confirmed": 0, "possible": 1}

#: Plain-language gloss of the reason codes whose meaning is unambiguous; the code itself is always printed too.
REASON_GLOSS = {
    "name-match": "matched by name only, the callee is not resolved",
    "resolution-unverified": "another possible callee of a call that already resolves",
    "runtime-built": "the key is built at run time",
    "target-not-mapped": "the target is in a unit that was never built",
    "example-config": "taken from an example configuration",
    "prompt-mention": "a mention in a prompt",
    "dynamic-import": "a dynamic import",
    "partial-key": "only part of the key matches",
    "unknown-key": "the key is not known",
    "field-truncated": "the field path was truncated",
}
GROUP_TITLE = {
    "other-possible-callees": "Other possible callees of one resolved call site",
    "runtime-built-access": "Runtime-built accesses",
}
PORT_ORIGIN = {"explicit": "stated explicitly", "env-default": "from an environment default", "compose": "from Compose"}
LAUNCH_ORDER = ("argv0", "service", "protocol", "port", "envVar", "mountPath")
DIRECTION_TITLE = {"impact": "What is affected by", "depends-on": "What depends on the answer to"}


def data(text: str, field_kind: str | None = None) -> str:
    """A data field's value as LLM-safe text: masked when it is an F-4 field (``field_kind``), then fenced (F-3)."""
    return fence_data(mask_identifier(field_kind, text) if field_kind else text)


def contract_kind(kind: str) -> str | None:
    return kind if kind in _CONTRACT_F4 else None


def reason_text(code: str) -> str:
    """``code`` with its gloss when it has one (``flow-unavailable:<lang>`` is glossed by its language)."""
    gloss = REASON_GLOSS.get(code)
    if gloss is None and code.startswith("flow-unavailable:"):
        gloss = f"no data-flow facts for {code.partition(':')[2]} in this unit, so the reach is by call only"
    return f"{code} ({gloss})" if gloss else code


def units_of(doc: dict) -> dict[str, dict]:
    return {u["unitId"]: u for u in doc["staleness"]["units"]}


def unit_label(units: dict[str, dict], unit_id: str) -> str:
    """The component label of a unit: its ``rootPath``, fenced."""
    unit = units.get(unit_id)
    return data(unit["rootPath"]) if unit else f"unit {unit_id}"


def all_items(doc: dict) -> Iterator[dict]:
    yield from doc["items"]
    for group in doc["groups"]:
        yield from group["members"]


def has_data_hop(item: dict) -> bool:
    """FR-10/§5.1: read off the chain, never off the stored ``viaData`` flag."""
    return any(h.get("edgeType") == "DATA" for h in item["chain"])


def component_groups(doc: dict, items: list[dict]) -> list[tuple[str, list[tuple[str, list[tuple[int, list[tuple[str, list[dict]]]]]]]]]:
    """``items`` grouped repository, component (unit), distance, then ``confirmed`` before ``possible`` (§6's ``items`` rule).

    Returns ``[(repoId, [(unitId, [(distance, [(status, [item, ...]), ...]), ...]), ...]), ...]``; ties keep document order.
    """
    units = units_of(doc)

    def key(pair):
        ix, it = pair
        return (it["repoId"], units.get(it["unitId"], {}).get("rootPath", ""), it["unitId"], it["distance"],
                _STATUS_RANK[it["status"]], ix)

    out: list = []
    for _, it in sorted(enumerate(items), key=key):
        repo, unit, dist, status = it["repoId"], it["unitId"], it["distance"], it["status"]
        if not out or out[-1][0] != repo:
            out.append((repo, []))
        comps = out[-1][1]
        if not comps or comps[-1][0] != unit:
            comps.append((unit, []))
        dists = comps[-1][1]
        if not dists or dists[-1][0] != dist:
            dists.append((dist, []))
        stats = dists[-1][1]
        if not stats or stats[-1][0] != status:
            stats.append((status, []))
        stats[-1][1].append(it)
    return out


# --- sentences ----------------------------------------------------------------------------------

def anchor_text(a: dict) -> str:
    if a["type"] == "symbol":
        text = f"{data(a['qname'])} in {data(a['path'])}"
        return text + (f", signature {data(a['signature'])}" if "signature" in a else "")
    if a["type"] == "contract":
        return f"{a['kind']} contract {data(a['keyNorm'], contract_kind(a['kind']))}"
    return f"configuration key {data(a['keyPath'], 'keyPath')} in {data(a['path'])}"


def request_text(req: dict) -> str:
    anchors = "; ".join(anchor_text(a) for a in req["anchors"])
    scope = req.get("scope")
    where = ""
    if scope and "repoIds" in scope:
        where = " Scope: repositories " + ", ".join(scope["repoIds"]) + "."
    elif scope:
        where = " Scope: paths " + ", ".join(data(p) for p in scope["paths"]) + "."
    return f"{DIRECTION_TITLE[req['direction']]}: {anchors}.{where}"


def interpretation_lines(doc: dict) -> list[str]:
    """The question as the engine understood it (FR-19), one line per fact."""
    interp = doc["interpretation"]
    lines: list[str] = []
    if interp["status"] == "resolved":
        lines.append(request_text(interp["request"]))
    elif interp["status"] == "ambiguous":
        lines.append(f"The question has several readings ({interp['ambiguityKind']}); none was chosen, so nothing was analysed.")
        for i, opt in enumerate(interp["options"]):
            rank = f" (rank {opt['rank']})" if "rank" in opt else ""
            lines.append(f"Reading {i + 1}{rank}: {request_text(opt)}")
    else:
        lines.append("The question named files that could not be used, so nothing was analysed.")
    for lk in interp.get("lookups", []):
        if lk["state"] == "skipped-file":
            lines.append(f"Not mapped: {data(lk['path'])} in repository {lk['repoId']} ({lk['reason']}).")
        else:
            lines.append(f"Not found: {data(lk['path'])} in repository {lk['repoId']}.")
    prov = interp.get("provenance")
    if prov:
        quant = f", quantization {data(prov['quantization'])}" if "quantization" in prov else ""
        cache = "interpretation cache bypassed" if prov.get("cacheBypass") else (
            "interpretation from the cache" if prov["cacheHit"] else "interpretation not cached")
        lines.append(f"Interpreted by model {data(prov['modelFingerprint'])} ({prov['endpoint']}{quant}); {cache}.")
    return lines


def staleness_lines(doc: dict) -> list[str]:
    lines = []
    for u in doc["staleness"]["units"]:
        head = f"{data(u['rootPath'])} ({u['lang']}, repository {u['repoId']})"
        if u["buildStatus"] == "fresh":
            lines.append(f"{head}: fresh, built {u['builtAt']} from commit {u['builtCommit'][:12]}.")
        elif u["buildStatus"] == "stale-kept":
            lines.append(f"{head}: stale, the build failed at {u['failedAt']} ({data(u['reason'])}); "
                         f"the view kept is from {u['builtAt']}, commit {u['builtCommit'][:12]}.")
        else:
            lines.append(f"{head}: not mapped, never built ({data(u['reason'])}).")
    return lines


def reach_note(doc: dict) -> str | None:
    """``depthBound.unexplored > 0``, read from the two fields (spec2 §4.2)."""
    db = doc["depthBound"]
    if db["unexplored"] > 0:
        return f"Reach stopped at distance {db['bound']} with {db['unexplored']} unexplored edges."
    return None


def counts_line(items: list[dict], groups: list[dict]) -> str:
    conf = sum(1 for i in items if i["status"] == "confirmed")
    line = f"{len(items)} items listed ({conf} confirmed, {len(items) - conf} possible)"
    if groups:
        line += f"; {len(groups)} folded groups with {sum(len(g['members']) for g in groups)} members, all listed below"
    return line + "."


def subject_text(item: dict, units: dict[str, dict]) -> str:
    s = item["subject"]
    if s["type"] == "symbol":
        lines = f"line {s['lineStart']}" if s["lineStart"] == s["lineEnd"] else f"lines {s['lineStart']}-{s['lineEnd']}"
        sig = f", signature {data(s['signature'])}" if "signature" in s else ""
        return f"{data(s['qname'])} in {data(s['path'])} {lines}{sig}"
    if s["type"] == "config":
        return f"configuration key {data(s['keyPath'], 'keyPath')} in {data(s['path'])} line {s['line']}"
    return f"never-built unit {unit_label(units, s['unitId'])}"


def status_text(item: dict) -> str:
    if item["status"] == "possible":
        return f"possible, {reason_text(item['reason'])}"
    return "confirmed"


def distance_text(item: dict) -> str:
    return f"distance {item['distance']} ({item['directness']})"


def launch_text(launch: dict) -> str:
    """Every field the launch fact carries, in a fixed order; an absent field says nothing (§7.1)."""
    parts = []
    for f in LAUNCH_ORDER:
        if f not in launch:
            continue
        v = launch[f]
        kind = f if f in ("mountPath", "service", "envVar") else None
        parts.append(f"{f} {v}" if f == "port" else f"{f} {data(v, kind)}")
    return f"launch {launch['launchKind']}" + (": " + ", ".join(parts) if parts else "")


def _where(h: dict) -> str:
    return f" in {data(h['path'])} line {h['line']}" if "path" in h else ""


def hop_text(h: dict, units: dict[str, dict]) -> str:
    """One hop on one line: what it is, where (file and line), how the walk arrived, and any caveat."""
    k = h["kind"]
    if k == "Symbol":
        what = f"function {data(h['qname'])}"
    elif k == "CallSite":
        what = f"call to {data(h['calleeName'])}"
    elif k == "ContractPoint":
        what = f"{h['contractKind']} contract {data(h['keyNorm'], contract_kind(h['contractKind']))}"
    elif k == "Cond":
        what = "condition" + (f" {data(h['code'])}" if "code" in h else "")
    elif k == "ConfigEntry":
        what = f"configuration key {data(h['keyPath'], 'keyPath')}"
    elif k == "Unit":
        what = f"never-built unit {unit_label(units, h['id'])}"
    elif k == "Link":
        what = "link"
    else:
        what = "parameter" if k == "Param" else "return value"
    if "accessPath" in h and k in ("Param", "Ret", "CallSite"):
        what += f", path {data(h['accessPath'])}"
    out = what + _where(h)
    if "edgeType" in h:
        out += f"; reached by {h['edgeType']} {h['dir']}"
        if h["edgeType"] == "DATA":
            out += " (path-matched data flow)" if h["state"] == "path" else " (data flow)"
        if h.get("candidateCount", 0) >= 2:
            out += f", may call any of {h['candidateCount']}"
        if h.get("status") == "possible":
            out += f"; possible, {reason_text(h['reason'])}"
    return out


def lifted_text(e: dict) -> str:
    extra = []
    if "urlPath" in e:
        extra.append(f"url {data(e['urlPath'])}")
    if "verbPath" in e:
        extra.append(f"verb {data(e['verbPath'])}")
    tail = f" ({', '.join(extra)})" if extra else ""
    return f"LIFTED_VIA {e['ord']}: {e['kind']} {data(e['name'])}{_where(e)}{tail}"


def port_text(b: dict) -> str:
    if "port" not in b:
        return ""
    origin = b["portOrigin"]
    why = f"from the `{b['packId']}` pack's default" if origin == "pack-default" else PORT_ORIGIN[origin]
    return f"; matched port {b['port']}, {why}"


def bound_text(b: dict) -> str:
    """A ``BOUND_BY`` entry; the matched port is the edge's own evidence, never derived from ``launch``."""
    what = f"configuration key {data(b['keyPath'], 'keyPath')}" if "keyPath" in b else launch_text(b["launch"])
    return f"BOUND_BY {b['ord']}: {what}{_where(b)}{port_text(b)}"


def link_detail_lines(h: dict) -> list[str]:
    """A ``Link`` hop expands to its ``LIFTED_VIA`` then ``BOUND_BY`` entries (sorted here: ``(siteId, ord)`` and ``ord``, spec2 §3.2)."""
    lifted = sorted(h.get("lifted", []), key=lambda e: (e["siteId"], e["ord"]))
    bound = sorted(h.get("bound", []), key=lambda b: b["ord"])
    return [lifted_text(e) for e in lifted] + [bound_text(b) for b in bound]


def launch_lines(h: dict) -> list[str]:
    return [launch_text(h["launch"])] if h["kind"] == "ConfigEntry" and "launch" in h else []


def hop_lines(h: dict, units: dict[str, dict]) -> tuple[str, list[str]]:
    """``(the hop's line, its detail lines)``."""
    return hop_text(h, units), link_detail_lines(h) if h["kind"] == "Link" else launch_lines(h)
