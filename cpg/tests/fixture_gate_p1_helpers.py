"""P1-41 dispatch B, pure helpers (no Docker, no store): the P1 row set, the expectations read from the manifests, and the row checkers.

Plan: `docs/plans/connected-code-graph-p1-gates.md` P1-41; `docs/test-plans/connected-code-graph-fixtures.md` §5-§7; `docs/test-plans/connected-code-graph-p1.md` §6.
Nothing here is derived from the code under test: every expected status, reason, provider set, distance and item comes from the manifests
(`cpg/fixtures/system1/expected/*.yaml`, or the directory named by ``CCG_GATE_P1_EXPECTED``, the mutation-test hook), through `fixtures/materialize.py`.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

CPG = Path(__file__).resolve().parents[1]


def load_materialize():
    import importlib.util
    import sys
    if "ccg_fixture_materialize_gate" in sys.modules:
        return sys.modules["ccg_fixture_materialize_gate"]
    spec = importlib.util.spec_from_file_location("ccg_fixture_materialize_gate", CPG / "fixtures" / "materialize.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


M = load_materialize()


def manifests():
    d = os.environ.get("CCG_GATE_P1_EXPECTED")
    return M.load_all(Path(d)) if d else M.load_all()


def merged_attr(held: dict, attr: str):
    return next((r[attr] for r in held.values() if r.get(attr) is not None), None)


def p1_rows(mfs) -> dict:
    """{row id: {repo: row}} of the rows whose earliest gate is P1 (distinct ids; every manifest that holds the id is merged)."""
    return {rid: h for rid, h in M.holders(mfs).items() if merged_attr(h, "gate") == "P1"}


def raw_counts(mfs) -> tuple[int, int]:
    """(raw row entries, raw entries with gate P1): the second definition (holder copies counted separately)."""
    rows = [r for m in mfs.values() for r in m["rows"]]
    return len(rows), sum(1 for r in rows if r.get("gate") == "P1")


def split_reason(reason) -> tuple[str | None, str | None]:
    """``runtime-built / unknown-key`` or ``{edge, link}`` -> (edge reason, link reason); a single reason is both."""
    if isinstance(reason, dict):
        return reason.get("edge"), reason.get("link")
    if isinstance(reason, str) and " / " in reason:
        a, b = reason.split(" / ", 1)
        return a.strip(), b.strip()
    return reason, reason


def expected_form(held: dict) -> dict:
    """The P1 form of a row: the `p1Form` where it carries one, the manifest form otherwise. Keys: status, edge, link, linked (False: no Link), key, consumes_fact (False: the consumer has no CONSUMES fact, so no edge reason)."""
    rows = list(held.values())
    p1 = next((r["p1Form"] for r in rows if r.get("p1Form")), None)
    src = p1 if p1 else {}
    status = src.get("status", src.get("st")) or merged_attr(held, "status") or merged_attr(held, "st")
    reason = src.get("reason") or merged_attr(held, "reason")
    edge, link = split_reason(reason)
    return {"status": status, "edge": edge, "link": link, "linked": src.get("linked", True), "key": src.get("key"), "p1Form": bool(p1),
            "consumes_fact": src.get("consumes_fact", True)}


def short(repo: str) -> str:
    return repo.removeprefix("fx-")


def loc(repo: str, path: str, qname: str) -> str:
    return f"{short(repo)}:{path}::{qname}"


def locs_of(mfs, rid: str, fields: tuple[str, ...]) -> list[str]:
    seen, out = set(), []
    for r in M.holders(mfs)[rid].values():
        for f, l in M.row_locators(r):
            if f in fields and l.get("qname"):
                s = loc(l["repo"], l["path"], l["qname"])
                if s not in seen:
                    seen.add(s)
                    out.append(s)
    return out


def consumers_of(mfs, rid):
    return locs_of(mfs, rid, ("consumer", "C", "locator"))


def providers_of(mfs, rid):
    out = locs_of(mfs, rid, ("provider", "P"))
    if out:
        return out
    true = []
    for r in M.holders(mfs)[rid].values():
        true += [loc(t["repoId"], t["path"], t["qname"].split("::", 1)[-1]) for t in r.get("trueTarget", []) if t.get("qname")]
    return true


# ---- link records (both instruments produce these) ------------------------------------------------------------------------------------
@dataclass(frozen=True)
class LinkRec:
    consumer: str          # "svc:path::qname"
    provider: str          # same form, or "unit:<id>" for a provider in an unmapped unit
    kind: str
    status: str
    reason: str
    bound: tuple = ()      # BOUND_BY ordinals, sorted
    via_key: str | None = None


def is_under(consumer: str, base: str) -> bool:
    return consumer == base or consumer.startswith(base + ".<anon>#")


def recs_from(recs, consumer_locs) -> list[LinkRec]:
    return [r for r in recs if any(is_under(r.consumer, c) for c in consumer_locs)]


#: BOUND_BY ordinals the P1 binding contract (`docs/plans/connected-code-graph-p1-linking.md` P1-53, P1-56) gives each bound row's every Link
BOUND_ORDS = {"LNK-45": [0, 1], "LNK-16": [0, 1], "LNK-02": [0], "LNK-06": [0], "LNK-12": [0], "AC6-EX-4": [0, 1, 2], "AC6-EX-1": [0, 1],
              "LNK-19": [], "LNK-20": []}
#: the rows asserted as a Link from the consumer to exactly the provider set, in the P1 form of the manifest
LINK_ROWS_PREFIX = ("LNK", "KN", "AC6-EX", "AC6-PM", "TW-2")
FANOUT_ROWS = ("AC6-URL-1", "AC6-URL-2", "AC6-URL-3")


def link_row_violations(rid: str, form: dict, consumers: list[str], providers: list[str], recs, *, expect_key: str | None = None,
                        superset_ok: bool = False) -> list[str]:
    """One Link-level row against the Link records of one instrument: the Link set from the consumer is exactly the provider set (or empty for a
    `linked: false` form), every Link at the P1 form's status and link reason, the BOUND_BY ordinals the contract names, the consumer key (KN rows,
    store instrument only)."""
    got = recs_from(recs, consumers)
    bad: list[str] = []
    if not form["linked"]:
        if got:
            bad.append(f"{rid}: expected no Link (p1Form linked: false), got {[(r.provider, r.status, r.reason) for r in got]}")
        return bad
    have = {r.provider for r in got}
    if (not set(providers) <= have) if superset_ok else (sorted(have) != sorted(providers)):
        bad.append(f"{rid}: Links from {consumers} go to {sorted(have)}, expected {'at least' if superset_ok else 'exactly'} {sorted(providers)}")
    for r in got:
        if (r.status, r.reason) != (form["status"], form["link"]):
            bad.append(f"{rid}: Link to {r.provider} is {r.status}/{r.reason}, expected {form['status']}/{form['link']}")
        if rid in BOUND_ORDS and list(r.bound) != BOUND_ORDS[rid]:
            bad.append(f"{rid}: Link to {r.provider} has BOUND_BY ordinals {list(r.bound)}, expected {BOUND_ORDS[rid]}")
        if expect_key is not None and r.via_key not in (None, expect_key):
            bad.append(f"{rid}: Link to {r.provider} carries consumer key {r.via_key!r}, expected {expect_key!r}")
    return bad


def fanout_violations(rid: str, form: dict, consumers: list[str], recs) -> list[str]:
    """AC6-URL-n: a keyless consumer: Links to more than one provider, every one `possible`/`unknown-key` (never `confirmed`)."""
    got = recs_from(recs, consumers)
    bad = []
    if len({r.provider for r in got}) < 2:
        bad.append(f"{rid}: expected a fan-out (>= 2 providers) from {consumers}, got {len({r.provider for r in got})}")
    for r in got:
        if (r.status, r.reason) != (form["status"], form["link"]):
            bad.append(f"{rid}: Link to {r.provider} is {r.status}/{r.reason}, expected {form['status']}/{form['link']}")
    return bad


# ---- closed world ---------------------------------------------------------------------------------------------------------------------
CONSUMER_FIELDS = ("consumer", "C", "locator", "X", "Y", "listed")


def manifest_locators(mfs) -> tuple[set[str], set[str]]:
    """(every locator of every row of every manifest at every gate, the subset in consumer-ish fields)."""
    anyl, cons = set(), set()
    for h in M.holders(mfs).values():
        for r in h.values():
            for f, l in M.row_locators(r):
                if not l.get("qname"):
                    continue
                s = loc(l["repo"], l["path"], l["qname"])
                anyl.add(s)
                if f in CONSUMER_FIELDS:
                    cons.add(s)
    return anyl, cons


def unplanted(recs, known: set[str]) -> list[LinkRec]:
    return [r for r in recs if not any(is_under(r.consumer, k) for k in known)]


# ---- walk rows ------------------------------------------------------------------------------------------------------------------------
def _key(repo: str, path: str, qname: str) -> tuple[str, str]:
    q = qname if qname.startswith(path + "::") else f"{path}::{qname}"
    return repo, q


def _from_loc(text: str) -> tuple[str, str]:
    d = M.parse_locator(text)
    return _key(d["repo"], d["path"], d["qname"])


@dataclass
class AskSpec:
    name: str
    owner: str                      # the row that carries the ask-level assertions (exact set, notices, bound)
    use_case: str
    anchor: tuple[str, str]
    items: dict = field(default_factory=dict)     # (repo, qname) -> (distance, status, reason, row id)
    absent: dict = field(default_factory=dict)    # (repo, qname) -> row id (a row with listed: false)
    notices: list = field(default_factory=list)


def _norm_reason(status: str, reason):
    return reason if reason else ("resolved-call" if status == "confirmed" else None)


def walk_specs(mfs) -> dict[str, AskSpec]:
    """The six asks of P1-41 (CF-DF, CF-DF3, CF-CD forward; RV-1..3 depends-on) and their expected item sets, from the rows' P1 forms."""
    held = M.holders(mfs)
    specs: dict[str, AskSpec] = {}

    def row(rid):
        return next((r for r in held[rid].values() if r.get("p1Form")), next(iter(held[rid].values())))

    def anchor_of(rid):
        c = next(r["changed"] for r in held[rid].values() if r.get("changed"))
        return _from_loc(c) if isinstance(c, str) else _key(c["repoId"], c["path"], c["qname"])

    def locd(d):
        return _key(d["repoId"], d["path"], d["qname"])

    def notices(p1):
        return sorted(l for n in p1.get("notices", []) for l in n["langs"])

    s = specs["CF-DF"] = AskSpec("CF-DF", "DF-1", "impact", anchor_of("DF-1"))
    for rid in ("DF-1", "DF-2", "DF-4"):
        p1 = row(rid)["p1Form"]
        s.items[locd(p1["locator"])] = (p1["distance"], p1["status"], p1["reason"], rid)
        s.notices = notices(p1)
    s = specs["CF-DF3"] = AskSpec("CF-DF3", "DF-3", "impact", anchor_of("DF-3"))
    p1 = row("DF-3")["p1Form"]
    s.items[_from_loc(p1["loc"])] = (p1["dist"], p1["st"], p1["reason"], "DF-3")
    for e in p1.get("extras", []) + p1.get("calleeRun", []):
        s.items[_from_loc(e["loc"])] = (e["dist"], e["st"], e["reason"], "DF-3")
    s.notices = notices(p1)
    s = specs["CF-CD"] = AskSpec("CF-CD", "CD-2", "impact", anchor_of("CD-2"))
    for rid in ("CD-1", "CD-2", "CD-3", "CD-4"):
        p1 = row(rid)["p1Form"]
        s.notices = notices(p1)
        if p1.get("listed") is False:
            s.absent[locd(next(iter(held[rid].values()))["locator"])] = rid
        if p1.get("caller"):
            c = p1["caller"]
            s.items[locd(c["locator"])] = (c["distance"], c["status"], c["reason"], rid)
        for c in p1.get("calleeRun", []):
            s.items[locd(c["locator"])] = (c["distance"], c["status"], c["reason"], rid)
    for rid in ("RV-1", "RV-2", "RV-3"):
        base = next(iter(held[rid].values()))
        p1 = row(rid)["p1Form"]
        sub = base["subject"]
        s = specs[rid] = AskSpec(rid, rid, "depends-on", _key(sub["repoId"], sub["path"], sub["qname"]))
        over = {locd(d["locator"]): d for d in p1.get("dependencies", [])}
        for d in base["dependencies"]:
            k = locd(d["locator"])
            d2 = over.get(k, d)
            s.items[k] = (d2["distance"], d2["status"], _norm_reason(d2["status"], d2.get("reason")), rid)
        for d in p1.get("extras", []):
            s.items[locd(d["locator"])] = (d["distance"], d["status"], d["reason"], rid)
        s.notices = notices(p1)
    return specs


def answer_items(doc: dict) -> dict:
    """{(repo, qname): (distance, status, reason)} of an answer's items; a confirmed item with no reason is `resolved-call`."""
    out = {}
    for i in doc["items"]:
        out[(i["repoId"], i["subject"]["qname"])] = (i["distance"], i["status"], _norm_reason(i["status"], i.get("reason")))
    return out


def walk_violations(spec: AskSpec, doc: dict, rid: str) -> list[str]:
    """Row-level: the row's own items are in the answer at the expected distance, status and reason. For the owner row, also: no other item, no
    item a `listed: false` row says is absent, the notices' languages, `depthBound.bound` 2."""
    got = answer_items(doc)
    bad = []
    for k, (d, st, rs, owner) in spec.items.items():
        if owner != rid:
            continue
        if k not in got:
            bad.append(f"{rid}: {spec.name}: expected item {k[1]} (d{d} {st}/{rs}) is not listed")
        elif got[k] != (d, st, rs):
            bad.append(f"{rid}: {spec.name}: item {k[1]} is d{got[k][0]} {got[k][1]}/{got[k][2]}, expected d{d} {st}/{rs}")
    for k, owner in spec.absent.items():
        if owner == rid and k in got:
            bad.append(f"{rid}: {spec.name}: {k[1]} is listed (d{got[k][0]}), the P1 form says not listed")
    if rid == spec.owner:
        for k in sorted(set(got) - set(spec.items)):
            bad.append(f"{rid}: {spec.name}: unplanted item {k[1]} (d{got[k][0]} {got[k][1]}/{got[k][2]}): a row's P1 form is exact")
        langs = sorted(l for n in doc.get("notices", []) if n["code"] == "flow-not-enumerated" for l in n["langs"])
        if langs != spec.notices:
            bad.append(f"{rid}: {spec.name}: notices flow-not-enumerated langs {langs}, expected {spec.notices}")
        if doc["depthBound"]["bound"] != 2:
            bad.append(f"{rid}: {spec.name}: depthBound.bound {doc['depthBound']['bound']}, expected 2")
        if spec.name == "CF-DF" and doc["depthBound"]["unexplored"] < 1:
            bad.append(f"{rid}: {spec.name}: depthBound.unexplored {doc['depthBound']['unexplored']}, expected >= 1")
    return bad


# ---- omitted files (SK) ---------------------------------------------------------------------------------------------------------------
def sk_expectation(held: dict) -> tuple[str, str, str, str]:
    """(repo, path, outcome, reason) of the OFF form (the build the runner makes: `jssrc2cpg` has no ignore override, so ON is not executable)."""
    row = next(iter(held.values()))
    repo = next(iter(held))
    p = row["path"]
    if isinstance(p, dict):          # fx-svc rows: file record, not listed
        return p["repoId"], p["path"], "included", "none"
    off = row["off"]
    return repo, p, off["outcome"], off["reason"]


def sk_violations(rid: str, exp, skipped: dict, files: set, units_roots: set) -> list[str]:
    """``skipped``: {(repo, path): [reasons]} of the store's SkippedFile nodes; ``files``: {(repo, path)} of File nodes."""
    repo, path, outcome, reason = exp
    got_reasons = skipped.get((repo, path), [])
    has_file = (repo, path) in files
    bad = []
    if outcome in ("listed", "listed_once"):
        if got_reasons != [reason]:
            bad.append(f"{rid}: {path} listed {got_reasons}, expected exactly [{reason!r}]")
        if has_file:
            bad.append(f"{rid}: {path} is listed but also has a file record")
    elif outcome == "included":
        if got_reasons:
            bad.append(f"{rid}: {path} is listed {got_reasons}, expected included with a file record")
        if not has_file:
            bad.append(f"{rid}: {path} has no file record, expected included")
    else:                                            # not_listed, not_in_unit_list
        if got_reasons:
            bad.append(f"{rid}: {path} is listed {got_reasons}, expected not listed")
        if has_file:
            bad.append(f"{rid}: {path} has a file record, expected none")
        if outcome == "not_in_unit_list" and any(path.startswith(r.rstrip('/') + "/") for r in units_roots if r not in ("", ".")):
            bad.append(f"{rid}: {path} lies under a unit root {sorted(r for r in units_roots if path.startswith(r.rstrip('/') + '/'))}, expected outside every unit")
    return bad
