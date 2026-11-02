"""S12 (P0-11b) -- slice assembly: cross-unit ``http`` link facts from fact bundles.

Contract: ``docs/plans/connected-code-graph-p0-2.md`` S12 row and SL-1/SL-2; bundle spec v1.3
(``docs/plans/connected-code-graph-bundle-spec3.md`` §5.1, §5.8-§5.12, §6, BF6); proposal §3.3-§3.4;
``docs/plans/connected-code-graph-eval-protocol2-ml.md`` §1.1 (``http`` ``keyNorm``).

The recognizers are hard-coded, the slice is not a pack runtime:

* **fetch consumer** -- every call site named ``fetch``: url at ``a0``, verb at ``a1.method``;
* **role lifting** -- a fixpoint over ``data`` records and call targets that makes the url (and verb)
  role of a ``fetch`` argument a role of the enclosing function's parameter, then of its callers'
  arguments, also through a callee whose return value is derived from a parameter;
* **FastAPI provider** -- ``<recv>.<verb>("<path>")`` decorators, composed with the ``prefix=`` of the
  ``include_router`` call that mounts the enclosing router factory (any depth), recording the
  conditions that control the mounting call.

Nothing else is recognized: no other client library (``axios``, ``requests``), no other framework,
no ``add_api_route``; a url that no value set resolves produces no link (``unresolvedConsumers``).

Usage: ``python assemble.py BUNDLE.jsonl [BUNDLE.jsonl ...] [--out FACTS.json]``
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # cpg/

from ccg.bundle import read_bundle  # noqa: E402
from ccg.ids import link_id  # noqa: E402
from ccg.keynorm import CONSUMER, PROVIDER, mask_http_segments  # noqa: E402

VERBS = frozenset({"GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "TRACE", "CONNECT"})
UNKNOWN = "\x00"  # an unresolved value piece inside a string under construction
MAX_ALTERNATIVES = 64
MAX_REF_DEPTH = 8

# ---------------------------------------------------------------------------------------------
# http keyNorm (spec3 §6 steps 1-8 + §1.8 step 8a). Step 3 (mount composition) is the caller's.
# ---------------------------------------------------------------------------------------------

_UNRESERVED = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~")
_ESCAPE = re.compile(r"%([0-9A-Fa-f]{2})")
_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.\-]*://")
_FIXED = frozenset({"{}", "{**}", CONSUMER, PROVIDER})


def _decode_unreserved(path: str) -> str:
    """Step 5, one pass: decode escapes of unreserved characters, uppercase the rest."""

    def one(m: re.Match) -> str:
        ch = chr(int(m.group(1), 16))
        return ch if ch in _UNRESERVED else "%" + m.group(1).upper()

    return _ESCAPE.sub(one, path)


def _collapse_and_resolve(path: str) -> str:
    """Step 6: collapse ``/`` runs, then RFC 3986 §5.2.4 dot-segment removal (a trailing ``.``/``..`` ends in ``/``)."""
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


_WHOLE_PARAM = (
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


def _param_segment(seg: str) -> str:
    """Step 7 for one segment: only a segment that is wholly a parameter is rewritten."""
    if seg in _FIXED:
        return seg
    for rx, repl in _WHOLE_PARAM:
        if rx.match(seg):
            return repl
    return seg


def normalize_path(path: str, placeholder: str) -> str:
    """Steps 4-7, the consumer-only step 8 (an ``UNKNOWN``-bearing segment), then 8a. ``path`` has no verb."""
    path = path.strip()
    if not path.startswith("/"):
        path = "/" + path
    path = _collapse_and_resolve(_decode_unreserved(path))
    segs = [_param_segment(s) for s in path.split("/")]
    segs = [CONSUMER if UNKNOWN in s else s for s in segs]
    return mask_http_segments("/".join(segs), placeholder)


def normalize_key(raw: str, form: str = "provider") -> str:
    """``keyNorm`` of a writer string ``"<verb> <path>"`` (EP2 §1.1, ``normalize`` in the given form).

    ``form`` is ``provider`` (``{#}``) or ``consumer`` (``{?}``). A string without a leading verb
    word is a bare path with the any-method verb ``*``.

    Step 2 (scheme, authority, query, fragment) is **not** done here: it lives only in
    :func:`url_path_of`, so never call this on a full URL, and do not copy this function as the
    whole writer-side normalizer.
    """
    raw = unicodedata.normalize("NFC", raw).strip()
    verb, sep, rest = raw.partition(" ")
    if sep and not raw.startswith("/") and (verb.upper() in VERBS or verb == "*"):
        v, path = verb.upper(), rest
    else:
        v, path = "*", raw
    if form not in ("provider", "consumer"):
        raise ValueError(f"form must be 'provider' or 'consumer', got {form!r}")
    placeholder = PROVIDER if form == "provider" else CONSUMER
    return f"{v} {normalize_path(path, placeholder)}"


def url_path_of(url: str) -> str:
    """Step 2: the path only -- scheme and authority, query and fragment are dropped."""
    m = _SCHEME.match(url)
    if m:
        rest = url[m.end():]
        slash = min((i for i in (rest.find("/"), rest.find("?"), rest.find("#")) if i >= 0), default=-1)
        url = "" if slash < 0 else rest[slash:]
    for ch in "?#":
        cut = url.find(ch)
        if cut >= 0:
            url = url[:cut]
    return url


# ---------------------------------------------------------------------------------------------
# The corpus: indexes over every record of every bundle.
# ---------------------------------------------------------------------------------------------


def _is_path_prefix(shorter: str, longer: str) -> bool:
    return longer == shorter or (longer.startswith(shorter) and longer[len(shorter)] in ".[")


class Corpus:
    def __init__(self, bundles: list[list[dict]]):
        self.symbols: dict[str, dict] = {}
        self.callsites: dict[str, dict] = {}
        self.params: dict[str, dict] = {}
        self.rets: dict[str, dict] = {}
        self.conds: dict[str, dict] = {}
        self.constants: dict[str, dict] = {}
        self.value_sets: dict[tuple[str, str], list] = {}
        self.data_by_use: dict[str, list[dict]] = defaultdict(list)
        self.decorators: list[dict] = []
        self.controlled_by: dict[str, list[str]] = defaultdict(list)
        self.rets_of: dict[str, list[str]] = defaultdict(list)
        self.callsites_of: dict[str, list[dict]] = defaultdict(list)
        self.callers_of: dict[str, list[dict]] = defaultdict(list)
        for records in bundles:
            for r in records:
                t = r["type"]
                if t == "symbol":
                    self.symbols[r["id"]] = r
                elif t == "callsite":
                    self.callsites[r["id"]] = r
                elif t == "param":
                    self.params[r["id"]] = r
                elif t == "ret":
                    self.rets[r["id"]] = r
                    self.rets_of[r["symbolId"]].append(r["id"])
                elif t == "cond":
                    self.conds[r["id"]] = r
                elif t == "constant":
                    self.constants[r["qname"]] = r
                elif t == "valueSet":
                    self.value_sets[(r["callSiteId"], r["arg"])] = r["values"]
                elif t == "data":
                    self.data_by_use[r["use"]].append(r)
                elif t == "decorator":
                    self.decorators.append(r)
                elif t == "controlledBy":
                    self.controlled_by[r["point"]].append(r["cond"])
        for cid in sorted(self.callsites):
            c = self.callsites[cid]
            self.callsites_of[c["symbolId"]].append(c)
            for t in c["targets"]:
                self.callers_of[t["symbolId"]].append(c)
        for lst in self.rets_of.values():
            lst.sort()

    # -- values ------------------------------------------------------------------------------
    def alternatives(self, element: dict, _seen: tuple = (), _depth: int = 0) -> list[str]:
        """Strings an ``element`` can take; ``UNKNOWN`` marks an unresolved piece."""
        if "const" in element:
            v = element["const"]
            return [str(v)]
        if "ref" in element:
            qn = element["ref"]
            const = self.constants.get(qn)
            if const is None or qn in _seen or _depth >= MAX_REF_DEPTH:
                return [UNKNOWN]
            return self.value_alternatives(const["values"], _seen + (qn,), _depth + 1)
        if "concat" in element or "template" in element:
            parts = [self.alternatives(p, _seen, _depth) for p in element.get("concat") or element["template"]]
            out: list[str] = []
            for combo in itertools.product(*parts):
                out.append("".join(combo))
                if len(out) >= MAX_ALTERNATIVES:
                    return sorted(set(out)) + [UNKNOWN]
            return sorted(set(out))
        return [UNKNOWN]  # importRef, unknown

    def value_alternatives(self, values: list[dict], _seen: tuple = (), _depth: int = 0) -> list[str]:
        out: set[str] = set()
        for el in values:
            out.update(self.alternatives(el, _seen, _depth))
        return sorted(out)

    def slot_alternatives(self, callsite_id: str, path: str) -> list[str] | None:
        values = self.value_sets.get((callsite_id, path))
        return None if values is None else self.value_alternatives(values)

    # -- flow --------------------------------------------------------------------------------
    def flows_into(self, use_id: str, slot_path: str):
        """Yield ``(data, def_path)`` for every ``data`` record that feeds ``use_id`` at ``slot_path``.

        A flow into a prefix of the slot (``a1`` for ``a1.method``) carries the remainder over.
        """
        for d in self.data_by_use.get(use_id, ()):
            if _is_path_prefix(d["toPath"], slot_path):
                yield d, d["fromPath"] + slot_path[len(d["toPath"]):]


# ---------------------------------------------------------------------------------------------
# Role lifting.
# ---------------------------------------------------------------------------------------------

_PARAM_PATH = re.compile(r"^p(\d+)(.*)$")
_RESULT_PATH = re.compile(r"^r(.*)$")


def lift_role(corpus: Corpus, seeds: list[tuple[str, str]]) -> dict[tuple, tuple | None]:
    """Fixpoint of a role from seed ``(callsiteId, argPath)`` slots; returns ``node -> predecessor``.

    Nodes: ``("slot", callsiteId, argPath)`` -- the value at that call argument has the role;
    ``("param", symbolId, "p<i>[.f]")`` -- the function's parameter has it. A predecessor is
    ``(node, kind, extra)`` pointing one step *towards the seed*.
    """
    pred: dict[tuple, tuple | None] = {}
    work: list[tuple] = []
    for cid, path in sorted(seeds):
        node = ("slot", cid, path)
        pred[node] = None
        work.append(node)

    def add(node: tuple, parent: tuple, kind: str, extra=None) -> None:
        if node not in pred:
            pred[node] = (parent, kind, extra)
            work.append(node)

    while work:
        node = work.pop(0)
        if node[0] == "slot":
            _, cid, path = node
            site = corpus.callsites[cid]
            for d, def_path in sorted(corpus.flows_into(cid, path), key=lambda x: x[0]["id"]):
                dp = corpus.params.get(d["def"])
                if dp is not None:
                    m = _PARAM_PATH.match(def_path)
                    if m:
                        add(("param", site["symbolId"], def_path), node, "param-def")
                    continue
                dcs = corpus.callsites.get(d["def"])
                m = _RESULT_PATH.match(def_path)
                if dcs is None or not m:
                    continue
                rest = m.group(1)  # "" or ".status": the call result's own path
                for t in sorted(dcs["targets"], key=lambda t: t["symbolId"]):
                    for rid in corpus.rets_of.get(t["symbolId"], ()):
                        for rd, rdef in sorted(corpus.flows_into(rid, "ret" + rest), key=lambda x: x[0]["id"]):
                            if rd["def"] not in corpus.params:
                                continue
                            pm = _PARAM_PATH.match(rdef)
                            if pm:
                                add(("slot", dcs["id"], f"a{pm.group(1)}{pm.group(2)}"), node, "call-ret",
                                    (t["symbolId"], rdef))
        else:
            _, sym_id, ppath = node
            pm = _PARAM_PATH.match(ppath)
            for caller in corpus.callers_of.get(sym_id, ()):
                add(("slot", caller["id"], f"a{pm.group(1)}{pm.group(2)}"), node, "caller")
    return pred


def lifted_hops(corpus: Corpus, pred: dict, node: tuple, verb_pred: dict | None) -> list[dict]:
    """The ordered ``LIFTED_VIA`` chain for the url slot ``node``: consumer site first, seed last.

    A callsite hop is the consumer call (or the seed); a symbol hop is a function whose parameter
    carries the role (``urlPath`` = that parameter path) or a callee through whose return value the
    role flows. A symbol hop reached through an inner call site names it (``viaSiteId``/``viaLine``).
    """
    chain: list[tuple[tuple, tuple | None]] = []
    cur: tuple | None = node
    while cur is not None:
        chain.append((cur, pred[cur]))
        cur = pred[cur][0] if pred[cur] else None
    verb_paths = _verb_paths(verb_pred)
    hops: list[dict] = []
    for i, (n, p) in enumerate(chain):
        if n[0] == "param":
            hops.append(_symbol_hop(corpus, n[1], n[2], verb_paths.get(("symbol", n[1]))))
            continue
        site = corpus.callsites[n[1]]
        if i == 0 or p is None:
            hops.append({
                "kind": "callsite", "id": site["id"], "path": corpus.symbols[site["symbolId"]]["path"],
                "line": site["line"], "callee": site["calleeName"], "urlPath": n[2],
                "verbPath": verb_paths.get(("callsite", site["id"])),
            })
        elif chain[i - 1][0][0] == "param":
            hops[-1]["viaSiteId"], hops[-1]["viaLine"] = site["id"], site["line"]
        if p is not None and p[1] == "call-ret":
            tid, rdef = p[2]
            hops.append(_symbol_hop(corpus, tid, rdef, verb_paths.get(("symbol", tid))))
    for i, h in enumerate(hops):
        h["ord"] = i
    return hops


def _symbol_hop(corpus: Corpus, sym_id: str, ppath: str, verb_path: str | None) -> dict:
    sym = corpus.symbols[sym_id]
    idx = int(_PARAM_PATH.match(ppath).group(1))
    line = next((p["line"] for p in corpus.params.values() if p["symbolId"] == sym_id and p["index"] == idx),
                sym["lineStart"])
    return {"kind": "symbol", "id": sym_id, "qname": sym["qname"], "path": sym["path"], "line": line,
            "lineStart": sym["lineStart"], "urlPath": ppath, "verbPath": verb_path}


def _verb_paths(verb_pred: dict | None) -> dict:
    """``{(kind, id): path}`` for every node the verb role reached (first path in sort order wins)."""
    out: dict = {}
    for vn in sorted(verb_pred or ()):
        key = ("callsite", vn[1]) if vn[0] == "slot" else ("symbol", vn[1])
        out.setdefault(key, vn[2])
    return out


# ---------------------------------------------------------------------------------------------
# Consumers.
# ---------------------------------------------------------------------------------------------


def consumers(corpus: Corpus) -> tuple[list[dict], list[dict]]:
    """``(consumer sites with a usable key, sites whose url is wholly unresolved)``."""
    seeds_url = [(c["id"], "a0") for c in corpus.callsites.values() if c["calleeName"] == "fetch"]
    seeds_verb = [(c["id"], "a1.method") for c in corpus.callsites.values() if c["calleeName"] == "fetch"]
    url_pred = lift_role(corpus, seeds_url)
    verb_pred = lift_role(corpus, seeds_verb)
    verb_slots: dict[str, list[tuple]] = defaultdict(list)
    for n in verb_pred:
        if n[0] == "slot":
            verb_slots[n[1]].append(n)
    seed_ids = {cid for cid, _ in seeds_url}
    found, unresolved = [], []
    has_child = {p[0] for p in url_pred.values() if p}
    for node in sorted(n for n in url_pred if n[0] == "slot"):
        _, cid, path = node
        site = corpus.callsites[cid]
        alts = corpus.slot_alternatives(cid, path)
        if alts is None or all(a == UNKNOWN for a in alts):
            if node in has_child:
                continue  # an intermediate hop of a longer chain, not a consumer
            unresolved.append({"siteId": cid, "path": corpus.symbols[site["symbolId"]]["path"], "line": site["line"]})
            continue
        chain_slots, cur = [], node
        while cur is not None:
            if cur[0] == "slot":
                chain_slots.append(cur[1])
            cur = url_pred[cur][0] if url_pred[cur] else None
        verbs, verb_partial = _verbs_at(corpus, chain_slots, verb_slots, seed_ids)
        keys = sorted({
            normalize_key(f"{v} {url_path_of(a)}", "consumer")
            for a in alts if a != UNKNOWN for v in verbs
        })
        found.append({
            "siteId": cid, "symbolId": site["symbolId"], "path": corpus.symbols[site["symbolId"]]["path"],
            "line": site["line"], "urlPath": path, "keys": keys,
            "partialAlternatives": verb_partial or UNKNOWN in alts,
            "liftedVia": lifted_hops(corpus, url_pred, node, verb_pred),
        })
    return found, unresolved


def _verbs_at(corpus: Corpus, chain_slots: list[str], verb_nodes: dict, seed_ids: set) -> tuple[list[str], bool]:
    """``(verbs, partial)`` for a consumer: the first call on its lifting chain (consumer side first)
    whose lifted ``...method`` slot resolves; else ``GET`` for a bare ``fetch(url)``; else ``*``.
    ``partial`` is true when that slot also holds an unresolved alternative (``[GET, unknown]``)."""
    for cid in chain_slots:
        alts: set[str] = set()
        partial = False
        for _, _, path in sorted(verb_nodes.get(cid, ())):
            for a in corpus.slot_alternatives(cid, path) or ():
                if a == UNKNOWN:
                    partial = True
                else:
                    v = a.upper()
                    alts.add(v if v in VERBS else "*")
        if alts:
            return sorted(alts), partial
    seed = chain_slots[-1]
    if seed in seed_ids and not any(k[0] == seed and k[1].startswith("a1") for k in corpus.value_sets) \
            and not any(d["toPath"].startswith("a1") for d in corpus.data_by_use.get(seed, ())):
        return ["GET"], False  # fetch(url) with no init object
    return ["*"], False


# ---------------------------------------------------------------------------------------------
# Providers.
# ---------------------------------------------------------------------------------------------

_ROUTE_VERBS = {"get", "head", "post", "put", "patch", "delete", "options", "trace"}


def _decorator_verb(name: str) -> str | None:
    parts = name.split(".")
    if len(parts) != 2:
        return None
    if parts[1] in _ROUTE_VERBS:
        return parts[1].upper()
    if parts[1] in ("api_route", "route"):
        return "*"
    return None


def _include_sites(corpus: Corpus) -> dict[str, list[dict]]:
    """``factory symbolId -> include_router sites that mount its result`` (a0 <- call result)."""
    out: dict[str, list[dict]] = defaultdict(list)
    for cid in sorted(corpus.callsites):
        site = corpus.callsites[cid]
        if site["calleeName"] != "include_router":
            continue
        for d in corpus.data_by_use.get(cid, ()):
            dcs = corpus.callsites.get(d["def"])
            if d["toPath"] == "a0" and d["fromPath"] == "r" and dcs is not None:
                for t in dcs["targets"]:
                    out[t["symbolId"]].append(site)
    return out


def _mount_chains(corpus: Corpus, factory_id: str, sites: dict, _stack=()) -> list[list[dict]]:
    chains = []
    for s in sites.get(factory_id, ()):
        if s["symbolId"] in _stack:
            continue
        outer = _mount_chains(corpus, s["symbolId"], sites, _stack + (factory_id,))
        for o in outer or [[]]:
            chains.append(o + [s])
    return chains


def providers(corpus: Corpus) -> tuple[list[dict], list[dict]]:
    """``(providers, dropped)``: one record per (handler, keyNorm); ``dropped`` are unresolved alternatives."""
    sites = _include_sites(corpus)
    out: dict[tuple[str, str], dict] = {}
    dropped: list[dict] = []
    for dec in sorted(corpus.decorators, key=lambda d: d["id"]):
        verb = _decorator_verb(dec["name"])
        route_arg = next((a for a in dec["args"] if a["arg"] == "a0"), None)
        if verb is None or route_arg is None:
            continue
        handler = corpus.symbols[dec["targetId"]]
        parent_id = handler.get("parentId")
        chains = _mount_chains(corpus, parent_id, sites) if parent_id else []
        if chains:
            mounts = [(c, "mounted") for c in chains]
        elif parent_id and any(c["calleeName"] == "FastAPI" for c in corpus.callsites_of.get(parent_id, ())):
            mounts = [([], "app")]
        else:
            mounts = [([], "unmounted-router")]
        routes = corpus.value_alternatives(route_arg["values"])
        for chain, how in mounts:
            prefix_sets = [_prefix_alternatives(corpus, s) for s in chain]
            for combo in itertools.product(*prefix_sets, routes):
                if any(UNKNOWN in part for part in combo):
                    dropped.append({"handlerId": handler["id"], "line": dec["line"], "reason": "unresolved-part"})
                    continue
                key = normalize_key(f"{verb} {''.join(combo)}", "provider")
                rec = out.setdefault((handler["id"], key), {
                    "handlerId": handler["id"], "qname": handler["qname"], "path": handler["path"],
                    "line": dec["line"], "decorator": dec["name"], "key": key, "mounts": [],
                })
                mount = {
                    "how": how,
                    "sites": [{"siteId": s["id"], "path": corpus.symbols[s["symbolId"]]["path"], "line": s["line"]}
                              for s in chain],
                    "conditions": [
                        {"siteId": s["id"], "line": corpus.conds[c]["line"], "code": corpus.conds[c]["code"]}
                        for s in chain for c in sorted(corpus.controlled_by.get(s["id"], ()))
                    ],
                }
                if mount not in rec["mounts"]:
                    rec["mounts"].append(mount)
    result = sorted(out.values(), key=lambda p: (p["key"], p["handlerId"]))
    for p in result:
        p["mounts"].sort(key=lambda m: json.dumps(m, sort_keys=True))
        p["status"], p["reason"] = (
            ("confirmed", "key-match") if any(m["how"] != "unmounted-router" for m in p["mounts"])
            else ("possible", "partial-key")
        )
    return result, dropped


def _prefix_alternatives(corpus: Corpus, site: dict) -> list[str]:
    values = corpus.value_sets.get((site["id"], "kw.prefix"))
    return [""] if values is None else corpus.value_alternatives(values)


# ---------------------------------------------------------------------------------------------
# Matching and links.
# ---------------------------------------------------------------------------------------------


def _split(key: str) -> tuple[str, list[str]]:
    verb, _, path = key.partition(" ")
    return verb, path.split("/")[1:]


def match_keys(provider_key: str, consumer_key: str) -> tuple[str, str, str] | None:
    """``(status, reason, matcher)`` if the keys relate, else ``None``.

    Both known and matching -> ``confirmed``; any ``{?}``, a ``*`` verb on either side or a trailing-slash
    difference -> ``possible``/``partial-key``. ``{#}`` matches only a consumer ``{?}``; ``{}`` one
    non-empty segment; ``{**}`` the one or more remaining segments.
    """
    pv, pp = _split(provider_key)
    cv, cp = _split(consumer_key)
    partial = False
    if pv != cv:
        if "*" not in (pv, cv):
            return None
        partial = True
    got = _match_segments(pp, cp)
    if got is None:
        return None
    partial = partial or got
    matcher = "exact" if provider_key == consumer_key else "path-template"
    return ("possible", "partial-key", matcher) if partial else ("confirmed", "key-match", matcher)


def _match_segments(p: list[str], c: list[str]) -> bool | None:
    """``None`` no match, ``False`` a full match, ``True`` a partial one."""
    partial = False
    i = 0
    while i < len(p) and i < len(c):
        ps, cs = p[i], c[i]
        if ps == "{**}":
            rest = c[i:]
            return partial or any(s == CONSUMER for s in rest)
        if cs == CONSUMER:
            partial = True
        elif ps == PROVIDER:
            return None
        elif ps == "{}":
            if cs == "":
                return None
        elif ps != cs:
            return None
        i += 1
    if i == len(p) == len(c):
        return partial
    tail_p, tail_c = p[i:], c[i:]
    if (tail_p == [""] and not tail_c) or (tail_c == [""] and not tail_p):
        return True  # `/x` against `/x/`
    return None


def assemble(bundles: list[list[dict]]) -> dict:
    """Cross-unit ``http`` link facts over the given bundles' records."""
    corpus = Corpus(bundles)
    sites, unresolved = consumers(corpus)
    provs, dropped = providers(corpus)
    grouped: dict[str, dict] = {}
    unmatched: list[dict] = []
    for site in sites:
        site_matched = False
        for ckey in site["keys"]:
            cands = [(p, m) for p in provs if (m := match_keys(p["key"], ckey))]
            for p, (status, reason, matcher) in cands:
                if status == "confirmed" and (p["status"] == "possible" or site["partialAlternatives"]):
                    status, reason = "possible", "partial-key"  # an unmounted router, or an unresolved alternative
                site_matched = True
                lid = link_id("http", p["key"], p["handlerId"], site["symbolId"])
                link = grouped.setdefault(lid, {
                    "type": "link", "id": lid, "kind": "http", "keyNorm": p["key"],
                    "provider": {"anchorId": p["handlerId"], "qname": p["qname"], "path": p["path"],
                                 "line": p["line"], "mounts": p["mounts"]},
                    "consumer": {"anchorId": site["symbolId"],
                                 "qname": corpus.symbols[site["symbolId"]]["qname"], "path": site["path"]},
                    "sites": [],
                })
                link["sites"].append({
                    "siteId": site["siteId"], "line": site["line"], "consumerKey": ckey, "status": status,
                    "reason": reason, "matcher": matcher,
                    "liftedVia": [dict(h, siteId=site["siteId"]) for h in site["liftedVia"]],
                })
        if not site_matched:
            unmatched.append({k: site[k] for k in ("siteId", "path", "line", "keys")})
    links = []
    for lid in sorted(grouped):
        link = grouped[lid]
        link["sites"].sort(key=lambda s: (s["siteId"], s["consumerKey"]))
        best = min(link["sites"], key=lambda s: (s["status"] != "confirmed", s["siteId"]))
        link["status"], link["reason"], link["matcher"] = best["status"], best["reason"], best["matcher"]
        links.append(link)
    return {
        "providers": provs, "consumers": sites, "links": links, "unmatchedConsumers": unmatched,
        "unresolvedConsumers": unresolved, "droppedProviderAlternatives": dropped,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("bundles", nargs="+")
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    facts = assemble([read_bundle(b) for b in args.bundles])
    text = json.dumps(facts, sort_keys=True, indent=1)
    if args.out:
        Path(args.out).write_text(text + "\n")
    else:
        print(text)
    print(f"links={len(facts['links'])} providers={len(facts['providers'])} consumers={len(facts['consumers'])} "
          f"unmatched={len(facts['unmatchedConsumers'])}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
