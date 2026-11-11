"""Merge a SCIP index into the call sites of a bundle (C-1001 P1-15).

The rule is the method note's (``docs/plans/connected-code-graph-p1-ml.md`` §2.2), fixed and never tuned. Per call
site it compares Joern's in-corpus targets ``J`` (the record's ``targets``) with SCIP's target ``S`` at the same
``(file, line, col)``; a target counts only if it is a symbol of the bundle (``known``: qname -> symbolId, §4.1 rule
8), so a SCIP ``local`` or external result is "SCIP external":

| case | result |
|---|---|
| ``S`` in ``J`` | ``S`` is the target, ``confirmed``/``resolved-call``; the other ``J`` members are candidates |
| ``J`` empty, ``S`` present | ``S`` is the target, ``confirmed``/``resolved-call`` |
| ``S`` not in ``J``, or SCIP external while ``J`` is in-corpus | no target; the site is ``possible``/``name-match``; every ``J`` member and ``S`` is a candidate |
| SCIP silent, or external with ``J`` empty | the record is unchanged |

**What a disagreement does to the bundle.** The site keeps ``possible``/``name-match`` and no target (the schema's
``callsite.reason`` has no ``resolution-unverified`` and the format is not changed). Joern's and SCIP's targets stay
reachable through ``CALLS_NAME`` -> ``Name`` -> ``RESOLVES_TO``, which the assembler derives from ``calleeName``;
the reader labels a candidate of a *resolved* site ``possible``/``resolution-unverified`` at read time. So
``MergeResult.candidates`` is a **merge-time report, not a bundle input**: the evidence run reports
``counts["disagree"]`` and the share of Joern targets demoted by it (a Joern target of a disagreeing site is no longer
a ``CALLS`` edge). Its entries carry ``possible``/``resolution-unverified`` as the report's own labels. Joern's targets
are always inside the record's targets plus the candidates (``R_m >= R_j`` for the function; for the pipeline it holds
through ``CALLS_NAME``, except for a candidate whose symbol name differs from ``calleeName``, an aliased import).

**SCIP is an unauthenticated writer.** The indexer ran staged code, so a hostile `.scip` can forge ``confirmed`` for
in-corpus symbols (it cannot point outside the corpus: only symbols of ``known`` count). The measures on the truth data
bound the benign case only.

**Reported, never minted.** A SCIP reference at a position with no Joern call site comes back in ``scip_only`` (a call
site's id needs the enclosing symbol and the ordinal; P1 drops these and counts them; they include non-call references
to callables, e.g. an import, which ``scip-typescript`` does not flag with the import role). Two distinct in-corpus SCIP
targets at one position are ``ambiguous``: the record is left alone. A site on a line with a character above U+FFFF
before the callee (``positions_from_bundle``'s ``astral`` set) is left alone and counted ``astral_skipped``: SCIP columns
are code points for BMP text on both tools (verified), UTF-16 units otherwise (unverified).

**Constructors.** The bundle names a constructor ``<class qname>.__init__`` (python) and ``<class qname>.<init>``
(javascript/typescript: the stored real bundle ``tests/data/scip/real_ts.bundle.jsonl`` has ``Box.<init>``).
``scip-python`` reports ``Box()`` as the type ``Box#``; ``scip-typescript`` reports ``new Box()`` as
``Box#`<constructor>`().``. Both map to the bundle's constructor symbol; a class that has none is counted
``ctor_unmapped`` and is not synthesised.
"""
from __future__ import annotations

import copy
from collections import Counter
from dataclasses import dataclass, field
from typing import Collection, Mapping, NamedTuple, Sequence

from extractors.indexers.scip import ScipIndex, ScipTarget

#: the constructor's name in the bundle, per language (stored real bundle: ``web/a.ts::Box.<init>``; python ``__init__``)
INIT_NAME = {"python": "__init__", "javascript": "<init>"}
SCIP_CTOR = "<constructor>"                    # scip-typescript's descriptor for a constructor
CANDIDATE_STATUS, CANDIDATE_REASON = "possible", "resolution-unverified"
DISAGREE_SITE_REASON = "name-match"          # the schema-valid site-level reason (see the module docstring)


@dataclass(frozen=True)
class ScipOnly:
    file: str
    line: int
    col: int
    qname: str
    symbolId: str


@dataclass
class MergeResult:
    callsites: list[dict]
    candidates: dict[str, list[dict]] = field(default_factory=dict)
    scip_only: list[ScipOnly] = field(default_factory=list)
    counts: Counter = field(default_factory=Counter)


def _resolve(t: ScipTarget, known: Mapping[str, str], prefix: str, lang: str,
             counts: Counter | None = None) -> tuple[str, str] | None:
    """``(qname, symbolId)`` of an in-corpus SCIP target, or None (local, external, or not a bundle symbol).
    A class (a type) and a ``<constructor>`` descriptor both resolve to the class's constructor symbol."""
    if t.local or t.file is None or not t.chain:
        return None
    base = t.file if prefix in ("", ".") else f"{prefix}/{t.file}"
    init = INIT_NAME.get(lang)
    chain = t.chain
    if chain[-1] == SCIP_CTOR and init:
        chain = chain[:-1] + (init,)
    q = f"{base}::{'.'.join(chain)}"
    if t.is_type and init:
        sid = known.get(f"{q}.{init}")
        if sid is None:
            if counts is not None and q in known:
                counts["ctor_unmapped"] += 1
            return None
        return f"{q}.{init}", sid
    sid = known.get(q)
    return (q, sid) if sid is not None else None


def _candidate(target: dict) -> dict:
    return {**target, "status": CANDIDATE_STATUS, "reason": CANDIDATE_REASON}


def merge_scip(callsites: Sequence[dict], positions: Mapping[str, tuple[str, int, int]], scip: ScipIndex,
               known: Mapping[str, str], *, prefix: str, lang: str, astral: Collection[str] = frozenset()) -> MergeResult:
    """Apply the rule to ``callsites`` (not mutated). ``positions`` maps ``callSiteId`` to its callee identifier's
    ``(file relative to the indexed directory, 1-based line, 0-based column)``; ``prefix`` is the unit's repo-relative
    root (``""``/``"."`` for a root unit), which turns an indexed-relative file into a bundle qname. ``astral`` holds the
    ids of sites whose line has a character above U+FFFF before the callee: they are left alone (``astral_skipped``)."""
    res = MergeResult([])
    sited = set(positions.values())
    for rec in callsites:
        pos = positions.get(rec["id"])
        if rec["id"] in astral:
            res.callsites.append(rec)
            res.counts["astral_skipped"] += 1
            continue
        refs = scip.refs.get(pos) if pos is not None else None
        if not refs:
            res.callsites.append(rec)
            res.counts["silent"] += 1
            continue
        in_corpus = {r[1]: r for r in (_resolve(t, known, prefix, lang, res.counts) for t in refs) if r is not None}
        joern = {t["symbolId"]: t for t in rec.get("targets", [])} if rec.get("resolution") == "resolved" else {}
        if len(in_corpus) > 1:
            res.callsites.append(rec)
            res.counts["ambiguous"] += 1
            continue
        if not in_corpus and not joern:
            res.callsites.append(rec)
            res.counts["silent"] += 1
            continue
        out = copy.deepcopy(rec)
        out.pop("externalTarget", None)
        if in_corpus:
            (sid, (q, _)), = in_corpus.items()
            if not joern or sid in joern:
                out.update(resolution="resolved", targets=[joern.get(sid, {"symbolId": sid, "qname": q})],
                           status="confirmed", reason="resolved-call")
                others = [_candidate(t) for k, t in sorted(joern.items()) if k != sid]
                res.counts["agree" if joern else "upgrade"] += 1
                if others:
                    res.candidates[rec["id"]] = others
                res.callsites.append(out)
                continue
            cands = {**joern, sid: {"symbolId": sid, "qname": q}}
        else:
            cands = joern
        out.update(resolution="unresolved", targets=[], status=CANDIDATE_STATUS, reason=DISAGREE_SITE_REASON)
        res.candidates[rec["id"]] = [_candidate(t) for _k, t in sorted(cands.items())]
        res.counts["disagree"] += 1
        res.callsites.append(out)
    for pos, targets in sorted(scip.refs.items()):
        if pos in sited:
            continue
        for t in targets:
            r = _resolve(t, known, prefix, lang) if (t.callable or t.is_type) else None
            if r is not None:
                res.scip_only.append(ScipOnly(pos[0], pos[1], pos[2], r[0], r[1]))
                res.counts["scip_only"] += 1
    return res


class Positions(NamedTuple):
    sites: dict[str, tuple[str, int, int]]       # callSiteId -> (file relative to the unit root, 1-based line, 0-based col)
    astral: frozenset[str]                       # sites with a character above U+FFFF before the callee on their line


def positions_from_bundle(records: Sequence[dict], report: Mapping) -> Positions:
    """The ``positions`` ``merge_scip`` needs, from a bundle's records and the ``assemble`` report's
    ``callsitePositions`` (``cs_info``: file, line, column of the callee's last identifier, columns in code points; the
    bundle's ``callsite`` carries only ``line``). A position for a call site the bundle does not hold, or whose line
    disagrees with the record's, is left out."""
    lines = {r["id"]: r["line"] for r in records if r.get("type") == "callsite"}
    sites, astral = {}, set()
    for cid, (file, line, col, wide) in report.get("callsitePositions", {}).items():
        if lines.get(cid) != line:
            continue
        sites[cid] = (file, line, col)
        if wide:
            astral.add(cid)
    return Positions(sites, frozenset(astral))
