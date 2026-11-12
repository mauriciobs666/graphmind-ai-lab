"""The staleness block and the stamp of an answer (C-1001 P1-39a; ``docs/plans/connected-code-graph-p1-answers.md`` P1-39 behaviours (1) and (2)).

Contract: ``-answer-spec.md`` §4.2 (``stamp``), §4.3 (``staleness.units``); ``-answer-spec2.md`` §3.5 (a unit crossed by ``TO_UNIT ->`` is a ``Unit`` hop and ``not-mapped``),
§4.3 (``builtAt``/``failedAt`` are in the answer but outside a 1.1 answer's ``facts_bytes``, which ``ccg.answer.facts_bytes`` applies); FR-7, FR-7a, AC-9, AC-9b.

**One staleness module.** ``ccg.staleness`` (P1-20) owns the closed ``FAIL_REASONS`` and the head comparison; this module imports the vocabulary and implements no second
rule. The answer's ``buildStatus`` is the state *refresh recorded* on the ``Unit`` node (``fresh``, ``stale-kept`` when a rebuild failed and the prior bundle was kept,
``not-mapped`` when no build ever passed), read as it is: the answer schema has no field for "behind the local head", so ``ccg.staleness.unit_staleness`` (which needs a clone)
is not called here.

**Which units.** The units on any chain of the items given (:func:`chain_units`): the unit fact of every chain node, plus the ``extra_units`` the builder collects from
``lifted``/``bound`` entries. Never the walk's ``units`` map: that holds every unit a flag gate read, chain or not, and V9 (``ccg.answer``) wants set equality.

**Commit and time.** Each entry carries its own ``Unit`` node's ``builtCommit`` and ``builtAt`` (and ``failedAt`` for ``stale-kept``), never the snapshot's. A ``stale-kept``
unit's are the last good build's; the failure is shown by ``failedAt`` and ``reason``. ``reason`` is the unit's ``failReason``, a value-free code of
:data:`ccg.staleness.FAIL_REASONS` (no free text is ever emitted).

**Stamp.** ``snapshotId`` from the ``:Snapshot`` node; ``verdictLogOffset`` 0 and the empty-log hash until P4 (``-p1-graph.md`` §3): the node's stored values must equal them or
the stamp is refused. No clock: the same snapshot gives the same bytes.

**Graph data is untrusted.** Every value read is checked against the shape the answer schema gives it and refused otherwise, never repaired, with a static message that names
no value (a unit id, commit or time is graph data).
"""
from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from datetime import datetime
from typing import Any

from ccg.answer import EMPTY_VERDICT_LOG_HASH
from ccg.engine.bfs import Item
from ccg.engine.reader import GraphDataError, Reader
from ccg.engine.state import Node
from ccg.staleness import FAIL_REASONS

#: ``stamp.verdictLogOffset`` until P4 owns the verdict log (``-answer-spec.md`` §4.2; the hash is ``ccg.answer.EMPTY_VERDICT_LOG_HASH``)
VERDICT_LOG_OFFSET = 0

_HEX32 = re.compile(r"[0-9a-f]{32}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_TIME = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z")
_REPO = re.compile(r"[a-z0-9._-]+")
_LANG = re.compile(r"[a-z][a-z0-9]*")
_STATUSES = ("fresh", "stale-kept", "not-mapped")


def _bad(what: str) -> GraphDataError:
    return GraphDataError(f"query 'units' returned {what}")


def _commit(v: Any) -> str:
    if not isinstance(v, str) or not _COMMIT.fullmatch(v):
        raise _bad("a malformed built commit")
    return v


def _time(v: Any) -> str:
    if not isinstance(v, str) or not _TIME.fullmatch(v):
        raise _bad("a malformed time")
    try:
        datetime.strptime(v, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        raise _bad("a malformed time") from None
    return v


def _code(v: Any) -> str:
    if not isinstance(v, str) or v not in FAIL_REASONS:
        raise _bad("a failure reason outside the closed set")
    return v


def _entry(row: tuple) -> dict[str, Any]:
    uid, repo, root, lang, status, commit, built, failed, reason, _flag = row
    if not isinstance(uid, str) or not _HEX32.fullmatch(uid):
        raise _bad("a malformed unit id")
    if not isinstance(repo, str) or not _REPO.fullmatch(repo):
        raise _bad("a malformed repository id")
    if not isinstance(root, str):
        raise _bad("a malformed root path")
    if not isinstance(lang, str) or not _LANG.fullmatch(lang):
        raise _bad("a malformed language")
    if not isinstance(status, str) or status not in _STATUSES:
        raise _bad("an unknown build status")
    out: dict[str, Any] = {"unitId": uid, "repoId": repo, "rootPath": root, "lang": lang, "buildStatus": status}
    if status == "not-mapped":
        if commit is not None or built is not None or failed is not None:
            raise _bad("a never-built unit with build fields")
        out["reason"] = _code(reason)
        return out
    out["builtCommit"], out["builtAt"] = _commit(commit), _time(built)
    if status == "fresh":
        if failed is not None or reason is not None:
            raise _bad("a fresh unit with a failure marker")
    else:
        out["failedAt"], out["reason"] = _time(failed), _code(reason)
    return out


def unit_entries(reader: Reader, unit_ids: Iterable[str]) -> list[dict[str, Any]]:
    """``staleness.units`` for ``unit_ids``: one entry each, read once from the ``units`` query, sorted by ``(repoId, rootPath, unitId)`` (code points; the canonical NFC order is
    ``ccg.answer.canonicalize``'s). A unit without a ``Unit`` row, or a malformed row, raises :class:`~ccg.engine.reader.GraphDataError`."""
    need = sorted(set(unit_ids))
    if not need:
        return []
    out: dict[str, dict[str, Any]] = {}
    for row in reader.read("units", {"ids": need}):
        if len(row) != 10 or not isinstance(row[0], str) or row[0] not in need or row[0] in out:
            raise _bad("a malformed row")
        out[row[0]] = _entry(tuple(row))
    if len(out) != len(need):
        raise GraphDataError("a unit named by the answer has no Unit row")
    return sorted(out.values(), key=lambda e: (e["repoId"], e["rootPath"], e["unitId"]))


def chain_units(items: Iterable[Item], facts: Mapping[Node, Mapping[str, Any]]) -> set[str]:
    """The unit ids on the chains of ``items``: the ``unit`` fact of each chain node that has one (a ``Link`` or ``ContractPoint`` has none)."""
    out: set[str] = set()
    for it in items:
        for step in it.chain:
            u = facts.get(step["node"], {}).get("unit")
            if u is None:
                continue
            if not isinstance(u, str) or not u:
                raise GraphDataError("a chain node carries a malformed unit")
            out.add(u)
    return out


def staleness_block(reader: Reader, items: Iterable[Item], facts: Mapping[Node, Mapping[str, Any]], extra_units: Iterable[str] = ()) -> dict[str, Any]:
    """The answer's ``staleness``: :func:`unit_entries` of the units on the chains of ``items`` and ``extra_units``. A unit that is the target of a ``Unit`` hop must be
    ``not-mapped`` (V9, ``-answer-spec2.md`` §3.5); a graph that says otherwise is refused here rather than at validation."""
    items = list(items)
    entries = unit_entries(reader, chain_units(items, facts) | set(extra_units))
    status = {e["unitId"]: e["buildStatus"] for e in entries}
    for it in items:
        for step in it.chain:
            if step["node"][0] == "Unit" and status.get(step["node"][1]) != "not-mapped":
                raise GraphDataError("a Unit hop's unit is not not-mapped")
    return {"units": entries}


def stamp(reader: Reader) -> dict[str, Any]:
    """The answer's ``stamp``: the ``:Snapshot`` node's id with the P1 verdict-log values. Exactly one snapshot, its stored values equal to the P1 ones."""
    rows = reader.read("snapshot")
    if len(rows) != 1 or len(rows[0]) != 5:
        raise GraphDataError("query 'snapshot' returned no single well-formed row")
    sid, offset, vhash = rows[0][:3]
    if not isinstance(sid, str) or not _HEX32.fullmatch(sid):
        raise GraphDataError("query 'snapshot' returned a malformed snapshot id")
    if type(offset) is not int or offset != VERDICT_LOG_OFFSET or vhash != EMPTY_VERDICT_LOG_HASH:
        raise GraphDataError("the snapshot's verdict log is not the P1 empty one")
    return {"snapshotId": sid, "verdictLogOffset": VERDICT_LOG_OFFSET, "verdictLogHash": EMPTY_VERDICT_LOG_HASH}
