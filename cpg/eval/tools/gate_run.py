#!/usr/bin/env python3
"""gate_run.py - the ask and run wiring of the AC-5 gate harness (P1-32b; ``docs/plans/connected-code-graph-p1-gates.md`` P1-32 32b, P1-37;
``docs/plans/connected-code-graph-p1-gate-marker-ml.md`` §1-§4 is the rule text of the marker and the resolve dry-run).

``gate.py`` is the library (``run_asks`` is its seam, ``score`` reads the ``run.json`` this module writes); this module adds only:

* :func:`make_ask`: the adapter that issues one structured engine ask exactly as ``ccg ask --request`` does (``clients.common.Client.ask``) from the
  gate's request. An anchor absent from the graph that does not resolve becomes a marked ``not-resolved`` document (a miss, as HA-3 scores it); an
  anchor IN the graph that does not resolve is a :class:`~gate.HarnessError` (never a synthesized document).
* :func:`resolve_preflight`: the read-only dry-run ``resolve`` of every asked in-graph anchor before the marker (counts by status only).
* :func:`run_gate` and the ``run`` verb: refuse unless the preflight is first-contact and clean (``G.run_context``), the list was not run, the probe
  map and the lists lie under the custodian area, the store does not already hold the gate system's live graph, the resolve dry-run passes; build the
  refresh Target at ``EVAL_COMMIT`` with PRIVATE ``work_root`` / ``keys_root`` / ``audit_dir`` under the custodian root (the developer's ``ccg`` area is
  never touched); take the marker (``started``: temp file + ``os.link``, so it is never empty or torn) under a per-list ``flock``; persist every answer
  (append + fsync) before the next ask; write ``run-<list key>.json`` (what ``gate.py score --run`` reads) and the marker ``asked``. The marker pins
  the flow flag, the withheld counts and the probe map's hash beside the other pins; ``score`` uses the run's flag, never the current decision.
  ``--resume`` continues a ``started`` / ``failed:transient`` run under strict equalities and never re-asks a recorded entry; a complete ``run.json``
  under a ``started`` marker (a kill between the two writes) only finalises the marker to ``asked``. No code path removes the marker.
* **Per-anchor failures** (:func:`ask_failure_entry`, the one policy function): a harness error on one anchor is recorded as
  ``{"anchorId", "harnessError": <class>}`` and an exception raised inside the engine's ask as ``{"anchorId", "outcome": "engine-error",
  "exceptionClass": <class>}``; the loop continues, the run completes and ``score`` gives ``blocked`` / a miss respectively. Transient and run-wide
  faults (credential, store, a refused snapshot, a full disk) still end the run (``failed:transient`` / ``failed:deterministic``).

Stdout and stderr carry counts and refusal reasons only: never gold, ids or item content.
Exit: 0 ok, 2 refused (incl. a resume refusal), 3 failed after the marker, deterministic (list spent), 4 bad input / failed before the marker (nothing
spent), 5 the pre-marker resolve dry-run refused (counts only; nothing spent, the failing anchors' clusters go to dev), 6 failed after the marker,
transient (resumable with ``--resume``).
"""
from __future__ import annotations

import argparse
import errno
import fcntl
import hashlib
import json
import os
import shutil
import signal
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parents[1])):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gate as G                           # noqa: E402
from gate import Refusal, HarnessError     # noqa: E402,F401

EXIT_OK, EXIT_REFUSED, EXIT_SPENT, EXIT_BAD, EXIT_RESOLVE, EXIT_RESUMABLE = 0, 2, 3, 4, 5, 6
PROBE_MAP_NAME = "s47-probe-map.json"


class ResolveRefusal(Refusal):
    """The pre-marker resolve dry-run failed for at least one anchor. ``counts`` is by status only (never an id); the list is not spent."""

    def __init__(self, counts: Mapping[str, int]):
        self.counts = dict(counts)
        super().__init__("the resolve dry-run refused: " + " ".join(f"{k}={self.counts[k]}" for k in ("ok", "none", "ambiguous", "id-mismatch", "total")))


class RunFailed(RuntimeError):
    """The run failed AFTER the marker; ``state`` is what was written (``failed:transient`` | ``failed:deterministic``). Class name only."""

    def __init__(self, state: str, error_class: str):
        self.state, self.error_class = state, error_class
        super().__init__(f"the run failed after the marker: {state} ({error_class})")


class EngineAskError(Exception):
    """Raised by :func:`make_ask` around the ENGINE's ``ask`` call and nowhere else: an exception that came out of the engine after the harness built and
    validated the request. ``__cause__`` is the engine's own exception. This wrapper is how the engine-error tier is decided by WHERE an exception was
    raised, never by its type or message (``docs/plans/connected-code-graph-p1-gate-marker-ml.md`` §3)."""


class _Terminated(BaseException):
    """SIGTERM turned into an exception by ``main`` (a transient stop: the marker is written ``failed:transient``)."""


# =============================================================================================
# the ask adapter
# =============================================================================================

def _mask_keypath_as_stored(key_path: str) -> str:
    """The ``keyPath`` in the form the build STORES (``gate.stored_keypath``): ``resolve_config`` matches the stored form exactly."""
    return G.stored_keypath(key_path)


def cli_request(req: Mapping) -> dict:
    """The gate's request (``gate.build_request``) as the CLI's structured request: the same use case and the same anchor, by the locator that
    resolves to its own id (``docs/reviews/connected-code-graph-p1-32b-impl.md`` M6, proven in ``tests/test_gate_run_ask_semantics.py``):
    a symbol by ``location`` (repo + path + qname: a qname defined in two files is not ambiguous), a config by its STORED (masked) key path,
    a contract by its ``keyNorm`` (an http key is already in the placeholder form the engine's consumer normalisation keeps)."""
    anchors = req.get("anchors") or []
    if len(anchors) != 1:
        raise HarnessError("a gate request holds exactly one anchor")
    a = anchors[0]
    if a["type"] == "symbol":
        spec = {"kind": "location", "repo": a["repoId"], "path": a["path"], "symbol": a["qname"]}
    elif a["type"] == "config":
        spec = {"kind": "config", "repo": a["repoId"], "path": a["path"], "keyPath": _mask_keypath_as_stored(a["keyPath"])}
    elif a["type"] == "contract":
        spec = {"kind": "contract", "key": a["keyNorm"], "contractKind": a["kind"]}
    else:
        raise HarnessError("a gate request never carries a unit anchor (it is asked from its probe)")
    return {"useCase": req["useCase"], "anchor": spec}


def make_ask(client, stamp: Mapping | None = None, in_graph: Callable[[Mapping], bool] | None = None) -> Callable[[dict], Mapping]:
    """``ask(gate_request) -> answer`` over ``client`` (a ``clients.common.Client``-shaped object). ``in_graph(request)`` says whether the asked anchor
    exists in the gate graph: an anchor that does NOT resolve and is absent from the graph becomes a marked ``not-resolved`` document (a miss for the
    item, scored with ``stamp``, not a C5 void); one that is in the graph (the default reading, fail closed) is a :class:`HarnessError`."""
    from clients.common import AnchorNotResolved, BadRequest

    def ask(req: dict) -> Mapping:
        spec = cli_request(req)                                       # the harness's own translation: a fault here is the harness's
        try:
            return client.ask(spec)
        except AnchorNotResolved as exc:
            if in_graph is None or in_graph(req):
                raise HarnessError("an anchor in the gate graph did not resolve") from None
            return {"interpretation": {"status": exc.detail.get("status") or "unresolved", "lookups": list(exc.detail.get("candidates") or [])},
                    "stamp": dict(stamp or {}), "synthesized": exc.code}
        except BadRequest:                                            # the engine refused the request the harness built
            raise HarnessError("the engine refused the request the harness built") from None
        except Exception as exc:
            if ask_failure_entry(exc) is None:                        # transient or run-wide: the run's to end, not the anchor's
                raise
            raise EngineAskError(type(exc).__name__) from exc         # from here on: raised inside the engine's ask
    return ask


def ask_failure_entry(exc: BaseException) -> dict | None:
    """THE policy for one anchor whose ask raised (``-p1-gate-marker-ml.md`` §3; the one place to change it). -> the entry to persist for that anchor, or
    ``None`` when the exception is not per-anchor and must propagate (the run then ends ``failed:transient`` / ``failed:deterministic``).

    * transient (a connection or timeout error, even one the engine raised) and run-wide faults (a refused credential, an unusable store or model, a
      :class:`~gate.Refusal` such as a changed snapshot) -> ``None``;
    * raised inside the engine's ask (:class:`EngineAskError`) -> ``{"outcome": "engine-error", "exceptionClass": <class>}``: a miss for the item, in the
      ``fail`` tier, never re-asked on a resume;
    * everything else (a :class:`~gate.HarnessError`, a translation fault, an exception that cannot be placed on the engine's side of the boundary) ->
      ``{"harnessError": <class>}``: the item is neither hit nor miss and the verdict is ``blocked``.

    Only a class name is ever recorded: a message may embed an id or a path."""
    from clients.common import ModelUnavailable, NoCredentials, NotBuilt, StoreAuthError
    inner = exc.__cause__ if isinstance(exc, EngineAskError) and exc.__cause__ is not None else exc
    if classify(inner) == "transient" or isinstance(exc, (Refusal, NoCredentials, StoreAuthError, ModelUnavailable, NotBuilt)) \
            or isinstance(inner, (Refusal, NoCredentials, StoreAuthError, ModelUnavailable, NotBuilt)):
        return None
    if isinstance(exc, EngineAskError):
        return {"outcome": G.ENGINE_ERROR, "exceptionClass": type(inner).__name__}
    return {"harnessError": type(exc).__name__}


_ID_KEY = {"symbol": "symbolId", "contract": "cpId", "config": "entryId"}
_LABEL = {"symbol": "Symbol", "contract": "ContractPoint", "config": "ConfigEntry"}


def _asked_anchor(req: Mapping) -> tuple[str, str]:
    a = req["anchors"][0]
    return a["type"], a[_ID_KEY[a["type"]]]


def _anchor_in_graph(snap, req: Mapping) -> bool:
    t, nid = _asked_anchor(req)
    if t == "contract":
        return snap.node_exists("ContractPoint", nid)
    return (snap.symbol(nid) if t == "symbol" else snap.config_entry(nid)) is not None


def resolve_preflight(records: Sequence[Mapping], probes: Mapping | None, snap, resolve: Callable[[dict], Mapping]) -> dict:
    """The read-only dry-run of ``-p1-gate-marker-ml.md`` §2 over every asked anchor that exists in the graph: exactly one candidate and its id equals
    the asked id. -> counts by status (``ok``, ``none``, ``ambiguous``, ``id-mismatch``, ``total``) and ``notInGraph``; raises
    :class:`ResolveRefusal` (counts only) unless every one is ``ok``. An anchor absent from the graph is not resolved."""
    locators = G.locator_index(records)
    counts = {"ok": 0, "none": 0, "ambiguous": 0, "id-mismatch": 0, "total": 0}
    absent = 0
    for aid, rec in G._asks(records, probes).items():                 # noqa: SLF001 - the one definition of "what is asked"
        req = G.build_request(rec, aid, locators)
        if not _anchor_in_graph(snap, req):
            absent += 1
            continue
        res = resolve(cli_request(req)["anchor"])
        cands = list(res.get("candidates") or [])
        t, nid = _asked_anchor(req)
        if res.get("status") == "resolved" and len(cands) == 1:
            status = "ok" if cands[0].get(_ID_KEY[t]) == nid else "id-mismatch"
        elif res.get("status") == "ambiguous" or len(cands) > 1:
            status = "ambiguous"
        else:
            status = "none"
        counts[status] += 1
        counts["total"] += 1
    if counts["ok"] != counts["total"]:
        raise ResolveRefusal(counts)
    return {**counts, "notInGraph": absent}


# =============================================================================================
# failure classes and the marker state machine
# =============================================================================================

def classify(exc: BaseException) -> str:
    """``transient`` is only a connection or timeout error to the store or engine, an ``OSError``, or an interrupt (``-p1-gate-marker-ml.md`` §1.1.3);
    a ``StoreError`` is transient only when what it wrapped was. EVERYTHING else, an unknown class included, is ``deterministic`` (fail closed)."""
    if isinstance(exc, (KeyboardInterrupt, _Terminated)):
        return "transient"
    names = {c.__name__ for c in type(exc).__mro__}
    if "AuthenticationError" in names:                                       # redis' is a ConnectionError subclass: a refused credential is not a blip
        return "deterministic"
    if isinstance(exc, OSError) and exc.errno in (errno.ENOSPC, errno.EDQUOT):   # a full disk fails the same way on a resume: run-wide, not a blip
        return "deterministic"
    if isinstance(exc, (OSError, ConnectionError, TimeoutError)):           # ConnectionError/TimeoutError are OSErrors; listed for the reader
        return "transient"
    if names & {"ConnectionError", "TimeoutError", "BusyLoadingError"} and "redis" in (type(exc).__module__ or ""):
        return "transient"
    from clients.common import StoreError
    if isinstance(exc, StoreError):
        inner = exc.__cause__ or exc.__context__
        while inner is not None:
            if classify(inner) == "transient":
                return "transient"
            inner = inner.__cause__ or inner.__context__
    return "deterministic"


def _atomic_write(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` so that a crash leaves the old file or the new one, never half of it (tmp + fsync + rename in the same directory)."""
    tmp = path.with_name("." + path.name + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(text.encode("utf-8"))
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _marker_path(cust: Path, key: str) -> Path:
    return cust / f"gate-run-{key}.marker"


def _read_marker(path: Path) -> dict:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise Refusal("the marker is unreadable: the run is refused (no marker is ever removed or rewritten by hand)") from None
    if not isinstance(doc, dict) or "state" not in doc or "pins" not in doc:
        raise Refusal("the marker is not a gate-run marker")
    return doc


def _write_marker(path: Path, doc: Mapping) -> None:
    _atomic_write(path, json.dumps(doc, sort_keys=True, separators=(",", ":")) + "\n")


def _req_id(req: Mapping) -> str:
    return _asked_anchor(req)[1]


def _append_line(path: Path, obj: Mapping) -> None:
    """One answer line, appended and fsynced (and the directory entry once, on creation) BEFORE the caller asks the next question."""
    new = not path.exists()
    with open(path, "ab") as fh:
        fh.write((json.dumps(obj, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"))
        fh.flush()
        os.fsync(fh.fileno())
    if new:
        dfd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)


def _valid_line(obj) -> bool:
    """An answer line: ``{anchorId, request, answer}``, a recorded harness error ``{anchorId, harnessError}`` or a recorded engine error
    ``{anchorId, outcome: engine-error, exceptionClass}``."""
    if not isinstance(obj, dict) or not isinstance(obj.get("anchorId"), str):
        return False
    return {"request", "answer"} <= set(obj) or "harnessError" in obj or obj.get("outcome") == G.ENGINE_ERROR


def load_answers(path: Path) -> dict:
    """The recorded entries ``{anchor id: {"request", "answer"} | {"harnessError"} | {"outcome", "exceptionClass"}}``. A last line that has no trailing
    newline or is not valid JSON is a torn write: it is cut from the file and its anchor counts as unrecorded. An invalid line anywhere else is a
    :class:`HarnessError`."""
    if not path.exists():
        return {}
    raw = path.read_bytes()
    parts = raw.split(b"\n")
    tail, complete = parts[-1], parts[:-1]
    good_len, out, torn = 0, {}, bool(tail)
    for i, line in enumerate(complete):
        try:
            obj = json.loads(line.decode("utf-8"))
            if not _valid_line(obj):
                raise ValueError("shape")
        except ValueError:                                          # UnicodeDecodeError is a ValueError
            if i == len(complete) - 1 and not tail:
                torn = True
                break
            raise HarnessError("the answers file has an invalid line that is not the last") from None
        out[obj["anchorId"]] = {k: v for k, v in obj.items() if k != "anchorId"}
        good_len += len(line) + 1
    if torn:
        with open(path, "r+b") as fh:
            fh.truncate(good_len)
            fh.flush()
            os.fsync(fh.fileno())
    return out


class _Persisting:
    """Wraps ``ask`` so that a recorded entry is returned and never re-asked, and a new one is on disk before it is returned. An ask that raises a
    per-anchor failure (:func:`ask_failure_entry`) is recorded as that entry and the run goes on with the next anchor; any other exception propagates."""

    def __init__(self, ask: Callable[[dict], Mapping], path: Path, recorded: dict):
        self.ask, self.path, self.recorded, self.asked_now = ask, path, recorded, 0

    def __call__(self, req: dict) -> Mapping:
        aid = _req_id(req)
        if aid in self.recorded:
            return self._returned(self.recorded[aid])
        try:
            doc = self.ask(req)
            entry = {"request": req, "answer": doc}
        except Exception as exc:                                    # noqa: BLE001 - classified by ask_failure_entry; a run-wide fault is re-raised
            entry = ask_failure_entry(exc)
            if entry is None:
                raise
        _append_line(self.path, {"anchorId": aid, **entry})
        self.recorded[aid] = entry
        self.asked_now += 1
        return self._returned(entry)

    @staticmethod
    def _returned(entry: Mapping):
        return entry["answer"] if "answer" in entry else G.AnchorFault(entry)


# =============================================================================================
# the run verb
# =============================================================================================

def _read_lists(lists: Sequence[Path], home: Path | None) -> tuple[list[dict], str]:
    """The held-out records, read only through the custodian guard; -> (records, the canonical list key, ``G.list_key``)."""
    records = []
    for f in lists:
        raw = G.within_custodian(f, home, "the held-out list").read_bytes()
        records += [json.loads(l) for l in raw.decode("utf-8").splitlines() if l.strip()]
    return records, G.list_key(records)


def _pins(ctx: Mapping, snapshot_id: str, obs: Mapping, *, flow_flag: bool, withheld: Mapping, probe_map_sha256: str) -> dict:
    """What a resume must find equal. The flow flag and the withheld counts decide WHICH anchors are asked, and the probe map decides which point a
    unit anchor is asked from: neither may change between two halves of one run (N3, N6)."""
    return {"recordCommit": ctx["recordCommit"], "harnessSha256": dict(ctx["harnessPin"]["sha256"]), "snapshotId": snapshot_id,
            "prereg": {k: obs.get(k) for k in G.PREREG_FIELDS}, "coreTreeHash": obs.get("coreTreeHash"), "label": ctx["label"],
            "flowFlag": bool(flow_flag), "withheld": dict(withheld), "probeMapSha256": probe_map_sha256}


PRIVATE_AREAS = ("work", "keys", "audit")


def _clean_leftover(cust: Path, dest: Path, prefix: str) -> None:
    """A gate-build directory a failed earlier attempt left (the run died before the marker, so nothing was spent) is removed and rebuilt. Only a direct
    child of the custodian root with the gate prefix, and never a symlink. Never a marker."""
    if dest.is_symlink():
        raise Refusal("a gate build directory is a symlink: refused")
    if dest.exists():
        if dest.parent.resolve() != cust.resolve() or not dest.name.startswith(prefix):
            raise Refusal("a gate build directory is not under the custodian root: refused")
        shutil.rmtree(dest)


def _private_build_areas(cust: Path, key: str) -> dict:
    """N4: the gate build is the system ``ccg`` (the clone's manifest says so), the same name as the developer's. Its bundles, unit state, HMAC key
    and sandbox audit therefore go to PRIVATE directories under the custodian root, passed to ``refresh`` as ``work_root`` / ``keys_root`` /
    ``audit_dir``, so the retention pass and the key store never see the developer's. Fresh per attempt (a leftover of a failed attempt is removed)."""
    out = {}
    for name, kw in zip(PRIVATE_AREAS, ("work_root", "keys_root", "audit_dir")):
        d = cust / f"gate-{name}-{key}"
        _clean_leftover(cust, d, f"gate-{name}-")
        d.mkdir(mode=0o700)
        out[kw] = d
    return out


class _Lock:
    """One live ``run`` / ``--resume`` per list: an exclusive non-blocking ``flock`` on ``gate-run-<key>.lock`` (never a marker; it holds nothing) for
    the whole run, so two processes cannot interleave answer lines or marker writes. Released by the OS if the process dies."""

    def __init__(self):
        self.fd = None

    def take(self, cust: Path, key: str) -> None:
        fd = os.open(cust / f"gate-run-{key}.lock", os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            raise Refusal("another run of this list holds the lock: refused (a second live run would interleave the answers)") from None
        self.fd = fd

    def release(self) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


def _finalise_if_complete(cust: Path, key: str, marker: Path, old: Mapping, n_records: int) -> dict | None:
    """N1: a kill after ``run-<key>.json`` was renamed in and before the marker became ``asked`` leaves a COMPLETE run under a ``started`` marker. A
    ``run.json`` that parses and whose ``listKey`` and ``snapshotId`` equal the marker's pins only finalises the marker to ``asked`` (nothing is asked,
    nothing is rebuilt); anything else is not a finalisation (-> ``None``: the ordinary resume checks refuse it)."""
    path = cust / f"run-{key}.json"
    if not path.is_file() or old["state"] not in ("started", "failed:transient"):
        return None
    try:
        run = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(run, dict) or not isinstance(run.get("answers"), dict) or run.get("listKey") != key \
            or run.get("snapshotId") != old["pins"].get("snapshotId"):
        return None
    _write_marker(marker, {**old, "state": "asked", "answered": len(run["answers"])})
    return {"records": n_records, "asked": len(run["answers"]), "label": old["label"], "finalised": True}


def _resume_checks(marker: Mapping, pins: Mapping, key: str, cust: Path, recorded: int, cache_len: Callable[[], int]) -> None:
    state = marker["state"]
    if state not in ("started", "failed:transient"):
        raise Refusal(f"resume refused: the marker state is {state} (only started or failed:transient resumes; a spent list needs a new list)")
    differ = sorted(k for k in set(pins) | set(marker["pins"]) if marker["pins"].get(k) != pins.get(k))
    if differ:
        raise Refusal("resume refused: the pinned values differ from the marker: " + ", ".join(differ))
    if (cust / f"run-{key}.json").exists():
        raise Refusal("resume refused: a run file already exists (score it; the run is complete)")
    if cache_len() > recorded + 1:
        raise Refusal("resume refused: the cache holds more than the one in-flight entry (C5)")


def run_gate(*args, **kw) -> dict:
    """The AC-5 gate run (:func:`_run_gate`) under the per-list lock."""
    lock = _Lock()
    try:
        return _run_gate(*args, lock=lock, **kw)
    finally:
        lock.release()


def _run_gate(repo: Path, gate_id: str, lists: Sequence[Path], custodian: Path, *, snapshot_factory: Callable[[Path], Any],
             ask_factory: Callable[[Path, Any], Callable[[dict], Mapping]], resolve_factory: Callable[[Path, Any], Callable[[dict], Mapping]],
             probe_map: Path | None = None, home: Path | None = None, observed: Mapping | None = None, tree: Mapping | None = None,
             committed_text=None, record_commit_sha: str | None = None, refresh_fn: Callable[..., Any] | None = None,
             cache_len: Callable[[], int] = lambda: 0, system: str = "ccg", engine_snapshot_id: Callable[[], str] | None = None,
             allow_after_adaptation: bool = False, resume: bool = False, flow_decision: Path | None = None,
              live_graph_present: Callable[[], bool] | None = None, lock: _Lock | None = None) -> dict:
    """The AC-5 gate run: every refusal happens before the marker and the first question. -> counts only."""
    repo = Path(repo)
    cust = G.within_custodian(custodian, home, "--custodian")
    pf = G.preflight(repo, gate_id, observed=observed, tree=tree, committed_text=committed_text, record_commit_sha=record_commit_sha)
    if not pf["firstContact"]["firstContact"]:
        raise Refusal("the preflight is not first-contact (prereg fields differ or cpg/ is not clean); the gate run is refused")
    ctx = G.run_context(repo, pf, allow_after_adaptation=allow_after_adaptation)
    pmap = G.within_custodian(Path(probe_map) if probe_map else cust / PROBE_MAP_NAME, home, "the S47 probe map")
    if not pmap.is_file():
        raise Refusal("the S47 probe map is missing under the custodian root: a custodian unit produces it (split_manifest.split(...)['probe']); the run is refused")
    probes = G.read_probe_map(pmap, home)
    all_records, key = _read_lists(lists, home)
    G.require_clean_renormalize(G.renormalize_stored_gold(all_records))
    flag = G.flow_flag(Path(flow_decision) if flow_decision else repo / G.FLOW_DECISION, committed_text)
    records, withheld = G.withhold_strata(all_records, G.STRATA if flag else G.STRATA_GATED)
    marker = _marker_path(cust, key)
    if lock is not None:
        lock.take(cust, key)
    if resume and not marker.exists():
        raise Refusal("nothing to resume: this list has no marker")
    if marker.exists() and not resume:
        raise Refusal("this list was already run: a re-run on the same list is refused (EP §12.1 step 7); a new list is required (or --resume of a started run)")
    old = _read_marker(marker) if resume else None
    if resume and (done := _finalise_if_complete(cust, key, marker, old, len(all_records))) is not None:
        return done
    eval_commit = (repo / "cpg" / "eval" / "EVAL_COMMIT").read_text(encoding="utf-8").strip()
    built = cust / f"gate-build-{key}"
    if not resume:
        if live_graph_present is not None and live_graph_present():
            raise Refusal("the store already holds the live graph of the gate system: the gate builds into its own empty throwaway store, never one in use")
        _clean_leftover(cust, built, "gate-build-")
        G.build_at_eval_commit(eval_commit, built, repo=repo, refresh_fn=refresh_fn, system=system, **_private_build_areas(cust, key))
    snap = snapshot_factory(built)
    sid = snap.snapshot_id()
    if engine_snapshot_id is not None and engine_snapshot_id() != sid:
        raise Refusal("the engine reader's :Snapshot id differs from the snapshot reader's: the store or system is not the one the run reads")
    answers_path = cust / f"run-{key}.answers.jsonl"
    recorded = load_answers(answers_path) if resume else {}
    obs = {**(dict(tree) if tree is not None else G.observe_tree(repo)), **(observed if observed is not None else G.engine_observed())}
    pins = _pins(ctx, sid, obs, flow_flag=flag, withheld=withheld, probe_map_sha256=hashlib.sha256(pmap.read_bytes()).hexdigest())
    if resume:
        _resume_checks(old, pins, key, cust, len(recorded), cache_len)
    else:
        G.check_cache_empty(cache_len)
    resolve_preflight(records, probes, snap, resolve_factory(built, snap))
    ask = ask_factory(built, snap)
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if resume:
        doc = {**old, "state": "started", "resumes": [*old["resumes"], {"answersBefore": len(recorded), "askedAfter": None}]}
        _write_marker(marker, doc)
    else:
        doc = {"state": "started", "gateId": gate_id, "records": len(all_records), "listSha256": key, "pins": pins, "head": ctx["head"],
               "label": ctx["label"], "startedAt": now, "resumes": [], "transientClasses": [], "withheld": withheld}
        G.first_run_marker(cust, key, doc)
    persisting = _Persisting(ask, answers_path, recorded)
    try:
        run = G.run_asks(records, snap, persisting, lambda: 0, probes=probes)
        run.update({"preflight": ctx, "withheld": withheld, "flowFlag": bool(flag), "listKey": key})
        doc["resumes"] = [*doc["resumes"][:-1], {**doc["resumes"][-1], "askedAfter": persisting.asked_now}] if resume else doc["resumes"]
        run["resume"] = {"resumed": bool(doc["resumes"]), "resumeCount": len(doc["resumes"]),
                         "answerCountsBefore": [r["answersBefore"] for r in doc["resumes"]], "answerCountsAfter": [r["askedAfter"] for r in doc["resumes"]],
                         "transientClasses": list(doc["transientClasses"]),
                         "statement": "answers recorded before a resume were not re-asked"}
        _atomic_write(cust / f"run-{key}.json", json.dumps(run, sort_keys=True) + "\n")
        _write_marker(marker, {**doc, "state": "asked", "answered": len(run["answers"])})
    except (Exception, KeyboardInterrupt, _Terminated) as exc:           # BaseException classes outside this list (a SIGKILL stand-in) write nothing
        klass = classify(exc)
        if resume:
            doc["resumes"] = [*doc["resumes"][:-1], {**doc["resumes"][-1], "askedAfter": persisting.asked_now}]
        fail = {**doc, "state": f"failed:{klass}", "errorClass": type(exc).__name__,
                "transientClasses": [*doc["transientClasses"], *([type(exc).__name__] if klass == "transient" else [])]}
        _write_marker(marker, fail)
        if isinstance(exc, (KeyboardInterrupt, _Terminated)):
            raise
        raise RunFailed(fail["state"], type(exc).__name__) from exc
    return {"records": len(all_records), "asked": len(run["answers"]), "label": ctx["label"]}


# =============================================================================================
# the default (production) wiring
# =============================================================================================

def _default_deps(a, *, connect: Callable[..., Any] | None = None, credential: Callable[[str, str], str] | None = None, refresh_module=None,
                  client_factory: Callable[..., Any] | None = None, db_factory: Callable[[str, int], Any] | None = None) -> dict:
    """The production seams, none of them injected by the caller: a refresh that loads into a ``Target`` of the system's ``build`` and ``verdict`` users
    on the explicit store ``--host``/``--port`` (``G.snapshot_graph``'s guard: loopback only, no missing or forbidden port, ``--graph`` equal to the
    system's live key, all BEFORE any connection), the snapshot reader, the engine ``Client`` for ask and resolve, and the engine-side ``:Snapshot`` id."""
    system = a.system or "ccg"
    live_key = G.snapshot_graph(a.host, a.port, a.graph, system, db_factory=lambda h, p: type("_Guard", (), {"select_graph": staticmethod(lambda k: k)})())
    box: dict[str, Any] = {}

    def client():
        if "c" not in box:
            if client_factory is not None:
                box["c"] = client_factory(system, host=a.host, port=a.port)
            else:
                from clients.common import Client
                box["c"] = Client(system, host=a.host, port=a.port)
        return box["c"]

    def refresh(sys_name, repo_root=None, **kw):
        from ccg import sysname
        from ccg.store import loader as L
        if refresh_module is None:
            from ccg import refresh as rm
        else:
            rm = refresh_module
        cred = credential
        if cred is None:
            from clients.common import _credential as cred
        conn = connect or L.connect
        build_db, verdict_db = (conn(a.host, a.port, username=sysname.acl_user(role, sys_name), password=cred(sys_name, role)) for role in ("build", "verdict"))
        root = Path(repo_root)
        return rm.refresh(sys_name, target=rm.Target(L.LiveStore(build_db, sys_name), build_db, verdict_db), repo_root=root,
                          manifest=root / "cpg" / "ccg" / "system.yaml", **kw)

    def snapshot_factory(built):
        from snapshot_export import RoRunner
        graph = G.snapshot_graph(a.host, a.port, a.graph, system, db_factory=db_factory)
        return G.StoreSnapshot(RoRunner(graph))

    def ask_factory(built, snap):
        stamp = {"snapshotId": snap.snapshot_id(), "verdictLogOffset": 0, "verdictLogHash": G.EMPTY_VERDICT_LOG_HASH}
        return make_ask(client(), stamp=stamp, in_graph=lambda req: _anchor_in_graph(snap, req))

    def live_graph_present():
        if db_factory is None:
            from falkordb import FalkorDB
            db = FalkorDB(host=a.host, port=a.port)
        else:
            db = db_factory(a.host, a.port)
        return live_key in list(db.list_graphs())

    def resolve_factory(built, snap):
        return client().resolve

    def engine_snapshot():
        rows = client().raw_query("MATCH (s:Snapshot) RETURN s.snapshotId")["rows"]
        if len(rows) != 1:
            raise HarnessError("the engine reader sees no single :Snapshot node")
        return rows[0][0]

    return {"snapshot_factory": snapshot_factory, "ask_factory": ask_factory, "resolve_factory": resolve_factory, "refresh_fn": refresh,
            "engine_snapshot_id": engine_snapshot, "system": system, "live_graph_present": live_graph_present}


def _msg(e: BaseException) -> str:
    text = str(e)
    return text[len("REFUSED: "):] if text.startswith("REFUSED: ") else text


def main(argv=None, deps: Mapping | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    for f in ("--gate-id", "--lists", "--custodian"):
        r.add_argument(f, required=True)
    r.add_argument("--repo", default=str(_HERE.parents[2]))
    r.add_argument("--system")
    r.add_argument("--host", default="127.0.0.1")
    r.add_argument("--port", type=int, help="the explicit port of the gate's own throwaway store (never 6379, 6390, 16390, 16400)")
    r.add_argument("--graph")
    r.add_argument("--probe-map", help=f"the S47 probe map under the custodian root (default <custodian>/{PROBE_MAP_NAME})")
    r.add_argument("--flow-decision", help="the committed flow decision record (default: the repository's)")
    r.add_argument("--resume", action="store_true", help="continue a started / failed:transient run of the same list under equal pins")
    r.add_argument("--allow-after-adaptation", action="store_true", help="run with the after-adaptation label (recorded in the marker and run.json)")
    a = ap.parse_args(argv)
    old_term = None
    try:
        old_term = signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(_Terminated()))
    except ValueError:                                           # not the main thread
        pass
    try:
        kw = dict(deps) if deps is not None else _default_deps(a)
        out = run_gate(Path(a.repo), a.gate_id, [Path(f) for f in a.lists.split(",")], Path(a.custodian),
                       probe_map=Path(a.probe_map) if a.probe_map else None, resume=a.resume, allow_after_adaptation=a.allow_after_adaptation,
                       flow_decision=Path(a.flow_decision) if a.flow_decision else None, **kw)
        print(json.dumps(out, sort_keys=True))
        return EXIT_OK
    except ResolveRefusal as e:
        print(f"REFUSED: {_msg(e)}", file=sys.stderr)
        return EXIT_RESOLVE
    except Refusal as e:
        print(f"REFUSED: {_msg(e)}", file=sys.stderr)
        return EXIT_REFUSED
    except RunFailed as e:
        print(str(e), file=sys.stderr)
        return EXIT_RESUMABLE if e.state == "failed:transient" else EXIT_SPENT
    except (KeyboardInterrupt, _Terminated):
        print("interrupted: the run is failed:transient (resumable with --resume) if the marker was taken", file=sys.stderr)
        return EXIT_RESUMABLE
    except Exception as e:                                        # before the marker: nothing is spent (a failure after it is a RunFailed)
        print(f"bad input: {type(e).__name__}", file=sys.stderr)
        return EXIT_BAD
    finally:
        if old_term is not None:
            signal.signal(signal.SIGTERM, old_term)


if __name__ == "__main__":
    sys.exit(main())
