#!/usr/bin/env python3
"""gate_run.py - the ask and run wiring of the AC-5 gate harness (P1-32b; ``docs/plans/connected-code-graph-p1-gates.md`` P1-32 32b, P1-37;
``docs/plans/connected-code-graph-p1-gate-marker-ml.md`` §1-§4 is the rule text of the marker and the resolve dry-run).

``gate.py`` is the library (``run_asks`` is its seam, ``score`` reads the ``run.json`` this module writes); this module adds only:

* :func:`make_ask`: the adapter that issues one structured engine ask exactly as ``ccg ask --request`` does (``clients.common.Client.ask``) from the
  gate's request. An anchor absent from the graph that does not resolve becomes a marked ``not-resolved`` document (a miss, as HA-3 scores it); an
  anchor IN the graph that does not resolve is a :class:`~gate.HarnessError` (never a synthesized document).
* :func:`resolve_preflight`: the read-only dry-run ``resolve`` of every asked in-graph anchor before the marker (counts by status only).
* :func:`run_gate` and the ``run`` verb: refuse unless the preflight is first-contact and clean (``G.run_context``), the list was not run, the probe
  map and the lists lie under the custodian area, the dev-gold probe result (``--dev-probe``, written by ``gate.py dev-probe`` from a dev rehearsal and bound to this commit, harness and build) attests every dev item of each gated
  stratum a hit (``gate.dev_probe_check``; exit 5), the store does not already hold the gate system's live graph, the resolve dry-run passes; build the
  refresh Target at ``EVAL_COMMIT`` with PRIVATE ``work_root`` / ``keys_root`` / ``audit_dir`` under the custodian root (the developer's ``ccg`` area is
  never touched); take the marker (``started``: temp file + ``os.link``, so it is never empty or torn) under a per-list ``flock``; persist every answer
  (append + fsync) before the next ask; write ``run-<list key>.json`` (what ``gate.py score --run`` reads) and the marker ``asked``. The marker pins
  the flow flag, the withheld counts and the probe map's hash beside the other pins; ``score`` uses the run's flag, never the current decision.
  ``--resume`` continues a ``started`` / ``failed:transient`` run under strict equalities and never re-asks a recorded entry; a complete ``run.json``
  under a ``started`` marker (a kill between the two writes) only finalises the marker to ``asked``. No code path removes the marker.
* **Per-anchor failures** (:func:`ask_failure_entry`, the one policy function): a harness error on one anchor is recorded as
  ``{"anchorId", "harnessError": <class>}`` and an exception raised inside the engine's ask as ``{"anchorId", "outcome": "engine-error",
  "exceptionClass": <class>}``; the loop continues, the run completes and ``score`` gives ``blocked`` / a miss respectively. Transient and run-wide
  faults (credential, store, a refused snapshot, the host: any OSError, MemoryError, RecursionError) still end the run (``failed:transient`` / ``failed:deterministic``).

Stdout and stderr carry counts and refusal reasons only: never gold, ids or item content.
Exit: 0 ok, 2 refused (incl. a resume refusal), 3 failed after the marker, deterministic (list spent), 4 bad input / failed before the marker (nothing
spent), 5 the pre-marker resolve dry-run or the dev-gold probe refused (counts only; nothing spent, the failing anchors' clusters go to dev), 6 failed after the marker,
transient (resumable with ``--resume``).
"""
from __future__ import annotations

import argparse
import contextlib
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
from ccg.engine import config as _engine_config   # noqa: E402
from gate import Refusal, HarnessError     # noqa: E402,F401

EXIT_OK, EXIT_REFUSED, EXIT_SPENT, EXIT_BAD, EXIT_RESOLVE, EXIT_RESUMABLE = 0, 2, 3, 4, 5, 6
PROBE_MAP_NAME = "s47-probe-map.json"
#: the repeat-stall conversion (``-p1-gate-marker-ml.md`` §1.1.3): k consecutive timeouts at one position, the deadline of the host-health canary and
#: the canary request. All three are pinned in the marker, so a resume with another value is refused.
STALL_K = 3
STALL_DEADLINE_S = _engine_config.QUERY_TIMEOUT_MS / 1000      # the engine's own per-query limit = the pinned store's TIMEOUT_MAX; ``raw_query`` refuses a larger one (D4):
#                                                                a store that cannot run this canary cannot run the engine's own asks either
CANARY_REQUEST = "MATCH (s:Snapshot) RETURN s.snapshotId"           # fixed, non-held-out: the same read the engine reader's snapshot check uses


class ResolveRefusal(Refusal):
    """The pre-marker resolve dry-run failed for at least one anchor. ``counts`` is by status only (never an id); the list is not spent."""

    def __init__(self, counts: Mapping[str, int]):
        self.counts = dict(counts)
        super().__init__("the resolve dry-run refused: " + " ".join(f"{k}={self.counts[k]}" for k in ("ok", "none", "ambiguous", "id-mismatch", "total"))
                         + ". " + G.STORE_REMEDY + "; fix the cause of the counts above (an anchor the engine cannot resolve) before the next run")


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


#: the host's, never the engine's: any ``OSError`` (a full disk, a quota, a descriptor limit, a permission), memory exhaustion, a blown stack. None of
#: them is placed on a held-out anchor; the run stops (``OSError`` is ``failed:transient``, resumable once the host is fixed).
CREDENTIAL_FAULTS = {"AuthenticationError", "AuthorizationError", "AuthenticationWrongNumberOfArgsError"}      # redis': a ConnectionError subclass that is not a blip
#: the store's own state, not the product's (``-p1-gate-marker-ml.md`` §3 rows 6-8): a refused credential, a role without the permission it was provisioned
#: with (``NOPERM``), a topology that is not the pinned one (``READONLY``, a failover). Each ends the run ``failed:deterministic`` with nothing recorded.
RUN_WIDE_FAULTS = CREDENTIAL_FAULTS | {"NoPermissionError", "ReadOnlyError"}
PROCESS_FAULTS = (OSError, MemoryError, RecursionError)
CHAIN_LIMIT = 16


def exception_chain(exc: BaseException) -> list[BaseException]:
    """``exc`` and what it was raised from or while handling (``__cause__``, else ``__context__``: the real ``Client`` raises ``from None``, which clears the cause
    and keeps the original in the context), outermost first, bounded and cycle-safe. The one walk every rule over a failure reads."""
    out: list[BaseException] = []
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen and len(out) < CHAIN_LIMIT:
        seen.add(id(cur))
        out.append(cur)
        cur = cur.__cause__ if cur.__cause__ is not None else cur.__context__
    return out


def _placed(exc: BaseException) -> list[BaseException]:
    """The links whose type says WHERE the failure happened. A ``StoreError`` is the client's wrapper (it carries what the store did, in its chain), so its
    whole chain counts; any other type is the raiser's own and stands for itself: an engine that raised its own ``ValueError`` while handling an
    ``OSError`` handled it. (A run-wide fault is read over the same links: :func:`_run_wide_links`.)"""
    from clients.common import StoreError
    chain = exception_chain(exc)
    return chain if isinstance(exc, StoreError) else chain[:1]


def _names(exc: BaseException) -> set[str]:
    return {c.__name__ for c in type(exc).__mro__}


def _is_redis(exc: BaseException) -> bool:
    module = type(exc).__module__ or ""
    return module == "redis" or module.startswith("redis.")


def _is_run_wide(exc: BaseException) -> bool:
    """One link: the driver's own credential/role/topology classes (by name AND the redis module: a product class that shares a name is the product's) or a
    harness/client class of the run-wide set."""
    from clients.common import ModelUnavailable, NoCredentials, NotBuilt, StoreAuthError
    return (_is_redis(exc) and bool(_names(exc) & RUN_WIDE_FAULTS)) or isinstance(exc, (Refusal, NoCredentials, StoreAuthError, ModelUnavailable, NotBuilt))


def _run_wide_links(exc: BaseException) -> list[BaseException]:
    """The links a run-wide fault is looked for on: the whole chain when the outer is the client's own wrapper (``StoreError`` or any ``clients.common`` class:
    the real ``Client`` carries what the store did in its chain), only the outer for any other type (a product that raised its own error while handling a
    credential fault has handled it: the tier follows where the fault was raised, never a name or a chain the product can arrange)."""
    from clients.common import ClientError
    chain = exception_chain(exc)
    return chain if isinstance(exc, ClientError) else chain[:1]


def _is_timeout(exc: BaseException) -> bool:
    """A hang: the socket's builtin ``TimeoutError`` or redis' own (by class name and module: ``redis`` stays a lazy import)."""
    return isinstance(exc, TimeoutError) or (_is_redis(exc) and "TimeoutError" in _names(exc))


def _is_host_fault(exc: BaseException) -> bool:
    """A connection or OS fault that is not the timeout itself: ``OSError`` (``ConnectionError`` included), redis' ``ConnectionError`` and ``BusyLoadingError``."""
    return not _is_timeout(exc) and (isinstance(exc, OSError) or (_is_redis(exc) and bool(_names(exc) & {"ConnectionError", "BusyLoadingError"})))


def _is_transient_leaf(exc: BaseException) -> bool:
    return isinstance(exc, (KeyboardInterrupt, _Terminated)) or _is_timeout(exc) or _is_host_fault(exc)


def ask_failure_entry(exc: BaseException) -> dict | None:
    """THE policy for one anchor whose ask raised (``-p1-gate-marker-ml.md`` §3; the one place to change it). -> the entry to persist for that anchor, or
    ``None`` when the exception is not per-anchor and must propagate (the run then ends ``failed:transient`` / ``failed:deterministic``). Decided on the
    whole cause chain, never on the outer type (the real ``Client`` wraps every fault in ``StoreError``); precedence: run-wide > connection/OS/host > rest.

    * run-wide faults anywhere on the chain (a refused credential, ``NOPERM``, ``READONLY``, an unusable store or model, a :class:`~gate.Refusal` such as a
      changed snapshot), a host fault (:data:`PROCESS_FAULTS`) and a transient one (:func:`classify`; a timeout included) -> ``None``;
    * raised inside the engine's ask (:class:`EngineAskError`) -> ``{"outcome": "engine-error", "exceptionClass": <the chain's innermost class under a ``StoreError`` outer, else the outer's own>}``: a miss for the
      item, in the ``fail`` tier, never re-asked on a resume;
    * everything else (a :class:`~gate.HarnessError`, a translation fault, an exception that cannot be placed on the engine's side of the boundary) ->
      ``{"harnessError": <class>}``: the item is neither hit nor miss and the verdict is ``blocked``.

    Only a class name is ever recorded: a message may embed an id or a path."""
    subject = exc.__cause__ if isinstance(exc, EngineAskError) and exc.__cause__ is not None else exc
    chain = exception_chain(subject)
    if any(_is_run_wide(c) for c in _run_wide_links(subject)) or any(isinstance(c, PROCESS_FAULTS) for c in _placed(subject)) or isinstance(exc, PROCESS_FAULTS) \
            or classify(subject) == "transient":
        return None
    if isinstance(exc, EngineAskError):
        from clients.common import StoreError
        return {"outcome": G.ENGINE_ERROR, "exceptionClass": type(chain[-1] if isinstance(subject, StoreError) else subject).__name__}
    return {"harnessError": type(exc).__name__}


def converts_on_repeat_stall(exc: BaseException) -> bool:
    """Whether a stop that ended the run on ``exc`` is a stall the repeat-stall rule may count (§1.1.3 (b)): a timeout is on the chain and nothing of a higher
    precedence is: no run-wide fault, no connection/OS/host fault (an ``OSError`` other than the timeout itself), no process fault, no interrupt."""
    chain = exception_chain(exc)                       # whole chain, both ways: a conversion is the irreversible direction, so any doubt keeps the stop plain
    return any(_is_timeout(c) for c in chain) and not any(
        _is_run_wide(c) or _is_host_fault(c) or isinstance(c, (MemoryError, RecursionError, KeyboardInterrupt, _Terminated)) for c in chain)


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
    a ``StoreError`` is transient only when something on its cause chain is. A run-wide fault anywhere on the chain (a refused credential, ``NOPERM``,
    ``READONLY``: redis' ``AuthenticationError`` is a ``ConnectionError`` subclass, not a blip) outranks everything. EVERYTHING else, an unknown class
    included, is ``deterministic`` (fail closed)."""
    if any(_is_run_wide(c) for c in _run_wide_links(exc)):
        return "deterministic"
    if any(_is_transient_leaf(c) for c in _placed(exc)):
        return "transient"
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


class _Stall:
    """The repeat-stall bookkeeping (§1.1.3). ``get``/``set`` read and write the ``stall`` field of the in-memory marker document: it reaches the disk
    only inside the marker write the run already makes when it stops (``failed:transient``), which is atomic, so no extra write can open a window.
    ``touched`` says whether THIS stop recorded a count; a stop that did not (any other failure) resets the count."""

    def __init__(self, get: Callable[[], Mapping | None], set_: Callable[[Mapping | None], None], canary: Callable[[], Any] | None, k: int):
        self.get, self.set, self.canary, self.k, self.touched = get, set_, canary, k, False

    def on_timeout(self, index: int) -> dict | None:
        """A ``TimeoutError`` inside the engine's ask at position ``index``. -> the entry to record when the k-th consecutive stop is reached with the
        canary answering; ``None`` for a plain stop (the caller re-raises): the canary failed or is absent (count reset), or the count is below k."""
        if self.canary is None:
            return None
        try:
            self.canary()
        except Exception:                                           # noqa: BLE001 - a failed canary is the host's: plain failed:transient, count reset
            return None
        prev = self.get() or {}
        count = prev["count"] + 1 if prev.get("index") == index else 1
        if count >= self.k:
            self.set(None)
            self.touched = True
            return {"outcome": G.ENGINE_ERROR, "exceptionClass": "TimeoutError", "via": G.REPEAT_STALL, "stalls": count}
        self.set({"index": index, "count": count})
        self.touched = True
        return None


class _Persisting:
    """Wraps ``ask`` so that a recorded entry is returned and never re-asked, and a new one is on disk before it is returned. An ask that raises a
    per-anchor failure (:func:`ask_failure_entry`) is recorded as that entry and the run goes on with the next anchor; any other exception propagates."""

    def __init__(self, ask: Callable[[dict], Mapping], path: Path, recorded: dict, stall: _Stall | None = None):
        self.ask, self.path, self.recorded, self.asked_now, self.stall = ask, path, recorded, 0, stall

    def __call__(self, req: dict) -> Mapping:
        aid = _req_id(req)
        if aid in self.recorded:
            return self._returned(self.recorded[aid])
        try:
            doc = self.ask(req)
            entry = {"request": req, "answer": doc}
        except Exception as exc:                                    # noqa: BLE001 - classified by ask_failure_entry; a run-wide fault is re-raised
            entry = ask_failure_entry(exc)
            if entry is None and self.stall is not None and converts_on_repeat_stall(exc):
                entry = self.stall.on_timeout(len(self.recorded))         # the ONE way a timeout becomes an engine error: k stops, canary answering
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


def _pins(ctx: Mapping, snapshot_id: str, obs: Mapping, *, flow_flag: bool, withheld: Mapping, probe_map_sha256: str, manifest_sha256: str,
          dev_probe_sha256: str | None, stall: Mapping) -> dict:
    """What a resume must find equal. The flow flag and the withheld counts decide WHICH anchors are asked, and the probe map decides which point a
    unit anchor is asked from: neither may change between two halves of one run (N3, N6)."""
    return {"recordCommit": ctx["recordCommit"], "harnessSha256": dict(ctx["harnessPin"]["sha256"]), "snapshotId": snapshot_id,
            "prereg": {k: obs.get(k) for k in G.PREREG_FIELDS}, "coreTreeHash": obs.get("coreTreeHash"), "label": ctx["label"],
            "flowFlag": bool(flow_flag), "withheld": dict(withheld), "probeMapSha256": probe_map_sha256,
            "manifestSha256": manifest_sha256, "devProbeSha256": dev_probe_sha256, "stall": dict(stall)}


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


_ALWAYS = object()


@contextlib.contextmanager
def _stranded_on_exit(active: bool, live: Any = _ALWAYS):
    """Marks whatever leaves the block (an exception or an interrupt) as one that leaves the built graph in the store (``exc.strandedStore``); :func:`main`
    then appends :data:`gate.STORE_REMEDY` to its line. ``active`` is false on a resume (no build ran in this process). With ``live`` (the live-graph probe)
    the mark is set only when the probe says the graph is live NOW: inside the build a failure before ``swap``'s rename leaves nothing, so the next fresh run
    is clean, and a probe that is absent or itself fails makes no claim about the store."""
    try:
        yield
    except BaseException as exc:                                    # noqa: BLE001 - re-raised unchanged
        if active and (live is _ALWAYS or (live is not None and _probe_says_live(live))):
            exc.strandedStore = True
        raise


def _probe_says_live(live: Callable[[], bool]) -> bool:
    """Whether the graph is live now. An unreadable store is no claim (False); a SECOND signal while the probe runs (it is called inside the handler of the
    first failure) leaves the answer unknown, so the remedy is kept (True) and the first exception stands: the probe never replaces it."""
    try:
        return bool(live())
    except Exception:                                               # noqa: BLE001 - an unreadable store: no claim about it
        return False
    except BaseException:                                           # noqa: BLE001 - KeyboardInterrupt / _Terminated / SystemExit during the probe
        return True


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
             allow_after_adaptation: bool = False, resume: bool = False, flow_decision: Path | None = None, dev_probe: Path | None = None,
              live_graph_present: Callable[[], bool] | None = None, lock: _Lock | None = None, canary: Callable[[], Any] | None = None,
              stall_k: int = STALL_K, stall_deadline_s: float = STALL_DEADLINE_S, canary_request: str = CANARY_REQUEST) -> dict:
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
    manifest = repo / G.SYSTEM_MANIFEST
    if not manifest.is_file():
        raise Refusal("the system manifest cpg/ccg/system.yaml is missing from the working tree: the build is given the working tree's manifest, because the clone at EVAL_COMMIT may hold none")
    declared = G.manifest_system(repo)
    if declared is not None and declared != system:
        raise Refusal("the system disagrees with the system manifest's own: pass the manifest's system, or none")
    all_records, key = _read_lists(lists, home)
    flag = G.flow_flag(Path(flow_decision) if flow_decision else repo / G.FLOW_DECISION, committed_text)
    if G.is_dev_rehearsal(repo, all_records):                    # a dev-only list needs no probe (it makes one): not read, and the marker pins none
        probe_doc, probe_sha = None, None
    else:
        probe_doc, probe_sha = G.read_dev_probe(dev_probe) if dev_probe else (None, None)
    G.score_inputs_check(repo, gate_id, pf, all_records, committed_text=committed_text,           # every refusal `score` already knows, before the marker
                         dev_probe=probe_doc, gated=G.gated_strata(flag))
    records, withheld = G.withhold_strata(all_records, G.gated_strata(flag))
    records, shapes = G.withhold_shapes(records, G.eval_commit_reader(repo))      # a withheld-shape scripts item is never asked or scored (the scoped AC-5 wording)
    G.check_withheld_cap(shapes)                                 # over the 25% cap: exit 5 here, before the lock, the build and the marker; nothing spent
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
            raise Refusal("the store already holds the live graph of the gate system (an earlier run built it, or the store is in use): the gate builds into its own "
                          "empty throwaway store only. Provision a new empty store (create its ACL users again) and rebuild; if an earlier run of this list was refused "
                          "after its build, the list is not spent (no marker was taken) and runs on the new store")
        _clean_leftover(cust, built, "gate-build-")
        with _stranded_on_exit(True, live_graph_present):          # a failure after ``swap``'s rename leaves the live graph in the store (P5-1)
            G.build_at_eval_commit(eval_commit, built, repo=repo, refresh_fn=refresh_fn, system=system, manifest=manifest, **_private_build_areas(cust, key))
    with _stranded_on_exit(not resume):                          # the build has loaded the live graph: every exit before the marker leaves the store stranded
        snap = snapshot_factory(built)
        sid = snap.snapshot_id()
        if probe_doc is not None:
            G.check_probe_snapshot(probe_doc, sid)             # the attestation vouches for THIS build (after the build: the stranded-store remedy applies)
        if engine_snapshot_id is not None and engine_snapshot_id() != sid:
            raise Refusal("the engine reader's :Snapshot id differs from the snapshot reader's: the store or system is not the one the run reads")
        answers_path = cust / f"run-{key}.answers.jsonl"
        recorded = load_answers(answers_path) if resume else {}
        obs = {**(dict(tree) if tree is not None else G.observe_tree(repo)), **(observed if observed is not None else G.engine_observed())}
        pins = _pins(ctx, sid, obs, flow_flag=flag, withheld=withheld, probe_map_sha256=hashlib.sha256(pmap.read_bytes()).hexdigest(),
                     manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                     dev_probe_sha256=probe_sha, stall={"k": stall_k, "deadlineS": stall_deadline_s, "canaryRequest": canary_request if canary is not None else None})
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
                   "label": ctx["label"], "startedAt": now, "resumes": [], "transientClasses": [], "withheld": withheld, "withheldShapes": shapes, "stall": None}
            G.first_run_marker(cust, key, doc)
    stall = _Stall(lambda: doc.get("stall"), lambda v: doc.__setitem__("stall", v), canary, stall_k)
    persisting = _Persisting(ask, answers_path, recorded, stall)
    try:
        run = G.run_asks(records, snap, persisting, lambda: 0, probes=probes)
        run.update({"preflight": ctx, "withheld": withheld, "withheldShapes": shapes, "flowFlag": bool(flag), "listKey": key})
        doc["resumes"] = [*doc["resumes"][:-1], {**doc["resumes"][-1], "askedAfter": persisting.asked_now}] if resume else doc["resumes"]
        run["resume"] = {"resumed": bool(doc["resumes"]), "resumeCount": len(doc["resumes"]),
                         "answerCountsBefore": [r["answersBefore"] for r in doc["resumes"]], "answerCountsAfter": [r["askedAfter"] for r in doc["resumes"]],
                         "transientClasses": list(doc["transientClasses"]),
                         "statement": "answers recorded before a resume were not re-asked"}
        _atomic_write(cust / f"run-{key}.json", json.dumps(run, sort_keys=True) + "\n")
        _write_marker(marker, {**doc, "state": "asked", "answered": len(run["answers"])})
    except (Exception, KeyboardInterrupt, _Terminated) as exc:           # BaseException classes outside this list (a SIGKILL stand-in) write nothing
        klass = classify(exc)
        if not stall.touched:
            doc["stall"] = None                                          # any stop that is not a counted timeout resets the count
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
    repo = Path(getattr(a, "repo", None) or _HERE.parents[2])
    system = a.system or G.manifest_system(repo) or "ccg"             # the manifest's own system name (D2), never a guess
    reader_db = db_factory or G.reader_db_factory(system, credential)    # the read-only ``raw`` role of the system, never the default user (D1)
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
                          **kw)

    def snapshot_factory(built):
        from snapshot_export import RoRunner
        graph = G.snapshot_graph(a.host, a.port, a.graph, system, db_factory=reader_db)
        return G.StoreSnapshot(RoRunner(graph))

    def ask_factory(built, snap):
        stamp = {"snapshotId": snap.snapshot_id(), "verdictLogOffset": 0, "verdictLogHash": G.EMPTY_VERDICT_LOG_HASH}
        return make_ask(client(), stamp=stamp, in_graph=lambda req: _anchor_in_graph(snap, req))

    def live_graph_present():
        """A read-only probe as the ``raw`` role: no reader role may run ``GRAPH.LIST``, and the server answers ``GRAPH.RO_QUERY`` on an absent key with
        "empty key". Any other failure (a refused credential, a dead store) is raised, never read as "absent"."""
        graph = reader_db(a.host, a.port).select_graph(live_key)
        try:
            graph.ro_query("MATCH (s:Snapshot) RETURN count(s)", None, _engine_config.QUERY_TIMEOUT_MS)
        except Exception as exc:                                   # noqa: BLE001 - classified below, never swallowed
            if type(exc).__name__ == "ResponseError" and "empty key" in str(exc):
                return False
            raise
        return True

    def canary_factory():
        """The host-health canary (§1.1.3): the fixed :data:`CANARY_REQUEST` through the engine reader under the pinned deadline."""
        return lambda: client().raw_query(CANARY_REQUEST, timeout_ms=round(STALL_DEADLINE_S * 1000))

    def resolve_factory(built, snap):
        return client().resolve

    def engine_snapshot():
        rows = client().raw_query("MATCH (s:Snapshot) RETURN s.snapshotId")["rows"]
        if len(rows) != 1:
            raise HarnessError("the engine reader sees no single :Snapshot node")
        return rows[0][0]

    return {"snapshot_factory": snapshot_factory, "ask_factory": ask_factory, "resolve_factory": resolve_factory, "refresh_fn": refresh,
            "engine_snapshot_id": engine_snapshot, "system": system, "live_graph_present": live_graph_present,
            "canary": canary_factory()}


def _msg(e: BaseException) -> str:
    text = str(e)
    return text[len("REFUSED: "):] if text.startswith("REFUSED: ") else text


def _remedy(e: BaseException, text: str) -> str:
    """The operator's remedy for a store the build stranded (:func:`_stranded_on_exit`), unless ``text`` already carries it."""
    return "" if not getattr(e, "strandedStore", False) or G.STORE_REMEDY in text else ". " + G.STORE_REMEDY


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
    r.add_argument("--dev-probe", help="the dev-gold probe `gate.py dev-probe` wrote from a dev rehearsal (an attestation bound to this commit, harness and build; pinned in the "
                                       "marker): refused at exit 5 unless every dev item of a gated stratum is a hit; omitted only for a list of dev items only")
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
                       flow_decision=Path(a.flow_decision) if a.flow_decision else None, dev_probe=Path(a.dev_probe) if a.dev_probe else None, **kw)
        print(json.dumps(out, sort_keys=True))
        return EXIT_OK
    except (ResolveRefusal, G.PrecheckRefusal) as e:
        print(f"REFUSED: {_msg(e)}{_remedy(e, str(e))}", file=sys.stderr)
        return EXIT_RESOLVE
    except Refusal as e:
        print(f"REFUSED: {_msg(e)}{_remedy(e, str(e))}", file=sys.stderr)
        return EXIT_REFUSED
    except RunFailed as e:
        print(str(e), file=sys.stderr)
        return EXIT_RESUMABLE if e.state == "failed:transient" else EXIT_SPENT
    except (KeyboardInterrupt, _Terminated) as e:
        print(f"interrupted: the run is failed:transient (resumable with --resume) if the marker was taken{_remedy(e, '')}", file=sys.stderr)
        return EXIT_RESUMABLE
    except Exception as e:                                        # before the marker: nothing is spent (a failure after it is a RunFailed)
        print(f"bad input: {type(e).__name__}{_remedy(e, '')}", file=sys.stderr)
        return EXIT_BAD
    finally:
        if old_term is not None:
            signal.signal(signal.SIGTERM, old_term)


if __name__ == "__main__":
    sys.exit(main())
