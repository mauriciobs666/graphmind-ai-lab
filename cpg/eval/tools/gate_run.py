#!/usr/bin/env python3
"""gate_run.py - the ask and run wiring of the AC-5 gate harness (P1-32b; ``docs/plans/connected-code-graph-p1-gates.md`` P1-32 32b, P1-37).

``gate.py`` is the library (``run_asks`` is its seam, ``score`` reads the ``run.json`` this module writes); this module adds only:

* :func:`make_ask`: the adapter that issues one structured engine ask exactly as ``ccg ask --request`` does (``clients.common.Client.ask``:
  ``{"useCase", "anchor": {...}}``, no interpreter and no model) from the gate's request (``gate.build_request``: resolved ids, ``anchors[]``),
  and returns the full answer JSON. An anchor the client refuses as ambiguous or unresolved becomes a marked ``not-resolved`` document (a miss for
  the item, as HA-3 scores it), not a crash; every other client error propagates.
* :func:`run_gate` and the ``run`` verb: refuse unless the preflight is first-contact (the four ``ccg-prereg-v1`` fields equal, clean ``cpg/``, the
  committed record present) and the list was not run before; read the held-out lists only from the custodian area; build the refresh Target at
  ``EVAL_COMMIT`` (``gate.build_at_eval_commit``, clone under the custodian root); take the first-run marker; ask; write ``run-<list sha16>.json`` (what ``gate.py score --run`` reads) into the custodian root.

Stdout and stderr carry counts and refusal reasons only: never gold, ids or item content. Exit: 0 ok, 2 refused, 4 bad input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parents[1])):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gate as G                           # noqa: E402
from gate import Refusal, HarnessError     # noqa: E402,F401


# =============================================================================================
# the ask adapter
# =============================================================================================

def cli_request(req: Mapping) -> dict:
    """The gate's request (``gate.build_request``) as the CLI's structured request: the same use case and the same anchor, by its locator."""
    anchors = req.get("anchors") or []
    if len(anchors) != 1:
        raise HarnessError("a gate request holds exactly one anchor")
    a = anchors[0]
    if a["type"] == "symbol":
        spec = {"kind": "symbol", "repo": a["repoId"], "qname": a["qname"]}
    elif a["type"] == "config":
        spec = {"kind": "config", "repo": a["repoId"], "path": a["path"], "keyPath": a["keyPath"]}
    elif a["type"] == "contract":
        spec = {"kind": "contract", "key": a["keyNorm"], "contractKind": a["kind"]}
    else:
        raise HarnessError("a gate request never carries a unit anchor (it is asked from its probe)")
    return {"useCase": req["useCase"], "anchor": spec}


def make_ask(client, stamp: Mapping | None = None) -> Callable[[dict], Mapping]:
    """``ask(gate_request) -> answer`` over ``client`` (a ``clients.common.Client``-shaped object). ``stamp`` is the gate snapshot's stamp, given to the
    ``not-resolved`` document so that it scores as a miss and not as a C5 void."""
    from clients.common import AnchorNotResolved

    def ask(req: dict) -> Mapping:
        try:
            return client.ask(cli_request(req))
        except AnchorNotResolved as exc:
            return {"interpretation": {"status": exc.detail.get("status") or "unresolved", "lookups": list(exc.detail.get("candidates") or [])},
                    "stamp": dict(stamp or {}), "synthesized": exc.code}
    return ask


# =============================================================================================
# the run verb
# =============================================================================================

def _within_custodian(path: Path, home: Path | None, what: str) -> Path:
    area = G._ac.custodian_area(home).resolve()
    p = Path(path).expanduser().resolve()
    if p != area and area not in p.parents:
        raise Refusal(f"REFUSED: {what} must be under the custodian area, never elsewhere")
    return p


def _read_lists(lists: Sequence[Path], home: Path | None) -> tuple[list[dict], str]:
    """The held-out records, read only through the custodian guard; -> (records, sha256 of the list bytes)."""
    h, records = hashlib.sha256(), []
    for f in lists:
        raw = _within_custodian(f, home, "the held-out list").read_bytes()
        h.update(raw)
        records += [json.loads(l) for l in raw.decode("utf-8").splitlines() if l.strip()]
    return records, h.hexdigest()


def run_gate(repo: Path, gate_id: str, lists: Sequence[Path], custodian: Path, *, snapshot_factory: Callable[[Path], Any],
             ask_factory: Callable[[Path, Any], Callable[[dict], Mapping]], home: Path | None = None, observed: Mapping | None = None,
             tree: Mapping | None = None, committed_text=None, record_commit_sha: str | None = None,
             refresh_fn: Callable[..., Any] | None = None, cache_len: Callable[[], int] = lambda: 0, system: str = "ccg") -> dict:
    """The AC-5 gate run: every refusal happens before the build, the marker and the first question. -> counts only."""
    repo = Path(repo)
    cust = _within_custodian(custodian, home, "--custodian")
    pf = G.preflight(repo, gate_id, observed=observed, tree=tree, committed_text=committed_text, record_commit_sha=record_commit_sha)
    if not pf["firstContact"]["firstContact"]:
        raise Refusal("the preflight is not first-contact (prereg fields differ or cpg/ is not clean); the gate run is refused")
    records, sha = _read_lists(lists, home)
    G.require_clean_renormalize(G.renormalize_stored_gold(records))
    marker = cust / f"gate-run-{sha[:16]}.marker"
    if marker.exists():
        raise Refusal("this list was already run: a re-run on the same list is refused (EP §12.1 step 7); a new list is required")
    eval_commit = (repo / "cpg" / "eval" / "EVAL_COMMIT").read_text(encoding="utf-8").strip()
    built = G.build_at_eval_commit(eval_commit, cust / f"gate-build-{sha[:16]}", repo=repo, refresh_fn=refresh_fn, system=system)
    snap = snapshot_factory(built)
    ask = ask_factory(built, snap)
    G.check_cache_empty(cache_len)
    G.first_run_marker(cust, sha, {"gateId": gate_id, "records": len(records), "listSha256": sha})
    run = G.run_asks(records, snap, ask, cache_len)
    G.write_custodian(cust, f"run-{sha[:16]}.json", run, home=home)
    return {"records": len(records), "asked": len(run["answers"]), "label": pf["label"]}


def _default_deps(a) -> dict:
    from falkordb import FalkorDB
    from clients.common import Client
    from snapshot_export import RoRunner

    def snapshot_factory(built):
        return G.StoreSnapshot(RoRunner(FalkorDB(host=a.host, port=a.port).select_graph(a.graph)))

    def ask_factory(built, snap):
        stamp = {"snapshotId": snap.snapshot_id(), "verdictLogOffset": 0, "verdictLogHash": G.EMPTY_VERDICT_LOG_HASH}
        return make_ask(Client(a.system, host=a.host, port=a.port), stamp=stamp)
    return {"snapshot_factory": snapshot_factory, "ask_factory": ask_factory, "system": a.system or "ccg"}


def main(argv=None, deps: Mapping | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    for f in ("--gate-id", "--lists", "--custodian"):
        r.add_argument(f, required=True)
    r.add_argument("--repo", default=str(_HERE.parents[2]))
    r.add_argument("--system")
    r.add_argument("--host", default="127.0.0.1")
    r.add_argument("--port", type=int)
    r.add_argument("--graph")
    a = ap.parse_args(argv)
    try:
        kw = dict(deps) if deps is not None else _default_deps(a)
        out = run_gate(Path(a.repo), a.gate_id, [Path(f) for f in a.lists.split(",")], Path(a.custodian), **kw)
        print(json.dumps(out, sort_keys=True))
        return 0
    except Refusal as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    except (OSError, ValueError, KeyError, TypeError, HarnessError) as e:
        print(f"bad input: {type(e).__name__}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    sys.exit(main())
