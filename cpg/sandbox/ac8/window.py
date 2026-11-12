#!/usr/bin/env python3
"""The AC-8 detector's window and the ``conn`` verdict (C-1001 P1-31; ``docs/plans/connected-code-graph-egress-spec3.md``
§8.1, P-d, and the detector control scenario).

The detector (``detector.pl``) writes, under ``<auditDir>/detector/``:

* ``conn.jsonl``    one ``conn`` record per connection: ``{ts, srcAddr, srcPort, dstAddr, dstPort, bytesOut, bytesIn}``;
* ``start.marker``  ``{"marker": "start", "ts": ...}``, before the first ``policy`` record;
* ``flush.marker``  ``{"marker": "flush", "ts": ..., "residue": N}``, after the last record; ``residue`` is the number of
  connections still unlogged (queued or being read) when the detector's drain stopped, and anything but 0 fails the verdict;
* ``final.marker``  ``{"marker": "final", "ts": ..., "residue": N}``, written by the SIGTERM drain just before the detector exits: N is
  what that last drain left unlogged. The harness stops the detector before judging, so the marker is required and N must be 0
  (a missing one means the detector was killed mid-drain or never stopped);
* ``probe.jsonl``   the harness's own positive-control connections: a conn record plus ``probe`` = ``<component>/<phase>``, the label
  of the single-use nonce the connection presented (counted and attributed by ``assertions.py``, never ``conn`` records).

Under option C no allow can produce a proxied stream, so the main run passes only with **zero** ``conn`` records: a ``conn``
inside the window fails it, and so does one outside it, and so does a missing marker (P-d). There is no ``conn`` <-> ``decision``
join in P1 (spec3 §6, input 9): the verdict is "none", never "each one matches a decision". A missing ``conn.jsonl`` is a failure,
not "zero connections": a detector that never started proves nothing.

Pure functions, no network and no Docker: ``judge`` takes parsed values; ``verdict_for_dir`` reads a detector directory;
``control_check`` is the detector control scenario's verdict-function half. ``python3 window.py verdict <detector dir>``
prints the failures and exits 1 on any; ``control <detector dir>`` exits 1 unless the control behaves.
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Sequence

CONN_FILE, PROBE_FILE = "conn.jsonl", "probe.jsonl"
MARKER_FILES = {"start": "start.marker", "flush": "flush.marker", "final": "final.marker"}
RESIDUE_MARKERS = ("flush", "final")   # the markers that carry a drain residue
CONN_FIELDS = ("ts", "srcAddr", "srcPort", "dstAddr", "dstPort", "bytesOut", "bytesIn")
PROBE_FIELDS = CONN_FIELDS + ("probe",)
# The detector control scenario's connections (run.sh `detector_control`): one token-less, one wrong-nonce and one replayed-nonce
# connection (all three are `conn`), a flood of CONTROL_FLOOD idle connections opened just before the flush (all must be logged before the
# flush marker: residue 0), and one correct-nonce connection (a probe record labelled CONTROL_PROBE_LABEL).
CONTROL_FLOOD = 40
CONTROL_CONNS = 3 + CONTROL_FLOOD
CONTROL_PROBE_LABEL = "control/ok"
_TS = re.compile(r"(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.(\d{1,6}))?Z")


def parse_ts(text: object) -> datetime | None:
    """An RFC 3339 UTC timestamp with ``Z`` (the audit form, with 1-6 fraction digits), else None."""
    if not isinstance(text, str):
        return None
    m = _TS.fullmatch(text)
    if not m:
        return None
    y, mo, d, h, mi, s = (int(x) for x in m.groups()[:6])
    us = int((m.group(7) or "0").ljust(6, "0"))
    try:
        return datetime(y, mo, d, h, mi, s, us, tzinfo=timezone.utc)
    except ValueError:
        return None


def format_ts(t: datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%S.") + f"{t.microsecond // 1000:03d}Z"


def _is_int(v: object, lo: int, hi: int | None = None) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and v >= lo and (hi is None or v <= hi)


def _conn_error(rec: object, fields: Sequence[str] = CONN_FIELDS) -> str | None:
    if not isinstance(rec, dict):
        return "not an object"
    if set(rec) != set(fields):
        return f"fields {sorted(rec)} are not exactly {sorted(fields)}"
    if "probe" in fields and not (isinstance(rec["probe"], str) and rec["probe"]):
        return "probe label is not a non-empty string"
    if parse_ts(rec["ts"]) is None:
        return "ts is not an RFC 3339 UTC timestamp"
    if not all(isinstance(rec[k], str) and rec[k] for k in ("srcAddr", "dstAddr")):
        return "srcAddr/dstAddr is not a non-empty string"
    if not (_is_int(rec["srcPort"], 0, 65535) and _is_int(rec["dstPort"], 0, 65535)):
        return "a port is not an integer in 0..65535"
    if not (_is_int(rec["bytesOut"], 0) and _is_int(rec["bytesIn"], 0)):
        return "a byte count is not a non-negative integer"
    return None


def read_jsonl(path: Path, fields: Sequence[str] = CONN_FIELDS) -> tuple[list[dict], list[str], bool]:
    """``(records, errors, exists)``. A line that is not a valid ``conn`` record is an error, never skipped, and so is a trailing
    partial line (the detector is stopped, or flushed, before a verdict is taken)."""
    try:
        data = Path(path).read_bytes()
    except OSError:
        return [], [], False
    recs: list[dict] = []
    errs: list[str] = []
    lines = data.split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()
    elif lines:
        errs.append(f"{Path(path).name}: trailing partial line")
        lines.pop()
    for n, raw in enumerate(lines, 1):
        try:
            rec = json.loads(raw)
        except ValueError:
            errs.append(f"{Path(path).name}:{n}: not JSON")
            continue
        why = _conn_error(rec, fields)
        if why:
            errs.append(f"{Path(path).name}:{n}: {why}")
        else:
            recs.append(rec)
    return recs, errs, True


def read_markers(detector_dir: Path) -> tuple[dict, list[str]]:
    """``({"start": t|None, "flush": t|None, "residue": n|None, "final": t|None, "final_residue": n|None}, errors)``: a marker file that
    is absent is None; one that is present but not exactly ``{"marker": <its name>, "ts": <timestamp>}`` (the flush and final markers
    also ``"residue": <int >= 0>``) is an error (and None)."""
    out: dict = {"start": None, "flush": None, "residue": None, "final": None, "final_residue": None}
    errs: list[str] = []
    for name, fname in MARKER_FILES.items():
        p = Path(detector_dir) / fname
        try:
            raw = p.read_text(encoding="utf-8")
        except OSError:
            continue
        try:
            doc = json.loads(raw)
        except ValueError:
            errs.append(f"{fname}: not JSON")
            continue
        t = parse_ts(doc.get("ts")) if isinstance(doc, dict) else None
        keys = {"marker", "ts"} | ({"residue"} if name in RESIDUE_MARKERS else set())
        if not (isinstance(doc, dict) and set(doc) == keys and doc["marker"] == name and t is not None
                and (name not in RESIDUE_MARKERS or _is_int(doc["residue"], 0))):
            errs.append(f"{fname}: not exactly {{marker: {name!r}, ts{', residue' if name in RESIDUE_MARKERS else ''}}}")
            continue
        out[name] = t
        if name == "flush":
            out["residue"] = doc["residue"]
        if name == "final":
            out["final_residue"] = doc["residue"]
    return out, errs


@dataclass
class Verdict:
    ok: bool
    failures: list[str] = field(default_factory=list)
    inside: list[dict] = field(default_factory=list)
    outside: list[dict] = field(default_factory=list)
    start: datetime | None = None
    flush: datetime | None = None


def judge(conns: Sequence[dict], start: datetime | None, flush: datetime | None, *, errors: Iterable[str] = (),
          policy_times: Sequence[datetime] = (), record_times: Sequence[datetime] = (),
          conn_file_exists: bool = True, residue: int | None = 0, final_residue: int | None = 0) -> Verdict:
    """The verdict of one run: ``ok`` iff there are no failures. Failure codes (each a string starting with the code):

    ``detector-log-missing`` · ``detector-record-malformed`` · ``marker-missing:start|flush`` · ``marker-malformed`` ·
    ``window-inverted`` · ``conn-in-window`` · ``conn-outside-window:before-start|after-flush`` ·
    ``window-starts-late`` (an audit record, a ``policy`` record first, precedes the start marker) · ``window-ends-early`` (a record follows the flush
    marker) · ``flush-residue`` (the detector's drain left connections unlogged: ``residue`` is not 0; ``None`` with a flush marker
    present is also a failure: the marker did not report one) · ``marker-missing:final`` (``final_residue`` is None: the detector never finished its
    termination drain) · ``final-residue`` (that drain left connections unlogged). ``policy_times`` and ``record_times`` are the audit records' timestamps (broker and sandbox records alike); give
    them to check P-d's "start marker before the first policy record, flush marker after the last record"."""
    fails: list[str] = []
    if not conn_file_exists:
        fails.append("detector-log-missing: conn.jsonl does not exist (a detector that never ran proves nothing)")
    fails += [f"detector-record-malformed: {e}" for e in errors if not e.startswith("marker")]
    fails += [f"marker-malformed: {e}" for e in errors if e.startswith("marker")]
    if start is None:
        fails.append("marker-missing:start")
    if flush is None:
        fails.append("marker-missing:flush")
    if start is not None and flush is not None and flush < start:
        fails.append("window-inverted: the flush marker precedes the start marker")
    if flush is not None and residue is None:
        fails.append("flush-residue: the flush marker reports no residue")
    elif residue:
        fails.append(f"flush-residue: {residue} connection(s) were still unlogged when the detector's drain stopped")
    if final_residue is None:
        fails.append("marker-missing:final: the detector did not finish its termination drain (it was killed, or never stopped)")
    elif final_residue:
        fails.append(f"final-residue: {final_residue} connection(s) were still unlogged when the termination drain stopped")
    inside: list[dict] = []
    outside: list[dict] = []
    for c in conns:
        t = parse_ts(c["ts"])
        assert t is not None
        if start is not None and t < start:
            outside.append(c)
            fails.append(f"conn-outside-window:before-start {c['srcAddr']}:{c['srcPort']} -> {c['dstAddr']}:{c['dstPort']} at {c['ts']}")
        elif flush is not None and t > flush:
            outside.append(c)
            fails.append(f"conn-outside-window:after-flush {c['srcAddr']}:{c['srcPort']} -> {c['dstAddr']}:{c['dstPort']} at {c['ts']}")
        elif start is None or flush is None:
            outside.append(c)  # no window to be inside of: the missing marker is already a failure
            fails.append(f"conn-outside-window:no-window {c['srcAddr']}:{c['srcPort']} -> {c['dstAddr']}:{c['dstPort']} at {c['ts']}")
        else:
            inside.append(c)
            fails.append(f"conn-in-window {c['srcAddr']}:{c['srcPort']} -> {c['dstAddr']}:{c['dstPort']} at {c['ts']}")
    early = [*policy_times, *record_times]
    if start is not None and early and min(early) < start:
        fails.append(f"window-starts-late: a record at {format_ts(min(early))} precedes the start marker")
    if flush is not None and record_times and max(record_times) > flush:
        fails.append(f"window-ends-early: a record at {format_ts(max(record_times))} follows the flush marker")
    return Verdict(not fails, fails, inside, outside, start, flush)


def verdict_for_dir(detector_dir: Path, *, policy_times: Sequence[datetime] = (), record_times: Sequence[datetime] = ()) -> Verdict:
    conns, errs, exists = read_jsonl(Path(detector_dir) / CONN_FILE)
    markers, merrs = read_markers(detector_dir)
    return judge(conns, markers["start"], markers["flush"], errors=errs + [f"marker {e}" for e in merrs],
                 policy_times=policy_times, record_times=record_times, conn_file_exists=exists, residue=markers["residue"],
                 final_residue=markers["final_residue"])


def control_check(detector_dir: Path) -> list[str]:
    """The detector control scenario (§8.1), the verdict function's half. ``detector_dir`` is a control run's directory: both
    markers (flush and final residue 0), the ``CONTROL_CONNS`` real ``conn`` records of ``run.sh detector-control`` (a token-less, a wrong-nonce
    and a replayed-nonce connection, and a flood of ``CONTROL_FLOOD`` idle connections opened just before the flush) all inside the
    window, and exactly one probe record, the correct nonce's, labelled ``CONTROL_PROBE_LABEL``. Returns the problems; ``[]`` means
    the control shows what it must:

    1. the detector routed each kind of connection right (the counts above) and the flood was all logged before the flush marker;
    2. a real ``conn`` is inside the window and the verdict FAILS with ``conn-in-window``;
    3. the same ``conn`` planted a second before the start marker fails with ``conn-outside-window:before-start``;
    4. planted a second after the flush marker it fails with ``conn-outside-window:after-flush``;
    5. the flush marker's residue reaches the verdict: a marker reporting pending connections fails with ``flush-residue``;
    6. the negative control: the same window with no ``conn`` passes (a verdict that always fails would satisfy 2-5)."""
    problems: list[str] = []
    d = Path(detector_dir)
    conns, errs, exists = read_jsonl(d / CONN_FILE)
    probes, perrs, pexists = read_jsonl(d / PROBE_FILE, PROBE_FIELDS)
    markers, merrs = read_markers(d)
    start, flush = markers["start"], markers["flush"]
    if errs or perrs or merrs or not exists or not pexists or start is None or flush is None or markers["final"] is None:
        return [f"control run is unusable: {errs + perrs + merrs or 'log or marker missing'}"]
    if markers["residue"] != 0:
        problems.append(f"the control's flush marker reports residue {markers['residue']}: the detector's drain left connections unlogged")
    if markers["final_residue"] != 0:
        problems.append(f"the control's final marker reports residue {markers['final_residue']}: the termination drain left connections unlogged")
    if len(conns) != CONTROL_CONNS:
        problems.append(f"the control holds {len(conns)} conn record(s), expected {CONTROL_CONNS} (token-less + wrong nonce + replayed "
                        f"nonce + a flood of {CONTROL_FLOOD}): the detector lost or mis-routed a connection")
    if [p["probe"] for p in probes] != [CONTROL_PROBE_LABEL]:
        problems.append(f"the control's probe records are {[p['probe'] for p in probes]}, expected exactly [{CONTROL_PROBE_LABEL!r}]: "
                        "the correct nonce must be filed once and a replay must not be")
    if not conns:
        return problems + ["control run holds no conn record: the component container's connection was not detected"]
    real = conns[0]
    for c in conns:
        t = parse_ts(c["ts"])
        if not (start <= t <= flush):
            problems.append("a control conn is not inside the window (a flood connection logged after the flush marker was not drained)")
            break
    cases = (
        ("real conn inside the window", [real], "conn-in-window"),
        ("conn before start", [{**real, "ts": format_ts(start - timedelta(seconds=1))}], "conn-outside-window:before-start"),
        ("conn after flush", [{**real, "ts": format_ts(flush + timedelta(seconds=1))}], "conn-outside-window:after-flush"),
    )
    for label, planted, code in cases:
        v = judge(planted, start, flush)
        if v.ok or not any(f.startswith(code) for f in v.failures):
            problems.append(f"{label}: the verdict did not fail with {code} (failures: {v.failures})")
    pend = judge([], start, flush, residue=3)
    if pend.ok or not any(f.startswith("flush-residue") for f in pend.failures):
        problems.append(f"a flush marker reporting 3 pending connections did not fail with flush-residue (failures: {pend.failures})")
    fin = judge([], start, flush, final_residue=3)
    if fin.ok or not any(f.startswith("final-residue") for f in fin.failures):
        problems.append(f"a final marker reporting 3 pending connections did not fail with final-residue (failures: {fin.failures})")
    nofin = judge([], start, flush, final_residue=None)
    if nofin.ok or not any(f.startswith("marker-missing:final") for f in nofin.failures):
        problems.append(f"a missing final marker did not fail with marker-missing:final (failures: {nofin.failures})")
    clean = judge([], start, flush)
    if not clean.ok:
        problems.append(f"zero conn records in a closed window did not pass: {clean.failures}")
    return problems


def _main(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[0] == "verdict":
        v = verdict_for_dir(Path(argv[1]))
        for f in v.failures:
            print(f"FAIL {f}")
        print("conn verdict:", "PASS (zero conn records, window closed)" if v.ok else "FAIL")
        return 0 if v.ok else 1
    if len(argv) == 2 and argv[0] == "control":
        problems = control_check(Path(argv[1]))
        for p in problems:
            print(f"FAIL {p}")
        print("detector control:", "PASS" if not problems else "FAIL")
        return 0 if not problems else 1
    print("usage: window.py verdict <detector dir> | control <detector dir>", file=sys.stderr)
    return 64


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
