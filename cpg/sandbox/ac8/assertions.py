#!/usr/bin/env python3
"""AC-8 part 1: rows 8.1-8.16, the preconditions P-a..P-g and the in-run controls, computed over a scenario's audit
(C-1001 P1-31; ``docs/plans/connected-code-graph-egress-spec3.md`` §8.1-§8.2, ``-egress-spec4.md``).

    python3 assertions.py <scenario dir> [--stub-resolver] [--json <file>]

Scenario directory layout (written by ``run.sh``; every verdict below is computed from these files, none from a probe's exit code)::

    audit/egress.jsonl  audit/sandbox.jsonl          the broker's and the runner's records (one scenario, never rotated)
    audit/detector/     conn.jsonl probe.jsonl start.marker flush.marker      (``window.py``)
    manifest.yaml       the scenario's system manifest
    facts.json          what the harness did: components, host snapshot, probe files, trace and event files (see ``Facts``)
    probes/*.txt        ``netprobe.pl`` output per phase and component; ``escape-container.txt`` (probe.pl via the runner)
    trace/              ``strace`` of the broker and of the positive-control program (row 8.16)
    docker-events.jsonl ``docker events`` captured for the scenario
    detector-control/   the detector control scenario's detector directory

Every row is **computed**, **vacuous** (nothing in P1 can make it fail, so the report says so; a thing that "cannot exist"
turning up is a ``fail``, not a vacuous pass) or **unrun** (row 8.16 without ``strace``: never reported as passed). A row that
is computed over an empty witness set is a fail: P-g exists so that no universal check runs over nothing.

Overall: ``fail`` (exit 1) if any row, precondition or control failed; else ``incomplete`` (exit 3) if any row is unrun **or the run
used a self-test stand-in** (``facts.json`` ``selfTest`` is non-empty: ``--stub-resolver``, ``--stub-model``, ``--store-standin``,
``--witness-parse``); else ``pass`` (exit 0). P1-40's acceptance is exit 0 with ``selfTest == []``.

    python3 assertions.py pre-e <manifest.yaml> [--stub-resolver]       P-e alone, before the run (spec3 §8.1: "before the run")
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence
from urllib.parse import urlsplit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import window  # noqa: E402  (the sibling module: the detector's window and the `conn` verdict)

HERE = Path(__file__).resolve().parent
PASS, FAIL, VACUOUS, UNRUN = "pass", "fail", "vacuous", "unrun"

# §3.3's rule ids abbreviated as §8's checks write them.
R1, R2, R3, R4, R5, R8, R9 = ("R1-LOCAL-ENDPOINT", "R2-GIT-OWN-REMOTE", "R3-VULNDB", "R4-NAMED-SERVICE",
                              "R5-SYSTEM-LOCAL-MODEL", "R8-CONTROLLED-REGISTRY", "R9-IMPORT-NAME-SERVICE")
LOCAL_RULES = (R1, R4)
JAR4IMPORT = "--jar4import-url"

# §7.6's forbidden names (documentary there, enforced here on a record's envKeys).
FORBIDDEN_ENV = re.compile(
    r"(?i:ENABLE_DEPENDENCY_RESOLVER_V2|JAVASRC_FETCH_DEPENDENCIES|JAVASRC_JDK_PATH|JAVA_TOOL_OPTIONS|_JAVA_OPTIONS|"
    r"JDK_JAVA_OPTIONS|.*_PROXY|NO_PROXY|GOPROXY|GOFLAGS|GOPATH|NUGET_.*|WSL_INTEROP|WSLENV|CCG_RUN_TOKEN|NODE_OPTIONS|NODE_PATH|"
    r"NODE_EXTRA_CA_CERTS|NODE_PRESERVE_SYMLINKS|NPM_CONFIG_.*|PYTHONPATH|PYTHONHOME|PYTHONSTARTUP|PYTHONUSERBASE|PIP_.*|UV_.*|"
    r"LD_PRELOAD|BASH_ENV)")

ESCAPE_REQUIRED = ("A1a", "A1b", "A1c", "A2a", "A2b", "A3a", "A3b", "A3c", "A4a:/src", "A4a:/out", "A4a:/tmp", "A4a:/home/ccg",
                   "A4a-memfd", "A4b", "A4c", "A5b", "A5c", "A5d", "A7")
ESCAPE_MIN_CHECKS = 60


@dataclass
class Row:
    id: str
    status: str
    title: str
    details: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"id": self.id, "status": self.status, "title": self.title, "details": self.details}


def rid(rule: object) -> str:
    return str(rule).split("-")[0]


def _ts(rec: dict) -> datetime | None:
    return window.parse_ts(rec.get("ts"))


# ------------------------------------------------------------------ loading


@dataclass
class Scenario:
    dir: Path
    egress: list[dict]
    sandbox: list[dict]
    invalid: list[str]            # schema failures and unparseable lines of both audit files (row 8.14)
    conns: list[dict]
    conn_errors: list[str]
    conn_exists: bool
    markers: dict
    marker_errors: list[str]
    facts: dict

    @property
    def start(self):
        return self.markers["start"]

    @property
    def flush(self):
        return self.markers["flush"]

    @property
    def detector_dir(self) -> Path:
        return self.dir / "audit" / "detector"

    def in_window(self, rec: dict) -> bool:
        t = _ts(rec)
        if t is None:
            return False
        if self.start is not None and t < self.start:
            return False
        if self.flush is not None and t > self.flush:
            return False
        return True

    def policies(self) -> list[dict]:
        return [r for r in self.egress if r.get("recordType") == "policy"]

    def decisions(self) -> list[dict]:
        return [r for r in self.egress if r.get("recordType") == "decision"]

    def closes(self) -> list[dict]:
        return [r for r in self.egress if r.get("recordType") == "close"]

    def policy_of(self, run_id: object) -> dict | None:
        return next((p for p in self.policies() if p.get("runId") == run_id), None)

    def local_set(self, policy: dict | None) -> set[str]:
        return {m["repoId"] for m in (policy or {}).get("modes", []) if m.get("effective") == "local-only"}

    def allows(self) -> list[dict]:
        return [d for d in self.decisions() if d.get("decision") == "allow"]


def _read_audit_file(path: Path, validate: Callable[[dict], list[str]]) -> tuple[list[dict], list[str]]:
    recs: list[dict] = []
    bad: list[str] = []
    try:
        data = path.read_bytes()
    except OSError:
        return recs, bad
    lines = data.split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()
    elif lines:
        lines.pop()  # a trailing partial line is skipped, as every reader of the audit skips it (§5.1)
    for n, raw in enumerate(lines, 1):
        if not raw:
            continue
        try:
            rec = json.loads(raw)
        except ValueError:
            bad.append(f"{path.name}:{n}: not JSON")
            continue
        if not isinstance(rec, dict):
            bad.append(f"{path.name}:{n}: not an object")
            continue
        for why in validate(rec):
            bad.append(f"{path.name}:{n}: {why}")
        recs.append(rec)
    return recs, bad


def _egress_errors(rec: dict) -> list[str]:
    from ccg.broker import audit
    import jsonschema
    try:
        audit.validate_record(rec)
    except jsonschema.ValidationError as exc:
        return [f"invalid broker record: {exc.message[:160]}"]
    return []


def _sandbox_errors(rec: dict) -> list[str]:
    from ccg import sandbox_audit
    return [f"invalid sandbox record: {e}" for e in sandbox_audit.record_errors(rec)]


def load(scenario_dir: Path) -> Scenario:
    d = Path(scenario_dir)
    eg, bad1 = _read_audit_file(d / "audit" / "egress.jsonl", _egress_errors)
    sb, bad2 = _read_audit_file(d / "audit" / "sandbox.jsonl", _sandbox_errors)
    conns, cerrs, exists = window.read_jsonl(d / "audit" / "detector" / window.CONN_FILE)
    markers, merrs = window.read_markers(d / "audit" / "detector")
    try:
        facts = json.loads((d / "facts.json").read_text())
    except (OSError, ValueError):
        facts = {}
    return Scenario(d, eg, sb, bad1 + bad2, conns, cerrs, exists, markers, merrs, facts)


def _read(path: Path) -> str | None:
    try:
        return path.read_text(errors="replace")
    except OSError:
        return None


# ------------------------------------------------------------------ the conn verdict and P-d


def conn_verdict(sc: Scenario) -> window.Verdict:
    times = [t for t in (_ts(r) for r in sc.egress + sc.sandbox) if t is not None]
    pol = [t for t in (_ts(r) for r in sc.policies()) if t is not None]
    return window.judge(sc.conns, sc.start, sc.flush, errors=sc.conn_errors + [f"marker {e}" for e in sc.marker_errors],
                        policy_times=pol, record_times=times, conn_file_exists=sc.conn_exists, residue=sc.markers["residue"],
                        final_residue=sc.markers["final_residue"])


# ------------------------------------------------------------------ controls (§8.1 "Controls inside the main run")


@dataclass
class Control:
    id: str
    what: str
    found: int
    needed: int = 1
    matches: list[dict] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.found >= self.needed

    def as_dict(self) -> dict:
        return {"id": self.id, "what": self.what, "found": self.found, "needed": self.needed, "ok": self.ok}


def _no_container_for(events_text: str | None, audit_id: str) -> tuple[bool, str]:
    """`docker events` shows no container created or started for ``audit_id`` (the runner names it ``ccg-sbx-<auditId>`` and labels it)."""
    if events_text is None:
        return False, "docker-events.jsonl is missing"
    n = 0
    for line in events_text.splitlines():
        if not line.strip():
            continue
        try:
            ev = json.loads(line)
        except ValueError:
            return False, "docker-events.jsonl has a line that is not JSON"
        n += 1
        attrs = (ev.get("Actor") or {}).get("Attributes") or {}
        if ev.get("Type") == "container" and (audit_id in json.dumps(attrs) or audit_id in str(ev.get("id", ""))):
            return False, f"a container event names the refused invocation: {ev.get('Action')}"
    if n == 0:
        return False, "docker-events.jsonl is empty: the capture proves nothing (the scenario's other containers produced no event)"
    return True, f"{n} container events captured, none for {audit_id}"


def find_controls(sc: Scenario) -> list[Control]:
    pols = {p["runId"]: p for p in sc.policies() if "runId" in p}
    decs = sc.decisions()

    def run_kind(d):
        return (pols.get(d.get("runId")) or {}).get("runKind")

    def run_local_only(d):
        return bool((pols.get(d.get("runId")) or {}).get("systemLocalOnly"))

    local_eps = {e.get("endpoint") for e in _manifest_egress(sc).get("local", [])}
    out: list[Control] = []

    def add(cid, what, matches, needed=1):
        out.append(Control(cid, what, len(matches), needed, matches))

    add("C1", "R4 from access.idp whose body has an extra field, and one naming a repository outside the system "
              "(payload-shape denies)",
        [d for d in decs if d.get("decision") == "deny" and rid(d.get("rule")) == "R4" and d.get("reason") == "payload-shape"
         and d.get("component") == "access.idp" and d.get("bytes", 0) > 0 and d.get("bodySha256")], 2)
    add("C2", "a hosted model call from answer.interpret in the mixed system (R5, system-local-only-model)",
        [d for d in decs if d.get("decision") == "deny" and d.get("rule") == R5 and d.get("component") == "answer.interpret"
         and d.get("reason") == "system-local-only-model" and run_local_only(d)])
    add("C3", "a git-fetch from build.git in a refresh run to a local endpoint with a listed path (R2, not-in-p1:git-fetch)",
        [d for d in decs if d.get("decision") == "deny" and d.get("rule") == R2 and d.get("component") == "build.git"
         and d.get("purpose") == "git-fetch" and d.get("reason") == "not-in-p1:git-fetch" and run_kind(d) == "refresh"
         and d.get("endpoint") in local_eps and d.get("target", {}).get("path")])
    add("C4", "a dependency-download from build.pkgmgr in a refresh run to a local endpoint with a listed path "
              "(R8, not-in-p1:controlled-download)",
        [d for d in decs if d.get("decision") == "deny" and d.get("rule") == R8 and d.get("component") == "build.pkgmgr"
         and d.get("purpose") == "dependency-download" and d.get("reason") == "not-in-p1:controlled-download"
         and run_kind(d) == "refresh" and d.get("endpoint") in local_eps and d.get("target", {}).get("path")])
    add("C5", "a request with no token (R0, token-invalid)",
        [d for d in decs if d.get("rule") == "R0-INVALID" and d.get("reason") == "token-invalid" and d.get("runId") is None])
    add("C6", "a request with codebases = [] (R0, codebases-empty)",
        [d for d in decs if d.get("rule") == "R0-INVALID" and d.get("reason") == "codebases-empty"])
    flag = [s for s in sc.sandbox if s.get("refused") is True and s.get("refusalReason") == "forbidden-flag"
            and s.get("exitCode") == 64 and JAR4IMPORT in s.get("toolArgv", []) and s.get("frontend") == "kotlin2cpg"]
    ok_ev, why_ev = (False, "no such record") if not flag else _no_container_for(_read(sc.dir / "docker-events.jsonl"),
                                                                                  flag[0]["auditId"])
    add("C7", "kotlin2cpg --jar4import-url: sandbox record refused, forbidden-flag, exit 64, and no container per docker "
              "events (" + why_ev + ")", flag if ok_ev else [])
    add("C8", "a refresh with --override hosted on a local-only repository (P2-OVERRIDE-REFUSED, effective local-only, "
              "overrideRefused)",
        [p for p in sc.policies() if p.get("override") == "hosted" and p.get("overrideRefused") is True
         and any(m.get("rule") == "P2-OVERRIDE-REFUSED" and m.get("default") == "local-only" and m.get("effective") == "local-only"
                 for m in p.get("modes", []))])
    add("C9", "a refresh with --override local on a hosted-default repository (P3-OVERRIDE-LOCAL, effective local-only)",
        [p for p in sc.policies() if p.get("override") == "local"
         and any(m.get("rule") == "P3-OVERRIDE-LOCAL" and m.get("default") == "hosted" and m.get("effective") == "local-only"
                 for m in p.get("modes", []))])
    return out


def _manifest_egress(sc: Scenario) -> dict:
    try:
        from ccg import manifest
        return dict(manifest.load_manifest(sc.dir / "manifest.yaml").egress)
    except Exception:
        return {"local": [], "outbound": [], "hosted": []}


# ------------------------------------------------------------------ probe text (netprobe.pl, probe.pl)


def parse_probe(text: str | None) -> dict:
    """``{done, checks, fail, ids: {id: [(PASS|FAIL, detail)]}, targets: (tcp, udp), sent}`` from a probe's output."""
    out: dict[str, Any] = {"done": False, "checks": 0, "fail": 0, "ids": {}, "targets": None, "sent": None}
    for line in (text or "").splitlines():
        m = re.match(r"CHECK (\S+) (PASS|FAIL) ?(.*)", line)
        if m:
            out["ids"].setdefault(m.group(1), []).append((m.group(2), m.group(3)))
            continue
        m = re.match(r"PROBE-DONE checks=(\d+) fail=(\d+)$", line)
        if m:
            out.update(done=True, checks=int(m.group(1)), fail=int(m.group(2)))
            continue
        m = re.match(r"TARGETS tcp=(\d+) udp=(\d+)$", line)
        if m:
            out["targets"] = (int(m.group(1)), int(m.group(2)))
            continue
        m = re.match(r"PROBE-SENT det=(\d+)$", line)
        if m:
            out["sent"] = int(m.group(1))
    return out


def _all_pass(p: dict, cid: str) -> bool:
    got = p["ids"].get(cid, [])
    return bool(got) and all(s == "PASS" for s, _ in got)


def component_probe_problems(text: str | None, snapshot: dict, probe_args: dict) -> dict[str, list[str]]:
    """Problems per precondition (``P-a``, ``P-b``, ``P-c``) of one ``netprobe.pl`` output."""
    prob: dict[str, list[str]] = {"P-a": [], "P-b": [], "P-c": []}
    p = parse_probe(text)
    if text is None:
        return {k: ["probe output missing"] for k in prob}
    if not p["done"]:
        return {k: ["no PROBE-DONE: the probe did not run to completion"] for k in prob}
    if p["checks"] != sum(len(v) for v in p["ids"].values()):
        prob["P-a"].append(f"PROBE-DONE says {p['checks']} checks but the output holds {sum(len(v) for v in p['ids'].values())}")
    if p["fail"] != sum(1 for v in p["ids"].values() for s, _ in v if s == "FAIL"):
        prob["P-a"].append("PROBE-DONE's fail count disagrees with the CHECK lines")
    for cid in ("A1b-v4", "A1b-v6"):
        if not _all_pass(p, cid):
            prob["P-a"].append(f"{cid}: {p['ids'].get(cid, 'not run')}")
    addrs = len(snapshot.get("v4", [])) + len(snapshot.get("v6", []))
    min_tcp = addrs * len(snapshot.get("tcp", [])) + 2
    min_udp = addrs * len(snapshot.get("udp", [])) + 2
    if not p["targets"] or p["targets"][0] < min_tcp or p["targets"][1] < min_udp:
        prob["P-a"].append(f"targets {p['targets']} fewer than the host snapshot needs (tcp>={min_tcp}, udp>={min_udp})")
    for cid in ("A1d-tcp", "A1d-udp"):
        bad = [d for s, d in p["ids"].get(cid, []) if s != "PASS"]
        if not p["ids"].get(cid):
            prob["P-a"].append(f"{cid}: no target probed")
        elif bad:
            prob["P-a"].append(f"{cid}: {len(bad)} target(s) not unreachable, e.g. {bad[0]}")
    for t in probe_args.get("pos", []):
        if not any(s == "PASS" and d.startswith(t + " ") for s, d in p["ids"].get("posctl-tcp", [])):
            prob["P-a"].append(f"positive control {t} did not accept (the probe is broken)")
    for t in probe_args.get("det", []):
        if not any(s == "PASS" and d.startswith(t + " ") for s, d in p["ids"].get("posctl-det", [])):
            prob["P-a"].append(f"positive control (detector) {t} did not accept (the probe is broken)")
    if not (probe_args.get("pos") or probe_args.get("det")):
        prob["P-a"].append("no positive-control target was given")
    for cid in ("A2-ahosts", "A2-localhost", "A2-embedded"):
        if not _all_pass(p, cid):
            prob["P-b"].append(f"{cid}: {p['ids'].get(cid, 'not run')}")
    for cid in ("A3-files", "A3-broker-present", "A3-named", "A3-abstract", "A3-connect", "A3-readonly", "A7"):
        if not _all_pass(p, cid):
            prob["P-c"].append(f"{cid}: {p['ids'].get(cid, 'not run')}")
    return prob


# ------------------------------------------------------------------ preconditions


def pre_a_b_c(sc: Scenario) -> tuple[dict[str, list[str]], int]:
    """P-a, P-b, P-c from every component's start and end probes; also returns the total detector-probe connections sent."""
    out: dict[str, list[str]] = {"P-a": [], "P-b": [], "P-c": []}
    probes = sc.facts.get("probes", [])
    comps = sc.facts.get("components", [])
    sent = 0
    if not comps:
        for k in out:
            out[k].append("facts.json names no component container")
    for comp in comps:
        for phase in ("start", "end"):
            entry = next((p for p in probes if p.get("component") == comp and p.get("phase") == phase), None)
            if entry is None:
                for k in out:
                    out[k].append(f"{comp}: no {phase} probe recorded")
                continue
            text = _read(sc.dir / entry.get("file", "probes/-missing-"))
            for k, v in component_probe_problems(text, entry.get("snapshot", {}), entry).items():
                out[k] += [f"{comp}/{phase}: {x}" for x in v]
            sent += parse_probe(text)["sent"] or 0
    return out, sent


PROBE_SLACK = timedelta(seconds=2)


def probe_attribution_problems(sc: Scenario, probes_logged: Sequence[dict]) -> list[str]:
    """Every ``probe.jsonl`` record matches exactly one expected (component, phase) probe, one record per detector target: its ``probe``
    label is ``<component>/<phase>``, its ``srcAddr`` is that component container's address (``facts.json`` ``probes[].ip``), its ``dstPort``
    one of the probe's detector targets (each once) and its time inside the probe's own ``t0``..``t1`` interval (plus a 2 s slack).
    Anything else, and any expected record that is absent, is a problem: a forged or replayed positive-control connection cannot hide."""
    probs: list[str] = []
    expected: dict[str, dict] = {}
    for comp in sc.facts.get("components", []):
        for phase in ("start", "end"):
            entry = next((p for p in sc.facts.get("probes", []) if p.get("component") == comp and p.get("phase") == phase), None)
            if entry is None:
                continue  # already a P-a/b/c problem
            ports = [int(str(t).rsplit(":", 1)[1]) for t in entry.get("det", []) if ":" in str(t)]
            t0, t1 = window.parse_ts(entry.get("t0")), window.parse_ts(entry.get("t1"))
            if not (entry.get("ip") and t0 and t1 and t0 <= t1 and ports):
                probs.append(f"{comp}/{phase}: facts.json lacks the probe's ip, t0/t1 or detector targets")
                continue
            expected[f"{comp}/{phase}"] = {"ip": entry["ip"], "ports": sorted(ports), "t0": t0, "t1": t1, "recs": []}
    for rec in probes_logged:
        e = expected.get(rec["probe"])
        if e is None:
            probs.append(f"probe record with label {rec['probe']!r} matches no expected (component, phase) probe")
            continue
        e["recs"].append(rec)
        t = window.parse_ts(rec["ts"])
        if rec["srcAddr"] != e["ip"]:
            probs.append(f"probe record {rec['probe']}: source {rec['srcAddr']} is not the component's address {e['ip']}")
        if t is None or not (e["t0"] - PROBE_SLACK <= t <= e["t1"] + PROBE_SLACK):
            probs.append(f"probe record {rec['probe']} at {rec['ts']} is outside that probe's own interval {entry_ts(e['t0'])}..{entry_ts(e['t1'])}")
    for label, e in expected.items():
        got = sorted(r["dstPort"] for r in e["recs"])
        if got != e["ports"]:
            probs.append(f"probe {label}: the detector filed records for ports {got}, expected exactly {e['ports']}")
    return probs


def entry_ts(t: datetime) -> str:
    return window.format_ts(t)


def pre_d(sc: Scenario) -> list[str]:
    v = conn_verdict(sc)
    probs = [f for f in v.failures if not (f.startswith("conn-in-window") or f.startswith("conn-outside-window"))]
    probes_logged, perrs, pexists = window.read_jsonl(sc.detector_dir / window.PROBE_FILE, window.PROBE_FIELDS)
    sent = pre_a_b_c(sc)[1]
    if not pexists:
        probs.append("probe.jsonl does not exist")
    probs += perrs
    if len(probes_logged) != sent:
        probs.append(f"the detector filed {len(probes_logged)} probe connection(s) but the probes report sending {sent}")
    probs += probe_attribution_problems(sc, probes_logged)
    return probs


def _non_local_hosts(egress: dict) -> list[tuple[str, dict, dict]]:
    return [(sec, e, urlsplit(e["endpoint"])) for sec in ("outbound", "hosted") for e in egress.get(sec, [])]


def pre_e_manifest(manifest_path: Path, resolver: Callable[[str], list[str]], iface: Callable[[], Any] | None = None) -> list[str]:
    """P-e over a manifest alone (so the harness can run it BEFORE the run): every non-local entry's host is under ``.invalid`` and the
    broker's own address path (``addr.validate`` with the given resolver) answers ``resolve-failed`` for it. Only an entry whose host passed
    the ``.invalid`` check is resolved (a real hostname in a manifest is reported, never looked up)."""
    from ccg import manifest
    from ccg.broker import addr
    probs: list[str] = []
    try:
        m = manifest.load_manifest(manifest_path)
    except Exception as exc:
        return [f"manifest.yaml does not load: {type(exc).__name__}: {exc}"]
    egress = dict(m.egress)
    iface = iface or addr.read_interfaces
    entries = _non_local_hosts(egress)
    invalid = []
    for sec, e, parts in entries:
        if not (parts.hostname or "").endswith(".invalid"):
            probs.append(f"{sec} entry {e['id']}: host {parts.hostname!r} is not under .invalid")
        else:
            invalid.append((sec, e, parts))
    if not any(sec == "hosted" and e["kind"] == "model" for sec, e, _ in entries):
        probs.append("the manifest has no hosted model entry")
    if not any(sec == "outbound" and e["kind"] == "idp" for sec, e, _ in entries):
        probs.append("the manifest has no outbound idp entry (the R4 controls need one)")

    def resolves_denied(res: Callable[[str], list[str]]) -> list[str]:
        """The entries the broker's own address path (`ccg.broker.addr.validate`) does NOT answer with resolve-failed."""
        leaks = []
        for sec, e, _ in invalid:
            try:
                addr.validate(egress, sec, e, resolver=res, iface=iface)
                leaks.append(f"{e['id']}: validated an address")
            except addr.AddressDenied as exc:
                if exc.reason != "resolve-failed":
                    leaks.append(f"{e['id']}: denied as {exc.reason}, not resolve-failed")
        return leaks

    probs += [f"{x}" for x in resolves_denied(resolver)]
    if invalid:  # positive control: a stub resolver that answers one of the names must be flagged
        target = invalid[0][2].hostname + "."
        flagged = resolves_denied(lambda name, _t=target: ["93.184.216.34"] if name == _t else resolver(name))
        if not any(invalid[0][1]["id"] in f for f in flagged):
            probs.append("positive control failed: a stub resolver answering a scenario name was not flagged")
    elif not entries:
        probs.append("the manifest has no non-local entry to check")
    return probs


def pre_e(sc: Scenario, resolver: Callable[[str], list[str]], iface: Callable[[], Any] | None = None) -> list[str]:
    """P-e after the run: ``pre_e_manifest`` over the scenario's manifest copy, and every policy record's allow-list digest is that manifest's."""
    from ccg import manifest
    from ccg.broker import policy
    probs = pre_e_manifest(sc.dir / "manifest.yaml", resolver, iface)
    try:
        digest = policy.LoadedSystem.from_manifest(manifest.load_manifest(sc.dir / "manifest.yaml")).digest
    except Exception as exc:
        probs.append(f"cannot compute the allow-list digest: {exc}")
    else:
        for p in sc.policies():
            if p.get("allowListDigest") != digest:
                probs.append(f"policy record {p.get('auditId')}: allowListDigest differs from the scenario manifest's")
    return probs


def pre_f(sc: Scenario) -> list[str]:
    try:
        from ccg import manifest
        m = manifest.load_manifest(sc.dir / "manifest.yaml")
    except Exception as exc:
        return [f"manifest.yaml does not load: {type(exc).__name__}"]
    modes = [r.policy.egress for r in m.repos]
    probs = []
    if "local-only" not in modes:
        probs.append("no local-only repository")
    if "hosted" not in modes:
        probs.append("no hosted-default repository")
    return probs


def pre_g(sc: Scenario, controls: list[Control]) -> list[str]:
    probs = []
    interp = [d for d in sc.allows() if d.get("component") == "answer.interpret" and rid(d.get("rule")) == "R1"]
    if not interp:
        probs.append("no R1 allow from answer.interpret")
    if sc.facts.get("summaryEnabled") and not [d for d in sc.allows() if d.get("component") == "answer.summarize"
                                               and rid(d.get("rule")) == "R1"]:
        probs.append("the summary is enabled but there is no R1 allow from answer.summarize")
    try:
        from ccg import manifest
        repos = [r.repo_id for r in manifest.load_manifest(sc.dir / "manifest.yaml").repos]
    except Exception:
        repos = []
    launched = [s for s in sc.sandbox if s.get("refused") is False]
    for repo in repos:
        if not any(repo in s.get("codebases", []) for s in launched):
            probs.append(f"no non-refused sandbox record for repository {repo}")
    if not repos:
        probs.append("the repositories of the system are unknown")
    for c in controls:
        if c.id in ("C8", "C9") and not c.ok:
            probs.append(f"override policy record missing ({c.id})")
    return probs


def pre_escape(sc: Scenario) -> list[str]:
    text = _read(sc.dir / sc.facts.get("escape", {}).get("file", "probes/escape-container.txt"))
    if text is None:
        return ["the §7.3 escape-axis probe output is missing (it runs inside the scenario)"]
    p = parse_probe(text)
    probs = []
    if not p["done"]:
        return ["no PROBE-DONE: the escape probe did not run to completion"]
    if p["checks"] < ESCAPE_MIN_CHECKS:
        probs.append(f"only {p['checks']} checks (< {ESCAPE_MIN_CHECKS})")
    if p["fail"] != 0:
        probs.append(f"{p['fail']} check(s) failed")
    for cid in ESCAPE_REQUIRED:
        if not _all_pass(p, cid):
            probs.append(f"{cid}: {p['ids'].get(cid, 'not run')}")
    for cid in ("A1d-tcp", "A1d-udp", "A1e-tcp", "A1e-udp"):
        if not p["ids"].get(cid) or any(s != "PASS" for s, _ in p["ids"][cid]):
            probs.append(f"{cid}: not all targets passed or none probed")
    if not any(d.startswith("1.1.1.1:443") for s, d in p["ids"].get("A1d-tcp", [])):
        probs.append("1.1.1.1:443 was not probed")
    return probs


def pre_detector_control(sc: Scenario) -> list[str]:
    return window.control_check(sc.dir / "detector-control")


# ------------------------------------------------------------------ the rows


def _backend_flag_ok(rec: dict) -> tuple[bool, str]:
    argv = rec.get("launcherArgv", [])
    if rec.get("backend") == "container":
        nets = [i for i, a in enumerate(argv) if a == "--network"]
        if len(nets) == 1 and nets[0] + 1 < len(argv) and argv[nets[0] + 1] == "none":
            return True, "--network none"
        return False, "container launcherArgv lacks exactly one `--network none`"
    if rec.get("backend") == "namespace":
        if "--unshare-all" in argv and "--share-net" not in argv:
            return True, "--unshare-all"
        return False, "namespace launcherArgv lacks --unshare-all or carries --share-net"
    return False, f"unknown backend {rec.get('backend')!r}"


def _run_argv_audit(frontend: str, argv: Sequence[str]) -> tuple[str | None, list[str], bool]:
    """``(digest, hits, ran)`` from the runner's own §7.5 function (``argv_audit.sh``)."""
    if any("\n" in str(a) for a in argv) or "\n" in frontend:
        return None, [], False  # a newline in an element could forge the helper's line protocol (DIGEST/HIT/DONE): never audit it
    try:
        cp = subprocess.run(["/bin/bash", "-p", str(HERE / "argv_audit.sh"), frontend, "--", *argv], capture_output=True,
                            text=True, timeout=60, env={"PATH": "/usr/bin:/bin"})
    except (OSError, subprocess.SubprocessError):
        return None, [], False
    digest, hits, done = None, [], False
    for line in cp.stdout.splitlines():
        if line.startswith("DIGEST "):
            digest = line[7:]
        elif line.startswith("HIT "):
            hits.append(line[4:])
        elif line == "DONE":
            done = True
    return digest, hits, done


def compute_rows(sc: Scenario, controls: list[Control], *, argv_auditor: Callable[[str, Sequence[str]], tuple] = _run_argv_audit,
                 trace_row: Row | None = None) -> list[Row]:
    ctl = {c.id: c for c in controls}
    pols = {p["runId"]: p for p in sc.policies()}
    allows = [d for d in sc.allows() if sc.in_window(d)]
    decs = [d for d in sc.decisions() if sc.in_window(d)]
    sbx = [s for s in sc.sandbox if sc.in_window(s)]
    launched = [s for s in sbx if s.get("refused") is False]
    rows: list[Row] = []

    def row(i, status, title, details=()):
        rows.append(Row(i, status, title, list(details)))

    def verdict_row(i, title, fails, witness_ok=True, details=()):
        st = PASS if not fails and witness_ok else FAIL
        det = list(details) + list(fails)
        if not witness_ok:
            det.append("the witness set is empty: the check ran over nothing")
        row(i, st, title, det)

    # 8.1
    v = conn_verdict(sc)
    fails = [f for f in v.failures if f.startswith(("conn-", "detector-log", "marker-", "window-", "flush-", "final-"))]
    wit = []
    for d in allows:
        run = pols.get(d.get("runId"))
        if set(d.get("codebases") or []) & sc.local_set(run):
            wit.append(d)
            if rid(d.get("rule")) not in ("R1", "R4"):
                fails.append(f"allow {d['auditId']} touching a local-only codebase has rule {rid(d.get('rule'))}")
    verdict_row("8.1", "A local-only codebase makes zero connections outside the allow-list", fails, bool(wit),
                [f"{len(sc.conns)} conn record(s); {len(wit)} allow(s) touch a local-only codebase"])
    # 8.2
    priv = [d for d in allows if d.get("privateAddress")]
    bad = [d for d in allows if rid(d.get("rule")) in ("R2", "R3", "R4")]
    row("8.2", FAIL if bad else VACUOUS, "Only own git remotes, vulndb, IdP, access-check or PR services (allow-listed) are outbound",
        [f"an {rid(d['rule'])} allow exists ({d['auditId']}): under option C and P-e it cannot, so this row is not vacuous" for d in bad]
        + [f"allow with privateAddress=true: {d['auditId']}" for d in priv]
        + ([] if bad else [f"vacuous under option C and P-e; {len(priv)} allow(s) with privateAddress=true (expected none)"]))
    # 8.3
    f83 = [] if ctl["C1"].ok else [f"the two R4 payload-shape controls: {ctl['C1'].found} of {ctl['C1'].needed} present"]
    for d in allows:
        if rid(d.get("rule")) == "R4":
            if not (d.get("transport") == "structured" and d.get("payloadClass") == "repo-names-and-identity"
                    and set(d.get("payloadFields") or []) <= {"repoNames", "identity.subject", "identity.issuer"}):
                f83.append(f"R4 allow {d['auditId']} carries more than repository names and identity")
    row("8.3", PASS if not f83 else FAIL, "Request bodies carry nothing beyond repository names and identity",
        f83 + ["negative half computed; the positive half (an R4 allow) is vacuous under P-e"] * (0 if f83 else 1))
    # 8.4
    vul = [p for p in sc.policies() if p.get("runKind") == "vulndb"]
    row("8.4", FAIL if vul else VACUOUS, "The vulnerability-database job is separate and never sees repository content",
        [f"a vulndb run exists ({p['runId']}): no P1 caller opens one" for p in vul] or ["vacuous: no vulndb run has a P1 caller"])
    # 8.5
    f85 = [] if ctl["C4"].ok else ["the dependency-download control (R8 deny) is absent"]
    f85 += [f"allow {d['auditId']} has purpose dependency-download" for d in allows if d.get("purpose") == "dependency-download"]
    if not launched:
        f85.append("no non-refused sandbox record in the window: the check ran over nothing")
    for s in launched:
        ok, why = _backend_flag_ok(s)
        if not ok:
            f85.append(f"sandbox record {s['auditId']}: {why}")
    row("8.5", PASS if not f85 else FAIL, "A local-only build makes no dependency download",
        f85 + [f"{len(launched)} non-refused sandbox record(s) checked; the proxied half is vacuous; `network` and "
               "`netnsDistinct` are constants and prove nothing"])
    # 8.6
    f86: list[str] = []
    if not launched:
        f86.append("no non-refused sandbox record in the window")
    for s in launched:
        aid = s["auditId"]
        digest, hits, ran = argv_auditor(s.get("frontend", ""), s.get("toolArgv", []))
        if not ran:
            f86.append(f"{aid}: the §7.5 re-run did not complete")
            continue
        if digest != (s.get("flagCheck") or {}).get("listDigest"):
            f86.append(f"{aid}: the flag lists changed since the record (listDigest differs): cannot re-run against the lists at listDigest")
        if hits:
            f86.append(f"{aid}: §7.5 re-run finds forbidden element(s) {hits}")
        if (s.get("flagCheck") or {}).get("forbiddenHits"):
            f86.append(f"{aid}: forbiddenHits is not empty")
        ok, why = _backend_flag_ok(s)
        if not ok:
            f86.append(f"{aid}: {why}")
        for k in s.get("envKeys", []):
            if FORBIDDEN_ENV.fullmatch(k):
                f86.append(f"{aid}: envKeys carries forbidden name {k}")
    if not ctl["C7"].ok:
        f86.append("the forbidden-flag control is absent or not refused with exit 64")
    row("8.6", PASS if not f86 else FAIL, "Every fetch-capable frontend is invoked with its fetch flags off", f86 + [f"{len(launched)} record(s) re-audited"])
    # 8.7
    interp = [d for d in allows if d.get("component") == "answer.interpret" and (pols.get(d.get("runId")) or {}).get("systemLocalOnly")]
    f87 = [f"allow {d['auditId']} from answer.interpret has rule {rid(d.get('rule'))}, not R1" for d in interp if rid(d.get("rule")) != "R1"]
    if not ctl["C2"].ok:
        f87.append("the hosted-model control (R5 deny) is absent")
    verdict_row("8.7", "A question touching only hosted repositories is still interpreted by a local model", f87, bool(interp),
                [f"{len(interp)} answer.interpret allow(s) in systemLocalOnly runs"])
    # 8.8
    f88, n88 = [], 0
    for p in sc.policies():
        if p.get("runKind") != "answer" or not (set(p.get("codebases", [])) & sc.local_set(p)):
            continue
        n88 += 1
        for d in decs:
            if d.get("runId") == p["runId"]:
                if d.get("requestMode") != "local-only":
                    f88.append(f"decision {d['auditId']} of run {p['runId']} has requestMode {d.get('requestMode')}")
                if d.get("decision") == "allow" and rid(d.get("rule")) not in ("R1", "R4"):
                    f88.append(f"allow {d['auditId']} of run {p['runId']} has rule {rid(d.get('rule'))}")
    verdict_row("8.8", "An answer touching a local-only repository is handled entirely locally", f88, n88 > 0,
                [f"{n88} answer run(s) touching a local-only repository"])
    # 8.9
    mc = [d for d in allows if d.get("purpose") == "model-call" and (pols.get(d.get("runId")) or {}).get("systemLocalOnly")]
    f89 = [f"model-call allow {d['auditId']} has rule {rid(d.get('rule'))}, not R1" for d in mc if rid(d.get("rule")) != "R1"]
    verdict_row("8.9", "In a system containing a local-only repository, every model call goes to a local endpoint", f89, bool(mc),
                [f"{len(mc)} model-call allow(s) in systemLocalOnly runs"])
    # 8.10, 8.11
    row("8.10", PASS if ctl["C8"].ok else FAIL, "A hosted override on a local-only codebase is refused",
        [] if ctl["C8"].ok else ["no policy record shows override=hosted, P2-OVERRIDE-REFUSED, effective local-only, overrideRefused"])
    row("8.11", PASS if ctl["C9"].ok else FAIL, "A hosted-default codebase run with a local override stays local",
        [] if ctl["C9"].ok else ["no policy record shows a hosted default, P3-OVERRIDE-LOCAL, effective local-only"])
    # 8.12
    bad12 = [d for d in allows if rid(d.get("rule")) in ("R8", "R9")]
    row("8.12", FAIL if bad12 else VACUOUS, "A hosted-default build downloads dependencies only through the controlled path",
        [f"an {rid(d['rule'])} allow exists ({d['auditId']})" for d in bad12] or ["vacuous (spec3 §6: the controlled path is deferred)"])
    # 8.13
    f813 = [] if ctl["C7"].ok else ["the --jar4import-url control is absent or not refused"]
    f813 += [f"sandbox record {s['auditId']} carries {JAR4IMPORT} and was not refused" for s in sbx
             if JAR4IMPORT in s.get("toolArgv", []) and s.get("refused") is not True]
    row("8.13", PASS if not f813 else FAIL, "No import-name service", f813 + ["residue computed; the controlled half is vacuous"])
    # 8.14
    bad14 = [x for x in sc.invalid]
    row("8.14", PASS if not bad14 and (sc.egress or sc.sandbox) else FAIL,
        "Every record validates against the closed-enum schemas", bad14[:20] + ([] if (sc.egress or sc.sandbox) else ["no records"])
        + [f"{len(sc.egress)} broker and {len(sc.sandbox)} sandbox records read"])
    # 8.15
    f815, n815 = [], 0
    for d in allows:
        p = pols.get(d.get("runId"))
        if p and p.get("runKind") == "answer" and sc.local_set(p):
            n815 += 1
            if d.get("codebases") == [] or d.get("codebases") is None:
                f815.append(f"allow {d['auditId']} in a run with a local-only codebase declares no codebases")
    f815 += [f"an {rid(d['rule'])} allow exists ({d['auditId']})" for d in allows if rid(d.get("rule")) in ("R8", "R9")]
    refresh_allows = [d for d in allows if (pols.get(d.get("runId")) or {}).get("runKind") == "refresh"]
    verdict_row("8.15", "No request can opt out of the sticky rule by under-declaring", f815, n815 > 0,
                [f"(i) computed over {n815} answer-run allow(s); (ii) and (iii) vacuous; {len(refresh_allows)} refresh-run allow(s) (informational)"])
    # 8.16
    rows.append(trace_row or Row("8.16", UNRUN, "The broker's own connections match its decisions", ["no trace evaluated"]))
    # added for P1: one close per allow
    f_close = []
    cl: dict[str, int] = {}
    for c in sc.closes():
        cl[c.get("decisionAuditId")] = cl.get(c.get("decisionAuditId"), 0) + 1
    for d in allows:
        n = cl.get(d["auditId"], 0)
        if n != 1:
            f_close.append(f"allow {d['auditId']} has {n} close record(s)" + (" (an interrupted exchange)" if n == 0 else ""))
    allow_ids = {d["auditId"] for d in sc.allows()}
    f_close += [f"close {c['auditId']} names no allow" for c in sc.closes() if c.get("decisionAuditId") not in allow_ids]
    row("close", PASS if not f_close else FAIL, "Every allow in the window has exactly one close", f_close + [f"{len(allows)} allow(s)"])
    return rows


# ------------------------------------------------------------------ row 8.16: the strace trace


_CALL = re.compile(r"^(?:\[pid\s+(\d+)\]\s+|(\d+)\s+)?(\d+\.\d+)\s+(connect|sendto|sendmsg|sendmmsg|socket)\((.*)$")


def parse_trace(text: str) -> list[dict]:
    """Network events of an ``strace -f -ttt -e trace=%network`` trace: ``{t, pid, call, family, addr, port, line}``.

    ``family`` is the ``sa_family_t`` named in the call (``AF_INET``, ``AF_INET6``, ``AF_UNIX``, ...; for ``socket`` the domain, with
    ``addr``/``port`` None; for a ``connect`` whose address cannot be parsed ``UNPARSED``, which ``check_trace`` reports); ``addr``/``port``
    are None for a family without them. A ``<... resumed>`` line is skipped (the arguments are in the ``<unfinished ...>`` half) and so is a
    ``sendto``/``sendmsg``/``sendmmsg`` with no destination address (a connected socket, whose ``connect`` was seen)."""
    events = []
    for line in text.splitlines():
        m = _CALL.match(line)
        if not m:
            continue
        pid = m.group(1) or m.group(2)
        call, rest = m.group(4), m.group(5)
        ev = {"t": float(m.group(3)), "pid": pid, "call": call, "family": None, "addr": None, "port": None, "line": line.strip()}
        if call == "socket":
            dom = re.match(r"(AF_\w+|PF_\w+|\d+)", rest)
            ev["family"] = dom.group(1).replace("PF_", "AF_") if dom else "UNPARSED"
            events.append(ev)
            continue
        fam = re.search(r"sa_family=(AF_\w+)", rest)
        if not fam:
            if call == "connect":
                ev["family"] = "UNPARSED"
                events.append(ev)
            continue
        ev["family"] = fam.group(1)
        if fam.group(1) == "AF_INET":
            a, p = re.search(r'sin_addr=inet_addr\("([^"]+)"\)', rest), re.search(r"sin_port=htons\((\d+)\)", rest)
            ev["addr"], ev["port"] = (a.group(1) if a else None), (int(p.group(1)) if p else None)
        elif fam.group(1) == "AF_INET6":
            a = re.search(r'inet_pton\(AF_INET6, "([^"]+)"', rest)
            p = re.search(r"sin6_port=htons\((\d+)\)", rest)
            ev["addr"], ev["port"] = (a.group(1) if a else None), (int(p.group(1)) if p else None)
        events.append(ev)
    return events


def _norm_addr(a: str | None) -> str | None:
    if a is None:
        return None
    import ipaddress
    try:
        return str(ipaddress.ip_address(a))
    except ValueError:
        return a


_NON_SOCKET_FAMILIES = {"AF_UNIX", "AF_LOCAL", "AF_NETLINK", "AF_UNSPEC"}


def check_trace(events: Sequence[dict], allows: Sequence[tuple[str, int, float]], t0: float, t1: float) -> list[str]:
    """Violations of §8.2 row 8.16 over ``events``: every non-AF_UNIX ``connect`` (and every addressed ``sendto``/``sendmsg``/``sendmmsg``)
    in ``[t0, t1]`` (epoch seconds) must equal the ``resolvedAddr`` and port of an allow decided at or before it, **each allow legitimising
    one call** (matched one-to-one, earliest first). Traffic to port 53 before any allow is a resolver query and is a violation by the same
    rule. A ``socket`` of any domain but AF_UNIX/AF_INET/AF_INET6/AF_NETLINK (AF_PACKET, raw...) and a ``connect`` whose address cannot be
    parsed are violations. AF_NETLINK (glibc's interface queries) and AF_UNSPEC (a UDP disconnect) carry no destination and are skipped.
    ``allows`` are ``(resolvedAddr, port, decision epoch seconds)``."""
    probs = []
    pool = [[_norm_addr(a), p, t, False] for a, p, t in allows]
    for ev in events:
        if not (t0 <= ev["t"] <= t1):
            continue
        if ev["call"] == "socket":
            if ev["family"] not in ("AF_UNIX", "AF_LOCAL", "AF_INET", "AF_INET6", "AF_NETLINK"):
                probs.append(f"socket() of domain {ev['family']} (not a stream/datagram socket the broker needs): {ev['line'][:120]}")
            continue
        if ev["family"] in _NON_SOCKET_FAMILIES:
            continue
        key = (_norm_addr(ev["addr"]), ev["port"])
        if ev["family"] not in ("AF_INET", "AF_INET6") or key[0] is None or key[1] is None:
            probs.append(f"{ev['call']} on an unexpected or unparsable family {ev['family']}: {ev['line'][:120]}")
            continue
        hit = next((x for x in pool if not x[3] and (x[0], x[1]) == key and ev["t"] >= x[2] - 0.002), None)
        if hit is None:
            kind = "resolver traffic" if key[1] == 53 else "a connection"
            probs.append(f"{kind} to {key[0]}:{key[1]} matches no unused allow decided before it ({ev['call']})")
        else:
            hit[3] = True
    return probs


def trace_row(sc: Scenario) -> Row:
    title = "The broker's own connections match its decisions"
    st = sc.facts.get("strace") or {}
    if not st.get("provisioned"):
        return Row("8.16", UNRUN, title, ["strace is not provisioned on this host: the row is unrun, not passed"])
    probs: list[str] = []
    trace_text = _read(sc.dir / st.get("trace", "trace/broker.strace"))
    ctl_text = _read(sc.dir / st.get("control", "trace/control.strace"))
    if trace_text is None:
        probs.append("the broker's trace file is missing")
    if ctl_text is None:
        probs.append("the positive-control trace is missing")
    if sc.start is None or sc.flush is None:
        probs.append("no window (P-d markers missing)")
    if probs:
        return Row("8.16", FAIL, title, probs)
    ev = parse_trace(trace_text)
    t0, t1 = sc.start.timestamp(), sc.flush.timestamp()
    allows = []
    for d in sc.allows():
        parts = urlsplit(d.get("endpoint") or "")
        port = parts.port or {"http": 80, "https": 443}.get(parts.scheme, 0)
        t = _ts(d)
        allows.append((d.get("resolvedAddr"), port, t.timestamp() if t else 0.0))
    probs += check_trace(ev, allows, t0, t1)
    # positive control: a trace of a small program dialling an unlisted address must be flagged by the same parser
    cev = parse_trace(ctl_text)
    cprobs = check_trace(cev, [], 0.0, float("inf"))
    dialed = st.get("controlTarget", "")
    if not cprobs or (dialed and not any(dialed in c for c in cprobs)):
        probs.append(f"positive control failed: the parser did not flag the dial to {dialed or 'the unlisted address'}")
    n_net = sum(1 for e in ev if e["call"] != "socket" and e["family"] in ("AF_INET", "AF_INET6") and t0 <= e["t"] <= t1)
    n_allow = sum(1 for d in sc.allows() if sc.in_window(d))
    if not ev:
        probs.append("the broker's trace holds no network call at all: a trace that captured nothing proves nothing")
    elif n_net < n_allow:
        probs.append(f"the trace shows {n_net} IP call(s) in the window for {n_allow} allow(s): every allow implies a connect, so the trace is incomplete")
    return Row("8.16", FAIL if probs else PASS, title,
               probs + [f"{len(ev)} traced call(s), {n_net} IP call(s) in the window, {len(allows)} allow(s); control flagged {len(cprobs)}"])


# ------------------------------------------------------------------ the report


def evaluate(scenario_dir: Path, *, resolver: Callable[[str], list[str]] | None = None, iface: Callable[[], Any] | None = None,
             argv_auditor: Callable[[str, Sequence[str]], tuple] = _run_argv_audit, resolver_name: str = "real",
             self_test: Sequence[str] = ()) -> dict:
    """``self_test``: stand-ins this evaluation itself used (the CLI's ``--stub-resolver``), added to ``facts.json``'s ``selfTest``."""
    from ccg.broker import addr
    sc = load(scenario_dir)
    recorded = sc.facts.get("selfTest")
    st_probs = [] if isinstance(recorded, list) and all(isinstance(x, str) for x in recorded) else [
        "facts.json has no `selfTest` list (run.sh always writes it, [] for an acceptance run)"]
    selftest = sorted({*(recorded if not st_probs else []), *self_test})
    controls = find_controls(sc)
    abc, _sent = pre_a_b_c(sc)
    pre = {
        "P-a": abc["P-a"], "P-b": abc["P-b"], "P-c": abc["P-c"],
        "P-d": pre_d(sc),
        "P-e": pre_e(sc, resolver or addr.default_resolver, iface),
        "P-f": pre_f(sc),
        "P-g": pre_g(sc, controls),
        "E-7.3": pre_escape(sc),
        "C-detector": pre_detector_control(sc),
        "S-facts": st_probs,
    }
    rows = compute_rows(sc, controls, argv_auditor=argv_auditor, trace_row=trace_row(sc))
    fail = (any(pre.values()) or any(not c.ok for c in controls) or any(r.status == FAIL for r in rows))
    unrun = any(r.status == UNRUN for r in rows)
    reasons = (["row 8.16 is unrun (strace is not provisioned)"] if unrun else []) + (
        [f"self-test stand-ins were used: {', '.join(selftest)} (an acceptance run has none)"] if selftest else [])
    return {
        "overall": "fail" if fail else ("incomplete" if reasons else "pass"),
        "incompleteReasons": [] if fail else reasons,
        "selfTest": selftest,
        "resolver": resolver_name,
        "preconditions": {k: {"ok": not v, "problems": v} for k, v in pre.items()},
        "controls": [c.as_dict() for c in controls],
        "rows": [r.as_dict() for r in rows],
    }


def render(rep: dict) -> str:
    out = [f"AC-8 part 1: {rep['overall'].upper()} (P-e resolver: {rep['resolver']}; self-test stand-ins: {', '.join(rep['selfTest']) or 'none'})"]
    out += [f"  incomplete because: {r}" for r in rep["incompleteReasons"]]
    out += ["", "Preconditions:"]
    for k, v in rep["preconditions"].items():
        out.append(f"  {k:11s} {'ok' if v['ok'] else 'FAIL'}")
        out += [f"      - {p}" for p in v["problems"][:8]]
    out += ["", "Controls:"]
    out += [f"  {c['id']} {'ok  ' if c['ok'] else 'FAIL'} ({c['found']}/{c['needed']}) {c['what']}" for c in rep["controls"]]
    out += ["", "Rows:"]
    for r in rep["rows"]:
        out.append(f"  {r['id']:6s} {r['status'].upper():8s} {r['title']}")
        out += [f"      - {d}" for d in r["details"][:8]]
    return "\n".join(out)


def _stub_resolver(_name: str) -> list[str]:  # answers nothing and sends nothing: no DNS query leaves the host
    raise OSError("stub resolver: NXDOMAIN")


def _main(argv: list[str]) -> int:
    args = list(argv)
    stub = "--stub-resolver" in args
    if stub:
        args.remove("--stub-resolver")
    if args[:1] == ["pre-e"]:
        if len(args) != 2:
            print("usage: assertions.py pre-e <manifest.yaml> [--stub-resolver]", file=sys.stderr)
            return 64
        from ccg.broker import addr
        probs = pre_e_manifest(Path(args[1]), _stub_resolver if stub else addr.default_resolver)
        for p in probs:
            print(f"FAIL P-e: {p}")
        print("P-e (resolver: %s): %s" % ("stub, no DNS query made" if stub else "real", "FAIL" if probs else "ok"))
        return 1 if probs else 0
    js = None
    if "--json" in args:
        i = args.index("--json")
        js = Path(args[i + 1])
        del args[i:i + 2]
    if len(args) != 1:
        print("usage: assertions.py <scenario dir> [--stub-resolver] [--json <file>] | pre-e <manifest.yaml> [--stub-resolver]", file=sys.stderr)
        return 64
    rep = evaluate(Path(args[0]), resolver=_stub_resolver if stub else None, resolver_name="stub (no DNS query made)" if stub else "real",
                   self_test=["stub-resolver"] if stub else [])
    print(render(rep))
    if js:
        js.write_text(json.dumps(rep, indent=1) + "\n")
    return {"pass": 0, "fail": 1, "incomplete": 3}[rep["overall"]]


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
