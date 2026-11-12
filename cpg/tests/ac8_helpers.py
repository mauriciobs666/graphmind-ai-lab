"""Shared builders for the P1-31 AC-8 harness tests (`test_ac8_window.py`, `test_ac8_assertions.py`).

`make_scenario` writes a complete, GOOD scenario directory in the layout `cpg/sandbox/ac8/assertions.py` reads (egress records
made by the real `Broker` through `broker_helpers.Harness`, sandbox records shaped as the runner writes them, probe output in
`netprobe.pl`'s and `probe.pl`'s format, a strace trace, markers), so a test breaks exactly one thing and reads the row it should
turn red. Expected values live in the tests as literals; nothing here computes one.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

from broker_helpers import FakeExchange, Harness, Resolver, fake_iface

# CCG_AC8_DIR: the mutation harness (sandbox/tests/mutants_ac8.py) points the tests at a mutated copy of cpg/sandbox/ac8.
AC8 = Path(os.environ.get("CCG_AC8_DIR") or Path(__file__).resolve().parent.parent / "sandbox" / "ac8")
SANDBOX = AC8.parent
sys.path.insert(0, str(AC8))
import assertions as A  # noqa: E402
import window as W  # noqa: E402

LOCAL = "http://127.0.0.1:18234"
IDP = "https://idp.ac8.invalid:443"
HOSTED = "https://api.ac8.invalid:443"
MODEL_PATH, IDP_PATH = "/v1/chat/completions", "/who"
SNAPSHOT = {"v4": ["192.168.0.64", "172.17.0.1"], "v6": ["fe80::1"], "tcp": [53, 6379], "udp": [53]}
DET_TARGETS = ["172.18.0.3:80", "172.18.0.3:443", "172.18.0.3:3128"]
POS_TARGETS = ["172.18.0.4:6379"]
ESCAPE_IDS = ("A1a", "A1b", "A1c", "A2a", "A2b", "A3a", "A3b", "A3c", "A4a:/src", "A4a:/out", "A4a:/tmp", "A4a:/home/ccg",
              "A4a-memfd", "A4b", "A4c", "A5b", "A5c", "A5d", "A7")


def egress_section() -> dict:
    return {"schemaVersion": 1,
            "local": [{"id": "stub", "kind": "model", "endpoint": LOCAL, "paths": [MODEL_PATH]}],
            "outbound": [{"id": "idp", "kind": "idp", "endpoint": IDP, "paths": [IDP_PATH]}],
            "hosted": [{"id": "hosted", "kind": "model", "endpoint": HOSTED, "paths": [MODEL_PATH]}]}


def write_manifest(path: Path, *, eg: dict | None = None, repos=(("front", "local-only"), ("docs", "hosted"))) -> None:
    doc = {"schemaVersion": 1, "system": "ac8smoke", "egress": eg if eg is not None else egress_section(),
           "repos": [{"repoId": r, "path": r, "policy": {"egress": m}} for r, m in repos]}
    path.write_text(yaml.safe_dump(doc))


def netprobe_text(snapshot=SNAPSHOT, pos=POS_TARGETS, det=DET_TARGETS, *, drop=(), fail=(), skip_pos=False, skip_det=False) -> str:
    """A passing `netprobe.pl` output. ``drop``: check ids left out; ``fail``: check ids reported FAIL."""
    lines: list[tuple[str, str, str]] = []  # (id, PASS|FAIL, detail)

    def add(cid, detail="x"):
        if cid not in drop:
            lines.append((cid, "FAIL" if cid in fail else "PASS", detail))
    add("A1b-v4")
    add("A1b-v6")
    n_t = n_u = 0
    for ip in snapshot["v4"] + snapshot["v6"]:
        for p in snapshot["tcp"]:
            add("A1d-tcp", f"{ip}:{p} ENETUNREACH")
            n_t += 1
        for p in snapshot["udp"]:
            add("A1d-udp", f"{ip}:{p} ENETUNREACH")
            n_u += 1
    for cid in ("A1d-tcp", "A1d-udp"):
        add(cid, "1.1.1.1:443 ENETUNREACH")
        add(cid, "[2606:4700:4700::1111]:443 ENETUNREACH")
    n_t += 2
    n_u += 2
    if not skip_pos:
        for t in pos:
            add("posctl-tcp", f"{t} CONNECTED")
    if not skip_det:
        for t in det:
            add("posctl-det", f"{t} CONNECTED")
    for cid in ("A2-ahosts", "A2-localhost", "A2-embedded", "A3-files", "A3-broker-present", "A3-named", "A3-abstract", "A3-connect", "A3-readonly", "A7"):
        add(cid)
    out = [f"CHECK {c} {s} {d}" for c, s, d in lines]
    out.insert(sum(1 for c, _, _ in lines if c.startswith("A1")) , f"TARGETS tcp={n_t} udp={n_u}")
    out.append(f"PROBE-SENT det={0 if skip_det else len(det)}")
    out.append(f"PROBE-DONE checks={len(lines)} fail={sum(1 for _, s, _ in lines if s == 'FAIL')}")
    return "\n".join(out) + "\n"


def escape_text(*, drop=(), fail=(), checks=134) -> str:
    out = []

    def add(cid, detail=""):
        if cid not in drop:
            out.append(f"CHECK {cid} {'FAIL' if cid in fail else 'PASS'} {detail}")
    for cid in ESCAPE_IDS:
        add(cid)
    add("A1d-tcp", "1.1.1.1:443 ENETUNREACH")
    add("A1d-udp", "1.1.1.1:443 ENETUNREACH")
    add("A1e-tcp", "127.0.0.1:1234 ECONNREFUSED")
    add("A1e-udp", "127.0.0.1:1234 ECONNREFUSED")
    n = len(out)
    out += [f"CHECK pad{i} PASS" for i in range(max(0, checks - n))]
    nf = sum(1 for line in out if " FAIL" in line)
    out.append(f"PROBE-DONE checks={len(out)} fail={nf}")
    return "\n".join(out) + "\n"


def list_digest() -> str:
    return hashlib.sha256((SANDBOX / "forbidden-flags.txt").read_bytes() + (SANDBOX / "allowed-flags.txt").read_bytes()).hexdigest()


def sandbox_record(aid: str, ts: str, *, repo="front", refused=False, frontend="pysrc2cpg", tool_argv=None, launcher=None,
                   env_keys=None, exit_code=None, reason=None, hits=(), digest=None, backend="container") -> dict:
    launched = not refused
    return {
        "schemaVersion": 1, "recordType": "sandbox", "auditId": aid, "ts": ts, "backend": backend,
        "toolchain": {"image": "ccg-toolchain:x", "imageId": "sha256:" + "a" * 64}, "job": "parse", "egressMode": "offline",
        "network": "none", "networkName": None,
        "launcherArgv": ([] if refused else launcher if launcher is not None else
                         ["docker", "run", "--rm", "--network", "none", "--hostname", "localhost", "--read-only"]),
        "toolArgv": tool_argv if tool_argv is not None else (["/src", "--language", "pythonsrc", "--output", f"/out/{aid}.cpg.bin"]
                                                              if launched else ["/src", "--language", "kotlin", "--jar4import-url", "http://x.invalid/"]),
        "frontend": frontend, "script": None,
        "flagCheck": {"listDigest": digest or list_digest(), "forbiddenHits": list(hits), "allowedHits": [], "liftedHits": []},
        "envKeys": env_keys if env_keys is not None else ([] if refused else ["PATH", "JAVA_HOME", "HOME", "TMPDIR", "LANG", "LC_ALL", "JAVA_OPTS"]),
        "mounts": [], "preconditions": {"interopPE": "disabled", "binfmtPinned": False, "seccomp": "builtin", "vsock": "blocked",
                                        "netnsDistinct": launched},
        "refused": refused, "refusalReason": reason if refused else None,
        "exitCode": exit_code if exit_code is not None else (64 if refused else 0), "codebases": [repo], "runId": None, "durationMs": 100,
    }


class Scenario:
    """A good scenario on disk; tests mutate files through the helpers and call `evaluate`."""

    def __init__(self, root: Path, *, strace: bool = True):
        self.dir = root / "scenario"
        self.audit = self.dir / "audit"
        self.det = self.audit / "detector"
        self.strace = strace
        for d in (self.det, self.dir / "probes", self.dir / "trace", self.dir / "detector-control"):
            d.mkdir(parents=True)
        write_manifest(self.dir / "manifest.yaml")
        self.build()

    # -- building
    def build(self):
        now = datetime.now(timezone.utc)
        self.start = now - timedelta(seconds=5)
        (self.dir.parent / "h").mkdir()
        h = Harness(self.dir.parent / "h", systems=None)
        # replace the harness's manifest with the scenario's
        from ccg import manifest as mf
        from ccg.broker import policy
        ls = policy.LoadedSystem.from_manifest(mf.load_manifest(self.dir / "manifest.yaml"))
        h.broker.systems = {ls.system: ls}
        h.broker.resolver = Resolver({})
        h.broker.iface = fake_iface()
        h.broker.exchanger = FakeExchange()
        self.h = h
        sysid = ls.system

        def rq(g, component, purpose, pc, method, endpoint, path, cbs, body=None, token="__t__"):
            r = h.broker.handle({"op": "request", "token": g["token"] if token == "__t__" else token,
                                 "request": {"component": component, "purpose": purpose, "payloadClass": pc,
                                             "target": {"method": method, "path": path}, "endpoint": endpoint, "codebases": cbs},
                                 "body": body})
            assert r["ok"], r
            return r
        model = {"messages": [{"role": "user", "content": "q"}]}
        ident = {"subject": "alice", "issuer": IDP}
        g = h.open("answer", "*", system=sysid)
        rq(g, "answer.interpret", "model-call", "repo-derived", "POST", LOCAL, MODEL_PATH, ["front"], model)
        rq(g, "access.idp", "idp", "repo-names-and-identity", "POST", IDP, IDP_PATH, ["front"],
           {"repoNames": ["front"], "identity": ident, "extra": 1})
        rq(g, "access.idp", "idp", "repo-names-and-identity", "POST", IDP, IDP_PATH, ["front"],
           {"repoNames": ["nope"], "identity": ident})
        rq(g, "answer.interpret", "model-call", "repo-derived", "POST", LOCAL, MODEL_PATH, ["front"], model, token=None)
        rq(g, "answer.interpret", "model-call", "repo-derived", "POST", LOCAL, MODEL_PATH, [], model)
        h.broker.handle({"op": "close_run", "token": g["token"]})
        g = h.open("answer", ["docs"], system=sysid)
        rq(g, "answer.interpret", "model-call", "repo-derived", "POST", LOCAL, MODEL_PATH, ["docs"], model)
        rq(g, "answer.interpret", "model-call", "repo-derived", "POST", HOSTED, MODEL_PATH, ["docs"], model)
        h.broker.handle({"op": "close_run", "token": g["token"]})
        g = h.open("refresh", "*", system=sysid)
        rq(g, "build.git", "git-fetch", "none", "GET", LOCAL, MODEL_PATH, ["front"])
        rq(g, "build.pkgmgr", "dependency-download", "none", "GET", LOCAL, MODEL_PATH, ["front"])
        h.broker.handle({"op": "close_run", "token": g["token"]})
        h.open("refresh", ["front"], "hosted", system=sysid)
        h.open("refresh", ["docs"], "local", system=sysid)
        self.audit.mkdir(exist_ok=True)
        (self.audit / "egress.jsonl").write_bytes((h.audit_dir / "egress.jsonl").read_bytes())
        ts = W.format_ts(datetime.now(timezone.utc))
        recs = [sandbox_record("11111111-1111-4111-8111-111111111111", ts, repo="front", frontend="pysrc2cpg"),
                sandbox_record("22222222-2222-4222-8222-222222222222", ts, repo="docs", frontend="pysrc2cpg"),
                sandbox_record("33333333-3333-4333-8333-333333333333", ts, repo="front", refused=True, frontend="kotlin2cpg",
                               reason="forbidden-flag", hits=["--jar4import-url"])]
        (self.audit / "sandbox.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
        self.flush = datetime.now(timezone.utc) + timedelta(seconds=5)
        self.write_markers(self.det, self.start, self.flush)
        (self.det / "conn.jsonl").write_text("")
        # probes: each (component, phase) is one probe interval; the detector filed one labelled record per target (ports 80, 443, 3128)
        probes, precs = [], []
        for ci, comp in enumerate(("answer", "front")):
            for pi, phase in enumerate(("start", "end")):
                f = f"probes/{phase}-{comp}.txt"
                (self.dir / f).write_text(netprobe_text())
                t0 = self.start - timedelta(seconds=3 - pi)
                t1 = t0 + timedelta(seconds=1)
                probes.append({"phase": phase, "component": comp, "file": f, "snapshot": SNAPSHOT, "pos": POS_TARGETS, "det": DET_TARGETS,
                               "ip": f"172.18.0.{10 + ci}", "t0": W.format_ts(t0), "t1": W.format_ts(t1)})
                for k, port in enumerate((80, 443, 3128)):
                    precs.append({"ts": W.format_ts(t0 + timedelta(seconds=0.5)), "srcAddr": f"172.18.0.{10 + ci}", "srcPort": 4000 + len(precs),
                                  "dstAddr": "172.18.0.3", "dstPort": port, "bytesOut": 0, "bytesIn": 20, "probe": f"{comp}/{phase}"})
        (self.det / "probe.jsonl").write_text("".join(json.dumps(r) + "\n" for r in precs))
        # detector control: the CONTROL_CONNS real conns inside the window and the one correct-nonce probe record, residue 0
        cs, cf = datetime(2026, 1, 1, tzinfo=timezone.utc), datetime(2026, 1, 1, 0, 0, 10, tzinfo=timezone.utc)
        ctl = self.dir / "detector-control"
        self.write_markers(ctl, cs, cf)
        (ctl / "conn.jsonl").write_text("".join(json.dumps({"ts": "2026-01-01T00:00:05.000Z", "srcAddr": "172.18.0.2", "srcPort": 5000 + i,
                                                            "dstAddr": "172.18.0.3", "dstPort": 80, "bytesOut": 0, "bytesIn": 18}) + "\n"
                                                for i in range(W.CONTROL_CONNS)))
        (ctl / "probe.jsonl").write_text(json.dumps({"ts": "2026-01-01T00:00:04.000Z", "srcAddr": "172.18.0.2", "srcPort": 4999,
                                                     "dstAddr": "172.18.0.3", "dstPort": 80, "bytesOut": 0, "bytesIn": 42,
                                                     "probe": W.CONTROL_PROBE_LABEL}) + "\n")
        (self.dir / "probes" / "escape-container.txt").write_text(escape_text())
        (self.dir / "docker-events.jsonl").write_text(json.dumps(
            {"Type": "container", "Action": "start", "Actor": {"ID": "c1", "Attributes": {"name": "ccg-ac8-x-detector"}}}) + "\n")
        allows = [json.loads(x) for x in (self.audit / "egress.jsonl").read_text().splitlines()]
        allows = [r for r in allows if r["recordType"] == "decision" and r["decision"] == "allow"]
        tr = "".join(f"100 {W.parse_ts(a['ts']).timestamp() + 0.001:.6f} connect(7, {{sa_family=AF_INET, sin_port=htons(18234), "
                     f"sin_addr=inet_addr(\"127.0.0.1\")}}, 16) = -1 EINPROGRESS (Operation now in progress)\n" for a in allows)
        tr += '100 1.000000 connect(3, {sa_family=AF_UNIX, sun_path="/var/run/nscd/socket"}, 110) = -1 ENOENT (No such file)\n'
        tr += '100 1.000001 sendto(6, "abc"..., 3, 0, NULL, 0) = 3\n'
        (self.dir / "trace" / "broker.strace").write_text(tr)
        (self.dir / "trace" / "control.strace").write_text(
            '200 5.000000 connect(3, {sa_family=AF_INET, sin_port=htons(48510), sin_addr=inet_addr("127.0.0.1")}, 16) = -1 EINPROGRESS (x)\n')
        facts = {"scenarioId": "t", "components": ["answer", "front"], "summaryEnabled": False, "probes": probes, "selfTest": [],
                 "escape": {"file": "probes/escape-container.txt", "backend": "container"}, "resolver": "stub",
                 "strace": ({"provisioned": True, "trace": "trace/broker.strace", "control": "trace/control.strace",
                             "controlTarget": "127.0.0.1:48510"} if self.strace else {"provisioned": False})}
        (self.dir / "facts.json").write_text(json.dumps(facts))

    @staticmethod
    def write_markers(d: Path, start: datetime, flush: datetime, residue: int = 0, final_residue: int | None = 0):
        """``final_residue=None`` writes no final marker (a detector that did not finish its termination drain)."""
        if final_residue is None:
            (d / "final.marker").unlink(missing_ok=True)
        else:
            (d / "final.marker").write_text(json.dumps({"marker": "final", "ts": W.format_ts(flush + timedelta(seconds=1)), "residue": final_residue}) + "\n")
        (d / "start.marker").write_text(json.dumps({"marker": "start", "ts": W.format_ts(start)}) + "\n")
        (d / "flush.marker").write_text(json.dumps({"marker": "flush", "ts": W.format_ts(flush), "residue": residue}) + "\n")

    # -- mutation helpers
    def egress(self) -> list[dict]:
        return [json.loads(x) for x in (self.audit / "egress.jsonl").read_text().splitlines()]

    def write_egress(self, recs: list[dict]):
        (self.audit / "egress.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))

    def sandbox(self) -> list[dict]:
        return [json.loads(x) for x in (self.audit / "sandbox.jsonl").read_text().splitlines()]

    def write_sandbox(self, recs: list[dict]):
        (self.audit / "sandbox.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))

    def evaluate(self, **kw) -> dict:
        kw.setdefault("resolver", _nxdomain)
        kw.setdefault("iface", _no_iface)
        kw.setdefault("argv_auditor", _clean_auditor)
        return A.evaluate(self.dir, **kw)


def _nxdomain(_name):
    raise OSError("NXDOMAIN")


def _no_iface():
    from ccg.broker import addr
    return addr.IfaceData()


def _clean_auditor(_fe, _argv):
    return list_digest(), [], True


def row(rep: dict, rid: str) -> dict:
    return next(r for r in rep["rows"] if r["id"] == rid)
