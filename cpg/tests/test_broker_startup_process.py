"""D9b — the broker as a real process (egress-spec5 section 7 items 6, 8, 9, 12 and the credential/fake-key scan).

Two ways to start it, both real subprocesses:
  * `python -m ccg.broker.server` (the product entry: closed-environment re-exec, trusted uids `{0}`, hosted check on)
    for every refusal the production entry makes. A conforming start needs a root-owned `/etc/llmgateway/ccg` chain, so it
    is tier 2 (`sudo` is not used here).
  * `tests/broker_launcher.py`, which calls `server.main` with the test-only keyword seams (`trusted_uids = {0, own uid}`)
    for conforming starts and for item 12 against a REAL holder subprocess (`llm-gateway/tests/holder_harness.py`, with
    `tests/holder_resolving_launcher.py` so the provider can have a dotted name that only the holder resolves).

The fake key `sk-ant-FAKE-...` is planted in the holder's key file only. A hosted entry cannot be a loopback literal (the
static check, which this entry point always runs), so the stand-in provider is `provider.example.test`.
"""
from __future__ import annotations

import base64
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

from broker_helpers import hosted_call, hosted_egress, write_manifest
from ccg.broker.audit import read_records
from ccg.broker.client import BrokerClient
from gateway_helpers import ScriptedHolder  # also puts llm-gateway/tests on sys.path for `holder_harness`
import holder_harness
from holder_harness import CALLER_SECRET, MODEL, Env, HolderProc, StandIn, body_for, tree_search

CPG = Path(__file__).resolve().parent.parent
LAUNCHER = Path(__file__).resolve().parent / "broker_launcher.py"
HOLDER_LAUNCHER = Path(__file__).resolve().parent / "holder_resolving_launcher.py"
ME = os.getuid()
REFUSAL = "application/vnd.llmgateway.refusal+json"
HOST = "provider.example.test"
SECRET_HEX = CALLER_SECRET.hex()


@pytest.fixture
def tmp_path():
    base = Path.home() / ".cache" / "ccg-broker-tests"
    base.mkdir(parents=True, exist_ok=True)
    d = Path(tempfile.mkdtemp(dir=base))
    yield d
    shutil.rmtree(d, ignore_errors=True)


# ------------------------------------------------------------------ starting the broker


class Proc:
    def __init__(self, popen: subprocess.Popen, sock: Path, audit: Path, errfile: Path):
        self.p, self.sock, self.audit, self.errfile = popen, sock, audit, errfile

    @property
    def err(self) -> str:
        return self.errfile.read_text() if self.errfile.exists() else ""

    def wait(self, timeout=20) -> int:
        return self.p.wait(timeout=timeout)

    def stop(self) -> int:
        if self.p.poll() is None:
            self.p.send_signal(signal.SIGTERM)
        try:
            return self.p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.p.kill()
            raise

    def records(self, kind=None):
        f = self.audit / "egress.jsonl"
        recs = list(read_records(f)) if f.exists() else []
        return [r for r in recs if kind is None or r["recordType"] == kind]


def spawn(tmp_path: Path, args: list[str], *, module=False, launcher_spec=None, wait=True, name="b") -> Proc:
    d = tmp_path / name
    d.mkdir(exist_ok=True, mode=0o700)
    audit, errfile = d / "audit", d.parent / f"{name}.stderr"
    sock = d / "broker.sock"
    argv = [str(a) for a in args]
    if "--socket" not in argv:
        argv += ["--socket", str(sock)]
    env = {"PATH": os.environ["PATH"], "CCG_AUDIT_DIR": str(audit)}
    if module:
        cmd = [sys.executable, "-m", "ccg.broker.server", *argv]
    else:
        spec = {"argv": argv, "trusted": [0, ME], **(launcher_spec or {})}
        cmd = [sys.executable, "-I", str(LAUNCHER), json.dumps(spec)]
    popen = subprocess.Popen(cmd, cwd=str(CPG) if module else str(d), env=env, stdout=subprocess.DEVNULL,
                             stderr=errfile.open("wb"))
    proc = Proc(popen, sock, audit, errfile)
    if wait:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if popen.poll() is not None:
                raise AssertionError(f"broker exited {popen.returncode}: {proc.err}")
            if sock.exists():
                try:
                    BrokerClient(str(sock), timeout=2).close_run("0" * 64)
                    return proc
                except Exception:
                    pass
            time.sleep(0.05)
        popen.kill()
        raise AssertionError("the broker did not listen in time")
    return proc


def manifest(d: Path, **kw) -> Path:
    return write_manifest(d, name=kw.pop("name", "installed.yaml"), **kw)


def cred(d: Path, text=SECRET_HEX + "\n", mode=0o600, name="broker.cred") -> Path:
    p = d / name
    p.write_text(text)
    p.chmod(mode)
    return p


def untouched(proc: Proc):
    assert not proc.sock.exists() and not (proc.sock.parent / "broker.lock").exists() and not proc.audit.exists()


# ------------------------------------------------------------------ item 8, the product entry: every refusal


@pytest.mark.parametrize("extra,rc,needle", [
    (["--key-holder-socket", "{h}", "--holder-uid", "{u}"], 64, "--require-owner-protected"),
    (["--key-holder-socket", "{h}", "--require-owner-protected"], 64, "--holder-uid"),
    (["--holder-cred", "{c}"], 64, "valid only with --key-holder-socket"),
    (["--holder-uid", "{u}"], 64, "valid only with --key-holder-socket"),
    (["--key-holder-socket", "{h}", "--require-owner-protected", "--holder-uid", "{me}", "--holder-cred", "{c}"], 64,
     "other than the broker's own"),
])
def test_the_product_entry_refuses_each_option_violation_with_no_socket_lock_or_audit(tmp_path, extra, rc, needle):
    d = tmp_path / "b"
    d.mkdir(mode=0o700)
    subst = {"{h}": str(tmp_path / "h.sock"), "{u}": str(ME + 1), "{me}": str(ME), "{c}": str(cred(tmp_path))}
    args = ["--system", manifest(d), *[subst.get(a, a) for a in extra]]
    p = spawn(tmp_path, args, module=True, wait=False)
    assert p.wait() == rc and needle in p.err
    untouched(p)


def test_the_product_entry_refuses_a_manifest_that_is_not_root_owned_with_69_naming_the_file(tmp_path):
    d = tmp_path / "b"
    d.mkdir(mode=0o700)
    m = manifest(d)
    p = spawn(tmp_path, ["--system", m, "--require-owner-protected"], module=True, wait=False)
    assert p.wait() == 69
    assert str(m) in p.err and "untrusted uid" in p.err  # trusted uids are `{0}` here and no flag changes that
    untouched(p)


def test_the_product_entry_has_no_way_to_widen_the_trusted_uids_or_to_skip_the_hosted_check(tmp_path):
    d = tmp_path / "b"
    d.mkdir(mode=0o700)
    m = manifest(d, eg=hosted_egress("https://127.0.0.1:8443"))
    for flag in ("--trusted-uids", "--check-hosted-entries", "--no-check-hosted-entries"):
        p = spawn(tmp_path, ["--system", m, flag, "0"], module=True, wait=False)
        assert p.wait() == 64 and "unrecognized arguments" in p.err
    env_p = subprocess.run([sys.executable, "-m", "ccg.broker.server", "--system", str(m), "--socket", str(d / "s.sock")],
                           cwd=CPG, env={"PATH": os.environ["PATH"], "CCG_AUDIT_DIR": str(d / "audit"),
                                         "CCG_TRUSTED_UIDS": "0", "CCG_CHECK_HOSTED_ENTRIES": "0"},
                           capture_output=True, timeout=30)
    assert env_p.returncode == 64 and b"forbidden address literal" in env_p.stderr


@pytest.mark.parametrize("text,rc", [("00" * 31, 64), ("00" * 33, 64), ("zz" * 32, 64), ("", 64)])
def test_the_product_entry_exits_64_for_a_malformed_credential_read_after_the_reexec(tmp_path, text, rc):
    d = tmp_path / "b"
    d.mkdir(mode=0o700)
    c = cred(tmp_path, text)
    args = ["--system", manifest(d), "--require-owner-protected", "--key-holder-socket", tmp_path / "h.sock",
            "--holder-uid", ME + 1, "--holder-cred", c]
    p = spawn(tmp_path, args, module=True, wait=False)
    assert p.wait() == rc and "exactly 32 bytes" in p.err and (not text or text not in p.err)
    untouched(p)


def test_the_product_entry_exits_69_for_each_unusable_credential_file(tmp_path):
    d = tmp_path / "b"
    d.mkdir(mode=0o700)
    good = cred(tmp_path)
    (tmp_path / "link").symlink_to(good)
    os.link(good, tmp_path / "twin")
    cases = {"missing": tmp_path / "nope", "symlink": tmp_path / "link", "two links": tmp_path / "twin",
             "mode 0640": cred(tmp_path, mode=0o640, name="loose")}
    for name, c in cases.items():
        args = ["--system", manifest(d), "--require-owner-protected", "--key-holder-socket", tmp_path / "h.sock",
                "--holder-uid", ME + 1, "--holder-cred", c]
        p = spawn(tmp_path, args, module=True, wait=False)
        assert (name, p.wait()) == (name, 69) and str(c) in p.err, (name, p.err)
        untouched(p)


def test_a_good_credential_passes_the_intake_and_the_manifest_check_is_what_stops_the_product_entry(tmp_path):
    d = tmp_path / "b"
    d.mkdir(mode=0o700)
    args = ["--system", manifest(d), "--require-owner-protected", "--key-holder-socket", tmp_path / "h.sock",
            "--holder-uid", ME + 1, "--holder-cred", cred(tmp_path)]
    p = spawn(tmp_path, args, module=True, wait=False)
    assert p.wait() == 69 and "installed.yaml" in p.err and "holder credential" not in p.err
    untouched(p)


def test_a_malformed_cred_is_found_by_the_reexeced_process_not_the_parent(tmp_path):
    """The re-exec'd child is `python -I -m ccg.broker.server`: a hung start shows its argv."""
    d = tmp_path / "b"
    d.mkdir(mode=0o700)
    fifo = tmp_path / "fifo.cred"
    os.mkfifo(fifo, 0o600)  # opened O_NONBLOCK and refused as not a regular file: never blocks the start
    args = ["--system", manifest(d), "--require-owner-protected", "--key-holder-socket", tmp_path / "h.sock",
            "--holder-uid", ME + 1, "--holder-cred", fifo]
    p = spawn(tmp_path, args, module=True, wait=False)
    assert p.wait() == 69 and "not a regular file" in p.err


# ------------------------------------------------------------------ item 8, conforming and failing starts via the launcher


def hosted_args(tmp_path, d, m, c=None, holder_sock=None, holder_uid=ME + 1):
    return ["--system", m, "--require-owner-protected", "--key-holder-socket", holder_sock or tmp_path / "h.sock",
            "--holder-uid", holder_uid, "--holder-cred", c or cred(tmp_path)]


def test_a_conforming_manifest_and_credential_start_without_a_holder_and_an_r7_allow_is_an_upstream_error(tmp_path):
    d = tmp_path / "b"
    d.mkdir(mode=0o700)
    eg = hosted_egress(f"https://{HOST}:443", paths=["/v1/messages"])
    m = manifest(d, repos=(("a", "hosted"),), eg=eg)
    p = spawn(tmp_path, hosted_args(tmp_path, d, m), launcher_spec={"fake_uid": True})  # the holder never started
    try:
        c = BrokerClient(str(p.sock), timeout=10)
        g = c.open_run("answer", "shop", ["a"])
        dec, resp = c.request(g.token, hosted_call(endpoint=f"https://{HOST}:443", codebases=["a"],
                                                   target={"method": "POST", "path": "/v1/messages"}), {"m": 1})
    finally:
        assert p.stop() == 0
    assert dec.allowed and dec.rule == "R7-HOSTED-SERVICE" and resp is None
    [close] = p.records("close")
    assert close["closeReason"] == "upstream-error" and close["bytesOut"] == 0  # the holder socket did not exist


@pytest.mark.parametrize("how", ["user-writable", "group-writable", "symlink", "writable-ancestor", "untrusted-owner",
                                 "fifo", "loopback-hosted", "http-less-name-ok"])
def test_the_launcher_start_refuses_each_nonconforming_manifest_with_69_and_starts_the_conforming_one(tmp_path, how):
    d = tmp_path / "b"
    d.mkdir(mode=0o700)
    trusted = [0, ME]
    m = manifest(d)
    want = 69
    if how == "user-writable":
        m.chmod(0o666)
    elif how == "group-writable":
        m.chmod(0o664)
    elif how == "symlink":
        link = d / "link.yaml"
        link.symlink_to(m)
        m = link
    elif how == "writable-ancestor":
        sub = d / "etc"
        sub.mkdir()
        m = manifest(sub)
        sub.chmod(0o770)
    elif how == "untrusted-owner":
        trusted = [0]  # production's `{0}`: this account is not trusted
    elif how == "fifo":
        m = d / "fifo.yaml"
        os.mkfifo(m, 0o644)
    elif how == "loopback-hosted":
        m = manifest(d, eg=hosted_egress("https://127.0.0.1:8443"), name="lo.yaml")
        want = 64
    else:
        want = 0
    p = spawn(tmp_path, ["--system", m, "--require-owner-protected"], launcher_spec={"trusted": trusted}, wait=want == 0)
    if want == 0:
        assert p.stop() == 0
        return
    assert p.wait() == want and str(m) in p.err
    untouched(p)


def test_the_manifest_is_loaded_once_a_later_edit_or_delete_changes_no_decision(tmp_path):
    d = tmp_path / "b"
    d.mkdir(mode=0o700)
    m = manifest(d)
    p = spawn(tmp_path, ["--system", m, "--require-owner-protected"])
    try:
        c = BrokerClient(str(p.sock), timeout=10)
        req = {"component": "answer.interpret", "purpose": "model-call", "payloadClass": "repo-derived",
               "endpoint": "http://127.0.0.1:1234", "target": {"method": "POST", "path": "/v1/chat/completions"},
               "codebases": ["docs"]}
        g = c.open_run("answer", "shop", ["docs"])
        before = c.request(g.token, req, {"m": 1})[0]
        m.write_text(m.read_text().replace("docs", "elsewhere"))  # then deleted: nothing is re-read
        (d / "installed.yaml").unlink()
        tracked = CPG / "ccg" / "system.yaml"
        assert tracked.exists()  # the tracked manifest is never consulted (it is not named by `--system`)
        g2 = c.open_run("answer", "shop", ["docs"])
        after = c.request(g2.token, req, {"m": 1})[0]
    finally:
        assert p.stop() == 0
    assert (before.decision, before.rule, before.reason) == (after.decision, after.rule, after.reason)


# ------------------------------------------------------------------ item 6 through the real process (R5/R6)


def test_r5_and_r6_refuse_through_the_real_process_and_the_holder_socket_sees_no_connection(tmp_path):
    h = ScriptedHolder(tmp_path / "run", lambda payload: None)
    d = tmp_path / "b"
    d.mkdir(mode=0o700)
    ep = f"https://{HOST}:443"
    m = manifest(d, repos=(("front", "local-only"), ("docs", "hosted")),
                 eg=hosted_egress(ep, paths=["/v1/messages"]))
    p = spawn(tmp_path, hosted_args(tmp_path, d, m, holder_sock=h.path, holder_uid=ME), launcher_spec={"fake_uid": True})
    extras = {"marker": "unrestricted", "scope": "ccg:repo:front", "unrestricted": True, "gatewayRunId": "eval-1"}
    try:
        c = BrokerClient(str(p.sock), timeout=10)
        tgt = {"method": "POST", "path": "/v1/messages"}
        g = c.open_run("answer", "shop", ["docs"])  # touches only hosted codebases of a system with a local-only one
        d5, r5 = c.request(g.token, hosted_call(endpoint=ep, codebases=["docs"], target=tgt, **extras), {"m": 1})
        g = c.open_run("answer", "shop", ["front"])
        d5b, _ = c.request(g.token, hosted_call(endpoint=ep, codebases=["front"], target=tgt, **extras), {"m": 1})
        d6, r6 = c.request(g.token, {"component": "answer.pr-adapter", "purpose": "pr-addon", "payloadClass": "repo-derived",
                                     "endpoint": ep, "target": tgt, "codebases": ["front"], **extras}, {"m": 1})
    finally:
        assert p.stop() == 0
        h.close()
    assert (d5.rule, d5.reason, r5) == ("R5-SYSTEM-LOCAL-MODEL", "system-local-only-model", None)
    assert (d5b.rule, d6.rule, d6.reason, r6) == ("R5-SYSTEM-LOCAL-MODEL", "R6-STICKY-LOCAL", "sticky-local-only", None)
    assert [r["decision"] for r in p.records("decision")] == ["deny", "deny", "deny"] and p.records("close") == []
    assert h.connections == 0


# ------------------------------------------------------------------ item 12, a real broker process in front of a real holder


def make_cert(directory: Path) -> tuple[str, str]:
    cert, key = directory / "server.pem", directory / "server.key"
    subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", str(key), "-out", str(cert),
                    "-days", "2", "-subj", f"/CN={HOST}", "-addext", f"subjectAltName=DNS:{HOST},IP:127.0.0.1"],
                   check=True, capture_output=True)
    return str(cert), str(key)


class Stack:
    """A holder subprocess (provider `HOST` -> the stand-in on loopback), its credential file, and a broker subprocess."""

    def __init__(self, tmp_path: Path, cert, monkeypatch, *, base_port_default=False, holder_table=None,
                 start_holder=True, start_broker=True):
        monkeypatch.setattr(holder_harness, "LAUNCHER", HOLDER_LAUNCHER)
        self.tmp = tmp_path
        self.standin = StandIn(*cert)
        self.env = Env(tmp_path / "root", self.standin, cert[0])
        self.port = self.standin.port
        self.endpoint = f"https://{HOST}:443" if base_port_default else f"https://{HOST}:{self.port}"
        base_url = f"https://{HOST}" if base_port_default else self.endpoint
        self.env.write_config(providers={"anth": self.env.provider(base_url=base_url)},
                              holder=self.env.holder_table(**holder_table) if holder_table else None)
        self.credentials = tmp_path / "cred"
        self.credentials.mkdir(mode=0o700)
        self.cred = cred(self.credentials)
        self.holder = HolderProc(self.env) if start_holder else None
        self.broker = None
        if start_broker:
            self.start_broker()

    def start_broker(self):
        d = self.tmp / "b"
        d.mkdir(mode=0o700)
        m = manifest(d, repos=(("a", "hosted"),), eg=hosted_egress(self.endpoint, paths=["/v1/messages"]))
        self.broker = spawn(self.tmp, ["--system", m, "--require-owner-protected", "--key-holder-socket",
                                       self.env.socket_path, "--holder-uid", ME, "--holder-cred", self.cred],
                            launcher_spec={"fake_uid": True})
        self.client = BrokerClient(str(self.broker.sock), timeout=30)

    def call(self, *, run_id="eval-1", body=None, path="/v1/messages", endpoint=None):
        g = self.client.open_run("answer", "shop", ["a"])
        req = hosted_call(endpoint=endpoint or self.endpoint, codebases=["a"], target={"method": "POST", "path": path})
        if run_id is not None:
            req["gatewayRunId"] = run_id
        return self.client.request(g.token, req, body if body is not None else json.loads(body_for()))

    def close(self):
        if self.broker is not None and self.broker.p.poll() is None:
            self.broker.stop()
        if self.holder is not None:
            self.holder.stop()
        self.standin.close()


@pytest.fixture(scope="module")
def cert(tmp_path_factory):
    return make_cert(tmp_path_factory.mktemp("tls"))


@pytest.fixture
def stack(tmp_path, cert, monkeypatch):
    made = []

    def make(**kw):
        s = Stack(tmp_path if not made else tmp_path / f"s{len(made)}", cert, monkeypatch, **kw)
        made.append(s)
        return s

    (tmp_path / "s1").mkdir()
    (tmp_path / "s2").mkdir()
    yield make
    for s in made:
        s.close()


def refusal_code(resp) -> str:
    assert resp.status == 499 and resp.content_type == REFUSAL
    return json.loads(resp.body)["holderRefusal"]


def test_a_200_settles_through_the_real_broker_and_the_key_is_in_none_of_its_files_argv_or_environment(stack):
    s = stack()
    dec, resp = s.call()
    assert dec.rule == "R7-HOSTED-SERVICE" and resp.status == 200
    assert json.loads(resp.body)["usage"]["output_tokens"] == 7
    [seen] = s.standin.seen  # the holder reached the provider with the key the broker never held
    assert seen.headers["x-api-key"] == s.env.marker and seen.path == "/v1/messages"
    [close] = s.broker.records("close")
    assert (close["httpStatus"], close["closeReason"]) == (200, "completed") and close["bytesOut"] > 0
    assert s.broker.records("decision")[0]["resolvedAddr"] is None
    s.broker.stop()
    scan(s, tmp_roots=[s.tmp / "b", s.tmp / "b.stderr"], broker_pid=None)


def proc_text(pid: int) -> bytes:
    return b"\0".join(Path(f"/proc/{pid}/{n}").read_bytes() for n in ("cmdline", "environ"))


def scan(s: Stack, *, tmp_roots, broker_pid):
    """The fake key and the credential (hex and raw bytes) appear in no broker file, stderr, argv or environment."""
    needles = [s.env.marker.encode(), SECRET_HEX.encode(), CALLER_SECRET]
    files = [r for r in tmp_roots]
    dirs = [str(r) for r in files if Path(r).is_dir()]
    for r in files:
        if Path(r).is_file():
            for n in needles:
                assert n not in Path(r).read_bytes(), (r, n[:6])
    for n in needles:
        assert tree_search(dirs, n) == []
    if broker_pid:
        for n in needles:
            assert n not in proc_text(broker_pid)


def test_the_scan_has_a_positive_control_a_planted_marker_is_found_in_files_argv_and_environment(tmp_path):
    marker = "sk-ant-FAKE-control"
    (tmp_path / "audit").mkdir()
    (tmp_path / "audit" / "planted.txt").write_text(marker)
    assert tree_search([str(tmp_path)], marker.encode()) == [str(tmp_path / "audit" / "planted.txt")]
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)", marker], env={"X": marker})
    try:
        time.sleep(0.3)
        assert marker.encode() in proc_text(p.pid)
        assert proc_text(p.pid).count(marker.encode()) == 2  # once in cmdline, once in environ
    finally:
        p.kill()
        p.wait()


def test_the_credential_and_the_key_are_in_no_live_broker_process_text_after_calls(stack):
    s = stack()
    s.call()
    s.call(run_id=None)  # a refusal too
    pid = s.broker.p.pid
    text = proc_text(pid)
    assert s.env.marker.encode() not in text and SECRET_HEX.encode() not in text and CALLER_SECRET not in text
    cmd = Path(f"/proc/{pid}/cmdline").read_bytes()
    assert str(s.cred).encode() in cmd and b"--holder-cred" in cmd  # the path travels in argv, never the content
    s.broker.stop()
    assert SECRET_HEX not in s.broker.err and s.env.marker not in s.broker.err
    # positive control: the same needles ARE found where they are deliberately kept
    assert tree_search([str(s.credentials)], SECRET_HEX.encode()) == [str(s.cred)]
    assert tree_search([str(s.env.root / "state" / "keys")], s.env.marker.encode()) != []


REFUSALS = {
    "run_id_missing": dict(run_id=None),
    "run_unmatched": dict(run_id="adhoc-1"),
    "model_not_pinned": dict(body=json.loads(body_for(model="claude-sonnet-4-5-latest"))),
    "limit_missing": dict(body={"model": MODEL, "messages": []}),
    "limit_over_ceiling": dict(body=json.loads(body_for(5000))),
    "body_invalid": dict(body=json.loads(body_for(100, tools=[{"name": "x"}]))),
}


def test_each_refusal_reaches_the_caller_intact_through_the_real_broker_and_nothing_reaches_the_provider(stack):
    s = stack()
    for code, kw in sorted(REFUSALS.items()):
        dec, resp = s.call(**kw)
        assert dec.allowed and refusal_code(resp) == code, code
    closes = s.broker.records("close")
    assert [(c["httpStatus"], c["closeReason"]) for c in closes] == [(499, "completed")] * len(REFUSALS)
    assert len(s.broker.records("decision")) == len(REFUSALS) and s.standin.seen == []
    for dec in s.broker.records("decision"):
        assert dec["decision"] == "allow" and dec["rule"] == "R7-HOSTED-SERVICE" and dec["resolvedAddr"] is None


def test_an_endpoint_the_holder_does_not_know_and_a_path_it_does_not_allow_are_refused_by_it(stack, monkeypatch):
    s = stack(start_broker=False)
    other = f"https://other.{HOST}:{s.port}"
    d = s.tmp / "b"
    d.mkdir(mode=0o700)
    eg = hosted_egress(s.endpoint, paths=["/v1/messages", "/v1/complete"])
    eg["hosted"].append({"id": "other", "kind": "model", "endpoint": other, "paths": ["/v1/messages"]})
    m = manifest(d, repos=(("a", "hosted"),), eg=eg)
    s.broker = spawn(s.tmp, ["--system", m, "--require-owner-protected", "--key-holder-socket", s.env.socket_path,
                             "--holder-uid", ME, "--holder-cred", s.cred], launcher_spec={"fake_uid": True})
    s.client = BrokerClient(str(s.broker.sock), timeout=30)
    assert refusal_code(s.call(endpoint=other)[1]) == "endpoint_unknown"
    assert refusal_code(s.call(path="/v1/complete")[1]) == "path_not_allowed"
    assert s.standin.seen == []


def test_a_hosted_entry_at_the_default_port_is_keyed_by_the_canonical_origin_and_ends_in_a_signed_upstream_error(stack):
    s = stack(base_port_default=True)  # manifest `https://HOST:443`, holder `base_url` written without the port
    dec, resp = s.call()
    assert dec.allowed and resp is None  # not a 499 `endpoint_unknown`: the holder found the provider
    [close] = s.broker.records("close")
    assert close["closeReason"] == "upstream-error" and close["bytesOut"] > 0 and close["bytesIn"] > 0
    assert close["httpStatus"] is None and s.standin.seen == []  # nothing listens on 443: the holder's connect was refused


def test_an_absent_run_id_is_refused_under_refuse_and_served_in_the_shared_bucket_under_cap(stack, tmp_path):
    s = stack()
    assert refusal_code(s.call(run_id=None)[1]) == "run_id_missing"
    s2 = stack(holder_table={"unmatched": "cap", "default_bound": 5000})
    dec, resp = s2.call(run_id=None)
    assert dec.allowed and resp.status == 200 and len(s2.standin.seen) == 1


def test_only_a_stopped_holder_gives_an_upstream_error_and_a_restarted_one_serves_under_the_same_credential(stack, monkeypatch):
    s = stack()
    assert s.call()[1].status == 200
    s.holder.stop()
    dec, resp = s.call()
    assert dec.allowed and resp is None
    assert [c["closeReason"] for c in s.broker.records("close")] == ["completed", "upstream-error"]
    s.holder = HolderProc(s.env)  # the holder restarts; the broker holds nothing else to refresh
    assert s.call()[1].status == 200


def test_the_credential_is_read_once_deleting_the_file_after_start_changes_nothing(stack):
    s = stack()
    assert s.call()[1].status == 200
    s.cred.unlink()
    assert s.call()[1].status == 200  # a per-call re-read would have been `unreadable`
    assert len(s.standin.seen) == 2


def test_a_wrong_credential_is_the_holders_unsigned_bad_credential_refusal_as_a_completed_499(stack, tmp_path):
    s = stack(start_broker=False)
    s.cred.write_text("ff" * 32 + "\n")
    s.start_broker()
    dec, resp = s.call()
    assert dec.allowed and refusal_code(resp) == "bad_credential" and s.standin.seen == []
    s.broker.stop()
    assert ("ff" * 32) not in s.broker.err

