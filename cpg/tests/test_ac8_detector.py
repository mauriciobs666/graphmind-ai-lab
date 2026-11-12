"""Docker-free tests of the AC-8 detector (`cpg/sandbox/ac8/detector.pl`, P1-31) on its OWN throwaway loopback ports: the process is
spawned by the test, listens on 127.0.0.1 (AC8_DET_ADDR / AC8_DET_PORTS test hooks) and is connected to only by the test; nothing here
touches a host address or port other than those. Covers the flush drain (M1), single-use per-probe nonces and their routing (M4)."""
from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path

import pytest

from ac8_helpers import AC8, W

pytestmark = pytest.mark.skipif(shutil.which("perl") is None, reason="perl is needed to run detector.pl on the host")
N1, N2 = "a" * 32, "b" * 32


def free_ports(n: int) -> list[int]:
    socks = [socket.socket() for _ in range(n)]
    for s in socks:
        s.bind(("127.0.0.1", 0))
    ports = [s.getsockname()[1] for s in socks]
    for s in socks:
        s.close()
    return ports


class Det:
    def __init__(self, tmp: Path, tokens: str = "", extra_env: dict | None = None):
        self.dir = tmp / "det"
        self.dir.mkdir()
        if tokens:
            (self.dir / "probe-tokens").write_text(tokens)
        self.ports = free_ports(3)
        env = {"PATH": "/usr/bin:/bin", "AC8_DET_ADDR": "127.0.0.1", "AC8_DET_PORTS": ",".join(map(str, self.ports)), **(extra_env or {})}
        self.proc = subprocess.Popen(["/usr/bin/perl", str(AC8 / "detector.pl"), str(self.dir)], env=env, stderr=subprocess.PIPE)
        for _ in range(100):
            if (self.dir / "ready").exists():
                break
            time.sleep(0.05)
        else:
            raise AssertionError("detector did not come up: " + self.proc.stderr.read().decode())

    def connect(self, data: bytes = b"", port: int | None = None, hold: bool = False) -> socket.socket:
        s = socket.create_connection(("127.0.0.1", port or self.ports[0]), timeout=3)
        if data:
            s.sendall(data)
        if not hold:
            s.close()
        return s

    def signal(self, sig, marker: str | None = None, wait: float = 20):
        self.proc.send_signal(sig)
        if marker:
            end = time.time() + wait
            while time.time() < end and not (self.dir / marker).exists():
                time.sleep(0.05)
            assert (self.dir / marker).exists(), f"{marker} was not written"

    def records(self, name: str) -> list[dict]:
        p = self.dir / name
        return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []

    def close(self):
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.wait()
        self.proc.stderr.close()


@pytest.fixture
def mkdet(tmp_path):
    made: list[Det] = []

    def make(tokens: str = "", **env) -> Det:
        assert not made, "one detector per test"
        d = Det(tmp_path, tokens, env)
        made.append(d)
        return d
    yield make
    for d in made:
        d.close()


def test_nonce_routing_tokenless_wrong_correct_and_replay(mkdet):
    d = mkdet(f"{N1} answer/start\n{N2} front/end\n")
    d.connect(b"hello\n")                                   # token-less
    d.connect(b"AC8PROBE:" + b"c" * 32 + b"\n")             # wrong nonce
    d.connect(b"AC8PROBE:" + N1.encode() + b"\n")           # correct, first use
    d.connect(b"AC8PROBE:" + N1.encode() + b"\n")           # replay of a used nonce
    d.connect(b"xx AC8PROBE:" + N2.encode() + b"\n")        # the nonce must be the first bytes
    d.signal(signal.SIGUSR2, "flush.marker")
    assert len(d.records("conn.jsonl")) == 4
    probes = d.records("probe.jsonl")
    assert [p["probe"] for p in probes] == ["answer/start"]
    assert set(probes[0]) == set(W.PROBE_FIELDS)


def test_no_token_file_every_connection_is_a_conn(mkdet):
    d = mkdet()
    d.connect(b"AC8PROBE:" + N1.encode() + b"\n")
    d.signal(signal.SIGUSR2, "flush.marker")
    assert len(d.records("conn.jsonl")) == 1 and d.records("probe.jsonl") == []


def test_markers_are_written_once_and_flush_reports_residue_zero(mkdet):
    d = mkdet()
    d.signal(signal.SIGUSR1, "start.marker")
    first = (d.dir / "start.marker").read_text()
    time.sleep(0.2)
    d.signal(signal.SIGUSR1)
    time.sleep(0.3)
    assert (d.dir / "start.marker").read_text() == first
    d.signal(signal.SIGUSR2, "flush.marker")
    assert json.loads((d.dir / "flush.marker").read_text())["residue"] == 0
    m, errs = W.read_markers(d.dir)
    assert not errs and m["start"] and m["flush"] and m["residue"] == 0


@pytest.mark.parametrize("flood", [40, 200])
def test_a_flood_across_the_flush_is_fully_logged_before_the_marker_and_fails_the_verdict(mkdet, flood):
    """M1: connections queued at the flush are all drained and logged first (residue 0), so the verdict sees them and FAILS; none is
    lost, however many (the drain is bounded by wall-clock, not by a count)."""
    d = mkdet()
    d.signal(signal.SIGUSR1, "start.marker")
    time.sleep(0.2)
    held = [d.connect(hold=True, port=d.ports[i % 3]) for i in range(flood)]
    d.signal(signal.SIGUSR2, "flush.marker", wait=40)
    try:
        assert json.loads((d.dir / "flush.marker").read_text())["residue"] == 0
        conns = d.records("conn.jsonl")
        assert len(conns) == flood
        v = W.verdict_for_dir(d.dir)
        assert not v.ok and sum(f.startswith("conn-in-window") for f in v.failures) == flood
        assert not any("residue" in f for f in v.failures)
    finally:
        for s in held:
            s.close()


def test_a_connection_after_the_flush_is_still_logged_so_the_verdict_fails_it(mkdet):
    d = mkdet()
    d.signal(signal.SIGUSR1, "start.marker")
    d.signal(signal.SIGUSR2, "flush.marker")
    time.sleep(0.1)
    d.connect(b"late\n")
    d.signal(signal.SIGTERM)
    assert d.proc.wait(timeout=20) == 0
    v = W.verdict_for_dir(d.dir)
    assert not v.ok and any(f.startswith("conn-outside-window:after-flush") for f in v.failures)


def test_sigterm_drains_then_exits_zero(mkdet):
    d = mkdet()
    held = [d.connect(hold=True) for _ in range(30)]
    d.signal(signal.SIGTERM)
    assert d.proc.wait(timeout=20) == 0
    assert len(d.records("conn.jsonl")) == 30
    final = json.loads((d.dir / "final.marker").read_text())
    assert set(final) == {"marker", "ts", "residue"} and final["marker"] == "final" and final["residue"] == 0
    for s in held:
        s.close()


def test_the_termination_drain_records_its_residue_in_a_final_marker_and_the_verdict_fails_on_it(mkdet):
    d = mkdet(AC8_DET_HOLD="5", AC8_DET_DRAIN_MAX="0.5")
    d.signal(signal.SIGUSR1, "start.marker")
    d.signal(signal.SIGUSR2, "flush.marker")
    held = [d.connect(b"x", hold=True) for _ in range(4)]
    time.sleep(0.2)
    d.proc.send_signal(signal.SIGTERM)
    try:
        assert d.proc.wait(timeout=20) == 0
        assert json.loads((d.dir / "final.marker").read_text())["residue"] == 4
        v = W.verdict_for_dir(d.dir)
        assert not v.ok and "final-residue: 4 connection(s) were still unlogged when the termination drain stopped" in v.failures
    finally:
        for s in held:
            s.close()


def test_a_final_marker_is_written_once_and_after_the_flush_marker(mkdet):
    d = mkdet()
    d.signal(signal.SIGUSR1, "start.marker")
    d.signal(signal.SIGUSR2, "flush.marker")
    assert not (d.dir / "final.marker").exists()
    d.proc.send_signal(signal.SIGTERM)
    assert d.proc.wait(timeout=20) == 0
    assert W.verdict_for_dir(d.dir).failures == []
    m, errs = W.read_markers(d.dir)
    assert errs == [] and m["flush"] <= m["final"] and m["final_residue"] == 0


def test_a_drain_that_runs_out_of_wall_clock_reports_its_residue_and_the_verdict_fails_on_it(mkdet):
    """The residue is the detector's own count of what it could not log in time (here: connections held longer than the drain's bound)."""
    d = mkdet(AC8_DET_HOLD="5", AC8_DET_DRAIN_MAX="0.5")
    d.signal(signal.SIGUSR1, "start.marker")
    time.sleep(0.2)
    held = [d.connect(b"x", hold=True) for _ in range(6)]
    time.sleep(0.2)
    d.signal(signal.SIGUSR2, "flush.marker")
    try:
        residue = json.loads((d.dir / "flush.marker").read_text())["residue"]
        assert residue == 6 and d.records("conn.jsonl") == []
        v = W.verdict_for_dir(d.dir)
        assert not v.ok and f"flush-residue: 6 connection(s) were still unlogged when the detector's drain stopped" in v.failures
    finally:
        for s in held:
            s.close()


def test_the_bytes_a_peer_sent_are_counted_and_the_detector_never_answers(mkdet):
    d = mkdet()
    s = d.connect(b"CONNECT example.com:443 HTTP/1.1\r\n\r\n", port=d.ports[2], hold=True)
    s.settimeout(1)
    d.signal(signal.SIGUSR2, "flush.marker")
    try:
        rec = d.records("conn.jsonl")[0]
        assert rec["bytesIn"] == len(b"CONNECT example.com:443 HTTP/1.1\r\n\r\n") and rec["bytesOut"] == 0 and rec["dstPort"] == d.ports[2]
        assert s.recv(10) == b""            # closed without forwarding or answering
    finally:
        s.close()
