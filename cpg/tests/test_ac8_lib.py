"""Docker-free tests of `cpg/sandbox/ac8/lib.sh` (P1-31): the bridge-address guard must read "no answer" as "unknown" (analyst F1 / security
M2), and the signal traps must clean up AND exit (F6 / M3); plus the argument checks of `run.sh` that fire before Docker is touched."""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from ac8_helpers import AC8

LIB = AC8 / "lib.sh"
RUN = AC8 / "run.sh"
PY = sys.executable


def guard(dev: str, ip_bin: str | None = None) -> int:
    env = {"PATH": "/usr/bin:/bin", "AC8_PY": PY}
    if ip_bin is not None:
        env["AC8_IP_BIN"] = ip_bin
    return subprocess.run(["/bin/bash", "-c", f'. "{LIB}"; bridge_has_address "$1"', "x", dev], env=env, capture_output=True).returncode


def fake_ip(tmp_path: Path, body: str, name: str = "ip") -> str:
    p = tmp_path / name
    p.write_text("#!/bin/sh\n" + body + "\n")
    p.chmod(0o755)
    return str(p)


def test_loopback_has_an_address():
    assert guard("lo") == 0


def test_a_device_that_does_not_exist_is_unknown_not_no_address():
    assert guard("br-doesnotexist00") == 2


def test_an_empty_device_name_is_unknown():
    assert guard("") == 2


@pytest.mark.parametrize("body", [
    "echo 'garbage, not json'",                                  # ip printed non-JSON
    "echo '[]'",                                                 # valid JSON, no interface at all
    "echo '{}'",                                                 # valid JSON, not a list
    "echo '[{\"ifname\":\"x\"}]'",                               # an interface without addr_info
    "echo '[{\"addr_info\":\"none\"}]'",                         # addr_info not a list
    "echo '[1,2]'",                                              # list of non-objects
    "echo '[{\"addr_info\":[]}]'; exit 1",                       # ip failed (exit 1) although it printed a perfect answer
    "exit 0",                                                    # ip succeeded and printed nothing
])
def test_garbage_or_failed_ip_output_is_unknown(tmp_path, body):
    assert guard("br-x", fake_ip(tmp_path, body)) == 2


def test_a_missing_ip_binary_is_unknown(tmp_path):
    assert guard("br-x", str(tmp_path / "no-such-ip")) == 2


def test_an_existing_device_with_an_empty_address_list_is_the_only_no_address(tmp_path):
    assert guard("br-x", fake_ip(tmp_path, "echo '[{\"ifname\":\"br-x\",\"addr_info\":[]}]'")) == 1


def test_an_existing_device_with_an_address_is_has_address(tmp_path):
    assert guard("br-x", fake_ip(tmp_path, "echo '[{\"ifname\":\"br-x\",\"addr_info\":[{\"local\":\"10.0.0.1\"}]}]'")) == 0


# ------------------------------------------------------------------ the traps


def trap_script(tmp_path: Path, tail: str, *, flag: str = "") -> tuple[Path, Path]:
    log = tmp_path / "log"
    script = tmp_path / "s.sh"
    script.write_text(f'''#!/bin/bash
set -uo pipefail
. "{LIB}"
cleanup() {{ echo cleanup >> "{log}"; sleep 0.3; echo cleanup-done >> "{log}"; }}
ac8_install_traps cleanup
echo ready > "{tmp_path}/ready"
{tail}
echo after-signal >> "{log}"
''')
    return script, log


def run_and_signal(tmp_path, sig, tail="sleep 30 & wait $!", extra=None):
    script, log = trap_script(tmp_path, tail)
    p = subprocess.Popen(["/bin/bash", str(script)], env={"PATH": "/usr/bin:/bin"})
    for _ in range(100):
        if (tmp_path / "ready").exists():
            break
        time.sleep(0.05)
    time.sleep(0.2)
    p.send_signal(sig)
    for s2 in (extra or ()):
        time.sleep(0.1)
        p.send_signal(s2)
    rc = p.wait(timeout=20)
    return rc, log.read_text().split() if log.exists() else []


@pytest.mark.parametrize("sig,rc", [(signal.SIGTERM, 143), (signal.SIGINT, 130), (signal.SIGHUP, 129)])
def test_a_signal_cleans_up_once_and_exits_nonzero_without_resuming(tmp_path, sig, rc):
    got_rc, log = run_and_signal(tmp_path, sig)
    assert got_rc == rc and log == ["cleanup", "cleanup-done"]     # cleaned up, exited, and the "after" line never ran


def test_a_second_signal_during_the_cleanup_does_not_cut_it_short_or_run_it_twice(tmp_path):
    got_rc, log = run_and_signal(tmp_path, signal.SIGTERM, extra=[signal.SIGTERM, signal.SIGINT])
    assert got_rc == 143 and log == ["cleanup", "cleanup-done"]


def test_a_normal_exit_runs_the_cleanup_exactly_once_and_keeps_its_exit_code(tmp_path):
    script, log = trap_script(tmp_path, "exit 7")
    assert subprocess.run(["/bin/bash", str(script)], env={"PATH": "/usr/bin:/bin"}).returncode == 7
    assert log.read_text().split() == ["cleanup", "cleanup-done"]


def test_a_die_path_exit_runs_the_cleanup_once(tmp_path):
    script, log = trap_script(tmp_path, "exit 70")
    assert subprocess.run(["/bin/bash", str(script)], env={"PATH": "/usr/bin:/bin"}).returncode == 70
    assert log.read_text().split() == ["cleanup", "cleanup-done"]


def test_the_old_pattern_cleanup_without_exit_resumes_the_script(tmp_path):
    """The bug F6/M3 describe, reproduced: a signal trap that only cleans up lets bash carry on with the interrupted flow."""
    log = tmp_path / "log"
    script = tmp_path / "old.sh"
    script.write_text(f'''#!/bin/bash
cleanup() {{ trap - EXIT INT TERM HUP; echo cleanup >> "{log}"; }}
trap cleanup EXIT INT TERM HUP
kill -TERM $$
echo resumed >> "{log}"
''')
    rc = subprocess.run(["/bin/bash", str(script)], env={"PATH": "/usr/bin:/bin"}).returncode
    assert log.read_text().split() == ["cleanup", "resumed"] and rc == 0


def isolated_or_die(dev: str, ip_bin: str | None) -> tuple[int, str]:
    env = {"PATH": "/usr/bin:/bin", "AC8_PY": PY}
    if ip_bin is not None:
        env["AC8_IP_BIN"] = ip_bin
    cp = subprocess.run(["/bin/bash", "-c", f'. "{LIB}"; die() {{ echo "DIED: $1"; exit 70; }}; bridge_isolated_or_die "$1"; echo RETURNED', "x", dev],
                        env=env, capture_output=True, text=True)
    return cp.returncode, cp.stdout


def test_the_isolation_guard_returns_only_for_an_existing_bridge_without_an_address(tmp_path):
    rc, out = isolated_or_die("br-x", fake_ip(tmp_path, "echo '[{\"ifname\":\"br-x\",\"addr_info\":[]}]'"))
    assert (rc, out) == (0, "RETURNED\n")


def test_the_isolation_guard_dies_70_on_a_bridge_with_an_address(tmp_path):
    rc, out = isolated_or_die("br-x", fake_ip(tmp_path, "echo '[{\"addr_info\":[{\"local\":\"10.0.0.1\"}]}]'"))
    assert rc == 70 and "carries an address" in out and "RETURNED" not in out


@pytest.mark.parametrize("dev,body", [("br-doesnotexist00", None), ("br-x", "echo garbage"), ("br-x", "exit 1"), ("br-x", "echo '[]'")])
def test_the_isolation_guard_dies_70_when_it_cannot_tell(tmp_path, dev, body):
    rc, out = isolated_or_die(dev, fake_ip(tmp_path, body) if body else None)
    assert rc == 70 and "cannot establish" in out and "RETURNED" not in out


def test_run_sh_installs_the_shared_traps_not_its_own():
    text = RUN.read_text()
    assert "ac8_install_traps cleanup" in text and "trap cleanup EXIT" not in text and "trap - EXIT INT TERM HUP" not in text
    assert 'bridge_isolated_or_die "$bridge"' in text and "ip -j address show dev" not in text


# ------------------------------------------------------------------ run.sh arguments that are refused before Docker is touched


def run_sh(*args):
    return subprocess.run(["/bin/bash", str(RUN), *args], env={"PATH": "/usr/bin:/bin"}, capture_output=True, text=True, timeout=60)


@pytest.mark.parametrize("comp", ["*", "../x", "a b", "a,,b", "A", "a,", "-x", "a;b"])
def test_components_are_validated_like_the_id(comp):
    r = run_sh("run", "--id", "t1", "--components", comp)
    assert r.returncode == 64 and "--components must match" in r.stderr


@pytest.mark.parametrize("opt", ["--store-container", "--store-port"])
def test_a_callers_container_cannot_be_attached_as_the_store(opt):
    r = run_sh("run", opt, "falkordb-dev")
    assert r.returncode == 64 and "not supported" in r.stderr


def test_the_usage_states_the_measured_scope_and_that_the_workload_is_unconfined():
    out = run_sh("--help").stdout
    assert "does NOT show that no component attempted any other" in out and "UNCONFINED" in out
    assert "No traffic leaves the host" not in out and "never `pass`" in out


def test_the_broker_strace_covers_the_whole_network_set_and_the_credentials_backstop_is_set():
    text = RUN.read_text()
    assert text.count("trace=%network") == 2 and "trace=connect" not in text       # the broker's and the positive control's
    assert "CCG_FORBID_REAL_CREDENTIALS=1" in text
    assert "token=" not in text.split("probe_components()")[1].split("main_run()")[0].replace("tokens=%s", "")   # no nonce on a command line


def test_facts_py_does_not_reset_an_unparsable_facts_file(tmp_path):
    f = tmp_path / "facts.json"
    f.write_text("{not json")
    cp = subprocess.run([PY, "-I", str(AC8 / "facts.py"), "set", str(f), "k", "1"], capture_output=True, text=True)
    assert cp.returncode == 70 and f.read_text() == "{not json"
    g = tmp_path / "new.json"          # a file that does not exist yet is created
    assert subprocess.run([PY, "-I", str(AC8 / "facts.py"), "set", str(g), "k", "1"]).returncode == 0 and '"k": 1' in g.read_text()


def test_an_argv_element_with_a_newline_is_never_audited_so_it_cannot_forge_the_helpers_protocol():
    import assertions as A
    assert A._run_argv_audit("pysrc2cpg", ["/src", "x\nDONE"]) == (None, [], False)
    assert A._run_argv_audit("pysrc2cpg", ["/src"])[2] is True


def test_the_cleanup_is_idempotent_when_invoked_twice(tmp_path):
    script, log = trap_script(tmp_path, "ac8_cleanup_once; ac8_cleanup_once; exit 0")
    assert subprocess.run(["/bin/bash", str(script)], env={"PATH": "/usr/bin:/bin"}).returncode == 0
    assert log.read_text().split() == ["cleanup", "cleanup-done"]


# ------------------------------------------------------------------ run.sh pins that need no Docker (security N1, N3, N4)

DETECTOR = AC8 / "detector.pl"


def _stop_detector(tmp_path: Path, docker_body: str) -> subprocess.CompletedProcess:
    """Runs `stop_detector` as extracted from run.sh, with a fake `docker` first on PATH (it logs its argv to $LOG)."""
    import re
    src = RUN.read_text()
    fn = re.search(r"^stop_detector\(\) \{.*?^\}", src, re.S | re.M).group(0)
    fake = tmp_path / "bin"
    fake.mkdir()
    d = fake / "docker"
    d.write_text('#!/bin/sh\necho "$@" >> "$LOG"\n' + docker_body + "\n")
    d.chmod(0o755)
    env = {"PATH": f"{fake}:/usr/bin:/bin", "LOG": str(tmp_path / "log")}
    cmd = f'die() {{ echo "DIE: $1" >&2; exit 1; }}\n{fn}\nstop_detector det1'
    return subprocess.run(["/bin/bash", "-c", cmd], env=env, capture_output=True, text=True)


FAKE_INSPECT = 'case "$*" in *State.Running*) echo false;; *State.ExitCode*) echo "$EXIT";; esac'


def test_stop_detector_waits_longer_than_the_detectors_drain_bound(tmp_path):
    import re
    drain = int(re.search(r"AC8_DET_DRAIN_MAX\} // (\d+)", DETECTOR.read_text()).group(1))
    r = _stop_detector(tmp_path, FAKE_INSPECT.replace("$EXIT", "0"))
    assert r.returncode == 0, r.stderr
    stop = next(ln for ln in (tmp_path / "log").read_text().splitlines() if ln.startswith("stop"))
    assert int(re.search(r"-t (\d+)", stop).group(1)) > drain


@pytest.mark.parametrize("code", ["137", "1", ""])
def test_stop_detector_dies_unless_the_detector_exited_zero(tmp_path, code):
    r = _stop_detector(tmp_path, FAKE_INSPECT.replace("$EXIT", code))
    assert r.returncode == 1 and "did not exit 0" in r.stderr


def test_stop_detector_dies_if_the_detector_is_still_running(tmp_path):
    r = _stop_detector(tmp_path, 'case "$*" in *State.Running*) echo true;; esac')
    assert r.returncode == 1 and "did not stop" in r.stderr


def _line(text: str, needle: str) -> str:
    hits = [ln for ln in text.splitlines() if needle in ln and not ln.lstrip().startswith("#")]
    assert len(hits) == 1, (needle, hits)
    return hits[0]


def test_the_row_8_16_control_is_traced_with_the_same_strace_flags_as_the_broker():
    import re
    src = RUN.read_text()
    broker = re.search(r"sarg=\(strace (.*?) -o ", _line(src, "sarg=(strace")).group(1)
    control = re.search(r"^\s*strace (.*?) -o \"\$SCEN/trace/control\.strace\"", src, re.M).group(1)
    assert "--seccomp-bpf" in broker and control == broker


@pytest.mark.parametrize("needle", ['assertions.py" "${pe[@]}"', 'controls.py" "${cargs[@]}"'])
def test_p_e_and_the_controls_run_with_real_credentials_forbidden(needle):
    assert "CCG_FORBID_REAL_CREDENTIALS=1" in _line(RUN.read_text(), needle)
