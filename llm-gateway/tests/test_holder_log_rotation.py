"""Size-based rotation of the holder's `calls.log` (security review Pass 7 minor; plan §3.4).

Real files in a temp state directory: one previous file is kept, the new file keeps the old one's
mode (and owner), no line is lost or split, and a rotation that cannot happen does not drop the
call."""

import json
import os
import stat

import pytest
from holder_harness import Clock, Env, StandIn, body_for, call, make_cert
from test_holder_core import FakeUpstream

LINE_BOUND = 400  # calls.log lines here are ~150 bytes


@pytest.fixture(scope="session")
def cert(tmp_path_factory):
    return make_cert(tmp_path_factory.mktemp("tls"))


@pytest.fixture
def env(tmp_path, cert):
    standin = StandIn(*cert)
    e = Env(tmp_path / "root", standin, cert[0])
    yield e
    standin.close()


def lines(path):
    with open(path) as f:
        return [json.loads(x) for x in f.read().splitlines()]


def drive(h, n, start=0):
    for i in range(start, start + n):
        assert call(h, run_id=f"eval-{i}", body=body_for(10)).status == 200


def test_the_log_rotates_at_the_size_keeping_one_previous_file(env):
    h = env.holder(clock=Clock(), upstream=FakeUpstream(), log_max_bytes=LINE_BOUND)
    drive(h, 12)
    h.close()
    cur, prev = f"{env.state_root}/calls.log", f"{env.state_root}/calls.log.1"
    assert not os.path.exists(cur + ".2")  # exactly one previous file
    assert 0 < os.path.getsize(cur) <= LINE_BOUND
    assert 0 < os.path.getsize(prev) <= LINE_BOUND
    run_ids = [r["run_id"] for r in lines(prev) + lines(cur)]
    assert run_ids == [f"eval-{i}" for i in range(12 - len(run_ids), 12)]  # contiguous tail
    assert len(run_ids) < 12  # older generations were dropped, not kept without bound


def test_nothing_is_lost_while_under_the_size(env):
    h = env.holder(clock=Clock(), upstream=FakeUpstream(), log_max_bytes=10_000)
    drive(h, 5)
    h.close()
    assert not os.path.exists(f"{env.state_root}/calls.log.1")
    assert [r["run_id"] for r in lines(f"{env.state_root}/calls.log")] == [
        f"eval-{i}" for i in range(5)
    ]


def test_the_rotated_file_keeps_the_mode_and_owner(env):
    h = env.holder(clock=Clock(), upstream=FakeUpstream(), log_max_bytes=LINE_BOUND)
    cur = f"{env.state_root}/calls.log"
    os.chmod(cur, 0o640)  # a deliberately different mode must carry over, not reset to 0600
    drive(h, 8)
    h.close()
    assert os.path.exists(cur + ".1")
    for path in (cur, cur + ".1"):
        st = os.stat(path)
        assert stat.S_IMODE(st.st_mode) == 0o640
        assert (st.st_uid, st.st_gid) == (os.getuid(), os.getgid())


def test_default_rotated_files_are_private(env):
    h = env.holder(clock=Clock(), upstream=FakeUpstream(), log_max_bytes=LINE_BOUND)
    drive(h, 8)
    h.close()
    for name in ("calls.log", "calls.log.1"):
        assert stat.S_IMODE(os.stat(f"{env.state_root}/{name}").st_mode) == 0o600


def test_a_failed_rotation_keeps_logging_to_the_current_file(env, monkeypatch, capfd):
    h = env.holder(clock=Clock(), upstream=FakeUpstream(), log_max_bytes=LINE_BOUND)

    def no(*a, **k):
        raise PermissionError("denied detail")

    monkeypatch.setattr(os, "replace", no)
    drive(h, 8)
    h.close()
    assert len(lines(f"{env.state_root}/calls.log")) == 8  # no call lost
    out, err = capfd.readouterr()
    assert "log rotation failed PermissionError" in err and "denied detail" not in err


def test_a_file_exactly_at_the_limit_is_not_rotated_until_the_next_line_would_exceed_it(env):
    probe = env.holder(clock=Clock(), upstream=FakeUpstream(), log_max_bytes=10_000)
    drive(probe, 1)
    probe.close()
    line = os.path.getsize(f"{env.state_root}/calls.log")  # run ids of one width: equal lines
    os.unlink(f"{env.state_root}/calls.log")
    h = env.holder(clock=Clock(), upstream=FakeUpstream(), log_max_bytes=2 * line)
    cur = f"{env.state_root}/calls.log"
    drive(h, 2)
    assert os.path.getsize(cur) == 2 * line and not os.path.exists(cur + ".1")
    drive(h, 1, start=2)
    h.close()
    assert [r["run_id"] for r in lines(cur + ".1")] == ["eval-0", "eval-1"]
    assert [r["run_id"] for r in lines(cur)] == ["eval-2"]


def test_the_new_file_is_chowned_to_the_old_files_owner_when_it_differs(env, monkeypatch):
    h = env.holder(clock=Clock(), upstream=FakeUpstream(), log_max_bytes=LINE_BOUND)
    old_fd, real_fstat, chowned = h._log_fd, os.fstat, []

    def fstat(fd):
        st = real_fstat(fd)
        if fd != old_fd or chowned:  # only the old log, until it has been rotated away
            return st
        fields = list(st)[:10]
        fields[4] += 1  # the old file reports another owner
        return os.stat_result(fields)

    monkeypatch.setattr(os, "fstat", fstat)
    monkeypatch.setattr(os, "fchown", lambda fd, uid, gid: chowned.append((uid, gid)))
    drive(h, 8)
    monkeypatch.undo()
    h.close()
    assert chowned and chowned[0] == (os.getuid() + 1, os.getgid())
