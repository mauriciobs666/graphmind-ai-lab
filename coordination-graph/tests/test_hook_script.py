"""The hook script `hooks/post-tool-sync.sh` (plan section 3.2, section 5 "Hook") against a recording fake of the
`coordgraph` entry point (`COORDGRAPH_HOOK_CLI`): what the shell decides, with no database and no Python of ours.

Every timing is measured to pipe EOF of `subprocess.run(capture_output=True)`, not just process exit (P2-M1).
The real CLI behind the script is `test_hook_live.py`."""
from __future__ import annotations

import fcntl
import os
import re
import subprocess
import threading
import time
from pathlib import Path

import pytest

HOOK = Path(__file__).resolve().parents[1] / "hooks" / "post-tool-sync.sh"
L = "a/docs/plans/x-coordination.md"
FAKE = """#!/bin/bash
# records every call; behaviour per subcommand from $FAKE_DIR/{rc,err,sleep}-<sub>
sub="$1"
echo "$sub $* | shared=${COORDGRAPH_SHARED_CONFIRM-unset} root=$COORDGRAPH_REPO_ROOT state=$COORDGRAPH_STATE_DIR" >>"$FAKE_DIR/calls.log"
[ -f "$FAKE_DIR/sleep-$sub" ] && sleep "$(cat "$FAKE_DIR/sleep-$sub")"
[ -f "$FAKE_DIR/err-$sub" ] && cat "$FAKE_DIR/err-$sub" >&2
echo "stdout noise"
[ -f "$FAKE_DIR/rc-$sub" ] && exit "$(cat "$FAKE_DIR/rc-$sub")"
exit 0
"""


class Hook:
    def __init__(self, tmp_path):
        self.root = tmp_path / "repo"
        self.state = tmp_path / "state"
        self.fake_dir = tmp_path / "fake"
        self.cli = tmp_path / "fakecli.sh"
        for d in (self.root, self.state, self.fake_dir):
            d.mkdir()
        self.cli.write_text(FAKE)
        self.cli.chmod(0o755)
        self.stamp = self.state / "stamp"
        self.extra: dict[str, str] = {}  # live tests: the ephemeral endpoints

    def put(self, rel: str, text: str = "x\n", mtime: float | None = None) -> Path:
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        if mtime is not None:
            os.utime(p, (mtime, mtime))
        return p

    def set_stamp(self, mtime: float) -> None:
        self.stamp.touch()
        os.utime(self.stamp, (mtime, mtime))

    def script(self, sub: str, *, rc=None, err=None, sleep=None) -> None:
        for kind, val in (("rc", rc), ("err", err), ("sleep", sleep)):
            if val is not None:
                (self.fake_dir / f"{kind}-{sub}").write_text(str(val))

    def run(self, payload: str = '{"tool_name":"Bash"}', **env):
        base = {"PATH": os.environ["PATH"], "HOME": os.environ.get("HOME", "/"),
                "CLAUDE_PROJECT_DIR": str(self.root), "COORDGRAPH_STATE_DIR": str(self.state),
                "COORDGRAPH_HOOK_CLI": str(self.cli), "FAKE_DIR": str(self.fake_dir)}
        base.update(self.extra)
        base.update({k: v for k, v in env.items() if v is not None})
        t = time.monotonic()
        r = subprocess.run([str(HOOK)], input=payload, capture_output=True, text=True, env=base, timeout=30)
        r.elapsed = time.monotonic() - t
        return r

    def calls(self) -> list[str]:
        f = self.fake_dir / "calls.log"
        return f.read_text().splitlines() if f.exists() else []

    def wait_calls(self, n: int, timeout=5.0) -> list[str]:
        end = time.time() + timeout
        while time.time() < end and len(self.calls()) < n:
            time.sleep(0.05)
        return self.calls()


@pytest.fixture
def h(tmp_path):
    hook = Hook(tmp_path)
    hook.set_stamp(time.time() - 100)
    return hook


def best_vs_baseline(run, n=20):
    """Best-of-n of `run` and of the kill-switched hook (bash start-up alone), interleaved in one loop: CPU load
    inflates both sides alike, so only a ratio is meaningful (an absolute wall time fails under oversubscription)."""
    base, real = [], []
    for _ in range(n):
        base.append(run(COORDGRAPH_HOOK_DISABLED="1").elapsed)
        r = run()
        assert r.returncode == 0 and r.stdout == "" and r.stderr == ""
        real.append(r.elapsed)
    return min(base), min(real)


def test_nothing_changed_is_exit_0_with_no_call_and_cheap_against_a_bare_bash_start(h):
    # structure pins "no Python"; the cost of the pruned `find` is judged against the same-run baseline. The
    # absolute figures (and the < 50 ms target) are measurements recorded in the README, not assertions.
    h.put(L, mtime=time.time() - 500)
    base, real = best_vs_baseline(h.run)
    assert h.calls() == []
    assert real < 6 * base + 0.03, (base, real)


def test_a_large_payload_does_not_slow_the_no_change_path(h):
    # bash reads a pipe a byte at a time with `read -d ''`: cost linear in the payload (~100 ms at 300 KB, ~25x a
    # small call). A relative measure of best-of-7 interleaved runs: load scales both sides alike.
    h.put(L, mtime=time.time() - 500)
    big = '{"tool_response":"' + "x" * 300_000 + '"}'
    small, large = [], []
    for _ in range(7):
        small.append(h.run().elapsed)
        r = h.run(payload=big)
        assert r.returncode == 0 and r.stderr == ""
        large.append(r.elapsed)
    assert h.calls() == []
    assert min(large) < 8 * min(small) + 0.02, (min(small), min(large))


# --- kill switch -------------------------------------------------------------------------------------------

@pytest.mark.parametrize("value", ["1", "true", "TRUE", " yes ", "on"])
def test_the_env_kill_switch_exits_0_with_a_pending_edit(h, value):
    h.put(L)
    assert h.run(COORDGRAPH_HOOK_DISABLED=value).returncode == 0 and h.calls() == []


@pytest.mark.parametrize("value", ["0", "false", "no", "off", ""])
def test_any_other_value_leaves_the_hook_on(h, value):
    h.put(L)
    h.run(COORDGRAPH_HOOK_DISABLED=value)
    assert len(h.calls()) == 1


def test_the_disabled_file_kills_the_hook(h):
    h.put(L)
    (h.state / "disabled").touch()
    assert h.run().returncode == 0 and h.calls() == []


# --- what counts as a mirrored file ---------------------------------------------------------------------------

@pytest.mark.parametrize("rel", [
    "a/docs/plans/x-coordination.md", "a/docs/reviews/r.md", "a/docs/requirements/q.md", "a/docs/BACKLOG.md",
    "docs/BACKLOG.md", "claude/cobb/kaizen/plan.md", "docs/plans/top-coordination.md"])
def test_each_covered_kind_triggers_a_sync_with_its_repo_relative_path(h, rel):
    h.put(rel)
    r = h.run()
    assert r.returncode == 0 and r.stderr == ""
    assert [c.split(" | ")[0] for c in h.calls()] == [f"hook-sync hook-sync --confirm coordination {rel}"]


@pytest.mark.parametrize("rel", [
    "a/docs/notes.md", "a/docs/plans/x.md", "archive/docs/plans/x-coordination.md",
    "a/archive/docs/BACKLOG.md", "deprecated/docs/BACKLOG.md", ".git/docs/BACKLOG.md", "n/node_modules/docs/BACKLOG.md",
    ".venv/docs/BACKLOG.md", "a/.hidden/docs/BACKLOG.md"])
def test_uncovered_and_pruned_paths_never_trigger(h, rel):
    h.put(rel)
    assert h.run().returncode == 0 and h.calls() == []


def test_a_registered_kb_file_triggers_and_an_unregistered_one_does_not(h):
    h.put("coordination-graph/kb-files.txt", "# registry\nkb/a.md\n\nkb/gone.md\n", mtime=time.time() - 500)
    h.put("kb/a.md")
    h.put("kb/other.md")
    h.run()
    assert [c.split(" | ")[0] for c in h.calls()] == ["hook-sync hook-sync --confirm coordination kb/a.md"]


def test_the_edit_is_in_the_graph_before_the_hook_returns_and_a_second_call_is_a_no_op(h):
    h.put(L)
    h.run()
    assert h.stamp.stat().st_mtime >= time.time() - 5
    h.run()
    assert len(h.calls()) == 1


def test_the_root_is_the_project_dir_never_the_payload_cwd(h, tmp_path):
    elsewhere = tmp_path / "wt"
    elsewhere.mkdir()
    (elsewhere / "a/docs/plans").mkdir(parents=True)
    (elsewhere / L).write_text("x")
    h.put(L)
    h.run(payload=f'{{"cwd":"{elsewhere}","tool_name":"Write","tool_input":{{"file_path":"{elsewhere / L}"}}}}')
    calls = h.calls()
    assert len(calls) == 1 and f"root={h.root} " in calls[0] and f"--confirm coordination {L} |" in calls[0]


# --- the confirmation (decision: option A) ------------------------------------------------------------------

def test_the_script_hard_codes_the_literal_key_and_never_reads_or_exports_one():
    text = HOOK.read_text()
    assert re.search(r"^CONFIRM_KEY=coordination$", text, re.M)
    assert "COORDGRAPH_KEY" not in text
    assert not re.search(r"export[^\n]*SHARED_CONFIRM", text)
    assert text.count("--confirm") >= 2 and not re.search(r'--confirm "\$\{?COORDGRAPH', text)


def test_the_flag_carries_the_literal_key_whatever_the_environment_says(h):
    h.put(L)
    h.run(COORDGRAPH_KEY="other", COORDGRAPH_SHARED_CONFIRM="other")
    (call,) = h.calls()
    assert "hook-sync --confirm coordination " in call
    assert "shared=unset" in call  # the hook neither exports nor passes on an ambient confirmation


def test_the_detached_retry_carries_the_same_flag(h):
    (h.state / "retry-due").write_text(f"{int(time.time()) - 5}\n")
    h.run()
    (call,) = h.wait_calls(1)
    assert call.startswith("retry-failed retry-failed --confirm coordination |") and "shared=unset" in call


# --- exit 2 and the hook's own faults ------------------------------------------------------------------------------

def test_a_hook_sync_exit_2_is_relayed_to_the_agent_and_the_stamp_still_advances(h):
    h.put(L)
    h.script("hook-sync", rc=2, err="coordgraph: 1 mirrored file(s) not synced:\n  " + L)
    r = h.run()
    assert r.returncode == 2 and L in r.stderr and "not synced" in r.stderr
    assert "stdout noise" not in r.stdout + r.stderr
    assert h.stamp.stat().st_mtime >= time.time() - 5
    assert h.run().returncode == 0 and len(h.calls()) == 1  # an unrelated later call neither warns nor syncs


@pytest.mark.parametrize("rc", [1, 3, 127])
def test_an_own_fault_is_never_read_as_nothing_to_sync(h, rc):
    h.put(L)
    h.script("hook-sync", rc=rc)
    r = h.run()
    assert r.returncode == 2 and L in r.stderr and f"exit {rc}" in r.stderr and "hook.log" in r.stderr
    assert (h.state / "failed.pending").read_text().split() == [L]
    assert int((h.state / "retry-due").read_text()) in range(int(time.time()) + 55, int(time.time()) + 66)
    assert h.stamp.stat().st_mtime >= time.time() - 5


def test_a_hook_sync_past_the_timeout_is_killed_and_treated_as_an_own_fault(h):
    h.put(L)
    h.script("hook-sync", sleep=30)
    r = h.run(COORDGRAPH_HOOK_TIMEOUT_S="1")
    assert r.returncode == 2 and r.elapsed < 10 and "exit 124" in r.stderr
    assert (h.state / "failed.pending").read_text().split() == [L]


def test_a_missing_entry_point_is_an_own_fault(h):
    h.put(L)
    r = h.run(COORDGRAPH_HOOK_CLI=str(h.root / "no-such-cli"))
    assert r.returncode == 2 and L in r.stderr


def test_the_shells_pending_append_waits_for_the_state_lock(h):
    h.put(L)
    h.script("hook-sync", rc=1)
    (h.state / "state.lock").touch()
    release = threading.Event()

    def hold():
        with open(h.state / "state.lock", "a") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            release.wait(10)

    t = threading.Thread(target=hold)
    t.start()
    time.sleep(0.2)
    threading.Timer(1.0, release.set).start()
    r = h.run()
    t.join()
    assert r.returncode == 2 and r.elapsed >= 0.8
    assert (h.state / "failed.pending").read_text().split() == [L]


def test_the_stamp_goes_to_the_calls_start_never_backwards(h):
    h.set_stamp(time.time() + 1000)  # another session already advanced it
    h.put(L, mtime=time.time() + 2000)
    h.run()
    assert h.stamp.stat().st_mtime > time.time() + 900


# --- the detached retry ---------------------------------------------------------------------------------------

def test_a_due_retry_is_launched_once_and_retry_due_is_bumped_first(h):
    (h.state / "retry-due").write_text(f"{int(time.time()) - 5}\n")
    h.run()
    assert len(h.wait_calls(1)) == 1
    due = int((h.state / "retry-due").read_text())
    assert due >= int(time.time()) + 55
    for _ in range(20):
        assert h.run().returncode == 0
    time.sleep(0.3)
    assert len(h.calls()) == 1  # a crashing retry costs one fork per minute, not one per call


def test_a_retry_that_is_not_due_is_not_launched(h):
    (h.state / "retry-due").write_text(f"{int(time.time()) + 300}\n")
    h.run()
    time.sleep(0.3)
    assert h.calls() == []


def test_a_slow_detached_retry_never_holds_the_hooks_pipes(h):
    (h.state / "retry-due").write_text(f"{int(time.time()) - 5}\n")
    h.script("retry-failed", sleep=4)
    r = h.run()
    assert r.returncode == 0 and r.elapsed < 1.5


# --- no stamp (first run) -----------------------------------------------------------------------------------------------

def test_no_stamp_runs_one_first_run_and_leaves_the_stamp_to_hook_sync(h):
    h.stamp.unlink()
    r = h.run()
    assert r.returncode == 0
    (call,) = h.calls()
    assert re.match(r"hook-sync hook-sync --confirm coordination --first-run --stamp-at \d+\.\d+ \|", call)
    assert not h.stamp.exists()


def test_no_stamp_and_a_retry_due_in_the_future_waits(h):
    h.stamp.unlink()
    (h.state / "retry-due").write_text(f"{int(time.time()) + 300}\n")
    assert h.run().returncode == 0 and h.calls() == []


def test_no_stamp_and_a_retry_due_in_the_past_runs_again(h):
    h.stamp.unlink()
    (h.state / "retry-due").write_text(f"{int(time.time()) - 1}\n")
    h.run()
    assert len(h.calls()) == 1


def test_a_first_run_that_faults_lists_every_discovered_file_creates_the_stamp_and_waits_a_minute(h):
    h.stamp.unlink()
    h.put(L, mtime=time.time() - 9000)
    h.put("b/docs/BACKLOG.md", mtime=time.time() - 9000)
    h.put("coordination-graph/kb-files.txt", "kb/a.md\n")
    h.put("kb/a.md")
    h.script("hook-sync", rc=1)
    r = h.run()
    assert r.returncode == 2
    assert sorted((h.state / "failed.pending").read_text().split()) == sorted(["b/docs/BACKLOG.md", "kb/a.md", L])
    assert h.stamp.exists()


# --- Write/Edit naming a file already in the failure state (R12) ---------------------------------------------

def edit_payload(path):
    return f'{{"hook_event_name":"PostToolUse","tool_name":"Edit","tool_input":{{"file_path":"{path}"}}}}'


def test_a_write_naming_a_failed_file_is_synced_even_when_the_stamp_is_past_it(h):
    h.put(L, mtime=time.time() - 500)
    (h.state / "failed.json").write_text(f'{{"{L}": {{"attempts": 1}}}}')
    r = h.run(payload=edit_payload(h.root / L))
    assert r.returncode == 0
    assert [c.split(" | ")[0] for c in h.calls()] == [f"hook-sync hook-sync --confirm coordination {L}"]


def test_a_relative_payload_path_in_the_pending_list_counts_too(h):
    h.put(L, mtime=time.time() - 500)
    (h.state / "failed.pending").write_text(f"{L}\n")
    h.run(payload=edit_payload(L))
    assert len(h.calls()) == 1


def test_a_write_naming_a_file_that_never_failed_or_lies_outside_the_root_does_nothing(h):
    h.put(L, mtime=time.time() - 500)
    (h.state / "failed.json").write_text('{"other.md": {}}')
    h.run(payload=edit_payload(h.root / L))
    h.run(payload=edit_payload("/elsewhere/a/docs/plans/x-coordination.md"))
    assert h.calls() == []


# --- hardening of the seams and the failure message ---------------------------------------------------------------

@pytest.mark.parametrize("bad", ["abc", "", "-5", "1.5", "5s"])
def test_a_non_numeric_timeout_falls_back_to_the_default_not_to_timeouts_exit_125(h, bad):
    h.put(L)
    r = h.run(COORDGRAPH_HOOK_TIMEOUT_S=bad)
    assert r.returncode == 0 and r.stderr == "" and len(h.calls()) == 1


def test_a_first_run_that_faults_names_a_few_files_and_the_count_not_all_of_them(h):
    h.stamp.unlink()
    for i in range(40):
        h.put(f"d{i}/docs/BACKLOG.md", mtime=time.time() - 9000)
    h.script("hook-sync", rc=1)
    r = h.run()
    assert r.returncode == 2 and len(r.stderr.splitlines()) < 30 and "and 2" in r.stderr and "more" in r.stderr
    assert len((h.state / "failed.pending").read_text().split()) == 40  # every file is still queued


def test_the_script_says_it_needs_bash_5_and_that_both_seams_are_never_set_in_production():
    head = HOOK.read_text().split("set -u")[0]
    assert "bash 5" in head.lower() and "never set in production" in head


def test_the_shell_discovery_equals_the_python_discovery_on_one_tree(h):
    from coordgraph.discover import discover, kb_files

    old = time.time() - 9000
    covered = ["a/docs/plans/x-coordination.md", "docs/plans/top-coordination.md", "a/b/docs/reviews/r-impl.md",
               "docs/reviews/r.md", "a/docs/requirements/q.md", "docs/BACKLOG.md", "a/docs/BACKLOG.md",
               "claude/cobb/kaizen/plan.md"]
    uncovered = ["a/docs/notes.md", "a/docs/plans/x.md", "archive/docs/BACKLOG.md", "a/archive/docs/reviews/r.md",
                 "deprecated/docs/BACKLOG.md", ".git/docs/BACKLOG.md", "n/node_modules/docs/BACKLOG.md",
                 ".venv/docs/BACKLOG.md", ".h/docs/BACKLOG.md", "x/claude/cobb/kaizen/plan.md",
                 "claude/cobb/kaizen/history.md", "a/docs/reviews/deep/r.md", "a/docs/reviews/r.txt"]
    for rel in covered + uncovered:
        h.put(rel, mtime=old)
    h.put("coordination-graph/kb-files.txt", "# r\nkb/a.md\n\nkb/gone.md\n", mtime=old)
    h.put("kb/a.md", mtime=old)
    h.stamp.unlink()
    h.set_stamp(old - 1000)
    h.run()
    (call,) = h.calls()
    shell = set(call.split(" | ")[0].split("--confirm coordination ")[1].split(" "))
    python = {f.path for f in discover(h.root)} | {p for p in kb_files(h.root) if (h.root / p).is_file()}
    assert shell == python and shell >= set(covered) and not shell & set(uncovered)
