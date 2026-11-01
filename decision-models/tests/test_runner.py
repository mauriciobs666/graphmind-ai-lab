"""`driver run`, the unattended runner, and `pause [--now]`/`resume` (typed-decision-models U6.3).

Rows: plan6 `docs/plans/typed-decision-models6.md` §5 U6.3 over §3.2, plus the contracts U5.6 left
for the runner (one `Campaign` per process, the `.exit` file, `interrupted` once, the lock).

Every launch goes to a fake `claude` this file writes (a Python script first on a `PATH` that holds
only it, `/usr/bin` and `/bin`), under a fake `HOME`: nothing calls the real CLI or reads the real
`~/.claude`. The fake records its argv, its raw environment (`/proc/self/environ`), its process
group and its open descriptors, then follows the per-call plan the test wrote: stage a transcript
shape with `test_campaign.stage` (so the audit sees exactly the shapes the U5.6 tests use), wait,
print, exit. Its control directory is baked into the script, because the runner passes it no
environment beyond the allowlist.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from test_campaign import GUARD_COMMAND, ROLES, SPAWN_TEMPLATE, Clock, write_fixture

from decisionmodels import driver, runner
from decisionmodels.campaign import Campaign

COMPONENT = Path(__file__).resolve().parents[1]
DRIVER_TEMPLATE = ("Make exactly one Agent call: subagent_type {agent_type}, with the prompt below "
                   "copied exactly, then reply done.\n\n{spawn_prompt}")
SETTINGS = {"model": "opus", "theme": "dark", "permissions": {"allow": ["Read"]},
            "modelSettings": {"effort": "high"}}

FAKE = '''#!{python}
import json, os, signal, subprocess, sys, time
from pathlib import Path

sys.dont_write_bytecode = True
CONTROL = Path({control!r})
COMPONENT = {component!r}

n = len(list(CONTROL.glob("call-*.json")))
plans = json.loads((CONTROL / "plan.json").read_text())
plan = plans[min(n, len(plans) - 1)]
if plan.get("ignore_term"):         # first, so a SIGTERM at any later point finds it ignored
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
argv = sys.argv[1:]
session = argv[argv.index("--session-id") + 1]
raw = Path("/proc/self/environ").read_bytes().decode()
env = dict(entry.split("=", 1) for entry in raw.split("\\0") if entry)
fds = []
for fd in os.listdir("/proc/self/fd"):
    try:
        fds.append(os.readlink(f"/proc/self/fd/{{fd}}"))
    except OSError:
        pass


def stage(shape):
    sys.path[:0] = [COMPONENT, COMPONENT + "/tests"]
    import test_campaign as tc
    from decisionmodels.campaign import Campaign
    camp = Campaign(Path.cwd().parent, claude_dir=Path(os.environ["HOME"]) / ".claude")
    (opaque,) = [p.name[:-len(".jsonl")] for p in (Path.cwd() / "req").glob("*.jsonl")]
    tc.stage(camp, session, opaque, shape, exit_code=None)
    if shape == "clean":
        tc.write_answer(camp, opaque, tc.answer_bytes(camp, opaque))


record = {{"argv": argv, "env": env, "cwd": os.getcwd(), "pid": os.getpid(),
          "pgid": os.getpgid(0), "sid": os.getsid(0), "fds": fds,
          "stdin": os.readlink("/proc/self/fd/0")}}
if plan.get("stage"):
    stage(plan["stage"])
if plan.get("grandchild"):
    # Dies on SIGTERM even when this process ignores it (an ignored signal survives exec).
    record["grandchild"] = subprocess.Popen(
        ["sleep", "60"], preexec_fn=lambda: signal.signal(signal.SIGTERM, signal.SIG_DFL)).pid
if plan.get("stubborn_grandchild"):
    # Ignores SIGTERM from before its exec, and inherits every descriptor (the lock's included).
    record["grandchild"] = subprocess.Popen(
        ["sleep", "60"], close_fds=False,
        preexec_fn=lambda: signal.signal(signal.SIGTERM, signal.SIG_IGN)).pid
partial = CONTROL / f".call-{{n}}.json"
partial.write_text(json.dumps(record))
os.replace(partial, CONTROL / f"call-{{n}}.json")
if plan.get("wait_for"):
    deadline = time.monotonic() + 60
    while not (CONTROL / plan["wait_for"]).exists() and time.monotonic() < deadline:
        time.sleep(0.02)
time.sleep(plan.get("sleep", 0))
if plan.get("final"):
    stage(plan["final"])
if "settings" in plan:
    settings = Path(os.environ["HOME"]) / ".claude" / "settings.json"
    settings.write_text(json.dumps(plan["settings"]))
sys.stdout.write(plan.get("stdout", ""))
sys.stderr.write(plan.get("stderr", ""))
sys.exit(plan.get("exit", 0))
'''

RUN_WITH_GRACE = ("import sys; from datetime import timedelta; from pathlib import Path; "
                  "from decisionmodels import runner; sys.exit(runner.run(Path(sys.argv[1]), "
                  "kill_grace=timedelta(seconds=float(sys.argv[2]))))")


class Bench:
    """A smoke campaign, a fake `HOME` with user settings, and the fake `claude` on `PATH`."""

    def __init__(self, tmp_path: Path, *, settings: dict[str, Any] | None = None,
                 batches: int = 1, batch_timeout: timedelta = timedelta(minutes=45)) -> None:
        self.home = tmp_path / "home"
        (self.home / ".claude").mkdir(parents=True)
        self.control = tmp_path / "control"
        self.control.mkdir()
        self.bin = tmp_path / "bin"
        self.bin.mkdir()
        fake = self.bin / "claude"
        fake.write_text(FAKE.format(python=sys.executable, control=str(self.control),
                                    component=str(COMPONENT)), "utf-8")
        fake.chmod(0o755)
        recorded = dict(SETTINGS if settings is None else settings)
        self.write_settings(recorded)
        self.camp = Campaign(tmp_path / "camp", claude_dir=self.home / ".claude", clock=Clock())
        self.camp.init(point="guard-judge", phase="smoke",
                       fixture=write_fixture(tmp_path / "fixture.jsonl", 2),
                       batch_timeout=batch_timeout)
        self.camp.record_expectations(
            guard_command=GUARD_COMMAND, roles=ROLES, driver_template=DRIVER_TEMPLATE,
            user_settings_digests=runner.user_settings_digests(recorded))
        for _ in range(batches - 1):
            self.camp.queue_batch("label", ["fx_0001"])
        self.path = f"{self.bin}:/usr/bin:/bin"

    @property
    def root(self) -> Path:
        return self.camp.root

    def write_settings(self, settings: dict[str, Any]) -> None:
        (self.home / ".claude" / "settings.json").write_text(json.dumps(settings), "utf-8")

    def plan(self, *calls: dict[str, Any]) -> None:
        (self.control / "plan.json").write_text(json.dumps(list(calls)), "utf-8")

    def calls(self) -> list[dict[str, Any]]:
        found = sorted(self.control.glob("call-*.json"), key=lambda p: int(p.stem.split("-")[1]))
        return [json.loads(path.read_text("utf-8")) for path in found]

    def call(self, n: int, timeout: float = 15) -> dict[str, Any]:
        path = self.control / f"call-{n}.json"
        wait_until(path.exists, timeout)
        return json.loads(path.read_text("utf-8"))

    def release(self) -> None:
        (self.control / "release").touch()

    def env(self) -> dict[str, str]:
        """The environment for a runner started as its own process."""
        env = {k: v for k, v in os.environ.items() if k.startswith("PYTHON")}
        env.update(PATH=self.path, HOME=str(self.home), LANG="C.UTF-8", LC_ALL="C.UTF-8")
        return env

    def start(self, kill_grace: timedelta | None = None) -> subprocess.Popen:
        """`driver run` as a separate process (for the signal and lock cases); with `kill_grace`,
        `runner.run` with that grace instead of `KILL_GRACE`."""
        argv = [sys.executable, "-m", "decisionmodels.driver", "run", "--root", str(self.root)]
        if kill_grace is not None:
            argv = [sys.executable, "-c", RUN_WITH_GRACE, str(self.root),
                    str(kill_grace.total_seconds())]
        return subprocess.Popen(argv, cwd=COMPONENT, env=self.env(),
                                stdin=subprocess.PIPE,     # never what the child gets
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    def events(self, kind: str | None = None) -> list[dict[str, Any]]:
        return [e for e in self.camp.events() if kind is None or e["event"] == kind]

    def outcomes(self) -> list[dict[str, Any]]:
        return self.events("outcome")

    def batches(self) -> dict[str, str]:
        return {b.batch_id: b.state for b in self.camp.state().batches.values()}


@pytest.fixture
def bench(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Bench:
    made = Bench(tmp_path)
    monkeypatch.setenv("HOME", str(made.home))
    monkeypatch.setenv("PATH", made.path)
    monkeypatch.setenv("LANG", "C.UTF-8")
    monkeypatch.setenv("LC_ALL", "C.UTF-8")
    return made


def wait_until(predicate: Callable[[], bool], timeout: float = 15) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out waiting for {predicate}")
        time.sleep(0.02)


def alive(pid: int) -> bool:
    """A live, non-zombie process."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text("ascii")
    except OSError:
        return False
    return stat.rsplit(")", 1)[1].split()[0] != "Z"


def run(bench: Bench, capsys: pytest.CaptureFixture, **kwargs: Any) -> tuple[int, str, str]:
    code = driver.main(["run", "--root", str(bench.root)]) if not kwargs else runner.run(
        bench.root, **kwargs)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def exit_file(bench: Bench, session: str) -> str:
    return (bench.camp.store / "runs" / f"{session}.exit").read_text("ascii")


# --- environment and argv ------------------------------------------------------------------------


def test_the_child_gets_the_allowlisted_environment_and_the_planned_command(
        bench: Bench, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    for name, value in {"USER": "ana", "TZ": "America/Sao_Paulo", "TMPDIR": "/tmp",
                        "ANTHROPIC_API_KEY": "sk-planted", "ANTHROPIC_MODEL": "planted",
                        "CLAUDE_CODE_EFFORT_LEVEL": "low", "CLAUDE_CODE_FORK_SUBAGENT": "0",
                        "OTHER": "x"}.items():
        monkeypatch.setenv(name, value)
    bench.plan({"final": "clean"})
    code, out, _ = run(bench, capsys)
    assert code == 0 and "stopped: done" in out

    (call,) = bench.calls()
    assert call["env"] == {"PATH": bench.path, "HOME": str(bench.home), "USER": "ana",
                           "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "TZ": "America/Sao_Paulo",
                           "TMPDIR": "/tmp", "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS": "1"}
    (issued,) = bench.events("issued")
    request = bench.root / "exchange" / "req" / f"{issued['opaque']}.jsonl"
    message = (DRIVER_TEMPLATE.replace("{agent_type}", "dm-labeler")
               .replace("{spawn_prompt}",
                        SPAWN_TEMPLATE.replace("{request_path}", str(request))))
    assert call["argv"] == [
        "-p", "--agent", "dm-driver", "--session-id", issued["sessionId"], "--strict-mcp-config",
        "--mcp-config", f"{bench.root}/exchange/empty-mcp.json", "--permission-mode",
        "acceptEdits", "--permission-prompts", "none", "--output-format", "json", message]
    assert "--fallback-model" not in call["argv"]
    assert call["cwd"] == str(bench.root / "exchange")
    # Its own process group (and session), so a kill reaches everything it started.
    assert call["pgid"] == call["pid"] == call["sid"]
    # The runner's lock descriptor is passed down (an orphan keeps the campaign locked).
    assert str(bench.root / "store" / "driver.lock") in call["fds"]
    assert call["stdin"] == "/dev/null"
    assert [o["outcome"] for o in bench.outcomes()] == ["ingested"]


def test_stdout_stderr_and_the_exit_code_are_the_launch_evidence(
        bench: Bench, capsys: pytest.CaptureFixture) -> None:
    """A launch that left no transcript: `launchFailed`, with what the child printed and its exit
    code recorded as is; `run` then stops (one launch per fire into a broken environment)."""
    # A second call, if the run ever made one, would succeed: the test fails rather than hangs.
    bench.plan({"stdout": '{"type": "result"}', "stderr": "auth failed\n", "exit": 3},
               {"final": "clean"})
    code, out, _ = run(bench, capsys)
    assert code == 0 and "launchFailed" in out
    assert len(bench.calls()) == 1
    (outcome,) = bench.outcomes()
    assert outcome["reason"] == "launchFailed"
    assert outcome["launch"] == {"exitCode": 3, "stdout": '{"type": "result"}',
                                 "stderr": "auth failed\n"}
    session = bench.events("issued")[0]["sessionId"]
    assert exit_file(bench, session) == "3\n"


def test_a_missing_claude_binary_is_a_launch_failure_not_a_crash(
        bench: Bench, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    monkeypatch.setenv("PATH", "/nonexistent")
    bench.plan({"final": "clean"})
    # Puts the fake back on PATH: a run that went on launching would reach it, and fail below.
    restore = threading.Timer(2, os.environ.__setitem__, ("PATH", bench.path))
    restore.start()
    try:
        code, _, _ = run(bench, capsys)
    finally:
        restore.cancel()
    assert code == 0 and bench.calls() == []
    (outcome,) = bench.outcomes()
    assert outcome["reason"] == "launchFailed" and outcome["launch"]["exitCode"] is None
    assert "the launch failed" in outcome["launch"]["stderr"]


def test_the_exit_file_is_written_before_ingest(bench: Bench,
                                                capsys: pytest.CaptureFixture) -> None:
    """A driver transcript with the prompt only and exit 0 is never spawned; the same with no
    observed exit would be `launchFailed`, so the ingest must already see `.exit`."""
    bench.plan({"final": "no-tool-use"}, {"final": "clean"})
    code, _, _ = run(bench, capsys)
    assert code == 0
    assert [o.get("reason") for o in bench.outcomes()] == ["neverSpawned", None]
    assert bench.batches() == {"bat_0001": "ingested"}


def test_the_exit_file_is_written_atomically(bench: Bench, monkeypatch: pytest.MonkeyPatch,
                                             capsys: pytest.CaptureFixture) -> None:
    replaced: list[tuple[str, str]] = []
    real = os.replace

    def spy(src: Any, dst: Any) -> None:
        replaced.append((Path(src).name, Path(dst).name))
        real(src, dst)

    monkeypatch.setattr(os, "replace", spy)
    bench.plan({"final": "clean"})
    run(bench, capsys)
    session = bench.events("issued")[0]["sessionId"]
    assert (f".{session}.exit.partial", f"{session}.exit") in replaced
    assert exit_file(bench, session) == "0\n"


def test_a_usage_stop_launches_once_and_exits(bench: Bench,
                                              capsys: pytest.CaptureFixture) -> None:
    bench.plan({"final": "rate-limit"})
    code, out, _ = run(bench, capsys)
    assert code == 0 and "usage-stop" in out
    assert len(bench.calls()) == 1
    assert [o["outcome"] for o in bench.outcomes()] == ["usage-stop"]


def test_one_campaign_instance_serves_the_whole_run(
        bench: Bench, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    """U5.6's contract: the usage-stop memory is the instance's, so a new instance per iteration
    would launch again inside a spent window."""
    made: list[Campaign] = []

    class Counting(Campaign):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            made.append(self)

    monkeypatch.setattr(runner, "Campaign", Counting)
    bench.plan({"final": "no-tool-use"}, {"final": "clean"})
    assert run(bench, capsys)[0] == 0
    assert len(bench.calls()) == 2 and len(made) == 1


def test_a_halt_stops_the_run(bench: Bench, capsys: pytest.CaptureFixture) -> None:
    bench.plan({"final": "opus-driver"})
    code, out, _ = run(bench, capsys)
    assert code == 0 and "halted" in out
    assert len(bench.calls()) == 1 and bench.outcomes()[0]["outcome"] == "halted"


def test_a_halt_on_ingest_stops_before_the_next_settings_check(
        bench: Bench, capsys: pytest.CaptureFixture) -> None:
    """The run stops on `IngestResult.stop` itself: a halting child that also edited the settings
    gives `stopped: halted` with the halt's reason, exit 0, and no refusal."""
    bench.plan({"final": "opus-driver", "settings": {**SETTINGS, "hooks": {}}})
    code, out, _ = run(bench, capsys)
    halt = bench.camp.state().halt
    assert code == 0
    assert f"stopped: halted ({halt['reason']} ({halt['voidClass']}) on bat_0001)" in out
    assert not (bench.root / "store" / "refusal").exists()


# --- user settings -------------------------------------------------------------------------------


RECOVERY = "revert it, or run `driver install` to accept it"
MARKER = "SECRET-MARKER-9f3a"


def refused(bench: Bench, capsys: pytest.CaptureFixture, reason: str) -> str:
    """A launch that refuses with `reason`: nothing launched, ledger byte-equal, the reason (with
    a timestamp) in `store/refusal`, and printed by `status`. Returns the refusal text."""
    bench.plan({"final": "clean"})
    before = bench.camp.ledger_path.read_bytes()
    code, _, err = run(bench, capsys)
    assert code == 1 and reason in err
    assert bench.calls() == []
    assert bench.camp.ledger_path.read_bytes() == before
    refusal = (bench.root / "store" / "refusal").read_text("utf-8")
    stamp, text = refusal.split(" ", 1)                         # a timestamp, then the reason
    assert datetime.fromisoformat(stamp).tzinfo is not None and reason in text
    assert driver.main(["status", "--root", str(bench.root)]) == 0
    assert any(line.startswith("REFUSED:") and reason in line
               for line in capsys.readouterr().out.splitlines())
    return refusal


FORBIDDEN_CASES = [("env", {"FOO": "1"}), ("hooks", {}), ("disableAllHooks", True),
                   ("autoContinueAtUsageLimit", True), ("fallbackModel", "sonnet"),
                   ("apiKeyHelper", "/bin/key")]


@pytest.mark.parametrize(("key", "value"), FORBIDDEN_CASES)
def test_a_forbidden_setting_refuses_the_launch(bench: Bench, capsys: pytest.CaptureFixture,
                                                key: str, value: Any) -> None:
    bench.write_settings({**SETTINGS, key: value})
    refused(bench, capsys, f"`{key}`")


@pytest.mark.parametrize(("key", "value"), FORBIDDEN_CASES)
def test_a_forbidden_setting_refuses_even_when_install_recorded_it(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture,
        key: str, value: Any) -> None:
    """As if `install` had seen it: recording it must not make it the campaign's baseline."""
    made = Bench(tmp_path, settings={**SETTINGS, key: value})
    monkeypatch.setenv("HOME", str(made.home))
    monkeypatch.setenv("PATH", made.path)
    refused(made, capsys, f"`{key}`")


def test_only_the_refusing_value_of_a_valued_forbidden_key_refuses() -> None:
    for key in ("disableAllHooks", "autoContinueAtUsageLimit"):
        assert runner.forbidden_settings_refusal({key: True}) is not None
        assert runner.forbidden_settings_refusal({key: False}) is None
    assert runner.forbidden_settings_refusal(SETTINGS) is None


@pytest.mark.parametrize(("edit", "key"), [
    ({"permissions": {"allow": ["Read"], "deny": ["Write"]}}, "permissions"),
    ({"permissions": {"allow": ["Read"], "ask": ["Bash"]}}, "permissions"),
    ({"permissions": {"allow": ["Read"], "additionalDirectories": ["/x"]}}, "permissions"),
    ({"modelSettings": {"effort": "low"}}, "modelSettings"),
    ({"effortLevel": "low"}, "effortLevel"),
    ({"agentPushNotifEnabled": True}, "agentPushNotifEnabled"),
    ({"outputStyle": "Explanatory"}, "outputStyle"),
    ({"enabledPlugins": {"x@y": True}}, "enabledPlugins"),
    ({"futureKey": 1}, "futureKey"),
])
def test_a_changed_recorded_setting_refuses_and_names_the_key(
        bench: Bench, capsys: pytest.CaptureFixture, edit: dict[str, Any], key: str) -> None:
    bench.write_settings({**SETTINGS, **edit})
    text = refused(bench, capsys, f"`{key}`")
    assert text.rstrip("\n").endswith(RECOVERY)


def test_a_key_removed_since_install_refuses(bench: Bench,
                                             capsys: pytest.CaptureFixture) -> None:
    bench.write_settings({k: v for k, v in SETTINGS.items() if k != "modelSettings"})
    text = refused(bench, capsys, "removed: `modelSettings`")
    assert text.rstrip("\n").endswith(RECOVERY)


def test_a_refusal_names_added_removed_and_changed_keys_and_shows_no_value(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    made = Bench(tmp_path, settings={**SETTINGS, "permissions": {"deny": ["Write"]}})
    monkeypatch.setenv("HOME", str(made.home))
    monkeypatch.setenv("PATH", made.path)
    made.write_settings({"model": "opus", "permissions": {"deny": [MARKER]},
                         "outputStyle": MARKER, "futureKey": MARKER})
    text = refused(made, capsys, "added: `futureKey`, `outputStyle`")
    assert "removed: `modelSettings`" in text and "changed: `permissions`" in text
    assert MARKER not in text
    assert MARKER not in (made.root / "store" / "refusal").read_text("utf-8")


def test_a_forbidden_refusal_shows_no_value(bench: Bench, capsys: pytest.CaptureFixture) -> None:
    bench.write_settings({**SETTINGS, "env": {"TOKEN": MARKER}, "apiKeyHelper": MARKER})
    assert MARKER not in refused(bench, capsys, "`env`")


def test_the_forbidden_check_runs_before_the_digest_comparison(
        bench: Bench, capsys: pytest.CaptureFixture) -> None:
    bench.write_settings({**SETTINGS, "hooks": {}, "futureKey": 1})
    text = refused(bench, capsys, "`hooks`")
    assert "futureKey" not in text


@pytest.mark.parametrize("edit", [
    {"permissions": {"allow": ["Read", "Write"]}},
    {"permissions": {"allow": ["Read"], "defaultMode": "plan"}},
    {"theme": "light"}, {"model": "sonnet"}, {"verbose": True}, {"$schema": "https://x/y.json"},
    {"switchModelsOnFlag": True}, {"switchModelsOnFlag": False},
    {"autoContinueAtUsageLimit": False}, {"disableAllHooks": False},
])
def test_ordinary_user_edits_do_not_refuse(bench: Bench, capsys: pytest.CaptureFixture,
                                           edit: dict[str, Any]) -> None:
    bench.write_settings({**SETTINGS, **edit})
    bench.plan({"final": "clean"})
    assert run(bench, capsys)[0] == 0
    assert len(bench.calls()) == 1 and bench.batches() == {"bat_0001": "ingested"}


@pytest.mark.parametrize(("installed", "now"), [
    ({"model": "opus"}, {"model": "opus", "permissions": {"allow": ["Read"]}}),
    ({"model": "opus"}, {"model": "opus", "permissions": {"defaultMode": "plan"}}),
    ({"model": "opus", "permissions": {"allow": ["Read"]}}, {"model": "opus"}),
    ({"model": "opus", "permissions": {"allow": ["Read"]}},
     {"model": "opus", "permissions": {}}),
])
def test_permissions_edits_that_leave_nothing_digested_do_not_refuse(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture,
        installed: dict[str, Any], now: dict[str, Any]) -> None:
    """A first allow rule (install saw no block), a lone `defaultMode`, and the last allow rule
    with its block removed: absent == empty."""
    made = Bench(tmp_path, settings=installed)
    monkeypatch.setenv("HOME", str(made.home))
    monkeypatch.setenv("PATH", made.path)
    made.write_settings(now)
    made.plan({"final": "clean"})
    assert run(made, capsys)[0] == 0 and len(made.calls()) == 1


def test_a_deny_rule_after_an_allow_only_install_refuses(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    made = Bench(tmp_path, settings={"permissions": {"allow": ["Read"]}})
    monkeypatch.setenv("HOME", str(made.home))
    monkeypatch.setenv("PATH", made.path)
    made.write_settings({"permissions": {"allow": ["Read"], "deny": ["Write"]}})
    refused(made, capsys, "added: `permissions`")


def test_reordered_and_reindented_settings_do_not_refuse(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    recorded = {"permissions": {"deny": ["Bash"], "ask": ["Edit"], "allow": ["Read"]},
                "modelSettings": {"effort": "high", "x": 1}}
    made = Bench(tmp_path, settings=recorded)
    monkeypatch.setenv("HOME", str(made.home))
    monkeypatch.setenv("PATH", made.path)
    reordered = {"modelSettings": {"x": 1, "effort": "high"},
                 "permissions": {"allow": ["Read"], "ask": ["Edit"], "deny": ["Bash"]}}
    (made.home / ".claude" / "settings.json").write_text(json.dumps(reordered, indent=4), "utf-8")
    made.plan({"final": "clean"})
    assert run(made, capsys)[0] == 0 and len(made.calls()) == 1


def test_the_settings_are_checked_before_every_launch(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    """The first child edits the user settings; the second launch is refused, mid-run."""
    made = Bench(tmp_path, batches=2)
    monkeypatch.setenv("HOME", str(made.home))
    monkeypatch.setenv("PATH", made.path)
    made.plan({"final": "clean", "settings": {**SETTINGS, "hooks": {}}}, {"final": "clean"})
    code, _, err = run(made, capsys)
    assert code == 1 and "`hooks`" in err
    assert len(made.calls()) == 1
    assert made.batches() == {"bat_0001": "ingested", "bat_0002": "pending"}
    assert (made.root / "store" / "refusal").is_file()


def test_the_settings_are_checked_before_next_reconciles(
        bench: Bench, capsys: pytest.CaptureFixture) -> None:
    """plan6 §3.2: a refusal appends nothing, so it comes before `next`, which would reconcile
    this issued attempt (no transcript, no `.exit`) into an outcome."""
    bench.camp.issue(bench.camp.next())
    bench.write_settings({**SETTINGS, "hooks": {}})
    before = bench.camp.ledger_path.read_bytes()
    code, _, err = run(bench, capsys)
    assert code == 1 and "`hooks`" in err
    assert bench.camp.ledger_path.read_bytes() == before and bench.calls() == []


def test_a_successful_launch_clears_the_refusal(bench: Bench,
                                                capsys: pytest.CaptureFixture) -> None:
    bench.write_settings({**SETTINGS, "hooks": {}})
    bench.plan({"final": "clean"})
    assert run(bench, capsys)[0] == 1
    assert (bench.root / "store" / "refusal").is_file()
    bench.write_settings(SETTINGS)
    assert run(bench, capsys)[0] == 0
    assert not (bench.root / "store" / "refusal").exists()


def test_a_missing_settings_file_matches_an_install_that_saw_none(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    made = Bench(tmp_path, settings={})
    (made.home / ".claude" / "settings.json").unlink()
    monkeypatch.setenv("HOME", str(made.home))
    monkeypatch.setenv("PATH", made.path)
    made.plan({"final": "clean"})
    assert run(made, capsys)[0] == 0 and len(made.calls()) == 1


def test_settings_that_are_not_an_object_refuse(bench: Bench,
                                                capsys: pytest.CaptureFixture) -> None:
    (bench.home / ".claude" / "settings.json").write_text("[]", "utf-8")
    code, _, err = run(bench, capsys)
    assert code == 1 and "not a JSON object" in err and bench.calls() == []


def test_the_digests_are_over_canonical_json() -> None:
    """plan6 §3.2 "canonical JSON" (sorted keys, no whitespace, `permissions` without `allow` and
    `defaultMode`): `install` records these, so a hand edit that only reorders keys must not
    refuse every launch."""
    settings = {"theme": "dark", "permissions": {"deny": ["Bash"], "allow": ["Read"],
                                                  "defaultMode": "plan"},
                "modelSettings": {"effort": "high", "a": "é"}}
    digests = runner.user_settings_digests(settings)
    assert set(digests) == {"permissions", "modelSettings"}
    assert digests["permissions"] == hashlib.sha256(b'{"deny":["Bash"]}').hexdigest()
    assert digests["modelSettings"] == hashlib.sha256(
        '{"a":"é","effort":"high"}'.encode()).hexdigest()
    reordered = {"modelSettings": {"a": "é", "effort": "high"},
                 "permissions": {"defaultMode": "plan", "allow": ["Read"], "deny": ["Bash"]}}
    assert runner.user_settings_digests(reordered) == digests


def test_the_digests_cover_each_recorded_key_and_only_those() -> None:
    settings = {**SETTINGS, "permissions": {"allow": ["Read"], "deny": ["Bash"]}}
    base = runner.user_settings_digests(settings)
    assert runner.user_settings_digests({**settings, "theme": "x", "model": "y"}) == base
    assert runner.user_settings_digests({**settings, "permissions": {"deny": ["x"]}}) != base
    assert runner.user_settings_digests({**settings, "modelSettings": None}) != base
    without = {k: v for k, v in settings.items() if k != "modelSettings"}
    assert set(runner.user_settings_digests(without)) == {"permissions"}


def test_permissions_with_nothing_left_after_the_strip_are_not_recorded() -> None:
    """Absent == empty: only `allow`/`defaultMode` leave nothing to digest."""
    for permissions in ({}, {"allow": ["Read"]}, {"defaultMode": "plan"},
                        {"allow": ["Read"], "defaultMode": "plan"}):
        assert "permissions" not in runner.user_settings_digests({"permissions": permissions})
    assert "permissions" in runner.user_settings_digests({"permissions": {"deny": []}})


def test_a_digest_is_over_its_own_keys_value() -> None:
    digests = runner.user_settings_digests({"outputStyle": "a", "language": "b"})
    assert digests["outputStyle"] == hashlib.sha256(b'"a"').hexdigest()
    assert digests["language"] == hashlib.sha256(b'"b"').hexdigest()


def test_the_unrecorded_set_is_pinned() -> None:
    """plan6 §3.2's closed list; adding to it is a plan change."""
    assert runner.UNRECORDED_SETTINGS == {
        "model", "switchModelsOnFlag", "autoContinueAtUsageLimit", "disableAllHooks", "$schema",
        "theme", "editorMode", "verbose", "viewMode", "tui", "spinnerTipsEnabled",
        "spinnerTipsOverride", "spinnerVerbs", "prefersReducedMotion",
        "terminalProgressBarEnabled", "terminalTitleFromRename", "showTurnDuration",
        "syntaxHighlightingDisabled", "timeFormat", "timeZone", "preferredNotifChannel",
        "inputNeededNotifEnabled", "autoScrollEnabled", "wheelScrollAccelerationEnabled",
        "emojiCompletionEnabled", "promptSuggestionEnabled", "maxProseWidth",
        "vimInsertModeRemaps", "awaySummaryEnabled", "voice", "voiceEnabled",
        "feedbackSurveyRate", "axScreenReader", "spellcheck", "keybindingFlavor"}
    assert set(runner.FORBIDDEN_SETTINGS) == {"env", "hooks", "disableAllHooks",
                                              "autoContinueAtUsageLimit", "fallbackModel",
                                              "apiKeyHelper"}


def test_run_needs_the_install_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                      capsys: pytest.CaptureFixture) -> None:
    made = Bench(tmp_path)
    monkeypatch.setenv("HOME", str(made.home))
    monkeypatch.setenv("PATH", made.path)
    made.camp.record_expectations(guard_command=GUARD_COMMAND, roles=ROLES)  # no driver record
    code, _, err = run(made, capsys)
    assert code == 1 and "driver install" in err and made.calls() == []


# --- root, pause and resume ----------------------------------------------------------------------


def test_a_root_inside_a_git_work_tree_is_refused(bench: Bench,
                                                  capsys: pytest.CaptureFixture) -> None:
    (bench.root.parent / ".git").mkdir()
    bench.plan({"final": "clean"})
    code, _, err = run(bench, capsys)
    assert code == 1 and "git work tree" in err and bench.calls() == []


def test_status_shows_a_root_refusal(bench: Bench, capsys: pytest.CaptureFixture) -> None:
    """`run` refuses the root before taking the lock, so no `store/refusal` is written: `status`
    reports the reason itself."""
    (bench.root.parent / "AGENTS.md").write_text("", "utf-8")
    code, _, err = run(bench, capsys)
    assert code == 1 and "AGENTS.md" in err and bench.calls() == []
    assert driver.main(["status", "--root", str(bench.root)]) == 0
    assert any(line.startswith("REFUSED:") and "AGENTS.md" in line
               for line in capsys.readouterr().out.splitlines())


def test_run_without_a_campaign_is_refused(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    assert driver.main(["run", "--root", str(tmp_path / "none")]) == 1
    assert "no campaign" in capsys.readouterr().err


def test_a_fire_while_paused_exits_at_once(bench: Bench, capsys: pytest.CaptureFixture) -> None:
    bench.plan({"final": "clean"})
    assert driver.main(["pause", "--root", str(bench.root)]) == 0
    before = bench.camp.ledger_path.read_bytes()
    code, out, _ = run(bench, capsys)
    assert code == 0 and "stopped: paused" in out and bench.calls() == []
    assert bench.camp.ledger_path.read_bytes() == before
    assert driver.main(["resume", "--root", str(bench.root)]) == 0
    assert not bench.camp.pause_path.exists()
    assert run(bench, capsys)[0] == 0 and len(bench.calls()) == 1


def test_a_fire_while_paused_exits_at_once_whatever_the_settings(
        bench: Bench, capsys: pytest.CaptureFixture) -> None:
    """Pausing to edit the user settings is the natural order: the pause wins, so the fire exits
    0 rather than refusing."""
    assert driver.main(["pause", "--root", str(bench.root)]) == 0
    bench.write_settings({**SETTINGS, "hooks": {}})
    before = bench.camp.ledger_path.read_bytes()
    code, out, _ = run(bench, capsys)
    assert code == 0 and "stopped: paused" in out and bench.calls() == []
    assert not (bench.root / "store" / "refusal").exists()
    assert bench.camp.ledger_path.read_bytes() == before


def test_pause_and_resume_need_a_campaign(tmp_path: Path) -> None:
    for command in ("pause", "resume"):
        assert driver.main([command, "--root", str(tmp_path)]) == 1
    assert not (tmp_path / "store").exists()


def test_the_kill_grace_is_thirty_seconds() -> None:
    assert runner.KILL_GRACE == timedelta(seconds=30)


# --- single flight, pause in flight, pause --now, orphans ----------------------------------------


def stat_start(pid: int) -> int:
    """Field 22 of `/proc/<pid>/stat`, computed here independently of `runner`."""
    fields = Path(f"/proc/{pid}/stat").read_text("ascii").rsplit(")", 1)[1].split()
    return int(fields[22 - 3])


def finish(process: subprocess.Popen, timeout: float = 15) -> tuple[int, str]:
    out, err = process.communicate(timeout=timeout)
    return process.returncode, out + err


def test_a_second_run_against_a_live_runner_exits_0_without_launching(
        bench: Bench, capsys: pytest.CaptureFixture) -> None:
    bench.plan({"wait_for": "release", "final": "clean"})
    first = bench.start()
    try:
        assert bench.call(0)["stdin"] == "/dev/null"     # not the runner's own (a pipe)
        record = json.loads((bench.root / "store" / "runner.json").read_text("utf-8"))
        assert record == {"runnerPid": first.pid, "runnerStartTime": stat_start(first.pid)}
        # Released by a watchdog if a blocking lock ever made the second run wait.
        watchdog = threading.Timer(5, bench.release)
        watchdog.start()
        code, out, _ = run(bench, capsys)
        watchdog.cancel()
        assert code == 0 and out.strip() == runner.LOCK_HELD
        assert len(bench.calls()) == 1
    finally:
        bench.release()
        code, output = finish(first)
    assert code == 0, output
    assert bench.batches() == {"bat_0001": "ingested"}


def test_pause_during_a_batch_lets_it_finish_and_the_run_exits(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    made = Bench(tmp_path, batches=2)
    monkeypatch.setenv("HOME", str(made.home))
    made.plan({"wait_for": "release", "final": "clean"}, {"final": "clean"})
    process = made.start()
    try:
        child = made.call(0)
        assert driver.main(["pause", "--root", str(made.root)]) == 0   # needs no lock
        assert alive(child["pid"]) and made.camp.pause_path.exists()
    finally:
        made.release()
        code, output = finish(process)
    assert code == 0 and "stopped: paused" in output
    assert len(made.calls()) == 1
    assert made.batches() == {"bat_0001": "ingested", "bat_0002": "pending"}
    assert driver.main(["status", "--root", str(made.root)]) == 0
    assert "PAUSED" in capsys.readouterr().out


def test_pause_now_stops_the_runner_and_its_child_and_records_interrupted_once(
        bench: Bench, capsys: pytest.CaptureFixture) -> None:
    bench.plan({"stage": "incomplete", "wait_for": "never", "grandchild": True},
               {"stage": "incomplete"}, {"final": "clean"})
    process = bench.start()
    try:
        child = bench.call(0)
        assert driver.main(["pause", "--root", str(bench.root), "--now"]) == 0
        assert "signalled runner" in capsys.readouterr().out
        code, output = finish(process)
    finally:
        if process.poll() is None:
            process.kill()
    assert code == 0 and "interrupted bat_0001" in output
    wait_until(lambda: not alive(child["pid"]) and not alive(child["grandchild"]), 5)
    assert [e["batch"] for e in bench.events("interrupted")] == ["bat_0001"]
    assert bench.outcomes() == []                       # the next run reconciles it
    session = bench.events("issued")[0]["sessionId"]
    assert exit_file(bench, session) == f"{-signal.SIGTERM}\n"
    assert bench.camp.pause_path.exists()

    # resume: the interrupted attempt is reconciled as incomplete, and a second incomplete for
    # the batch does not halt, because the first is not counted toward the cap.
    assert driver.main(["resume", "--root", str(bench.root)]) == 0
    assert run(bench, capsys)[0] == 0
    assert [(o["outcome"], o.get("reason")) for o in bench.outcomes()] == [
        ("reissued", "incomplete"), ("reissued", "incomplete"), ("ingested", None)]
    assert len(bench.events("interrupted")) == 1
    assert bench.batches() == {"bat_0001": "ingested"}


def ignores_sigterm(pid: int) -> bool:
    mask = next(line for line in Path(f"/proc/{pid}/status").read_text("ascii").splitlines()
                if line.startswith("SigIgn:"))
    return bool(int(mask.split()[1], 16) & (1 << (signal.SIGTERM - 1)))


def test_a_descendant_that_survives_sigterm_is_killed_after_the_grace(
        bench: Bench, capsys: pytest.CaptureFixture) -> None:
    """plan6 §3.2 step 4: the SIGKILL goes to the group whenever a member is left, not only when
    the leader is. Here the leader dies on SIGTERM and its child ignores it while holding the
    inherited lock: a kill that skipped it would leave every later fire on the lock."""
    bench.plan({"stage": "incomplete", "wait_for": "never", "stubborn_grandchild": True},
               {"final": "clean"})
    process = bench.start(kill_grace=timedelta(seconds=0.5))
    grandchild = None
    try:
        child = bench.call(0)
        grandchild = child["grandchild"]
        lock = str(bench.root / "store" / "driver.lock")
        assert ignores_sigterm(grandchild)
        assert lock in [os.readlink(f) for f in Path(f"/proc/{grandchild}/fd").iterdir()]
        signalled = time.monotonic()
        assert driver.main(["pause", "--root", str(bench.root), "--now"]) == 0
        code, output = finish(process)
        elapsed = time.monotonic() - signalled
        assert code == 0 and "interrupted bat_0001" in output
        assert elapsed >= 0.5                                  # the grace, waited out
        wait_until(lambda: not alive(grandchild), 5)
    finally:
        if process.poll() is None:
            process.kill()
        if grandchild and alive(grandchild):
            os.kill(grandchild, signal.SIGKILL)
    assert driver.main(["resume", "--root", str(bench.root)]) == 0
    code, out, _ = run(bench, capsys)
    assert code == 0 and "stopped: done" in out


def test_sigint_stops_the_runner_like_pause_now(bench: Bench) -> None:
    """Ctrl-C on a hand-started `run`: the child has its own session, so the runner kills it."""
    bench.plan({"stage": "incomplete", "wait_for": "never"}, {"final": "clean"})
    process = bench.start()
    try:
        child = bench.call(0)
        process.send_signal(signal.SIGINT)
        code, output = finish(process)
    finally:
        if process.poll() is None:
            process.kill()
    assert code == 0 and "stopped: interrupted bat_0001" in output
    wait_until(lambda: not alive(child["pid"]), 5)
    assert [e["batch"] for e in bench.events("interrupted")] == ["bat_0001"]
    session = bench.events("issued")[0]["sessionId"]
    assert exit_file(bench, session) == f"{-signal.SIGTERM}\n"


def signal_self_after(monkeypatch: pytest.MonkeyPatch, method: str) -> None:
    """`Campaign.<method>` sends SIGTERM to this process once it returns, as `pause --now`
    landing at that moment would; the runner's handler must already be in place."""
    real = getattr(Campaign, method)

    def wrapped(self: Campaign, *args: Any, **kwargs: Any) -> Any:
        result = real(self, *args, **kwargs)
        assert callable(signal.getsignal(signal.SIGTERM))     # never the default action
        os.kill(os.getpid(), signal.SIGTERM)
        return result

    monkeypatch.setattr(Campaign, method, wrapped)


def test_a_signal_between_batches_stops_before_the_next_launch(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    made = Bench(tmp_path, batches=2)
    monkeypatch.setenv("HOME", str(made.home))
    monkeypatch.setenv("PATH", made.path)
    made.plan({"final": "clean"})
    signal_self_after(monkeypatch, "ingest")
    code, out, _ = run(made, capsys)
    assert code == 0 and out.splitlines()[-1] == "stopped: signalled"
    assert len(made.calls()) == 1 and made.events("interrupted") == []
    assert made.batches() == {"bat_0001": "ingested", "bat_0002": "pending"}


def test_a_signal_while_next_runs_launches_nothing(
        bench: Bench, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    bench.plan({"final": "clean"})
    signal_self_after(monkeypatch, "next")
    code, out, _ = run(bench, capsys)
    assert code == 0 and out.splitlines()[-1] == "stopped: signalled"
    assert bench.calls() == [] and bench.events("issued") == []
    assert "launched" not in out


def test_pause_now_survives_a_runner_that_exits_after_the_check(
        bench: Bench, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    """The start time matched, then the runner exited before the signal."""
    gone = subprocess.Popen(["true"])
    gone.wait()
    (bench.root / "store" / "runner.json").write_text(json.dumps(
        {"runnerPid": gone.pid, "runnerStartTime": 7}))
    monkeypatch.setattr(runner, "process_start_time", lambda pid: 7)
    assert driver.main(["pause", "--root", str(bench.root), "--now"]) == 0
    assert "nothing signalled" in capsys.readouterr().out
    assert bench.camp.pause_path.exists()


def test_pause_now_with_no_runner_signals_nothing(bench: Bench,
                                                  capsys: pytest.CaptureFixture) -> None:
    assert driver.main(["pause", "--root", str(bench.root), "--now"]) == 0
    assert "nothing signalled" in capsys.readouterr().out
    assert bench.camp.pause_path.exists()


def test_pause_now_never_signals_a_process_with_another_start_time(
        bench: Bench, capsys: pytest.CaptureFixture) -> None:
    """A recycled PID: alive, but not the runner `runner.json` recorded."""
    stranger = subprocess.Popen(["sleep", "30"])
    try:
        (bench.root / "store" / "runner.json").write_text(json.dumps(
            {"runnerPid": stranger.pid, "runnerStartTime": stat_start(stranger.pid) + 1}))
        assert driver.main(["pause", "--root", str(bench.root), "--now"]) == 0
        assert "nothing signalled" in capsys.readouterr().out
        time.sleep(0.2)
        assert stranger.poll() is None
    finally:
        stranger.kill()
        stranger.wait()


def test_an_orphan_keeps_the_campaign_locked_until_it_exits(
        bench: Bench, capsys: pytest.CaptureFixture) -> None:
    bench.plan({"stage": "incomplete", "wait_for": "release", "final": "clean"})
    process = bench.start()
    try:
        child = bench.call(0)
        process.kill()                                   # the runner dies; its child lives on
        process.communicate(timeout=15)
        assert alive(child["pid"])

        before = bench.camp.ledger_path.read_bytes()
        code, out, _ = run(bench, capsys)
        assert code == 0 and out.strip() == runner.LOCK_HELD
        assert len(bench.calls()) == 1 and bench.camp.ledger_path.read_bytes() == before
        # The dead runner's record names no live runner: the orphan is not signalled.
        assert driver.main(["pause", "--root", str(bench.root), "--now"]) == 0
        assert "nothing signalled" in capsys.readouterr().out
        assert driver.main(["resume", "--root", str(bench.root)]) == 0
        time.sleep(0.2)
        assert alive(child["pid"])
    finally:
        bench.release()
    wait_until(lambda: not alive(child["pid"]), 15)

    code, out, _ = run(bench, capsys)
    assert code == 0 and "stopped: done" in out
    assert len(bench.calls()) == 1
    assert [o["outcome"] for o in bench.outcomes()] == ["ingested"]


# --- timeout -------------------------------------------------------------------------------------


def test_a_child_past_batch_timeout_is_killed_as_a_group_and_the_batch_ingested(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    """The child ignores SIGTERM, so it takes the SIGKILL after the grace, and the run lasts at
    least the timeout plus the grace; its grandchild does not, so only a group kill reaches it.
    The timeout is 3 s, not 1 s, so a child still starting under CPU load is not killed before it
    records its call (the fake ignores SIGTERM as its first statement)."""
    made = Bench(tmp_path, batch_timeout=timedelta(seconds=3))
    monkeypatch.setenv("HOME", str(made.home))
    monkeypatch.setenv("PATH", made.path)
    made.plan({"stage": "incomplete", "grandchild": True, "ignore_term": True, "sleep": 60},
              {"final": "clean"})
    fired = threading.Event()

    def watchdog() -> None:          # unblocks a runner that never kills, and fails the test
        fired.set()
        for call in made.calls():
            for pid in (call["pid"], call.get("grandchild")):
                if pid:
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass

    timer = threading.Timer(20, watchdog)
    timer.start()
    started = time.monotonic()
    try:
        code, out, _ = run(made, capsys, kill_grace=timedelta(seconds=1))
    finally:
        timer.cancel()
    elapsed = time.monotonic() - started
    first = made.calls()[0]
    try:
        assert not fired.is_set() and code == 0 and "timeout" in out
        assert 3 + 1 <= elapsed < 18                    # batchTimeout + the grace
        wait_until(lambda: not alive(first["pid"]) and not alive(first["grandchild"]), 5)
        session = made.events("issued")[0]["sessionId"]
        assert exit_file(made, session) == f"{-signal.SIGKILL}\n"
        assert [(o["outcome"], o.get("reason")) for o in made.outcomes()] == [
            ("reissued", "incomplete"), ("ingested", None)]
    finally:
        for pid in (first["pid"], first["grandchild"]):
            if alive(pid):
                os.kill(pid, signal.SIGKILL)
