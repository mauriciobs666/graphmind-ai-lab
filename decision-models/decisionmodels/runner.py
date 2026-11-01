"""`driver run`, the unattended runner, and the lock-free stop switch `pause [--now]`/`resume`
(plan `docs/plans/typed-decision-models6.md` §3.1, §3.2).

**The loop.** One `Campaign` instance serves the whole process (its usage-stop memory is the
instance's, `campaign.py`'s docstring): `next` reconciles and names the pending batch or stops;
`issue` records `issued {batch, opaque, sessionId}` and writes the request; the child
`claude -p --agent dm-driver` session runs in its own process group under `batchTimeout`, killed
as a group on expiry: SIGTERM to the group, then SIGKILL to the group once `KILL_GRACE` has passed
with any member left, so a descendant that ignores SIGTERM (and holds the inherited lock) never
outlives the run; the wait ends early once the group is empty. Its exit code is written to
`store/runs/<sessionId>.exit` atomically after `wait()` and before `ingest`; then `ingest`. The
loop exits 0 when `next` stops (done, halted, usage stop, `store/pause`), when `ingest` returns
`IngestResult.stop`, and after a `launchFailed` outcome, so a fire into a broken environment costs
one launch, never a hot loop of them (§3.2 "Scheduling": a fire in a spent window costs at most
one short driver turn). `store/pause` is checked at the top of every iteration, before the
settings check, so a fire while paused exits 0 at once whatever the settings hold.

**The child** gets `CHILD_ENV_ALLOWLIST` from the runner's environment plus
`CLAUDE_CODE_DISABLE_BACKGROUND_TASKS=1`, nothing else; `exchange/` as its working directory;
stdin from `/dev/null`; stdout and stderr in `store/runs/<sessionId>.json` and `.err`; and the
lock's descriptor (`pass_fds`), so an orphan whose runner died keeps the campaign locked until it
exits. `--fallback-model` and `--setting-sources` are never passed.

**User settings** (`~/.claude/settings.json`, which applies to the child) are checked before each
launch: an `env` or `hooks` key, `autoContinueAtUsageLimit: true`, or a `permissions`/
`modelSettings` hash other than the one `install` recorded each refuse. The check runs before
`next`, so a refusal appends nothing, not even a reconciliation. Every refusal `run` makes after
taking the lock overwrites `store/refusal` with an ISO 8601 timestamp and its reason (which
`status` prints) and exits 1; a successful launch deletes the file. The root and no-campaign
refusals come before the lock and go to stderr only: `status` refuses a missing campaign the same
way, and reports a root refusal itself, computed live (`Campaign.status`).

**Single flight.** `run` holds `store/driver.lock` (`LOCK_EX | LOCK_NB`) for its whole life and
exits 0 with one line when it is held. On taking it, before any launch, it writes
`store/runner.json`: `{runnerPid, runnerStartTime}`, the start time being field 22 of
`/proc/<pid>/stat`. SIGTERM (`pause --now`) and SIGINT stop it: a running child group is killed as
on a timeout, `interrupted {batch}` is appended once, and `run` exits 0 without ingesting; the
next `run` reconciles the batch. A signal that lands before a launch (between batches, or while
`next` reconciles or prepares) issues nothing: `run` prints `stopped: signalled` and exits 0.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from datetime import timedelta
from pathlib import Path
from typing import Any, TextIO

from .campaign import (
    Campaign,
    CampaignRefused,
    Serve,
    Stop,
    _halt_text,
    _write_atomic,
    root_refusal,
)
from .ingest import AnswerRefused

#: plan6 §3.2: the child's environment, beyond which only the background-task switch is set.
CHILD_ENV_ALLOWLIST = ("PATH", "HOME", "USER", "LANG", "LC_ALL", "TZ", "TMPDIR")
CHILD_ENV_SET = {"CLAUDE_CODE_DISABLE_BACKGROUND_TASKS": "1"}
#: plan6 §3.2 step 4: SIGKILL follows SIGTERM after 30 s.
KILL_GRACE = timedelta(seconds=30)
#: The user-settings keys whose values `install` hashes (plan6 §3.2 "User settings").
HASHED_SETTINGS = ("permissions", "modelSettings")
REFUSED_SETTINGS = ("env", "hooks")
LOCK_HELD = ("another `driver run` holds store/driver.lock (a live run, or an orphaned child); "
             "exiting")


def user_settings_path() -> Path:
    return Path.home() / ".claude" / "settings.json"


def read_user_settings(path: Path) -> dict[str, Any]:
    """The settings object; a missing file is `{}`. Anything that is not a JSON object refuses."""
    if not path.is_file():
        return {}
    try:
        settings = json.loads(path.read_text("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CampaignRefused(f"{path} is not JSON ({exc})") from None
    if not isinstance(settings, dict):
        raise CampaignRefused(f"{path} is not a JSON object")
    return settings


def user_settings_sha256(settings: Mapping[str, Any]) -> str:
    """sha256 over the canonical JSON of the `permissions` and `modelSettings` values present
    (sorted keys, no whitespace): what `install` records and `run` compares."""
    hashed = {key: settings[key] for key in HASHED_SETTINGS if key in settings}
    canonical = json.dumps(hashed, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def settings_refusal(settings: Mapping[str, Any], recorded_sha256: str) -> str | None:
    """Why the user settings refuse a launch (plan6 §3.2), or `None`."""
    for key in REFUSED_SETTINGS:
        if key in settings:
            return f"~/.claude/settings.json has the `{key}` key"
    if settings.get("autoContinueAtUsageLimit") is True:
        return "~/.claude/settings.json sets autoContinueAtUsageLimit: true"
    if user_settings_sha256(settings) != recorded_sha256:
        return ("~/.claude/settings.json's permissions/modelSettings differ from what "
                "`driver install` recorded")
    return None


def child_env(environ: Mapping[str, str]) -> dict[str, str]:
    env = {name: environ[name] for name in CHILD_ENV_ALLOWLIST if name in environ}
    env.update(CHILD_ENV_SET)
    return env


def driver_message(template: str, serve: Serve) -> str:
    return (template.replace("{agent_type}", serve.agent_type)
            .replace("{spawn_prompt}", serve.spawn_prompt))


def child_argv(camp: Campaign, session_id: str, message: str) -> list[str]:
    """plan6 §3.2's command, verbatim."""
    return ["claude", "-p", "--agent", "dm-driver", "--session-id", session_id,
            "--strict-mcp-config", "--mcp-config", str(camp.exchange_dir / "empty-mcp.json"),
            "--permission-mode", "acceptEdits", "--permission-prompts", "none",
            "--output-format", "json", message]


def process_start_time(pid: int) -> int | None:
    """Field 22 of `/proc/<pid>/stat` (clock ticks since boot), or `None` for no such process."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text("ascii")
    except OSError:
        return None
    # Field 2 is `(comm)`, which may hold spaces or parentheses: count from after its last `)`.
    return int(stat.rsplit(")", 1)[1].split()[19])


# --- run -----------------------------------------------------------------------------------------


class Runner:
    """One `driver run` process (module docstring)."""

    def __init__(self, root: Path, *, kill_grace: timedelta = KILL_GRACE, poll: float = 0.05,
                 out: TextIO | None = None) -> None:
        self.camp = Campaign(root)
        self.kill_grace = kill_grace.total_seconds()
        self.poll = poll
        self.out = out if out is not None else sys.stdout
        self.signalled: int | None = None
        self.lock_fd: int | None = None

    def say(self, line: str) -> None:
        print(line, file=self.out, flush=True)

    def run(self) -> int:
        camp = self.camp
        refusal = root_refusal(camp.root)
        if refusal is not None:
            raise CampaignRefused(refusal)
        if not camp.ledger_path.is_file():
            raise CampaignRefused(f"no campaign at {camp.root} (run `driver init`)")
        fd = os.open(camp.store / "driver.lock", os.O_RDWR | os.O_CREAT, 0o644)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                self.say(LOCK_HELD)
                return 0
            self.lock_fd = fd
            previous = {sig: signal.signal(sig, self._on_signal)
                        for sig in (signal.SIGTERM, signal.SIGINT)}
            try:
                pid = os.getpid()
                _write_atomic(camp.store / "runner.json", json.dumps(
                    {"runnerPid": pid, "runnerStartTime": process_start_time(pid)}).encode())
                return self._refusing(self._loop)
            finally:
                for sig, handler in previous.items():
                    signal.signal(sig, handler)
        finally:
            os.close(fd)

    def _refusing(self, body: Callable[[], int]) -> int:
        """Every refusal on a campaign leaves its reason in `store/refusal` for `status`."""
        try:
            return body()
        except (CampaignRefused, AnswerRefused) as exc:
            stamp = self.camp.clock().isoformat()
            _write_atomic(self.camp.store / "refusal", f"{stamp} {exc}\n".encode("utf-8"))
            raise CampaignRefused(str(exc)) from None

    def _on_signal(self, signum: int, _frame: Any) -> None:
        self.signalled = signum

    def _loop(self) -> int:
        camp = self.camp
        while self.signalled is None:
            if camp.pause_path.exists():
                # Before the settings check: a fire while paused exits 0 at once (plan6 §3.2).
                return self._stopped(Stop("paused", "store/pause exists"))
            template, recorded = self._install_record()
            refusal = settings_refusal(read_user_settings(user_settings_path()), recorded)
            if refusal is not None:
                raise CampaignRefused(f"launch refused: {refusal}")
            action = camp.next()
            if isinstance(action, Stop):
                return self._stopped(action)
            if self.signalled is not None:
                break
            stop = self._batch(action, template)
            if stop is not None:
                return self._stopped(stop)
        self.say("stopped: signalled")
        return 0

    def _stopped(self, stop: Stop) -> int:
        self.say(f"stopped: {stop.reason}" + (f" ({stop.detail})" if stop.detail else ""))
        return 0

    def _install_record(self) -> tuple[str, str]:
        expectations = self.camp.state().expectations or {}
        template = expectations.get("driverTemplate")
        recorded = expectations.get("userSettingsSha256")
        if not isinstance(template, str) or not isinstance(recorded, str):
            raise CampaignRefused("`driver install` recorded no driver template or user-settings "
                                  "hash (run `driver install`)")
        return template, recorded

    def _batch(self, serve: Serve, template: str) -> Stop | None:
        """plan6 §3.2 steps 2–5 for one batch; returns why the run stops, or `None`."""
        camp = self.camp
        session = camp.issue(serve)
        exited = self._launch(session, child_argv(camp, session, driver_message(template, serve)))
        if exited is not None:
            returncode, stopped = exited
            _write_atomic(camp.store / "runs" / f"{session}.exit",
                          f"{returncode}\n".encode("ascii"))
            if stopped == "interrupted":
                camp.record_interrupted(serve.batch)
                return Stop(f"interrupted {serve.batch}")
        result = camp.ingest(serve.opaque)
        if result.stop is not None:
            halt = camp.state().halt if result.stop == "halted" else None
            return Stop(result.stop, _halt_text(halt) if halt is not None else
                        (result.outcome or {}).get("reason", ""))
        if result.outcome is not None and result.outcome.get("reason") == "launchFailed":
            return Stop("launchFailed", "the launch produced nothing; the next fire retries")
        return None

    def _launch(self, session: str, argv: list[str]) -> tuple[int, str | None] | None:
        """Start the child and wait for it: its return code and why it was stopped early (if it
        was), or `None` when it never started."""
        runs = self.camp.store / "runs"
        runs.mkdir(parents=True, exist_ok=True)
        with (open(runs / f"{session}.json", "wb") as out,
              open(runs / f"{session}.err", "wb") as err):
            try:
                child = subprocess.Popen(
                    argv, cwd=self.camp.exchange_dir, env=child_env(os.environ),
                    stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                    start_new_session=True, pass_fds=(self.lock_fd,))
            except OSError as exc:
                # No `.exit`: an exit never observed, so the ingest gives `launchFailed` with
                # this text as its stderr evidence.
                err.write(f"driver run: the launch failed: {exc}\n".encode("utf-8"))
                return None
            (self.camp.store / "refusal").unlink(missing_ok=True)
            self.say(f"launched session {session}")
            return self._wait(child)

    def _wait(self, child: subprocess.Popen) -> tuple[int, str | None]:
        """Wait under `batchTimeout`; on expiry or a signal, kill the child's group."""
        timeout = self.camp.state().init["batchTimeout"]
        deadline = time.monotonic() + timeout
        while True:
            try:
                return child.wait(timeout=self.poll), None
            except subprocess.TimeoutExpired:
                pass
            if self.signalled is not None:
                stopped = "interrupted"
            elif time.monotonic() >= deadline:
                stopped = "timeout"
            else:
                continue
            self.say(f"{stopped}: killing the child's process group {child.pid}")
            return self._kill_group(child), stopped

    def _kill_group(self, child: subprocess.Popen) -> int:
        """SIGTERM to the group, then SIGKILL to the group if any member (not only the leader)
        is left once the grace has passed; returns the leader's return code."""
        _killpg(child.pid, signal.SIGTERM)
        deadline = time.monotonic() + self.kill_grace
        while time.monotonic() < deadline:
            child.poll()                 # reap the leader, whose zombie is still a member
            if not _group_alive(child.pid):
                return child.wait()
            time.sleep(self.poll)
        _killpg(child.pid, signal.SIGKILL)
        return child.wait()


def _killpg(pgid: int, sig: int) -> None:
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        pass


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    return True


# --- the stop switch -----------------------------------------------------------------------------


def _campaign(root: Path) -> Campaign:
    camp = Campaign(root)
    if not camp.ledger_path.is_file():
        raise CampaignRefused(f"no campaign at {camp.root} (run `driver init`)")
    return camp


def pause(root: Path, *, now: bool = False) -> list[str]:
    """`driver pause [--now]`: create `store/pause` (no lock, no ledger); with `now`, SIGTERM the
    runner `runner.json` names, only if that PID is alive with the recorded start time."""
    camp = _campaign(root)
    camp.pause_path.touch()
    lines = ["paused: store/pause exists; a running `driver run` stops after its batch"]
    if now:
        lines.append(_signal_runner(camp))
    return lines


def _signal_runner(camp: Campaign) -> str:
    path = camp.store / "runner.json"
    try:
        record = json.loads(path.read_text("utf-8"))
        pid, started = record["runnerPid"], record["runnerStartTime"]
    except (OSError, ValueError, KeyError, TypeError):
        return "no runner recorded in store/runner.json; nothing signalled"
    if not isinstance(pid, int) or pid <= 0 or process_start_time(pid) != started:
        return f"runner {pid} is not running; nothing signalled"
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:       # it exited after the start-time check
        return f"runner {pid} exited; nothing signalled"
    return f"signalled runner {pid}; it kills its child, records `interrupted` and exits"


def resume(root: Path) -> list[str]:
    """`driver resume`: delete `store/pause`."""
    camp = _campaign(root)
    camp.pause_path.unlink(missing_ok=True)
    return ["resumed: store/pause removed; the next `driver run` serves"]


def run(root: Path, *, kill_grace: timedelta = KILL_GRACE, poll: float = 0.05,
        out: TextIO | None = None) -> int:
    return Runner(root, kill_grace=kill_grace, poll=poll, out=out).run()
