"""`driver install` and the agent sources (typed-decision-models U5.7; plan6
`docs/plans/typed-decision-models6.md` §3.2 "User settings", §3.3 and the §5 U5.7 row over plan4
§3.4, §3.5).

Nothing here calls the real `claude` or reads the real `~/.claude`: `install` takes the user
settings path, and the runner cases use `test_runner.Bench` (a fake `HOME`, a fake `claude` on
`PATH`). Expected values are written in the tests, never read from the constants they check.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest
from test_audit import EXPECTED, attachment, audit_of, clean_run, found, insert_before_reply
from test_campaign import ROLES, Clock, write_fixture
from test_runner import Bench

from decisionmodels import audit, driver, install, render, runner, smoke
from decisionmodels.campaign import Campaign, CampaignRefused

pytestmark = pytest.mark.usefixtures("fake_claude")
AGENTS = Path(__file__).resolve().parents[1] / "agents"
SETTINGS = {"model": "opus", "theme": "dark", "permissions": {"allow": ["Read"]},
            "outputStyle": "default"}
GUARD_COMMAND = 'python3 "$CLAUDE_PROJECT_DIR/.claude/guard_hook.py"'
INSTALLED = ("agents/dm-driver.md", "agents/dm-generator.md", "agents/dm-labeler.md",
             "spawn-generator.txt", "spawn-labeler.txt", "guard_hook.py", "settings.json")
SOURCE_OF = {"agents/dm-driver.md": "dm-driver.md", "agents/dm-generator.md": "dm-generator.md",
             "agents/dm-labeler.md": "dm-labeler.md"}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass
class Installed:
    camp: Campaign
    home: Path
    lines: list[str]

    @property
    def claude(self) -> Path:
        return self.camp.exchange_dir / ".claude"

    @property
    def settings(self) -> Path:
        return self.home / ".claude" / "settings.json"

    def install(self, **kwargs: Any) -> list[str]:
        return install.install(self.camp.root, settings_path=self.settings, **kwargs)

    def expectations(self) -> list[dict[str, Any]]:
        return [e for e in self.camp.events() if e["event"] == "expectations"]


def fresh_campaign(tmp_path: Path, settings: dict[str, Any] | None = None) -> Installed:
    home = tmp_path / "home"
    (home / ".claude").mkdir(parents=True)
    (home / ".claude" / "settings.json").write_text(json.dumps(SETTINGS if settings is None
                                                               else settings), "utf-8")
    camp = Campaign(tmp_path / "camp", claude_dir=home / ".claude", clock=Clock())
    camp.init(point="guard-judge", phase="smoke",
              fixture=write_fixture(tmp_path / "fixture.jsonl", 2))
    return Installed(camp, home, [])


def installed(tmp_path: Path, settings: dict[str, Any] | None = None) -> Installed:
    made = fresh_campaign(tmp_path, settings)
    made.lines = made.install()
    return made


def agents_copy(tmp_path: Path, **edits: Callable[[str], str]) -> Path:
    """A copy of `agents/` with each named file's text passed through its function."""
    copy = tmp_path / "agents-copy"
    shutil.copytree(AGENTS, copy)
    for name, edit in edits.items():
        path = copy / name
        path.write_text(edit(path.read_text("utf-8")), "utf-8")
    return copy


def refuses(made: Installed, match: str, **kwargs: Any) -> None:
    before = made.camp.ledger_path.read_bytes()
    with pytest.raises(CampaignRefused, match=match):
        made.install(**kwargs)
    assert made.camp.ledger_path.read_bytes() == before


# --- the sources ---------------------------------------------------------------------------------


def front(name: str) -> dict[str, str]:
    return install.frontmatter((AGENTS / name).read_text("utf-8"))


def test_dm_driver_pins_sonnet_has_no_bash_and_three_turns() -> None:
    fields = front("dm-driver.md")
    assert fields["model"] == "claude-sonnet-5-5"
    assert fields["tools"] == "Agent(dm-generator, dm-labeler)"
    assert "Bash" not in fields["tools"]
    assert fields["maxTurns"] == "3"
    assert "initialPrompt" not in fields and "effort" not in fields


@pytest.mark.parametrize(("name", "model", "effort", "turns"), [
    ("dm-generator.md", "claude-opus-5-5", "medium", "6"),
    ("dm-labeler.md", "claude-sonnet-5-5", "high", "14"),
])
def test_the_subagents_pin_model_effort_tools_and_turns(
        name: str, model: str, effort: str, turns: str) -> None:
    fields = front(name)
    assert (fields["model"], fields["effort"], fields["maxTurns"]) == (model, effort, turns)
    assert fields["tools"] == "Read, Write"
    assert (fields["omitClaudeMd"], fields["permissionMode"]) == ("true", "acceptEdits")


@pytest.mark.parametrize("name", ["dm-driver.md", "dm-generator.md", "dm-labeler.md"])
def test_no_definition_carries_hooks_or_an_mcp_tool(name: str) -> None:
    fields = front(name)
    assert "hooks" not in fields
    assert "mcp__" not in fields["tools"]
    assert fields["name"] == name.removesuffix(".md")


def test_the_source_settings_wire_the_guard_once_and_pin_the_usage_limit() -> None:
    settings = json.loads((AGENTS / "settings.json").read_text("utf-8"))
    assert settings["autoContinueAtUsageLimit"] is False
    assert settings["hooks"]["PreToolUse"] == [{"matcher": "*", "hooks": [
        {"type": "command", "command": GUARD_COMMAND, "timeout": 30, "onFailure": "block"}]}]
    assert set(settings) == {"autoContinueAtUsageLimit", "hooks"}


@pytest.mark.parametrize("name", ["spawn-generator.txt", "spawn-labeler.txt", "spawn-driver.txt"])
def test_a_template_has_no_trailing_newline(name: str) -> None:
    assert not (AGENTS / name).read_bytes().endswith(b"\n")


def test_the_spawn_template_is_plan4s_text() -> None:
    expected = ("Read {request_path} with offset 1, limit 1. Its `pages` field lists "
                "[offset, limit] pairs: read each remaining page with exactly that offset and "
                "limit. Follow the instructions on line 2, write the answer where the header "
                "says, then reply `done`.")
    for name in ("spawn-generator.txt", "spawn-labeler.txt"):
        assert (AGENTS / name).read_text("utf-8") == expected


# --- guard_command -------------------------------------------------------------------------------


def wiring(**edits: Any) -> dict[str, Any]:
    handler = {"type": "command", "command": "python3 g.py", "onFailure": "block"}
    handler.update(edits.pop("handler", {}))
    settings: dict[str, Any] = {"hooks": {"PreToolUse": [{"matcher": "*", "hooks": [handler]}]}}
    settings.update(edits)
    return settings


def test_guard_command_is_the_exact_command_string() -> None:
    assert install.guard_command(wiring()) == "python3 g.py"
    assert install.guard_command(json.dumps(wiring())) == "python3 g.py"


@pytest.mark.parametrize("settings", [
    wiring(handler={"onFailure": "continue"}),
    wiring(handler={"onFailure": None}),
    {"hooks": {"PreToolUse": [{"matcher": "*", "hooks": [
        {"type": "command", "command": "python3 g.py"}]}]}},
    wiring(handler={"type": "http"}),
    wiring(handler={"command": ""}),
    {"hooks": {"PreToolUse": [{"matcher": "Write", "hooks": [
        {"type": "command", "command": "g", "onFailure": "block"}]}]}},
    {"hooks": {"PreToolUse": [{"matcher": "*", "hooks": [
        {"type": "command", "command": "g", "onFailure": "block"},
        {"type": "command", "command": "h", "onFailure": "block"}]}]}},
    {"hooks": {"PreToolUse": [
        {"matcher": "*", "hooks": [{"type": "command", "command": "g", "onFailure": "block"}]},
        {"matcher": "*", "hooks": [{"type": "command", "command": "g", "onFailure": "block"}]}]}},
    {"hooks": {**wiring()["hooks"], "Stop": []}},
    {"hooks": {}},
    {},
    [],
])
def test_guard_command_refuses_any_wiring_but_the_one_blocking_guard(settings: Any) -> None:
    with pytest.raises(CampaignRefused):
        install.guard_command(settings)


def test_guard_command_refuses_text_that_is_not_json() -> None:
    with pytest.raises(CampaignRefused, match="not JSON"):
        install.guard_command("{")


# --- a first install -----------------------------------------------------------------------------


def test_install_byte_copies_the_sources_and_the_hashes_equal_the_sources(tmp_path: Path) -> None:
    made = installed(tmp_path)
    for name in INSTALLED:
        source = AGENTS / SOURCE_OF.get(name, name)
        assert (made.claude / name).read_bytes() == source.read_bytes()
    (event,) = made.expectations()
    assert event["promptSha256"] == {
        "agentDefinition": {
            "driver": digest((AGENTS / "dm-driver.md").read_bytes()),
            "generator": digest((AGENTS / "dm-generator.md").read_bytes()),
            "labeler": digest((AGENTS / "dm-labeler.md").read_bytes())},
        "spawnTemplate": {
            "driver": digest((AGENTS / "spawn-driver.txt").read_bytes()),
            "generator": digest((AGENTS / "spawn-generator.txt").read_bytes()),
            "labeler": digest((AGENTS / "spawn-labeler.txt").read_bytes())},
        "guardHook": digest((AGENTS / "guard_hook.py").read_bytes())}
    assert event["settingsSha256"] == digest((AGENTS / "settings.json").read_bytes())


def test_the_driver_template_is_stored_and_its_hash_is_of_what_is_stored(tmp_path: Path) -> None:
    (event,) = installed(tmp_path).expectations()
    stored = event["driverTemplate"]
    assert stored == (AGENTS / "spawn-driver.txt").read_text("utf-8")
    assert "{agent_type}" in stored and "{spawn_prompt}" in stored
    assert event["promptSha256"]["spawnTemplate"]["driver"] == digest(stored.encode("utf-8"))


def test_the_expectations_declare_the_pins_templates_and_an_empty_instructions_set(
        tmp_path: Path) -> None:
    (event,) = installed(tmp_path).expectations()
    template = (AGENTS / "spawn-labeler.txt").read_text("utf-8")
    assert event["guardCommand"] == GUARD_COMMAND
    assert event["roles"] == {
        "generator": {"agentType": "dm-generator", "model": "claude-opus-5-5",
                      "effort": "medium", "spawnTemplate": template, "instructions": []},
        "labeler": {"agentType": "dm-labeler", "model": "claude-sonnet-5-5",
                    "effort": "high", "spawnTemplate": template, "instructions": []}}


def test_the_installed_settings_pin_autocontinue_off(tmp_path: Path) -> None:
    made = installed(tmp_path)
    assert json.loads((made.claude / "settings.json").read_text("utf-8")
                      )["autoContinueAtUsageLimit"] is False


def test_the_recorded_digests_equal_the_runners_of_the_planted_file(tmp_path: Path) -> None:
    made = installed(tmp_path)
    (event,) = made.expectations()
    assert event["userSettingsDigests"] == runner.user_settings_digests(SETTINGS)
    assert set(event["userSettingsDigests"]) == {"outputStyle"}      # model, theme, allow: exempt


def test_the_output_names_the_recorded_keys_and_shows_no_value(tmp_path: Path) -> None:
    marker = "marker-value-7731"
    made = installed(tmp_path, {**SETTINGS, "outputStyle": marker, "futureKey": [marker]})
    text = "\n".join(made.lines)
    assert "recorded user-settings keys: `futureKey`, `outputStyle`" in text
    assert marker not in text
    assert GUARD_COMMAND in text


def test_no_recorded_keys_prints_none(tmp_path: Path) -> None:
    assert "recorded user-settings keys: (none)" in "\n".join(
        installed(tmp_path, {"theme": "dark"}).lines)


def test_a_missing_user_settings_file_records_nothing(tmp_path: Path) -> None:
    made = fresh_campaign(tmp_path)
    made.settings.unlink()
    made.install()
    assert made.expectations()[0]["userSettingsDigests"] == {}


def test_install_leaves_the_exchange_holding_only_the_dot_claude_additions(tmp_path: Path) -> None:
    made = installed(tmp_path)
    names = sorted(p.name for p in made.camp.exchange_dir.iterdir())
    assert names == [".claude", "ans", "empty-mcp.json", "req"]
    assert sorted(p.name for p in (made.claude / "agents").iterdir()) == [
        "dm-driver.md", "dm-generator.md", "dm-labeler.md"]


# --- forbidden user settings ---------------------------------------------------------------------


@pytest.mark.parametrize("key", sorted(runner.FORBIDDEN_SETTINGS))
def test_each_forbidden_key_refuses_install_and_appends_nothing(tmp_path: Path, key: str) -> None:
    value = {"disableAllHooks": True, "autoContinueAtUsageLimit": True}.get(key, "x")
    made = fresh_campaign(tmp_path, {**SETTINGS, key: value})
    refuses(made, f"forbidden `{key}` setting.*remove it")
    assert not (made.camp.exchange_dir / ".claude").exists()


def test_the_six_forbidden_keys_are_the_plans() -> None:
    assert set(runner.FORBIDDEN_SETTINGS) == {"env", "hooks", "disableAllHooks",
                                              "autoContinueAtUsageLimit", "fallbackModel",
                                              "apiKeyHelper"}


@pytest.mark.parametrize("settings", [
    {"disableAllHooks": False}, {"autoContinueAtUsageLimit": False}, {"model": "opus"}])
def test_the_safe_values_of_a_forbidden_class_key_install(tmp_path: Path,
                                                          settings: dict[str, Any]) -> None:
    assert len(installed(tmp_path, settings).expectations()) == 1


def test_install_applies_the_runners_forbidden_check_not_its_own(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    made = fresh_campaign(tmp_path)
    monkeypatch.setattr(runner, "forbidden_settings_refusal", lambda settings: "planted refusal")
    refuses(made, "planted refusal")


def test_user_settings_that_are_not_json_refuse_install(tmp_path: Path) -> None:
    made = fresh_campaign(tmp_path)
    made.settings.write_text("{", "utf-8")
    refuses(made, "not JSON")


# --- re-running install --------------------------------------------------------------------------


def test_rerun_after_a_refused_edit_appends_a_second_event_and_the_next_run_launches(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bench = Bench(tmp_path, settings=SETTINGS)
    monkeypatch.setenv("HOME", str(bench.home))
    monkeypatch.setenv("PATH", bench.path)
    made = Installed(bench.camp, bench.home, [])
    first = made.expectations()[0]          # the bench's own record, before any install
    edited = {**SETTINGS, "outputStyle": "explanatory"}
    bench.write_settings(edited)
    with pytest.raises(CampaignRefused, match="outputStyle.*run `driver install` to accept it"):
        runner.run(bench.root)
    assert bench.calls() == []
    ledger = bench.camp.ledger_path.read_bytes()
    made.install()
    events = made.expectations()
    assert len(events) == 2 and "promptSha256" in events[-1] and "promptSha256" not in first
    assert events[-1]["userSettingsDigests"] == runner.user_settings_digests(edited)
    assert events[-1]["userSettingsDigests"] != first["userSettingsDigests"]
    assert bench.camp.ledger_path.read_bytes().startswith(ledger)
    bench.plan({"stage": "clean"})
    assert runner.run(bench.root) == 0
    (call,) = bench.calls()
    message = call["argv"][-1]
    assert message.startswith("Make exactly one Agent call with subagent_type dm-labeler")
    assert "PROMPT:\nRead " in message


def test_rerun_with_nothing_changed_appends_an_equal_fresh_event(tmp_path: Path) -> None:
    made = installed(tmp_path)
    made.install()
    first, second = made.expectations()
    strip = {k: v for k, v in second.items() if k != "at"}
    assert strip == {k: v for k, v in first.items() if k != "at"} and second["at"] != first["at"]


# --- the version floor --------------------------------------------------------------------------


@pytest.mark.parametrize("text", ["2.1.294 (Claude Code)\n", "2.0.999\n", "1.9.9999\n",
                                  "no version here\n", "", "2.1\n", "v2.1\n"])
def test_a_claude_below_the_floor_or_with_no_version_refuses_install_before_any_write(
        tmp_path: Path, fake_claude: Callable[[str | None], None], text: str) -> None:
    made = fresh_campaign(tmp_path)
    fake_claude(text)
    refuses(made, "older than 2.1.295|gave no version")
    assert not (made.camp.exchange_dir / ".claude").exists()


def test_a_claude_that_exits_non_zero_refuses_install_whatever_it_prints(
        tmp_path: Path, fake_claude: Callable[..., None]) -> None:
    made = fresh_campaign(tmp_path)
    fake_claude("2.1.316\n", 1)
    refuses(made, "gave no version \\(exit 1\\)")


def test_a_missing_claude_binary_refuses_install(
        tmp_path: Path, fake_claude: Callable[[str | None], None]) -> None:
    made = fresh_campaign(tmp_path)
    fake_claude(None)
    refuses(made, "could not run")
    assert not (made.camp.exchange_dir / ".claude").exists()


@pytest.mark.parametrize("text", ["2.1.295 (Claude Code)\n", "2.1.316\n", "2.1.1000\n",
                                  "2.10.0\n", "10.0.0\n", "Claude Code 3.0.1 build 7\n"])
def test_a_claude_at_or_above_the_floor_installs_as_an_integer_tuple(
        tmp_path: Path, fake_claude: Callable[[str | None], None], text: str) -> None:
    made = fresh_campaign(tmp_path)
    fake_claude(text)
    made.install()
    assert len(made.expectations()) == 1


def test_the_version_is_read_from_stderr_too(tmp_path: Path) -> None:
    binary = Path(__import__("shutil").which("claude"))
    binary.write_text("#!/bin/sh\necho '2.1.316 (Claude Code)' >&2\n", "utf-8")
    assert runner.claude_version_refusal(__import__("os").environ) is None
    binary.write_text("#!/bin/sh\necho '2.1.294' >&2\n", "utf-8")
    assert "older than 2.1.295" in runner.claude_version_refusal(__import__("os").environ)


def test_a_hung_claude_version_is_a_refusal_not_a_raised_timeout(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    binary = Path(__import__("shutil").which("claude"))
    binary.write_text("#!/bin/sh\nsleep 30\n", "utf-8")
    real = runner.subprocess.run
    monkeypatch.setattr(runner.subprocess, "run",
                        lambda argv, **kw: real(argv, **{**kw, "timeout": 0.3}))
    message = runner.claude_version_refusal(__import__("os").environ)
    assert message is not None and "could not run (TimeoutExpired)" in message
    made = fresh_campaign(tmp_path)
    refuses(made, "TimeoutExpired")


def test_the_floor_is_2_1_295() -> None:
    assert runner.MIN_CLAUDE_VERSION == (2, 1, 295)


def test_the_version_check_runs_under_the_child_environment(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        fake_claude: Callable[[str | None], None]) -> None:
    seen: dict[str, str] = {}
    real = runner.subprocess.run

    def spy(argv: Any, **kwargs: Any) -> Any:
        seen.update(kwargs["env"])
        return real(argv, **kwargs)
    monkeypatch.setattr(runner.subprocess, "run", spy)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setenv("CLAUDE_CODE_EFFORT_LEVEL", "low")
    assert runner.claude_version_refusal(__import__("os").environ) is None
    assert "ANTHROPIC_API_KEY" not in seen and "CLAUDE_CODE_EFFORT_LEVEL" not in seen
    assert seen["CLAUDE_CODE_DISABLE_BACKGROUND_TASKS"] == "1" and "PATH" in seen


# --- a repin survives a re-run ------------------------------------------------------------------


def test_a_repinned_model_and_instructions_set_survive_a_reinstall(tmp_path: Path) -> None:
    from test_campaign import F_AGENT, F_SESSION, outcomes, run_batch
    made = installed(tmp_path)
    camp = made.camp
    run_batch(camp, "refusal")                                    # a halt: safeguardStop
    camp.resolve(decision="move labelling", repin_model=("labeler", "claude-haiku-9"))
    action, session = run_batch(camp, "refusal", ingest=False)
    agent = camp.transcript_path(session).with_suffix("") / "subagents" / f"agent-{F_AGENT}.jsonl"
    lines = agent.read_text("utf-8").splitlines()
    planted = attachment("instructions", files=[
        {"path": "/campaign/CLAUDE.md", "type": "Project", "content": "Be brief."}])
    lines[-2:-2] = [json.dumps(planted).replace("5e551011-0000-4000-8000-0000000000d1", session)]
    agent.write_text("\n".join(lines) + "\n", "utf-8")
    assert F_SESSION != session
    camp.ingest(action.opaque)
    assert outcomes(camp)[-1]["outcome"] == "halted"
    camp.resolve(decision="declare it", repin_instructions=True)
    pair = [["/campaign/CLAUDE.md", digest(b"Be brief.")]]
    assert camp.state().instructions["labeler"] == pair
    assert camp.state().pins["labeler"] == "claude-haiku-9"
    made.install()
    state = camp.state()
    assert len(made.expectations()) == 2
    assert (state.pins["labeler"], state.instructions["labeler"]) == ("claude-haiku-9", pair)
    assert state.pins["generator"] == "claude-opus-5-5"          # an unrepinned role is unchanged


# --- the hash checks -----------------------------------------------------------------------------


@pytest.mark.parametrize("name", INSTALLED)
def test_a_one_byte_edit_to_an_installed_file_refuses_install(tmp_path: Path, name: str) -> None:
    made = installed(tmp_path)
    path = made.claude / name
    path.write_bytes(path.read_bytes() + b" ")
    refuses(made, "differs from its source")
    path.write_bytes(path.read_bytes()[:-1])
    made.install()
    assert len(made.expectations()) == 2


def test_a_one_byte_change_that_keeps_the_length_also_refuses(tmp_path: Path) -> None:
    made = installed(tmp_path)
    path = made.claude / "guard_hook.py"
    data = bytearray(path.read_bytes())
    data[data.index(b"deny")] = ord("D")
    path.write_bytes(bytes(data))
    refuses(made, "differs from its source")


def test_an_installed_settings_without_onfailure_block_refuses_install(tmp_path: Path) -> None:
    made = installed(tmp_path)
    settings = json.loads((made.claude / "settings.json").read_text("utf-8"))
    del settings["hooks"]["PreToolUse"][0]["hooks"][0]["onFailure"]
    (made.claude / "settings.json").write_text(json.dumps(settings), "utf-8")
    refuses(made, "differs from its source")


def test_a_source_changed_since_the_first_install_refuses_even_with_no_copy_left(
        tmp_path: Path) -> None:
    made = installed(tmp_path)
    shutil.rmtree(made.claude)
    copy = agents_copy(tmp_path, **{"dm-labeler.md": lambda text: text.replace("high", "low")})
    refuses(made, "no longer hash to what this campaign recorded", agents_dir=copy)
    assert not made.claude.exists()


def test_a_stray_file_among_the_installed_definitions_refuses(tmp_path: Path) -> None:
    made = installed(tmp_path)
    (made.claude / "agents" / "other.md").write_text("---\nname: other\n---\n", "utf-8")
    refuses(made, "not agent definitions")


def test_a_local_settings_file_refuses(tmp_path: Path) -> None:
    made = installed(tmp_path)
    (made.claude / "settings.local.json").write_text("{}", "utf-8")
    refuses(made, "second settings channel")


def test_a_copy_that_does_not_hash_to_its_source_refuses_the_install(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    made = fresh_campaign(tmp_path)
    monkeypatch.setattr(install, "_write_atomic", lambda path, data: (
        path.parent.mkdir(parents=True, exist_ok=True), path.write_bytes(data + b"x")))
    refuses(made, "does not hash to its source after the copy")


def test_a_symlinked_installed_file_refuses(tmp_path: Path) -> None:
    made = installed(tmp_path)
    target = tmp_path / "elsewhere.py"
    target.write_bytes((AGENTS / "guard_hook.py").read_bytes())
    (made.claude / "guard_hook.py").unlink()
    (made.claude / "guard_hook.py").symlink_to(target)
    refuses(made, "not a regular file")


# --- refusals of the sources ---------------------------------------------------------------------


def test_a_source_settings_without_onfailure_block_is_refused_before_anything_is_written(
        tmp_path: Path) -> None:
    made = fresh_campaign(tmp_path)
    copy = agents_copy(tmp_path, **{"settings.json": lambda t: t.replace('"onFailure"', '"onX"')})
    refuses(made, "onFailure", agents_dir=copy)
    assert not (made.camp.exchange_dir / ".claude").exists()


def test_a_source_settings_whose_command_does_not_run_the_guard_is_refused(tmp_path: Path) -> None:
    made = fresh_campaign(tmp_path)
    copy = agents_copy(tmp_path, **{"settings.json": lambda t: t.replace("guard_hook.py",
                                                                         "other.py")})
    refuses(made, "does not run guard_hook.py", agents_dir=copy)


def test_a_source_settings_without_the_usage_limit_pin_is_refused(tmp_path: Path) -> None:
    made = fresh_campaign(tmp_path)
    copy = agents_copy(tmp_path, **{"settings.json": lambda t: t.replace("false", "true", 1)})
    refuses(made, "autoContinueAtUsageLimit", agents_dir=copy)


def test_a_definition_carrying_hooks_is_refused(tmp_path: Path) -> None:
    made = fresh_campaign(tmp_path)
    copy = agents_copy(tmp_path, **{"dm-labeler.md": lambda t: t.replace(
        "tools: Read, Write", "tools: Read, Write\nhooks: x")})
    refuses(made, "carries `hooks`", agents_dir=copy)


def test_a_template_with_a_trailing_newline_is_refused(tmp_path: Path) -> None:
    made = fresh_campaign(tmp_path)
    copy = agents_copy(tmp_path, **{"spawn-labeler.txt": lambda t: t + "\n"})
    refuses(made, "no trailing newline", agents_dir=copy)


def swap(old: str, new: str) -> Callable[[str], str]:
    def edit(text: str) -> str:
        assert old in text, old
        return text.replace(old, new, 1)
    return edit


def drop_key(key: str) -> Callable[[str], str]:
    return lambda text: "\n".join(line for line in text.split("\n")
                                  if not line.startswith(f"{key}:"))


DEFINITION_PLANTS = [
    ("misspelt omitClaudeMd", "dm-labeler.md", swap("omitClaudeMd", "omitClaudeMD"),
     "omitClaudeMD"),
    ("misspelt maxTurns", "dm-generator.md", swap("maxTurns", "maxTurn"), "maxTurn"),
    ("misspelt maxTurns on the driver", "dm-driver.md", swap("maxTurns", "maxturns"),
     "maxturns"),
    ("missing model", "dm-labeler.md", drop_key("model"), "model"),
    ("missing effort", "dm-generator.md", drop_key("effort"), "effort"),
    ("empty effort", "dm-labeler.md", swap("effort: high", "effort:"), "effort"),
    ("missing omitClaudeMd", "dm-generator.md", drop_key("omitClaudeMd"), "omitClaudeMd"),
    ("missing permissionMode", "dm-labeler.md", drop_key("permissionMode"), "permissionMode"),
    ("driver with Bash", "dm-driver.md", swap("Agent(dm-generator, dm-labeler)",
                                               "Agent(dm-generator, dm-labeler), Bash"), "tools"),
    ("driver model off the pin", "dm-driver.md", swap("claude-sonnet-5-5", "claude-opus-5-5"),
     "model"),
    ("driver maxTurns off the pin", "dm-driver.md", swap("maxTurns: 3", "maxTurns: 4"),
     "maxTurns"),
    ("driver with an effort key", "dm-driver.md", swap("maxTurns: 3", "maxTurns: 3\neffort: low"),
     "effort"),
    ("driver missing description", "dm-driver.md", drop_key("description"), "description"),
    ("subagent tools Read only", "dm-labeler.md", swap("tools: Read, Write", "tools: Read"),
     "tools"),
    ("subagent tools with Bash", "dm-generator.md", swap("tools: Read, Write",
                                                         "tools: Read, Write, Bash"), "tools"),
    ("omitClaudeMd false", "dm-labeler.md", swap("omitClaudeMd: true", "omitClaudeMd: false"),
     "omitClaudeMd"),
    ("permissionMode default", "dm-generator.md", swap("permissionMode: acceptEdits",
                                                       "permissionMode: default"),
     "permissionMode"),
    ("maxTurns not an integer", "dm-labeler.md", swap("maxTurns: 14", "maxTurns: many"),
     "maxTurns"),
    ("maxTurns zero", "dm-labeler.md", swap("maxTurns: 14", "maxTurns: 0"), "maxTurns"),
    ("a hooks key on the labeler", "dm-labeler.md", swap("tools: Read, Write",
                                                         "tools: Read, Write\nhooks: x"), "hooks"),
    ("a hooks key on the driver", "dm-driver.md", swap("maxTurns: 3", "maxTurns: 3\nhooks: x"),
     "hooks"),
    ("maxTurns repeated, 14 then 2", "dm-labeler.md", swap("maxTurns: 14",
                                                           "maxTurns: 14\nmaxTurns: 2"),
     "repeats the frontmatter key `maxTurns`"),
    ("model repeated on the driver", "dm-driver.md", swap("maxTurns: 3",
                                                          "maxTurns: 3\nmodel: claude-sonnet-5-5"),
     "repeats the frontmatter key `model`"),
    ("maxTurns repeated with a space before the colon", "dm-labeler.md",
     swap("maxTurns: 14", "maxTurns: 14\nmaxTurns : 2"), "repeats the frontmatter key `maxTurns`"),
    ("labeler named dm-labelerx", "dm-labeler.md", swap("name: dm-labeler", "name: dm-labelerx"),
     "`name` must be `dm-labeler`"),
    ("generator named as the labeler", "dm-generator.md", swap("name: dm-generator",
                                                               "name: dm-labeler"),
     "`name` must be `dm-generator`"),
    ("driver named dm-generator", "dm-driver.md", swap("name: dm-driver", "name: dm-generator"),
     "`name` must be `dm-driver`"),
    ("an unknown key", "dm-generator.md", swap("maxTurns: 6", "maxTurns: 6\ncolor: red"), "color"),
]


@pytest.mark.parametrize(("why", "name", "edit", "mention"), DEFINITION_PLANTS,
                         ids=[plant[0] for plant in DEFINITION_PLANTS])
def test_a_planted_definition_is_refused_before_anything_is_written(
        tmp_path: Path, why: str, name: str, edit: Callable[[str], str], mention: str) -> None:
    made = fresh_campaign(tmp_path)
    copy = agents_copy(tmp_path, **{name: edit})
    refuses(made, f"{name}.*{mention}", agents_dir=copy)
    assert not (made.camp.exchange_dir / ".claude").exists()


def test_the_shipped_definitions_pass_the_check() -> None:
    for role, name in install.DEFINITIONS.items():
        assert install.definition_error(role, front(name)) is None


def test_a_missing_source_is_refused(tmp_path: Path) -> None:
    made = fresh_campaign(tmp_path)
    copy = agents_copy(tmp_path)
    (copy / "dm-driver.md").unlink()
    refuses(made, "is missing", agents_dir=copy)


def test_a_smoke_copy_with_two_turns_installs_and_records_its_own_hash(tmp_path: Path) -> None:
    made = fresh_campaign(tmp_path)
    copy = agents_copy(tmp_path, **{"dm-labeler.md": lambda t: t.replace("maxTurns: 14",
                                                                         "maxTurns: 2")})
    made.install(agents_dir=copy)
    assert "maxTurns: 2" in (made.claude / "agents" / "dm-labeler.md").read_text("utf-8")
    (event,) = made.expectations()
    assert event["promptSha256"]["agentDefinition"]["labeler"] == digest(
        (copy / "dm-labeler.md").read_bytes())
    assert event["promptSha256"]["agentDefinition"]["labeler"] != digest(
        (AGENTS / "dm-labeler.md").read_bytes())


# --- the lock and the campaign -------------------------------------------------------------------


def test_install_refuses_while_the_lock_is_held_and_appends_nothing(tmp_path: Path) -> None:
    made = fresh_campaign(tmp_path)
    before = made.camp.ledger_path.read_bytes()
    with made.camp.locked():
        with pytest.raises(CampaignRefused, match="locked by a running"):
            made.install()
    assert made.camp.ledger_path.read_bytes() == before
    assert not (made.camp.exchange_dir / ".claude").exists()


def test_install_without_a_campaign_is_refused(tmp_path: Path) -> None:
    with pytest.raises(CampaignRefused, match="no campaign"):
        install.install(tmp_path / "none", settings_path=tmp_path / "s.json")
    assert not (tmp_path / "none").exists()


# --- the command line ----------------------------------------------------------------------------


def test_the_install_command_prints_and_exits_0(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                capsys: pytest.CaptureFixture) -> None:
    made = fresh_campaign(tmp_path)
    monkeypatch.setenv("HOME", str(made.home))
    assert driver.main(["install", "--root", str(made.camp.root)]) == 0
    out = capsys.readouterr().out
    assert "recorded user-settings keys: `outputStyle`" in out and "guard command:" in out
    assert len(made.expectations()) == 1


def test_a_refused_install_exits_1_with_one_line_on_stderr(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    made = fresh_campaign(tmp_path, {"hooks": {}})
    monkeypatch.setenv("HOME", str(made.home))
    assert driver.main(["install", "--root", str(made.camp.root)]) == 1
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err.startswith("refused: ")
    assert captured.err.count("\n") == 1 and "`hooks`" in captured.err


# --- declared expectations, each against a planted-red transcript --------------------------------


def declared(made: Installed) -> audit.Expected:
    """The audit's expectations as the install's ledger event gives them."""
    state = made.camp.state()
    spec = state.expectations["roles"]["labeler"]
    return replace(EXPECTED, agent_type=spec["agentType"], model=state.pins["labeler"],
                   effort=spec["effort"], spawn_template=spec["spawnTemplate"],
                   guard_command=state.expectations["guardCommand"],
                   instructions=frozenset(tuple(p) for p in state.instructions["labeler"]))


def with_installed_guard(run: Any, command: str) -> None:
    """Re-key the fixture's guard entries to the installed command string."""
    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("command") == EXPECTED.guard_command:
                node["command"] = command
            for child in node.values():
                walk(child)
        elif isinstance(node, list):
            for child in node:
                walk(child)
    walk(run.driver)
    walk(run.agent)


def planted(tmp_path: Path, plant: Callable[[Any], None]) -> list[tuple[str, str]] | str:
    made = installed(tmp_path)
    run = clean_run()
    with_installed_guard(run, made.camp.state().expectations["guardCommand"])
    plant(run)
    result = audit_of(tmp_path / "audit", run, expected=declared(made))
    return result.action if result.action == audit.CLEAN else found(result)


def test_the_installed_expectations_pass_a_clean_run(tmp_path: Path) -> None:
    """The positive control: every red below is the plant, not the install's values."""
    assert planted(tmp_path, lambda run: None) == audit.CLEAN


def test_an_instructions_file_breaks_the_declared_empty_set(tmp_path: Path) -> None:
    def plant(run: Any) -> None:
        insert_before_reply(run, attachment("instructions", files=[
            {"path": "/campaign/CLAUDE.md", "type": "Project", "content": "x"}]))
    assert planted(tmp_path, plant) == [(audit.ENVIRONMENTAL, "instructionsMismatch")]


def test_a_git_status_key_breaks_the_declared_expectation(tmp_path: Path) -> None:
    def plant(run: Any) -> None:
        run.agent[1]["attachment"]["context"]["gitStatus"] = "Current branch: main"
    assert planted(tmp_path, plant) == [(audit.ENVIRONMENTAL, "gitStatus")]


def test_an_mcp_attachment_breaks_the_declared_expectation(tmp_path: Path) -> None:
    def plant(run: Any) -> None:
        insert_before_reply(run, attachment("mcp_instructions_delta", addedNames=["x"],
                                            addedBlocks=["…"], removedNames=[]))
    assert planted(tmp_path, plant) == [(audit.ENVIRONMENTAL, "mcpInstructions")]


def test_an_effort_other_than_the_installed_pin_is_environmental(tmp_path: Path) -> None:
    def plant(run: Any) -> None:
        run.agent[3]["params"]["output_config"]["effort"] = "medium"     # the generator's pin
    assert planted(tmp_path, plant) == [(audit.ENVIRONMENTAL, "effortOverride")]


def test_a_model_other_than_the_installed_pin_is_environmental(tmp_path: Path) -> None:
    def plant(run: Any) -> None:
        run.assistant("msg_s0003")[1]["message"]["model"] = "claude-opus-5-5"
    assert planted(tmp_path, plant) == [(audit.ENVIRONMENTAL, "modelMismatch")]


def test_a_guard_command_other_than_the_installed_one_is_not_a_guard_entry(
        tmp_path: Path) -> None:
    made = installed(tmp_path)
    run = clean_run()           # still keyed to the fixture's "python3 guard_hook.py"
    result = audit_of(tmp_path / "audit", run, expected=declared(made))
    assert set(found(result)) == {(audit.ENVIRONMENTAL, "hookMissing")}


# --- the label smoke's fixture -------------------------------------------------------------------


def test_the_smoke_fixture_is_fifty_items_each_rendering_at_the_cap() -> None:
    items = smoke.at_cap_items()
    assert len(items) == 50
    assert {len(render.render_guard_item(item)) for item in items} == {6000}
    assert len({json.dumps(item) for item in items}) == 50


def test_a_fixture_item_that_misses_the_cap_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(smoke, "PHRASE", "curta")
    with pytest.raises(ValueError, match="not 6000"):
        smoke.at_cap_items(2)


def test_smoke_fixture_writes_items_that_init_accepts_and_next_batches_whole(
        tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    out = tmp_path / "items.jsonl"
    assert driver.main(["smoke-fixture", "--out", str(out)]) == 0
    assert "wrote 50 items" in capsys.readouterr().out
    assert len(out.read_text("utf-8").splitlines()) == 50
    root = tmp_path / "smoke-root"
    camp = Campaign(root, claude_dir=tmp_path / "claude", clock=Clock())
    camp.init(point="guard-judge", phase="smoke", fixture=out)
    camp.record_expectations(guard_command="g", roles=ROLES)
    action = camp.next()
    camp.issue(action)
    request = (root / "exchange" / "req" / f"{action.opaque}.jsonl").read_text("utf-8")
    assert len(request.splitlines()) == 52             # header, rubric, 50 items
    assert json.loads(request.splitlines()[0])["pages"][0] == [1, 1]
    assert camp.state().batches["bat_0001"].entries == [f"fx_{n:04d}" for n in range(1, 51)]


def test_smoke_fixture_refuses_an_existing_file_and_a_bad_count(
        tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    out = tmp_path / "items.jsonl"
    out.write_text("keep", "utf-8")
    assert driver.main(["smoke-fixture", "--out", str(out)]) == 1
    assert out.read_text("utf-8") == "keep"
    assert driver.main(["smoke-fixture", "--out", str(tmp_path / "n.jsonl"), "--count", "0"]) == 1
    assert "refused" in capsys.readouterr().err
